import os
import tempfile

import numpy as np
import pytest
from omfiles import OmFilePyReader, OmFilePyWriter

from virtualizarr.readers.omfiles import OmFilesVirtualBackend


# Helper functions to create test OM files
def create_simple_om_file():
    """Create a simple OM file with a 2D array."""
    with tempfile.NamedTemporaryFile(suffix='.om', delete=False) as f:
        filepath = f.name

    writer = OmFilePyWriter(filepath)

    # Create a simple 2D array with dimensions 10x5
    data = np.arange(50, dtype=np.float32).reshape(10, 5)
    chunks = (5, 5)  # Chunk size

    # Add dimension information as attributes
    dims_attr = writer.write_scalar("blub_0,blub_1", "_ARRAY_DIMENSIONS")

    # Write the data array
    root = writer.write_array(
        data=data,
        chunks=chunks,
        name="data",
        compression="pfor_delta_2d",
        children=[dims_attr]
    )

    # Close the writer
    writer.close(root)

    return filepath


def create_om_file_with_coordinates():
    """Create an OM file with coordinates."""
    with tempfile.NamedTemporaryFile(suffix='.om', delete=False) as f:
        filepath = f.name

    writer = OmFilePyWriter(filepath)

    # Create coordinate arrays
    lat = np.linspace(0, 90, 10, dtype=np.float32)
    lon = np.linspace(-180, 180, 5, dtype=np.float32)

    # Write coordinates
    lat_dim_attr = writer.write_scalar("lat", "_ARRAY_DIMENSIONS")
    lat_var = writer.write_array(
        data=lat,
        chunks=(10,),
        name="lat",
        compression="pfor_delta_2d",
        children=[lat_dim_attr]
    )
    lon_dim_attr = writer.write_scalar("lon", "_ARRAY_DIMENSIONS")
    lon_var = writer.write_array(
        data=lon,
        chunks=(5,),
        name="lon",
        compression="pfor_delta_2d",
        children=[lon_dim_attr]
    )

    # Add dimension information
    dim_attr = writer.write_scalar("lat,lon", "_ARRAY_DIMENSIONS")

    # Create a data array
    data = np.random.rand(10, 5).astype(np.float32)
    data_var = writer.write_array(
        data=data,
        chunks=(5, 5),
        name="temperature",
        compression="pfor_delta_2d",
        children=[dim_attr]
    )

    # Create the root variable
    root = writer.write_group("", [lat_var, lon_var, data_var])

    # Close the writer
    writer.close(root)

    return filepath


def create_om_file_with_scalar_attrs():
    """Create an OM file with scalar attributes."""
    with tempfile.NamedTemporaryFile(suffix='.om', delete=False) as f:
        filepath = f.name

    writer = OmFilePyWriter(filepath)

    # Create a simple array
    data = np.arange(10, dtype=np.float32)
    data_var = writer.write_array(
        data=data,
        chunks=(5,),
        name="data",
        compression="pfor_delta_2d"
    )

    # Add scalar attributes
    string_attr = writer.write_scalar("Example attribute", "string_attr")
    int_attr = writer.write_scalar(42, "int_attr")
    float_attr = writer.write_scalar(3.14, "float_attr")

    # Add dimension info
    dim_attr = writer.write_scalar("dim_0", "_ARRAY_DIMENSIONS")

    # Create the root variable
    root = writer.write_group("root", [data_var, string_attr, int_attr, float_attr, dim_attr])

    # Close the writer
    writer.close(root)

    return filepath


# Fixtures for the test files
@pytest.fixture
def simple_om_file():
    """Fixture for a simple OM file."""
    filepath = create_simple_om_file()
    yield filepath
    os.unlink(filepath)  # Clean up after test


@pytest.fixture
def om_file_with_coords():
    """Fixture for an OM file with coordinates."""
    filepath = create_om_file_with_coordinates()
    yield filepath
    os.unlink(filepath)  # Clean up after test


@pytest.fixture
def om_file_with_attrs():
    """Fixture for an OM file with attributes."""
    filepath = create_om_file_with_scalar_attrs()
    yield filepath
    os.unlink(filepath)  # Clean up after test


class TestOpenVirtualDataset:
    def test_coord_names(
        self,
        om_file_with_coords,
    ):
        vds = OmFilesVirtualBackend.open_virtual_dataset(om_file_with_coords)

        assert set(vds.coords) == {"lat", "lon"}
        assert vds.coords["lat"].shape == (10,)
        assert vds.coords["lon"].shape == (5,)


class TestOMChunkManifest:
    def test_no_chunking(self, simple_om_file):
        """Test creating a chunk manifest from an OM file without explicit chunking."""
        reader = OmFilePyReader(simple_om_file)
        variables = reader.get_flat_variable_metadata()

        # Find the data variable
        data_var = next((var for name, var in variables.items() if name == "data"), None)
        assert data_var is not None

        # Create manifest
        manifest = OmFilesVirtualBackend._dataset_chunk_manifest(
            path=simple_om_file,
            var=data_var,
            var_reader=reader
        )

        assert manifest.shape_chunk_grid == (1,)

        reader.close()

    @pytest.mark.skip("Need to implement proper support for chunking per variable")
    def test_chunked(self, simple_om_file):
        chunked_om_file = simple_om_file # FIXME
        reader = OmFilePyReader(chunked_om_file)
        variables = reader.get_flat_variable_metadata()

        # Find the data variable
        data_var = next((var for name, var in variables.items() if name == "data"), None)
        assert data_var is not None

        var_reader = reader.init_from_variable(data_var)

        # Create manifest
        manifest = OmFilesVirtualBackend._dataset_chunk_manifest(
            path=chunked_om_file,
            var=data_var,
            var_reader=var_reader
        )
        assert manifest.shape_chunk_grid == (2, 2)

        reader.close()

    # TODO:
    # def test_empty_chunks(self, empty_chunks_hdf5_file):
    #     f = h5py.File(empty_chunks_hdf5_file)
    #     ds = f["data"]
    #     with pytest.raises(ValueError, match="chunked but contains no chunks"):
    #         HDFVirtualBackend._dataset_chunk_manifest(
    #             path=empty_chunks_hdf5_file, dataset=ds
    #         )

class TestOMDimensionHandling:
    def test_extract_dimensions(self, simple_om_file):
        ds = OmFilePyReader(simple_om_file)
        dims = OmFilesVirtualBackend._get_dimensions(ds, "data")
        assert dims == ["blub_0", "blub_1"]

        ds.close()


# class TestOMDatasetToVariable:
#     """Test converting OM variables to xarray Variables."""

#     def test_om_array_to_variable(self, simple_om_file):
#         """Test converting an OM array to an xarray Variable."""
#         # This would test our conversion from OM arrays to xarray variables
#         # Implementation will depend on our final approach
#         reader = OmFilePyReader(simple_om_file)
#         variables = reader.get_flat_variable_metadata()

#         # Find the data variable
#         data_var = next((var for name, var in variables.items() if name == "data"), None)
#         assert data_var is not None

#         # For now, just check we can access the data
#         var_reader = reader.init_from_variable(data_var)
#         data = var_reader[:]
#         assert data.shape == (10, 5)

#         reader.close()


# class TestOMAttributeExtraction:
#     """Test extracting attributes from OM files."""

#     def test_extract_scalar_attributes(self, om_file_with_attrs):
#         """Test extracting scalar attributes from an OM file."""
#         # This would test our attribute extraction logic
#         reader = OmFilePyReader(om_file_with_attrs)
#         variables = reader.get_flat_variable_metadata()

#         # Check for some known attributes
#         string_attr_var = next((var for name, var in variables.items() if name == "string_attr"), None)
#         assert string_attr_var is not None

#         # Read the attribute value
#         attr_reader = reader.init_from_variable(string_attr_var)
#         assert attr_reader.is_scalar
#         value = attr_reader.get_scalar()
#         assert value == "Example attribute"

#         reader.close()


# class TestOpenVirtualDataset:
#     """Test opening OM files as virtual datasets."""

#     def test_open_simple_dataset(self, simple_om_file):
#         """Test opening a simple OM file as a virtual dataset."""
#         # This will test our open_virtual_dataset function
#         # Since we haven't implemented it yet, this is a placeholder
#         # that we'll update as we make progress

#         # Direct reading with OmFilePyReader to compare later
#         reader = OmFilePyReader(simple_om_file)
#         variables = reader.get_flat_variable_metadata()
#         data_var = next((var for name, var in variables.items() if name == "data"), None)
#         var_reader = reader.init_from_variable(data_var)
#         expected_data = var_reader[:]
#         reader.close()

#         # For now, assert that we can read the expected data
#         assert expected_data.shape == (10, 5)

#         # This is where we'd test our virtualizarr implementation when ready
#         # vds = open_virtual_dataset(simple_om_file, backend=OMVirtualBackend)
#         # assert isinstance(vds, xr.Dataset)
#         # assert "data" in vds.variables
#         # assert isinstance(vds["data"].data, ManifestArray)

#     def test_open_dataset_with_coords(self, om_file_with_coords):
#         """Test opening an OM file with coordinates as a virtual dataset."""
#         # Similar to above, this is a placeholder test

#         # Direct reading to verify test data
#         reader = OmFilePyReader(om_file_with_coords)
#         variables = reader.get_flat_variable_metadata()

#         # Check that the temperature variable exists
#         temp_var = next((var for name, var in variables.items() if name == "temperature"), None)
#         assert temp_var is not None

#         # Check that coordinates exist
#         lat_var = next((var for name, var in variables.items() if name == "lat"), None)
#         lon_var = next((var for name, var in variables.items() if name == "lon"), None)
#         assert lat_var is not None
#         assert lon_var is not None

#         reader.close()

#         # This is where we'd test our virtualizarr implementation when ready
#         # vds = open_virtual_dataset(om_file_with_coords, backend=OMVirtualBackend)
#         # assert isinstance(vds, xr.Dataset)
#         # assert "temperature" in vds.variables
#         # assert set(vds.coords) == {"lat", "lon"}


# # Placeholder for integration tests
# def test_read_data_roundtrip(simple_om_file):
#     """Test reading data with the virtual dataset matches direct reading."""
#     # This will be an integration test that compares reading through
#     # virtualizarr with direct reading from the file

#     # Direct reading
#     reader = OmFilePyReader(simple_om_file)
#     variables = reader.get_flat_variable_metadata()
#     data_var = next((var for name, var in variables.items() if name == "data"), None)
#     var_reader = reader.init_from_variable(data_var)
#     expected_data = var_reader[:]
#     reader.close()

#     # For now, just assert on the direct reading
#     assert expected_data.shape == (10, 5)

#     # Once implemented, we'd compare with virtual reading:
#     # vds = open_virtual_dataset(simple_om_file, backend=OMVirtualBackend)
#     # np.testing.assert_array_equal(vds["data"].values, expected_data)
