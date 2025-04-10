import os
from typing import Dict, List, Optional, Union, Any, Iterable, Mapping, Hashable, Tuple

import numpy as np
import xarray as xr
import fsspec

from virtualizarr.manifests import ChunkManifest, ManifestArray, ChunkEntry
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.readers.common import construct_fully_virtual_dataset, replace_virtual_with_loadable_vars
from virtualizarr.types import ChunkKey
from virtualizarr.utils import soft_import

# omfiles = soft_import("omfiles", "For reading Open-Meteo files", strict=False)
import omfiles

# Constants
DIMENSION_KEY = "_ARRAY_DIMENSIONS"


class OmFilesVirtualBackend(VirtualBackend):
    """Backend for Open-Meteo (.om) file format."""

    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, xr.Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> xr.Dataset:
        """
        Open an OM file as a virtual dataset.

        Args:
            filepath: Path to the OM file
            group: Group within the OM file to open (not used for OM, but kept for API consistency)
            drop_variables: Variables to exclude
            loadable_variables: Variables to include
            decode_times: Whether to decode times
            indexes: Dictionary of indexes to use
            virtual_backend_kwargs: Additional arguments for the virtual backend
            reader_options: Additional arguments for the reader

        Returns:
            xr.Dataset: Virtual dataset
        """
        if omfiles is None:
            raise ImportError("omfiles is required for using the OmFilesVirtualBackend")

        if virtual_backend_kwargs:
            raise NotImplementedError(
                "OM reader does not understand any virtual_backend_kwargs"
            )

        _drop_vars: list[Hashable] = (
            [] if drop_variables is None else list(drop_variables)
        )

        # Extract virtual variables from the OM file
        virtual_vars = OmFilesVirtualBackend._virtual_vars_from_om(
            path=filepath,
            reader_options=reader_options,
        )

        print("Virtual variables:")
        print(virtual_vars)

        # Get root attributes
        attrs = OmFilesVirtualBackend._get_root_attrs(
            path=filepath,
            reader_options=reader_options,
        )

        print(attrs)

        # Look for coordinates attribute
        coordinates_attr = attrs.pop("coordinates", "")
        coord_names = coordinates_attr.split() if coordinates_attr else []

        # Create a fully virtual dataset
        fully_virtual_dataset = construct_fully_virtual_dataset(
            virtual_vars=virtual_vars,
            coord_names=coord_names,
            attrs=attrs,
        )

        # TODO: # Replace virtual variables with loadable variables if specified
        # vds = replace_virtual_with_loadable_vars(
        #     fully_virtual_dataset,
        #     filepath,
        #     loadable_variables=loadable_variables,
        #     reader_options=reader_options,
        #     indexes=indexes,
        #     decode_times=decode_times,
        # )
        vds = fully_virtual_dataset

        # Drop specified variables
        return vds.drop_vars(_drop_vars)

    @staticmethod
    def _extract_attrs(reader: 'omfiles.OmFilePyReader', variable_path: str) -> Dict[str, Any]:
        """Extract attributes for a variable by finding its direct scalar children."""
        attrs = {}
        variables = reader.get_flat_variable_metadata()
        print(variables)

        # Normalize the path to ensure consistent matching
        if not variable_path.endswith('/') and variable_path:
            variable_path += '/'
        elif variable_path == '':
            # Root variable has no path prefix
            pass

        # Find all scalar children of this variable
        for name, var in variables.items():
            print(name, var)
            # Skip the variable itself
            if name == variable_path.rstrip('/'):
                continue

            # Check if this is a direct child of the variable
            if name.startswith(variable_path) and '/' not in name[len(variable_path):]:
                var_reader = reader.init_from_variable(var)
                if var_reader and var_reader.is_scalar:
                    # Use the basename as the attribute name
                    attr_name = name[len(variable_path):]
                    attrs[attr_name] = var_reader.get_scalar()

        return attrs

    @staticmethod
    def _get_dimensions(reader: 'omfiles.OmFilePyReader', variable_path: str) -> List[str]:
        """
        Get dimension names for a variable.

        Args:
            reader: The OM file reader
            variable_path: Path to the variable

        Returns:
            List of dimension names
        """
        # First check if the variable has an _ARRAY_DIMENSIONS attribute
        attrs = OmFilesVirtualBackend._extract_attrs(reader, variable_path)

        if DIMENSION_KEY in attrs:
            # Parse the comma-separated dimension names
            dim_str = attrs[DIMENSION_KEY]
            if isinstance(dim_str, str):
                dimensions = dim_str.split(',')
                return dimensions

        # Fallback to generic dimension names
        return [f"dim_{i}" for i in range(len(reader.shape))]

    @staticmethod
    def _dataset_chunk_manifest(
        path: str,
        var: 'omfiles.OmVariable',
        var_reader: 'omfiles.OmFilePyReader',
    ) -> ChunkManifest:
        """Create a chunk manifest for a variable."""
        # For the initial implementation, treat each variable as a single chunk
        # This is a simplification - OM files support chunking but we'll handle that later

        chunk_entry = ChunkEntry.with_validation(
            path=path,
            offset=var.offset,
            length=var.size
        )

        # Use a single chunk key for the entire variable
        chunk_key = ChunkKey("0")
        chunk_entries = {chunk_key: chunk_entry}

        return ChunkManifest(entries=chunk_entries)

    @staticmethod
    def _dataset_to_variable(
        path: str,
        reader: 'omfiles.OmFilePyReader',
        var: 'omfiles.OmVariable',
        var_reader: 'omfiles.OmFilePyReader',
    ) -> Optional[xr.Variable]:
        """Convert an OM variable to an xarray Variable with virtual data."""
        if var_reader.is_scalar or var_reader.is_group:
            return None

        # Get variable attributes
        attrs = OmFilesVirtualBackend._extract_attrs(reader, var_reader.name)

        # Get dimensions
        dims = OmFilesVirtualBackend._get_dimensions(var_reader, var_reader.name)

        # OM files can have chunks, but for simplicity we'll treat each variable as a single chunk for now
        # In the future, we can extract actual chunk information from the OM format
        chunks = var_reader.shape

        # Create array metadata for zarr
        metadata = create_v3_array_metadata(
            shape=var_reader.shape,
            data_type=var_reader.dtype,
            chunk_shape=chunks,
        )

        # Create chunk manifest
        manifest = OmFilesVirtualBackend._dataset_chunk_manifest(path, var, var_reader)

        # Create manifest array
        marray = ManifestArray(metadata=metadata, chunkmanifest=manifest)

        # Create xarray variable
        return xr.Variable(data=marray, dims=dims, attrs=attrs)

    @staticmethod
    def _virtual_vars_from_om(
        path: str,
        reader_options: Optional[dict] = None,
    ) -> Dict[str, xr.Variable]:
        """Extract virtual variables from an OM file."""
        # Open the OM file
        fs = fsspec.filesystem("file")
        with omfiles.OmFilePyReader.from_path(path) if fs.protocol == "file" else omfiles.OmFilePyReader.from_fsspec(fs.open(path)) as reader:
            variables = reader.get_flat_variable_metadata()

            # Create virtual variables dict
            virtual_vars = {}

            # Process each variable
            for var_name, var in variables.items():
                var_reader = reader.init_from_variable(var)

                # Skip scalar variables and groups for now
                if var_reader.is_scalar or var_reader.is_group:
                    continue

                # Convert to xarray Variable
                variable = OmFilesVirtualBackend._dataset_to_variable(
                    path=path,
                    reader=reader,
                    var=var,
                    var_reader=var_reader,
                )

                if variable is not None:
                    virtual_vars[var_name] = variable

            return virtual_vars

    @staticmethod
    def _get_root_attrs(
        path: str,
        reader_options: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Get global attributes from the OM file."""
        fs = fsspec.filesystem("file")
        with omfiles.OmFilePyReader.from_path(path) if fs.protocol == "file" else omfiles.OmFilePyReader.from_fsspec(fs.open(path)) as reader:
            # Extract attributes from the root
            return OmFilesVirtualBackend._extract_attrs(reader, "root")
