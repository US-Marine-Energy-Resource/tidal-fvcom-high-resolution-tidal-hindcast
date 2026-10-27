#!/usr/bin/env python3
"""
Stitch together monthly HSDS files into a single yearly file.
Performs sanity checking on timestamps and ensures data continuity.
"""

import argparse
import gc
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

from tidal_fvcom.config import config
from tidal_fvcom.file_manager import get_hsds_temp_dir, get_hsds_final_file_path
from tidal_fvcom.nc_manager import calculate_optimal_chunk_sizes

# HDF5 cache settings for optimal performance
HDF5_READ_CACHE = config["hdf5_cache"]["read_cache_bytes"]
HDF5_WRITE_CACHE = config["hdf5_cache"]["stitch_write_cache_bytes"]

# ---------------------------------------------------------------------------
# Write-cache strategy for the face-block stitch
#
# The stitch writes each variable one COMPLETE face-block at a time (the whole
# time axis for a contiguous range of faces), so every HDF5 chunk is written
# once, in full. There is no read-modify-write for the chunk cache to absorb,
# so the old multi-GB write cache bought us nothing here -- the working set per
# variable (tens of GB) always dwarfed it. We deliberately keep the write cache
# SMALL (a few dozen chunks) but fix the two knobs that actually matter:
#   * rdcc_nslots: the chunk-cache hash table. h5py's default of 521 hash-
#     thrashes once the cache holds more than a few hundred chunks. We set a
#     prime well above the chunks we keep resident.
#   * rdcc_w0=1.0: prefer evicting fully-written chunks first (we never revisit
#     a chunk once its block is written), so the cache never holds dead weight.
# The cache size is derived from the REAL chunk size via
# calculate_optimal_chunk_sizes -- the same function that created the datasets.
# ---------------------------------------------------------------------------
STITCH_WRITE_RDCC_TARGET_CHUNKS = 128  # keep ~128 chunks resident
STITCH_WRITE_RDCC_NSLOTS = 12809  # prime > 128 chunks * 100
STITCH_WRITE_RDCC_W0 = 1.0

# Default RAM budget (GB) for the single face-block accumulator held in memory
# at a time. Block width is derived from this at runtime, so peak memory is
# bounded regardless of mesh size (SE Alaska = 1.1M faces, Puget = 1.7M).
# Overridable via --memory-budget-gb.
DEFAULT_MEMORY_BUDGET_GB = 32.0

# Global NaN replacement value
NAN_FILL_VALUE = -999.0


def representative_chunk_bytes(file_structure):
    """
    Bytes-per-chunk for a representative 2D (time, face) variable, computed with
    calculate_optimal_chunk_sizes -- the same chunking the datasets were created
    with. Used to size the write cache. Falls back to the config target size if
    no 2D time-varying variable is present.
    """
    for var_info in file_structure["variable_info"].values():
        if var_info.get("is_static", False):
            continue
        yearly_shape = var_info.get("yearly_shape")
        if yearly_shape is None or len(yearly_shape) != 2:
            continue
        chunks = calculate_optimal_chunk_sizes(
            shape=yearly_shape,
            dims=["time", "face"],
            dtype=var_info["dtype"],
            config=config,
        )
        return int(np.prod(chunks)) * np.dtype(var_info["dtype"]).itemsize
    return int(config["dataset"]["encoding"]["chunk_spec"]["target_size_mb"] * 1024 * 1024)

# Variables to exclude from final stitched output
SKIP_VARIABLES = {
    "vap_water_column_max_sea_water_power_density",
    "vap_water_column_max_sea_water_speed",
    "vap_water_column_mean_sea_water_power_density",
    "vap_water_column_mean_sea_water_speed",
    "vap_water_column_mean_sea_water_to_direction",
    "vap_water_column_mean_u",
    "vap_water_column_mean_v",
    "u_sigma_layer_01",
    "u_sigma_layer_02",
    "u_sigma_layer_03",
    "u_sigma_layer_04",
    "u_sigma_layer_05",
    "u_sigma_layer_06",
    "u_sigma_layer_07",
    "u_sigma_layer_08",
    "u_sigma_layer_09",
    "u_sigma_layer_10",
    "v_sigma_layer_01",
    "v_sigma_layer_02",
    "v_sigma_layer_03",
    "v_sigma_layer_04",
    "v_sigma_layer_05",
    "v_sigma_layer_06",
    "v_sigma_layer_07",
    "v_sigma_layer_08",
    "v_sigma_layer_09",
    "v_sigma_layer_10",
}


def compute_expected_full_year_timeline(location_config):
    """
    Compute the complete expected full-year timeline for a location.

    Shared by structure creation and the resume path so both agree on the
    yearly dimensions without re-creating the output file.

    Returns:
        tuple: (expected_timeline: pd.DatetimeIndex, expected_time_steps: int)
    """
    start_date = pd.to_datetime(location_config["start_date_utc"])
    delta_t_seconds = location_config["expected_delta_t_seconds"]
    expected_end_date = (
        start_date + pd.DateOffset(years=1) - pd.Timedelta(seconds=delta_t_seconds)
    )
    expected_timeline = pd.date_range(
        start=start_date,
        end=expected_end_date,
        freq=pd.Timedelta(seconds=delta_t_seconds),
        tz="UTC",
    )
    return expected_timeline, len(expected_timeline)


def replace_nans_inplace(data):
    """
    Replace NaNs with NAN_FILL_VALUE in place (no copy) for plain float arrays.

    The stitch hot path reads a fresh array from disk each time, so in-place
    editing is safe and avoids the full-array copy that
    replace_nans_with_fill_value makes. Non-float / structured arrays are
    returned unchanged. Returns data for chaining.
    """
    if data.dtype.names is None and np.issubdtype(data.dtype, np.floating):
        np.copyto(data, NAN_FILL_VALUE, where=np.isnan(data))
    return data


def replace_nans_with_fill_value(data, fill_value=NAN_FILL_VALUE):
    """
    Replace all NaN values in data with fill_value.

    Handles both floating point arrays and structured arrays (for meta compound dataset).

    Args:
        data: numpy array (can be regular array or structured array)
        fill_value: Value to replace NaNs with (default: -999.0)

    Returns:
        numpy array with NaNs replaced
    """
    # Make a copy to avoid modifying original
    data_filled = data.copy()

    # Handle structured arrays (meta compound dataset)
    if data.dtype.names is not None:
        for field_name in data.dtype.names:
            field_data = data_filled[field_name]
            # Only process numeric fields that can have NaNs
            if np.issubdtype(field_data.dtype, np.floating):
                nan_mask = np.isnan(field_data)
                nan_count = np.sum(nan_mask)
                if nan_count > 0:
                    field_data[nan_mask] = fill_value
                    print(
                        f"    Replaced {nan_count} NaNs in field '{field_name}' with {fill_value}"
                    )
    # Handle regular arrays
    elif np.issubdtype(data.dtype, np.floating):
        nan_mask = np.isnan(data_filled)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            data_filled[nan_mask] = fill_value
            print(f"    Replaced {nan_count} NaNs with {fill_value}")

    return data_filled


def add_fill_value_attr(dataset, fill_value=NAN_FILL_VALUE):
    """
    Add _FillValue attribute to dataset.

    Args:
        dataset: h5py Dataset object
        fill_value: Fill value to document in attributes
    """
    try:
        # Convert fill_value to appropriate dtype
        if np.issubdtype(dataset.dtype, np.floating):
            dataset.attrs["_FillValue"] = np.float32(fill_value)
            print(f"    Added _FillValue={fill_value} attribute")
        elif np.issubdtype(dataset.dtype, np.integer):
            dataset.attrs["_FillValue"] = int(fill_value)
            print(f"    Added _FillValue={int(fill_value)} attribute")
    except Exception as e:
        print(f"    Warning: Could not set _FillValue attribute: {e}")


def pad_to_full_year(output_file, location_config, file_structure):
    """
    Ensure the yearly file has exactly 1 year of data with no gaps.

    This function generates the complete expected timeline for a full year,
    compares it against actual data, and fills any missing timestamps (at
    beginning, middle, or end of year) with NaN fill values.

    Parameters:
    - output_file: Path to the yearly H5 file
    - location_config: Location configuration from config.py
    - file_structure: File structure dict from analyze_temporal_file_structure
    """
    print("\n" + "=" * 80)
    print("PADDING TO FULL YEAR (FILLING ANY GAPS)")
    print("=" * 80)

    # Parse configuration
    start_date = pd.to_datetime(location_config["start_date_utc"])
    delta_t_seconds = location_config["expected_delta_t_seconds"]

    # Calculate expected end date using pandas date offset (handles leap years automatically)
    # Add 1 year, then subtract one timestep
    expected_end_date = (
        start_date + pd.DateOffset(years=1) - pd.Timedelta(seconds=delta_t_seconds)
    )

    print(f"Start date:        {start_date}")
    print(f"Expected end date: {expected_end_date}")
    print(f"Timestep:          {delta_t_seconds} seconds")

    # Generate complete expected timeline for full year using pandas
    print("\nGenerating expected full-year timeline...")
    expected_timeline = pd.date_range(
        start=start_date,
        end=expected_end_date,
        freq=pd.Timedelta(seconds=delta_t_seconds),
        tz="UTC",  # Ensure UTC timezone to match actual data
    )

    expected_count = len(expected_timeline)
    print(f"Expected timesteps: {expected_count}")
    print(f"First expected:     {expected_timeline[0]}")
    print(f"Last expected:      {expected_timeline[-1]}")

    # Read actual timeline from file
    print("\nReading actual timeline from file...")
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        actual_time_index = h5f["time_index"][:]
        actual_count = len(actual_time_index)

        # Decode and parse actual timestamps using pandas
        if actual_time_index.dtype.kind == "S":
            actual_time_strings = [t.decode("utf-8") for t in actual_time_index]
        else:
            actual_time_strings = [str(t) for t in actual_time_index]

        actual_timeline = pd.to_datetime(actual_time_strings)

    print(f"Actual timesteps:   {actual_count}")
    print(f"First actual:       {actual_timeline[0]}")
    print(f"Last actual:        {actual_timeline[-1]}")

    # Use pandas to find missing timestamps
    print("\nAnalyzing timeline gaps...")

    # Create DataFrames for comparison
    expected_df = pd.DataFrame(
        {"timestamp": expected_timeline, "expected_idx": range(expected_count)}
    )
    actual_df = pd.DataFrame(
        {"timestamp": actual_timeline, "actual_idx": range(actual_count)}
    )

    # Merge to find which expected timestamps exist in actual data
    merged_df = expected_df.merge(actual_df, on="timestamp", how="left")

    # Find missing timestamps
    missing_mask = merged_df["actual_idx"].isna()
    missing_count = missing_mask.sum()

    if missing_count == 0:
        print("✓ No gaps found - timeline is complete")
        print("=" * 80 + "\n")
        return

    print(f"Found {missing_count} missing timesteps")
    print(f"Missing percentage: {100 * missing_count / expected_count:.2f}%")

    # Show gap ranges using pandas
    missing_indices = merged_df[missing_mask].index.tolist()
    if len(missing_indices) > 0:
        print("\nGap locations:")
        # Group consecutive indices to show gap ranges
        gap_starts = [missing_indices[0]]
        gap_ends = []

        for i in range(1, len(missing_indices)):
            if missing_indices[i] != missing_indices[i - 1] + 1:
                gap_ends.append(missing_indices[i - 1])
                gap_starts.append(missing_indices[i])
        gap_ends.append(missing_indices[-1])

        for start, end in zip(gap_starts, gap_ends):
            gap_size = end - start + 1
            print(f"  Gap: index {start} to {end} ({gap_size} timesteps)")
            print(f"       {expected_timeline[start]} to {expected_timeline[end]}")

    # Create mapping array: for each expected index, what is the actual index (or None if missing)
    print("\nCreating timestamp mapping...")
    expected_to_actual_map = merged_df[
        "actual_idx"
    ].values  # Will have NaN for missing timestamps
    mapped_count = (~pd.isna(expected_to_actual_map)).sum()
    print(f"Mapped {mapped_count}/{expected_count} timestamps")

    # Rebuild datasets with complete timeline
    print("\nRebuilding datasets with complete timeline...")

    with h5py.File(output_file, "a", rdcc_nbytes=HDF5_WRITE_CACHE) as h5f:
        # 1. Update time_index
        print("\n  Updating time_index...")
        # Use pandas to format timestamps consistently
        expected_time_strings = expected_timeline.strftime(
            "%Y-%m-%d %H:%M:%S+00:00"
        ).tolist()

        # Delete old dataset and create new one
        del h5f["time_index"]

        time_dtype = h5py.string_dtype(encoding="utf-8", length=26)
        h5f.create_dataset("time_index", data=expected_time_strings, dtype=time_dtype)
        print(f"    Updated time_index: {actual_count} → {expected_count} timesteps")

        # 2. Update all time-varying variables (chunked processing for memory efficiency)
        print("\n  Rebuilding time-varying variables...")
        for var_name, var_info in file_structure["variable_info"].items():
            if var_info.get("is_static", False):
                print(f"    {var_name}: SKIPPED (static variable)")
                continue

            print(f"    Processing {var_name}...")

            # Get metadata from old dataset
            old_dataset = h5f[var_name]
            actual_shape = old_dataset.shape
            dtype = old_dataset.dtype
            attrs = dict(old_dataset.attrs)
            chunks = old_dataset.chunks

            # Determine new shape
            if len(actual_shape) == 1:
                new_shape = (expected_count,)
            else:
                new_shape = (expected_count,) + actual_shape[1:]

            print(f"      Old shape: {actual_shape}, New shape: {new_shape}")

            # Create temporary dataset with full shape, filled with NaN
            temp_name = f"{var_name}_temp"
            new_dataset = h5f.create_dataset(
                temp_name,
                shape=new_shape,
                dtype=dtype,
                chunks=chunks,
                fillvalue=NAN_FILL_VALUE,
            )

            # Copy actual data in chunks to correct positions
            # Process in chunks of time steps to avoid memory issues
            CHUNK_SIZE = 500  # Process 500 timesteps at a time (~3.3 GB per chunk for Puget Sound)

            valid_mask = ~pd.isna(expected_to_actual_map)
            valid_expected_indices = np.where(valid_mask)[0]
            valid_actual_indices = expected_to_actual_map[valid_mask].astype(int)

            print(
                f"      Copying {len(valid_expected_indices)} timesteps in chunks of {CHUNK_SIZE}..."
            )

            # Group indices by chunks for efficient processing
            for chunk_start in range(0, len(valid_expected_indices), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(valid_expected_indices))
                chunk_expected = valid_expected_indices[chunk_start:chunk_end]
                chunk_actual = valid_actual_indices[chunk_start:chunk_end]

                # Read chunk of actual data
                if len(actual_shape) == 1:
                    # 1D time series
                    actual_chunk = old_dataset[chunk_actual]
                    new_dataset[chunk_expected] = actual_chunk
                else:
                    # 2D (time, face) - read all faces for this time chunk
                    actual_chunk = old_dataset[chunk_actual, :]
                    new_dataset[chunk_expected, :] = actual_chunk

                if (chunk_start // CHUNK_SIZE + 1) % 10 == 0:
                    progress = (chunk_end / len(valid_expected_indices)) * 100
                    print(f"        Progress: {progress:.1f}%")

            # Restore attributes to new dataset
            for attr_name, attr_value in attrs.items():
                new_dataset.attrs[attr_name] = attr_value

            # Delete old dataset and rename temp to original name
            del h5f[var_name]
            h5f[var_name] = new_dataset
            del h5f[temp_name]

            print(
                f"      ✓ Complete: {actual_shape} → {new_shape}, filled {missing_count} gaps with {NAN_FILL_VALUE}"
            )

    # Validation
    print("\nValidating final timeline...")
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        final_time_index = h5f["time_index"][:]
        final_count = len(final_time_index)

        if final_count != expected_count:
            raise ValueError(
                f"Validation failed: expected {expected_count} timesteps, got {final_count}"
            )

        # Parse and validate using pandas
        if final_time_index.dtype.kind == "S":
            final_time_strings = [t.decode("utf-8") for t in final_time_index]
        else:
            final_time_strings = [str(t) for t in final_time_index]

        final_timeline = pd.to_datetime(final_time_strings)

        # Check monotonicity using pandas
        is_monotonic = final_timeline.is_monotonic_increasing

        if not is_monotonic:
            raise ValueError(
                "Validation failed: timestamps are not monotonically increasing"
            )

        print(f"✓ Validation passed: {final_count} timesteps, monotonically increasing")

    print(f"\n✓ Successfully filled {missing_count} gaps with {NAN_FILL_VALUE}")
    print("=" * 80 + "\n")


def stitch_single_b1_file_for_hsds(
    monthly_dir,
    output_file,
    location_name,
    location_config,
    perform_checks=True,
    memory_budget_gb=DEFAULT_MEMORY_BUDGET_GB,
):
    """
    Stitch together temporal HSDS files into single yearly file

    Parameters:
    - monthly_dir: Directory containing temporal H5 files
    - output_file: Output path for yearly H5 file
    - location_name: Location name for file pattern matching
    - location_config: Location configuration from config.py (needed for padding)
    - perform_checks: Whether to perform sanity checks
    - memory_budget_gb: RAM budget for one face-block accumulator (bounds peak memory)
    """
    monthly_dir = Path(monthly_dir)

    # Find all temporal files using consistent naming pattern (chunk_XXX)
    file_pattern = f"{location_name}_chunk_*_hsds.h5"
    metadata_pattern = f"{location_name}_chunk_*_metadata.json"

    # Find all temporal files and metadata
    temporal_files = sorted(list(monthly_dir.glob(file_pattern)))
    metadata_files = sorted(list(monthly_dir.glob(metadata_pattern)))

    if len(temporal_files) == 0:
        raise ValueError(
            f"No temporal HSDS files found matching pattern '{file_pattern}' in {monthly_dir}"
        )

    print(f"Found {len(temporal_files)} temporal files to stitch together")

    # Step 1: Load and validate metadata
    if perform_checks and len(metadata_files) > 0:
        print("Step 1: Validating temporal file metadata...")
        validate_temporal_metadata(metadata_files)

    # Step 2: Analyze structure from first file
    print("Step 2: Analyzing file structure...")
    file_structure = analyze_temporal_file_structure(temporal_files)

    # Step 3: Create yearly file structure with full year dimensions
    print("Step 3: Creating yearly H5 file with full year dimensions")
    create_yearly_file_structure(
        output_file, temporal_files[0], file_structure, location_config
    )

    # Step 4: Stitch data with timestamp mapping (gaps automatically filled)
    print("Step 4: Stitching temporal data with timestamp mapping...")
    stitch_data_into_yearly_file(
        output_file, temporal_files, file_structure, memory_budget_gb=memory_budget_gb
    )

    # Step 5: Final validation
    if perform_checks:
        print("Step 5: Performing final validation...")
        validate_yearly_file(output_file, file_structure)

    print(f"Successfully created yearly HSDS file: {output_file}")
    return output_file


def validate_temporal_metadata(metadata_files):
    """Validate temporal metadata for continuity and completeness"""
    temporal_metadata = []
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            temporal_metadata.append(metadata)

    # Sort by chunk number
    temporal_metadata.sort(key=lambda x: x["chunk"])
    found_chunks = {m["chunk"] for m in temporal_metadata}
    expected_chunks = set(range(len(metadata_files)))

    # Check for sequential chunks
    if found_chunks != expected_chunks:
        raise ValueError(
            f"Missing or duplicate chunks. Expected {sorted(expected_chunks)}, found {sorted(found_chunks)}"
        )

    # All chunks must have same variables
    all_variables = [set(m["variables"]) for m in temporal_metadata]
    if len(set(frozenset(vars) for vars in all_variables)) > 1:
        raise ValueError("Variable lists differ between chunks")

    print(f"Metadata validation complete for {len(metadata_files)} chunks")


def analyze_temporal_file_structure(temporal_files):
    """Analyze structure of temporal files to determine yearly file dimensions"""
    print("Analyzing temporal file structure...")

    print(f"Reading structure from first file: {temporal_files[0].name}\n")

    with h5py.File(temporal_files[0], "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        # Get variable information
        variable_info = {}
        meta_field_info = {}  # Face-only variables that will become meta fields
        total_time_steps = 0
        n_faces = None

        # First pass: collect basic info and identify face-only variables
        # Skip special datasets that are handled separately
        skip_datasets = {"time_index", "sigma_layer", "sigma_level"}

        print("\nCollecting variable info from first temporal file:")
        print("Validating NLR spec compliance (only 1D or 2D variables allowed)...")

        invalid_3d_vars = []

        for var_name in h5f.keys():
            if var_name not in skip_datasets:
                # Check if this variable should be excluded from final output
                if var_name in SKIP_VARIABLES:
                    print(f"  {var_name}: SKIPPED (in exclude list)")
                    continue

                var_dataset = h5f[var_name]
                dims = var_dataset.dims
                shape = var_dataset.shape
                dtype = var_dataset.dtype

                print(f"  {var_name}: shape={shape}, dtype={dtype}")

                # NLR spec validation: only 1D or 2D variables allowed
                if len(shape) > 2:
                    invalid_3d_vars.append((var_name, shape))

                # Determine if this is a face-only variable for meta
                if len(shape) == 1:
                    # This is a 1D variable - could be face-only (for meta) or time-only
                    # We'll determine this by checking if size matches n_faces
                    if n_faces is None:
                        # First 1D variable - assume this is n_faces
                        # We need latitude/longitude to always exist
                        if var_name == "latitude" or var_name == "longitude":
                            n_faces = shape[0]
                            print(f"  Determined n_faces={n_faces} from {var_name}")

                variable_info[var_name] = {
                    "dims": [d.label for d in dims] if dims else None,
                    "shape_template": shape,
                    "dtype": dtype,
                    "attrs": dict(var_dataset.attrs),
                }

        # Validate we found n_faces
        if n_faces is None:
            raise ValueError(
                "Could not determine n_faces - latitude or longitude variable not found"
            )

        print(f"\nNumber of faces: {n_faces}")

        # Raise error if any 3D variables found
        if invalid_3d_vars:
            error_msg = (
                "\n" + "=" * 80 + "\n"
                "ERROR: NLR spec violation - 3D variables detected in H5 file!\n"
                "=" * 80 + "\n"
                "The NLR standardization spec requires all data variables to be 1D or 2D.\n"
                "3D variables (time, sigma, face) must be split into separate 2D variables.\n\n"
                "Found the following 3D variables:\n"
            )
            for var_name, shape in invalid_3d_vars:
                error_msg += f"  - {var_name}: shape={shape}\n"
            error_msg += (
                "\nTo fix this issue:\n"
                "1. Ensure convert_b1_vap_tidal_data_for_hsds.py is splitting 3D variables\n"
                "2. Re-run the conversion step to regenerate temporal H5 files\n"
                "3. Each 3D variable should be split into N separate 2D variables:\n"
                "   {var_name}_sigma_layer_1, {var_name}_sigma_layer_2, ..., {var_name}_sigma_layer_N\n"
                "=" * 80
            )
            raise ValueError(error_msg)

    # Second pass: calculate total time steps across all files
    print("\nCalculating total time steps...")
    for temporal_file in temporal_files:
        with h5py.File(temporal_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
            time_steps = len(h5f["time_index"])
            total_time_steps += time_steps

    print(f"Total time steps across all files: {total_time_steps}")

    # Third pass: separate face-only variables (for meta) from time-varying variables
    print("\nSeparating face-only variables (meta fields) from time-varying variables:")

    # Standard field ordering for meta compound dataset
    meta_field_order = [
        "latitude",
        "longitude",
        "water_depth",
        "timezone",
        "jurisdiction",
        "element_vertex_1_lat",
        "element_vertex_1_lon",
        "element_vertex_2_lat",
        "element_vertex_2_lon",
        "element_vertex_3_lat",
        "element_vertex_3_lon",
    ]

    for var_name, var_info in list(variable_info.items()):
        shape_template = var_info["shape_template"]

        # Identify face-only variables: 1D with size == n_faces
        if len(shape_template) == 1 and shape_template[0] == n_faces:
            # This is a face-only variable - goes into meta
            print(f"  {var_name}: shape={shape_template} → META FIELD")
            meta_field_info[var_name] = var_info
            # Remove from regular variable_info (will be in meta instead)
            del variable_info[var_name]
        elif len(shape_template) == 2 and shape_template[1] == n_faces:
            # Time-varying 2D variable (time, face)
            yearly_shape = (total_time_steps, n_faces)
            var_info["yearly_shape"] = yearly_shape
            var_info["is_static"] = False
            print(
                f"  {var_name}: shape={shape_template} → Time-varying (yearly: {yearly_shape})"
            )
        else:
            # Other patterns - keep as-is (shouldn't happen in this dataset)
            var_info["yearly_shape"] = shape_template
            var_info["is_static"] = True
            print(f"  {var_name}: shape={shape_template} → Keeping as-is")

    # Sort meta fields by standard order, then alphabetically for any extras
    sorted_meta_fields = {}
    # First add fields in standard order
    for field_name in meta_field_order:
        if field_name in meta_field_info:
            sorted_meta_fields[field_name] = meta_field_info[field_name]

    # Then add any remaining fields alphabetically
    remaining_fields = sorted(
        [f for f in meta_field_info.keys() if f not in meta_field_order]
    )
    for field_name in remaining_fields:
        sorted_meta_fields[field_name] = meta_field_info[field_name]

    print(f"\nMeta will contain {len(sorted_meta_fields)} fields:")
    for field_name in sorted_meta_fields.keys():
        print(f"  - {field_name}")

    print(f"\nTime-varying variables: {len(variable_info)}")

    if SKIP_VARIABLES:
        print(f"\nExcluded variables (not in output): {len(SKIP_VARIABLES)}")
        for skip_var in sorted(SKIP_VARIABLES):
            print(f"  - {skip_var}")

    return {
        "n_faces": n_faces,
        "total_time_steps": total_time_steps,
        "n_temporal_files": len(temporal_files),
        "variable_info": variable_info,
        "meta_field_info": sorted_meta_fields,
    }


def assemble_h5_data_for_meta(template_file, meta_field_info, n_faces):
    """
    Read individual face-only datasets from template file and assemble into compound array.

    Args:
        template_file: Path to first temporal H5 file
        meta_field_info: Dict of field names to their info (dtype, attrs, etc.)
        n_faces: Number of faces

    Returns:
        numpy structured array ready for meta dataset
    """
    print("\nAssembling meta compound dataset:")

    # Build compound dtype from meta_field_info
    meta_dtype = []
    for field_name, field_info in meta_field_info.items():
        dtype = field_info["dtype"]
        meta_dtype.append((field_name, dtype))
        print(f"  {field_name}: {dtype}")

    # Create empty structured array
    meta_data = np.empty(n_faces, dtype=meta_dtype)

    # Read data from template file and populate structured array
    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        for field_name in meta_field_info.keys():
            if field_name not in h5f:
                raise ValueError(
                    f"Field '{field_name}' not found in template file. "
                    f"Available datasets: {list(h5f.keys())}"
                )

            field_data = h5f[field_name][:]
            meta_data[field_name] = field_data
            print(
                f"  Loaded {field_name}: shape={field_data.shape}, dtype={field_data.dtype}"
            )

    # Replace NaNs in meta compound array
    print("\n  Replacing NaNs in meta fields:")
    meta_data = replace_nans_with_fill_value(meta_data)

    print(
        f"\nMeta compound array assembled: shape={meta_data.shape}, {len(meta_dtype)} fields"
    )
    return meta_data


def assemble_attrs_for_meta(meta_field_info, existing_global_attrs):
    """
    Collect attributes from meta fields and format as global H5 attributes.

    Attributes are stored with pattern: meta:{field_name}:{attr_name} = attr_value

    Args:
        meta_field_info: Dict of field names to their info (including attrs)
        existing_global_attrs: Dict of existing global attributes to check for conflicts

    Returns:
        dict: Global attributes to add for meta fields

    Raises:
        ValueError: If attribute name conflicts with existing global attribute
    """
    print("\nAssembling attributes for meta fields:")

    meta_attrs = {}

    for field_name, field_info in meta_field_info.items():
        field_attrs = field_info.get("attrs", {})

        if field_attrs:
            print(f"  {field_name}: {len(field_attrs)} attributes")
            for attr_name, attr_value in field_attrs.items():
                # Format: meta:{field}:{attr}
                global_attr_name = f"meta:{field_name}:{attr_name}"

                # Check for conflicts with existing global attributes
                if global_attr_name in existing_global_attrs:
                    raise ValueError(
                        f"Attribute conflict: '{global_attr_name}' already exists in global attributes. "
                        f"Existing value: {existing_global_attrs[global_attr_name]}, "
                        f"New value: {attr_value}"
                    )

                meta_attrs[global_attr_name] = attr_value
                print(f"    {global_attr_name} = {attr_value}")
        else:
            print(f"  {field_name}: no attributes")

    print(f"Total meta attributes: {len(meta_attrs)}")
    return meta_attrs


def assemble_hsds_meta(template_file, meta_field_info, n_faces, existing_global_attrs):
    """
    Orchestrate meta compound dataset creation and attribute assembly.

    Args:
        template_file: Path to first temporal H5 file
        meta_field_info: Dict of field names to their info
        n_faces: Number of faces
        existing_global_attrs: Dict of existing global attributes

    Returns:
        tuple: (meta_data, meta_attrs)
            - meta_data: numpy structured array for meta dataset
            - meta_attrs: dict of global attributes for meta fields
    """
    print("\n" + "=" * 80)
    print("ASSEMBLING HSDS META DATASET")
    print("=" * 80)

    # Assemble the compound data array
    meta_data = assemble_h5_data_for_meta(template_file, meta_field_info, n_faces)

    # Assemble the attributes
    meta_attrs = assemble_attrs_for_meta(meta_field_info, existing_global_attrs)

    print("=" * 80)
    print("META ASSEMBLY COMPLETE")
    print("=" * 80 + "\n")

    return meta_data, meta_attrs


def create_yearly_file_structure(
    output_file, template_file, file_structure, location_config
):
    """Create the structure of the yearly H5 file with meta compound dataset and full year dimensions"""

    # Calculate expected full-year dimensions (with padding)
    expected_timeline, expected_time_steps = compute_expected_full_year_timeline(
        location_config
    )

    print("\nFull year dimensions:")
    print(f"  Expected timesteps (full year): {expected_time_steps}")
    print(f"  Actual timesteps (from files):  {file_structure['total_time_steps']}")
    if expected_time_steps > file_structure["total_time_steps"]:
        print(
            f"  → Will pad {expected_time_steps - file_structure['total_time_steps']} timesteps with fill values"
        )

    # Store for later use in stitching
    file_structure["expected_time_steps"] = expected_time_steps
    file_structure["expected_timeline"] = expected_timeline

    # First, assemble meta outside of file context
    existing_global_attrs = {}
    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as template_h5:
        existing_global_attrs = dict(template_h5.attrs)

    meta_data, meta_attrs = assemble_hsds_meta(
        template_file,
        file_structure["meta_field_info"],
        file_structure["n_faces"],
        existing_global_attrs,
    )

    with h5py.File(template_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as template_h5:
        with h5py.File(output_file, "w", rdcc_nbytes=HDF5_WRITE_CACHE) as yearly_h5:
            # Copy global attributes from template
            for attr_name, attr_value in template_h5.attrs.items():
                yearly_h5.attrs[attr_name] = attr_value

            # Add meta field attributes as global attributes
            for attr_name, attr_value in meta_attrs.items():
                yearly_h5.attrs[attr_name] = attr_value

            # Create meta compound dataset
            meta_dataset = yearly_h5.create_dataset("meta", data=meta_data)
            # Add _FillValue as global attribute for meta (compound datasets can't have _FillValue directly)
            yearly_h5.attrs["meta:_FillValue"] = NAN_FILL_VALUE
            print(
                f"Created meta dataset: shape={meta_data.shape}, {len(meta_data.dtype.names)} fields"
            )
            print(f"Added global attribute meta:_FillValue={NAN_FILL_VALUE}")

            # Copy sigma coordinates (same for all months)
            if "sigma_layer" in template_h5:
                sigma_layer_data = template_h5["sigma_layer"][:]
                yearly_h5.create_dataset("sigma_layer", data=sigma_layer_data)
                print(f"Copied sigma_layer: {sigma_layer_data.shape}")

            if "sigma_level" in template_h5:
                sigma_level_data = template_h5["sigma_level"][:]
                yearly_h5.create_dataset("sigma_level", data=sigma_level_data)
                print(f"Copied sigma_level: {sigma_level_data.shape}")

            # Create empty time_index with FULL YEAR size
            time_dtype = template_h5["time_index"].dtype
            yearly_h5.create_dataset(
                "time_index",
                shape=(expected_time_steps,),
                dtype=time_dtype,
            )

            # Create empty datasets for all variables with FULL YEAR dimensions
            print("\nCreating yearly datasets (with full year dimensions):")
            for var_name, var_info in file_structure["variable_info"].items():
                # Use full year dimensions instead of actual data dimensions
                actual_yearly_shape = var_info["yearly_shape"]
                is_static = var_info.get("is_static", False)

                # Replace time dimension with expected_time_steps
                if len(actual_yearly_shape) > 1:
                    yearly_shape = (expected_time_steps,) + actual_yearly_shape[1:]
                else:
                    yearly_shape = (expected_time_steps,)

                # Store both shapes for later use
                var_info["full_year_shape"] = yearly_shape

                print(f"\n  Variable: {var_name}")
                print(f"    Shape: {yearly_shape}")
                print(f"    Dtype: {var_info['dtype']}")
                print(f"    Is static: {is_static}")

                # Chunk sizes come from the central chunking strategy
                # (calculate_optimal_chunk_sizes), which sizes chunks per the
                # config chunk_spec. dim names tell it which axis to chunk along
                # (the configured preferred dimension). The NLR spec guarantees
                # 1D/2D variables, so we derive the names from the number of
                # dimensions rather than from the source file.
                num_dims = len(yearly_shape)
                if num_dims == 2:
                    dims = ["time", "face"]
                elif num_dims == 1:
                    dims = ["face"]
                else:
                    raise ValueError(
                        f"Variable '{var_name}' has {num_dims} dimensions "
                        f"(shape {yearly_shape}); the NLR spec only allows "
                        "1D or 2D variables."
                    )

                chunks = calculate_optimal_chunk_sizes(
                    shape=yearly_shape,
                    dims=dims,
                    dtype=var_info["dtype"],
                    config=config,
                )
                print(f"    Chunks: {chunks}")

                # Informational chunk-size estimate
                dtype_size = np.dtype(var_info["dtype"]).itemsize
                chunk_bytes = int(np.prod(chunks)) * dtype_size
                print(
                    f"    Estimated chunk size: {chunk_bytes / (1024**2):.3f} MB "
                    f"({chunk_bytes:,} bytes)"
                )

                print("    Creating dataset...")
                dataset = yearly_h5.create_dataset(
                    var_name,
                    shape=yearly_shape,
                    dtype=var_info["dtype"],
                    chunks=chunks,
                    fillvalue=NAN_FILL_VALUE,  # Gaps automatically filled with NaN
                )
                print("    ✓ Dataset created successfully (gaps pre-filled)")

                # Copy attributes
                for attr_name, attr_value in var_info["attrs"].items():
                    try:
                        dataset.attrs[attr_name] = attr_value
                    except Exception as e:
                        print(f"    Warning: Could not set attribute {attr_name}: {e}")

                # Add _FillValue attribute for numeric datasets
                add_fill_value_attr(dataset)


def _face_block_width(expected_time_steps, itemsize, chunk_faces, n_faces, budget_bytes):
    """
    Largest face-block width (number of faces) whose full-time accumulator fits
    the RAM budget, aligned DOWN to the dataset's face-chunk boundary so every
    block write covers complete chunks. Always at least one chunk wide.
    """
    raw = max(1, budget_bytes // (expected_time_steps * itemsize))
    width = (raw // chunk_faces) * chunk_faces
    width = max(chunk_faces, width)
    return int(min(width, n_faces))


def _stitch_2d_variable_blockstreamed(
    dataset, var_name, per_file_maps, expected_time_steps, budget_bytes
):
    """
    Stream a 2D (time, face) variable into the yearly dataset one face-block at a
    time: assemble the variable's COMPLETE time axis for a contiguous range of
    faces in memory, then write that block in a single assignment.

    Because each block is aligned to the face-chunk boundary and spans all
    timesteps, every HDF5 chunk is written exactly once, in full -- no
    read-modify-write. Peak memory is one block accumulator (~budget). Each input
    file is read once per block, so total read volume is 1x. Returns block count.
    """
    n_faces = dataset.shape[1]
    dtype = dataset.dtype
    itemsize = np.dtype(dtype).itemsize
    chunk_faces = dataset.chunks[1] if dataset.chunks else n_faces

    block_width = _face_block_width(
        expected_time_steps, itemsize, chunk_faces, n_faces, budget_bytes
    )
    block_starts = list(range(0, n_faces, block_width))
    acc_gb = expected_time_steps * block_width * itemsize / (1024**3)
    print(
        f"      {len(block_starts)} face-block(s) of up to {block_width} faces "
        f"(~{acc_gb:.1f} GB/block)"
    )

    for bi, f0 in enumerate(block_starts, start=1):
        f1 = min(f0 + block_width, n_faces)
        # Full time axis for this face range; gap timesteps stay at fill value.
        acc = np.full((expected_time_steps, f1 - f0), NAN_FILL_VALUE, dtype=dtype)
        for monthly_file, local_actual, expected_idx in per_file_maps:
            with h5py.File(monthly_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as mh5:
                data = mh5[var_name][:, f0:f1]
            replace_nans_inplace(data)
            acc[expected_idx, :] = data[local_actual, :]
        # One contiguous, complete-chunk write for the whole block.
        dataset[:, f0:f1] = acc
        print(f"        block {bi}/{len(block_starts)}: faces {f0}:{f1} written")
        del acc
        gc.collect()

    return len(block_starts)


def _stitch_1d_time_variable(dataset, var_name, per_file_maps, expected_time_steps):
    """Assemble a 1D (time,) variable across all files in RAM and write once."""
    acc = np.full((expected_time_steps,), NAN_FILL_VALUE, dtype=dataset.dtype)
    for monthly_file, local_actual, expected_idx in per_file_maps:
        with h5py.File(monthly_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as mh5:
            data = mh5[var_name][:]
        replace_nans_inplace(data)
        acc[expected_idx] = data[local_actual]
    dataset[:] = acc


def stitch_data_into_yearly_file(
    output_file, monthly_files, file_structure, memory_budget_gb=DEFAULT_MEMORY_BUDGET_GB
):
    """
    Stitch data from monthly files into the yearly file, one variable at a time,
    streaming each 2D (time, face) variable in contiguous FACE-BLOCKS.

    Why face-blocks instead of month-by-month: the yearly datasets are chunked by
    face with each chunk spanning the whole time axis. A month-at-a-time write
    puts only ~240 of a chunk's ~17,520 timesteps, forcing HDF5 to read-modify-
    write every chunk on every month (~70x write amplification). Instead we
    assemble a variable's complete time axis for a range of faces in memory and
    write it once, so each chunk is written exactly once, in full.

    Peak memory is bounded by memory_budget_gb (one block accumulator),
    independent of mesh size. Each input file's data is read once (partitioned
    across blocks). Gaps remain at NAN_FILL_VALUE.
    """
    expected_timeline = file_structure["expected_timeline"]
    expected_time_steps = file_structure["expected_time_steps"]

    # ---- First pass: per-file timestamps + step counts -> timestamp mapping ----
    print("\nBuilding timestamp mapping...")
    actual_timestamps = []
    file_time_counts = []
    for monthly_file in monthly_files:
        with h5py.File(monthly_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as monthly_h5:
            time_data = monthly_h5["time_index"][:]
            if time_data.dtype.kind == "S":
                time_strings = [t.decode("utf-8") for t in time_data]
            else:
                time_strings = [str(t) for t in time_data]
            actual_timestamps.extend(pd.to_datetime(time_strings))
            file_time_counts.append(len(time_data))

    actual_timestamps = pd.DatetimeIndex(actual_timestamps)
    print(f"  Collected {len(actual_timestamps)} actual timestamps")

    expected_df = pd.DataFrame(
        {"timestamp": expected_timeline, "expected_idx": range(expected_time_steps)}
    )
    actual_df = pd.DataFrame(
        {"timestamp": actual_timestamps, "actual_idx": range(len(actual_timestamps))}
    )
    merged_df = expected_df.merge(actual_df, on="timestamp", how="left")

    has_data_mask = ~merged_df["actual_idx"].isna()
    expected_indices_with_data = merged_df[has_data_mask]["expected_idx"].values
    actual_indices_for_data = merged_df[has_data_mask]["actual_idx"].values.astype(int)

    gap_count = int((~has_data_mask).sum())
    print(f"  Mapped {len(expected_indices_with_data)} timesteps with data")
    if gap_count > 0:
        print(f"  Found {gap_count} gaps (pre-filled with {NAN_FILL_VALUE})")

    # ---- Precompute, per file, (local rows -> expected year rows) ----
    # Each input file contributes a contiguous block of global actual indices;
    # convert those to file-local row indices plus the matching year indices.
    # Pure-gap files (no mapped data) are skipped.
    per_file_maps = []  # list of (path, local_actual_idx, expected_idx)
    actual_offset = 0
    for monthly_file, n_steps in zip(monthly_files, file_time_counts):
        file_mask = (actual_indices_for_data >= actual_offset) & (
            actual_indices_for_data < actual_offset + n_steps
        )
        if file_mask.any():
            local_actual = actual_indices_for_data[file_mask] - actual_offset
            expected_idx = expected_indices_with_data[file_mask]
            per_file_maps.append((monthly_file, local_actual, expected_idx))
        actual_offset += n_steps

    variable_info = file_structure["variable_info"]

    # ---- Size the write cache from the REAL chunk size (see module header) ----
    chunk_bytes = representative_chunk_bytes(file_structure)
    write_rdcc_nbytes = STITCH_WRITE_RDCC_TARGET_CHUNKS * chunk_bytes
    budget_bytes = int(memory_budget_gb * (1024**3))
    print(
        f"\nWrite cache: {write_rdcc_nbytes / (1024**2):.0f} MB "
        f"({STITCH_WRITE_RDCC_TARGET_CHUNKS} chunks x {chunk_bytes / (1024**2):.2f} MB), "
        f"nslots={STITCH_WRITE_RDCC_NSLOTS}, w0={STITCH_WRITE_RDCC_W0}"
    )

    with h5py.File(
        output_file,
        "a",
        rdcc_nbytes=write_rdcc_nbytes,
        rdcc_nslots=STITCH_WRITE_RDCC_NSLOTS,
        rdcc_w0=STITCH_WRITE_RDCC_W0,
    ) as yearly_h5:
        # Write the complete expected timeline to time_index.
        print("\nWriting full year timeline to time_index...")
        expected_time_strings = expected_timeline.strftime(
            "%Y-%m-%d %H:%M:%S+00:00"
        ).tolist()
        yearly_h5["time_index"][:] = expected_time_strings
        print(f"  ✓ Wrote {expected_time_steps} timestamps")

        print("\nStitching data (per-variable, face-block streaming)...")
        print(f"Processing {len(variable_info)} variables")
        print(f"Memory budget per block: {memory_budget_gb:.1f} GB")

        n_vars = len(variable_info)
        for vi, var_name in enumerate(variable_info.keys(), start=1):
            dataset = yearly_h5[var_name]
            print(f"  [{vi}/{n_vars}] {var_name} (shape {dataset.shape})")

            if dataset.ndim == 1:
                _stitch_1d_time_variable(
                    dataset, var_name, per_file_maps, expected_time_steps
                )
            else:
                _stitch_2d_variable_blockstreamed(
                    dataset,
                    var_name,
                    per_file_maps,
                    expected_time_steps,
                    budget_bytes,
                )

        yearly_h5.flush()

    print(
        f"\n✓ Stitching complete: {len(expected_indices_with_data)} timesteps written, {gap_count} gaps filled"
    )


def validate_yearly_file(output_file, file_structure):
    """Perform final validation on the yearly file"""
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as yearly_h5:
        # Check time index continuity
        time_index = yearly_h5["time_index"][:]

        # Time index is now string format: 'YYYY-MM-DD HH:MM:SS+00:00'
        # Decode bytes to strings and parse
        time_strings = [
            t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in time_index
        ]
        dt_timestamps = pd.to_datetime(time_strings)

        # Timestamps must be monotonic
        is_monotonic = dt_timestamps.is_monotonic_increasing
        if not is_monotonic:
            raise ValueError("Timestamps are not monotonic")

        # Check data completeness - should match full year dimensions
        expected_time_steps = file_structure["expected_time_steps"]
        actual_steps = len(yearly_h5["time_index"])
        if actual_steps != expected_time_steps:
            raise ValueError(
                f"Time step mismatch. Expected (full year): {expected_time_steps}, Got: {actual_steps}"
            )

        # Verify no gaps in timeline (all timestamps should be present)
        expected_timeline = file_structure["expected_timeline"]
        if not dt_timestamps.equals(expected_timeline):
            raise ValueError("Timeline does not match expected full-year timeline")

        print(
            f"✓ Validation complete: {actual_steps} timesteps, full year coverage, monotonically increasing"
        )


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Stitch temporal HSDS files into yearly file"
    )
    parser.add_argument("--location", type=str, required=True, help="Location name")
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Directory with temporal H5 files (default from config)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output yearly H5 file path (default from config)",
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip validation checks"
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=DEFAULT_MEMORY_BUDGET_GB,
        help=(
            "RAM budget (GB) for the single in-memory face-block accumulator. "
            "Larger = fewer, bigger blocks (fewer file opens); smaller = lower "
            f"peak memory. Default: {DEFAULT_MEMORY_BUDGET_GB:g}."
        ),
    )

    args = parser.parse_args()

    # Get location config
    location_config = config["location_specification"][args.location]

    # Determine paths using file_manager
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = get_hsds_temp_dir(config, location_config)

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = get_hsds_final_file_path(config, location_config)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing output file if it exists to start with a clean slate
    if output_file.exists():
        print(f"Removing existing output file: {output_file}")
        output_file.unlink()

    print(f"Stitching temporal files from {temp_dir}")
    print(f"Output: {output_file}")

    # Perform stitching
    stitch_single_b1_file_for_hsds(
        monthly_dir=temp_dir,
        output_file=output_file,
        location_name=args.location,
        location_config=location_config,
        perform_checks=not args.skip_checks,
        memory_budget_gb=args.memory_budget_gb,
    )

    print(f"Stitching complete! Final file: {output_file}")

    # Print final file info
    with h5py.File(output_file, "r", rdcc_nbytes=HDF5_READ_CACHE) as h5f:
        print(f"Final file size: {output_file.stat().st_size / (1024**3):.2f} GB")
        print(
            f"Variables: {[k for k in h5f.keys() if k not in ['meta', 'time_index']]}"
        )
        print(f"Time steps: {len(h5f['time_index'])}")


if __name__ == "__main__":
    main()
