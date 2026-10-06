"""Pipeline step worker: partition (a1_std -> a2_std_partition).

Reads the standardize-step tracking parquet and repartitions the standardized
dataset by time (per the location's ``partition_frequency``) into a2_std_partition.

Usage:
    python run_partition.py <location>
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scalar.*")

from pathlib import Path

import pandas as pd

from tidal_fvcom.config import config
from tidal_fvcom.cli import location_arg_parser
from tidal_fvcom.file_manager import get_tracking_output_dir
from tidal_fvcom.partition_by_time import partition_by_time


def main():
    args = location_arg_parser(
        config, "Partition a standardized FVCOM dataset by time (a2_std_partition)."
    ).parse_args()
    location = config["location_specification"][args.location]

    tracking_path = Path(
        get_tracking_output_dir(config, location),
        f"{location['output_name']}_standardize_step_tracking.parquet",
    )
    if not tracking_path.exists():
        raise FileNotFoundError(
            f"[partition] Missing standardize tracking file: {tracking_path}. "
            "Run the standardize step first (python run_standardize.py <location>)."
        )

    print(f"[partition] Loading standardize tracking: {tracking_path}")
    valid_std_files_df = pd.read_parquet(tracking_path)

    print("[partition] Partitioning standardized dataset by time...")
    partition_by_time(config, args.location, valid_std_files_df, force_reprocess=True)
    print("[partition] Done.")


if __name__ == "__main__":
    main()
