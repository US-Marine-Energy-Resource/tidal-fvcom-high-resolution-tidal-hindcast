"""Pipeline step worker: standardize (raw -> a1_std).

Reads the verify-step tracking parquet and writes standardized NetCDF (a1_std)
plus the standardize-step tracking parquet that ``run_partition.py`` consumes.

Usage:
    python run_standardize.py <location>
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scalar.*")

from pathlib import Path

import pandas as pd

from config import config
from src.cli import location_arg_parser
from src.file_manager import get_tracking_output_dir
from src.standardize import standardize_dataset


def main():
    args = location_arg_parser(
        config, "Standardize a verified FVCOM dataset (a1_std)."
    ).parse_args()
    location = config["location_specification"][args.location]

    tracking_path = Path(
        get_tracking_output_dir(config, location),
        f"{location['output_name']}_verify_step_tracking.parquet",
    )
    if not tracking_path.exists():
        raise FileNotFoundError(
            f"[standardize] Missing verify tracking file: {tracking_path}. "
            "Run the verify step first (python run_verify.py <location>)."
        )

    print(f"[standardize] Loading verify tracking: {tracking_path}")
    valid_timestamps_df = pd.read_parquet(tracking_path)

    print("[standardize] Standardizing dataset...")
    standardize_dataset(config, args.location, valid_timestamps_df, skip_if_verified=False)
    print("[standardize] Done.")


if __name__ == "__main__":
    main()
