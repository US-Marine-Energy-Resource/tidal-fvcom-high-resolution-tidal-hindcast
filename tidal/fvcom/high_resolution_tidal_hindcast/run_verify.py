"""Pipeline step worker: verify (raw -> tracking).

Verifies raw FVCOM outputs (time continuity, coordinate system, attributes) and
writes the verify-step tracking parquet that ``run_standardize.py`` consumes.

Usage:
    python run_verify.py <location>
"""

import warnings

# Suppress pyproj/numpy deprecation spam before importing the processing stack.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scalar.*")

from config import config
from src.cli import location_arg_parser
from src.file_manager import get_specified_nc_files
from src.verify import verify_dataset


def main():
    args = location_arg_parser(config, "Verify raw FVCOM dataset integrity.").parse_args()
    location = config["location_specification"][args.location]

    print(f"[verify] Finding raw .nc files for {args.location}...")
    nc_files = get_specified_nc_files(config, location)
    print(f"[verify] Found {len(nc_files)} files.")

    print("[verify] Verifying dataset integrity...")
    verify_dataset(config, location, nc_files, skip_if_verified=False)
    print("[verify] Done.")


if __name__ == "__main__":
    main()
