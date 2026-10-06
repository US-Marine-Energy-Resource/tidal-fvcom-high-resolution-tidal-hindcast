"""Pipeline step worker: summary parquet products (b3 -> b4 / b5).

Converts the yearly-mean NetCDF summary (b3) into the tabular summary parquet (b4)
and, optionally, the combined atlas geoparquet (b5). The upstream monthly/yearly
means (b2/b3) are produced separately by the summarize array sub-chain.

  --product summary_parquet  -> b4 only   (create_combined_atlas_output=False)
  --product atlas            -> b4 + b5   (create_combined_atlas_output=True)
  --product both (default)   -> b4 + b5

Usage:
    python run_summary.py <location> [--product {summary_parquet,atlas,both}]
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scalar.*")

from tidal_fvcom.config import config
from tidal_fvcom.cli import location_arg_parser
from tidal_fvcom.vap_create_parquet_summary import convert_nc_summary_to_parquet


def main():
    parser = location_arg_parser(
        config, "Create summary (b4) and atlas (b5) parquet products from b3."
    )
    parser.add_argument(
        "--product",
        choices=["summary_parquet", "atlas", "both"],
        default="both",
        help="Which parquet product(s) to create (default: both).",
    )
    args = parser.parse_args()

    # b5 (atlas) is built from b4, so any product that includes the atlas also
    # regenerates b4 in the same pass.
    create_atlas = args.product in ("atlas", "both")

    print(f"[summary] Creating parquet products (product={args.product})...")
    convert_nc_summary_to_parquet(
        config, args.location, create_combined_atlas_output=create_atlas
    )
    print("[summary] Done.")


if __name__ == "__main__":
    main()
