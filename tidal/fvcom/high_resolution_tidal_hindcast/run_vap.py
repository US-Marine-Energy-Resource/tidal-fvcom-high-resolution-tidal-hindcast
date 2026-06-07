"""Pipeline step worker: vap (a2_std_partition -> b1_vap).

Computes face-centered precalculations, then derives value-added products
(speed, direction, power density, ...) for each time-partitioned file.

Usage:
    python run_vap.py <location>
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*scalar.*")

from config import config
from src.cli import location_arg_parser
from src.derive_vap_fvcom import (
    calculate_and_save_face_center_precalculations,
    derive_vap,
)


def main():
    args = location_arg_parser(
        config, "Derive value-added products from partitioned FVCOM data (b1_vap)."
    ).parse_args()

    print("[vap] Calculating face-center precalculations...")
    calculate_and_save_face_center_precalculations(
        config, args.location, skip_if_precalculated=True
    )

    print("[vap] Deriving value-added products...")
    derive_vap(config, args.location, skip_if_output_files_exist=False)
    print("[vap] Done.")


if __name__ == "__main__":
    main()
