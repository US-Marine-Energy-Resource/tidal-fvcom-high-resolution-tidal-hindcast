"""Command line argument helpers for the FVCOM data processing workflow.

Provides three parsers used across the entry points:
    - ``location_arg_parser`` — a validated ``location`` positional (extend it with
      extra args); used by the per-step ``run_<step>.py`` workers.
    - ``parse_args`` — location + ``--output-type`` (used by older scripts).
    - ``parse_partition_args`` — location + batch/product options for the array
      workers (``partition_dataset.py``, ``summarize_dataset.py``).

The unified workflow orchestrator is ``run.py``; see ``python run.py --list-steps``.

Example:
    parser = location_arg_parser(config, "Run one step.")
    args = parser.parse_args()
    location_config = config["location_specification"][args.location]
"""

import argparse


def validate_location(config):
    """Create a validation function closure with access to config."""

    def _validate_location(location):
        if location not in config["location_specification"]:
            valid_locations = list(config["location_specification"].keys())
            raise argparse.ArgumentTypeError(
                f"Invalid location: {location}. Must be one of: {', '.join(valid_locations)}"
            )
        return location

    return _validate_location


def validate_output_type(output_type):
    """Validate output type and return standardized value."""
    valid_types = ["summary", "std", "all"]
    output_type = output_type.lower()
    if output_type not in valid_types:
        raise argparse.ArgumentTypeError(
            f"Invalid output type: {output_type}. Must be one of: {', '.join(valid_types)}"
        )
    return output_type


def parse_args(config):
    """Parse command line arguments using provided config."""
    parser = argparse.ArgumentParser(
        description="Process FVCOM data for specified location and output type."
    )

    parser.add_argument(
        "location",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )

    parser.add_argument(
        "--output-type",
        type=validate_output_type,
        default="all",
        help="Output type to generate (summary, std, or all). Defaults to all.",
    )

    return parser.parse_args()


def location_arg_parser(config, description):
    """Return an ArgumentParser with a validated ``location`` positional.

    Shared by the per-step ``run_<step>.py`` workers. Callers may add extra
    arguments to the returned parser before calling ``parse_args()``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "location",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )
    return parser


def parse_partition_args(config):
    """Parse command line arguments using provided config."""
    parser = argparse.ArgumentParser(
        description="Process FVCOM data for specified location and output type."
    )
    parser.add_argument(
        "location",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )
    # Add batch size argument
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to process in each batch (default: 100)",
    )
    # Add batch number argument
    parser.add_argument(
        "--batch-num", type=int, default=1, help="Batch number to process (default: 1)"
    )
    # Add skip existing argument
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip processing faces whose output files already exist",
    )
    # Which summary product(s) to compute (used by summarize_dataset.py; ignored
    # by other batch workers). monthly -> b2, yearly -> b3, both -> b2 and b3.
    parser.add_argument(
        "--product",
        choices=["monthly", "yearly", "both"],
        default="both",
        help="Summary product(s) to compute: monthly (b2), yearly (b3), or both",
    )

    return parser.parse_args()
