from tidal_fvcom.config import config
from tidal_fvcom.cli import parse_partition_args

from tidal_fvcom.calculate_vap_average import (
    calculate_vap_yearly_average,
    calculate_vap_monthly_average,
)


if __name__ == "__main__":
    args = parse_partition_args(config)

    # Access the location config
    location = config["location_specification"][args.location]

    batch_size = args.batch_size
    batch_number = args.batch_num
    product = args.product

    print(
        f"Create VAP Summary Dataset ({product}) for {args.location} for batch "
        f"{batch_number} of size {batch_size}..."
    )

    if product in ("yearly", "both"):
        calculate_vap_yearly_average(
            config, args.location, batch_size=batch_size, batch_number=batch_number
        )

    if product in ("monthly", "both"):
        calculate_vap_monthly_average(
            config, args.location, batch_size=batch_size, batch_number=batch_number
        )
