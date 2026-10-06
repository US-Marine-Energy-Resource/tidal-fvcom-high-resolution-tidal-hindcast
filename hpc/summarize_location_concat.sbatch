#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov
#SBATCH --time=24:00:00
#SBATCH --partition=standard

# Get parameters from environment
LOCATION=${LOCATION:-cook_inlet}

# Source shared environment setup
source hpc/sbatch_env_setup.sh

echo "Starting concatenation for $LOCATION..."

# Run your concatenation script here
python -m tidal_fvcom.steps.concat_summarized_datasets $LOCATION --batch-num 0

echo "Concatenation for $LOCATION completed successfully."
