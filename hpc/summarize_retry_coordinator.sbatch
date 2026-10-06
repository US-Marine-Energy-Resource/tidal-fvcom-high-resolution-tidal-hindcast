#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem=8G

# Retry coordinator job
# Checks for missing batch files and submits retries to standard partition
# Uses --wait so this job blocks until all retries complete

# Get parameters from environment
LOCATION=${LOCATION:-cook_inlet}

# Source shared environment setup
source hpc/sbatch_env_setup.sh

# Run the coordinator script
python -m tidal_fvcom.steps.summarize_retry_coordinator $LOCATION

exit_code=$?

echo "Summarize retry completed with exit code $exit_code"

exit $exit_code
