#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=stitch_hsds_%j.out
#SBATCH --job-name=stitch_hsds
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov
#SBATCH --partition=standard
#SBATCH --time=47:00:00
##SBATCH --partition=debug
##SBATCH --time=59

# Prepare for python script execution
source hpc/sbatch_env_setup.sh

# Run the stitching script
python -m tidal_fvcom.steps.stitch_individual_hsds_tidal_data_into_single_file --location ${LOCATION}
