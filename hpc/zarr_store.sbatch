#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov

# Zarr store build. run.py submits this three times per location:
#   PHASE=init      single job
#   PHASE=write     array job, one task per face block
#   PHASE=finalize  single job
# Partition, memory and time come from the sbatch CLI (config["zarr"]["slurm"]).

LOCATION=${LOCATION:?LOCATION must be set}
PHASE=${PHASE:?PHASE must be set (init|write|finalize)}

source hpc/sbatch_env_setup.sh

if [ "$PHASE" = "write" ]; then
    python -m tidal_fvcom.steps.build_zarr_store $LOCATION --phase write --block ${SLURM_ARRAY_TASK_ID:?}
else
    python -m tidal_fvcom.steps.build_zarr_store $LOCATION --phase $PHASE
fi
