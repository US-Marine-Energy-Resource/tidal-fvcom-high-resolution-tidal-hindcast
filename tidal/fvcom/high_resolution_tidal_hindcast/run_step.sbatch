#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov

# Generic single-node wrapper for the unified workflow's in-process steps.
# The orchestrator (run.py) submits this with --export setting STEP/LOCATION and
# passes --job-name/--time/--partition on the sbatch CLI.
#
#   STEP      one of: verify, standardize, partition, vap, summary
#   LOCATION  config location key (e.g. cook_inlet)
#   PRODUCT   optional, forwarded to run_summary.py (summary_parquet|atlas|both)

LOCATION=${LOCATION:-cook_inlet}
STEP=${STEP:?STEP must be set (verify|standardize|partition|vap|summary)}
PRODUCT=${PRODUCT:-}

# Prepare for python script execution
source sbatch_env_setup.sh

echo "Running workflow step '$STEP' for $LOCATION..."

if [ "$STEP" = "summary" ] && [ -n "$PRODUCT" ]; then
    python run_${STEP}.py $LOCATION --product $PRODUCT
else
    python run_${STEP}.py $LOCATION
fi

echo "Workflow step '$STEP' for $LOCATION completed."
