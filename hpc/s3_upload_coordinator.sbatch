#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%j.out
#SBATCH --job-name=s3_upload_coordinator
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov
#SBATCH --time=4:00:00
#SBATCH --partition=standard

# Upload coordinator for the unified workflow's `upload` step.
#
# The orchestrator (run.py) submits this gated `--dependency=afterok` on the
# vap_data_products and summary terminals, so it runs only after the publishable
# data products exist. From inside the job it builds the S3 manifest(s) and
# dispatches the upload array(s) via dispatch_s3_upload_v2.py (sbatch-from-within-
# a-job, the same pattern summarize_retry_coordinator.sbatch uses).
#
#   LOCATION     config location key (e.g. cook_inlet)
#   DATA_LEVELS  colon-separated data level dir names (run.py joins them with ':'
#                to avoid the comma --export uses as its own separator)

LOCATION=${LOCATION:?LOCATION must be set}
DATA_LEVELS=${DATA_LEVELS:?DATA_LEVELS must be set}

# Convert the ':'-delimited list back into space-separated args.
DATA_LEVELS="${DATA_LEVELS//:/ }"

source hpc/sbatch_env_setup.sh

echo "Dispatching S3 upload for $LOCATION (levels: $DATA_LEVELS)..."

python -m tidal_fvcom.steps.dispatch_s3_upload_v2 $LOCATION --data-levels $DATA_LEVELS --skip-if-uploaded

echo "S3 upload dispatch for $LOCATION complete."
