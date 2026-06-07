#!/bin/bash
#SBATCH --account=hindcastra
#SBATCH --output=%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrew.simms@nlr.gov
#SBATCH --time=12:00:00
#SBATCH --partition=standard

# Array worker for the point-parquet sub-product of vap_data_products:
# partitions b1_vap into the per-point time-series parquet dataset
# (b1_vap_by_point_partition). The orchestrator (run.py) sets --array and
# exports LOCATION/BATCH_SIZE.
#
#   LOCATION    config location key (e.g. cook_inlet)
#   BATCH_SIZE  faces per batch (array index = batch number)

LOCATION=${LOCATION:-cook_inlet}
BATCH_SIZE=${BATCH_SIZE:-10000}

source sbatch_env_setup.sh

echo "Point-parquet batch $SLURM_ARRAY_TASK_ID for $LOCATION (batch size $BATCH_SIZE)..."

python partition_dataset.py $LOCATION --batch-size $BATCH_SIZE --batch-num $SLURM_ARRAY_TASK_ID --skip-existing

echo "Point-parquet batch $SLURM_ARRAY_TASK_ID for $LOCATION completed."
