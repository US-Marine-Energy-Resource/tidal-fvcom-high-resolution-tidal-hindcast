#!/bin/bash
# Shared environment setup sourced by every hpc/*.sbatch job.
#
# Jobs must be submitted from the repository root (run.py does this), so
# SLURM_SUBMIT_DIR is the repo root and `python -m tidal_fvcom...` resolves.

echo "Initializing tidal hindcast sbatch environment..."

# Load necessary modules
module load conda/2024.06.1
conda deactivate
conda activate tidal_fvcom

# Run from the repository root
cd "${SLURM_SUBMIT_DIR}"

# Make the tidal_fvcom package importable from scripts outside it (e.g. atlas/)
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

echo "Environment setup complete."
