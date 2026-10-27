#!/bin/bash
# Shared environment setup sourced by every hpc/*.sbatch job.
#
# Jobs must be submitted from the repository root (run.py does this), so
# SLURM_SUBMIT_DIR is the repo root and `python -m tidal_fvcom...` resolves.
#
# Kestrel RHEL9 notes:
#   source conda.sh so conda activate works in a batch shell
#   deactivate twice, the module can auto-activate base
#   activate by full path, not by name

TIDAL_FVCOM_CONDA_ENV="${TIDAL_FVCOM_CONDA_ENV:-/home/asimms/.conda-envs/tidal_fvcom}"

echo "Initializing tidal hindcast sbatch environment on $(hostname)..."

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda deactivate || true
conda activate "${TIDAL_FVCOM_CONDA_ENV}"

# Run from the repository root
cd "${SLURM_SUBMIT_DIR}"

# Make the tidal_fvcom package importable from scripts outside it (e.g. atlas/)
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Force Python to flush print statements immediately
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Environment setup complete."
