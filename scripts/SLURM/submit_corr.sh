#!/bin/bash
#SBATCH --job-name=wildcat_corr
#SBATCH --account=bisc019342
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=logs/%x_%j.out

# Usage:  mkdir -p logs && sbatch submit_corr.sh
#         sbatch submit_corr.sh basic
#
# Writes results_wild/corr_<model>.pdf, the parameter correlation matrix from
# the Godambe H and J already stored in results_wild/claic_<model>.pkl by
# `wildcat_pipeline.py report`. Nothing is refitted and no bootstrap gradient is
# recomputed, so this takes seconds; it needs a job only because the login
# nodes are not for running things.
#
# The printed columns are a check: se/value must reproduce the stored se_log
# behind the published intervals. If they disagree, the pickle does not match
# the fit in the tables.

set -euo pipefail

export OMP_NUM_THREADS=1
export WILDCAT_OUTDIR="${WILDCAT_OUTDIR:-results_wild}"

set +u
source /user/work/gb22703/miniforge3/etc/profile.d/conda.sh
conda activate dadi
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python plot_corr.py "$@"