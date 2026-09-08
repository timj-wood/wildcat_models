#!/bin/bash
#SBATCH --job-name=wildcat_sfs
#SBATCH --account=bisc019342
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out

# Usage:  mkdir -p logs && sbatch submit_sfs.sh
#
# SLURM opens the --output file before this script runs, so logs/ must already
# exist at submission time. The mkdir below only covers later runs.
#
# Parses the MSMC files, writes results_wild/data.fs (33 x 13, folded: 16
# wild-caught Scottish x 6 domestic) and 100 block bootstrap replicates under
# results_wild/boots/, plus results_wild/meta.json.
#
# Run once, before any fitting. About 5 minutes and well under 8G; the memory
# goes on the data dictionary, which holds all 145,512 biallelic sites at once
# so that fragment_data_dict can split it into blocks.

set -euo pipefail

export OMP_NUM_THREADS=1

# conda's activation scripts are not reliably safe under `set -u`.
set +u
source /user/work/gb22703/miniforge3/etc/profile.d/conda.sh
conda activate dadi
set -u

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python wildcat_pipeline.py sfs