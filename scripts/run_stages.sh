#!/bin/bash
#SBATCH --job-name=wc_driver
#SBATCH --account=bisc019342
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:10:00
#SBATCH --output=logs/driver_%j.out
#SBATCH --error=logs/driver_%j.err
#
# run_stages.sh -- submit a staged optimisation as a chain of dependent jobs.
#
#   sbatch run_stages.sh basic            submit rounds 1-4
#   sbatch run_stages.sh basic 4          resume at round 4
#
# The optional second argument is the round to start from. Resuming needs the
# previous round's JSON to exist already: run the best stage for that round
# first, with
#
#   sbatch --export=ALL,MODEL=basic,ROUND=3 submit_stage.sh best
#
# This is a driver, not a computation: it calls sbatch eight times and exits in
# about a second. It is written as a job so that nothing has to be run by hand
# on a login node. Eight jobs go into the queue at once, each waiting on the one
# before, so the whole schedule can be submitted and left alone.
#
# Round 1 explores widely from the model's P0. Each later round perturbs less
# around the best point found so far, which concentrates the search where it is
# worth searching. The tolerances also tighten, because a loose tolerance is
# only acceptable while the aim is still to find the right basin.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

# Must match wildcat_pipeline.py, which reads the same variable. Exported so
# that every job in the chain inherits it.
export WILDCAT_OUTDIR="${WILDCAT_OUTDIR:-results_wild}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not on PATH on this node, so the driver cannot submit the" >&2
    echo "chain from inside a job. Run it on a login node instead:" >&2
    echo "    bash run_stages.sh ${1:-basic}" >&2
    exit 1
fi

MODEL="${1:-basic}"
START="${2:-1}"
NTASK="${NTASK:-50}"

# Which limits actually bite depends on the optimiser, and the optimiser is
# chosen by whether the model carries the TA > max(TB, TD) constraint:
#
#   sec_contact       unconstrained -> optimize_log_fmin -> honours ONLY --maxiter
#   basic, growth     constrained   -> nlopt COBYLA      -> honours ONLY --maxtime
#                                                           and --maxeval
#
# Every argument is passed to every model regardless, and each optimiser ignores
# what it does not honour, so getting this wrong is silent: the run completes,
# just not under the limits you thought you had set. growth is CONSTRAINED --
# _dennis_spec defaults to constrain_times=True -- so it belongs with basic.
if [ "${MODEL}" = "sec_contact" ]; then
    WALL=6:00:00
    ITERS=(800 2000 3000 4000)
else
    WALL=6:00:00
    ITERS=(300 300 300 300)
fi

FOLDS=(2 1 0.5 0.25)
FTOLS=(1e-2 1e-3 1e-4 1e-4)
SEEDS=(2000 3000 4000 5000)

# run_one_restart runs TWO rounds per restart, each capped at MAXTIME, so the
# per-task ceiling is 2*MAXTIME plus startup. That has to sit well inside WALL:
# at 10800 the arithmetic came to exactly 6 h against a 6 h wall clock, and a
# task killed on the wall clock writes no pickle at all, which is the outcome
# maxtime exists to prevent. 7200 leaves two hours of headroom.
#
# These are also generous now: the wild-only spectrum needs a [42, 52, 62] grid
# rather than [62, 72, 82], and integration cost climbs steeply with grid size.
MAXTIMES=(5400 5400 7200 7200)
MAXEVALS=(20000 20000 40000 40000)

mkdir -p logs "${WILDCAT_OUTDIR}"

echo "model=${MODEL}  tasks/round=${NTASK}  walltime/task=${WALL}"
echo "outdir=${WILDCAT_OUTDIR}"
dep=""

for i in 0 1 2 3; do
    r=$((i + 1))
    if [ ${r} -lt ${START} ]; then
        continue
    fi
    start=""
    if [ ${i} -gt 0 ]; then
        start="${WILDCAT_OUTDIR}/best_${MODEL}_r${i}.json"
        if [ ${r} -eq ${START} ] && [ ! -f "${start}" ]; then
            echo "cannot resume at round ${r}: ${start} does not exist." >&2
            echo "run the best stage for round ${i} first:" >&2
            echo "  sbatch --export=ALL,MODEL=${MODEL},ROUND=${i} submit_stage.sh best" >&2
            exit 1
        fi
    fi

    fit=$(MODEL="${MODEL}" ROUND="${r}" SEEDBASE="${SEEDS[$i]}" \
          FOLD="${FOLDS[$i]}" FTOL="${FTOLS[$i]}" MAXITER="${ITERS[$i]}" \
          MAXTIME="${MAXTIMES[$i]}" MAXEVAL="${MAXEVALS[$i]}" STARTFROM="${start}" \
          sbatch --parsable ${dep} \
                 --job-name="wc_${MODEL}_r${r}" \
                 --array=1-${NTASK}%${NTASK} \
                 --time=${WALL} \
                 submit_stage.sh fit)
    echo "  round ${r} fit  : ${fit}  (fold=${FOLDS[$i]} maxiter=${ITERS[$i]} maxtime=${MAXTIMES[$i]})"

    # afterany, not afterok: the point of staging is that the round's best point
    # is worth having even when some restarts failed.
    bst=$(MODEL="${MODEL}" ROUND="${r}" \
          sbatch --parsable --dependency=afterany:${fit} \
                 --job-name="wc_${MODEL}_r${r}_best" \
                 --time=0:20:00 \
                 submit_stage.sh best)
    echo "  round ${r} best : ${bst}"

    # afterok here, though: without the JSON the next round has nothing to
    # perturb around and would silently restart from P0.
    dep="--dependency=afterok:${bst}"
done

echo
echo "submitted. watch with:  squeue -u ${USER}"
echo "final answer will be:   ${WILDCAT_OUTDIR}/best_${MODEL}_r4.json"