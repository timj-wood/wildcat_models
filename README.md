# Wildcat demographic inference with dadi

Fits and compares demographic models for Scottish wildcat (*Felis silvestris*)
and domestic cat (*F. catus*) from a folded joint site frequency spectrum, using
[dadi](https://dadi.readthedocs.io) 2.4.4. Model comparison uses CLAIC rather than AIC,
since the likelihood is composite over linked sites and overstates its own information
content.

Summer research internship, University of Bristol.

## Contents

| Item | Purpose |
|---|---|
| `pipeline/wildcat_pipeline.py` | Main CLI: `sfs`, `fit`, `best`, `report`, `plot` |
| `pipeline/wildcat_models.py` | Demographic model definitions, bounds, time-ordering constraint |
| `pipeline/claic.py` | CLAIC via a Godambe information matrix from block bootstraps |
| `SLURM/run_stages.sh` | Driver: submits a four-round staged optimisation as a dependent chain |
| `SLURM/submit_stage.sh` | One round — array of restarts (`fit`), then pick the best (`best`) |
| `SLURM/submit_report.sh` | Polish, CLAIC, CSVs and figures, once fitting is done |
| `results/data.fs` | The folded 53 × 13 JSFS the models are scored against |
| `results/boots/` | 100 block bootstrap replicates, required for CLAIC |
| `results/*.csv`, `results/*.png` | Fitted parameters and fit diagnostics |
| `environment.yml` | Conda environment |

**The scripts expect a flat working directory.** `wildcat_pipeline.py` writes to a
relative `results/`, and the SLURM scripts `cd` to the directory `sbatch` was called
from. The `pipeline/` and `SLURM/` split here is for readability only — to run
anything, copy the contents of both into a single directory alongside `results/`.

## Models

| Name | Params | Description |
|---|---|---|
| `sec_contact` | 6 | dadi's `sec_contact_asym_mig`: split, isolation, then secondary contact |
| `basic` | 11 | Structured wildcat/domestic model with instantaneous size changes |
| `growth` | 13 | `basic` with exponential rather than instantaneous size change |

`basic` and `growth` need TA > max(TB, TD), an inequality constraint, so they are fitted
with COBYLA. `sec_contact` is unconstrained and uses log-space
Nelder-Mead, which handles the several orders of magnitude between times and sizes
considerably better.

## Results

| Model | Params (+θ) | log-likelihood | Effective params | CLAIC | ΔCLAIC |
|---|---|---|---|---|---|
| `growth` | 14 | −2423.57 | 370.2 | **5587.56** | 0 |
| `basic` | 12 | −2699.57 | 322.1 | 6043.37 | 455.8 |
| `sec_contact` | 7 | −3143.07 | 257.5 | 6801.24 | 1213.7 |

At eps = 0.01. `growth` is preferred, and the ordering is not close: the −2ll gaps
dominate the penalty gaps by roughly 5:1, so the ranking survives substantial error in
the effective-parameter estimate.

Note the size of the effective parameter counts — 257 to 370, against nominal counts of
7 to 14. That excess is the whole reason for using CLAIC here: it measures directly how
much linkage has inflated the confidence of the composite likelihood. Unadjusted AIC
would rank the models the same way but for the wrong reasons.

Per-model parameter estimates, in cats and years, are in `results/<model>_results.csv`.

## Quick start

```bash
conda env create -f environment.yml
conda activate dadi
```

`results/data.fs` and the bootstrap replicates are included, so `fit`, `report` and
`plot` run without the raw genomic data:

```bash
# One optimisation restart, to check the pipeline works
python wildcat_pipeline.py fit --model sec_contact --restarts 1 --verbose

# Combine restarts, polish the best, compute CLAIC
python wildcat_pipeline.py report --models sec_contact basic growth

# Data / model / residual figures
python wildcat_pipeline.py plot --models sec_contact basic growth
```

Only `sfs` needs the original MSMC multihetsep files, which aren't distributed here:

```bash
python wildcat_pipeline.py sfs --input chr1.txt chr2.txt --nboot 100
```

## Running on a cluster

Reliable convergence needs hundreds of restarts. `run_stages.sh` submits four rounds of
50 restarts as a chain of dependent jobs and exits in about a second:

```bash
sbatch run_stages.sh growth          # rounds 1-4
sbatch run_stages.sh basic 3         # resume at round 3
```

Round 1 explores widely from the model's default starting point; each later round
perturbs less around the best point found so far and tightens the tolerance. Chaining
uses `afterany` for the fitting arrays, so a few failed restarts don't stall the round.

Then, once fitting has finished:

```bash
sbatch --partition=compute --time=24:00:00 submit_report.sh sec_contact basic growth
```

`report` caches finished models, so an expensive model fitted in an earlier job still
appears in the comparison without being redone.

### Notes on the arguments

- `--seed-base` must differ between batches, or later restarts duplicate the seeds — and
  therefore the optima — of earlier ones.
- `--eps` takes several finite-difference steps for the Godambe matrices. The first is
  reported, the rest are a stability check. Smaller is not better: below about 1e-3 the
  second differences are dominated by numerical noise. Effective parameters varying by
  more than ~15% across steps means the estimate should not be reported.
- Pass `--no-cached` to ignore earlier runs, or delete `results/claic_<model>.pkl` before
  re-reporting after new fits.
- `--maxtime` and `--polish-maxtime` cap a pathological restart that would otherwise
  stall on a collapsed diffusion timestep.

## Data

46 diploid individuals — 26 Scottish wildcat, 6 domestic, 14 mainland European — over
chromosomes 1 and 2, 22,488,648 callable sites. Only the Scottish × domestic pair is
analysed here. Projected to a folded 53 × 13 spectrum, with 100 block bootstrap
replicates over 1 Mb blocks. μ = 0.86 × 10⁻⁸ per site per generation, 3 years per
generation.

**The underlying genomic data are not included in this repository.**

## Notes

Things worth knowing if you extend this:

- Extrapolate before folding, and fold after projection. Log-likelihoods are comparable
  only across models scored on the identical spectrum at the identical projection.
- Migrants per generation is `N_receiving × m`, not `N_ref × m`.
- `Nref` is a per-model scaling constant from the fitted θ, not an estimate of ancestral
  size. Convert to years before comparing across models.
- A non-positive-definite Hessian at the optimum means CLAIC should not be reported.
  CLAIC is computed on the linear scale for this reason — in log space dadi's finite
  differences put a negative eigenvalue into H on this data.
- Effective information is limited by the number of genomic blocks (~22 at 1 Mb), not by
  the number of bootstrap replicates.
- In dadi 2.4.4, `perturb_params` uses numpy's legacy global `RandomState` — seed with
  `np.random.seed()`, not `np.random.default_rng()`, or restarts aren't reproducible.
- `optimize_log_fmin` honours only `--maxiter`; COBYLA honours only `--maxtime` and
  `--maxeval`. They are not interchangeable, and a restart capped by the wrong one has no
  cap at all short of the SLURM wall clock.
- COBYLA raises `RoundoffLimited` when a parameter pins against a zero bound, so
  migration lower bounds are floored at 1e-4 rather than 0.

## References

Gutenkunst RN, Hernandez RD, Williamson SH, Bustamante CD (2009). Inferring the joint
demographic history of multiple populations from multidimensional SNP frequency data.
*PLoS Genetics* 5(10): e1000695.

Varin C, Vidoni P (2005). A note on composite likelihood inference and model selection.
*Biometrika* 92(3): 519–528.

Howard-McCombe J, Ward D, Kitchener AC, Lawson D, Senn HV, Beaumont M (2021). On the
use of genome-wide data to model and date the time of anthropogenic hybridisation: an
example from the Scottish wildcat. *Molecular Ecology* 30(15): 3688–3702.

## Acknowledgements

Supervised by Mark Beaumont, Dennis Prangle, and Grace Yan. University of Bristol - School of Biological Sciences & School of Mathematics
