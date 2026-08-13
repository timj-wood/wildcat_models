# Scottish wildcat / domestic cat demography

Fitting two-population demographic models to a joint site frequency spectrum
with [dadi](https://dadi.readthedocs.io) 2.4.4, and comparing them with CLAIC.

Summer research internship, University of Bristol, supervised by Mark Beaumont.
Model definitions (`wildcat_models.py`) and the CLAIC implementation
(`claic.py`) are Dennis Prangle's.

## Data

Whole-genome SNPs from 46 cats, chromosomes 1 and 2, in MSMC multihetsep
format. The analysed pair is:

| Population | Individuals | Haplotypes |
|---|---|---|
| Scottish wildcat (wild-caught) | 16 | 32 |
| Domestic | 6 | 12 |

10 captive-bred Scottish cats and 14 mainland European wildcats are in the
input files but are not analysed. Hudson FST between the wild-caught and
captive Scottish cats is 0.085, so they are not treated as one population.

Constants: L = 22,488,648 callable sites, mu = 0.86e-8 per bp per generation
(Wang et al. 2022), generation time 3 years (Howard-McCombe et al. 2021).

The spectrum is folded, 33 x 13, with 123,348 segregating sites. 204
multiallelic sites are dropped.

## Models

| Name | Function | Parameters |
|---|---|---|
| `sec_contact` | `dadi.Demographics2D.sec_contact_asym_mig` | 6 |
| `basic` | `wildcat_domestic` | 11 |
| `growth` | `wildcat_domestic_growth` | 13 |

`basic` is isolation with migration plus a size change in each branch, with the
constraint TA > max(TB, TD). `growth` is the same with exponential size changes
instead of instantaneous ones. Both are constrained, so they are fitted with
COBYLA; `sec_contact` is unconstrained and uses Nelder-Mead in log space.

## Running it

Everything goes through SLURM. Build the spectrum once:

    mkdir -p logs && sbatch submit_sfs.sh

Then fit a model. This submits four rounds of 50 restarts as a chain of
dependent jobs, each round perturbing less around the best point from the last:

    sbatch run_stages.sh sec_contact
    sbatch run_stages.sh basic
    sbatch run_stages.sh growth

The answer for each model ends up in `results_wild/best_<model>_r4.json`.
Then compare:

    sbatch submit_report.sh

which writes per-model CSVs, `model_comparison.csv` and fit figures.

## Files

    wildcat_pipeline.py    everything: spectrum, fitting, rescaling, report
    wildcat_models.py      model functions and bounds (Dennis)
    claic.py               CLAIC (Dennis)
    submit_sfs.sh          build the spectrum and bootstraps
    run_stages.sh          submit a four-round staged optimisation
    submit_stage.sh        one round, run as a job array
    submit_report.sh       CLAIC comparison and figures

Output directory is set by `WILDCAT_OUTDIR`, default `results_wild`.

## Status

Work in progress. `sec_contact` is running; `basic` and `growth` to follow.

An earlier version of this analysis pooled the wild-caught and captive Scottish
cats. Those results are not comparable to these, because the spectrum is a
different shape and composite log-likelihoods do not carry across.

## Notes

Some things that cost time to work out, in case they are useful:

- `dadi.Inference.opt(log_opt=True)` returns the exponentiated starting point,
  not the optimum. The likelihood is still correct.
- `dadi.Misc.perturb_params` uses numpy's legacy global RandomState.
  `np.random.default_rng()` is silently ignored, so seed with
  `np.random.seed()`.
- `optimize_log_fmin` honours only `maxiter`. COBYLA honours only `maxtime` and
  `maxeval`. Passing the wrong one fails silently.
- COBYLA raises `RoundoffLimited` when a parameter pins against a bound of
  exactly 0, so migration lower bounds are floored at 1e-4.
- CLAIC needs a properly converged optimum or the Hessian is unstable. The
  report stage re-polishes the best restart before computing it.
