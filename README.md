# Scottish wildcat / domestic cat demography

Fitting two-population demographic models to a joint site frequency spectrum
with [dadi](https://dadi.readthedocs.io) 2.4.4, and comparing them with CLAIC.

Summer research internship, University of Bristol, supervised by Mark Beaumont, Dennis Prangle, and Grace Yan

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

<img src="plots/jsfs.png" alt="Folded joint site frequency spectrum, Scottish wild-caught x domestic" width="450">

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

## Results

Each model was fitted from 50 restarts in each of four rounds, then compared
with CLAIC over 100 block bootstraps. Effective parameter counts are
tr(J.H^-1), and are far larger than the number of free parameters because the
composite likelihood treats linked sites as independent.

| Model | ll | k | eff. k | CLAIC | dCLAIC |
|---|---|---|---|---|---|
| `growth` | -1376.22 | 13 | 310.6 | 3373.67 | 0.00 |
| `basic` | -1411.14 | 11 | 292.7 | 3407.74 | 34.07 |
| `sec_contact` | -1768.69 | 6 | 338.3 | 4214.04 | 840.37 |

`sec_contact` is rejected. It loses by 840 CLAIC units, and its effective
parameter count is the largest of the three despite having the fewest
parameters, which is what a sandwich estimator does when a model is
misspecified. Its migration rates are also at or above the point where dadi's
diffusion grid stops being reliable, with m12 pinned on its upper bound.

`basic` and `growth` are not distinguishable. The gap is 34 CLAIC units, but
the effective parameter counts are near 300 and carry roughly 10-15% sampling
noise, so the penalty term alone is uncertain by tens of units. `growth` also
fits nuS against its upper bound, which makes that optimum constrained rather
than maximal and its CLAIC not formally valid, and its top-10 restart spread of
0.90 is marginal.

`basic` is reported here. It is the more parsimonious of the two, it converged
cleanly (top-10 spread 0.107, no parameter on a bound), and it puts the
domestication size change at a date the archaeology supports.

| Parameter | Value |
|---|---|
| Nref | 13,514 |
| Split of silvestris and lybica, TA | 485,000 years |
| Domestic size change, TD | 10,600 years |
| Wildcat size change, TB | 673 years |
| Wildcat Ne after TB | 2,711 |
| Domestic Ne after TD | 22,413 |
| Domestic into wildcat | 1.88 individuals per generation |
| Wildcat into domestic | 5.03 individuals per generation |

<img src="results/fit_basic.png" alt="basic model fit and residuals" width="650">

TD is close to the archaeological estimate for cat domestication in the Near
East, which is around 10,000 years and is not an input to the fit. TA is
roughly twice the usual published figure for the silvestris/lybica split. Deep
parameters are weakly constrained by a site frequency spectrum, and both models
agree on it, so this is a property of the data rather than of either
parameterisation. TB is far too recent for the landbridge it was meant to
represent, and is more likely picking up the recent decline of the Scottish
population.

m2_ds, recent gene flow into the wildcat, sits within 10% of its upper bound
and above the rate at which the diffusion grid strains, so 18.76 is better read
as the data wanting a high rate than as an estimate of one.

Residuals for `basic` and `growth` are near identical and still structured,
mostly along the low domestic frequency edge and the fixed-in-domestic column.

Confidence intervals from the Godambe matrix are not computed, so the values
above carry no uncertainties.

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
