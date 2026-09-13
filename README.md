# Scottish wildcat / domestic cat demography

## Introduction

Scottish wildcats (*Felis silvestris*) have hybridised extensively with domestic cats (*F. catus*). The hybridisation is severe enough that the wild population is at risk of being genetically swamped: enough domestic ancestry enters the population each generation that the wildcat genome is progressively replaced rather than the two forms remaining distinct. Interpreting how far that has gone needs a demographic baseline. The code in this repo was used to try and establish how the two lineages separated, how much gene flow has passed between them since, and how large the populations were before and after the recent collapse of the Scottish one.

This project estimates that baseline by fitting two-population demographic models to the joint site frequency spectrum (JSFS) with [dadi](https://dadi.readthedocs.io) 2.4.4, comparing the models with CLAIC, and putting confidence intervals on the fitted parameters with the Godambe information matrix.

dadi takes a different route to the same question as the ABC and simulation-based approaches used elsewhere on this dataset. Rather than simulating replicate datasets and comparing summary statistics, it numerically solves a [diffusion approximation](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695) to the Wright-Fisher process to get the expected JSFS under a set of demographic parameters, and scores that expectation against the observed spectrum with a Poisson likelihood over bins. It is fast, and it returns a likelihood, so models can be ranked directly.

However, the likelihood being composite presents initial issues. Linked sites are not independent, but the Poisson calculation treats every site as though it were, so the likelihood surface is correctly located but far too sharply peaked. Parameter estimates are still consistent, but standard errors and likelihood differences are not. Both are corrected here using the [Godambe information matrix](https://doi.org/10.1093/molbev/msv255) estimated from a block bootstrap, which is also what CLAIC uses in place of the AIC penalty. The effective parameter counts in the results below give a sense of the size of the problem: eleven free parameters behave like nearly three hundred.

## Data

Whole-genome SNPs from 46 cats, chromosomes A1 and A2, in MSMC multihetsep format. The analysed pair is:

| Population | Individuals | Haplotypes |
|---|---|---|
| Scottish wildcat (wild-caught) | 16 | 32 |
| Domestic | 6 | 12 |

10 captive-bred Scottish cats and 14 mainland European wildcats are in the input files but are not analysed. Hudson FST between the wild-caught and captive Scottish cats is 0.085, and the captive cats have 21% lower nucleotide diversity, so they are not treated as one population. The comparison of pooled and wild-only spectra in Section 2.2 of the report (site counts, Tajima's D, FST, intermediate-frequency mass, pi) is reproduced by `scripts/pooling_check.py`.

Constants: L = 22,488,648 callable sites, mu = 0.86e-8 per bp per generation (Wang et al. 2022), generation time 3 years (Howard-McCombe et al. 2021).

Three site counts:

| Count | Value | Where it comes from |
|---|---|---|
| Raw segregating sites | 145,716 | lines in the two multihetsep files |
| Biallelic | 145,512 | after dropping 204 multiallelic |
| In the fitted spectrum | 123,348 | `Spectrum.S()` on the 33 x 13 array |

The last gap is not filtering. 22,164 sites are polymorphic across all 92 haplotypes but monomorphic within the 32 Scottish and 12 domestic haplotypes analysed, so they land in the masked corner.

The spectrum is folded, 33 x 13. 209 of its 429 bins are masked: the monomorphic corner, plus the 208 redundant entries above the folding diagonal at i + j = 22, whose counts are already carried by their reflected partners. The likelihood is therefore evaluated over 220 bins. Because the spectrum is folded, the axes are minor-allele counts (minor across the pooled 44 haplotypes), not derived-allele counts.

<img src="plots/jsfs.png" alt="Folded joint site frequency spectrum, Scottish wild-caught x domestic" width="450">

## Models

| Name | Function | Parameters |
|---|---|---|
| `sec_contact` | `dadi.Demographics2D.sec_contact_asym_mig` | 6 |
| `basic` | `wildcat_domestic` | 11 |
| `growth` | `wildcat_domestic_growth` | 13 |

All three models involve the ancestral population splitting into a *silvestris* branch and a *lybica* branch, where the two exchange migrants after the split. Model differences lie in what occurs to the population sizes after the split, and in when the migration rates are allowed to change.

`basic` works as follows:

* An ancestral population of size NA exists until TA, when it splits into the
  lineage leading to the Scottish wildcat and the lineage leading to the domestic
  cat.
* The two branches exchange migrants continuously and asymmetrically from TA
  onwards. Each direction has an early rate and a recent rate: into the wildcat,
  m_ls from TA to TB and then m2_ds to the present; into the domestic, m_sl from
  TA to TD and then m2_sd to the present.
* The domestic branch changes size instantaneously at TD. The fit places this at
  about 10,600 years, near the archaeological evidence for early cat-human
  association, but with a confidence interval too wide to treat as a date.
* The Scottish branch changes size instantaneously at TB, which the fit places
  several hundred years ago and estimates far more tightly than anything else in
  the model.
* The two size changes are constrained to postdate the split, TA > max(TB, TD).

See the [model schematic](plots/model.pdf) for the full parameterisation.

`growth` is the same model with exponential size changes in place of the instantaneous ones, which adds two present-day sizes. `sec_contact` is simpler: the branches are completely isolated after the split for a duration T1 and then exchange migrants for a duration T2, at two asymmetric rates m12 and m21, with one size per branch.

`basic` and `growth` are constrained, so they are fitted with COBYLA, whereas `sec_contact` is unconstrained and uses Nelder-Mead in log space.

Migration subscripts name the receiving population first in `sec_contact`, following dadi, and the source population first in `basic` and `growth`, following the original specification. Sizes in `basic` and `growth` are ratios to NA, which is fixed at 1 and absorbed into theta, which is why `basic` has 11 free parameters rather than 12.

`basic` is not a special case of `growth`: sizes in `growth` are continuous at TB and TD by construction, so no choice of its parameters reproduces the jump that `basic` places there. They are non-nested, hence CLAIC rather than a likelihood ratio test.

## Running it

This work was carried out using the computational facilities of the Advanced Computing Research Centre, University of Bristol - http://www.bristol.ac.uk/acrc/.

Build the spectrum once:

    mkdir -p logs && sbatch scripts/submit_sfs.sh

Then fit a model. This submits four rounds of 50 restarts as a chain of dependent jobs, each round perturbing less around the best point from the last:

    sbatch scripts/run_stages.sh sec_contact
    sbatch scripts/run_stages.sh basic
    sbatch scripts/run_stages.sh growth

The answer for each model ends up in `results_wild/best_<model>_r4.json`. Then compare:

    sbatch scripts/submit_report.sh

which writes per-model CSVs, `model_comparison.csv`, confidence intervals, fit figures, and `claic_<model>.pkl` holding the Godambe H and J for each model. Finally, the correlation matrices:

    sbatch scripts/submit_corr.sh

which reads the stored H and J, forms the sandwich covariance H^-1 J H^-1 (the same matrix whose diagonal gave the intervals), and writes `results/correlation/corr_<model>.pdf` and `corr_<model>.csv`. Nothing is refitted and no bootstrap gradient is recomputed, so this takes seconds. The job log prints se/value against the stored standard errors as a check that the pickle matches the fit in the tables.

Run parameters are written to a metadata file alongside each set of results, including the seed used to draw the 100 bootstrap replicates. The bootstrap is the only stochastic step between the data and the reported CLAIC values and intervals, so recording the seed means those can be regenerated exactly rather than approximately.

## Files

    scripts/
      wildcat_pipeline.py    everything: spectrum, fitting, rescaling, report
      wildcat_models.py      model functions and bounds (Dennis)
      claic.py               CLAIC (Dennis)
      plot_corr.py           correlation matrices from the stored Godambe H and J
      pooling_check.py       Table 2 of the report: pooled vs wild-only spectra, pi
      submit_sfs.sh          build the spectrum and bootstraps
      run_stages.sh          submit a four-round staged optimisation
      submit_stage.sh        one round, run as a job array
      submit_report.sh       CLAIC comparison, intervals and figures
      submit_corr.sh         correlation matrices (runs plot_corr.py)
    data/                    input multihetsep files
    environment.yml          conda environment (dadi 2.4.4)
    plots/                   spectrum and model schematic
    results/
      models/                fits, intervals, model_comparison.csv, fit figures
      correlation/           corr_<model>.csv and corr_<model>.pdf

Output directory is set by `WILDCAT_OUTDIR`, default `results_wild`.

## Results

Four rounds of 50 restarts per model, compared with CLAIC over 100 block bootstraps. Effective parameter counts are tr(J.H^-1) over the parameter vector including theta, so read against k+1. They far exceed the free parameter count because the composite likelihood treats linked sites as independent.

| Model | ll | k | eff. k | CLAIC | dCLAIC |
|---|---|---|---|---|---|
| `growth` | -1376.22 | 13 | 310.6 | 3373.67 | 0.00 |
| `basic` | -1411.14 | 11 | 292.7 | 3407.74 | 34.07 |
| `sec_contact` | -1768.69 | 6 | 338.3 | 4214.04 | 840.37 |

`sec_contact` scores 840 units worse than `growth` and 806 worse than `basic`. `basic` and `growth` are not distinguishable: the `growth` fit ended with N_S on its upper bound, where CLAIC is not defined, and in any case the 34-unit gap is smaller than the noise expected on the penalty terms from 100 replicates. `basic` is reported as the primary result: it is the only model with no parameter on a bound, and its effective parameter count is the most stable to the finite-difference step size (0.56 against 2.51 for `growth` and 21.15 for `sec_contact`). All three models converged in the sense that the top-10 spread in the final round was small (0.000, 0.107 and 0.903 for `sec_contact`, `basic` and `growth`).

### Parameters of `basic`

95% intervals from the Godambe matrix on the log scale, so multiplicative and asymmetric. Physical intervals convert each limit at the point estimate of Nref, and so do **not** carry the uncertainty in theta.

| Parameter | Value | 95% CI | CI width |
|---|---|---|---|
| Nref | 13,514 | 6,144 - 29,718 | 4.8 |
| Split of silvestris and lybica, TA | 485,000 yr | 186,000 - 1,264,000 | 6.8 |
| Domestic size change, TD | 10,600 yr | 3,200 - 35,100 | 11.0 |
| Wildcat size change, TB | 673 yr | 648 - 700 | 1.08 |
| Wildcat Ne after TB | 2,711 | 2,191 - 3,354 | 1.53 |
| Domestic Ne after TD | 22,413 | 8,661 - 58,002 | 6.7 |
| Domestic into wildcat, m2_ds | 18.76 | 18.03 - 19.51 | 1.08 |
| Wildcat into domestic, m2_sd | 6.07 | 4.49 - 8.20 | 1.83 |

Migrant counts are 1.88 individuals per generation into the wildcat and 5.03 into the domestic, with no interval, being products of two correlated parameters. Rates and counts point opposite ways, because the count scales with the receiving population.

<img src="results/models/fit_basic.png" alt="basic model fit and residuals" width="650">

### Parameter correlations

The interval widths above are marginal and should not be read independently. The correlation matrix of the sandwich covariance shows that the deep parameters of `basic` lie along a single ridge: N_S, N_L, N_D and T_A correlate with one another at +0.95 to +0.98 and with theta at -0.95 to -0.99, and the early migration rates correlate negatively with the sizes and positively with theta. Raising the ancestral sizes and lengthening the split while lowering theta and the early gene flow leaves the composite likelihood almost unchanged, which is why those intervals are wide.

The narrow intervals are not independent either. T_B and m2_ds correlate at +0.987, so what the data fix is a combination of the length of the recent epoch and the rate of gene flow within it, not each separately. Of the four tightly estimated parameters, only the wildcat size after T_B and m2_sd are free of strong correlations (nothing above 0.49 and 0.20 respectively).

H has a condition number of order 1e15, so correlations near +/-1 should be read as saying two parameters are not separately identified, not as precise measures of covariation.

[parameter correlation matrix for basic](results/correlation/corr_basic.pdf)

Matrices for all three models are in `results/correlation/corr_<model>.csv`, with parameter order in the header row. Order follows `wildcat_models.py`, so in `growth` the present-day sizes come fifth and sixth.
