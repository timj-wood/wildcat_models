# Demographic inference for Scottish wildcats and domestic cats using ∂a∂i

## Introduction

Scottish wildcats (*Felis silvestris grampia*) have undergone extensive hybridisation with domestic cats (*F. catus*). The hybridisation is now advanced enough that the wild population is at risk of being genetically "swamped", which would amount to extinction of the wildcat as a distinct genetic entity even if wild-living cats remain.

This project uses [∂a∂i](https://dadi.readthedocs.io/) to fit and compare demographic models for the two populations. ∂a∂i is a composite-likelihood method: rather than simulating datasets and comparing summaries, as in Approximate Bayesian Computation, it solves a diffusion approximation to the allele-frequency distribution under a given demographic model and computes the expected [joint site frequency spectrum](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000695) (JSFS). The observed JSFS is then treated as a multinomial sample from that expectation. This makes model fitting fast enough to compare many models and to optimise from many starting points, at the cost of treating linked sites as independent — an assumption that has to be corrected for before likelihoods can be compared or uncertainties quoted (see *Model comparison and uncertainty* below).

The original motivation was that a simple isolation-with-migration (IM) model appeared to fit the data poorly, suggesting model misspecification. Later work suggested that IM may in fact fit reasonably well, but the aim of the project is unchanged: with a larger dataset and a set of alternative models, it is possible to compare likelihoods properly, identify which features of the demography the data can and cannot resolve, and quantify how well the preferred model reproduces the observed spectrum.

The work also serves as a composite-likelihood baseline for Grace Yan's PhD, which develops [simulation-based inference](https://www.pnas.org/doi/10.1073/pnas.1912789117) (SBI) methods for genetic data. Documented parameter estimates and confidence intervals from a well-understood likelihood method give a point of comparison for the SBI results on the same dataset.

## The data

The dataset consists of chromosomes 1 and 2 from wild-caught Scottish wildcats and domestic cats:

* **16 wild-caught Scottish wildcats** and **6 domestic cats**.
* **22,488,648 callable sites** after filtering.
* A **folded 33 × 13 joint site frequency spectrum**. The spectrum is folded because the ancestral allele cannot be reliably assigned, so counts are recorded as minor-allele frequencies.
* Mutation rate μ = 0.86 × 10⁻⁸ per site per generation, from a domestic cat pedigree (Wang et al. 2022), and a generation time of approximately 3 years. These are used to convert scaled parameters into individuals and years.

Captive Scottish wildcats were excluded rather than pooled with the wild-caught samples. Hudson's *F*<sub>ST</sub> between the two groups is 0.085, and the pooled spectrum carries substantially more intermediate-frequency mass, which biases divergence-time and migration estimates.

## The models

Three two-population models are fitted and compared. Population sizes are given in individuals and times in generations before the present; migration subscripts name the **source** population first.

* **`basic`** (11 parameters). An ancestral population splits into a wildcat and a domestic lineage. Each lineage has its own size, and the domestic lineage undergoes a change in size at domestication. The Scottish wildcat lineage undergoes a recent collapse. Migration is symmetric in form but estimated separately in each direction, and is allowed to differ before and after the recent events.
* **`growth`** (13 parameters). As `basic`, but with exponential size change in the wildcat lineage rather than a step change.
* **`sec_contact`** (6 parameters). A secondary-contact model in ∂a∂i's own parameterisation: divergence in isolation followed by a period of migration. Included as a simpler alternative with a different qualitative history.

`basic` and `growth` impose a time-ordering constraint, TA > max(TB, TD), so that the ancestral split precedes both the wildcat collapse and domestication. The constraint is enforced during optimisation using [COBYLA](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/), which handles inequality constraints directly.

Because `sec_contact` is written in ∂a∂i's own notation (ν as a ratio to N<sub>ref</sub>, epoch durations, migration named by receiving population) and `basic` and `growth` follow Mark's notation (absolute sizes, times before present, source-first migration subscripts), the two conventions are kept separate throughout rather than harmonised. Translating between them silently is an easy way to introduce sign and direction errors.

## Fitting

Optimisation is staged, in four rounds of 50 restarts each. Each round perturbs the best parameters from the previous round by a shrinking factor (2 → 1 → 0.5 → 0.25 fold) and tightens the convergence tolerance. Convergence is judged by the spread of the top ten log-likelihoods rather than by the optimiser's own exit status: a narrow spread across independent starts is evidence that the optimum is real, and a parameter that drifts monotonically towards its bound across rounds is evidence that it is not identified by the data.

## Model comparison and uncertainty

Composite likelihood treats linked sites as independent, so it overstates the information in the data — here by roughly 67-fold. Raw AIC and likelihood-ratio tests are therefore invalid. Two corrections are applied, both built from a **block bootstrap** (100 replicates, 1 Mb blocks, giving around 415 blocks across the two chromosomes):

* **CLAIC** for model comparison, using the [Godambe information matrix](https://academic.oup.com/mbe/article/33/2/591/2579696) to rescale the effective number of parameters.
* **Confidence intervals** from the Godambe sandwich covariance, H⁻¹JH⁻¹, computed on the log scale via the delta method so that intervals on sizes and times stay positive.

It is the number of independent blocks, not the number of bootstrap replicates, that limits the effective information, so increasing the replicate count does not narrow the intervals.

## Repository contents

| File | Purpose |
| --- | --- |
| `wildcat_pipeline.py` | Data loading, spectrum construction, staged optimisation, bootstrapping, CLAIC and confidence intervals |
| `wildcat_models.py` | Model definitions (`basic`, `growth`, `sec_contact`) and parameter bounds |
| `claic.py` | Godambe information matrix and CLAIC calculation |
| `plot_demography.py` | Converts fitted parameters to effective population size against years before present for both lineages, with confidence shading; exports long-format CSV for replotting |
| `report/` | LaTeX source for the write-up |

Output directories are set through the `WILDCAT_OUTDIR` environment variable, read by both the Python pipeline and the submission scripts, so that results from different sample sets cannot be mixed.

## Running

Fits are run on the University of Bristol BluePebble cluster under SLURM. All jobs are submitted with `sbatch`; nothing is run interactively on the login node. Short diagnostics go to the `short` partition and full staged fits to `compute`.

The environment is a conda environment containing ∂a∂i 2.4.4 and its dependencies:

```
conda env create -f environment.yml
conda activate dadi
```

A typical run:

```
export WILDCAT_OUTDIR=results_wild
sbatch scripts/fit_model.sh basic
sbatch scripts/bootstrap.sh
sbatch scripts/claic.sh --fitdir results_wild/round4
```

## References

Gutenkunst, R. N., Hernandez, R. D., Williamson, S. H., & Bustamante, C. D. (2009). Inferring the joint demographic history of multiple populations from multidimensional SNP frequency data. *PLoS Genetics*, 5(10), e1000695.

Coffman, A. J., Hsieh, P. H., Gravel, S., & Gutenkunst, R. N. (2016). Computationally efficient composite likelihood statistics for demographic inference. *Molecular Biology and Evolution*, 33(2), 591–593.
