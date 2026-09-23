# wildcat_models

Demographic inference for the Scottish wildcat (*Felis silvestris*) and the domestic cat (*F. catus*) from a folded joint site frequency spectrum, using [dadi](https://dadi.readthedocs.io) 2.4.4.

Three two-population models are fitted, compared with CLAIC, and given confidence intervals from the Godambe information matrix (100 block-bootstrap replicates). Background, methods and results are in the accompanying report ([report](dadi_report.pdf)).

## Requirements

A SLURM cluster and conda. Create the environment with:

    conda env create -f environment.yml

## Data

`data/` holds MSMC multihetsep files for 46 cats on chromosomes A1 and A2. Only two groups are analysed:

| Population | Individuals | Haplotype columns |
|---|---|---|
| Scottish wildcat (wild-caught) | 16 | 0–31 |
| Domestic | 6 | 52–63 |

Captive Scottish cats (32–51) and mainland European wildcats (64–91) are in the files but excluded. See Section 2.2 of the report for why.

Fixed constants: callable length L = 22,488,648 bp, µ = 0.86e-8 per bp per generation, generation time 3 years.

## Models

| Name | Function | Free parameters |
|---|---|---|
| `sec_contact` | `dadi.Demographics2D.sec_contact_asym_mig` | 6 |
| `basic` | `wildcat_domestic` | 11 |
| `growth` | `wildcat_domestic_growth` | 13 |

`basic` and `growth` are constrained (TA > max(TB, TD)) and fitted with COBYLA; `sec_contact` uses Nelder-Mead in log space. Full parameterisations are in Table 3 of the report and the [model schematic](plots/model.pdf).

Note that migration subscripts name the receiving population first in `sec_contact` (dadi convention) but the source population first in `basic` and `growth`.

## Usage

1. Build the spectrum and bootstrap replicates:

       mkdir -p logs && sbatch scripts/submit_sfs.sh

2. Fit each model (four chained rounds of 50 restarts):

       sbatch scripts/run_stages.sh sec_contact
       sbatch scripts/run_stages.sh basic
       sbatch scripts/run_stages.sh growth

   The best fit for each model is written to `results_wild/best_<model>_r4.json`.

3. Compare models and compute intervals:

       sbatch scripts/submit_report.sh

4. Compute correlation matrices from the stored Godambe matrices (no refitting, takes seconds):

       sbatch scripts/submit_corr.sh

The output directory is set by `WILDCAT_OUTDIR` (default `results_wild`).

## Outputs

- `model_comparison.csv`: log-likelihoods, effective parameter counts and CLAIC
- Per-model parameter CSVs with 95% confidence intervals and physical values
- `fit_<model>.png`: observed and expected spectra with Anscombe residuals
- `claic_<model>.pkl`: Godambe H and J matrices
- `correlation/corr_<model>.csv` and `.pdf`: parameter correlation matrices (parameter order follows `wildcat_models.py`)

Each run writes a metadata file recording its parameters, including the bootstrap seed, so CLAIC values and intervals can be reproduced exactly.

## Files

    scripts/
      wildcat_pipeline.py    spectrum, fitting, rescaling, report
      wildcat_models.py      model functions and bounds
      claic.py               CLAIC calculation
      plot_corr.py           correlation matrices from stored H and J
      pooling_check.py       pooled vs wild-only spectra (report Table 2)
      submit_sfs.sh          build spectrum and bootstraps
      run_stages.sh          submit four-round staged optimisation
      submit_stage.sh        single round, as a job array
      submit_report.sh       CLAIC comparison, intervals, figures
      submit_corr.sh         correlation matrices
    data/                    input multihetsep files
    plots/                   spectrum and model schematic
    results/
      models/                fits, intervals, comparison, fit figures
      correlation/           correlation matrices
    environment.yml          conda environment

## Credits

This project was supervised by Mark Beaumont, Dennis Prangle and Grace Yan. Model specification by Mark Beaumont. `wildcat_models.py` and `claic.py` specification by Dennis Prangle. 

This work was carried out using the computational facilities of the [Advanced Computing Research Centre](http://www.bristol.ac.uk/acrc/), University of Bristol.
