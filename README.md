# FIG-AIME-2026: figure reproduction

This repo reproduces the four figures in the Fractional Information Gain (FIG) paper:
[Fractional Information Gain: a student-level, differentiable personalization metric for black-box
predictive models](https://aclanthology.org/2026.aimecon-main.12/), AIME-2026.
It uses the [FIG library](https://github.com/RenaissancePhilanthropy/fractional-information-gain)
as a dependency, and only contains the experiments and plotting code for reproducing the
results in the paper. If you want to use FIG in your own work, we encourage you to use the [FIG library](https://github.com/RenaissancePhilanthropy/fractional-information-gain).

Each figure is drawn by one R script in `R/`. Figures 1 & 3 are self-contained (they
generate their own synthetic data) and draw immediately. Figures 2 & 4 read an input CSV
that must first be computed from raw datasets by the Python pipeline in `pipeline/`.
Everything is seeded, and the paper's figures are included in `figures/reference/` for
comparison. Output PDFs go to `figures/`.

## How to run

Install the R dependencies once (paste straight into bash from the repo root):

```bash
# CRAN packages (plus remotes, for the GitHub install below)
Rscript -e 'install.packages(c("tidyverse","cowplot","pROC","remotes"), repos="https://cloud.r-project.org")'
# figmetric: the R implementation of the FIG metric
Rscript -e 'remotes::install_github("RenaissancePhilanthropy/fractional-information-gain", subdir="r")'
```

To plot Figures 1 and 3:

```bash
make plots
```

To plot all figures, first download the AS'15 and KDD Cup datasets as instructed in
`pipeline/DATASETS.md` (the KDD Cup ones require making an account), and then run

```bash
uv sync                         # create ./.venv from pyproject.toml + uv.lock (installs fig too)
uv run make data                # synthetic data + open real datasets + item-only table
uv run make train               # train SAKT (Fig 4 curve + Fig 2 bars)
uv run make plots               # draw all four figures -> figures/
```

If you run the above without those datasets, the plots will still render, but Figure 2 will be
missing the corresponding bars.

**The datasets `make data` downloads are not covered by this repo's licence.** They come with
their own conditions on commercial use, redistribution and citation; see `pipeline/DATASETS.md`.
