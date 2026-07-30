# Producing the figure-input data

This pipeline computes the input CSVs for Figures 2 and 4 from raw data, writing them to
`out/`. Everything is seeded, so rerunning it reproduces the paper's figure inputs. (Model
training is floating-point sensitive, so a different platform can produce slightly
different trained numbers.)

**Normally you drive this through the `Makefile` one level up** (`make data`, `make train`,
`make plots`; see the top-level README). The `python ...` commands below are what those
targets run, for when you want a single step by hand.

## Environment

Dependencies (numpy, pandas, scikit-learn, torch, and `fig`) are declared in
`../pyproject.toml` and locked in `../uv.lock`. Create the environment with `uv` (from the
repo root), then run steps inside it:

```bash
uv sync                 # creates ./.venv with the locked dependencies (incl. fig)
uv run make data        # or: make data PY=.venv/bin/python
```

`fig` (the `fractional-information-gain` package) is installed from PyPI.

## Figure 4: `calibration_curves.csv` (synthetic, seeded) -- runs end to end

```bash
python pipeline/synthetic/generate_1pl_5skill.py        # -> datasets/1pl/training_1pl_5000_students_200_items.csv
python pipeline/fig4_calibration_curves.py              # trains SAKT, evals Skill-3 partial-input -> out/calibration_curves.csv
```

Trains a SAKT model on the 1PL 5-skill data and evaluates it under the Skill-3-input
partial protocol, computing FIG-V via `fig.fig_torch`. Training takes an hour or two on a CPU.
Then plot it: `make plots`.

## Figure 2: `baseline_vs_sota_computed.csv`

```bash
python pipeline/synthetic/generate_sim_irt.py           # -> datasets/sim_irt/sim_irt_unif{2,5}/...
python pipeline/compute_table.py                        # item-only + reference -> out/baseline_vs_sota_computed.csv
# Trained SAKT bars (per dataset), then re-run compute_table.py to merge them in:
python pipeline/train_sakt_long.py --train <..._train.csv> --test <..._test.csv> --dataset sim_u2
python pipeline/compute_table.py
```

The two synthetic columns (sim_u2, sim_u5) run fully with no downloads. The four real
columns need their datasets prepared first (only NIPS34 auto-downloads; AS'15 and the KDD Cup
pair are gated) -- see `DATASETS.md`.

## Files

```
synthetic/generate_1pl_5skill.py   5-skill 1PL generator (Fig 4 data), seeded
synthetic/generate_sim_irt.py      dense 1PL IRT generator (Fig 2 sim columns), seeded
fig4_calibration_curves.py         train SAKT + Skill-3 partial-input eval -> calibration_curves.csv
compute_table.py                   item-only baseline + reference -> baseline_vs_sota_computed.csv
train_sakt_long.py                 train SAKT on one dataset -> out/trained_sakt_ci.csv
sakt.py                            SAKT model + training helpers (shared by the two trainers)
baserate_baseline.py               item base-rate baseline (item-only metrics, via fig)
bootstrap_ci.py                    student-bootstrap CIs
dataset_utils.py                   canonical long-format + by-student split helpers
DATASETS.md                        per-dataset sources, access status, prepare-format spec
datasets/
  assistments2015/prepare_assistments2015.py   as15 (gated raw file -> convert)
  nips34/prepare_nips34.py                      nips34/Eedi (auto-download + convert)
  prepare_kddcup.py                             al2005 + bd2006 (gated raw file -> convert)
  <name>/*.csv, *.zip, ...                       generated/downloaded data (gitignored)
out/                               pipeline outputs (gitignored)
```
