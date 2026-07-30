# Datasets for the Figure 2 pipeline

Figure 2 (`item_only_vs_trained_perstudent.pdf`) compares the item-only baseline, a trained
SAKT model, and a reference across six dataset columns. Two are synthetic (generated here,
seeded); four are real knowledge-tracing benchmarks.

## Canonical long format

Every dataset is consumed as two CSVs, `<name>_train.csv` and `<name>_test.csv`, split
**by student** (15% of students held out, seed 42, via `dataset_utils.split_by_student`).
Required columns:

| column       | meaning                                              |
|--------------|------------------------------------------------------|
| `student_id` | learner id (int)                                     |
| `item_id`    | item / question / KC id at the granularity used (int)|
| `score`      | correctness in {0, 1}                                |
| `order_idx`  | 0-based per-student response order (for sequences)   |
| `true_prob`  | (sim only) generating probability sigmoid(theta - d) |

Files live at `pipeline/datasets/<dir>/<prefix>_{train,test}.csv` where `<dir>`/`<prefix>`
are listed in `compute_table.py:COLUMNS`.

## Synthetic columns (generated here, seeded)

| key    | generator                              | shape            | reproduces |
|--------|----------------------------------------|------------------|------------|
| sim_u2 | `synthetic/generate_sim_irt.py` (U[-2,2]) | 4000 x 300 dense | item-only + IRT-oracle reference are deterministic given the seeded generator |
| sim_u5 | `synthetic/generate_sim_irt.py` (U[-5,5]) | 4000 x 300 dense | same |

## Real columns

Each real dataset has a `prepare_*.py` that emits `<prefix>_{train,test}.csv` in the canonical
format. `compute_table.py` skips any dataset whose files are absent, so you can prepare them
incrementally. Only NIPS34 auto-downloads; AS'15 and the two KDD Cup sets are gated, so you fetch
those from their distributors yourself (see below).

| key    | dataset                         | item granularity | prepare script | source / access |
|--------|---------------------------------|------------------|----------------|-----------------|
| as15   | ASSISTments 2015 skill builder  | sequence_id      | `datasets/assistments2015/prepare_assistments2015.py --raw <file>` | ASSISTments (gated) |
| nips34 | Eedi NeurIPS 2020 Tasks 3&4     | QuestionId       | `datasets/nips34/prepare_nips34.py` (auto-download) | Azure `dqanonymousdata.blob.core.windows.net/neurips-public/data.zip` (657 MB) |
| al2005 | KDD Cup 2010 Algebra 2005-2006  | KC               | `datasets/prepare_kddcup.py --dataset al2005 --raw <file>` | PSLC DataShop (gated) |
| bd2006 | KDD Cup 2010 Bridge-to-Algebra 2006 | KC          | `datasets/prepare_kddcup.py --dataset bd2006 --raw <file>` | PSLC DataShop (gated) |

The AS'15 and NIPS34 prepare scripts reproduce the paper's item-only metrics **exactly**: the
item-only baseline is order-independent, so it is fully determined by the by-student split and
item granularity.

### Licences

The datasets are not part of this repo; each `prepare_*.py` fetches or converts its own copy, and
they come with their own terms. Eedi describes the NIPS34 data and its licence conditions at
<https://www.eedischool.com/projects/neurips-education-challenge>. The terms for AS'15 and for the
KDD Cup pair are linked from their sections below.

### ASSISTments 2015 -- where to get the gated data

AS'15 is not auto-downloaded. ASSISTments distribute it from their own dataset page, and set out
terms of use on the same site:

- Dataset page: <https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data>
- Terms of use: <https://sites.google.com/site/assistmentsdata/termsofuseforusingdata>

**Auto-discovery: copy the downloaded CSV to its canonical path**, then just re-run `make data`:
```
pipeline/datasets/assistments2015/2015_100_skill_builders_main_problems.csv
```

Equivalently, prepare it explicitly (a custom path can be passed with `--raw`):
```bash
python pipeline/datasets/assistments2015/prepare_assistments2015.py
```

### KDD Cup 2010 -- where to get the gated data

AL'05 and BD'06 are not downloadable unattended. Get them from the KDD Cup 2010 EDM Challenge
via PSLC DataShop (free account + data-use agreement):

- Download page: <https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp>
- `al2005` -> "Algebra I 2005-2006", raw training file `algebra_2005_2006_train.txt`
- `bd2006` -> "Bridge to Algebra 2006-2007", raw training file `bridge_to_algebra_2006_2007_train.txt`

**Auto-discovery: copy each downloaded raw file to its canonical path**, then just re-run
`make data` (it prepares them automatically; `make train` and `make plots` then pick up the
new columns):
```
pipeline/datasets/kddcup_al2005/algebra_2005_2006_train.txt
pipeline/datasets/kddcup_bd2006/bridge_to_algebra_2006_2007_train.txt
```

Equivalently, prepare them explicitly (a custom path can be passed with `--raw`):
```bash
python pipeline/datasets/prepare_kddcup.py --dataset al2005   # reads the canonical path above
python pipeline/datasets/prepare_kddcup.py --dataset bd2006
```
The converter uses the first `KC(<model>)` column as the item (KC-level, as in the paper),
`Correct First Attempt` as the score, and `First Transaction Time` as the order. Check that the
item-only AUC matches the paper's value (al2005 0.723, bd2006 0.685) to confirm your data is
prepared the same way; if it differs, the KC handling is what to adjust (`--kc-col`, or the
multi-KC policy in `prepare_kddcup.py`).

### Notes on the real columns

1. **Trained SAKT bars require training.** Run `train_sakt_long.py` per dataset, or `make train`
   for the full sequence.
2. **Reference bars are published numbers, not computed.** For the real columns the "Reference"
   bar is a hardcoded best-ID-only pyKT benchmark value (`compute_table.py:BEST_REPORTED`), so
   it needs no data -- only the item-only and SAKT bars do.
