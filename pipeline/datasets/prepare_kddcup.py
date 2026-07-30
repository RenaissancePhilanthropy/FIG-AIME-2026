"""Prepare the KDD Cup 2010 datasets (keys: al2005, bd2006) into the canonical long format.

These two are GATED: the raw files come from PSLC DataShop / the KDD Cup 2010 EDM Challenge,
which requires a free account and accepting a data-use agreement. They cannot be downloaded
unattended. Obtain them here, then pass the raw file with --raw:

  KDD Cup 2010 download page:  https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp
  - al2005  -> "Algebra I 2005-2006"        raw training file: algebra_2005_2006_train.txt
  - bd2006  -> "Bridge to Algebra 2006-2007" raw training file: bridge_to_algebra_2006_2007_train.txt

Raw format: tab-separated, one row per student-step transaction, with columns including
"Anon Student Id", "Correct First Attempt" (0/1), "First Transaction Time", and one or more
knowledge-component columns named "KC(<model>)". The paper predicts these at the **KC level**
(the raw problems/steps are far too sparse), so item_id = the KC(<model>) field.

Canonical output: student_id and item_id (integer-encoded from Anon Student Id and the KC
field), score (=Correct First Attempt), order_idx (by First Transaction Time), split by
student (15% test, seed 42). Writes <name>_{train,test}.csv into pipeline/datasets/kddcup_<name>/.

Key assumption to VERIFY: a step's full KC(...)
string is treated as ONE composite item (steps tagged with multiple KCs joined by "~~" become
their own item). After preparing, run baserate_baseline.py and confirm the item-only AUC
matches the target (al2005 ~ 0.723, bd2006 ~ 0.685); if it differs, the KC-handling or KC-model
column is the thing to adjust (see --kc-col).

Auto-discovery: copy each downloaded raw file to its canonical location and it is picked up
automatically (by `make data` and by this script with no --raw):
  pipeline/datasets/kddcup_al2005/algebra_2005_2006_train.txt
  pipeline/datasets/kddcup_bd2006/bridge_to_algebra_2006_2007_train.txt

Run, e.g. (a custom path can still be passed with --raw):
  python pipeline/datasets/prepare_kddcup.py --dataset al2005
  python pipeline/datasets/prepare_kddcup.py --dataset bd2006 --raw /some/other/path.txt
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(_HERE)
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
from dataset_utils import add_order_idx, write_splits  # noqa: E402

# key -> (output subdirectory, output file prefix, expected raw-file name).
# Drop the downloaded raw file at datasets/<subdir>/<raw name> for auto-discovery.
DATASETS = {
    "al2005": ("kddcup_al2005", "al2005", "algebra_2005_2006_train.txt"),
    "bd2006": ("kddcup_bd2006", "bd2006", "bridge_to_algebra_2006_2007_train.txt"),
}
STUDENT_COL = "Anon Student Id"
SCORE_COL = "Correct First Attempt"
TIME_COL = "First Transaction Time"


def default_raw_path(dataset: str) -> str:
    """Canonical location to drop the raw file so it is auto-discovered."""
    subdir, _, raw_name = DATASETS[dataset]
    return os.path.join(_HERE, subdir, raw_name)


def convert(raw_path: str, dataset: str, kc_col: str | None) -> None:
    """Convert a raw KDD Cup txt file to the canonical long-format split."""
    subdir, prefix, _ = DATASETS[dataset]
    out_dir = os.path.join(_HERE, subdir)

    df = pd.read_csv(raw_path, sep="\t", encoding="latin-1", low_memory=False)
    if kc_col is None:
        kc_cols = [c for c in df.columns if c.startswith("KC(")]
        if not kc_cols:
            raise ValueError(f"no KC(...) column found; columns are: {list(df.columns)}")
        kc_col = kc_cols[0]
        print(f"using KC column: {kc_col}")

    df = df[[STUDENT_COL, kc_col, SCORE_COL, TIME_COL]].copy()
    df = df.dropna(subset=[STUDENT_COL, kc_col, SCORE_COL])  # drop steps with no KC
    df = df[df[SCORE_COL].isin([0, 1])]

    # KC string -> contiguous item ids; student string -> contiguous ids.
    df["item_id"] = pd.factorize(df[kc_col].astype(str))[0].astype("int64")
    df["student_id"] = pd.factorize(df[STUDENT_COL].astype(str))[0].astype("int64")
    df["score"] = df[SCORE_COL].astype("int64")
    order = pd.to_datetime(df[TIME_COL], errors="coerce")

    df = add_order_idx(df[["student_id", "item_id", "score"]], order_key=order.to_numpy())
    df = df[["student_id", "item_id", "score", "order_idx"]]
    write_splits(df, out_dir=out_dir, name=prefix, test_frac=0.15, seed=42)


def main() -> None:
    """Parse args and convert one KDD Cup dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--raw",
        default=None,
        help="path to the raw KDD Cup *_train.txt file "
        "(default: the auto-discovered location under datasets/<subdir>/)",
    )
    parser.add_argument(
        "--kc-col",
        default=None,
        help="KC model column to use as item_id (default: the first 'KC(...)' column)",
    )
    args = parser.parse_args()
    raw = args.raw or default_raw_path(args.dataset)
    if not os.path.isfile(raw):
        raise SystemExit(
            f"raw file not found: {raw}\n"
            "Download it from https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp "
            "and copy it to that path (or pass --raw /your/path.txt)."
        )
    convert(raw, args.dataset, args.kc_col)


if __name__ == "__main__":
    main()
