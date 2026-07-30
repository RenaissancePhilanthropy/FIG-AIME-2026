"""Prepare ASSISTments 2015 (key: as15) into the canonical long format.

This one is GATED: it is not downloaded for you. ASSISTments distribute the data from their own
dataset page, and set out terms of use on the same site:

  2015 ASSISTments Skill Builder Data:
    https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data
  Terms of use:
    https://sites.google.com/site/assistmentsdata/termsofuseforusingdata

Save the CSV to the canonical path below and it is auto-discovered (by `make data` and by this
script with no --raw):

  pipeline/datasets/assistments2015/2015_100_skill_builders_main_problems.csv

Raw columns: user_id, log_id, sequence_id, correct. ASSISTments 2015 has no problem ids --
only sequence_id (100 skill builders) -- so the item granularity is sequence_id (matching the
KT literature and the paper's table). Partial-credit rows (correct not in {0, 1}) are dropped.

Canonical output: student_id, item_id (=sequence_id), score (=correct), order_idx (by log_id),
split by student (15% test, seed 42). Writes assistments2015_{train,test}.csv next to this file.

Run (a custom path can be passed with --raw):
  python pipeline/datasets/assistments2015/prepare_assistments2015.py
"""

import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(os.path.dirname(_HERE))
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
from dataset_utils import add_order_idx, write_splits  # noqa: E402

RAW_CSV = os.path.join(_HERE, "2015_100_skill_builders_main_problems.csv")
PAGE = "https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data"


def convert(raw: str) -> None:
    """Convert the raw CSV to canonical long format and write the by-student split."""
    df = pd.read_csv(raw, encoding="latin-1")
    df = df.rename(
        columns={"user_id": "student_id", "sequence_id": "item_id", "correct": "score"}
    )
    # Keep binary responses only (drop partial credit), then canonical columns.
    df = df[df["score"].isin([0, 1])].copy()
    df["student_id"] = df["student_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["score"] = df["score"].astype("int64")
    df = add_order_idx(df[["student_id", "item_id", "score"]], order_key=df["log_id"])
    df = df[["student_id", "item_id", "score", "order_idx"]]
    write_splits(df, out_dir=_HERE, name="assistments2015", test_frac=0.15, seed=42)


def main() -> None:
    """Parse args and convert the raw ASSISTments 2015 CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default=RAW_CSV,
        help="path to the raw 2015_100_skill_builders_main_problems.csv "
        "(default: the auto-discovered location next to this script)",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.raw):
        raise SystemExit(
            f"raw file not found: {args.raw}\n"
            f"Download the 2015 skill builder data from {PAGE} "
            "and copy it to that path (or pass --raw /your/path.csv)."
        )
    convert(args.raw)


if __name__ == "__main__":
    main()
