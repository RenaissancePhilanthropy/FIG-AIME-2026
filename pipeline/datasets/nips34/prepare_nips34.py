"""Prepare the Eedi NeurIPS 2020 Tasks 3 & 4 dataset (key: nips34) into long format.

Source (no login): the NeurIPS 2020 Education Challenge public data bundle
  https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip   (~657 MB)
Eedi describes the dataset and its licence conditions at
  https://www.eedischool.com/projects/neurips-education-challenge

Inside the zip, Tasks 3 & 4 share the training file ``train_data/train_task_3_4.csv`` with
columns UserId, QuestionId, AnswerId, IsCorrect (+ CorrectAnswer, AnswerValue), and per-answer
timestamps in ``metadata/answer_metadata_task_3_4.csv`` (AnswerId, DateAnswered). The item
granularity is the question (QuestionId), matching the paper's table and the pyKT NIPS34
benchmark.

Canonical output: student_id (=UserId), item_id (=QuestionId), score (=IsCorrect),
order_idx (by DateAnswered when available, else AnswerId), split by student (15% test, seed 42).
Writes nips34_{train,test}.csv next to this file.

Run: python pipeline/datasets/nips34/prepare_nips34.py
NOTE: downloads ~657 MB; set --zip to a pre-downloaded data.zip to skip the download.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(os.path.dirname(_HERE))
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
from dataset_utils import add_order_idx, write_splits  # noqa: E402

URL = "https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
ZIP_PATH = os.path.join(_HERE, "data.zip")


def _member(zf: zipfile.ZipFile, suffix: str) -> str:
    """Return the first archive member whose name ends with ``suffix``."""
    return next(n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix))


def download(zip_path: str) -> None:
    """Download the challenge data.zip to ``zip_path`` if not already present."""
    if os.path.exists(zip_path):
        print(f"using existing zip: {zip_path}")
        return
    print(f"downloading {URL} (~657 MB) -> {zip_path}")
    urllib.request.urlretrieve(URL, zip_path)
    print("download complete")


def main() -> None:
    """Download, convert Tasks 3&4 to canonical long format, write by-student split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=ZIP_PATH, help="path to data.zip (downloaded if absent)")
    args = parser.parse_args()

    download(args.zip)
    with zipfile.ZipFile(args.zip) as zf:
        with zf.open(_member(zf, "train_task_3_4.csv")) as f:
            df = pd.read_csv(f, usecols=["UserId", "QuestionId", "IsCorrect", "AnswerId"])
        # Per-answer timestamp for chronological ordering (optional but preferred).
        try:
            with zf.open(_member(zf, "answer_metadata_task_3_4.csv")) as f:
                meta = pd.read_csv(f, usecols=["AnswerId", "DateAnswered"])
            df = df.merge(meta, on="AnswerId", how="left")
        except (StopIteration, ValueError, KeyError):
            pass

    df = df.rename(
        columns={"UserId": "student_id", "QuestionId": "item_id", "IsCorrect": "score"}
    )
    df = df[df["score"].isin([0, 1])].copy()
    df["student_id"] = df["student_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["score"] = df["score"].astype("int64")
    # Order by timestamp when the metadata merged, else by AnswerId (computed after the
    # filter above so it stays aligned with df).
    if "DateAnswered" in df:
        order = pd.to_datetime(df["DateAnswered"], errors="coerce")
    else:
        order = df["AnswerId"]
    df = add_order_idx(df[["student_id", "item_id", "score"]], order_key=order.to_numpy())
    df = df[["student_id", "item_id", "score", "order_idx"]]
    write_splits(df, out_dir=_HERE, name="nips34", test_frac=0.15, seed=42)


if __name__ == "__main__":
    main()
