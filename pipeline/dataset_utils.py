"""Shared helpers for preparing knowledge-tracing datasets.

Canonical long format used across this folder: one row per response with at
least the columns ``student_id``, ``item_id`` and ``score`` (in {0, 1}).
Datasets are split into train/test *by student* (the same student never
appears in both) so the base-rate baseline never sees a test student's
responses.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd


def add_order_idx(df: pd.DataFrame, order_key) -> pd.DataFrame:
    """Sort responses into per-student chronological order and add ``order_idx``.

    ``order_key`` is a column (or array) that increases with time. The returned
    frame is sorted by (student, order_key) and carries a 0-based per-student
    ``order_idx`` so sequence models can reconstruct each student's history.
    """
    df = df.copy()
    df["_order_key"] = order_key
    df = df.sort_values(["student_id", "_order_key"], kind="stable").reset_index(drop=True)
    df["order_idx"] = df.groupby("student_id").cumcount()
    return df.drop(columns="_order_key")


def split_by_student(
    df: pd.DataFrame,
    test_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-format frame into train/test by holding out whole students."""
    rng = np.random.default_rng(seed)
    students = df["student_id"].unique()
    n_test = round(len(students) * test_frac)
    test_students = set(rng.choice(students, size=n_test, replace=False).tolist())

    is_test = df["student_id"].isin(test_students)
    return df[~is_test].copy(), df[is_test].copy()


def write_splits(
    df: pd.DataFrame,
    out_dir: str,
    name: str,
    test_frac: float,
    seed: int,
) -> Tuple[str, str]:
    """Split ``df`` by student and write ``<name>_train.csv`` / ``<name>_test.csv``."""
    train_df, test_df = split_by_student(df, test_frac=test_frac, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"{name}_train.csv")
    test_path = os.path.join(out_dir, f"{name}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"[{name}] {df['student_id'].nunique()} students, {len(df)} responses, "
        f"{df['item_id'].nunique()} items, base rate {df['score'].mean():.3f}"
    )
    print(
        f"[{name}] train: {len(train_df)} rows / {train_df['student_id'].nunique()} students -> {train_path}"
    )
    print(
        f"[{name}] test:  {len(test_df)} rows / {test_df['student_id'].nunique()} students -> {test_path}"
    )
    return train_path, test_path
