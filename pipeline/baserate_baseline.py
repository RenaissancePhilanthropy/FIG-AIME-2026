"""Item base-rate baseline for knowledge-tracing datasets.

Given a train set and a test set in long format (one row per response with
columns student_id, item_id, score in {0,1}), this predicts every test
response using only the per-item proportion-correct estimated on the train
set. Items unseen in the train set fall back to the global train base rate.

Reports AUC, accuracy (threshold 0.5), and per-student FIG-V, each with a 95%
student-bootstrap CI, on the test set. This is the "how much can you do with
nothing but item difficulty" floor that any real knowledge-tracing model
should beat.
"""

import argparse
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from fig import fractional_information_gain_validation
from fig._utils import BaselineData, TrainData, compute_baseline

# Allow importing the sibling bootstrap_ci module when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from bootstrap_ci import student_bootstrap_metrics  # noqa: E402


@dataclass
class BaselineResult:
    dataset: str
    item_col: str
    n_train_rows: int
    n_test_rows: int
    n_train_items: int
    n_test_items: int
    # Test rows whose item never appeared in train (predicted via global mean).
    n_unseen_item_rows: int
    frac_unseen_item_rows: float
    train_base_rate: float
    test_base_rate: float
    auc: float
    accuracy: float
    # Accuracy of always predicting the majority class on the test set, for
    # context (a base-rate model should at least match this).
    majority_class_accuracy: float
    # Fractional Information Gain (per-student) of these item base-rate
    # predictions, measured against the FIG library's (shrunk) item baseline. For
    # the item-only model this is ~0 by construction; it is the reference floor a
    # trained model beats.
    fig_v_per_student: float
    # 95% student-bootstrap CIs (lo, hi) for the metrics that get error bars.
    auc_lo: float
    auc_hi: float
    acc_lo: float
    acc_hi: float
    # Per-student (macro-average) FIG-V CI -- the paper's primary FIG variant.
    figs_lo: float
    figs_hi: float


def compute_baserate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_col: str = "item_id",
    score_col: str = "score",
    dataset: str = "",
) -> BaselineResult:
    """Predict each test response with the train per-item proportion correct.

    The prediction is the empirical-Bayes-shrunk per-item rate (the exact
    baseline the FIG library uses), so AUC, accuracy and FIG all describe the
    same item-only predictor. Shrinkage (alpha=beta=2) is negligible for
    well-sampled items and only regularizes items with few training responses;
    it makes FIG of the item-only model ~0 by construction (it *is* the FIG
    baseline) rather than catastrophic on sparse items.
    """
    train_items = train_df[item_col].to_numpy()
    train_y = train_df[score_col].to_numpy()
    baseline: BaselineData = compute_baseline(TrainData(y=train_y, item_id=train_items))
    global_rate = float(train_df[score_col].mean())

    # Map each test row's item to its shrunk train baseline; unseen -> global.
    n_items = len(baseline.item_vocab)
    test_items = test_df[item_col].to_numpy()
    enc = np.fromiter(
        (baseline.item_vocab.get(x, n_items) for x in test_items),
        dtype=np.int64,
        count=len(test_items),
    )
    unseen_mask = enc == n_items
    preds = baseline.item_baselines[enc].astype(float)

    y_true = test_df[score_col].to_numpy(dtype=int)

    # AUC is undefined if the test set has a single class.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, preds))

    accuracy = float(accuracy_score(y_true, (preds >= 0.5).astype(int)))
    majority = float(max(y_true.mean(), 1.0 - y_true.mean()))

    # FIG-V of the item base-rate predictions (clipped to (0,1) for the log).
    fig_preds = np.clip(preds, 1e-12, 1 - 1e-12)
    fig = fractional_information_gain_validation(
        y_pred_eval=fig_preds,
        y_eval=y_true,
        item_id_eval=test_df[item_col].to_numpy(),
        student_id_eval=test_df["student_id"].to_numpy(),
        y_train=train_df[score_col].to_numpy(),
        item_id_train=train_df[item_col].to_numpy(),
    )

    # 95% student-bootstrap CIs (same predictor; AUC/accuracy/per-student FIG-V).
    ci = student_bootstrap_metrics(
        y_pred=preds, y=y_true, item_id=test_items,
        student_id=test_df["student_id"].to_numpy(),
        y_train=train_y, item_id_train=train_items,
    )

    return BaselineResult(
        dataset=dataset,
        item_col=item_col,
        n_train_rows=int(len(train_df)),
        n_test_rows=int(len(test_df)),
        n_train_items=int(train_df[item_col].nunique()),
        n_test_items=int(test_df[item_col].nunique()),
        n_unseen_item_rows=int(unseen_mask.sum()),
        frac_unseen_item_rows=float(unseen_mask.mean()),
        train_base_rate=global_rate,
        test_base_rate=float(y_true.mean()),
        auc=auc,
        accuracy=accuracy,
        majority_class_accuracy=majority,
        fig_v_per_student=float(fig["fig_v"]),
        auc_lo=float(ci["auc"][1]), auc_hi=float(ci["auc"][2]),
        acc_lo=float(ci["accuracy"][1]), acc_hi=float(ci["accuracy"][2]),
        figs_lo=float(ci["fig_student"][1]), figs_hi=float(ci["fig_student"][2]),
    )


def run_from_files(
    train_path: str,
    test_path: str,
    item_col: str = "item_id",
    score_col: str = "score",
    dataset: str = "",
) -> BaselineResult:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return compute_baserate_baseline(
        train_df, test_df, item_col=item_col, score_col=score_col, dataset=dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="train CSV (long format)")
    parser.add_argument("--test", required=True, help="test CSV (long format)")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()

    result = run_from_files(
        args.train, args.test, args.item_col, args.score_col, args.dataset
    )
    for key, value in asdict(result).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
