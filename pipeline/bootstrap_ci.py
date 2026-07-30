"""Student-level bootstrap confidence intervals for KT metrics.

Students are the independent unit (responses within a student are correlated),
so we resample whole students with replacement and recompute each metric on the
resample. This gives sampling/evaluation uncertainty (not training/seed
variance) for AUC, accuracy and FIG-V (per-student), consistently across the
item-only baseline and the trained model.

FIG-V and accuracy reduce to per-student sums, so they are bootstrapped by
resampling those sums (fast): FIG-V is the mean of each resampled student's own
gain ratio. AUC needs the actual rows, so it resamples the per-student row sets
and recomputes roc_auc each replicate.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from fig._utils import TrainData, compute_baseline

EPS = 1e-12


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def student_bootstrap_metrics(
    y_pred: np.ndarray,
    y: np.ndarray,
    item_id: np.ndarray,
    student_id: np.ndarray,
    y_train: np.ndarray,
    item_id_train: np.ndarray,
    B: int = 500,
    seed: int = 0,
    ci: float = 95.0,
) -> dict[str, tuple[float, float, float]]:
    """Return {metric: (point, lo, hi)} for auc, accuracy, fig_student.

    The FIG baseline (per-item shrunk train rate) is computed exactly as the FIG
    library does, so the bootstrap point estimates match the reported metrics.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)

    # Per-row FIG baseline probability (shrunk train item rate; unseen -> global).
    baseline = compute_baseline(TrainData(y=y_train, item_id=item_id_train))
    n_items = len(baseline.item_vocab)
    enc = np.fromiter(
        (baseline.item_vocab.get(x, n_items) for x in np.asarray(item_id)),
        dtype=np.int64,
        count=len(item_id),
    )
    baseline_prob = baseline.item_baselines[enc]

    # Per-row contributions.
    bce = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    hbase = _binary_entropy(baseline_prob)
    correct = ((y_pred >= 0.5).astype(np.float64) == y).astype(np.float64)

    # Per-student aggregates + row index sets.
    _, inv = np.unique(np.asarray(student_id), return_inverse=True)
    n_students = inv.max() + 1
    s_bce = np.bincount(inv, weights=bce, minlength=n_students)
    s_h = np.bincount(inv, weights=hbase, minlength=n_students)
    s_correct = np.bincount(inv, weights=correct, minlength=n_students)
    s_n = np.bincount(inv, minlength=n_students).astype(np.float64)

    order = np.argsort(inv, kind="stable")
    splits = np.cumsum(s_n.astype(int))[:-1]
    student_rows = np.split(order, splits)

    # Point estimates on the full sample. fig_student is the student-weighted
    # macro-average (each student's own gain ratio, then averaged) -- the
    # paper's primary FIG-V.
    point = {
        "fig_student": float(np.mean(1.0 - s_bce / s_h)),
        "accuracy": s_correct.sum() / s_n.sum(),
        "auc": float(roc_auc_score(y, y_pred)) if len(np.unique(y)) > 1 else float("nan"),
    }

    rng = np.random.default_rng(seed)
    figs_b = np.empty(B)
    acc_b = np.empty(B)
    auc_b = np.empty(B)
    for b in range(B):
        samp = rng.integers(0, n_students, size=n_students)
        figs_b[b] = np.mean(1.0 - s_bce[samp] / s_h[samp])
        acc_b[b] = s_correct[samp].sum() / s_n[samp].sum()
        idx = np.concatenate([student_rows[s] for s in samp])
        yb = y[idx]
        auc_b[b] = roc_auc_score(yb, y_pred[idx]) if len(np.unique(yb)) > 1 else np.nan

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "fig_student": (float(point["fig_student"]), *np.percentile(figs_b, [lo_q, hi_q])),
        "accuracy": (float(point["accuracy"]), *np.percentile(acc_b, [lo_q, hi_q])),
        "auc": (float(point["auc"]), *np.nanpercentile(auc_b, [lo_q, hi_q])),
    }
