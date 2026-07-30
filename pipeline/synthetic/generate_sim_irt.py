"""Generate the simulated 1PL IRT datasets (sim_u2, sim_u5) for the Figure 2 pipeline.

These are the two synthetic "validation" columns of the item-only-vs-trained table: a
dense 1PL IRT response matrix where every student answers every item. They bracket the
real datasets and validate the metric -- widening item difficulty from U[-2,2] to U[-5,5]
spreads items apart and lifts the item-only AUC even though the base rate stays ~0.5.

Generative model (1PL / Rasch): student ability theta ~ N(0, 1); item difficulty
d ~ U[lo, hi]; for every (student, item) pair, true_prob = sigmoid(theta - d) and
score ~ Bernoulli(true_prob). The data is dense (n_students * n_items rows).

Per variant it writes a by-student train/test split (15% test, seed 42) to
``datasets/sim_irt/<name>/<name>_{train,test}.csv`` with columns
``student_id, item_id, score, true_prob, order_idx``.
"""

import os
import sys

import numpy as np
import pandas as pd

# dataset_utils lives one directory up (pipeline/); reuse its canonical splitter.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(_HERE)
if _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)
from dataset_utils import add_order_idx, write_splits  # noqa: E402

# Fixed generation parameters.
SEED = 42
N_STUDENTS = 4000
N_ITEMS = 300
OUT_DIR = os.path.join(_PIPELINE, "datasets", "sim_irt")
# (name, difficulty lower bound, upper bound) for each simulated column.
VARIANTS = [
    ("sim_irt_unif2", -2.0, 2.0),
    ("sim_irt_unif5", -5.0, 5.0),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_students: int,
    n_items: int,
    diff_lo: float,
    diff_hi: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a dense 1PL IRT response table (every student answers every item)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=n_students)
    difficulty = rng.uniform(diff_lo, diff_hi, size=n_items)

    # Dense outer grid: true_prob[s, i] = sigmoid(theta_s - d_i).
    true_prob = sigmoid(theta[:, None] - difficulty[None, :])
    scores = rng.binomial(1, true_prob)

    student_id = np.repeat(np.arange(n_students), n_items)
    item_id = np.tile(np.arange(n_items), n_students)
    df = pd.DataFrame(
        {
            "student_id": student_id.astype(np.int64),
            "item_id": item_id.astype(np.int64),
            "score": scores.reshape(-1).astype(np.int64),
            "true_prob": true_prob.reshape(-1).astype(np.float64),
        }
    )
    # Per-student order index (items are already in ascending order within student).
    return add_order_idx(df, order_key=df["item_id"].to_numpy())


def main() -> None:
    """Generate both sim variants and write their by-student splits."""
    for name, lo, hi in VARIANTS:
        df = generate(N_STUDENTS, N_ITEMS, diff_lo=lo, diff_hi=hi, seed=SEED)
        write_splits(df, out_dir=os.path.join(OUT_DIR, name), name=name, test_frac=0.15, seed=SEED)


if __name__ == "__main__":
    main()
