"""Compute the item-only baseline table (AUC, accuracy, FIG-V) that Figure 2 plots.

For each dataset we run the item base-rate baseline on the prepared splits (the
per-dataset item granularity is listed in pipeline/DATASETS.md), then:

* AUC / Accuracy / FIG-V (per-student) come from baserate_baseline, whose
  item-only predictions ARE the FIG (shrunk) item baseline -- so the same
  predictor drives every metric.
* For the simulated IRT datasets the "best achievable" reference is the IRT
  oracle: the AUC/accuracy of the true response probabilities sigmoid(theta-d)
  stored in the `true_prob` column (the Bayes-optimal full-information model).
* For real datasets the best-reported AUC/accuracy are published deep-KT results
  (see the note above BEST_REPORTED below).

Writes out/baseline_vs_sota_computed.csv (the Figure 2 input).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from fig import fractional_information_gain_validation

HERE = os.path.dirname(os.path.abspath(__file__))
# Allow importing the sibling baserate_baseline module when run as a script.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from baserate_baseline import run_from_files  # noqa: E402

DATA = os.path.join(HERE, "datasets")
RESULTS = os.path.join(HERE, "out")

# key, dataset dir, file prefix. (Display names live in
# R/fig2_item_only_vs_trained.R's dataset_labels, keyed by this same `key`.)
COLUMNS = [
    ("sim_u2", "sim_irt/sim_irt_unif2", "sim_irt_unif2"),
    ("sim_u5", "sim_irt/sim_irt_unif5", "sim_irt_unif5"),
    ("as15", "assistments2015", "assistments2015"),
    ("al2005", "kddcup_al2005", "al2005"),
    ("bd2006", "kddcup_bd2006", "bd2006"),
    ("nips34", "nips34", "nips34"),
]
# AS2009 is not included: its KC-level evaluation suffers the multi-skill
# row-duplication label leakage the pyKT benchmark warns about, and at problem
# level it is too sparse.

# Reference = the best ID-ONLY model in the pyKT benchmark (Liu et al., 2022) --
# models using only item/KC IDs + responses, the same information class as our
# item-only baseline and SAKT; models with extra inputs are excluded (AKT/simpleKT/
# IEKT use Rasch question-difficulty, GKT a KC graph, DKT-F forgetting time).
# These are the values used for the paper's Figure 2 reference bars: DKT+ on
# ASSISTments/KDD, SAINT on NIPS34. Sim columns = IRT oracle (computed instead).
BEST_REPORTED: dict[str, dict] = {
    "as15": {"auc": 0.728, "acc": 0.750, "model": "DKT+"},
    "al2005": {"auc": 0.817, "acc": 0.810, "model": "DKT+"},
    "bd2006": {"auc": 0.802, "acc": 0.855, "model": "DKT+"},
    "nips34": {"auc": 0.786, "acc": 0.718, "model": "SAINT"},
}


def fmt(x) -> str:
    return "---" if x is None or pd.isna(x) else f"{x:.3f}"


def main() -> None:
    rows = []
    for key, ddir, prefix in COLUMNS:
        train = os.path.join(DATA, ddir, f"{prefix}_train.csv")
        test = os.path.join(DATA, ddir, f"{prefix}_test.csv")
        # Skip datasets whose prepared splits are absent (e.g. real-dataset rows
        # whose source data is not available); the table is built from whatever
        # is present. See pipeline/DATASETS.md.
        if not (os.path.exists(train) and os.path.exists(test)):
            print(f"[skip] {key}: prepared files not found ({prefix}_train/test.csv)")
            continue
        r = run_from_files(train, test, dataset=key)

        # Bayes-optimal reference for the simulated datasets.
        best_auc = best_acc = best_figs = None
        if key.startswith("sim_"):
            # The true generating probability true_prob = sigma(theta - d) is the
            # Bayes-optimal predictor (minimises expected log-loss, perfectly
            # calibrated). Its AUC/accuracy/FIG are the maximum achievable given
            # the irreducible Bernoulli noise -- the ceiling, measured through the
            # same item-base-rate baseline as every other FIG here.
            tdf = pd.read_csv(test)
            train_df = pd.read_csv(train)
            y = tdf["score"].to_numpy()
            p = tdf["true_prob"].to_numpy()
            best_auc = float(roc_auc_score(y, p))
            best_acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
            bfig = fractional_information_gain_validation(
                y_pred_eval=np.clip(p, 1e-12, 1 - 1e-12), y_eval=y,
                item_id_eval=tdf["item_id"].to_numpy(),
                student_id_eval=tdf["student_id"].to_numpy(),
                y_train=train_df["score"].to_numpy(),
                item_id_train=train_df["item_id"].to_numpy(),
            )
            best_figs = float(bfig["fig_v"])
        else:
            br = BEST_REPORTED.get(key, {})
            best_auc, best_acc = br.get("auc"), br.get("acc")

        rows.append(
            {
                "key": key,
                "auc": r.auc, "accuracy": r.accuracy,
                "fig_student": r.fig_v_per_student,
                # item-only 95% student-bootstrap CIs
                "auc_lo": r.auc_lo, "auc_hi": r.auc_hi,
                "acc_lo": r.acc_lo, "acc_hi": r.acc_hi,
                "figs_lo": r.figs_lo, "figs_hi": r.figs_hi,
                "best_auc": best_auc,
                "best_acc": best_acc,
                "best_figs": best_figs,
            }
        )
        print(
            f"{key:8s} AUC={r.auc:.3f} ACC={r.accuracy:.3f} "
            f"FIGs={r.fig_v_per_student:+.4f} "
            f"bestAUC={fmt(best_auc)} bestACC={fmt(best_acc)} bestFIGs={fmt(best_figs)}"
        )

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)

    # Trained SAKT results (if present), keyed by the column key used as --dataset.
    trained_path = os.path.join(RESULTS, "trained_sakt_ci.csv")
    trained = {}
    if os.path.exists(trained_path):
        tdf = pd.read_csv(trained_path).drop_duplicates("dataset", keep="last")
        trained = tdf.set_index("dataset").to_dict("index")
    t_cols = ["t_auc", "t_acc", "t_figs",
              "t_auc_lo", "t_auc_hi", "t_acc_lo", "t_acc_hi",
              "t_figs_lo", "t_figs_hi"]
    for col in t_cols:
        df[col] = None
    for i, rr in df.iterrows():
        t = trained.get(rr["key"])
        if t:
            df.at[i, "t_auc"] = t["auc"]
            df.at[i, "t_acc"] = t["accuracy"]
            df.at[i, "t_figs"] = t["fig_v_per_student"]
            # SAKT 95% CI bounds
            for c in ("auc_lo", "auc_hi", "acc_lo", "acc_hi", "figs_lo", "figs_hi"):
                df.at[i, "t_" + c] = t.get(c)

    # The Fig-2 input: item-only + trained-SAKT + best-reported columns, one row per dataset.
    out_csv = os.path.join(RESULTS, "baseline_vs_sota_computed.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")



if __name__ == "__main__":
    main()
