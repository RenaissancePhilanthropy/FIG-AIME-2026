"""Train SAKT on a long-format dataset and report test AUC / accuracy / FIG-V.

Reads the canonical long-format splits produced under ``datasets/`` (student_id,
item_id, score, order_idx), reconstructs each student's chronological sequence
(windowed to a fixed length), trains SAKT with BCE, fits a temperature on a
held-out validation split, and evaluates on the test students with the same
train-derived item baseline the item-only row uses -- so the Trained row is an
apples-to-apples comparison against the item-only row.

Appends one row per run to pipeline/out/trained_sakt_ci.csv.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn, optim

from fig.fig_torch import (
    fractional_information_gain_validation_torch,
    precompute_baseline_torch,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "out")
OUT_CSV = os.path.join(RESULTS, "trained_sakt_ci.csv")  # one row appended per dataset

# Fixed training configuration.
ITEM_COL = "item_id"
MAX_SEQ_LEN = 200
BATCH_SIZE = 256
DIM = 64
LR = 3e-4
VAL_FRAC = 0.1
N_BOOT = 500        # student-bootstrap replicates for the 95% CIs
SEED = 42
MAX_EPOCHS = 50     # --max-epochs default (overridable for a quick run)
PATIENCE = 5        # --patience default

# Allow importing the sibling bootstrap_ci module when run as a script.
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from bootstrap_ci import student_bootstrap_metrics  # noqa: E402
from sakt import SAKT, SAKTSequenceDataset, collate_fn, fit_temperature, get_device  # noqa: E402


def build_ordered_sequences(
    df: pd.DataFrame, item_enc: np.ndarray, max_seq_len: int
) -> list:
    """One (student_id, [(item_enc, score), ...]) per <=max_seq_len chronological window."""
    work = pd.DataFrame(
        {
            "student_id": df["student_id"].to_numpy(),
            "item_enc": item_enc,
            "score": df["score"].to_numpy().astype(int),
            "order_idx": df["order_idx"].to_numpy(),
        }
    ).sort_values(["student_id", "order_idx"], kind="stable")

    sequences = []
    for sid, g in work.groupby("student_id", sort=False):
        items = list(zip(g["item_enc"].tolist(), g["score"].tolist()))
        for i in range(0, len(items), max_seq_len):
            sequences.append((int(sid), items[i : i + max_seq_len]))
    return sequences


def encode_items(train_df: pd.DataFrame, test_df: pd.DataFrame, item_col: str):
    """Map item ids to 0..N-1 over the train vocab; unseen test items -> N (unknown)."""
    vocab = {v: i for i, v in enumerate(sorted(train_df[item_col].unique()))}
    n_train_items = len(vocab)
    train_enc = train_df[item_col].map(vocab).to_numpy()
    test_enc = test_df[item_col].map(lambda x: vocab.get(x, n_train_items)).to_numpy()
    # The returned count includes the extra "unknown" slot so unseen test items are valid.
    return train_enc.astype(np.int64), test_enc.astype(np.int64), n_train_items + 1


@torch.no_grad()
def evaluate(model, loader, device, temperature, baseline):
    model.eval()
    probs, ys, items, students = [], [], [], []
    for batch in loader:
        logits = model(batch["input_in"].to(device), batch["input_ex"].to(device))
        mask = batch["mask"].to(device)
        p = torch.sigmoid(logits[mask] / temperature)
        sid = batch["student_ids"].to(device).unsqueeze(1).expand_as(batch["input_ex"].to(device))
        probs.append(p.cpu())
        ys.append(batch["targets"].to(device)[mask].cpu())
        items.append((batch["input_ex"].to(device)[mask] - 1).cpu())  # back to item_enc
        students.append(sid[mask].cpu())

    probs = torch.cat(probs)
    ys = torch.cat(ys)
    items = torch.cat(items)
    students = torch.cat(students)

    auc = float(roc_auc_score(ys.numpy(), probs.numpy())) if ys.unique().numel() > 1 else float("nan")
    acc = float(accuracy_score(ys.numpy(), (probs.numpy() >= 0.5).astype(int)))
    fig = fractional_information_gain_validation_torch(
        y_pred_eval=probs, y_eval=ys, item_id_eval=items,
        student_id_eval=students, baseline=baseline,
    )
    return {
        "auc": auc, "accuracy": acc,
        "fig_v_per_student": float(fig["fig_v"].item()),
        "n_eval": int(ys.numel()),
        # per-row arrays (for the student bootstrap on the test set)
        "y_pred": probs.numpy(), "y": ys.numpy(),
        "item_enc": items.numpy(), "student": students.numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)  # sakt.py's variable_context truncation draws from stdlib random
    device = get_device()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    train_enc, test_enc, n_questions = encode_items(train_df, test_df, ITEM_COL)

    # Item baseline from the FULL train set (matches the item-only row).
    baseline = precompute_baseline_torch(
        torch.tensor(train_df["score"].to_numpy(), dtype=torch.float32),
        torch.tensor(train_enc, dtype=torch.long),
    )

    # Split train students into train/val for temperature + early stopping.
    rng = np.random.default_rng(SEED)
    students = train_df["student_id"].unique()
    val_students = set(rng.choice(students, size=max(1, round(len(students) * VAL_FRAC)), replace=False).tolist())
    is_val = train_df["student_id"].isin(val_students).to_numpy()

    train_seqs = build_ordered_sequences(train_df[~is_val], train_enc[~is_val], MAX_SEQ_LEN)
    val_seqs = build_ordered_sequences(train_df[is_val], train_enc[is_val], MAX_SEQ_LEN)
    test_seqs = build_ordered_sequences(test_df, test_enc, MAX_SEQ_LEN)

    train_ds = SAKTSequenceDataset(train_seqs, n_items=n_questions, max_seq_len=MAX_SEQ_LEN, variable_context=True)
    val_ds = SAKTSequenceDataset(val_seqs, n_items=n_questions, max_seq_len=MAX_SEQ_LEN)
    test_ds = SAKTSequenceDataset(test_seqs, n_items=n_questions, max_seq_len=MAX_SEQ_LEN)

    dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sh, collate_fn=collate_fn)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    model = SAKT(n_questions=n_questions, seq_len=MAX_SEQ_LEN, dim=DIM, heads=8, dropout=0.3, context_dropout=0.3).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    print(f"[{args.dataset}] device={device} n_questions={n_questions} "
          f"train_seqs={len(train_seqs)} val_seqs={len(val_seqs)} test_seqs={len(test_seqs)}")

    best_val_auc, best_state, since_improve = -1.0, None, 0
    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["input_in"].to(device), batch["input_ex"].to(device))
            mask = batch["mask"].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], batch["targets"].to(device)[mask]
            )
            loss.backward()
            opt.step()

        val = evaluate(model, val_loader, device, 1.0, baseline)
        print(f"[{args.dataset}] epoch {epoch:02d} val_auc={val['auc']:.4f} val_fig={val['fig_v_per_student']:+.4f}")
        if val["auc"] > best_val_auc + 1e-4:
            best_val_auc, best_state, since_improve = val["auc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"[{args.dataset}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    temperature = fit_temperature(model, val_loader, device=device)
    test = evaluate(model, test_loader, device, temperature, baseline)
    print(f"[{args.dataset}] TEST auc={test['auc']:.4f} acc={test['accuracy']:.4f} "
          f"fig_student={test['fig_v_per_student']:+.4f} T={temperature:.3f}")

    # 95% student-bootstrap CIs on the test set (same students bootstrapped).
    ci = student_bootstrap_metrics(
        y_pred=test["y_pred"], y=test["y"], item_id=test["item_enc"],
        student_id=test["student"], y_train=train_df["score"].to_numpy(),
        item_id_train=train_enc, B=N_BOOT,
    )
    print(f"[{args.dataset}] CI auc[{ci['auc'][1]:.3f},{ci['auc'][2]:.3f}] "
          f"fig[{ci['fig_student'][1]:+.4f},{ci['fig_student'][2]:+.4f}]")

    os.makedirs(RESULTS, exist_ok=True)
    row = {
        "dataset": args.dataset, "temperature": temperature,
        "val_auc": best_val_auc, "auc": test["auc"], "accuracy": test["accuracy"],
        "fig_v_per_student": test["fig_v_per_student"],
        "n_eval": test["n_eval"],
        "auc_lo": ci["auc"][1], "auc_hi": ci["auc"][2],
        "acc_lo": ci["accuracy"][1], "acc_hi": ci["accuracy"][2],
        "figs_lo": ci["fig_student"][1], "figs_hi": ci["fig_student"][2],
    }
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{args.dataset}] wrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
