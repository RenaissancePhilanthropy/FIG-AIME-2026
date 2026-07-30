"""Compute calibration-curve data for the 'FIG increases with observations' figure (Fig 4).

Trains a SAKT model on the 1PL 5-skill synthetic dataset and evaluates it under the
Skill-3-input protocol, computing FIG-V via the installed ``fig``
library. Writes pipeline/out/calibration_curves.csv -- the input that R/fig4_fig_vs_observations.R
plots.

Seeded (SEED=42); rerunning reproduces the paper's figure (see pipeline/README.md).
"""

import argparse
import csv
import random
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim

from fig._utils import BaselineData
from fig.fig_torch import (
    fractional_information_gain_validation_torch,
    precompute_baseline_torch,
)
from sakt import SAKT, SAKTSequenceDataset, collate_fn, fit_temperature, get_device

# ---------------------------------------------------------------------------
# Paths (relative to this file: pipeline/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(
    _HERE, "datasets", "1pl", "training_1pl_5000_students_200_items.csv",
)
_OUT_DIR = os.path.join(_HERE, "out")
_OUT_CSV = os.path.join(_OUT_DIR, "calibration_curves.csv")

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
DIM = 64
HEADS = 8
DOUT = 0.3
CONTEXT_DOUT = 0.3
LR = 3e-4
BATCH = 256
GRAD_CLIP = 1.0
PATIENCE = 15
# The figure needs a few hundred epochs of training; past roughly 320 it stops
# changing appreciably.
MAX_EPOCHS = 440
VAL_FRAC = 0.1
SEED = 42

# Dataset constants
N_ITEMS = 200
ITEMS_PER_SKILL = 40   # 200 items / 5 skills
TARGET_SKILL_ID = 3    # Skill 3 = items 120-159

# Columns of out/calibration_curves.csv -- exactly what R/fig4_fig_vs_observations.R
# plots: the observation count and the per-student FIG-V (with 95% CI lo/hi) for the
# Skill-3, other-skills, and all-items groups of the Skill-3-input protocol.
FIELDNAMES = [
    "ct",
    "skill3_fig_v", "skill3_fig_v_lo", "skill3_fig_v_hi",
    "other_fig_v", "other_fig_v_lo", "other_fig_v_hi",
    "all_fig_v", "all_fig_v_lo", "all_fig_v_hi",
]


# ---------------------------------------------------------------------------
# Skill-3 partial-input eval
# ---------------------------------------------------------------------------


def build_skill_partial_eval_tensors(
    sequences: list,
    n_items: int,
    n_input: int,
    max_seq_len: int,
    rng: np.random.Generator,
    skill_id: int,
    items_per_skill: int,
) -> dict:
    """Build tensors for skill-specific partial-input evaluation."""
    batch_size = len(sequences)
    input_in = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    input_ex = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    targets = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    eval_mask_skill = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    eval_mask_other = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    student_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, (student_id, seq) in enumerate(sequences):
        student_ids[i] = int(student_id)
        seq_len = min(len(seq), max_seq_len)
        seq_trim = seq[:seq_len]

        skill_items = [
            (item_id, correct)
            for item_id, correct in seq_trim
            if (item_id // items_per_skill) == skill_id
        ]
        other_items = [
            (item_id, correct)
            for item_id, correct in seq_trim
            if (item_id // items_per_skill) != skill_id
        ]

        rng.shuffle(skill_items)
        rng.shuffle(other_items)

        input_items = skill_items[: min(n_input, len(skill_items))]
        remaining_skill = skill_items[min(n_input, len(skill_items)) :]
        ordered = input_items + remaining_skill + other_items

        for t in range(len(ordered)):
            item_id, correct = ordered[t]

            if t > 0 and t <= len(input_items):
                prev_item, prev_correct = ordered[t - 1]
                if prev_correct == 1:
                    input_in[i, t] = prev_item + n_items + 1
                else:
                    input_in[i, t] = prev_item + 1

            input_ex[i, t] = item_id + 1
            targets[i, t] = float(correct)

            if t >= len(input_items) and (item_id // items_per_skill) == skill_id:
                eval_mask_skill[i, t] = True
            if t >= len(input_items) and (item_id // items_per_skill) != skill_id:
                eval_mask_other[i, t] = True

    return {
        "input_in": input_in,
        "input_ex": input_ex,
        "targets": targets,
        "eval_mask_skill": eval_mask_skill,
        "eval_mask_other": eval_mask_other,
        "student_ids": student_ids,
    }


def evaluate_skill_partial_inputs(
    model: SAKT,
    sequences: list,
    n_items: int,
    n_input: int,
    max_seq_len: int,
    baseline: BaselineData,
    device: str,
    seed: int,
    skill_id: int,
    items_per_skill: int,
    temperature: float = 1.0,
) -> dict:
    """Evaluate skill-specific partial-input protocol on holdout sequences."""
    model.eval()
    rng = np.random.default_rng(seed)
    batch = build_skill_partial_eval_tensors(
        sequences,
        n_items,
        n_input,
        max_seq_len,
        rng=rng,
        skill_id=skill_id,
        items_per_skill=items_per_skill,
    )
    input_in = batch["input_in"].to(device)
    input_ex = batch["input_ex"].to(device)
    targets = batch["targets"].to(device)
    eval_mask_skill = batch["eval_mask_skill"].to(device)
    eval_mask_other = batch["eval_mask_other"].to(device)
    student_ids = batch["student_ids"].to(device)

    with torch.no_grad():
        logits = model(input_in, input_ex)
        probs = torch.sigmoid(logits / temperature)

        def collect(mask: torch.Tensor) -> dict:
            # Collect valid predictions and targets for items under this mask.
            valid_targets = targets[mask]
            valid_probs = probs[mask]
            valid_item_ids = input_ex[mask] - 1
            student_ids_expanded = student_ids.unsqueeze(1).expand_as(input_ex)
            valid_student_ids = student_ids_expanded[mask]

            if valid_targets.numel() == 0:
                return {
                    "fig_v": float("nan"),
                    "fig_v_lo": float("nan"),
                    "fig_v_hi": float("nan"),
                    "n_targets": 0,
                }

            # Call FIG-V directly to access per-student FIG values for bootstrap CIs.
            fig_v_result = fractional_information_gain_validation_torch(
                y_pred_eval=valid_probs,
                y_eval=valid_targets,
                item_id_eval=valid_item_ids,
                student_id_eval=valid_student_ids,
                baseline=baseline,
            )
            fig_v = float(fig_v_result["fig_v"].item())

            # Student bootstrap: resample students with replacement to get 95% CI.
            fig_by_student = fig_v_result["fig_v_by_student"].cpu().numpy()
            vals = fig_by_student[np.isfinite(fig_by_student)]
            bs_rng = np.random.default_rng(12345)
            boot_means = np.array([
                bs_rng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(2000)
            ])
            fig_v_lo = float(np.percentile(boot_means, 2.5))
            fig_v_hi = float(np.percentile(boot_means, 97.5))

            return {
                "fig_v": fig_v,
                "fig_v_lo": fig_v_lo,
                "fig_v_hi": fig_v_hi,
                "n_targets": int(valid_targets.numel()),
            }

        skill_metrics = collect(eval_mask_skill)
        other_metrics = collect(eval_mask_other)
        # Union mask: all held-out items (revealed skill items are excluded by
        # construction; this is every item the model must predict in this setting).
        all_metrics = collect(eval_mask_skill | eval_mask_other)

    return {"skill": skill_metrics, "other": other_metrics, "all": all_metrics}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max training epochs (default {MAX_EPOCHS}; lower for a quick run).")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early-stopping patience (default {PATIENCE}).")
    args = parser.parse_args()
    max_epochs, patience = args.max_epochs, args.patience

    torch.manual_seed(SEED)
    random.seed(SEED)  # sakt.py's variable_context truncation draws from stdlib random
    device = get_device()
    print(f"device={device}")

    # Load data -- columns: student_id, item_id, score, prob_correct
    df = pd.read_csv(_DATASET)
    print(f"loaded {len(df)} rows, {df['student_id'].nunique()} students, {df['item_id'].nunique()} items")

    # Build per-student sequences, shuffling each with a fixed seed: the generator writes
    # a student's responses skill-blocked, and training should see mixed-skill order,
    # like the eval-time sequences.
    shuffle_rng = np.random.default_rng(SEED)
    sequences: list[tuple[int, list[tuple[int, int]]]] = []
    for sid, g in df.sort_values("student_id", kind="stable").groupby("student_id", sort=False):
        items = list(zip(g["item_id"].astype(int), g["score"].astype(int)))
        shuffle_rng.shuffle(items)
        sequences.append((int(sid), items))

    n_students = len(sequences)
    max_seq_len = max(len(seq) for _, seq in sequences)
    print(f"n_students={n_students} max_seq_len={max_seq_len}")

    # Deterministic split: 80% train pool, 20% holdout.
    perm = torch.randperm(n_students, generator=torch.Generator().manual_seed(SEED)).tolist()
    split_idx = int(0.8 * n_students)
    train_pool_indices = set(perm[:split_idx])
    holdout_indices = set(perm[split_idx:])

    train_pool_seqs = [sequences[i] for i in range(n_students) if i in train_pool_indices]
    holdout_seqs = [sequences[i] for i in range(n_students) if i in holdout_indices]
    print(f"train_pool={len(train_pool_seqs)} holdout={len(holdout_seqs)}")

    # Split train pool into inner train / inner val for early stopping + temperature.
    rng = np.random.default_rng(SEED)
    n_inner_val = max(1, round(len(train_pool_seqs) * VAL_FRAC))
    val_choice = rng.choice(len(train_pool_seqs), size=n_inner_val, replace=False)
    val_set = set(val_choice.tolist())
    inner_train_seqs = [s for i, s in enumerate(train_pool_seqs) if i not in val_set]
    inner_val_seqs = [s for i, s in enumerate(train_pool_seqs) if i in val_set]
    print(f"inner_train={len(inner_train_seqs)} inner_val={len(inner_val_seqs)}")

    # FIG baseline from all train-pool responses (4000 students x 200 items = 800k).
    y_train_list, item_id_train_list = [], []
    for _, seq in train_pool_seqs:
        for item_id, correct in seq:
            item_id_train_list.append(item_id)
            y_train_list.append(correct)
    y_train = torch.tensor(y_train_list, dtype=torch.float32)
    item_id_train = torch.tensor(item_id_train_list, dtype=torch.long)
    baseline = precompute_baseline_torch(y_train, item_id_train)
    print(f"baseline computed from {len(y_train)} train-pool observations")

    # Datasets and loaders for training.
    os.makedirs(_OUT_DIR, exist_ok=True)
    train_ds = SAKTSequenceDataset(inner_train_seqs, n_items=N_ITEMS, max_seq_len=max_seq_len, variable_context=True)
    val_ds = SAKTSequenceDataset(inner_val_seqs, n_items=N_ITEMS, max_seq_len=max_seq_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    # Build SAKT model -- n_questions=N_ITEMS because all item ids are 0..199 contiguous.
    model = SAKT(n_questions=N_ITEMS, seq_len=max_seq_len, dim=DIM, heads=HEADS,
                 dropout=DOUT, context_dropout=CONTEXT_DOUT).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    # Training loop with early stopping on validation AUC.
    best_val_auc, best_state, since_improve = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["input_in"].to(device), batch["input_ex"].to(device))
            mask = batch["mask"].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], batch["targets"].to(device)[mask]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # Evaluate on inner val to track AUC for early stopping.
        model.eval()
        val_logits_list, val_targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                lg = model(batch["input_in"].to(device), batch["input_ex"].to(device))
                mk = batch["mask"].to(device)
                val_logits_list.append(lg[mk].cpu())
                val_targets_list.append(batch["targets"].to(device)[mk].cpu())
        val_logits_cat = torch.cat(val_logits_list)
        val_targets_cat = torch.cat(val_targets_list)
        val_probs = torch.sigmoid(val_logits_cat)
        val_auc = float(roc_auc_score(val_targets_cat.numpy(), val_probs.numpy())) if val_targets_cat.unique().numel() > 1 else float("nan")

        print(f"epoch {epoch:03d} val_auc={val_auc:.4f}", flush=True)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                print(f"early stop at epoch {epoch} (best_val_auc={best_val_auc:.4f})")
                break

    # Restore best weights and fit temperature on inner val.
    if best_state is not None:
        model.load_state_dict(best_state)
    temperature = fit_temperature(model, val_loader, device=device)
    print(f"temperature={temperature:.4f}")

    # ---------------------------------------------------------------------------
    # Evaluate on 1000 holdout students
    # ---------------------------------------------------------------------------
    rows: list[dict] = []

    # Skill-3 partial-input eval: reveal n_input of each holdout student's 40 Skill-3
    # responses and evaluate on the held-out rest.
    for n_input in [1, 2, 4, 8, 16, 32]:
        skill_result = evaluate_skill_partial_inputs(
            model=model,
            sequences=holdout_seqs,
            n_items=N_ITEMS,
            n_input=n_input,
            max_seq_len=max_seq_len,
            baseline=baseline,
            device=device,
            seed=2000 + n_input,
            skill_id=TARGET_SKILL_ID,
            items_per_skill=ITEMS_PER_SKILL,
            temperature=temperature,
        )
        skill_m = skill_result["skill"]
        other_m = skill_result["other"]
        all_m = skill_result["all"]
        print(
            f"val_skill3_input ct={n_input}: "
            f"skill3_fig_v={skill_m['fig_v']:+.4f} skill3_n={skill_m['n_targets']} "
            f"other_fig_v={other_m['fig_v']:+.4f} other_n={other_m['n_targets']} "
            f"all_fig_v={all_m['fig_v']:+.4f} all_n={all_m['n_targets']}"
        )
        rows.append({
            "ct": n_input,
            "skill3_fig_v": skill_m["fig_v"],
            "skill3_fig_v_lo": skill_m["fig_v_lo"],
            "skill3_fig_v_hi": skill_m["fig_v_hi"],
            "other_fig_v": other_m["fig_v"],
            "other_fig_v_lo": other_m["fig_v_lo"],
            "other_fig_v_hi": other_m["fig_v_hi"],
            "all_fig_v": all_m["fig_v"],
            "all_fig_v_lo": all_m["fig_v_lo"],
            "all_fig_v_hi": all_m["fig_v_hi"],
        })

    # Write output CSV.
    with open(_OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {_OUT_CSV}")


if __name__ == "__main__":
    main()
