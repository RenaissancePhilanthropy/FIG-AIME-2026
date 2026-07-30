"""SAKT model and training helpers, shared by the two SAKT scripts.

Defined once here; both train_sakt_long.py (the Fig 2 "Trained SAKT" bars) and
fig4_calibration_curves.py (the Fig 4 curve) import from it. The two scripts share
the model and these helpers but differ in their data, training settings, and
evaluation protocol.
"""

import random

import torch
from torch import nn, optim


def get_device() -> str:
    """Return the best available device string."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SAKT(nn.Module):
    """Self-attentive knowledge tracing (Pandey and Karypis 2019, arXiv:1907.06837).

    One attention block over a student's answer history. Keys and values are the
    interactions already seen -- a token encoding both the question and whether it
    was answered correctly, hence a vocabulary twice the question count -- while the
    query is the embedding of the question being asked. Position t holds the response
    to question t-1, and a causal mask allows attention only up to t, so a question is
    never scored using its own answer. Index 0 means padding in both vocabularies,
    which is why each is one slot wider than its contents.

    With context_dropout > 0, training also blanks a random share of the history, so
    the model stays usable when only part of a sequence is revealed. That is what the
    Figure 4 protocol does.
    """

    def __init__(self, n_questions, seq_len, dim, heads, dropout, context_dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.context_dropout = context_dropout

        self.interaction_embedding = nn.Embedding(2 * n_questions + 1, dim)
        self.question_embedding = nn.Embedding(n_questions + 1, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)

        self.value_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)

        # batch_first=False: this block works in (seq, batch, dim) order.
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)

        self.ffn_hidden = nn.Linear(dim, dim)
        self.ffn_output = nn.Linear(dim, dim)
        self.readout = nn.Linear(dim, 1)

        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, interactions, questions):
        if self.training and self.context_dropout > 0:
            keep = torch.bernoulli(
                torch.full_like(interactions, 1 - self.context_dropout, dtype=torch.float32)
            ).to(interactions.dtype)
            interactions = interactions * keep

        positions = torch.arange(self.seq_len, device=interactions.device).unsqueeze(0)
        history = self.interaction_embedding(interactions) + self.position_embedding(positions)
        asked = self.question_embedding(questions)

        value = self.value_proj(history).permute(1, 0, 2)
        key = self.key_proj(history).permute(1, 0, 2)
        query = self.query_proj(asked).permute(1, 0, 2)

        attended, _ = self.attention(query, key, value, attn_mask=self.causal_mask,
                                     key_padding_mask=questions == 0)
        attended = self.attention_norm(query + attended).permute(1, 0, 2)

        projected = self.ffn_dropout(self.ffn_output(torch.relu(self.ffn_hidden(attended))))
        return self.readout(self.ffn_norm(projected + attended)).squeeze(-1)


class SAKTSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, student_sequences, n_items, max_seq_len, variable_context=False):
        self.student_sequences = student_sequences
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.variable_context = variable_context

    def __len__(self):
        return len(self.student_sequences)

    def __getitem__(self, idx):
        student_id, seq = self.student_sequences[idx]
        seq_len = min(len(seq), self.max_seq_len)
        input_in = torch.zeros(self.max_seq_len, dtype=torch.long)
        input_ex = torch.zeros(self.max_seq_len, dtype=torch.long)
        targets = torch.zeros(self.max_seq_len, dtype=torch.float32)
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        for t in range(seq_len):
            item_id, correct = seq[t]
            if t > 0:
                prev_item, prev_correct = seq[t - 1]
                input_in[t] = prev_item + self.n_items + 1 if prev_correct == 1 else prev_item + 1
            input_ex[t] = item_id + 1
            targets[t] = float(correct)
            if t > 0:
                mask[t] = True
        if self.variable_context and seq_len > 1:
            k = random.randint(0, seq_len - 1)
            input_in[k + 1 :] = 0
        return {"input_in": input_in, "input_ex": input_ex, "targets": targets, "mask": mask, "student_id": student_id}


def collate_fn(batch):
    return {
        "input_in": torch.stack([b["input_in"] for b in batch]),
        "input_ex": torch.stack([b["input_ex"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "student_ids": torch.tensor([b["student_id"] for b in batch]),
    }


def fit_temperature(model, loader, device):
    """Fit a single temperature on validation data to minimise NLL (LBFGS)."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_in"].to(device), batch["input_ex"].to(device))
            mask = batch["mask"].to(device)
            all_logits.append(logits[mask].cpu())
            all_targets.append(batch["targets"].to(device)[mask].cpu())
    logits_all, targets_all = torch.cat(all_logits), torch.cat(all_targets)
    log_t = nn.Parameter(torch.zeros(1))
    opt = optim.LBFGS([log_t], lr=0.01, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits_all / log_t.exp(), targets_all)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
