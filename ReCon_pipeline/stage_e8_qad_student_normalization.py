"""
stage_e8_qad_normalization.py
─────────────────────────────────────────────────────────────
True E8 Lattice QAD for TinyBERT, with the paper's row-wise dynamic-
range NORMALIZATION (Kohli et al., Section 3.1.3 / Algorithm 1)
applied to the student -- to directly test advisor's question of
whether normalization helps or hurts QAD, matching the same
methodology already used to test this for QAT.

Kept as a SEPARATE file from stage_e8_qad.py (the original,
non-normalized QAD, left completely untouched) -- per project
convention, no shared imports between stage scripts.

Supports BOTH of advisor's requested experiments via one flag below:
  EXPERIMENT = "A": teacher trained normally (plain FP32, as before)
                    + student trained WITH normalization
  EXPERIMENT = "B": teacher ALSO trained with normalization
                    (stage2b_train_teacher_normalization.py)
                    + student trained WITH normalization

NORMALIZATION (identical formula/derivation to
stage2b_train_teacher_normalization.py -- see that file's docstring
for the full derivation and the Y_max-grounding fix):
  beta_i = Y_max / (C_b * sigma_i), per output ROW, Y_max grounded
  empirically in the codebook's own coordinate range, C_b = 5.0.
  Row split directly into 8-D blocks (no padding needed -- both
  BERT-Base's 768 and TinyBERT's 312 row widths divide evenly by 8).

DIAGNOSTIC INSTRUMENTATION (per advisor's request): codebook
coordinate range printed once. After every epoch, ONE aggregated
number (all quantized student layers combined) reports how many
post-normalization weight values fall outside that range.

Saves the FOLDED (quantized) state_dict on every best-F1 checkpoint,
and the final test evaluation does NOT re-wrap the reloaded
checkpoint -- both fixes already validated on stage_e8_qad.py.
"""

import json
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from itertools import product

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ═══════════════════════════════════════════════════════════
# EXPERIMENT SWITCH -- change this one line to run either of
# advisor's two requested experiments. Each writes to its own
# distinctly-named output dir, so both can exist side by side.
# ═══════════════════════════════════════════════════════════
EXPERIMENT = "A"    # "A" or "B" -- see module docstring

if EXPERIMENT == "A":
    TEACHER_MODEL_DIR   = "teacher_model"                                    # existing, plain FP32
    TEACHER_TRAIN_PROBS = "pipeline_data/teacher_train_probs.json"           # existing
    STUDENT_SAVE_DIR    = "student_model_lattice_e8_normalization_expA"
elif EXPERIMENT == "B":
    TEACHER_MODEL_DIR   = "teacher_model_normalization"                      # from stage2b_train_teacher_normalization.py
    TEACHER_TRAIN_PROBS = "pipeline_data/teacher_train_probs_normalization.json"
    STUDENT_SAVE_DIR    = "student_model_lattice_e8_normalization_expB"
else:
    raise ValueError(f"EXPERIMENT must be 'A' or 'B', got {EXPERIMENT!r}")

STUDENT_MODEL     = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH        = 128
BATCH_SIZE        = 16
# NOTE (advisor's task 2 -- convergence check): same epoch count as
# the original (non-normalized) QAD script. Watch the printed
# per-epoch loss below to judge convergence -- deliberately not
# silently bumped here; see stage2b_train_teacher_normalization.py's
# matching note for the same reasoning.
EPOCHS            = 5
LR                = 4e-5
TEMPERATURE       = 2.0
ALPHA_DISTILL     = 0.5
C_B               = 5.0     # paper's fixed row-safety constant

TRAIN_FILE = "pipeline_data/train_clean.jsonl"
TEST_FILE  = "pipeline_data/test_clean.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# E8 codebook (verbatim copy of stage_e8_qad.py's construction)
# ─────────────────────────────────────────────────────────────
def is_e8_point(x, tol=1e-6):
    x = np.array(x, dtype=float)
    is_integer = np.all(np.abs(x - np.round(x)) < tol)
    is_half_integer = np.all(np.abs(x - np.round(x - 0.5) - 0.5) < tol)
    sum_val = np.sum(x)
    is_even_sum = (np.abs(sum_val - np.round(sum_val)) < tol and
                   int(np.round(sum_val)) % 2 == 0)
    return (is_integer or is_half_integer) and is_even_sum


def generate_e8_8bit_codebook():
    print("Generating E8 lattice points for 8-bit codebook layout...")
    candidates = []
    for coords in product([-2, -1, 0, 1, 2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    for coords in product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    candidates.sort(key=lambda item: item[0])
    codebook = [item[1] for item in candidates[:256]]
    return np.array(codebook, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Normalized E8-quantized Linear layer (identical to
# stage2b_train_teacher_normalization.py's -- duplicated here per
# project convention of self-contained stage scripts)
# ─────────────────────────────────────────────────────────────
class E8LatticeQuantizer(nn.Module):
    def __init__(self, codebook_np):
        super().__init__()
        self.register_buffer("codebook", torch.tensor(codebook_np, dtype=torch.float32))

    def forward(self, x):
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        cb_norm = torch.sum(self.codebook ** 2, dim=1, keepdim=True).T
        distances = x_norm - 2 * torch.matmul(x, self.codebook.T) + cb_norm
        indices = torch.argmin(distances, dim=1)
        return self.codebook[indices]


class E8QuantizedLinearNorm(nn.Module):
    def __init__(self, orig_linear, codebook_np, C_b=5.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.weight = nn.Parameter(orig_linear.weight.data.clone())
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        self.quantizer = E8LatticeQuantizer(codebook_np)
        self.C_b = C_b
        y_max = float(np.max(np.abs(codebook_np)))
        self.register_buffer("Y_max", torch.tensor(y_max))

    def forward(self, x):
        W = self.weight
        O, I = W.shape
        assert I % 8 == 0, f"Row width {I} not divisible by 8 -- would need padding"

        sigma = W.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
        beta = self.Y_max / (self.C_b * sigma)
        W_tilde = W * beta

        W_8d = W_tilde.reshape(-1, 8)
        W_quant_8d = self.quantizer(W_8d)
        W_quant = W_quant_8d.reshape(O, I)

        W_reconstructed = W_quant / beta

        W_final = W + (W_reconstructed - W).detach()
        return F.linear(x, W_final, self.bias)


def transform_to_e8_lattice_norm(model, codebook_np, C_b=5.0):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if "classifier" in name:
                continue
            setattr(model, name, E8QuantizedLinearNorm(child, codebook_np, C_b))
        else:
            transform_to_e8_lattice_norm(child, codebook_np, C_b)


def get_quantized_state_dict_norm(model):
    sd = model.state_dict()
    quantized_sd = {k: v.clone() for k, v in sd.items()}
    for name, module in model.named_modules():
        if isinstance(module, E8QuantizedLinearNorm):
            W = module.weight.data
            O, I = W.shape
            sigma = W.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
            beta = module.Y_max / (module.C_b * sigma)
            W_tilde = W * beta
            W_8d = W_tilde.reshape(-1, 8)
            W_quant_8d = module.quantizer(W_8d)
            W_quant = W_quant_8d.reshape(O, I)
            W_folded = W_quant / beta

            weight_key = f"{name}.weight"
            assert weight_key in quantized_sd, f"key {weight_key} missing"
            quantized_sd[weight_key] = W_folded.detach().cpu().clone()
    return quantized_sd


def compute_overload_stats(model, codebook_min, codebook_max):
    """ONE aggregated number across all quantized student layers."""
    total_outside = 0
    total_weights = 0
    for module in model.modules():
        if isinstance(module, E8QuantizedLinearNorm):
            W = module.weight.data
            sigma = W.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
            beta = module.Y_max / (module.C_b * sigma)
            W_tilde = W * beta
            outside = ((W_tilde < codebook_min) | (W_tilde > codebook_max)).sum().item()
            total_outside += outside
            total_weights += W_tilde.numel()
    pct = 100.0 * total_outside / total_weights if total_weights > 0 else 0.0
    return total_outside, total_weights, pct


# ─────────────────────────────────────────────────────────────
# Dataset / eval (unchanged from stage_e8_qad.py)
# ─────────────────────────────────────────────────────────────
class QADDataset(Dataset):
    def __init__(self, items, tokenizer, max_len, teacher_probs=None):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.teacher_probs = teacher_probs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }
        if self.teacher_probs is not None:
            out["teacher_prob"] = torch.tensor(self.teacher_probs[idx], dtype=torch.float32)
        return out


def evaluate_model(model, loader, description="Evaluation"):
    model.eval()
    all_labels, all_preds = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            all_labels.extend(batch["label"].tolist())
            all_preds.extend(preds)

    latency = ((time.time() - start_time) / len(loader.dataset)) * 1000
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n[{description}]")
    print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}%")
    print(f"  Recall:   {rec*100:.2f}% | F1 Score:  {f1*100:.2f}%")
    print(f"  Latency:  {latency:.3f} ms/sample")
    print(f"  Confusion Matrix:\n{cm}")
    return {"f1": f1, "accuracy": acc}


def main():
    print("=" * 65)
    print(f"STAGE 3b (NORMALIZED): E8 Lattice QAD -- Experiment {EXPERIMENT}")
    print(f"  Teacher dir: {TEACHER_MODEL_DIR}/")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

    raw_train_items = []
    with open(TRAIN_FILE, "r") as f:
        for line in f:
            raw_train_items.append(json.loads(line))

    with open(TEACHER_TRAIN_PROBS, "r") as f:
        teacher_data = json.load(f)
    train_probs = teacher_data["probs"]

    val_size = int(0.1 * len(raw_train_items))
    train_size = len(raw_train_items) - val_size
    train_items = raw_train_items[:train_size]
    val_items   = raw_train_items[train_size:]

    print(f"Aligned Slice Sizes -> Train: {len(train_items)} | Teacher Targets: {len(train_probs)}")
    assert len(train_items) == len(train_probs), "Index sizes out of sync with teacher layout!"

    train_dataset = QADDataset(train_items, tokenizer, MAX_LENGTH, teacher_probs=train_probs)
    val_dataset   = QADDataset(val_items, tokenizer, MAX_LENGTH)

    test_items = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            test_items.append(json.loads(line))
    test_dataset = QADDataset(test_items, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    labels = [item["label"] for item in train_items]
    neg_count = labels.count(0)
    pos_count = labels.count(1)
    class_weights = torch.tensor([1.0, float(neg_count) / pos_count]).to(device)
    print(f"Computed Imbalance Penalties -> NonLeak: 1.0, Leak: {class_weights[1].item():.3f}")

    codebook_np = generate_e8_8bit_codebook()
    codebook_min = float(codebook_np.min())
    codebook_max = float(codebook_np.max())
    print(f"\nCodebook coordinate range: [{codebook_min:.4f}, {codebook_max:.4f}]")
    print(f"C_b = {C_B}, Y_max (empirical, from codebook) = "
          f"{float(np.max(np.abs(codebook_np))):.4f}")

    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL, num_labels=2)
    transform_to_e8_lattice_norm(student, codebook_np, C_B)
    student = student.to(device)
    n_quant = sum(1 for m in student.modules() if isinstance(m, E8QuantizedLinearNorm))
    print(f"QAT-wrapped {n_quant} linear layers with normalized E8 quantization "
          f"(classifier head kept full precision)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_val_f1 = 0.0
    t_start = time.time()
    epoch_losses = []

    os.makedirs(STUDENT_SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target_labels = batch["label"].to(device)
            t_probs = batch["teacher_prob"].to(device)

            outputs = student(input_ids=ids, attention_mask=mask)
            s_logits = outputs.logits

            loss_ce = F.cross_entropy(s_logits, target_labels, weight=class_weights)
            s_log_probs = F.log_softmax(s_logits / TEMPERATURE, dim=1)
            loss_kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (TEMPERATURE ** 2)

            loss = (1.0 - ALPHA_DISTILL) * loss_ce + ALPHA_DISTILL * loss_kl
            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 300 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] | Step {step}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
        val_metrics = evaluate_model(student, val_loader, description=f"Validation Epoch {epoch+1}")

        n_out, n_total, pct_out = compute_overload_stats(student, codebook_min, codebook_max)
        print(f"    Post-normalization overload check: {n_out:,} / {n_total:,} "
              f"weights outside codebook range [{codebook_min:.4f}, {codebook_max:.4f}] "
              f"({pct_out:.2f}%)")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            quantized_sd = get_quantized_state_dict_norm(student)
            student.save_pretrained(STUDENT_SAVE_DIR, state_dict=quantized_sd)
            print(f"  New best checkpoint (quantized weights folded)! Best Val F1: {best_val_f1*100:.2f}%")

    print(f"\nTraining completed in {(time.time() - t_start)/60:.2f} minutes.")
    if len(epoch_losses) >= 2:
        last_change_pct = 100.0 * abs(epoch_losses[-1] - epoch_losses[-2]) / (abs(epoch_losses[-2]) + 1e-12)
        print(f"Loss change, final epoch vs previous: {last_change_pct:.2f}% "
              f"(large value suggests more epochs may help; small value "
              f"suggests roughly converged)")

    print("\n" + "=" * 65)
    print(f"RUNNING FINAL TEST EVALUATION ON BEST CHECKPOINT (Experiment {EXPERIMENT})")
    print("=" * 65)
    # Checkpoint already contains FOLDED weights -- do NOT re-wrap
    # with transform_to_e8_lattice_norm here (same bug class already
    # fixed in stage_e8_qad.py's final-eval section).
    best_student = AutoModelForSequenceClassification.from_pretrained(STUDENT_SAVE_DIR)
    best_student = best_student.to(device)
    evaluate_model(best_student, test_loader,
                  description=f"Final E8 Lattice QAD (Normalized) Student Test -- Experiment {EXPERIMENT}")

    print(f"\nSaved best normalized QAD student to {STUDENT_SAVE_DIR}/")
    print("STAGE 3b (NORMALIZED) COMPLETE")


if __name__ == "__main__":
    main()