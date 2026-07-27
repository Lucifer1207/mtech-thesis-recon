"""
stage2b_train_teacher_normalization.py
─────────────────────────────────────────────────────────────
Fine-tunes BERT-Base teacher with E8 LATTICE QUANTIZATION + the
paper's row-wise dynamic-range NORMALIZATION applied during training
(QAT), rather than plain FP32 fine-tuning.

This produces the "normalized teacher" checkpoint needed for QAD
Experiment B (teacher ALSO trained with normalization + normalized
student), to test whether a quantization-aware teacher changes how
well a normalized student distills.

Kept as a SEPARATE file from stage2b_train_teacher.py (the original,
plain-FP32 teacher, still used as-is for Experiment A) -- per project
convention, no shared imports between stage scripts, each fully
self-contained.

NORMALIZATION (Kohli et al., Section 3.1.3 / Algorithm 1):
  For weight row i:  sigma_i = std(W_i,:)
                      beta_i = Y_max / (C_b * sigma_i)
                      W_tilde_i,: = beta_i * W_i,:
  Quantize each 8-D sub-block of the scaled row against the E8
  codebook, then invert:  W_hat_i,: = quantized / beta_i.

  Y_max is derived EMPIRICALLY from the codebook's own coordinate
  range (Y_max = codebook.abs().max()), NOT from an assumed (q,M)
  formula -- this is the exact fix already validated for the
  BERT-Base/TinyBERT QAT normalization pipeline (an assumed (q,M)
  pair gave a Y_max 127x too large for this fixed 256-point codebook,
  causing total training collapse the first time this was tried).

  C_b = 5.0 (paper's fixed row-safety constant).

  Since beta is now PER-ROW (not per 32-element block), every 8-D
  sub-vector within a row shares the same beta -- so unlike the
  on-device export format (which needs 32-element blocks with ONE
  shared scale each), training here can split each row directly into
  8-D blocks with no padding at all: both BERT-Base's row width (768)
  and TinyBERT's (312) divide evenly by 8. Export-time 32-element
  block-alignment (if these checkpoints are later pushed on-device)
  is a separate, later concern -- not needed for this training script.

DIAGNOSTIC INSTRUMENTATION (per advisor's request): after building
the codebook, its coordinate range is printed once. After every
epoch, ONE aggregated number (across all quantized layers combined)
reports how many post-normalization weight values (W_tilde, BEFORE
quantization) fall outside that codebook range -- directly testing
whether normalization is actually bringing weights into the
representable region, independent of whether final accuracy is good
or bad.

Saves the FOLDED (quantized, on-lattice) state_dict on every best-F1
checkpoint -- same fix already validated for stage_e8_qad.py (the
live weight is a continuous fp32 shadow; save_pretrained on the raw
model would otherwise persist the wrong, unquantized values).
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

MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
# NOTE (advisor's task 2 -- convergence check): this is the SAME
# epoch count as the original (non-normalized) teacher/student
# scripts. Advisor's hypothesis is that low epoch count MIGHT be why
# normalized results look worse -- watch the printed per-epoch loss
# below; if it's still dropping meaningfully at the final epoch,
# that's evidence favoring "needs more epochs" over "normalization
# itself is the problem". Bump this and re-run if so -- deliberately
# NOT silently increased here, since that's an experimental decision
# for you/advisor to make from the evidence, not one to bake in.
EPOCHS       = 3
LR           = 2e-5
VAL_FRACTION = 0.1
C_B          = 5.0     # paper's fixed row-safety constant

TRAIN_FILE = "pipeline_data/train_clean.jsonl"
TEST_FILE  = "pipeline_data/test_clean.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# E8 codebook (verbatim copy of stage_e8_qad.py's construction --
# must match exactly so the QAD student trained against this teacher
# is using the identical fixed lattice).
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
# Normalized E8-quantized Linear layer (paper's Algorithm 1)
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
    """
    E8-quantized Linear layer with the paper's row-wise dynamic-range
    normalization: beta_i = Y_max / (C_b * sigma_i), applied per
    output row before the 8-D lattice projection, inverted after.
    """
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
        # Y_max grounded EMPIRICALLY in the codebook's own coordinate
        # range -- NOT an assumed (q,M) formula. See module docstring.
        y_max = float(np.max(np.abs(codebook_np)))
        self.register_buffer("Y_max", torch.tensor(y_max))

    def forward(self, x):
        W = self.weight
        O, I = W.shape
        assert I % 8 == 0, f"Row width {I} not divisible by 8 -- would need padding"

        sigma = W.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)  # (O,1)
        beta = self.Y_max / (self.C_b * sigma)                              # (O,1)
        W_tilde = W * beta                                                   # (O,I)

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
    """
    Folds every E8QuantizedLinearNorm's weight into its quantized
    (on-lattice, normalization-inverted) value, on a CLONED state_dict
    -- mirrors stage_e8_qad.py's get_quantized_state_dict() fix
    exactly. Never applied in-place to the live model.
    """
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
    """
    ONE aggregated number (all quantized layers combined, per
    advisor's request): how many post-normalization weight values
    (W_tilde, BEFORE quantization) fall outside the codebook's fixed
    coordinate range.
    """
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
# Dataset / eval (unchanged from stage2b_train_teacher.py)
# ─────────────────────────────────────────────────────────────
class ReConDataset(Dataset):
    def __init__(self, items, tokenizer, max_length):
        self.texts  = [x["text"]  for x in items]
        self.labels = [x["label"] for x in items]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            max_length=self.max_length, padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def print_cm(cm):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"    Confusion Matrix:")
    print(f"    {'':>14} {'Pred Non-leak':>15} {'Pred Leak':>12}")
    print(f"    {'Actual Non-leak':>14} {tn:>15,} {fp:>12,}")
    print(f"    {'Actual Leak':>14} {fn:>15,} {tp:>12,}")
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


def evaluate(model, loader, name=""):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            preds_all.extend(preds.tolist())
            labels_all.extend(batch["label"].numpy().tolist())
            probs_all.extend(probs.tolist())
    elapsed = time.time() - t0

    acc  = accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, zero_division=0)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec  = recall_score(labels_all, preds_all, zero_division=0)
    cm   = confusion_matrix(labels_all, preds_all, labels=[0, 1])

    print(f"\n  [{name}]")
    print(f"    Accuracy:  {acc*100:.2f}%  |  Precision: {prec*100:.2f}%")
    print(f"    Recall:    {rec*100:.2f}%  |  F1:        {f1*100:.2f}%")
    print_cm(cm)
    print(f"    Latency: {(elapsed/len(labels_all))*1000:.3f} ms/sample")

    return {
        "accuracy": acc, "f1": f1, "precision": prec,
        "recall": rec, "confusion_matrix": cm.tolist(),
        "probs": probs_all, "labels": labels_all
    }


def main():
    print("=" * 65)
    print("STAGE 2b (NORMALIZED): BERT-Base teacher, E8 + row-wise")
    print("normalization (Kohli et al. Algorithm 1) applied via QAT")
    print("=" * 65)
    print(f"\nDevice: {device}")

    codebook_np = generate_e8_8bit_codebook()
    codebook_min = float(codebook_np.min())
    codebook_max = float(codebook_np.max())
    print(f"\nCodebook coordinate range: [{codebook_min:.4f}, {codebook_max:.4f}]")
    print(f"C_b = {C_B}, Y_max (empirical, from codebook) = "
          f"{float(np.max(np.abs(codebook_np))):.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    transform_to_e8_lattice_norm(model, codebook_np, C_B)
    model = model.to(device)
    n_quant = sum(1 for m in model.modules() if isinstance(m, E8QuantizedLinearNorm))
    print(f"QAT-wrapped {n_quant} linear layers with normalized E8 quantization "
          f"(classifier head kept full precision)")

    full_train = load_jsonl(TRAIN_FILE)
    test_items = load_jsonl(TEST_FILE)

    random.shuffle(full_train)
    val_n = int(len(full_train) * VAL_FRACTION)
    val_items   = full_train[:val_n]
    train_items = full_train[val_n:]

    print("\nClass distribution per split:")
    for name, items in [("TRAIN", train_items),
                        ("VAL",   val_items),
                        ("TEST",  test_items)]:
        n = len(items)
        leaks = sum(1 for x in items if x["label"] == 1)
        print(f"  {name}: {n:,} total | "
              f"PII={leaks:,} ({100*leaks/n:.2f}%) | "
              f"Non-PII={n-leaks:,} ({100*(n-leaks)/n:.2f}%)")

    train_ds = ReConDataset(train_items, tokenizer, MAX_LENGTH)
    val_ds   = ReConDataset(val_items,   tokenizer, MAX_LENGTH)
    test_ds  = ReConDataset(test_items,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )

    print(f"\nFine-tuning {EPOCHS} epochs...")
    t_start = time.time()
    best_val_f1 = 0.0
    epoch_losses = []

    os.makedirs("teacher_model_normalization", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            ep_loss += out.loss.item()
            if step % 200 == 0:
                print(f"  Ep{epoch+1} step {step}/{len(train_loader)} "
                      f"loss={out.loss.item():.4f}")

        avg = ep_loss / len(train_loader)
        epoch_losses.append(avg)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

        val_res = evaluate(model, val_loader, name=f"Validation epoch {epoch+1}")

        # Diagnostic: aggregated overload stats, this epoch's state
        n_out, n_total, pct_out = compute_overload_stats(model, codebook_min, codebook_max)
        print(f"    Post-normalization overload check: {n_out:,} / {n_total:,} "
              f"weights outside codebook range [{codebook_min:.4f}, {codebook_max:.4f}] "
              f"({pct_out:.2f}%)")

        if val_res["f1"] > best_val_f1:
            best_val_f1 = val_res["f1"]
            quantized_sd = get_quantized_state_dict_norm(model)
            model.save_pretrained("teacher_model_normalization", state_dict=quantized_sd)
            tokenizer.save_pretrained("teacher_model_normalization")
            print(f"    New best val F1={best_val_f1*100:.2f}% -- "
                  f"checkpoint saved (quantized weights folded)")

    print(f"\nTraining time: {(time.time()-t_start)/60:.1f} min")
    if len(epoch_losses) >= 2:
        last_change_pct = 100.0 * abs(epoch_losses[-1] - epoch_losses[-2]) / (abs(epoch_losses[-2]) + 1e-12)
        print(f"Loss change, final epoch vs previous: {last_change_pct:.2f}% "
              f"(large value here suggests loss was still moving -- "
              f"more epochs may help; small value suggests it had "
              f"roughly converged)")

    print("\n" + "=" * 65)
    print("FINAL TEST SET RESULTS (best checkpoint, reloaded)")
    print("=" * 65)
    # Checkpoint already contains FOLDED (quantized) weights -- do NOT
    # re-wrap with transform_to_e8_lattice_norm here (same class of
    # bug already fixed in stage_e8_qad.py: re-wrapping already-
    # quantized weights re-derives normalization from values that are
    # no longer raw, corrupting every layer).
    best_model = AutoModelForSequenceClassification.from_pretrained(
        "teacher_model_normalization"
    ).to(device)
    test_res = evaluate(best_model, test_loader, name="Test (normalized teacher)")
    model_mb = sum(p.numel()*4 for p in best_model.parameters()) / (1024**2)
    print(f"\n  Model size (FP32 weights): {model_mb:.2f} MB")

    with open("pipeline_data/teacher_test_probs_normalization.json", "w") as f:
        json.dump({"probs": test_res["probs"], "labels": test_res["labels"]}, f)

    print("\nGenerating teacher probabilities on TRAIN set for QAD (Experiment B)...")
    best_model.eval()
    train_probs_all, train_labels_all = [], []
    train_ds_ordered = ReConDataset(train_items, tokenizer, MAX_LENGTH)
    train_ordered_loader = DataLoader(train_ds_ordered, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for batch in train_ordered_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = best_model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            train_probs_all.extend(probs.tolist())
            train_labels_all.extend(batch["label"].numpy().tolist())

    with open("pipeline_data/teacher_train_probs_normalization.json", "w") as f:
        json.dump({"probs": train_probs_all, "labels": train_labels_all,
                   "n_train": len(train_items)}, f)

    print(f"  Saved {len(train_probs_all):,} train probs to "
          f"teacher_train_probs_normalization.json")
    print("\nSaved normalized teacher to teacher_model_normalization/")
    print("STAGE 2b (NORMALIZED) COMPLETE")


if __name__ == "__main__":
    main()