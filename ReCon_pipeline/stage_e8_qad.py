"""
stage3b_e8_lattice_qad.py (v2 -- fixed checkpoint-save bug)
─────────────────────────────────────────────────────────────
True E8 Lattice Quantization-Aware Distillation (QAD) Script.
Replaces scalar uniform simulation with explicit 8D block projection.

Block Format (C++ Struct Aligned):
  - 32 weights group together.
  - 1 floating-point scale factor Δ per 32 elements.
  - Divided into 4 consecutive 8D vectors mapped directly onto 
    the 256-element E8 8-bit codebook.
  - Uses Straight-Through Estimation (STE) for autograd.

FIX vs v1:
  E8QuantizedLinear.forward() computes quantization transiently on
  every forward pass (STE), but never wrote that quantized value back
  into self.weight. student.save_pretrained(...) therefore saved the
  raw, continuous, UNQUANTIZED shadow weight -- not what
  evaluate_model() was actually measuring live. Downstream export
  scripts loading that checkpoint were doing fresh post-hoc
  quantization on non-lattice-aligned weights, which is why SNR came
  out low (~3.87dB) despite training/eval F1 numbers being correct.

  v2 adds get_quantized_state_dict(), which folds every
  E8QuantizedLinear's weight into its quantized (on-lattice)
  reconstruction on a CLONED copy of the state_dict -- never in-place
  on the live model, since doing so in-place would destroy the fp32
  shadow weight needed for further AdamW updates (the same class of
  bug already fixed once in this thesis's stage_qat_all.py). The
  checkpoint saved via save_pretrained(..., state_dict=quantized_sd)
  now genuinely contains lattice-quantized values, directly exportable
  and decomposable losslessly (matching the approach validated for
  the BERT-Base/TinyBERT QAT export pipeline).
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

# ─────────────────────────────────────────────────────────────
# 1. Reproducibility & Config
# ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

STUDENT_MODEL     = "huawei-noah/TinyBERT_General_4L_312D"
TEACHER_MODEL_DIR = "teacher_model"
MAX_LENGTH        = 128
BATCH_SIZE        = 16
EPOCHS            = 5
LR                = 4e-5
TEMPERATURE       = 2.0
ALPHA_DISTILL     = 0.5

TRAIN_FILE = "pipeline_data/train_clean.jsonl"
TEST_FILE  = "pipeline_data/test_clean.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# 2. Native E8 Codebook Generator Setup
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
    # Integer shell tracking
    for coords in product([-2, -1, 0, 1, 2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    # Half-integer shell tracking
    for coords in product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
            
    # Sort points by norm-squared energy shells and slice the lowest 256
    candidates.sort(key=lambda item: item[0])
    codebook = [item[1] for item in candidates[:256]]
    return np.array(codebook, dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# 3. Custom E8 Lattice PyTorch Layers
# ─────────────────────────────────────────────────────────────
class E8LatticeQuantizer(nn.Module):
    def __init__(self, codebook_np):
        super().__init__()
        self.register_buffer("codebook", torch.tensor(codebook_np, dtype=torch.float32))
        
    def forward(self, x):
        # Vectorized matrix distance mapping on GPU: ||x - c||^2
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        cb_norm = torch.sum(self.codebook ** 2, dim=1, keepdim=True).T
        distances = x_norm - 2 * torch.matmul(x, self.codebook.T) + cb_norm
        indices = torch.argmin(distances, dim=1)
        return self.codebook[indices]

class E8QuantizedLinear(nn.Module):
    def __init__(self, orig_linear, codebook_np):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.weight = nn.Parameter(orig_linear.weight.data.clone())
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        # Allocate independent unique quantizer instance to prevent save-time tensor sharing errors
        self.quantizer = E8LatticeQuantizer(codebook_np)

    def forward(self, x):
        W = self.weight
        orig_shape = W.shape
        
        # Flatten and group into standard 32-element blocks (1 scale per block)
        W_flat = W.reshape(-1, 32)
        
        # Compute dynamic scale factor Δ per block
        max_vals = torch.max(torch.abs(W_flat), dim=1, keepdim=True)[0]
        scales = max_vals / 2.0
        scales = torch.clamp(scales, min=1e-8)
        
        # Project vectors into normalized lattice space
        W_norm = W_flat / scales
        W_8d = W_norm.reshape(-1, 8)
        
        # Distance map against our real GPU codebook
        W_quant_8d = self.quantizer(W_8d)
        
        # Reconstruct structural dimensions back
        W_quant_flat = W_quant_8d.reshape(-1, 32)
        W_reconstructed = (W_quant_flat * scales).reshape(orig_shape)
        
        # Straight-Through Estimation (STE) operator bypass
        W_final = W + (W_reconstructed - W).detach()
        return F.linear(x, W_final, self.bias)

def transform_to_e8_lattice(model, codebook_np):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if "classifier" in name:
                continue  # Avoid target task head collapse
            setattr(model, name, E8QuantizedLinear(child, codebook_np))
        else:
            transform_to_e8_lattice(child, codebook_np)


def get_quantized_state_dict(model):
    """
    Returns a NEW state_dict in which every E8QuantizedLinear's weight
    has been folded into its quantized (on-lattice) reconstruction --
    the identical computation E8QuantizedLinear.forward() applies
    transiently every forward pass via STE, but persisted here so the
    SAVED checkpoint actually matches what evaluate_model() measures
    live (which always re-quantizes on the fly, regardless of what
    save_pretrained happens to write to disk).

    IMPORTANT: this computes on a CLONED copy of each state_dict
    tensor and must NEVER be applied in-place to the live model's
    module.weight.data during training. Doing so would permanently
    destroy the fp32 shadow weight that AdamW needs for further
    updates -- the exact class of bug already fixed once in this
    thesis's QAT pipeline (stage_qat_all.py's original hook-based
    quantizer). Here, the live `model` is left completely untouched;
    only the returned state_dict differs from model.state_dict().
    """
    sd = model.state_dict()
    quantized_sd = {k: v.clone() for k, v in sd.items()}

    for name, module in model.named_modules():
        if isinstance(module, E8QuantizedLinear):
            W = module.weight.data
            orig_shape = W.shape
            W_flat = W.reshape(-1, 32)
            max_vals = torch.max(torch.abs(W_flat), dim=1, keepdim=True)[0]
            scales = torch.clamp(max_vals / 2.0, min=1e-8)
            W_8d = (W_flat / scales).reshape(-1, 8)
            W_quant_8d = module.quantizer(W_8d)
            W_quant_flat = W_quant_8d.reshape(-1, 32)
            W_folded = (W_quant_flat * scales).reshape(orig_shape)

            weight_key = f"{name}.weight"
            assert weight_key in quantized_sd, f"key {weight_key} missing from state_dict"
            quantized_sd[weight_key] = W_folded.detach().cpu().clone()
            # bias (if any) is intentionally left untouched -- it was
            # never quantized during training either.

    return quantized_sd

# ─────────────────────────────────────────────────────────────
# 4. Dataset Class Pipeline
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

# ─────────────────────────────────────────────────────────────
# 5. Training and Evaluation Modules
# ─────────────────────────────────────────────────────────────
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
    print("STAGE 3b: True E8 Lattice QAD Pipeline Running...")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    
    # 1. Load data records safely
    raw_train_items = []
    with open(TRAIN_FILE, "r") as f:
        for line in f:
            raw_train_items.append(json.loads(line))
            
    # 2. Load teacher soft target distributions
    with open("pipeline_data/teacher_train_probs.json", "r") as f:
        teacher_data = json.load(f)
    train_probs = teacher_data["probs"]
    
    # 3. Apply exact deterministic validation fractions
    val_size = int(0.1 * len(raw_train_items))
    train_size = len(raw_train_items) - val_size
    
    train_items = raw_train_items[:train_size]
    val_items   = raw_train_items[train_size:]
    
    print(f"Aligned Slice Sizes -> Train: {len(train_items)} | Teacher Targets: {len(train_probs)}")
    assert len(train_items) == len(train_probs), "Index sizes out of sync with teacher layout!"
    
    # Initialize separate loaders with safe, isolated scopes
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
    
    # Imbalance weights processing
    labels = [item["label"] for item in train_items]
    neg_count = labels.count(0)
    pos_count = labels.count(1)
    class_weights = torch.tensor([1.0, float(neg_count) / pos_count]).to(device)
    print(f"Computed Imbalance Penalties -> NonLeak: 1.0, Leak: {class_weights[1].item():.3f}")

    # Build model & convert standard linear weights to lattice projection modules
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL, num_labels=2)
    codebook_np = generate_e8_8bit_codebook()
    
    # Structural injection happens BEFORE moving to GPU device
    transform_to_e8_lattice(student, codebook_np)
    student = student.to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_val_f1 = 0.0
    t_start = time.time()
    
    # ─────────────────────────────────────────────────────────────
    # Distillation Loop
    # ─────────────────────────────────────────────────────────────
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
            
            # Weighted loss evaluation
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
                
        print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f} ---")
        val_metrics = evaluate_model(student, val_loader, description=f"Validation Epoch {epoch+1}")
        
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            quantized_sd = get_quantized_state_dict(student)
            student.save_pretrained("student_model_lattice_e8", state_dict=quantized_sd)
            print(f"  🆕 Checkpoint Saved (quantized weights folded)! Best Val F1: {best_val_f1*100:.2f}%")

    print(f"\nTraining completed in {(time.time() - t_start)/60:.2f} minutes.")
    
    # ─────────────────────────────────────────────────────────────
    # Final Test Set Evaluation
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RUNNING FINAL TEST EVALUATION ON BEST CHECKPOINT")
    print("=" * 65)
    # NOTE: the saved checkpoint already contains FOLDED (quantized,
    # on-lattice) weights -- get_quantized_state_dict() baked them in
    # before save_pretrained(). Do NOT call transform_to_e8_lattice()
    # here: that would wrap already-quantized weights in a fresh
    # E8QuantizedLinear, whose forward() re-derives scale via
    # max(|block|)/2 on values that are already on-lattice -- the same
    # naive-re-derivation bug fixed in the QAT export pipeline (it
    # only recovers the true scale if a codeword's max-abs component
    # happens to be exactly 2, which most don't). Applied on every
    # layer, every forward pass, that mismatch compounds into the
    # complete classifier collapse seen in the first run of this
    # script (F1=0.00%). A plain nn.Linear forward pass on already-
    # folded weights already IS the quantized computation -- nothing
    # left to re-derive.
    best_student = AutoModelForSequenceClassification.from_pretrained("student_model_lattice_e8")
    best_student = best_student.to(device)
    evaluate_model(best_student, test_loader, description="Final E8 Lattice QAD Student Test")

if __name__ == "__main__":
    main()
