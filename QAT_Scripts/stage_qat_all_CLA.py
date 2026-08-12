"""
stage_qat_all.py (v4 — unified scale convention with export/on-device format)
─────────────────────────────────────────────────────────────
QAT for BERT-Base and TinyBERT with E8 and Z8 quantization.

FIX vs v3:
  v3 computed one scale per 8-element sub-vector (4 independent
  scales per 32-weight block). The export script and the C++
  on-device harness both store exactly ONE scale per 32-element
  block (struct block_e8_8 { float scale; uint8_t idx[4]; }).
  This mismatch meant the weights actually shipped to the phone
  would not be the same quantized values validated on the GPU.

  v4 computes ONE shared scale per 32-element block (S = max(|block|)/2),
  then splits that block into its 4 sub-vectors of 8 for the E8/Z8
  codebook lookup, reusing the same shared scale for all 4. This is
  now byte-for-byte consistent with export_e8_8bit/export_z8_8bit
  and with the on-device block_e8_8/block_z8_8 struct — no export or
  struct changes needed on top of this, only a re-run of QAT.

FIX vs v2 (carried over from v3):
  v2 used a forward_pre_hook that overwrote `module.weight.data`
  with the quantized value on every forward pass. This destroyed
  the full-precision "shadow" weight the optimizer needs: every
  AdamW update (lr=2e-5) was smaller than the gap between lattice
  codewords, so the very next forward pass snapped the weight
  right back to where it started. Net effect: the model could
  never learn (loss stuck at ln(2), F1 stuck at 0).

  v3 replaces every nn.Linear with a QATLinear module that keeps
  a real fp32 nn.Parameter as the trainable master weight, and
  quantizes it ONLY inside forward() via the STE autograd
  Function. Gradients flow straight through to the fp32 master,
  so updates actually accumulate across steps.

  Also: the checkpoint is now always guaranteed to be written
  (best-F1 state kept in memory, with a last-epoch fallback), so
  a stalled run can no longer crash on an empty save_dir.

CODEBOOK FORMATION: explicitly shown, using verified is_e8_point()
QUANTIZATION: nearest codebook entry via vectorized distance computation
STE: gradient passes straight through during backward pass
"""

import json, time, os, random, copy
from itertools import product
from collections import defaultdict
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

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════
# SECTION 1: E8 CODEBOOK FORMATION  (unchanged — this part was fine)
# ═══════════════════════════════════════════════════════════

def is_e8_point(x, tol=1e-6):
    """
    Check E8 lattice membership.
    Case 1: all integer coords, even sum
    Case 2: all half-integer coords, even sum
    """
    x = np.array(x, dtype=float)
    is_int  = np.all(np.abs(x - np.round(x)) < tol)
    is_half = np.all(np.abs(x - np.round(x - 0.5) - 0.5) < tol)
    s = np.sum(x)
    even = (np.abs(s - np.round(s)) < tol and int(np.round(s)) % 2 == 0)
    return (is_int or is_half) and even


def build_e8_codebook(bits, device):
    """
    CODEBOOK FORMATION — E8 Lattice
    Searches for valid E8 points, sorts by norm, takes 2^bits closest.
    Returns GPU tensor of shape (2^bits, 8).
    """
    print(f"\n  ── E8 CODEBOOK FORMATION ({bits}-bit) ──────────────")
    print(f"  Step 1: Search integer coords in [-2,2]^8 for E8 points...")
    candidates = []
    for coords in product([-2,-1,0,1,2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x,x)), x))

    print(f"  Step 2: Search half-integer coords in [-1.5,1.5]^8...")
    for coords in product([-1.5,-0.5,0.5,1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x,x)), x))

    print(f"  Total valid E8 points found: {len(candidates):,}")

    # Step 3: Sort by norm shell, shuffle within same norm (unbiased)
    rng = np.random.default_rng(seed=42)
    norm_groups = defaultdict(list)
    for norm_sq, x in candidates:
        norm_groups[round(norm_sq, 4)].append(x)

    ordered = []
    for norm in sorted(norm_groups.keys()):
        grp = norm_groups[norm]
        for i in rng.permutation(len(grp)):
            ordered.append(grp[i])

    # Step 4: Take 2^bits closest points as codebook
    N = 2 ** bits
    codebook = torch.tensor(
        np.array(ordered[:N], dtype=np.float32), device=device
    )
    shell1 = sum(1 for p in ordered[:N] if abs(np.dot(p,p)-2.0)<1e-4)
    print(f"  Step 4: Codebook = {N} closest E8 points")
    print(f"  Shell-1 points (norm=2) included: {shell1}")
    print(f"  ── E8 CODEBOOK FORMATION COMPLETE ──────────────────")
    return codebook  # (N, 8), on GPU


def build_z8_codebook(bits, device):
    """
    CODEBOOK FORMATION — Z8 (integer lattice, no parity constraint)
    Used as comparison baseline against E8.
    """
    print(f"\n  ── Z8 CODEBOOK FORMATION ({bits}-bit) ──────────────")
    candidates = []
    for coords in product(range(-3,4), repeat=8):
        x = np.array(coords, dtype=float)
        candidates.append((float(np.dot(x,x)), x))

    rng = np.random.default_rng(seed=42)
    norm_groups = defaultdict(list)
    for ns, x in candidates:
        norm_groups[round(ns,4)].append(x)

    ordered = []
    N = 2 ** bits
    for norm in sorted(norm_groups.keys()):
        grp = norm_groups[norm]
        for i in rng.permutation(len(grp)):
            ordered.append(grp[i])
        if len(ordered) >= N: break

    codebook = torch.tensor(
        np.array(ordered[:N], dtype=np.float32), device=device
    )
    print(f"  Codebook = {N} closest Z8 integer points")
    print(f"  ── Z8 CODEBOOK FORMATION COMPLETE ──────────────────")
    return codebook  # (N, 8), on GPU


# ═══════════════════════════════════════════════════════════
# SECTION 2: VECTORIZED STE QUANTIZATION
# All operations run on GPU — no Python loops over blocks.
# ═══════════════════════════════════════════════════════════

class LatticeQuantizeFn(torch.autograd.Function):
    """
    Vectorized lattice quantization with STE.

    FORWARD (all on GPU, no Python loops):
      1. Flatten weight to 1D, pad to multiple of 32
      2. Reshape to (num_blocks, 32)
      3. Compute ONE scale per 32-element block: S = max(|block|) / 2
      4. Scale block: block_scaled = block / S
      5. Split each scaled 32-block into 4 sub-vectors of 8, and for
         each sub-vector compute distances to ALL codebook entries:
         dist[i,j] = ||subvec_i - codebook[j]||^2
      6. Find nearest codeword index per sub-vector: argmin over codebook dim
      7. Gather nearest codewords, reassemble the 4 sub-vectors back
         into a 32-block, multiply by the ONE shared scale S
      8. Reshape back to original weight shape

    This block/scale layout is now identical to export_e8_8bit /
    export_z8_8bit and to the on-device block_e8_8/block_z8_8 struct
    (float scale + 4x uint8 index = 8 bytes/32 weights = 2.0 bits/weight).

    BACKWARD: gradient passes straight through (STE) to whatever
    tensor was passed in as `weight` — this is only useful for
    training if that tensor is itself a real, persistent
    nn.Parameter (see QATLinear below). Applying this to a
    Parameter's `.data` and writing the result back into `.data`
    (as the old hook-based version did) throws the gradient away
    with nowhere persistent to land.
    """
    @staticmethod
    def forward(ctx, weight, codebook):
        orig_shape = weight.shape
        flat = weight.detach().reshape(-1).float()

        # Pad to multiple of 32 — block size now matches the on-device
        # format exactly: ONE shared scale per 32 elements, split into
        # 4 sub-vectors of 8 for the E8/Z8 codebook lookup.
        rem = flat.shape[0] % 32
        pad = (32 - rem) % 32
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

        blocks32 = flat.reshape(-1, 32)       # (B32, 32)

        # ONE scale per 32-element block: S = max(|block|) / 2
        # (identical formula/granularity to export_e8_8bit / export_z8_8bit)
        S = blocks32.abs().max(dim=1, keepdim=True).values / 2.0
        S = S.clamp(min=1e-8)                 # (B32, 1)
        blocks32_scaled = blocks32 / S        # (B32, 32), normalised by shared scale

        # Split each 32-block into its 4 sub-vectors of 8 for lookup
        B32 = blocks32_scaled.shape[0]
        subvecs = blocks32_scaled.reshape(B32 * 4, 8)   # (B32*4, 8)

        # Vectorized nearest-neighbour:
        # dist(i,j) = ||b_i - c_j||^2 = ||b_i||^2 - 2*b_i·c_j + ||c_j||^2
        CB = codebook.float()                 # (N, 8)
        b_norm = (subvecs ** 2).sum(dim=1, keepdim=True)          # (B32*4, 1)
        c_norm = (CB ** 2).sum(dim=1).unsqueeze(0)                # (1, N)
        dot    = subvecs @ CB.T                                    # (B32*4, N)
        dists  = b_norm - 2 * dot + c_norm                        # (B32*4, N)

        best_idx  = dists.argmin(dim=1)                           # (B32*4,)
        best_code = CB[best_idx]                                   # (B32*4, 8)

        # Reassemble the 4 sub-vectors back into 32-blocks and apply
        # the ONE shared scale (broadcasts across all 32 elements)
        best_code_32 = best_code.reshape(B32, 32)                 # (B32, 32)
        result = best_code_32 * S                                  # (B32, 32)

        # Reshape back
        result_flat = result.reshape(-1)
        if pad:
            result_flat = result_flat[:-pad]

        return result_flat.reshape(orig_shape).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None   # STE: gradient unchanged


def quantize_lattice(weight, codebook):
    return LatticeQuantizeFn.apply(weight, codebook)


class QATLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear.

    Keeps a real fp32 `nn.Parameter` (self.weight) as the trainable
    master copy — this is what the optimizer actually updates.
    Quantization happens ONLY inside forward(), via the STE
    autograd Function, so:
      - the forward math uses the quantized (on-lattice) weight,
        exactly reflecting what the deployed model will compute
      - the backward pass sends gradient straight through to the
        fp32 master weight (that's what STE means), so small
        updates accumulate across steps instead of being wiped
        out by re-quantization on the very next forward pass.
    """
    def __init__(self, orig_linear, codebook):
        super().__init__()
        self.in_features  = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.weight = nn.Parameter(orig_linear.weight.data.clone())
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)
        # not persistent: we never save a QATLinear model directly,
        # we "fold" quantized weights into a plain HF model instead
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x):
        w_q = quantize_lattice(self.weight, self.codebook)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def replace_linear_with_qat(model, codebook, exclude_substrings=("classifier",)):
    """
    Replaces every nn.Linear submodule (except ones whose full
    dotted name contains a string in exclude_substrings) with a
    QATLinear. The classification head is excluded by default:
    it's tiny (negligible effect on model size/compression) and
    starts from a random init, so quantizing it from step 0 adds
    instability for essentially no benefit. Pass
    exclude_substrings=() if your thesis requires quantizing it too.
    """
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                targets.append((module, child_name))
    for parent, child_name in targets:
        orig = getattr(parent, child_name)
        setattr(parent, child_name, QATLinear(orig, codebook))
    return len(targets)


def get_quantized_state_dict(qat_model):
    """
    Builds a state_dict where every QATLinear's weight has been
    folded down to its actual quantized (on-lattice) value — i.e.
    exactly what the model computes with at inference time — while
    every other parameter (embeddings, layernorms, the excluded
    classifier head, etc.) is copied through unchanged.
    """
    sd = {}
    for name, module in qat_model.named_modules():
        if isinstance(module, QATLinear):
            with torch.no_grad():
                w_q = quantize_lattice(module.weight, module.codebook).detach().clone()
            sd[f"{name}.weight"] = w_q
            if module.bias is not None:
                sd[f"{name}.bias"] = module.bias.detach().clone()
    full_sd = qat_model.state_dict()
    for k, v in full_sd.items():
        if k not in sd:
            sd[k] = v.detach().clone()
    return sd


# ═══════════════════════════════════════════════════════════
# SECTION 3: DATASET + EVALUATION
# ═══════════════════════════════════════════════════════════

class ReConDataset(Dataset):
    def __init__(self, items, tokenizer, ml=128):
        self.texts  = [x["text"]  for x in items]
        self.labels = [x["label"] for x in items]
        self.tok = tokenizer; self.ml = ml

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True,
                       max_length=self.ml, padding="max_length",
                       return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def load_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f]


def print_cm(cm):
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    print(f"    Confusion Matrix:")
    print(f"    {'':>14} {'Pred Non-leak':>15} {'Pred Leak':>12}")
    print(f"    {'Actual Non-leak':>14} {tn:>15,} {fp:>12,}")
    print(f"    {'Actual Leak':>14} {fn:>15,} {tp:>12,}")
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


def evaluate(model, loader, device, name=""):
    model.eval(); pa, la = [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device))
            pa.extend(torch.argmax(out.logits,1).cpu().tolist())
            la.extend(b["label"].tolist())
    el = time.time()-t0
    acc  = accuracy_score(la,pa)
    f1   = f1_score(la,pa,zero_division=0)
    prec = precision_score(la,pa,zero_division=0)
    rec  = recall_score(la,pa,zero_division=0)
    cm   = confusion_matrix(la,pa,labels=[0,1])
    print(f"\n  [{name}]")
    print(f"    Accuracy:{acc*100:.2f}%  Precision:{prec*100:.2f}%")
    print(f"    Recall:  {rec*100:.2f}%  F1:       {f1*100:.2f}%")
    print_cm(cm)
    print(f"    Latency: {el/len(la)*1000:.3f} ms/sample")
    return {"accuracy":acc,"f1":f1,"precision":prec,"recall":rec,
            "confusion_matrix":cm.tolist()}


# ═══════════════════════════════════════════════════════════
# SECTION 4: QAT TRAINING
# ═══════════════════════════════════════════════════════════

def run_qat(model_name, model_source, codebook, quant_type,
            train_items, val_items, test_items,
            device, epochs=3, lr=2e-5, batch_size=16):

    print(f"\n{'='*65}")
    print(f"QAT: {model_name} + {quant_type}")
    print(f"{'='*65}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=2).to(device)

    n = replace_linear_with_qat(model, codebook, exclude_substrings=("classifier",))
    print(f"  QAT-wrapped {n} linear layers (classifier head kept full precision) | quant type: {quant_type}")

    nl = sum(1 for x in train_items if x["label"]==0)
    lk = sum(1 for x in train_items if x["label"]==1)
    nt = nl+lk
    w  = torch.tensor([nt/(2*nl), nt/(2*lk)]).to(device)
    print(f"  Class weights: NonLeak={w[0]:.3f} Leak={w[1]:.3f}")

    tds = ReConDataset(train_items, tokenizer)
    vds = ReConDataset(val_items,   tokenizer)
    eds = ReConDataset(test_items,  tokenizer)
    tl = DataLoader(tds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(vds, batch_size=batch_size, shuffle=False)
    el = DataLoader(eds, batch_size=batch_size, shuffle=False)

    ce   = nn.CrossEntropyLoss(weight=w)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr)
    sch  = get_linear_schedule_with_warmup(opt,0,len(tl)*epochs)

    save_dir = f"qat_{model_name.replace('-','_').lower()}_{quant_type.lower()}_v4_epochs{epochs}"
    os.makedirs(save_dir, exist_ok=True)

    best_f1  = -1.0          # -1 so even an epoch scoring 0.0 F1 gets kept as a fallback
    best_qsd = None
    t0 = time.time()

    for ep in range(epochs):
        model.train(); ep_loss=0.0
        for step, batch in enumerate(tl):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbs  = batch["label"].to(device)
            opt.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask)
            loss = ce(out.logits, lbs)
            loss.backward(); opt.step(); sch.step()
            ep_loss += loss.item()
            if step % 300 == 0:
                print(f"  Ep{ep+1} step{step}/{len(tl)} loss={loss.item():.4f}")
        print(f"  Epoch {ep+1} avg: {ep_loss/len(tl):.4f}")
        vr = evaluate(model, vl, device, f"Val ep{ep+1}")
        if vr["f1"] > best_f1:
            best_f1 = vr["f1"]
            best_qsd = {k: v.cpu() for k, v in get_quantized_state_dict(model).items()}
            print(f"    ✅ New best val F1={best_f1*100:.2f}% (kept in memory)")

    print(f"\n  Training: {(time.time()-t0)/60:.1f} min")

    if best_qsd is None:
        # Should not happen anymore, but guarantees we never crash
        # trying to reload an empty checkpoint dir.
        print("  ⚠️  No epoch beat the -1.0 baseline (unexpected) — falling back to last-epoch weights.")
        best_qsd = {k: v.cpu() for k, v in get_quantized_state_dict(model).items()}

    # Fold the actual quantized weights into a fresh, plain HF model
    # (real nn.Linear layers) so it saves/loads with the standard
    # AutoModelForSequenceClassification API and is directly usable
    # for on-device (Android) export later.
    best = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=2).to(device)
    best.load_state_dict({k: v.to(device) for k, v in best_qsd.items()}, strict=True)
    best.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Saved quantized checkpoint -> {save_dir}/")

    print(f"\n  ── FINAL TEST: {model_name} {quant_type} ──")
    res = evaluate(best, el, device, f"Test {model_name} {quant_type}")
    mb  = sum(p.numel()*4 for p in best.parameters())/(1024**2)
    print(f"  Size: {mb:.2f} MB")
    return res, save_dir


# ═══════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # FIX: epochs was hardcoded to 3 with no CLI override, and save_dir
    # had no epoch tag ("qat_{model}_{quant}_v4"), so every run at any
    # epoch count overwrote the same directory and analyze_checkpoint.py
    # (which expects "qat_{model}_{quant}_v4_epochs{N}") could never find
    # a real checkpoint to analyze. Both fixed below.
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full_train = load_jsonl("pipeline_data/train_clean.jsonl")
    test_items = load_jsonl("pipeline_data/test_clean.jsonl")
    random.shuffle(full_train)
    val_n = int(len(full_train)*0.1)
    val_items, train_items = full_train[:val_n], full_train[val_n:]
    print(f"Train:{len(train_items):,} Val:{len(val_items):,} Test:{len(test_items):,}")

    # Build both codebooks explicitly on GPU
    print("\n"+"="*65)
    print("BUILDING CODEBOOKS")
    print("="*65)
    BITS = 8
    e8_cb = build_e8_codebook(BITS, device)
    z8_cb = build_z8_codebook(BITS, device)
    print(f"\nE8 codebook on GPU: {e8_cb.shape}")
    print(f"Z8 codebook on GPU: {z8_cb.shape}")

    # NEW: persist both codebooks to disk. Previously only held as
    # non-persistent buffers inside each QATLinear (never saved, never
    # reconstructable except by re-running build_e8_codebook/
    # build_z8_codebook with the same seed=42). Saved once here since
    # both codebooks are shared across all 4 configs below.
    np.save("codebook_e8_v4.npy", e8_cb.detach().cpu().numpy())
    np.save("codebook_z8_v4.npy", z8_cb.detach().cpu().numpy())
    print(f"  Saved codebooks -> codebook_e8_v4.npy, codebook_z8_v4.npy")

    configs = [
        ("BERT-Base", "teacher_model",                         e8_cb, "E8"),
        ("BERT-Base", "teacher_model",                         z8_cb, "Z8"),
        ("TinyBERT",  "huawei-noah/TinyBERT_General_4L_312D", e8_cb, "E8"),
        ("TinyBERT",  "huawei-noah/TinyBERT_General_4L_312D", z8_cb, "Z8"),
    ]

    all_results = []
    for mn, ms, cb, qt in configs:
        res, sd = run_qat(mn, ms, cb, qt,
                          train_items, val_items, test_items,
                          device, epochs=args.epochs)
        all_results.append((mn, qt, res, sd))

    print("\n"+"="*65)
    print("FINAL SUMMARY — ALL QAT EXPERIMENTS")
    print("="*65)
    print(f"\n{'Model':<22} {'Quant':>6} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("-"*60)
    for mn, qt, res, _ in all_results:
        print(f"  {mn:<20} {qt:>6} "
              f"{res['accuracy']*100:>7.2f}% "
              f"{res['f1']*100:>7.2f}% "
              f"{res['precision']*100:>7.2f}% "
              f"{res['recall']*100:>7.2f}%")

    print("\nSaved model dirs:")
    for _,_,_,d in all_results: print(f"  {d}/")
    print("\nNext: push models to phone for RSS RAM + latency")
