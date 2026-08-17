"""
stage_qat_all.py (v5 — paper-based row-wise dynamic-range calibration)
─────────────────────────────────────────────────────────────
QAT for BERT-Base and TinyBERT with E8 and Z8 quantization.

FIX vs v4:
  v4 computed ONE scale per 32-element block: S = max(|block|)/2.
  This matched the on-device export format but did NOT match the
  normalization method in Kohli et al., "Lattice-Based Vector
  Quantization for Low-Bit Quantization-Aware Training" (Section 3.1.3,
  Algorithm 1) -- the paper this thesis's E8/Z8 QAT approach is built
  on. The paper's calibration is instead:

      Y_max   = Delta0 * (q^M - 1) / 2
      sigma_i = std(row i of the ORIGINAL weight matrix)
      beta_i  = Y_max / (C_b * sigma_i)      -- ONE beta per output ROW
      W_tilde_i,: = beta_i * W_i,:            -- scale that row
      (quantize each 8-D sub-block of the scaled row against the
       codebook, same nearest-neighbour lookup as before)
      W_hat_i,: = (quantized row) / beta_i    -- invert the scaling

  v5 implements this precisely: PER-ROW calibration (std-based, tied
  to the paper's Y_max/C_b), replacing the per-32-block max/2 scale.
  Our fixed 256-entry (8-bit) codebook corresponds to the paper's
  q=256, M=1 (one direct nearest-of-256 lookup, no hierarchical
  residual digits) -- this is exactly the paper's own "fused" QAT
  simplification (collapsing HNLQ to a single nearest-lattice-point
  projection), not an approximation of it.

  IMPORTANT structural fix that comes with this: since scale is now
  per-ROW, a storage block (which shares one scale across 4 sub-
  vectors) must never span two rows with different scales. v4 padded
  the WHOLE flattened tensor to a multiple of 32 (which happened to
  be safe only because BERT-Base's row width 768 and TinyBERT's total
  element count 97344 are divisible by 32 -- NOT TinyBERT's per-row
  width of 312). v5 instead pads EACH ROW individually to a multiple
  of 32 before chunking, which is correct regardless of row width and
  is a no-op for BERT-Base (768 already divides evenly).

FIX vs v3 (carried over from v4):
  v3 computed one scale per 8-element sub-vector (4 independent
  scales per 32-weight block); v4 unified this to one shared scale
  per 32-element block to match the on-device export format.

FIX vs v2 (carried over from v3/v4):
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

import json, time, os, random, copy, argparse
from itertools import product
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")   # headless -- no display needed, just saves PNG files
import matplotlib.pyplot as plt
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
# Row-wise dynamic-range calibration constants
# (Kohli et al., Section 3.1.3 / Algorithm 1)
#
#   Y_max = Delta0 * (q^M - 1) / 2
#   beta_i = Y_max / (C_b * sigma_i)      -- per output row i
#
# FIX: Y_max is now derived EMPIRICALLY from the codebook's own
# coordinate range (computed inside forward(), from the `codebook`
# tensor it already receives), instead of assumed from a (q,M) pair.
#
# Why: our fixed 256-entry codebook only contains E8/Z8 points with
# small coordinates (max |coordinate| = 1.5 for our E8-256 codebook).
# The paper's (q,M)-derived Y_max formula assumes a codebook/lattice
# search that scales WITH Y_max (either a properly hierarchical HNLQ
# construction, or an unbounded lattice search that can represent
# arbitrarily large points). Our codebook is a small FIXED lookup
# table, so plugging in q=256 gave Y_max=191.25 -- 127x larger than
# what the codebook can represent (confirmed empirically: max
# |coordinate| across the 256-point codebook is 1.5). Every row-scaled
# weight then landed far outside the codebook's coverage and snapped
# to a boundary codeword regardless of its true value, destroying all
# signal -- this is exactly what caused the F1=0.00%/constant-
# majority-class collapse in the first normalized run.
#
# Grounding Y_max in the codebook's own range keeps the paper's core
# mechanism (row-wise beta = Y_max/(C_b*sigma)) exactly as specified,
# while guaranteeing scaled weights actually land within what the
# codebook can represent.
# ═══════════════════════════════════════════════════════════
C_B      = 5.0     # paper's fixed row-safety constant ("5-sigma range per row")

# LITERAL-Y_max VARIANT: unlike stage_qat_all_normalization_CLA.py, which
# grounds Y_max empirically in codebook.abs().max() (a workaround for the
# small fixed 256-point table), THIS script follows the paper's Appendix
# A.3 formula exactly: Y_max = Delta0*(q^M - 1)/2. Defaults below match
# q=8, M=1 -- a real config used throughout the paper's own tables
# (e.g. Table 1/2/4/5's "3.00b(8,1)" row) -- rather than q=256, which
# never appears anywhere in the paper (the paper states q in [2,8],
# M in [1,4] in Section 4; the lone exception is q=16 appearing once,
# in Appendix Table 6). These are reassigned from CLI args in __main__,
# same pattern as C_B above.
DELTA0   = 1.5
Q        = 8
M        = 1
CLIP_TAU = None    # optional clip (paper: "optional element-wise clipping for
                   # numerical stability"); set a float to enable, e.g. 6.0

# ═══════════════════════════════════════════════════════════
# SECTION 2: VECTORIZED STE QUANTIZATION
# All operations run on GPU — no Python loops over blocks.
# ═══════════════════════════════════════════════════════════

class LatticeQuantizeFn(torch.autograd.Function):
    """
    Vectorized lattice quantization with STE, using the paper's
    row-wise dynamic-range calibration (Kohli et al., Algorithm 1).

    FORWARD (all on GPU, no Python loops):
      1. Compute Y_max = Delta0*(q^M - 1)/2 (fixed, doesn't depend on data)
      2. Compute sigma_i = std(row i of weight), per output row
      3. Compute beta_i = Y_max / (C_b * sigma_i), per output row
      4. Scale each row: W_tilde_i,: = beta_i * W_i,:
      5. Pad EACH ROW individually to a multiple of 32 (so a shared-
         scale storage block never spans two rows with different betas)
      6. Split into 8-D sub-vectors, find nearest codeword for each
         (same nearest-neighbour lookup against the fixed codebook
         as before -- only the scale feeding into it has changed)
      7. Reassemble rows, drop row padding, invert scaling: Ŵ = Ŵ̃ / beta
      8. Optional clip to [-CLIP_TAU, CLIP_TAU]

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
        orig_shape = weight.shape           # (O, I)
        O, I = orig_shape
        W = weight.detach().float()          # (O, I)

        # ---- Paper's row-wise dynamic-range calibration ----
        # LITERAL variant: Y_max follows the paper's Appendix A.3
        # formula exactly (Delta0*(q^M-1)/2), unlike the CLA script's
        # empirical codebook.abs().max() workaround. Since the fixed
        # 256-point codebook only spans a small coordinate range, this
        # is expected to push most scaled weights outside the table's
        # coverage -- that overload/collapse behavior is the entire
        # point of this diagnostic variant.
        Y_max = DELTA0 * (Q ** M - 1) / 2.0
        sigma = W.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)  # (O,1)
        beta  = Y_max / (C_B * sigma)                                       # (O,1)
        W_tilde = W * beta                                                    # (O,I)

        # Pad EACH ROW individually to a multiple of 32 -- guarantees
        # a storage block never spans two rows (rows can have
        # different betas; a shared-scale block must not cross a
        # scale boundary). No-op for BERT-Base (768 already divides
        # evenly); matters for TinyBERT (312 does not).
        rem = I % 32
        pad = (32 - rem) % 32
        if pad:
            W_tilde = torch.cat(
                [W_tilde, torch.zeros(O, pad, device=W.device, dtype=W_tilde.dtype)],
                dim=1)
        I_padded = I + pad

        flat = W_tilde.reshape(-1)              # row-major; rows stay block-aligned
        subvecs = flat.reshape(-1, 8)             # (O*I_padded/8, 8)

        # Vectorized nearest-neighbour against the fixed codebook
        # (unchanged from before -- only the input scale has changed)
        CB = codebook.float()                    # (N, 8)
        b_norm = (subvecs ** 2).sum(dim=1, keepdim=True)
        c_norm = (CB ** 2).sum(dim=1).unsqueeze(0)
        dot    = subvecs @ CB.T
        dists  = b_norm - 2 * dot + c_norm
        best_idx  = dists.argmin(dim=1)
        best_code = CB[best_idx]

        W_hat_tilde = best_code.reshape(O, I_padded)
        if pad:
            W_hat_tilde = W_hat_tilde[:, :I]     # drop row padding back off

        # ---- Invert row-wise scaling ----
        W_hat = W_hat_tilde / beta                # (O, I)

        if CLIP_TAU is not None:
            W_hat = W_hat.clamp(-CLIP_TAU, CLIP_TAU)

        return W_hat.to(weight.dtype)

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
            device, epochs=5, lr=2e-5, batch_size=16, cb_used=None):

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

    # FIX: save_dir now includes epochs AND C_b -- the old
    # "..._v5" (no epoch/cb tag) meant every different --epochs or
    # --cb run overwrote the SAME directory, silently corrupting or
    # losing previous runs' checkpoints when run sequentially, and
    # risking concurrent-write corruption if run in parallel (as
    # discovered when the epoch-sweep experiments were already
    # underway). This does not affect any already-reported F1/loss
    # numbers (those came from live evaluation / epoch-tagged JSON
    # files, both already fine) -- only the actual saved checkpoint
    # weights on disk were at risk.
    cb_tag = f"{cb_used:g}" if cb_used is not None else "unknown"
    save_dir = (f"qat_{model_name.replace('-','_').lower()}_{quant_type.lower()}"
                f"_v5lit_epochs{epochs}_cb{cb_tag}_d{DELTA0:g}_q{Q}_m{M}")
    os.makedirs(save_dir, exist_ok=True)

    # Persist the exact codebook used to train this checkpoint. It was
    # previously only held as a non-persistent buffer (register_buffer(...,
    # persistent=False)), so it never landed in state_dict() or on disk.
    # Saving it here makes future codeword-usage analysis and mobile
    # deployment self-contained -- no need to re-derive it from
    # build_e8_codebook/build_z8_codebook with the same seed=42.
    np.save(os.path.join(save_dir, "codebook.npy"), codebook.detach().cpu().numpy())
    print(f"  Saved codebook ({tuple(codebook.shape)}) -> {save_dir}/codebook.npy")

    best_f1  = -1.0          # -1 so even an epoch scoring 0.0 F1 gets kept as a fallback
    best_qsd = None
    best_beta_values = None  # NEW: beta snapshot matching whichever epoch is best
    epoch_losses = []        # NEW: one avg training loss per epoch, for the error-vs-epoch graph
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
        epoch_losses.append(ep_loss/len(tl))   # NEW: same value just printed, saved for the graph
        vr = evaluate(model, vl, device, f"Val ep{ep+1}")
        if vr["f1"] > best_f1:
            best_f1 = vr["f1"]
            best_qsd = {k: v.cpu() for k, v in get_quantized_state_dict(model).items()}
            # NEW: snapshot beta values at the SAME moment as the best
            # checkpoint (not after the full loop finishes) -- the best
            # epoch isn't always the last one, so capturing beta only
            # after training ends could describe a DIFFERENT model
            # state than the one actually saved/reported.
            best_beta_values = collect_beta_values(model)
            print(f"    ✅ New best val F1={best_f1*100:.2f}% (kept in memory)")

    print(f"\n  Training: {(time.time()-t0)/60:.1f} min")

    if best_qsd is None:
        # Should not happen anymore, but guarantees we never crash
        # trying to reload an empty checkpoint dir.
        print("  ⚠️  No epoch beat the -1.0 baseline (unexpected) — falling back to last-epoch weights.")
        best_qsd = {k: v.cpu() for k, v in get_quantized_state_dict(model).items()}
        best_beta_values = collect_beta_values(model)

    # NEW: beta-value histogram, using the snapshot captured at the
    # SAME epoch as the best checkpoint (see note above -- NOT a
    # fresh computation from whatever the live model looks like after
    # the full loop, which could be a later, different epoch).
    save_beta_histogram(best_beta_values, model_name, quant_type, epochs, cb_used)

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
    return res, save_dir, epoch_losses


# ═══════════════════════════════════════════════════════════
# SECTION 4b: BETA-VALUE HISTOGRAM (NEW -- advisor's request)
# ═══════════════════════════════════════════════════════════

def collect_beta_values(model):
    """
    Computes beta_i = Y_max / (C_B * sigma_i) for every output row of
    every QATLinear layer in `model`, using its LIVE (unfolded)
    weight -- the exact same computation LatticeQuantizeFn.forward()
    performs every forward pass, just read out here for inspection
    rather than used to quantize. Returns one flat list of beta
    values, pooled across every layer's every row (aggregated, same
    convention already used for the QAD overload diagnostics).
    """
    all_betas = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, QATLinear):
                W = module.weight.data.float()
                codebook = module.codebook
                Y_max = DELTA0 * (Q ** M - 1) / 2.0
                sigma = W.std(dim=1, unbiased=False).clamp(min=1e-8)  # (O,)
                beta = Y_max / (C_B * sigma)                          # (O,)
                all_betas.extend(beta.cpu().tolist())
    return all_betas


def save_beta_histogram(beta_values, model_name, quant_type, epochs_used, cb_used):
    """
    Saves the raw beta values (JSON) and a histogram plot (PNG) for
    one model+quant+epoch+C_b combination. Filenames encode all four,
    so different C_b / epoch experiments never overwrite each other.
    """
    cb_tag = f"{cb_used:g}"
    tag = f"{model_name.replace('-','_').lower()}_{quant_type.lower()}_epochs{epochs_used}_cb{cb_tag}"

    json_path = f"beta_hist_{tag}.json"
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "quant_type": quant_type,
                   "epochs": epochs_used, "C_b": cb_used,
                   "beta_values": beta_values}, f)
    print(f"  Saved beta values -> {json_path}  (n={len(beta_values):,})")

    plt.figure(figsize=(7, 5))
    plt.hist(beta_values, bins=80, color="tab:blue", alpha=0.75, edgecolor="black", linewidth=0.3)
    plt.xlabel("beta_i value")
    plt.ylabel("Count (rows, pooled across all QATLinear layers)")
    plt.title(f"{model_name} + {quant_type} -- beta histogram\n"
             f"({epochs_used} epochs, C_b={cb_used})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"beta_hist_{tag}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved beta histogram -> {png_path}")


# ═══════════════════════════════════════════════════════════
# SECTION 4c: ERROR-VS-EPOCH GRAPHING
# ═══════════════════════════════════════════════════════════

def save_and_plot_loss_curves(loss_histories, epochs_used, out_prefix="qat_normalization"):
    """
    loss_histories: dict like {"BERT-Base_E8": [ep1_loss, ep2_loss, ...], ...}
    epochs_used:    the EPOCHS value used for this run (3, 5, etc.) -- baked
                    into the output filenames so a 3-epoch run and a 5-epoch
                    run never overwrite each other's saved data/plots.

    Saves:
      {out_prefix}_loss_history_epochs{N}.json  -- raw numbers, for later
                                                     comparison across runs
      {out_prefix}_loss_vs_epoch_epochs{N}.png  -- immediate plot for THIS run
    """
    json_path = f"{out_prefix}_loss_history_epochs{epochs_used}.json"
    with open(json_path, "w") as f:
        json.dump({"epochs": epochs_used, "histories": loss_histories}, f, indent=2)
    print(f"\n  Saved loss history -> {json_path}")

    plt.figure(figsize=(8, 5))
    for label, losses in loss_histories.items():
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title(f"QAT + Normalization -- Training loss vs epoch ({epochs_used} epochs)")
    plt.xticks(range(1, epochs_used + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"{out_prefix}_loss_vs_epoch_epochs{epochs_used}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved plot -> {png_path}")


# ═══════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ═══════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Command-line argument parsing for EPOCHS_USED
    parser = argparse.ArgumentParser(description="Run QAT with variable epochs and C_b.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--cb", type=float, default=5.0,
                       help="Row-safety constant C_b in beta_i=Y_max/(C_b*sigma_i) "
                            "(default: 5.0, the paper's fixed value)")
    parser.add_argument("--delta0", type=float, default=1.5,
                       help="Base scale Delta0 in Y_max=Delta0*(q^M-1)/2 (default: 1.5)")
    parser.add_argument("--q", type=int, default=8,
                       help="Radix q (default: 8, matches paper's own '3.00b(8,1)' config; "
                            "paper states q in [2,8] -- NOT 256)")
    parser.add_argument("--m", type=int, default=1,
                       help="Digit levels M (default: 1)")
    args = parser.parse_args()

    EPOCHS_USED = args.epochs
    C_B = args.cb       # reassigns the module-level global; LatticeQuantizeFn.forward
                        # looks up C_B by name at call time, so this takes effect for
                        # every training step that follows.
    DELTA0 = args.delta0
    Q = args.q
    M = args.m
    Y_max_preview = DELTA0 * (Q ** M - 1) / 2.0
    print(f" Running QAT script (LITERAL Y_max variant) for {EPOCHS_USED} epochs, "
          f"C_b={C_B}, Delta0={DELTA0}, q={Q}, M={M}  ->  Y_max={Y_max_preview:.4f}")
    print(f" NOTE: fixed 256-point codebook's own max|coord| is ~1.5 -- "
          f"expect heavy overload/collapse if Y_max is much larger than that.")

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

    configs = [
        ("BERT-Base", "teacher_model",                         e8_cb, "E8"),
        ("BERT-Base", "teacher_model",                         z8_cb, "Z8"),
        ("TinyBERT",  "huawei-noah/TinyBERT_General_4L_312D", e8_cb, "E8"),
        ("TinyBERT",  "huawei-noah/TinyBERT_General_4L_312D", z8_cb, "Z8"),
    ]

    all_results = []
    loss_histories = {}   # {"BERT-Base_E8": [...], ...} for the graph
    for mn, ms, cb, qt in configs:
        res, sd, epoch_losses = run_qat(mn, ms, cb, qt,
                          train_items, val_items, test_items,
                          device, epochs=EPOCHS_USED, cb_used=C_B)
        all_results.append((mn, qt, res, sd))
        loss_histories[f"{mn}_{qt}"] = epoch_losses

    # Save the raw loss numbers + an immediate error-vs-epoch plot
    save_and_plot_loss_curves(loss_histories, EPOCHS_USED)

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