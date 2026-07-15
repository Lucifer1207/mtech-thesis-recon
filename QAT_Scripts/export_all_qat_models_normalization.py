"""
export_all_models_v3.py
────────────────────────────────────────────────────────────────
Exports the 4 actual QAT-trained checkpoints (BERT-Base/TinyBERT x
E8/Z8) to on-device binary block files for full_model_benchmark_recon.cpp.

FIXES vs export_all_models_v2.py:
  1. Loads the REAL trained checkpoints (qat_*_v4/) via
     AutoModelForSequenceClassification, instead of fresh generic
     HF pretrained weights via AutoModel.
  2. Adds Z8 codebook generation + Z8 export (v2 only had E8).
  3. Only lattice-quantizes the same Linear-layer .weight tensors
     that stage_qat_all.py actually QAT-wrapped (every nn.Linear
     except ones with "classifier" in the name). Everything else
     (embeddings, layernorms, all Linear .bias tensors, classifier
     weight/bias) is reported separately as FP32 -- exactly mirroring
     what was actually trained. The old script instead flattened and
     quantized the ENTIRE state_dict, which would have quantized
     things that were deliberately kept full-precision during QAT.
  4. Codebook generation is copied verbatim (same enumeration, same
     seed=42 shuffle/tie-break) from stage_qat_all.py's
     build_e8_codebook / build_z8_codebook, so the 256-point set
     exactly matches what training used. This matters because we are
     re-deriving (scale, index) from weights that are ALREADY
     quantized -- a mismatched codebook could snap them to a
     different lattice point and introduce real (avoidable) error.
  5. Block format matches stage_qat_all.py v4 exactly: ONE scale per
     32-element block, split into 4 sub-vectors of 8 for the
     E8/Z8 codebook lookup (float scale + 4x uint8 idx = 8 bytes/
     32 weights = 2.0 bits/weight for the quantized core).

Output per model+quant combo: one block_e8_8-format .bin file
containing ONLY the quantized Linear-layer core (this is what's
pushed to the phone and benchmarked). Two codebook .bin files
(E8, Z8). The FP32 remainder (embeddings/layernorms/classifier) is
NOT written to disk -- the benchmark harness only exercises the
quantized core, and the remainder is reported for total-footprint
bookkeeping only.
"""

import json
import struct
import os
from collections import defaultdict
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

BLOCK_SIZE = 32   # weights per block -- matches stage_qat_all.py v4

CHECKPOINTS = [
    {"name": "BERT-Base", "quant": "E8", "dir": "qat_bert_base_e8_v5",
     "out": "bert_base_e8_quant.bin", "rows": 768, "cols": 768},
    {"name": "BERT-Base", "quant": "Z8", "dir": "qat_bert_base_z8_v5",
     "out": "bert_base_z8_quant.bin", "rows": 768, "cols": 768},
    {"name": "TinyBERT",  "quant": "E8", "dir": "qat_tinybert_e8_v5",
     "out": "tinybert_e8_quant.bin",   "rows": 312, "cols": 312},
    {"name": "TinyBERT",  "quant": "Z8", "dir": "qat_tinybert_z8_v5",
     "out": "tinybert_z8_quant.bin",   "rows": 312, "cols": 312},
]

BITS = 8


# ─────────────────────────────────────────────────────────────
# 1. Codebook builders -- copied verbatim from stage_qat_all.py
#    so the 256-point set exactly matches what training used.
# ─────────────────────────────────────────────────────────────
def is_e8_point(x, tol=1e-6):
    x = np.array(x, dtype=float)
    is_int  = np.all(np.abs(x - np.round(x)) < tol)
    is_half = np.all(np.abs(x - np.round(x - 0.5) - 0.5) < tol)
    s = np.sum(x)
    even = (np.abs(s - np.round(s)) < tol and int(np.round(s)) % 2 == 0)
    return (is_int or is_half) and even


def build_e8_codebook_np(bits):
    print(f"\n  ── E8 CODEBOOK FORMATION ({bits}-bit) ──────────────")
    candidates = []
    for coords in product([-2, -1, 0, 1, 2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    for coords in product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    print(f"  Total valid E8 points found: {len(candidates):,}")

    rng = np.random.default_rng(seed=42)
    norm_groups = defaultdict(list)
    for norm_sq, x in candidates:
        norm_groups[round(norm_sq, 4)].append(x)
    ordered = []
    for norm in sorted(norm_groups.keys()):
        grp = norm_groups[norm]
        for i in rng.permutation(len(grp)):
            ordered.append(grp[i])

    N = 2 ** bits
    codebook = np.array(ordered[:N], dtype=np.float32)
    print(f"  Codebook = {N} closest E8 points")
    print(f"  ── E8 CODEBOOK FORMATION COMPLETE ──────────────────")
    return codebook


def build_z8_codebook_np(bits):
    print(f"\n  ── Z8 CODEBOOK FORMATION ({bits}-bit) ──────────────")
    candidates = []
    for coords in product(range(-3, 4), repeat=8):
        x = np.array(coords, dtype=float)
        candidates.append((float(np.dot(x, x)), x))

    rng = np.random.default_rng(seed=42)
    norm_groups = defaultdict(list)
    for ns, x in candidates:
        norm_groups[round(ns, 4)].append(x)

    ordered = []
    N = 2 ** bits
    for norm in sorted(norm_groups.keys()):
        grp = norm_groups[norm]
        for i in rng.permutation(len(grp)):
            ordered.append(grp[i])
        if len(ordered) >= N:
            break

    codebook = np.array(ordered[:N], dtype=np.float32)
    print(f"  Codebook = {N} closest Z8 integer points")
    print(f"  ── Z8 CODEBOOK FORMATION COMPLETE ──────────────────")
    return codebook


# ─────────────────────────────────────────────────────────────
# 2. Identify which .weight tensors were actually QAT-wrapped
#    (mirrors stage_qat_all.py's replace_linear_with_qat exactly)
# ─────────────────────────────────────────────────────────────
def get_qat_target_names(model, exclude_substrings=("classifier",)):
    targets = set()
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                targets.add(f"{full_name}.weight")
    return targets


# ─────────────────────────────────────────────────────────────
# 3. Block export: EXACT decomposition of already-quantized blocks.
#
#    IMPORTANT: we do NOT re-derive scale via max(|block|)/2 here.
#    That formula is only valid applied ONCE to the ORIGINAL raw
#    weights; applied a second time to an ALREADY-quantized block
#    (== scale_true * codeword) it systematically underestimates
#    scale_true, because most codewords don't have max-abs component
#    exactly 2 (only some shell-1 points do). That bug is what
#    produced the ~6dB SNR in the first export attempt -- it's a
#    real, systematic error, not floating-point noise near ties.
#
#    Instead: since block == scale_true * codebook[idx] EXACTLY (up
#    to float32 rounding) for each of the 4 sub-vectors sharing one
#    scale, we recover (scale, idx) losslessly via a least-squares
#    scalar fit against every codeword and taking the one with ~zero
#    residual. This is an exact inverse, not an approximation -- SNR
#    should now come out very high (near float32 precision limits).
# ─────────────────────────────────────────────────────────────
def export_lattice_8bit(all_flat, codebook, out_file, chunk=20000):
    assert len(all_flat) % BLOCK_SIZE == 0, (
        f"Flat weight count {len(all_flat)} not a multiple of {BLOCK_SIZE} "
        f"-- block alignment would not match training. Check that every "
        f"QAT-wrapped Linear layer's weight.numel() is itself a multiple "
        f"of 32."
    )
    codebook_norms = np.sum(codebook ** 2, axis=1)              # (256,)
    num_blocks = len(all_flat) // BLOCK_SIZE
    blocks = all_flat.reshape(num_blocks, 4, 8)                  # (B,4,8)

    out_dtype = np.dtype([("scale", "<f4"), ("idx", "u1", (4,))])
    total_sq_err = 0.0
    total_sq_sig = 0.0

    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", num_blocks))
        for start in range(0, num_blocks, chunk):
            end = min(start + chunk, num_blocks)
            blk = blocks[start:end]                              # (b,4,8)
            b = blk.shape[0]
            if start % (chunk * 20) == 0 and start > 0:
                print(f"      block {start:,}/{num_blocks:,}...")

            # Use whichever of the 4 sub-vectors has the largest norm
            # as the reference for recovering the shared scale --
            # avoids numerical issues from near-zero sub-vectors.
            norms4 = np.sum(blk ** 2, axis=2)                     # (b,4)
            ref = np.argmax(norms4, axis=1)                        # (b,)
            v_ref = blk[np.arange(b), ref, :]                       # (b,8)

            # Least-squares scalar fit of v_ref against every codeword:
            # alpha_j = (v_ref . cb_j) / ||cb_j||^2
            # residual^2 = ||v_ref||^2 - alpha_j * (v_ref . cb_j)
            #
            # IMPORTANT: training's scale is always POSITIVE by construction
            # (S = max(|block|)/2 >= 0). Without constraining alpha > 0 here,
            # a codeword's negation can tie on residual for this ONE
            # sub-vector (since alpha_j * cb_j == (-alpha_j) * (-cb_j)), and
            # if the Z8 codebook isn't perfectly symmetric under negation at
            # its truncated boundary shell (it isn't -- it's built via random
            # shuffle-and-cut to hit exactly 256 points), picking the
            # negative-alpha branch here silently corrupts the OTHER 3
            # sub-vectors, which all share this one scale. Restricting to
            # alpha > 0 removes this spurious tie entirely. (E8 was already
            # unaffected by this, but the constraint is harmless there too.)
            dots = v_ref @ codebook.T                                # (b,256)
            alphas = dots / (codebook_norms[None, :] + 1e-12)        # (b,256)
            vref_norm2 = np.sum(v_ref ** 2, axis=1, keepdims=True)    # (b,1)
            resid = vref_norm2 - alphas * dots                        # (b,256)
            resid = np.where(alphas > 1e-9, resid, np.inf)             # positive-scale only
            best_j = np.argmin(resid, axis=1)                          # (b,)
            S = alphas[np.arange(b), best_j]                            # (b,)
            S = np.where(~np.isfinite(S) | (np.abs(S) < 1e-12), 1e-8, S)

            idxs = np.zeros((b, 4), dtype=np.uint8)
            idxs[np.arange(b), ref] = best_j.astype(np.uint8)

            # Decode the other 3 sub-vectors using the now-known scale
            for i in range(4):
                mask = (ref != i)
                if not np.any(mask):
                    continue
                v = blk[mask, i, :]
                s_m = S[mask][:, None]
                scaled = v / s_m
                d = codebook_norms[None, :] - 2 * (scaled @ codebook.T)
                idxs[mask, i] = np.argmin(d, axis=1).astype(np.uint8)

            recon = codebook[idxs.reshape(-1)].reshape(b, 4, 8) * S[:, None, None]
            total_sq_err += float(np.sum((recon - blk) ** 2))
            total_sq_sig += float(np.sum(blk ** 2))

            rec = np.zeros(b, dtype=out_dtype)
            rec["scale"] = S.astype(np.float32)
            rec["idx"] = idxs
            f.write(rec.tobytes())

    size_bytes = os.path.getsize(out_file)
    snr_db = 10 * np.log10(total_sq_sig / (total_sq_err + 1e-12))
    print(f"    -> {out_file}  ({num_blocks:,} blocks, {size_bytes:,} bytes, "
          f"{size_bytes/1024/1024:.2f} MB, exact-decomposition SNR: {snr_db:.2f} dB "
          f"[should now be very high / near-lossless])")
    return num_blocks, size_bytes, snr_db


# ─────────────────────────────────────────────────────────────
# 4. Build & save both codebooks
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Building codebooks (identical to stage_qat_all.py)...")
print("=" * 60)
codebook_e8 = build_e8_codebook_np(BITS)
codebook_z8 = build_z8_codebook_np(BITS)

codebook_e8.tofile("bert_codebook_e8_8bit.bin")
codebook_z8.tofile("bert_codebook_z8_8bit.bin")
print(f"\n  E8 codebook saved -> bert_codebook_e8_8bit.bin "
      f"({os.path.getsize('bert_codebook_e8_8bit.bin')} bytes)")
print(f"  Z8 codebook saved -> bert_codebook_z8_8bit.bin "
      f"({os.path.getsize('bert_codebook_z8_8bit.bin')} bytes)\n")

# ─────────────────────────────────────────────────────────────
# 5. Export each checkpoint
# ─────────────────────────────────────────────────────────────
summary = []
for ck in CHECKPOINTS:
    print(f"{'='*60}")
    print(f"  {ck['name']} + {ck['quant']}   ({ck['dir']}/)")
    print(f"{'='*60}")

    model = AutoModelForSequenceClassification.from_pretrained(ck["dir"])
    model.eval()

    qat_names = get_qat_target_names(model)
    sd = model.state_dict()

    quant_parts, plain_parts = [], []
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        if name in qat_names:
            # v5 training pads EACH ROW individually to a multiple of
            # 32 (scale is now per-ROW, not per-block -- a storage
            # block must never span two rows with different scales).
            # Export must mirror this exactly, or a 32-element block
            # could straddle two rows here even though it never did
            # during training, breaking the exact-decomposition step
            # below (same failure mode fixed earlier for Z8, but this
            # time from a row/scale mismatch rather than a codebook
            # sign ambiguity). No-op whenever a tensor's column count
            # is already 32-aligned (e.g. BERT-Base's 768, 3072);
            # matters whenever it isn't (e.g. TinyBERT's 312, 1200).
            assert t.dim() == 2, (
                f"{name} expected a 2D Linear weight, got shape {tuple(t.shape)}"
            )
            O, I = t.shape
            rem = I % BLOCK_SIZE
            pad = (BLOCK_SIZE - rem) % BLOCK_SIZE
            if pad:
                t = torch.cat([t, torch.zeros(O, pad, dtype=t.dtype)], dim=1)
            arr = t.numpy().astype(np.float32).flatten()
            assert arr.size % BLOCK_SIZE == 0, (
                f"{name} has {arr.size} elements after row-padding, "
                f"not a multiple of 32"
            )
            quant_parts.append(arr)
        else:
            arr = t.numpy().astype(np.float32).flatten()
            plain_parts.append(arr)

    quant_flat = np.concatenate(quant_parts)
    plain_flat = np.concatenate(plain_parts) if plain_parts else np.array([], dtype=np.float32)

    print(f"  Quantized core : {len(qat_names)} Linear-layer weight tensors, "
          f"{quant_flat.size:,} params")
    print(f"  FP32 remainder : {len(sd) - len(qat_names)} tensors, "
          f"{plain_flat.size:,} params (embeddings/layernorms/classifier/biases)")

    codebook = codebook_e8 if ck["quant"] == "E8" else codebook_z8
    nb, size_bytes, snr = export_lattice_8bit(quant_flat, codebook, ck["out"])

    plain_mb = plain_flat.size * 4 / (1024 * 1024)
    total_mb = size_bytes / (1024 * 1024) + plain_mb
    print(f"  Total model footprint (quantized core + FP32 remainder): "
          f"{total_mb:.2f} MB  (remainder not written to disk, "
          f"not pushed to device -- benchmark harness only exercises "
          f"the quantized core)")

    summary.append({
        "name": ck["name"], "quant": ck["quant"], "file": ck["out"],
        "blocks": nb, "size_mb": size_bytes/(1024*1024),
        "snr_db": snr, "total_mb": total_mb,
        "rows": ck["rows"], "cols": ck["cols"],
    })
    del model
    print()

# ─────────────────────────────────────────────────────────────
# 6. Summary + adb push commands
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  Export Summary")
print("=" * 90)
print(f"  {'Model':<12} {'Quant':>6} {'Blocks':>10} {'Core MB':>10} "
      f"{'SNR (dB)':>10} {'Total MB':>10} {'Shape':>12}")
print(f"  {'-'*88}")
for r in summary:
    print(f"  {r['name']:<12} {r['quant']:>6} {r['blocks']:>10,} "
          f"{r['size_mb']:>10.2f} {r['snr_db']:>10.2f} {r['total_mb']:>10.2f} "
          f"{r['rows']}x{r['cols']:>7}")
print("=" * 90)

print("\nPush to device (adb push):")
for r in summary:
    print(f"  adb push {r['file']} /data/local/tmp/")
print("  adb push bert_codebook_e8_8bit.bin /data/local/tmp/")
print("  adb push bert_codebook_z8_8bit.bin /data/local/tmp/")