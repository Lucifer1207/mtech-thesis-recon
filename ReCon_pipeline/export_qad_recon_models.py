"""
export_qad_recon_models.py (v2 -- corrected)
────────────────────────────────────────────────────────────────
Exports the TinyBERT E8-lattice QAD student (from stage_e8_qad.py v2)
to a binary block file for full_model_benchmark_qad.cpp.

FIXES vs v1:
  1. Loaded via AutoModel instead of AutoModelForSequenceClassification
     -- inconsistent with how the checkpoint was actually created
     (a sequence-classification model), silently dropping/ignoring
     the classifier head's structure.
  2. Built its OWN E8 codebook using a DIFFERENT construction
     (lexicographic tie-break + a norm²≤4 cutoff filter) than
     stage_e8_qad.py's training-time codebook (pure norm-sort, no
     shuffle, no cutoff). Even though both aim for "256 closest E8
     points", a different tie-break rule at the shell boundary can
     select a different subset -- meaning indices wouldn't mean the
     same lattice points as during training. Codebook-building code
     here is now copied verbatim from stage_e8_qad.py.
  3. Re-derived scale via max(|already-quantized block|)/2, which
     only works on RAW unquantized weights (this is now moot since
     v2 of stage_e8_qad.py saves genuinely folded/quantized weights,
     but the same exact-decomposition method validated for the
     BERT-Base/TinyBERT QAT export is used here anyway, for identical
     lossless-recovery guarantees and to stay consistent with the
     rest of the pipeline).
  4. Was quantizing the ENTIRE state_dict rather than only the Linear
     layers that were actually QAT-wrapped (excludes "classifier",
     matching stage_e8_qad.py's transform_to_e8_lattice exactly).

Output: tinybert_e8_qad_quant.bin (block_e8_8 format, 8 bytes/32
weights) + tinybert_e8_qad_codebook_8bit.bin (256x8 float32, this
QAD-specific codebook -- NOT the same file as the QAT pipeline's
E8 codebook, since the two are built with different code and are
not guaranteed to be identical; mixing them would silently corrupt
decoding).
"""

import struct
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

BLOCK_SIZE = 32
CKPT_DIR   = "student_model_lattice_e8"
OUT_QUANT  = "tinybert_e8_qad_quant.bin"
OUT_CODEBOOK = "tinybert_e8_qad_codebook_8bit.bin"


# ─────────────────────────────────────────────────────────────
# 1. Codebook builder -- copied verbatim from stage_e8_qad.py so
#    the 256-point set exactly matches what training used.
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
    print("Building E8 codebook (identical to stage_e8_qad.py)...")
    candidates = []
    for coords in product([-2, -1, 0, 1, 2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    for coords in product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    candidates.sort(key=lambda item: item[0])   # matches stage_e8_qad.py exactly
    codebook = [item[1] for item in candidates[:256]]
    print(f"  Codebook = 256 closest E8 points")
    return np.array(codebook, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2. Identify which .weight tensors were actually QAT/QAD-wrapped
#    (mirrors stage_e8_qad.py's transform_to_e8_lattice exactly:
#    every nn.Linear except a child literally named "classifier")
# ─────────────────────────────────────────────────────────────
def get_qad_target_names(model):
    targets = set()
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                if "classifier" in child_name:
                    continue
                full_name = f"{name}.{child_name}" if name else child_name
                targets.add(f"{full_name}.weight")
    return targets


# ─────────────────────────────────────────────────────────────
# 3. Exact decomposition of already-quantized blocks (identical
#    method validated for the BERT-Base/TinyBERT QAT export --
#    least-squares scalar fit against every codeword, positive-scale
#    constrained, using whichever of the 4 sub-vectors has the
#    largest norm as the reference for recovering the shared scale).
# ─────────────────────────────────────────────────────────────
def export_lattice_8bit(all_flat, codebook, out_file, chunk=20000):
    assert len(all_flat) % BLOCK_SIZE == 0, (
        f"Flat weight count {len(all_flat)} not a multiple of {BLOCK_SIZE}"
    )
    codebook_norms = np.sum(codebook ** 2, axis=1)
    num_blocks = len(all_flat) // BLOCK_SIZE
    blocks = all_flat.reshape(num_blocks, 4, 8)

    out_dtype = np.dtype([("scale", "<f4"), ("idx", "u1", (4,))])
    total_sq_err = 0.0
    total_sq_sig = 0.0

    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", num_blocks))
        for start in range(0, num_blocks, chunk):
            end = min(start + chunk, num_blocks)
            blk = blocks[start:end]
            b = blk.shape[0]

            norms4 = np.sum(blk ** 2, axis=2)
            ref = np.argmax(norms4, axis=1)
            v_ref = blk[np.arange(b), ref, :]

            dots = v_ref @ codebook.T
            alphas = dots / (codebook_norms[None, :] + 1e-12)
            vref_norm2 = np.sum(v_ref ** 2, axis=1, keepdims=True)
            resid = vref_norm2 - alphas * dots
            resid = np.where(alphas > 1e-9, resid, np.inf)   # positive-scale only
            best_j = np.argmin(resid, axis=1)
            S = alphas[np.arange(b), best_j]
            S = np.where(~np.isfinite(S) | (np.abs(S) < 1e-12), 1e-8, S)

            idxs = np.zeros((b, 4), dtype=np.uint8)
            idxs[np.arange(b), ref] = best_j.astype(np.uint8)

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
    print(f"  -> {out_file}  ({num_blocks:,} blocks, {size_bytes:,} bytes, "
          f"{size_bytes/1024/1024:.2f} MB, exact-decomposition SNR: {snr_db:.2f} dB "
          f"[should be very high / near-lossless])")
    return num_blocks, size_bytes, snr_db


# ─────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Loading QAD checkpoint from {CKPT_DIR}/ ...")
    print("=" * 60)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
    model.eval()

    codebook = generate_e8_8bit_codebook()
    codebook.tofile(OUT_CODEBOOK)
    print(f"  Codebook saved -> {OUT_CODEBOOK} ({os.path.getsize(OUT_CODEBOOK)} bytes)\n")

    qad_names = get_qad_target_names(model)
    sd = model.state_dict()

    quant_parts, plain_parts = [], []
    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        if name in qad_names:
            assert t.dim() == 2, f"{name} expected a 2D Linear weight, got {tuple(t.shape)}"
            O, I = t.shape
            rem = I % BLOCK_SIZE
            pad = (BLOCK_SIZE - rem) % BLOCK_SIZE
            if pad:
                # Per-row padding keeps block boundaries clean whenever
                # a layer's row width wouldn't otherwise divide evenly
                # by 32 (e.g. TinyBERT's 312), consistent with the
                # careful handling used elsewhere in this pipeline.
                t = torch.cat([t, torch.zeros(O, pad, dtype=t.dtype)], dim=1)
            arr = t.numpy().astype(np.float32).flatten()
            assert arr.size % BLOCK_SIZE == 0
            quant_parts.append(arr)
        else:
            plain_parts.append(t.numpy().astype(np.float32).flatten())

    quant_flat = np.concatenate(quant_parts)
    plain_flat = np.concatenate(plain_parts) if plain_parts else np.array([], dtype=np.float32)

    print(f"Quantized core : {len(qad_names)} Linear-layer weight tensors, "
          f"{quant_flat.size:,} params")
    print(f"FP32 remainder : {len(sd) - len(qad_names)} tensors, "
          f"{plain_flat.size:,} params (embeddings/layernorms/classifier/biases)\n")

    nb, size_bytes, snr = export_lattice_8bit(quant_flat, codebook, OUT_QUANT)

    plain_mb = plain_flat.size * 4 / (1024 * 1024)
    total_mb = size_bytes / (1024 * 1024) + plain_mb
    print(f"\nTotal model footprint (quantized core + FP32 remainder): "
          f"{total_mb:.2f} MB  (remainder not written to disk, "
          f"not pushed to device)")

    print("\nPush to device (adb push):")
    print(f"  adb push {OUT_QUANT} /data/local/tmp/")
    print(f"  adb push {OUT_CODEBOOK} /data/local/tmp/")


if __name__ == "__main__":
    main()
