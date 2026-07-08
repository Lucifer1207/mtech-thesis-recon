"""export_all_models.py
────────────────────────────────────────────────────────────────
Exports ALL models to binary files including BOTH E8 variants:

  1. BERT-Base FP32         → bert_full_fp32.bin
  2. BERT-Base E8 16-bit    → bert_full_e8_16bit.bin  (3.0 bits/weight)
  3. BERT-Base E8 8-bit     → bert_full_e8_8bit.bin   (2.0 bits/weight)
  4. DistilBERT FP32        → distilbert_full_fp32.bin
  5. MobileBERT FP32        → mobilebert_full_fp32.bin
  6. TinyBERT (4L) FP32     → tinybert_full_fp32.bin
  7. bert_codebook_e8_16bit.bin  (65536 × 8D × 4B = 2 MB)
  8. bert_codebook_e8_8bit.bin   (256   × 8D × 4B = 8 KB)

Block formats:
  FP32 block    : 32 × float32                        = 128 bytes
  E8 16-bit blk : float32 scale + 4 × uint16 indices  =  12 bytes
  E8 8-bit  blk : float32 scale + 4 × uint8  indices  =   8 bytes

Effective bits/weight:
  E8 16-bit : (32 scale bits + 4×16 index bits) / 32 = 3.0 bits/weight
  E8 8-bit  : (32 scale bits + 4×8  index bits) / 32 = 2.0 bits/weight

NOTE (v2 change): file sizes are now read directly from disk using
os.path.getsize() instead of being computed manually, and BOTH MiB
(1024-based) and MB (1000-based) are printed to avoid ambiguity.
"""

import numpy as np
import struct
import itertools
import os
from transformers import AutoModel
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

BLOCK_SIZE = 32   # weights per block

# ─────────────────────────────────────────────────────────────
#  Helper: get true file size from disk, in both MiB and MB
# ─────────────────────────────────────────────────────────────
def get_size_mib_mb(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mib = size_bytes / (1024 * 1024)
    size_mb  = size_bytes / (1000 * 1000)
    return size_bytes, size_mib, size_mb

# ─────────────────────────────────────────────────────────────
#  Helper: flatten all state_dict weights → zero-pad → return
# ─────────────────────────────────────────────────────────────
def get_flat_weights(model):
    all_flat = np.concatenate([
        t.detach().cpu().numpy().astype(np.float32).flatten()
        for t in model.state_dict().values()
    ])
    rem = len(all_flat) % BLOCK_SIZE
    if rem:
        all_flat = np.concatenate(
            [all_flat, np.zeros(BLOCK_SIZE - rem, dtype=np.float32)])
    return all_flat

# ─────────────────────────────────────────────────────────────
#  Helper: write FP32 binary
# ─────────────────────────────────────────────────────────────
def export_fp32(all_flat, out_file):
    num_blocks = len(all_flat) // BLOCK_SIZE
    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", num_blocks))
        f.write(all_flat.tobytes())
    size_bytes, size_mib, size_mb = get_size_mib_mb(out_file)
    print(f"    FP32 → {out_file}  ({num_blocks:,} blocks, "
          f"{size_bytes:,} bytes, {size_mib:.2f} MiB, {size_mb:.2f} MB)")
    return num_blocks, size_mib, size_mb

# ─────────────────────────────────────────────────────────────
#  Helper: write E8 16-bit binary
#  Block: float32 scale (4B) + 4 × uint16 indices (8B) = 12B
# ─────────────────────────────────────────────────────────────
def export_e8_16bit(all_flat, codebook, codebook_norms, out_file):
    num_blocks = len(all_flat) // BLOCK_SIZE
    blocks = all_flat.reshape(num_blocks, BLOCK_SIZE)
    total_sq_err = 0.0
    total_sq_sig = 0.0

    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", num_blocks))
        for i, block in enumerate(blocks):
            if i % 200000 == 0:
                print(f"      E8-16bit block {i:,}/{num_blocks:,}...")
            max_val = float(np.max(np.abs(block)))
            scale   = float(max_val / 2.0) if max_val > 1e-8 else 1e-8
            subvecs = (block / scale).reshape(4, 8)
            dots    = np.dot(codebook, subvecs.T)
            dists   = codebook_norms[:, None] - 2 * dots
            best    = np.argmin(dists, axis=0).astype(np.uint16)
            recon   = codebook[best].flatten() * scale
            total_sq_err += float(np.sum((recon - block)**2))
            total_sq_sig += float(np.sum(block**2))
            f.write(struct.pack("<f", scale))
            f.write(best.tobytes())

    size_bytes, size_mib, size_mb = get_size_mib_mb(out_file)
    snr_db = 10 * np.log10(total_sq_sig / (total_sq_err + 1e-12))
    print(f"    E8-16bit → {out_file}  ({num_blocks:,} blocks, "
          f"{size_bytes:,} bytes, {size_mib:.2f} MiB, {size_mb:.2f} MB, "
          f"SNR: {snr_db:.2f} dB)")
    return size_mib, size_mb, snr_db

# ─────────────────────────────────────────────────────────────
#  Helper: write E8 8-bit binary
#  Block: float32 scale (4B) + 4 × uint8 indices (4B) = 8B
#  Uses first 256 entries of the 16-bit codebook.
# ─────────────────────────────────────────────────────────────
def export_e8_8bit(all_flat, codebook_256, codebook_norms_256, out_file):
    num_blocks = len(all_flat) // BLOCK_SIZE
    blocks = all_flat.reshape(num_blocks, BLOCK_SIZE)
    total_sq_err = 0.0
    total_sq_sig = 0.0

    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", num_blocks))
        for i, block in enumerate(blocks):
            if i % 200000 == 0:
                print(f"      E8-8bit block {i:,}/{num_blocks:,}...")
            max_val = float(np.max(np.abs(block)))
            scale   = float(max_val / 2.0) if max_val > 1e-8 else 1e-8
            subvecs = (block / scale).reshape(4, 8)
            dots    = np.dot(codebook_256, subvecs.T)
            dists   = codebook_norms_256[:, None] - 2 * dots
            best    = np.argmin(dists, axis=0).astype(np.uint8)
            recon   = codebook_256[best].flatten() * scale
            total_sq_err += float(np.sum((recon - block)**2))
            total_sq_sig += float(np.sum(block**2))
            f.write(struct.pack("<f", scale))
            f.write(best.tobytes())

    size_bytes, size_mib, size_mb = get_size_mib_mb(out_file)
    snr_db = 10 * np.log10(total_sq_sig / (total_sq_err + 1e-12))
    print(f"    E8-8bit  → {out_file}  ({num_blocks:,} blocks, "
          f"{size_bytes:,} bytes, {size_mib:.2f} MiB, {size_mb:.2f} MB, "
          f"SNR: {snr_db:.2f} dB)")
    return size_mib, size_mb, snr_db

# ─────────────────────────────────────────────────────────────
#  1. Build E8 codebooks
# ─────────────────────────────────────────────────────────────
print("="*60)
print("Building E8 codebooks...")
print("="*60)

e8_points = []
for p in itertools.product(range(-2, 3), repeat=8):
    if sum(p) % 2 == 0 and sum(x**2 for x in p) <= 4:
        e8_points.append(p)
for p in itertools.product([-1.5, -0.5, 0.5, 1.5], repeat=8):
    if int(sum(2*x for x in p)) % 2 == 0 and sum(x**2 for x in p) <= 4:
        e8_points.append(p)

e8_points = sorted(e8_points, key=lambda v: (sum(x**2 for x in v), v))

# 16-bit codebook: 65536 entries
codebook_16 = np.array(e8_points[:65536], dtype=np.float32)
if len(codebook_16) < 65536:
    codebook_16 = np.vstack([codebook_16,
                  np.zeros((65536 - len(codebook_16), 8), dtype=np.float32)])
norms_16 = np.sum(codebook_16**2, axis=1)

# 8-bit codebook: first 256 entries of 16-bit codebook
codebook_8  = codebook_16[:256].copy()
norms_8     = norms_16[:256].copy()

# Save both codebooks
codebook_16.tofile("bert_codebook_e8_16bit.bin")
codebook_8.tofile("bert_codebook_e8_8bit.bin")

_, cb16_mib, cb16_mb = get_size_mib_mb("bert_codebook_e8_16bit.bin")
_, cb8_mib, cb8_mb = get_size_mib_mb("bert_codebook_e8_8bit.bin")
print(f"  16-bit codebook saved → bert_codebook_e8_16bit.bin  "
      f"({cb16_mib:.2f} MiB, {cb16_mb:.2f} MB)")
print(f"  8-bit  codebook saved → bert_codebook_e8_8bit.bin   "
      f"({cb8_mib:.4f} MiB, {cb8_mb:.4f} MB)\n")

# ─────────────────────────────────────────────────────────────
#  2. Export all models
# ─────────────────────────────────────────────────────────────
MODELS = [
    {"name": "BERT-Base",      "id": "bert-base-uncased",
     "fp32": "bert_full_fp32.bin",
     "e8_16": "bert_full_e8_16bit.bin",
     "e8_8" : "bert_full_e8_8bit.bin",
     "do_e8": True},

    {"name": "TinyBERT (4L)",  "id": "huawei-noah/TinyBERT_General_4L_312D",
     "fp32": "tinybert_full_fp32.bin",
     "e8_16": "tinybert_full_e8_16bit.bin",
     "e8_8" : "tinybert_full_e8_8bit.bin",
     "do_e8": True},
]

summary = []
for m in MODELS:
    print(f"{'='*60}")
    print(f"  {m['name']}  ({m['id']})")
    print(f"{'='*60}")

    model       = AutoModel.from_pretrained(m["id"])
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    all_flat = get_flat_weights(model)
    del model

    # FP32
    nb, fp32_mib, fp32_mb = export_fp32(all_flat, m["fp32"])

    e8_16_mib = e8_16_mb = e8_8_mib = e8_8_mb = snr_16 = snr_8 = None
    if m["do_e8"]:
        e8_16_mib, e8_16_mb, snr_16 = export_e8_16bit(
            all_flat, codebook_16, norms_16, m["e8_16"])
        e8_8_mib, e8_8_mb, snr_8 = export_e8_8bit(
            all_flat, codebook_8,  norms_8,  m["e8_8"])

    summary.append({
        "name": m["name"], "params": total_params,
        "fp32_mib": fp32_mib, "fp32_mb": fp32_mb,
        "e8_16_mib": e8_16_mib, "e8_16_mb": e8_16_mb, "snr_16": snr_16,
        "e8_8_mib": e8_8_mib, "e8_8_mb": e8_8_mb, "snr_8": snr_8,
    })
    del all_flat
    print()

# ─────────────────────────────────────────────────────────────
#  3. Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*96)
print("  Export Summary  (sizes shown as MiB / MB)")
print("="*96)
print(f"  {'Model':<18} {'Params':>12} {'FP32':>18} {'E8-16bit':>18} "
      f"{'SNR-16':>8} {'E8-8bit':>18} {'SNR-8':>8}")
print(f"  {'-'*94}")
for r in summary:
    fp32_str = f"{r['fp32_mib']:.2f}/{r['fp32_mb']:.2f} MB"
    e16 = f"{r['e8_16_mib']:.2f}/{r['e8_16_mb']:.2f} MB" if r['e8_16_mib'] else "—"
    s16 = f"{r['snr_16']:.2f} dB" if r['snr_16'] else "—"
    e8  = f"{r['e8_8_mib']:.2f}/{r['e8_8_mb']:.2f} MB" if r['e8_8_mib'] else "—"
    s8  = f"{r['snr_8']:.2f} dB" if r['snr_8'] else "—"
    print(f"  {r['name']:<18} {r['params']:>12,} "
          f"{fp32_str:>18} {e16:>18} {s16:>8} {e8:>18} {s8:>8}")
print("="*96)

print("\nPush to emulator (use full adb path):")
files = [
    "bert_full_fp32.bin", "bert_full_e8_16bit.bin", "bert_full_e8_8bit.bin",
    "bert_codebook_e8_16bit.bin", "bert_codebook_e8_8bit.bin",
    "distilbert_full_fp32.bin", "mobilebert_full_fp32.bin",
    "tinybert_full_fp32.bin",
]
for fn in files:
    print(f"  adb push {fn} /data/local/tmp/")