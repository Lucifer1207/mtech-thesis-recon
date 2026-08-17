"""
analyze_checkpoint.py
─────────────────────────────────────────────────────────────
Post-hoc analysis of an ALREADY-TRAINED, ALREADY-SAVED QAT checkpoint
-- no retraining needed (per advisor's guidance). Computes, from the
saved (folded/quantized) weights plus the EXACT codebook used during
that training run:

  1. Overload check: how many post-scaling weight values fall outside
     the codebook's exact coordinate range (aggregated across all
     quantized layers) -- same diagnostic already used for QAD
     (stage_e8_qad_overload_diagnostic.py), now for QAT checkpoints.
  2. Value ranges: weight (on-disk), beta (if normalized),
     post-scaling (W_tilde), and codebook coordinate ranges.
  3. Codeword usage histogram: for each of the 256 codebook entries,
     how many quantized 8-D sub-vectors across the model map to it --
     shows whether the model uses the codebook efficiently/uniformly
     or concentrates on a handful of codewords. (New task.)

IMPORTANT CAVEAT: the saved checkpoint contains FOLDED (already-
quantized, on-lattice) weights, not the original raw/shadow fp32
weights from mid-training. This script recomputes sigma/beta by
treating the folded weight AS IF it were the raw weight -- a very
close approximation (this project's export pipelines have repeatedly
validated 100+dB SNR between folded and raw values elsewhere), but
not bit-exact to the live training-time sigma. Results computed this
way will be extremely close to, but not perfectly identical to, what
was measured live during training.

Codebook precision: rebuilt here with the IDENTICAL construction
(same seed=42 shuffle) used by the training scripts -- ma'am
specifically asked for the codebook range to be exact, not
approximated, which is why this is reproduced verbatim rather than
hardcoded from memory.

Supports TWO checkpoint types:
  --mode normalized     -- row-wise beta_i=Y_max/(C_b*sigma_i)
                            (stage_qat_all_normalization_CLA.py /
                            stage_qat_algorithm3_exact.py checkpoints)
                            requires --cb
  --mode nonnormalized  -- block-wise max(|32-block|)/2 scaling
                            (stage_qat_all_CLA.py checkpoints)

Usage (single checkpoint):
    python3 analyze_checkpoint.py \\
        --checkpoint-dir qat_bert_base_e8_v5_epochs20_cb5 \\
        --mode normalized --cb 5

Usage (batch -- auto-discovers all 4 model+quant combos, matching
each script's known naming convention):
    python3 analyze_checkpoint.py --batch normalized --epoch 20 --cb 5
    python3 analyze_checkpoint.py --batch normalized --epoch 20 --cb 4
    python3 analyze_checkpoint.py --batch normalized --epoch 20 --cb 3
    python3 analyze_checkpoint.py --batch nonnormalized --epoch 20
"""

import json, argparse
from itertools import product
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────
# Codebook builders -- copied verbatim from stage_qat_all_normalization_CLA.py
# / stage_qat_all_CLA.py so the EXACT same 256-point codebook (same
# seed=42 shuffle) is used here.
# ─────────────────────────────────────────────────────────────
def is_e8_point(x, tol=1e-6):
    x = np.array(x, dtype=float)
    is_int  = np.all(np.abs(x - np.round(x)) < tol)
    is_half = np.all(np.abs(x - np.round(x - 0.5) - 0.5) < tol)
    s = np.sum(x)
    even = (np.abs(s - np.round(s)) < tol and int(np.round(s)) % 2 == 0)
    return (is_int or is_half) and even


def build_e8_codebook(bits=8):
    candidates = []
    for coords in product([-2,-1,0,1,2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x,x)), x))
    for coords in product([-1.5,-0.5,0.5,1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x,x)), x))
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
    return np.array(ordered[:N], dtype=np.float32)


def build_z8_codebook(bits=8):
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
    return np.array(ordered[:N], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Which layers were actually quantized (same exclusion rule used
# throughout this project's training scripts)
# ─────────────────────────────────────────────────────────────
def get_target_layer_names(model, exclude_substrings=("classifier",)):
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                targets.append(full_name)
    return targets


# ─────────────────────────────────────────────────────────────
# Core analysis for one checkpoint
# ─────────────────────────────────────────────────────────────
def plot_codeword_histogram(counts, title, out_png):
    """Bar chart of per-codeword usage counts (log y-axis -- usage is
    almost always extremely non-uniform, so linear scale hides
    everything but the tallest bar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = np.asarray(counts)
    N = len(counts)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(N), counts, width=1.0, color="tab:blue", edgecolor="none")
    plt.xlabel("Codeword index (0..255)")
    plt.ylabel("Usage count (8-D sub-vectors assigned)")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    used = int((counts > 0).sum())
    print(f"  Saved histogram -> {out_png}  (codewords used: {used}/{N})")


def analyze_one_checkpoint(checkpoint_dir, mode, codebook, cb_value=None,
                            delta0=None, q=None, m=None):
    print(f"\n{'='*70}")
    label = f"mode={mode}" + (f", C_b={cb_value}" if cb_value is not None else "")
    print(f"Analyzing: {checkpoint_dir}  ({label})")
    print(f"{'='*70}")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    target_names = get_target_layer_names(model)
    sd = model.state_dict()

    codebook_min = float(codebook.min())
    codebook_max = float(codebook.max())
    codebook_norms = np.sum(codebook**2, axis=1)
    Y_max = float(np.max(np.abs(codebook)))  # only used in normalized mode
    Y_max_literal = (delta0 * (q ** m - 1) / 2.0) if mode == "literal" else None

    all_weight_vals, all_beta_vals, all_wtilde_vals = [], [], []
    total_outside = 0
    total_weights = 0
    codeword_counts = np.zeros(len(codebook), dtype=np.int64)

    for name in target_names:
        W = sd[f"{name}.weight"].detach().cpu().numpy().astype(np.float32)
        O, I = W.shape
        all_weight_vals.append(W.reshape(-1))

        if mode == "literal":
            # Same fixed-codebook row-wise scheme as "normalized", but
            # Y_max follows the paper's literal Appendix A.3 formula
            # (Delta0*(q^M-1)/2) instead of codebook.abs().max().
            sigma = np.clip(W.std(axis=1, ddof=0), 1e-8, None)
            beta = Y_max_literal / (cb_value * sigma)             # (O,)
            all_beta_vals.append(beta)
            W_tilde = W * beta[:, None]                           # (O,I)
        elif mode == "normalized":
            sigma = np.clip(W.std(axis=1, ddof=0), 1e-8, None)
            beta = Y_max / (cb_value * sigma)                    # (O,)
            all_beta_vals.append(beta)
            W_tilde = W * beta[:, None]                           # (O,I)
        else:  # nonnormalized: block-wise max(|32-block|)/2
            assert (O*I) % 32 == 0, f"{name}: total size not divisible by 32"
            W_flat = W.reshape(-1, 32)
            max_vals = np.max(np.abs(W_flat), axis=1, keepdims=True)
            scales = np.clip(max_vals / 2.0, 1e-8, None)
            W_tilde = (W_flat / scales).reshape(O, I)

        all_wtilde_vals.append(W_tilde.reshape(-1))

        outside = np.sum((W_tilde < codebook_min) | (W_tilde > codebook_max))
        total_outside += int(outside)
        total_weights += W_tilde.size

        # Codeword usage: nearest-codeword index for every 8-D sub-vector
        subvecs = W_tilde.reshape(-1, 8)
        dots = subvecs @ codebook.T
        dists = codebook_norms[None,:] - 2*dots
        nearest = np.argmin(dists, axis=1)
        counts = np.bincount(nearest, minlength=len(codebook))
        codeword_counts += counts

    all_weight_vals = np.concatenate(all_weight_vals)
    all_wtilde_vals = np.concatenate(all_wtilde_vals)
    pct_outside = 100.0 * total_outside / total_weights

    result = {
        "checkpoint_dir": checkpoint_dir,
        "mode": mode,
        "C_b": cb_value,
        "codebook_range": [codebook_min, codebook_max],
        "weight_value_range": [float(all_weight_vals.min()), float(all_weight_vals.max())],
        "post_scaling_value_range": [float(all_wtilde_vals.min()), float(all_wtilde_vals.max())],
        "overload_count": total_outside,
        "overload_total": total_weights,
        "overload_pct": pct_outside,
        "codeword_usage_counts": codeword_counts.tolist(),
        "codewords_used": int(np.sum(codeword_counts > 0)),
        "codewords_total": len(codebook),
        "codeword_usage_min": int(codeword_counts.min()),
        "codeword_usage_max": int(codeword_counts.max()),
    }
    if mode in ("normalized", "literal"):
        all_beta_vals = np.concatenate(all_beta_vals)
        result["beta_value_range"] = [float(all_beta_vals.min()), float(all_beta_vals.max())]
        if mode == "literal":
            result["Y_max_literal"] = Y_max_literal

    print(f"  Codebook range          : [{codebook_min:.4f}, {codebook_max:.4f}]")
    print(f"  Weight value range      : [{result['weight_value_range'][0]:.4f}, {result['weight_value_range'][1]:.4f}]")
    if mode == "literal":
        print(f"  Y_max (literal formula) : {Y_max_literal:.4f}")
    if mode in ("normalized", "literal"):
        print(f"  Beta value range        : [{result['beta_value_range'][0]:.4f}, {result['beta_value_range'][1]:.4f}]")
    print(f"  Post-scaling (W_tilde) range : [{result['post_scaling_value_range'][0]:.4f}, {result['post_scaling_value_range'][1]:.4f}]")
    print(f"  Overload                : {total_outside:,} / {total_weights:,}  ({pct_outside:.2f}%)")
    print(f"  Codewords used          : {result['codewords_used']} / {result['codewords_total']}")
    print(f"  Codeword usage count range : [{result['codeword_usage_min']}, {result['codeword_usage_max']}]  (min/max across the 256 codewords)")

    return result


# ─────────────────────────────────────────────────────────────
# Batch discovery -- matches each training script's known naming
# ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    ("BERT-Base", "bert_base"),
    ("TinyBERT",  "tinybert"),
]
QUANT_TYPES = ["E8", "Z8"]


def discover_checkpoints(batch_type, epoch, cb=None, delta0=None, q=None, m=None):
    dirs = []
    for model_label, model_tag in MODEL_CONFIGS:
        for quant in QUANT_TYPES:
            quant_tag = quant.lower()
            if batch_type == "normalized":
                d = f"qat_{model_tag}_{quant_tag}_v5_epochs{epoch}_cb{cb:g}"
            elif batch_type == "nonnormalized":
                d = f"qat_{model_tag}_{quant_tag}_v4_epochs{epoch}"
            elif batch_type == "literal":
                d = f"qat_{model_tag}_{quant_tag}_v5lit_epochs{epoch}_cb{cb:g}_d{delta0:g}_q{q}_m{m}"
            elif batch_type == "exact":
                d = f"qat_exact_{model_tag}_{quant_tag}_epochs{epoch}_cb{cb:g}_d{delta0:g}_q{q}_m{m}"
            else:
                raise ValueError(batch_type)
            dirs.append((model_label, quant, d))
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Post-hoc QAT checkpoint analysis (no retraining).")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Single checkpoint directory to analyze.")
    parser.add_argument("--mode", type=str, choices=["normalized", "nonnormalized", "literal"],
                       help="Required with --checkpoint-dir.")
    parser.add_argument("--cb", type=float, default=None,
                       help="C_b used for this checkpoint (required if --mode normalized/literal).")
    parser.add_argument("--quant-type", type=str, choices=["E8", "Z8"], default="E8",
                       help="Which codebook to use for --checkpoint-dir mode.")

    parser.add_argument("--batch", type=str,
                       choices=["normalized", "nonnormalized", "literal", "exact"], default=None,
                       help="Auto-discover and analyze all 4 model+quant checkpoints for a given config. "
                            "'literal' = fixed-codebook + Y_max=Delta0*(q^M-1)/2 "
                            "(stage_qat_all_normalization_literal_ymax.py).")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch count (for --batch).")
    parser.add_argument("--delta0", type=float, default=1.5, help="Delta0 (for --batch exact/literal).")
    parser.add_argument("--q", type=int, default=8,
                       help="q (for --batch exact/literal). Default 8 -- matches the paper's own "
                            "config (q in [2,8]; q=256 never appears in the paper).")
    parser.add_argument("--m", type=int, default=1, help="M (for --batch exact/literal).")

    args = parser.parse_args()

    e8_codebook = build_e8_codebook()
    z8_codebook = build_z8_codebook()

    results = []

    if args.batch:
        if args.epoch is None:
            raise SystemExit("--batch requires --epoch")
        if args.batch in ("normalized", "exact", "literal") and args.cb is None:
            raise SystemExit(f"--batch {args.batch} requires --cb")

        mode = {"nonnormalized": "nonnormalized", "literal": "literal"}.get(args.batch, "normalized")
        dirs = discover_checkpoints(args.batch, args.epoch, args.cb, args.delta0, args.q, args.m)
        for model_label, quant, d in dirs:
            codebook = e8_codebook if quant == "E8" else z8_codebook
            try:
                r = analyze_one_checkpoint(
                    d, mode, codebook,
                    cb_value=args.cb if mode in ("normalized", "literal") else None,
                    delta0=args.delta0, q=args.q, m=args.m)
                r["model"] = model_label
                r["quant_type"] = quant
                results.append(r)
                plot_codeword_histogram(
                    r["codeword_usage_counts"],
                    f"{d}\n({mode}, {model_label}+{quant})",
                    f"codeword_hist_{d}.png")
            except OSError as e:
                print(f"\n  Could not load {d}/ -- skipping ({e})")

        out_json = f"analysis_{args.batch}_epoch{args.epoch}"
        if args.cb is not None:
            out_json += f"_cb{args.cb:g}"
        out_json += ".json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nSaved combined results -> {out_json}")

        print(f"\n{'='*70}")
        print("BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<12}{'Quant':>6}{'Overload%':>12}{'CodewordsUsed':>16}")
        for r in results:
            print(f"{r['model']:<12}{r['quant_type']:>6}{r['overload_pct']:>11.2f}%"
                  f"{r['codewords_used']:>10}/{r['codewords_total']}")

    elif args.checkpoint_dir:
        if args.mode is None:
            raise SystemExit("--checkpoint-dir requires --mode")
        if args.mode in ("normalized", "literal") and args.cb is None:
            raise SystemExit(f"--mode {args.mode} requires --cb")
        codebook = e8_codebook if args.quant_type == "E8" else z8_codebook
        r = analyze_one_checkpoint(
            args.checkpoint_dir, args.mode, codebook,
            cb_value=args.cb if args.mode in ("normalized", "literal") else None,
            delta0=args.delta0, q=args.q, m=args.m)
        out_json = f"analysis_{args.checkpoint_dir}.json"
        with open(out_json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved -> {out_json}")
        plot_codeword_histogram(
            r["codeword_usage_counts"],
            f"{args.checkpoint_dir}\n({args.mode})",
            f"codeword_hist_{args.checkpoint_dir}.png")

    else:
        raise SystemExit("Provide either --checkpoint-dir (+--mode) or --batch (+--epoch).")


if __name__ == "__main__":
    main()