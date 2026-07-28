"""
plot_epoch_comparison.py (v2 -- auto-discovers however many epoch
experiments have been run, not just a hardcoded 3-vs-5 pair)
─────────────────────────────────────────────────────────────
Overlays training loss curves from ALL qat_normalization_loss_history_
epochs*.json files found in the current directory -- one subplot per
model+quant combo (BERT-Base+E8, BERT-Base+Z8, TinyBERT+E8,
TinyBERT+Z8), with one line per epoch-count experiment found (3, 5,
10, 12, or whatever combination you've actually run).

FIX vs v1: v1 hardcoded exactly two epoch counts (3 and 5) via a
fixed FILES dict. Now that stage_qat_all_normalization_CLA.py takes
--epochs as a command-line argument (so you can run experiments at
any epoch count, e.g. --epochs 10, --epochs 12), this script instead
GLOBS for every qat_normalization_loss_history_epochs*.json file
present and plots whatever it finds -- no hardcoded list to update
each time you run a new epoch count.

Run this AFTER stage_qat_all_normalization_CLA.py has been run at
however many different --epochs values you want to compare, e.g.:
    python3 stage_qat_all_normalization_CLA.py --epochs 3
    python3 stage_qat_all_normalization_CLA.py --epochs 5
    python3 stage_qat_all_normalization_CLA.py --epochs 10
    python3 stage_qat_all_normalization_CLA.py --epochs 12
    python3 plot_epoch_comparison.py
"""

import json
import glob
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATTERN = "qat_normalization_loss_history_epochs*.json"
OUT_PNG = "qat_normalization_loss_comparison_all_epochs.png"

# A fixed, distinct color per epoch-count line so the same epoch count
# always gets the same color across every subplot -- makes the legend
# easy to read even with many experiments overlaid.
COLOR_CYCLE = ["tab:orange", "tab:blue", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def main():
    paths = sorted(glob.glob(PATTERN))
    if not paths:
        print(f"  No files matching {PATTERN} found in the current directory.")
        print(f"  Run stage_qat_all_normalization_CLA.py with --epochs <N> "
              f"at least once first.")
        return

    # Parse the epoch count out of each filename (e.g. "...epochs10.json" -> 10)
    loaded = {}   # {epoch_count: {"BERT-Base_E8": [...], ...}}
    for path in paths:
        m = re.search(r"epochs(\d+)\.json$", path)
        if not m:
            print(f"  Skipping {path} -- couldn't parse an epoch count from its name.")
            continue
        n_epochs = int(m.group(1))
        with open(path) as f:
            loaded[n_epochs] = json.load(f)["histories"]

    if len(loaded) < 1:
        print("  No valid, parseable files found. Stopping without writing a plot.")
        return

    epoch_counts = sorted(loaded.keys())
    print(f"  Found {len(epoch_counts)} epoch-count experiment(s): {epoch_counts}")

    # All 4 model+quant combos, gathered across whichever runs have them
    model_keys = sorted(set().union(*(set(h.keys()) for h in loaded.values())))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, key in enumerate(model_keys[:4]):
        ax = axes[i]
        for j, n_epochs in enumerate(epoch_counts):
            color = COLOR_CYCLE[j % len(COLOR_CYCLE)]
            if key in loaded[n_epochs]:
                losses = loaded[n_epochs][key]
                ax.plot(range(1, len(losses) + 1), losses,
                       marker="o", color=color, label=f"{n_epochs} epochs")
        ax.set_title(key.replace("_", " + "))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"QAT + Normalization: training loss vs epoch "
                f"({', '.join(str(e) for e in epoch_counts)} epochs)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"\n  Saved comparison plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
