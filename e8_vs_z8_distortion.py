import numpy as np
from itertools import product
from collections import defaultdict

np.random.seed(42)  # reproducibility

# ─────────────────────────────────────────────────────────────
# PART 1: E8 lattice generation (same logic as our earlier code)
# ─────────────────────────────────────────────────────────────

def is_e8_point(x, tol=1e-6):
    x = np.array(x, dtype=float)
    is_integer = np.all(np.abs(x - np.round(x)) < tol)
    is_half_integer = np.all(np.abs(x - np.round(x - 0.5) - 0.5) < tol)
    sum_val = np.sum(x)
    is_even_sum = (np.abs(sum_val - np.round(sum_val)) < tol and
                   int(np.round(sum_val)) % 2 == 0)
    return (is_integer or is_half_integer) and is_even_sum


def generate_all_e8_points():
    candidates = []
    for coords in product([-2, -1, 0, 1, 2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    for coords in product([-1.5, -0.5, 0.5, 1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            candidates.append((float(np.dot(x, x)), x))
    return candidates


def generate_e8_codebook(bits, all_points):
    """Sort by norm, shuffle within same-norm group for unbiased
    selection (same fix we applied earlier to avoid sign bias)."""
    num_codewords = 2 ** bits
    rng = np.random.default_rng(seed=42)

    norm_groups = defaultdict(list)
    for norm_sq, x in all_points:
        norm_groups[round(norm_sq, 4)].append(x)

    sorted_norms = sorted(norm_groups.keys())
    ordered_points = []
    for norm in sorted_norms:
        group = norm_groups[norm]
        indices = rng.permutation(len(group))
        for i in indices:
            ordered_points.append(group[i])

    return np.array(ordered_points[:num_codewords])


# ─────────────────────────────────────────────────────────────
# PART 2: Z8 lattice generation
# Z8 = all-integer points in 8D.
# ─────────────────────────────────────────────────────────────

def generate_z8_codebook(bits, search_range=2):
    num_codewords = 2 ** bits
    rng = np.random.default_rng(seed=42)

    candidates = []
    for coords in product(range(-search_range, search_range + 1), repeat=8):
        x = np.array(coords, dtype=float)
        norm_sq = float(np.dot(x, x))
        candidates.append((norm_sq, x))

    norm_groups = defaultdict(list)
    for norm_sq, x in candidates:
        norm_groups[round(norm_sq, 4)].append(x)

    sorted_norms = sorted(norm_groups.keys())
    ordered_points = []
    for norm in sorted_norms:
        group = norm_groups[norm]
        indices = rng.permutation(len(group))
        for i in indices:
            ordered_points.append(group[i])
        if len(ordered_points) >= num_codewords:
            break

    return np.array(ordered_points[:num_codewords])


# ─────────────────────────────────────────────────────────────
# PART 3: Quantization + distortion measurement
# ─────────────────────────────────────────────────────────────

def compute_scale_factor(weight_vector):
    """S = ||x||2 / sqrt(8) -- same formula used throughout."""
    x = np.array(weight_vector, dtype=float)
    return np.linalg.norm(x) / np.sqrt(8)


def quantize_and_measure(vectors, codebook, scale_per_vector=True):
    """
    For each input vector:
      1. Compute scale factor (per-vector, matching our standard approach)
      2. Scale vector into lattice's natural range
      3. Find nearest codeword
      4. Reconstruct (scale back up)
      5. Measure squared error

    Returns: total squared error, total squared signal, list of errors
    """
    total_sq_err = 0.0
    total_sq_sig = 0.0
    errors = []

    for vec in vectors:
        S = compute_scale_factor(vec) if scale_per_vector else 1.0
        if S < 1e-8:
            S = 1e-8
        scaled = vec / S

        diff = codebook - scaled
        sq_dist = np.sum(diff ** 2, axis=1)
        best_idx = np.argmin(sq_dist)
        reconstructed = S * codebook[best_idx]

        err = float(np.sum((vec - reconstructed) ** 2))
        sig = float(np.sum(vec ** 2))

        total_sq_err += err
        total_sq_sig += sig
        errors.append(err)

    return total_sq_err, total_sq_sig, errors


def snr_db(sig_power, err_power):
    return 10 * np.log10(sig_power / (err_power + 1e-12))


# ─────────────────────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("E8 vs Z8 Distortion Comparison at Same Code Book sizes ")
    print("=" * 70)

    print("\nGenerating E8 lattice points...")
    e8_points = generate_all_e8_points()
    print(f"  Total valid E8 points found: {len(e8_points)}")
    NUM_TEST_VECTORS = 10000
    print(f"\nGenerating {NUM_TEST_VECTORS} random 8D test vectors "
          f"(Gaussian, mean=0, std=1)...")
    test_vectors = np.random.randn(NUM_TEST_VECTORS, 8)

    print(f"\n{'Bits':<6} {'Codewords':<12} {'E8 SNR (dB)':<14} "
          f"{'Z8 SNR (dB)':<14} {'E8 Advantage':<14}")
    print("-" * 62)

    results = []
    for bits in [2, 3, 4, 6, 8]:
        e8_codebook = generate_e8_codebook(bits, e8_points)
        z8_codebook = generate_z8_codebook(bits)

        e8_err, e8_sig, _ = quantize_and_measure(test_vectors, e8_codebook)
        z8_err, z8_sig, _ = quantize_and_measure(test_vectors, z8_codebook)

        e8_snr = snr_db(e8_sig, e8_err)
        z8_snr = snr_db(z8_sig, z8_err)
        advantage_db = e8_snr - z8_snr

        print(f"{bits:<6} {2**bits:<12} {e8_snr:<14.3f} "
              f"{z8_snr:<14.3f} {advantage_db:+.3f} dB")

        results.append({
            "bits": bits, "e8_snr": e8_snr, "z8_snr": z8_snr,
            "advantage_db": advantage_db,
            "e8_codewords": len(e8_codebook), "z8_codewords": len(z8_codebook)
        })
    
    return results


if __name__ == "__main__":
    main()
