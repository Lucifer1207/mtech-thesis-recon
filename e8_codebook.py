import numpy as np
from itertools import product

def is_e8_point(x, tol=1e-6):
    """
    Check if an 8D vector x is a valid E8 lattice point.
    Condition 1: all integers with even sum
    Condition 2: all half-integers with even sum
    """
    x = np.array(x, dtype=float)
    is_integer = np.all(np.abs(x - np.round(x)) < tol)
    is_half_integer = np.all(
        np.abs(x - np.round(x - 0.5) - 0.5) < tol
    )
    sum_val = np.sum(x)
    is_even_sum = (np.abs(sum_val - np.round(sum_val)) < tol and
                   int(np.round(sum_val)) % 2 == 0)
    return (is_integer or is_half_integer) and is_even_sum


def generate_all_e8_points():
    """
    Generate ALL valid E8 points in our search range.
    Returns list of (norm_sq, point) — unsorted.
    
    We search:
    - Integer coords in [-2, 2]
    - Half-integer coords in [-1.5, 1.5]
    """
    candidates = []

    # Integer coordinate candidates
    for coords in product([-2,-1,0,1,2], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            norm_sq = float(np.dot(x, x))
            candidates.append((norm_sq, x))

    # Half-integer coordinate candidates
    for coords in product([-1.5,-0.5,0.5,1.5], repeat=8):
        x = np.array(coords, dtype=float)
        if is_e8_point(x):
            norm_sq = float(np.dot(x, x))
            candidates.append((norm_sq, x))

    return candidates


def generate_e8_codebook(bits, all_points=None):
    """
    Generate E8 codebook for given bit budget.
    
    KEY FIX: Sort by norm only — no lexicographic bias.
    Within same norm, order doesn't matter for correctness
    because ALL shell-1 points should be included before
    any shell-2 points.
    
    This ensures symmetric coverage — positive and negative
    coordinates both represented fairly.
    """
    num_codewords = 2 ** bits
    print(f"\nGenerating {bits}-bit E8 codebook ({num_codewords} codewords)...")

    # Generate points if not provided
    if all_points is None:
        all_points = generate_all_e8_points()

    # Sort by norm ONLY — no lexicographic bias
    # np.random with fixed seed for reproducibility within same norm
    rng = np.random.default_rng(seed=42)
    
    # Group by norm, shuffle within each norm group
    from collections import defaultdict
    norm_groups = defaultdict(list)
    for norm_sq, x in all_points:
        key = round(norm_sq, 4)
        norm_groups[key].append(x)
    
    # Sort norm groups and shuffle within each group
    sorted_norms = sorted(norm_groups.keys())
    ordered_points = []
    for norm in sorted_norms:
        group = norm_groups[norm]
        # Shuffle within same-norm group for unbiased selection
        indices = rng.permutation(len(group))
        for i in indices:
            ordered_points.append((norm, group[i]))

    # Take first num_codewords
    codebook = np.array([p[1] for p in ordered_points[:num_codewords]])

    # Count shell-1 points
    shell1 = sum(1 for p in ordered_points[:num_codewords]
                 if abs(p[0] - 2.0) < 1e-4)
    print(f"  Shell-1 points in codebook: {shell1} / {num_codewords}")

    return codebook


def compute_scale_factor(weight_vector):
    """
    S = ||x||₂ / √8
    """
    x = np.array(weight_vector, dtype=float)
    return np.linalg.norm(x) / np.sqrt(8)


def find_nearest_codeword(weight_vector, codebook, S):
    """
    1. Scale: x' = x / S
    2. Find: argmin ||x' - c||²
    3. Reconstruct: x̂ = S × c*
    4. Error: ||x - x̂||²
    """
    x = np.array(weight_vector, dtype=float)
    scaled = x / S

    diff = codebook - scaled           # (num_codewords, 8)
    sq_dist = np.sum(diff**2, axis=1)  # (num_codewords,)

    best_idx = int(np.argmin(sq_dist))
    best_codeword = codebook[best_idx]
    reconstructed = S * best_codeword
    error = float(np.sum((x - reconstructed)**2))

    return best_idx, reconstructed, error


def main():

    # Generate all E8 points once — reuse across demos
    print("Pre-generating all E8 points (this runs once)...")
    all_points = generate_all_e8_points()
    print(f"Total valid E8 points found: {len(all_points)}")

    # ─────────────────────────────────────
    # DEMO 1: Codebook sizes
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 1: Codebook sizes at different bit levels")
    print("=" * 60)

    for bits in [2, 3, 4, 8]:
        cb = generate_e8_codebook(bits, all_points)
        print(f"  {bits}-bit → {len(cb)} codewords\n")

    # ─────────────────────────────────────
    # DEMO 2: Quantize sample weight block
    # ─────────────────────────────────────
    print("=" * 60)
    print("DEMO 2: Quantizing a sample weight block")
    print("=" * 60)

    sample_weights = np.array([0.92, -0.45,  1.13, -0.87,
                                0.34, -1.20,  0.67, -0.55])

    S = compute_scale_factor(sample_weights)
    print(f"\nOriginal weights : {sample_weights}")
    print(f"Scale factor S   : {S:.4f}")
    print(f"Scaled weights   : {np.round(sample_weights/S, 4)}")
    print()

    for bits in [2, 3, 4, 8]:
        cb = generate_e8_codebook(bits, all_points)
        idx, recon, err = find_nearest_codeword(sample_weights, cb, S)

        print(f"  {bits}-bit quantization:")
        print(f"    Nearest codeword  : {cb[idx]}")
        print(f"    Reconstructed     : {np.round(recon, 4)}")
        print(f"    Quantization err  : {err:.6f}")
        print()

    # ─────────────────────────────────────
    # DEMO 3: Subset relationship
    # ─────────────────────────────────────
    print("=" * 60)
    print("DEMO 3: Is 2-bit codebook subset of 4-bit?")
    print("=" * 60)

    cb_2 = generate_e8_codebook(2, all_points)
    cb_4 = generate_e8_codebook(4, all_points)

    all_found = True
    for point in cb_2:
        found = any(np.allclose(point, p) for p in cb_4)
        if not found:
            all_found = False
            break

    if all_found:
        print("\n✅ 2-bit IS a subset of 4-bit — confirmed")
    else:
        print("\n❌ Not a subset")

    # ─────────────────────────────────────
    # DEMO 4: Accuracy vs compression
    # ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 4: Accuracy vs compression tradeoff")
    print("=" * 60)

    print(f"\n{'Bits':<6} {'Codewords':<12} {'Error':<14} {'Compression'}")
    print("-" * 48)

    original_bits = 8 * 32  # 8 weights × 32 bits

    for bits in [2, 3, 4, 8]:
        cb = generate_e8_codebook(bits, all_points)
        _, _, err = find_nearest_codeword(sample_weights, cb, S)
        compressed_bits = bits + 16  # index + float16 scale
        ratio = original_bits / compressed_bits
        print(f"{bits:<6} {2**bits:<12} {err:<14.6f} {ratio:.1f}×")


if __name__ == "__main__":
    main()