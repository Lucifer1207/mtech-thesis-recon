# Codeword Usage Histograms — What They Show

## Which file is which

| Filename | Condition | Model + Lattice |
|---|---|---|
| `codeword_hist_qat_bert_base_e8_v4_epochs20.png` | Without normalization | BERT-Base + E8 |
| `codeword_hist_qat_bert_base_z8_v4_epochs20.png` | Without normalization | BERT-Base + Z8 |
| `codeword_hist_qat_tinybert_e8_v4_epochs20.png` | Without normalization | TinyBERT + E8 |
| `codeword_hist_qat_tinybert_z8_v4_epochs20.png` | Without normalization | TinyBERT + Z8 |
| `codeword_hist_qat_bert_base_e8_v5_epochs20_cb5.png` | Normalized, our Y_max=max() | BERT-Base + E8 |
| `codeword_hist_qat_bert_base_z8_v5_epochs20_cb5.png` | Normalized, our Y_max=max() | BERT-Base + Z8 |
| `codeword_hist_qat_tinybert_e8_v5_epochs20_cb5.png` | Normalized, our Y_max=max() | TinyBERT + E8 |
| `codeword_hist_qat_tinybert_z8_v5_epochs20_cb5.png` | Normalized, our Y_max=max() | TinyBERT + Z8 |
| `codeword_hist_qat_bert_base_e8_v5lit_epochs20_cb5_d1.5_q8_m1.png` | Normalized, literal Y_max (paper formula) | BERT-Base + E8 |
| `codeword_hist_qat_bert_base_z8_v5lit_epochs20_cb5_d1.5_q8_m1.png` | Normalized, literal Y_max (paper formula) | BERT-Base + Z8 |
| `codeword_hist_qat_tinybert_e8_v5lit_epochs20_cb5_d1.5_q8_m1.png` | Normalized, literal Y_max (paper formula) | TinyBERT + E8 |
| `codeword_hist_qat_tinybert_z8_v5lit_epochs20_cb5_d1.5_q8_m1.png` | Normalized, literal Y_max (paper formula) | TinyBERT + Z8 |

The pattern described below for each condition is consistent across
all 4 model/lattice combinations — the shape doesn't change model to
model, only the exact numbers do (noted where relevant).

## What these graphs are

For each of the 4 model/quantization combinations (BERT-Base+E8,
BERT-Base+Z8, TinyBERT+E8, TinyBERT+Z8), under 3 different training
conditions, we counted — after training, across every quantized
weight in the model — how many times each of the 256 fixed codebook
entries was the *nearest* match. That gives one bar chart per
combination per condition: x-axis is the codeword index (0–255),
y-axis is how many 8-D weight sub-vectors landed on that codeword.

The 3 conditions:
1. **Without normalization** — the original scheme; each 32-weight
   block gets its own simple min/max scale, no row-level awareness.
2. **With normalization (our fix)** — row-wise scaling, using
   Y_max = codebook's own max coordinate (an empirical choice we
   made because the codebook is small and fixed).
3. **With normalization (paper's literal formula)** — same row-wise
   scaling, but Y_max computed exactly as the paper specifies:
   Δ0·(qᴹ−1)/2, with q=8, M=1, Δ0=1.5 (paper-matching values).

## What each condition's shape tells us

**Without normalization:** usage is close to *uniform* across all
256 codewords — no single codeword dominates. This scheme has no
built-in pull toward the origin, so weights spread out fairly evenly
over the whole codebook.

**With normalization (our Y_max):** one codeword — the one nearest
the origin — absorbs the *large majority* of usage. For BERT-Base
this single codeword accounts for roughly 55% of all usage under E8
and roughly 89% under Z8; the rest of the near-origin codewords
share a smaller, still-substantial slice, and usage falls off
sharply for codewords farther from the origin, with the farthest
ones used only a handful of times. This is the expected, "efficient"
picture: weights cluster near zero, and the codebook's near-origin
points do almost all the work.

**With normalization (literal Y_max):** the pattern flips —
codeword 0 is *under*-used, and usage actually *grows* toward the
far/boundary codewords instead of shrinking. This is the visual
signature of overload: the literal formula's Y_max (5.25) is larger
than the codebook's own actual range (E8 tops out at 1.5, Z8 at
1.0), so even ordinary weight values get scaled past what the table
can represent, and end up snapping to whichever boundary codeword is
nearest — not just the rare outliers.

## A structural detail worth noting

In both normalized conditions, the point where usage drops off
sharply is not arbitrary — for **E8 it happens right around codeword
index 240**, and for **Z8 right around index 128**, in both
BERT-Base and TinyBERT versions. These are exactly the number of
lattice points in the *first shell* around the origin for each
lattice: E8's nearest-neighbor shell has exactly 240 points (a known
constant, E8's "kissing number"), and Z8's first two shells combined
have 128 points (16 at distance 1, 112 at distance √2). That the
drop-off lines up with this known lattice geometry, consistently
across both models, is a good sign that the quantization is behaving
the way the underlying math predicts, not an artifact of training.

## What question this answers

The original ask was: *how efficiently does the model use the fixed
256-entry codebook, and does that differ between the normalized and
non-normalized training schemes?* The answer: yes, substantially.
Without normalization, the codebook is used broadly but without any
particular structure. With normalization (our empirical Y_max), the
codebook is used *very* efficiently — a small number of near-origin
codewords cover almost all the weight mass, meaning most of the 256
entries contribute little. With the literal-formula Y_max, the
codebook is used inefficiently in the opposite direction — overload
pushes weights toward the boundary rather than the center.

## Conclusion

Our empirically-grounded Y_max (codebook.abs().max()) matches how a
small, fixed codebook actually gets used: concentrated near the
origin, matching the real distribution of trained weight values.
The paper's literal Y_max formula is the mathematically "correct"
one for an *unbounded* quantizer, but on a small fixed table it
overshoots and pushes usage toward the boundary instead — visible
directly in the histogram shape, not just in accuracy numbers. That
said, accuracy under the literal-Y_max condition did **not** collapse
the way the shifted usage pattern might suggest — the network
still reached comparable F1 to the empirical-Y_max condition,
suggesting STE training can partly adapt around a poorly-matched
quantization range even when the codebook is clearly being used
inefficiently. This is worth treating as a genuine, reportable
finding rather than smoothing over: matching the paper's formula
exactly does not automatically mean better results on a small,
fixed codebook — the two aren't the same design, and the histograms
are direct visual evidence of that difference.
