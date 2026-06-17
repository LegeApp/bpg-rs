I inspected the Rust port against x265 4.1. My read is: the quality gap is probably **not primarily an RDOQ-rate-model problem anymore**. The more important remaining issue is that still265 is spending residual bits to repair worse prediction decisions, especially at the smallest intra blocks.

## Main diagnosis

The most concrete missing x265 parity item I found is **8×8 intra `PartNxN` support**.

In x265, intra analysis explicitly tests `SIZE_NxN` at 8×8 minimum CUs:

```cpp
if (cuGeom.log2CUSize == 3 && cuData.m_slice->m_sps->quadtreeTULog2MinSize < 3)
    md.pred[PRED_INTRA_NxN].cu.initSubCU(..., SIZE_NxN);
```

That path allows one 8×8 CU to be predicted as **four 4×4 luma prediction units**, each with its own luma intra mode. This is valuable for edges, fine texture, text-like detail, and small directional changes.

In still265, the decoder supports `PartNxN`, but the encoder appears never to emit it. In `crates/still265/src/encoder/write.rs`, the min-CU intra partition mode is effectively always written as `2Nx2N`:

```rust
if log2_cb_size == 3 {
    cabac.encode_bin(&mut writer.writer, 1, CabacCtx::PartMode)?;
}
```

Your decoder confirms the meaning:

```rust
if bin != 0 {
    PartMode::Part2Nx2N
} else {
    PartMode::PartNxN
}
```

So still265 can split the **transform tree** into 4×4 TUs, but it still uses one luma prediction mode for the whole 8×8 CU. x265 can use four separate 4×4 prediction modes. That is a real coding-efficiency difference, not just a speed/primitive difference.

That likely explains the “encoding more data more slowly” symptom: the extra data is probably going into **luma residual coefficients** for small blocks whose prediction should have been cheaper if represented as `PartNxN`. You are spending CABAC residual bits to correct a prediction model x265 avoids.

## Other contributors

There are three other meaningful contributors, but I would rank them below `PartNxN` as explanations for the unexplained 1 dB.

### 1. SAO mismatch

`remaining-gaps.md` already estimates about **0.3 dB** from missing SAO in the comparison. That sounds plausible. x265 placebo/default uses SAO unless disabled. still265 has SAO support, but the backend default is off.

So a direct comparison of:

```text
x265 -m9 with SAO
vs
still265 Best without SAO
```

is not tool-equivalent.

SAO does not explain the whole gap, but it is enough to make the remaining problem look larger and noisier.

### 2. x265 placebo is not PSNR-pure by default

x265 placebo enables aggressive RD behavior and normally has psy-RD active unless using `--tune psnr`. In `source/common/param.cpp`, x265 defaults include SAO and psy-RD, while placebo enables `rdLevel=6`, `rdoqLevel=2`, `transformSkip`, and `psyRdoq=1.0`.

For metric diagnosis, the fairer x265 comparator is closer to:

```text
-m 9
--tune psnr
--no-sao        # for one ablation
--no-transform-skip
```

Then add tools back one at a time.

That said, because x265 is still ahead in PSNR, psy is probably not the whole story. But you need the ablation to avoid chasing the wrong missing feature.

### 3. still265 Best is not the same as still265 Reference

Your own effort templates matter. `Best` is not just “Reference but faster.” It has:

```rust
chroma_during_luma_trials: false
select_rdoq_single_scan: true
parallel_analysis: true
```

while `Reference`/`Placebo` use more exact paths.

So before changing algorithmic code, run the quality sweep with the most exact tier, not just `Best`. If `Reference` materially narrows the gap, then part of the loss is in the practical Best shortcuts. If it does not, the missing coding tools are the dominant issue.

One small code smell: `best2_chroma_gate` currently looks hard to disable by env var because the parsed value is filtered to `> 0.0` and then falls back to the default margin. For ablation, patch it to `None`; do not assume `BPG_BEST2_CHROMA_GATE=0` disables it.

## What is probably not the main cause

The docs already make this fairly clear: **RDOQ parity did not buy much**.

You implemented x265-like lambda2, sign-data hiding, and RDOQ level-2-style rate modeling. Sign hiding helped. RDOQ level modeling was almost neutral on photos. That suggests the gap is upstream of coefficient refinement: prediction/mode/partition/TU decisions are producing a worse residual before RDOQ gets a chance to polish it.

So I would not put more effort into RDOQ first unless your new traces specifically show coefficient-level decisions diverging after prediction parity.

## Where to look for the extra bits

Instrument final encoded CUs/TUs and bucket:

```text
luma residual bits by CU size: 64, 32, 16, 8
luma residual bits by TU size: 32, 16, 8, 4
chroma residual bits separately
mode syntax bits separately
CBF / split / transform-tree syntax separately
distortion by CU/TU size
```

The hypothesis predicts:

```text
still265 has too many residual bits and/or too much distortion in 8×8 CUs and 4×4 TUs.
```

Then decode x265’s bitstream with your Rust decoder and count actual `PartNxN` use. Since your decoder already supports `PartNxN`, this should be straightforward. Add counters where `decode_part_mode()` returns `PartNxN`.

If x265 uses `PartNxN` frequently on blocks where still265 has high residual cost, that is your smoking gun.

## Closure plan, in order

### 1. Run a strict ablation matrix

Do this before implementing more code.

Compare:

```text
A. x265 placebo default
B. x265 placebo --no-sao
C. x265 placebo --tune psnr
D. x265 placebo --tune psnr --no-sao
E. x265 placebo --tune psnr --no-sao --no-transform-skip
```

Against:

```text
F. still265 Best, SAO off
G. still265 Best, SAO on
H. still265 Reference, SAO off
I. still265 Reference, SAO on
```

The key question is whether the remaining gap after `D`/`E` versus `H`/`I` is still around 0.7–1.0 dB. If yes, missing intra partition/mode behavior is the likely center.

### 2. Implement 8×8 `PartNxN`

This is the highest-value missing feature I saw.

Implementation shape:

```text
At log2_cb_size == 3:
    evaluate normal Part2Nx2N
    evaluate PartNxN
    choose by full RD cost
```

For `PartNxN`:

```text
- One 8×8 CU.
- Four 4×4 luma PUs.
- Four luma intra modes.
- For 4:2:0 / 4:2:2, one chroma mode for the CU.
- Force intra_split_flag / transform-tree split behavior as required.
- Reconstruct each 4×4 PU before analyzing the next one, because neighboring reconstructed samples affect later prediction.
- Code all mode syntax and residual bits into the RD cost, including the cheaper/more expensive part-mode syntax.
```

Do this first only in `Reference` or a new experimental mode. Do not optimize it initially. Prove the BD-rate/PSNR effect first.

Expected outcome: if this is the missing piece, still265 should produce fewer luma residual bits in fine-detail 8×8 areas, even if encode time worsens temporarily.

### 3. Make SAO production-grade and default-on for x265 parity

Your current SAO design appears conservative and two-pass. That is acceptable for correctness testing, but for parity with x265 it needs to be part of the normal high-quality pipeline.

This will probably close the documented ~0.3 dB portion and improve equal-size comparisons. It may also reduce residual pressure indirectly at the same target size because post-filtering improves reconstructed quality without spending transform bits.

### 4. Revisit Best shortcuts only after Reference parity

Once Reference is closer, migrate the feature back into Best with pruning:

```text
- Only test PartNxN when 8×8 SATD suggests directional disagreement across quadrants.
- Always include MPMs.
- Include top K angular modes per 4×4 PU.
- Use x265-style candidate count: roughly 8–10 in high RD modes.
- Cache 4×4 predictors and SATDs aggressively.
```

Do not let Best and Reference diverge in kind. Best should be a pruned version of the same search space, not a different encoder.

### 5. SIMD after the quality hypothesis is proven

SIMD can close much of the `.7s → .3s` gap, but it will not fix the 1 dB gap unless the missing SIMD is hiding a logic difference, which seems unlikely.

Highest-value SIMD/primitive targets after `PartNxN`:

```text
- 4×4 DST forward/inverse
- quant/dequant
- inverse transform
- 4×4/8×8 SATD
- angular intra prediction/all-angles path
- residual/subtract and SSD
- CABAC residual estimation call reduction/caching, not just vectorization
```

For current Best, the profiler pointing at residual-bit estimation means you are call-bound on coefficient CABAC simulation. SIMD helps surrounding math, but the larger win is reducing bad candidate evaluations and making the candidates closer to x265’s.

## Practical explanation of the 1 dB

My current explanation is:

```text
~0.3 dB: SAO/tool mismatch.
~0.1–0.3 dB: Best shortcuts, context approximation, chroma/luma trial differences, transform skip/psy comparison noise.
remaining large chunk: missing 8×8 PartNxN intra prediction and related x265 intra search behavior.
```

The important inversion is this:

```text
still265 is not worse because it writes too little data.
It is worse because it spends data on residuals that x265 avoids through better local prediction/partition choices.
```

So the next serious quality milestone should not be “more exact RDOQ.” It should be:

```text
decode x265 output → count PartNxN → instrument still265 residual bits by 8×8/4×4 → implement PartNxN → rerun equal-size BD sweep.
```

That will tell you whether the remaining 1 dB is a missing coding-tool gap or a collection of smaller mature-x265 heuristics.
