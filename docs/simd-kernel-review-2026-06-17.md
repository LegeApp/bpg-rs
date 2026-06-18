# SIMD kernel review (no code changes) — 2026-06-17

Review of the advice's remaining SIMD targets against the current
`crates/still265/src/primitives/` state. Verdict per item: what exists, and the
remaining opportunity.

## 4×4 / 8×8 SATD — DONE (tiled), minor headroom

- 8-bit: `simd_x86::satd_u8_sse2` — SSE2, processes the block as 4×4 Hadamard
  tiles (`step_by(4)`), so 8×8/16×16/32×32 are already covered by tiling.
- 10/12-bit: `wide_simd::satd_u16` — same 4×4-tiled Hadamard in portable `wide`.
- Both byte-identical to `hadamard4_satd` (tests + end-to-end).

Remaining opportunity (optimization, not a gap): an **AVX2 8-bit path** doing two
4×4 tiles per iteration, and an **8×8-native Hadamard** (one 8-pt butterfly +
fewer transposes than 4× 4×4 tiles). Expected modest; SATD is no longer the
dominant cost on Best.

## Angular intra / all-angles — PARTIAL (clearest remaining win)

`wide_simd::intra_pred_allangs` vectorizes only the **vertical** modes
(`mode >= 18`) via `predict_vertical_simd`; the **horizontal** modes (2..17)
fall back to scalar `predict_angular`. The two halves are transposes of each
other, so a `predict_horizontal_simd` mirror would roughly **double** the
all-angles vectorized coverage. This is the single best remaining SIMD
opportunity in the list (the rough-mode luma search prices all 33 angles).

## residual / subtract & SSD — DONE

- `wide_simd::sub_residual` (↔ x265 `pixel_sub_ps`) and `wide_simd::ssd_u16`
  (↔ x265 `pixel_ssd`) are both dispatched, byte-identical, 8-wide row loops +
  scalar tail, all block sizes.
- Minor: an 8-bit-specific SSD with i16 lane accumulation could shave a little,
  but the i32-lane `ssd_u16` already handles 8-bit correctly. Low priority.

## CABAC residual estimation — already call-reduced/cached (not a SIMD target)

`estimate_residual_bits` is intentionally **not** vectorizable and is already
optimized exactly as the advice asks ("call reduction/caching, not just
vectorization"):

- Table-driven cost model (`cabac::ENTROPY_BITS` = x265 `g_entropyBits`); no
  arithmetic-coder simulation.
- Each `CodedBlock` memoizes its `frac_bits`, so the transform-tree estimator
  re-prices only split/cbf flags, never the residual.
- `ResidualEstimateCache` does **incremental** sub-block-boundary re-pricing for
  RDOQ candidate evaluation — only the changed sub-block is re-walked,
  bit-identical to a full estimate.

The remaining lever here is *fewer / closer candidates* (search shape), not
SIMD. PartNxN added per-PU search calls, so this is where profiling should focus
next — e.g. pruning PartNxN trials by 8×8 SATD quadrant disagreement (advice
step 4), not vectorizing the estimator.

## Summary

| target | status | next step |
|---|---|---|
| 4×4/8×8 SATD | done (tiled) | optional AVX2 / 8×8-native |
| angular all-angles | **partial** (vertical only) | **add horizontal SIMD** |
| residual/sub + SSD | done | — |
| CABAC residual est. | cached/call-reduced | reduce candidates, not SIMD |
</content>
