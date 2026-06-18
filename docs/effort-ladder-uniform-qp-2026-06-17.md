# Effort ladder reshape: uniform-QP "pruned-Best" tiers (2026-06-17)

Goal: make every preset an *increasingly faster, less-exhaustive version of
`Best`* rather than a different encoder. Three changes:

## 1. Uniform QP across the ladder (AQ off by default)

The speed tiers (Fastest/Fast/Balanced/Good) previously ran per-CU variance
**adaptive quantization** while `Best` was uniform-QP — so they diverged from
`Best` in kind, and a corpus sweep showed AQ *hurt*: MS-SSIM ≈ 3 dB worse than
`Best` and a **scrambled quality ladder** (Fastest scored higher than
Fast/Balanced at equal QP). This matches the project's prior finding that the
preanalysis AQ signal is RD-neutral-to-negative on photos.

`aq_active` now returns `false` for the speed tiers unless the new
`StillHevcConfig::adaptive_qp` opt-in is set (`Best` keeps its `BPG_BEST_AQ`
experiment; reference tiers and monochrome stay uniform). AQ became a config
field (not env) so it is race-free under the parallel test harness.

## 2. PartNxN on all quality tiers (Fast..Best)

8x8 `PartNxN` now works on the plan/final-code path (`final_code_cu_nxn` +
`CuLeafPlan::nxn`), not just the winner-direct `best2_cu_reuse` path. It is
enabled for Fast/Balanced/Good/Best (each prunes the per-PU search via its own
RD budget); off for Fastest (speed floor) and the exact reference tiers.

## 3. Whole ladder runs the parallel wavefront

With AQ off, the frozen-slice-init + WPP wavefront (previously `Best`-only) is
safe for every non-reference tier, so all of them now build in parallel and
scale with cores. `Placebo` keeps its static frozen (no-WPP) reference template;
`Reference` stays serial. `effort_template` became an owned (Copy) value so the
parallel flag can be set per-encode.

## Result (1024x768 photo, QP 28, 4:2:0, SAO on)

| tier | size (equal QP) | PSNR-Y | MS-SSIM | warm time |
|---|---:|---:|---:|---:|
| fastest  | 90374 | 38.81 | 0.99246 | 0.30s |
| fast     | 81366 | 39.08 | 0.99283 | 0.48s |
| balanced | 80223 | 39.14 | 0.99293 | 0.50s |
| good     | 78535 | 39.18 | 0.99299 | 0.81s |
| best     | 74496 | 39.11 | 0.99289 | 0.94s |

Monotonic on every axis: size shrinks and time grows with effort; quality climbs
(Best trades a hair of equal-QP PSNR for size via its fast paths, but wins on the
rate-quality curve). Per-image BD-rate vs x265 m9 is also monotonic
(fastest > fast > balanced > good > best). All tiers bit-exact
(`all_effort_tiers_round_trip`, textured 4:2:0; stock `bpgdec` agrees).
</content>
