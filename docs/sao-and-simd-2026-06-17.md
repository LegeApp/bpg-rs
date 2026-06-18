# SAO production-grade rework + SIMD kernels (2026-06-17)

Follow-on to `docs/partnxn-investigation-2026-06-17.md`. Implements the
actionable items from `docs/sao.md` and the SIMD targets from
`docs/quality-discrepancy-with-x265-advice.md`.

## SAO: remove the double full-RDO encode (the headline fix)

`docs/sao.md` diagnosed SAO's ~7x slowdown as a two-full-encode pipeline (encode
without SAO → decide map → throw away state → encode *again* with SAO). Fixed by
splitting the existing parallel path into **build** and **write** halves and
replaying:

- `build_slice_trees_parallel` — CTU-wavefront-parallel analysis, reconstructs
  `state.frame`, marks deblock edges, returns the per-CTU `CuNode` trees.
- `write_slice_from_trees` — pure serial CABAC pass over the cached trees,
  optionally prefixing each CTU with `sao()` syntax. It does **not** touch the
  frame (reconstruction already happened in build), so SAO can be decided on the
  deblocked frame *between* build and write.

`encode_with_stats` SAO path (parallel/Best tiers): build → deblock → decide SAO
→ replay-write → apply SAO. No second RDO pass. Serial (non-Best) tiers keep the
old two-pass path (their build is interleaved with the running write context, so
build/write can't be split byte-identically).

**Result:** SAO overhead on Best dropped from **~+700% to ~0%** (1.87s → 1.79s,
within noise on a 1024x768 photo at QP28). Quality +0.27 dB / +0.9% size at QP28.
Bit-exact: new `best_sao_replay_round_trip` test; stock `bpgdec` accepts streams.

Deferred (smaller `docs/sao.md` items): lambda*bits charging in the SAO RD
decision (would trim the +0.9%), parallelizing the SAO map decision, BO SIMD,
specialized E1/E2/E3 loops, `apply_sao` line buffers.

## SIMD kernels (portable `wide`, byte-identical, dispatched via `PRIMITIVES`)

Both new kernels follow the project invariant: byte-identical to a scalar
reference (enforced by tests + end-to-end md5), selectable via `BPG_PRIMITIVES`.

- **`dequantize`** (RDO): inverse quantization in place (x265 `dequant_normal`),
  run on every TU reconstruction. `i32x8` multiply/add/arith-shift/clamp.
  `transform.rs::dequantize` delegates to it. Test: `dequantize_matches_scalar`
  over all qp/log2/bit_depth + tails.
- **`sao_stats_e0`** (SAO): horizontal edge-offset statistics (x265
  `saoCuStatsE0`). Signs via inherent `is_negative`/`is_positive` masks; per-
  category accumulation via masked `blend`; per-row `i32x8` accumulators reduced
  once. Used by `sao_eo_stats` for interior CTBs (scalar fallback at
  plane/source edges). Test: `sao_stats_e0_matches_scalar` + md5 equality.

`wide-simd` is default-on for `bpg-tools`, so both are active in release builds.

## Combined quality state (7-image test-set, 4:2:0, QP 24-36, equal-size vs x265 m9)

| config | PSNR gap | size | SSIM gap |
|---|---:|---:|---:|
| Best baseline (pre-session) | -0.676 dB | +2.49% | -0.0040 |
| Best + PartNxN | -0.470 dB | -1.86% | -0.0013 |
| **Best + PartNxN + SAO** | **-0.215 dB** | **-0.36%** | **+0.0006** |

With PartNxN (default-on) + SAO (`--sao`), still265 Best is now within ~0.2 dB of
x265 m9 PSNR at equal size, *smaller* in bytes, and slightly *better* on SSIM —
about two-thirds of the original ~1 dB gap closed.

## SAO is now default-on for Best

The CLI defaults SAO **on for `--effort best`** (where the replay makes it
~free) and off for the speed tiers (which still use the slower two-pass SAO).
`--sao` / `--no-sao` force the choice. Quality gap is considered meaningfully
closed for now: SSIM is at parity (rust slightly ahead on the 7-image mean),
size is ~equal, and the residual PSNR gap (~0.2 dB) is not necessarily visually
meaningful — see the expanded metric panel below.

## Heads-up tester: perceptual metric panel

`scripts/metrics.py` + `scripts/headsup_quality.py` now report a layered panel
beyond PSNR/SSIM-RGB: PSNR-Y, MS-SSIM (+dB), VIFp, PSNR-HVS-M (DCT-domain CSF +
masking), VMAF and XPSNR (via ffmpeg), with optional Butteraugli (external bin)
and piq learned metrics (LPIPS/DISTS/HaarPSI) when available. New outputs:
`metric_aggregates.csv` (mean/median/worst-image per metric) and
`bdrate_by_image.csv` (Bjontegaard BD-rate, rust vs x265, per metric). The panel
is self-contained (numpy/scipy/PIL); `--no-extra-metrics` / `--no-vmaf` skip it.
</content>
