# 8x8 PartNxN gap — confirmed diagnosis (2026-06-17)

## Goal context

Closing the still265-Best-vs-x265-m9 quality gap. Per-user scope: **Best vs
x265 m9 only**, **SAO off**, equal-size comparison (Reference/Placebo ignored).

Baseline (`docs/quality-baseline-2026-06-17.md`): Best SAO-off vs x265 m9 is
**-0.676 dB / +2.49% size** at equal size (worse on both axes).

## Smoking gun

The outside-agent advice (`docs/quality-discrepancy-with-x265-advice.md`) names
missing **8x8 `PartNxN` intra** as the highest-value gap. Confirmed by decoding
real x265 m9 streams with the Rust decoder (added env-gated counters
`BPG_PARTNXN_STATS=1` in `crates/bpg-hevc-decode/src/hevc/ctu.rs`):

| image (QP28, 4:2:0) | x265 8x8 CUs | x265 PartNxN | share |
|---|---:|---:|---:|
| 20240502_151356 | 5128 | 2189 | 42.7% |
| 20240501_110934 | ~9100 | 5181 | 56.8% |
| 20240502_184356 | ~3070 | 1228 | 40.0% |
| 20240503_105655 | ~7840 | 4031 | 51.4% |

still265 Best uses PartNxN on **0%** of its 8x8 CUs (the encoder never emits it;
`write.rs` always writes part_mode=2Nx2N at log2_cb_size==3). The decoder already
fully supports decoding PartNxN.

x265 also tends to have *more* 8x8 CUs (it splits down to 8x8 and then predicts
four independent 4x4 luma PUs). This is consistent with "still265 spends residual
bits to repair worse prediction on small blocks."

## Decision

Implement 8x8 PartNxN in the still265 encoder for the **Best** tier. This is the
single highest-value missing coding tool for the photo corpus.

## Implementation (done)

Default-on for **Best, 4:2:0**, interior CUs, on the winner-direct
`best2_cu_reuse` path (no plan/final-recode round-trip needed). Override with
`BPG_PARTNXN=0`. Files:

- `encoder/types.rs`: `NxnInfo { luma_modes:[u8;4], mpms:[[..;3];4] }` + `nxn`
  field on `CuLeaf`.
- `encoder/rdo.rs`: `build_cu_leaf_nxn` (per-PU rough+RD luma mode select,
  reconstruct in z-order, one chroma TU at the CU), `decide_cu_8x8_part`
  (RD-compare Part2Nx2N vs PartNxN), gate in `decide_cu`'s `!can_split` branch,
  `estimate_cu_leaf_nxn_bits`.
- `encoder/write.rs`: `write_cu_nxn` (part_mode=0, four prev-flags, four
  mpm/rem, chroma, forced-split TT) — mirrors the decoder's syntax order.
- 4:2:2/4:4:4 deliberately excluded (per-PU/stacked chroma not wired);
  `cat == 1` gate.

## Result (7-image test-set, 4:2:0, QP 24-36, equal-size vs x265 m9)

| metric (rust - x265, equal-size) | Best baseline | Best + PartNxN |
|---|---:|---:|
| PSNR gap   | -0.676 dB | **-0.470 dB** |
| size       | +2.49 %   | **-1.86 %**   |
| SSIM gap   | -0.0040   | **-0.0013**   |

- PSNR gap closed by **+0.207 dB**; size flipped from +2.49% to **-1.86%**
  (still265 Best is now *smaller* than x265 at the matched point).
- Equal-QP rust avg: 34.439 -> 34.645 dB (+0.21 dB), 94924 -> 90991 B (-4.1%),
  encode 0.80 -> 1.77s (~2.2x; speed deferred per session scope).
- PartNxN fires on ~40-57% of 8x8 CUs, matching x265.

## Validation

- 25/25 `encode_roundtrip` tests pass with PartNxN default-on (encoder recon ==
  in-tree decoder, bit-exact), incl. new `best_partnxn_round_trip`.
- Stock C `bpgdec.exe` decodes PartNxN streams to the same PSNR (valid stream).
- Parallel == serial output, byte-identical (`BPG_ENC_THREADS=1` vs default).
- `BPG_PARTNXN=0` reproduces the exact pre-change baseline bytes.
</content>
