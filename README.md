# bpg-rs

A pure-Rust implementation of the [BPG image format](https://bellard.org/bpg/)
— Fabrice Bellard's HEVC-intra-based still-image container. It provides an
encoder (`still265`, a from-scratch port of x265's still-picture intra path), a
decoder (`bpg-hevc-decode`), and a `bpg-tools` CLI that ties them together.

The entire workspace is pure Rust: no C/C++ is built or linked, and there is no
`cmake`/libde265/libx265 dependency. It reads PNG/JPEG, writes BPG, and decodes
BPG/HEIC back to JPEG or PNG.

## What it's for

BPG stores a single still image as an all-intra HEVC frame in a compact
container — roughly JPEG-class file sizes at noticeably higher quality, or much
smaller files at equal quality. The reference implementation (`libbpg` +
`bpgenc -e x265`) is C and depends on a full x265 build. `bpg-rs` is a
dependency-free Rust alternative that is byte-compatible with the format: files
it writes decode in stock `bpgdec`, and it decodes third-party BPG/HEIC files.

### How it compares to x265 / C `bpgenc`

`still265`'s default `slow` tier is now competitive with C `bpgenc -e x265` in
both speed and quality. Across the rate range the remaining quality deficit is
about **4% BD-rate** — i.e. it needs ~4% more bits than x265 for the same
objective quality. Timing is within ~2–2.5× of x265 on a multicore machine.

Measured 2026-06-30, 20-core machine, 4:2:0, 8-bit, all SIMD + threads on, one
native photo per resolution, `bpg-highres-compare --native`, built with the full
LTO `release-lto` profile. Times are averages of 10 Rust runs per effort; the C
baseline is averaged over the 30 paired C `bpgenc` runs produced by the harness.
**Comparison is at equal *actual* HEVC QP** (`still265 -q28` vs `bpgenc -q31`;
see the QP-offset note below), so the file-size deltas are a true
compression-efficiency comparison. PSNR is the decoded reconstruction vs.
source. Note these rows are matched on QP, not on rate — `slow` runs slightly
*higher* PSNR than C here, so the size deltas overstate the ~4% BD-rate figure.

**12 MP (4000×3000)** — C `bpgenc -m9`: 1.43 s, 606,046 B, 40.39 dB Y / 37.11 dB RGB

| effort  | encode s | vs C   | bytes   | vs C size | psnr Y | psnr RGB |
|---------|---------:|-------:|--------:|----------:|-------:|---------:|
| fast    |     1.36 |  0.95× | 698,669 |   +15.3%  | 40.17  |  37.09   |
| slow    |     2.40 |  1.7×  | 640,292 |    +5.7%  | 40.48  |  37.21   |
| placebo |     9.01 |  6.3×  | 643,587 |    +6.2%  | 40.41  |  37.21   |

**50 MP (8160×6120)** — C `bpgenc -m9`: 3.03 s, 405,500 B, 46.26 dB Y / 43.88 dB RGB

| effort  | encode s | vs C   | bytes   | vs C size | psnr Y | psnr RGB |
|---------|---------:|-------:|--------:|----------:|-------:|---------:|
| fast    |     5.84 |  1.9×  | 532,946 |   +31.4%  | 46.16  |  44.08   |
| slow    |     7.11 |  2.3×  | 456,733 |   +12.6%  | 46.39  |  44.11   |
| placebo |    23.58 |  7.8×  | 459,911 |   +13.4%  | 46.37  |  44.17   |

At these matched quantizers `slow` already edges x265 on PSNR while spending
6–13% more bytes; `placebo` (exhaustive search) does not beat `slow` on these
photographic images, so `slow` is the recommended quality/speed point. Per-MP
encode time *drops* as resolution grows because the tiled encoder spreads larger
pictures across cores.

> **QP-mapping offset.** `bpgenc -qN` encodes at actual HEVC QP `N − 3` (the
> offset lives inside x265's `tune=ssim` preset), whereas `still265 -qN` uses HEVC
> QP `N` directly. This is verified bit-exact (bpgenc `-q31` and still265 `-q28`
> produce identical coefficient levels). Always compare at equal *actual* QP or by
> BD-rate, never at equal nominal `-q`.

## Crates

| Crate | Purpose |
|---|---|
| `bpg-bitstream` | `ue7` base-128 varints (container fields) and an MSB-first bit reader/writer with Exp-Golomb `ue(v)` (HEVC RBSP). |
| `bpg-image` | `Image`/`Plane<u16>`, color spaces, RGB↔YCbCr (BT.601/709/2020, full/limited range, 8/10/12-bit), 4:4:4→4:2:0/4:2:2 subsampling, CTU padding. |
| `bpg-format` | The fixed BPG container header (read/write). |
| `bpg-hevc` | Annex-B NAL parsing/emulation-prevention and the VPS/SPS rewriting for BPG's "modified HEVC" stream. |
| `bpg-hevc-decode` | Pure-Rust HEVC still-picture intra decoder (CABAC, intra prediction, transform/dequant, deblocking, SAO). |
| `bpg-decode` | BPG/HEIC container → Annex-B → decode → RGB(A)/BGR(A). |
| `bpg-encode` | `HevcEncoder` trait + still-image encode orchestration (pad → encode → rewrite headers → container). |
| `still265` | The Rust-native still HEVC intra encoder (the bulk of this project — see below). |
| `toojpeg` | Vendored minimal JPEG writer, used for BPG→JPEG decode output. |
| `bpg-tools` | `clap`-based CLI exposing `encode` and `decode`. |

## Building and testing

```bash
cargo build --release
cargo test --release
```

Release builds use thin LTO by default. For maximum standalone binary
optimization, use `cargo build --profile release-lto`. Cargo profiles are chosen
by the root application, so downstream callers should enable LTO in their own
release profile; OpenArc already does this for its `release` profile. Pure Rust,
no `cmake`/C toolchain. Edition 2021, `cargo`/`rustc` 1.94+.

## CLI

Encode a PNG or JPEG to BPG:

```bash
cargo run -p bpg-tools --release -- encode input.png -o out.bpg \
  --qp 28 --format 420 --effort slow
```

- `--qp` (0–51): the CQP quantizer (1:1 with HEVC QP).
- `--effort` (`fast` / `slow` / `placebo`, default `slow`): RD-search budget.
  - `fast` — aggressive, x265-style early-outs (zero/low-residual CU and TU
    early termination, narrowed rough-mode sweeps). Smallest encode time, ~10–30%
    larger files.
  - `slow` — the default archival tier: progressive rough-mode decision,
    full-RD luma/chroma, TU/CU-split RD, NxN PU, multi-pass RDOQ. The recommended
    quality/speed point.
  - `placebo` — exhaustive reference search (all 35 rough modes, exact greedy
    RDOQ, no pruning), CTU-wavefront parallel. Mainly a regression oracle; rarely
    beats `slow` on photos. `BPG_ENC_THREADS` caps the worker count.

  Older ladder names (`fastest`, `balanced`, `best`, `slowplus`, …) are accepted
  as aliases and collapse onto these three.
- `--format` (gray/420/422/444), `--bit-depth` (8/10/12; 10/12 needs 16-bit PNG
  input to be useful).
- `--aq` (default `off`): adaptive quantization — see below.
- `--no-sao` / `--no-deblock`: in-loop filters are **on by default**.
- `--color-space` (ycbcr/rgb/ycgco/bt709/bt2020) and `--limited-range`.
  RGB/YCgCo encode requires `--format 444`.

`encode` prints the output size, effort, and wall-clock time on completion.

Decode a BPG (or HEIC/HEIF) — JPEG output by default, PNG via a `.png` extension:

```bash
cargo run -p bpg-tools --release -- decode in.bpg -o out.jpg          # JPEG
cargo run -p bpg-tools --release -- decode in.bpg -o out.png --format rgba
```

JPEG output preserves the source's native YCbCr (no RGB round-trip) for the
common BT.601 full-range 8-bit case. The decoder supports gray/4:2:0/4:2:2/4:4:4,
8/10-bit, and non-CTU-aligned sizes.

The `BPG_PRIMITIVES` env var (`scalar`/`simd`/`auto`) overrides SIMD primitive
selection for A/B testing.

## Status

- Still-image, intra-only, lossy (CQP) encode and decode. No animation, alpha,
  or lossless mode.
- Color: YCbCr (BT.601/709/2020), RGB/GBR, and YCgCo end-to-end for 4:4:4.
- Chroma: gray, 4:2:0, 4:2:2, 4:4:4 on both encoder and decoder.
- Bit depth: 8/10/12-bit, round-trip tested.
- In-loop filters: deblocking and SAO (band + edge offset) both implemented and
  on by default; the single-pass replay path makes SAO nearly free. The decoder
  implements both fully for third-party files.
- `HevcEncoder::caps()` reports a backend's supported depths/formats/filters, so
  unsupported requests fail up front with a clear error instead of panicking.

## What was ported from x265 — and what wasn't

`still265` is a port of x265's **still / all-intra** path. The goal was a faithful
reproduction of x265's coding tools, not its frame-level machinery.

**Ported:**

- The CABAC arithmetic engine and context models, and x265's table-driven bit
  estimator (`g_entropyBits`) used to price RD candidates — no arithmetic-coder
  simulation in the search loop.
- NAL/RBSP writer, VPS/SPS/PPS, and slice-segment header.
- Intra prediction: all 35 modes, reference-sample substitution and smoothing.
- Forward/inverse transforms (DCT 4–32, 4×4 DST) as butterfly stages, plus
  quant/dequant.
- RDOQ (rate-distortion optimized quantization) and sign data hiding (on by
  default for all tiers).
- Mode decision: SATD rough-mode decision, full-RD luma/chroma search, TU-split
  RD, CU-split RD, and NxN PU search.
- Spatial adaptive quantization (see below).
- Multithreading: tiled analysis, plus CTU-wavefront-parallel analysis for the
  exhaustive tier.
- Hot kernels (SATD, SSD, residual subtraction, forward DCT) have pure-Rust SIMD
  implementations that are **bit-identical** to the scalar reference (unit-tested
  and verified by byte-identical end-to-end encodes); `BPG_PRIMITIVES=scalar`
  forces the scalar path.

**Not ported (out of scope for a still encoder):**

- Motion estimation, inter prediction, B/P frames, the DPB, and lookahead.
- Rate control beyond CQP — no ABR/CRF/VBV or bitrate-targeted 2-pass.
- `psy-rd` and `ssim-rd`: the search uses the plain `D + λ·R` RD objective. The
  SSIM-RD energy term from x265's `tune=ssim` is **not** ported.

The known ~4% BD-rate gap to x265 is concentrated in luma coefficient-coding
efficiency (RDOQ / coefficient decimation) at equal QP, not in partitioning,
search depth, transforms, or the in-loop filters — those have been ruled out by
direct A/B comparison.

## BPG container & the `tune=ssim` AQ bug

The `libbpg` container/color layers were ported in full, and `bpgenc`'s
parameter setup was reproduced — including its request for x265's `tune=ssim`.

`tune=ssim` is meant to enable spatial adaptive quantization. But x265 force-clears
`aqMode`/`hevcAq`/`cuTree`/`aqStrength` whenever the rate-control mode is CQP, and
BPG always encodes in CQP. So in the C reference the `tune=ssim` AQ never actually
runs — every `bpgenc` still is effectively uniform-QP. (x265 itself warns that
SSIM tuning with AQ disabled is not meaningful.)

`bpg-rs` fixes this: a still encode in CQP can still emit legal per-quantization-
group `cu_qp_delta` syntax, so spatial AQ genuinely works. It is exposed via
`--aq` and defaults to **off** (uniform QP) to match stock `bpgenc` output; turn
it on to actually exercise perceptual AQ.

## Adaptive quantization

The `--aq` presets implement perceptual variance AQ following **Prangnell 2017**,
redistributing QP around the picture mean by local activity:

- `perceptual` / `perceptual-mild` — luma-activity AQ (the `-mild` variant halves
  the strength).
- `perceptual-chroma` / `perceptual-chroma-mild` — adds chroma activity, the
  luma+chroma model from Prangnell 2017.
- `two-pass` — measures coded complexity in pass 1, redistributes QP in pass 2,
  and keeps the result only when it is a perceptual RD win over uniform QP.

`--aq-strength` and `--aq-clamp` tune any preset. `legacy-shrink`, `psnr-probe`,
and `positive-probe` are diagnostics, not quality presets.

## References

- M. T. Prangnell, "Spatiotemporal Adaptive Quantization for the Perceptual Video
  Coding of HEVC" (2017) — the perceptual variance AQ (luma + chroma activity)
  behind the `--aq perceptual*` presets.

Earlier versions cited fast-encoder literature (light-weight CU-split early
termination; angular intra-mode exclusion; the Kvazaar accumulated-cost
branch-and-bound prune). Those experiments were not retained in the production
search path — `fast`'s early-outs are now plain x265-style residual/cost
heuristics — so they are no longer cited as in-use.
