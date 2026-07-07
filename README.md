# bpg-rs

A pure-Rust implementation of the [BPG image format](https://bellard.org/bpg/),
Fabrice Bellard's HEVC-intra still-image container.

The workspace provides:

- `still265`: a Rust-native HEVC still-image encoder, shaped after x265's
  all-intra still-picture path;
- `bpg-hevc-decode`: a pure-Rust HEVC still-image decoder;
- `bpg-tools`: a CLI for encoding and decoding images;
- supporting crates for the BPG container, HEVC bitstream handling, image
  conversion, and codec plumbing.

No C or C++ encoder/decoder is built or linked. There is no `cmake`, `libx265`,
`libde265`, or `libbpg` dependency in the Rust workspace. The tools read PNG/JPEG,
write BPG, and decode BPG/HEIC/HEIF back to JPEG or PNG.

## Current state

`still265` is now in practical speed parity range with C `bpgenc -e x265 -m9`
for the tested still-image workload. It is not uniformly faster, but the old
1.5–2.5× Rust-vs-C gap is gone: the latest 7-image sweep is **1.016× aggregate**
against `bpgenc -m9`, with two images faster than C and the remaining tail mostly
in textured 12 MP inputs.

Latest documented sweep:

- encoder: `still265` `slow`, QP 28, native mode;
- baseline: on-PATH x265 4.1 `bpgenc -m9`;
- build: `release-lto`;
- protocol: best-of-2, idle desktop;
- dataset: seven 12 MP / 50 MP native photos from `test-set-large`.

| image | MP | still265 | bpgenc/x265 | ratio |
|---|---:|---:|---:|---:|
| 150624 | 50 | 2.253 s | 2.596 s | **0.87×** |
| 155124 | 50 | 3.138 s | 3.164 s | **0.99×** |
| 155040 | 50 | 3.652 s | 3.498 s | 1.04× |
| 203230 | 12 | 1.105 s | 1.060 s | 1.04× |
| 155515 | 12 | 1.261 s | 1.196 s | 1.05× |
| 154651 | 12 | 1.428 s | 1.291 s | 1.11× |
| 201839 | 12 | 1.527 s | 1.331 s | 1.15× |
| **aggregate** |  | **14.36 s** | **14.14 s** | **1.016×** |

The mean of the per-image ratios was 1.036× and the median was 1.044×. Smooth
50 MP images are already faster than, or effectively tied with, C `bpgenc` on
this machine. The remaining gap is not a general kernel-throughput deficit: the
latest pinned/user-CPU check found still265 per-thread work at parity or better
on the worst 12 MP image. The residual speed difference is mainly parallel
scheduling efficiency and image-content sensitivity.

The latest parity table is a speed table. Earlier equal-QP quality sweeps showed
still265 `slow` producing slightly higher Y-PSNR than x265 while spending more
bytes on the tested photos. Most later speed work was either byte-identical or
BD-neutral, but a fresh BD-rate table should be used before making a precise
compression-efficiency claim.

> **QP note.** `still265 --qp N` uses actual HEVC QP `N`. C `bpgenc`'s nominal
> `-q` value is offset by its x265 `tune=ssim` setup, so benchmark comparisons
> should use the harness's native/equal-QP mode or a BD-rate sweep, not equal
> command-line `-q` numbers.

## What it is for

BPG stores a single still image as an all-intra HEVC frame in a compact
container. It can deliver JPEG-class file sizes at higher quality, or smaller
files at similar quality, while remaining compatible with stock BPG decoders.

`bpg-rs` exists as a Rust-native implementation of that stack. It is intended
for applications that want BPG encode/decode without shipping a C/C++ codec
build, and for still-image archival or conversion pipelines where CQP HEVC intra
coding is acceptable.

It is not a video encoder. There is no motion estimation, inter prediction,
lookahead, DPB management, ABR/CRF/VBV rate control, animation support, alpha
encode path, or lossless mode.

## Crates

| Crate | Purpose |
|---|---|
| `bpg-bitstream` | BPG `ue7` container varints and MSB-first bit I/O with HEVC Exp-Golomb support. |
| `bpg-image` | Image planes, color spaces, RGB↔YCbCr conversion, chroma subsampling, bit-depth handling, and CTU padding. |
| `bpg-format` | BPG container header read/write. |
| `bpg-hevc` | Annex-B NAL parsing, emulation-prevention handling, and VPS/SPS rewriting for BPG's modified HEVC stream. |
| `bpg-hevc-decode` | Pure-Rust HEVC still-picture intra decoder: CABAC, intra prediction, transform/dequant, deblocking, and SAO. |
| `bpg-decode` | BPG/HEIC/HEIF container decode to RGB(A), BGR(A), PNG, or JPEG output paths. |
| `bpg-encode` | `HevcEncoder` trait and still-image encode orchestration. |
| `still265` | Rust-native HEVC intra still encoder. |
| `toojpeg` | Vendored minimal JPEG writer used by decode output. |
| `bpg-tools` | CLI exposing `encode`, `decode`, and comparison/development tools. |

## Building and testing

```bash
cargo build --release
cargo test --release
```

Release builds use thin LTO. The benchmarked parity numbers use the explicit
fat-LTO profile:

```bash
cargo build --profile release-lto -p bpg-tools
```

The documented thin-vs-fat LTO check found only noise-level differences, so
`cargo build --release` is appropriate for development and ordinary use;
`release-lto` is the comparison profile for benchmark tables.

The workspace is pure Rust. Development notes in this repository use Rust 2024
formatting conventions and recent measurements were run with rustc 1.94.x.

## CLI

Encode a PNG or JPEG to BPG:

```bash
cargo run -p bpg-tools --release -- encode input.png -o out.bpg \
  --qp 28 --format 420 --effort slow
```

Important encode options:

- `--qp 0..51`: constant HEVC quantizer. Lower is larger/higher quality.
- `--effort fast|slow|placebo`: RD-search budget. Default: `slow`.
- `--format gray|420|422|444`: output chroma format.
- `--bit-depth 8|10|12`: encode bit depth. 10/12-bit paths need suitable input
  to be useful.
- `--aq`: optional adaptive quantization preset. Defaults to `off` to match
  stock `bpgenc`'s effective uniform-QP behavior.
- `--no-sao` / `--no-deblock`: disable in-loop filters. Both are on by default.
- `--color-space ycbcr|rgb|ycgco|bt709|bt2020` and `--limited-range`: color
  handling options. RGB/YCgCo encode requires `--format 444`.

Effort tiers:

- `fast`: reduced search for lower latency and larger files. It keeps the same
  codec tools but uses narrower decisions and earlier termination.
- `slow`: the production tier and current benchmark target. This is the tier
  that reached practical speed parity with `bpgenc -m9` on the documented sweep.
- `placebo`: exhaustive/reference tier. Useful for regression and comparison;
  not normally the best quality/speed point for photos.

Older ladder names such as `fastest`, `balanced`, `best`, and `slowplus` are
accepted as aliases and collapse onto the current effort tiers.

Decode a BPG, HEIC, or HEIF file:

```bash
cargo run -p bpg-tools --release -- decode in.bpg -o out.jpg
cargo run -p bpg-tools --release -- decode in.bpg -o out.png --format rgba
```

JPEG output preserves the source's native YCbCr path for the common BT.601
full-range 8-bit case instead of forcing an RGB round trip. The decoder supports
gray, 4:2:0, 4:2:2, and 4:4:4; 8/10-bit; in-loop filters; and non-CTU-aligned
image sizes.

## Encoder architecture

`still265` is similar to x265 where it matters for still-image coding, but it is
not a direct transplant of x265's frame encoder.

### x265-shaped pieces

The encoder carries over the core all-intra HEVC machinery:

- CABAC context models and table-driven RD bit estimation;
- VPS/SPS/PPS and slice writing for BPG-compatible modified HEVC;
- all 35 intra modes, reference substitution, and reference filtering;
- forward/inverse DCT/DST transforms, quant/dequant, RDOQ, and sign-data hiding;
- rough SATD mode decision followed by full-RD luma/chroma decisions;
- transform-tree and coding-unit split RD;
- NxN PU handling;
- deblocking and SAO;
- CQP operation compatible with BPG's still-image encode model.

The current `slow` path intentionally tracks x265's still-picture search shape
more closely than older versions did. Notable changes include the x265-style
64×64 CTU-root force split, fixed-point RDOQ cost arithmetic, and reuse of the
root transform-unit result already found during the cheap luma stage.

### Rust-side architecture

The recent parity work changed the encoder's structure substantially:

- **Contiguous CTU canvas.** The old reconstruction overlay patch stack was
  replaced by a per-CTU working canvas. Border reads are now straight top/left
  canvas or strip reads instead of patch-stack scans. This shipped byte-identical
  and measured a 9–11% single-thread win in the documented 12 MP case.
- **Frameless WPP workers.** WPP workers no longer clone or own full
  reconstruction frames. They operate from row strips, small maps, and CTU-local
  canvas state, then publish CTU rectangles to the master frame. This removed the
  old O(workers × frame) memory behavior; the documented 50 MP working set fell
  from 3855 MB to 999 MB in that A/B.
- **WPP row-segment handoff.** Rows are owner-free and can be parked/resumed when
  dependencies stall or the frontier should move. The path is byte-identical and
  run-to-run deterministic; release policy is geometry-aware so short rows avoid
  excessive voluntary handoff traffic.
- **Topology-aware worker budget.** On hybrid CPUs, narrow WPP rows default to the
  performance-core thread count instead of blindly using every logical worker.
  `BPG_ENC_THREADS` still overrides the automatic budget.
- **Parallel filters.** SAO decision/application and deblocking have parallel
  paths. These are byte-identical changes and also benefit decode-side SAO where
  applicable.
- **Descent gate.** `slow` uses a parent-only split-descent gate in smooth areas.
  It is a measured, BD-guarded still-image optimization rather than an x265
  feature; Fast/Placebo do not rely on it.
- **Root-TU reuse.** The cheap-stage luma winner can be replayed in the exact
  stage when all preconditions match. The shipped path has a verify mode that
  recomputes and asserts equality; the documented hit rate was 100% on the 12 MP
  canonical encode.

The result is no longer well described as "Rust is doing the same work as x265,
but slower." In the latest audit, per-thread work was already at parity or
better on the checked 12 MP case. The remaining differences are mostly parallel
utilization, still-image-specific pruning, and some smaller unmerged dedup/SIMD
opportunities.

## SIMD and threading controls

Useful environment variables:

| Variable | Purpose |
|---|---|
| `BPG_ENC_THREADS=N` | Explicitly cap encoder worker count. Overrides automatic WPP budgeting. |
| `BPG_PRIMITIVES=auto|simd|scalar` | Select primitive dispatch for A/B testing. |
| `BPG_WPP_HANDOFF=0` | Disable row-segment handoff and use the whole-row WPP fallback. |
| `BPG_WPP_HANDOFF_RELEASE=auto|stall-only|1|2|4|N` | Tune voluntary row release cadence. Default is geometry-aware. |
| `BPG_WPP_HYBRID_CAP=0` | Disable the hybrid-CPU worker cap. |
| `BPG_STILLSEARCH_DESCENT_GATE=0` | Disable the Slow-tier descent gate for A/B. |
| `BPG_STILLSEARCH_ROOT_TU_REUSE=0|verify` | Disable or verify root-TU reuse. |

These are primarily benchmark and development controls. Normal users should
start with the defaults.

## Status and compatibility

Implemented:

- still-image lossy CQP encode and decode;
- stock BPG-compatible output streams;
- decode of third-party BPG/HEIC/HEIF still images covered by the supported HEVC
  intra feature set;
- gray, 4:2:0, 4:2:2, and 4:4:4;
- 8/10/12-bit paths, with round-trip tests;
- YCbCr BT.601/709/2020, RGB/GBR, and YCgCo paths where the format supports it;
- deblocking and SAO on by default;
- `HevcEncoder::caps()` so unsupported encode requests fail up front.

Not implemented / out of scope:

- video encode or decode features beyond still intra pictures;
- motion estimation, inter prediction, B/P frames, lookahead, or DPB management;
- ABR, CRF, VBV, or bitrate-targeted two-pass rate control;
- animation;
- alpha encode path;
- lossless encode mode.

Recent speed-path changes were gated with the still265 suite, byte-identity
sweeps across 12/50 MP and QP points where applicable, stock decoder round trips,
serial/tile/WPP fallback checks, and run-to-run determinism checks for the WPP
handoff path.

## Adaptive quantization

BPG's reference pipeline requests x265 `tune=ssim`, but BPG encodes in CQP and
x265 clears its AQ state under CQP. In practice, stock `bpgenc` still images are
uniform-QP even when the preset name suggests SSIM tuning.

`bpg-rs` exposes real still-image AQ via legal per-quantization-group
`cu_qp_delta` syntax, while keeping `--aq off` as the default for stock-`bpgenc`
comparability.

Available AQ presets:

- `perceptual` / `perceptual-mild`: luma-activity variance AQ;
- `perceptual-chroma` / `perceptual-chroma-mild`: luma+chroma activity AQ;
- `two-pass`: measures coded complexity in a first pass, redistributes QP in a
  second pass, and keeps the result only when it wins the perceptual RD check.

`--aq-strength` and `--aq-clamp` tune the presets. `legacy-shrink`,
`psnr-probe`, and `positive-probe` are diagnostics rather than recommended
quality modes.

## References

- Fabrice Bellard, BPG image format.
- x265 4.1, all-intra still-picture path used by `bpgenc -e x265 -m9`.
- M. T. Prangnell, "Spatiotemporal Adaptive Quantization for the Perceptual Video
  Coding of HEVC" (2017), for the perceptual variance AQ model.
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
