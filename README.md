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
# Installable end-user command (also available as `cargo run -p bpg-tools`).
cargo run -p bpg-tools --release -- input.png -o out.bpg \
  --qp 28 --format 420 --effort slow --aq
```

The installed binary is `bpgenc`, so the equivalent command is
`bpgenc input.png -o out.bpg --qp 28 --format 420 --effort slow --aq`.
Install it locally with `cargo install --path crates/bpg-tools --bin bpgenc`.
`bpg-tools encode ...` remains as a compatible explicit-subcommand form; use
`bpgenc decode ...` (or `bpg-tools decode ...`) for decoding.

Important encode options:

- `--qp 0..51`: constant HEVC quantizer. Lower is larger/higher quality.
- `--effort fast|slow|placebo`: RD-search budget. Default: `slow`.
- `--format gray|420|422|444`: output chroma format.
- `--bit-depth 8|10|12`: encode bit depth. 10/12-bit paths need suitable input
  to be useful.
- `--aq [MODE]`: optional adaptive quantization. Bare `--aq` chooses the
  recommended measured two-pass mode; omit it (or use `--aq off`) for
  stock-`bpgenc`-comparable uniform QP. Pass a mode explicitly to experiment
  with a single-pass alternative.
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
`cu_qp_delta` syntax. AQ remains opt-in for stock-`bpgenc` comparability, but
when enabled bare `--aq` selects the measured two-pass path by default.

Available AQ presets:

- `two-pass` (the bare `--aq` default): measures coded complexity in a first
  pass, redistributes QP in a second pass, and keeps the result only when it
  wins the perceptual RD check;
- `perceptual` / `perceptual-mild`: luma-activity variance AQ;
- `perceptual-chroma` / `perceptual-chroma-mild`: luma+chroma activity AQ;
- `auto-variance` / `auto-variance-biased`: x265-style single-pass AQ modes.

`--aq-strength` and `--aq-clamp` tune the presets. `legacy-shrink`,
`psnr-probe`, and `positive-probe` are diagnostics rather than recommended
quality modes.

## References

- Fabrice Bellard, BPG image format.
- x265 4.1, all-intra still-picture path used by `bpgenc -e x265 -m9`.
- M. T. Prangnell, "Spatiotemporal Adaptive Quantization for the Perceptual Video
  Coding of HEVC" (2017), for the perceptual variance AQ model.
