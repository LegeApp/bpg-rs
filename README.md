# bpg-rs

A pure-Rust implementation of the [BPG image format](https://bellard.org/bpg/)
(Fabrice Bellard's HEVC-intra-based still image container): an encoder
(`still265`, a from-scratch port of x265's still-picture intra path) and a
decoder (`bpg-hevc-decode`), tied together by a `bpg-tools` CLI.

The entire workspace is pure Rust — no C/C++ code is built or linked.

## What's here

| Crate | Purpose |
|---|---|
| `bpg-bitstream` | `ue7` base-128 varints (container header fields) and MSB-first `BitReader`/`BitWriter` with Exp-Golomb `ue(v)` (HEVC RBSP). |
| `bpg-image` | `Image`/`Plane<u16>`, `ChromaFormat`, `ColorSpace`, RGB->YCbCr conversion (BT.601/709/2020, full/limited range, 8/10/12-bit), 4:4:4->4:2:0/4:2:2 chroma subsampling, CTU padding. |
| `bpg-format` | The fixed-size BPG container header (read/write). |
| `bpg-hevc` | Annex-B NAL parsing/emulation-prevention handling, VPS/SPS rewriting for BPG's "modified HEVC" stream, encode- and decode-side stream rebuilding. |
| `bpg-hevc-decode` | Vendored pure-Rust HEVC still-picture intra decoder (CABAC, intra prediction, transform/dequant, deblocking, SAO). |
| `bpg-decode` | BPG container -> Annex-B -> `bpg-hevc-decode` -> RGB/RGBA/BGR(A) decode orchestration. |
| `bpg-encode` | `HevcEncoder` trait + still-image encode orchestration (pad -> encode -> rewrite HEVC headers -> BPG container). |
| `still265` | The Rust-native still-picture HEVC intra encoder: CABAC, NAL/RBSP writers, VPS/SPS/PPS, slice header, CTU/CU/TU recursion, intra prediction, transform/quant, RDOQ, and full-RD mode decision. |
| `bpg-tools` | `clap`-based CLI exposing `encode` and `decode` subcommands. |

Dependency order: `bpg-bitstream` -> `bpg-format`, `bpg-hevc`; `bpg-image` is
standalone; `bpg-encode` depends on `bpg-image` + `bpg-format` + `bpg-hevc`;
`still265` implements `bpg-encode::HevcEncoder`; `bpg-hevc-decode` is
standalone; `bpg-decode` depends on `bpg-bitstream` + `bpg-format` +
`bpg-hevc` + `bpg-hevc-decode`; `bpg-tools` ties everything together.

## Building and testing

```bash
cargo build --release
cargo test --release
```

Pure Rust, no `cmake`/C toolchain required. Toolchain: edition 2021,
`cargo`/`rustc` 1.94+.

## Using the CLI

Encode a PNG to BPG:

```bash
cargo run -p bpg-tools --release -- encode input.png -o output.bpg \
  --qp 28 --format 420 --effort balanced
```

- `--qp` (0-51): quantizer for CQP-style rate control.
- `--bit-depth` (8/10/12): output sample depth; 16-bit PNG input is required
  to benefit from 10/12-bit output.
- `--format` (420/422/444): chroma subsampling. The `still265` encoder
  currently supports 4:2:0 and 4:4:4; 4:2:2 encode is not yet implemented.
- `--effort` (fast/balanced/best): RD-search effort, trading encode time for
  rate-distortion quality.
- `--color-space` (ycbcr/bt709/bt2020) and `--limited-range`.

Decode a BPG to PNG:

```bash
cargo run -p bpg-tools --release -- decode input.bpg -o output.png \
  --format rgba
```

The decoder supports 4:2:0, 4:2:2, and 4:4:4 chroma, 8/10-bit depths, and
non-CTU-aligned image sizes.

## Status

- Still-image, intra-only, lossy (CQP) encoding and decoding. No animation,
  no alpha plane, no lossless mode.
- Color space: YCbCr (BT.601/709/2020 matrices). RGB and YCgCo color spaces
  are unimplemented stubs.
- Chroma: encoder supports 4:2:0 and 4:4:4 (4:2:2 encode pending); decoder
  supports 4:2:0, 4:2:2, and 4:4:4.
- Bit depth: 8 and 10-bit are implemented end-to-end; 12-bit is a stub.
- In-loop filters: SAO and deblocking are signalled off by the encoder
  (conformant bitstreams, just not yet used for extra rate-distortion gain).
  The decoder implements both for decoding third-party BPG files that use
  them.
- Mode decision: full-RD luma/chroma intra mode search, TU-split RD, CU-split
  RD, and multi-pass RDOQ with effort-tier-selected last-significant-position
  optimization.

### Quality vs. the C/x265 reference

Historical comparisons against the original C `bpgenc -e x265` on a
3200x2528 8-bit photo (4:2:0, RGB-only PSNR/SSIM):

| Encoder | Config | Size (bytes) | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| `still265` (Rust) | qp30 420 balanced | 899,661 | 36.30 | 0.9770 |
| C `bpgenc -e x265` | qp30 420 | 992,634 | 38.08 | 0.9812 |
| C `bpgenc -e jctvc` | qp30 420 | 873,219 | 36.16 | 0.9719 |

At this quantizer the Rust encoder already produces smaller files than C
x265, but at lower PSNR/SSIM — x265 is not yet beaten on a quality-per-bit
basis. Closing that gap (primarily SAO, deblocking, and finer RD search) is
the main remaining quality work.

## Repository layout

```text
bpg-rs/
  crates/            # workspace member crates (see table above)
```
