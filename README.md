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
  to benefit from 10/12-bit output. The `still265` backend supports all three.
- `--format` (420/422/444): chroma subsampling. The `still265` encoder
  supports all three.
- `--effort` (floor/floor-plus/floor-plus2/floor-shallow/slow/slow-plus/fastest/fast/balanced/good/best/placebo/reference):
  RD-search effort ladder
  (HandBrake/x265-style), trading encode time for rate-distortion quality.
  Higher tiers run more RD trials — i.e. call the CABAC bit estimator more
  often, which is the dominant cost. `balanced` is the default.
  - `fastest`/`fast`/`balanced` apply encoder-side, bitstream-neutral search
    pruning (CU-split early termination, liang2013; angular intra-mode
    exclusion, heindel2016) at decreasing aggressiveness. Averaged over a
    16-image 1024x768 photo set (4:2:0, vs. the unpruned encoder): `fastest`
    ~30-50% faster (+0.3-1.8% size), `fast` ~20-40% faster (+0.1-0.6%),
    `balanced` ~10% faster (≈0% size).
  - `fastest` is intentionally Kvazaar-like and aggressive: sparse progressive
    SATD RMD, one luma RD candidate, chroma DM-only, approximate residual bits,
    plain-quant trials, zero-residual CU early stop, and fixed-leaf small TUs.
  - `slow` is the current high-res quality experiment: a shallow 64→32-first
    still-image CTU search with progressive RMD, final-winner RDOQ, and
    classifier-steered split/search pruning. `slow-plus` is a deliberately
    separate experimental successor: it skips the root 64×64 leaf below QP35
    (matching x265's all-intra root behavior), widens the progressive RMD budget,
    and enables conservative/selective PartNxN islands by default.
  - `fast`/`balanced`/`good` additionally apply **source-derived preanalysis
    steering**: a cheap per-32x32-cell structure map (variance, edge density,
    gradient-orientation entropy, noise, chroma activity) spends search budget
    non-uniformly — forcing angular pruning on flat/gradient cells, guarding it
    on edge/text cells, and (for `good`) spending extra RD candidates on
    text/colour-critical cells. Bitstream-neutral; decode-validated against stock
    `bpgdec`. On an 8-image photo subset (qp28, 4:2:0) it gives `balanced`/`good`
    ~6% faster at ≈equal size; the protective guards mainly help on
    screenshot/scan/line-art content. `fastest` is left unsteered (its local
    pruning is already maximal). See `docs/remaining-gaps.md`.
  - `fastest`/`fast`/`balanced`/`good` are also QP-aware: at high QP they narrow
    rough-mode sweeps, luma/chroma RD-candidate counts, limited-RDOQ coverage,
    and selected split early-outs. `best`/`placebo`/`reference` stay exhaustive
    at rough-mode search.
  - `best` is the practical max-quality tier: exhaustive rough-mode search,
    full-RD luma/chroma trials, the full chroma candidate set, exact residual-bit
    pricing, exact final coding, and uniform QP, while still using the
    non-reference luma-only candidate path and single-scan RDOQ. It is meant to
    sit between `good` and `placebo`.
  - `placebo` and `reference` run the byte-exact exhaustive reference family
    (all 35 rough modes, full chroma RD during luma trials, exact greedy RDOQ,
    no pruning). `reference` is single-threaded with exact running-context RD —
    the slowest, bit-reproducible regression reference. `placebo` is the same
    exhaustive search with CTU-wavefront-**parallel** analysis (RD priced against
    a frozen slice-init context, which enables the parallelism), giving a large
    multicore speedup (~3x on a 12-CTU-row 1024x768 image) for a small size cost
    vs `reference`. `BPG_ENC_THREADS` caps the worker count (set to 1 to
    reproduce `placebo`'s output single-threaded).
- `--sao`: enable the conservative SAO encoder (off by default; see Status).
- `--no-deblock`: disable the in-loop deblocking filter (on by default).
- `--color-space` (ycbcr/rgb/ycgco/bt709/bt2020) and `--limited-range`.

On completion `encode` prints the output size, the effort tier, and the
wall-clock encode time, e.g. `wrote out.bpg (220654 bytes, effort Placebo,
1180.34s)` — useful for gauging the cost of the slower tiers (`reference`/
`placebo` on a large camera image can take many minutes).

The `BPG_PRIMITIVES` env var (`scalar`/`simd`/`auto`) overrides primitive
backend selection for A/B testing; `BPG_RDOQ_SINGLESCAN` overrides the RDOQ
path.

Decode a BPG to PNG:

```bash
cargo run -p bpg-tools --release -- decode input.bpg -o output.png \
  --format rgba
```

The decoder supports gray, 4:2:0, 4:2:2, and 4:4:4 chroma, 8/10-bit depths,
and non-CTU-aligned image sizes.

## Status

- Still-image, intra-only, lossy (CQP) encoding and decoding. No animation,
  no alpha plane, no lossless mode.
- Color space: YCbCr (BT.601/709/2020 matrices), RGB/GBR, and YCgCo are
  implemented end-to-end for 4:4:4. RGB/YCgCo encode currently requires
  `--format 444`.
- Chroma: encoder and decoder both support gray, 4:2:0, 4:2:2, and 4:4:4.
- Bit depth: 8, 10, and 12-bit are implemented end-to-end for gray/4:2:0/4:2:2/4:4:4
  (encode -> decode round-trip tested in `still265/tests/encode_roundtrip.rs`).
- In-loop filters: deblocking is implemented and enabled by default (signalled
  in the PPS/slice header with zero beta/tc offsets, applied post-CABAC using
  the same TU-edge marks the decoder would derive). SAO is implemented as a
  conservative band-offset/edge-offset encoder, **off by default**
  (`--sao` / `RustStillHevcEncoder::with_sao`); identical neighbour CTBs are
  emitted with SAO merge syntax. The decoder implements both filters fully for
  decoding third-party BPG files.
- Primitives: hot kernels dispatch through a `still265::primitives::Primitives`
  function-pointer table chosen at startup from CPU features + `BPG_PRIMITIVES`.
  The scalar implementations are canonical. Optimized kernels: 8-bit SATD
  (SSE2, `x86_64`, always built); and, in normal `bpg-tools` builds through the
  pure-Rust `wide-simd` feature, 10/12-bit SATD, SSD (RD distortion), residual
  subtraction, and the forward 1-D DCT. Every optimized kernel is
  **bit-identical** to scalar (unit-tested, and verified by byte-identical
  end-to-end encodes across 8/10/12-bit × 4:2:0/4:2:2/4:4:4 ×
  fast/balanced/best). The build remains pure Rust with no external C/C++/asm;
  `BPG_PRIMITIVES=scalar` remains available for A/B testing. On the
  analysis-bound effort tiers it measures ~8–20% faster (the 8-bit `best` tier
  is CABAC-bound, so it moves little). Remaining kernels are listed in
  `docs/remaining-gaps.md`.
- `bpg_encode::HevcEncoder::caps()` reports a backend's supported bit depths,
  chroma formats, and in-loop-filter/lossless/alpha support; `bpg-tools`
  checks the request against this before doing any image I/O or encoding
  work, so unsupported combinations (e.g. `--format 422`) fail with a clear
  error up front rather than panicking deep in the encoder.
- Mode decision: full-RD luma/chroma intra mode search, TU-split RD, CU-split
  RD, and multi-pass RDOQ with effort-tier-selected last-significant-position
  optimization. The `--effort` tiers
  (floor/floor-plus/floor-plus2/floor-shallow/slow/slow-plus/fastest/fast/
  balanced/good/best/placebo/reference)
  scale the search breadth — rough-mode shortlist size, luma/chroma RD-candidate
  counts, single-scan-vs-exact RDOQ, and high-QP pruning for the non-reference
  tiers. Multi-candidate non-reference CUs choose luma with a DM-chroma proxy,
  then run chroma mode search once for the winning luma mode; `placebo`/
  `reference` keep the exhaustive joint search. RD costs are priced by a
  table-driven
  CABAC estimator (x265's `g_entropyBits`, no arithmetic-coder simulation), and
  each coded block's residual bit cost is memoized on the block so the
  transform-tree estimator never re-prices residuals — so the encoder is
  call-bound (estimator invocations scale with search breadth), which is why the
  effort ladder is the primary speed/quality control.

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
basis. (These numbers predate the deblocking and SAO encode support and the
RDOQ stage-(c) work; the table has not been re-measured against the full
3200x2528 photo since.) Finer RD search and enabling SAO/deblock in the
default quality path remain the main quality work.

### High-resolution multi-effort timing (native 12 MP / 50 MP)

Measured 2026-06-22 on a 20-core machine, all SIMD + rayon enabled, multi-threaded,
QP 28, 4:2:0, 8-bit, on two native photos from `test-set/test-set-large`
(`bpg-highres-compare --native`). C is `bpgenc 0.9.8` (x265 4.1) at `-m 9`. PSNR is
the Rust reconstruction vs. source (Y); the file-size deltas are at equal QP, so
they indicate compression efficiency, not a full rate-distortion match (no C PSNR
is decoded here, and the C/Rust QP scales are not guaranteed identical).

**12.0 MP (3000×4000)** — C: 1.56 s, 789,285 B

| effort | encode s | s/MP | vs C time | bytes | vs C size | psnr_y |
|---|---:|---:|---:|---:|---:|---:|
| best | 4.83 | 0.40 | 3.1× | 743,974 | −5.7% | 40.75 |
| slow | 2.77 | 0.23 | 1.8× | 838,590 | +6.2% | 40.62 |
| fastadaptive | 2.06 | 0.17 | 1.3× | 867,772 | +9.9% | 40.38 |

**49.9 MP (8160×6120)** — C: 2.90 s, 570,912 B

| effort | encode s | s/MP | vs C time | bytes | vs C size | psnr_y |
|---|---:|---:|---:|---:|---:|---:|
| best | 10.70 | 0.21 | 3.7× | 655,311 | +14.8% | 46.84 |
| slow | 8.20 | 0.16 | 2.8× | 674,751 | +18.2% | 47.07 |
| fastadaptive | 5.70 | 0.11 | 1.9× | 675,772 | +18.4% | 46.90 |

Notes:

- **Tiling / parallel scaling:** per-MP encode time *drops* as resolution grows
  (`best` 0.40 → 0.21 s/MP, `slow` 0.23 → 0.16, `fastadaptive` 0.17 → 0.11),
  i.e. the tiled encoder now spreads larger pictures across cores instead of doing
  flat work-per-pixel. Part of the drop is also content (the 50 MP photo is smoother,
  ~3× fewer CU trials/MP), so the two effects are not fully separable from this
  two-image run, but the direction confirms tiling is engaging at high resolution.
- **Slow vs. Best:** `best` adds NxN-PU search and exhaustive rough-mode search
  (≈15× the CU trials of `slow`) for ~1.3–1.7× the wall-clock. The payoff is
  size: at 12 MP `best` is ~11% smaller than `slow` at equal/slightly-better PSNR;
  at 50 MP the gap shrinks to ~3% smaller while `slow` actually edges PSNR
  (47.07 vs 46.84 dB), so on smooth high-res content `slow` is close to RD-parity
  with `best` at meaningfully lower cost.
- **vs. C:** still265 stays ~1.3–3.7× slower than x265 `-m 9` and the gap widens
  with resolution (C parallelizes more aggressively per-MP), but `best` produces a
  *smaller* file than C on the 12 MP photo at this QP.

## Repository layout

```text
bpg-rs/
  crates/            # workspace member crates (see table above)
```

## References

Encoder-side techniques adapted from the literature (used in the `still265`
effort tiers):

- F. Liang, X. Peng, and J. Xu, "A Light-Weight HEVC Encoder for Image Coding"
  (Microsoft Research Asia / USTC) — CU-split early termination, used in the
  `fastest`/`fast`/`balanced` tiers.
- A. Heindel, C. Pylinski, and A. Kaup, "Two-Stage Exclusion of Angular Intra
  Prediction Modes for Fast Mode Decision in HEVC" (FAU Erlangen-Nürnberg) —
  global angular intra-mode exclusion, used in the `fastest`/`fast`/`balanced`
  tiers.
- A. Lemmetti, E. Kallio, M. Viitanen, J. Vanne, and T. D. Hämäläinen,
  "Rate-Distortion-Complexity Optimized Coding Scheme for Kvazaar HEVC Intra
  Encoder," 2018 Data Compression Conference (DCC), Tampere University of
  Technology — accumulated-cost (branch-and-bound) CU-split termination. Wired
  into the CU-split RD decision for **all** tiers as an *exact*,
  bitstream-neutral prune: the four children are built one at a time and the
  split is abandoned as soon as the accumulated child RD cost provably exceeds
  the leaf cost. Output is byte-identical to full evaluation; it removes ~2-3%
  of CU-split RD trials, but wall-clock is largely unchanged on photographic
  input because the prune only fires on smooth CUs (whose skipped children are
  cheap), while residual bit-estimation/RDOQ dominates the cost. See
  `docs/remaining-gaps.md`.
