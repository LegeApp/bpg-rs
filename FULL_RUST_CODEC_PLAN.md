# Full Rust BPG Codec Plan

## Goal

Turn `bpg-rs` into a self-contained Rust BPG encoder/decoder.

Short term: vendor the Rust HEVC decoder currently referenced from
`heic-decoder-rs` and remove the external path dependency.

Long term: replace the current x265 FFI backend with a Rust still-picture HEVC
intra encoder derived from x265, while keeping only x265's original assembly
primitive codecs behind thin ABI wrappers.

## Current State

`bpg-rs` currently has:

- `bpg-bitstream`: `ue7` varints and bit-level I/O.
- `bpg-image`: RGB to YCbCr conversion, chroma subsampling, and padding.
- `bpg-format`: BPG header write/read.
- `bpg-hevc`: encode-side modified HEVC builder and decode-side BPG payload to
  Annex-B rebuild.
- `bpg-encode`: generic `HevcEncoder` trait plus BPG still encode
  orchestration.
- `bpg-x265-sys` / `bpg-x265`: current C/C++ x265 backend.
- `bpg-decode`: current BPG decode API, still depending on external
  `heic-decoder-rs`.

The existing `x265-port-plan.md` has the right strategic target: do not port
the full x265 video encoder. Port a still-image intra HEVC encoder that keeps
x265's proven intra/RDO/CABAC/primitive behavior and drops video-only
machinery.

The HEIC decoder source provides the immediate decoder module source:
`bitstream`, `params`, `slice`, `cabac`, `ctu`, `intra`, `residual`,
`transform`, `deblock`, `sao`, and `picture`.

## Target Crate Layout

```text
bpg-rs/
  crates/
    bpg-bitstream
    bpg-image
    bpg-format
    bpg-hevc
    bpg-hevc-decode      # vendored/adapted HEIC HEVC decoder
    bpg-decode           # BPG decode API using bpg-hevc-decode
    bpg-encode           # stable BPG encode orchestration
    bpg-x265             # temporary oracle/backend, eventually optional
    bpg-x265-sys         # temporary oracle/backend, eventually optional
    bpg-still-hevc       # Rust x265-derived still encoder
    x265-primitives
    x265-primitives-sys  # original asm + C ABI shims only
    bpg-tools
```

## Progress

### Phase 1 COMPLETE — decoder vendored, repo self-contained

The external `heic_decoder_original` path dependency is gone. The HEVC decode
module (`heic-decoder-rs/src/hevc/*`) is vendored as the new
`crates/bpg-hevc-decode` crate:

- `src/hevc/*` copied verbatim (no edits to the decoder logic), so the module
  still cross-checks against its upstream.
- `crate::error::HevcError` is provided by a local `src/error.rs` (the
  `HevcError` enum only — `HeicError` and the `enough`/`whereat`
  cancellation/location machinery were left behind).
- `crate::heif::HevcDecoderConfig` is provided by a minimal `src/heif.rs` shim
  (just `nal_units` + `length_size_minus_one`, the only fields the decoder
  references) so the length-prefixed `decode_with_config`/`get_info_from_config`
  entry points still compile. BPG decode itself only uses `hevc::decode(annexb)`.
- External deps reduced to `archmage` + `safe_unaligned_simd` (the SIMD crates),
  edition 2024 (the decoder uses let-chains).

`bpg-decode` now depends on `bpg-hevc-decode` instead of the external path; its
public API (`DecoderConfig`, `PixelLayout`, `DecodeOutput`, `ImageInfo`,
`Limits`, `DecodeError`) is unchanged.

Fixture tests added in `crates/bpg-decode/tests/fixtures.rs`:
- `html/lena512color.bpg` decodes to 512×512 RGBA/RGB.
- `html/clock.bpg` returns `Unsupported("animation")`.
- `tests/fixtures/alpha64.bpg` (generated via `bpgenc`) returns
  `Unsupported(alpha)`; its alpha flag is still visible via `ImageInfo`.

Verification: `cargo check --workspace` builds with no path dependency outside
this repo; `cargo test --workspace` passes 45 tests (incl. 9 vendored decoder
unit tests + 4 new fixtures). Decoded `lena` matches stock `bpgdec` at ~45.9 dB
PSNR (the heic-decoder-rs chroma-upsampling/YCbCr→RGB path is not bit-exact with
libbpg, but this is the decoder's pre-existing fidelity, unchanged by
vendoring). A full self-contained round trip — `bpg-tools encode --backend x265`
then `bpg-decode` — succeeds.

#### Phase 1 follow-up COMPLETE: decoder now supports 4:2:0, 4:2:2, 4:4:4

The Phase 2 oracle surfaced that the vendored decoder was **4:2:0-only** —
4:2:2/4:4:4 decoded to ~5–8 dB garbage (its transform-tree/residual/intra path
was written and tested only for 4:2:0, the dominant HEIC case). This has now
been fixed; the decoder reconstructs all three chroma formats, validated
against stock `bpgdec`:

- **4:4:4** (`ChromaArrayType==3`): chroma transform tree mirrors luma — cbf_cb/
  cbf_cr decoded at 4x4 too, chroma TBs at luma size/position decoded at every
  leaf, per-PU chroma modes for NxN, and the log2==3 directional chroma scan.
- **4:2:2** (`ChromaArrayType==2`): two vertically-stacked chroma TBs per luma
  TU, each with its own cbf (second cbf decoded per spec 7.3.8.8), and the
  Table 8-3 chroma intra mode remap.
- **Non-CU-aligned sizes** (all formats): BPG codes `pic_width/height` at the
  display size, so edge min-CUs straddle the picture boundary. The planes are
  now allocated at the CU-aligned coded size and cropped on output — this also
  fixed pre-existing 4:2:0 odd-size corruption (~26 dB → ~49 dB).

Results vs bpgdec (oracle `rust_psnr`): 4:4:4 49–99 dB, 4:2:2 30–99 dB, all
high (residual differences are the final YCbCr→RGB rounding / chroma-upsampling
choice, not reconstruction error). Odd 131×97: 4:2:0 49 dB, 4:2:2 51 dB, 4:4:4
70 dB (4:4:4 was previously a panic). `bpg-decode` no longer rejects any chroma
format. **Known remaining decoder gaps**: alpha plane, MPEG2 chroma siting
(`c_h_phase==0`), RGB/YCgCo color spaces, `separate_colour_plane_flag` — all
still `Unsupported`.

### Phase 2 COMPLETE — x265 oracle harness + regression corpus

New `crates/bpg-oracle` binary (`gen` / `check`):

- **Deterministic corpus** (`corpus.rs`): flat, gradient, checkerboard, line
  art, seeded noise, a 16-bit HDR gradient, and a center crop of `dusk.png`
  (8-bit + 16-bit) — the content classes the tuning roadmap targets. Generated
  from code with fixed seeds, so the corpus and manifest are reproducible.
- **Config matrix**: 8-bit × {4:2:0, 4:2:2, 4:4:4} × {qp 24, 32}; 16-bit ×
  {4:2:0, 4:4:4} × {qp 28} (exercises the linked 10-bit lib). 40 encodes / 8
  images.
- **Locked still-image settings** asserted by `bpg-x265` (intra-only, CQP,
  `bRepeatHeaders`, AMP, BT.601, JPEG siting) — written to `oracle/SETTINGS.md`
  by `gen`.
- For each encode it records: bpg bytes, rebuilt Annex-B (`*.hevc`), the
  ground-truth **stock-bpgdec** reference decode (`*.decoded.png`), and two
  metric axes — **encode-quality PSNR** (bpgdec vs source) and the **Rust
  decoder status/fidelity** (`ok`/`unsupported`/`fail` + PSNR vs bpgdec).
- `manifest.csv` + `SETTINGS.md` are committed; `corpus/` and `out/` are
  `.gitignore`d (regenerable). `check` regenerates and diffs `bpg_bytes` (exact)
  and encode-quality PSNR (±0.01 dB) against the committed manifest, assuming
  the same vendored x265 build (ENABLE_ASSEMBLY=OFF) and bpgdec.

Result: encode-quality PSNR is high (36–99 dB) across **all** chroma formats,
confirming the encoder is sound and giving the Phase 3 Rust encoder a concrete,
regenerable comparison baseline. 4 oracle unit tests (corpus determinism +
PSNR) run under `cargo test`.

**Next:** Phase 3 (`bpg-still-hevc` syntax skeleton) — the corpus + manifest are
the baseline its output will be measured against.

## Phase 1: Vendor The HEIC HEVC Decoder

1. Create `crates/bpg-hevc-decode`.
2. Copy only `heic-decoder-rs/src/hevc/*`, not the HEIF container code.
3. Replace `heic_decoder_original` error dependencies with local error types
   or a small Rust error enum.
4. Move BPG-specific decode flow into this shape:

   ```text
   BPG bytes
   -> bpg-format::BpgFile
   -> bpg-hevc::rebuild_annexb_from_bpg_payload
   -> bpg-hevc-decode::decode_annexb
   -> bpg-decode output layouts
   ```

5. Preserve the current `bpg-decode` public API:
   - `DecoderConfig`
   - `PixelLayout`
   - `DecodeOutput`
   - `ImageInfo`
   - `Limits`
6. Add fixture tests:
   - `html/lena512color.bpg` decodes.
   - current `bpg-tools encode` output decodes.
   - `clock.bpg` returns unsupported animation.
   - alpha/W-plane samples return unsupported.
7. Remove the external path dependency to
   `/mnt/Samsung980_1TB/Rust-projects/openarc/heic-decoder-rs`.

Acceptance: `cargo check --workspace` works with no dependency on an external
local path outside this repo.

## Phase 2: Establish The x265 Oracle

1. Keep `bpg-x265` as the reference backend.
2. Add an oracle test harness that records:
   - input image metadata
   - x265 params
   - raw Annex-B HEVC
   - final BPG bytes
   - decoded PNG
   - size and quality metrics
3. Build a small regression corpus:
   - flat color
   - gradients
   - checkerboards
   - line art
   - noisy patch
   - natural photo crop
   - 4:2:0, 4:2:2, and 4:4:4
   - 8-bit and 10-bit where supported
4. Lock the still-image x265 settings:
   - one frame
   - all intra
   - CQP
   - repeat headers
   - no lookahead
   - no scenecut
   - no B-frames
   - BPG-compatible SPS constraints

Acceptance: the oracle corpus can be regenerated and compared for decode
validity, size, and quality metrics.

## Phase 3: Port The x265 Syntax/Core Skeleton

Create `bpg-still-hevc` with a Rust-native still encoder API:

```rust
pub struct StillHevcEncoder;

pub struct StillHevcConfig {
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
    pub qp: u8,
    pub effort: Effort,
    pub sao: SaoMode,
    pub deblock: DeblockMode,
}
```

Port first:

- bitstream writer
- NAL writer
- VPS/SPS/PPS
- profile/level handling
- slice header writer
- CABAC shell and context initialization

Do not port motion, lookahead, rate control, B/P frames, or DPB beyond minimal
stubs.

Acceptance: the Rust encoder can emit syntactically valid headers and a
skeleton intra access unit that the decoder rejects only because slice data is
incomplete, not because headers are malformed.

## Phase 4: Build The Assembly Primitive Boundary

1. Create `x265-primitives-sys`.
2. Keep original x265 assembly files unchanged.
3. Add tiny C/C++ shims only where needed to expose stable unmangled symbols.
4. Create `x265-primitives` as the safe Rust wrapper with a dispatch table for:
   - SATD/SAD
   - DCT/IDCT
   - quant/dequant
   - intra prediction
   - pixel copy/variance
   - SAO/deblock kernels
5. Add scalar Rust fallbacks where practical, but treat x265 assembly output as
   the oracle.

Acceptance: primitive tests match x265 for fixed vectors across supported bit
depths.

## Phase 5: Port The Still Intra Encoder

Port only intra-relevant x265 logic:

- frame/picture memory
- CTU/CU/TU structures
- intra prediction
- transform
- quant/RDOQ
- residual coding
- CABAC syntax
- CU split/mode decision
- optional deblock/SAO

Split x265's mixed video files into Rust modules:

```text
frame.rs
ctu.rs
cu.rs
tu.rs
intra_pred.rs
transform.rs
quant.rs
rdo.rs
cabac_writer.rs
slice_writer.rs
filters.rs
```

Acceptance milestones:

1. Encode one 64x64 CTU, 8-bit 4:2:0, no SAO/deblock.
2. Encode a full image, 8-bit 4:2:0.
3. Add 4:4:4.
4. Add 10-bit.
5. Add SAO/deblock.
6. Compare against the x265 oracle for decode validity, size, and PSNR/SSIM.

## Phase 6: Integrate The Rust Encoder Into BPG

1. Implement `bpg_encode::HevcEncoder` for `bpg-still-hevc`.
2. Add CLI backend selection:

   ```text
   bpg-tools encode input.png -o out.bpg --backend rust
   bpg-tools encode input.png -o out.bpg --backend x265
   ```

3. Keep the x265 backend as an oracle until the Rust backend reaches parity.
4. Once stable, make the x265 backend an optional feature:

   ```toml
   default = ["rust-encoder"]
   oracle-x265 = ["bpg-x265", "bpg-x265-sys"]
   ```

Acceptance: `--backend rust` produces BPG files decoded by `bpg-decode` and
stock `bpgdec`.

## Phase 7: Remove Video Baggage

After parity, explicitly do not port or remove:

- motion estimation
- reference frames
- B/P frames
- lookahead
- scenecut
- ABR/VBV/two-pass rate control
- frame threading
- CLI-style x265 param parser
- HDR/video SEIs

Replace rate control with still-image controls:

- fixed QP
- target-size binary search
- optional per-block QP map
- presets for photo, grain, screenshot, line art, flat illustration, and
  lossless

## Implementation Order

1. Vendor `bpg-hevc-decode` and remove the external HEIC dependency.
2. Add decoder fixture tests.
3. Add the x265 oracle corpus and test harness.
4. Create the `bpg-still-hevc` syntax skeleton.
5. Create `x265-primitives-sys` and `x265-primitives`.
6. Port single-CTU intra encode.
7. Port full-image intra encode.
8. Add filters and bit-depth/chroma variants.
9. Wire the Rust encoder as a `HevcEncoder`.
10. Make the x265 backend an optional oracle.

## Key Risks

- CABAC correctness: requires dense tests and oracle comparison.
- CU/TU recursion: port incrementally by block size.
- Assembly ABI: isolate unsafe code to primitive wrapper crates.
- Bit-depth specialization: prefer separate monomorphized paths or const
  generics over pervasive runtime branching.
- Licensing: x265-derived Rust code and retained assembly inherit x265
  licensing constraints.

## Recommended Next Step

Start with Phase 1 only: create `bpg-hevc-decode`, copy the HEVC module from
`heic-decoder-rs`, remove the external path dependency, and make current decode
tests pass.

That makes the repo self-contained before the larger x265 port begins.
