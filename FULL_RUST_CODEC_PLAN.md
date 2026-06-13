# Full Rust BPG Codec Plan

## Goal

Turn `bpg-rs` into a self-contained Rust BPG encoder/decoder.

Short term: vendor the Rust HEVC decoder currently referenced from
`heic-decoder-rs` and remove the external path dependency.

Long term: replace the current x265 FFI backend with a Rust still-picture HEVC
intra encoder derived from x265. The Rust encoder becomes the production
backend; upstream x265 remains only as an oracle/dev dependency for regression
comparison, quality benchmarking, syntax checks, and phase-gate validation.

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
    bpg-x265             # oracle/dev backend only once Rust backend is wired
    bpg-x265-sys         # oracle/dev backend only once Rust backend is wired
    bpg-still-hevc       # Rust x265-derived still encoder
    x265-primitives
    x265-primitives-sys  # temporary x265 primitive oracle/shim; not production
    bpg-hevc-core        # future shared HEVC math/syntax helpers
    bpg-tools
```

## Roadmap Correction: Correctness First, Then RD Analysis

The Rust still encoder is now a valid intra encoder skeleton with expanding
coverage, not yet an x265-class encoder. SAO, deblocking, and 4:2:2 are the
remaining easy correctness/completeness items, but they are not the center of
compression efficiency. The hard part is the x265-style analysis loop:
SATD-based rough mode pruning, full RD mode decisions, recursive CU/TU split
search, chroma mode decision, RDOQ/sign hiding, and eventually hot-path
allocation removal plus Rust/SIMD primitive replacement.

The corrected order is:

1. Wire `bpg-still-hevc` into `bpg-encode::HevcEncoder` and `bpg-tools` as
   `--backend rust`; make Rust the default once it can emit BPGs that stock
   `bpgdec` and `bpg-decode` accept. Keep `--backend x265` as oracle/dev.
2. Fix the Rust backend source-geometry boundary before adding new chroma
   formats: visible dimensions, coded dimensions, per-plane width/height,
   stride, bit depth, and chroma format must be explicit instead of inferred
   from a single padded image width.
3. Complete Rust-still-encoder 4:2:2 correctness.
4. Add deblocking, using the existing decoder filter path at first.
5. Add SAO syntax with every CTB/component signalling SAO-off, then add real
   SAO RD search as a separate step.
6. Replace the temporary x265 C++ primitive shim with scalar Rust primitives,
   then `std::arch` SIMD, then narrow hand-written assembly only where profiling
   proves it is needed.
7. Port x265-like intra analysis/RD: rough SATD mode decision, CABAC bit
   estimation, full RD mode decision, CU split recursion, TU split recursion,
   chroma mode decision, RDOQ, sign hiding, and scratch-buffer reuse.
8. Only after the Rust backend meets explicit quality/size/speed gates, remove
   upstream x265 from normal builds and keep it only in oracle tooling.

Definition of materially complete:

- Rust backend encodes BPG without x265 in the production path.
- 8-bit and 10-bit 4:2:0, 4:2:2, and 4:4:4 pass exact
  encoder-reconstruction-vs-decoder-output tests, including odd dimensions.
- Deblock and SAO can be enabled and decode exactly.
- Upstream x265 is not required for normal builds.
- Rust output is within the chosen size/quality band against x265 on the oracle
  corpus, initially 5-15% file size at similar quality and tightened later.
- Production primitive dispatch has no C++ x265 dependency.
- Unsupported BPG features fail explicitly.

Current reproducibility issue: `bpg-rs/x265_4.1` is a symlink to an absolute
local path (`/mnt/Samsung980_1TB/isolated-dev/BPG/x265_4.1`). Before claiming
the repo is self-contained, replace this with a real vendored tree, a git
submodule, a documented oracle setup step, or remove upstream x265 from normal
builds.

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

### Phase 3 COMPLETE — `bpg-still-hevc` syntax skeleton

New `crates/bpg-still-hevc`:

- `bpg-bitstream` gained `read_se_golomb`/`write_se_golomb` (signed Exp-Golomb
  `se(v)`, H.265 9.2.2 codeNum mapping 0,1,-1,2,-2 <-> 0,1,2,3,4).
- `nal.rs`: Annex-B NAL writer (`write_annexb_nal`) — start code + 2-byte NAL
  header + emulation-prevention `00 00 03` insertion, generalized from
  `bpg-hevc`'s `make_nal`.
- `params.rs`: `write_vps`/`write_sps`/`write_pps` + a shared
  `write_profile_tier_level` helper (Main profile, level 4.0 ==
  `general_level_idc=120`, `max_sub_layers_minus1=0` only). Field order/
  presence mirrors `bpg-hevc-decode`'s `parse_vps`/`parse_sps`/`parse_pps`
  exactly; constants anchored to a real x265 BPG-still encode
  (`oracle/out/checkerboard__420_8bit_qp24.hevc`, dumped via
  `tests/dump_oracle_params.rs`). Two simplifications vs. that oracle stream
  (chosen to keep the slice-header writer simple):
  `sample_adaptive_offset_enabled_flag=false` and
  `entropy_coding_sync_enabled_flag=false`. The SPS `vui_parameters()` is
  written through to its end (not just the fields `parse_sps` reads), so a
  conformant decoder (stock `bpgdec`/libde265, used for Phase 5 verification)
  doesn't misread `rbsp_trailing_bits()` as further VUI flags.
- `slice.rs`: `write_slice_segment_header` for a single first-slice IDR
  I-slice — `slice_type=I`, `slice_qp_delta` (derived from `config.qp`),
  `slice_loop_filter_across_slices_enabled_flag`, then `byte_alignment()`.
  Field order mirrors `SliceHeader::parse`.
- `cabac.rs`: full CABAC encoder shell ported from x265's `Entropy` class
  (`entropy.cpp`) — `ContextModel::new` (`sbacInit`), `encode_bin`,
  `encode_bin_ep`/`encode_bins_ep`/`encode_bin_trm`, `finish`, using the same
  `g_lpsTable`/state-transition tables as the decoder's CABAC. Not yet wired
  into slice-data emission (no slice data is written in the skeleton).
  `tests/cabac_roundtrip.rs` round-trips a sequence of regular, bypass, and
  terminate bins through `bpg-hevc-decode`'s `CabacDecoder`, checking
  bin-for-bin output and context (state, mps) evolution match — the
  arithmetic-coder/state-table port is validated independently of residual
  syntax.
- Public API (`lib.rs`): `StillHevcEncoder::syntax_skeleton`,
  `StillHevcConfig` (incl. `width`/`height`, added beyond the literal struct
  in this plan since SPS needs picture dimensions), `Effort`, `SaoMode`,
  `DeblockMode`.
- `tests/skeleton_roundtrip.rs`: feeds `syntax_skeleton`'s output through
  `bpg-hevc-decode`'s `parse_vps`/`parse_sps`/`parse_pps`/`SliceHeader::parse`
  — all parse successfully, `slice_qp_y` matches `config.qp`, and
  `data_offset` consumes the whole (empty) slice payload, i.e. the header is
  well-formed and rejected only for lack of slice data, not malformed syntax.
  A second test, `syntax_skeleton_decode_fails_on_missing_slice_data`, runs
  the full `bpg_hevc_decode::hevc::decode` on the skeleton and confirms it
  returns `HevcError::CabacError("...short...")` from `SliceContext::new`
  (slice-data CABAC init needs >= 2 bytes and gets zero) — i.e. the failure
  is in slice-data decode, not parameter-set or slice-header parsing,
  satisfying the Phase 3 acceptance criterion in full.

Motion, lookahead, rate control, B/P frames, and DPB are not implemented (no
stubs needed — the skeleton is IDR-I-frame-only by construction).

**Next:** Phase 5 (port the still intra encoder / CABAC slice-data emission),
per the implementation order below. (Phase 4, the assembly primitive boundary,
is now complete across 8-bit and 10-bit — see its progress section.)

### Phase 5 progress (in flight)

Foundations for the still intra encoder, each validated bit-exactly against
`bpg-hevc-decode` (the decoder is the oracle — the encoder's job is to emit a
stream the decoder reconstructs to the same samples the encoder reconstructed):

- **`contexts.rs`**: the full 170-entry CABAC context store (indices +
  `INIT_VALUES`) mirrored from the decoder's `cabac::context`/`INIT_VALUES`,
  initialized per H.265 9.3.2.2 from the slice QP.
- **`residual.rs`**: `encode_residual` — the complete `residual_coding()`
  CABAC writer, the exact inverse of `decode_residual` (last-sig prefix/suffix,
  coded_sub_block_flag, sig_coeff_flag context derivation, greater1/greater2
  context evolution, sign bits + sign data hiding, Golomb-Rice/EGk
  coeff_abs_level_remaining with adaptive Rice). `tests/residual_roundtrip.rs`
  round-trips coefficient blocks (4x4..32x32, all scan orders, luma+chroma,
  SDH) through the decoder's `decode_residual` — all exact.
- **`transform.rs`**: forward DCT/DST + x265 quantizer (forward), and a
  dequant + inverse DCT/DST reconstruction path that is **bit-identical** to
  the decoder's `transform::dequantize` + `inverse_transform`
  (`tests/transform_recon.rs`: equal across 4x4..32x32, every QP, and DST-4x4).
  This guarantees encoder-reconstructed neighbours match the decoder's.

- **`encoder.rs`**: the slice-data coding-tree writer + reconstruction engine.
  Walks the coding tree exactly as the decoder's `ctu` parses it
  (`split_cu_flag`, `coding_unit` with MPM-based intra-mode signalling,
  `transform_tree` forced 64->32 split + cbf hierarchy, `cbf_luma`,
  `residual_coding`, per-CTU `end_of_slice`), reconstructing each TB into a
  `DecodedFrame` via the decoder's `predict_intra` + the verified inverse
  path. Assembles VPS/SPS/PPS + slice into an Annex-B IDR access unit.

Decoder visibility-only changes (no logic edits) to support the above:
`hevc::{intra, residual, transform}` made `pub`; `bpg-hevc-decode` is now a
normal (not dev) dependency of `bpg-still-hevc` for the reconstruction engine.

#### Phase 5 milestones 1 & 2 COMPLETE — encode one CTU and a full image (8-bit 4:2:0)

`tests/encode_roundtrip.rs`: the encoder emits an IDR access unit that
`bpg_hevc_decode::hevc::decode` decodes to **exactly** the encoder's
reconstruction (luma + Cb + Cr planes bit-identical), across QPs 18/27/32/40
and a 192x128 (3x2-CTU) image. Luma PSNR vs source is 56 dB (qp18) down to
41 dB (qp40); the multi-CTU image is 38.6 dB at qp30. PPS now disables the
in-loop deblocking filter (control-present + disabled), so the slice header
omits `slice_loop_filter_across_slices_enabled_flag`, and SDH is off
(`sign_data_hiding_enabled_flag = 0`) — both matching milestone-1 scope.

Milestone-1/2 simplifications still open: fixed Planar luma + DM chroma (no
real mode decision yet), one 64x64 CU per CTB (no `split_cu_flag` recursion),
and 64-aligned dimensions only (no boundary CU splits).

#### Phase 5 milestone 4 COMPLETE — 10-bit

`transform.rs` dispatches forward/inverse DCT/DST to the `x265-primitives`
8-bit or `bitdepth10` primitives by `bit_depth`; the encoder threads the
bit-depth QP offset (dequant QP = SliceQpY + 6*(bd-8), CABAC contexts still
init from SliceQpY) and builds a 10-bit `DecodedFrame`.
`tests/encode_roundtrip.rs::ten_bit_round_trip` round-trips 10-bit pictures
through the decoder exactly (PSNR 56.7 dB @ qp22 .. 45 dB @ qp38).

#### Phase 5 milestones 3 & 5 COMPLETE for the Phase-6 integration scope — 4:4:4, luma mode decision, and boundary CU splits

The encoder now supports 4:4:4 (`ChromaArrayType == 3`) in addition to 4:2:0:
chroma transform blocks mirror luma position/size, chroma cbf syntax is coded
for 4x4 leaves as required, and `tests/encode_roundtrip.rs::yuv444_round_trip`
round-trips both 8-bit and 10-bit 4:4:4 pictures through the decoder with
bit-exact reconstruction.

The encoder also picks each CU's luma mode by lowest prediction SSE over all 35
modes (evaluated on the CU's top-left TB against the source), instead of fixed
Planar. Decode stays exact, quality improves and size drops (192x128 8-bit:
38.6 dB/1263 B -> 39.1 dB/1039 B; 128x128 10-bit: 40.5 dB/643 B -> 41.4 dB/
476 B), and the angular intra + per-mode-scan-order paths are exercised
end-to-end through `encode_residual` and verified against the decoder.

Boundary CU splitting is now implemented for non-64-aligned pictures. The
encoder mirrors H.265 `coding_quadtree()`/the decoder's decision tree:
`split_cu_flag` is coded only for fully-inside CUs, CUs that straddle the
picture boundary are force-split down to the 8x8 minimum coding-block size,
and the split-flag context increment is derived from left/above neighbour CU
depths. Internally, reconstruction uses min-CU-aligned coded planes while the
SPS keeps display dimensions; source reads at the padded edge are replicated.
`tests/encode_roundtrip.rs::non_ctu_aligned_boundary_splits` round-trips a
131x97 4:2:0 picture exactly against the encoder reconstruction.

SAO and deblocking remain intentionally signalled off (`sao_enabled_flag = 0`,
PPS deblocking filter disabled). This is conformant, but the corrected roadmap
now treats the easy filters as near-term work before the hard analysis/RD
phase. 4:2:2 in the Rust still encoder is likewise near-term correctness work;
the production x265-backed BPG path already supports 4:2:2, but the Rust
backend must support it before it can be called production-complete.

#### Phase 5 verification / standard cross-check

The important Phase 5 syntax and reconstruction pieces have been checked
against the local H.265 markdown:

- Table 6-1 chroma geometry: 4:2:0 uses `SubWidthC=2, SubHeightC=2`; 4:4:4
  uses `SubWidthC=1, SubHeightC=1`; odd-size 4:2:0 fixtures use ceil chroma
  dimensions.
- Clause 7.3.8 coding-tree syntax: boundary CUs are force-split without
  coding `split_cu_flag`; fully-inside split flags use neighbour-depth context
  derivation.
- Clause 8.4 intra prediction: luma mode signalling uses MPM derivation, and
  chroma uses DM in the supported path.
- Clause 8.6 transform/reconstruction: forward quantization plus inverse
  dequant/transform is tested bit-exactly against the decoder reconstruction.
- Clause 9.3 CABAC: context initialization uses `SliceQpY` (not the bit-depth
  adjusted transform QP), residual syntax round-trips through the decoder, and
  terminate bins mark CTU slice completion.

Verification run: `cargo test --workspace` passes; `cargo run -p bpg-oracle --
check` passes all 40 x265-backed oracle encodes. The oracle command validates
the existing x265 backend corpus; Rust-still-encoder quality parity against
x265 remains Phase 6/7 quality work after integration.

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

### Phase 4 scope decisions

Phase 4 intentionally starts with the intra still-image encode path, not the
full x265 `EncoderPrimitives` table. The first vertical slice is:

- forward/inverse DCT for 4x4, 8x8, 16x16, and 32x32 transform blocks
- SAD/SATD for square luma blocks used by intra mode decision cost
- DC and Planar intra prediction

The follow-up intra families are quant/dequant, residual/reconstruction helpers
(`sub_ps`, `add_ps`), pixel copy/fill helpers, variance/SSE where the RDO path
needs them, then optional SAO/deblock kernels.

Motion-estimation and motion-compensation primitives are out of Phase 4's still
encoder scope: `PU` inter search helpers (`sad_x3`/`sad_x4`, `ads`),
interpolation filters (`luma_hpp`/`hps`/`vpp`), `pixelavg`, and `addAvg` are
video inter-path machinery and are deliberately omitted unless a later oracle
test proves an intra still-image path needs one.

Bit-depth scope is staged. The first implementation and tests are 8-bit only
(`HIGH_BIT_DEPTH=0`, `X265_DEPTH=8`) to match Phase 5 milestone 1. A 10-bit
build must be added before Phase 4 is considered complete across supported BPG
bit depths; pixel-typed primitives need the same dual-compile pattern as the
existing x265 backend.

### Phase 4 COMPLETE — assembly primitive boundary (8-bit + 10-bit)

The Phase 4 intra primitive slice is linked and tested end-to-end through
`x265-primitives-sys` and `x265-primitives` across **both** supported BPG bit
depths (8-bit and 10-bit; 12-bit remains deferred — its x265 lib is not built):

- `x265-primitives-sys` builds an independent scalar x265 primitive archive
  from `common/constants.cpp`, `common/dct.cpp`, `common/pixel.cpp`, and
  `common/intrapred.cpp`, plus the local C ABI shim. It does not link the full
  `bpg-x265-sys` multilib. The same source subset is dual-compiled under
  `HIGH_BIT_DEPTH=1`, `X265_DEPTH=10`, `X265_NS=x265_10bit` for the 10-bit
  surface, following x265's own multilib namespace split.
- The C ABI bodies live in one shared, prefix-parameterized header
  (`shim/wrapper_impl.h`); `shim/wrapper.cpp` includes it with `BPG_PREFIX
  bpgprim_` (8-bit), `shim/wrapper10.cpp` with `BPG_PREFIX bpgprim10_`
  (10-bit). The two depths therefore emit the *identical* function set with no
  per-function duplication and cannot drift. `bpgprim10_*` now mirrors the full
  `bpgprim_*` surface (previously only a narrow DCT4x4/SAD/SATD/DC/add_ps
  probe).
- The safe wrapper exposes, **for each of 8-bit and 10-bit**,
  4x4/8x8/16x16/32x32 DCT/IDCT, 4x4 DST/IDST, square luma SAD/SATD/SA8D,
  DC/Planar/angular intra prediction plus intra reference filtering, residual
  subtraction (`sub_ps`), reconstruction (`add_ps`), block copy helpers
  (`copy_pp`, `copy_ps`, `copy_sp`, `copy_ss`), shift-copy helpers, transpose,
  block fill, variance/SSE/SSD, scalar quant/nquant, normal/scaling dequant,
  count-nonzero, and copy-count helpers. The 8-bit surface is the crate-root
  `dct`/`pixel`/`intra`/`recon`/`quant` modules (`pixel == u8`); the 10-bit
  surface is the identically-shaped `bitdepth10::{dct,pixel,intra,recon,quant}`
  submodules (`pixel == u16`, reconstruction/`copy_sp` clip ceiling 1023). Both
  depths are generated from the same crate-level macros, so they stay in lock
  step.
- Tests include absolute fixed-vector checks for DCT DC-only output, SAD/SATD
  constant differences, uniform intra prediction, residual/reconstruction,
  clipping, copy helpers, SSE, variance/SA8D/SSD, shift-copy/transpose,
  quant/dequant, count-nonzero, and copy-count, plus DCT/IDCT round trips — at
  8-bit, and a mirror set across the 10-bit surface (DCT DC scaling,
  near-lossless 32x32 round trip, intra prediction, residual/reconstruction
  with 10-bit clipping, copy/distortion, and quant/count helpers). Phase 4's
  acceptance criterion (primitive tests match x265 for fixed vectors across
  supported bit depths) is met. 21 tests across the two crates.

Deferred (not blocking Phase 4): any additional RDO helper primitives Phase 5
proves it needs (added on demand once the encoder exists), optional SAO/deblock
kernels, and the 12-bit surface (gated on building a `MAIN12` x265 lib, which
the project defers to much later alongside lossless).

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
- deblock/SAO filter syntax and reconstruction

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
5. Add non-64-aligned boundary CU splitting.
6. Add 4:2:2, including the two vertically-stacked chroma transform blocks
   required for 4:2:2 and odd-dimension tests.
7. Add deblocking with config-driven PPS/slice syntax and encoder-side
   post-reconstruction filtering.
8. Add SAO syntax with all CTBs/components off, then real SAO RD search.
9. Compare against the x265 oracle for decode validity, size, PSNR/SSIM, and
   speed.

Phase 5 is correctness-first. It may produce valid but inefficient streams.
Do not treat Phase 5 as x265-quality parity; that belongs to the later analysis
phase.

## Phase 6: Integrate The Rust Encoder Into BPG

1. Implement `bpg_encode::HevcEncoder` for `bpg-still-hevc`.
2. Make source geometry explicit at the backend boundary:

   ```rust
   pub struct PlaneRef<'a> {
       pub data: &'a [u16],
       pub width: usize,
       pub height: usize,
       pub stride: usize,
   }

   pub struct Source<'a> {
       pub width: usize,
       pub height: usize,
       pub coded_width: usize,
       pub coded_height: usize,
       pub bit_depth: u8,
       pub chroma_format: ChromaFormat,
       pub y: PlaneRef<'a>,
       pub cb: Option<PlaneRef<'a>>,
       pub cr: Option<PlaneRef<'a>>,
   }
   ```

   BPG stores visible dimensions while HEVC coding may use padded/coded planes;
   do not infer all plane geometry from a single padded luma width.
2. Add CLI backend selection:

   ```text
   bpg-tools encode input.png -o out.bpg --backend rust
   bpg-tools encode input.png -o out.bpg --backend x265
   ```

3. Make `--backend rust` the default once it produces BPG files accepted by
   `bpg-decode` and stock `bpgdec`.
4. Keep the x265 backend as an oracle/dev feature only:

   ```toml
   default = ["rust-encoder"]
   oracle-x265 = ["bpg-x265", "bpg-x265-sys"]
   ```

Acceptance: `--backend rust` produces BPG files decoded by `bpg-decode` and
stock `bpgdec`.

## Phase 7: Easy Filters And Remaining Format Correctness

Implement the remaining syntax/format features before deep RD analysis:

1. Rust-still 4:2:2:
   - `chroma_format_idc == 2`
   - chroma plane geometry = half width, full height
   - two vertically stacked chroma transform blocks where required
   - 8-bit/10-bit round-trip tests, odd sizes, and BPG container encode/decode
2. Deblocking:
   - parameterize PPS and slice loop-filter syntax
   - record TU boundaries and QP maps during encode
   - apply the existing decoder deblock implementation to encoder
     reconstruction initially
   - exact decoder-vs-encoder reconstruction tests
3. SAO stage 1:
   - set `sample_adaptive_offset_enabled_flag = 1`
   - write slice SAO flags
   - emit per-CTB SAO type index 0 for all components
   - confirm output is identical to SAO-off
4. SAO stage 2:
   - BO and EO candidate search per CTB/component
   - RD cost = distortion after SAO + lambda * estimated SAO bits
   - merge-left/up decisions after independent candidates work
   - allow RD to choose off freely for line art/text-like content

## Phase 8: Intra Analysis And RD Parity

This is the main compression-efficiency phase. Filters and 4:2:2 do not make
the Rust encoder x265-class by themselves.

Port or reimplement the x265 intra analysis path:

- SATD/Hadamard rough intra mode decision over Planar, DC, angular modes, and
  MPMs.
- CABAC bit estimation for analysis, separate from final writing.
- Full RD mode decision:

  ```text
  rd_cost = distortion + lambda * estimated_bits
  ```

- Recursive CU split search: evaluate leaf vs four children down to min CU.
- Recursive TU split search: evaluate unsplit vs split transform tree.
- Chroma mode decision over DM, Planar, DC, horizontal/vertical/angular
  candidates where legal.
- RDOQ, last-significant-position decisions, sign hiding, and later transform
  skip if needed.
- Hot-path allocation removal: per-CTU scratch buffers, fixed-size stack arrays
  where practical, and no allocation inside block loops.

Acceptance: Rust output is decode-valid on the full corpus, exact
decoder-vs-encoder reconstruction holds, and size/quality is within the
current parity gate against x265.

## Phase 9: Primitive Replacement

The current `x265-primitives-sys` compiles upstream scalar C++ primitive files
with assembly off. That is acceptable as an oracle/bring-up shim, but it
conflicts with the production goal.

Replacement path:

1. Define a Rust primitive trait/table for the subset the encoder needs.
2. Implement scalar Rust primitives first.
3. Add `std::arch` SIMD for x86_64/aarch64 hot paths.
4. Add narrow handwritten assembly only where profiling proves `std::arch` is
   insufficient.
5. Keep upstream x265 primitive wrappers only in oracle/benchmark builds.

Prioritize SAD/SATD, intra prediction, DCT/IDCT/DST, quant/dequant,
residual/reconstruction, copy/fill, pixel SSE/SSD, then SAO/deblock.

## Phase 10: Remove Video Baggage

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
8. Wire the Rust encoder as a `HevcEncoder` and add `--backend rust`.
9. Fix explicit source/coded plane geometry at the backend boundary.
10. Add Rust-still 4:2:2.
11. Add deblocking.
12. Add SAO syntax-off, then SAO RD search.
13. Port x265-style intra analysis/RD decisions.
14. Replace x265 C++ primitive shims with Rust scalar/SIMD primitives.
15. Make x265 optional oracle/dev only, then remove it from normal builds.

## Key Risks

- CABAC correctness: requires dense tests and oracle comparison.
- CU/TU recursion: port incrementally by block size.
- Assembly ABI: isolate unsafe code to primitive wrapper crates.
- Bit-depth specialization: prefer separate monomorphized paths or const
  generics over pervasive runtime branching.
- Analysis quality: valid syntax is not enough; x265-class compression depends
  on SATD/RD mode decisions, CU/TU split search, chroma decisions, and RDOQ.
- Primitive dependency drift: upstream C++ primitive shims are useful oracles
  but must not remain in the production path.
- Reproducibility: `x265_4.1` must not remain an absolute symlink if x265 is
  required by normal builds.
- Licensing: x265-derived Rust code and retained assembly inherit x265
  licensing constraints.

## Recommended Next Step

Current next step: integrate the Rust still encoder as a real `HevcEncoder`
backend and make the source geometry explicit. Then do the easy filters and
format coverage in order: Rust 4:2:2, deblock, SAO syntax-off, SAO RD search.
After that, start the hard intra-analysis/RD phase.
Here is the consolidated, non-redundant project plan for `bpg-rs`. It merges the historical context, architectural decisions, completed work, and remaining roadmap into a single source of truth.

---

# `bpg-rs`: Rust BPG Codec Master Plan

## 1. Project Vision & Goal
Turn `bpg-rs` into a self-contained, high-performance, Rust-native BPG image encoder and decoder. 

**The core strategy: Extract an x265-derived *still-picture intra encoder*, and delete the video baggage.** 
We are not porting all of x265. We are building a dedicated BPG/HEVC still-image intra encoder that retains x265's proven RDO (Rate-Distortion Optimization), CABAC, and primitive logic, while discarding motion estimation, P/B frames, rate control over time, and threading complexities irrelevant to still images.

**Philosophy:**
*   **Correctness First:** Achieve bit-exact decode validity and syntax correctness before optimizing for compression efficiency.
*   **Archival & Source-Aware, not Metric-Chasing:** Avoid destructive preprocessing just to boost PSNR/SSIM. Use a conservative "tuning" system based on image character (e.g., film grain, line art, photo) rather than blind parameter searches.
*   **Upstream x265 as an Oracle:** Stock x265 is used strictly as an oracle/dev dependency to validate Rust outputs (syntax, file size, quality) via regression testing. Once the Rust backend achieves parity, x265 will be removed from standard builds.

---

## 2. Current State: What is Done
A massive amount of the foundation and intermediate integration is complete. 

### Core Infrastructure & BPG Container
*   **`bpg-bitstream`:** Fully implemented bit-level I/O (`ue7`, `ue(v)`, signed Exp-Golomb).
*   **`bpg-image`:** Color conversion (BT.601/709/2020), chroma subsampling (4:2:0, 4:2:2, 4:4:4), and padding implemented. 8-bit and 10-bit pipelines are fully supported.
*   **`bpg-format`:** BPG container header read/write complete.
*   **`bpg-hevc`:** NAL extraction and HEVC/SPS stream rewriting/modification for BPG payload rules is done.
*   **CLI & Orchestration:** `bpg-tools` (CLI) and `bpg-encode` (orchestrator) are fully functional.

### Decoder Integration
*   **Vendored Decoder (`bpg-hevc-decode`):** The external HEIC HEVC decoder was successfully vendored. External path dependencies are gone.
*   **Format Support:** Decodes 8-bit and 10-bit across 4:2:0, 4:2:2, and 4:4:4 chroma formats. Odd-dimension geometry issues are fixed.

### The x265 Oracle & Testing
*   **`bpg-x265-sys` / `bpg-x265`:** C/C++ FFI wrappers are built. Can dynamically link and dual-compile 8-bit and 10-bit x265 static libraries via `build.rs`.
*   **Oracle Harness (`bpg-oracle`):** Deterministic test corpus generator in place. Automatically encodes, decodes, and records metrics (PSNR, file size, syntax validity) against stock x265.

### Rust Still Encoder Scaffolding (`bpg-still-hevc`)
*   **Syntax Skeleton:** NAL NAL, VPS/SPS/PPS, slice headers, and CABAC context initialization are ported and generate decoder-valid bitstreams.
*   **Assembly Primitive Boundary (`x265-primitives`):** Safe Rust wrappers over x265 C++ ASM kernels for both 8-bit and 10-bit (DCT/IDCT, SATD, SAD, intra prediction, pixel copy, quant/dequant).
*   **Intra Foundations:** `contexts`, `residual`, `transform`, and slice-data coding tree writers are implemented. 
*   **Milestones Met:** The Rust encoder can currently encode single and multi-CTU images (8-bit and 10-bit, 4:2:0 and 4:4:4) with luma mode decision and boundary CU splits. **Reconstruction is bit-exact against the decoder.**

---

## 3. Target Architecture & Crate Layout
```text
bpg-rs/
  crates/
    bpg-bitstream/       # Varints and bit-level I/O
    bpg-image/           # RGB <-> YCbCr, subsampling, geometry
    bpg-format/          # BPG container header parsing/writing
    bpg-hevc/            # BPG payload to Annex-B conversions
    bpg-hevc-decode/     # Vendored, pure-Rust HEVC decoder
    bpg-decode/          # High-level BPG decode API
    bpg-encode/          # Orchestration and HevcEncoder trait
    bpg-still-hevc/      # PRODUCTION: Pure Rust x265-derived intra encoder
    x265-primitives/     # Safe Rust primitive dispatch
    x265-primitives-sys/ # Temporary ASM shim for primitive kernels
    bpg-analyze/         # FUTURE: Image feature extraction
    bpg-tune/            # FUTURE: Source-aware parameter resolution
    bpg-tools/           # CLI binary
    
    # DEV / ORACLE ONLY:
    bpg-x265/            # FFI binding to upstream x265
    bpg-x265-sys/        # Upstream x265 CMake build
    bpg-oracle/          # Regression testing and benchmarking
```

---

## 4. Engineering Rules: What to Keep vs. Drop from x265

### KEEP (Port to Rust)
*   **HEVC Coding Machinery:** CTU/CU/TU structures, recursive partitioning, Intra prediction, DCT/DST, Quantization, RDO/RDOQ, and CABAC.
*   **Assembly Primitives:** The core speed of x265. Kept behind a clean Rust FFI, to be gradually replaced by `std::arch` SIMD later.
*   **10-bit Path:** Crucial for high-quality gradients.

### DROP (Do Not Port)
*   **Inter-Prediction & Motion:** Motion estimation, references, DPB, B/P frames, temporal MVP, weighted prediction.
*   **Video Rate Control:** ABR, VBV, 2-pass, Cutree, Lookahead, Scenecut. (Replaced with CQP and Target-Size search).
*   **Extraneous Systems:** Frame threading, video/HDR SEIs, dynamic HDR10, CLI parser, VMAF/CSV logging. 
*   *Note: Alpha channel and animation are officially dropped from the bpg-rs roadmap to focus on core archival photography.*

---

## 5. The Roadmap: What is Left to Do

The project is currently transitioning out of the foundational syntax/scaffolding phase and entering the integration and compression-efficiency phase.

### Phase A: Rust Encoder Integration & Correctness
1. **Wire the Backend:** Implement `HevcEncoder` for `bpg-still-hevc`. Make `--backend rust` the default CLI choice once it successfully emits standard BPG files. Keep `--backend x265` as an oracle flag.
2. **Explicit Source Geometry:** Fix the backend boundary so visible dimensions, coded dimensions, per-plane width/height, stride, and chroma format are explicitly passed, rather than inferred from a padded luma width.
3. **Rust 4:2:2 Support:** Port 4:2:2 vertical-stack transform geometry to the Rust encoder (already works in the Oracle/Decoder, needs to be written into the Rust encoder).
4. **Deblocking:** Add deblocking, utilizing the existing decoder filter path applied to encoder reconstruction.
5. **SAO (Sample Adaptive Offset):** 
   * *Stage 1:* Add SAO syntax with every CTB signalling SAO-off.
   * *Stage 2:* Add actual SAO RD (Rate-Distortion) search for Edge/Band offsets.

### Phase B: Intra Analysis & RD Parity (The Hard Part)
*Syntax correctness does not equal compression efficiency. This phase brings the Rust encoder to x265-level quality/size ratios.*
1. **Rough Mode Decision:** Implement SATD/Hadamard rough intra mode decision over Planar, DC, angular modes, and MPMs.
2. **Bit Estimation:** Port CABAC bit estimation for RD analysis (separate from final writing).
3. **Full RD Mode Decision:** Calculate `rd_cost = distortion + lambda * estimated_bits`.
4. **Recursion Searches:** Implement recursive CU split search (evaluate leaf vs. 4 children) and TU split search.
5. **Chroma & Residuals:** Implement Chroma mode decision, RDOQ (Rate-Distortion Optimized Quantization), and sign data hiding.
6. **Allocation Purge:** Remove per-block `Vec` allocations. Use hot-path scratch buffers and fixed stack arrays.

### Phase C: Primitive Replacement & Decoupling
1. Define a pure Rust primitive trait/table.
2. Write scalar Rust fallbacks for all needed primitives (DCT, SATD, Intra pred, etc.).
3. Implement `std::arch` SIMD for x86_64/aarch64 hot paths.
4. Keep the C++ `x265-primitives-sys` shim *only* for oracle comparisons. Make it completely optional.

### Phase D: The Tuning System (Replacing Rate Control)
*Replaces traditional x265 CLI flags with an archival-first policy system.*
1. **`bpg-analyze`:** Build feature extraction (edge density, noise/grain estimates, flat-region fractions).
2. **`bpg-tune`:** Map features to a `Tune` (e.g., `Neutral`, `Photo`, `FilmGrain`, `Slide`, `LineArt`, `Screenshot`).
3. **`ArchivalPolicy`:** Enforce hard rules (e.g., "never strip ICC", "no destructive denoising", "allow 4:2:0 subsampling").

### Phase E: Final Cleanup
1. **Sever Upstream x265:** Remove the `bpg-x265` and `bpg-x265-sys` crates from standard builds. They become standalone dev-tools.
2. Resolve any remaining absolute symlinks/paths to x265 source directories to ensure fully reproducible, clean public builds.

---

## 6. Deferred to "Much Later"
*Features that the architecture supports adding eventually, but are zero-priority for 1.0:*
*   **12-bit depth:** (Requires compiling a `MAIN12` x265 lib and adding `u16` paths).
*   **Lossless mode:** (`bLossless=1` bypasses transforms/quantization).
*   **RGB / YCgCo color spaces.**
*   **MPEG2 Chroma Siting** (`c_h_phase==0`).
