//! Rust-native still-picture HEVC intra encoder (Phase 3 syntax skeleton).
//!
//! This crate is a line-by-line port of the *syntax* layer of x265's still
//! (single-frame, all-intra) encode path: the CABAC arithmetic engine
//! (`cabac`), the NAL/RBSP writer (`nal`), the VPS/SPS/PPS parameter-set
//! writers (`params`), and the slice segment header writer (`slice`). Motion,
//! lookahead, rate control, B/P frames, and the DPB are explicitly out of
//! scope (see `bpg-rs/FULL_RUST_CODEC_PLAN.md` Phase 3/5).
//!
//! The pixel-domain encoder (CTU/CU/TU recursion, intra prediction,
//! transform/quant, residual CABAC syntax) is Phase 5 and not implemented
//! here; [`StillHevcEncoder`] currently only assembles parameter-set NALs and
//! a minimal slice header.

pub mod analysis_cache;
pub mod backend;
pub mod cabac;
pub mod contexts;
pub mod effort;
pub mod encoder;
pub mod nal;
pub mod params;
pub mod plan;
pub mod preanalysis;
mod primitives;
mod rdoq;
pub mod residual;
pub mod sao;
pub mod slice;
pub mod trace;
pub mod transform;

pub use bpg_image::ChromaFormat;

/// How much effort the mode-decision search spends, as a seven-step ladder
/// (HandBrake/x265-style naming). Higher tiers run more RD trials — i.e. call
/// the (already table-driven, already memoized) CABAC bit estimator more often,
/// which is the dominant cost — in exchange for quality:
///
/// - [`Effort::Fastest`]: sparsest rough-mode shortlist, one RD candidate, no
///   chroma mode search beyond DM, no RDOQ refinement, approximate residual
///   bits, fixed-leaf small TUs, and the most aggressive paper-driven search
///   pruning (CU-split early termination + angular mode exclusion). The
///   cheapest preset.
/// - [`Effort::Fast`]: a second luma RD candidate only on the larger (16x16+)
///   CUs, a sparser rough-mode sweep than `Balanced`, and moderate search
///   pruning — a midpoint between `Fastest` and `Balanced`.
/// - [`Effort::Balanced`] (default): the archival sweet spot — even-stepped
///   rough modes, two luma RD candidates, single-scan RDOQ on small blocks,
///   plus *light* (near-lossless) search pruning (zero-residual CU early
///   termination + near-flat angular exclusion).
/// - [`Effort::Good`]: denser rough modes, more luma/chroma RD candidates, and
///   wider single-scan RDOQ coverage than `Balanced` — between `Balanced` and
///   `Best`.
///
/// `Fast`/`Balanced`/`Good` additionally apply **source-derived preanalysis
/// steering** (see [`crate::preanalysis`]): a cheap per-32x32-cell structure map
/// shifts search budget away from flat/smooth cells (prune angular) and toward
/// edge/text cells (protect angular search; `Good` also spends extra RD
/// candidates on text/colour-critical cells). `Fastest` is left unsteered (its
/// local pruning is already maximal, so steering only adds overhead), as are the
/// `Placebo`/`Reference` uniform/exact references.
/// `Fastest`/`Fast`/`Balanced`/`Good` also trim search breadth as QP rises
/// (narrower rough-mode sweeps, fewer RD candidates, fewer limited-RDOQ passes,
/// and looser fast-tier early termination), because high quantization erases
/// many fine mode/partition distinctions. `Best`/`Placebo`/`Reference` stay
/// exhaustive.
/// - [`Effort::Best`]: a practical max-quality tier: exhaustive rough-mode
///   search, full-RD luma/chroma trials, full chroma candidate set, exact
///   residual-bit pricing, exact final coding, and uniform QP, but with the
///   non-reference luma-only candidate path and single-scan RDOQ. Its output is
///   not bit-identical to [`Effort::Placebo`].
/// - [`Effort::Placebo`]: the full exhaustive reference search (all 35 rough
///   modes, full chroma RD during luma trials, exact greedy per-coefficient
///   RDOQ, no pruning) run with CTU-wavefront-**parallel** analysis. RD is
///   priced against a frozen slice-init CABAC context, which is what allows the
///   parallelism.
/// - [`Effort::Reference`]: identical exhaustive search to [`Effort::Placebo`]
///   but **single-threaded** with exact running-context RD. The slowest,
///   bit-reproducible regression-reference preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    Fastest,
    Fast,
    #[default]
    Balanced,
    Good,
    Best,
    Placebo,
    Reference,
}

/// Sample Adaptive Offset mode (placeholder; Phase 3 emits `sao_enabled_flag
/// = 0` regardless).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaoMode {
    #[default]
    Off,
    On,
}

/// In-loop deblocking filter mode (placeholder; Phase 3 emits
/// `slice_deblocking_filter_disabled_flag = 1` regardless).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblockMode {
    #[default]
    Off,
    On,
}

/// The byte-exact reference tiers ([`Effort::Placebo`]/[`Effort::Reference`]).
/// They
/// are immune to adaptive quantization and source-derived search steering, so
/// their output stays bit-reproducible — the single choke point every
/// quality-trading branch consults, so the "reference tiers never change"
/// invariant holds structurally.
#[inline]
pub fn is_reference_tier(effort: Effort) -> bool {
    matches!(effort, Effort::Placebo | Effort::Reference)
}

/// Whether adaptive quantization (per-CU `cu_qp_delta`) is active for a config.
/// This is the single gate consulted by both the PPS writer
/// ([`params::write_pps`]) and the encoder, so the emitted
/// `cu_qp_delta_enabled_flag` and the per-CU QP plan can never disagree.
///
/// Adaptive QP — the chief size/time lever for lossy still encoding — is on for
/// the speed/size-oriented tiers and off for `Best` plus the byte-exact
/// references ([`is_reference_tier`]). The
/// writer and decoder both gate `cu_qp_delta` on luma CBF plus every coded
/// chroma CBF, including 4:2:2's second stacked Cb/Cr blocks.
///
/// TODO: monochrome (`ChromaFormat::Gray`) is excluded pending a fix. With no
/// chroma CBF to trigger an early `cu_qp_delta`, flat luma defers the delta past
/// the first TU, and the QG QP prediction then diverges from stock/in-tree
/// decoders on non-CTU-aligned pictures (the `gray_non_ctu_aligned` round-trip).
/// 4:2:0/4:2:2/4:4:4 round-trip cleanly (smooth + textured, aligned + not,
/// deblock/SAO, 8/10/12-bit); monochrome falls back to uniform QP until the
/// monochrome QG-prediction path is reconciled with the decoder.
#[inline]
pub fn aq_active(config: &StillHevcConfig) -> bool {
    !matches!(
        config.effort,
        Effort::Best | Effort::Placebo | Effort::Reference
    ) && config.chroma != ChromaFormat::Gray
}

/// Still-picture HEVC encoder configuration.
#[derive(Debug, Clone)]
pub struct StillHevcConfig {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
    pub qp: u8,
    pub effort: Effort,
    pub sao: SaoMode,
    pub deblock: DeblockMode,
}

/// Rust-native still-picture HEVC intra encoder.
///
/// The Phase 3 skeleton only implements [`StillHevcEncoder::syntax_skeleton`],
/// which emits VPS+SPS+PPS+slice-segment-header NALs with no slice data (CTU
/// coding is Phase 5).
pub struct StillHevcEncoder;

impl StillHevcEncoder {
    /// Emit an Annex-B byte stream containing VPS, SPS, PPS, and an IDR
    /// slice NAL whose header is complete but whose slice data is empty.
    ///
    /// This is the Phase 3 acceptance artifact: parameter sets and the slice
    /// header are syntactically valid and round-trip through
    /// `bpg-hevc-decode`'s parsers, but the slice contains no coding-tree
    /// data (Phase 5).
    pub fn syntax_skeleton(config: &StillHevcConfig) -> Vec<u8> {
        let mut out = Vec::new();
        nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
        nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
        nal::write_annexb_nal(&mut out, nal::NalType::Pps, &params::write_pps(config));
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::IdrWRadl,
            &slice::write_slice_segment_header(config),
        );
        out
    }
}
