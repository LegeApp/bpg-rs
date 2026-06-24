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
pub mod batch;
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
pub mod tu_diag;

pub use bpg_image::ChromaFormat;

/// How much effort the mode-decision search spends, as a nine-step ladder
/// (HandBrake/x265-style naming). Higher tiers run more RD trials — i.e. call
/// the (already table-driven, already memoized) CABAC bit estimator more often,
/// which is the dominant cost — in exchange for quality:
///
/// - [`Effort::Floor`]: experimental speed-floor baseline. It deliberately
///   under-searches CTUs by forcing leaf CUs where legal, disabling PartNxN,
///   using one luma candidate, and forcing leaf TUs where legal. This is for
///   timing diagnostics, not quality.
/// - [`Effort::FloorPlus`]: photo-oriented floor-first repair preset. It starts
///   from `Floor`, then selectively repairs failed 64x64 leaves with richer
///   mode/TU search or one terminal 64->32 split.
/// - [`Effort::FloorPlus2`]: FloorPlus with a tiny best-first CTU repair
///   auction and odds-gated mode expansion. It keeps PartNxN disabled and avoids
///   recursive CTU search.
/// - [`Effort::FloorShallow`]: experimental shallow-CU preset. At each 64×64
///   CTU chooses among three candidates by real RD cost: the plain floor leaf,
///   an enhanced 64×64 leaf (same FloorPlus mode/TU budget), and a terminal
///   64→32 split whose four 32×32 children each receive the same enhanced budget
///   (not the bare MPM-only terminal-child budget used by FloorPlus). No
///   recursive CU split below 32×32, no PartNxN. Tests the hypothesis that
///   giving split children proper prediction+TU search recovers the remaining
///   Floor→Best gap without the complexity of FloorPlus2.
/// - [`Effort::SlowPlus`]: experimental Slow successor. It keeps Slow's shallow
///   64→32-first architecture and reinvests budget into denser progressive RMD
///   (coarse_step 3, top 3 regions), one extra luma/chroma RD candidate, and
///   conservative/selective PartNxN islands (adaptive-fidelity, RDOQ on the top
///   PUs) in structured 8×8 leaves. At QP28 this RD-beats `Slow` (smaller *and*
///   higher PSNR) on native 12/50 MP photos. The x265-style "skip the root
///   64×64 leaf at low QP" rule is wired but **off by default** — it regressed
///   size on smooth high-res content (see `slowplus_skip_root_leaf`).
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
    Floor,
    FloorPlus,
    FloorPlus2,
    FloorShallow,
    /// Fuzzy shallow quality mode: progressive RMD, terminal 64→32 split,
    /// final-winner RDOQ, QP-aware search. The experimental quality tier
    /// between the floor family and Best — recovers most of the byte gap
    /// without full recursive CTU search.
    Slow,
    /// Experimental Slow successor: denser progressive RMD, one extra
    /// luma/chroma RD candidate, and conservative/selective PartNxN enabled by
    /// default. RD-beats `Slow` at QP28. (The x265-style no-root-64-leaf rule
    /// is wired but off by default — it regressed size on smooth high-res.)
    SlowPlus,
    Fastest,
    /// QP-adaptive hybrid: uses [`FloorPlus`]'s search at low QP (< 32) and
    /// [`Fastest`]'s search at high QP (>= 32). Intended as the new "Fast"
    /// tier — combines thorough coding at low compression with speed at high
    /// compression.
    FastAdaptive,
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
    if config.chroma == ChromaFormat::Gray {
        return false;
    }
    match config.effort {
        // Byte-reproducible reference tiers never quantize adaptively.
        Effort::Placebo | Effort::Reference => false,
        // The whole ladder is now uniform-QP: each tier is a progressively
        // pruned version of `Best` (same coding tools — PartNxN, SAO, uniform
        // QP — with less search), so quality scales monotonically with effort.
        // The previous per-CU variance AQ on the speed tiers diverged badly from
        // `Best` (≈3 dB MS-SSIM and a scrambled quality ladder; see
        // docs/effort-ladder-uniform-qp-2026-06-17.md), and the project's own
        // measurements found the preanalysis AQ signal RD-neutral-to-negative.
        // `Best` can still opt into the experimental variance AQ via
        // [`best_aq_params`] (`BPG_BEST_AQ`).
        Effort::Best => best_aq_params().is_some(),
        // Speed tiers: opt-in via the config field (default off → uniform QP).
        _ => config.adaptive_qp,
    }
}

/// Experimental variance-AQ configuration for [`Effort::Best`], parsed from the
/// environment. Returns `Some((perceptual, strength))` when enabled:
/// `BPG_BEST_AQ=perceptual` (SSIM-leaning, x265 aq-mode-2 direction) or
/// `BPG_BEST_AQ=psnr` (SSE-leaning, opposite sign; the default for any other
/// non-empty value). `BPG_BEST_AQ_STRENGTH` scales the offset (default 1.0,
/// x265 `--aq-strength`). Unset / `0` / empty ⇒ `None` (Best stays uniform-QP).
pub fn best_aq_params() -> Option<(bool, f32)> {
    let mode = std::env::var("BPG_BEST_AQ").ok()?;
    let mode = mode.trim();
    if mode.is_empty() || mode == "0" {
        return None;
    }
    let perceptual = mode.eq_ignore_ascii_case("perceptual");
    let strength = std::env::var("BPG_BEST_AQ_STRENGTH")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
        .unwrap_or(1.0);
    Some((perceptual, strength))
}

/// Whether HEVC sign-data-hiding is applied for a config. Single source of
/// truth consulted by both the PPS writer ([`params::write_pps`], which emits
/// `sign_data_hiding_enabled_flag`) and the encoder (which makes each coding
/// group's hidden sign parity-consistent), so the flag and the coded levels
/// can never disagree.
///
/// Scoped to [`Effort::Best`] (the x265-parity tier); the byte-reproducible
/// reference tiers ([`Effort::Placebo`]/[`Effort::Reference`]) stay SDH-free so
/// their output remains bit-stable. `BPG_BEST2_SDH=0` reverts for A/B testing.
#[inline]
pub fn sdh_active(config: &StillHevcConfig) -> bool {
    config.effort == Effort::Best
        && std::env::var("BPG_BEST2_SDH")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
}

/// Experimental PPS chroma QP offset for rate-allocation diagnostics. Defaults
/// to x265/libbpg parity (`0`).
#[inline]
pub fn chroma_qp_offset() -> i8 {
    std::env::var("BPG_CHROMA_QP_OFFSET")
        .ok()
        .and_then(|v| v.trim().parse::<i8>().ok())
        .map(|v| v.clamp(-12, 12))
        .unwrap_or(0)
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
    /// Per-CU adaptive quantization (`cu_qp_delta`) for the speed/size-oriented
    /// tiers. **Off by default**: the effort ladder is uniform-QP (each tier a
    /// pruned `Best`), which the project's measurements found beats the per-CU
    /// variance AQ on photos. Opt in for size-at-equal-QP on flat content. No
    /// effect on `Best` (use `BPG_BEST_AQ`) or the reference tiers.
    pub adaptive_qp: bool,
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
        let tiles = params::effective_tile_dims(config);
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::Pps,
            &params::write_pps(config, tiles),
        );
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::IdrWRadl,
            &slice::write_slice_segment_header(config, tiles, &[]),
        );
        out
    }
}
