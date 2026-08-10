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

/// How much search work the still encoder spends.
///
/// The production model is intentionally small:
///
/// - [`Effort::Fast`]: same pipeline shape as Slow, with cheaper search budgets.
/// - [`Effort::Slow`]: the default archival-quality preset.
/// - [`Effort::Placebo`]: expensive experimental/perceptual playground for
///   tools that prove they beat Slow.
///
/// Older CLI names are accepted by the tools as aliases, but the crate API only
/// exposes these three budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    Fast,
    #[default]
    Slow,
    Placebo,
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

/// Whether an effort is the inert high-quality comparison tier.
#[inline]
pub fn is_reference_tier(effort: Effort) -> bool {
    matches!(effort, Effort::Placebo)
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
    resolve_aq_mode(config) != AqMode::Off || preanalysis::external_aq_map_requested()
}

/// Adaptive-quantization map selection. The per-QG QP offset is derived from one
/// of these strategies (see [`crate::preanalysis`] for the offset math). `Off` is
/// uniform QP — the default for every tier; AQ is an explicit opt-in. The two
/// `*Probe` variants are experiment/control modes, not user-quality features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AqMode {
    /// Uniform QP (no `cu_qp_delta`). Default.
    #[default]
    Off,
    /// Legacy one-directional importance shrink ([`preanalysis::aq_qp_offset`]):
    /// always raises QP, most on flat/noisy regions. The old `adaptive_qp = true`
    /// behaviour, kept as a compatibility mode.
    LegacyShrink,
    /// Bidirectional luma-variance perceptual AQ (x265 aq-mode-2 direction):
    /// high-variance up, flat down. MS-SSIM-leaning.
    Perceptual,
    /// Perceptual AQ with chroma-aware activity (Prangnell 2017 luma+chroma).
    PerceptualChroma,
    /// Port of x265 aq-mode 2 (`AUTO_VARIANCE`): per-QG offset
    /// `strength·avg_adj·(qp_adj0 − avg_adj_final)` on the AC-energy power law
    /// `qp_adj0 = (energy + 1)^0.1`, self-normalized by the picture's energy
    /// statistics ([`preanalysis::aq_qp_offset_autovar`]).
    AutoVariance,
    /// Port of x265 aq-mode 3 (`AUTO_VARIANCE_BIASED`): [`Self::AutoVariance`]
    /// plus a dark/flat bias `strength·(1 − 11/qp_adj0²)` that pushes
    /// low-energy QGs further negative (extra bits on flat/dark regions).
    AutoVarianceBiased,
    /// Experiment control: PSNR/SSE-leaning bidirectional AQ (opposite sign of
    /// `Perceptual`). Not RD-efficient at equal bytes — diagnostic only.
    PsnrProbe,
    /// Experiment control: rank-based positive-only shrinker (effective higher
    /// QP). Not a quality feature — diagnostic/size-bias only.
    PositiveProbe,
    /// Two-pass measured AQ: pass-1 encodes uniformly and measures per-QG coded
    /// complexity (an actual CTU-search outcome); pass-2 applies a perceptual
    /// QP redistribution derived from it. The only AQ that reliably works (source
    /// heuristics don't predict where AQ pays — see project history). Opt-in for
    /// Slow and Placebo. Driven by [`crate::encoder::encode_two_pass`].
    TwoPassMeasured,
}

impl AqMode {
    /// The recommended opt-in AQ mode. It measures coded complexity in a
    /// uniform first pass, then applies AQ only when its candidate gate finds a
    /// perceptual rate-distortion win. `Default` deliberately remains `Off` so
    /// applications must opt into the extra encode work explicitly.
    #[inline]
    pub const fn recommended() -> Self {
        Self::TwoPassMeasured
    }

    /// Whether this mode requires the two-pass encode driver
    /// ([`crate::encoder::encode_two_pass`]) rather than a single in-tree pass.
    pub fn is_two_pass(self) -> bool {
        matches!(self, AqMode::TwoPassMeasured)
    }
}

/// Tunables for the recommended measured two-pass AQ mode. Keeping this named
/// API next to [`AqMode::recommended`] lets applications opt in without
/// duplicating the CLI's tested strength and clamp.
#[inline]
pub const fn recommended_aq_preset() -> (AqMode, f32, u8) {
    (AqMode::recommended(), 1.00, 4)
}

/// Resolve a named adaptive-quantization preset to `(mode, strength, clamp)` —
/// the single source of truth shared by the CLI `--aq` flag and external library
/// callers (so the API exposes exactly the same options as the command line).
/// Returns `None` for an unknown name. Names use hyphens, e.g.
/// `perceptual-chroma-mild`. `two-pass` is the recommended measured mode.
pub fn aq_preset(name: &str) -> Option<(AqMode, f32, u8)> {
    Some(match name.trim().to_ascii_lowercase().as_str() {
        "off" => (AqMode::Off, 0.0, 2),
        "legacy-shrink" | "legacy" => (AqMode::LegacyShrink, 0.0, 8),
        "perceptual-mild" => (AqMode::Perceptual, 0.20, 2),
        "perceptual" => (AqMode::Perceptual, 0.35, 2),
        "perceptual-chroma-mild" => (AqMode::PerceptualChroma, 0.20, 2),
        "perceptual-chroma" => (AqMode::PerceptualChroma, 0.35, 2),
        // x265's native aq-strength scale is ~1.0; clamp 6 covers the offsets
        // aq-strength 1.0 actually produces without truncating the tails.
        "auto-variance" | "autovar" => (AqMode::AutoVariance, 1.0, 6),
        "auto-variance-biased" | "autovar-biased" => (AqMode::AutoVarianceBiased, 1.0, 6),
        "psnr-probe" | "psnr" => (AqMode::PsnrProbe, 0.35, 2),
        "positive-probe" | "positive" => (AqMode::PositiveProbe, 0.35, 2),
        // Measured optimum on the 7-image test set (2026-07 sweeps): s1.0 is the
        // peak of the strength curve (s0.6 gets roughly half the BD-rate gain,
        // s1.4+ declines); clamp 4 — 6 was indistinguishable at s1.0.
        "two-pass" | "twopass" => recommended_aq_preset(),
        _ => return None,
    })
}

/// Resolve the effective [`AqMode`] for an encode. Resolution order:
/// 1. `BPG_PLACEBO_AQ` env override (Placebo only) — back-compat experiment hook.
/// 2. `config.aq_mode` (the `--aq` flag).
/// 3. `config.adaptive_qp = true` ⇒ [`AqMode::LegacyShrink`] (deprecated alias).
/// 4. Otherwise [`AqMode::Off`].
pub fn resolve_aq_mode(config: &StillHevcConfig) -> AqMode {
    if config.effort == Effort::Placebo {
        if let Some((mode, _)) = placebo_aq_env() {
            return mode;
        }
    }
    if config.aq_mode != AqMode::Off {
        return config.aq_mode;
    }
    if config.adaptive_qp {
        return AqMode::LegacyShrink;
    }
    AqMode::Off
}

/// Resolve `(mode, strength, clamp)` for an encode, honouring the
/// `BPG_PLACEBO_AQ` / `BPG_PLACEBO_AQ_STRENGTH` env override on Placebo and the
/// `config.aq_*` fields otherwise.
pub fn resolve_aq(config: &StillHevcConfig) -> (AqMode, f32, f32) {
    if config.effort == Effort::Placebo {
        if let Some((mode, strength)) = placebo_aq_env() {
            // The legacy env path keeps the wider legacy clamp.
            return (mode, strength, preanalysis::VAR_AQ_CLAMP);
        }
    }
    let mode = resolve_aq_mode(config);
    (mode, config.aq_strength, config.aq_clamp.max(1) as f32)
}

/// Parse the legacy `BPG_PLACEBO_AQ` / `BPG_PLACEBO_AQ_STRENGTH` experiment
/// override into `(mode, strength)`. Recognised values: `perceptual`,
/// `perceptual-chroma`, `psnr`, `positive`; any other non-empty value falls back
/// to `psnr` (its historical default). Unset / `0` / empty ⇒ `None`.
fn placebo_aq_env() -> Option<(AqMode, f32)> {
    let mode = std::env::var("BPG_PLACEBO_AQ").ok()?;
    let mode = mode.trim();
    if mode.is_empty() || mode == "0" {
        return None;
    }
    let aq_mode = if mode.eq_ignore_ascii_case("perceptual") {
        AqMode::Perceptual
    } else if mode.eq_ignore_ascii_case("perceptual-chroma")
        || mode.eq_ignore_ascii_case("perceptual_chroma")
    {
        AqMode::PerceptualChroma
    } else if mode.eq_ignore_ascii_case("positive") {
        AqMode::PositiveProbe
    } else {
        AqMode::PsnrProbe
    };
    let strength = std::env::var("BPG_PLACEBO_AQ_STRENGTH")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
        .unwrap_or(1.0);
    Some((aq_mode, strength))
}

/// Whether HEVC sign-data-hiding is applied for a config. Single source of
/// truth consulted by both the PPS writer ([`params::write_pps`], which emits
/// `sign_data_hiding_enabled_flag`) and the encoder (which makes each coding
/// group's hidden sign parity-consistent), so the flag and the coded levels
/// can never disagree.
///
/// Enabled by default for every effort: `bpgenc -m9` (x265 placebo) always codes
/// with `signhide`, and an A/B on the slow/x265shape parity tier showed SDH is a
/// clean equal-bytes BD-rate win (~+0.04–0.055 dB on textured content) that is
/// decode-safe (the `recon==decode` round-trip holds with it on). Previously this
/// was scoped to the old Best tier only. `BPG_SDH=0` (or the legacy
/// `BPG_BEST2_SDH=0`) disables it for A/B testing.
#[inline]
pub fn sdh_active(config: &StillHevcConfig) -> bool {
    let _ = config;
    if let Ok(v) = std::env::var("BPG_SDH") {
        return v.trim() != "0";
    }
    std::env::var("BPG_BEST2_SDH")
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
    /// **Deprecated** compatibility alias for `aq_mode = AqMode::LegacyShrink`.
    /// When `aq_mode` is `Off` and this is `true`, the encoder uses the legacy
    /// one-directional importance shrink. Prefer setting [`Self::aq_mode`].
    pub adaptive_qp: bool,
    /// Adaptive-quantization strategy. **`Off` by default** (uniform QP): the
    /// effort ladder is uniform-QP (each tier a pruned `Best`), which the
    /// project's measurements found beats per-CU variance AQ on photos. AQ is an
    /// explicit opt-in via the `--aq` flag for the Slow tier (single pass) and an
    /// experiment hook on Placebo (`BPG_PLACEBO_AQ` env override).
    pub aq_mode: AqMode,
    /// AQ strength (x265 `--aq-strength`): scales the per-QG QP deviation from
    /// the picture mean. Ignored when `aq_mode == AqMode::Off`.
    pub aq_strength: f32,
    /// Maximum magnitude (QP steps) the AQ offset is clamped to. The `--aq`
    /// presets use `2`. Ignored when `aq_mode == AqMode::Off`.
    pub aq_clamp: u8,
    /// Two-pass AQ candidate-compare gate (default `true`). When the resolved AQ
    /// mode is [`AqMode::TwoPassMeasured`], keep the AQ candidate only if it is a
    /// perceptual rate-distortion win over the uniform pass; otherwise emit the
    /// uniform encode. Makes two-pass AQ safe to enable by default — it can
    /// improve a picture but never regress one. Ignored for non-two-pass modes.
    pub two_pass_gate: bool,
    /// Psychovisual RD strength (x265 `--psy-rd`, sane range 0..=5, x265
    /// default 2.0). Adds an AC-energy-preservation term to mode/TU/CU RD
    /// decisions. `0.0` (the default) is byte-identical to the plain SSE+rate
    /// cost. Clamped on resolve; see `encoder::stillsearch::env::psy_rd`.
    pub psy_rd: f32,
    /// Psychovisual RDOQ strength (x265 `--psy-rdoq`, sane range 0..=50).
    /// Biases RDOQ level decisions toward preserving AC coefficient energy.
    /// `0.0` (the default) disables it. Clamped on resolve; see
    /// `encoder::stillsearch::env::psy_rdoq`.
    pub psy_rdoq: f32,
    /// Adaptive-quantization quantization-group size in luma samples: `32`
    /// (default, PPS `diff_cu_qp_delta_depth = 1`) or `16` (opt-in,
    /// `diff_cu_qp_delta_depth = 2`; yeo2013 found 16x16 the best AQ
    /// granularity for BD-rate). Any other value resolves to 32. Applies to
    /// the single-pass AQ modes and the external `BPG_AQ_OFFSET_MAP`;
    /// [`AqMode::TwoPassMeasured`] always uses 32x32 QGs (its pass-1 activity
    /// accumulator is 32x32-only for now). Ignored when AQ is off.
    pub aq_qg: u8,
}

/// Resolve the effective quantization-group size (log2, luma samples) for an
/// encode: `5` (32x32, the default) or `4` (16x16, `aq_qg = 16` opt-in). The
/// single source of truth shared by the PPS writer (`diff_cu_qp_delta_depth =
/// CTB_LOG2 − qg_log2`) and the encoder's QP-prediction mirror, so the
/// bitstream and the per-CU QP plan can never disagree on the QG grid.
///
/// [`AqMode::TwoPassMeasured`] is forced to 32x32 regardless of `aq_qg`: its
/// pass-1 per-QG activity accumulator and candidate gate are defined on the
/// 32x32 grid (16x16 QGs there are out of scope for now).
#[inline]
pub fn resolve_aq_qg_log2(config: &StillHevcConfig) -> u8 {
    if config.aq_qg == 16 && resolve_aq_mode(config) != AqMode::TwoPassMeasured {
        4
    } else {
        5
    }
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
        nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps(config));
        nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
        let wpp = params::effective_wpp_enabled(config);
        let tiles = params::effective_tile_dims(config);
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::Pps,
            &params::write_pps(config, tiles, wpp),
        );
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::IdrWRadl,
            &slice::write_slice_segment_header(config, tiles, wpp, &[]),
        );
        out
    }
}
