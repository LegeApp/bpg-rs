//! Centralized StillSearch environment gates.
//!
//! Keep StillSearch-only env parsing here so experimental knobs are discoverable
//! without grepping the search implementation. These are process-global and are
//! cached on first use; set them before starting an encode.
//!
//! Each function accepts a template fallback value so the search code reads from
//! the effort template by default but a developer can still override via env var.

use std::sync::OnceLock;

use super::angular::{AngularExclusionConfig, AngularExclusionMode};
use crate::rdoq::RdoqPricingMode;

/// Set to any value to collect `EncodeStats::stillsearch_ledger_ns` bucket
/// timings in addition to call counts.
pub(super) const PROFILE: &str = "BPG_STILLSEARCH_PROFILE";

/// `0` restores the old behavior: every rough-shortlisted luma mode gets full
/// recursive TU search. Any other value / unset enables the x265-style cheap
/// luma-only first pass followed by full TU search for one winner.
pub(super) const LUMA_CHEAP: &str = "BPG_STILLSEARCH_LUMA_CHEAP";

/// Diagnostic RD lambda multiplier. Values below 1.0 spend more bits for lower
/// distortion; values above 1.0 save bits. Defaults to 1.0.
pub(super) const LAMBDA_SCALE: &str = "BPG_STILLSEARCH_LAMBDA_SCALE";

/// Experimental SSIM-RD-style candidate cost. This is deliberately separate
/// from AQ: it only changes mode/TU/CBF ranking by adding a source-normalized
/// residual-energy term to the normal SSE+rate cost. Off by default.
pub(super) const SSIM_RD: &str = "BPG_STILLSEARCH_SSIM_RD";

/// Use RDOQ, not hard quantization, during search/materialization trials.
/// x265 placebo (`-m9`) has rdoq-level=2 enabled during RD search, while
/// still265 historically ran RDOQ only as a final winner recode. `1` forces
/// RDOQ trials everywhere, `0` forces hard quant everywhere; unset (or
/// `effort`) defers to the effort template's per-stage `TrialQuant` policy
/// (Rdoq on slow/placebo, HardQuant on fast). Measured 2026-07-02:
/// −2.8% Y / −2.7% RGB BD-rate on the small native set at +46% encode time.
pub(super) const RDOQ_SEARCH: &str = "BPG_STILLSEARCH_RDOQ_SEARCH";

/// One-pass exact evaluation: `decide_tt` (Phase 1) quantizes with the effort
/// policy's trial quant (RDOQ on slow/placebo), and the Phase-2 RDOQ re-run of
/// exact candidates is skipped as redundant. This halves exact-stage work and
/// explores TU splits under RDOQ (closer to x265 rdoq-level 2, which never
/// evaluates a candidate twice). `0` restores the two-phase hard-quant→RDOQ
/// flow. On by default.
pub(super) const RDOQ_ONE_PASS: &str = "BPG_STILLSEARCH_RDOQ_ONE_PASS";

/// Experimental post-RDOQ residual-pricing model:
/// `frozen|running|hybrid|exact-cleanup|bounded-trellis`, or `0|1|2|3|4`.
/// Defaults to `frozen`, preserving the current single-scan RDOQ output.
pub(super) const RDOQ_PRICING: &str = "BPG_RDOQ_PRICING";

/// Beam width for `BPG_RDOQ_PRICING=bounded-trellis` (default 4, clamped
/// internally by the optimizer).
pub(super) const RDOQ_TRELLIS_BEAM: &str = "BPG_RDOQ_TRELLIS_BEAM";

/// Diagnostic: disable winner-final RDOQ recode and commit hard-quantized
/// search coefficients. x265 rdoq-level=0 A/B is a useful quant/RDO parity
/// reference; this isolates still265's final RDOQ contribution.
pub(super) const FINAL_RDOQ: &str = "BPG_STILLSEARCH_FINAL_RDOQ";

/// Diagnostic only: sample RDOQ-coded block residual pricing rows. Set
/// `BPG_STILLSEARCH_RDOQ_AUDIT=1` and optionally
/// `BPG_STILLSEARCH_RDOQ_AUDIT_MOD=N` (default 1024; 1 prints every block).
pub(super) const RDOQ_AUDIT: &str = "BPG_STILLSEARCH_RDOQ_AUDIT";
pub(super) const RDOQ_AUDIT_MOD: &str = "BPG_STILLSEARCH_RDOQ_AUDIT_MOD";

/// Single-block forensic dump target. `BPG_STILL_DUMP=x,y,log2,c_idx` makes
/// `eval_component_8_from_src_pred` print the full residual pipeline (source,
/// prediction, residual, forward coeffs, quantised levels, dequant, inverse
/// residual, recon, distortion) for every candidate mode whose block geometry
/// matches, so it can be diffed coefficient-by-coefficient against x265. The
/// fourth field (c_idx) is optional and defaults to 0 (luma).
pub(super) const STILL_DUMP: &str = "BPG_STILL_DUMP";

/// Diagnostic transform-tree split controls. These target the observed x265
/// parity gap where C emits many more luma TUs, especially 4x4 TUs in textured
/// regions. All are off by default.
pub(super) const TU_MIN_SPLIT_LOG2: &str = "BPG_STILLSEARCH_TU_MIN_SPLIT_LOG2";
pub(super) const TU_SPLIT_BIAS_PER_PX: &str = "BPG_STILLSEARCH_TU_SPLIT_BIAS_PER_PX";
pub(super) const TU_4X4_BIAS_PER_PX: &str = "BPG_STILLSEARCH_TU_4X4_BIAS_PER_PX";
pub(super) const TU_FORCE_SPLIT_LOG2: &str = "BPG_STILLSEARCH_TU_FORCE_SPLIT_LOG2";

/// Targeted 8x8->4x4 TU split admission. Unlike the global min-split/bias
/// probes, this only admits extra 4x4 split evaluation when local evidence says
/// the 8x8 leaf is a poor fit.
pub(super) const TU_TARGET_4X4: &str = "BPG_STILLSEARCH_TU_TARGET_4X4";
pub(super) const TU_TARGET_VAR_MIN: &str = "BPG_STILLSEARCH_TU_TARGET_VAR_MIN";
pub(super) const TU_TARGET_COST_MIN: &str = "BPG_STILLSEARCH_TU_TARGET_COST_MIN";
pub(super) const TU_TARGET_MEAN_RANGE_MIN: &str = "BPG_STILLSEARCH_TU_TARGET_MEAN_RANGE_MIN";

/// Diagnostic CU split controls. These increase the number of luma CUs/TUs to
/// probe the x265 gap where bpgenc emits many more small partitions.
pub(super) const CU_SPLIT_BIAS_PER_PX: &str = "BPG_STILLSEARCH_CU_SPLIT_BIAS_PER_PX";
pub(super) const CU_FORCE_SPLIT_LOG2: &str = "BPG_STILLSEARCH_CU_FORCE_SPLIT_LOG2";

/// Decision-identical CU split branch-and-bound: abort remaining split
/// children once the partial split cost already exceeds the leaf-first bound
/// (the split can no longer win the leaf-vs-split compare). `0` disables.
pub(super) const CU_SPLIT_BOUND: &str = "BPG_STILLSEARCH_CU_SPLIT_BOUND";

/// Same branch-and-bound at the transform-tree level (`decide_tt` leaf-first
/// split evaluation). `0` disables.
pub(super) const TU_SPLIT_BOUND: &str = "BPG_STILLSEARCH_TU_SPLIT_BOUND";

/// Diagnostic PartNxN controls. PartNxN creates four 4x4 luma PUs/TUs inside
/// an 8x8 CU, matching one path by which x265 emits many more 4x4 luma TUs.
pub(super) const NXN_BIAS_PER_PX: &str = "BPG_STILLSEARCH_NXN_BIAS_PER_PX";
pub(super) const NXN_FORCE: &str = "BPG_STILLSEARCH_NXN_FORCE";

/// Multiplier for the experimental SSIM-RD residual-energy term.
pub(super) const SSIM_RD_WEIGHT: &str = "BPG_STILLSEARCH_SSIM_RD_WEIGHT";

/// Residual syntax pricing mode used only by the luma-cheap first pass.
/// Defaults to `exact`. Set to `skip` or `0` to rank cheap luma candidates
/// without exact residual syntax bits; the selected winner is still remeasured
/// with exact pricing in the full pass.
pub(super) const LUMA_CHEAP_RESIDUAL_PRICE: &str = "BPG_STILLSEARCH_LUMA_CHEAP_RESIDUAL_PRICE";

/// Experimental: when set (`1`/`on`), the cheap-pass exact residual pricing uses
/// the static-context (frozen-state, x265 `EstBitsSbac`-style) bit estimate
/// instead of the evolving-CABAC re-encode. A/B knob for the residual-pricing
/// speed/quality lever; off by default.
pub(super) const CHEAP_BITS_STATIC: &str = "BPG_STILLSEARCH_CHEAP_BITS_STATIC";

/// Diagnostic angular-mode prefilter for rough luma search:
/// `off|game|iame|tsame`. Defaults to off. `BPG_BEST_ANGULAR_EXCLUSION` is
/// accepted as a compatibility alias for old rdo2 experiments.
pub(super) const ANGULAR_EXCLUSION: &str = "BPG_STILLSEARCH_ANGULAR_EXCLUSION";
pub(super) const ANGULAR_EXCLUSION_LEGACY: &str = "BPG_BEST_ANGULAR_EXCLUSION";

/// Skip PartNxN for 8×8 CUs whose rough SATD (best mode, from RoughLuma) is
/// below this threshold. A rough SATD of T means the block's best-mode
/// prediction error is < T / 64 per pixel on average.
/// Default 1000: smooth blocks skip NxN (PSNR is unchanged or slightly better
/// for smooth content; saves ~3s on a 12MP encode). Set to 0 to disable.
pub(super) const NXN_SKIP_SATD: &str = "BPG_STILLSEARCH_NXN_SKIP_SATD";
pub(super) const ANGULAR_GAME_VAR: &str = "BPG_ANGULAR_GAME_VAR";
pub(super) const ANGULAR_IAME_FACTOR: &str = "BPG_ANGULAR_IAME_FACTOR";
pub(super) const ANGULAR_MIN_KEEP: &str = "BPG_ANGULAR_MIN_KEEP";
pub(super) const ANGULAR_MIN_LOG2: &str = "BPG_ANGULAR_MIN_LOG2";

/// x265-parity CABAC context evolution during StillSearch RD pricing.
///
/// x265 chains entropy contexts through the coding tree during analysis: each
/// child CU is priced from the previous sibling's post-coding contexts
/// (`compressIntraCU`'s `nextContext`), `codeIntraLumaQT` evolves the coder
/// TU-to-TU with per-depth save/restore, and the RDOQ rate model reads those
/// evolved states. still265 historically priced every trial from a context
/// frozen at CTU entry (`price_base`), overestimating residual bits mid-CTU by
/// a measured 12–37%.
///
/// Values: `1`/unset = full (search + finalize), `0`/`off` = disabled
/// (byte-identical to the frozen-context behavior), `search` = search-time
/// chaining only, `finalize` = decoder-order final-RDOQ contexts only.
pub(super) const CTX_EVOLVE: &str = "BPG_STILLSEARCH_CTX_EVOLVE";

/// x265-parity CU-header bit pricing: include `intra_chroma_pred_mode` bits
/// for every CU leaf and `part_mode` bits for every 8x8 CU in the leaf RD
/// cost, matching x265 `checkIntra`'s full-syntax `totalBits`. Historically
/// these were only priced inside the 8x8 2Nx2N-vs-NxN comparison, so CU
/// splits (4 headers vs 1) were under-priced. `0` restores the old pricing.
pub(super) const CU_HEADER_BITS: &str = "BPG_STILLSEARCH_CU_HEADER_BITS";

/// x265-parity CU-level chroma intra mode search (`estIntraPredChromaQT`):
/// full-RD evaluation of the five allowed chroma modes
/// {planar, vertical, horizontal, DC, DM} for each CU leaf, instead of
/// always signalling DM. `0` disables (DM-only, the historical behavior).
pub(super) const CHROMA_MODE_SEARCH: &str = "BPG_STILLSEARCH_CHROMA_MODES";

/// Chroma-mode search optimizer knobs. The mode search gate above still
/// controls whether chroma modes are searched at all.
///
/// `CHROMA_ROUGH=0` restores exhaustive exact RD over all non-DM modes.
/// `CHROMA_ROUGH_K` is the number of non-DM modes promoted from the rough SATD
/// pass to exact RD (1..=4; default from the effort template).
/// `CHROMA_SKIP_SATD` is a per-chroma-sample rough-SATD threshold below which
/// DM is kept immediately (0 disables; default from the effort template).
/// `CHROMA_REUSE_DM=0` restores the old behavior of rebuilding DM winners
/// instead of replaying the already-materialized DM residual contexts.
pub(super) const CHROMA_ROUGH: &str = "BPG_STILLSEARCH_CHROMA_ROUGH";
pub(super) const CHROMA_ROUGH_K: &str = "BPG_STILLSEARCH_CHROMA_ROUGH_K";
pub(super) const CHROMA_SKIP_SATD: &str = "BPG_STILLSEARCH_CHROMA_SKIP_SATD";
pub(super) const CHROMA_REUSE_DM: &str = "BPG_STILLSEARCH_CHROMA_REUSE_DM";

/// Rough-pass audit: when set, logs per-CU statistics about rough vs exact
/// ranking and angular-mode inclusion to stderr.  Sampled to avoid flooding
/// large images; the sampling stride is controlled by `ROUGH_AUDIT_MOD`.
pub(super) const ROUGH_AUDIT: &str = "BPG_STILLSEARCH_ROUGH_AUDIT";

/// When set, the rough luma search uses scalar `predict_into` for angular
/// modes instead of the batched `predict_all_angular_into_u16` primitive.
/// Slower, but isolates whether the batched angular predictor path is
/// numerically biased.
pub(super) const SCALAR_ANGULAR_ROUGH: &str = "BPG_STILLSEARCH_SCALAR_ANGULAR_ROUGH";

/// Rough-stage mode-bit cost weight (default 1.0).  Set to 0 to disable mode
/// bits in rough ranking, isolating whether planar/DC signalling advantage
/// is suppressing angular candidates too early.
pub(super) const ROUGH_MODE_BIT_WEIGHT: &str = "BPG_STILLSEARCH_ROUGH_MODE_BIT_WEIGHT";

/// Diagnostic x265shape RMD admission window. x265 uses 1.25. Larger values
/// are not canonical; they test whether Still265's current rough cost model is
/// losing oracle winners before simple RDO can rank them.
pub(super) const X265_RMD_WINDOW: &str = "BPG_STILLSEARCH_X265_RMD_WINDOW";

/// Diagnostic x265shape RMD candidate cap. x265 placebo uses
/// `2 + rdLevel + (depth >> 1)`. Larger values are not canonical; they test
/// whether Still265 rough ranking is pushing oracle winners below x265's cap.
pub(super) const X265_RMD_CAP: &str = "BPG_STILLSEARCH_X265_RMD_CAP";

/// Diagnostic only: when set alongside the luma oracle, print one `RMD_AUDIT:`
/// line per oracle sample comparing Still265's fractional mode-bit RMD rank
/// with x265-like integer `bitsCodeBin()` RMD ranks, including carry brackets.
pub(super) const X265_RMD_AUDIT: &str = "BPG_STILLSEARCH_X265_RMD_AUDIT";

/// Diagnostic only: use the no-carry integer x265 `calcRdSADCost` approximation
/// for x265shape RMD candidate admission. This is not the default because
/// x265's real `bitsCodeBin()` depends on the current fractional CABAC carry.
pub(super) const X265_RMD_INTEGER_COST: &str = "BPG_STILLSEARCH_X265_RMD_INTEGER_COST";

/// Diagnostic x265shape exact remeasure expansion. Default top=1 matches x265shape;
/// higher values remeasure multiple simple-RDO-ranked modes with split-enabled
/// exact search, optionally gated by variance and close-cost margin.
pub(super) const X265SHAPE_EXACT_TOP: &str = "BPG_STILLSEARCH_X265SHAPE_EXACT_TOP";
pub(super) const X265SHAPE_EXACT_MARGIN: &str = "BPG_STILLSEARCH_X265SHAPE_EXACT_MARGIN";
pub(super) const X265SHAPE_EXACT_VAR_MIN: &str = "BPG_STILLSEARCH_X265SHAPE_EXACT_VAR_MIN";

/// Diagnostic only: when set alongside the luma oracle, print one
/// `MISS_AUDIT:` line per sampled oracle miss with RMD admission and
/// simple-RDO ranking context.
pub(super) const LUMA_MISS_AUDIT: &str = "BPG_STILLSEARCH_LUMA_MISS_AUDIT";

/// Oracle diagnostic: when set, sampled CUs get full `decide_tt()` run on the
/// entire shortlist AND on all 35 modes, reporting the true-best winner.
/// Output is CSV-compatible lines on stderr bearing the `ORACLE:` prefix.
pub(super) const LUMA_ORACLE: &str = "BPG_STILLSEARCH_LUMA_ORACLE";

/// Sampling modulus for the luma oracle (default 64).  Every N-th CU is
/// sampled; higher values are less noisy but converge more slowly.
pub(super) const LUMA_ORACLE_MOD: &str = "BPG_STILLSEARCH_LUMA_ORACLE_MOD";

#[inline]
pub(super) fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(PROFILE).is_some())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CtxEvolve {
    search: bool,
    finalize: bool,
}

#[inline]
fn ctx_evolve() -> CtxEvolve {
    static VALUE: OnceLock<CtxEvolve> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(CTX_EVOLVE).unwrap_or_default();
        match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "none" => CtxEvolve {
                search: false,
                finalize: false,
            },
            "search" => CtxEvolve {
                search: true,
                finalize: false,
            },
            "finalize" => CtxEvolve {
                search: false,
                finalize: true,
            },
            _ => CtxEvolve {
                search: true,
                finalize: true,
            },
        }
    })
}

/// Search-time CABAC context chaining (CU/TU sibling-to-sibling pricing).
#[inline]
pub(super) fn ctx_evolve_search() -> bool {
    ctx_evolve().search
}

/// Decoder-order evolving contexts for the winner-only RDOQ finalize pass.
#[inline]
pub(super) fn ctx_evolve_finalize() -> bool {
    ctx_evolve().finalize
}

/// Full CU-header (`part_mode` + `intra_chroma_pred_mode`) bit pricing in CU
/// leaf RD costs (x265 `checkIntra` parity). Default on.
#[inline]
pub(super) fn cu_header_bits_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(CU_HEADER_BITS)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// CU-level full-RD chroma intra mode search (x265 `estIntraPredChromaQT`
/// parity). Default on.
#[inline]
pub(super) fn chroma_mode_search_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(CHROMA_MODE_SEARCH)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Enable SATD rough pre-selection for chroma mode search. Default on.
#[inline]
pub(super) fn chroma_rough_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(CHROMA_ROUGH)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Number of non-DM chroma modes to exact-evaluate after rough ranking.
#[inline]
pub(super) fn chroma_rough_k(template_k: u8) -> usize {
    static VALUE: OnceLock<Option<usize>> = OnceLock::new();
    let env_val = *VALUE.get_or_init(|| {
        std::env::var(CHROMA_ROUGH_K)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|v| v.clamp(1, 4))
    });
    env_val.unwrap_or(usize::from(template_k.clamp(1, 4)))
}

/// Per-sample SATD threshold below which chroma search keeps DM immediately.
#[inline]
pub(super) fn chroma_skip_satd(template_threshold: f64) -> f64 {
    static VALUE: OnceLock<Option<f64>> = OnceLock::new();
    let env_val = *VALUE.get_or_init(|| {
        std::env::var(CHROMA_SKIP_SATD)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
    });
    env_val.unwrap_or(template_threshold.max(0.0))
}

/// Reuse the already-materialized DM chroma blocks for the baseline cost and
/// avoid a full DM rebuild when DM wins. Default on.
#[inline]
pub(super) fn chroma_reuse_dm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(CHROMA_REUSE_DM)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

#[inline]
pub(super) fn x265_rmd_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(X265_RMD_AUDIT).is_some())
}

#[inline]
pub(super) fn x265_rmd_integer_cost_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(X265_RMD_INTEGER_COST).is_some())
}

#[inline]
pub(super) fn rdoq_audit_sample(x0: u32, y0: u32, log2_size: u8, c_idx: u8, mode: u8) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os(RDOQ_AUDIT).is_some()) {
        return false;
    }

    static MOD: OnceLock<u64> = OnceLock::new();
    let modulus = *MOD.get_or_init(|| {
        std::env::var(RDOQ_AUDIT_MOD)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1024)
    });
    if modulus <= 1 {
        return true;
    }

    // Stable integer mix: enough to spread neighboring blocks/modes without
    // pulling in a RNG or process-global state.
    let mut h = u64::from(x0 / 4).wrapping_mul(0x9E37_79B1);
    h ^= u64::from(y0 / 4).wrapping_mul(0x85EB_CA77);
    h ^= u64::from(log2_size).wrapping_mul(0xC2B2_AE3D);
    h ^= u64::from(c_idx).wrapping_mul(0x27D4_EB2F);
    h ^= u64::from(mode).wrapping_mul(0x1656_67B1);
    h % modulus == 0
}

#[inline]
pub(super) fn x265shape_exact_top() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(X265SHAPE_EXACT_TOP)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(1)
    })
}

#[inline]
pub(super) fn x265shape_exact_margin() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(X265SHAPE_EXACT_MARGIN)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 1.0)
            .unwrap_or(f64::INFINITY)
    })
}

#[inline]
pub(super) fn x265shape_exact_var_min() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(X265SHAPE_EXACT_VAR_MIN)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0)
    })
}

#[inline]
pub(super) fn luma_miss_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(LUMA_MISS_AUDIT).is_some())
}

/// Returns `true` if luma cheap pass is enabled.
/// Falls back to `template_enabled` when env var is not set.
#[inline]
pub(super) fn luma_cheap_enabled(template_enabled: bool) -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_val = *ENV.get_or_init(|| std::env::var(LUMA_CHEAP).ok().map(|v| v.trim() != "0"));
    env_val.unwrap_or(template_enabled)
}

/// RD lambda multiplier used by [`super::price::rd_lambda`].
#[inline]
pub(super) fn lambda_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var(LAMBDA_SCALE)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

/// Target CTU `(x,y)` pixel origin for candidate-cost dumps (`BPG_DUMP_CTU="x,y"`,
/// same env as the patched x265 `BPGCAND` dump). `None` when unset/malformed.
pub(super) fn dump_ctu_target() -> Option<(u32, u32)> {
    static V: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *V.get_or_init(|| {
        let s = std::env::var("BPG_DUMP_CTU").ok()?;
        let (a, b) = s.split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

/// Diagnostic scale on the SEARCH-side residual-bit estimate
/// (`BPG_RESIDUAL_BITS_SCALE`, default 1.0). Applies only to trial RD pricing,
/// never to emitted bits. <1 approximates the within-CTU CABAC context drift
/// (mid-CTU blocks code ~12-37% cheaper than the frozen CTU-entry `price_base`
/// estimates), to test whether pricing residual closer to its real evolving-
/// context cost shifts textured leaf/split/CBF decisions toward more coding.
pub(super) fn residual_bits_search_scale() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("BPG_RESIDUAL_BITS_SCALE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

/// Parsed `BPG_STILL_DUMP` forensic target: `(x, y, log2_size, c_idx)`.
/// Cached; `None` when unset or malformed (the hot path pays one branch).
#[inline]
pub(super) fn dump_target() -> Option<(u32, u32, u8, u8)> {
    static VALUE: OnceLock<Option<(u32, u32, u8, u8)>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var(STILL_DUMP).ok()?;
        let mut it = raw.split(',').map(|s| s.trim());
        let x = it.next()?.parse::<u32>().ok()?;
        let y = it.next()?.parse::<u32>().ok()?;
        let log2 = it.next()?.parse::<u8>().ok()?;
        let c_idx = it.next().and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
        Some((x, y, log2, c_idx))
    })
}

/// Override the smallest TU log2 size that may be reached by optional splits.
/// `2` enables 8x8 -> 4x4 split evaluation; default policy usually stops at 8x8.
#[inline]
pub(super) fn tu_min_split_log2_override() -> Option<u8> {
    static VALUE: OnceLock<Option<u8>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_MIN_SPLIT_LOG2)
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|&v| (2..=5).contains(&v))
    })
}

/// Subtract this many RD-cost units per luma pixel from split candidates.
#[inline]
pub(super) fn tu_split_bias_per_px() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_SPLIT_BIAS_PER_PX)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

/// Extra split bias for 8x8 -> 4x4 split decisions.
#[inline]
pub(super) fn tu_4x4_bias_per_px() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_4X4_BIAS_PER_PX)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

/// Force optional TU splits at the given parent log2 size (for example `3`
/// forces legal 8x8 -> 4x4 splits).
#[inline]
pub(super) fn tu_force_split_log2() -> Option<u8> {
    static VALUE: OnceLock<Option<u8>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_FORCE_SPLIT_LOG2)
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|&v| (3..=6).contains(&v))
    })
}

/// Subtract this many RD-cost units per luma pixel from CU split candidates.
#[inline]
pub(super) fn cu_split_bias_per_px() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(CU_SPLIT_BIAS_PER_PX)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

/// Leaf-first CU split branch-and-bound abort. Default on; decision- and
/// byte-identical to full evaluation (only skips children that cannot change
/// the leaf-vs-split winner).
#[inline]
pub(super) fn cu_split_bound_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(CU_SPLIT_BOUND)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Leaf-first TU split branch-and-bound abort. Default on; decision- and
/// byte-identical to full evaluation (only skips TU children that cannot
/// change the leaf-vs-split winner).
#[inline]
pub(super) fn tu_split_bound_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(TU_SPLIT_BOUND)
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// Force CU splits at the given parent log2 size (for example `4` forces
/// legal 16x16 -> 8x8 CU splits).
#[inline]
pub(super) fn cu_force_split_log2() -> Option<u8> {
    static VALUE: OnceLock<Option<u8>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(CU_FORCE_SPLIT_LOG2)
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|&v| (4..=6).contains(&v))
    })
}

/// Subtract this many RD-cost units per luma pixel from legal PartNxN candidates.
#[inline]
pub(super) fn nxn_bias_per_px() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(NXN_BIAS_PER_PX)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
    })
}

/// Force legal PartNxN candidates to win after evaluation. Diagnostic only.
#[inline]
pub(super) fn nxn_force_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(NXN_FORCE)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none" || v.is_empty())
            })
            .unwrap_or(false)
    })
}

/// Enable targeted 8x8->4x4 TU split admission.
#[inline]
pub(super) fn tu_target_4x4_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_TARGET_4X4)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none" || v.is_empty())
            })
            .unwrap_or(false)
    })
}

#[inline]
pub(super) fn tu_target_var_min() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_TARGET_VAR_MIN)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(200.0)
    })
}

#[inline]
pub(super) fn tu_target_cost_min() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_TARGET_COST_MIN)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(50.0)
    })
}

#[inline]
pub(super) fn tu_target_mean_range_min() -> f64 {
    static VALUE: OnceLock<f64> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(TU_TARGET_MEAN_RANGE_MIN)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(16.0)
    })
}

/// True unless `BPG_STILLSEARCH_FINAL_RDOQ=0/off/none` disables final RDOQ.
#[inline]
pub(super) fn final_rdoq_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(FINAL_RDOQ)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none")
            })
            .unwrap_or(true)
    })
}

/// True unless `BPG_STILLSEARCH_RDOQ_ONE_PASS=0/off/none` restores the
/// two-phase (hard-quant Phase 1 + RDOQ Phase 2) exact evaluation.
#[inline]
pub(super) fn rdoq_one_pass_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(RDOQ_ONE_PASS)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none")
            })
            .unwrap_or(true)
    })
}

/// RDOQ-in-search override: `Some(true)` forces RDOQ trials at every search
/// decision site, `Some(false)` forces hard quant, `None` (unset or `effort`)
/// defers to the effort template's per-stage [`crate::effort::TrialQuant`].
#[inline]
pub(super) fn rdoq_search_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let v = std::env::var(RDOQ_SEARCH).ok()?;
        let v = v.trim().to_ascii_lowercase();
        if v.is_empty() || v == "effort" {
            return None;
        }
        Some(!(v == "0" || v == "off" || v == "none"))
    })
}

/// Post-RDOQ pricing experiment mode. See [`RDOQ_PRICING`].
#[inline]
pub(super) fn rdoq_pricing_mode() -> RdoqPricingMode {
    static MODE: OnceLock<RdoqPricingMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var(RDOQ_PRICING).ok();
        match raw
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("1" | "running" | "run") => RdoqPricingMode::Running,
            Some("2" | "hybrid") => RdoqPricingMode::Hybrid,
            Some("3" | "exact" | "exact-cleanup" | "cleanup") => RdoqPricingMode::ExactCleanup,
            Some("4" | "trellis" | "bounded-trellis" | "bounded_trellis") => {
                RdoqPricingMode::BoundedTrellis
            }
            _ => RdoqPricingMode::Frozen,
        }
    })
}

/// Beam width for `BPG_RDOQ_PRICING=bounded-trellis`.
#[inline]
pub(super) fn rdoq_trellis_beam() -> usize {
    static BEAM: OnceLock<usize> = OnceLock::new();
    *BEAM.get_or_init(|| {
        std::env::var(RDOQ_TRELLIS_BEAM)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(4)
    })
}

/// True when experimental SSIM-RD-style candidate ranking is enabled.
#[inline]
pub(super) fn ssim_rd_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(SSIM_RD)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none" || v.is_empty())
            })
            .unwrap_or(false)
    })
}

/// Weight for the experimental SSIM-RD-style residual-energy term.
#[inline]
pub(super) fn ssim_rd_weight() -> f64 {
    static WEIGHT: OnceLock<f64> = OnceLock::new();
    *WEIGHT.get_or_init(|| {
        std::env::var(SSIM_RD_WEIGHT)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(1.0 / 16384.0)
    })
}

/// Residual syntax pricing mode used only by the luma-cheap first pass.
/// Falls back to `template_exact` when env var is not set.
#[inline]
pub(super) fn luma_cheap_residual_price_exact(template_exact: bool) -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_val = *ENV.get_or_init(|| {
        std::env::var(LUMA_CHEAP_RESIDUAL_PRICE).ok().map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "skip" || v == "none" || v == "off")
        })
    });
    env_val.unwrap_or(template_exact)
}

/// True when the cheap pass should price residual bits with the static-context
/// estimate (frozen-state x265 `EstBitsSbac` model) instead of the evolving
/// CABAC re-encode. Off by default; see [`CHEAP_BITS_STATIC`].
pub(super) fn cheap_bits_static() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var(CHEAP_BITS_STATIC)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "off" || v == "none" || v.is_empty())
            })
            .unwrap_or(false)
    })
}

#[inline]
pub(super) fn angular_exclusion_config() -> AngularExclusionConfig {
    static CFG: OnceLock<AngularExclusionConfig> = OnceLock::new();
    *CFG.get_or_init(parse_angular_exclusion_config)
}

fn parse_angular_exclusion_config() -> AngularExclusionConfig {
    let raw = std::env::var(ANGULAR_EXCLUSION)
        .or_else(|_| std::env::var(ANGULAR_EXCLUSION_LEGACY))
        .ok();
    let mode = match raw
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("game") => AngularExclusionMode::Game,
        Some("iame") => AngularExclusionMode::Iame,
        Some("tsame") => AngularExclusionMode::Tsame,
        _ => AngularExclusionMode::Off,
    };
    AngularExclusionConfig {
        mode,
        game_ref_var_threshold: parse_env_or(ANGULAR_GAME_VAR, 8.0),
        iame_factor: parse_env_or(ANGULAR_IAME_FACTOR, 0.5),
        min_angular_keep: parse_env_or(ANGULAR_MIN_KEEP, 6),
        min_log2_size: parse_env_or(ANGULAR_MIN_LOG2, 4),
        protect_mpm: true,
    }
}

/// Threshold below which PartNxN is skipped for 8×8 CUs.
/// Falls back to `template_val` when env var is not set.
#[inline]
pub(super) fn nxn_skip_satd_threshold(template_val: f64) -> f64 {
    static ENV: OnceLock<Option<f64>> = OnceLock::new();
    let env_val = *ENV.get_or_init(|| {
        std::env::var(NXN_SKIP_SATD)
            .ok()
            .and_then(|s| s.parse().ok())
    });
    env_val.unwrap_or(template_val)
}

/// Returns `true` if the rough-stage audit should be collected.
#[inline]
pub(super) fn rough_audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(ROUGH_AUDIT).is_some())
}

/// Returns `true` if scalar (non-batched) angular prediction should be used
/// for the rough pass.
#[inline]
pub(super) fn scalar_angular_rough_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(SCALAR_ANGULAR_ROUGH)
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(false)
    })
}

/// Weight applied to the mode-bit cost in rough ranking (default 1.0).
/// Returns `None` when the env var is not set, letting the caller use its
/// default weight.
#[inline]
pub(super) fn rough_mode_bit_weight_override() -> Option<f64> {
    static V: OnceLock<Option<f64>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var(ROUGH_MODE_BIT_WEIGHT)
            .ok()
            .and_then(|s| s.parse().ok())
    })
}

/// Returns `true` if the sampled luma oracle diagnostic is enabled.
#[inline]
pub(super) fn luma_oracle_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(LUMA_ORACLE).is_some())
}

/// Returns the oracle sampling modulus (default 64).
#[inline]
pub(super) fn luma_oracle_mod() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var(LUMA_ORACLE_MOD)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64)
            .max(1)
    })
}

#[inline]
pub(super) fn x265_rmd_window() -> f64 {
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| parse_env_or(X265_RMD_WINDOW, 1.25_f64).max(1.0))
}

#[inline]
pub(super) fn x265_rmd_cap(default_cap: usize) -> usize {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    V.get_or_init(|| {
        std::env::var(X265_RMD_CAP)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|n| n.clamp(1, 35))
    })
    .unwrap_or(default_cap)
}

fn parse_env_or<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}
