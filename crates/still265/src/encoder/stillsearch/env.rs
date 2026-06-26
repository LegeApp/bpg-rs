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

/// Set to any value to collect `EncodeStats::stillsearch_ledger_ns` bucket
/// timings in addition to call counts.
pub(super) const PROFILE: &str = "BPG_STILLSEARCH_PROFILE";

/// `0` restores the old behavior: every rough-shortlisted luma mode gets full
/// recursive TU search. Any other value / unset enables the x265-style cheap
/// luma-only first pass followed by full TU search for one winner.
pub(super) const LUMA_CHEAP: &str = "BPG_STILLSEARCH_LUMA_CHEAP";

/// Residual syntax pricing mode used only by the luma-cheap first pass.
/// Defaults to `exact`. Set to `skip` or `0` to rank cheap luma candidates
/// without exact residual syntax bits; the selected winner is still remeasured
/// with exact pricing in the full pass.
pub(super) const LUMA_CHEAP_RESIDUAL_PRICE: &str = "BPG_STILLSEARCH_LUMA_CHEAP_RESIDUAL_PRICE";

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

/// Returns `true` if luma cheap pass is enabled.
/// Falls back to `template_enabled` when env var is not set.
#[inline]
pub(super) fn luma_cheap_enabled(template_enabled: bool) -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env_val = *ENV.get_or_init(|| std::env::var(LUMA_CHEAP).ok().map(|v| v.trim() != "0"));
    env_val.unwrap_or(template_enabled)
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

fn parse_env_or<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}
