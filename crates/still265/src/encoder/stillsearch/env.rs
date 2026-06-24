//! Centralized StillSearch environment gates.
//!
//! Keep StillSearch-only env parsing here so experimental knobs are discoverable
//! without grepping the search implementation. These are process-global and are
//! cached on first use; set them before starting an encode.

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

/// Number of cheap-ranked luma candidates to remeasure with full recursive TU
/// search. `1` is fastest; larger values move toward the old exhaustive exact
/// shortlist. Defaults to 1 (dropped 3→2→1 after sweeps showed no quality loss).
pub(super) const LUMA_CHEAP_EXACT_TOP: &str = "BPG_STILLSEARCH_LUMA_CHEAP_EXACT_TOP";

/// Diagnostic angular-mode prefilter for rough luma search:
/// `off|game|iame|tsame`. Defaults to off. `BPG_BEST_ANGULAR_EXCLUSION` is
/// accepted as a compatibility alias for old rdo2 experiments.
pub(super) const ANGULAR_EXCLUSION: &str = "BPG_STILLSEARCH_ANGULAR_EXCLUSION";
pub(super) const ANGULAR_EXCLUSION_LEGACY: &str = "BPG_BEST_ANGULAR_EXCLUSION";

/// Max rough-pass candidates carried into the cheap luma pass (before MPM
/// union). Default 2 (was 4→3→2; sweeps showed no quality loss at each step).
/// Increasing to 3 or 4 trades ~1s speed for minor quality margin insurance.
pub(super) const ROUGH_RD_CANDS_ENV: &str = "BPG_STILLSEARCH_ROUGH_RD_CANDS";

/// Skip PartNxN for 8×8 CUs whose rough SATD (best mode, from RoughLuma) is
/// below this threshold. A rough SATD of T means the block's best-mode
/// prediction error is < T / 64 per pixel on average.
/// Default 1000: smooth blocks skip NxN (PSNR is unchanged or slightly better
/// for smooth content; saves ~3s on a 12MP encode). Set to 0 to disable.
pub(super) const NXN_SKIP_SATD: &str = "BPG_STILLSEARCH_NXN_SKIP_SATD";

/// Angular exclusion parameters retained from the paper-derived rdo2 experiment.
pub(super) const ANGULAR_GAME_VAR: &str = "BPG_ANGULAR_GAME_VAR";
pub(super) const ANGULAR_IAME_FACTOR: &str = "BPG_ANGULAR_IAME_FACTOR";
pub(super) const ANGULAR_MIN_KEEP: &str = "BPG_ANGULAR_MIN_KEEP";
pub(super) const ANGULAR_MIN_LOG2: &str = "BPG_ANGULAR_MIN_LOG2";

#[inline]
pub(super) fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os(PROFILE).is_some())
}

#[inline]
pub(super) fn luma_cheap_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(LUMA_CHEAP)
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

#[inline]
pub(super) fn luma_cheap_exact_top() -> usize {
    static TOP: OnceLock<usize> = OnceLock::new();
    *TOP.get_or_init(|| parse_env_or(LUMA_CHEAP_EXACT_TOP, 1usize).max(1))
}

#[inline]
pub(super) fn luma_cheap_residual_price_exact() -> bool {
    static EXACT: OnceLock<bool> = OnceLock::new();
    *EXACT.get_or_init(|| {
        std::env::var(LUMA_CHEAP_RESIDUAL_PRICE)
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "skip" || v == "none" || v == "off")
            })
            .unwrap_or(true)
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

#[inline]
pub(super) fn rough_rd_cands() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    // Cap at 4 (ROUGH_RD_CANDS) so the fixed-size shortlist arrays are never overflowed.
    *V.get_or_init(|| parse_env_or(ROUGH_RD_CANDS_ENV, 2usize).clamp(1, 4))
}

#[inline]
pub(super) fn nxn_skip_satd_threshold() -> f64 {
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| parse_env_or(NXN_SKIP_SATD, 1000.0_f64))
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
