//! Centralized search-effort templates and four-level pipeline configuration.
//!
//! The search pipeline has four explicit evaluation levels:
//!
//! ```text
//! Level 0 — Rough    (SATD-based mode scoring, shortlist construction)
//! Level 1 — Cheap    (hard-quant luma trials on a shortlist)
//! Level 2 — Exact    (full RD search on promoted modes)
//! Level 3 — Final    (winner-only RDOQ recode + syntax emission)
//! ```
//!
//! Each level has its own policy struct and evidence types.  Presets (Fast / Slow /
//! Placebo) assign specific budgets to each level — the pipeline itself stays the
//! same.

use std::fmt;

use crate::preanalysis::{RegionClass, SearchPolicy};
use crate::{Effort, is_reference_tier};

// ──────────────────────────────────────────────
// Shared enums
// ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntropyContextMode {
    Running,
    FrozenSliceInit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmdModeSet {
    MpmPlanarDcOnly,
    Step4,
    Step3,
    Step2,
    Dense,
    Exhaustive,
    Progressive {
        coarse_step: u8,
        top_regions: u8,
        refine_radius: u8,
    },
}

impl RmdModeSet {
    /// Build the mode list for the RMD (rough mode decision) angular scan.
    pub fn modes(self, mpm: [u8; 3]) -> Vec<u8> {
        let mut modes = Vec::with_capacity(35);
        // Planar = mode 0, DC = mode 1
        if self != RmdModeSet::MpmPlanarDcOnly {
            modes.push(0);
            modes.push(1);
        }
        match self {
            RmdModeSet::MpmPlanarDcOnly => {
                // No angular modes. MPMs handle the rest.
            }
            RmdModeSet::Step4 => {
                for mode in (2..=34).step_by(4) {
                    modes.push(mode);
                }
            }
            RmdModeSet::Step3 => {
                for mode in (2..=34).step_by(3) {
                    modes.push(mode);
                }
            }
            RmdModeSet::Step2 => {
                for mode in (2..=34).step_by(2) {
                    modes.push(mode);
                }
            }
            RmdModeSet::Dense => {
                for mode in 2..=34 {
                    modes.push(mode);
                }
            }
            RmdModeSet::Exhaustive => {
                for mode in 0..=34 {
                    modes.push(mode);
                }
            }
            RmdModeSet::Progressive {
                coarse_step,
                top_regions,
                refine_radius,
            } => {
                // Phase 1: coarse scan over the full angular range.
                let coarse: Vec<u8> = (2..=34).step_by(coarse_step as usize).collect();
                // Sort coarse modes by cost (placeholder — real sorting
                // happens in the search; here we keep the physical order).
                // The expensive step is to keep the top N regions.
                let top_n = top_regions.min(3); // max 3 directional regions
                let mut kept: Vec<u8> = Vec::new();
                // Group into H (2–9), D (10–25), V (26–34).
                for (lo, hi) in [(2u8, 9u8), (10, 25), (26, 34)] {
                    let in_region: Vec<u8> = coarse
                        .iter()
                        .copied()
                        .filter(|&m| m >= lo && m <= hi)
                        .collect();
                    if !in_region.is_empty() && kept.len() < top_n as usize {
                        kept.extend(&in_region);
                    }
                }
                // Fallback: if no coarse modes in top regions, keep best 2.
                if kept.is_empty() {
                    kept.extend(coarse.iter().take(2));
                }
                modes.extend(&kept);
                // Phase 2: refine around kept modes.
                if refine_radius > 0 {
                    let refine_set: Vec<u8> = kept
                        .iter()
                        .flat_map(|&m| {
                            let lo = m.saturating_sub(refine_radius).max(2);
                            let hi = (m + refine_radius).min(34);
                            (lo..=hi).filter(|&c| !modes.contains(&c))
                        })
                        .collect();
                    // Add refinement modes in order, deduplicated.
                    for m in refine_set {
                        if !modes.contains(&m) {
                            modes.push(m);
                        }
                    }
                }
            }
        }
        // Always include MPM candidates (may be duplicate — caller deduplicates).
        modes.extend_from_slice(&mpm);
        modes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitSearch {
    ForceLeaf,
    PreferLeaf,
    EvaluateBoth,
    PreferSplit,
    ForceSplit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentKind {
    Luma,
    ChromaCb,
    ChromaCr,
}

impl ComponentKind {
    pub fn from_c_idx(c_idx: u8) -> Self {
        match c_idx {
            0 => ComponentKind::Luma,
            1 => ComponentKind::ChromaCb,
            _ => ComponentKind::ChromaCr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AngularFamily {
    Horizontal,
    Diagonal,
    Vertical,
}

impl AngularFamily {
    pub fn classify(mode: u8) -> Option<Self> {
        match mode {
            2..=9 => Some(AngularFamily::Horizontal),
            10..=25 => Some(AngularFamily::Diagonal),
            26..=34 => Some(AngularFamily::Vertical),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuEarlyTerminateRule {
    Disabled,
    Balanced,
    Fast,
    Fastest,
}

// ──────────────────────────────────────────────
// RDOQ trial policy
// ──────────────────────────────────────────────

/// Controls whether analysis-stage RDOQ trials are run on exact candidates.
///
/// x265's `--rdoq-level=1` applies RDOQ during the search (not just at final
/// write).  This enum controls the same boundary: RDOQ should influence mode
/// and TU split ranking, not just final coefficient rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialRdoqMode {
    Off,
    ExactOnly,
    ExactCloseOnly,
    CheapCloseAndExact,
    PlaceboAllExact,
}

/// Policy for analysis-stage RDOQ trials.
#[derive(Clone, Copy, Debug)]
pub struct RdoqTrialPolicy {
    pub mode: TrialRdoqMode,
    /// Re-run RDOQ if a candidate is within this cost ratio of the best
    /// hard-quant candidate (only meaningful in `ExactCloseOnly` / `CheapCloseAndExact`).
    pub close_margin: f64,
    /// Maximum RDOQ trial candidates per luma mode search.
    pub max_rdoq_modes: u8,
    /// RDOQ level (1 = coefficient-level rounding only).
    pub level: u8,
}

// ──────────────────────────────────────────────
// TU split policy
// ──────────────────────────────────────────────

/// How the exact stage explores the residual quad-tree (TU split).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuSplitMode {
    Disabled,
    LeafFirstEarlyTerminate,
    EvaluateBoth,
    ForceSplit,
}

/// Configuration for exact-stage TU split search.
#[derive(Clone, Copy, Debug)]
pub struct TuExactPolicy {
    pub split_mode: TuSplitMode,
    pub max_extra_depth: u8,
    pub min_split_log2: u8,
    /// Test split after leaf unless leaf is clearly terminal.
    pub leaf_first: bool,
    /// For Fast only.
    pub zero_residual_early_terminate: bool,
    pub low_residual_early_terminate: bool,
    pub low_residual_bits_per_px: f64,
    pub low_distortion_per_px: f64,
    /// For Slow/Placebo — apply RDOQ during split evaluation.
    pub rdoq_split_trials: bool,
}

// ──────────────────────────────────────────────
// Core level enums
// ──────────────────────────────────────────────

/// Identifies which search level the engine is currently executing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchLevel {
    Rough,
    Cheap,
    Exact,
    Final,
}

/// Whether a trial evaluates only luma or full luma+chroma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentScope {
    LumaOnly,
    FullComponents,
}

/// Quantizer mode for a trial pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialQuant {
    HardQuant,
    Rdoq,
}

/// How residual bits are priced during RD cost computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualPriceLevel {
    None,
    Approx,
    Exact,
}

/// When chroma is evaluated relative to luma mode search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaTiming {
    Never,
    WinnerOnly,
    DuringExactTrials,
}

// ══════════════════════════════════════════════
// Level 0 — Rough
// ══════════════════════════════════════════════

/// Configuration for the rough SATD-based mode scoring pass.
#[derive(Clone, Copy, Debug)]
pub struct RoughLumaPolicy {
    pub mode_set: RmdModeSet,
    pub score_all_modes: bool,
    pub use_mode_bits: bool,
    pub angular_family_detection: bool,
}

/// Evidence collected during the rough pass.
///
/// The rough pass scores available intra modes by SATD cost and classifies
/// them into families.  Downstream stages consume this evidence to build
/// the shortlist and later the exact promotion set.
#[derive(Clone, Debug)]
pub struct RoughBlockEvidence {
    /// Best-scoring mode overall.
    pub best_global: u8,
    /// Best planar (0) or DC (1) mode, if either beat the angular modes.
    pub best_planar_dc: Option<u8>,
    /// Best angular mode (2–34), if any angular was scored.
    pub best_angular: Option<u8>,
    /// Best angular per family (H/D/V).
    pub best_angular_by_family: [Option<u8>; 3],
    /// Full rough-cost list for all scored modes.
    pub rough_costs: Vec<ModeCost>,
    /// SATD cost of the best mode (used for split/NxN gating).
    pub best_satd: f64,
    /// Block activity / variance measure.
    pub activity: u32,
    /// Source sample range (max − min).
    pub range: u16,
    /// Directional energy strength.
    pub directional_strength: u32,
}

/// A single mode scored during the rough pass.
#[derive(Clone, Copy, Debug)]
pub struct ModeCost {
    pub mode: u8,
    pub cost: f64,
    pub satd: u32,
    pub class: ModeClass,
    pub family: Option<AngularFamily>,
}

/// Classification of an intra mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeClass {
    Planar,
    Dc,
    Angular,
}

// ══════════════════════════════════════════════
// Level 1 — Cheap
// ══════════════════════════════════════════════

/// Configuration for the cheap luma-only trial pass.
#[derive(Clone, Copy, Debug)]
pub struct CheapLumaPolicy {
    pub enabled: bool,
    pub max_ranked_modes: u8,
    pub scope: ComponentScope,
    pub allow_optional_tu_split: bool,
    pub residual_price: ResidualPriceLevel,
    pub quant: TrialQuant,
    /// When true, the cheap pass adds an approximate chroma SATD cost so that
    /// the cheap ranking better predicts the full-RDO winner, matching x265's
    /// leaf-RDO signal and allowing `max_exact_modes` to be set as low as 1.
    pub chroma_satd_in_cheap: bool,
}

/// Result of a cheap (hard-quant, luma-only) trial.
#[derive(Clone, Copy, Debug)]
pub struct CheapMode {
    pub mode: u8,
    pub cost: f64,
    pub luma_cbf: bool,
    pub residual_bits: u64,
    pub distortion: u64,
    pub rough_rank: usize,
}

/// Compact result from a simple-RDO (luma-only, no-TU-split, no-overlay)
/// evaluation. Intended for the Phase 2 lightweight evaluator used in
/// x265shape cheap ranking.
#[derive(Clone, Copy, Debug)]
pub struct SimpleRdoResult {
    pub mode: u8,
    pub cost: f64,
    pub dist: u64,
    pub bits: u64,
    pub cbf: bool,
    pub rough_rank: usize,
}

// ══════════════════════════════════════════════
// Level 2 — Exact
// ══════════════════════════════════════════════

/// Whether the exact TT search pass runs after cheap ranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactUsage {
    /// Skip exact entirely; use the cheap winner's plan directly.
    /// RDOQ finalization recodes the coefficients.
    Disabled,
    /// Run exact only for modes promoted from cheap ranking.
    PromotedModes,
    /// Run exact for the full shortlist (bypass cheap ranking).
    AllShortlist,
    /// x265-shaped Slow audit path: RMD window, simple luma RDO for candidates,
    /// then one split-enabled winner remeasure.
    X265Shape,
}

/// Configuration for the exact RD search pass on promoted modes.
#[derive(Clone, Copy, Debug)]
pub struct ExactLumaPolicy {
    pub max_modes: u8,
    pub scope: ComponentScope,
    pub residual_price: ResidualPriceLevel,
    pub quant: TrialQuant,
    pub promote: ExactPromotionPolicy,
    /// Controls whether the exact pass runs at all after cheap ranking.
    pub exact_usage: ExactUsage,
}

/// Controls which modes are promoted from the shortlist (and optional
/// cheap ranking) into the exact pass.
#[derive(Clone, Copy, Debug)]
pub struct ExactPromotionPolicy {
    pub max_exact_modes: u8,
    pub include_cheap_winner: bool,
    pub include_best_rough_angular_if_pd_wins_cheap: bool,
    pub include_best_rough_pd_if_angular_wins_cheap: bool,
    pub cheap_close_margin: f64,
}

impl ExactPromotionPolicy {
    pub const fn all_shortlist() -> Self {
        Self {
            max_exact_modes: 255,
            include_cheap_winner: false,
            include_best_rough_angular_if_pd_wins_cheap: false,
            include_best_rough_pd_if_angular_wins_cheap: false,
            cheap_close_margin: 1.0,
        }
    }
}

/// The winner of the exact search pass for one CU.
///
/// `Tt` is the transform-tree plan type (e.g. [`crate::plan::TtPlan`]).
/// `Saved` is the detached reconstruction overlay for the block.
#[derive(Clone, Debug)]
pub struct ExactModeWinner<Tt, Saved> {
    pub mode: u8,
    pub tt: Tt,
    pub recon: Saved,
    pub cost: f64,
}

// ══════════════════════════════════════════════
// Level 3 — Final
// ══════════════════════════════════════════════

/// Configuration for the winner-only RDOQ finalization pass.
#[derive(Clone, Copy, Debug)]
pub struct FinalSearchPolicy {
    pub quant: TrialQuant,
    pub residual_price: ResidualPriceLevel,
    pub rdoq_final: bool,
    pub sign_data_hiding: bool,
}

// ══════════════════════════════════════════════
// Pipeline policy structs
// ══════════════════════════════════════════════

/// Controls the representation-aware shortlist built from rough evidence.
#[derive(Clone, Copy, Debug)]
pub struct LumaShortlistPolicy {
    pub max_modes: u8,
    pub include_best_global: bool,
    pub include_best_planar_dc: bool,
    pub include_best_angular: bool,
    pub include_mpm: bool,
    pub angular_family_slots: u8,
    pub angular_neighbor_radius: u8,
}

/// Full luma search funnel policy (rough → shortlist → cheap → exact).
#[derive(Clone, Copy, Debug)]
pub struct LumaSearchPolicy {
    pub rough: RoughLumaPolicy,
    pub shortlist: LumaShortlistPolicy,
    pub cheap: CheapLumaPolicy,
    pub exact: ExactLumaPolicy,
}

/// Transform-tree (TU) search policy.
#[derive(Clone, Copy, Debug)]
pub struct TuSearchPolicy {
    pub min_split_log2: u8,
    pub leaf_first: bool,
    pub split_search: SplitSearch,
    pub zero_residual_early_terminate: bool,
    pub low_residual_early_terminate: bool,
    pub low_residual_bits_per_px: f64,
    pub low_distortion_per_px: f64,
}

/// Coding-unit (CU) search policy.
#[derive(Clone, Copy, Debug)]
pub struct CuSearchPolicy {
    pub split_search: SplitSearch,
    pub early_terminate: CuEarlyTerminateRule,
    pub use_preanalysis_force_leaf: bool,
    pub use_preanalysis_force_split: bool,
    /// Force legal CUs at this parent log2 size to split without evaluating
    /// the 2Nx2N leaf. `Some(6)` mirrors x265's still-image intra path, which
    /// never tests a 64x64 CU leaf at the CTU root.
    pub force_split_log2: Option<u8>,
    pub leaf_first: bool,
    /// Descent-termination gate: after the 2Nx2N leaf is evaluated (and the
    /// residual-based early-termination declined), skip the split branch
    /// entirely when the leaf's RD total is at most
    /// `t * (rd_lambda(qp) / rd_lambda(qp28)) * pixels` — the leaf already
    /// codes the block so cheaply per pixel that a split win is implausible.
    /// Calibrated offline against full leaf-vs-split DSC logs
    /// (`BPG_STILLSEARCH_DESCENT_LOG`, see x265-speed-parity-audit.md §10.5):
    /// at the shipped thresholds the wrongly skipped split wins are <~1% and
    /// their forgone RD is <~0.01% of node RD, while 15-70% of 16x16 and
    /// 1-34% of 32x32 descents are skipped (rising with QP and content
    /// smoothness). `0.0` disables the gate at that size.
    /// `BPG_STILLSEARCH_DESCENT_GATE=t16,t32|off` overrides.
    pub descent_gate_t16: f64,
    pub descent_gate_t32: f64,
}

/// PartNxN search policy.
#[derive(Clone, Copy, Debug)]
pub struct NxnSearchPolicy {
    pub enabled: bool,
    pub rough_gate_enabled: bool,
    pub rough_satd_threshold: f64,
    pub require_directional_or_texture: bool,
    pub exact_eval: bool,
}

/// Chroma mode/timing policy.
#[derive(Clone, Copy, Debug)]
pub struct ChromaSearchPolicy {
    pub timing: ChromaTiming,
    pub max_candidates: u8,
    pub residual_price: ResidualPriceLevel,
    pub quant: TrialQuant,
    /// Number of non-DM chroma modes promoted from the SATD rough pass to exact
    /// RD. 4 preserves exhaustive x265-parity search.
    pub rough_k: u8,
    /// Per-chroma-sample rough SATD threshold below which chroma mode search
    /// keeps DM without exact non-DM trials. 0 disables the skip.
    pub skip_satd_per_sample: f64,
}

/// Pre-analysis steering for per-block decision making.
#[derive(Clone, Copy, Debug)]
pub struct PreanalysisTemplate {
    pub class_steering: bool,
    pub allow_candidate_expansion: bool,
    pub importance_rmd_prune_factor: Option<f64>,
    pub importance_force_leaf: bool,
}

impl PreanalysisTemplate {
    pub const fn oracle_disabled() -> Self {
        Self {
            class_steering: false,
            allow_candidate_expansion: false,
            importance_rmd_prune_factor: None,
            importance_force_leaf: false,
        }
    }
}

// ══════════════════════════════════════════════
// EffortTemplate — composable preset
// ══════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct EffortTemplate {
    pub name: Effort,
    pub oracle: bool,
    pub entropy_context: EntropyContextMode,
    pub parallel_analysis: bool,

    pub luma: LumaSearchPolicy,
    pub tu: TuSearchPolicy,
    pub tu_exact: TuExactPolicy,
    pub cu: CuSearchPolicy,
    pub nxn: NxnSearchPolicy,
    pub chroma: ChromaSearchPolicy,
    pub final_pass: FinalSearchPolicy,
    pub rdoq_trials: RdoqTrialPolicy,

    pub preanalysis: PreanalysisTemplate,
}

impl EffortTemplate {
    /// True if this template is the expensive comparison/oracle tier.
    pub fn is_reference(&self) -> bool {
        matches!(self.name, Effort::Placebo)
    }
}

// ══════════════════════════════════════════════
// Canonical mapping & template selection
// ══════════════════════════════════════════════

/// Map the old wide `Effort` enum to the three canonical presets.
pub fn canonical_effort(e: Effort) -> Effort {
    e
}

/// Return the canonical template for a given effort.
/// All old effort names map to one of FAST / SLOW / PLACEBO.
#[inline]
pub fn template(effort: Effort) -> &'static EffortTemplate {
    match canonical_effort(effort) {
        Effort::Fast => &FAST,
        Effort::Slow => &SLOW,
        Effort::Placebo => &PLACEBO,
    }
}

/// Return the encode-time template for the stable preset.
///
/// Production presets are no longer silently mutated by audit environment
/// variables; diagnostics should be expressed as explicit experiment overlays.
#[inline]
pub fn template_for_encode(effort: Effort) -> EffortTemplate {
    *template(effort)
}

// ══════════════════════════════════════════════
// Three canonical presets
// ══════════════════════════════════════════════

/// Fast — minimalistic robust preset for quick / bulk encodes.
///
/// Same staged pipeline as Slow, but spends work only where
/// evidence is strong.  The cheap winner normally decides exact
/// promotion; a second mode is tried only if it is category-different
/// (planar/DC vs angular) and within 5 % cost margin.
///
/// Target: old minimal-search speed with corrected architecture.
///
/// Do not tune this by trying to match Slow.  Fast is allowed to
/// give up the last 5-10 % of compression — it is designed to avoid
/// search volume, not to chase oracle quality.
///
/// ### Key budgets (from presets2.md)
/// - Shortlist: 4-5 modes, protect best global/angular/planar/MPM, no family diversity.
/// - Exact: 1 mode, rarely 2 (close category-different challenger only).
/// - TU: leaf-first, aggressive zero/low-residual early terminate.
/// - CU: leaf-first, aggressive early terminate, preanalysis force-leaf on smooth.
/// - NxN: strongly gated (directional/texture required, rough SATD ≥ 1000).
/// - Chroma: winner-only, one candidate, approx residual pricing.
/// - Final: RDOQ finalization always on.
pub(crate) static FAST: EffortTemplate = EffortTemplate {
    name: Effort::Fast,
    oracle: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: true,

    luma: LumaSearchPolicy {
        rough: RoughLumaPolicy {
            // Step-4 coarse angular scan + Planar/DC.  The coarse grid
            // (every 4th angular mode) catches directional trends without
            // scoring all 33 angular modes.  MPMs are still added later.
            mode_set: RmdModeSet::Step4,
            score_all_modes: true,
            use_mode_bits: true,
            angular_family_detection: true,
        },
        shortlist: LumaShortlistPolicy {
            max_modes: 5,
            include_best_global: true,
            include_best_planar_dc: true,
            include_best_angular: true,
            include_mpm: true,
            angular_family_slots: 0,
            angular_neighbor_radius: 0,
        },
        cheap: CheapLumaPolicy {
            enabled: true,
            max_ranked_modes: 4,
            scope: ComponentScope::LumaOnly,
            allow_optional_tu_split: false,
            residual_price: ResidualPriceLevel::Approx,
            quant: TrialQuant::HardQuant,
            chroma_satd_in_cheap: false,
        },
        exact: ExactLumaPolicy {
            max_modes: 2,
            scope: ComponentScope::LumaOnly,
            residual_price: ResidualPriceLevel::Approx,
            quant: TrialQuant::HardQuant,
            promote: ExactPromotionPolicy {
                max_exact_modes: 1,
                include_cheap_winner: true,
                include_best_rough_angular_if_pd_wins_cheap: false,
                include_best_rough_pd_if_angular_wins_cheap: false,
                cheap_close_margin: 1.05,
            },
            exact_usage: ExactUsage::Disabled,
        },
    },

    tu: TuSearchPolicy {
        min_split_log2: 3,
        leaf_first: true,
        split_search: SplitSearch::PreferLeaf,
        zero_residual_early_terminate: true,
        low_residual_early_terminate: true,
        low_residual_bits_per_px: 0.05,
        low_distortion_per_px: 1.0,
    },

    tu_exact: TuExactPolicy {
        split_mode: TuSplitMode::LeafFirstEarlyTerminate,
        max_extra_depth: 1,
        min_split_log2: 3,
        leaf_first: true,
        zero_residual_early_terminate: true,
        low_residual_early_terminate: true,
        low_residual_bits_per_px: 0.05,
        low_distortion_per_px: 1.0,
        rdoq_split_trials: false,
    },

    cu: CuSearchPolicy {
        split_search: SplitSearch::PreferLeaf,
        early_terminate: CuEarlyTerminateRule::Fast,
        use_preanalysis_force_leaf: true,
        use_preanalysis_force_split: false,
        force_split_log2: None,
        leaf_first: true,
        descent_gate_t16: 0.0,
        descent_gate_t32: 0.0,
    },

    nxn: NxnSearchPolicy {
        enabled: true,
        rough_gate_enabled: true,
        rough_satd_threshold: 1000.0,
        require_directional_or_texture: true,
        exact_eval: true,
    },

    chroma: ChromaSearchPolicy {
        timing: ChromaTiming::WinnerOnly,
        max_candidates: 1,
        residual_price: ResidualPriceLevel::Approx,
        quant: TrialQuant::HardQuant,
        rough_k: 1,
        skip_satd_per_sample: 0.0,
    },

    final_pass: FinalSearchPolicy {
        quant: TrialQuant::Rdoq,
        residual_price: ResidualPriceLevel::Exact,
        rdoq_final: true,
        sign_data_hiding: true,
    },

    rdoq_trials: RdoqTrialPolicy {
        mode: TrialRdoqMode::Off,
        close_margin: 1.00,
        max_rdoq_modes: 0,
        level: 0,
    },

    preanalysis: PreanalysisTemplate {
        class_steering: true,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: Some(0.8),
        importance_force_leaf: true,
    },
};

/// Slow — main quality / archival preset.
///
/// This is the measured x265shape search budget: one stable pipeline, precise
/// enough for production uniform-QP encodes, without requiring diagnostic env
/// routing.
pub(crate) static SLOW: EffortTemplate = EffortTemplate {
    name: Effort::Slow,
    oracle: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: true,

    luma: LumaSearchPolicy {
        rough: RoughLumaPolicy {
            mode_set: RmdModeSet::Exhaustive,
            score_all_modes: true,
            use_mode_bits: true,
            angular_family_detection: true,
        },
        shortlist: LumaShortlistPolicy {
            max_modes: 8,
            include_best_global: true,
            include_best_planar_dc: true,
            include_best_angular: true,
            include_mpm: true,
            angular_family_slots: 2,
            angular_neighbor_radius: 1,
        },
        cheap: CheapLumaPolicy {
            enabled: true,
            max_ranked_modes: 0,
            scope: ComponentScope::LumaOnly,
            allow_optional_tu_split: false,
            residual_price: ResidualPriceLevel::Exact,
            quant: TrialQuant::Rdoq,
            chroma_satd_in_cheap: false,
        },
        exact: ExactLumaPolicy {
            max_modes: 4,
            scope: ComponentScope::LumaOnly,
            residual_price: ResidualPriceLevel::Exact,
            quant: TrialQuant::Rdoq,
            promote: ExactPromotionPolicy {
                max_exact_modes: 1,
                include_cheap_winner: true,
                include_best_rough_angular_if_pd_wins_cheap: false,
                include_best_rough_pd_if_angular_wins_cheap: false,
                cheap_close_margin: 1.10,
            },
            exact_usage: ExactUsage::X265Shape,
        },
    },

    tu: TuSearchPolicy {
        min_split_log2: 2,
        leaf_first: true,
        split_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: true,
        low_residual_early_terminate: true,
        low_residual_bits_per_px: 0.02,
        low_distortion_per_px: 0.5,
    },

    tu_exact: TuExactPolicy {
        split_mode: TuSplitMode::EvaluateBoth,
        max_extra_depth: 2, // Must not exceed MAX_INTRA_TT_DEPTH (2) — the write path uses that constant.
        min_split_log2: 2,
        leaf_first: true,
        zero_residual_early_terminate: true,
        low_residual_early_terminate: true,
        low_residual_bits_per_px: 0.015,
        low_distortion_per_px: 0.35,
        rdoq_split_trials: false,
    },

    cu: CuSearchPolicy {
        split_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Balanced,
        use_preanalysis_force_leaf: true,
        use_preanalysis_force_split: true,
        // x265 placebo/`bpgenc -m9` always splits the CTU root: its intra
        // analysis never evaluates a 64x64 2Nx2N leaf. Slow is the production
        // x265-shaped preset, so skip that extra depth by default. Set
        // `BPG_STILLSEARCH_CU_FORCE_SPLIT_LOG2=0` to restore the old search.
        force_split_log2: Some(6),
        leaf_first: true,
        descent_gate_t16: 4.0,
        descent_gate_t32: 1.0,
    },

    nxn: NxnSearchPolicy {
        enabled: true,
        rough_gate_enabled: true,
        rough_satd_threshold: 1000.0,
        require_directional_or_texture: false,
        exact_eval: true,
    },

    chroma: ChromaSearchPolicy {
        timing: ChromaTiming::WinnerOnly,
        max_candidates: 2,
        residual_price: ResidualPriceLevel::Exact,
        quant: TrialQuant::Rdoq,
        rough_k: 1,
        skip_satd_per_sample: 0.0,
    },

    final_pass: FinalSearchPolicy {
        quant: TrialQuant::Rdoq,
        residual_price: ResidualPriceLevel::Exact,
        rdoq_final: true,
        sign_data_hiding: true,
    },

    rdoq_trials: RdoqTrialPolicy {
        mode: TrialRdoqMode::ExactOnly,
        close_margin: 1.00,
        max_rdoq_modes: 35,
        level: 2,
    },

    preanalysis: PreanalysisTemplate {
        class_steering: true,
        allow_candidate_expansion: true,
        importance_rmd_prune_factor: None,
        importance_force_leaf: false,
    },
};

/// Placebo — oracle / reference preset.
///
/// Answers the question: "What would the staged architecture choose
/// if pruning were mostly disabled?"  Not intended for normal encode
/// speed.  Used to validate Slow and measure what pruning misses.
///
/// All pruning mechanisms are disabled through policy (not through
/// separate code paths) so Slow and Fast can be validated against it.
///
/// ### Key budgets (from presets2.md)
/// - Shortlist: 12+ modes, full representation + family diversity.
/// - Cheap: disabled (diagnostic only).
/// - Exact: all shortlisted modes, full components (luma + chroma).
/// - TU: leaf-first OK but no early termination; split evaluated when legal.
/// - CU: no early termination; leaf and split both evaluated when legal.
/// - NxN: evaluated whenever legal.
/// - Chroma: during exact trials.
/// - Final: RDOQ finalization always on.
pub(crate) static PLACEBO: EffortTemplate = EffortTemplate {
    name: Effort::Placebo,
    oracle: true,
    entropy_context: EntropyContextMode::FrozenSliceInit,
    parallel_analysis: true,

    luma: LumaSearchPolicy {
        rough: RoughLumaPolicy {
            mode_set: RmdModeSet::Exhaustive,
            score_all_modes: true,
            use_mode_bits: true,
            angular_family_detection: true,
        },
        shortlist: LumaShortlistPolicy {
            max_modes: 12,
            include_best_global: true,
            include_best_planar_dc: true,
            include_best_angular: true,
            include_mpm: true,
            angular_family_slots: 3,
            angular_neighbor_radius: 2,
        },
        cheap: CheapLumaPolicy {
            enabled: false,
            max_ranked_modes: 0,
            scope: ComponentScope::LumaOnly,
            allow_optional_tu_split: false,
            residual_price: ResidualPriceLevel::Approx,
            quant: TrialQuant::Rdoq,
            chroma_satd_in_cheap: false,
        },
        exact: ExactLumaPolicy {
            max_modes: 12,
            scope: ComponentScope::FullComponents,
            residual_price: ResidualPriceLevel::Exact,
            quant: TrialQuant::Rdoq,
            promote: ExactPromotionPolicy::all_shortlist(),
            exact_usage: ExactUsage::AllShortlist,
        },
    },

    tu: TuSearchPolicy {
        min_split_log2: 2,
        leaf_first: true,
        split_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
        low_residual_early_terminate: false,
        low_residual_bits_per_px: 0.0,
        low_distortion_per_px: 0.0,
    },

    tu_exact: TuExactPolicy {
        split_mode: TuSplitMode::EvaluateBoth,
        max_extra_depth: 2,
        min_split_log2: 2,
        leaf_first: true,
        zero_residual_early_terminate: false,
        low_residual_early_terminate: false,
        low_residual_bits_per_px: 0.0,
        low_distortion_per_px: 0.0,
        rdoq_split_trials: true,
    },

    cu: CuSearchPolicy {
        split_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
        use_preanalysis_force_leaf: false,
        use_preanalysis_force_split: false,
        force_split_log2: None,
        leaf_first: true,
        descent_gate_t16: 0.0,
        descent_gate_t32: 0.0,
    },

    nxn: NxnSearchPolicy {
        enabled: true,
        rough_gate_enabled: false,
        rough_satd_threshold: 0.0,
        require_directional_or_texture: false,
        exact_eval: true,
    },

    chroma: ChromaSearchPolicy {
        timing: ChromaTiming::DuringExactTrials,
        max_candidates: 3,
        residual_price: ResidualPriceLevel::Exact,
        quant: TrialQuant::Rdoq,
        rough_k: 4,
        skip_satd_per_sample: 0.0,
    },

    final_pass: FinalSearchPolicy {
        quant: TrialQuant::Rdoq,
        residual_price: ResidualPriceLevel::Exact,
        rdoq_final: true,
        sign_data_hiding: true,
    },

    rdoq_trials: RdoqTrialPolicy {
        mode: TrialRdoqMode::Off,
        close_margin: 999.0,
        max_rdoq_modes: 0,
        level: 1,
    },

    preanalysis: PreanalysisTemplate::oracle_disabled(),
};

// ══════════════════════════════════════════════
// Constants
// ══════════════════════════════════════════════

const FORCE_LEAF_IMPORTANCE: u16 = 24;
const FORCE_SPLIT_IMPORTANCE: u16 = 224;
const CLOSE_CALL_MARGIN: f64 = 0.02;

// ══════════════════════════════════════════════
// Policy resolution (template → BlockSearchBudget)
// ══════════════════════════════════════════════

impl EffortTemplate {
    pub fn resolve_policy(&self, region: RegionClass, importance_q8: u16) -> SearchPolicy {
        let preanalysis = self.preanalysis;
        if self.is_reference() {
            return crate::preanalysis::INERT;
        }

        let mut policy = if preanalysis.class_steering {
            policy_for(region)
        } else {
            crate::preanalysis::INERT
        };

        if !preanalysis.allow_candidate_expansion {
            policy.luma_rd_bias = policy.luma_rd_bias.min(0);
            policy.chroma_rd_bias = policy.chroma_rd_bias.min(0);
        }

        policy.rmd_prune_factor = preanalysis.importance_rmd_prune_factor;
        if preanalysis.importance_force_leaf && importance_q8 < FORCE_LEAF_IMPORTANCE {
            policy.force_leaf = true;
        }
        if matches!(region, RegionClass::TextLike) && importance_q8 >= FORCE_SPLIT_IMPORTANCE {
            policy.force_split = true;
        }

        policy
    }

    pub fn resolve(&self, desc: BlockDesc) -> BlockSearchBudget {
        let policy = desc.policy;
        let effective = qp_budget_effort(self.name, desc.qp);
        let effective_template = template(effective);
        let rmd_mode_set = effective_template.luma.rough.mode_set;
        let base_luma = self.luma_rd_candidates_base(desc.log2_size, desc.qp);
        let luma_rd_cap = match canonical_effort(self.name) {
            Effort::Placebo => self.luma.exact.max_modes,
            Effort::Slow => 8,
            _ => 4,
        };
        let luma_rd_candidates =
            (base_luma as i8 + policy.luma_rd_bias).clamp(1, luma_rd_cap as i8) as u8;
        let base_chroma = self.chroma_rd_candidates_base(desc.qp);
        let chroma_min = if base_chroma == 0 { 0 } else { 1 };
        let chroma_rd_candidates =
            (base_chroma as i8 + policy.chroma_rd_bias).clamp(chroma_min, 5) as u8;
        let split_policy = if policy.force_leaf {
            SplitSearch::ForceLeaf
        } else if policy.force_split {
            SplitSearch::ForceSplit
        } else if policy.prefer_split {
            SplitSearch::PreferSplit
        } else {
            self.cu.split_search
        };

        let canonical = canonical_effort(self.name);
        let zero_residual_tu_early_terminate = match canonical {
            Effort::Fast => true,
            Effort::Slow => desc.qp >= 38,
            Effort::Placebo => false,
        };

        let tu_split = if canonical == Effort::Placebo {
            self.tu.split_search
        } else if zero_residual_tu_early_terminate {
            SplitSearch::PreferLeaf
        } else if matches!(
            split_policy,
            SplitSearch::ForceSplit | SplitSearch::PreferSplit
        ) {
            split_policy
        } else {
            self.tu.split_search
        };

        BlockSearchBudget {
            effort: self.name,
            angular_prune: policy.angular_prune,
            angular_prune_var_threshold_8bit: angular_prune_var_threshold_8bit(self.name),
            rmd_mode_set,
            luma_rd_candidates_base: base_luma,
            luma_rd_candidates,
            chroma_rd_candidates_base: base_chroma,
            chroma_rd_candidates,
            luma_trial_quality: if canonical_effort(self.name) == Effort::Placebo {
                TrialQuality::Final
            } else if canonical_effort(self.name) == Effort::Slow
                && self.luma.exact.residual_price == ResidualPriceLevel::Exact
            {
                TrialQuality::FullRd
            } else {
                TrialQuality::FastRd
            },
            chroma_trial_quality: if canonical_effort(self.name) == Effort::Placebo {
                TrialQuality::Final
            } else {
                TrialQuality::FastRd
            },
            final_quality: TrialQuality::Final,
            cu_split: split_policy,
            tu_split,
            close_call_margin: CLOSE_CALL_MARGIN,
            exact_residual_bits_for_trials: matches!(
                effective_template.luma.exact.residual_price,
                ResidualPriceLevel::Exact
            ),
            rdoq_for_trials: false,
            rdoq_pass_budget: RdoqPassBudget {
                effort: self.name,
                qp: desc.qp,
                canonical: canonical_effort(self.name),
            },
            cu_early_terminate: self.cu.early_terminate,
            zero_residual_tu_early_terminate,
            rmd_prune_factor: policy.rmd_prune_factor,
            allow_cu_early_terminate: policy.allow_early_term,
        }
    }

    fn luma_rd_candidates_base(&self, log2_size: u8, qp: i32) -> u8 {
        if self.is_reference() {
            return self.luma.exact.max_modes;
        }
        match canonical_effort(self.name) {
            Effort::Fast => {
                if log2_size >= 4 {
                    self.luma.exact.max_modes
                } else {
                    1
                }
            }
            Effort::Slow => {
                if qp >= 44 {
                    1
                } else if qp >= 38 {
                    2
                } else {
                    self.luma.exact.max_modes
                }
            }
            Effort::Placebo => self.luma.exact.max_modes,
        }
    }

    fn chroma_rd_candidates_base(&self, qp: i32) -> u8 {
        match canonical_effort(self.name) {
            Effort::Fast => self.chroma.max_candidates,
            Effort::Slow => {
                if qp >= 44 {
                    1
                } else if qp >= 38 {
                    2
                } else {
                    self.chroma.max_candidates
                }
            }
            Effort::Placebo => self.chroma.max_candidates,
        }
    }
}

// ──────────────────────────────────────────────
// QP-dependent effort downgrade
// ──────────────────────────────────────────────

/// Apply QP-dependent effort downgrading for per-CU budget resolution.
#[inline]
fn qp_budget_effort(effort: Effort, qp: i32) -> Effort {
    if is_reference_tier(effort) {
        return effort;
    }
    match (canonical_effort(effort), qp) {
        (Effort::Slow, q) if q >= 44 => Effort::Fast,
        (Effort::Slow, q) if q >= 38 => Effort::Fast,
        (Effort::Fast, q) if q >= 38 => Effort::Fast,
        _ => effort,
    }
}

#[inline]
fn angular_prune_var_threshold_8bit(effort: Effort) -> Option<i64> {
    match canonical_effort(effort) {
        Effort::Fast => Some(32),
        Effort::Slow => None,
        Effort::Placebo => None,
    }
}

// ──────────────────────────────────────────────
// policy_for() — preanalysis region-to-policy
// ──────────────────────────────────────────────

fn policy_for(class: RegionClass) -> SearchPolicy {
    match class {
        RegionClass::Flat | RegionClass::Gradient => SearchPolicy {
            angular_prune: Some(true),
            allow_early_term: true,
            luma_rd_bias: -1,
            chroma_rd_bias: 0,
            ..crate::preanalysis::INERT
        },
        RegionClass::DirectionalEdge => SearchPolicy {
            angular_prune: Some(false),
            allow_early_term: false,
            luma_rd_bias: 0,
            chroma_rd_bias: 0,
            prefer_split: true,
            ..crate::preanalysis::INERT
        },
        RegionClass::TextLike => SearchPolicy {
            angular_prune: Some(false),
            allow_early_term: false,
            luma_rd_bias: 1,
            chroma_rd_bias: 0,
            prefer_split: true,
            ..crate::preanalysis::INERT
        },
        RegionClass::Texture => SearchPolicy {
            angular_prune: None,
            allow_early_term: true,
            luma_rd_bias: 0,
            chroma_rd_bias: 0,
            ..crate::preanalysis::INERT
        },
        RegionClass::Noisy => SearchPolicy {
            angular_prune: Some(true),
            allow_early_term: true,
            luma_rd_bias: 0,
            chroma_rd_bias: 0,
            ..crate::preanalysis::INERT
        },
        RegionClass::ChromaCritical => SearchPolicy {
            angular_prune: None,
            allow_early_term: true,
            luma_rd_bias: 0,
            chroma_rd_bias: 1,
            ..crate::preanalysis::INERT
        },
    }
}

// ══════════════════════════════════════════════
// Legacy compatibility wrappers
// ══════════════════════════════════════════════

/// Quality/fidelity of a trial evaluation.
///
/// Kept for backward compat with `BlockPlan` and `eval.rs`; the new template
/// system controls trial quality through `TrialQuant` + `ResidualPriceLevel`
/// instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialQuality {
    Rough,
    FastRd,
    FullRd,
    Final,
}

/// Resolved per-block search budget.
///
/// Produced by [`EffortTemplate::resolve`] and consumed by the stillsearch
/// engine.  This is a legacy compatibility type — new code should read the
/// template fields directly.
#[derive(Clone, Copy, Debug)]
pub struct BlockSearchBudget {
    pub effort: Effort,
    pub angular_prune: Option<bool>,
    pub angular_prune_var_threshold_8bit: Option<i64>,
    pub rmd_mode_set: RmdModeSet,
    pub luma_rd_candidates_base: u8,
    pub luma_rd_candidates: u8,
    pub chroma_rd_candidates_base: u8,
    pub chroma_rd_candidates: u8,
    pub luma_trial_quality: TrialQuality,
    pub chroma_trial_quality: TrialQuality,
    pub final_quality: TrialQuality,
    pub cu_split: SplitSearch,
    pub tu_split: SplitSearch,
    pub close_call_margin: f64,
    pub exact_residual_bits_for_trials: bool,
    pub rdoq_for_trials: bool,
    pub rdoq_pass_budget: RdoqPassBudget,
    pub cu_early_terminate: CuEarlyTerminateRule,
    pub zero_residual_tu_early_terminate: bool,
    pub rmd_prune_factor: Option<f64>,
    pub allow_cu_early_terminate: bool,
}

impl BlockSearchBudget {
    #[inline]
    pub fn rough_luma_modes(self, mpm: [u8; 3], prune_angular: bool) -> Vec<u8> {
        if prune_angular {
            RmdModeSet::MpmPlanarDcOnly.modes(mpm)
        } else {
            self.rmd_mode_set.modes(mpm)
        }
    }

    #[inline]
    pub fn rdoq_passes(self, log2_size: u8, component: ComponentKind, nnz: u32) -> u8 {
        self.rdoq_pass_budget.passes(log2_size, component, nnz)
    }

    #[inline]
    pub fn tu_split_early_terminate(self, luma_cbf: bool) -> bool {
        !luma_cbf && self.zero_residual_tu_early_terminate
    }

    pub fn should_early_terminate_cu(
        self,
        has_residual: bool,
        leaf_bits_whole: u64,
        log2_cb_size: u8,
        search_qp: i32,
        bit_depth: u8,
        source_luma_range: impl FnOnce() -> u16,
    ) -> bool {
        match self.cu_early_terminate {
            CuEarlyTerminateRule::Disabled => false,
            CuEarlyTerminateRule::Balanced => {
                if !has_residual {
                    return true;
                }
                if search_qp < 38 {
                    return false;
                }
                let bits = leaf_bits_whole;
                let area = 1u64 << (2 * log2_cb_size as u64);
                let k = if search_qp >= 44 { 4 } else { 8 };
                bits * k < area
            }
            CuEarlyTerminateRule::Fast => {
                if !has_residual {
                    return true;
                }
                let bits = leaf_bits_whole;
                let area = 1u64 << (2 * log2_cb_size as u64);
                let k = if search_qp >= 44 {
                    2
                } else if search_qp >= 38 {
                    3
                } else {
                    4
                };
                bits * k < area
            }
            CuEarlyTerminateRule::Fastest => {
                if !has_residual {
                    return true;
                }
                let bits = leaf_bits_whole;
                let area = 1u64 << (2 * log2_cb_size as u64);
                let k = if search_qp >= 44 { 1 } else { 2 };
                if bits * k < area {
                    return true;
                }
                let r_th_8: u16 = if search_qp >= 44 {
                    80
                } else if search_qp >= 38 {
                    64
                } else {
                    48
                };
                let r_th = r_th_8 << (bit_depth - 8);
                source_luma_range() < r_th
            }
        }
    }
}

/// RDOQ pass budget per-block.
#[derive(Clone, Copy, Debug)]
pub struct RdoqPassBudget {
    effort: Effort,
    qp: i32,
    pub(crate) canonical: Effort,
}

impl RdoqPassBudget {
    pub fn passes(self, log2_size: u8, _component: ComponentKind, nnz: u32) -> u8 {
        match self.canonical {
            Effort::Fast => 0,
            Effort::Slow => {
                if self.qp < 38 {
                    if log2_size <= 4 && nnz <= 64 { 1 } else { 0 }
                } else if self.qp >= 44 {
                    0
                } else if log2_size <= 3 && nnz <= 32 {
                    1
                } else {
                    0
                }
            }
            Effort::Placebo => {
                if log2_size <= 4 && nnz <= 128 {
                    2
                } else {
                    1
                }
            }
        }
    }
}

// ══════════════════════════════════════════════
// Debug display
// ══════════════════════════════════════════════

pub fn describe_effort(effort: Effort, qp: i32) -> EffortDescription {
    let t = template(effort);
    let texture = t.resolve(describe_block_desc(t, qp, RegionClass::Texture, 256));
    let flat_low_importance = t.resolve(describe_block_desc(t, qp, RegionClass::Flat, 0));
    EffortDescription {
        template: t,
        original_effort: effort,
        texture,
        flat_low_importance,
        qp,
    }
}

fn describe_block_desc(
    template: &EffortTemplate,
    qp: i32,
    region: RegionClass,
    importance_q8: u16,
) -> BlockDesc {
    BlockDesc {
        x: 0,
        y: 0,
        log2_size: 5,
        qp,
        region,
        importance_q8,
        component: ComponentKind::Luma,
        policy: template.resolve_policy(region, importance_q8),
    }
}

/// Per-block descriptor fed into the budget resolution logic.
#[derive(Clone, Copy, Debug)]
pub struct BlockDesc {
    pub x: u32,
    pub y: u32,
    pub log2_size: u8,
    pub qp: i32,
    pub region: RegionClass,
    pub importance_q8: u16,
    pub component: ComponentKind,
    pub policy: SearchPolicy,
}

pub struct EffortDescription {
    template: &'static EffortTemplate,
    original_effort: Effort,
    texture: BlockSearchBudget,
    flat_low_importance: BlockSearchBudget,
    qp: i32,
}

impl fmt::Display for EffortDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "effort: {:?}  entropy: {:?}  parallel_analysis: {}",
            self.original_effort, self.template.entropy_context, self.template.parallel_analysis
        )?;
        writeln!(f, "  neutral texture budget:")?;
        write_budget(f, self.qp, self.texture)?;
        writeln!(f, "  flat low-importance budget:")?;
        write_budget(f, self.qp, self.flat_low_importance)
    }
}

fn write_budget(f: &mut fmt::Formatter<'_>, qp: i32, budget: BlockSearchBudget) -> fmt::Result {
    writeln!(
        f,
        "    rmd: {:?}  luma_rd_candidates@qp{}: {}  chroma_rd_candidates@qp{}: {}",
        budget.rmd_mode_set, qp, budget.luma_rd_candidates, qp, budget.chroma_rd_candidates
    )?;
    writeln!(
        f,
        "    trial_quality: luma={:?} chroma={:?} final={:?}",
        budget.luma_trial_quality, budget.chroma_trial_quality, budget.final_quality
    )?;
    writeln!(
        f,
        "    split: cu={:?} tu={:?}  residual: rdoq_trials={} exact_bits_trials={}",
        budget.cu_split,
        budget.tu_split,
        budget.rdoq_for_trials,
        budget.exact_residual_bits_for_trials
    )?;
    writeln!(
        f,
        "    angular_prune={:?} rmd_prune_factor={:?} allow_cu_early_term={}",
        budget.angular_prune, budget.rmd_prune_factor, budget.allow_cu_early_terminate
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_slow_is_x265shape() {
        assert_eq!(SLOW.luma.rough.mode_set, RmdModeSet::Exhaustive);
        assert!(SLOW.luma.rough.score_all_modes);
        assert_eq!(SLOW.luma.cheap.max_ranked_modes, 0);
        assert_eq!(SLOW.luma.cheap.residual_price, ResidualPriceLevel::Exact);
        assert_eq!(SLOW.luma.exact.exact_usage, ExactUsage::X265Shape);
        assert_eq!(SLOW.luma.exact.promote.max_exact_modes, 1);
        assert!(!SLOW.luma.cheap.chroma_satd_in_cheap);
        assert!(
            !SLOW
                .luma
                .exact
                .promote
                .include_best_rough_angular_if_pd_wins_cheap
        );
        assert!(
            !SLOW
                .luma
                .exact
                .promote
                .include_best_rough_pd_if_angular_wins_cheap
        );
        assert_eq!(SLOW.nxn.rough_satd_threshold, 1000.0);
        assert_eq!(SLOW.rdoq_trials.mode, TrialRdoqMode::ExactOnly);
        assert_eq!(SLOW.rdoq_trials.max_rdoq_modes, 35);
        assert_eq!(SLOW.rdoq_trials.level, 2);
    }
}
