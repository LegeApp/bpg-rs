//! Centralized search-effort templates and per-block budget resolution.
//!
//! This module is deliberately data-first: tiers describe search quantities and
//! policies here, while the encoder consumes a resolved [`BlockSearchBudget`].
//! Later decision-plan work can add cheaper trial implementations without
//! scattering new `match Effort` branches through CU/TU/mode coding.

use std::fmt;

use crate::preanalysis::{RegionClass, SearchPolicy};
use crate::{is_reference_tier, Effort};

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
    pub fn modes(self, mpm: [u8; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(35);
        out.extend_from_slice(&mpm);
        out.extend_from_slice(&[0, 1]);

        match self {
            RmdModeSet::MpmPlanarDcOnly => {}
            RmdModeSet::Step4 => out.extend_from_slice(&[2, 6, 10, 14, 18, 22, 26, 30, 34]),
            RmdModeSet::Step3 => {
                out.extend_from_slice(&[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34])
            }
            RmdModeSet::Step2 => out.extend_from_slice(&[
                2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
            ]),
            RmdModeSet::Dense => out.extend_from_slice(&[
                2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                28, 30, 32, 34,
            ]),
            RmdModeSet::Exhaustive => out.extend(0..=34),
            RmdModeSet::Progressive { .. } => {
                // Dynamic progressive scoring is implemented in the encoder,
                // where rough SATD scores are available. This static fallback is
                // used only by generic callers.
                out.extend_from_slice(&[
                    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
                ]);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrialQuality {
    Rough,
    FastRd,
    FullRd,
    Final,
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
    #[inline]
    pub fn from_c_idx(c_idx: u8) -> Self {
        match c_idx {
            0 => ComponentKind::Luma,
            1 => ComponentKind::ChromaCb,
            2 => ComponentKind::ChromaCr,
            _ => unreachable!("component index must be 0, 1, or 2"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RmdTemplate {
    pub mode_set: RmdModeSet,
    pub max_luma_rd_candidates: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct LumaSearchTemplate {
    pub trial_quality: TrialQuality,
    pub chroma_during_luma_trials: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ChromaSearchTemplate {
    pub trial_quality: TrialQuality,
    pub max_rd_candidates: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct ResidualSearchTemplate {
    pub rdoq_during_trials: bool,
    pub exact_bits_during_trials: bool,
    pub final_quality: TrialQuality,
}

#[derive(Clone, Copy, Debug)]
pub struct CuSplitTemplate {
    pub default_search: SplitSearch,
    pub early_terminate: CuEarlyTerminateRule,
}

#[derive(Clone, Copy, Debug)]
pub struct TuSplitTemplate {
    pub default_search: SplitSearch,
    pub zero_residual_early_terminate: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PreanalysisTemplate {
    pub class_steering: bool,
    pub allow_candidate_expansion: bool,
    pub importance_rmd_prune_factor: Option<f64>,
    pub importance_force_leaf: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct EffortTemplate {
    pub name: Effort,
    pub reference: bool,
    pub entropy_context: EntropyContextMode,
    pub parallel_analysis: bool,
    pub rmd: RmdTemplate,
    pub luma: LumaSearchTemplate,
    pub chroma: ChromaSearchTemplate,
    pub residual: ResidualSearchTemplate,
    pub cu_split: CuSplitTemplate,
    pub tu_split: TuSplitTemplate,
    pub preanalysis: PreanalysisTemplate,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CuEarlyTerminateRule {
    Disabled,
    Balanced,
    Fast,
    Fastest,
}

#[derive(Clone, Copy, Debug)]
pub struct RdoqPassBudget {
    effort: Effort,
    qp: i32,
}

impl RdoqPassBudget {
    pub fn passes(self, log2_size: u8, _component: ComponentKind, nnz: u32) -> u8 {
        match self.effort {
            Effort::Fastest | Effort::Fast => 0,
            Effort::Balanced => {
                if self.qp < 38 && log2_size <= 3 && nnz <= 32 {
                    1
                } else {
                    0
                }
            }
            Effort::Good => {
                if self.qp >= 44 {
                    0
                } else if self.qp >= 38 {
                    if log2_size <= 3 && nnz <= 32 {
                        1
                    } else {
                        0
                    }
                } else if log2_size <= 4 && nnz <= 64 {
                    1
                } else {
                    0
                }
            }
            Effort::Best | Effort::Placebo | Effort::Reference => {
                if log2_size <= 4 && nnz <= 128 {
                    2
                } else {
                    1
                }
            }
        }
    }
}

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

impl EffortTemplate {
    pub fn resolve_policy(&self, region: RegionClass, importance_q8: u16) -> SearchPolicy {
        let preanalysis = self.preanalysis;
        if self.reference {
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
        let rmd_mode_set = effective_template.rmd.mode_set;
        let base_luma = self.luma_rd_candidates_base(desc.log2_size, desc.qp);
        let luma_rd_cap = if self.name == Effort::Best { 9 } else { 4 };
        let luma_rd_candidates =
            (base_luma as i8 + policy.luma_rd_bias).clamp(1, luma_rd_cap) as u8;
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
            self.cu_split.default_search
        };
        let zero_residual_tu_early_terminate = match self.name {
            Effort::Fastest | Effort::Fast => true,
            Effort::Balanced => desc.qp >= 38,
            Effort::Good | Effort::Best | Effort::Placebo | Effort::Reference => false,
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
            luma_trial_quality: effective_template.luma.trial_quality,
            chroma_trial_quality: effective_template.chroma.trial_quality,
            final_quality: effective_template.residual.final_quality,
            cu_split: split_policy,
            tu_split: if self.name == Effort::Fastest && desc.log2_size <= 4 {
                SplitSearch::ForceLeaf
            } else if zero_residual_tu_early_terminate {
                SplitSearch::PreferLeaf
            } else if matches!(
                split_policy,
                SplitSearch::ForceSplit | SplitSearch::PreferSplit
            ) {
                split_policy
            } else {
                self.tu_split.default_search
            },
            close_call_margin: CLOSE_CALL_MARGIN,
            exact_residual_bits_for_trials: effective_template.residual.exact_bits_during_trials,
            rdoq_for_trials: effective_template.residual.rdoq_during_trials,
            rdoq_pass_budget: RdoqPassBudget {
                effort: self.name,
                qp: desc.qp,
            },
            cu_early_terminate: self.cu_split.early_terminate,
            zero_residual_tu_early_terminate,
            rmd_prune_factor: policy.rmd_prune_factor,
            allow_cu_early_terminate: policy.allow_early_term,
        }
    }

    fn luma_rd_candidates_base(&self, log2_size: u8, qp: i32) -> u8 {
        if self.reference {
            return self.rmd.max_luma_rd_candidates;
        }

        match self.name {
            Effort::Fast => {
                if log2_size >= 4 {
                    self.rmd.max_luma_rd_candidates
                } else {
                    1
                }
            }
            Effort::Balanced => {
                if qp >= 38 {
                    1
                } else {
                    self.rmd.max_luma_rd_candidates
                }
            }
            Effort::Good => {
                if qp >= 44 {
                    1
                } else if qp >= 38 {
                    2
                } else {
                    self.rmd.max_luma_rd_candidates
                }
            }
            Effort::Fastest | Effort::Best | Effort::Placebo | Effort::Reference => {
                self.rmd.max_luma_rd_candidates
            }
        }
    }

    fn chroma_rd_candidates_base(&self, qp: i32) -> u8 {
        match self.name {
            Effort::Good => {
                if qp >= 44 {
                    1
                } else if qp >= 38 {
                    2
                } else {
                    self.chroma.max_rd_candidates
                }
            }
            Effort::Fastest
            | Effort::Fast
            | Effort::Balanced
            | Effort::Best
            | Effort::Placebo
            | Effort::Reference => self.chroma.max_rd_candidates,
        }
    }
}

const FORCE_LEAF_IMPORTANCE: u16 = 24;
const FORCE_SPLIT_IMPORTANCE: u16 = 224;
const CLOSE_CALL_MARGIN: f64 = 0.02;

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

static FASTEST: EffortTemplate = EffortTemplate {
    name: Effort::Fastest,
    reference: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Progressive {
            coarse_step: 8,
            top_regions: 1,
            refine_radius: 1,
        },
        max_luma_rd_candidates: 1,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        max_rd_candidates: 0,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: false,
        exact_bits_during_trials: false,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Fastest,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: true,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: false,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: Some(1.10),
        importance_force_leaf: true,
    },
};

static FAST: EffortTemplate = EffortTemplate {
    name: Effort::Fast,
    reference: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Progressive {
            coarse_step: 4,
            top_regions: 1,
            refine_radius: 1,
        },
        max_luma_rd_candidates: 2,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        max_rd_candidates: 1,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: false,
        exact_bits_during_trials: false,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Fast,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: true,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: true,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: Some(1.10),
        importance_force_leaf: true,
    },
};

static BALANCED: EffortTemplate = EffortTemplate {
    name: Effort::Balanced,
    reference: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Progressive {
            coarse_step: 4,
            top_regions: 2,
            refine_radius: 2,
        },
        max_luma_rd_candidates: 2,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FastRd,
        max_rd_candidates: 1,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: false,
        exact_bits_during_trials: false,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Balanced,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: true,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: Some(1.10),
        importance_force_leaf: true,
    },
};

static GOOD: EffortTemplate = EffortTemplate {
    name: Effort::Good,
    reference: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Progressive {
            coarse_step: 3,
            top_regions: 3,
            refine_radius: 2,
        },
        max_luma_rd_candidates: 3,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        max_rd_candidates: 3,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: true,
        exact_bits_during_trials: true,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: true,
        allow_candidate_expansion: true,
        importance_rmd_prune_factor: Some(1.10),
        importance_force_leaf: true,
    },
};

static BEST: EffortTemplate = EffortTemplate {
    name: Effort::Best,
    reference: false,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Exhaustive,
        max_luma_rd_candidates: 8,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        max_rd_candidates: 5,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: true,
        exact_bits_during_trials: true,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: false,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: None,
        importance_force_leaf: false,
    },
};

/// `Best` with the `Placebo` frozen-slice-init CTU-wavefront parallel path
/// (`encode_slice_data_parallel`). Identical search budget to `BEST`; only the
/// entropy-context mode (frozen vs running) and `parallel_analysis` differ, so
/// output drifts ~0.1% from serial `Best` (same rationale that lets `Placebo`
/// differ from `Reference`). Selected by default for `Best`
/// (`BPG_BEST2_PARALLEL=0` reverts to serial). Keeps `name: Effort::Best` so
/// all `Best` search levers still apply.
pub(crate) static BEST_PARALLEL: EffortTemplate = EffortTemplate {
    name: Effort::Best,
    reference: false,
    entropy_context: EntropyContextMode::FrozenSliceInit,
    parallel_analysis: true,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Exhaustive,
        max_luma_rd_candidates: 8,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        chroma_during_luma_trials: false,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::FullRd,
        max_rd_candidates: 5,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: true,
        exact_bits_during_trials: true,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: false,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: None,
        importance_force_leaf: false,
    },
};

static PLACEBO: EffortTemplate = EffortTemplate {
    name: Effort::Placebo,
    reference: true,
    entropy_context: EntropyContextMode::FrozenSliceInit,
    parallel_analysis: true,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Exhaustive,
        max_luma_rd_candidates: 3,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::Final,
        chroma_during_luma_trials: true,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::Final,
        max_rd_candidates: 5,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: true,
        exact_bits_during_trials: true,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: false,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: None,
        importance_force_leaf: false,
    },
};

static REFERENCE: EffortTemplate = EffortTemplate {
    name: Effort::Reference,
    reference: true,
    entropy_context: EntropyContextMode::Running,
    parallel_analysis: false,
    rmd: RmdTemplate {
        mode_set: RmdModeSet::Exhaustive,
        max_luma_rd_candidates: 3,
    },
    luma: LumaSearchTemplate {
        trial_quality: TrialQuality::Final,
        chroma_during_luma_trials: true,
    },
    chroma: ChromaSearchTemplate {
        trial_quality: TrialQuality::Final,
        max_rd_candidates: 5,
    },
    residual: ResidualSearchTemplate {
        rdoq_during_trials: true,
        exact_bits_during_trials: true,
        final_quality: TrialQuality::Final,
    },
    cu_split: CuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        early_terminate: CuEarlyTerminateRule::Disabled,
    },
    tu_split: TuSplitTemplate {
        default_search: SplitSearch::EvaluateBoth,
        zero_residual_early_terminate: false,
    },
    preanalysis: PreanalysisTemplate {
        class_steering: false,
        allow_candidate_expansion: false,
        importance_rmd_prune_factor: None,
        importance_force_leaf: false,
    },
};

#[inline]
pub fn template(effort: Effort) -> &'static EffortTemplate {
    match effort {
        Effort::Fastest => &FASTEST,
        Effort::Fast => &FAST,
        Effort::Balanced => &BALANCED,
        Effort::Good => &GOOD,
        Effort::Best => &BEST,
        Effort::Placebo => &PLACEBO,
        Effort::Reference => &REFERENCE,
    }
}

#[inline]
pub fn select_rdoq_single_scan(effort: Effort) -> bool {
    match std::env::var("BPG_RDOQ_SINGLESCAN").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => !is_reference_tier(effort),
    }
}

#[inline]
fn angular_prune_var_threshold_8bit(effort: Effort) -> Option<i64> {
    match effort {
        Effort::Fastest => Some(128),
        Effort::Fast => Some(32),
        Effort::Balanced => Some(8),
        Effort::Good | Effort::Best | Effort::Placebo | Effort::Reference => None,
    }
}

#[inline]
fn qp_budget_effort(effort: Effort, qp: i32) -> Effort {
    if is_reference_tier(effort) {
        return effort;
    }
    match (effort, qp) {
        (Effort::Good, q) if q >= 44 => Effort::Fast,
        (Effort::Good, q) if q >= 38 => Effort::Balanced,
        (Effort::Balanced, q) if q >= 44 => Effort::Fastest,
        (Effort::Balanced, q) if q >= 38 => Effort::Fast,
        (Effort::Fast, q) if q >= 38 => Effort::Fastest,
        _ => effort,
    }
}

pub fn describe_effort(effort: Effort, qp: i32) -> EffortDescription {
    let t = template(effort);
    let texture = t.resolve(describe_block_desc(t, qp, RegionClass::Texture, 256));
    let flat_low_importance = t.resolve(describe_block_desc(t, qp, RegionClass::Flat, 0));
    EffortDescription {
        template: t,
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

pub struct EffortDescription {
    template: &'static EffortTemplate,
    texture: BlockSearchBudget,
    flat_low_importance: BlockSearchBudget,
    qp: i32,
}

impl fmt::Display for EffortDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "effort: {:?}  entropy: {:?}  parallel_analysis: {}",
            self.template.name, self.template.entropy_context, self.template.parallel_analysis
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
