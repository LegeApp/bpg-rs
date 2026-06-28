//! CTU-local workspace and scratch buffers.

use std::time::Instant;

use super::arena::CoeffArena;
use super::ledger::StillSearchLedger;
use crate::contexts::Contexts;
use crate::rdoq::RdoqScratch;
use crate::residual::ResidualPricingScratch;

/// Per-CTU substage profile accumulators for `eval_component_8`.
/// Populated only when `BPG_STILLSEARCH_PROFILE=1`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SubstageProfile {
    pub predict_ns: u64,
    pub forward_xform_ns: u64,
    pub quant_ns: u64,
    pub recon_dist_ns: u64,
    pub residual_price_ns: u64,
    pub calls: u64,
}

pub(super) struct CtuWorkspace {
    pub(super) coeffs: CoeffArena,
    pub(super) ledger: StillSearchLedger,
    pub(super) block_scratch: BlockScratch,
    /// Frozen CTU-entry context for trial residual/flag pricing (the standard
    /// HM/x265 RDO approximation: contexts are not re-coded across candidates).
    /// Re-seeded per CTU from the live writer context at coding-tree entry.
    pub(super) price_base: Contexts,
    pub(super) price_scratch: ResidualPricingScratch,
    /// Reusable scratch for the retained-block single-scan RDOQ refinement.
    pub(super) rdoq_scratch: RdoqScratch,
    /// Best rough SATD score for the current 8×8 CU (set by decide_cu_luma_mode
    /// when log2_cb_size==3, consumed by decide_cu_min_leaf_or_nxn).
    pub(super) last_8x8_rough_satd: f64,
    /// Per-CTU substage profile (eval_component_8 breakdown). Zero when
    /// profiling is disabled.
    pub(super) substage: SubstageProfile,
    /// Number of TU split evaluations skipped due to zero-residual early
    /// termination in this CTU.
    pub(super) tu_split_early_terminations: u64,
    /// Per-TU-depth evaluation work for this CTU, indexed by `log2_size`
    /// (TU sizes 4/8/16/32 = log2 2/3/4/5). Counts `eval_tt_leaf` /
    /// `eval_tt_split` calls. Merged into `EncodeStats` in `build_ctu`.
    pub(super) tu_leaf_by_log2: [u64; 7],
    pub(super) tu_split_by_log2: [u64; 7],
}

impl Default for CtuWorkspace {
    fn default() -> Self {
        Self {
            coeffs: CoeffArena::default(),
            ledger: StillSearchLedger::default(),
            block_scratch: BlockScratch::default(),
            price_base: Contexts::new(0),
            price_scratch: ResidualPricingScratch::default(),
            rdoq_scratch: RdoqScratch::default(),
            last_8x8_rough_satd: f64::INFINITY,
            substage: SubstageProfile::default(),
            tu_split_early_terminations: 0,
            tu_leaf_by_log2: [0; 7],
            tu_split_by_log2: [0; 7],
        }
    }
}

impl CtuWorkspace {
    pub(super) fn reset(&mut self) {
        self.coeffs.clear();
        self.ledger.clear_ctu();
        self.block_scratch.clear_ctu();
        self.substage = SubstageProfile::default();
        self.tu_split_early_terminations = 0;
        self.tu_leaf_by_log2 = [0; 7];
        self.tu_split_by_log2 = [0; 7];
    }

    /// Seed the trial-pricing entry context with the live CTU-entry context.
    pub(super) fn set_price_context(&mut self, ctxs: &Contexts) {
        self.price_base = ctxs.clone();
    }
}

/// Fixed CTU-local scratch for final bridge block coding and future trial
/// evaluators. Search paths must reuse these buffers rather than allocate per
/// candidate.
#[derive(Default, Debug)]
pub(super) struct BlockScratch {
    pub(super) residual_i16_4x4: Vec<i16>,
    pub(super) residual_i16_8x8: Vec<i16>,
    pub(super) residual_i16_16x16: Vec<i16>,
    pub(super) residual_i16_32x32: Vec<i16>,
    pub(super) coeff_i16: Vec<i16>,
    pub(super) transform_tmp_i16: Vec<i16>,
    pub(super) levels_i16: Vec<i16>,
    pub(super) dequant_coeff_i16: Vec<i16>,
    pub(super) recon_residual_i16: Vec<i16>,
    pub(super) rough_angular_batch_u16: Vec<u16>,
    pub(super) rough_pred_u8: Vec<u8>,
    pub(super) component_src_u8: Vec<u8>,
    pub(super) component_pred_u8: Vec<u8>,
    pub(super) component_recon_u8: Vec<u8>,
    pub(super) component_pred_tmp_u16: Vec<u16>,
}

impl BlockScratch {
    pub(super) fn clear_ctu(&mut self) {
        self.residual_i16_4x4.clear();
        self.residual_i16_8x8.clear();
        self.residual_i16_16x16.clear();
        self.residual_i16_32x32.clear();
        self.coeff_i16.clear();
        self.transform_tmp_i16.clear();
        self.levels_i16.clear();
        self.dequant_coeff_i16.clear();
        self.recon_residual_i16.clear();
        self.rough_angular_batch_u16.clear();
        self.rough_pred_u8.clear();
        self.component_src_u8.clear();
        self.component_pred_u8.clear();
        self.component_recon_u8.clear();
        self.component_pred_tmp_u16.clear();
    }

    pub(super) fn residual_i16_for_log2(&mut self, log2_size: u8) -> &mut Vec<i16> {
        match log2_size {
            2 => &mut self.residual_i16_4x4,
            3 => &mut self.residual_i16_8x8,
            4 => &mut self.residual_i16_16x16,
            5 => &mut self.residual_i16_32x32,
            _ => &mut self.residual_i16_32x32,
        }
    }
}
