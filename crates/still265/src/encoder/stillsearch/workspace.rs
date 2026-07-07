//! CTU-local workspace and scratch buffers.

use super::arena::CoeffArena;
use super::ledger::StillSearchLedger;
use crate::contexts::Contexts;
use crate::rdoq::RdoqScratch;
use crate::residual::ResidualPricingScratch;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SsimRdNorm {
    pub(super) valid: bool,
    pub(super) qp: i32,
    pub(super) f_dc_den: u64,
    pub(super) f_ac_den: u64,
}

/// Per-CTU substage profile accumulators for `eval_component_8`.
/// Populated only when `BPG_STILLSEARCH_PROFILE=1`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SubstageProfile {
    pub border_ns: u64,
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
    /// Frozen CTU-entry context, re-seeded per CTU from the live writer
    /// context at coding-tree entry. Kept as the stable reference state:
    /// `price_cur` is re-seeded from it for the finalize pass, and it is the
    /// exact pre-CTU writer state either way.
    pub(super) price_base: Contexts,
    /// Evolving trial-pricing context (x265 `m_rqt[depth].cur` parity). All
    /// pricing reads go through this. When `BPG_STILLSEARCH_CTX_EVOLVE` is
    /// enabled, committed winner syntax evolves it sibling-to-sibling through
    /// the coding tree (with save/restore around alternatives); when disabled
    /// it stays equal to `price_base`, reproducing the frozen behavior.
    pub(super) price_cur: Contexts,
    /// When true, `eval_component*` write their selected outcome's context
    /// updates (CBF bin + residual syntax) back into `price_cur`. Set only
    /// around committed-winner evaluation scopes (materialize/finalize), never
    /// during candidate ranking.
    pub(super) commit_ctx: bool,
    /// When true alongside `commit_ctx`, chroma (`c_idx != 0`) evaluations do
    /// not commit context updates. Used by search-time materialization while
    /// the CU-level chroma mode search is enabled: the chroma blocks coded
    /// there are provisional DM blocks; the chroma-mode rebuild pass commits
    /// the final chroma syntax instead.
    pub(super) commit_ctx_skip_chroma: bool,
    pub(super) price_scratch: ResidualPricingScratch,
    /// Reusable scratch for the retained-block single-scan RDOQ refinement.
    pub(super) rdoq_scratch: RdoqScratch,
    /// Best rough SATD score for the current 8×8 CU (set by decide_cu_luma_mode
    /// when log2_cb_size==3, consumed by decide_cu_min_leaf_or_nxn).
    pub(super) last_8x8_rough_satd: f64,
    /// Best rough (SATD + mode bits) score of the most recent
    /// `decide_cu_luma_mode` call at any size — the per-CU rough evidence the
    /// descent-termination instrumentation/gate reads (see `cu.rs`).
    pub(super) last_rough_best_cost: f64,
    /// Per-CTU substage profile (eval_component_8 breakdown). Zero when
    /// profiling is disabled.
    pub(super) substage: SubstageProfile,
    /// Per-component CTU source normalization for the opt-in SSIM-RD cost.
    pub(super) ssim_rd_norm: [SsimRdNorm; 3],
    /// Number of TU split evaluations skipped due to zero-residual early
    /// termination in this CTU.
    pub(super) tu_split_early_terminations: u64,
    /// TU split evaluations aborted by the leaf-first branch-and-bound
    /// (decision-identical; see `TuSplitBound`).
    pub(super) tu_split_bound_aborts: u64,
    /// Per-TU-depth evaluation work for this CTU, indexed by `log2_size`
    /// (TU sizes 4/8/16/32 = log2 2/3/4/5). Counts `eval_tt_leaf` /
    /// `eval_tt_split` calls. Merged into `EncodeStats` in `build_ctu`.
    pub(super) tu_leaf_by_log2: [u64; 7],
    pub(super) tu_split_by_log2: [u64; 7],
    /// When true, the batched cheap simple-RDO evaluator captures each
    /// candidate's full root-leaf luma outcome into `root_tu_candidates`
    /// (slot i = shortlist candidate i). Set only around the x265-shape
    /// cheap ranking of a root-eligible CU (8-bit, `log2 <= MAX_TB_LOG2`).
    pub(super) root_tu_capture: bool,
    /// Per-candidate root-leaf capture slots for the current cheap ranking.
    /// Buffers are reused across CUs.
    pub(super) root_tu_candidates: Vec<RootTuCandidate>,
    /// The cheap winner's captured root-leaf luma outcome, promoted by
    /// `rank_x265_simple_rdo_modes` and consumed (at most once) by the winner
    /// materialization's `decide_tt_with_config` when every input matches.
    pub(super) root_tu_cache: Option<RootTuCache>,
    /// Consumed root-TU captures this CTU (merged into
    /// `EncodeStats::root_tu_reuse_hits`).
    pub(super) root_tu_reuse_hits: u64,
}

/// One captured root-TU luma evaluation outcome: everything
/// `eval_component_8_from_src_pred` computed for a candidate, sufficient to
/// replay its side effects (overlay push, coeff retention, context commit)
/// and return value without re-running transform/quant/pricing.
#[derive(Default, Debug)]
pub(super) struct RootTuCandidate {
    pub(super) cbf: bool,
    pub(super) cost: f64,
    pub(super) dist: u64,
    pub(super) frac_bits: u64,
    pub(super) rd_frac_bits: u64,
    /// Block to push to the overlay: reconstruction when coded, the intra
    /// prediction when the zero-residual outcome won.
    pub(super) push_block: Vec<u8>,
    /// Final (post sign-data-hiding) quantized levels; empty when `!cbf`.
    pub(super) levels: Vec<i16>,
}

impl RootTuCandidate {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn set(
        &mut self,
        cbf: bool,
        cost: f64,
        dist: u64,
        frac_bits: u64,
        rd_frac_bits: u64,
        push_block: &[u8],
        levels: &[i16],
    ) {
        self.cbf = cbf;
        self.cost = cost;
        self.dist = dist;
        self.frac_bits = frac_bits;
        self.rd_frac_bits = rd_frac_bits;
        self.push_block.clear();
        self.push_block.extend_from_slice(push_block);
        self.levels.clear();
        self.levels.extend_from_slice(levels);
    }
}

/// The promoted cheap-winner root-TU capture plus every input that must match
/// at the consumption site for the replay to be byte-identical to a fresh
/// evaluation.
pub(super) struct RootTuCache {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) log2_size: u8,
    pub(super) mode: u8,
    pub(super) qp: i32,
    pub(super) lambda_bits: u64,
    pub(super) quant: super::eval::QuantMode,
    pub(super) sdh: bool,
    /// `price_cur` as seen by the cheap evaluation (candidates do not commit
    /// context updates, so one snapshot covers the whole ranking). Compared
    /// against the live `price_cur` at consumption time — before the
    /// materialization commits any syntax — so any interleaved context
    /// evolution disables the reuse instead of corrupting it.
    pub(super) ctx: Contexts,
    pub(super) cand: RootTuCandidate,
}

impl Default for CtuWorkspace {
    fn default() -> Self {
        Self {
            coeffs: CoeffArena::default(),
            ledger: StillSearchLedger::default(),
            block_scratch: BlockScratch::default(),
            price_base: Contexts::new(0),
            price_cur: Contexts::new(0),
            commit_ctx: false,
            commit_ctx_skip_chroma: false,
            price_scratch: ResidualPricingScratch::default(),
            rdoq_scratch: RdoqScratch::default(),
            last_8x8_rough_satd: f64::INFINITY,
            last_rough_best_cost: f64::INFINITY,
            substage: SubstageProfile::default(),
            ssim_rd_norm: [SsimRdNorm::default(); 3],
            tu_split_early_terminations: 0,
            tu_split_bound_aborts: 0,
            tu_leaf_by_log2: [0; 7],
            tu_split_by_log2: [0; 7],
            root_tu_capture: false,
            root_tu_candidates: Vec::new(),
            root_tu_cache: None,
            root_tu_reuse_hits: 0,
        }
    }
}

impl CtuWorkspace {
    pub(super) fn reset(&mut self) {
        self.coeffs.clear();
        self.ledger.clear_ctu();
        self.block_scratch.clear_ctu();
        self.substage = SubstageProfile::default();
        self.ssim_rd_norm = [SsimRdNorm::default(); 3];
        self.tu_split_early_terminations = 0;
        self.tu_split_bound_aborts = 0;
        self.tu_leaf_by_log2 = [0; 7];
        self.tu_split_by_log2 = [0; 7];
        self.root_tu_capture = false;
        self.root_tu_cache = None;
        self.root_tu_reuse_hits = 0;
    }

    /// Seed the trial-pricing entry context with the live CTU-entry context.
    pub(super) fn set_price_context(&mut self, ctxs: &Contexts) {
        self.price_base = ctxs.clone();
        self.price_cur = ctxs.clone();
        self.commit_ctx = false;
        self.commit_ctx_skip_chroma = false;
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
    pub(super) component_src_u16: Vec<u16>,
    pub(super) component_pred_u16: Vec<u16>,
    pub(super) component_recon_u16: Vec<u16>,
    pub(super) component_pred_tmp_u16: Vec<u16>,
    /// Reusable per-mode accumulators for the batched simple-RDO ranker.
    pub(super) simple_rdo_accum: Vec<super::tu::SimpleRdoAccum>,
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
        self.component_src_u16.clear();
        self.component_pred_u16.clear();
        self.component_recon_u16.clear();
        self.component_pred_tmp_u16.clear();
        self.simple_rdo_accum.clear();
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
