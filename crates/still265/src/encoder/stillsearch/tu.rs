//! Transform-tree decisions.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::effort::{ChromaTiming, SplitSearch};
use crate::encoder::Encoder;

use super::depth::StillSearchDepth;
use super::emit;
use super::eval::{QuantMode, ResidualPricingMode};
use super::ledger::WorkBucket;
use super::plan::{ParentChromaPlan, TtPlan};
use super::price::split_flag_bits;
use super::source::CtuSourceCache;
use crate::encoder::types::{
    MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, MIN_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb,
};

/// Per-mode accumulator for the batched, leaf-major simple-RDO ranker
/// ([`StillSearchDepth::eval_simple_rdo_luma_modes`]). One per candidate mode;
/// each TU leaf adds its luma RD cost (plus the `split=false` flag bits) and ORs
/// in its coded-block flag.
#[derive(Clone, Copy, Debug)]
pub(super) struct SimpleRdoAccum {
    pub(super) mode: u8,
    pub(super) cost: f64,
    pub(super) cbf: bool,
}

impl SimpleRdoAccum {
    #[inline]
    fn new(mode: u8) -> Self {
        Self {
            mode,
            cost: 0.0,
            cbf: false,
        }
    }

    #[inline]
    fn add_leaf(&mut self, cost: f64, cbf: bool) {
        self.cost += cost;
        self.cbf |= cbf;
    }
}

/// Leaf-first TU split abort bound. `split_loses` evaluates the exact float
/// expression of `decide_tt`'s final leaf-vs-split compare on the partial
/// child-cost sum; because `fl(p - bias)` is monotone nondecreasing in `p`
/// and the remaining terms (children, parent chroma, split-flag bits) are
/// non-negative, a `true` here proves the fully-evaluated split would lose.
#[derive(Clone, Copy)]
pub(super) struct TuSplitBound {
    bias: f64,
    leaf_cost: f64,
}

impl TuSplitBound {
    #[inline]
    pub(super) fn new(bias: f64, leaf_cost: f64) -> Self {
        Self { bias, leaf_cost }
    }

    #[inline]
    fn split_loses(&self, partial_split_cost: f64) -> bool {
        partial_split_cost - self.bias >= self.leaf_cost
    }
}

/// Scope of transform-tree evaluation: luma-only (for fast mode/TU search)
/// or full luma+chroma (for winner re-evaluation and finalize).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TtEvalScope {
    LumaOnly,
    FullComponents,
}

/// Complete configuration for one exact transform-tree evaluation.
///
/// Carries quant/residual-pricing/scope/split-policy/retain-coeff
/// explicitly instead of inferring them from `state.effort_template`,
/// enabling selective RDOQ trials and differing TU split policies
/// during analysis vs. finalize.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExactEvalConfig {
    pub quant: super::eval::QuantMode,
    pub residual_pricing: super::eval::ResidualPricingMode,
    pub scope: TtEvalScope,
    pub tu: crate::effort::TuExactPolicy,
    pub retain_coeff: bool,
    /// Root-TU dedup (audit §10.13): at `trafo_depth == 0`, consume the cheap
    /// stage's captured winning-mode root-leaf luma outcome instead of
    /// re-evaluating, when every captured input matches. Set only by the
    /// x265-shape winner materialization.
    pub reuse_root_luma: bool,
}

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
{
    #[inline]
    fn effective_tu_min_split_log2(&self, base: u8) -> u8 {
        super::env::tu_min_split_log2_override().unwrap_or(base)
    }

    #[inline]
    fn tu_split_cost_bias(&self, log2_size: u8) -> f64 {
        let pixels = 1u64 << (2 * log2_size);
        let per_px = super::env::tu_split_bias_per_px()
            + if log2_size == 3 {
                super::env::tu_4x4_bias_per_px()
            } else {
                0.0
            };
        per_px * pixels as f64
    }

    #[inline]
    fn force_tu_split(&self, log2_size: u8) -> bool {
        super::env::tu_force_split_log2().is_some_and(|v| v == log2_size)
    }

    /// Update the `split_transform_flag` context in the evolving trial context
    /// for the branch currently being explored (x265 codes the flag before the
    /// node's content, so its context evolution precedes the children's).
    #[inline]
    fn commit_tt_split_flag_ctx(&mut self, log2_size: u8, is_split: bool) {
        let ci = crate::contexts::ctx::SPLIT_TRANSFORM_FLAG
            + (5usize.saturating_sub(log2_size as usize)).min(2);
        self.workspace.price_cur.models[ci].update(is_split as u8);
    }

    /// Quantization for the plain `decide_tt` exact pass. With one-pass RDOQ
    /// (default) this follows the exact-stage `TrialQuant` policy so candidates
    /// and TU splits are evaluated under RDOQ directly and the Phase-2 re-run
    /// is skipped; with `BPG_STILLSEARCH_RDOQ_ONE_PASS=0` it stays hard-quant
    /// (the legacy two-phase flow).
    #[inline]
    fn tt_search_quant(&self, state: &Encoder<'_>) -> QuantMode {
        if super::env::rdoq_one_pass_enabled() {
            super::eval::search_trial_quant(state.effort_template.luma.exact.quant)
        } else {
            QuantMode::HardQuantSearch
        }
    }

    #[inline]
    fn target_4x4_possible(&self, log2_size: u8, syntax_can_split: bool) -> bool {
        syntax_can_split && log2_size == 3 && super::env::tu_target_4x4_enabled()
    }

    fn target_4x4_admit(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        leaf_tt: &TtPlan,
        leaf_cost: f64,
    ) -> bool {
        if leaf_has_no_luma_residual(leaf_tt) {
            return false;
        }
        let (var, mean_range) = self.source_stats_8x8(state, x0, y0);
        if var < super::env::tu_target_var_min() {
            return false;
        }
        let cost_per_px = leaf_cost / 64.0;
        cost_per_px >= super::env::tu_target_cost_min()
            || mean_range >= super::env::tu_target_mean_range_min()
    }

    fn source_stats_8x8(&self, state: &Encoder<'_>, x0: u32, y0: u32) -> (f64, f64) {
        let shift = state.bit_depth.saturating_sub(8) as u32;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut qsum = [0.0f64; 4];
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = (self.source.sample(0, x0 + x, y0 + y) >> shift) as f64;
                sum += v;
                sum2 += v * v;
                let q = ((y >= 4) as usize) * 2 + (x >= 4) as usize;
                qsum[q] += v;
            }
        }
        let mean = sum / 64.0;
        let var = (sum2 / 64.0) - mean * mean;
        let mut min_q = f64::INFINITY;
        let mut max_q = f64::NEG_INFINITY;
        for s in qsum {
            let m = s / 16.0;
            min_q = min_q.min(m);
            max_q = max_q.max(m);
        }
        (var.max(0.0), max_q - min_q)
    }

    /// Decide one transform-tree node: full leaf vs four split children, by RD
    /// cost. On return the overlay holds exactly the winner's recon patches for
    /// this region (loser patches are rewound).
    pub(super) fn decide_tt(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
    ) -> (TtPlan, f64) {
        let must_split = log2_size > MAX_TB_LOG2;
        let syntax_can_split = !must_split
            && state.cat != 2
            && log2_size > MIN_TB_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH;
        let can_split = syntax_can_split
            && log2_size
                > self.effective_tu_min_split_log2(state.effort_template.tu.min_split_log2);
        let target_4x4_possible =
            !can_split && self.target_4x4_possible(log2_size, syntax_can_split);

        // Determine evaluation scope from the effort template. Luma-only search
        // skips chroma evaluation during luma mode/TU trials; chroma is
        // re-attached only for the winner. Placebo templates enable
        // full chroma evaluation for the oracle.
        let scope = if state.effort_template.chroma.timing == ChromaTiming::DuringExactTrials {
            TtEvalScope::FullComponents
        } else {
            TtEvalScope::LumaOnly
        };

        // Read residual pricing from the exact policy template.
        let residual_pricing = match state.effort_template.luma.exact.residual_price {
            crate::effort::ResidualPriceLevel::Exact => ResidualPricingMode::Exact,
            _ => ResidualPricingMode::Skip,
        };
        let tt_quant = self.tt_search_quant(state);

        if must_split {
            return self.eval_tt_split(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                false,
                lambda,
                None,
            );
        }
        if !can_split && !target_4x4_possible {
            return self.eval_tt_leaf(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                luma_mode,
                false,
                lambda,
                tt_quant,
                residual_pricing,
                true,
                scope,
                None,
            );
        }

        // Leaf-first evaluation: evaluate the leaf candidate before the split.
        // With evolving trial contexts, each alternative's coded syntax
        // evolves `price_cur` from the same node-entry snapshot; the winner's
        // exit state survives (x265 `codeIntraLumaQT` save/restore parity).
        let flag_priced = can_split || target_4x4_possible;
        let ctx_entry = self
            .workspace
            .commit_ctx
            .then(|| self.workspace.price_cur.clone());
        if self.workspace.commit_ctx && flag_priced {
            self.commit_tt_split_flag_ctx(log2_size, false);
        }
        let mark0 = self.overlay.mark();
        let (leaf_tt, leaf_cost) = self.eval_tt_leaf(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            luma_mode,
            can_split || target_4x4_possible,
            lambda,
            tt_quant,
            residual_pricing,
            true,
            scope,
            None,
        );
        let leaf_saved = self.overlay.detach_from(mark0);
        let target_4x4_admit =
            target_4x4_possible && self.target_4x4_admit(state, x0, y0, &leaf_tt, leaf_cost);
        if !can_split && !target_4x4_admit {
            self.overlay.reattach(leaf_saved);
            return (leaf_tt, leaf_cost);
        }

        // TU early termination: skip split search when the leaf is good
        // enough.  Three levels of gate:
        //
        // 1. Zero-residual: no luma coded block → skip (clean block).
        // 2. Low-residual: leaf cost per pixel below threshold → skip.
        // 3. PreferLeaf policy: even with residual, skip unless evidence
        //    suggests split would help.
        {
            let tu_policy = &state.effort_template.tu;
            let num_px = 1u64 << (2 * log2_size);
            let leaf_cost_per_px = leaf_cost / num_px as f64;

            let skip = !self.force_tu_split(log2_size)
                && !target_4x4_admit
                && (tu_policy.zero_residual_early_terminate && leaf_has_no_luma_residual(&leaf_tt)
                    || tu_policy.low_residual_early_terminate
                        && leaf_cost_per_px < tu_policy.low_distortion_per_px
                    || tu_policy.split_search == SplitSearch::ForceLeaf);

            if skip {
                self.workspace.tu_split_early_terminations += 1;
                self.overlay.reattach(leaf_saved);
                return (leaf_tt, leaf_cost);
            }
        }

        let ctx_leaf = ctx_entry.as_ref().map(|entry| {
            let leaf_state = self.workspace.price_cur.clone();
            self.workspace.price_cur = entry.clone();
            if flag_priced {
                self.commit_tt_split_flag_ctx(log2_size, true);
            }
            leaf_state
        });

        let bias = self.tu_split_cost_bias(log2_size);
        let bound = (super::env::tu_split_bound_enabled() && !self.force_tu_split(log2_size))
            .then_some(TuSplitBound::new(bias, leaf_cost));
        let (split_tt, split_cost) = self.eval_tt_split(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            true,
            lambda,
            bound,
        );

        let split_cmp_cost = split_cost - bias;
        if self.force_tu_split(log2_size) || split_cmp_cost < leaf_cost {
            #[cfg(test)]
            super::api::SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            super::api::LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(leaf_state) = ctx_leaf {
                self.workspace.price_cur = leaf_state;
            }
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            (leaf_tt, leaf_cost)
        }
    }

    /// Evaluate this node as a leaf TU (luma + chroma), returning its internal
    /// plan and RD cost and pushing its recon to the overlay.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_tt_leaf(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
        quant_mode: QuantMode,
        residual_pricing: ResidualPricingMode,
        retain_coeff: bool,
        scope: TtEvalScope,
        root_luma_replay: Option<&super::workspace::RootTuCandidate>,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuLeaf);
        self.workspace.tu_leaf_by_log2[log2_size as usize] += 1;

        let cat = state.cat;
        let qp_y = state.cur_qp_y;
        let luma_pred = IntraPredMode::from_u8(luma_mode).unwrap_or(IntraPredMode::Dc);
        let luma = if let Some(cand) = root_luma_replay {
            self.replay_root_luma_component(state, x0, y0, log2_size, luma_pred, retain_coeff, cand)
        } else {
            self.eval_component(
                state,
                x0,
                y0,
                log2_size,
                0,
                luma_pred,
                qp_y,
                trafo_depth,
                lambda,
                quant_mode,
                residual_pricing,
                retain_coeff,
            )
        };
        let mut cost = luma.cost;

        let mut cb = emit::empty_block();
        let mut cr = emit::empty_block();
        let mut cb1 = emit::empty_block();
        let mut cr1 = emit::empty_block();
        let mut chroma_log2 = 0;
        let chroma_mode_for_syntax = chroma_mode;

        if scope == TtEvalScope::FullComponents && has_chroma_tb(cat, log2_size) {
            if let Some((cx, cy, clog2, count)) = chroma_tb_geom(cat, x0, y0, log2_size) {
                chroma_log2 = clog2;
                let cmode = IntraPredMode::from_u8(chroma_pred_mode(cat, chroma_mode_for_syntax))
                    .unwrap_or(IntraPredMode::Dc);
                let qp_c = state.cur_qp_c;
                let step = 1u32 << clog2;

                let cb_e = self.eval_component(
                    state,
                    cx,
                    cy,
                    clog2,
                    1,
                    cmode,
                    qp_c,
                    trafo_depth,
                    lambda,
                    quant_mode,
                    ResidualPricingMode::Exact,
                    retain_coeff,
                );
                cost += cb_e.cost;
                cb = cb_e.into_plan_block();

                let cr_e = self.eval_component(
                    state,
                    cx,
                    cy,
                    clog2,
                    2,
                    cmode,
                    qp_c,
                    trafo_depth,
                    lambda,
                    quant_mode,
                    ResidualPricingMode::Exact,
                    retain_coeff,
                );
                cost += cr_e.cost;
                cr = cr_e.into_plan_block();

                if count > 1 {
                    let y1 = cy + step;
                    let cb1_e = self.eval_component(
                        state,
                        cx,
                        y1,
                        clog2,
                        1,
                        cmode,
                        qp_c,
                        trafo_depth,
                        lambda,
                        quant_mode,
                        ResidualPricingMode::Exact,
                        retain_coeff,
                    );
                    cost += cb1_e.cost;
                    cb1 = cb1_e.into_plan_block();

                    let cr1_e = self.eval_component(
                        state,
                        cx,
                        y1,
                        clog2,
                        2,
                        cmode,
                        qp_c,
                        trafo_depth,
                        lambda,
                        quant_mode,
                        ResidualPricingMode::Exact,
                        retain_coeff,
                    );
                    cost += cr1_e.cost;
                    cr1 = cr1_e.into_plan_block();
                }
            }
        }

        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, false);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        let tt = emit::leaf_tu(
            log2_size,
            chroma_log2,
            trafo_depth,
            luma_mode,
            chroma_mode_for_syntax,
            luma.into_plan_block(),
            cb,
            cr,
            cb1,
            cr1,
        );
        (tt, cost)
    }

    /// Evaluate a luma-mode candidate with x265's first-pass intra shape:
    /// forced transform splits are honored, but optional TU splits are not
    /// explored. The full recursive TU search is then run only for the selected
    /// best mode. This is cheaper than `decide_tt` and provides the
    /// StillSearch `LumaCheap` stage.
    pub(super) fn decide_tt_luma_no_optional_split(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
    ) -> (TtPlan, f64) {
        let must_split = log2_size > MAX_TB_LOG2;
        if must_split {
            return self.eval_tt_forced_split_no_optional(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                lambda,
            );
        }

        let split_flag_coded = state.cat != 2
            && log2_size
                > self.effective_tu_min_split_log2(state.effort_template.tu.min_split_log2)
            && trafo_depth < MAX_INTRA_TT_DEPTH;
        self.eval_tt_luma_leaf(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            split_flag_coded,
            lambda,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_tt_luma_leaf(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuLeaf);
        self.workspace.tu_leaf_by_log2[log2_size as usize] += 1;

        let luma_pred = IntraPredMode::from_u8(luma_mode).unwrap_or(IntraPredMode::Dc);
        let luma = self.eval_component(
            state,
            x0,
            y0,
            log2_size,
            0,
            luma_pred,
            state.cur_qp_y,
            trafo_depth,
            lambda,
            super::eval::search_trial_quant(state.effort_template.luma.cheap.quant),
            if super::env::luma_cheap_residual_price_exact(
                state.effort_template.luma.cheap.residual_price
                    == crate::effort::ResidualPriceLevel::Exact,
            ) {
                ResidualPricingMode::Exact
            } else {
                ResidualPricingMode::Skip
            },
            false,
        );
        let mut cost = luma.cost;

        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, false);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }
        let tt = emit::leaf_tu(
            log2_size,
            0,
            trafo_depth,
            luma_mode,
            luma_mode,
            luma.into_plan_block(),
            emit::empty_block(),
            emit::empty_block(),
            emit::empty_block(),
            emit::empty_block(),
        );
        (tt, cost)
    }

    /// Lightweight luma-leaf evaluation that skips overlay push and TtPlan
    /// construction. Returns (cost, cbf) — enough for cheap-mode ranking.
    /// Matches [`eval_tt_luma_leaf`] in cost semantics.
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_luma_leaf_cheap(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
    ) -> (f64, bool) {
        let luma_pred = IntraPredMode::from_u8(luma_mode).unwrap_or(IntraPredMode::Dc);
        let luma = self.eval_component_no_overlay(
            state,
            x0,
            y0,
            log2_size,
            0,
            luma_pred,
            state.cur_qp_y,
            trafo_depth,
            lambda,
            super::eval::search_trial_quant(state.effort_template.luma.cheap.quant),
            if super::env::luma_cheap_residual_price_exact(
                state.effort_template.luma.cheap.residual_price
                    == crate::effort::ResidualPriceLevel::Exact,
            ) {
                ResidualPricingMode::Exact
            } else {
                ResidualPricingMode::Skip
            },
            false,
        );
        let mut cost = luma.cost;
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, false);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }
        (cost, luma.cbf)
    }

    /// Lightweight rank-only evaluation that handles forced-split and leaf TU
    /// cases. Does not push to overlay, does not allocate TtPlan.
    /// Returns (RD_cost, luma_cbf).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_simple_rdo_luma(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
    ) -> (f64, bool) {
        let must_split = log2_size > MAX_TB_LOG2;
        if must_split {
            let half = 1u32 << (log2_size - 1);
            let kid_log2 = log2_size - 1;
            let kid_depth = trafo_depth + 1;
            let (c0, _) =
                self.eval_simple_rdo_luma(state, x0, y0, kid_log2, kid_depth, luma_mode, lambda);
            let (c1, _) = self.eval_simple_rdo_luma(
                state,
                x0 + half,
                y0,
                kid_log2,
                kid_depth,
                luma_mode,
                lambda,
            );
            let (c2, _) = self.eval_simple_rdo_luma(
                state,
                x0,
                y0 + half,
                kid_log2,
                kid_depth,
                luma_mode,
                lambda,
            );
            let (c3, _) = self.eval_simple_rdo_luma(
                state,
                x0 + half,
                y0 + half,
                kid_log2,
                kid_depth,
                luma_mode,
                lambda,
            );
            (c0 + c1 + c2 + c3, false)
        } else {
            let split_flag_coded = state.cat != 2
                && log2_size > state.effort_template.tu.min_split_log2
                && trafo_depth < MAX_INTRA_TT_DEPTH;
            self.eval_tt_luma_leaf_cheap(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                split_flag_coded,
                lambda,
            )
        }
    }

    /// Batched, leaf-major x265-shape simple RDO. Equivalent to calling
    /// [`eval_simple_rdo_luma`] once per mode, but each concrete TU leaf samples
    /// its source and builds its intra reference border **once**, then evaluates
    /// every candidate mode against them — eliminating the per-mode source
    /// resample and border rebuild that dominated `LumaCheap`. Byte-identical to
    /// the per-mode path (same leaf cost, same `split_false` bits, same child sum
    /// order).
    pub(super) fn eval_simple_rdo_luma_modes(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        modes: &[u8],
        lambda: f64,
        out: &mut Vec<SimpleRdoAccum>,
    ) {
        out.clear();
        out.extend(modes.iter().copied().map(SimpleRdoAccum::new));
        self.eval_simple_rdo_luma_modes_accum(state, x0, y0, log2_size, trafo_depth, lambda, out);
    }

    fn eval_simple_rdo_luma_modes_accum(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        lambda: f64,
        accum: &mut [SimpleRdoAccum],
    ) {
        if log2_size > MAX_TB_LOG2 {
            // Forced split: recurse children in decoder/z-order, accumulating
            // each child leaf's per-mode cost into the same accumulators.
            let half = 1u32 << (log2_size - 1);
            let kid_log2 = log2_size - 1;
            let kid_depth = trafo_depth + 1;
            self.eval_simple_rdo_luma_modes_accum(
                state, x0, y0, kid_log2, kid_depth, lambda, accum,
            );
            self.eval_simple_rdo_luma_modes_accum(
                state,
                x0 + half,
                y0,
                kid_log2,
                kid_depth,
                lambda,
                accum,
            );
            self.eval_simple_rdo_luma_modes_accum(
                state,
                x0,
                y0 + half,
                kid_log2,
                kid_depth,
                lambda,
                accum,
            );
            self.eval_simple_rdo_luma_modes_accum(
                state,
                x0 + half,
                y0 + half,
                kid_log2,
                kid_depth,
                lambda,
                accum,
            );
            return;
        }

        let split_flag_coded = state.cat != 2
            && log2_size > state.effort_template.tu.min_split_log2
            && trafo_depth < MAX_INTRA_TT_DEPTH;
        let split_false_cost = if split_flag_coded {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, false);
            lambda * bits as f64 / CabacEstimator::SCALE as f64
        } else {
            0.0
        };

        self.eval_tt_luma_leaf_cheap_modes(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            lambda,
            split_false_cost,
            accum,
        );
    }

    /// Leaf-major batched cheap leaf evaluator. Mirrors the per-mode
    /// `eval_tt_luma_leaf_cheap`, which does not bump `TuLeaf` (cheap-path work is
    /// accounted under `LumaCheap`), so this doesn't either.
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_luma_leaf_cheap_modes(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        lambda: f64,
        split_false_cost: f64,
        accum: &mut [SimpleRdoAccum],
    ) {
        if state.bit_depth == 8 {
            self.eval_tt_luma_leaf_cheap_modes_8(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                lambda,
                split_false_cost,
                accum,
            );
            return;
        }
        // High-bit-depth keeps the per-mode path (less hot).
        for a in accum.iter_mut() {
            let (cost, cbf) = self.eval_tt_luma_leaf_cheap(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                a.mode,
                false,
                lambda,
            );
            a.add_leaf(cost + split_false_cost, cbf);
        }
    }

    /// 8-bit leaf-major inner loop: sample source + build refs once, then predict
    /// and evaluate each mode against the shared source/refs.
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_luma_leaf_cheap_modes_8(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        lambda: f64,
        split_false_cost: f64,
        accum: &mut [SimpleRdoAccum],
    ) {
        let size = 1usize << log2_size;
        let n = size * size;

        // Source and reference borders are identical for every candidate mode at
        // this leaf (simple RDO never pushes candidate recon to the overlay).
        let mut src = std::mem::take(&mut self.workspace.block_scratch.component_src_u8);
        src.resize(n, 0);
        self.source.sample_block_u8(0, x0, y0, size, &mut src);
        let refs = self.build_intra_refs_for_block(state, x0, y0, log2_size, 0);

        let residual_pricing = if super::env::luma_cheap_residual_price_exact(
            state.effort_template.luma.cheap.residual_price
                == crate::effort::ResidualPriceLevel::Exact,
        ) {
            ResidualPricingMode::Exact
        } else {
            ResidualPricingMode::Skip
        };

        let mut pred = std::mem::take(&mut self.workspace.block_scratch.component_pred_u8);
        pred.resize(n, 0);

        // Root-TU capture (audit §10.13): record each candidate's full leaf
        // outcome so the ranking winner's materialization can replay it
        // instead of re-evaluating. Only armed by the x265-shape ranker for
        // root-eligible CUs.
        let capture = self.workspace.root_tu_capture && trafo_depth == 0;
        let mut caps = std::mem::take(&mut self.workspace.root_tu_candidates);
        if capture && caps.len() < accum.len() {
            caps.resize_with(accum.len(), Default::default);
        }

        for (i, a) in accum.iter_mut().enumerate() {
            let mode = IntraPredMode::from_u8(a.mode).unwrap_or(IntraPredMode::Dc);
            self.predict_exact_from_refs_u8(&refs, log2_size, 0, mode, &mut pred);
            let trial = self.eval_component_8_from_src_pred(
                state,
                x0,
                y0,
                log2_size,
                0,
                mode,
                state.cur_qp_y,
                trafo_depth,
                lambda,
                super::eval::search_trial_quant(state.effort_template.luma.cheap.quant),
                residual_pricing,
                false,
                false,
                &src,
                &pred,
                if capture { Some(&mut caps[i]) } else { None },
            );
            a.add_leaf(trial.cost + split_false_cost, trial.cbf);
        }

        self.workspace.root_tu_candidates = caps;
        self.workspace.block_scratch.component_src_u8 = src;
        self.workspace.block_scratch.component_pred_u8 = pred;
    }

    fn eval_tt_forced_split_no_optional(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);
        self.workspace.tu_split_by_log2[log2_size as usize] += 1;

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let (k0, c0) = self.decide_tt_luma_no_optional_split(
            state, x0, y0, kid_log2, kid_depth, luma_mode, lambda,
        );
        let (k1, c1) = self.decide_tt_luma_no_optional_split(
            state,
            x0 + half,
            y0,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
        );
        let (k2, c2) = self.decide_tt_luma_no_optional_split(
            state,
            x0,
            y0 + half,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
        );
        let (k3, c3) = self.decide_tt_luma_no_optional_split(
            state,
            x0 + half,
            y0 + half,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
        );
        let kids = vec![k0, k1, k2, k3];
        let tt = TtPlan::Split {
            log2_size,
            trafo_depth,
            cbf_cb: false,
            cbf_cr: false,
            cbf_cb1: false,
            cbf_cr1: false,
            parent_chroma: None,
            kids,
        };
        (tt, c0 + c1 + c2 + c3)
    }

    fn eval_tt_split_parent_chroma(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
        quant_mode: QuantMode,
        retain_coeff: bool,
    ) -> Option<(ParentChromaPlan, f64)> {
        if log2_size != 3 || !matches!(state.cat, 1 | 2) {
            return None;
        }
        let cx = x0 / 2;
        let cy = if state.cat == 1 { y0 / 2 } else { y0 };
        let clog2 = 2;
        let step = 1u32 << clog2;
        let chroma_mode = chroma_pred_mode(state.cat, luma_mode);
        let pred_mode = IntraPredMode::from_u8(chroma_mode).unwrap_or(IntraPredMode::Dc);
        let qp_c = state.cur_qp_c;

        let cb0 = self.eval_component(
            state,
            cx,
            cy,
            clog2,
            1,
            pred_mode,
            qp_c,
            trafo_depth,
            lambda,
            quant_mode,
            ResidualPricingMode::Exact,
            retain_coeff,
        );
        let mut cost = cb0.cost;
        let cb1 = if state.cat == 2 {
            let cb = self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                1,
                pred_mode,
                qp_c,
                trafo_depth,
                lambda,
                quant_mode,
                ResidualPricingMode::Exact,
                retain_coeff,
            );
            cost += cb.cost;
            Some(cb)
        } else {
            None
        };

        let cr0 = self.eval_component(
            state,
            cx,
            cy,
            clog2,
            2,
            pred_mode,
            qp_c,
            trafo_depth,
            lambda,
            quant_mode,
            ResidualPricingMode::Exact,
            retain_coeff,
        );
        cost += cr0.cost;
        let cr1 = if state.cat == 2 {
            let cr = self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                2,
                pred_mode,
                qp_c,
                trafo_depth,
                lambda,
                quant_mode,
                ResidualPricingMode::Exact,
                retain_coeff,
            );
            cost += cr.cost;
            Some(cr)
        } else {
            None
        };

        Some((
            ParentChromaPlan {
                log2_size: clog2,
                chroma_mode,
                cb: cb0.into_plan_block(),
                cb1: cb1
                    .map(super::eval::BlockTrial::into_plan_block)
                    .unwrap_or_else(emit::empty_block),
                cr: cr0.into_plan_block(),
                cr1: cr1
                    .map(super::eval::BlockTrial::into_plan_block)
                    .unwrap_or_else(emit::empty_block),
            },
            cost,
        ))
    }

    /// Evaluate this node as a split into four child transform trees.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_tt_split(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
        bound: Option<TuSplitBound>,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);
        self.workspace.tu_split_by_log2[log2_size as usize] += 1;

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let mut kids = Vec::with_capacity(4);
        let mut cost = 0.0;
        for (kx, ky) in [
            (x0, y0),
            (x0 + half, y0),
            (x0, y0 + half),
            (x0 + half, y0 + half),
        ] {
            let (k, c) = self.decide_tt(state, kx, ky, kid_log2, kid_depth, luma_mode, lambda);
            kids.push(k);
            cost += c;
            if let Some(b) = &bound
                && b.split_loses(cost)
            {
                self.workspace.tu_split_bound_aborts += 1;
                return (
                    self.tt_split_plan(log2_size, trafo_depth, None, kids),
                    f64::INFINITY,
                );
            }
        }

        let mut parent_chroma = None;
        if let Some((pc, pc_cost)) = self.eval_tt_split_parent_chroma(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            lambda,
            self.tt_search_quant(state),
            true,
        ) {
            cost += pc_cost;
            parent_chroma = Some(pc);
        }
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, true);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        (
            self.tt_split_plan(log2_size, trafo_depth, parent_chroma, kids),
            cost,
        )
    }

    fn tt_split_plan(
        &self,
        log2_size: u8,
        trafo_depth: u8,
        parent_chroma: Option<ParentChromaPlan>,
        kids: Vec<TtPlan>,
    ) -> TtPlan {
        TtPlan::Split {
            log2_size,
            trafo_depth,
            cbf_cb: kids.iter().any(TtPlan::cbf_cb),
            cbf_cr: kids.iter().any(TtPlan::cbf_cr),
            cbf_cb1: kids.iter().any(TtPlan::cbf_cb1),
            cbf_cr1: kids.iter().any(TtPlan::cbf_cr1),
            parent_chroma,
            kids,
        }
    }

    // ── Config-aware variants ──────────────────────────────────────

    /// Decide one transform-tree node using an explicit [`ExactEvalConfig`]
    /// instead of inferring parameters from `state.effort_template`.
    /// Used for analysis-stage RDOQ trials and custom TU split policies.
    pub(super) fn decide_tt_with_config(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
        cfg: ExactEvalConfig,
    ) -> (TtPlan, f64) {
        let must_split = log2_size > MAX_TB_LOG2;
        let syntax_can_split = !must_split
            && state.cat != 2
            && log2_size > MIN_TB_LOG2
            && trafo_depth < cfg.tu.max_extra_depth;
        let can_split =
            syntax_can_split && log2_size > self.effective_tu_min_split_log2(cfg.tu.min_split_log2);
        let target_4x4_possible =
            !can_split && self.target_4x4_possible(log2_size, syntax_can_split);

        // Root-TU dedup: take the cheap winner's captured root-leaf luma
        // outcome iff every input matches, *before* any context mutation
        // below (the capture snapshot is compared against the live
        // `price_cur`). In verify mode the fresh evaluation still runs and
        // the capture is asserted against it afterwards.
        let root_cache = (cfg.reuse_root_luma && trafo_depth == 0)
            .then(|| self.take_root_tu_cache(state, x0, y0, log2_size, luma_mode, lambda, &cfg))
            .flatten();
        let verify_mode = super::env::root_tu_reuse_mode() == super::env::RootTuReuseMode::Verify;
        let (root_replay, root_verify) = if verify_mode {
            (None, root_cache)
        } else {
            (root_cache, None)
        };

        if must_split
            || can_split
                && (matches!(cfg.tu.split_mode, crate::effort::TuSplitMode::ForceSplit)
                    || self.force_tu_split(log2_size))
        {
            return self.eval_tt_split_with_config(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                false,
                lambda,
                cfg,
                None,
            );
        }
        if (!can_split && !target_4x4_possible)
            || matches!(cfg.tu.split_mode, crate::effort::TuSplitMode::Disabled)
        {
            let out = self.eval_tt_leaf(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                luma_mode,
                false,
                lambda,
                cfg.quant,
                cfg.residual_pricing,
                cfg.retain_coeff,
                cfg.scope,
                root_replay.as_ref().map(|c| &c.cand),
            );
            if let Some(cache) = &root_verify {
                self.verify_root_tu_capture(&out.0, cache);
            }
            return out;
        }

        // Leaf-first evaluation. Trial-context save/restore mirrors decide_tt.
        let flag_priced = can_split || target_4x4_possible;
        let ctx_entry = self
            .workspace
            .commit_ctx
            .then(|| self.workspace.price_cur.clone());
        if self.workspace.commit_ctx && flag_priced {
            self.commit_tt_split_flag_ctx(log2_size, false);
        }
        let mark0 = self.overlay.mark();
        let (leaf_tt, leaf_cost) = self.eval_tt_leaf(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            luma_mode,
            can_split || target_4x4_possible,
            lambda,
            cfg.quant,
            cfg.residual_pricing,
            cfg.retain_coeff,
            cfg.scope,
            root_replay.as_ref().map(|c| &c.cand),
        );
        if let Some(cache) = &root_verify {
            self.verify_root_tu_capture(&leaf_tt, cache);
        }
        let leaf_saved = self.overlay.detach_from(mark0);
        let target_4x4_admit =
            target_4x4_possible && self.target_4x4_admit(state, x0, y0, &leaf_tt, leaf_cost);
        if !can_split && !target_4x4_admit {
            self.overlay.reattach(leaf_saved);
            return (leaf_tt, leaf_cost);
        }

        // TU early termination gates (driven by config).
        {
            let num_px = 1u64 << (2 * log2_size);
            let leaf_cost_per_px = leaf_cost / num_px as f64;
            let skip = !self.force_tu_split(log2_size)
                && !target_4x4_admit
                && (cfg.tu.zero_residual_early_terminate && leaf_has_no_luma_residual(&leaf_tt)
                    || cfg.tu.low_residual_early_terminate
                        && leaf_cost_per_px < cfg.tu.low_distortion_per_px);
            if skip {
                self.workspace.tu_split_early_terminations += 1;
                self.overlay.reattach(leaf_saved);
                return (leaf_tt, leaf_cost);
            }
        }

        let ctx_leaf = ctx_entry.as_ref().map(|entry| {
            let leaf_state = self.workspace.price_cur.clone();
            self.workspace.price_cur = entry.clone();
            if flag_priced {
                self.commit_tt_split_flag_ctx(log2_size, true);
            }
            leaf_state
        });

        let bias = self.tu_split_cost_bias(log2_size);
        let bound =
            super::env::tu_split_bound_enabled().then_some(TuSplitBound::new(bias, leaf_cost));
        let (split_tt, split_cost) = self.eval_tt_split_with_config(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            true,
            lambda,
            cfg,
            bound,
        );

        let split_cmp_cost = split_cost - bias;
        if split_cmp_cost < leaf_cost {
            #[cfg(test)]
            super::api::SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            super::api::LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(leaf_state) = ctx_leaf {
                self.workspace.price_cur = leaf_state;
            }
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            (leaf_tt, leaf_cost)
        }
    }

    /// Take the promoted cheap-winner root-TU capture iff every input the
    /// evaluation depends on matches this call. Any mismatch leaves the cache
    /// unconsumed (it is overwritten by the next ranking) and falls back to a
    /// fresh evaluation, so a `Some` here is byte-identical by construction.
    fn take_root_tu_cache(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        lambda: f64,
        cfg: &ExactEvalConfig,
    ) -> Option<super::workspace::RootTuCache> {
        let cache = self.workspace.root_tu_cache.as_ref()?;
        let matches = cache.x0 == x0
            && cache.y0 == y0
            && cache.log2_size == log2_size
            && cache.mode == luma_mode
            && cache.qp == state.cur_qp_y
            && cache.lambda_bits == lambda.to_bits()
            && cache.quant == cfg.quant
            && cfg.residual_pricing == ResidualPricingMode::Exact
            && cache.sdh == state.sign_data_hiding
            && cache.ctx == self.workspace.price_cur;
        if !matches {
            return None;
        }
        self.workspace.root_tu_reuse_hits += 1;
        self.workspace.root_tu_cache.take()
    }

    /// Verify-mode check: the freshly evaluated root leaf's luma plan block
    /// must equal the cheap-stage capture field for field (and level for
    /// level). Panics on divergence — this is the validation gate for the
    /// root-TU reuse fast path.
    fn verify_root_tu_capture(&self, tt: &TtPlan, cache: &super::workspace::RootTuCache) {
        let TtPlan::Leaf(leaf) = tt else {
            panic!("root-TU verify: fresh evaluation did not produce a leaf");
        };
        let cand = &cache.cand;
        let luma = &leaf.luma;
        let levels: &[i16] = luma
            .coeff
            .map(|id| self.workspace.coeffs.get(id))
            .unwrap_or(&[]);
        assert!(
            luma.cbf == cand.cbf
                && luma.frac_bits == cand.frac_bits
                && luma.dist == cand.dist
                && luma.rd_frac_bits == cand.rd_frac_bits
                && levels == &cand.levels[..],
            "root-TU verify mismatch at ({},{}) log2={} mode={}: \
             fresh cbf={} frac={} dist={} rd_frac={} nlev={} vs \
             captured cbf={} frac={} dist={} rd_frac={} nlev={}",
            cache.x0,
            cache.y0,
            cache.log2_size,
            cache.mode,
            luma.cbf,
            luma.frac_bits,
            luma.dist,
            luma.rd_frac_bits,
            levels.len(),
            cand.cbf,
            cand.frac_bits,
            cand.dist,
            cand.rd_frac_bits,
            cand.levels.len(),
        );
    }

    /// Evaluate a split node using an explicit config, recursing via
    /// [`decide_tt_with_config`].
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_split_with_config(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
        cfg: ExactEvalConfig,
        bound: Option<TuSplitBound>,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);
        self.workspace.tu_split_by_log2[log2_size as usize] += 1;

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let mut kids = Vec::with_capacity(4);
        let mut cost = 0.0;
        for (kx, ky) in [
            (x0, y0),
            (x0 + half, y0),
            (x0, y0 + half),
            (x0 + half, y0 + half),
        ] {
            let (k, c) = self
                .decide_tt_with_config(state, kx, ky, kid_log2, kid_depth, luma_mode, lambda, cfg);
            kids.push(k);
            cost += c;
            if let Some(b) = &bound
                && b.split_loses(cost)
            {
                self.workspace.tu_split_bound_aborts += 1;
                return (
                    self.tt_split_plan(log2_size, trafo_depth, None, kids),
                    f64::INFINITY,
                );
            }
        }

        let mut parent_chroma = None;
        if cfg.scope == TtEvalScope::FullComponents {
            if let Some((pc, pc_cost)) = self.eval_tt_split_parent_chroma(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                lambda,
                cfg.quant,
                cfg.retain_coeff,
            ) {
                cost += pc_cost;
                parent_chroma = Some(pc);
            }
        }
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_cur, log2_size, true);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        (
            self.tt_split_plan(log2_size, trafo_depth, parent_chroma, kids),
            cost,
        )
    }
}

/// Returns `true` when the TT plan is a leaf whose luma block has no coded
/// residual (CBF=false). Used by leaf-first early termination.
fn leaf_has_no_luma_residual(tt: &TtPlan) -> bool {
    matches!(tt, TtPlan::Leaf(l) if !l.luma.cbf)
}
