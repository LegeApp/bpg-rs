//! Transform-tree decisions.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::effort::{ChromaTiming, SplitSearch};
use crate::encoder::Encoder;

use super::depth::StillSearchDepth;
use super::emit;
use super::eval::{QuantMode, ResidualPricingMode};
use super::ledger::WorkBucket;
use super::overlay::OverlayCache;
use super::plan::TtPlan;
use super::price::split_flag_bits;
use super::source::CtuSourceCache;
use crate::encoder::types::{
    MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb,
};

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
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
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
        let can_split = !must_split
            && state.cat != 2
            && log2_size > state.effort_template.tu.min_split_log2
            && trafo_depth < MAX_INTRA_TT_DEPTH;

        // Determine evaluation scope from the effort template. Luma-only search
        // skips chroma evaluation during luma mode/TU trials; chroma is
        // re-attached only for the winner. Reference/Placebo templates enable
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
            );
        }
        if !can_split {
            return self.eval_tt_leaf(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                false,
                lambda,
                QuantMode::HardQuantSearch,
                residual_pricing,
                true,
                scope,
            );
        }

        // Leaf-first evaluation: evaluate the leaf candidate before the split.
        let mark0 = self.overlay.mark();
        let (leaf_tt, leaf_cost) = self.eval_tt_leaf(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            true,
            lambda,
            QuantMode::HardQuantSearch,
            residual_pricing,
            true,
            scope,
        );
        let leaf_saved = self.overlay.detach_from(mark0);

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

            let skip = tu_policy.zero_residual_early_terminate
                && leaf_has_no_luma_residual(&leaf_tt)
                || tu_policy.low_residual_early_terminate
                    && leaf_cost_per_px < tu_policy.low_distortion_per_px
                || tu_policy.split_search == SplitSearch::ForceLeaf;

            if skip {
                self.workspace.tu_split_early_terminations += 1;
                self.overlay.reattach(leaf_saved);
                return (leaf_tt, leaf_cost);
            }
        }

        let (split_tt, split_cost) = self.eval_tt_split(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            true,
            lambda,
        );

        if split_cost < leaf_cost {
            #[cfg(test)]
            super::api::SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            super::api::LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            (leaf_tt, leaf_cost)
        }
    }

    /// Evaluate this node as a leaf TU (luma + chroma), returning its internal
    /// plan and RD cost and pushing its recon to the overlay.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_tt_leaf(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
        quant_mode: QuantMode,
        residual_pricing: ResidualPricingMode,
        retain_coeff: bool,
        scope: TtEvalScope,
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuLeaf);

        let cat = state.cat;
        let qp_y = state.cur_qp_y;
        let luma_pred = IntraPredMode::from_u8(luma_mode).unwrap_or(IntraPredMode::Dc);
        let luma = self.eval_component(
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
        );
        let mut cost = luma.cost;

        let mut cb = emit::empty_block();
        let mut cr = emit::empty_block();
        let mut cb1 = emit::empty_block();
        let mut cr1 = emit::empty_block();
        let mut chroma_log2 = 0;
        let chroma_mode_for_syntax = luma_mode;

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
            let bits = split_flag_bits(&self.workspace.price_base, log2_size, false);
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
            && log2_size > state.effort_template.tu.min_split_log2
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
            QuantMode::HardQuantSearch,
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
            let bits = split_flag_bits(&self.workspace.price_base, log2_size, false);
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

    /// Evaluate this node as a split into four child transform trees.
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
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let (k0, c0) = self.decide_tt(state, x0, y0, kid_log2, kid_depth, luma_mode, lambda);
        let (k1, c1) = self.decide_tt(state, x0 + half, y0, kid_log2, kid_depth, luma_mode, lambda);
        let (k2, c2) = self.decide_tt(state, x0, y0 + half, kid_log2, kid_depth, luma_mode, lambda);
        let (k3, c3) = self.decide_tt(
            state,
            x0 + half,
            y0 + half,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
        );

        let kids = vec![k0, k1, k2, k3];
        let cbf_cb = kids.iter().any(TtPlan::cbf_cb);
        let cbf_cr = kids.iter().any(TtPlan::cbf_cr);
        let cbf_cb1 = kids.iter().any(TtPlan::cbf_cb1);
        let cbf_cr1 = kids.iter().any(TtPlan::cbf_cr1);

        let mut cost = c0 + c1 + c2 + c3;
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_base, log2_size, true);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        let tt = TtPlan::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cr,
            cbf_cb1,
            cbf_cr1,
            parent_chroma: None,
            kids,
        };
        (tt, cost)
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
        let can_split = !must_split
            && state.cat != 2
            && log2_size > cfg.tu.min_split_log2
            && trafo_depth < cfg.tu.max_extra_depth;

        if must_split || matches!(cfg.tu.split_mode, crate::effort::TuSplitMode::ForceSplit) {
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
            );
        }
        if !can_split || matches!(cfg.tu.split_mode, crate::effort::TuSplitMode::Disabled) {
            return self.eval_tt_leaf(
                state,
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                false,
                lambda,
                cfg.quant,
                cfg.residual_pricing,
                cfg.retain_coeff,
                cfg.scope,
            );
        }

        // Leaf-first evaluation.
        let mark0 = self.overlay.mark();
        let (leaf_tt, leaf_cost) = self.eval_tt_leaf(
            state,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            true,
            lambda,
            cfg.quant,
            cfg.residual_pricing,
            cfg.retain_coeff,
            cfg.scope,
        );
        let leaf_saved = self.overlay.detach_from(mark0);

        // TU early termination gates (driven by config).
        {
            let num_px = 1u64 << (2 * log2_size);
            let leaf_cost_per_px = leaf_cost / num_px as f64;
            let skip = cfg.tu.zero_residual_early_terminate && leaf_has_no_luma_residual(&leaf_tt)
                || cfg.tu.low_residual_early_terminate
                    && leaf_cost_per_px < cfg.tu.low_distortion_per_px
                || matches!(
                    cfg.tu.split_mode,
                    crate::effort::TuSplitMode::LeafFirstEarlyTerminate
                );
            if skip {
                self.workspace.tu_split_early_terminations += 1;
                self.overlay.reattach(leaf_saved);
                return (leaf_tt, leaf_cost);
            }
        }

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
        );

        if split_cost < leaf_cost {
            #[cfg(test)]
            super::api::SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            super::api::LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            (leaf_tt, leaf_cost)
        }
    }

    /// Evaluate a split node using an explicit config, recursing via
    /// [`decide_tt_with_config`].
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
    ) -> (TtPlan, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let (k0, c0) =
            self.decide_tt_with_config(state, x0, y0, kid_log2, kid_depth, luma_mode, lambda, cfg);
        let (k1, c1) = self.decide_tt_with_config(
            state,
            x0 + half,
            y0,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
            cfg,
        );
        let (k2, c2) = self.decide_tt_with_config(
            state,
            x0,
            y0 + half,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
            cfg,
        );
        let (k3, c3) = self.decide_tt_with_config(
            state,
            x0 + half,
            y0 + half,
            kid_log2,
            kid_depth,
            luma_mode,
            lambda,
            cfg,
        );

        let kids = vec![k0, k1, k2, k3];
        let cbf_cb = kids.iter().any(TtPlan::cbf_cb);
        let cbf_cr = kids.iter().any(TtPlan::cbf_cr);
        let cbf_cb1 = kids.iter().any(TtPlan::cbf_cb1);
        let cbf_cr1 = kids.iter().any(TtPlan::cbf_cr1);

        let mut cost = c0 + c1 + c2 + c3;
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_base, log2_size, true);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        let tt = TtPlan::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cr,
            cbf_cb1,
            cbf_cr1,
            parent_chroma: None,
            kids,
        };
        (tt, cost)
    }
}

/// Returns `true` when the TT plan is a leaf whose luma block has no coded
/// residual (CBF=false). Used by leaf-first early termination.
fn leaf_has_no_luma_residual(tt: &TtPlan) -> bool {
    matches!(tt, TtPlan::Leaf(l) if !l.luma.cbf)
}
