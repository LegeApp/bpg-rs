//! Transform-tree decisions.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::encoder::Encoder;

use super::depth::StillSearchDepth;
use super::emit;
use super::ledger::WorkBucket;
use super::overlay::OverlayCache;
use super::plan::TtPlan;
use super::price::split_flag_bits;
use super::source::CtuSourceCache;
use crate::encoder::types::{
    MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb,
};

/// Lowest `log2` luma TU size StillSearch will split down to. Keeping this at
/// 8x8 (`> 3`) guarantees ordinary 2Nx2N leaves carry chroma; PartNxN handles
/// the special forced 8x8->4x4 split parent chroma separately.
const MIN_SPLIT_LOG2: u8 = 3;

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
            && log2_size > MIN_SPLIT_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH;

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
            );
        }

        let mark0 = self.overlay.mark();
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
        let split_end = self.overlay.mark();
        let (leaf_tt, leaf_cost) = self.eval_tt_leaf(
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
            self.overlay.truncate(split_end);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            super::api::LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.overlay.drain_range(mark0, split_end);
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
        );
        let mut cost = luma.cost;

        let mut cb = emit::empty_block();
        let mut cr = emit::empty_block();
        let mut cb1 = emit::empty_block();
        let mut cr1 = emit::empty_block();
        let mut chroma_log2 = 0;
        let chroma_mode_for_syntax = luma_mode;

        if has_chroma_tb(cat, log2_size) {
            if let Some((cx, cy, clog2, count)) = chroma_tb_geom(cat, x0, y0, log2_size) {
                chroma_log2 = clog2;
                let cmode = IntraPredMode::from_u8(chroma_pred_mode(cat, chroma_mode_for_syntax))
                    .unwrap_or(IntraPredMode::Dc);
                let qp_c = state.cur_qp_c;
                let step = 1u32 << clog2;

                let cb_e =
                    self.eval_component(state, cx, cy, clog2, 1, cmode, qp_c, trafo_depth, lambda);
                cost += cb_e.cost;
                cb = cb_e.into_plan_block();

                let cr_e =
                    self.eval_component(state, cx, cy, clog2, 2, cmode, qp_c, trafo_depth, lambda);
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
}
