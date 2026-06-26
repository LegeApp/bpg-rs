//! Winner-only RDOQ finalize pass.
//!
//! Search and screening (`decide_*`, `rough`, `nxn`) run entirely on hard
//! quantization. After the CU/TU structure for a CTU is chosen, this pass walks
//! the selected `CuPlan`/`TtPlan` in decoder (z-)order and re-codes each block
//! with [`QuantMode::RdoqFinal`], reusing the same shared component evaluator,
//! transform, sign-data-hiding, recon, and residual-pricing path. Only the
//! quantizer differs.
//!
//! The pass makes no decisions: the partition, modes, and TU structure are
//! fixed; it only replaces the coefficients/recon/CBFs with RDOQ-optimized
//! ones. It updates the recon overlay after each finalized block so later
//! blocks predict from finalized samples (avoiding the inconsistency of
//! predicting from hard-quant recon while committing RDOQ recon).

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::encoder::Encoder;
use crate::encoder::types::chroma_pred_mode;

use super::depth::StillSearchDepth;
use super::emit;
use super::eval::{QuantMode, ResidualPricingMode};
use super::overlay::OverlayCache;
use super::plan::{CuLeafPlan, CuPlan, ParentChromaPlan, TtPlan};
use super::source::CtuSourceCache;

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// Re-code the selected `plan` with RDOQ in decoder order, returning the
    /// finalized plan (RDOQ coeffs/CBFs) and leaving RDOQ recon in the overlay.
    pub(super) fn finalize_cu(
        &mut self,
        state: &Encoder<'_>,
        plan: CuPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        lambda: f64,
    ) -> CuPlan {
        match plan {
            CuPlan::Split { kids } => {
                let half = 1u32 << (log2_cb_size - 1);
                let kid_log2 = log2_cb_size - 1;
                let mut out = Vec::with_capacity(4);

                // Boundary CTUs omit children that start wholly outside the
                // display rectangle. Mirror `decide_cu_split`/`write_cu`'s
                // conditional child order exactly; a fixed four-offset zip
                // misplaces right/bottom-edge children and can commit patches
                // past the coded stride.
                let x1 = x0 + half;
                let y1 = y0 + half;
                let mut kids = kids.into_iter();

                if let Some(kid) = kids.next() {
                    out.push(self.finalize_cu(state, kid, x0, y0, kid_log2, lambda));
                }
                if x1 < state.display_width {
                    if let Some(kid) = kids.next() {
                        out.push(self.finalize_cu(state, kid, x1, y0, kid_log2, lambda));
                    }
                }
                if y1 < state.display_height {
                    if let Some(kid) = kids.next() {
                        out.push(self.finalize_cu(state, kid, x0, y1, kid_log2, lambda));
                    }
                }
                if x1 < state.display_width && y1 < state.display_height {
                    if let Some(kid) = kids.next() {
                        out.push(self.finalize_cu(state, kid, x1, y1, kid_log2, lambda));
                    }
                }
                CuPlan::Split { kids: out }
            }
            CuPlan::Leaf(leaf) => {
                CuPlan::Leaf(self.finalize_cu_leaf(state, leaf, x0, y0, log2_cb_size, lambda))
            }
        }
    }

    fn finalize_cu_leaf(
        &mut self,
        state: &Encoder<'_>,
        leaf: CuLeafPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        lambda: f64,
    ) -> CuLeafPlan {
        let tt = self.finalize_tt(state, leaf.tt, x0, y0, log2_cb_size, lambda);
        CuLeafPlan { tt, ..leaf }
    }

    fn finalize_tt(
        &mut self,
        state: &Encoder<'_>,
        tt: TtPlan,
        x0: u32,
        y0: u32,
        log2_size: u8,
        lambda: f64,
    ) -> TtPlan {
        match tt {
            // A leaf carries its own luma mode (a PartNxN PU stores its per-PU
            // mode here); re-code it (and any leaf-level chroma) with RDOQ. The
            // `code_split_flag` argument only affects the returned cost, which
            // the finalizer discards, so `false` is fine.
            TtPlan::Leaf(l) => {
                let (new_tt, _) = self.eval_tt_leaf(
                    state,
                    x0,
                    y0,
                    l.log2_size,
                    l.trafo_depth,
                    l.luma_mode,
                    false,
                    lambda,
                    QuantMode::RdoqFinal,
                    ResidualPricingMode::Exact,
                    true,
                    super::tu::TtEvalScope::FullComponents,
                );
                new_tt
            }
            TtPlan::Split {
                log2_size: ls,
                trafo_depth,
                parent_chroma,
                kids,
                ..
            } => {
                let half = 1u32 << (log2_size - 1);
                let kid_log2 = log2_size - 1;
                let offs = [(0, 0), (half, 0), (0, half), (half, half)];
                let mut new_kids = Vec::with_capacity(4);
                for (kid, (dx, dy)) in kids.into_iter().zip(offs) {
                    new_kids.push(self.finalize_tt(state, kid, x0 + dx, y0 + dy, kid_log2, lambda));
                }
                // Chroma at the split node (PartNxN 4:2:0/4:2:2 parent chroma)
                // is coded after the luma sub-blocks, matching decoder order.
                let new_parent =
                    parent_chroma.map(|pc| self.finalize_parent_chroma(state, x0, y0, pc, lambda));
                let cbf_cb = new_kids.iter().any(TtPlan::cbf_cb)
                    || new_parent.as_ref().is_some_and(|p| p.cb.cbf);
                let cbf_cr = new_kids.iter().any(TtPlan::cbf_cr)
                    || new_parent.as_ref().is_some_and(|p| p.cr.cbf);
                let cbf_cb1 = new_kids.iter().any(TtPlan::cbf_cb1)
                    || new_parent.as_ref().is_some_and(|p| p.cb1.cbf);
                let cbf_cr1 = new_kids.iter().any(TtPlan::cbf_cr1)
                    || new_parent.as_ref().is_some_and(|p| p.cr1.cbf);
                TtPlan::Split {
                    log2_size: ls,
                    trafo_depth,
                    cbf_cb,
                    cbf_cr,
                    cbf_cb1,
                    cbf_cr1,
                    parent_chroma: new_parent,
                    kids: new_kids,
                }
            }
        }
    }

    /// Re-code a PartNxN parent chroma group with RDOQ, mirroring
    /// `eval_part_nxn_parent_chroma_subsampled`'s geometry.
    fn finalize_parent_chroma(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        pc: ParentChromaPlan,
        lambda: f64,
    ) -> ParentChromaPlan {
        let clog2 = pc.log2_size;
        let cx = x0 / 2;
        let cy = if state.cat == 1 { y0 / 2 } else { y0 };
        let step = 1u32 << clog2;
        let pred_mode = IntraPredMode::from_u8(chroma_pred_mode(state.cat, pc.chroma_mode))
            .unwrap_or(IntraPredMode::Dc);
        let qp_c = state.cur_qp_c;
        let m = QuantMode::RdoqFinal;

        let cb0 = self.eval_component(
            state,
            cx,
            cy,
            clog2,
            1,
            pred_mode,
            qp_c,
            0,
            lambda,
            m,
            ResidualPricingMode::Exact,
            true,
        );
        let cb1 = if state.cat == 2 {
            Some(self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                1,
                pred_mode,
                qp_c,
                0,
                lambda,
                m,
                ResidualPricingMode::Exact,
                true,
            ))
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
            0,
            lambda,
            m,
            ResidualPricingMode::Exact,
            true,
        );
        let cr1 = if state.cat == 2 {
            Some(self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                2,
                pred_mode,
                qp_c,
                0,
                lambda,
                m,
                ResidualPricingMode::Exact,
                true,
            ))
        } else {
            None
        };

        ParentChromaPlan {
            log2_size: clog2,
            chroma_mode: pc.chroma_mode,
            cb: cb0.into_plan_block(),
            cb1: cb1
                .map(super::eval::BlockTrial::into_plan_block)
                .unwrap_or_else(emit::empty_block),
            cr: cr0.into_plan_block(),
            cr1: cr1
                .map(super::eval::BlockTrial::into_plan_block)
                .unwrap_or_else(emit::empty_block),
        }
    }
}
