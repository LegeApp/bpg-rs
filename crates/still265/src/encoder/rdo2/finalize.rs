//! Writer-compatible final materialization for rdo2-selected winners.
//!
//! These routines own the final replay boundary for the staged rdo2 search:
//! they produce the existing `Tt`/`CuNode` structures expected by `write.rs`
//! while coding committed transform blocks through `rdo2_eval_leaf_block`
//! instead of re-entering the legacy block-coding path.

use bpg_hevc_decode::hevc::intra::fill_mpm_candidates;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use super::super::types::{
    chroma_pred_mode, chroma_tb_geom, CodedBlock, CuLeaf, CuNode, LeafTu, NxnInfo, ParentChromaTu,
    Tt,
};
use super::policy::{EvalKind, EvalPolicy};
use crate::contexts::Contexts;
use crate::plan::{CuLeafPlan, CuPlan, DecisionConfidence, TtPlan};
use crate::trace::WorkBucket;

impl<'a> super::super::Encoder<'a> {
    pub(in crate::encoder) fn rdo2_final_parent_chroma_tu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Option<ParentChromaTu> {
        if self.cat != 1 || log2_size != 3 {
            return None;
        }
        let (cx, cy, clog2, _count) = chroma_tb_geom(self.cat, x0, y0, log2_size)?;
        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let policy = EvalPolicy::for_kind(EvalKind::Final).with_bucket(WorkBucket::FinalReplay);
        let cb = self
            .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 1, pred, self.cur_qp_c, policy)
            .coded;
        let cr = self
            .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 2, pred, self.cur_qp_c, policy)
            .coded;
        Some(ParentChromaTu {
            log2_size: clog2,
            chroma_mode,
            cb,
            cr,
        })
    }

    pub(in crate::encoder) fn rdo2_final_code_tt_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let policy = EvalPolicy::for_kind(EvalKind::Final).with_bucket(WorkBucket::FinalReplay);
        let luma = self
            .rdo2_eval_leaf_block(ctxs, x0, y0, log2_size, 0, luma_mode, self.cur_qp_y, policy)
            .coded;

        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let (chroma_log2, cb, cr, cb1, cr1) = match chroma_tb_geom(self.cat, x0, y0, log2_size) {
            Some((cx, cy, clog2, count)) => {
                let size = 1u32 << clog2;
                let cb = self
                    .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 1, pred, self.cur_qp_c, policy)
                    .coded;
                let cr = self
                    .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 2, pred, self.cur_qp_c, policy)
                    .coded;
                let (cb1, cr1) = if count > 1 {
                    let ty = cy + size;
                    (
                        self.rdo2_eval_leaf_block(
                            ctxs,
                            cx,
                            ty,
                            clog2,
                            1,
                            pred,
                            self.cur_qp_c,
                            policy,
                        )
                        .coded,
                        self.rdo2_eval_leaf_block(
                            ctxs,
                            cx,
                            ty,
                            clog2,
                            2,
                            pred,
                            self.cur_qp_c,
                            policy,
                        )
                        .coded,
                    )
                } else {
                    (CodedBlock::empty(), CodedBlock::empty())
                };
                (clog2, cb, cr, cb1, cr1)
            }
            None => (
                0,
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
            ),
        };

        Tt::Leaf(LeafTu {
            log2_size,
            chroma_log2,
            trafo_depth,
            luma_mode,
            chroma_mode,
            luma,
            cb,
            cr,
            cb1,
            cr1,
        })
    }

    pub(in crate::encoder) fn rdo2_final_code_tt(
        &mut self,
        plan: &TtPlan,
        x0: u32,
        y0: u32,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        match plan {
            TtPlan::Leaf {
                log2_size,
                trafo_depth,
            } => self.rdo2_final_code_tt_leaf(
                x0,
                y0,
                *log2_size,
                *trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
            TtPlan::Split {
                log2_size,
                trafo_depth,
                kids,
                parent_chroma,
            } => {
                let half = 1u32 << (*log2_size - 1);
                let child_pos = [
                    (x0, y0),
                    (x0 + half, y0),
                    (x0, y0 + half),
                    (x0 + half, y0 + half),
                ];
                let coded_kids: Vec<Tt> = kids
                    .iter()
                    .zip(child_pos)
                    .map(|(kid, (kx, ky))| {
                        self.rdo2_final_code_tt(kid, kx, ky, luma_mode, chroma_mode, ctxs)
                    })
                    .collect();
                let coded_parent_chroma = parent_chroma.as_ref().and_then(|_| {
                    self.rdo2_final_parent_chroma_tu(x0, y0, *log2_size, chroma_mode, ctxs)
                });
                let cbf_cb = coded_parent_chroma
                    .as_ref()
                    .map(|c| c.cb.cbf)
                    .unwrap_or_else(|| coded_kids.iter().any(|k| k.cbf_cb() || k.cbf_cb1()));
                let cbf_cr = coded_parent_chroma
                    .as_ref()
                    .map(|c| c.cr.cbf)
                    .unwrap_or_else(|| coded_kids.iter().any(|k| k.cbf_cr() || k.cbf_cr1()));
                let cbf_cb1 = coded_kids.iter().any(|k| k.cbf_cb1());
                let cbf_cr1 = coded_kids.iter().any(|k| k.cbf_cr1());
                Tt::Split {
                    log2_size: *log2_size,
                    trafo_depth: *trafo_depth,
                    cbf_cb,
                    cbf_cr,
                    cbf_cb1,
                    cbf_cr1,
                    parent_chroma: coded_parent_chroma,
                    kids: coded_kids,
                }
            }
        }
    }

    pub(in crate::encoder) fn rdo2_final_code_cu_nxn(
        &mut self,
        x0: u32,
        y0: u32,
        luma_modes: [u8; 4],
        chroma_mode_idx: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        self.set_ct_depth(x0, y0, 3, ct_depth);
        let positions = [(x0, y0), (x0 + 4, y0), (x0, y0 + 4), (x0 + 4, y0 + 4)];
        let mut mpms = [[IntraPredMode::Dc; 3]; 4];
        let mut kids: Vec<Tt> = Vec::with_capacity(4);
        for (i, &(px, py)) in positions.iter().enumerate() {
            let cand_a = self.neighbor_left_mode(px, py);
            let cand_b = self.neighbor_above_mode(px, py);
            mpms[i] = fill_mpm_candidates(cand_a, cand_b);
            self.store_mode(px, py, 2, luma_modes[i]);
            kids.push(self.rdo2_final_code_tt_leaf(
                px,
                py,
                2,
                1,
                luma_modes[i],
                luma_modes[i],
                ctxs,
            ));
        }
        let chroma_mode = Self::chroma_mode_from_idx(luma_modes[0], chroma_mode_idx);
        let parent_chroma = self.rdo2_final_parent_chroma_tu(x0, y0, 3, chroma_mode, ctxs);
        let cbf_cb = parent_chroma.as_ref().map(|c| c.cb.cbf).unwrap_or(false);
        let cbf_cr = parent_chroma.as_ref().map(|c| c.cr.cbf).unwrap_or(false);
        let tt = Tt::Split {
            log2_size: 3,
            trafo_depth: 0,
            cbf_cb,
            cbf_cr,
            cbf_cb1: false,
            cbf_cr1: false,
            parent_chroma,
            kids,
        };
        CuNode::Leaf(CuLeaf {
            mpm: mpms[0],
            luma_mode: luma_modes[0],
            chroma_mode_idx,
            confidence: DecisionConfidence::Clear,
            tt,
            nxn: Some(NxnInfo { luma_modes, mpms }),
        })
    }

    pub(in crate::encoder) fn rdo2_final_code_cu(
        &mut self,
        plan: &CuPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        if !self.aq.active {
            return self.rdo2_final_code_cu_inner(plan, x0, y0, log2_cb_size, ct_depth, ctxs);
        }

        let saved_policy = self.cur_policy;
        let saved_qp = (self.cur_qp_y, self.cur_qp_c);
        self.cur_policy = Some(self.region_policy(x0, y0, log2_cb_size));
        self.aq_set_cu_qp(x0, y0);
        let node = self.rdo2_final_code_cu_inner(plan, x0, y0, log2_cb_size, ct_depth, ctxs);
        self.cur_policy = saved_policy;
        (self.cur_qp_y, self.cur_qp_c) = saved_qp;
        node
    }

    fn rdo2_final_code_cu_inner(
        &mut self,
        plan: &CuPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        match plan {
            CuPlan::Leaf(CuLeafPlan {
                // `mpm` from the plan is the screen-time list; the final replay
                // recomputes it from the final neighbour modes below.
                mpm: _,
                luma_mode,
                chroma_mode_idx,
                chroma_mode,
                tt,
                nxn,
            }) => {
                if let Some(luma_modes) = nxn {
                    return self.rdo2_final_code_cu_nxn(
                        x0,
                        y0,
                        *luma_modes,
                        *chroma_mode_idx,
                        ct_depth,
                        ctxs,
                    );
                }
                self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
                // Recompute the MPM list from the *final* neighbour modes rather
                // than trusting the screen-time `mpm` carried in the plan. The
                // decoder derives MPM from the reconstructed neighbour modes, so
                // the written `mpm_idx`/`rem_intra` is only consistent if we use
                // the same final-neighbour MPM here (the PartNxN path already does
                // this). A stale screen-time MPM can mis-signal the luma mode when
                // the winning neighbour mode differs from what the screen saw.
                let cand_a = self.neighbor_left_mode(x0, y0);
                let cand_b = self.neighbor_above_mode(x0, y0);
                let mpm = fill_mpm_candidates(cand_a, cand_b);
                self.store_mode(x0, y0, log2_cb_size, *luma_mode);
                let tt = self.rdo2_final_code_tt(tt, x0, y0, *luma_mode, *chroma_mode, ctxs);
                CuNode::Leaf(CuLeaf {
                    mpm,
                    luma_mode: *luma_mode,
                    chroma_mode_idx: *chroma_mode_idx,
                    confidence: DecisionConfidence::Clear,
                    tt,
                    nxn: None,
                })
            }
            CuPlan::Split { kids } => {
                self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
                let half = (1u32 << log2_cb_size) / 2;
                let x1 = x0 + half;
                let y1 = y0 + half;
                let mut kid_iter = kids.iter();
                let mut coded = Vec::new();
                coded.push(self.rdo2_final_code_cu(
                    kid_iter.next().expect("split CU has top-left child"),
                    x0,
                    y0,
                    log2_cb_size - 1,
                    ct_depth + 1,
                    ctxs,
                ));
                if x1 < self.display_width {
                    coded.push(self.rdo2_final_code_cu(
                        kid_iter.next().expect("split CU has top-right child"),
                        x1,
                        y0,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                if y1 < self.display_height {
                    coded.push(self.rdo2_final_code_cu(
                        kid_iter.next().expect("split CU has bottom-left child"),
                        x0,
                        y1,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                if x1 < self.display_width && y1 < self.display_height {
                    coded.push(self.rdo2_final_code_cu(
                        kid_iter.next().expect("split CU has bottom-right child"),
                        x1,
                        y1,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                CuNode::Split { kids: coded }
            }
        }
    }
}
