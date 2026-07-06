//! 8x8 PartNxN search path.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::encoder::Encoder;
use crate::primitives::intra_angs;
use crate::primitives::{satd_u8, satd_u16};

use super::canvas::CanvasSaved;
use super::depth::StillSearchDepth;
use super::emit;
use super::eval::{BlockTrial, ResidualPricingMode};
use super::ledger::{StillSearchLedger, WorkBucket};
use super::plan::{CuPlan, NxnInfoPlan, ParentChromaPlan, TtPlan};
use super::price::luma_mode_bits;
use super::rough::build_luma_shortlist;
use super::source::CtuSourceCache;
use crate::effort::{AngularFamily, ModeClass, ModeCost};
use crate::encoder::types::{CHROMA_DM_IDX, chroma_pred_mode};

/// Rough-mode cache for the four PartNxN PUs. Uses a fixed-size array
/// large enough for any shortlist policy max_modes + MPMs.
const NXN_MAX_SHORTLIST: usize = 16;

pub(super) struct NxnRoughSet {
    shortlist: [[u8; NXN_MAX_SHORTLIST]; 4],
    len: [usize; 4],
}

impl NxnRoughSet {
    fn new() -> Self {
        Self {
            shortlist: [[0; NXN_MAX_SHORTLIST]; 4],
            len: [0; 4],
        }
    }

    fn set(&mut self, pu: usize, modes: &[u8]) {
        self.len[pu] = modes.len();
        self.shortlist[pu][..modes.len()].copy_from_slice(modes);
    }

    fn modes(&self, pu: usize) -> &[u8] {
        &self.shortlist[pu][..self.len[pu]]
    }
}

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
{
    /// Build and price an 8x8 `PartNxN` candidate. Luma is coded as four 4x4
    /// PUs/TUs in decoder z-order; each PU's selected recon is pushed before the
    /// next PU so legal intra references see prior siblings through the overlay.
    /// For Phase 7 this supports 4:2:0 parent chroma, 4:4:4 per-PU chroma,
    /// and grayscale. 4:2:2 stacked parent chroma is still deferred.
    pub(super) fn decide_cu_part_nxn(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        ct_depth: u8,
        lambda: f64,
    ) -> (CuPlan, f64) {
        state.stats.partnxn_cu_trials += 1;
        state.set_ct_depth(x0, y0, 3, ct_depth);
        if state.aq.active {
            state.aq_set_cu_qp(x0, y0);
        }

        let pu_xy = [(x0, y0), (x0 + 4, y0), (x0, y0 + 4), (x0 + 4, y0 + 4)];
        let mut luma_modes = [IntraPredMode::Dc.as_u8(); 4];
        let mut mpms = [[IntraPredMode::Dc; 3]; 4];
        let mut kids = Vec::with_capacity(4);
        let mut cost = 0.0;
        let mut rough_set = NxnRoughSet::new();

        let evolve = super::env::ctx_evolve_search();
        for (pu, &(px, py)) in pu_xy.iter().enumerate() {
            let mpm = super::intra_mpm(state, px, py);
            mpms[pu] = mpm;
            self.fill_nxn_rough_set(state, px, py, pu, mpm, lambda, &mut rough_set);
            let (mode, luma, recon, pu_cost) =
                self.eval_nxn4_batch(state, px, py, pu, mpm, lambda, &rough_set);
            self.overlay.reattach(recon);
            luma_modes[pu] = mode;
            state.store_mode(px, py, 2, mode);
            cost += pu_cost;
            if evolve {
                // Commit the winning PU's luma syntax (mode trials never
                // commit): CBF bin + residual context evolution, so later
                // PUs price from post-PU contexts like x265's coder does.
                self.workspace.price_cur.models[crate::contexts::ctx::CBF_LUMA]
                    .update(luma.cbf as u8);
                if luma.cbf {
                    if let Some(id) = luma.coeff {
                        let scan = crate::residual::get_scan_order(2, mode, 0, state.cat);
                        let _ = crate::residual::estimate_residual_bits(
                            &mut self.workspace.price_cur,
                            self.workspace.coeffs.get(id),
                            2,
                            0,
                            scan,
                            state.sign_data_hiding,
                        );
                    }
                }
            }

            let mut cb = emit::empty_block();
            let mut cr = emit::empty_block();
            let chroma_log2 = if state.cat == 3 { 2 } else { 0 };
            if state.cat == 3 {
                let prev_commit = self.workspace.commit_ctx;
                let prev_skip = self.workspace.commit_ctx_skip_chroma;
                self.workspace.commit_ctx = evolve;
                self.workspace.commit_ctx_skip_chroma = false;
                let chroma_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);
                let cb_e = self.eval_component(
                    state,
                    px,
                    py,
                    2,
                    1,
                    chroma_mode,
                    state.cur_qp_c,
                    1,
                    lambda,
                    super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
                    ResidualPricingMode::Exact,
                    true,
                );
                cost += cb_e.cost;
                cb = cb_e.into_plan_block();
                let cr_e = self.eval_component(
                    state,
                    px,
                    py,
                    2,
                    2,
                    chroma_mode,
                    state.cur_qp_c,
                    1,
                    lambda,
                    super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
                    ResidualPricingMode::Exact,
                    true,
                );
                cost += cr_e.cost;
                cr = cr_e.into_plan_block();
                self.workspace.commit_ctx = prev_commit;
                self.workspace.commit_ctx_skip_chroma = prev_skip;
            }

            let tt = emit::leaf_tu(
                2,
                chroma_log2,
                1,
                mode,
                mode,
                luma.into_plan_block(),
                cb,
                cr,
                emit::empty_block(),
                emit::empty_block(),
            );
            kids.push(tt);
        }

        let mut parent_chroma = None;
        if matches!(state.cat, 1 | 2) {
            let prev_commit = self.workspace.commit_ctx;
            let prev_skip = self.workspace.commit_ctx_skip_chroma;
            self.workspace.commit_ctx = evolve;
            self.workspace.commit_ctx_skip_chroma = false;
            let (pc, pc_cost) =
                self.eval_part_nxn_parent_chroma_subsampled(state, x0, y0, luma_modes[0], lambda);
            self.workspace.commit_ctx = prev_commit;
            self.workspace.commit_ctx_skip_chroma = prev_skip;
            cost += pc_cost;
            parent_chroma = Some(pc);
        }

        let (cbf_cb, cbf_cb1, cbf_cr, cbf_cr1) = if state.cat == 3 {
            (
                kids.iter().any(TtPlan::cbf_cb),
                false,
                kids.iter().any(TtPlan::cbf_cr),
                false,
            )
        } else {
            (
                parent_chroma.as_ref().is_some_and(|c| c.cb.cbf),
                parent_chroma.as_ref().is_some_and(|c| c.cb1.cbf),
                parent_chroma.as_ref().is_some_and(|c| c.cr.cbf),
                parent_chroma.as_ref().is_some_and(|c| c.cr1.cbf),
            )
        };
        let tt = TtPlan::Split {
            log2_size: 3,
            trafo_depth: 0,
            cbf_cb,
            cbf_cr,
            cbf_cb1,
            cbf_cr1,
            parent_chroma,
            kids,
        };

        if evolve {
            // NxN CU header syntax: part_mode=NxN, four prev_intra flags,
            // then the chroma mode(s). Header models are disjoint from
            // content models, so applying them after content preserves the
            // writer's per-model bin sequences.
            let ctxs = &mut self.workspace.price_cur;
            ctxs.models[crate::contexts::ctx::PART_MODE].update(0);
            for pu in 0..4 {
                let in_mpm = mpms[pu].iter().any(|m| m.as_u8() == luma_modes[pu]);
                ctxs.models[crate::contexts::ctx::PREV_INTRA_LUMA_PRED_FLAG].update(in_mpm as u8);
            }
            if state.cat != 0 {
                let n = if state.cat == 3 { 4 } else { 1 };
                for _ in 0..n {
                    ctxs.models[crate::contexts::ctx::INTRA_CHROMA_PRED_MODE].update(0);
                }
            }
        }

        let mut leaf = emit::cu_leaf(mpms[0], luma_modes[0], tt);
        leaf.chroma_mode_idx = CHROMA_DM_IDX;
        leaf.nxn = Some(NxnInfoPlan {
            luma_modes,
            mpms,
            chroma_mode_idx: [CHROMA_DM_IDX; 4],
        });
        (CuPlan::Leaf(leaf), cost)
    }

    fn eval_part_nxn_parent_chroma_subsampled(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        chroma_mode: u8,
        lambda: f64,
    ) -> (ParentChromaPlan, f64) {
        debug_assert!(matches!(state.cat, 1 | 2));
        let cx = x0 / 2;
        let cy = if state.cat == 1 { y0 / 2 } else { y0 };
        let clog2 = 2;
        let step = 1u32 << clog2;
        let pred_mode = IntraPredMode::from_u8(chroma_pred_mode(state.cat, chroma_mode))
            .unwrap_or(IntraPredMode::Dc);

        let cb0 = self.eval_component(
            state,
            cx,
            cy,
            clog2,
            1,
            pred_mode,
            state.cur_qp_c,
            0,
            lambda,
            super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
            ResidualPricingMode::Exact,
            true,
        );
        let mut cost = cb0.cost;
        let mut cb1 = None;
        if state.cat == 2 {
            let cb = self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                1,
                pred_mode,
                state.cur_qp_c,
                0,
                lambda,
                super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
                ResidualPricingMode::Exact,
                true,
            );
            cost += cb.cost;
            cb1 = Some(cb);
        }

        let cr0 = self.eval_component(
            state,
            cx,
            cy,
            clog2,
            2,
            pred_mode,
            state.cur_qp_c,
            0,
            lambda,
            super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
            ResidualPricingMode::Exact,
            true,
        );
        cost += cr0.cost;
        let mut cr1 = None;
        if state.cat == 2 {
            let cr = self.eval_component(
                state,
                cx,
                cy + step,
                clog2,
                2,
                pred_mode,
                state.cur_qp_c,
                0,
                lambda,
                super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
                ResidualPricingMode::Exact,
                true,
            );
            cost += cr.cost;
            cr1 = Some(cr);
        }

        let parent = ParentChromaPlan {
            log2_size: clog2,
            chroma_mode,
            cb: cb0.into_plan_block(),
            cb1: cb1
                .map(BlockTrial::into_plan_block)
                .unwrap_or_else(emit::empty_block),
            cr: cr0.into_plan_block(),
            cr1: cr1
                .map(BlockTrial::into_plan_block)
                .unwrap_or_else(emit::empty_block),
        };
        (parent, cost)
    }

    fn fill_nxn_rough_set(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        pu: usize,
        mpm: [IntraPredMode; 3],
        lambda: f64,
        set: &mut NxnRoughSet,
    ) {
        let timer = StillSearchLedger::start_timer();
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let scale = CabacEstimator::SCALE as f64;
        let size = 4usize;
        let lambda_sad = lambda.sqrt();
        let mut rough = [(0.0f64, 0u8); 35];
        // Build the 4x4 angular predictions (modes 2..=34) once from a single
        // overlay-aware reference border, mirroring the CU-level rough batch.
        // Planar/DC (0/1) stay on the per-mode predictor.
        let n = size * size;
        if state.bit_depth == 8 {
            // Predict directly to u8 — one batch narrow, no per-mode narrow loop.
            let mut batch_u8 = [0u8; intra_angs::ANGULAR_MODES * 16];
            self.predict_all_angular_into_u8(
                state,
                x0,
                y0,
                2,
                &mut batch_u8[..intra_angs::ANGULAR_MODES * n],
            );
            let mut src = [0u8; 16];
            self.source.sample_block_u8(0, x0, y0, size, &mut src);
            let mut pred = [0u8; 16];
            let mut tmp_u16 = Vec::with_capacity(16);
            for (idx, m) in (0u8..=1).enumerate() {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into_u8(state, x0, y0, 2, 0, mode, &mut pred, &mut tmp_u16);
                let satd = satd_u8(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_cur, mpm_u8, m);
                rough[idx] = (satd as f64 + lambda_sad * mbits as f64 / scale, m);
            }
            for m in 2u8..=34 {
                let off = intra_angs::slot_offset(m, 2);
                let satd = satd_u8(&src, size, &batch_u8[off..off + n], size, size);
                let mbits = luma_mode_bits(&self.workspace.price_cur, mpm_u8, m);
                rough[m as usize] = (satd as f64 + lambda_sad * mbits as f64 / scale, m);
            }
        } else {
            let mut batch = [0u16; intra_angs::ANGULAR_MODES * 16];
            self.predict_all_angular_into_u16(
                state,
                x0,
                y0,
                2,
                &mut batch[..intra_angs::ANGULAR_MODES * n],
            );
            let mut src = [0u16; 16];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self.source.sample(0, x0 + x as u32, y0 + y as u32);
                }
            }
            let mut pred = [0u16; 16];
            for (idx, m) in (0u8..=1).enumerate() {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into(state, x0, y0, 2, 0, mode, &mut pred);
                let satd = satd_u16(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_cur, mpm_u8, m);
                rough[idx] = (satd as f64 + lambda_sad * mbits as f64 / scale, m);
            }
            for m in 2u8..=34 {
                let off = intra_angs::slot_offset(m, 2);
                let slot = &batch[off..off + n];
                let satd = satd_u16(&src, size, slot, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_cur, mpm_u8, m);
                rough[m as usize] = (satd as f64 + lambda_sad * mbits as f64 / scale, m);
            }
        }
        rough.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        self.workspace.ledger.bump(WorkBucket::NxnRough);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::NxnRough, timer);

        let sl_policy = &state.effort_template.luma.shortlist;
        fn classify_nxn(m: u8) -> (ModeClass, Option<AngularFamily>) {
            if m == 0 {
                (ModeClass::Planar, None)
            } else if m == 1 {
                (ModeClass::Dc, None)
            } else {
                (ModeClass::Angular, AngularFamily::classify(m))
            }
        }
        let rough_costs: Vec<ModeCost> = rough
            .iter()
            .map(|&(cost, m)| {
                let (class, family) = classify_nxn(m);
                ModeCost {
                    mode: m,
                    cost,
                    satd: 0,
                    class,
                    family,
                }
            })
            .collect();
        let modes = build_luma_shortlist(&rough_costs, mpm_u8, sl_policy);
        set.set(pu, &modes);
    }

    /// Exact-evaluate the selected PU's rough set. The name is intentionally the
    /// planned batch boundary; this first version still evaluates one PU at a
    /// time while preserving the later function shape.
    fn eval_nxn4_batch(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        pu: usize,
        mpm: [IntraPredMode; 3],
        lambda: f64,
        rough_set: &NxnRoughSet,
    ) -> (u8, BlockTrial, CanvasSaved, f64) {
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let scale = CabacEstimator::SCALE as f64;
        let mut best: Option<(u8, BlockTrial, CanvasSaved, f64)> = None;
        for &mode in rough_set.modes(pu) {
            let timer = StillSearchLedger::start_timer();
            let mark = self.overlay.mark();
            let pred_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);
            let trial = self.eval_component(
                state,
                x0,
                y0,
                2,
                0,
                pred_mode,
                state.cur_qp_y,
                1,
                lambda,
                super::eval::search_trial_quant(state.effort_template.luma.exact.quant),
                ResidualPricingMode::Exact,
                true,
            );
            let mbits = luma_mode_bits(&self.workspace.price_cur, mpm_u8, mode);
            let total = trial.cost + lambda * mbits as f64 / scale;
            let recon = self.overlay.detach_from(mark);
            match &best {
                Some((_, _, _, best_total)) if *best_total <= total => {}
                _ => best = Some((mode, trial, recon, total)),
            }
            self.workspace.ledger.bump(WorkBucket::NxnBatch);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::NxnBatch, timer);
        }

        let (mode, trial, recon, total) = best.expect("PartNxN shortlist is never empty");
        state.stats.partnxn_code_block_calls += 1;
        (mode, trial, recon, total)
    }
}
