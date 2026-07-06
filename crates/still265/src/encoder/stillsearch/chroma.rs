//! CU-level chroma intra mode search (x265 `estIntraPredChromaQT` parity).
//!
//! x265's rd 5/6 path full-RD evaluates the five allowed chroma modes
//! {planar, vertical, horizontal, DC, DM} once per CU after the luma
//! decision, then signals the winner through `intra_chroma_pred_mode`.
//! StillSearch historically hardwired DM. This pass keeps the luma/TT
//! structure fixed (like x265: chroma follows the luma TT), re-prices the
//! CU's chroma blocks under candidate modes, and rewrites the plan's chroma
//! blocks/mode fields, overlay recon, and (with evolving contexts) commits the
//! winner's chroma context updates that materialization deferred. Slow/fast
//! templates use a SATD rough pass to promote only the best non-DM candidate to
//! exact RD; `BPG_STILLSEARCH_CHROMA_ROUGH=0` or placebo restores exhaustive
//! exact evaluation.
//!
//! Gated by `BPG_STILLSEARCH_CHROMA_MODES` (default on). 4:2:2 keeps DM-only
//! coding until the pre-existing parent-chroma mode-mapping quirks there are
//! resolved.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::encoder::Encoder;
use crate::encoder::types::{CHROMA_DM_IDX, chroma_pred_mode, chroma_tb_geom, has_chroma_tb};
use crate::primitives::{sa8d_u8, sa8d_u16, satd_u8, satd_u16};
use crate::residual::get_scan_order;

use super::depth::StillSearchDepth;
use super::eval::ResidualPricingMode;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::plan::{CuLeafPlan, PlanBlock, TtPlan};
use super::price::entropy_bits;
use super::source::CtuSourceCache;

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
{
    /// Re-decide this 2Nx2N CU leaf's chroma intra mode across the five
    /// allowed modes. Returns the RD-cost delta relative to the incoming DM
    /// chroma coding; the caller adds it to the leaf cost. The winner rebuild
    /// updates plan blocks/overlay and commits deferred chroma contexts; the
    /// optimized rough path may replay already-materialized DM contexts instead
    /// when DM wins.
    pub(super) fn select_cu_chroma_mode(
        &mut self,
        state: &mut Encoder<'_>,
        leaf: &mut CuLeafPlan,
        x0: u32,
        y0: u32,
        _log2_cb_size: u8,
        lambda: f64,
    ) -> f64 {
        // Grayscale has no chroma; 4:2:2 stays DM-only (see module docs).
        if state.cat == 0 || state.cat == 2 || leaf.nxn.is_some() {
            return 0.0;
        }
        let luma_mode = leaf.luma_mode;
        // HEVC Table 8-3 allowed list: planar/vertical/horizontal/DC, with a
        // slot equal to the luma (DM) mode replaced by angular-34; DM last.
        let mut list = [0u8, 26, 10, 1];
        for m in list.iter_mut() {
            if *m == luma_mode {
                *m = 34;
            }
        }

        let scale = CabacEstimator::SCALE as f64;
        let chroma_model = self.workspace.price_cur.models[ctx::INTRA_CHROMA_PRED_MODE];
        let dm_header = lambda * entropy_bits(&chroma_model, 0) as f64 / scale;
        let non_dm_header = lambda * (entropy_bits(&chroma_model, 1) as f64 + 2.0 * scale) / scale;

        let rough_enabled = super::env::chroma_rough_enabled();
        let rough_k = if rough_enabled {
            super::env::chroma_rough_k(state.effort_template.chroma.rough_k)
        } else {
            4
        };
        // Keep exhaustive modes byte-stable with the old path. Rebuild elision
        // is only allowed when rough pruning already makes this an optimized,
        // intentionally non-identical search path.
        let reuse_dm = super::env::chroma_reuse_dm_enabled() && rough_enabled && rough_k < 4;
        let dm_cost =
            self.tt_chroma_mode_cost(state, &leaf.tt, x0, y0, luma_mode, lambda) + dm_header;
        let mut best_idx = CHROMA_DM_IDX;
        let mut best_sig = luma_mode;
        let mut best_cost = dm_cost;

        if rough_enabled {
            let skip_satd =
                super::env::chroma_skip_satd(state.effort_template.chroma.skip_satd_per_sample);
            if skip_satd > 0.0 {
                let (dm_satd, samples) =
                    self.tt_chroma_rough_satd(state, &leaf.tt, x0, y0, luma_mode);
                if samples > 0 && dm_satd as f64 / samples as f64 <= skip_satd {
                    state.stats.chroma_mode_rough_skips += 1;
                    if reuse_dm {
                        self.tt_chroma_commit_ctx(state, &leaf.tt);
                    } else {
                        self.tt_chroma_mode_rebuild(state, &mut leaf.tt, x0, y0, luma_mode, lambda);
                    }
                    leaf.chroma_mode_idx = CHROMA_DM_IDX;
                    return 0.0;
                }
            }
        }

        let finalists: Vec<(usize, u8)> = if rough_enabled && rough_k < 4 {
            let mut rough: Vec<(f64, usize, u8)> = list
                .iter()
                .enumerate()
                .map(|(idx, &sig)| {
                    let (satd, _) = self.tt_chroma_rough_satd(state, &leaf.tt, x0, y0, sig);
                    (satd as f64 + non_dm_header, idx, sig)
                })
                .collect();
            rough.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            rough
                .into_iter()
                .take(rough_k)
                .map(|(_, idx, sig)| (idx, sig))
                .collect()
        } else {
            list.iter().copied().enumerate().collect()
        };

        for (idx, sig) in finalists {
            state.stats.chroma_mode_exact_candidates += 1;
            let cost =
                self.tt_chroma_mode_cost(state, &leaf.tt, x0, y0, sig, lambda) + non_dm_header;
            if cost < best_cost {
                best_cost = cost;
                best_idx = idx as u8;
                best_sig = sig;
            }
        }

        if best_idx == CHROMA_DM_IDX && reuse_dm {
            self.tt_chroma_commit_ctx(state, &leaf.tt);
        } else {
            // Rebuild the winner's chroma blocks in coding order (plan blocks,
            // coeff arena, overlay recon), committing chroma context updates when
            // evolving contexts are active.
            let prev_commit = self.workspace.commit_ctx;
            let prev_skip = self.workspace.commit_ctx_skip_chroma;
            self.workspace.commit_ctx = super::env::ctx_evolve_search();
            self.workspace.commit_ctx_skip_chroma = false;
            self.tt_chroma_mode_rebuild(state, &mut leaf.tt, x0, y0, best_sig, lambda);
            self.workspace.commit_ctx = prev_commit;
            self.workspace.commit_ctx_skip_chroma = prev_skip;
        }
        leaf.chroma_mode_idx = best_idx;

        best_cost - dm_cost
    }

    /// Prediction-only rough SATD over every chroma block in the TT for one
    /// signalled chroma mode. Returns (SATD sum, source-sample count).
    fn tt_chroma_rough_satd(
        &mut self,
        state: &mut Encoder<'_>,
        tt: &TtPlan,
        x0: u32,
        y0: u32,
        sig_mode: u8,
    ) -> (u64, u64) {
        let rough_timer = StillSearchLedger::start_timer();
        let out = self.tt_chroma_rough_satd_inner(state, tt, x0, y0, sig_mode);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::ChromaRough, rough_timer);
        out
    }

    fn tt_chroma_rough_satd_inner(
        &mut self,
        state: &mut Encoder<'_>,
        tt: &TtPlan,
        x0: u32,
        y0: u32,
        sig_mode: u8,
    ) -> (u64, u64) {
        match tt {
            TtPlan::Leaf(l) => {
                let mut satd = 0u64;
                let mut samples = 0u64;
                if has_chroma_tb(state.cat, l.log2_size) {
                    if let Some((cx, cy, clog2, count)) =
                        chroma_tb_geom(state.cat, x0, y0, l.log2_size)
                    {
                        let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                            .unwrap_or(IntraPredMode::Dc);
                        let step = 1u32 << clog2;
                        for c_idx in [1u8, 2u8] {
                            for i in 0..count as u32 {
                                satd += self.chroma_block_rough_satd(
                                    state,
                                    cx,
                                    cy + i * step,
                                    clog2,
                                    c_idx,
                                    pred,
                                ) as u64;
                                samples += 1u64 << (2 * clog2);
                            }
                        }
                    }
                }
                (satd, samples)
            }
            TtPlan::Split {
                log2_size,
                parent_chroma,
                kids,
                ..
            } => {
                let half = 1u32 << (log2_size - 1);
                let offs = [(0, 0), (half, 0), (0, half), (half, half)];
                let mut satd = 0u64;
                let mut samples = 0u64;
                for (kid, (dx, dy)) in kids.iter().zip(offs) {
                    let (s, n) =
                        self.tt_chroma_rough_satd_inner(state, kid, x0 + dx, y0 + dy, sig_mode);
                    satd += s;
                    samples += n;
                }
                if parent_chroma.is_some() {
                    let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                        .unwrap_or(IntraPredMode::Dc);
                    let (cx, cy) = parent_chroma_origin(state, x0, y0);
                    for c_idx in [1u8, 2u8] {
                        satd += self.chroma_block_rough_satd(state, cx, cy, 2, c_idx, pred) as u64;
                        samples += 16;
                    }
                }
                (satd, samples)
            }
        }
    }

    fn chroma_block_rough_satd(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        pred_mode: IntraPredMode,
    ) -> u32 {
        self.workspace.ledger.bump(WorkBucket::ChromaRough);
        state.stats.chroma_rough_predictions += 1;
        let size = 1usize << log2_size;
        let n = size * size;
        if state.bit_depth == 8 {
            let mut src = vec![0u8; n];
            self.source.sample_block_u8(c_idx, x0, y0, size, &mut src);
            let mut pred = vec![0u8; n];
            let mut tmp_u16 = Vec::with_capacity(n);
            self.predict_into_u8(
                state,
                x0,
                y0,
                log2_size,
                c_idx,
                pred_mode,
                &mut pred,
                &mut tmp_u16,
            );
            if log2_size >= 3 {
                sa8d_u8(&src, size, &pred, size, size)
            } else {
                satd_u8(&src, size, &pred, size, size)
            }
        } else {
            let mut src = vec![0u16; n];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self.source.sample(c_idx, x0 + x as u32, y0 + y as u32);
                }
            }
            let mut pred = vec![0u16; n];
            self.predict_into(state, x0, y0, log2_size, c_idx, pred_mode, &mut pred);
            if log2_size >= 3 {
                sa8d_u16(&src, size, &pred, size, size)
            } else {
                satd_u16(&src, size, &pred, size, size)
            }
        }
    }

    /// Commit deferred DM chroma contexts from already-materialized plan blocks
    /// without re-predicting, transforming, quantizing, or touching the overlay.
    fn tt_chroma_commit_ctx(&mut self, state: &Encoder<'_>, tt: &TtPlan) {
        if !super::env::ctx_evolve_search() {
            return;
        }
        self.tt_chroma_commit_ctx_inner(state, tt);
    }

    fn tt_chroma_commit_ctx_inner(&mut self, state: &Encoder<'_>, tt: &TtPlan) {
        match tt {
            TtPlan::Leaf(l) => {
                self.commit_plan_chroma_block(
                    state,
                    &l.cb,
                    l.chroma_log2,
                    1,
                    l.chroma_mode,
                    l.trafo_depth,
                );
                self.commit_plan_chroma_block(
                    state,
                    &l.cb1,
                    l.chroma_log2,
                    1,
                    l.chroma_mode,
                    l.trafo_depth,
                );
                self.commit_plan_chroma_block(
                    state,
                    &l.cr,
                    l.chroma_log2,
                    2,
                    l.chroma_mode,
                    l.trafo_depth,
                );
                self.commit_plan_chroma_block(
                    state,
                    &l.cr1,
                    l.chroma_log2,
                    2,
                    l.chroma_mode,
                    l.trafo_depth,
                );
            }
            TtPlan::Split {
                trafo_depth,
                parent_chroma,
                kids,
                ..
            } => {
                for kid in kids {
                    self.tt_chroma_commit_ctx_inner(state, kid);
                }
                if let Some(pc) = parent_chroma {
                    self.commit_plan_chroma_block(
                        state,
                        &pc.cb,
                        pc.log2_size,
                        1,
                        pc.chroma_mode,
                        *trafo_depth,
                    );
                    self.commit_plan_chroma_block(
                        state,
                        &pc.cb1,
                        pc.log2_size,
                        1,
                        pc.chroma_mode,
                        *trafo_depth,
                    );
                    self.commit_plan_chroma_block(
                        state,
                        &pc.cr,
                        pc.log2_size,
                        2,
                        pc.chroma_mode,
                        *trafo_depth,
                    );
                    self.commit_plan_chroma_block(
                        state,
                        &pc.cr1,
                        pc.log2_size,
                        2,
                        pc.chroma_mode,
                        *trafo_depth,
                    );
                }
            }
        }
    }

    fn commit_plan_chroma_block(
        &mut self,
        state: &Encoder<'_>,
        block: &PlanBlock,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        trafo_depth: u8,
    ) {
        if block.rd_frac_bits == 0 && !block.cbf {
            return;
        }
        let levels: Vec<i16> = block
            .coeff
            .map(|id| self.workspace.coeffs.get(id))
            .unwrap_or(&[])
            .to_vec();
        let scan = get_scan_order(log2_size, mode, c_idx, state.cat);
        self.commit_component_ctx(
            block.cbf,
            &levels,
            log2_size,
            c_idx,
            trafo_depth,
            scan,
            state.sign_data_hiding,
        );
    }

    /// Sum the RD cost of every chroma block in this transform tree when
    /// predicted with `sig_mode` (a signalled, pre-4:2:2-mapping mode).
    /// Pure pricing: no overlay pushes, no plan mutation, no context commits.
    fn tt_chroma_mode_cost(
        &mut self,
        state: &Encoder<'_>,
        tt: &TtPlan,
        x0: u32,
        y0: u32,
        sig_mode: u8,
        lambda: f64,
    ) -> f64 {
        match tt {
            TtPlan::Leaf(l) => {
                let mut cost = 0.0;
                if has_chroma_tb(state.cat, l.log2_size) {
                    if let Some((cx, cy, clog2, count)) =
                        chroma_tb_geom(state.cat, x0, y0, l.log2_size)
                    {
                        let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                            .unwrap_or(IntraPredMode::Dc);
                        let step = 1u32 << clog2;
                        for c_idx in [1u8, 2u8] {
                            for i in 0..count as u32 {
                                let t = self.eval_component_no_overlay(
                                    state,
                                    cx,
                                    cy + i * step,
                                    clog2,
                                    c_idx,
                                    pred,
                                    state.cur_qp_c,
                                    l.trafo_depth,
                                    lambda,
                                    super::eval::search_trial_quant(
                                        state.effort_template.chroma.quant,
                                    ),
                                    ResidualPricingMode::Exact,
                                    false,
                                );
                                cost += t.cost;
                            }
                        }
                    }
                }
                cost
            }
            TtPlan::Split {
                log2_size,
                trafo_depth,
                parent_chroma,
                kids,
                ..
            } => {
                let half = 1u32 << (log2_size - 1);
                let offs = [(0, 0), (half, 0), (0, half), (half, half)];
                let mut cost = 0.0;
                for (kid, (dx, dy)) in kids.iter().zip(offs) {
                    cost +=
                        self.tt_chroma_mode_cost(state, kid, x0 + dx, y0 + dy, sig_mode, lambda);
                }
                if parent_chroma.is_some() {
                    let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                        .unwrap_or(IntraPredMode::Dc);
                    let (cx, cy) = parent_chroma_origin(state, x0, y0);
                    for c_idx in [1u8, 2u8] {
                        let t = self.eval_component_no_overlay(
                            state,
                            cx,
                            cy,
                            2,
                            c_idx,
                            pred,
                            state.cur_qp_c,
                            *trafo_depth,
                            lambda,
                            super::eval::search_trial_quant(state.effort_template.chroma.quant),
                            ResidualPricingMode::Exact,
                            false,
                        );
                        cost += t.cost;
                    }
                }
                cost
            }
        }
    }

    /// Recode every chroma block in this transform tree with `sig_mode`,
    /// replacing plan blocks (coeffs retained in the arena), pushing recon to
    /// the overlay in coding order, updating chroma mode fields and split-node
    /// chroma CBF flags. Context commits follow `workspace.commit_ctx`.
    fn tt_chroma_mode_rebuild(
        &mut self,
        state: &Encoder<'_>,
        tt: &mut TtPlan,
        x0: u32,
        y0: u32,
        sig_mode: u8,
        lambda: f64,
    ) {
        match tt {
            TtPlan::Leaf(l) => {
                if has_chroma_tb(state.cat, l.log2_size) {
                    if let Some((cx, cy, clog2, count)) =
                        chroma_tb_geom(state.cat, x0, y0, l.log2_size)
                    {
                        let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                            .unwrap_or(IntraPredMode::Dc);
                        let step = 1u32 << clog2;
                        // Writer residual order at a leaf: cb, cb1, cr, cr1.
                        l.cb = self
                            .eval_component(
                                state,
                                cx,
                                cy,
                                clog2,
                                1,
                                pred,
                                state.cur_qp_c,
                                l.trafo_depth,
                                lambda,
                                super::eval::search_trial_quant(state.effort_template.chroma.quant),
                                ResidualPricingMode::Exact,
                                true,
                            )
                            .into_plan_block();
                        if count > 1 {
                            l.cb1 = self
                                .eval_component(
                                    state,
                                    cx,
                                    cy + step,
                                    clog2,
                                    1,
                                    pred,
                                    state.cur_qp_c,
                                    l.trafo_depth,
                                    lambda,
                                    super::eval::search_trial_quant(
                                        state.effort_template.chroma.quant,
                                    ),
                                    ResidualPricingMode::Exact,
                                    true,
                                )
                                .into_plan_block();
                        }
                        l.cr = self
                            .eval_component(
                                state,
                                cx,
                                cy,
                                clog2,
                                2,
                                pred,
                                state.cur_qp_c,
                                l.trafo_depth,
                                lambda,
                                super::eval::search_trial_quant(state.effort_template.chroma.quant),
                                ResidualPricingMode::Exact,
                                true,
                            )
                            .into_plan_block();
                        if count > 1 {
                            l.cr1 = self
                                .eval_component(
                                    state,
                                    cx,
                                    cy + step,
                                    clog2,
                                    2,
                                    pred,
                                    state.cur_qp_c,
                                    l.trafo_depth,
                                    lambda,
                                    super::eval::search_trial_quant(
                                        state.effort_template.chroma.quant,
                                    ),
                                    ResidualPricingMode::Exact,
                                    true,
                                )
                                .into_plan_block();
                        }
                        l.chroma_mode = sig_mode;
                    }
                }
            }
            TtPlan::Split {
                log2_size,
                trafo_depth,
                cbf_cb,
                cbf_cr,
                cbf_cb1,
                cbf_cr1,
                parent_chroma,
                kids,
            } => {
                let half = 1u32 << (*log2_size - 1);
                let offs = [(0, 0), (half, 0), (0, half), (half, half)];
                for (kid, (dx, dy)) in kids.iter_mut().zip(offs) {
                    self.tt_chroma_mode_rebuild(state, kid, x0 + dx, y0 + dy, sig_mode, lambda);
                }
                if let Some(pc) = parent_chroma {
                    let pred = IntraPredMode::from_u8(chroma_pred_mode(state.cat, sig_mode))
                        .unwrap_or(IntraPredMode::Dc);
                    let (cx, cy) = parent_chroma_origin(state, x0, y0);
                    pc.cb = self
                        .eval_component(
                            state,
                            cx,
                            cy,
                            2,
                            1,
                            pred,
                            state.cur_qp_c,
                            *trafo_depth,
                            lambda,
                            super::eval::search_trial_quant(state.effort_template.chroma.quant),
                            ResidualPricingMode::Exact,
                            true,
                        )
                        .into_plan_block();
                    pc.cr = self
                        .eval_component(
                            state,
                            cx,
                            cy,
                            2,
                            2,
                            pred,
                            state.cur_qp_c,
                            *trafo_depth,
                            lambda,
                            super::eval::search_trial_quant(state.effort_template.chroma.quant),
                            ResidualPricingMode::Exact,
                            true,
                        )
                        .into_plan_block();
                    // Store the mapped prediction mode, matching
                    // `eval_tt_split_parent_chroma`'s convention (the writer
                    // derives the parent scan from this field directly).
                    pc.chroma_mode = chroma_pred_mode(state.cat, sig_mode);
                }
                let parent = parent_chroma.as_ref();
                *cbf_cb = kids.iter().any(TtPlan::cbf_cb) || parent.is_some_and(|p| p.cb.cbf);
                *cbf_cr = kids.iter().any(TtPlan::cbf_cr) || parent.is_some_and(|p| p.cr.cbf);
                *cbf_cb1 = kids.iter().any(TtPlan::cbf_cb1) || parent.is_some_and(|p| p.cb1.cbf);
                *cbf_cr1 = kids.iter().any(TtPlan::cbf_cr1) || parent.is_some_and(|p| p.cr1.cbf);
            }
        }
    }
}

/// Chroma origin of the subsampled parent-chroma group at an 8x8 split node.
fn parent_chroma_origin(state: &Encoder<'_>, x0: u32, y0: u32) -> (u32, u32) {
    let cx = x0 / 2;
    let cy = if state.cat == 1 { y0 / 2 } else { y0 };
    (cx, cy)
}
