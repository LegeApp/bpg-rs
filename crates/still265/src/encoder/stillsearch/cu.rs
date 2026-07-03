//! Coding-unit decisions.

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::effort::{CuEarlyTerminateRule, SplitSearch};
use crate::encoder::Encoder;
use crate::encoder::types::QG_LOG2;
use crate::plan::DecisionConfidence;

use super::depth::StillSearchDepth;
use super::emit;
use super::overlay::OverlayCache;
use super::plan::{CuPlan, PlanBlock, TtPlan};
use super::price::{chroma_dm_bits, entropy_bits, part_mode_bits, rd_lambda};
use super::source::CtuSourceCache;

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    #[inline]
    fn cu_split_cost_bias(&self, log2_cb_size: u8) -> f64 {
        let pixels = 1u64 << (2 * log2_cb_size);
        super::env::cu_split_bias_per_px() * pixels as f64
    }

    #[inline]
    fn force_cu_split(&self, log2_cb_size: u8) -> bool {
        super::env::cu_force_split_log2().is_some_and(|v| v == log2_cb_size)
    }

    /// Update the `split_cu_flag` context in the evolving trial context for
    /// the branch being explored (the writer codes it before the CU content).
    #[inline]
    fn commit_cu_split_flag_ctx(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        ct_depth: u8,
        is_split: bool,
    ) {
        let ci = ctx::SPLIT_CU_FLAG + state.split_ctx_inc(x0, y0, ct_depth);
        self.workspace.price_cur.models[ci].update(is_split as u8);
    }

    /// Commit a 2Nx2N CU leaf's header syntax (`part_mode` at min CU size,
    /// `prev_intra_luma_pred_flag`, `intra_chroma_pred_mode`) into the
    /// evolving trial context. Content (CBF/residual) contexts were already
    /// committed by the winner's materialization; header models are disjoint
    /// from content models, so applying them here yields the same final
    /// per-model state as the writer's coding order.
    fn commit_cu_leaf_header_ctx(
        &mut self,
        state: &Encoder<'_>,
        mpm: [bpg_hevc_decode::hevc::slice::IntraPredMode; 3],
        mode: u8,
        log2_cb_size: u8,
        chroma_mode_idx: u8,
    ) {
        let ctxs = &mut self.workspace.price_cur;
        if log2_cb_size == 3 {
            ctxs.models[ctx::PART_MODE].update(1);
        }
        let in_mpm = mpm.iter().any(|m| m.as_u8() == mode);
        ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG].update(in_mpm as u8);
        if state.cat != 0 {
            ctxs.models[ctx::INTRA_CHROMA_PRED_MODE]
                .update((chroma_mode_idx != crate::encoder::types::CHROMA_DM_IDX) as u8);
        }
    }

    /// Decide one CU: code it as a single `Leaf2Nx2N`, legal `PartNxN`, or
    /// split into four sub-CUs. On return the overlay holds exactly the winner's
    /// recon for this region and `mode_map`/`ct_depth_map` reflect it.
    pub(super) fn decide_cu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> (CuPlan, f64) {
        let cb_size = 1u32 << log2_cb_size;
        let fully_inside =
            x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
        let can_split = log2_cb_size > 3;

        if !fully_inside && can_split {
            return self.decide_cu_split(state, x0, y0, log2_cb_size, ct_depth);
        }
        // AQ targets are defined per 32x32 quantization group. Keeping larger
        // leaf CUs would quantize the whole leaf with its top-left QG target,
        // while the bitstream/decoder QP prediction can change at QG
        // boundaries. For AQ-active experiments, force the CU tree down to QG
        // size so analysis, writer, and decoder agree.
        if state.aq.active && can_split && log2_cb_size > QG_LOG2 {
            if super::env::ctx_evolve_search() {
                self.commit_cu_split_flag_ctx(state, x0, y0, ct_depth, true);
            }
            return self.decide_cu_split(state, x0, y0, log2_cb_size, ct_depth);
        }
        if !can_split {
            if fully_inside && self.part_nxn_legal(state, log2_cb_size) {
                return self.decide_cu_min_leaf_or_nxn(state, x0, y0, ct_depth);
            }
            return self.decide_cu_leaf(state, x0, y0, log2_cb_size, ct_depth);
        }

        // Snapshot the split-flag pricing model and (with evolving contexts)
        // the CU-entry context before the alternatives touch either. Neighbor
        // ct-depths feeding `split_ctx_inc` come from CUs decided before this
        // one, so computing the index up front matches the writer.
        let evolve = super::env::ctx_evolve_search();
        let ci = ctx::SPLIT_CU_FLAG + state.split_ctx_inc(x0, y0, ct_depth);
        let flag_model = self.workspace.price_cur.models[ci];
        let ctx_entry = evolve.then(|| self.workspace.price_cur.clone());
        if evolve {
            self.commit_cu_split_flag_ctx(state, x0, y0, ct_depth, false);
        }

        let mark0 = self.overlay.mark();
        let (leaf_plan, leaf_cost) = self.decide_cu_leaf(state, x0, y0, log2_cb_size, ct_depth);
        let leaf_saved = self.overlay.detach_from(mark0);

        // CU early termination: skip split evaluation when the leaf is
        // good enough.  Evidence includes luma residual presence, source
        // activity, and the effort template's aggressiveness.
        let cu_policy = &state.effort_template.cu;
        let leaf_has_residual = match &leaf_plan {
            CuPlan::Leaf(l) => l.tt.cbf_luma(),
            _ => true,
        };

        let should_skip_split = match cu_policy.early_terminate {
            CuEarlyTerminateRule::Disabled => false,
            CuEarlyTerminateRule::Fastest => true,
            CuEarlyTerminateRule::Fast => {
                // Zero-residual leaf → always accept.
                if !leaf_has_residual {
                    true
                } else if cu_policy.split_search == SplitSearch::PreferLeaf {
                    // PreferLeaf + some residual → accept if source is smooth.
                    let activity = self.source_activity(state, x0, y0, log2_cb_size);
                    activity < 8.0
                } else {
                    false
                }
            }
            CuEarlyTerminateRule::Balanced => {
                if !leaf_has_residual {
                    true
                } else if cu_policy.split_search == SplitSearch::PreferLeaf {
                    let activity = self.source_activity(state, x0, y0, log2_cb_size);
                    activity < 4.0
                } else {
                    false
                }
            }
        };

        let should_skip_split = should_skip_split && !self.force_cu_split(log2_cb_size);

        if should_skip_split {
            state.stats.cu_early_term_by_log2[log2_cb_size as usize] += 1;
            self.overlay.reattach(leaf_saved);
            if let CuPlan::Leaf(ref leaf) = leaf_plan {
                state.store_mode(x0, y0, log2_cb_size, leaf.luma_mode);
                state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
            }
            return (leaf_plan, leaf_cost);
        }

        let ctx_leaf = ctx_entry.as_ref().map(|entry| {
            let leaf_state = self.workspace.price_cur.clone();
            self.workspace.price_cur = entry.clone();
            self.commit_cu_split_flag_ctx(state, x0, y0, ct_depth, true);
            leaf_state
        });

        let (split_plan, split_cost) = self.decide_cu_split(state, x0, y0, log2_cb_size, ct_depth);

        let lambda = rd_lambda(state.cur_qp_y);
        let scale = CabacEstimator::SCALE as f64;
        let leaf_total = leaf_cost + lambda * entropy_bits(&flag_model, 0) as f64 / scale;
        let split_total = split_cost + lambda * entropy_bits(&flag_model, 1) as f64 / scale;

        if let Some((tx, ty)) = super::env::dump_ctu_target() {
            if x0 & !63 == tx && y0 & !63 == ty {
                let mode = match &leaf_plan {
                    CuPlan::Leaf(l) => l.luma_mode as i32,
                    _ => -1,
                };
                let leaf_dist = cu_plan_dist(&leaf_plan);
                let split_dist = cu_plan_dist(&split_plan);
                let leaf_frac_bits = cu_plan_rd_frac_bits(&leaf_plan);
                let split_frac_bits = cu_plan_rd_frac_bits(&split_plan);
                let leaf_bits = rd_real_bits(leaf_total, leaf_dist, lambda);
                let split_bits = rd_real_bits(split_total, split_dist, lambda);
                let n = 1u32 << log2_cb_size;
                eprintln!(
                    "STILLCAND cu={x0},{y0} {n}x{n} leaf_mode={mode} leaf_dist={leaf_dist} leaf_frac_bits={leaf_frac_bits} leaf_bits={leaf_bits:.1} leaf_total={leaf_total:.0} split_dist={split_dist} split_frac_bits={split_frac_bits} split_bits={split_bits:.1} split_total={split_total:.0} -> {}",
                    if split_total < leaf_total {
                        "SPLIT"
                    } else {
                        "LEAF"
                    }
                );
            }
        }

        let split_cmp_total = split_total - self.cu_split_cost_bias(log2_cb_size);
        if self.force_cu_split(log2_cb_size) || split_cmp_total < leaf_total {
            state.stats.cu_splits_taken_by_log2[log2_cb_size as usize] += 1;
            #[cfg(test)]
            super::api::CU_SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_plan, split_total)
        } else {
            if let Some(leaf_state) = ctx_leaf {
                self.workspace.price_cur = leaf_state;
            }
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            if let CuPlan::Leaf(ref leaf) = leaf_plan {
                state.store_mode(x0, y0, log2_cb_size, leaf.luma_mode);
                state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
            }
            (leaf_plan, leaf_total)
        }
    }

    /// Rough source activity: sum of absolute differences from the mean
    /// for the luma block, normalized per pixel.  Low values indicate
    /// smooth regions where leaf-only search is likely sufficient.
    fn source_activity(&self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) -> f64 {
        let size = 1usize << log2_cb_size;
        let n = size * size;
        if state.bit_depth == 8 {
            let mut buf = vec![0u8; n];
            self.source.sample_block_u8(0, x0, y0, size, &mut buf);
            let mean = buf.iter().map(|&s| s as u64).sum::<u64>() / n as u64;
            let sad = buf
                .iter()
                .map(|&s| (s as i64 - mean as i64).unsigned_abs())
                .sum::<u64>();
            sad as f64 / n as f64
        } else {
            let mut sad: u64 = 0;
            let mut sum: u64 = 0;
            for y in 0..size {
                for x in 0..size {
                    let v = self.source.sample(0, x0 + x as u32, y0 + y as u32) as u64;
                    sum += v;
                }
            }
            let mean = sum / n as u64;
            for y in 0..size {
                for x in 0..size {
                    let v = self.source.sample(0, x0 + x as u32, y0 + y as u32) as u64;
                    sad += (v as i64 - mean as i64).unsigned_abs();
                }
            }
            sad as f64 / n as f64
        }
    }

    fn part_nxn_legal(&self, state: &Encoder<'_>, log2_cb_size: u8) -> bool {
        let nxn_policy = &state.effort_template.nxn;
        if !nxn_policy.enabled {
            return false;
        }
        // 4:2:2 PartNxN parent chroma has a known second-stacked-CBF/replay
        // edge at very low QP. Keep PartNxN enabled for 4:2:0 and 4:4:4, but
        // gate 4:2:2 until that syntax/recon path is fixed.
        if !state.part_nxn_enabled || log2_cb_size != 3 || !matches!(state.cat, 0 | 1 | 3) {
            return false;
        }
        true
    }

    fn save_8x8_modes(&self, state: &Encoder<'_>, x0: u32, y0: u32) -> [u8; 4] {
        [
            state.mode_at(x0, y0),
            state.mode_at(x0 + 4, y0),
            state.mode_at(x0, y0 + 4),
            state.mode_at(x0 + 4, y0 + 4),
        ]
    }

    fn restore_8x8_modes(&self, state: &mut Encoder<'_>, x0: u32, y0: u32, modes: [u8; 4]) {
        state.store_mode(x0, y0, 2, modes[0]);
        state.store_mode(x0 + 4, y0, 2, modes[1]);
        state.store_mode(x0, y0 + 4, 2, modes[2]);
        state.store_mode(x0 + 4, y0 + 4, 2, modes[3]);
    }

    /// Decide an 8x8 CU between regular `2Nx2N` and `PartNxN`. Both candidates
    /// are evaluated against the same incoming overlay/mode-map state, and only
    /// the winner's recon patches/modes remain on return.
    fn decide_cu_min_leaf_or_nxn(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        ct_depth: u8,
    ) -> (CuPlan, f64) {
        state.stats.partnxn_attempts += 1;
        let lambda = rd_lambda(state.cur_qp_y);
        let scale = CabacEstimator::SCALE as f64;
        let incoming_modes = self.save_8x8_modes(state, x0, y0);
        let mark0 = self.overlay.mark();

        // Header-bit models snapshotted before either alternative can evolve
        // them. With `cu_header_bits` on, `decide_cu_leaf` prices the leaf's
        // own part_mode/chroma header, so only the NxN side adds them here.
        let header_bits_in_leaf = super::env::cu_header_bits_enabled();
        let part_leaf = part_mode_bits(&self.workspace.price_cur, false);
        let part_nxn = part_mode_bits(&self.workspace.price_cur, true);
        let chroma_bits = chroma_dm_bits(&self.workspace.price_cur, state.cat);
        let evolve = super::env::ctx_evolve_search();
        let ctx_entry = evolve.then(|| self.workspace.price_cur.clone());

        let (leaf_plan, leaf_cost) = self.decide_cu_leaf(state, x0, y0, 3, ct_depth);
        let leaf_saved = self.overlay.detach_from(mark0);
        let leaf_header_extra = if header_bits_in_leaf {
            0.0
        } else {
            lambda * (part_leaf + chroma_bits) as f64 / scale
        };

        // Rough SATD below the env threshold means the block is too smooth for
        // PartNxN to improve over the 2Nx2N leaf — skip it.
        let nxn_policy = &state.effort_template.nxn;
        let nxn_threshold = super::env::nxn_skip_satd_threshold(nxn_policy.rough_satd_threshold);
        if nxn_policy.rough_gate_enabled
            && nxn_threshold > 0.0
            && self.workspace.last_8x8_rough_satd < nxn_threshold
        {
            state.stats.partnxn_skips += 1;
            state.stats.partnxn_losses += 1;
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            if let CuPlan::Leaf(ref leaf) = leaf_plan {
                state.store_mode(x0, y0, 3, leaf.luma_mode);
                state.set_ct_depth(x0, y0, 3, ct_depth);
            }
            return (leaf_plan, leaf_cost + leaf_header_extra);
        }

        let ctx_leaf = ctx_entry
            .as_ref()
            .map(|entry| std::mem::replace(&mut self.workspace.price_cur, entry.clone()));

        self.restore_8x8_modes(state, x0, y0, incoming_modes);
        let (nxn_plan, nxn_cost) = self.decide_cu_part_nxn(state, x0, y0, ct_depth, lambda);

        let nxn_chroma_bits = if state.cat == 3 {
            4 * chroma_bits
        } else {
            chroma_bits
        };
        let leaf_total = leaf_cost + leaf_header_extra;
        let nxn_total = nxn_cost + lambda * (part_nxn + nxn_chroma_bits) as f64 / scale;
        let nxn_cmp_total = nxn_total - super::env::nxn_bias_per_px() * 64.0;

        if let Some((tx, ty)) = super::env::dump_ctu_target() {
            if x0 & !63 == tx && y0 & !63 == ty {
                let mode = match &leaf_plan {
                    CuPlan::Leaf(l) => l.luma_mode as i32,
                    _ => -1,
                };
                let leaf_dist = cu_plan_dist(&leaf_plan);
                let nxn_dist = cu_plan_dist(&nxn_plan);
                let leaf_frac_bits = cu_plan_rd_frac_bits(&leaf_plan);
                let nxn_frac_bits = cu_plan_rd_frac_bits(&nxn_plan);
                let leaf_bits = rd_real_bits(leaf_total, leaf_dist, lambda);
                let nxn_bits = rd_real_bits(nxn_total, nxn_dist, lambda);
                eprintln!(
                    "STILLNXN cu={x0},{y0} 8x8 leaf_mode={mode} leaf_dist={leaf_dist} leaf_frac_bits={leaf_frac_bits} leaf_bits={leaf_bits:.1} leaf_total={leaf_total:.0} nxn_dist={nxn_dist} nxn_frac_bits={nxn_frac_bits} nxn_bits={nxn_bits:.1} nxn_total={nxn_total:.0} -> {}",
                    if nxn_cmp_total < leaf_total {
                        "NXN"
                    } else {
                        "LEAF"
                    }
                );
            }
        }

        if super::env::nxn_force_enabled() || nxn_cmp_total < leaf_total {
            state.stats.partnxn_wins += 1;
            if state.cat == 2 && nxn_plan.has_parent_second_chroma_cbf() {
                state.stats.partnxn_422_parent_chroma_second_cbf += 1;
            }
            (nxn_plan, nxn_total)
        } else {
            state.stats.partnxn_losses += 1;
            if let Some(leaf_state) = ctx_leaf {
                self.workspace.price_cur = leaf_state;
            }
            self.overlay.truncate(mark0);
            self.overlay.reattach(leaf_saved);
            if let CuPlan::Leaf(ref leaf) = leaf_plan {
                state.store_mode(x0, y0, 3, leaf.luma_mode);
                state.set_ct_depth(x0, y0, 3, ct_depth);
            }
            (leaf_plan, leaf_total)
        }
    }

    fn decide_cu_split(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> (CuPlan, f64) {
        let half = (1u32 << log2_cb_size) / 2;
        let x1 = x0 + half;
        let y1 = y0 + half;
        let kid_log2 = log2_cb_size - 1;
        let kd = ct_depth + 1;
        let mut kids = Vec::with_capacity(4);
        let mut cost = 0.0;

        let (k, c) = self.decide_cu(state, x0, y0, kid_log2, kd);
        kids.push(k);
        cost += c;
        if x1 < state.display_width {
            let (k, c) = self.decide_cu(state, x1, y0, kid_log2, kd);
            kids.push(k);
            cost += c;
        }
        if y1 < state.display_height {
            let (k, c) = self.decide_cu(state, x0, y1, kid_log2, kd);
            kids.push(k);
            cost += c;
        }
        if x1 < state.display_width && y1 < state.display_height {
            let (k, c) = self.decide_cu(state, x1, y1, kid_log2, kd);
            kids.push(k);
            cost += c;
        }
        (CuPlan::Split { kids }, cost)
    }

    fn decide_cu_leaf(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> (CuPlan, f64) {
        state.stats.cu_trials += 1;
        state.stats.cu_trials_by_log2[log2_cb_size as usize] += 1;
        state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let mpm = super::intra_mpm(state, x0, y0);

        if state.aq.active {
            state.aq_set_cu_qp(x0, y0);
        }
        let lambda = rd_lambda(state.cur_qp_y);

        let winner = self.decide_cu_luma_mode(state, x0, y0, log2_cb_size, mpm, lambda);
        state.store_mode(x0, y0, log2_cb_size, winner.mode);
        self.overlay.reattach(winner.recon);

        state.stats.final_coded_blocks += 1;
        let mut leaf = emit::cu_leaf(mpm, winner.mode, winner.tt);
        leaf.confidence = DecisionConfidence::Clear;

        let mut cost = winner.cost;
        // x265 `checkIntra` prices the complete CU syntax (part_mode, pred
        // info incl. the chroma mode) into every mode's rdcost, so CU splits
        // pay for their extra per-CU headers. Price them here for parity;
        // the 8x8 2Nx2N-vs-NxN comparison skips its own copy when this is on.
        if super::env::cu_header_bits_enabled() {
            let scale = CabacEstimator::SCALE as f64;
            let mut hbits = chroma_dm_bits(&self.workspace.price_cur, state.cat);
            if log2_cb_size == 3 {
                hbits += part_mode_bits(&self.workspace.price_cur, false);
            }
            cost += lambda * hbits as f64 / scale;
        }

        // CU-level chroma mode search (x265 estIntraPredChromaQT parity):
        // re-decide the winner's chroma blocks across the five allowed chroma
        // modes, updating plan blocks, overlay recon, and the leaf cost.
        if super::env::chroma_mode_search_enabled() {
            cost += self.select_cu_chroma_mode(state, &mut leaf, x0, y0, log2_cb_size, lambda);
        }

        if super::env::ctx_evolve_search() {
            let chroma_idx = leaf.chroma_mode_idx;
            self.commit_cu_leaf_header_ctx(state, mpm, winner.mode, log2_cb_size, chroma_idx);
        }

        (CuPlan::Leaf(leaf), cost)
    }
}

fn rd_real_bits(cost: f64, dist: u64, lambda: f64) -> f64 {
    if lambda <= 0.0 {
        0.0
    } else {
        ((cost - dist as f64) / lambda).max(0.0)
    }
}

fn plan_block_dist(block: &PlanBlock) -> u64 {
    block.dist
}

fn plan_block_rd_frac_bits(block: &PlanBlock) -> u64 {
    block.rd_frac_bits
}

fn tt_plan_dist(tt: &TtPlan) -> u64 {
    match tt {
        TtPlan::Leaf(l) => {
            plan_block_dist(&l.luma)
                + plan_block_dist(&l.cb)
                + plan_block_dist(&l.cr)
                + plan_block_dist(&l.cb1)
                + plan_block_dist(&l.cr1)
        }
        TtPlan::Split {
            parent_chroma,
            kids,
            ..
        } => {
            let parent = parent_chroma.as_ref().map_or(0, |p| {
                plan_block_dist(&p.cb)
                    + plan_block_dist(&p.cb1)
                    + plan_block_dist(&p.cr)
                    + plan_block_dist(&p.cr1)
            });
            parent + kids.iter().map(tt_plan_dist).sum::<u64>()
        }
    }
}

fn tt_plan_rd_frac_bits(tt: &TtPlan) -> u64 {
    match tt {
        TtPlan::Leaf(l) => {
            plan_block_rd_frac_bits(&l.luma)
                + plan_block_rd_frac_bits(&l.cb)
                + plan_block_rd_frac_bits(&l.cr)
                + plan_block_rd_frac_bits(&l.cb1)
                + plan_block_rd_frac_bits(&l.cr1)
        }
        TtPlan::Split {
            parent_chroma,
            kids,
            ..
        } => {
            let parent = parent_chroma.as_ref().map_or(0, |p| {
                plan_block_rd_frac_bits(&p.cb)
                    + plan_block_rd_frac_bits(&p.cb1)
                    + plan_block_rd_frac_bits(&p.cr)
                    + plan_block_rd_frac_bits(&p.cr1)
            });
            parent + kids.iter().map(tt_plan_rd_frac_bits).sum::<u64>()
        }
    }
}

fn cu_plan_dist(plan: &CuPlan) -> u64 {
    match plan {
        CuPlan::Leaf(leaf) => tt_plan_dist(&leaf.tt),
        CuPlan::Split { kids } => kids.iter().map(cu_plan_dist).sum(),
    }
}

fn cu_plan_rd_frac_bits(plan: &CuPlan) -> u64 {
    match plan {
        CuPlan::Leaf(leaf) => tt_plan_rd_frac_bits(&leaf.tt),
        CuPlan::Split { kids } => kids.iter().map(cu_plan_rd_frac_bits).sum(),
    }
}
