//! Coding-unit decisions.

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::effort::{CuEarlyTerminateRule, SplitSearch};
use crate::encoder::Encoder;
use crate::plan::DecisionConfidence;

use super::depth::StillSearchDepth;
use super::emit;
use super::overlay::OverlayCache;
use super::plan::CuPlan;
use super::price::{chroma_dm_bits, entropy_bits, part_mode_bits, rd_lambda};
use super::source::CtuSourceCache;

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
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
        if !can_split {
            if fully_inside && self.part_nxn_legal(state, log2_cb_size) {
                return self.decide_cu_min_leaf_or_nxn(state, x0, y0, ct_depth);
            }
            return self.decide_cu_leaf(state, x0, y0, log2_cb_size, ct_depth);
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

        if should_skip_split {
            self.overlay.reattach(leaf_saved);
            if let CuPlan::Leaf(ref leaf) = leaf_plan {
                state.store_mode(x0, y0, log2_cb_size, leaf.luma_mode);
                state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
            }
            return (leaf_plan, leaf_cost);
        }

        let (split_plan, split_cost) = self.decide_cu_split(state, x0, y0, log2_cb_size, ct_depth);

        let lambda = rd_lambda(state.cur_qp_y);
        let scale = CabacEstimator::SCALE as f64;
        let ci = ctx::SPLIT_CU_FLAG + state.split_ctx_inc(x0, y0, ct_depth);
        let model = &self.workspace.price_base.models[ci];
        let leaf_total = leaf_cost + lambda * entropy_bits(model, 0) as f64 / scale;
        let split_total = split_cost + lambda * entropy_bits(model, 1) as f64 / scale;

        if split_total < leaf_total {
            #[cfg(test)]
            super::api::CU_SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (split_plan, split_total)
        } else {
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
        if !state.part_nxn_enabled || log2_cb_size != 3 || !matches!(state.cat, 0 | 1 | 2 | 3) {
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

        let (leaf_plan, leaf_cost) = self.decide_cu_leaf(state, x0, y0, 3, ct_depth);
        let leaf_saved = self.overlay.detach_from(mark0);

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
            let part_leaf = part_mode_bits(&self.workspace.price_base, false);
            let chroma_bits = chroma_dm_bits(&self.workspace.price_base, state.cat);
            return (
                leaf_plan,
                leaf_cost + lambda * (part_leaf + chroma_bits) as f64 / scale,
            );
        }

        self.restore_8x8_modes(state, x0, y0, incoming_modes);
        let (nxn_plan, nxn_cost) = self.decide_cu_part_nxn(state, x0, y0, ct_depth, lambda);

        let part_leaf = part_mode_bits(&self.workspace.price_base, false);
        let part_nxn = part_mode_bits(&self.workspace.price_base, true);
        let chroma_bits = chroma_dm_bits(&self.workspace.price_base, state.cat);
        let nxn_chroma_bits = if state.cat == 3 {
            4 * chroma_bits
        } else {
            chroma_bits
        };
        let leaf_total = leaf_cost + lambda * (part_leaf + chroma_bits) as f64 / scale;
        let nxn_total = nxn_cost + lambda * (part_nxn + nxn_chroma_bits) as f64 / scale;

        if nxn_total < leaf_total {
            state.stats.partnxn_wins += 1;
            if state.cat == 2 && nxn_plan.has_parent_second_chroma_cbf() {
                state.stats.partnxn_422_parent_chroma_second_cbf += 1;
            }
            (nxn_plan, nxn_total)
        } else {
            state.stats.partnxn_losses += 1;
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
        (CuPlan::Leaf(leaf), winner.cost)
    }
}
