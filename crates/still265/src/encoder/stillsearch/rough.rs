//! Rough and exact luma mode search.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::encoder::Encoder;
use crate::primitives;
use crate::primitives::intra_angs;
use crate::primitives::{satd_u8, satd_u16};

use super::depth::StillSearchDepth;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::overlay::OverlayCache;
use super::plan::TtPlan;
use super::price::{ROUGH_RD_CANDS, luma_mode_bits};
use super::source::CtuSourceCache;
use crate::encoder::types::MAX_TB_LOG2;

pub(super) struct ExactModeWinner<Saved> {
    pub(super) mode: u8,
    pub(super) tt: TtPlan,
    pub(super) recon: Saved,
    pub(super) cost: f64,
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// Choose the CU's luma intra mode: an x265-shaped rough SATD pass over all
    /// 35 modes builds an MPM-protected shortlist (within 25% of the best rough
    /// cost), which is then evaluated by exact transform-tree RD. The retained
    /// exact winner carries its TT plan and detached recon patches; callers
    /// reattach it instead of replaying the winning mode.
    pub(super) fn decide_cu_luma_mode(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm: [IntraPredMode; 3],
        lambda: f64,
    ) -> ExactModeWinner<O::Saved> {
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let scale = CabacEstimator::SCALE as f64;

        let rough_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let size = 1usize << rough_log2;
        let n = size * size;
        let lambda_sad = lambda.sqrt();
        let mut rough: Vec<(f64, u8)> = Vec::with_capacity(35);
        let rough_timer = StillSearchLedger::start_timer();
        if state.bit_depth == 8 {
            let mut src = vec![0u8; n];
            self.source.sample_block_u8(0, x0, y0, size, &mut src);
            let src_u16;
            let mode_list =
                if angular_config().enabled() && rough_log2 >= angular_config().min_log2_size {
                    src_u16 = src.iter().map(|&v| u16::from(v)).collect::<Vec<_>>();
                    angular_filtered_modes(state, x0, y0, rough_log2, &src_u16, mpm)
                } else {
                    None
                };
            let mut pred = vec![0u8; n];
            let mut tmp_u16 = Vec::with_capacity(n);
            let modes = mode_list.as_deref().unwrap_or(ALL_INTRA_MODES.as_slice());
            for &m in modes.iter().filter(|&&m| m <= 1) {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into_u8(state, x0, y0, rough_log2, 0, mode, &mut pred, &mut tmp_u16);
                let satd = satd_u8(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
            }
            if modes.iter().any(|&m| (2..=34).contains(&m)) {
                let mut batch =
                    std::mem::take(&mut self.workspace.block_scratch.rough_angular_batch_u16);
                batch.resize(intra_angs::ANGULAR_MODES * n, 0);
                self.predict_all_angular_into_u16(state, x0, y0, rough_log2, &mut batch);
                let mut pred8 = std::mem::take(&mut self.workspace.block_scratch.rough_pred_u8);
                pred8.resize(n, 0);
                for &m in modes.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let off = intra_angs::slot_offset(m, rough_log2);
                    let slot = &batch[off..off + n];
                    for (d, &s) in pred8.iter_mut().zip(slot.iter()) {
                        *d = s.min(u8::MAX as u16) as u8;
                    }
                    let satd = satd_u8(&src, size, &pred8, size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
                }
                self.workspace.block_scratch.rough_pred_u8 = pred8;
                self.workspace.block_scratch.rough_angular_batch_u16 = batch;
            }
        } else {
            let mut src = vec![0u16; n];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self.source.sample(0, x0 + x as u32, y0 + y as u32);
                }
            }
            let mode_list = angular_filtered_modes(state, x0, y0, rough_log2, &src, mpm);
            let mut pred = vec![0u16; n];
            let modes = mode_list.as_deref().unwrap_or(ALL_INTRA_MODES.as_slice());
            for &m in modes.iter().filter(|&&m| m <= 1) {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into(state, x0, y0, rough_log2, 0, mode, &mut pred);
                let satd = satd_u16(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
            }
            if modes.iter().any(|&m| (2..=34).contains(&m)) {
                let mut batch =
                    std::mem::take(&mut self.workspace.block_scratch.rough_angular_batch_u16);
                batch.resize(intra_angs::ANGULAR_MODES * n, 0);
                self.predict_all_angular_into_u16(state, x0, y0, rough_log2, &mut batch);
                for &m in modes.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let off = intra_angs::slot_offset(m, rough_log2);
                    let slot = &batch[off..off + n];
                    let satd = satd_u16(&src, size, slot, size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
                }
                self.workspace.block_scratch.rough_angular_batch_u16 = batch;
            }
        }
        self.workspace.ledger.bump(WorkBucket::RoughLuma);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::RoughLuma, rough_timer);

        // Store the best rough score for the NxN skip heuristic (consumed by
        // decide_cu_min_leaf_or_nxn for 8×8 CUs).
        if log2_cb_size == 3 {
            self.workspace.last_8x8_rough_satd = rough[0].0;
        }

        // Cost-ascending, mode-ascending tiebreak so the shortlist is identical
        // regardless of the order modes were scored (batched angular vs the
        // per-mode exclusion path), matching the historical low-mode-first tie
        // resolution.
        rough.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let best_rough = rough[0].0;

        let rough_rd_cands = super::env::rough_rd_cands();
        let mut shortlist: Vec<u8> = Vec::with_capacity(ROUGH_RD_CANDS + 3);
        for &(cost, m) in rough.iter() {
            if shortlist.len() >= rough_rd_cands || cost > best_rough * 1.25 {
                break;
            }
            shortlist.push(m);
        }
        for &mm in mpm_u8.iter() {
            if !shortlist.contains(&mm) {
                shortlist.push(mm);
            }
        }

        if !super::env::luma_cheap_enabled() {
            let mut best: Option<ExactModeWinner<O::Saved>> = None;
            for &mode in &shortlist {
                let exact_timer = StillSearchLedger::start_timer();
                let mark = self.overlay.mark();
                let (tt, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
                let total = tt_cost + lambda * mbits as f64 / scale;
                let recon = self.overlay.detach_from(mark);
                self.workspace.ledger.bump(WorkBucket::LumaExact);
                self.workspace
                    .ledger
                    .finish_timer(WorkBucket::LumaExact, exact_timer);
                match &best {
                    Some(best) if best.cost <= total => {}
                    _ => {
                        best = Some(ExactModeWinner {
                            mode,
                            tt,
                            recon,
                            cost: total,
                        });
                    }
                }
            }
            let winner = best.expect("luma shortlist is never empty");
            #[cfg(test)]
            if winner.mode != IntraPredMode::Dc.as_u8() {
                super::api::LUMA_NONDC_PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return winner;
        }

        let mut cheap_ranked: Vec<(f64, u8)> = Vec::with_capacity(shortlist.len());
        for &mode in &shortlist {
            let cheap_timer = StillSearchLedger::start_timer();
            let mark = self.overlay.mark();
            let tt_cost =
                self.decide_tt_luma_no_optional_split(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let total = tt_cost + lambda * mbits as f64 / scale;
            self.overlay.truncate(mark);
            self.workspace.ledger.bump(WorkBucket::LumaCheap);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::LumaCheap, cheap_timer);
            cheap_ranked.push((total, mode));
        }
        cheap_ranked.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        // Quality-first variant of x265's second pass: remeasure the top-N
        // cheap-ranked modes with optional TU splits enabled, then retain the
        // best exact plan/recon. N defaults to 2; N=1 is fastest but showed
        // small rate losses on production-size smoke tests.
        let mut winner: Option<ExactModeWinner<O::Saved>> = None;
        let exact_top = super::env::luma_cheap_exact_top().min(cheap_ranked.len());
        for &(_, mode) in cheap_ranked.iter().take(exact_top) {
            let exact_timer = StillSearchLedger::start_timer();
            let mark = self.overlay.mark();
            let (tt, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let cost = tt_cost + lambda * mbits as f64 / scale;
            let recon = self.overlay.detach_from(mark);
            self.workspace.ledger.bump(WorkBucket::LumaExact);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::LumaExact, exact_timer);
            match &winner {
                Some(best) if best.cost <= cost => {}
                _ => {
                    winner = Some(ExactModeWinner {
                        mode,
                        tt,
                        recon,
                        cost,
                    });
                }
            }
        }

        let winner = winner.expect("cheap luma shortlist is never empty");
        #[cfg(test)]
        if winner.mode != IntraPredMode::Dc.as_u8() {
            super::api::LUMA_NONDC_PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        winner
    }

    pub(super) fn predict_all_angular_into_u16(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        dst: &mut [u16],
    ) {
        let tile_bounds = state.tile_clamp_bounds(x0, y0, 0);
        let overlay = &self.overlay;
        let (uf, ft, center, bit_depth) =
            bpg_hevc_decode::hevc::intra::build_reference_borders_with_reader(
                &state.frame,
                x0,
                y0,
                log2_size,
                0,
                true,
                |c, rx, ry| {
                    if let Some((tx0, ty0, tx1, ty1)) = tile_bounds {
                        if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                            return Some(bpg_hevc_decode::hevc::UNINIT_SAMPLE);
                        }
                    }
                    overlay.sample(c, rx, ry)
                },
            );
        primitives::intra_pred_allangs(dst, &uf, &ft, center, log2_size, 0, bit_depth);
    }
}

const ALL_INTRA_MODES: [u8; 35] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
];

fn angular_config() -> super::angular::AngularExclusionConfig {
    super::env::angular_exclusion_config()
}

fn angular_filtered_modes(
    state: &mut Encoder<'_>,
    x0: u32,
    y0: u32,
    log2_size: u8,
    src: &[u16],
    mpm: [IntraPredMode; 3],
) -> Option<Vec<u8>> {
    let cfg = angular_config();
    if !cfg.enabled() || log2_size < cfg.min_log2_size {
        return None;
    }

    // Diagnostic-only for now: this uses the committed-frame border builder
    // rather than StillSearch's overlay-aware prediction reader. Keep it
    // env-gated until an overlay-aware border builder and BD sweep prove it out.
    let (uf, ft, center, _) = bpg_hevc_decode::hevc::intra::build_reference_borders(
        &state.frame,
        x0,
        y0,
        log2_size,
        0,
        true,
    );
    let exclusion =
        super::angular::angular_exclusion_mask(cfg, src, &uf, &ft, center, log2_size, mpm);
    let mut modes: Vec<u8> = ALL_INTRA_MODES
        .iter()
        .copied()
        .filter(|&m| exclusion.mask.contains(m))
        .collect();
    let before_angular = 33usize;
    let after_angular = modes.iter().filter(|&&m| (2..=34).contains(&m)).count();
    if after_angular < before_angular {
        state.stats.angular_exclusions += 1;
        state.stats.rdo2_angular_exclusion_blocks += 1;
        state.stats.rdo2_angular_modes_before += before_angular as u64;
        state.stats.rdo2_angular_modes_after += after_angular as u64;
        state.stats.rdo2_angular_modes_removed += (before_angular - after_angular) as u64;
        state.stats.rdo2_angular_game_blocks += u64::from(exclusion.game_changed);
        state.stats.rdo2_angular_iame_blocks += u64::from(exclusion.iame_changed);
    }
    if modes.is_empty() {
        modes.extend([0, 1, mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()]);
        modes.sort_unstable();
        modes.dedup();
    }
    Some(modes)
}
