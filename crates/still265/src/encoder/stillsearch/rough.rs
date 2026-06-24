//! Rough and exact luma mode search.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::encoder::Encoder;
use crate::primitives::{satd_u8, satd_u16};

use super::depth::StillSearchDepth;
use super::ledger::WorkBucket;
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
        state: &Encoder<'_>,
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
        if state.bit_depth == 8 {
            let mut src = vec![0u8; n];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self
                        .source
                        .sample(0, x0 + x as u32, y0 + y as u32)
                        .min(u8::MAX as u16) as u8;
                }
            }
            let mut pred = vec![0u8; n];
            let mut tmp_u16 = Vec::with_capacity(n);
            for m in 0u8..=34 {
                let mode = IntraPredMode::from_u8(m).expect("0..=34 are valid intra modes");
                self.predict_into_u8(state, x0, y0, rough_log2, 0, mode, &mut pred, &mut tmp_u16);
                let satd = satd_u8(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
            }
        } else {
            let mut src = vec![0u16; n];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self.source.sample(0, x0 + x as u32, y0 + y as u32);
                }
            }
            let mut pred = vec![0u16; n];
            for m in 0u8..=34 {
                let mode = IntraPredMode::from_u8(m).expect("0..=34 are valid intra modes");
                self.predict_into(state, x0, y0, rough_log2, 0, mode, &mut pred);
                let satd = satd_u16(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
            }
        }
        self.workspace.ledger.bump(WorkBucket::RoughLuma);

        rough.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_rough = rough[0].0;

        let mut shortlist: Vec<u8> = Vec::with_capacity(ROUGH_RD_CANDS + 3);
        for &(cost, m) in rough.iter() {
            if shortlist.len() >= ROUGH_RD_CANDS || cost > best_rough * 1.25 {
                break;
            }
            shortlist.push(m);
        }
        for &mm in mpm_u8.iter() {
            if !shortlist.contains(&mm) {
                shortlist.push(mm);
            }
        }

        let mut best: Option<ExactModeWinner<O::Saved>> = None;
        for &mode in &shortlist {
            let mark = self.overlay.mark();
            let (tt, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let total = tt_cost + lambda * mbits as f64 / scale;
            let recon = self.overlay.detach_from(mark);
            self.workspace.ledger.bump(WorkBucket::LumaExact);
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
        winner
    }
}
