//! StillSearch public-in-crate entry points.
//!
//! Phase 4 (Local TU/RQT): the luma+chroma transform tree is decided by an
//! overlay-based full-vs-split RD search. Trial blocks predict into local
//! scratch via [`predict_intra_into_with_reader`] (reading the committed frame
//! immutably, overlay-first) and write their reconstructed samples into a
//! CTU-local [`ReconOverlay`]; only the winning branch's recon is committed to
//! the shared frame, exactly once per CU. No trial mutates `state.frame`.
//!
//! Scope note: luma TUs are not split below 8x8 yet (`log2_size > 3`), so every
//! leaf bears its chroma TB and the `parent_chroma` RQT special case never
//! arises. Coefficient retention still clones winning `levels` into the writer
//! `CodedBlock`; moving that into the `CoeffArena` is deferred to the speed
//! phase.

use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::UNINIT_SAMPLE;
use bpg_hevc_decode::hevc::intra;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::{CabacEstimator, ContextModel};
use crate::contexts::{Contexts, ctx};
use crate::plan::DecisionConfidence;
use crate::primitives::{satd_u16, ssd_u16};
use crate::residual::{apply_sign_data_hiding, estimate_residual_bits_into, get_scan_order};
use crate::transform;

use super::emit;
use super::ledger::WorkBucket;
use super::overlay::{ReconOverlay8, ReconOverlay16};
use super::workspace::{BlockScratch, CtuWorkspace};
use crate::encoder::Encoder;
use crate::encoder::syntax::{CodedBlock, CuNode, Tt};
use crate::encoder::types::{
    MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb,
};

/// Lowest `log2` luma TU size StillSearch will split down to in Phase 4. Keeping
/// this at 8x8 (`> 3`) guarantees every leaf TU carries chroma, deferring the
/// `parent_chroma` RQT case to a later phase.
const MIN_SPLIT_LOG2: u8 = 3;

#[cfg(test)]
pub(in crate::encoder) static SPLIT_WINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(in crate::encoder) static LEAF_WINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Count of CU luma-mode decisions that selected a non-DC mode (test guard for
/// the rough+RD mode search actually doing work).
#[cfg(test)]
pub(in crate::encoder) static LUMA_NONDC_PICKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(in crate::encoder) struct StillSearch {
    imp: StillSearchImpl,
}

enum StillSearchImpl {
    EightBit(StillSearchDepth<CtuSource8, ReconOverlay8>),
    HighBit(StillSearchDepth<CtuSource16, ReconOverlay16>),
}

struct StillSearchDepth<S, O> {
    workspace: CtuWorkspace,
    source: S,
    overlay: O,
}

impl StillSearch {
    pub(in crate::encoder) fn new(bit_depth: u8) -> Self {
        let imp = if bit_depth == 8 {
            StillSearchImpl::EightBit(StillSearchDepth::default())
        } else {
            StillSearchImpl::HighBit(StillSearchDepth::default())
        };
        Self { imp }
    }

    pub(in crate::encoder) fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        match &mut self.imp {
            StillSearchImpl::EightBit(search) => {
                search.build_ctu(state, x0, y0, log2_cb_size, ct_depth)
            }
            StillSearchImpl::HighBit(search) => {
                search.build_ctu(state, x0, y0, log2_cb_size, ct_depth)
            }
        }
    }
}

impl<S, O> Default for StillSearchDepth<S, O>
where
    S: Default,
    O: Default,
{
    fn default() -> Self {
        Self {
            workspace: CtuWorkspace::default(),
            source: S::default(),
            overlay: O::default(),
        }
    }
}

/// CTU-local reconstruction overlay used for non-mutating trials. Trials push
/// candidate recon patches; the winning branch commits once to `state.frame`.
trait OverlayCache {
    fn clear(&mut self);
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16>;
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]);
    fn mark(&self) -> usize;
    fn truncate(&mut self, mark: usize);
    fn drain_range(&mut self, start: usize, end: usize);
    fn commit_to_frame(&self, frame: &mut DecodedFrame);
}

impl OverlayCache for ReconOverlay8 {
    fn clear(&mut self) {
        ReconOverlay8::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay8::sample(self, c_idx, x, y).map(u16::from)
    }
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]) {
        ReconOverlay8::push_block(self, c_idx, x, y, width, height, samples);
    }
    fn mark(&self) -> usize {
        ReconOverlay8::mark(self)
    }
    fn truncate(&mut self, mark: usize) {
        ReconOverlay8::truncate(self, mark);
    }
    fn drain_range(&mut self, start: usize, end: usize) {
        ReconOverlay8::drain_range(self, start, end);
    }
    fn commit_to_frame(&self, frame: &mut DecodedFrame) {
        ReconOverlay8::commit_to_frame(self, frame);
    }
}

impl OverlayCache for ReconOverlay16 {
    fn clear(&mut self) {
        ReconOverlay16::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay16::sample(self, c_idx, x, y)
    }
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]) {
        ReconOverlay16::push_block(self, c_idx, x, y, width, height, samples);
    }
    fn mark(&self) -> usize {
        ReconOverlay16::mark(self)
    }
    fn truncate(&mut self, mark: usize) {
        ReconOverlay16::truncate(self, mark);
    }
    fn drain_range(&mut self, start: usize, end: usize) {
        ReconOverlay16::drain_range(self, start, end);
    }
    fn commit_to_frame(&self, frame: &mut DecodedFrame) {
        ReconOverlay16::commit_to_frame(self, frame);
    }
}

trait CtuSourceCache {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8);
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16;
}

#[derive(Default)]
struct CtuSource8 {
    planes: [CachedPlane8; 3],
}

#[derive(Default)]
struct CtuSource16 {
    planes: [CachedPlane16; 3],
}

#[derive(Default)]
struct CachedPlane8 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

#[derive(Default)]
struct CachedPlane16 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u16>,
}

impl CtuSourceCache for CtuSource8 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane.samples.push(
                        state
                            .src_sample(c_idx, px + dx, py + dy)
                            .min(u8::MAX as u16) as u8,
                    );
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y) as u16)
            .unwrap_or(128)
    }
}

impl CtuSourceCache for CtuSource16 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane
                        .samples
                        .push(state.src_sample(c_idx, px + dx, py + dy));
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y))
            .unwrap_or(128)
    }
}

impl CachedPlane8 {
    fn sample(&self, x: u32, y: u32) -> u8 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}

impl CachedPlane16 {
    fn sample(&self, x: u32, y: u32) -> u16 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}

/// Result of evaluating one transform block (luma or one chroma component).
struct BlockTrial {
    levels: Vec<i16>,
    cbf: bool,
    frac_bits: u64,
    cost: f64,
}

impl BlockTrial {
    fn into_coded_block(self) -> CodedBlock {
        CodedBlock {
            levels: self.levels,
            cbf: self.cbf,
            frac_bits: self.frac_bits,
        }
    }
}

/// HEVC luma RD lambda (HM/x265 `0.57 * 2^((qp-12)/3)`), in pixel-SSE units.
fn rd_lambda(qp: i32) -> f64 {
    0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0)
}

/// Estimated `split_transform_flag` entropy bits at this node (1/32768-bit
/// units), mirroring the context selection in `write.rs`.
fn split_flag_bits(ctxs: &Contexts, log2_size: u8, is_split: bool) -> u64 {
    let ci = ctx::SPLIT_TRANSFORM_FLAG + (5usize.saturating_sub(log2_size as usize)).min(2);
    ctxs.models[ci].entropy_bits(is_split as u8) as u64
}

#[inline]
fn entropy_bits(model: &ContextModel, bin: u8) -> u64 {
    model.entropy_bits(bin) as u64
}

/// Max rough-search candidates (within the x265 25% threshold) carried into the
/// exact RD luma-mode pass, before MPMs are unioned in. Tunable per effort tier.
const ROUGH_RD_CANDS: usize = 4;

/// Estimated `prev_intra_luma_pred_flag` + `mpm_idx`/`rem_intra_luma_pred_mode`
/// bits (1/32768-bit units) for signaling `mode` given the MPM list, mirroring
/// `write_intra_luma_mode`.
fn luma_mode_bits(ctxs: &Contexts, mpm_u8: [u8; 3], mode: u8) -> u64 {
    let prev = &ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let bypass = CabacEstimator::SCALE;
    match mpm_u8.iter().position(|&m| m == mode) {
        Some(0) => entropy_bits(prev, 1) + bypass,
        Some(_) => entropy_bits(prev, 1) + 2 * bypass,
        None => entropy_bits(prev, 0) + 5 * bypass,
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        self.workspace.reset();
        self.workspace.set_price_qp(state.cur_qp_y);
        self.source.reset_from_ctu(state, x0, y0, log2_cb_size);
        self.overlay.clear();
        self.build_cu(state, x0, y0, log2_cb_size, ct_depth)
    }

    fn build_cu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        let cb_size = 1u32 << log2_cb_size;
        let fully_inside =
            x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
        if !fully_inside && log2_cb_size > 3 {
            let half = cb_size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            // Final syntax only. Candidate search must use PlanId/TtPlanId arenas.
            let mut kids = Vec::with_capacity(4);
            kids.push(self.build_cu(state, x0, y0, log2_cb_size - 1, ct_depth + 1));
            if x1 < state.display_width {
                kids.push(self.build_cu(state, x1, y0, log2_cb_size - 1, ct_depth + 1));
            }
            if y1 < state.display_height {
                kids.push(self.build_cu(state, x0, y1, log2_cb_size - 1, ct_depth + 1));
            }
            if x1 < state.display_width && y1 < state.display_height {
                kids.push(self.build_cu(state, x1, y1, log2_cb_size - 1, ct_depth + 1));
            }
            return CuNode::Split { kids };
        }

        state.stats.cu_trials += 1;
        state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let mpm = intra::fill_mpm_candidates(
            state.neighbor_left_mode(x0, y0),
            state.neighbor_above_mode(x0, y0),
        );

        if state.aq.active {
            state.aq_set_cu_qp(x0, y0);
        }
        let lambda = rd_lambda(state.cur_qp_y);

        // Phase 5: choose the luma mode by rough SATD shortlist + exact RD.
        self.overlay.clear();
        let luma_mode = self.decide_cu_luma_mode(state, x0, y0, log2_cb_size, mpm, lambda);
        state.store_mode(x0, y0, log2_cb_size, luma_mode);

        // Re-run the winning mode to populate the overlay with its recon; this is
        // the single final transform-tree decision committed to the frame.
        self.overlay.clear();
        let (tt, _cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, luma_mode, lambda);

        // Commit the winning branch's recon to the shared frame, exactly once.
        self.overlay.commit_to_frame(&mut state.frame);
        self.overlay.clear();
        self.workspace.ledger.bump(WorkBucket::FinalCommit);

        state.stats.final_coded_blocks += 1;
        CuNode::Leaf(emit::cu_leaf(mpm, luma_mode, tt)).tap_confidence(DecisionConfidence::Clear)
    }

    /// Choose the CU's luma intra mode: an x265-shaped rough SATD pass over all
    /// 35 modes builds an MPM-protected shortlist (within 25% of the best rough
    /// cost), which is then evaluated by exact transform-tree RD. The overlay is
    /// left empty (each trial's patches are rewound).
    fn decide_cu_luma_mode(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm: [IntraPredMode; 3],
        lambda: f64,
    ) -> u8 {
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let scale = CabacEstimator::SCALE as f64;

        // Rough pass: SATD of each mode's prediction over a representative block
        // (CU size, capped at the max TU) from the committed neighbours.
        let rough_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let size = 1usize << rough_log2;
        let n = size * size;
        let mut src = vec![0u16; n];
        for y in 0..size {
            for x in 0..size {
                src[y * size + x] = self.source.sample(0, x0 + x as u32, y0 + y as u32);
            }
        }
        let lambda_sad = lambda.sqrt();
        let mut pred = vec![0u16; n];
        let mut rough: Vec<(f64, u8)> = Vec::with_capacity(35);
        for m in 0u8..=34 {
            let mode = IntraPredMode::from_u8(m).expect("0..=34 are valid intra modes");
            self.predict_into(state, x0, y0, rough_log2, 0, mode, &mut pred);
            let satd = satd_u16(&src, size, &pred, size, size);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
            rough.push((satd as f64 + lambda_sad * mbits as f64 / scale, m));
        }
        self.workspace.ledger.bump(WorkBucket::RoughLuma);

        rough.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let best_rough = rough[0].0;

        // Shortlist: the cheapest rough candidates within 25% of the best, then
        // union the MPMs so a context-cheap mode is never dropped.
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

        // Exact pass: full transform-tree RD per shortlist mode (+ mode bits).
        let mut best_mode = shortlist[0];
        let mut best_total = f64::INFINITY;
        for &mode in &shortlist {
            let mark = self.overlay.mark();
            let (_tt, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let total = tt_cost + lambda * mbits as f64 / scale;
            self.overlay.truncate(mark);
            self.workspace.ledger.bump(WorkBucket::LumaExact);
            if total < best_total {
                best_total = total;
                best_mode = mode;
            }
        }
        #[cfg(test)]
        if best_mode != IntraPredMode::Dc.as_u8() {
            LUMA_NONDC_PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        best_mode
    }

    /// Predict a block into caller-owned `dst` (`size*size`, stride `size`),
    /// reading the committed frame immutably with overlay-first reference
    /// samples and tile-boundary clamping. Never mutates the shared frame.
    fn predict_into(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: IntraPredMode,
        dst: &mut [u16],
    ) {
        let size = 1usize << log2_size;
        let tile_bounds = state.tile_clamp_bounds(x0, y0, c_idx);
        let overlay = &self.overlay;
        intra::predict_intra_into_with_reader(
            &state.frame,
            x0,
            y0,
            log2_size,
            mode,
            c_idx,
            true,
            dst,
            size,
            |c, rx, ry| {
                if let Some((tx0, ty0, tx1, ty1)) = tile_bounds {
                    if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                        return Some(UNINIT_SAMPLE);
                    }
                }
                overlay.sample(c, rx, ry)
            },
        );
    }

    /// Decide one transform-tree node: full leaf vs four split children, by RD
    /// cost. On return the overlay holds exactly the winner's recon patches for
    /// this region (loser patches are rewound).
    fn decide_tt(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        lambda: f64,
    ) -> (Tt, f64) {
        let must_split = log2_size > MAX_TB_LOG2;
        // 4:2:2 (cat 2) RD split is deferred: its stacked second chroma TB needs
        // the parent-chroma cbf-propagation special case at split nodes, which is
        // not yet modeled. Force-leaf there (the pre-Phase-4 behavior). 4:2:0 and
        // 4:4:4 split cleanly because chroma follows luma at every leaf.
        let can_split = !must_split
            && state.cat != 2
            && log2_size > MIN_SPLIT_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH;

        if must_split {
            return self.eval_tt_split(state, x0, y0, log2_size, trafo_depth, luma_mode, false, lambda);
        }
        if !can_split {
            return self.eval_tt_leaf(state, x0, y0, log2_size, trafo_depth, luma_mode, false, lambda);
        }

        // Both candidates are legal. Evaluate SPLIT first, then LEAF, keeping
        // only the winner's overlay patches.
        //
        // Order matters for reference-sample correctness: the leaf's recon patch
        // covers the whole region, so if it were pushed first, a split child's
        // extended references (planar bottom-left/top-right, angular) into a
        // not-yet-coded sibling sub-region would read the leaf's phantom recon
        // instead of being unavailable — diverging from the decoder. A leaf
        // never reads inside its own region, so evaluating it after the split
        // (whose patches sit underneath) is safe.
        let mark0 = self.overlay.mark();
        let (split_tt, split_cost) =
            self.eval_tt_split(state, x0, y0, log2_size, trafo_depth, luma_mode, true, lambda);
        let split_end = self.overlay.mark();
        let (leaf_tt, leaf_cost) =
            self.eval_tt_leaf(state, x0, y0, log2_size, trafo_depth, luma_mode, true, lambda);

        if split_cost < leaf_cost {
            #[cfg(test)]
            SPLIT_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Drop the leaf patch; the split children's patches remain visible.
            self.overlay.truncate(split_end);
            (split_tt, split_cost)
        } else {
            #[cfg(test)]
            LEAF_WINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Drop the split children's patches; the leaf patch remains.
            self.overlay.drain_range(mark0, split_end);
            (leaf_tt, leaf_cost)
        }
    }

    /// Evaluate this node as a leaf TU (luma + chroma), returning its final
    /// syntax and RD cost and pushing its recon to the overlay.
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_leaf(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
    ) -> (Tt, f64) {
        self.workspace.ledger.bump(WorkBucket::TuLeaf);

        let cat = state.cat;
        let qp_y = state.cur_qp_y;
        let luma_pred = IntraPredMode::from_u8(luma_mode).unwrap_or(IntraPredMode::Dc);
        let luma = self.eval_component(state, x0, y0, log2_size, 0, luma_pred, qp_y, trafo_depth, lambda);
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

                let cb_e = self.eval_component(state, cx, cy, clog2, 1, cmode, qp_c, trafo_depth, lambda);
                cost += cb_e.cost;
                cb = cb_e.into_coded_block();

                let cr_e = self.eval_component(state, cx, cy, clog2, 2, cmode, qp_c, trafo_depth, lambda);
                cost += cr_e.cost;
                cr = cr_e.into_coded_block();

                if count > 1 {
                    let y1 = cy + step;
                    let cb1_e = self.eval_component(state, cx, y1, clog2, 1, cmode, qp_c, trafo_depth, lambda);
                    cost += cb1_e.cost;
                    cb1 = cb1_e.into_coded_block();

                    let cr1_e = self.eval_component(state, cx, y1, clog2, 2, cmode, qp_c, trafo_depth, lambda);
                    cost += cr1_e.cost;
                    cr1 = cr1_e.into_coded_block();
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
            luma.into_coded_block(),
            cb,
            cr,
            cb1,
            cr1,
        );
        (tt, cost)
    }

    /// Evaluate this node as a split into four child transform trees.
    #[allow(clippy::too_many_arguments)]
    fn eval_tt_split(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        code_split_flag: bool,
        lambda: f64,
    ) -> (Tt, f64) {
        self.workspace.ledger.bump(WorkBucket::TuSplit);

        let half = 1u32 << (log2_size - 1);
        let kid_log2 = log2_size - 1;
        let kid_depth = trafo_depth + 1;
        let (k0, c0) = self.decide_tt(state, x0, y0, kid_log2, kid_depth, luma_mode, lambda);
        let (k1, c1) = self.decide_tt(state, x0 + half, y0, kid_log2, kid_depth, luma_mode, lambda);
        let (k2, c2) = self.decide_tt(state, x0, y0 + half, kid_log2, kid_depth, luma_mode, lambda);
        let (k3, c3) =
            self.decide_tt(state, x0 + half, y0 + half, kid_log2, kid_depth, luma_mode, lambda);

        let kids = vec![k0, k1, k2, k3];
        let cbf_cb = kids.iter().any(Tt::cbf_cb);
        let cbf_cr = kids.iter().any(Tt::cbf_cr);
        let cbf_cb1 = kids.iter().any(Tt::cbf_cb1);
        let cbf_cr1 = kids.iter().any(Tt::cbf_cr1);

        let mut cost = c0 + c1 + c2 + c3;
        if code_split_flag {
            let bits = split_flag_bits(&self.workspace.price_base, log2_size, true);
            cost += lambda * bits as f64 / CabacEstimator::SCALE as f64;
        }

        let tt = Tt::Split {
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

    /// Evaluate a single transform block (one component) without mutating the
    /// shared frame. Predicts into local scratch (overlay-first reference
    /// reads), prices the coded and null-CBF candidates, and pushes the winning
    /// recon to the overlay. Returns the retained levels/cbf and RD cost.
    #[allow(clippy::too_many_arguments)]
    fn eval_component(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: IntraPredMode,
        qp: i32,
        trafo_depth: u8,
        lambda: f64,
    ) -> BlockTrial {
        let size = 1usize << log2_size;
        let n = size * size;
        let bit_depth = state.bit_depth;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;

        // 1. Gather the source block.
        let mut src = vec![0u16; n];
        for y in 0..size {
            for x in 0..size {
                src[y * size + x] = self.source.sample(c_idx, x0 + x as u32, y0 + y as u32);
            }
        }

        // 2. Predict into local scratch, reading the committed frame immutably
        //    with overlay-first reference samples (no frame mutation).
        let mut pred = vec![0u16; n];
        self.predict_into(state, x0, y0, log2_size, c_idx, mode, &mut pred);

        // 3. Residual -> forward transform -> quant (+ optional sign hiding) and
        //    reconstruct the coded candidate, all in CTU-local scratch.
        let mut levels_vec = Vec::new();
        let mut nnz;
        let mut recon_coded: Option<Vec<u16>> = None;
        let dist_zero = ssd_u16(&src, size, &pred, size, size);
        let mut dist_coded = dist_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);
            for i in 0..n {
                residual[i] =
                    (src[i] as i32 - pred[i] as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }

            transform::forward_transform_into(residual, log2_size, is_dst4, bit_depth, coeff, tmp);
            nnz = transform::quantize_into(coeff, log2_size, qp, bit_depth, levels);
            if sdh_enabled && nnz > 0 {
                let (scale, qbits) = transform::quant_params(log2_size, qp, bit_depth);
                nnz =
                    apply_sign_data_hiding(levels, coeff, log2_size, scan, scale, qbits, nnz);
            }

            if nnz > 0 {
                transform::reconstruct_residual_into(
                    levels,
                    log2_size,
                    qp,
                    bit_depth,
                    is_dst4,
                    dequant,
                    recon_residual,
                );
                let max_sample = (1i32 << bit_depth) - 1;
                let mut recon = vec![0u16; n];
                for i in 0..n {
                    recon[i] = (pred[i] as i32 + recon_residual[i] as i32).clamp(0, max_sample)
                        as u16;
                }
                dist_coded = ssd_u16(&src, size, &recon, size, size);
                levels_vec = levels.clone();
                recon_coded = Some(recon);
            }
        }

        // 4. Price the coded vs null-CBF candidates and pick the cheaper.
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 0);
        let cbf1_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 1);

        let scale = CabacEstimator::SCALE as f64;
        let cost_zero = dist_zero as f64 + lambda * cbf0_bits as f64 / scale;

        let coded = if nnz > 0 {
            let residual_bits = estimate_residual_bits_into(
                &self.workspace.price_base,
                &levels_vec,
                log2_size,
                c_idx,
                scan,
                sdh_enabled,
                &mut self.workspace.price_scratch,
            );
            self.workspace.ledger.bump(WorkBucket::ResidualPrice);
            let cost = dist_coded as f64 + lambda * (residual_bits + cbf1_bits) as f64 / scale;
            Some((residual_bits, cost))
        } else {
            None
        };

        match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                let recon = recon_coded.expect("coded candidate has recon");
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &recon);
                BlockTrial {
                    levels: levels_vec,
                    cbf: true,
                    frac_bits: residual_bits,
                    cost: cost_coded,
                }
            }
            _ => {
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &pred);
                BlockTrial {
                    levels: Vec::new(),
                    cbf: false,
                    frac_bits: 0,
                    cost: cost_zero,
                }
            }
        }
    }
}

type BlockScratchParts<'a> = (
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
);

fn scratch_parts_for_log2(scratch: &mut BlockScratch, log2_size: u8) -> BlockScratchParts<'_> {
    match log2_size {
        2 => (
            &mut scratch.residual_i16_4x4,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        3 => (
            &mut scratch.residual_i16_8x8,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        4 => (
            &mut scratch.residual_i16_16x16,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        _ => (
            &mut scratch.residual_i16_32x32,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
    }
}

trait TapConfidence {
    fn tap_confidence(self, confidence: DecisionConfidence) -> Self;
}

impl TapConfidence for CuNode {
    fn tap_confidence(mut self, confidence: DecisionConfidence) -> Self {
        if let CuNode::Leaf(leaf) = &mut self {
            leaf.confidence = confidence;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    fn scan_eval_function_bodies(src: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let bytes = src.as_bytes();
        let mut i = 0;
        while let Some(pos) = src[i..].find("fn eval_") {
            let fn_start = i + pos;
            let Some(open_rel) = src[fn_start..].find('{') else {
                break;
            };
            let open = fn_start + open_rel;
            let mut depth = 0i32;
            let mut end = open;
            for (off, b) in bytes[open..].iter().enumerate() {
                match *b {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + off + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            out.push(&src[fn_start..end]);
            i = end;
        }
        out
    }

    #[test]
    fn luma_mode_search_selects_directional_mode() {
        use crate::encoder::{Source, encode};
        use crate::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};
        use std::sync::atomic::Ordering;

        // Vertical stripes: DC is a poor predictor; the rough+RD search (with
        // MPM-protected shortlist) must pick a non-DC mode for these CUs.
        let (w, h) = (128usize, 128usize);
        let mut y = vec![0u16; w * h];
        for px in y.iter_mut().enumerate() {
            let i = px.0 % w;
            *px.1 = if (i / 4) & 1 == 0 { 60 } else { 200 };
        }
        let chroma = vec![128u16; (w / 2) * (h / 2)];
        let config = StillHevcConfig {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 27,
            effort: Effort::Balanced,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
            adaptive_qp: false,
        };
        super::LUMA_NONDC_PICKS.store(0, Ordering::Relaxed);
        let _ = encode(
            &config,
            Source {
                y: &y,
                cb: &chroma,
                cr: &chroma,
            },
        );
        let nondc = super::LUMA_NONDC_PICKS.load(Ordering::Relaxed);
        assert!(
            nondc > 0,
            "luma mode search never picked a non-DC mode on vertical-stripe content"
        );
    }

    #[test]
    fn tu_rd_split_wins_on_localized_detail() {
        use crate::encoder::{Source, encode};
        use crate::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};
        use std::sync::atomic::Ordering;

        // 64x64 mostly-flat luma with a high-frequency patch in the top-left
        // 16x16: a 32x32 DC TU must code that detail through one large
        // transform, while splitting isolates it (three near-zero quadrants +
        // one coded), so RD split should win at least once.
        let (w, h) = (64usize, 64usize);
        let mut y = vec![128u16; w * h];
        for j in 0..16 {
            for i in 0..16 {
                y[j * w + i] = if (i ^ j) & 1 == 0 { 40 } else { 210 };
            }
        }
        let chroma = vec![128u16; (w / 2) * (h / 2)];

        let config = StillHevcConfig {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 27,
            effort: Effort::Balanced,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
            adaptive_qp: false,
        };

        super::SPLIT_WINS.store(0, Ordering::Relaxed);
        super::LEAF_WINS.store(0, Ordering::Relaxed);
        let _ = encode(
            &config,
            Source {
                y: &y,
                cb: &chroma,
                cr: &chroma,
            },
        );
        let splits = super::SPLIT_WINS.load(Ordering::Relaxed);
        let leaves = super::LEAF_WINS.load(Ordering::Relaxed);
        assert!(
            splits > 0,
            "TU RD split never won on localized-detail content (splits={splits}, leaves={leaves})"
        );
    }

    #[test]
    fn search_eval_helpers_do_not_mutate_shared_frame() {
        for body in scan_eval_function_bodies(include_str!("api.rs")) {
            assert!(
                !body.contains("state.frame.plane_mut") && !body.contains("predict_intra_tiled"),
                "eval_* functions must write to overlays/scratch, not mutate state.frame directly:\n{body}"
            );
        }
    }
}
