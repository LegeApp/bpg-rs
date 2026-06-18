//! Still-image intra slice-data encoder (Phase 5 milestone 1/2).
//!
//! Emits a full IDR access unit (VPS/SPS/PPS + slice) for an 8-bit/10-bit
//! 4:2:0 or 4:4:4 picture. The CABAC
//! slice data is produced by walking the coding tree exactly as
//! `bpg-hevc-decode`'s `ctu` decoder parses it, in lockstep:
//! `split_cu_flag`, `coding_unit` (intra mode
//! signalling), `transform_tree` (forced 64->32 split, cbf flags), and
//! `residual_coding` (via [`crate::residual::encode_residual`]).
//!
//! Reconstruction reuses the decoder's `predict_intra` + the dequant/inverse
//! transform path in [`crate::transform`] (proven bit-identical to the
//! decoder), so each transform block is predicted from, and reconstructed
//! into, a `DecodedFrame` in coding order — guaranteeing the decoder
//! reproduces the encoder's reconstruction sample-for-sample.
//!
//! Milestone-1 simplifications (all produce a valid, decodable stream):
//! fixed DM chroma, `sign_data_hiding = false`. Deblocking (`config.deblock`),
//! monochrome/4:2:0/4:2:2/4:4:4, and conservative SAO (`config.sao`, see
//! [`Encoder::decide_sao_map`]) are implemented. Full RDO CU/TU splitting
//! comes in a later milestone.

mod aq;
mod rdo;
mod snapshot;
mod types;
mod write;

use bpg_hevc_decode::hevc::sao::{apply_sao, SaoMap};
use bpg_hevc_decode::hevc::slice::IntraPredMode;
use bpg_hevc_decode::DecodedFrame;

use crate::analysis_cache::{AnalysisCache, CacheDecisionConfidence};
use crate::contexts::Contexts;
use crate::effort::{BlockDesc, BlockSearchBudget, EffortTemplate};
use crate::plan::DecisionConfidence;
use crate::sao;
use crate::{nal, params, slice, DeblockMode, Effort, SaoMode, StillHevcConfig};

use self::aq::AqState;
use self::types::*;
pub use self::types::{EncodeStats, Source};
use self::write::{
    build_slice_trees_parallel, build_slice_trees_serial, encode_slice_data,
    encode_slice_data_parallel, write_slice_from_trees,
};

struct Encoder<'a> {
    display_width: u32,
    display_height: u32,
    cat: u8,
    qp_y: i32,
    qp_c: i32,
    bit_depth: u8,
    effort: Effort,
    effort_template: EffortTemplate,
    src: Source<'a>,
    frame: DecodedFrame,
    mode_map: Vec<u8>,
    mode_stride: usize,
    ct_depth_map: Vec<u8>,
    ct_depth_stride: usize,
    tu_depth_map: Vec<u8>,
    tu_depth_stride: usize,
    single_scan_rdoq: bool,
    deblock: bool,
    scratch_residual: Vec<i16>,
    scratch_coeffs: Vec<i16>,
    scratch_transform_tmp: Vec<i16>,
    scratch_src8: Vec<u8>,
    scratch_pred: Vec<u16>,
    scratch_pred8: Vec<u8>,
    scratch_scored: Vec<(u64, u8)>,
    /// Batch buffer for `intra_pred_allangs` (33 angular slots), pooled per CU.
    scratch_allangs: Vec<u16>,
    analysis: std::sync::Arc<crate::preanalysis::AnalysisMaps>,
    analysis_cache: AnalysisCache,
    cur_policy: Option<crate::preanalysis::SearchPolicy>,
    cur_qp_y: i32,
    cur_qp_c: i32,
    luma_trial_quality_override: Option<crate::effort::TrialQuality>,
    best2_chroma_gate: Option<f64>,
    best2_chroma_protect: bool,
    best2_luma_fastrd: bool,
    best2_rough_lambda: bool,
    best2_rd_lambda: bool,
    best2_rdoq2: bool,
    sign_data_hiding: bool,
    best2_wpp: bool,
    best_luma_leaf_screen: bool,
    best_tu_neighbor_limit: bool,
    /// `Some((perceptual, strength))` when the experimental bidirectional
    /// variance AQ is active for `Best` (drives [`Encoder::aq_qg_target`]).
    best_aq: Option<(bool, f32)>,
    best2_tt_reuse: bool,
    best2_cu_reuse: bool,
    /// Experimental: evaluate 8x8 `PartNxN` (four 4x4 luma PUs) against the
    /// normal `Part2Nx2N` CU and pick the RD winner. Env-gated (`BPG_PARTNXN`);
    /// only effective on the winner-direct `best2_cu_reuse` path.
    partnxn: bool,
    aq: AqState,
    prof: Profiler,
    trace: crate::trace::SearchTrace,
    stats: EncodeStats,
}

impl<'a> Encoder<'a> {
    fn src_luma_stride(&self) -> usize {
        self.display_width as usize
    }

    fn src_chroma_dims(&self) -> (u32, u32) {
        match self.cat {
            0 => (0, 0),
            1 => (
                self.display_width.div_ceil(2),
                self.display_height.div_ceil(2),
            ),
            2 => (self.display_width.div_ceil(2), self.display_height),
            3 => (self.display_width, self.display_height),
            _ => unreachable!(),
        }
    }

    fn src_chroma_stride(&self) -> usize {
        self.src_chroma_dims().0 as usize
    }

    fn src_plane(&self, c_idx: u8) -> (&[u16], usize, u32, u32) {
        match c_idx {
            0 => (
                self.src.y,
                self.src_luma_stride(),
                self.display_width,
                self.display_height,
            ),
            1 => {
                let (cw, ch) = self.src_chroma_dims();
                (self.src.cb, self.src_chroma_stride(), cw, ch)
            }
            2 => {
                let (cw, ch) = self.src_chroma_dims();
                (self.src.cr, self.src_chroma_stride(), cw, ch)
            }
            _ => (&[], 0, 0, 0),
        }
    }

    fn src_sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        let (plane, stride, w, h) = match c_idx {
            0 => (
                self.src.y,
                self.src_luma_stride(),
                self.display_width,
                self.display_height,
            ),
            1 => {
                let (cw, ch) = self.src_chroma_dims();
                (self.src.cb, self.src_chroma_stride(), cw, ch)
            }
            2 => {
                let (cw, ch) = self.src_chroma_dims();
                (self.src.cr, self.src_chroma_stride(), cw, ch)
            }
            _ => return 0,
        };
        let sx = x.min(w.saturating_sub(1));
        let sy = y.min(h.saturating_sub(1));
        plane[sy as usize * stride + sx as usize]
    }

    fn sao_eo_stats(
        &self,
        c_idx: u8,
        eo_class: u8,
        x_start: u32,
        y_start: u32,
        x_end: u32,
        y_end: u32,
    ) -> sao::EoStats {
        let (plane, stride) = self.frame.plane(c_idx);
        let (plane_w, plane_h) = self.frame.component_dims(c_idx);

        // Horizontal EO over an interior region (left/right neighbours valid and
        // the source unclamped) uses the dispatched SIMD kernel. Plane-edge /
        // source-padding CTBs fall through to the generic scalar loop below.
        if eo_class == 0 {
            let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);
            if x_start >= 1
                && x_end > x_start
                && x_end + 1 <= plane_w
                && x_end <= sw
                && y_end <= plane_h
                && y_end <= sh
            {
                let mut stats = sao::EoStats::default();
                crate::primitives::sao_stats_e0(
                    plane,
                    stride,
                    src_plane,
                    src_stride,
                    x_start,
                    y_start,
                    x_end - x_start,
                    y_end - y_start,
                    &mut stats.sum,
                    &mut stats.count,
                );
                return stats;
            }
        }

        let (dx0, dy0, dx1, dy1) = sao::EO_OFFSETS[eo_class as usize & 3];

        let mut stats = sao::EoStats::default();
        for y in y_start..y_end {
            for x in x_start..x_end {
                let nx0 = x as i32 + dx0;
                let ny0 = y as i32 + dy0;
                let nx1 = x as i32 + dx1;
                let ny1 = y as i32 + dy1;
                if nx0 < 0
                    || nx0 >= plane_w as i32
                    || ny0 < 0
                    || ny0 >= plane_h as i32
                    || nx1 < 0
                    || nx1 >= plane_w as i32
                    || ny1 < 0
                    || ny1 >= plane_h as i32
                {
                    continue;
                }

                let idx = y as usize * stride + x as usize;
                let recon = plane[idx] as i32;
                let n0 = plane[ny0 as usize * stride + nx0 as usize] as i32;
                let n1 = plane[ny1 as usize * stride + nx1 as usize] as i32;

                let sign0 = (recon - n0).signum();
                let sign1 = (recon - n1).signum();
                let edge_idx = (2 + sign0 + sign1) as usize;
                if edge_idx == 2 {
                    continue;
                }

                let src = self.src_sample(c_idx, x, y) as i64;
                stats.sum[edge_idx] += src - recon as i64;
                stats.count[edge_idx] += 1;
            }
        }
        stats
    }

    fn sao_bo_stats(
        &self,
        c_idx: u8,
        x_start: u32,
        y_start: u32,
        x_end: u32,
        y_end: u32,
    ) -> sao::BoStats {
        let (plane, stride) = self.frame.plane(c_idx);
        let band_shift = self.bit_depth.saturating_sub(5);
        let mut stats = sao::BoStats::default();
        for y in y_start..y_end {
            for x in x_start..x_end {
                let idx = y as usize * stride + x as usize;
                let recon = plane[idx] as i32;
                let src = self.src_sample(c_idx, x, y) as i64;
                let band = ((recon as u16 >> band_shift) & 31) as usize;
                stats.sum[band] += src - recon as i64;
                stats.count[band] += 1;
            }
        }
        stats
    }

    fn best_sao_component(
        &self,
        c_idx: u8,
        x_start: u32,
        y_start: u32,
        x_end: u32,
        y_end: u32,
    ) -> (u8, u8, [i8; 4], u8, i64) {
        let mut best = (0u8, 0u8, [0i8; 4], 0u8, 0i64);

        let bo_stats = self.sao_bo_stats(c_idx, x_start, y_start, x_end, y_end);
        let (band_pos, bo_offsets, bo_reduction) =
            sao::bo_offsets_and_reduction(&bo_stats, self.bit_depth);
        if bo_reduction > best.4 {
            best = (1, 0, bo_offsets, band_pos, bo_reduction);
        }

        for eo_class in 0..4u8 {
            let stats = self.sao_eo_stats(c_idx, eo_class, x_start, y_start, x_end, y_end);
            let (offsets, reduction) = sao::eo_offsets_and_reduction(&stats, self.bit_depth);
            if reduction > best.4 {
                best = (2, eo_class, offsets, 0, reduction);
            }
        }

        best
    }

    fn decide_sao_map(&self, ctb_size: u32) -> SaoMap {
        let ctbs_x = self.display_width.div_ceil(ctb_size);
        let ctbs_y = self.display_height.div_ceil(ctb_size);
        let mut map = SaoMap::new(ctbs_x, ctbs_y);
        let (sub_x, sub_y) = self.frame.chroma_subsampling();

        let (luma_w, luma_h) = self.frame.component_dims(0);
        let (chroma_w, chroma_h) = if self.cat == 0 {
            (0, 0)
        } else {
            self.frame.component_dims(1)
        };

        for ctb_y in 0..ctbs_y {
            for ctb_x in 0..ctbs_x {
                let x0 = ctb_x * ctb_size;
                let y0 = ctb_y * ctb_size;
                let info = map.get_mut(ctb_x, ctb_y);

                let lx_end = (x0 + ctb_size).min(luma_w);
                let ly_end = (y0 + ctb_size).min(luma_h);
                let best = self.best_sao_component(0, x0, y0, lx_end, ly_end);
                if best.4 > 0 {
                    info.sao_type_idx[0] = best.0;
                    info.sao_eo_class[0] = best.1;
                    info.sao_offset_val[0] = best.2;
                    info.sao_band_position[0] = best.3;
                }

                if self.cat != 0 {
                    let cx0 = x0 / sub_x;
                    let cy0 = y0 / sub_y;
                    let cx_end = (cx0 + ctb_size / sub_x).min(chroma_w);
                    let cy_end = (cy0 + ctb_size / sub_y).min(chroma_h);
                    let mut best_chroma = (0u8, 0u8, 0u8, 0u8, [0i8; 4], [0i8; 4], 0i64);

                    let cb_bo_stats = self.sao_bo_stats(1, cx0, cy0, cx_end, cy_end);
                    let cr_bo_stats = self.sao_bo_stats(2, cx0, cy0, cx_end, cy_end);
                    let (cb_band, cb_offsets, cb_reduction) =
                        sao::bo_offsets_and_reduction(&cb_bo_stats, self.bit_depth);
                    let (cr_band, cr_offsets, cr_reduction) =
                        sao::bo_offsets_and_reduction(&cr_bo_stats, self.bit_depth);
                    let total = cb_reduction + cr_reduction;
                    if total > best_chroma.6 {
                        best_chroma = (1, 0, cb_band, cr_band, cb_offsets, cr_offsets, total);
                    }

                    for eo_class in 0..4u8 {
                        let cb_stats = self.sao_eo_stats(1, eo_class, cx0, cy0, cx_end, cy_end);
                        let cr_stats = self.sao_eo_stats(2, eo_class, cx0, cy0, cx_end, cy_end);
                        let (cb_offsets, cb_reduction) =
                            sao::eo_offsets_and_reduction(&cb_stats, self.bit_depth);
                        let (cr_offsets, cr_reduction) =
                            sao::eo_offsets_and_reduction(&cr_stats, self.bit_depth);
                        let total = cb_reduction + cr_reduction;
                        if total > best_chroma.6 {
                            best_chroma = (2, eo_class, 0, 0, cb_offsets, cr_offsets, total);
                        }
                    }
                    if best_chroma.6 > 0 {
                        info.sao_type_idx[1] = best_chroma.0;
                        info.sao_type_idx[2] = best_chroma.0;
                        info.sao_eo_class[1] = best_chroma.1;
                        info.sao_eo_class[2] = best_chroma.1;
                        info.sao_band_position[1] = best_chroma.2;
                        info.sao_band_position[2] = best_chroma.3;
                        info.sao_offset_val[1] = best_chroma.4;
                        info.sao_offset_val[2] = best_chroma.5;
                    }
                }
            }
        }
        map
    }

    fn frame_plane_dims(&self, c_idx: u8) -> (usize, usize) {
        match c_idx {
            0 => (self.frame.width as usize, self.frame.height as usize),
            1 | 2 => match self.cat {
                0 => (0, 0),
                1 => (
                    self.frame.width.div_ceil(2) as usize,
                    self.frame.height.div_ceil(2) as usize,
                ),
                2 => (
                    self.frame.width.div_ceil(2) as usize,
                    self.frame.height as usize,
                ),
                3 => (self.frame.width as usize, self.frame.height as usize),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn fork_worker(&self) -> Encoder<'a> {
        Encoder {
            display_width: self.display_width,
            display_height: self.display_height,
            cat: self.cat,
            qp_y: self.qp_y,
            qp_c: self.qp_c,
            bit_depth: self.bit_depth,
            effort: self.effort,
            effort_template: self.effort_template,
            src: self.src,
            frame: self.frame.clone(),
            mode_map: self.mode_map.clone(),
            mode_stride: self.mode_stride,
            ct_depth_map: self.ct_depth_map.clone(),
            ct_depth_stride: self.ct_depth_stride,
            tu_depth_map: self.tu_depth_map.clone(),
            tu_depth_stride: self.tu_depth_stride,
            single_scan_rdoq: self.single_scan_rdoq,
            deblock: self.deblock,
            scratch_residual: Vec::new(),
            scratch_coeffs: Vec::new(),
            scratch_transform_tmp: Vec::new(),
            scratch_src8: Vec::new(),
            scratch_pred: Vec::new(),
            scratch_pred8: Vec::new(),
            scratch_scored: Vec::new(),
            scratch_allangs: Vec::new(),
            analysis: self.analysis.clone(),
            analysis_cache: self.analysis_cache.empty_like(),
            cur_policy: self.cur_policy,
            cur_qp_y: self.cur_qp_y,
            cur_qp_c: self.cur_qp_c,
            luma_trial_quality_override: None,
            best2_chroma_gate: self.best2_chroma_gate,
            best2_chroma_protect: self.best2_chroma_protect,
            best2_luma_fastrd: self.best2_luma_fastrd,
            best2_rough_lambda: self.best2_rough_lambda,
            best2_rd_lambda: self.best2_rd_lambda,
            best2_rdoq2: self.best2_rdoq2,
            sign_data_hiding: self.sign_data_hiding,
            best2_wpp: self.best2_wpp,
            best_luma_leaf_screen: self.best_luma_leaf_screen,
            best_tu_neighbor_limit: self.best_tu_neighbor_limit,
            best_aq: self.best_aq,
            best2_tt_reuse: self.best2_tt_reuse,
            best2_cu_reuse: self.best2_cu_reuse,
            partnxn: self.partnxn,
            aq: AqState::inert(),
            prof: Profiler {
                on: self.prof.on,
                ..Default::default()
            },
            trace: crate::trace::SearchTrace::default(),
            stats: EncodeStats::default(),
        }
    }

    fn region_policy(&self, x0: u32, y0: u32, log2_size: u8) -> crate::preanalysis::SearchPolicy {
        self.analysis.policy_at(x0, y0, log2_size, self.effort)
    }

    fn region_policy_cached(
        &self,
        x0: u32,
        y0: u32,
        log2_size: u8,
    ) -> crate::preanalysis::SearchPolicy {
        self.cur_policy
            .unwrap_or_else(|| self.region_policy(x0, y0, log2_size))
    }

    fn block_desc(&self, x0: u32, y0: u32, log2_size: u8, c_idx: u8) -> BlockDesc {
        let region = self.analysis.region_class_at(x0, y0, log2_size);
        let importance_q8 = self.analysis.importance_at(x0, y0, log2_size);
        BlockDesc {
            x: x0,
            y: y0,
            log2_size,
            qp: self.search_qp(),
            region,
            importance_q8,
            component: crate::effort::ComponentKind::from_c_idx(c_idx),
            policy: self.region_policy_cached(x0, y0, log2_size),
        }
    }

    fn block_budget(&self, x0: u32, y0: u32, log2_size: u8, c_idx: u8) -> BlockSearchBudget {
        self.effort_template
            .resolve(self.block_desc(x0, y0, log2_size, c_idx))
    }

    fn record_luma_winner_rank(&mut self, rank: usize) {
        bump_rank(&mut self.stats.luma_winner_rank_counts, rank);
    }

    fn record_chroma_winner_rank(&mut self, rank: usize) {
        bump_rank(&mut self.stats.chroma_winner_rank_counts, rank);
    }

    fn record_tu_winner(&mut self, x0: u32, y0: u32, log2_size: u8, split: bool) {
        let idx = self.analysis.region_class_at(x0, y0, log2_size).index();
        if split {
            self.stats.tu_split_wins_by_region[idx] += 1;
        } else {
            self.stats.tu_leaf_wins_by_region[idx] += 1;
        }
        if let Some(prior_depth) = self.tu_neighbor_prior(x0, y0) {
            self.trace.note_tu_neighbor_prior(prior_depth, split);
        }
        self.store_tu_depth(x0, y0, log2_size, split);
    }

    fn record_cu_winner(&mut self, x0: u32, y0: u32, log2_size: u8, split: bool) {
        let idx = self.analysis.region_class_at(x0, y0, log2_size).index();
        if split {
            self.stats.cu_split_wins_by_region[idx] += 1;
        } else {
            self.stats.cu_leaf_wins_by_region[idx] += 1;
        }
    }

    fn record_close_call(&mut self, best: f64, runner_up: f64, margin: f64) {
        if Self::is_close_call(best, runner_up, margin) {
            self.stats.full_rd_close_calls += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_split_decision(
        &mut self,
        kind: crate::trace::DecisionKind,
        x0: u32,
        y0: u32,
        log2_size: u8,
        leaf_cost: f64,
        split_cost: f64,
        leaf_exact: u32,
        split_exact: u32,
    ) {
        if !self.trace.enabled {
            return;
        }
        self.trace.note_decision(
            kind,
            x0,
            y0,
            log2_size,
            true,
            &[
                crate::trace::CandRec::new(leaf_cost, leaf_exact),
                crate::trace::CandRec::new(split_cost, split_exact),
            ],
        );
    }

    fn tt_luma_approx_delta(&self, ctxs: &Contexts, tt: &Tt) -> i64 {
        match tt {
            Tt::Leaf(l) => {
                if l.luma.cbf {
                    let approx =
                        self.approx_residual_frac_bits(ctxs, &l.luma.levels, l.log2_size, 0);
                    approx as i64 - l.luma.frac_bits as i64
                } else {
                    0
                }
            }
            Tt::Split { kids, .. } => kids
                .iter()
                .map(|k| self.tt_luma_approx_delta(ctxs, k))
                .sum(),
        }
    }

    fn is_close_call(best: f64, runner_up: f64, margin: f64) -> bool {
        margin > 0.0
            && best.is_finite()
            && runner_up.is_finite()
            && runner_up <= best * (1.0 + margin)
    }

    fn decision_confidence(best: f64, runner_up: f64, margin: f64) -> DecisionConfidence {
        if Self::is_close_call(best, runner_up, margin) {
            DecisionConfidence::Close
        } else {
            DecisionConfidence::Clear
        }
    }

    fn cache_decision_confidence(confidence: DecisionConfidence) -> CacheDecisionConfidence {
        match confidence {
            DecisionConfidence::Clear => CacheDecisionConfidence::Clear,
            DecisionConfidence::Close => CacheDecisionConfidence::Close,
        }
    }

    fn cu_leaf_node(mut leaf: CuLeaf, confidence: DecisionConfidence) -> CuNode {
        leaf.confidence = confidence;
        CuNode::Leaf(leaf)
    }

    fn annotate_root_cu_confidence(node: &mut CuNode, confidence: DecisionConfidence) {
        if let CuNode::Leaf(leaf) = node {
            leaf.confidence = confidence;
        }
    }

    fn split_ctx_inc(&self, x0: u32, y0: u32, ct_depth: u8) -> usize {
        let mut inc = 0usize;
        if x0 > 0 {
            let d = self.ct_depth_at(x0 - 1, y0);
            if d != 0xFF && d > ct_depth {
                inc += 1;
            }
        }
        if y0 > 0 {
            let d = self.ct_depth_at(x0, y0 - 1);
            if d != 0xFF && d > ct_depth {
                inc += 1;
            }
        }
        inc
    }

    fn ct_depth_at(&self, x: u32, y: u32) -> u8 {
        let idx = (y / 8) as usize * self.ct_depth_stride + (x / 8) as usize;
        self.ct_depth_map.get(idx).copied().unwrap_or(0xFF)
    }

    fn set_ct_depth(&mut self, x0: u32, y0: u32, log2_size: u8, ct_depth: u8) {
        let n = (1u32 << log2_size).div_ceil(8);
        let sx = x0 / 8;
        let sy = y0 / 8;
        for dy in 0..n {
            for dx in 0..n {
                let idx = (sy + dy) as usize * self.ct_depth_stride + (sx + dx) as usize;
                if idx < self.ct_depth_map.len() {
                    self.ct_depth_map[idx] = ct_depth;
                }
            }
        }
    }

    fn store_tu_depth(&mut self, x0: u32, y0: u32, log2_size: u8, split: bool) {
        let depth = u8::from(split);
        let n = (1u32 << log2_size) / 4;
        let sx = x0 / 4;
        let sy = y0 / 4;
        for dy in 0..n {
            for dx in 0..n {
                let idx = (sy + dy) as usize * self.tu_depth_stride + (sx + dx) as usize;
                if idx < self.tu_depth_map.len() {
                    self.tu_depth_map[idx] = depth;
                }
            }
        }
    }

    fn tu_depth_at(&self, x: u32, y: u32) -> Option<u8> {
        let idx = (y / 4) as usize * self.tu_depth_stride + (x / 4) as usize;
        self.tu_depth_map
            .get(idx)
            .copied()
            .filter(|&depth| depth != 0xFF)
    }

    fn tu_neighbor_prior(&self, x0: u32, y0: u32) -> Option<u8> {
        let mut max_depth = None;
        if x0 > 0 {
            max_depth = self.tu_depth_at(x0 - 1, y0);
        }
        if y0 > 0 {
            if let Some(depth) = self.tu_depth_at(x0, y0 - 1) {
                max_depth = Some(max_depth.map_or(depth, |m| m.max(depth)));
            }
        }
        max_depth
    }

    fn should_limit_tu_to_neighbor_leaf(&self, x0: u32, y0: u32, log2_size: u8, leaf: &Tt) -> bool {
        if !self.best_tu_neighbor_limit || log2_size < 4 {
            return false;
        }
        if self.tu_neighbor_prior(x0, y0) != Some(0) {
            return false;
        }
        if tt_has_residual(leaf) {
            return false;
        }
        matches!(
            self.analysis.region_class_at(x0, y0, log2_size),
            crate::preanalysis::RegionClass::Flat | crate::preanalysis::RegionClass::Gradient
        )
    }

    fn record_tu_neighbor_leaf_skip(&mut self, x0: u32, y0: u32, log2_size: u8) {
        self.stats.tu_neighbor_leaf_skips += 1;
        self.trace.note_tu_neighbor_leaf_skip(x0, y0, log2_size);
    }

    fn record_analysis_cache_cu_node(
        &mut self,
        node: &CuNode,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) {
        match node {
            CuNode::Leaf(leaf) => {
                self.analysis_cache.record_cu_region(
                    x0,
                    y0,
                    log2_cb_size,
                    ct_depth,
                    Self::cache_decision_confidence(leaf.confidence),
                );
                self.analysis_cache.record_leaf_modes(
                    x0,
                    y0,
                    log2_cb_size,
                    leaf.luma_mode,
                    leaf.chroma_mode_idx,
                );
                self.stats.analysis_cache_cu_records += 1;
                self.stats.analysis_cache_leaf_records += 1;
                self.record_analysis_cache_tt(&leaf.tt, x0, y0);
            }
            CuNode::Split { kids } => {
                let half = 1u32 << (log2_cb_size - 1);
                let x1 = x0 + half;
                let y1 = y0 + half;
                let log2_kid = log2_cb_size - 1;
                let mut kids = kids.iter();
                let kid = kids.next().expect("split CU has first child");
                self.record_analysis_cache_cu_node(kid, x0, y0, log2_kid, ct_depth + 1);
                if x1 < self.display_width {
                    let kid = kids.next().expect("split CU has right child");
                    self.record_analysis_cache_cu_node(kid, x1, y0, log2_kid, ct_depth + 1);
                }
                if y1 < self.display_height {
                    let kid = kids.next().expect("split CU has bottom child");
                    self.record_analysis_cache_cu_node(kid, x0, y1, log2_kid, ct_depth + 1);
                }
                if x1 < self.display_width && y1 < self.display_height {
                    let kid = kids.next().expect("split CU has bottom-right child");
                    self.record_analysis_cache_cu_node(kid, x1, y1, log2_kid, ct_depth + 1);
                }
            }
        }
    }

    fn record_analysis_cache_tt(&mut self, tt: &Tt, x0: u32, y0: u32) {
        match tt {
            Tt::Leaf(leaf) => {
                self.analysis_cache
                    .record_tu_region(x0, y0, leaf.log2_size, leaf.trafo_depth);
                self.stats.analysis_cache_tu_records += 1;
            }
            Tt::Split {
                log2_size, kids, ..
            } => {
                let half = 1u32 << (log2_size - 1);
                self.record_analysis_cache_tt(&kids[0], x0, y0);
                self.record_analysis_cache_tt(&kids[1], x0 + half, y0);
                self.record_analysis_cache_tt(&kids[2], x0, y0 + half);
                self.record_analysis_cache_tt(&kids[3], x0 + half, y0 + half);
            }
        }
    }

    fn store_mode(&mut self, x0: u32, y0: u32, log2_size: u8, mode: u8) {
        let n = (1u32 << log2_size) / 4;
        let sx = x0 / 4;
        let sy = y0 / 4;
        for dy in 0..n {
            for dx in 0..n {
                let idx = (sy + dy) as usize * self.mode_stride + (sx + dx) as usize;
                if idx < self.mode_map.len() {
                    self.mode_map[idx] = mode;
                }
            }
        }
    }

    fn mode_at(&self, x: u32, y: u32) -> u8 {
        let idx = (y / 4) as usize * self.mode_stride + (x / 4) as usize;
        self.mode_map.get(idx).copied().unwrap_or(1)
    }

    fn neighbor_left_mode(&self, x0: u32, y0: u32) -> IntraPredMode {
        if x0 == 0 {
            return IntraPredMode::Dc;
        }
        IntraPredMode::from_u8(self.mode_at(x0 - 1, y0)).unwrap_or(IntraPredMode::Dc)
    }

    fn neighbor_above_mode(&self, x0: u32, y0: u32) -> IntraPredMode {
        if y0 == 0 {
            return IntraPredMode::Dc;
        }
        let ctb = 1u32 << CTB_LOG2;
        let ctb_y_start = (y0 / ctb) * ctb;
        if y0 - 1 < ctb_y_start {
            return IntraPredMode::Dc;
        }
        IntraPredMode::from_u8(self.mode_at(x0, y0 - 1)).unwrap_or(IntraPredMode::Dc)
    }

    fn source_block(&self, c_idx: u8, x0: u32, y0: u32, size: usize) -> Vec<u16> {
        let mut block = vec![0u16; size * size];
        for j in 0..size {
            for i in 0..size {
                block[j * size + i] = self.src_sample(c_idx, x0 + i as u32, y0 + j as u32);
            }
        }
        block
    }

    fn source_luma_range(&self, x0: u32, y0: u32, size: usize) -> u16 {
        let mut lo = u16::MAX;
        let mut hi = 0u16;
        for j in 0..size {
            for i in 0..size {
                let v = self.src_sample(0, x0 + i as u32, y0 + j as u32);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        hi.saturating_sub(lo)
    }

    fn search_qp(&self) -> i32 {
        (self.cur_qp_y - self.aq.qp_bd_offset).clamp(0, 51)
    }
}

pub fn encode_with_stats(
    config: &StillHevcConfig,
    src: Source<'_>,
) -> (Vec<u8>, DecodedFrame, EncodeStats) {
    assert!(
        matches!(config.bit_depth, 8 | 10 | 12),
        "supported bit depths: 8, 10, 12"
    );
    let cat = chroma_array_type(config.chroma);
    let width = config.width;
    let height = config.height;
    let min_cb = 8;
    let coded_width = width.div_ceil(min_cb) * min_cb;
    let coded_height = height.div_ceil(min_cb) * min_cb;
    let bd = config.bit_depth;
    let qp_bd_offset = 6 * (bd as i32 - 8);
    let slice_qp_y = config.qp as i32;
    let qp_y = slice_qp_y + qp_bd_offset;
    let qpi_c = slice_qp_y.clamp(-qp_bd_offset, 57);
    let qp_c = chroma_qp_from_luma(qpi_c) + qp_bd_offset;

    let mode_stride = width.div_ceil(4) as usize;
    let mode_height = height.div_ceil(4) as usize;
    let ct_depth_stride = width.div_ceil(8) as usize;
    let ct_depth_height = height.div_ceil(8) as usize;
    let deblock = config.deblock == DeblockMode::On;
    let aq_active = crate::aq_active(config);
    let mut frame = DecodedFrame::with_params(coded_width, coded_height, bd, cat);
    if deblock {
        frame.qp_map.fill(qp_y as i8);
    }
    // Parallel `Best` (default-on): reuse the Placebo frozen-slice CTU wavefront
    // for the Best search budget, ~3x faster than serial. Costs the inherent
    // frozen-vs-running CABAC drift (~0.1 dB / ~0.4% size on the photo test-set);
    // accepted as Best's default operating point. `BPG_BEST2_PARALLEL=0` reverts
    // to the serial running-context path (the higher-efficiency comparison
    // anchor). Determinism oracle (`BPG_ENC_THREADS=1`) holds.
    // The frozen-slice-init WPP path was built for uniform-QP `Best`; its
    // per-worker frozen context does not carry the per-QG QP-prediction state
    // across worker boundaries, so it is incompatible with adaptive QP. When the
    // experimental variance AQ opts `Best` in, fall back to the serial path
    // (same rationale as `Placebo` + `--sao`).
    let parallel_enabled = std::env::var("BPG_BEST2_PARALLEL")
        .ok()
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    // The whole non-reference ladder now runs the CTU-wavefront-parallel path
    // (each tier is a pruned `Best`, all uniform-QP), so wall-clock scales
    // monotonically with effort. Gated off when AQ is opted back in
    // (`BPG_LADDER_AQ`/`BPG_BEST_AQ`), since the frozen-slice worker context does
    // not carry per-QG QP prediction across worker boundaries. `Placebo` keeps
    // its own static parallel (frozen, no WPP) reference template; `Reference`
    // stays serial.
    let best2_parallel = !crate::is_reference_tier(config.effort) && !aq_active && parallel_enabled;
    let mut effort_template = if config.effort == Effort::Best && best2_parallel {
        crate::effort::BEST_PARALLEL
    } else {
        *crate::effort::template(config.effort)
    };
    if best2_parallel {
        effort_template.parallel_analysis = true;
    }
    // WPP-style analysis context propagation for the parallel ladder: price each
    // CTU against a context propagated along the wavefront (raster-within-row,
    // seeded from the row above's 2nd CTU) instead of one cold frozen context.
    // Recovers essentially all of the parallel frozen-vs-running drift. Never
    // enabled for `Placebo` (keeps it byte-identical to its frozen reference).
    let best2_wpp = best2_parallel;
    let best2_luma_fastrd = config.effort == Effort::Best
        && !aq_active
        && std::env::var_os("BPG_BEST2_LUMA_FASTRD").is_some();
    // x265-parity rough mode cost (SATD + lambda_sad*bits, ported from
    // `calcRdSADCost`/`m_lambda`). Default-on for `Best`; A/B verified
    // quality-neutral-to-positive on the photo test-set (PSNR Δ ∈ [−0.004,
    // +0.020] dB, size Δ ≤ +0.19%). `BPG_BEST2_ROUGH_LAMBDA=0` reverts to the
    // legacy SSE-domain rough cost for comparison.
    let best2_rough_lambda = config.effort == Effort::Best
        && std::env::var("BPG_BEST2_ROUGH_LAMBDA")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
    // x265-parity SSE-domain RD/RDOQ lambda. The legacy HM intra factor (0.57)
    // under-penalizes bits ~16-24% vs `x265_lambda2_tab` at the same QP, pushing
    // still265 to a higher-rate/higher-quality operating point than x265 placebo
    // at equal QP. Default-on for `Best` (the x265-parity comparison tier);
    // `BPG_BEST2_RD_LAMBDA=0` reverts to the legacy formula for A/B.
    let best2_rd_lambda = config.effort == Effort::Best
        && std::env::var("BPG_BEST2_RD_LAMBDA")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
    // x265-parity RDOQ level-2 rate model (greater1/greater2 context tracking +
    // adapted Rice in `rdoq_single_scan`). Default-on for `Best`
    // (`BPG_BEST2_RDOQ2=0` reverts to the frozen-context single-scan model).
    let best2_rdoq2 = config.effort == Effort::Best
        && std::env::var("BPG_BEST2_RDOQ2")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
    // AQ-gated: under the experimental variance AQ, `Best` falls back to the
    // same coding path the lower AQ tiers use. The leaf-screen / TU-neighbour /
    // TT-reuse fast paths bypass `final_code_tt`'s per-CU QP bookkeeping, which
    // the decoder's QG-QP prediction mirror depends on, so they must stay off
    // when QP varies per quantization group.
    let best_luma_leaf_screen = config.effort == Effort::Best && !aq_active;
    let best_tu_neighbor_limit = config.effort == Effort::Best && !aq_active;
    // Phase 2 chroma rough-margin gate: validated quality-neutral and promoted
    // into default `Best` (see docs/intra-search-ledger-findings.md "Best2
    // experiment 1"). Margin 0.10 dominated 0.20/0.40 (bigger call cut, same
    // quality). `BPG_BEST2_CHROMA_GATE` still overrides for any non-reference
    // tier (A/B testing / other-tier experiments).
    const BEST_CHROMA_GATE_MARGIN: f64 = 0.10;
    let analysis_cache = AnalysisCache::new(
        width,
        height,
        bd,
        config.chroma,
        config.effort,
        config.qp,
        source_hash(width, height, bd, config.chroma, src),
    );
    let needs_analysis = aq_active
        || best_tu_neighbor_limit
        || effort_template.preanalysis.class_steering
        || effort_template.preanalysis.importance_force_leaf
        || effort_template
            .preanalysis
            .importance_rmd_prune_factor
            .is_some();
    let analysis = std::sync::Arc::new(if needs_analysis {
        crate::preanalysis::analyze(width, height, bd, cat, src)
    } else {
        crate::preanalysis::AnalysisMaps::empty()
    });
    let trace = {
        let ctb = 1u32 << CTB_LOG2;
        let trace_analysis = if needs_analysis {
            Some(analysis.clone())
        } else {
            Some(std::sync::Arc::new(crate::preanalysis::analyze(
                width, height, bd, cat, src,
            )))
        };
        crate::trace::SearchTrace::from_env(
            CTB_LOG2,
            width,
            height,
            width.div_ceil(ctb),
            height.div_ceil(ctb),
            trace_analysis,
            &format!("{:?}", config.effort),
            slice_qp_y,
            &format!("{:?}", config.chroma),
        )
    };
    let mut state = Encoder {
        display_width: width,
        display_height: height,
        cat,
        qp_y,
        qp_c,
        bit_depth: bd,
        effort: config.effort,
        effort_template,
        src,
        frame,
        mode_map: vec![1u8; mode_stride * mode_height],
        mode_stride,
        ct_depth_map: vec![0xFF; ct_depth_stride * ct_depth_height],
        ct_depth_stride,
        tu_depth_map: vec![0xFF; mode_stride * mode_height],
        tu_depth_stride: mode_stride,
        single_scan_rdoq: crate::effort::select_rdoq_single_scan(config.effort),
        deblock,
        scratch_residual: Vec::new(),
        scratch_coeffs: Vec::new(),
        scratch_transform_tmp: Vec::new(),
        scratch_src8: Vec::new(),
        scratch_pred: Vec::new(),
        scratch_pred8: Vec::new(),
        scratch_scored: Vec::new(),
        scratch_allangs: Vec::new(),
        analysis: analysis.clone(),
        analysis_cache,
        cur_policy: None,
        cur_qp_y: qp_y,
        cur_qp_c: qp_c,
        luma_trial_quality_override: None,
        best2_chroma_gate: if crate::is_reference_tier(config.effort) {
            None
        } else {
            std::env::var("BPG_BEST2_CHROMA_GATE")
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .filter(|&m| m > 0.0)
                .or((config.effort == Effort::Best).then_some(BEST_CHROMA_GATE_MARGIN))
        },
        // Advisor gap #9: protect ChromaCritical regions from the chroma gate.
        // Neutral on photos (ChromaCritical is rare there), protective on
        // chroma-detailed content. Default-on; `BPG_BEST2_CHROMA_PROTECT=0`
        // disables (kept toggleable per the neutral-fix integration rule).
        best2_chroma_protect: std::env::var("BPG_BEST2_CHROMA_PROTECT")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        best2_luma_fastrd,
        best2_rough_lambda,
        best2_rd_lambda,
        best2_rdoq2,
        sign_data_hiding: crate::sdh_active(config),
        best2_wpp,
        best_luma_leaf_screen,
        best_tu_neighbor_limit,
        best_aq: if config.effort == Effort::Best {
            crate::best_aq_params()
        } else {
            None
        },
        best2_tt_reuse: config.effort == Effort::Best && !aq_active && !best2_luma_fastrd,
        best2_cu_reuse: config.effort == Effort::Best && !aq_active && !best2_luma_fastrd,
        // PartNxN is a real coding tool, so it belongs on every quality-leaning
        // tier (each prunes the per-PU search via its own RD budget) — Best down
        // to Fast. Off for Fastest (speed floor) and the exact reference tiers.
        partnxn: std::env::var("BPG_PARTNXN").map(|v| v != "0").unwrap_or(matches!(
            config.effort,
            Effort::Best | Effort::Good | Effort::Balanced | Effort::Fast
        )),
        prof: Profiler {
            on: std::env::var_os("BPG_PROFILE").is_some(),
            ..Default::default()
        },
        trace,
        aq: AqState {
            active: aq_active,
            qp_bd_offset,
            slice_qp_y,
            current_qpy: slice_qp_y,
            last_qpy_prev_qg: slice_qp_y,
            qg_x: -1,
            qg_y: -1,
            coded: false,
            cu_qp_delta: 0,
            pred: slice_qp_y,
            cu_target: slice_qp_y,
            target_qg: (-1, -1),
            target: slice_qp_y,
        },
        stats: EncodeStats::default(),
    };

    let ctb = 1u32 << CTB_LOG2;

    let slice_data = if config.sao == SaoMode::On {
        // Production SAO (docs/sao.md): build every CTU tree once (reconstructing
        // the frame), deblock, decide SAO on the deblocked frame, then *replay*
        // the cached trees with SAO syntax — no second RDO pass. Parallel tiers
        // build via the wavefront; serial tiers build interleaved (the throwaway
        // bitstream only evolves the running context their RD pricing needs).
        let trees = if state.effort_template.parallel_analysis {
            build_slice_trees_parallel(&mut state, slice_qp_y)
        } else {
            build_slice_trees_serial(&mut state, slice_qp_y)
        };
        if state.deblock {
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
        }
        let sao_map = state.decide_sao_map(ctb);
        let bytes = write_slice_from_trees(&mut state, &trees, Some(&sao_map), slice_qp_y);
        // The frame is already deblocked; SAO is the final in-loop step.
        apply_sao(&mut state.frame, &sao_map, ctb);
        bytes
    } else {
        let bytes = if state.effort_template.parallel_analysis {
            encode_slice_data_parallel(&mut state, slice_qp_y)
        } else {
            encode_slice_data(&mut state, None, slice_qp_y)
        };
        if state.deblock {
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
        }
        bytes
    };

    state.stats.region_class_counts = analysis.class_counts();

    if state.trace.enabled {
        let total = state.stats.residual_bit_estimates;
        if let Err(e) = state.trace.write_reports(total) {
            eprintln!("search-trace: failed to write reports: {e}");
        }
    }

    let mut payload = slice::write_slice_segment_header(config);
    payload.extend_from_slice(&slice_data);

    let mut out = Vec::new();
    nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
    nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
    nal::write_annexb_nal(&mut out, nal::NalType::Pps, &params::write_pps(config));
    nal::write_annexb_nal(&mut out, nal::NalType::IdrWRadl, &payload);

    if state.prof.on {
        let p = &state.prof;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "profile (serial-path inner timers, ms): snapshot+restore={:.1}  \
             code_block={:.1} (of which rdoq={:.1}, exact_residual_bits={:.1})",
            ms(p.snapshot),
            ms(p.code_block),
            ms(p.rdoq),
            ms(p.residual_bits),
        );
    }

    (out, state.frame, state.stats)
}

pub fn encode(config: &StillHevcConfig, src: Source<'_>) -> (Vec<u8>, DecodedFrame) {
    let (bytes, recon, _) = encode_with_stats(config, src);
    (bytes, recon)
}
