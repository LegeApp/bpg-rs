//! Still-image HEVC intra encoder facade.
//!
//! Search and split decisions live under [`stillsearch`]. This module keeps the
//! public encode API, final writer state, and unavoidable syntax/availability
//! helpers small while the StillSearch v2 implementation grows behind it.

mod aq;
mod stillsearch;
mod syntax;
mod types;
mod write;

/// Temporary overlay-probe accessor (feature `overlay-probe`): returns
/// (sample_calls, patch_iters) from the CTU recon overlay scans.
#[cfg(feature = "overlay-probe")]
pub(crate) fn overlay_probe_counts() -> (u64, u64, u64) {
    stillsearch::overlay_probe_counts()
}

use std::sync::Arc;

use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::sao::{SaoInfo, SaoMap, apply_sao_threads};
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::{DeblockMode, SaoMode, StillHevcConfig, nal, params, sao, slice};

use self::aq::AqState;
use self::syntax::CuNode;
use self::types::*;
pub use self::types::{EncodeStats, Source};
use self::write::{
    build_slice_trees_parallel, build_slice_trees_serial, encode_slice_data, write_slice_from_trees,
};

pub(super) struct Encoder<'a> {
    pub(super) display_width: u32,
    pub(super) display_height: u32,
    pub(super) cat: u8,
    pub(super) bit_depth: u8,
    pub(super) tile_grid: bpg_hevc_decode::hevc::tile::TileGrid,
    pub(super) src: Source<'a>,
    pub(super) frame: DecodedFrame,
    pub(super) mode_map: Vec<u8>,
    pub(super) mode_stride: usize,
    pub(super) ct_depth_map: Vec<u8>,
    pub(super) ct_depth_stride: usize,
    pub(super) deblock: bool,
    pub(super) sign_data_hiding: bool,
    pub(super) aq: AqState,
    pub(super) cur_qp_y: i32,
    pub(super) cur_qp_c: i32,
    /// Resolved adaptive-quantization strategy + strength + clamp for this
    /// encode (see [`crate::resolve_aq`]). [`crate::AqMode::Off`] means uniform
    /// QP. Ignored when [`Self::aq_offset_map`] is present (the external-map
    /// experiment path takes priority).
    pub(super) aq_mode: crate::AqMode,
    pub(super) aq_strength: f32,
    pub(super) aq_clamp: f32,
    pub(super) aq_offset_map: Option<Arc<crate::preanalysis::AqOffsetMap>>,
    pub(super) part_nxn_enabled: bool,
    pub(super) analysis: Arc<crate::preanalysis::AnalysisMaps>,
    pub(super) stats: EncodeStats,
    pub(super) effort_template: crate::effort::EffortTemplate,
}

impl<'a> Encoder<'a> {
    fn src_chroma_dims(&self) -> (u32, u32) {
        match self.cat {
            0 => (0, 0),
            1 => (
                self.display_width.div_ceil(2),
                self.display_height.div_ceil(2),
            ),
            2 => (self.display_width.div_ceil(2), self.display_height),
            3 => (self.display_width, self.display_height),
            _ => (0, 0),
        }
    }

    fn src_chroma_stride(&self) -> usize {
        self.src_chroma_dims().0 as usize
    }

    pub(in crate::encoder) fn src_sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        let (plane, stride, w, h) = match c_idx {
            0 => (
                self.src.y,
                self.display_width as usize,
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
        if plane.is_empty() || w == 0 || h == 0 || stride == 0 {
            return 1u16 << self.bit_depth.saturating_sub(1);
        }
        let sx = x.min(w.saturating_sub(1));
        let sy = y.min(h.saturating_sub(1));
        plane
            .get(sy as usize * stride + sx as usize)
            .copied()
            .unwrap_or(1u16 << self.bit_depth.saturating_sub(1))
    }

    /// Source plane view `(samples, stride, width, height)` for SAO statistics,
    /// mirroring [`Self::src_sample`]'s component-to-plane mapping.
    fn src_plane(&self, c_idx: u8) -> (&[u16], usize, u32, u32) {
        match c_idx {
            0 => (
                self.src.y,
                self.display_width as usize,
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

    /// Accumulate Edge-Offset statistics for one component and `eo_class` over a
    /// CTB region of the deblocked reconstruction vs the source. Categories with
    /// no edge (index 2) are dropped, matching the decoder's offset application.
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

        let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);

        // Use dispatched SIMD kernels for interior regions where all neighbours
        // are within the plane and source bounds. Plane-edge / source-padding
        // CTBs fall through to the generic scalar loop.
        let interior = x_end > x_start && y_end > y_start && x_end <= sw && y_end <= sh;

        if interior {
            match eo_class {
                0 if x_start >= 1 && x_end + 1 <= plane_w => {
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
                1 if y_start >= 1 && y_end + 1 <= plane_h && x_end <= plane_w => {
                    let mut stats = sao::EoStats::default();
                    crate::primitives::sao_stats_e1(
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
                2 if x_start >= 1
                    && y_start >= 1
                    && x_end + 1 <= plane_w
                    && y_end + 1 <= plane_h =>
                {
                    let mut stats = sao::EoStats::default();
                    crate::primitives::sao_stats_e2(
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
                3 if x_end + 1 <= plane_w
                    && y_start >= 1
                    && x_start >= 1
                    && y_end + 1 <= plane_h =>
                {
                    let mut stats = sao::EoStats::default();
                    crate::primitives::sao_stats_e3(
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
                _ => {}
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

    /// Accumulate Band-Offset statistics for one component over a CTB region.
    fn sao_bo_stats(
        &self,
        c_idx: u8,
        x_start: u32,
        y_start: u32,
        x_end: u32,
        y_end: u32,
    ) -> sao::BoStats {
        let (plane, stride) = self.frame.plane(c_idx);
        let (plane_w, plane_h) = self.frame.component_dims(c_idx);
        let band_shift = self.bit_depth.saturating_sub(5) as u8;
        let mut stats = sao::BoStats::default();

        let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);
        if x_end <= plane_w && y_end <= plane_h && x_end <= sw && y_end <= sh {
            crate::primitives::sao_stats_bo(
                plane,
                stride,
                src_plane,
                src_stride,
                x_start,
                y_start,
                x_end - x_start,
                y_end - y_start,
                band_shift,
                &mut stats.sum,
                &mut stats.count,
            );
            return stats;
        }

        let band_shift = band_shift as u32;
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

    /// Pick the best SAO type for one component (off / band / edge) by distortion
    /// reduction. Returns `(type_idx, eo_class, offsets, band_position, reduction)`;
    /// `reduction == 0` means SAO-off wins.
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

    /// Decide SAO parameters for every CTB. Each CTB's decision reads only
    /// the committed reconstruction + source (`&self`) and writes only its own
    /// map cell, so rows are distributed round-robin over scoped threads —
    /// byte-identical to the serial scan.
    fn decide_sao_map(&self, ctb_size: u32) -> SaoMap {
        let ctbs_x = self.display_width.div_ceil(ctb_size);
        let ctbs_y = self.display_height.div_ceil(ctb_size);
        let mut map = SaoMap::new(ctbs_x, ctbs_y);

        let workers = write::parallel_thread_count().min(ctbs_y as usize).max(1);
        if workers <= 1 {
            for (ctb_y, row) in map.data.chunks_mut(ctbs_x as usize).enumerate() {
                self.decide_sao_row(ctb_size, ctb_y as u32, row);
            }
            return map;
        }
        let mut worker_rows: Vec<Vec<(u32, &mut [SaoInfo])>> =
            (0..workers).map(|_| Vec::new()).collect();
        for (ctb_y, row) in map.data.chunks_mut(ctbs_x as usize).enumerate() {
            worker_rows[ctb_y % workers].push((ctb_y as u32, row));
        }
        std::thread::scope(|scope| {
            for rows in worker_rows {
                scope.spawn(move || {
                    for (ctb_y, row) in rows {
                        self.decide_sao_row(ctb_size, ctb_y, row);
                    }
                });
            }
        });
        map
    }

    fn decide_sao_row(&self, ctb_size: u32, ctb_y: u32, row: &mut [SaoInfo]) {
        let (sub_x, sub_y) = self.frame.chroma_subsampling();

        let (luma_w, luma_h) = self.frame.component_dims(0);
        let (chroma_w, chroma_h) = if self.cat == 0 {
            (0, 0)
        } else {
            self.frame.component_dims(1)
        };

        {
            for (ctb_x, info) in row.iter_mut().enumerate() {
                let ctb_x = ctb_x as u32;
                let x0 = ctb_x * ctb_size;
                let y0 = ctb_y * ctb_size;

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
    }

    pub(in crate::encoder) fn same_tile_px(&self, ax: u32, ay: u32, bx: u32, by: u32) -> bool {
        self.tile_grid.same_tile_px(ax, ay, bx, by, CTB_LOG2)
    }

    pub(in crate::encoder) fn plane_shifts(&self, c_idx: u8) -> (u8, u8) {
        bpg_hevc_decode::hevc::tile::plane_shifts(c_idx, self.cat)
    }

    pub(in crate::encoder) fn tile_clamp_bounds(
        &self,
        x: u32,
        y: u32,
        c_idx: u8,
    ) -> Option<(u32, u32, u32, u32)> {
        if self.tile_grid.is_single() {
            return None;
        }
        let (sx, sy) = self.plane_shifts(c_idx);
        Some(self.tile_grid.tile_plane_bounds(x, y, CTB_LOG2, sx, sy))
    }

    pub(in crate::encoder) fn split_ctx_inc(&self, x0: u32, y0: u32, ct_depth: u8) -> usize {
        let mut inc = 0usize;
        if x0 > 0 && self.same_tile_px(x0 - 1, y0, x0, y0) {
            let d = self.ct_depth_at(x0 - 1, y0);
            if d != 0xFF && d > ct_depth {
                inc += 1;
            }
        }
        if y0 > 0 && self.same_tile_px(x0, y0 - 1, x0, y0) {
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

    pub(in crate::encoder) fn set_ct_depth(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ct_depth: u8,
    ) {
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

    pub(in crate::encoder) fn store_mode(&mut self, x0: u32, y0: u32, log2_size: u8, mode: u8) {
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

    pub(in crate::encoder) fn neighbor_left_mode(&self, x0: u32, y0: u32) -> IntraPredMode {
        if x0 == 0 || !self.same_tile_px(x0 - 1, y0, x0, y0) {
            return IntraPredMode::Dc;
        }
        IntraPredMode::from_u8(self.mode_at(x0 - 1, y0)).unwrap_or(IntraPredMode::Dc)
    }

    pub(in crate::encoder) fn neighbor_above_mode(&self, x0: u32, y0: u32) -> IntraPredMode {
        if y0 == 0 || !self.same_tile_px(x0, y0 - 1, x0, y0) {
            return IntraPredMode::Dc;
        }
        let ctb = 1u32 << CTB_LOG2;
        let ctb_y_start = (y0 / ctb) * ctb;
        if y0 - 1 < ctb_y_start {
            return IntraPredMode::Dc;
        }
        IntraPredMode::from_u8(self.mode_at(x0, y0 - 1)).unwrap_or(IntraPredMode::Dc)
    }

    pub(in crate::encoder) fn record_analysis_cache_cu_node(
        &mut self,
        node: &CuNode,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        _ct_depth: u8,
    ) {
        match node {
            CuNode::Leaf(_) => {
                let idx = self.analysis.region_class_at(x0, y0, log2_cb_size).index();
                self.stats.cu_leaf_wins_by_region[idx] += 1;
            }
            CuNode::Split { kids } => {
                let idx = self.analysis.region_class_at(x0, y0, log2_cb_size).index();
                self.stats.cu_split_wins_by_region[idx] += 1;
                let half = 1u32 << (log2_cb_size - 1);
                let x1 = x0 + half;
                let y1 = y0 + half;
                let mut kids = kids.iter();
                if let Some(k) = kids.next() {
                    self.record_analysis_cache_cu_node(k, x0, y0, log2_cb_size - 1, 0);
                }
                if x1 < self.display_width {
                    if let Some(k) = kids.next() {
                        self.record_analysis_cache_cu_node(k, x1, y0, log2_cb_size - 1, 0);
                    }
                }
                if y1 < self.display_height {
                    if let Some(k) = kids.next() {
                        self.record_analysis_cache_cu_node(k, x0, y1, log2_cb_size - 1, 0);
                    }
                }
                if x1 < self.display_width && y1 < self.display_height {
                    if let Some(k) = kids.next() {
                        self.record_analysis_cache_cu_node(k, x1, y1, log2_cb_size - 1, 0);
                    }
                }
            }
        }
    }
}

pub fn encode_with_stats(
    config: &StillHevcConfig,
    src: Source<'_>,
) -> (Vec<u8>, DecodedFrame, EncodeStats) {
    // Two-pass AQ runs the encoder twice via its own driver; everything else is
    // a single pass.
    if crate::resolve_aq_mode(config).is_two_pass() && config.chroma != crate::ChromaFormat::Gray {
        return encode_two_pass(config, src);
    }
    let (bytes, frame, stats, _) = encode_inner(config, src, None, false);
    (bytes, frame, stats)
}

/// Pass-1 per-QG measurements for two-pass AQ. Each `Vec` is indexed
/// `cy * cells_x + cx` over the 32x32 quantization-group grid.
struct QgMeasurements {
    cells_x: u32,
    cells_y: u32,
    /// `Σ|level|` coded coefficient energy (luma + chroma).
    energy: Vec<f64>,
    /// Real coded bits (`frac_bits / SCALE`, luma + chroma).
    bits: Vec<f64>,
    /// Luma SSE between source and the final pass-1 reconstruction.
    sse: Vec<f64>,
}

/// Single encode pass. `aq_override` injects an in-memory per-QG offset map
/// (pass 2 of two-pass AQ); when `None`, the external `BPG_AQ_OFFSET_MAP` is
/// loaded if AQ is active. `collect_activity` measures per-QG coded energy +
/// bits (from the built CU trees) and luma SSE (source vs reconstruction) —
/// the pass-1 measurements that drive two-pass AQ.
fn encode_inner(
    config: &StillHevcConfig,
    src: Source<'_>,
    aq_override: Option<Arc<crate::preanalysis::AqOffsetMap>>,
    collect_activity: bool,
) -> (Vec<u8>, DecodedFrame, EncodeStats, Option<QgMeasurements>) {
    let encode_start = std::time::Instant::now();
    if !matches!(config.bit_depth, 8 | 10 | 12) || config.width == 0 || config.height == 0 {
        let bd = if matches!(config.bit_depth, 8 | 10 | 12) {
            config.bit_depth
        } else {
            8
        };
        let frame = DecodedFrame::with_params(config.width.max(1), config.height.max(1), bd, 1);
        return (Vec::new(), frame, EncodeStats::default(), None);
    }

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
    let qpi_c = (slice_qp_y + crate::chroma_qp_offset() as i32).clamp(-qp_bd_offset, 57);
    let qp_c = chroma_qp_from_luma(qpi_c, cat) + qp_bd_offset;

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

    let stream_wpp = params::effective_wpp_enabled(config);
    let build_wpp = params::effective_wpp_build_enabled(config);
    let tiles = params::effective_tile_dims(config);
    let tile_grid = {
        let ctb = 1u32 << CTB_LOG2;
        let ctbs_x = width.div_ceil(ctb);
        let ctbs_y = height.div_ceil(ctb);
        match tiles {
            Some((cols, rows)) => {
                bpg_hevc_decode::hevc::tile::TileGrid::uniform(ctbs_x, ctbs_y, cols, rows)
            }
            None => bpg_hevc_decode::hevc::tile::TileGrid::single(ctbs_x, ctbs_y),
        }
    };

    let eff_t = crate::effort::template_for_encode(config.effort);
    let aq_offset_map = if aq_override.is_some() {
        aq_override
    } else if aq_active {
        crate::preanalysis::load_external_aq_offset_map(width, height).map(Arc::new)
    } else {
        None
    };
    let (aq_mode, aq_strength, aq_clamp) = crate::resolve_aq(config);

    let analysis = if aq_active || !eff_t.oracle {
        Arc::new(crate::preanalysis::analyze(width, height, bd, cat, src))
    } else {
        Arc::new(crate::preanalysis::AnalysisMaps::empty())
    };

    let mut state = Encoder {
        display_width: width,
        display_height: height,
        cat,
        bit_depth: bd,
        tile_grid,
        src,
        frame,
        mode_map: vec![1u8; mode_stride * mode_height],
        mode_stride,
        ct_depth_map: vec![0xFF; ct_depth_stride * ct_depth_height],
        ct_depth_stride,
        deblock,
        sign_data_hiding: crate::sdh_active(config),
        aq: AqState {
            active: aq_active,
            qp_bd_offset,
            slice_qp_y,
            current_qpy: slice_qp_y,
            last_qpy_prev_qg: slice_qp_y,
            qpy_pred: slice_qp_y,
            first_qp_group: true,
            qg_x: -1,
            qg_y: -1,
            coded: false,
            cu_qp_delta: 0,
            pred: slice_qp_y,
            cu_target: slice_qp_y,
            cu_x: 0,
            cu_y: 0,
            cu_log2_size: 0,
            target_qg: (-1, -1),
            target: slice_qp_y,
        },
        cur_qp_y: qp_y,
        cur_qp_c: qp_c,
        aq_mode,
        aq_strength,
        aq_clamp,
        aq_offset_map,
        part_nxn_enabled: eff_t.nxn.enabled,
        analysis,
        stats: EncodeStats::default(),
        effort_template: eff_t,
    };

    let ctb = 1u32 << CTB_LOG2;
    let phase_start = std::time::Instant::now();
    // Two-pass AQ measures per-QG coded energy from the built CU trees, so it
    // needs the tree-building path (the direct `encode_slice_data` path retains
    // no tree to walk).
    let mut qg_energy_bits: Option<(u32, u32, Vec<f64>, Vec<f64>)> = None;
    let (slice_data, entry_sizes) = if config.sao == SaoMode::On
        || !state.tile_grid.is_single()
        || stream_wpp
        || build_wpp
        || collect_activity
    {
        let trees = if build_wpp {
            let allow_aq_build = state.aq.active;
            build_slice_trees_parallel(&mut state, slice_qp_y, allow_aq_build)
        } else {
            build_slice_trees_serial(&mut state, slice_qp_y)
        };
        state.stats.phase_build_us += phase_start.elapsed().as_micros() as u64;
        if collect_activity {
            qg_energy_bits = Some(collect_qg_activity(
                &trees,
                state.display_width,
                state.display_height,
            ));
        }
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter_threads(
                &mut state.frame,
                0,
                0,
                0,
                0,
                write::parallel_thread_count(),
            );
            state.stats.phase_deblock_us += phase_start.elapsed().as_micros() as u64;
        }
        let sao_map = if config.sao == SaoMode::On {
            let phase_start = std::time::Instant::now();
            let map = state.decide_sao_map(ctb);
            state.stats.phase_sao_decide_us += phase_start.elapsed().as_micros() as u64;
            Some(map)
        } else {
            None
        };
        let phase_start = std::time::Instant::now();
        let (bytes, entries) =
            write_slice_from_trees(&mut state, &trees, sao_map.as_ref(), slice_qp_y, stream_wpp);
        state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
        if let Some(map) = &sao_map {
            let phase_start = std::time::Instant::now();
            apply_sao_threads(&mut state.frame, map, ctb, write::parallel_thread_count());
            state.stats.phase_sao_apply_us += phase_start.elapsed().as_micros() as u64;
        }
        (bytes, entries)
    } else {
        let phase_start = std::time::Instant::now();
        let bytes = encode_slice_data(&mut state, None, slice_qp_y);
        state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter_threads(
                &mut state.frame,
                0,
                0,
                0,
                0,
                write::parallel_thread_count(),
            );
            state.stats.phase_deblock_us += phase_start.elapsed().as_micros() as u64;
        }
        (bytes, Vec::new())
    };

    state.stats.region_class_counts = state.analysis.class_counts();

    // Two-pass AQ pass-1: now that the reconstruction is final (deblocked +
    // SAO'd), measure per-QG luma distortion and bundle the full measurement.
    let activity_out = qg_energy_bits.map(|(cx, cy, energy, bits)| {
        let sse = collect_qg_sse(
            state.src.y,
            &state.frame,
            state.display_width,
            state.display_height,
            cx,
            cy,
        );
        QgMeasurements {
            cells_x: cx,
            cells_y: cy,
            energy,
            bits,
            sse,
        }
    });

    let mut payload = slice::write_slice_segment_header(config, tiles, stream_wpp, &entry_sizes);
    payload.extend_from_slice(&slice_data);

    let mut out = Vec::new();
    nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
    nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
    nal::write_annexb_nal(
        &mut out,
        nal::NalType::Pps,
        &params::write_pps(config, tiles, stream_wpp),
    );
    nal::write_annexb_nal(&mut out, nal::NalType::IdrWRadl, &payload);

    state.stats.phase_total_us = encode_start.elapsed().as_micros() as u64;
    (out, state.frame, state.stats, activity_out)
}

/// Two-pass measured AQ ([`crate::AqMode::TwoPassMeasured`]). Pass 1 encodes at
/// uniform QP and measures per-QG coded coefficient energy (an actual
/// CTU-search outcome). Pass 2 re-encodes with a perceptual QP redistribution
/// derived from that measurement.
///
/// **Candidate-compare gate** ([`StillHevcConfig::two_pass_gate`], default on):
/// AQ helps some images and hurts others, and source statistics can't predict
/// which (project history). So the encoder keeps the AQ (pass 2) candidate only
/// when it is actually a perceptual rate-distortion win over the uniform (pass
/// 1) candidate — measured directly with luma SSIM and coded size — and
/// otherwise returns the uniform result. This makes two-pass AQ a safe default:
/// it can improve a picture but never regress one.
pub fn encode_two_pass(
    config: &StillHevcConfig,
    src: Source<'_>,
) -> (Vec<u8>, DecodedFrame, EncodeStats) {
    // Pass 1: uniform QP, collecting per-QG coded energy, bits, and SSE.
    let mut pass1 = config.clone();
    pass1.aq_mode = crate::AqMode::Off;
    pass1.adaptive_qp = false;
    let (bytes_u, frame_u, stats_u, measurements) = encode_inner(&pass1, src, None, true);

    let (_mode, strength, clamp) = crate::resolve_aq(config);
    let map = measurements.and_then(|m| {
        let activity = two_pass_activity(&m, config.qp as i32);
        // Skip when the picture has no usable spread (uniformly flat/busy) so
        // two-pass never regresses — pass 2 is then a plain uniform encode.
        let (lo, hi) = activity
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        if !(hi > lo) {
            return None;
        }
        Some(Arc::new(crate::preanalysis::AqOffsetMap::from_qg_activity(
            m.cells_x,
            m.cells_y,
            &activity,
            strength,
            clamp as u8,
        )))
    });

    // No usable map → the uniform pass 1 is already the answer.
    let Some(map) = map else {
        return (bytes_u, frame_u, stats_u);
    };

    // Pass 2: apply the measured AQ map.
    let (bytes_a, frame_a, stats_a, _) = encode_inner(config, src, Some(map), false);

    if !two_pass_gate_enabled(config) {
        return (bytes_a, frame_a, stats_a);
    }

    // Candidate-compare gate (3-pass equal-byte test). Pass 3 encodes uniform at
    // a neighbour QP so the AQ bitrate is bracketed by two uniform points; the
    // uniform quality is interpolated at the AQ byte count and AQ is kept only if
    // it beats that — i.e. AQ is genuinely above the uniform RD curve at equal
    // bytes (the same comparison BD-rate makes). Quality is a blended luma
    // SSIM+PSNR score (`blended_quality_db`), so AQ must be a net perceptual AND
    // fidelity win, not just an SSIM win at the expense of PSNR. No tuning
    // constant beyond the perceptual/fidelity weight.
    let bd = config.bit_depth;
    let ssim_w = std::env::var("BPG_2PASS_GATE_SSIM_W")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(0.5);
    let quality = |frame: &DecodedFrame| -> f64 {
        let (ssim, mse) = luma_fidelity(src.y, frame, config.width, config.height, bd);
        blended_quality_db(ssim, mse, bd, ssim_w)
    };
    let q_u = quality(&frame_u);
    let q_a = quality(&frame_a);
    // AQ usually saves bytes (raises effective QP), so the bracketing uniform
    // point is at the next-higher QP; if AQ grew, bracket with the lower QP.
    let q = config.qp as i32;
    let q3 = if bytes_a.len() <= bytes_u.len() {
        (q + 1).min(51)
    } else {
        (q - 1).max(0)
    };
    // Rate value: how much AQ's byte savings are allowed to offset a sub-curve
    // blended-quality loss. The 3-pass measures the uniform curve's own
    // quality-shed when it drops from `bytes_u` to AQ's `bytes_a`
    // (`q_u − uniform_q_at_aq_rate`); we credit AQ a `(rate_value−1)` fraction of
    // that as tolerance, so substantial savings excuse a minor loss while a large
    // loss still reverts. `rate_value = 1` is the strict equal-byte test.
    let rate_value = std::env::var("BPG_2PASS_GATE_RATE_VALUE")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 1.0)
        .unwrap_or(2.0);
    let keep_aq = if q3 == q {
        q_a >= q_u
    } else {
        let mut pass3 = pass1.clone();
        pass3.qp = q3 as u8;
        let (bytes_u3, frame_u3, _, _) = encode_inner(&pass3, src, None, false);
        let q_u3 = quality(&frame_u3);
        let uniform_q_at_aq_rate =
            interp_log_rate(bytes_u.len(), q_u, bytes_u3.len(), q_u3, bytes_a.len());
        // Quality uniform sheds to reach AQ's (smaller) byte count, i.e. the
        // value of AQ's savings in blended-quality dB; 0 if AQ didn't save.
        let savings_value = (q_u - uniform_q_at_aq_rate).max(0.0);
        let credit = (rate_value - 1.0) * savings_value;
        let keep = q_a >= uniform_q_at_aq_rate - credit;
        if std::env::var("BPG_2PASS_DEBUG").is_ok_and(|v| v.trim() != "0") {
            eprintln!(
                "2PASS-GATE w={ssim_w} rv={rate_value} q_a={q_a:.4} uniform@aq_rate={uniform_q_at_aq_rate:.4} credit={credit:.4} thresh={:.4} (q_u={q_u:.4}@{} q_u3={q_u3:.4}@{} bytes_a={}) -> {}",
                uniform_q_at_aq_rate - credit,
                bytes_u.len(),
                bytes_u3.len(),
                bytes_a.len(),
                if keep { "KEEP-AQ" } else { "REVERT-UNIFORM" }
            );
        }
        keep
    };

    if keep_aq {
        (bytes_a, frame_a, stats_a)
    } else {
        (bytes_u, frame_u, stats_u)
    }
}

/// Whether the two-pass candidate-compare gate is active. `config` default is on
/// (`true`); `BPG_2PASS_GATE=0` force-disables it for A/B (always keep AQ).
fn two_pass_gate_enabled(config: &StillHevcConfig) -> bool {
    if let Ok(v) = std::env::var("BPG_2PASS_GATE") {
        return v.trim() != "0";
    }
    config.two_pass_gate
}

/// Interpolate a quality value at `target_bytes` over **log-rate** between two
/// measured (bytes, quality) points — the rate axis BD-rate uses. Clamps to the
/// endpoints when `target` falls outside the bracket (AQ saved/grew more than a
/// QP step).
fn interp_log_rate(bytes0: usize, q0: f64, bytes1: usize, q1: f64, target_bytes: usize) -> f64 {
    let (l0, l1, lt) = (
        (bytes0.max(1) as f64).ln(),
        (bytes1.max(1) as f64).ln(),
        (target_bytes.max(1) as f64).ln(),
    );
    if (l1 - l0).abs() < f64::EPSILON {
        return (q0 + q1) * 0.5;
    }
    let t = ((lt - l0) / (l1 - l0)).clamp(0.0, 1.0);
    q0 + t * (q1 - q0)
}

/// Luma fidelity of a reconstruction vs the source for the two-pass gate:
/// `(mean 8x8-block SSIM, mean squared error)`, computed in one pass. SSIM is
/// Wang et al. 2004 (perceptual, what the AQ targets); MSE is the fidelity term
/// (PSNR). The gate blends both because SSIM and PSNR can disagree — AQ
/// systematically trades PSNR for SSIM, so an SSIM-only gate can keep an encode
/// whose fidelity collapsed.
fn luma_fidelity(
    src_y: &[u16],
    recon: &DecodedFrame,
    disp_w: u32,
    disp_h: u32,
    bit_depth: u8,
) -> (f64, f64) {
    let l = ((1u32 << bit_depth) - 1) as f64;
    let c1 = (0.01 * l).powi(2);
    let c2 = (0.03 * l).powi(2);
    let src_stride = disp_w as usize;
    let rec_stride = recon.width as usize;
    let mut ssim_acc = 0.0f64;
    let mut blocks = 0u32;
    let mut sse = 0.0f64;
    let mut total = 0.0f64;
    let mut by = 0u32;
    while by < disp_h {
        let mut bx = 0u32;
        while bx < disp_w {
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy, mut n) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for y in by..(by + 8).min(disp_h) {
                let sr = y as usize * src_stride;
                let rr = y as usize * rec_stride;
                for x in bx..(bx + 8).min(disp_w) {
                    let a = src_y[sr + x as usize] as f64;
                    let b = recon.y_plane[rr + x as usize] as f64;
                    sx += a;
                    sy += b;
                    sxx += a * a;
                    syy += b * b;
                    sxy += a * b;
                    let d = a - b;
                    sse += d * d;
                    n += 1.0;
                }
            }
            if n > 0.0 {
                let mx = sx / n;
                let my = sy / n;
                let vx = (sxx / n - mx * mx).max(0.0);
                let vy = (syy / n - my * my).max(0.0);
                let cxy = sxy / n - mx * my;
                ssim_acc += ((2.0 * mx * my + c1) * (2.0 * cxy + c2))
                    / ((mx * mx + my * my + c1) * (vx + vy + c2));
                blocks += 1;
                total += n;
            }
            bx += 8;
        }
        by += 8;
    }
    let ssim = if blocks == 0 {
        1.0
    } else {
        ssim_acc / blocks as f64
    };
    let mse = if total == 0.0 { 0.0 } else { sse / total };
    (ssim, mse)
}

/// Blended perceptual/fidelity quality score in dB for the two-pass gate:
/// `w·SSIM_dB + (1−w)·PSNR_dB`. SSIM is mapped to dB (`−10·log10(1−SSIM)`) so it
/// shares the PSNR scale and the two terms are comparable. `w`
/// (`BPG_2PASS_GATE_SSIM_W`, default 0.5) trades perceptual vs fidelity: 1.0 =
/// SSIM-only, 0.0 = PSNR-only. Higher SSIM and higher PSNR both raise the score.
fn blended_quality_db(ssim: f64, mse: f64, bit_depth: u8, ssim_w: f64) -> f64 {
    let peak = ((1u32 << bit_depth) - 1) as f64;
    let ssim_db = -10.0 * (1.0 - ssim).max(1e-6).log10();
    let psnr_db = if mse <= 1e-9 {
        // Lossless block: cap at a high but finite PSNR so the blend stays sane.
        99.0
    } else {
        10.0 * (peak * peak / mse).log10()
    };
    ssim_w * ssim_db + (1.0 - ssim_w) * psnr_db
}

pub fn encode(config: &StillHevcConfig, src: Source<'_>) -> (Vec<u8>, DecodedFrame) {
    let (bytes, recon, _) = encode_with_stats(config, src);
    (bytes, recon)
}

/// Form the per-QG two-pass activity signal from pass-1 measurements. The
/// `from_qg_activity` map then raises QP where this is high (complex / maskable)
/// and lowers it where low (flat / banding-prone), around the picture mean.
///
/// Default is `energy`: `Σ|level|` coded coefficient energy per QG — i.e. how
/// much residual detail the encoder actually coded there. A 7-image A/B (vs the
/// rate-distortion-slope signal `D + λR`, pure distortion `sse`, and coded
/// `bits`) found **energy wins decisively** — more MS-SSIM gain and more
/// per-image wins (5/7) — because it most cleanly isolates maskable coded
/// texture. The RD-cost / distortion signals dilute that with terms that behave
/// oppositely in textured regions, so they were measured and rejected (kept
/// behind the env selector for the record). `BPG_2PASS_SIGNAL` selects
/// `energy` (default) / `rd` / `bits` / `sse` / `bits_per_sse`.
fn two_pass_activity(m: &QgMeasurements, slice_qp: i32) -> Vec<f64> {
    let signal = std::env::var("BPG_2PASS_SIGNAL")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    // HEVC luma RD lambda in pixel-SSE units (HM/x265 `0.57 * 2^((qp-12)/3)`).
    let lambda = 0.57f64 * 2f64.powf((slice_qp as f64 - 12.0) / 3.0);
    let n = m.energy.len();
    (0..n)
        .map(|i| match signal.as_str() {
            // Measured Lagrangian RD cost D + λR (rejected — see above).
            "rd" => m.sse[i] + lambda * m.bits[i],
            "bits" => m.bits[i],
            "sse" => m.sse[i],
            // Bits spent per unit residual distortion remaining.
            "bits_per_sse" => m.bits[i] / (1.0 + m.sse[i]),
            // Default: coded coefficient energy.
            _ => m.energy[i],
        })
        .collect()
}

/// Measure per-QG (32x32) coded coefficient energy from the built CU trees —
/// the two-pass AQ pass-1 signal. Energy is `Σ|level|` over every transform
/// block (luma + chroma), attributed to the quantization group containing each
/// transform unit's top-left. Mirrors [`write::write_cu`]'s position recursion.
fn collect_qg_activity(
    trees: &[Option<CuNode>],
    display_width: u32,
    display_height: u32,
) -> (u32, u32, Vec<f64>, Vec<f64>) {
    use self::syntax::{CodedBlock, Tt};
    const QG: u32 = 1 << QG_LOG2;
    let cells_x = display_width.div_ceil(QG).max(1);
    let cells_y = display_height.div_ceil(QG).max(1);
    let mut energy = vec![0.0f64; (cells_x * cells_y) as usize];
    let mut bits = vec![0.0f64; (cells_x * cells_y) as usize];

    // Σ|level| and real coded bits (`frac_bits / SCALE`) over a leaf TU's TBs.
    let leaf_stats = |l: &self::syntax::LeafTu| -> (f64, f64) {
        let blocks = [&l.luma, &l.cb, &l.cr, &l.cb1, &l.cr1];
        let e: i64 = blocks
            .iter()
            .flat_map(|b: &&CodedBlock| b.levels.iter())
            .map(|&c| (c as i64).abs())
            .sum();
        let fb: u64 = blocks.iter().map(|b| b.frac_bits).sum();
        (
            e as f64,
            fb as f64 / crate::cabac::CabacEstimator::SCALE as f64,
        )
    };

    // Recurse a transform tree, attributing each leaf TU to its QG.
    fn tt_walk(
        tt: &Tt,
        x: u32,
        y: u32,
        energy: &mut [f64],
        bits: &mut [f64],
        add: &impl Fn(&mut [f64], &mut [f64], u32, u32, f64, f64),
        leaf_stats: &impl Fn(&self::syntax::LeafTu) -> (f64, f64),
    ) {
        match tt {
            Tt::Leaf(l) => {
                let (e, b) = leaf_stats(l);
                add(energy, bits, x, y, e, b);
            }
            Tt::Split {
                log2_size, kids, ..
            } => {
                let half = 1u32 << (log2_size - 1);
                for (i, kid) in kids.iter().enumerate() {
                    let kx = x + (i as u32 & 1) * half;
                    let ky = y + (i as u32 >> 1) * half;
                    tt_walk(kid, kx, ky, energy, bits, add, leaf_stats);
                }
            }
        }
    }

    fn cu_walk(
        node: &CuNode,
        x: u32,
        y: u32,
        log2_cb: u8,
        energy: &mut [f64],
        bits: &mut [f64],
        add: &impl Fn(&mut [f64], &mut [f64], u32, u32, f64, f64),
        leaf_stats: &impl Fn(&self::syntax::LeafTu) -> (f64, f64),
    ) {
        match node {
            CuNode::Leaf(leaf) => tt_walk(&leaf.tt, x, y, energy, bits, add, leaf_stats),
            CuNode::Split { kids } => {
                let half = 1u32 << (log2_cb - 1);
                for (i, kid) in kids.iter().enumerate() {
                    let kx = x + (i as u32 & 1) * half;
                    let ky = y + (i as u32 >> 1) * half;
                    cu_walk(kid, kx, ky, log2_cb - 1, energy, bits, add, leaf_stats);
                }
            }
        }
    }

    let add = |energy: &mut [f64], bits: &mut [f64], x: u32, y: u32, e: f64, b: f64| {
        if x < display_width && y < display_height {
            let cx = (x / QG).min(cells_x - 1);
            let cy = (y / QG).min(cells_y - 1);
            let idx = (cy * cells_x + cx) as usize;
            energy[idx] += e;
            bits[idx] += b;
        }
    };

    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = display_width.div_ceil(ctb);
    for (idx, tree) in trees.iter().enumerate() {
        if let Some(node) = tree {
            let cx = (idx as u32 % ctbs_x) * ctb;
            let cy = (idx as u32 / ctbs_x) * ctb;
            cu_walk(
                node,
                cx,
                cy,
                CTB_LOG2,
                &mut energy,
                &mut bits,
                &add,
                &leaf_stats,
            );
        }
    }
    (cells_x, cells_y, energy, bits)
}

/// Per-QG luma SSE between the source and the (final, deblocked + SAO'd) pass-1
/// reconstruction — the distortion half of the two-pass RD-slope signal.
fn collect_qg_sse(
    src_y: &[u16],
    recon: &DecodedFrame,
    display_width: u32,
    display_height: u32,
    cells_x: u32,
    cells_y: u32,
) -> Vec<f64> {
    const QG: u32 = 1 << QG_LOG2;
    let mut sse = vec![0.0f64; (cells_x * cells_y) as usize];
    let src_stride = display_width as usize;
    let rec_stride = recon.width as usize;
    for y in 0..display_height {
        let cy = (y / QG).min(cells_y - 1);
        let srow = y as usize * src_stride;
        let rrow = y as usize * rec_stride;
        for x in 0..display_width {
            let cx = (x / QG).min(cells_x - 1);
            let d = src_y[srow + x as usize] as i64 - recon.y_plane[rrow + x as usize] as i64;
            sse[(cy * cells_x + cx) as usize] += (d * d) as f64;
        }
    }
    sse
}
