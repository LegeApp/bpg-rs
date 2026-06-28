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
pub(crate) fn overlay_probe_counts() -> (u64, u64) {
    stillsearch::overlay_probe_counts()
}

use std::sync::Arc;

use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::sao::{SaoMap, apply_sao};
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::{DeblockMode, SaoMode, StillHevcConfig, nal, params, sao, slice};

use self::aq::AqState;
use self::syntax::CuNode;
use self::types::*;
pub use self::types::{EncodeStats, Source};
use self::write::{build_slice_trees_serial, encode_slice_data, write_slice_from_trees};

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
    pub(super) best_aq: Option<(bool, f32)>,
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
    let encode_start = std::time::Instant::now();
    if !matches!(config.bit_depth, 8 | 10 | 12) || config.width == 0 || config.height == 0 {
        let bd = if matches!(config.bit_depth, 8 | 10 | 12) {
            config.bit_depth
        } else {
            8
        };
        let frame = DecodedFrame::with_params(config.width.max(1), config.height.max(1), bd, 1);
        return (Vec::new(), frame, EncodeStats::default());
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
            qg_x: -1,
            qg_y: -1,
            coded: false,
            cu_qp_delta: 0,
            pred: slice_qp_y,
            cu_target: slice_qp_y,
            target_qg: (-1, -1),
            target: slice_qp_y,
        },
        cur_qp_y: qp_y,
        cur_qp_c: qp_c,
        best_aq: if config.effort == crate::Effort::Best {
            crate::best_aq_params()
        } else {
            None
        },
        part_nxn_enabled: eff_t.nxn.enabled,
        analysis,
        stats: EncodeStats::default(),
        effort_template: eff_t,
    };

    let ctb = 1u32 << CTB_LOG2;
    let phase_start = std::time::Instant::now();
    let (slice_data, entry_sizes) = if config.sao == SaoMode::On || !state.tile_grid.is_single() {
        let trees = build_slice_trees_serial(&mut state, slice_qp_y);
        state.stats.phase_build_us += phase_start.elapsed().as_micros() as u64;
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
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
            write_slice_from_trees(&mut state, &trees, sao_map.as_ref(), slice_qp_y);
        state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
        if let Some(map) = &sao_map {
            let phase_start = std::time::Instant::now();
            apply_sao(&mut state.frame, map, ctb);
            state.stats.phase_sao_apply_us += phase_start.elapsed().as_micros() as u64;
        }
        (bytes, entries)
    } else {
        let phase_start = std::time::Instant::now();
        let bytes = encode_slice_data(&mut state, None, slice_qp_y);
        state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
            state.stats.phase_deblock_us += phase_start.elapsed().as_micros() as u64;
        }
        (bytes, Vec::new())
    };

    state.stats.region_class_counts = state.analysis.class_counts();

    let mut payload = slice::write_slice_segment_header(config, tiles, &entry_sizes);
    payload.extend_from_slice(&slice_data);

    let mut out = Vec::new();
    nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
    nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
    nal::write_annexb_nal(
        &mut out,
        nal::NalType::Pps,
        &params::write_pps(config, tiles),
    );
    nal::write_annexb_nal(&mut out, nal::NalType::IdrWRadl, &payload);

    state.stats.phase_total_us = encode_start.elapsed().as_micros() as u64;
    (out, state.frame, state.stats)
}

pub fn encode(config: &StillHevcConfig, src: Source<'_>) -> (Vec<u8>, DecodedFrame) {
    let (bytes, recon, _) = encode_with_stats(config, src);
    (bytes, recon)
}
