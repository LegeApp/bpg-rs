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

use std::sync::Arc;

use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::sao::{SaoMap, apply_sao};
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::{DeblockMode, SaoMode, StillHevcConfig, nal, params, slice};

use self::aq::AqState;
use self::syntax::CuNode;
use self::types::*;
pub use self::types::{EncodeStats, Source};
use self::write::{build_slice_trees_serial, encode_slice_data, write_slice_from_trees};

pub(super) struct Encoder<'a> {
    display_width: u32,
    display_height: u32,
    cat: u8,
    bit_depth: u8,
    tile_grid: bpg_hevc_decode::hevc::tile::TileGrid,
    src: Source<'a>,
    frame: DecodedFrame,
    mode_map: Vec<u8>,
    mode_stride: usize,
    ct_depth_map: Vec<u8>,
    ct_depth_stride: usize,
    tu_depth_map: Vec<u8>,
    tu_depth_stride: usize,
    deblock: bool,
    sign_data_hiding: bool,
    aq: AqState,
    cur_qp_y: i32,
    cur_qp_c: i32,
    best_aq: Option<(bool, f32)>,
    analysis: Arc<crate::preanalysis::AnalysisMaps>,
    stats: EncodeStats,
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

    fn decide_sao_map(&self, ctb_size: u32) -> SaoMap {
        let width_ctbs = self.display_width.div_ceil(ctb_size);
        let height_ctbs = self.display_height.div_ceil(ctb_size);
        SaoMap::new(width_ctbs, height_ctbs)
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

    pub(in crate::encoder) fn predict_intra_tiled(
        &mut self,
        x: u32,
        y: u32,
        log2_size: u8,
        mode: IntraPredMode,
        c_idx: u8,
    ) {
        if let Some((tx0, ty0, tx1, ty1)) = self.tile_clamp_bounds(x, y, c_idx) {
            bpg_hevc_decode::hevc::intra::predict_intra_with_reader(
                &mut self.frame,
                x,
                y,
                log2_size,
                mode,
                c_idx,
                true,
                move |_c, rx, ry| {
                    if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                        Some(bpg_hevc_decode::hevc::UNINIT_SAMPLE)
                    } else {
                        None
                    }
                },
            );
        } else {
            bpg_hevc_decode::hevc::intra::predict_intra(
                &mut self.frame,
                x,
                y,
                log2_size,
                mode,
                c_idx,
                true,
            );
        }
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

    pub(in crate::encoder) fn store_tu_depth(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        split: bool,
    ) {
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

    let analysis = if aq_active || config.effort != crate::Effort::Best {
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
        tu_depth_map: vec![0xFF; mode_stride * mode_height],
        tu_depth_stride: mode_stride,
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
        analysis,
        stats: EncodeStats::default(),
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
