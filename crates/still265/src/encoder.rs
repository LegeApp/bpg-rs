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
//! fixed DM chroma, `sign_data_hiding = false`, no SAO/deblock. Full RDO CU/TU
//! splitting, 4:2:2, and filters come in later milestones.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::hevc::intra::{fill_mpm_candidates, predict_intra};
use bpg_hevc_decode::hevc::slice::IntraPredMode;
use bpg_hevc_decode::DecodedFrame;
use std::fmt;
use std::mem::size_of;

use crate::cabac::{CabacEncoder, CabacEstimator};
use crate::contexts::{ctx, Contexts};
use crate::primitives;
use crate::rdoq;
use crate::residual::{
    encode_residual, estimate_residual_bits, get_scan_order, ResidualEstimateCache,
};
use crate::{nal, params, slice, transform, ChromaFormat, Effort, StillHevcConfig};

/// `chroma_format_idc` / `ChromaArrayType` (H.265 Table 6-1), for the formats
/// this encoder supports.
fn chroma_array_type(chroma: ChromaFormat) -> u8 {
    match chroma {
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv444 => 3,
        _ => unimplemented!("still265: only 4:2:0 and 4:4:4 are supported"),
    }
}

/// Geometry of the chroma transform block(s) associated with a luma TB at
/// `(x0, y0, log2_size)`, or `None` if this leaf has no chroma TB (4:2:0,
/// `log2_size == 2`: chroma is coded by the parent 8x8 node instead).
///
/// - 4:2:0 (`cat == 1`): one half-size chroma TB at `(x0/2, y0/2)`, only when
///   `log2_size >= 3`.
/// - 4:4:4 (`cat == 3`): chroma TBs mirror luma 1:1 (same position and size),
///   at every leaf including `log2_size == 2` (H.265 7.3.8.8: chroma cbf and
///   transform_unit chroma blocks are present whenever `ChromaArrayType ==
///   3`, regardless of `log2_size`).
fn chroma_tb_geom(cat: u8, x0: u32, y0: u32, log2_size: u8) -> Option<(u32, u32, u8)> {
    match cat {
        1 if log2_size >= 3 => Some((x0 / 2, y0 / 2, log2_size - 1)),
        1 => None,
        3 => Some((x0, y0, log2_size)),
        _ => unreachable!(),
    }
}

/// Whether a leaf TB at `log2_size` carries its own chroma TB(s) (see
/// [`chroma_tb_geom`]).
fn has_chroma_tb(cat: u8, log2_size: u8) -> bool {
    cat == 3 || log2_size >= 3
}

const CTB_LOG2: u8 = 6;
const MAX_TB_LOG2: u8 = 5;
const MIN_TB_LOG2: u8 = 2;
const MAX_INTRA_TT_DEPTH: u8 = 2;
const CHROMA_DM_IDX: u8 = 4;

/// Number of SATD-shortlisted luma mode candidates that get a full
/// TU-split RD trial per CU, by encode effort. Mirrors x265 presets'
/// `rdLevel`/candidate-count tradeoff: faster presets commit to the best-SATD
/// mode immediately, slower presets spend extra full-RD trials on the
/// next-best SATD candidates.
/// Pick the RDOQ algorithm for an effort level.
///
/// `Best` keeps the exact greedy per-coefficient RDOQ as a slow, high-quality
/// reference preset (akin to x265's placebo); `Balanced` (and `Fast`, which
/// applies no RDOQ refinement anyway) use the ~30%-faster single-scan RDOQ,
/// which is RD-equivalent at archival bitrates. `BPG_RDOQ_SINGLESCAN=1`/`=0`
/// forces the choice (for benchmarking / A-B).
fn select_rdoq_single_scan(effort: Effort) -> bool {
    match std::env::var("BPG_RDOQ_SINGLESCAN").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => !matches!(effort, Effort::Best),
    }
}

fn luma_rd_candidates(effort: Effort) -> usize {
    match effort {
        Effort::Fast => 1,
        Effort::Balanced => 2,
        Effort::Best => 3,
    }
}

fn rough_luma_modes(effort: Effort, mpm: [IntraPredMode; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(35);
    for m in mpm {
        out.push(m.as_u8());
    }
    out.extend_from_slice(&[0, 1]);

    match effort {
        Effort::Fast => out.extend_from_slice(&[2, 6, 10, 14, 18, 22, 26, 30, 34]),
        Effort::Balanced => out.extend_from_slice(&[
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
        ]),
        Effort::Best => out.extend(0..=34),
    }

    out.sort_unstable();
    out.dedup();
    out
}

/// Number of SATD-shortlisted chroma mode candidates (out of the 5 legal
/// H.265 Table 8-2 choices: Planar, Vertical, Horizontal, DC, DM) that get a
/// full residual-coding RD trial per CU, by encode effort. `Effort::Fast`
/// keeps the original SATD-only decision (no extra transform/RDOQ work);
/// `Effort::Best` full-RD-evaluates all 5 candidates.
fn chroma_rd_candidates(effort: Effort) -> usize {
    match effort {
        Effort::Fast => 1,
        Effort::Balanced => 1,
        Effort::Best => 5,
    }
}

/// H.265 Table 8-22 chroma QP mapping for 4:2:0 (mirrors the decoder's
/// `chroma_qp_from_luma`).
fn chroma_qp_from_luma(qpi: i32) -> i32 {
    static TAB: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        TAB[(qpi - 30) as usize]
    }
}

struct CodedBlock {
    levels: Vec<i16>,
    cbf: bool,
    frac_bits: u64,
}

impl CodedBlock {
    fn empty() -> Self {
        Self {
            levels: Vec::new(),
            cbf: false,
            frac_bits: 0,
        }
    }
}

/// A reconstructed-and-recorded leaf transform unit's coded data.
struct LeafTu {
    log2_size: u8,
    /// `log2` size of the chroma TBs at this leaf, if any (see
    /// [`chroma_tb_geom`]); unused (left at 0) when [`has_chroma_tb`] is false.
    chroma_log2: u8,
    trafo_depth: u8,
    luma_mode: u8,
    chroma_mode: u8,
    luma: CodedBlock,
    cb: CodedBlock,
    cr: CodedBlock,
}

/// Chroma transform data carried by a split node in the H.265 special case
/// where subsampled chroma is coded at the parent after split luma children.
struct ParentChromaTu {
    log2_size: u8,
    chroma_mode: u8,
    cb: CodedBlock,
    cr: CodedBlock,
}

enum Tt {
    Split {
        log2_size: u8,
        trafo_depth: u8,
        cbf_cb: bool,
        cbf_cr: bool,
        parent_chroma: Option<ParentChromaTu>,
        kids: Vec<Tt>,
    },
    Leaf(LeafTu),
}

impl Tt {
    fn cbf_cb(&self) -> bool {
        match self {
            Tt::Split { cbf_cb, .. } => *cbf_cb,
            Tt::Leaf(l) => l.cb.cbf,
        }
    }
    fn cbf_cr(&self) -> bool {
        match self {
            Tt::Split { cbf_cr, .. } => *cbf_cr,
            Tt::Leaf(l) => l.cr.cbf,
        }
    }
}

/// A coding unit, fully coded (intra mode + transform tree).
struct CuLeaf {
    mpm: [IntraPredMode; 3],
    luma_mode: u8,
    chroma_mode_idx: u8,
    tt: Tt,
}

/// A coding quadtree node: either a coded leaf CU, or a `split_cu_flag`
/// split into (up to four, fewer at picture boundaries) child nodes.
enum CuNode {
    Split { kids: Vec<CuNode> },
    Leaf(CuLeaf),
}

/// Source planes (full-range YCbCr, 4:2:0 or 4:4:4 per [`StillHevcConfig::chroma`])
/// for the picture being encoded. For 4:2:0 `cb`/`cr` are half-resolution
/// (`width/2 x height/2`); for 4:4:4 they are full resolution (`width x height`),
/// same as `y`.
pub struct Source<'a> {
    pub y: &'a [u16],
    pub cb: &'a [u16],
    pub cr: &'a [u16],
}

#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    pub ctu_count: u64,
    pub cu_trials: u64,
    pub luma_rough_predictions: u64,
    pub chroma_rough_predictions: u64,
    pub code_block_calls: u64,
    pub forward_transforms: u64,
    pub inverse_transforms: u64,
    pub residual_bit_estimates: u64,
    pub cache_builds: u64,
    pub cache_fast_hits: u64,
    pub cache_fallbacks: u64,
    pub frame_snapshots: u64,
    pub frame_restores: u64,
    pub map_snapshots: u64,
    pub map_restores: u64,
    pub bytes_snapshotted: u64,
}

impl fmt::Display for EncodeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "encode stats:")?;
        writeln!(f, "  ctu_count: {}", self.ctu_count)?;
        writeln!(f, "  cu_trials: {}", self.cu_trials)?;
        writeln!(
            f,
            "  luma_rough_predictions: {}",
            self.luma_rough_predictions
        )?;
        writeln!(
            f,
            "  chroma_rough_predictions: {}",
            self.chroma_rough_predictions
        )?;
        writeln!(f, "  code_block_calls: {}", self.code_block_calls)?;
        writeln!(f, "  forward_transforms: {}", self.forward_transforms)?;
        writeln!(f, "  inverse_transforms: {}", self.inverse_transforms)?;
        writeln!(
            f,
            "  residual_bit_estimates: {}",
            self.residual_bit_estimates
        )?;
        writeln!(f, "  cache_builds: {}", self.cache_builds)?;
        writeln!(f, "  cache_fast_hits: {}", self.cache_fast_hits)?;
        writeln!(f, "  cache_fallbacks: {}", self.cache_fallbacks)?;
        writeln!(f, "  frame_snapshots: {}", self.frame_snapshots)?;
        writeln!(f, "  frame_restores: {}", self.frame_restores)?;
        writeln!(f, "  map_snapshots: {}", self.map_snapshots)?;
        writeln!(f, "  map_restores: {}", self.map_restores)?;
        writeln!(f, "  bytes_snapshotted: {}", self.bytes_snapshotted)
    }
}

struct Encoder<'a> {
    display_width: u32,
    display_height: u32,
    /// `ChromaArrayType` / `chroma_format_idc`: 1 (4:2:0) or 3 (4:4:4).
    cat: u8,
    qp_y: i32,
    qp_c: i32,
    bit_depth: u8,
    /// Controls how many SATD-shortlisted luma intra modes get a full
    /// TU-split RD trial per CU (see [`luma_rd_candidates`]).
    effort: Effort,
    src: Source<'a>,
    frame: DecodedFrame,
    /// Luma intra mode per 4x4 min-PU (for MPM neighbour lookup).
    mode_map: Vec<u8>,
    mode_stride: usize,
    /// CU depth per 8x8 min-CB (for `split_cu_flag` context derivation).
    ct_depth_map: Vec<u8>,
    ct_depth_stride: usize,
    /// Use single-scan RDOQ (`rdoq::rdoq_single_scan`) instead of the exact
    /// greedy refinement. Selected by [`select_rdoq_single_scan`]: on for
    /// `Balanced`, off for `Best` (which keeps the exact greedy RDOQ as the
    /// slow high-quality reference preset). `BPG_RDOQ_SINGLESCAN` overrides.
    single_scan_rdoq: bool,
    stats: EncodeStats,
}

struct PlaneSnapshot {
    c_idx: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    data: Vec<u16>,
}

struct FrameSnapshot {
    planes: Vec<PlaneSnapshot>,
}

#[derive(Clone)]
struct MapSnapshot {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<u8>,
}

fn snapshot_map(
    map: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> MapSnapshot {
    let rows = map.len().div_ceil(stride);
    let x = x.min(stride);
    let y = y.min(rows);
    let width = width.min(stride.saturating_sub(x));
    let height = height.min(rows.saturating_sub(y));
    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = (y + row) * stride + x;
        data.extend_from_slice(&map[start..start + width]);
    }
    MapSnapshot {
        x,
        y,
        width,
        height,
        stride,
        data,
    }
}

fn restore_map(map: &mut [u8], snapshot: &MapSnapshot) {
    for row in 0..snapshot.height {
        let src = row * snapshot.width;
        let dst = (snapshot.y + row) * snapshot.stride + snapshot.x;
        map[dst..dst + snapshot.width].copy_from_slice(&snapshot.data[src..src + snapshot.width]);
    }
}

impl<'a> Encoder<'a> {
    fn src_luma_stride(&self) -> usize {
        self.display_width as usize
    }

    fn src_chroma_dims(&self) -> (u32, u32) {
        match self.cat {
            1 => (
                self.display_width.div_ceil(2),
                self.display_height.div_ceil(2),
            ),
            3 => (self.display_width, self.display_height),
            _ => unreachable!(),
        }
    }

    fn src_chroma_stride(&self) -> usize {
        self.src_chroma_dims().0 as usize
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
            _ => {
                let (cw, ch) = self.src_chroma_dims();
                (self.src.cr, self.src_chroma_stride(), cw, ch)
            }
        };
        let sx = x.min(w.saturating_sub(1));
        let sy = y.min(h.saturating_sub(1));
        plane[sy as usize * stride + sx as usize]
    }

    fn frame_plane_dims(&self, c_idx: u8) -> (usize, usize) {
        match c_idx {
            0 => (self.frame.width as usize, self.frame.height as usize),
            1 | 2 => match self.cat {
                1 => (
                    self.frame.width.div_ceil(2) as usize,
                    self.frame.height.div_ceil(2) as usize,
                ),
                3 => (self.frame.width as usize, self.frame.height as usize),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn snapshot_plane(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: usize,
        height: usize,
    ) -> PlaneSnapshot {
        let (plane_width, plane_height) = self.frame_plane_dims(c_idx);
        let x = (x as usize).min(plane_width);
        let y = (y as usize).min(plane_height);
        let width = width.min(plane_width.saturating_sub(x));
        let height = height.min(plane_height.saturating_sub(y));
        let (plane, stride) = self.frame.plane(c_idx);
        let mut data = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = (y + row) * stride + x;
            data.extend_from_slice(&plane[start..start + width]);
        }
        let snapshot = PlaneSnapshot {
            c_idx,
            x,
            y,
            width,
            height,
            data,
        };
        self.stats.frame_snapshots += 1;
        self.stats.bytes_snapshotted += (snapshot.data.len() * size_of::<u16>()) as u64;
        snapshot
    }

    fn restore_plane(&mut self, snapshot: &PlaneSnapshot) {
        self.stats.frame_restores += 1;
        let (plane, stride) = self.frame.plane_mut(snapshot.c_idx);
        for row in 0..snapshot.height {
            let src = row * snapshot.width;
            let dst = (snapshot.y + row) * stride + snapshot.x;
            plane[dst..dst + snapshot.width]
                .copy_from_slice(&snapshot.data[src..src + snapshot.width]);
        }
    }

    fn snapshot_frame_region(&mut self, x0: u32, y0: u32, log2_size: u8) -> FrameSnapshot {
        let size = 1usize << log2_size;
        let mut planes = Vec::with_capacity(3);
        planes.push(self.snapshot_plane(0, x0, y0, size, size));
        if let Some((cx, cy, clog2)) = chroma_tb_geom(self.cat, x0, y0, log2_size) {
            let csize = 1usize << clog2;
            planes.push(self.snapshot_plane(1, cx, cy, csize, csize));
            planes.push(self.snapshot_plane(2, cx, cy, csize, csize));
        }
        FrameSnapshot { planes }
    }

    fn restore_frame_region(&mut self, snapshot: &FrameSnapshot) {
        for plane in &snapshot.planes {
            self.restore_plane(plane);
        }
    }

    fn snapshot_mode_region(&mut self, x0: u32, y0: u32, log2_size: u8) -> MapSnapshot {
        let x = (x0 / 4) as usize;
        let y = (y0 / 4) as usize;
        let width = ((1u32 << log2_size) / 4) as usize;
        let height = width;
        let snapshot = snapshot_map(&self.mode_map, self.mode_stride, x, y, width, height);
        self.stats.map_snapshots += 1;
        self.stats.bytes_snapshotted += snapshot.data.len() as u64;
        snapshot
    }

    fn restore_mode_region(&mut self, snapshot: &MapSnapshot) {
        self.stats.map_restores += 1;
        restore_map(&mut self.mode_map, snapshot);
    }

    fn snapshot_ct_depth_region(&mut self, x0: u32, y0: u32, log2_size: u8) -> MapSnapshot {
        let x = (x0 / 8) as usize;
        let y = (y0 / 8) as usize;
        let width = (1u32 << log2_size).div_ceil(8) as usize;
        let height = width;
        let snapshot = snapshot_map(
            &self.ct_depth_map,
            self.ct_depth_stride,
            x,
            y,
            width,
            height,
        );
        self.stats.map_snapshots += 1;
        self.stats.bytes_snapshotted += snapshot.data.len() as u64;
        snapshot
    }

    fn restore_ct_depth_region(&mut self, snapshot: &MapSnapshot) {
        self.stats.map_restores += 1;
        restore_map(&mut self.ct_depth_map, snapshot);
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

    fn predicted_block(&self, c_idx: u8, x0: u32, y0: u32, size: usize) -> Vec<u16> {
        let (plane, stride) = self.frame.plane(c_idx);
        let mut block = vec![0u16; size * size];
        for j in 0..size {
            for i in 0..size {
                block[j * size + i] = plane[(y0 as usize + j) * stride + x0 as usize + i];
            }
        }
        block
    }

    fn satd_block_cost(&self, src: &[u16], src8: Option<&[u8]>, pred: &[u16], size: usize) -> u32 {
        match self.bit_depth {
            8 => {
                let src8 = src8.expect("8-bit source scratch must be present");
                let pred8: Vec<u8> = pred.iter().map(|&v| v.min(255) as u8).collect();
                primitives::satd_u8(src8, size, &pred8, size, size)
            }
            10 => primitives::satd_u16(src, size, pred, size, size),
            _ => unreachable!("supported bit depths are checked at encode entry"),
        }
    }

    /// Shortlist luma intra mode candidates for a CU by x265-style rough SATD
    /// analysis: all 35 H.265 luma intra modes are predicted from
    /// reconstructed neighbours and scored against the source transform block
    /// (SATD plus the estimated `prev_intra_luma_pred`/`mpm_idx`/`rem_intra_
    /// luma_pred_mode` bits from the current CABAC context snapshot). The
    /// best [`luma_rd_candidates`] modes by that rough cost are returned, in
    /// ascending-cost order, for `build_cu_leaf` to evaluate with full
    /// TU-split RD. The predictions this writes are transient — `build_tt`
    /// re-predicts with whichever mode wins.
    fn choose_luma_mode_candidates(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ctxs: &Contexts,
    ) -> Vec<u8> {
        let size = 1usize << log2_size;
        let src = self.source_block(0, x0, y0, size);
        let src8 = (self.bit_depth == 8)
            .then(|| src.iter().map(|&v| v.min(255) as u8).collect::<Vec<u8>>());

        let cand_a = self.neighbor_left_mode(x0, y0);
        let cand_b = self.neighbor_above_mode(x0, y0);
        let mpm = fill_mpm_candidates(cand_a, cand_b);

        // These trial predictions write real sample values over not-yet-coded
        // neighbours that `build_tt`'s later RD trials still rely on being
        // `UNINIT_SAMPLE` (for intra reference-sample availability/
        // substitution). Snapshot and restore the frame so this analysis pass
        // leaves no residue.
        let base_frame = self.snapshot_plane(0, x0, y0, size, size);

        let mut scored: Vec<(u64, u8)> = Vec::with_capacity(35);
        for m in rough_luma_modes(self.effort, mpm) {
            self.stats.luma_rough_predictions += 1;
            self.restore_plane(&base_frame);
            let mode = IntraPredMode::from_u8(m).unwrap();
            predict_intra(&mut self.frame, x0, y0, log2_size, mode, 0, true);
            let pred = self.predicted_block(0, x0, y0, size);
            let satd = self.satd_block_cost(&src, src8.as_deref(), &pred, size) as u64;
            let mode_bits = estimate_intra_luma_mode_bits(ctxs, mpm, m);
            let cost = satd * CabacEstimator::SCALE + mode_bits;
            scored.push((cost, m));
        }
        self.restore_plane(&base_frame);

        scored.sort_by_key(|&(cost, _)| cost);
        let n = luma_rd_candidates(self.effort);
        scored.into_iter().take(n).map(|(_, m)| m).collect()
    }

    fn chroma_mode_from_idx(luma_mode: u8, mode_idx: u8) -> u8 {
        if mode_idx == CHROMA_DM_IDX {
            return luma_mode;
        }

        let candidate = match mode_idx {
            0 => IntraPredMode::Planar,
            1 => IntraPredMode::Angular26,
            2 => IntraPredMode::Angular10,
            3 => IntraPredMode::Dc,
            _ => unreachable!("invalid chroma mode index"),
        }
        .as_u8();

        if candidate == luma_mode {
            IntraPredMode::Angular34.as_u8()
        } else {
            candidate
        }
    }

    /// Choose the CU-level chroma intra prediction syntax mode from the legal
    /// H.265 Table 8-2 candidates: Planar, Vertical, Horizontal, DC, and DM.
    ///
    /// All 5 candidates are first scored by rough SATD + mode-signaling bits
    /// (as before). The best [`chroma_rd_candidates`] of those (by `effort`)
    /// then get a full RD trial: `code_block` actually transforms/quantizes/
    /// RDOQs the CU-level chroma TB(s) for each candidate mode, and the
    /// lowest distortion+bits cost wins. `Effort::Fast` shortlists a single
    /// candidate, so this degenerates to the original SATD-only decision.
    fn choose_chroma_mode(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> (u8, u8) {
        let Some((cx, cy, clog2)) = chroma_tb_geom(self.cat, x0, y0, log2_size) else {
            return (luma_mode, CHROMA_DM_IDX);
        };
        let size = 1usize << clog2;
        let cb_src = self.source_block(1, cx, cy, size);
        let cr_src = self.source_block(2, cx, cy, size);
        let cb_src8 = (self.bit_depth == 8).then(|| {
            cb_src
                .iter()
                .map(|&v| v.min(255) as u8)
                .collect::<Vec<u8>>()
        });
        let cr_src8 = (self.bit_depth == 8).then(|| {
            cr_src
                .iter()
                .map(|&v| v.min(255) as u8)
                .collect::<Vec<u8>>()
        });

        // See the matching comment in `choose_luma_mode`: snapshot/restore so
        // these trial predictions don't leave non-`UNINIT_SAMPLE` residue over
        // not-yet-coded neighbours.
        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);

        let mut scored: Vec<(u64, u8)> = Vec::with_capacity(5);
        for mode_idx in 0..=CHROMA_DM_IDX {
            self.stats.chroma_rough_predictions += 1;
            self.restore_frame_region(&base_frame);
            let mode = Self::chroma_mode_from_idx(luma_mode, mode_idx);
            let pred_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);

            predict_intra(&mut self.frame, cx, cy, clog2, pred_mode, 1, true);
            let cb_pred = self.predicted_block(1, cx, cy, size);
            let cb_cost = self.satd_block_cost(&cb_src, cb_src8.as_deref(), &cb_pred, size) as u64;

            predict_intra(&mut self.frame, cx, cy, clog2, pred_mode, 2, true);
            let cr_pred = self.predicted_block(2, cx, cy, size);
            let cr_cost = self.satd_block_cost(&cr_src, cr_src8.as_deref(), &cr_pred, size) as u64;

            let mode_bits = estimate_intra_chroma_mode_bits(ctxs, mode_idx);
            let cost = (cb_cost + cr_cost) * CabacEstimator::SCALE + mode_bits;
            scored.push((cost, mode_idx));
        }
        self.restore_frame_region(&base_frame);

        scored.sort_by_key(|&(cost, _)| cost);
        let n = chroma_rd_candidates(self.effort);
        if n <= 1 {
            let idx = scored[0].1;
            return (Self::chroma_mode_from_idx(luma_mode, idx), idx);
        }

        let mut best_idx = scored[0].1;
        let mut best_mode = Self::chroma_mode_from_idx(luma_mode, best_idx);
        let mut best_cost = f64::MAX;
        for &(_, mode_idx) in scored.iter().take(n) {
            let mode = Self::chroma_mode_from_idx(luma_mode, mode_idx);

            self.restore_frame_region(&base_frame);
            let cb = self.code_block(ctxs, cx, cy, clog2, 1, mode, self.qp_c);
            let cb_distortion = self.distortion_block(1, cx, cy, clog2);
            let cr = self.code_block(ctxs, cx, cy, clog2, 2, mode, self.qp_c);
            let cr_distortion = self.distortion_block(2, cx, cy, clog2);

            let mode_bits = estimate_intra_chroma_mode_bits(ctxs, mode_idx);
            let cost = self.rd_cost(
                cb_distortion + cr_distortion,
                cb.frac_bits + cr.frac_bits + mode_bits,
            );

            if cost < best_cost {
                best_cost = cost;
                best_idx = mode_idx;
                best_mode = mode;
            }
        }
        self.restore_frame_region(&base_frame);

        (best_mode, best_idx)
    }

    /// Predict + transform + quantize + reconstruct one transform block, in
    /// place in the frame; returns the quantized levels and the cbf.
    fn code_block(
        &mut self,
        ctxs: &Contexts,
        x: u32,
        y: u32,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        qp: i32,
    ) -> CodedBlock {
        self.stats.code_block_calls += 1;
        let size = 1usize << log2_size;
        let pred_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);

        // Prediction into the frame plane (reads reconstructed neighbours).
        predict_intra(&mut self.frame, x, y, log2_size, pred_mode, c_idx, true);

        let (plane, stride) = self.frame.plane(c_idx);
        let mut residual = vec![0i16; size * size];
        for j in 0..size {
            for i in 0..size {
                let pred = plane[(y as usize + j) * stride + x as usize + i] as i32;
                let s = self.src_sample(c_idx, x + i as u32, y + j as u32) as i32;
                residual[j * size + i] = (s - pred) as i16;
            }
        }

        let is_dst = log2_size == 2 && c_idx == 0;
        self.stats.forward_transforms += 1;
        let coeffs = transform::forward_transform(&residual, log2_size, is_dst, self.bit_depth);
        let (levels, nnz) = if self.single_scan_rdoq {
            let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
            rdoq::rdoq_single_scan(
                ctxs,
                &coeffs,
                log2_size,
                c_idx,
                qp,
                self.bit_depth,
                scan,
                self.lambda(),
            )
        } else {
            let (levels, _) = transform::quantize(&coeffs, log2_size, qp, self.bit_depth);
            let nnz = levels.iter().filter(|&&v| v != 0).count() as u32;
            let passes = self.rdoq_passes(log2_size, c_idx, nnz);
            if passes == 0 {
                (levels, nnz)
            } else {
                self.refine_levels_rdoq_limited(
                    ctxs, &coeffs, levels, log2_size, c_idx, mode, qp, passes,
                )
            }
        };
        let cbf = nnz > 0;

        if cbf {
            self.stats.inverse_transforms += 1;
            let res =
                transform::reconstruct_residual(&levels, log2_size, qp, self.bit_depth, is_dst);
            let max_val = (1i32 << self.bit_depth) - 1;
            let (plane, stride) = self.frame.plane_mut(c_idx);
            for j in 0..size {
                for i in 0..size {
                    let idx = (y as usize + j) * stride + x as usize + i;
                    let pred = plane[idx] as i32;
                    plane[idx] = (pred + res[j * size + i] as i32).clamp(0, max_val) as u16;
                }
            }
        }

        let frac_bits = if cbf {
            self.residual_frac_bits(ctxs, &levels, log2_size, c_idx, mode)
        } else {
            0
        };

        CodedBlock {
            levels,
            cbf,
            frac_bits,
        }
    }

    fn distortion_block(&self, c_idx: u8, x: u32, y: u32, log2_size: u8) -> u64 {
        let size = 1usize << log2_size;
        let (plane, stride) = self.frame.plane(c_idx);
        let mut sse = 0u64;
        for j in 0..size {
            for i in 0..size {
                let recon = plane[(y as usize + j) * stride + x as usize + i] as i64;
                let src = self.src_sample(c_idx, x + i as u32, y + j as u32) as i64;
                let d = src - recon;
                sse += (d * d) as u64;
            }
        }
        sse
    }

    fn distortion_tt_region(&self, x0: u32, y0: u32, log2_size: u8) -> u64 {
        let mut sse = self.distortion_block(0, x0, y0, log2_size);
        if let Some((cx, cy, clog2)) = chroma_tb_geom(self.cat, x0, y0, log2_size) {
            sse += self.distortion_block(1, cx, cy, clog2);
            sse += self.distortion_block(2, cx, cy, clog2);
        }
        sse
    }

    fn rd_cost(&self, distortion: u64, frac_bits: u64) -> f64 {
        // x265's intra analysis uses a QP-derived lambda. This keeps the first
        // Rust RD decisions on that same curve while CABAC costs are in
        // 1/32768-bit fixed-point units.
        distortion as f64 + self.lambda() * (frac_bits as f64 / CabacEstimator::SCALE as f64)
    }

    fn lambda(&self) -> f64 {
        0.57f64 * 2f64.powf((self.qp_y as f64 - 12.0) / 3.0)
    }

    fn residual_frac_bits(
        &mut self,
        ctxs: &Contexts,
        levels: &[i16],
        log2_size: u8,
        c_idx: u8,
        mode: u8,
    ) -> u64 {
        if levels.iter().all(|&v| v == 0) {
            return 0;
        }
        self.stats.residual_bit_estimates += 1;
        let mut ctxs = ctxs.clone();
        let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
        estimate_residual_bits(&mut ctxs, levels, log2_size, c_idx, scan, false)
    }


    fn rdoq_passes(&self, log2_size: u8, _c_idx: u8, nnz: u32) -> u8 {
        match self.effort {
            Effort::Fast => 0,
            Effort::Balanced => {
                if log2_size <= 3 && nnz <= 32 {
                    1
                } else {
                    0
                }
            }
            Effort::Best => {
                if log2_size <= 4 && nnz <= 128 {
                    2
                } else {
                    1
                }
            }
        }
    }

    fn refine_levels_rdoq_limited(
        &mut self,
        ctxs: &Contexts,
        coeffs: &[i16],
        mut levels: Vec<i16>,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        qp: i32,
        max_passes: u8,
    ) -> (Vec<i16>, u32) {
        // A single greedy pass can leave improvements on the table: changing
        // one coefficient's level shifts the significant_coeff_flag/last-
        // position context for its neighbours, which can make a previously
        // rejected candidate for *another* coefficient become the better
        // choice (including driving the whole block to all-zero, i.e.
        // cbf == 0). Repeat the per-coefficient pass until it converges
        // (no level changes) or a small pass cap is hit.
        //
        // Coefficient distortion (`coeff_distortion`) is separable: changing
        // one level only changes that coefficient's term. We track a running
        // total `cur_dist` and update it by the per-coefficient delta instead
        // of re-summing the whole block per candidate. The CABAC bit estimate
        // is *not* separable (contexts ripple along the scan), so it is still
        // recomputed via `residual_frac_bits` — the dominant cost, see the
        // RDOQ profiling notes.
        let lambda = self.lambda();
        let scale = CabacEstimator::SCALE as f64;
        let dq = transform::DequantParams::new(log2_size, qp, self.bit_depth);
        let coeff_dist = |idx: usize, level: i16| -> u64 {
            let d = coeffs[idx] as i64 - dq.apply(level) as i64;
            (d * d) as u64
        };
        let mut cur_dist: u64 = (0..levels.len()).map(|i| coeff_dist(i, levels[i])).sum();

        // Exact sub-block-boundary cache for candidate bit costs: replaces the
        // per-candidate full `residual_coding()` re-encode with replaying only
        // the changed coefficient's sub-block and the lower-frequency tail.
        // Rebuilt whenever a level is actually changed (so it always reflects
        // the current `levels`). See `ResidualEstimateCache`.
        let scan_order = get_scan_order(log2_size, mode, c_idx, self.cat);
        self.stats.cache_builds += 1;
        let mut cache =
            ResidualEstimateCache::build(ctxs, &levels, log2_size, c_idx, scan_order, false);

        for _ in 0..max_passes {
            let mut changed = false;
            // `total_bits()` is the exact full cost of the current levels (for a
            // non-all-zero block); when the block is all-zero the inner loop
            // refines nothing, so `best_cost` is never consulted.
            let mut best_cost = cur_dist as f64 + lambda * (cache.total_bits() as f64 / scale);
            for idx in 0..levels.len() {
                let level = levels[idx];
                if level == 0 {
                    continue;
                }
                let sign = level.signum();
                let abs = level.unsigned_abs() as i16;
                let mut candidates = [0i16; 4];
                let mut len = 0usize;
                candidates[len] = level;
                len += 1;
                candidates[len] = 0;
                len += 1;
                if abs > 1 {
                    candidates[len] = sign * (abs - 1);
                    len += 1;
                }
                if abs < 32767 {
                    candidates[len] = sign * (abs + 1);
                    len += 1;
                }

                let original = levels[idx];
                let orig_dist = coeff_dist(idx, original);
                let mut best_level = original;
                for &candidate in &candidates[..len] {
                    if candidate == original {
                        continue;
                    }
                    // Fast exact path via the cache; fall back to a full
                    // estimate when it declines (4x4, or zeroing the last
                    // significant coefficient).
                    let est = cache.estimate_one_change(&levels, idx, candidate);
                    let bits = match est {
                        Some(b) => {
                            self.stats.cache_fast_hits += 1;
                            b
                        }
                        None => {
                            self.stats.cache_fallbacks += 1;
                            levels[idx] = candidate;
                            let b =
                                self.residual_frac_bits(ctxs, &levels, log2_size, c_idx, mode);
                            levels[idx] = original;
                            b
                        }
                    };
                    let cand_dist = cur_dist - orig_dist + coeff_dist(idx, candidate);
                    let cost = cand_dist as f64 + lambda * (bits as f64 / scale);
                    if cost < best_cost {
                        best_cost = cost;
                        best_level = candidate;
                    }
                }
                if best_level != original {
                    levels[idx] = best_level;
                    changed = true;
                    cur_dist = cur_dist - orig_dist + coeff_dist(idx, best_level);
                    // Incrementally update the cache for the accepted change;
                    // only fall back to a full rebuild when the last position
                    // moved (or 4x4).
                    if !cache.apply_change(&levels, idx) {
                        self.stats.cache_builds += 1;
                        cache = ResidualEstimateCache::build(
                            ctxs, &levels, log2_size, c_idx, scan_order, false,
                        );
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let nnz = levels.iter().filter(|&&v| v != 0).count() as u32;
        (levels, nnz)
    }

    fn can_split_tt(&self, log2_size: u8, trafo_depth: u8) -> bool {
        if !(log2_size <= MAX_TB_LOG2
            && log2_size > MIN_TB_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH)
        {
            return false;
        }
        // 4:2:0's 8x8 luma TU splits to four 4x4 leaves with no per-leaf
        // chroma TB; chroma is instead coded once at this (8x8) node via
        // `build_parent_chroma_tu`. That special path isn't yet validated
        // against the decoder's recon, so keep 8x8 leaves unsplit for 4:2:0.
        !(self.cat == 1 && log2_size == 3)
    }

    fn build_parent_chroma_tu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Option<ParentChromaTu> {
        if self.cat == 3 || log2_size != 3 {
            return None;
        }
        let (cx, cy, clog2) = chroma_tb_geom(self.cat, x0, y0, log2_size)?;
        let cb = self.code_block(ctxs, cx, cy, clog2, 1, chroma_mode, self.qp_c);
        let cr = self.code_block(ctxs, cx, cy, clog2, 2, chroma_mode, self.qp_c);
        Some(ParentChromaTu {
            log2_size: clog2,
            chroma_mode,
            cb,
            cr,
        })
    }

    fn build_tt_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let luma = self.code_block(ctxs, x0, y0, log2_size, 0, luma_mode, self.qp_y);

        let (chroma_log2, cb, cr) = match chroma_tb_geom(self.cat, x0, y0, log2_size) {
            Some((cx, cy, clog2)) => {
                let cb = self.code_block(ctxs, cx, cy, clog2, 1, chroma_mode, self.qp_c);
                let cr = self.code_block(ctxs, cx, cy, clog2, 2, chroma_mode, self.qp_c);
                (clog2, cb, cr)
            }
            None => (0, CodedBlock::empty(), CodedBlock::empty()),
        };

        Tt::Leaf(LeafTu {
            log2_size,
            chroma_log2,
            trafo_depth,
            luma_mode,
            chroma_mode,
            luma,
            cb,
            cr,
        })
    }

    fn build_tt_split(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let half = 1u32 << (log2_size - 1);
        let kids = vec![
            self.build_tt(
                x0,
                y0,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
            self.build_tt(
                x0 + half,
                y0,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
            self.build_tt(
                x0,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
            self.build_tt(
                x0 + half,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
        ];
        let parent_chroma = self.build_parent_chroma_tu(x0, y0, log2_size, chroma_mode, ctxs);
        let cbf_cb = parent_chroma
            .as_ref()
            .map(|c| c.cb.cbf)
            .unwrap_or_else(|| kids.iter().any(|k| k.cbf_cb()));
        let cbf_cr = parent_chroma
            .as_ref()
            .map(|c| c.cr.cbf)
            .unwrap_or_else(|| kids.iter().any(|k| k.cbf_cr()));
        Tt::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cr,
            parent_chroma,
            kids,
        }
    }

    /// Build a transform (sub)tree: reconstruct in coding order, record levels.
    fn build_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            return self.build_tt_split(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }

        if !self.can_split_tt(log2_size, trafo_depth) {
            return self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }

        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);

        self.restore_frame_region(&base_frame);
        let leaf = self.build_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
        let leaf_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let leaf_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);

        self.restore_frame_region(&base_frame);
        let split =
            self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
        let split_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let split_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
        let split_cost = self.rd_cost(split_distortion, split_bits);

        if split_cost < leaf_cost {
            split
        } else {
            self.restore_frame_region(&leaf_frame);
            leaf
        }
    }

    /// Build a coding unit at `(x0, y0)`/`log2_cb_size`: choose intra modes
    /// and build its transform tree (mirrors the leaf body of the former
    /// `write_coding_quadtree`).
    ///
    /// The luma mode is chosen from a SATD-shortlist (see
    /// [`choose_luma_mode_candidates`]); when the shortlist has more than one
    /// entry (`Effort::Balanced`/`Best`), each candidate gets its own
    /// `choose_chroma_mode` + full `build_tt` (TU-split RD) trial, and the
    /// candidate with the lowest CU-level RD cost wins. `Effort::Fast` always
    /// shortlists a single mode, so this degenerates to one trial.
    fn build_cu_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuLeaf {
        self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let tt_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let candidates = self.choose_luma_mode_candidates(x0, y0, tt_log2, ctxs);
        let cand_a = self.neighbor_left_mode(x0, y0);
        let cand_b = self.neighbor_above_mode(x0, y0);
        let mpm = fill_mpm_candidates(cand_a, cand_b);

        let base_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let base_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);

        let mut best_leaf: Option<CuLeaf> = None;
        let mut best_cost = f64::MAX;
        let mut best_frame: Option<FrameSnapshot> = None;
        let mut best_mode_map: Option<MapSnapshot> = None;

        for luma_mode in candidates {
            self.stats.cu_trials += 1;
            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);

            let (chroma_mode, chroma_mode_idx) =
                self.choose_chroma_mode(x0, y0, tt_log2, luma_mode, ctxs);
            self.store_mode(x0, y0, log2_cb_size, luma_mode);

            let tt = self.build_tt(x0, y0, log2_cb_size, 0, luma_mode, chroma_mode, ctxs);
            let leaf = CuLeaf {
                mpm,
                luma_mode,
                chroma_mode_idx,
                tt,
            };

            let distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
            let bits = estimate_cu_leaf_bits(ctxs, &leaf, log2_cb_size, self.cat);
            let cost = self.rd_cost(distortion, bits);

            if cost < best_cost {
                best_cost = cost;
                best_frame = Some(self.snapshot_frame_region(x0, y0, log2_cb_size));
                best_mode_map = Some(self.snapshot_mode_region(x0, y0, log2_cb_size));
                best_leaf = Some(leaf);
            }
        }

        self.restore_frame_region(
            best_frame
                .as_ref()
                .expect("choose_luma_mode_candidates returns at least one candidate"),
        );
        self.restore_mode_region(
            best_mode_map
                .as_ref()
                .expect("choose_luma_mode_candidates returns at least one candidate"),
        );
        best_leaf.expect("choose_luma_mode_candidates returns at least one candidate")
    }

    /// Build the (up to four, fewer at picture boundaries) children of a
    /// split coding quadtree node; mirrors the recursion in [`write_cu`].
    fn build_cu_kids(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> Vec<CuNode> {
        let half = (1u32 << log2_cb_size) / 2;
        let x1 = x0 + half;
        let y1 = y0 + half;
        let mut kids = Vec::new();
        kids.push(self.build_cu(x0, y0, log2_cb_size - 1, ct_depth + 1, ctxs));
        if x1 < self.display_width {
            kids.push(self.build_cu(x1, y0, log2_cb_size - 1, ct_depth + 1, ctxs));
        }
        if y1 < self.display_height {
            kids.push(self.build_cu(x0, y1, log2_cb_size - 1, ct_depth + 1, ctxs));
        }
        if x1 < self.display_width && y1 < self.display_height {
            kids.push(self.build_cu(x1, y1, log2_cb_size - 1, ct_depth + 1, ctxs));
        }
        kids
    }

    /// Build a coding quadtree node: RD-choose between coding `(x0, y0)` at
    /// `log2_cb_size` as a single CU, or splitting into four
    /// `log2_cb_size - 1` children. Splitting is forced at picture
    /// boundaries (so the CU stays inside the picture) and unsplit is forced
    /// at the minimum CB size (`log2_cb_size == 3`).
    fn build_cu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        let cb_size = 1u32 << log2_cb_size;
        let fully_inside =
            x0 + cb_size <= self.display_width && y0 + cb_size <= self.display_height;
        let can_split = log2_cb_size > 3;

        if !can_split {
            return CuNode::Leaf(self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs));
        }
        if !fully_inside {
            return CuNode::Split {
                kids: self.build_cu_kids(x0, y0, log2_cb_size, ct_depth, ctxs),
            };
        }

        let ctx_inc = self.split_ctx_inc(x0, y0, ct_depth);

        let base_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let base_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let base_ct_depth_map = self.snapshot_ct_depth_region(x0, y0, log2_cb_size);

        let leaf = self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs);
        let leaf_distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
        let leaf_bits = estimate_cu_leaf_bits(ctxs, &leaf, log2_cb_size, self.cat)
            + estimate_split_cu_flag_bits(ctxs, ctx_inc, false);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let leaf_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let leaf_ct_depth_map = self.snapshot_ct_depth_region(x0, y0, log2_cb_size);

        self.restore_frame_region(&base_frame);
        self.restore_mode_region(&base_mode_map);
        self.restore_ct_depth_region(&base_ct_depth_map);

        let kids = self.build_cu_kids(x0, y0, log2_cb_size, ct_depth, ctxs);
        let split_distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
        let split_bits = estimate_cu_kids_bits(ctxs, &kids, x0, y0, log2_cb_size, ct_depth, self)
            + estimate_split_cu_flag_bits(ctxs, ctx_inc, true);
        let split_cost = self.rd_cost(split_distortion, split_bits);

        if split_cost < leaf_cost {
            CuNode::Split { kids }
        } else {
            self.restore_frame_region(&leaf_frame);
            self.restore_mode_region(&leaf_mode_map);
            self.restore_ct_depth_region(&leaf_ct_depth_map);
            CuNode::Leaf(leaf)
        }
    }
}

/// Write the transform tree's CABAC syntax (mirrors `decode_transform_tree_inner`).
#[allow(clippy::too_many_arguments)]
fn write_tt(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    node: &Tt,
    cat: u8,
    intra_split_flag: bool,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
) {
    let (log2_size, trafo_depth) = match node {
        Tt::Split {
            log2_size,
            trafo_depth,
            ..
        } => (*log2_size, *trafo_depth),
        Tt::Leaf(l) => (l.log2_size, l.trafo_depth),
    };
    let is_split = matches!(node, Tt::Split { .. });

    // split_transform_flag: coded only in the decodable window.
    let max_trafo_depth = MAX_INTRA_TT_DEPTH + intra_split_flag as u8;
    let split_coded = log2_size <= MAX_TB_LOG2
        && log2_size > MIN_TB_LOG2
        && trafo_depth < max_trafo_depth
        && !(intra_split_flag && trafo_depth == 0);
    if split_coded {
        let ci = ctx::SPLIT_TRANSFORM_FLAG + (5 - log2_size as usize).min(2);
        enc.encode_bin(w, is_split as u8, ctxs.get(ci));
    }

    // chroma cbf (H.265 7.3.8.8: `log2TrafoSize > 2 || ChromaArrayType == 3`).
    let decode_chroma_cbf = log2_size > 2 || cat == 3;
    let (cbf_cb, cbf_cr) = (node.cbf_cb(), node.cbf_cr());
    if decode_chroma_cbf {
        let ci = ctx::CBF_CBCR + trafo_depth as usize;
        if trafo_depth == 0 || parent_cbf_cb {
            enc.encode_bin(w, cbf_cb as u8, ctxs.get(ci));
        }
        if trafo_depth == 0 || parent_cbf_cr {
            enc.encode_bin(w, cbf_cr as u8, ctxs.get(ci));
        }
    }

    match node {
        Tt::Split {
            kids,
            parent_chroma,
            ..
        } => {
            for kid in kids {
                write_tt(enc, w, ctxs, kid, cat, intra_split_flag, cbf_cb, cbf_cr);
            }
            if let Some(c) = parent_chroma {
                if c.cb.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 1, cat);
                    encode_residual(enc, w, ctxs, &c.cb.levels, c.log2_size, 1, scan, false);
                }
                if c.cr.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &c.cr.levels, c.log2_size, 2, scan, false);
                }
            }
        }
        Tt::Leaf(l) => {
            // The chroma cbf in effect at this leaf (inherited if not coded here).
            let eff_cbf_cb = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cb) {
                cbf_cb
            } else {
                parent_cbf_cb
            };
            let eff_cbf_cr = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cr) {
                cbf_cr
            } else {
                parent_cbf_cr
            };

            // cbf_luma (always coded for intra).
            let ctx_off = if trafo_depth == 0 { 1 } else { 0 };
            enc.encode_bin(w, l.luma.cbf as u8, ctxs.get(ctx::CBF_LUMA + ctx_off));

            if l.luma.cbf {
                let scan = get_scan_order(l.log2_size, l.luma_mode, 0, cat);
                encode_residual(enc, w, ctxs, &l.luma.levels, l.log2_size, 0, scan, false);
            }
            if has_chroma_tb(cat, l.log2_size) {
                let clog2 = l.chroma_log2;
                if eff_cbf_cb {
                    let scan = get_scan_order(clog2, l.chroma_mode, 1, cat);
                    encode_residual(enc, w, ctxs, &l.cb.levels, clog2, 1, scan, false);
                }
                if eff_cbf_cr {
                    let scan = get_scan_order(clog2, l.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &l.cr.levels, clog2, 2, scan, false);
                }
            }
        }
    }
}

fn estimate_tt_bits(
    ctxs: &Contexts,
    node: &Tt,
    cat: u8,
    intra_split_flag: bool,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
) -> u64 {
    let mut ctxs = ctxs.clone();
    let mut est = CabacEstimator::new();
    estimate_tt_inner(
        &mut est,
        &mut ctxs,
        node,
        cat,
        intra_split_flag,
        parent_cbf_cb,
        parent_cbf_cr,
    );
    est.frac_bits()
}

#[allow(clippy::too_many_arguments)]
fn estimate_tt_inner(
    est: &mut CabacEstimator,
    ctxs: &mut Contexts,
    node: &Tt,
    cat: u8,
    intra_split_flag: bool,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
) {
    let (log2_size, trafo_depth) = match node {
        Tt::Split {
            log2_size,
            trafo_depth,
            ..
        } => (*log2_size, *trafo_depth),
        Tt::Leaf(l) => (l.log2_size, l.trafo_depth),
    };
    let is_split = matches!(node, Tt::Split { .. });

    let max_trafo_depth = MAX_INTRA_TT_DEPTH + intra_split_flag as u8;
    let split_coded = log2_size <= MAX_TB_LOG2
        && log2_size > MIN_TB_LOG2
        && trafo_depth < max_trafo_depth
        && !(intra_split_flag && trafo_depth == 0);
    if split_coded {
        let ci = ctx::SPLIT_TRANSFORM_FLAG + (5 - log2_size as usize).min(2);
        est.encode_bin(is_split as u8, ctxs.get(ci));
    }

    let decode_chroma_cbf = log2_size > 2 || cat == 3;
    let (cbf_cb, cbf_cr) = (node.cbf_cb(), node.cbf_cr());
    if decode_chroma_cbf {
        let ci = ctx::CBF_CBCR + trafo_depth as usize;
        if trafo_depth == 0 || parent_cbf_cb {
            est.encode_bin(cbf_cb as u8, ctxs.get(ci));
        }
        if trafo_depth == 0 || parent_cbf_cr {
            est.encode_bin(cbf_cr as u8, ctxs.get(ci));
        }
    }

    match node {
        Tt::Split {
            kids,
            parent_chroma,
            ..
        } => {
            for kid in kids {
                estimate_tt_inner(est, ctxs, kid, cat, intra_split_flag, cbf_cb, cbf_cr);
            }
            if let Some(c) = parent_chroma {
                if c.cb.cbf {
                    est.add_frac_bits(c.cb.frac_bits);
                }
                if c.cr.cbf {
                    est.add_frac_bits(c.cr.frac_bits);
                }
            }
        }
        Tt::Leaf(l) => {
            let eff_cbf_cb = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cb) {
                cbf_cb
            } else {
                parent_cbf_cb
            };
            let eff_cbf_cr = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cr) {
                cbf_cr
            } else {
                parent_cbf_cr
            };

            let ctx_off = if trafo_depth == 0 { 1 } else { 0 };
            est.encode_bin(l.luma.cbf as u8, ctxs.get(ctx::CBF_LUMA + ctx_off));

            if l.luma.cbf {
                est.add_frac_bits(l.luma.frac_bits);
            }
            if has_chroma_tb(cat, l.log2_size) {
                if eff_cbf_cb {
                    est.add_frac_bits(l.cb.frac_bits);
                }
                if eff_cbf_cr {
                    est.add_frac_bits(l.cr.frac_bits);
                }
            }
        }
    }
}

/// Emit `prev_intra_luma_pred_flag` + `mpm_idx`/`rem_intra_luma_pred_mode` for
/// a chosen luma `mode`, mirroring the decoder's MPM derivation.
fn write_intra_luma_mode(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    mpm: [IntraPredMode; 3],
    mode: u8,
) {
    let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
    let in_mpm = mpm_u8.iter().position(|&m| m == mode);

    if let Some(idx) = in_mpm {
        enc.encode_bin(w, 1, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        // mpm_idx as truncated unary in bypass: 0 -> "0", 1 -> "10", 2 -> "11".
        match idx {
            0 => enc.encode_bin_ep(w, 0),
            1 => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 0);
            }
            _ => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 1);
            }
        }
    } else {
        enc.encode_bin(w, 0, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        // rem = mode reduced past the sorted MPM values (inverse of decode).
        let mut sorted = mpm_u8;
        sorted.sort_unstable();
        let mut rem = mode as i32;
        for &v in sorted.iter().rev() {
            if rem >= v as i32 {
                rem -= 1;
            }
        }
        // 5 fixed-length bypass bits, MSB first.
        for b in (0..5).rev() {
            enc.encode_bin_ep(w, ((rem >> b) & 1) as u8);
        }
    }
}

fn estimate_intra_luma_mode_bits(ctxs: &Contexts, mpm: [IntraPredMode; 3], mode: u8) -> u64 {
    let mut ctxs = ctxs.clone();
    let mut est = CabacEstimator::new();
    let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
    let in_mpm = mpm_u8.iter().position(|&m| m == mode);

    if let Some(idx) = in_mpm {
        est.encode_bin(1, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        match idx {
            0 => est.encode_bin_ep(0),
            1 => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(0);
            }
            _ => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(1);
            }
        }
    } else {
        est.encode_bin(0, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        for _ in 0..5 {
            est.encode_bin_ep(0);
        }
    }

    est.frac_bits()
}

/// Emit `intra_chroma_pred_mode` for H.265 Table 8-2 mode indexes:
/// 0=Planar, 1=Vertical, 2=Horizontal, 3=DC, 4=DM.
fn write_intra_chroma_mode(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    mode_idx: u8,
) {
    if mode_idx == CHROMA_DM_IDX {
        enc.encode_bin(w, 0, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
        return;
    }

    enc.encode_bin(w, 1, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
    enc.encode_bin_ep(w, (mode_idx >> 1) & 1);
    enc.encode_bin_ep(w, mode_idx & 1);
}

fn estimate_intra_chroma_mode_bits(ctxs: &Contexts, mode_idx: u8) -> u64 {
    let mut ctxs = ctxs.clone();
    let mut est = CabacEstimator::new();
    if mode_idx == CHROMA_DM_IDX {
        est.encode_bin(0, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
    } else {
        est.encode_bin(1, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
        est.encode_bin_ep((mode_idx >> 1) & 1);
        est.encode_bin_ep(mode_idx & 1);
    }
    est.frac_bits()
}

/// Estimated bits for a `split_cu_flag` bin at `ctx_inc` with value `value`.
fn estimate_split_cu_flag_bits(ctxs: &Contexts, ctx_inc: usize, value: bool) -> u64 {
    let mut ctxs = ctxs.clone();
    let mut est = CabacEstimator::new();
    est.encode_bin(value as u8, ctxs.get(ctx::SPLIT_CU_FLAG + ctx_inc));
    est.frac_bits()
}

/// Estimated bits for a coded CU leaf: `part_mode` (at the minimum CB size),
/// the intra luma/chroma mode syntax, and the transform tree.
fn estimate_cu_leaf_bits(ctxs: &Contexts, leaf: &CuLeaf, log2_cb_size: u8, cat: u8) -> u64 {
    let mut bits = 0u64;
    if log2_cb_size == 3 {
        let mut c = ctxs.clone();
        let mut est = CabacEstimator::new();
        est.encode_bin(1, c.get(ctx::PART_MODE));
        bits += est.frac_bits();
    }
    bits += estimate_intra_luma_mode_bits(ctxs, leaf.mpm, leaf.luma_mode);
    bits += estimate_intra_chroma_mode_bits(ctxs, leaf.chroma_mode_idx);
    bits += estimate_tt_bits(ctxs, &leaf.tt, cat, false, true, true);
    bits
}

/// Estimated bits for a coding quadtree node (leaf or split), mirroring
/// [`write_cu`]'s traversal so `split_cu_flag` contexts line up.
fn estimate_cu_node_bits(
    ctxs: &Contexts,
    node: &CuNode,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
    state: &Encoder<'_>,
) -> u64 {
    let cb_size = 1u32 << log2_cb_size;
    let fully_inside = x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
    let can_split = log2_cb_size > 3;
    let split_coded = fully_inside && can_split;

    let mut bits = 0u64;
    if split_coded {
        let ctx_inc = state.split_ctx_inc(x0, y0, ct_depth);
        bits += estimate_split_cu_flag_bits(ctxs, ctx_inc, matches!(node, CuNode::Split { .. }));
    }
    match node {
        CuNode::Split { kids } => {
            bits += estimate_cu_kids_bits(ctxs, kids, x0, y0, log2_cb_size, ct_depth, state);
        }
        CuNode::Leaf(leaf) => {
            bits += estimate_cu_leaf_bits(ctxs, leaf, log2_cb_size, state.cat);
        }
    }
    bits
}

/// Estimated bits for the (up to four) children of a split coding quadtree
/// node; mirrors the recursion in [`write_cu`]/[`Encoder::build_cu_kids`].
fn estimate_cu_kids_bits(
    ctxs: &Contexts,
    kids: &[CuNode],
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
    state: &Encoder<'_>,
) -> u64 {
    let half = (1u32 << log2_cb_size) / 2;
    let x1 = x0 + half;
    let y1 = y0 + half;
    let mut bits = 0u64;
    let mut kids = kids.iter();
    bits += estimate_cu_node_bits(
        ctxs,
        kids.next().unwrap(),
        x0,
        y0,
        log2_cb_size - 1,
        ct_depth + 1,
        state,
    );
    if x1 < state.display_width {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x1,
            y0,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    if y1 < state.display_height {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x0,
            y1,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    if x1 < state.display_width && y1 < state.display_height {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x1,
            y1,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    bits
}

/// Write a coding quadtree node's CABAC syntax: `split_cu_flag` (when
/// coded), then either the four children or the leaf CU's `coding_unit`
/// (intra mode signalling + transform tree). Mirrors
/// [`Encoder::build_cu`]/[`Encoder::build_cu_kids`]'s traversal.
#[allow(clippy::too_many_arguments)]
fn write_cu(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    node: &CuNode,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
) {
    let cb_size = 1u32 << log2_cb_size;
    let fully_inside = x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
    let can_split = log2_cb_size > 3;
    let split_coded = fully_inside && can_split;

    if split_coded {
        let ci = ctx::SPLIT_CU_FLAG + state.split_ctx_inc(x0, y0, ct_depth);
        enc.encode_bin(w, matches!(node, CuNode::Split { .. }) as u8, ctxs.get(ci));
    }

    match node {
        CuNode::Split { kids } => {
            let half = cb_size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            let mut kids = kids.iter();

            write_cu(
                state,
                enc,
                w,
                ctxs,
                kids.next().unwrap(),
                x0,
                y0,
                log2_cb_size - 1,
                ct_depth + 1,
            );
            if x1 < state.display_width {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x1,
                    y0,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
            if y1 < state.display_height {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x0,
                    y1,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
            if x1 < state.display_width && y1 < state.display_height {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x1,
                    y1,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
        }
        CuNode::Leaf(leaf) => {
            // coding_unit: intra. At minimum CU size, part_mode is coded;
            // choose 2Nx2N (first bin 1), not NxN.
            if log2_cb_size == 3 {
                enc.encode_bin(w, 1, ctxs.get(ctx::PART_MODE));
            }
            write_intra_luma_mode(enc, w, ctxs, leaf.mpm, leaf.luma_mode);
            write_intra_chroma_mode(enc, w, ctxs, leaf.chroma_mode_idx);
            write_tt(enc, w, ctxs, &leaf.tt, state.cat, false, true, true);
        }
    }
}

fn write_coding_quadtree(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
) {
    let cu = state.build_cu(x0, y0, log2_cb_size, ct_depth, ctxs);
    write_cu(state, enc, w, ctxs, &cu, x0, y0, log2_cb_size, ct_depth);
}

/// Encode an 8-bit or 10-bit 4:2:0/4:4:4 picture to a complete Annex-B IDR
/// access unit. Returns the bytes, the encoder's reconstruction (which a
/// conforming decoder reproduces exactly), and analysis counters.
pub fn encode_with_stats(
    config: &StillHevcConfig,
    src: Source<'_>,
) -> (Vec<u8>, DecodedFrame, EncodeStats) {
    assert!(
        config.bit_depth == 8 || config.bit_depth == 10,
        "supported bit depths: 8, 10"
    );
    let cat = chroma_array_type(config.chroma);

    let width = config.width;
    let height = config.height;
    let min_cb = 8;
    let coded_width = width.div_ceil(min_cb) * min_cb;
    let coded_height = height.div_ceil(min_cb) * min_cb;
    let bd = config.bit_depth;
    // H.265 8.6.1: dequant QP includes the bit-depth offset (6*(bd-8)).
    let qp_bd_offset = 6 * (bd as i32 - 8);
    let slice_qp_y = config.qp as i32;
    let qp_y = slice_qp_y + qp_bd_offset;
    let qpi_c = slice_qp_y.clamp(-qp_bd_offset, 57);
    let qp_c = chroma_qp_from_luma(qpi_c) + qp_bd_offset;

    let mode_stride = width.div_ceil(4) as usize;
    let mode_height = height.div_ceil(4) as usize;
    let ct_depth_stride = width.div_ceil(8) as usize;
    let ct_depth_height = height.div_ceil(8) as usize;
    let mut state = Encoder {
        display_width: width,
        display_height: height,
        cat,
        qp_y,
        qp_c,
        bit_depth: bd,
        effort: config.effort,
        src,
        frame: DecodedFrame::with_params(coded_width, coded_height, bd, cat),
        mode_map: vec![1u8; mode_stride * mode_height],
        mode_stride,
        ct_depth_map: vec![0xFF; ct_depth_stride * ct_depth_height],
        ct_depth_stride,
        single_scan_rdoq: select_rdoq_single_scan(config.effort),
        stats: EncodeStats::default(),
    };

    // --- Slice data (CABAC) ---
    let mut w = BitWriter::new();
    let mut enc = CabacEncoder::new();
    // CABAC contexts are initialized from SliceQpY (without the bit-depth
    // offset), per H.265 9.3.2.2.
    let mut ctxs = Contexts::new(slice_qp_y);

    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = width.div_ceil(ctb);
    let ctbs_y = height.div_ceil(ctb);
    let total = ctbs_x * ctbs_y;
    let mut done = 0u32;

    for cy in 0..ctbs_y {
        for cx in 0..ctbs_x {
            let x0 = cx * ctb;
            let y0 = cy * ctb;
            state.stats.ctu_count += 1;

            write_coding_quadtree(&mut state, &mut enc, &mut w, &mut ctxs, x0, y0, CTB_LOG2, 0);

            // end_of_slice_segment_flag.
            done += 1;
            enc.encode_bin_trm(&mut w, (done == total) as u8);
        }
    }

    enc.finish(&mut w);
    w.write_bit(1); // rbsp_stop_one_bit
    w.byte_align();
    let slice_data = w.into_bytes();

    // --- Assemble Annex-B access unit ---
    let mut payload = slice::write_slice_segment_header(config);
    payload.extend_from_slice(&slice_data);

    let mut out = Vec::new();
    nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
    nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
    nal::write_annexb_nal(&mut out, nal::NalType::Pps, &params::write_pps());
    nal::write_annexb_nal(&mut out, nal::NalType::IdrWRadl, &payload);

    (out, state.frame, state.stats)
}

/// Encode an 8-bit or 10-bit 4:2:0/4:4:4 picture
/// to a complete Annex-B IDR access unit. Returns the bytes and the encoder's
/// reconstruction (which a conforming decoder reproduces exactly).
pub fn encode(config: &StillHevcConfig, src: Source<'_>) -> (Vec<u8>, DecodedFrame) {
    let (bytes, recon, _) = encode_with_stats(config, src);
    (bytes, recon)
}
