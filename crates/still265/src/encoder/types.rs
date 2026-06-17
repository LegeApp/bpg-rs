//! Data types, geometry helpers, and statistics for the still-image intra
//! slice-data encoder.
//!
//! Ported from `bpgenc.c` (structs, `chroma_tb_geom`, `has_chroma_tb`,
//! `decode_second_cbf`, `chroma_pred_mode`, `chroma_qp_from_luma`).

use bpg_hevc_decode::hevc::intra::map_chroma_mode_422;
use bpg_hevc_decode::hevc::slice::IntraPredMode;
use std::fmt;

use crate::plan::DecisionConfidence;
use crate::ChromaFormat;

/// `chroma_format_idc` / `ChromaArrayType` (H.265 Table 6-1), for the formats
/// this encoder supports.
pub(super) fn chroma_array_type(chroma: ChromaFormat) -> u8 {
    match chroma {
        ChromaFormat::Gray => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// Geometry of the chroma transform block(s) associated with a luma TB at
/// `(x0, y0, log2_size)`, or `None` if this leaf has no chroma TB (4:2:0/
/// 4:2:2, `log2_size == 2`: chroma is coded by the parent 8x8 node instead).
///
/// Returns `(cx, cy, clog2, count)`: `count` square chroma TBs of size
/// `1 << clog2`, stacked vertically starting at `(cx, cy)` with stride
/// `1 << clog2` (i.e. TB `i` is at `(cx, cy + i * (1 << clog2))`).
///
/// - 4:2:0 (`cat == 1`): one half-size chroma TB at `(x0/2, y0/2)`, only when
///   `log2_size >= 3`.
/// - 4:2:2 (`cat == 2`): two half-width, half-height-of-luma (i.e. square)
///   chroma TBs stacked at `(x0/2, y0)` and `(x0/2, y0 + 2^(log2_size-1))`,
///   only when `log2_size >= 3` (mirrors `decode_chroma_tus`'s `cat == 2`
///   case in the decoder).
/// - 4:4:4 (`cat == 3`): chroma TBs mirror luma 1:1 (same position and size),
///   at every leaf including `log2_size == 2` (H.265 7.3.8.8: chroma cbf and
///   transform_unit chroma blocks are present whenever `ChromaArrayType ==
///   3`, regardless of `log2_size`).
pub(super) fn chroma_tb_geom(
    cat: u8,
    x0: u32,
    y0: u32,
    log2_size: u8,
) -> Option<(u32, u32, u8, u8)> {
    match cat {
        0 => None,
        1 if log2_size >= 3 => Some((x0 / 2, y0 / 2, log2_size - 1, 1)),
        2 if log2_size >= 3 => Some((x0 / 2, y0, log2_size - 1, 2)),
        1 | 2 => None,
        3 => Some((x0, y0, log2_size, 1)),
        _ => unreachable!(),
    }
}

/// Whether a leaf TB at `log2_size` carries its own chroma TB(s) (see
/// [`chroma_tb_geom`]).
pub(super) fn has_chroma_tb(cat: u8, log2_size: u8) -> bool {
    cat != 0 && (cat == 3 || log2_size >= 3)
}

/// Whether a second stacked chroma TB/cbf is coded for this transform-tree
/// node (H.265 7.3.8.7: 4:2:2 only, wherever the chroma residual is coded at
/// this level — a leaf, or an 8x8 node splitting to 4x4 luma).
pub(super) fn decode_second_cbf(cat: u8, log2_size: u8, is_split: bool) -> bool {
    cat == 2 && (!is_split || log2_size == 3)
}

/// The intra prediction mode actually used for a chroma transform block,
/// given the signalled `intra_chroma_pred_mode` syntax value. 4:2:2 remaps
/// via H.265 Table 8-3 (`map_chroma_mode_422`); other formats use the
/// signalled mode unchanged.
pub(super) fn chroma_pred_mode(cat: u8, chroma_mode: u8) -> u8 {
    if cat == 2 {
        map_chroma_mode_422(chroma_mode)
    } else {
        chroma_mode
    }
}

pub(super) const CTB_LOG2: u8 = 6;
/// log2 of the adaptive-QP quantization-group size (32x32), matching the PPS
/// `diff_cu_qp_delta_depth = 1` (`Log2MinCuQpDeltaSize = CTB_LOG2 − 1`) and the
/// 32x32 preanalysis cell grid ([`crate::preanalysis::CELL_LOG2`]).
pub(super) const QG_LOG2: u8 = CTB_LOG2 - 1;
pub(super) const MAX_TB_LOG2: u8 = 5;
pub(super) const MIN_TB_LOG2: u8 = 2;
pub(super) const MAX_INTRA_TT_DEPTH: u8 = 2;
pub(super) const CHROMA_DM_IDX: u8 = 4;

/// H.265 Table 8-22 chroma QP mapping for 4:2:0 (mirrors the decoder's
/// `chroma_qp_from_luma`).
pub(super) fn chroma_qp_from_luma(qpi: i32) -> i32 {
    static TAB: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        TAB[(qpi - 30) as usize]
    }
}

#[derive(Clone)]
pub(super) struct CodedBlock {
    pub(super) levels: Vec<i16>,
    pub(super) cbf: bool,
    pub(super) frac_bits: u64,
}

impl CodedBlock {
    pub(super) fn empty() -> Self {
        Self {
            levels: Vec::new(),
            cbf: false,
            frac_bits: 0,
        }
    }
}

pub(super) struct BlockTrial {
    pub(super) estimate: crate::plan::BlockEstimate,
    pub(super) coded: CodedBlock,
}

/// A reconstructed-and-recorded leaf transform unit's coded data.
pub(super) struct LeafTu {
    pub(super) log2_size: u8,
    /// `log2` size of the chroma TBs at this leaf, if any (see
    /// [`chroma_tb_geom`]); unused (left at 0) when [`has_chroma_tb`] is false.
    pub(super) chroma_log2: u8,
    pub(super) trafo_depth: u8,
    pub(super) luma_mode: u8,
    pub(super) chroma_mode: u8,
    pub(super) luma: CodedBlock,
    pub(super) cb: CodedBlock,
    pub(super) cr: CodedBlock,
    /// Second stacked chroma TB (4:2:2 only, [`chroma_tb_geom`]'s `count == 2`);
    /// [`CodedBlock::empty`] (with `cbf == false`) otherwise.
    pub(super) cb1: CodedBlock,
    pub(super) cr1: CodedBlock,
}

/// Chroma transform data carried by a split node in the H.265 special case
/// where subsampled chroma is coded at the parent after split luma children.
pub(super) struct ParentChromaTu {
    pub(super) log2_size: u8,
    pub(super) chroma_mode: u8,
    pub(super) cb: CodedBlock,
    pub(super) cr: CodedBlock,
}

pub(super) enum Tt {
    Split {
        log2_size: u8,
        trafo_depth: u8,
        cbf_cb: bool,
        cbf_cr: bool,
        /// Second stacked-chroma-TB cbf (4:2:2 only); always `false` for
        /// 4:2:0/4:4:4.
        cbf_cb1: bool,
        cbf_cr1: bool,
        parent_chroma: Option<ParentChromaTu>,
        kids: Vec<Tt>,
    },
    Leaf(LeafTu),
}

impl Tt {
    pub(super) fn cbf_cb(&self) -> bool {
        match self {
            Tt::Split { cbf_cb, .. } => *cbf_cb,
            Tt::Leaf(l) => l.cb.cbf,
        }
    }
    pub(super) fn cbf_cr(&self) -> bool {
        match self {
            Tt::Split { cbf_cr, .. } => *cbf_cr,
            Tt::Leaf(l) => l.cr.cbf,
        }
    }
    pub(super) fn cbf_cb1(&self) -> bool {
        match self {
            Tt::Split { cbf_cb1, .. } => *cbf_cb1,
            Tt::Leaf(l) => l.cb1.cbf,
        }
    }
    pub(super) fn cbf_cr1(&self) -> bool {
        match self {
            Tt::Split { cbf_cr1, .. } => *cbf_cr1,
            Tt::Leaf(l) => l.cr1.cbf,
        }
    }
}

/// Whether a transform tree codes any residual (any luma/chroma `cbf` set).
pub(super) fn tt_has_residual(tt: &Tt) -> bool {
    match tt {
        Tt::Leaf(l) => l.luma.cbf || l.cb.cbf || l.cr.cbf || l.cb1.cbf || l.cr1.cbf,
        Tt::Split {
            parent_chroma,
            kids,
            ..
        } => {
            parent_chroma
                .as_ref()
                .is_some_and(|pc| pc.cb.cbf || pc.cr.cbf)
                || kids.iter().any(tt_has_residual)
        }
    }
}

/// Whether a coded CU leaf has any non-zero residual. A zero-residual CU is
/// reconstructed exactly from its prediction, so splitting it (liang2013 §3.2
/// CU early termination) almost never lowers RD cost — it only adds
/// `split_cu_flag` bits.
pub(super) fn cu_leaf_has_residual(leaf: &CuLeaf) -> bool {
    tt_has_residual(&leaf.tt)
}

/// Per-PU data for an 8x8 CU coded as `PartNxN` (four 4x4 luma prediction
/// units, each with its own intra mode). Mirrors the decoder's
/// `decode_coding_unit` `PartNxN` branch. PU order is z-scan (TL, TR, BL, BR).
pub(super) struct NxnInfo {
    /// The four 4x4-PU luma intra modes (z-order).
    pub(super) luma_modes: [u8; 4],
    /// The MPM candidate list used to code each PU's luma mode, derived during
    /// build with each prior PU's mode already stored (so PU1/PU2/PU3 see their
    /// sibling neighbours, exactly like the decoder).
    pub(super) mpms: [[IntraPredMode; 3]; 4],
}

/// A coding unit, fully coded (intra mode + transform tree).
pub(super) struct CuLeaf {
    pub(super) mpm: [IntraPredMode; 3],
    pub(super) luma_mode: u8,
    pub(super) chroma_mode_idx: u8,
    pub(super) confidence: DecisionConfidence,
    pub(super) tt: Tt,
    /// `Some` when this 8x8 CU is coded as `PartNxN`. `luma_mode`/`mpm` then
    /// describe PU0 (used for chroma DM and as this CU's neighbour mode for
    /// later CUs); the four per-PU modes live in [`NxnInfo`]. `None` is the
    /// normal single-PU `Part2Nx2N` CU.
    pub(super) nxn: Option<NxnInfo>,
}

/// A coding quadtree node: either a coded leaf CU, or a `split_cu_flag`
/// split into (up to four, fewer at picture boundaries) child nodes.
pub(super) enum CuNode {
    Split { kids: Vec<CuNode> },
    Leaf(CuLeaf),
}

/// Source planes (full-range YCbCr, 4:2:0 or 4:4:4 per [`crate::StillHevcConfig::chroma`])
/// for the picture being encoded. For 4:2:0 `cb`/`cr` are half-resolution
/// (`width/2 x height/2`); for 4:4:4 they are full resolution (`width x height`),
/// same as `y`.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub y: &'a [u16],
    pub cb: &'a [u16],
    pub cr: &'a [u16],
}

pub(super) fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub(super) fn source_hash(
    width: u32,
    height: u32,
    bit_depth: u8,
    chroma: ChromaFormat,
    src: Source<'_>,
) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    hash = fnv1a_update(hash, &width.to_le_bytes());
    hash = fnv1a_update(hash, &height.to_le_bytes());
    hash = fnv1a_update(hash, &[bit_depth, chroma as u8]);
    for plane in [src.y, src.cb, src.cr] {
        hash = fnv1a_update(hash, &(plane.len() as u64).to_le_bytes());
        for &sample in plane {
            hash = fnv1a_update(hash, &sample.to_le_bytes());
        }
    }
    hash
}

#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    pub ctu_count: u64,
    pub cu_trials: u64,
    pub cu_early_terminations: u64,
    pub cu_split_bound_aborts: u64,
    pub angular_exclusions: u64,
    pub region_class_counts: [u64; crate::preanalysis::NUM_CLASSES],
    pub policy_angular_forced: u64,
    pub policy_angular_guarded: u64,
    pub policy_early_term_suppressed: u64,
    pub cu_force_leaf: u64,
    pub tu_split_early_terminations: u64,
    pub tu_neighbor_leaf_skips: u64,
    pub analysis_cache_cu_records: u64,
    pub analysis_cache_leaf_records: u64,
    pub analysis_cache_tu_records: u64,
    pub rmd_prunes: u64,
    pub luma_candidate_expansions: u64,
    pub chroma_candidate_expansions: u64,
    pub final_coded_blocks: u64,
    pub trial_coded_blocks: u64,
    pub final_rdoq_blocks: u64,
    pub trial_rdoq_blocks: u64,
    pub luma_winner_rank_counts: [u64; 5],
    pub chroma_winner_rank_counts: [u64; 6],
    pub cu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub cu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub full_rd_close_calls: u64,
    pub luma_close_call_escalations: u64,
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

impl EncodeStats {
    pub fn merge(&mut self, other: &EncodeStats) {
        self.ctu_count += other.ctu_count;
        self.cu_trials += other.cu_trials;
        self.cu_early_terminations += other.cu_early_terminations;
        self.cu_split_bound_aborts += other.cu_split_bound_aborts;
        self.angular_exclusions += other.angular_exclusions;
        self.policy_angular_forced += other.policy_angular_forced;
        self.policy_angular_guarded += other.policy_angular_guarded;
        self.policy_early_term_suppressed += other.policy_early_term_suppressed;
        self.cu_force_leaf += other.cu_force_leaf;
        self.tu_split_early_terminations += other.tu_split_early_terminations;
        self.tu_neighbor_leaf_skips += other.tu_neighbor_leaf_skips;
        self.analysis_cache_cu_records += other.analysis_cache_cu_records;
        self.analysis_cache_leaf_records += other.analysis_cache_leaf_records;
        self.analysis_cache_tu_records += other.analysis_cache_tu_records;
        self.rmd_prunes += other.rmd_prunes;
        self.luma_candidate_expansions += other.luma_candidate_expansions;
        self.chroma_candidate_expansions += other.chroma_candidate_expansions;
        self.final_coded_blocks += other.final_coded_blocks;
        self.trial_coded_blocks += other.trial_coded_blocks;
        self.final_rdoq_blocks += other.final_rdoq_blocks;
        self.trial_rdoq_blocks += other.trial_rdoq_blocks;
        merge_array(
            &mut self.luma_winner_rank_counts,
            &other.luma_winner_rank_counts,
        );
        merge_array(
            &mut self.chroma_winner_rank_counts,
            &other.chroma_winner_rank_counts,
        );
        merge_array(
            &mut self.cu_leaf_wins_by_region,
            &other.cu_leaf_wins_by_region,
        );
        merge_array(
            &mut self.cu_split_wins_by_region,
            &other.cu_split_wins_by_region,
        );
        merge_array(
            &mut self.tu_leaf_wins_by_region,
            &other.tu_leaf_wins_by_region,
        );
        merge_array(
            &mut self.tu_split_wins_by_region,
            &other.tu_split_wins_by_region,
        );
        self.full_rd_close_calls += other.full_rd_close_calls;
        self.luma_close_call_escalations += other.luma_close_call_escalations;
        self.luma_rough_predictions += other.luma_rough_predictions;
        self.chroma_rough_predictions += other.chroma_rough_predictions;
        self.code_block_calls += other.code_block_calls;
        self.forward_transforms += other.forward_transforms;
        self.inverse_transforms += other.inverse_transforms;
        self.residual_bit_estimates += other.residual_bit_estimates;
        self.cache_builds += other.cache_builds;
        self.cache_fast_hits += other.cache_fast_hits;
        self.cache_fallbacks += other.cache_fallbacks;
        self.frame_snapshots += other.frame_snapshots;
        self.frame_restores += other.frame_restores;
        self.map_snapshots += other.map_snapshots;
        self.map_restores += other.map_restores;
        self.bytes_snapshotted += other.bytes_snapshotted;
    }
}

impl fmt::Display for EncodeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "encode stats:")?;
        writeln!(f, "  ctu_count: {}", self.ctu_count)?;
        writeln!(f, "  cu_trials: {}", self.cu_trials)?;
        writeln!(f, "  cu_early_terminations: {}", self.cu_early_terminations)?;
        writeln!(f, "  cu_split_bound_aborts: {}", self.cu_split_bound_aborts)?;
        writeln!(f, "  angular_exclusions: {}", self.angular_exclusions)?;
        writeln!(f, "  region_class_counts: {:?}", self.region_class_counts)?;
        writeln!(f, "  policy_angular_forced: {}", self.policy_angular_forced)?;
        writeln!(
            f,
            "  policy_angular_guarded: {}",
            self.policy_angular_guarded
        )?;
        writeln!(
            f,
            "  policy_early_term_suppressed: {}",
            self.policy_early_term_suppressed
        )?;
        writeln!(f, "  cu_force_leaf: {}", self.cu_force_leaf)?;
        writeln!(
            f,
            "  tu_split_early_terminations: {}",
            self.tu_split_early_terminations
        )?;
        writeln!(
            f,
            "  tu_neighbor_leaf_skips: {}",
            self.tu_neighbor_leaf_skips
        )?;
        writeln!(
            f,
            "  analysis_cache_cu_records: {}",
            self.analysis_cache_cu_records
        )?;
        writeln!(
            f,
            "  analysis_cache_leaf_records: {}",
            self.analysis_cache_leaf_records
        )?;
        writeln!(
            f,
            "  analysis_cache_tu_records: {}",
            self.analysis_cache_tu_records
        )?;
        writeln!(f, "  rmd_prunes: {}", self.rmd_prunes)?;
        writeln!(
            f,
            "  luma_candidate_expansions: {}",
            self.luma_candidate_expansions
        )?;
        writeln!(
            f,
            "  chroma_candidate_expansions: {}",
            self.chroma_candidate_expansions
        )?;
        writeln!(f, "  final_coded_blocks: {}", self.final_coded_blocks)?;
        writeln!(f, "  trial_coded_blocks: {}", self.trial_coded_blocks)?;
        writeln!(f, "  final_rdoq_blocks: {}", self.final_rdoq_blocks)?;
        writeln!(f, "  trial_rdoq_blocks: {}", self.trial_rdoq_blocks)?;
        writeln!(
            f,
            "  luma_winner_rank_counts: {:?}",
            self.luma_winner_rank_counts
        )?;
        writeln!(
            f,
            "  chroma_winner_rank_counts: {:?}",
            self.chroma_winner_rank_counts
        )?;
        writeln!(
            f,
            "  cu_leaf_wins_by_region: {:?}",
            self.cu_leaf_wins_by_region
        )?;
        writeln!(
            f,
            "  cu_split_wins_by_region: {:?}",
            self.cu_split_wins_by_region
        )?;
        writeln!(
            f,
            "  tu_leaf_wins_by_region: {:?}",
            self.tu_leaf_wins_by_region
        )?;
        writeln!(
            f,
            "  tu_split_wins_by_region: {:?}",
            self.tu_split_wins_by_region
        )?;
        writeln!(f, "  full_rd_close_calls: {}", self.full_rd_close_calls)?;
        writeln!(
            f,
            "  luma_close_call_escalations: {}",
            self.luma_close_call_escalations
        )?;
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

pub(super) fn merge_array<const N: usize>(dst: &mut [u64; N], src: &[u64; N]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

pub(super) fn bump_rank<const N: usize>(counts: &mut [u64; N], rank: usize) {
    if N == 0 {
        return;
    }
    let idx = rank.min(N - 1);
    counts[idx] += 1;
}

/// Coarse wall-time accumulators for the hot inner operations. Only armed when
/// `BPG_PROFILE` is set; otherwise every `enter`/`add` is a predictable
/// `if false` branch. Serial-path (non-parallel tiers) measurement tool.
#[derive(Clone, Copy, Default)]
pub(super) struct Profiler {
    pub(super) on: bool,
    pub(super) snapshot: std::time::Duration,
    pub(super) code_block: std::time::Duration,
    pub(super) residual_bits: std::time::Duration,
    pub(super) rdoq: std::time::Duration,
}

pub(super) struct PlaneSnapshot {
    pub(super) c_idx: u8,
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) data: Vec<u16>,
}

pub(super) struct FrameSnapshot {
    pub(super) planes: Vec<PlaneSnapshot>,
}

#[derive(Clone)]
pub(super) struct MapSnapshot {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) stride: usize,
    pub(super) data: Vec<u8>,
}

/// One CTU's parallel-analysis output: its coding-quadtree plus owned
/// snapshots of the reconstruction and neighbour-map regions to merge into the
/// shared global state after the wavefront step's barrier.
pub(super) struct CtuBuild {
    pub(super) cx: u32,
    pub(super) cy: u32,
    pub(super) cu: CuNode,
    pub(super) fsnap: FrameSnapshot,
    pub(super) msnap: MapSnapshot,
    pub(super) csnap: MapSnapshot,
}
