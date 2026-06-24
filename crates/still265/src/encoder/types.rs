//! Public encoder API types, final-syntax helpers, and lightweight statistics.

use bpg_hevc_decode::hevc::intra::map_chroma_mode_422;
use std::fmt;

use crate::ChromaFormat;

/// `chroma_format_idc` / `ChromaArrayType` (H.265 Table 6-1).
pub(super) fn chroma_array_type(chroma: ChromaFormat) -> u8 {
    match chroma {
        ChromaFormat::Gray => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// Geometry of chroma TB(s) associated with a luma TB.
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
        _ => None,
    }
}

pub(super) fn has_chroma_tb(cat: u8, log2_size: u8) -> bool {
    cat != 0 && (cat == 3 || log2_size >= 3)
}

pub(super) fn decode_second_cbf(cat: u8, log2_size: u8, is_split: bool) -> bool {
    cat == 2 && (!is_split || log2_size == 3)
}

pub(super) fn chroma_pred_mode(cat: u8, chroma_mode: u8) -> u8 {
    if cat == 2 {
        map_chroma_mode_422(chroma_mode)
    } else {
        chroma_mode
    }
}

pub(super) const CTB_LOG2: u8 = 6;
pub(super) const QG_LOG2: u8 = CTB_LOG2 - 1;
pub(super) const MAX_TB_LOG2: u8 = 5;
pub(super) const MIN_TB_LOG2: u8 = 2;
pub(super) const MAX_INTRA_TT_DEPTH: u8 = 2;
pub(super) const CHROMA_DM_IDX: u8 = 4;

/// H.265 §8.6.1 chroma QP derivation.
pub(super) fn chroma_qp_from_luma(qpi: i32, cat: u8) -> i32 {
    if cat != 1 {
        return qpi.min(51);
    }
    static TAB: [i32; 13] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
    if qpi < 30 {
        qpi
    } else if qpi >= 43 {
        qpi - 6
    } else {
        TAB[(qpi - 30) as usize]
    }
}

/// Source planes (full-range YCbCr) for the picture being encoded.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub y: &'a [u16],
    pub cb: &'a [u16],
    pub cr: &'a [u16],
}

/// Encoder statistics. Search-era counters are retained as zero-valued public
/// fields for tooling compatibility while StillSearch grows its own ledger.
#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    pub ctu_count: u64,
    pub cu_trials: u64,
    pub cu_early_terminations: u64,
    pub cu_split_bound_aborts: u64,
    pub floorplus_ctus: u64,
    pub floorplus_repair_attempts: u64,
    pub floorplus_enhanced_leaf_wins: u64,
    pub floorplus_shallow_split_wins: u64,
    pub floorplus_floor_leaf_kept: u64,
    pub floorplus_repair_skips_no_residual: u64,
    pub floorplus_repair_skips_low_cost: u64,
    pub floorplus_bytes_saved_estimate: u64,
    pub floorplus2_ctus: u64,
    pub floorplus2_floor_kept: u64,
    pub floorplus2_repair_attempts: u64,
    pub floorplus2_bids_generated: u64,
    pub floorplus2_bids_executed: u64,
    pub floorplus2_bids_accepted: u64,
    pub floorplus2_bids_rejected: u64,
    pub floorplus2_enhanced_leaf_bids: u64,
    pub floorplus2_enhanced_leaf_wins: u64,
    pub floorplus2_split64_bids: u64,
    pub floorplus2_split64_wins: u64,
    pub floorplus2_child_repair_bids: u64,
    pub floorplus2_child_repair_wins: u64,
    pub floorplus2_repair_skips_no_residual: u64,
    pub floorplus2_repair_skips_low_cost: u64,
    pub floorplus2_odds_mode_k_sum: u64,
    pub floorplus2_odds_mode_k_max: u64,
    pub floorplus2_odds_bid_k_sum: u64,
    pub floorplus2_odds_bid_k_max: u64,
    pub floorplus2_bytes_saved_estimate: u64,
    pub floorshallow_ctus: u64,
    pub floorshallow_repair_attempts: u64,
    pub floorshallow_enhanced_leaf_wins: u64,
    pub floorshallow_enhanced_split_wins: u64,
    pub floorshallow_floor_kept: u64,
    pub floorshallow_repair_skips_no_residual: u64,
    pub floorshallow_repair_skips_low_cost: u64,
    pub floorshallow_bytes_saved_estimate: u64,
    pub cu_force_leaf: u64,
    pub tu_split_early_terminations: u64,
    pub rmd_prunes: u64,
    pub luma_candidate_expansions: u64,
    pub chroma_candidate_expansions: u64,
    pub partnxn_attempts: u64,
    pub partnxn_skips: u64,
    pub partnxn_wins: u64,
    pub partnxn_losses: u64,
    pub partnxn_cu_trials: u64,
    pub partnxn_code_block_calls: u64,
    pub final_coded_blocks: u64,
    pub trial_coded_blocks: u64,
    pub final_rdoq_blocks: u64,
    pub trial_rdoq_blocks: u64,
    pub best_tt_cheap_tu_decisions: u64,
    pub best_tt_escalated_tu_decisions: u64,
    pub best_tt_escalation_changed_winner: u64,
    pub best_tt_full_trial_rdoq_blocks_saved: u64,
    pub best_tt_exact_residual_estimates_saved: u64,
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
    pub bytes_restored: u64,
    pub phase_total_us: u64,
    pub phase_build_us: u64,
    pub phase_parallel_restore_us: u64,
    pub phase_deblock_us: u64,
    pub phase_sao_decide_us: u64,
    pub phase_sao_apply_us: u64,
    pub phase_write_us: u64,
    pub angular_exclusions: u64,
    pub rdo2_angular_exclusion_blocks: u64,
    pub rdo2_angular_game_blocks: u64,
    pub rdo2_angular_iame_blocks: u64,
    pub rdo2_angular_modes_before: u64,
    pub rdo2_angular_modes_after: u64,
    pub rdo2_angular_modes_removed: u64,
    pub policy_angular_forced: u64,
    pub policy_angular_guarded: u64,
    pub policy_early_term_suppressed: u64,
    pub region_class_counts: [u64; crate::preanalysis::NUM_CLASSES],
    pub luma_winner_rank_counts: [u64; 5],
    pub chroma_winner_rank_counts: [u64; 6],
    pub cu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub cu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub partnxn_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
}

impl EncodeStats {
    pub fn merge(&mut self, other: &Self) {
        self.ctu_count += other.ctu_count;
        self.phase_total_us += other.phase_total_us;
        self.phase_build_us += other.phase_build_us;
        self.phase_deblock_us += other.phase_deblock_us;
        self.phase_sao_decide_us += other.phase_sao_decide_us;
        self.phase_sao_apply_us += other.phase_sao_apply_us;
        self.phase_write_us += other.phase_write_us;
    }
}

impl fmt::Display for EncodeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EncodeStats:")?;
        writeln!(f, "  ctu_count: {}", self.ctu_count)?;
        writeln!(f, "  final_coded_blocks: {}", self.final_coded_blocks)?;
        writeln!(f, "  phase_total_us: {}", self.phase_total_us)?;
        writeln!(f, "  phase_build_us: {}", self.phase_build_us)?;
        writeln!(f, "  phase_write_us: {}", self.phase_write_us)?;
        writeln!(f, "  phase_deblock_us: {}", self.phase_deblock_us)?;
        writeln!(f, "  phase_sao_decide_us: {}", self.phase_sao_decide_us)?;
        writeln!(f, "  phase_sao_apply_us: {}", self.phase_sao_apply_us)
    }
}
