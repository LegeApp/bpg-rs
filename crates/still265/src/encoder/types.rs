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
    /// Split-bound aborts by how many children were fully evaluated before
    /// the abort fired (index 1..=3; 0 and 4 are impossible).
    pub cu_split_bound_abort_by_child: [u64; 5],
    /// Per-CU-depth evaluation work, indexed directly by `log2_cb_size`
    /// (CU sizes 8/16/32/64 = log2 3/4/5/6; TU sizes 4/8/16/32 = log2 2/3/4/5).
    /// These mirror x265's `DETAILED_CU_STATS` "Intra RDO calls per depth" so the
    /// two encoders' per-depth work is directly comparable. Printed via
    /// `--debug-stats`.
    pub cu_trials_by_log2: [u64; 7],
    pub cu_splits_taken_by_log2: [u64; 7],
    pub cu_early_term_by_log2: [u64; 7],
    pub tu_leaf_by_log2: [u64; 7],
    pub tu_split_by_log2: [u64; 7],
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
    pub tu_split_bound_aborts: u64,
    pub rmd_prunes: u64,
    pub luma_candidate_expansions: u64,
    pub chroma_candidate_expansions: u64,
    pub partnxn_attempts: u64,
    pub partnxn_skips: u64,
    pub partnxn_wins: u64,
    pub partnxn_losses: u64,
    pub partnxn_cu_trials: u64,
    pub partnxn_code_block_calls: u64,
    /// Final 4:2:2 PartNxN winners whose parent chroma used a second stacked
    /// Cb/Cr coded block flag (`cb1` or `cr1`). This is a StillSearch-specific
    /// correctness/coverage counter, not a legacy RDO stat.
    pub partnxn_422_parent_chroma_second_cbf: u64,
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
    pub chroma_mode_exact_candidates: u64,
    pub chroma_mode_rough_skips: u64,
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
    /// Exact-pass `tt_cost` sum × 1000 (fixed-point) and candidate count,
    /// tracked per mode category.  Reset each frame.  Printed via `--debug-stats`.
    pub luma_exact_planar_tt_cost_x1000: u64,
    pub luma_exact_planar_count: u64,
    pub luma_exact_dc_tt_cost_x1000: u64,
    pub luma_exact_dc_count: u64,
    pub luma_exact_angular_tt_cost_x1000: u64,
    pub luma_exact_angular_count: u64,
    /// Sampled luma oracle diagnostics. Populated only when
    /// `BPG_STILLSEARCH_LUMA_ORACLE=1`.
    pub luma_oracle_samples: u64,
    pub luma_oracle_mode_misses: u64,
    pub luma_oracle_shortlist_hits: u64,
    pub luma_oracle_cheap_top_hits: u64,
    pub luma_oracle_rough_rank_sum: u64,
    pub luma_oracle_cheap_rank_sum: u64,
    pub luma_oracle_cheap_rank_count: u64,
    pub luma_oracle_delta_cost_x1000: i64,
    pub luma_oracle_miss_delta_cost_x1000: i64,
    pub region_class_counts: [u64; crate::preanalysis::NUM_CLASSES],
    pub luma_winner_rank_counts: [u64; 5],
    pub chroma_winner_rank_counts: [u64; 6],
    pub cu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub cu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_leaf_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub tu_split_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    pub partnxn_wins_by_region: [u64; crate::preanalysis::NUM_CLASSES],
    /// Picture-level StillSearch work ledger aggregated from CTU-local ledgers.
    /// Bucket order is internal to StillSearch for now; legacy counters above
    /// remain compatibility-only and new search work should not write rdo2-era
    /// fields.
    pub stillsearch_ledger: [u64; 16],
    /// Optional StillSearch per-bucket wall-clock nanoseconds. Populated only
    /// when `BPG_STILLSEARCH_PROFILE=1`; otherwise all buckets remain zero.
    /// Bucket order matches [`Self::stillsearch_ledger`].
    pub stillsearch_ledger_ns: [u64; 16],
    /// Per-CTU-substage accumulators for `eval_component_8` breakdown.
    /// Populated only when `BPG_STILLSEARCH_PROFILE=1`.
    pub substage_border_ns: u64,
    pub substage_predict_ns: u64,
    pub substage_forward_xform_ns: u64,
    pub substage_quant_ns: u64,
    pub substage_recon_dist_ns: u64,
    pub substage_residual_price_ns: u64,
    pub substage_calls: u64,
}

impl EncodeStats {
    pub fn merge(&mut self, other: &Self) {
        self.ctu_count += other.ctu_count;
        self.cu_trials += other.cu_trials;
        self.cu_early_terminations += other.cu_early_terminations;
        self.cu_split_bound_aborts += other.cu_split_bound_aborts;
        self.tu_split_bound_aborts += other.tu_split_bound_aborts;
        for (dst, src) in self
            .cu_split_bound_abort_by_child
            .iter_mut()
            .zip(other.cu_split_bound_abort_by_child.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .cu_trials_by_log2
            .iter_mut()
            .zip(other.cu_trials_by_log2.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .cu_splits_taken_by_log2
            .iter_mut()
            .zip(other.cu_splits_taken_by_log2.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .cu_early_term_by_log2
            .iter_mut()
            .zip(other.cu_early_term_by_log2.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .tu_leaf_by_log2
            .iter_mut()
            .zip(other.tu_leaf_by_log2.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .tu_split_by_log2
            .iter_mut()
            .zip(other.tu_split_by_log2.iter())
        {
            *dst += *src;
        }
        self.phase_total_us += other.phase_total_us;
        self.phase_build_us += other.phase_build_us;
        self.phase_parallel_restore_us += other.phase_parallel_restore_us;
        self.phase_deblock_us += other.phase_deblock_us;
        self.phase_sao_decide_us += other.phase_sao_decide_us;
        self.phase_sao_apply_us += other.phase_sao_apply_us;
        self.phase_write_us += other.phase_write_us;
        self.frame_snapshots += other.frame_snapshots;
        self.frame_restores += other.frame_restores;
        self.map_snapshots += other.map_snapshots;
        self.map_restores += other.map_restores;
        self.bytes_snapshotted += other.bytes_snapshotted;
        self.bytes_restored += other.bytes_restored;
        for (dst, src) in self
            .stillsearch_ledger
            .iter_mut()
            .zip(other.stillsearch_ledger.iter())
        {
            *dst += *src;
        }
        for (dst, src) in self
            .stillsearch_ledger_ns
            .iter_mut()
            .zip(other.stillsearch_ledger_ns.iter())
        {
            *dst = dst.saturating_add(*src);
        }
        self.luma_exact_planar_tt_cost_x1000 += other.luma_exact_planar_tt_cost_x1000;
        self.luma_exact_planar_count += other.luma_exact_planar_count;
        self.luma_exact_dc_tt_cost_x1000 += other.luma_exact_dc_tt_cost_x1000;
        self.luma_exact_dc_count += other.luma_exact_dc_count;
        self.luma_exact_angular_tt_cost_x1000 += other.luma_exact_angular_tt_cost_x1000;
        self.luma_exact_angular_count += other.luma_exact_angular_count;
        self.luma_oracle_samples += other.luma_oracle_samples;
        self.luma_oracle_mode_misses += other.luma_oracle_mode_misses;
        self.luma_oracle_shortlist_hits += other.luma_oracle_shortlist_hits;
        self.luma_oracle_cheap_top_hits += other.luma_oracle_cheap_top_hits;
        self.luma_oracle_rough_rank_sum += other.luma_oracle_rough_rank_sum;
        self.luma_oracle_cheap_rank_sum += other.luma_oracle_cheap_rank_sum;
        self.luma_oracle_cheap_rank_count += other.luma_oracle_cheap_rank_count;
        self.luma_oracle_delta_cost_x1000 += other.luma_oracle_delta_cost_x1000;
        self.luma_oracle_miss_delta_cost_x1000 += other.luma_oracle_miss_delta_cost_x1000;
        self.chroma_rough_predictions += other.chroma_rough_predictions;
        self.chroma_mode_exact_candidates += other.chroma_mode_exact_candidates;
        self.chroma_mode_rough_skips += other.chroma_mode_rough_skips;
        self.substage_border_ns = self
            .substage_border_ns
            .saturating_add(other.substage_border_ns);
        self.substage_predict_ns = self
            .substage_predict_ns
            .saturating_add(other.substage_predict_ns);
        self.substage_forward_xform_ns = self
            .substage_forward_xform_ns
            .saturating_add(other.substage_forward_xform_ns);
        self.substage_quant_ns = self
            .substage_quant_ns
            .saturating_add(other.substage_quant_ns);
        self.substage_recon_dist_ns = self
            .substage_recon_dist_ns
            .saturating_add(other.substage_recon_dist_ns);
        self.substage_residual_price_ns = self
            .substage_residual_price_ns
            .saturating_add(other.substage_residual_price_ns);
        self.substage_calls += other.substage_calls;
    }
}

impl fmt::Display for EncodeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "EncodeStats:")?;
        writeln!(f, "  ctu_count: {}", self.ctu_count)?;
        writeln!(f, "  final_coded_blocks: {}", self.final_coded_blocks)?;
        writeln!(f, "  phase_total_us: {}", self.phase_total_us)?;
        writeln!(f, "  phase_build_us: {}", self.phase_build_us)?;
        writeln!(
            f,
            "  phase_parallel_restore_us: {}",
            self.phase_parallel_restore_us
        )?;
        writeln!(f, "  phase_write_us: {}", self.phase_write_us)?;
        writeln!(f, "  phase_deblock_us: {}", self.phase_deblock_us)?;
        writeln!(f, "  phase_sao_decide_us: {}", self.phase_sao_decide_us)?;
        writeln!(f, "  phase_sao_apply_us: {}", self.phase_sao_apply_us)?;
        fn avg(n: u64, sum: u64) -> String {
            if n == 0 {
                "N/A".into()
            } else {
                format!("{:.3}", sum as f64 / n as f64 / 1000.0)
            }
        }
        writeln!(
            f,
            "  luma_exact_planar: count={} avg_tt_cost={}",
            self.luma_exact_planar_count,
            avg(
                self.luma_exact_planar_count,
                self.luma_exact_planar_tt_cost_x1000
            )
        )?;
        writeln!(
            f,
            "  luma_exact_dc:     count={} avg_tt_cost={}",
            self.luma_exact_dc_count,
            avg(self.luma_exact_dc_count, self.luma_exact_dc_tt_cost_x1000)
        )?;
        writeln!(
            f,
            "  luma_exact_angular:count={} avg_tt_cost={}",
            self.luma_exact_angular_count,
            avg(
                self.luma_exact_angular_count,
                self.luma_exact_angular_tt_cost_x1000
            )
        )?;
        if self.luma_oracle_samples > 0 {
            let samples = self.luma_oracle_samples as f64;
            let pct = |n: u64| n as f64 / samples * 100.0;
            let avg_delta = self.luma_oracle_delta_cost_x1000 as f64 / samples / 1000.0;
            let avg_miss_delta = if self.luma_oracle_mode_misses == 0 {
                0.0
            } else {
                self.luma_oracle_miss_delta_cost_x1000 as f64
                    / self.luma_oracle_mode_misses as f64
                    / 1000.0
            };
            let avg_rough_rank = self.luma_oracle_rough_rank_sum as f64 / samples;
            let avg_cheap_rank = if self.luma_oracle_cheap_rank_count == 0 {
                f64::NAN
            } else {
                self.luma_oracle_cheap_rank_sum as f64 / self.luma_oracle_cheap_rank_count as f64
            };
            let shortlist_misses = self
                .luma_oracle_samples
                .saturating_sub(self.luma_oracle_shortlist_hits);
            let simple_rdo_misses = self
                .luma_oracle_shortlist_hits
                .saturating_sub(self.luma_oracle_cheap_top_hits);
            writeln!(
                f,
                "  luma_oracle: samples={} misses={} ({:.1}%) shortlist_hits={} ({:.1}%) cheap_top_hits={} ({:.1}%)",
                self.luma_oracle_samples,
                self.luma_oracle_mode_misses,
                pct(self.luma_oracle_mode_misses),
                self.luma_oracle_shortlist_hits,
                pct(self.luma_oracle_shortlist_hits),
                self.luma_oracle_cheap_top_hits,
                pct(self.luma_oracle_cheap_top_hits),
            )?;
            writeln!(
                f,
                "  luma_oracle: shortlist_misses={} ({:.1}%) simple_rdo_misses={} ({:.1}% of admitted)",
                shortlist_misses,
                pct(shortlist_misses),
                simple_rdo_misses,
                if self.luma_oracle_shortlist_hits == 0 {
                    0.0
                } else {
                    simple_rdo_misses as f64 / self.luma_oracle_shortlist_hits as f64 * 100.0
                },
            )?;
            writeln!(
                f,
                "  luma_oracle: avg_rough_rank={:.2} avg_cheap_rank={:.2} avg_delta_cost={:.1} avg_miss_delta_cost={:.1}",
                avg_rough_rank, avg_cheap_rank, avg_delta, avg_miss_delta,
            )?;
        }
        // Per-bucket wall times (only populated when BPG_STILLSEARCH_PROFILE=1).
        let total_ns: u64 = self.stillsearch_ledger_ns.iter().sum();
        if total_ns > 0 {
            const BUCKETS: [&str; 16] = [
                "RoughLuma",
                "LumaCheap",
                "LumaExact",
                "TuLeaf",
                "TuSplit",
                "NxnRough",
                "NxnBatch",
                "ChromaRough",
                "ChromaTrial",
                "Rdoq",
                "RdoqTrial",
                "ResidualPrice",
                "FinalCommit",
                "Writer",
                "Deblock",
                "Sao",
            ];
            writeln!(
                f,
                "  stillsearch_profile (total_wall={:.3}s):",
                total_ns as f64 / 1e9
            )?;
            for (name, (&calls, &ns)) in BUCKETS.iter().zip(
                self.stillsearch_ledger
                    .iter()
                    .zip(self.stillsearch_ledger_ns.iter()),
            ) {
                if ns > 0 || calls > 0 {
                    writeln!(
                        f,
                        "    {:14} calls={:8}  wall={:7.3}s  ({:5.1}%)",
                        name,
                        calls,
                        ns as f64 / 1e9,
                        ns as f64 / total_ns as f64 * 100.0,
                    )?;
                }
            }
            // Substage breakdown (eval_component_8 internals).
            let sub_ns = self.substage_border_ns
                + self.substage_predict_ns
                + self.substage_forward_xform_ns
                + self.substage_quant_ns
                + self.substage_recon_dist_ns
                + self.substage_residual_price_ns;
            if self.substage_calls > 0 && sub_ns > 0 {
                writeln!(f, "  substage_profile ({} calls):", self.substage_calls)?;
                for (name, ns) in [
                    ("BorderBuild", self.substage_border_ns),
                    ("Predict", self.substage_predict_ns),
                    ("ForwardXform", self.substage_forward_xform_ns),
                    ("Quant+SDH", self.substage_quant_ns),
                    ("Recon+Dist", self.substage_recon_dist_ns),
                    ("ResidualPrice", self.substage_residual_price_ns),
                ] {
                    writeln!(
                        f,
                        "    {:14}  wall={:7.3}s  ({:5.1}%)",
                        name,
                        ns as f64 / 1e9,
                        ns as f64 / sub_ns as f64 * 100.0,
                    )?;
                }
            }
        }
        writeln!(
            f,
            "  tu_split_early_terminations: {}",
            self.tu_split_early_terminations
        )?;
        Ok(())
    }
}
