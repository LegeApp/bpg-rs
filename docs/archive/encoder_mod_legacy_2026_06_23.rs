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
mod rdo2;
mod rdo_legacy;
mod snapshot;
mod types;
mod write;

use bpg_hevc_decode::hevc::sao::{apply_sao, SaoMap};
use bpg_hevc_decode::hevc::slice::IntraPredMode;
use bpg_hevc_decode::DecodedFrame;

use crate::analysis_cache::{AnalysisCache, CacheDecisionConfidence};
use crate::effort::{BlockDesc, BlockSearchBudget, EffortTemplate, RmdModeSet, SplitSearch};
use crate::plan::DecisionConfidence;
use crate::sao;
use crate::{nal, params, slice, DeblockMode, Effort, SaoMode, StillHevcConfig};

use self::aq::AqState;
use self::types::*;
pub use self::types::{EncodeStats, Source};
use self::write::{
    build_slice_trees_parallel, build_slice_trees_serial, encode_slice_data, write_slice_from_trees,
};

#[derive(Clone, Copy)]
enum FloorPlusBudget {
    EnhancedLeaf,
    FloorPlus2EnhancedLeaf,
    TerminalChild,
    SlowLeaf,
    SlowDeep16To8,
}

struct Encoder<'a> {
    display_width: u32,
    display_height: u32,
    cat: u8,
    qp_y: i32,
    qp_c: i32,
    bit_depth: u8,
    effort: Effort,
    effort_template: EffortTemplate,
    /// Tile partition geometry. Single-tile grid when tiles are disabled (the
    /// default), so tile-scan order is plain raster and availability is
    /// unrestricted. Drives tile-scan encode order + cross-tile availability,
    /// matching the decoder.
    tile_grid: bpg_hevc_decode::hevc::tile::TileGrid,
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
    /// Best-only, env-gated angular-mode exclusion before rough SATD scoring.
    best_angular_exclusion: rdo2::angular_exclusion::AngularExclusionConfig,
    /// Pooled buffers for the staged `rdo2` TU search (refactor-rdo.md).
    search_scratch: rdo2::scratch::SearchScratch,
    /// Gate: route the transform-tree decision through the new `rdo2` staged
    /// engine (`BPG_RDO2_TU`). Default on for Best.
    rdo2_tu: bool,
    /// Gate: screen luma mode candidates with cheap trials, then exact-recheck
    /// close top-two decisions (`BPG_RDO2_LUMA`). Default on for Best.
    rdo2_luma: bool,
    /// Gate: use non-committing scratch block eval for Best luma leaf-screen
    /// candidate ranking (`BPG_RDO2_LUMA_SCRATCH`). Default on for Best luma.
    rdo2_luma_scratch: bool,
    /// Multiplier for rdo2 luma close-call exact rechecks
    /// (`BPG_RDO2_LUMA_CLOSE_MULT`). Default 2.0 for Best, 1.0 otherwise.
    rdo2_luma_close_mult: f64,
    /// Temporary fallback: use the legacy materializing exact luma escalation
    /// even when the scratch luma screen is active
    /// (`BPG_RDO2_LUMA_LEGACY_ESCALATE`). Default off.
    rdo2_luma_legacy_escalate: bool,
    /// Gate: screen chroma mode RD candidates with cheap trials, exact-recheck
    /// close calls, and re-code the winner exactly (`BPG_RDO2_CHROMA`). Default off.
    rdo2_chroma: bool,
    /// Gate: when `BPG_RDO2_CHROMA` is active, use the rdo2 non-committing
    /// evaluator for one-block chroma RD candidate costs
    /// (`BPG_RDO2_CHROMA_SCRATCH`). Stacked 4:2:2 chroma remains on the
    /// materializing fallback until scratch-overlay prediction exists.
    rdo2_chroma_scratch: bool,
    /// Gate: bound the 8x8 `PartNxN` search by the exact `Part2Nx2N` cost,
    /// aborting (skipping the remaining PUs) once the accumulated per-PU luma RD
    /// cost — a true lower bound on the full `PartNxN` cost — can no longer beat
    /// it (`BPG_RDO2_NXN`). Exact, no quality loss. Default on for Best.
    rdo2_nxn: bool,
    /// Gate (Phase 8): when set, `rdo2_analyze_tt` screens the luma+chroma
    /// transform tree with a **native** rdo2 recursion that costs each leaf
    /// through `rdo2_eval_leaf_block` (explicit `EvalKind::CheapTrial` policy)
    /// instead of re-entering the legacy recursive `build_tt` under the implicit
    /// `best_tt_cheap_trial` flag (`BPG_RDO2_TT_NATIVE`). Default on for Best:
    /// validated byte-identical to the legacy screen across 4:2:0/4:2:2/4:4:4,
    /// multiple QPs, and 2000x1500, so the structure decision and exact final
    /// replay are unchanged. Escape with `BPG_RDO2_TT_NATIVE=0`.
    rdo2_tt_native: bool,
    /// Re-entry guard: true while inside `rdo2_analyze_tt`'s cheap screen, so the
    /// recursive `build_tt` calls run the legacy path (cheap) instead of
    /// re-entering the gate.
    in_rdo2: bool,
    /// Internal Phase-6 rdo2 optimization: while replaying the root CTU winner,
    /// final residual syntax will be emitted by the writer, so analysis can skip
    /// the duplicate exact residual bit estimate.
    elide_final_residual_pricing: bool,
    analysis: std::sync::Arc<crate::preanalysis::AnalysisMaps>,
    analysis_cache: AnalysisCache,
    cur_policy: Option<crate::preanalysis::SearchPolicy>,
    cur_qp_y: i32,
    cur_qp_c: i32,
    luma_trial_quality_override: Option<crate::effort::TrialQuality>,
    floorplus_budget: Option<FloorPlusBudget>,
    floorplus_repair_bits_bpp: f64,
    floorplus_repair_dist_px: f64,
    /// Slow sweep knobs (env `BPG_SLOW_REPAIR_THRESH`, `BPG_SLOW_LUMA_CANDS`):
    /// D/E worst-child & deep-island save-frac threshold (F uses half), and the
    /// SlowLeaf/SlowDeep luma RD candidate count. Defaults reproduce the current
    /// behaviour (0.01, 4).
    slow_repair_thresh: f64,
    slow_luma_cands: u8,
    /// SlowPlus spends one extra luma RD candidate by default, but remains
    /// env-tunable independently so Slow's baseline stays frozen.
    slowplus_luma_cands: u8,
    /// SlowPlus skips the root 64×64 leaf trial below QP35 (x265's all-intra
    /// low-QP root rule). Default **off**: a 2026-06-22 ablation on native
    /// 12/50 MP photos at QP28 showed it costs bytes (+0.8% at 12 MP, +7.3% at
    /// 50 MP) with no PSNR gain — on smooth high-res content the big-CU leaf is
    /// frequently optimal, so forcing the 64→32 split wastes split-flag bits.
    /// Kept env-tunable (`BPG_SLOWPLUS_SKIP_ROOT_LEAF=1`) for revisiting at
    /// higher QP, where the heuristic may yet pay off.
    slowplus_skip_root_leaf: bool,
    /// SlowPlus: on a close-call 64-leaf vs 64→32-split decision in
    /// `decide_cu_slow_64`, re-price both branches with the exact final-replay
    /// coder (RDOQ recon + serial-CABAC bits) before committing — the same
    /// fidelity Best uses. The shallow auction otherwise prices the two branches
    /// with plain-quant recon and frozen-context approximate bits.
    ///
    /// Default **off**. A 2026-06-23 ablation (QP28, native photos) showed this
    /// does **not** close the gap to Best and is a content-dependent wash: it
    /// fires on ~20% of CTUs and flips ~16% of those, but the flips are
    /// greedy-local (priced on a frozen cross-CTU CABAC context) so they help
    /// only marginally on smooth high-res (50 MP −0.10%) while slightly *hurting*
    /// the textured mid-res content where Best actually RD-dominates (4 MP +0.04%,
    /// 12 MP +0.01%). Conclusion: the ~12% gap to Best is **not** in the
    /// top-level leaf-vs-split decision fidelity. Kept env-tunable
    /// (`BPG_SLOWPLUS_CU_ESCALATE=1`) for the smooth-content case and future work.
    slowplus_cu_escalate: bool,
    /// SlowPlus: let the D (worst-child) and E (env-island) 32→16 repair splits in
    /// `decide_cu_slow_64` use the `SlowDeep16To8` budget, so their 16×16
    /// grandchildren can recurse to 8×8 and evaluate 4×4 NxN PU (the F island
    /// already does this). Default **on**. A 2026-06-23 stat diff vs Best isolated
    /// SlowPlus's ~12% size gap to under-reaching depth/NxN (Best codes ~2.5× more
    /// CUs, wins 4×4 NxN ~9× more); extending island depth is a clean Pareto win at
    /// QP28 (4 MP −2.9%/+0.11 dB, 12 MP −2.5%/+0.10 dB, 50 MP −1.1%/+0.05 dB). Slow
    /// keeps leaf-only islands (baseline frozen); disable via
    /// `BPG_SLOWPLUS_DEEP_REPAIR=0`.
    slowplus_deep_repair: bool,
    floorplus_split: bool,
    floorplus_tu: bool,
    floorplus_modes: RmdModeSet,
    floorplus2_repair_bits_bpp: f64,
    floorplus2_repair_dist_px: f64,
    floorplus2_split: bool,
    floorplus2_tu: bool,
    floorplus2_modes: RmdModeSet,
    floorplus2_mode_limit: u8,
    floorplus2_max_bids_per_ctu: usize,
    floorplus2_max_accepted_bids_per_ctu: usize,
    floorplus2_odds_bid_threshold: f64,
    floorplus2_odds_mode_threshold: f64,
    floorplus2_child_repair: bool,
    best2_chroma_gate: Option<f64>,
    best2_chroma_protect: bool,
    best2_luma_fastrd: bool,
    best2_rough_lambda: bool,
    best2_rd_lambda: bool,
    best2_rdoq2: bool,
    best_trial_approx_bits: bool,
    best_trial_rdoq_gate: BestTrialRdoqGate,
    best_tt_close_escalation: bool,
    best_tt_escalation_margin: f64,
    best_tt_cheap_trial: bool,
    best_tt_exact_trial: bool,
    sign_data_hiding: bool,
    best2_wpp: bool,
    best_luma_leaf_screen: bool,
    best_tu_neighbor_limit: bool,
    /// Experimental speed lever (`BPG_SMOOTH_TU_LEAF`): when `Some(min_log2)`,
    /// large luma TUs (`log2_size >= min_log2`) in flat/smooth regions skip the
    /// split-subtree evaluation and commit the leaf directly. Targets the
    /// measured smooth-region TU over-split (62–69% of smooth cells use a smaller
    /// TU than x265, for ~no byte benefit) — the split evaluation is wasted work.
    /// Default `None` (off).
    smooth_tu_leaf_min_log2: Option<u8>,
    /// `Some((perceptual, strength))` when the experimental bidirectional
    /// variance AQ is active for `Best` (drives [`Encoder::aq_qg_target`]).
    best_aq: Option<(bool, f32)>,
    /// Gate (`BPG_BEST2_TT_REUSE=1`, Best, non-AQ): decide the transform-tree
    /// leaf-vs-split structure directly on RDOQ (exact) cost in the native screen
    /// and keep that winning `Tt`, instead of deciding on a cheap plain-quant
    /// screen and then re-coding the winner in a separate final RDOQ replay.
    /// Output-changing (RDOQ-based structure decision); validate via BD sweep.
    best2_tt_reuse: bool,
    /// Transient: true only while the native TT screen is running under
    /// `best2_tt_reuse`, so the screen leaf coders code at `EvalKind::Final`
    /// (committed RDOQ) instead of `EvalKind::CheapTrial`.
    tt_screen_final: bool,
    best2_cu_reuse: bool,
    /// Experimental: evaluate 8x8 `PartNxN` (four 4x4 luma PUs) against the
    /// normal `Part2Nx2N` CU and pick the RD winner. Env-gated (`BPG_PARTNXN`);
    /// only effective on the winner-direct `best2_cu_reuse` path.
    partnxn: bool,
    partnxn_prune: PartNxnPrune,
    /// Gate (`BPG_NXN_PU_RDOQ_TOP=K`): in `build_cu_leaf_nxn`, RDOQ-price only the
    /// top-`K` SATD-ranked candidate modes per 4x4 PU instead of every candidate,
    /// then pick the RDOQ winner among those. The rough search already ranks the
    /// candidate list by SATD, so this reuses that ordering to cut the dominant
    /// `NxnPuExact` RDOQ volume. `None` = price every candidate (default,
    /// byte-identical). Output-changing when set; BD-validate.
    nxn_pu_rdoq_top: Option<usize>,
    /// PartNxN per-PU screen diagnostic gates, parsed once at construction
    /// instead of via `std::env::var` inside the per-sub-PU hot loop
    /// (`BPG_NXN_CLOSE_MULT`/`BPG_NXN_EXACT`/`BPG_NXN_ADAPTIVE`/`BPG_NXN_APPROX`).
    nxn_close_mult: f64,
    nxn_exact: bool,
    nxn_adaptive: bool,
    nxn_approx: bool,
    /// PartNxN winner carry-forward (`BPG_NXN_CARRY=1`, default-off). When set,
    /// the default full-RDOQ PartNxN screen retains the winning PU's quantised
    /// coefficients and commits them via `commit_nxn_pu_luma` instead of
    /// re-RDOQ'ing the winner at `EvalKind::Final` — byte-identical, removes the
    /// redundant per-PU forward-transform + RDOQ (~514K blocks at 4K).
    nxn_carry: bool,
    /// Diagnostic gate (`BPG_CU_EARLY_DIAG=1`): in `decide_cu`, compute a
    /// candidate normal-QP force-leaf predicate and log how often it would fire
    /// (and how often it would be wrong) per CU size, without changing the
    /// decision. Used to confirm the resolution-scaling lever before acting.
    cu_early_diag: bool,
    /// Diagnostic gate (`BPG_CABAC_CTX_DIAG=1`): compare frozen-entry CU split
    /// pricing with serial sibling-context syntax pricing. Read-only.
    cabac_ctx_diag: bool,
    /// Best-only CU early termination: apply the conservative force-leaf
    /// predicate only at 16x16 CUs. The larger 32x32/64x64 force-leaf cases had
    /// high diagnostic mistake rates on real photos.
    best_cu_early_16: bool,
    /// Experimental Best-only force-split (`BPG_BEST_CU_FORCE_SPLIT=edge64-lowqp`
    /// or `edge-lowqp`): skip the large-CU leaf path in edge/text/chroma-critical
    /// regions at QP <= 28. `Some(6)` = 64x64 only; `Some(5)` = 32x32+64x64.
    best_cu_force_split_edge_lowqp_min_log2: Option<u8>,
    /// Best-only x265-parity rule: below QP35, skip the non-split 64x64 intra
    /// CU candidate and always descend to 32x32 children at max CU size. x265's
    /// normal intra path does not call `checkIntra()` when
    /// `log2CUSize == MAX_LOG2_CU_SIZE`; still265 keeps the slower 64x64 leaf
    /// option at QP35+ because the measured high-QP size penalty was larger.
    best_cu_no_64_leaf: bool,
    /// Best-only zero-residual force-leaf at 32x32 (`BPG_BEST_CU_ZERO_LEAF_32=1`).
    /// When a 32x32 cheap leaf has no coded residual and is not in a text/edge/
    /// chroma-critical region, skip the split comparison. Split can only win via
    /// mode/syntax tradeoffs, not distortion improvement. Diagnostic confirmed
    /// <5% mistake rates across QP 24-36 on real 4K photos; population grows to
    /// ~30% of 32x32 evals at QP36 with near-zero mistake rates.
    best_cu_zero_leaf_32: bool,
    /// Stage-2 top-down rough-SATD skip-leaf gate (`BPG_BEST_CU_ROUGH_SKIP`):
    /// `Some(min_log2)` skips the large-CU leaf RDO and commits straight to split
    /// for fully-inside CUs at `log2_cb_size >= min_log2` whose rough parent-vs-
    /// children probe predicts a split (see `cu_rough_skip_leaf`, threshold
    /// `cu_rough_m_split`). `BPG_BEST_CU_ROUGH_SKIP=32` -> `Some(5)`, `=16` ->
    /// `Some(4)`; unset / `off` / `0` -> `None` (no behavior change). Validated at
    /// 32x32; the diagnostic rejected force-leaf and 16x16 skip. Default-off
    /// pending a BD sweep.
    best_cu_rough_skip_min_log2: Option<u8>,
    /// Stage-2 source-activity force-split gate (`BPG_BEST_CU_SRC_SPLIT`):
    /// `Some(min_log2)` skips the large-CU leaf RDO and commits to split for
    /// fully-inside CUs at `log2_cb_size >= min_log2` whose cheap source
    /// between-child variance fraction clears `cu_src_t_split` (see
    /// `cu_src_force_split`). Unlike the rough probe this is faithful (source-only)
    /// and cheap (one variance pass). `=32`->`Some(5)`, `=16`->`Some(4)`; unset /
    /// `off` / `0` -> `None`. Default-off pending a BD sweep; ~12% irreducible
    /// mistake floor at 32x32, so expect a real (small) BD cost.
    best_cu_src_split_min_log2: Option<u8>,
    /// Best-only experimental tunable cheap-leaf early termination
    /// (`BPG_CU_EARLY_K`, `BPG_CU_EARLY_MIN_LOG2`). When `k > 0`, a CU at
    /// `log2_cb_size >= min_log2` is forced to leaf (the split descent is skipped)
    /// when it is not in a text/edge/chroma-critical region and `bits*k < area`.
    /// Unlike the k=16 16x16 rule (`best_cu_early_16`) and the QP>=38-gated effort
    /// rule, this fires at all QPs and all sizes >= min_log2 — the resolution-
    /// scaling lever. `k=8` mirrors the x265 Balanced threshold. Default-ON at
    /// `k=8` for Best when the frame is >= 4 MP (resolution-gated; the trade is a
    /// pure BD loss at low res), `0` (disabled) otherwise. `BPG_CU_EARLY_K`
    /// overrides explicitly, `=0` force-disables.
    best_cu_early_term_k: u64,
    best_cu_early_term_min_log2: u8,
    /// QP floor for the tunable cheap-leaf knob (`BPG_CU_EARLY_MIN_QP`). The
    /// diagnostic shows 32x32 split-win rate falls with QP (49%@28 -> 22%@36), so
    /// the aggressive predicate is only safe at the high end of Best's range.
    /// `0` = fire at all QPs (default).
    best_cu_early_term_min_qp: i32,
    /// Thresholds (percent ratio) for the diagnostic top-down rough-SATD split
    /// predictor in `cu_early_diag_log`. `cu_rough_m_leaf`: rough-force-leaf
    /// fires when `children_rough * 100 >= parent_rough * m_leaf` (children fail
    /// to beat the parent by enough). `cu_rough_m_split`: rough-skip-leaf fires
    /// when `children_rough * 100 <= parent_rough * m_split` (children much
    /// cheaper). Parsed once from `BPG_CU_ROUGH_M_LEAF` / `BPG_CU_ROUGH_M_SPLIT`;
    /// only read when `cu_early_diag` is set, so default encodes never probe.
    cu_rough_m_leaf: u64,
    cu_rough_m_split: u64,
    /// Percent thresholds for the diagnostic source-activity split predictor
    /// (`cu_src_activity` in cu.rs). `cu_src_t_leaf`: source-leaf fires when the
    /// between-child variance fraction `between_dev * 100 <= parent_dev * t_leaf`
    /// (homogeneous block). `cu_src_t_split`: source-split fires when
    /// `between_dev * 100 >= parent_dev * t_split` on a non-flat parent
    /// (structure at this scale). From `BPG_CU_SRC_T_LEAF` / `BPG_CU_SRC_T_SPLIT`;
    /// only read when `cu_early_diag` is set.
    cu_src_t_leaf: u64,
    cu_src_t_split: u64,
    aq: AqState,
    prof: Profiler,
    trace: crate::trace::SearchTrace,
    stats: EncodeStats,
}

impl<'a> Encoder<'a> {
    fn with_tt_trial_flags<R>(
        &mut self,
        cheap_trial: bool,
        exact_trial: bool,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        debug_assert!(
            !(cheap_trial && exact_trial),
            "TU trial mode cannot be both cheap and exact"
        );
        let prev_cheap = self.best_tt_cheap_trial;
        let prev_exact = self.best_tt_exact_trial;
        self.best_tt_cheap_trial = cheap_trial;
        self.best_tt_exact_trial = exact_trial;
        let result = f(self);
        self.best_tt_cheap_trial = prev_cheap;
        self.best_tt_exact_trial = prev_exact;
        result
    }

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
            tile_grid: self.tile_grid.clone(),
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
            best_angular_exclusion: self.best_angular_exclusion,
            search_scratch: rdo2::scratch::SearchScratch::default(),
            rdo2_tu: self.rdo2_tu,
            rdo2_luma: self.rdo2_luma,
            rdo2_luma_scratch: self.rdo2_luma_scratch,
            rdo2_luma_close_mult: self.rdo2_luma_close_mult,
            rdo2_luma_legacy_escalate: self.rdo2_luma_legacy_escalate,
            rdo2_chroma: self.rdo2_chroma,
            rdo2_chroma_scratch: self.rdo2_chroma_scratch,
            rdo2_nxn: self.rdo2_nxn,
            rdo2_tt_native: self.rdo2_tt_native,
            in_rdo2: false,
            elide_final_residual_pricing: false,
            analysis: self.analysis.clone(),
            analysis_cache: self.analysis_cache.empty_like(),
            cur_policy: self.cur_policy,
            cur_qp_y: self.cur_qp_y,
            cur_qp_c: self.cur_qp_c,
            luma_trial_quality_override: None,
            floorplus_budget: None,
            floorplus_repair_bits_bpp: self.floorplus_repair_bits_bpp,
            floorplus_repair_dist_px: self.floorplus_repair_dist_px,
            slow_repair_thresh: self.slow_repair_thresh,
            slow_luma_cands: self.slow_luma_cands,
            slowplus_luma_cands: self.slowplus_luma_cands,
            slowplus_skip_root_leaf: self.slowplus_skip_root_leaf,
            slowplus_cu_escalate: self.slowplus_cu_escalate,
            slowplus_deep_repair: self.slowplus_deep_repair,
            floorplus_split: self.floorplus_split,
            floorplus_tu: self.floorplus_tu,
            floorplus_modes: self.floorplus_modes,
            floorplus2_repair_bits_bpp: self.floorplus2_repair_bits_bpp,
            floorplus2_repair_dist_px: self.floorplus2_repair_dist_px,
            floorplus2_split: self.floorplus2_split,
            floorplus2_tu: self.floorplus2_tu,
            floorplus2_modes: self.floorplus2_modes,
            floorplus2_mode_limit: self.floorplus2_mode_limit,
            floorplus2_max_bids_per_ctu: self.floorplus2_max_bids_per_ctu,
            floorplus2_max_accepted_bids_per_ctu: self.floorplus2_max_accepted_bids_per_ctu,
            floorplus2_odds_bid_threshold: self.floorplus2_odds_bid_threshold,
            floorplus2_odds_mode_threshold: self.floorplus2_odds_mode_threshold,
            floorplus2_child_repair: self.floorplus2_child_repair,
            best2_chroma_gate: self.best2_chroma_gate,
            best2_chroma_protect: self.best2_chroma_protect,
            best2_luma_fastrd: self.best2_luma_fastrd,
            best2_rough_lambda: self.best2_rough_lambda,
            best2_rd_lambda: self.best2_rd_lambda,
            best2_rdoq2: self.best2_rdoq2,
            best_trial_approx_bits: self.best_trial_approx_bits,
            best_trial_rdoq_gate: self.best_trial_rdoq_gate,
            best_tt_close_escalation: self.best_tt_close_escalation,
            best_tt_escalation_margin: self.best_tt_escalation_margin,
            best_tt_cheap_trial: false,
            best_tt_exact_trial: false,
            sign_data_hiding: self.sign_data_hiding,
            best2_wpp: self.best2_wpp,
            best_luma_leaf_screen: self.best_luma_leaf_screen,
            best_tu_neighbor_limit: self.best_tu_neighbor_limit,
            smooth_tu_leaf_min_log2: self.smooth_tu_leaf_min_log2,
            best_aq: self.best_aq,
            best2_tt_reuse: self.best2_tt_reuse,
            tt_screen_final: false,
            best2_cu_reuse: self.best2_cu_reuse,
            partnxn: self.partnxn,
            partnxn_prune: self.partnxn_prune,
            nxn_pu_rdoq_top: self.nxn_pu_rdoq_top,
            nxn_close_mult: self.nxn_close_mult,
            nxn_exact: self.nxn_exact,
            nxn_adaptive: self.nxn_adaptive,
            nxn_approx: self.nxn_approx,
            nxn_carry: self.nxn_carry,
            cu_early_diag: self.cu_early_diag,
            cabac_ctx_diag: self.cabac_ctx_diag,
            best_cu_early_16: self.best_cu_early_16,
            best_cu_force_split_edge_lowqp_min_log2: self.best_cu_force_split_edge_lowqp_min_log2,
            best_cu_no_64_leaf: self.best_cu_no_64_leaf,
            best_cu_zero_leaf_32: self.best_cu_zero_leaf_32,
            best_cu_rough_skip_min_log2: self.best_cu_rough_skip_min_log2,
            best_cu_src_split_min_log2: self.best_cu_src_split_min_log2,
            best_cu_early_term_k: self.best_cu_early_term_k,
            best_cu_early_term_min_log2: self.best_cu_early_term_min_log2,
            best_cu_early_term_min_qp: self.best_cu_early_term_min_qp,
            cu_rough_m_leaf: self.cu_rough_m_leaf,
            cu_rough_m_split: self.cu_rough_m_split,
            cu_src_t_leaf: self.cu_src_t_leaf,
            cu_src_t_split: self.cu_src_t_split,
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
        let mut budget = self
            .effort_template
            .resolve(self.block_desc(x0, y0, log2_size, c_idx));
        match self.floorplus_budget {
            Some(FloorPlusBudget::EnhancedLeaf) => {
                budget.rmd_mode_set = self.floorplus_modes;
                budget.luma_rd_candidates_base = 2;
                budget.luma_rd_candidates = 2;
                budget.chroma_rd_candidates_base = 0;
                budget.chroma_rd_candidates = 0;
                budget.cu_split = SplitSearch::ForceLeaf;
                budget.tu_split = if self.floorplus_tu {
                    SplitSearch::EvaluateBoth
                } else {
                    SplitSearch::ForceLeaf
                };
            }
            Some(FloorPlusBudget::FloorPlus2EnhancedLeaf) => {
                budget.rmd_mode_set = self.floorplus2_modes;
                budget.luma_rd_candidates_base = self.floorplus2_mode_limit;
                budget.luma_rd_candidates = self.floorplus2_mode_limit;
                budget.chroma_rd_candidates_base = 0;
                budget.chroma_rd_candidates = 0;
                budget.cu_split = SplitSearch::ForceLeaf;
                budget.tu_split = if self.floorplus2_tu {
                    SplitSearch::EvaluateBoth
                } else {
                    SplitSearch::ForceLeaf
                };
            }
            Some(FloorPlusBudget::TerminalChild) => {
                budget.rmd_mode_set = RmdModeSet::MpmPlanarDcOnly;
                budget.luma_rd_candidates_base = 1;
                budget.luma_rd_candidates = 1;
                budget.chroma_rd_candidates_base = 0;
                budget.chroma_rd_candidates = 0;
                budget.cu_split = SplitSearch::ForceLeaf;
                budget.tu_split = SplitSearch::ForceLeaf;
            }
            Some(FloorPlusBudget::SlowLeaf) => {
                let t = crate::effort::template(self.effort);
                let cands = if self.effort == Effort::SlowPlus {
                    self.slowplus_luma_cands
                } else {
                    self.slow_luma_cands
                };
                budget.rmd_mode_set = t.rmd.mode_set;
                budget.luma_rd_candidates_base = cands;
                budget.luma_rd_candidates = cands;
                budget.chroma_rd_candidates_base = t.chroma.max_rd_candidates;
                budget.chroma_rd_candidates = t.chroma.max_rd_candidates;
                budget.cu_split = SplitSearch::ForceLeaf;
                budget.tu_split = SplitSearch::EvaluateBoth;
            }
            Some(FloorPlusBudget::SlowDeep16To8) => {
                let t = crate::effort::template(self.effort);
                let cands = if self.effort == Effort::SlowPlus {
                    self.slowplus_luma_cands
                } else {
                    self.slow_luma_cands
                };
                budget.rmd_mode_set = t.rmd.mode_set;
                budget.luma_rd_candidates_base = cands;
                budget.luma_rd_candidates = cands;
                budget.chroma_rd_candidates_base = t.chroma.max_rd_candidates;
                budget.chroma_rd_candidates = t.chroma.max_rd_candidates;
                budget.cu_split = if log2_size == 4 {
                    SplitSearch::EvaluateBoth
                } else {
                    SplitSearch::ForceLeaf
                };
                budget.tu_split = SplitSearch::EvaluateBoth;
            }
            None => {}
        }
        budget
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

    /// Whether luma pixel `(ax, ay)` lies in the same tile as `(bx, by)`. Always
    /// true when tiles are disabled (the common case).
    fn same_tile_px(&self, ax: u32, ay: u32, bx: u32, by: u32) -> bool {
        self.tile_grid.same_tile_px(ax, ay, bx, by, CTB_LOG2)
    }

    /// Plane subsampling shifts `(x, y)` for component `c_idx` under the active
    /// chroma array type. Delegates to the shared
    /// [`bpg_hevc_decode::hevc::tile::plane_shifts`] so encoder and decoder agree.
    fn plane_shifts(&self, c_idx: u8) -> (u8, u8) {
        bpg_hevc_decode::hevc::tile::plane_shifts(c_idx, self.cat)
    }

    /// Plane-pixel tile bounds `(x0, y0, x1, y1)` (exclusive upper) for the block
    /// at plane coords `(x, y)` in `c_idx`, or `None` when tiles are disabled.
    /// Reference samples outside these bounds belong to a different tile and must
    /// be forced unavailable so the encoder's reconstruction matches the decoder.
    fn tile_clamp_bounds(&self, x: u32, y: u32, c_idx: u8) -> Option<(u32, u32, u32, u32)> {
        if self.tile_grid.is_single() {
            return None;
        }
        let (sx, sy) = self.plane_shifts(c_idx);
        Some(self.tile_grid.tile_plane_bounds(x, y, CTB_LOG2, sx, sy))
    }

    fn split_ctx_inc(&self, x0: u32, y0: u32, ct_depth: u8) -> usize {
        let mut inc = 0usize;
        // Neighbours across a tile boundary are unavailable for context
        // derivation (matches the decoder).
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

    fn tu_neighbor_prior_pair(&self, x0: u32, y0: u32) -> (Option<u8>, Option<u8>) {
        let left = (x0 > 0).then(|| self.tu_depth_at(x0 - 1, y0)).flatten();
        let above = (y0 > 0).then(|| self.tu_depth_at(x0, y0 - 1)).flatten();
        (left, above)
    }

    fn tu_neighbor_state_index(depth: Option<u8>) -> usize {
        match depth {
            None => 0,
            Some(0) => 1,
            Some(_) => 2,
        }
    }

    /// Speed lever ([`Self::smooth_tu_leaf_min_log2`]): force a large luma TU in a
    /// flat/smooth region to its leaf, skipping the split-subtree evaluation. The
    /// decoder-derived TU map showed still265 splits 62–69% of smooth cells below
    /// x265 for ~no byte benefit, so the split RD work there is largely wasted.
    fn should_force_smooth_tu_leaf(&self, x0: u32, y0: u32, log2_size: u8) -> bool {
        let Some(min_log2) = self.smooth_tu_leaf_min_log2 else {
            return false;
        };
        if log2_size < min_log2 {
            return false;
        }
        matches!(
            self.analysis.region_class_at(x0, y0, log2_size),
            crate::preanalysis::RegionClass::Flat | crate::preanalysis::RegionClass::Gradient
        )
    }

    fn should_limit_tu_to_neighbor_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        leaf: &Tt,
    ) -> bool {
        let i = log2_size as usize;
        if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
            self.stats.tu_neighbor_limit_calls_by_log2[i] += 1;
        }
        if !self.best_tu_neighbor_limit || log2_size < 4 {
            if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                if !self.best_tu_neighbor_limit {
                    self.stats.tu_neighbor_limit_disabled_by_log2[i] += 1;
                } else {
                    self.stats.tu_neighbor_limit_small_by_log2[i] += 1;
                }
            }
            return false;
        }
        let (left_prior, above_prior) = self.tu_neighbor_prior_pair(x0, y0);
        if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
            let combo = Self::tu_neighbor_state_index(left_prior) * 3
                + Self::tu_neighbor_state_index(above_prior);
            self.stats.tu_neighbor_limit_prior_combo_by_log2[i][combo] += 1;
        }
        match left_prior.into_iter().chain(above_prior).max() {
            Some(0) => {
                if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                    self.stats.tu_neighbor_limit_prior_leaf_by_log2[i] += 1;
                }
            }
            Some(_) => {
                if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                    self.stats.tu_neighbor_limit_prior_split_by_log2[i] += 1;
                }
                return false;
            }
            None => {
                if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                    self.stats.tu_neighbor_limit_prior_none_by_log2[i] += 1;
                }
                return false;
            }
        }
        if tt_has_residual(leaf) {
            if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                self.stats.tu_neighbor_limit_residual_reject_by_log2[i] += 1;
            }
            return false;
        }
        if !matches!(
            self.analysis.region_class_at(x0, y0, log2_size),
            crate::preanalysis::RegionClass::Flat | crate::preanalysis::RegionClass::Gradient
        ) {
            if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
                self.stats.tu_neighbor_limit_region_reject_by_log2[i] += 1;
            }
            return false;
        }
        if i < self.stats.tu_neighbor_limit_calls_by_log2.len() {
            self.stats.tu_neighbor_limit_accept_by_log2[i] += 1;
        }
        true
    }

    fn record_tu_neighbor_mixed_leaf_diag(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        leaf: &Tt,
        split_won: bool,
    ) {
        if !self.best_tu_neighbor_limit || log2_size < 4 {
            return;
        }
        let (left_prior, above_prior) = self.tu_neighbor_prior_pair(x0, y0);
        let mixed = matches!(
            (
                Self::tu_neighbor_state_index(left_prior),
                Self::tu_neighbor_state_index(above_prior),
            ),
            (1, 2) | (2, 1)
        );
        if !mixed || tt_has_residual(leaf) {
            return;
        }
        if !matches!(
            self.analysis.region_class_at(x0, y0, log2_size),
            crate::preanalysis::RegionClass::Flat | crate::preanalysis::RegionClass::Gradient
        ) {
            return;
        }
        let i = log2_size as usize;
        if i >= self.stats.tu_neighbor_mixed_leaf_pred_by_log2.len() {
            return;
        }
        self.stats.tu_neighbor_mixed_leaf_pred_by_log2[i] += 1;
        if split_won {
            self.stats.tu_neighbor_mixed_leaf_mistake_by_log2[i] += 1;
        }
    }

    fn record_tu_neighbor_both_split_diag(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        split_won: bool,
    ) {
        if !self.best_tu_neighbor_limit || log2_size < 4 {
            return;
        }
        let (left_prior, above_prior) = self.tu_neighbor_prior_pair(x0, y0);
        let both_split = matches!(
            (
                Self::tu_neighbor_state_index(left_prior),
                Self::tu_neighbor_state_index(above_prior),
            ),
            (2, 2)
        );
        if !both_split {
            return;
        }
        let i = log2_size as usize;
        if i >= self.stats.tu_neighbor_both_split_pred_by_log2.len() {
            return;
        }
        self.stats.tu_neighbor_both_split_pred_by_log2[i] += 1;
        if !split_won {
            self.stats.tu_neighbor_both_split_mistake_by_log2[i] += 1;
        }
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
        // Left neighbour is unavailable across a tile boundary (the above
        // neighbour's CTB-row guard already covers tile row boundaries).
        if !self.same_tile_px(x0 - 1, y0, x0, y0) {
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
    let encode_start = std::time::Instant::now();
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
    let qpi_c = (slice_qp_y + crate::chroma_qp_offset() as i32).clamp(-qp_bd_offset, 57);
    let qp_c = chroma_qp_from_luma(qpi_c, cat) + qp_bd_offset;

    // Resolve composite efforts (e.g. FastAdaptive, which combines FloorPlus and
    // Fastest by slice QP — see `effort::composition`) to a concrete base, so all
    // downstream encoder flags see a real base template. Also collapse
    // Fast/Balanced/Good onto Best (rdo2 development phase).
    let resolved_effort = match config.effort {
        Effort::Fast | Effort::Balanced | Effort::Good => Effort::Best,
        other => crate::effort::resolve_for_qp(other, slice_qp_y),
    };
    let mut coerced_config;
    let config = if resolved_effort != config.effort {
        coerced_config = config.clone();
        coerced_config.effort = resolved_effort;
        &coerced_config
    } else {
        config
    };

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
    // Historical diagnostics: `BPG_BEST_SCHEDULER=balanced` used to combine the
    // small-luma trial-RDOQ gate with approximate trial residual bits. The
    // current rdo2 Best path makes the scheduler/RDOQ gate counters identical
    // to `off`, so keep the explicit knobs below as diagnostics only.
    if best2_parallel {
        effort_template.parallel_analysis = true;
    }
    // WPP-style analysis context propagation for the parallel ladder: price each
    // CTU against a context propagated along the wavefront (raster-within-row,
    // seeded from the row above's 2nd CTU) instead of one cold frozen context.
    // Recovers essentially all of the parallel frozen-vs-running drift. Never
    // enabled for `Placebo` (keeps it byte-identical to its frozen reference).
    let best2_wpp = best2_parallel
        && std::env::var("BPG_WPP")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
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
    // Default-on for `Best`; for any other tier it stays on the legacy lambda
    // unless `BPG_BEST2_RD_LAMBDA=1` is set, so the x265-parity lambda can be
    // A/B'd on Slow/SlowPlus (the gap to C is luma, and the legacy HM 0.57 factor
    // is a prime suspect — see docs/c-side-psnr-luma-gap-2026-06-23.md).
    let best2_rd_lambda = std::env::var("BPG_BEST2_RD_LAMBDA")
        .ok()
        .map(|v| v.trim() != "0")
        .unwrap_or(config.effort == Effort::Best);
    // x265-parity RDOQ level-2 rate model (greater1/greater2 context tracking +
    // adapted Rice in `rdoq_single_scan`). Default-on for `Best`
    // (`BPG_BEST2_RDOQ2=0` reverts to the frozen-context single-scan model).
    let best2_rdoq2 = config.effort == Effort::Best
        && std::env::var("BPG_BEST2_RDOQ2")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
    let best_cu_force_split_edge_lowqp_min_log2 = if config.effort == Effort::Best {
        std::env::var("BPG_BEST_CU_FORCE_SPLIT").ok().and_then(|v| {
            match v.trim().to_ascii_lowercase().as_str() {
                "edge64-lowqp" | "edge64" => Some(6),
                "edge-lowqp" | "edge32-lowqp" | "edge" => Some(5),
                _ => None,
            }
        })
    } else {
        None
    };
    let best_trial_approx_bits = config.effort == Effort::Best
        && std::env::var("BPG_BEST_TRIAL_APPROX_BITS")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(false);
    let best_trial_rdoq_gate = if config.effort == Effort::Best {
        match std::env::var("BPG_BEST_TRIAL_RDOQ_GATE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("0") | Some("off") | Some("false") => BestTrialRdoqGate::Off,
            Some("1") | Some("on") | Some("true") | Some("small-luma") => {
                BestTrialRdoqGate::SmallLuma
            }
            Some("chroma") => BestTrialRdoqGate::Chroma,
            Some("luma32") | Some("32") => BestTrialRdoqGate::Luma32,
            Some("large") | Some("large-luma") => BestTrialRdoqGate::Large,
            _ => BestTrialRdoqGate::Off,
        }
    } else {
        BestTrialRdoqGate::Off
    };
    let best_tt_close_escalation = config.effort == Effort::Best
        && std::env::var("BPG_BEST_TT_CLOSE_ESCALATION")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(false);
    let best_tt_escalation_margin = std::env::var("BPG_BEST_TT_ESCALATION_MARGIN")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.02)
        .max(0.0);
    // AQ-gated: under the experimental variance AQ, `Best` falls back to the
    // same coding path the lower AQ tiers use. The leaf-screen / TU-neighbour /
    // TT-reuse fast paths bypass `final_code_tt`'s per-CU QP bookkeeping, which
    // the decoder's QG-QP prediction mirror depends on, so they must stay off
    // when QP varies per quantization group.
    let best_luma_leaf_screen = config.effort == Effort::Best && !aq_active;
    let best_tu_neighbor_limit = config.effort == Effort::Best && !aq_active;
    // Experimental smooth-region TU force-leaf speed lever. `BPG_SMOOTH_TU_LEAF`:
    // unset/`0` ⇒ off; `1` ⇒ min log2 4 (16×16+); a number 2..=5 ⇒ that min log2.
    // Disabled under AQ (per-QG QP bookkeeping, like the other fast TU paths).
    let smooth_tu_leaf_min_log2: Option<u8> = if aq_active {
        None
    } else {
        std::env::var("BPG_SMOOTH_TU_LEAF").ok().and_then(|v| {
            match v.trim() {
                "" | "0" => None,
                "1" => Some(4),
                n => n.parse::<u8>().ok().filter(|m| (2..=5).contains(m)),
            }
        })
    };
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
    let cu_early_diag_enabled = std::env::var_os("BPG_CU_EARLY_DIAG").is_some();
    let cabac_ctx_diag = std::env::var_os("BPG_CABAC_CTX_DIAG").is_some();
    let needs_analysis = aq_active
        || cu_early_diag_enabled
        || best_cu_force_split_edge_lowqp_min_log2.is_some()
        || best_tu_neighbor_limit
        || smooth_tu_leaf_min_log2.is_some()
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
    let env_bool_or = |key: &str, default: bool| {
        std::env::var(key)
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(default)
    };
    let env_f64_or = |key: &str, default: f64| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(default)
    };
    let floorplus_modes = match std::env::var("BPG_FLOORPLUS_MODES")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("step3") => RmdModeSet::Step3,
        Some("step4") => RmdModeSet::Step4,
        _ if config.effort == Effort::Slow => crate::effort::template(Effort::Slow).rmd.mode_set,
        _ => RmdModeSet::MpmPlanarDcOnly,
    };
    let floorplus2_modes = match std::env::var("BPG_FLOORPLUS2_MODES")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("mpm") => RmdModeSet::MpmPlanarDcOnly,
        Some("step3") => RmdModeSet::Step3,
        _ => RmdModeSet::Step4,
    };
    // `config.effort` is already the resolved base here. `Fastest` is on the
    // rdo2 path (it is FastAdaptive's high-QP base; routing it through the legacy
    // rdo1 engine would expose the open boundary-CABAC desync). `FastAdaptive`
    // never appears post-resolution.
    let rdo2_best_luma_default = matches!(
        config.effort,
        Effort::Best
            | Effort::Slow
            | Effort::SlowPlus
            | Effort::Fastest
            | Effort::Floor
            | Effort::FloorPlus
            | Effort::FloorPlus2
            | Effort::FloorShallow
    );
    let rdo2_tu = rdo2_best_luma_default || env_bool_or("BPG_RDO2_TU", false);
    let rdo2_nxn = rdo2_best_luma_default || env_bool_or("BPG_RDO2_NXN", false);
    let rdo2_luma = rdo2_best_luma_default || env_bool_or("BPG_RDO2_LUMA", false);
    let rdo2_luma_scratch =
        rdo2_best_luma_default || env_bool_or("BPG_RDO2_LUMA_SCRATCH", rdo2_luma);
    let rdo2_luma_close_mult = env_f64_or(
        "BPG_RDO2_LUMA_CLOSE_MULT",
        if rdo2_best_luma_default { 2.0 } else { 1.0 },
    );
    let rdo2_chroma = rdo2_best_luma_default || env_bool_or("BPG_RDO2_CHROMA", false);
    let rdo2_chroma_scratch =
        rdo2_best_luma_default || env_bool_or("BPG_RDO2_CHROMA_SCRATCH", rdo2_chroma);
    let best_angular_exclusion = if config.effort == Effort::Best {
        let mode = match std::env::var("BPG_BEST_ANGULAR_EXCLUSION")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("game") => rdo2::angular_exclusion::AngularExclusionMode::Game,
            Some("iame") => rdo2::angular_exclusion::AngularExclusionMode::Iame,
            Some("tsame") | Some("1") | Some("on") | Some("true") => {
                rdo2::angular_exclusion::AngularExclusionMode::Tsame
            }
            _ => rdo2::angular_exclusion::AngularExclusionMode::Off,
        };
        rdo2::angular_exclusion::AngularExclusionConfig {
            mode,
            game_ref_var_threshold: std::env::var("BPG_ANGULAR_GAME_VAR")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(8.0),
            iame_factor: std::env::var("BPG_ANGULAR_IAME_FACTOR")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.5),
            min_angular_keep: std::env::var("BPG_ANGULAR_MIN_KEEP")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .filter(|&v| v > 0)
                .unwrap_or(6),
            min_log2_size: std::env::var("BPG_ANGULAR_MIN_LOG2")
                .ok()
                .and_then(|v| v.trim().parse::<u8>().ok())
                .unwrap_or(4),
            protect_mpm: true,
        }
    } else {
        rdo2::angular_exclusion::AngularExclusionConfig::default()
    };
    let slow_partnxn = match config.effort {
        Effort::Slow => std::env::var("BPG_SLOW_PARTNXN")
            .map(|v| v != "0")
            .unwrap_or(false),
        Effort::SlowPlus => std::env::var("BPG_SLOWPLUS_PARTNXN")
            .or_else(|_| std::env::var("BPG_SLOW_PARTNXN"))
            .map(|v| v != "0")
            .unwrap_or(true),
        _ => false,
    };
    // Single source of truth for the tile partition, computed once and threaded
    // to the encoder, `write_pps`, and the slice header so they cannot disagree
    // (and the env-driven `effective_tile_dims` is parsed only once per encode).
    let tiles = params::effective_tile_dims(config);
    let tile_grid = {
        let ctb = 1u32 << CTB_LOG2;
        let ctbs_x = width.div_ceil(ctb);
        let ctbs_y = height.div_ceil(ctb);
        match tiles {
            // Uniform spacing (matches `write_pps`'s uniform_spacing_flag=1 and
            // the decoder's H.265 6.5.1 derivation — same shared constructor).
            Some((cols, rows)) => {
                bpg_hevc_decode::hevc::tile::TileGrid::uniform(ctbs_x, ctbs_y, cols, rows)
            }
            None => bpg_hevc_decode::hevc::tile::TileGrid::single(ctbs_x, ctbs_y),
        }
    };

    // `BPG_TILE_SKEW`: read-only estimator of how spatially skewed the per-CTU
    // encode cost is, to decide whether cost-aware non-uniform sizing (FAST,
    // Phase 5b) can repay its complexity. Cost is proxied from the source-only
    // preanalysis (no search). Under the auto-grid (one tile per core), makespan
    // = the heaviest tile's cost, so the best case FAST could achieve is the
    // mean tile cost → potential speedup ≈ max/mean. A ratio near 1.0 means the
    // uniform grid is already balanced and 5b is not worth it.
    if std::env::var_os("BPG_TILE_SKEW").is_some() {
        let ctb = 1u32 << CTB_LOG2;
        let ctbs_x = width.div_ceil(ctb);
        let ctbs_y = height.div_ceil(ctb);
        // Per-CTU cost estimators (source-derived, pre-search):
        //   var_lin: sum of cell variance over the CTU (raw local complexity).
        //   var_log: sum of log2(1+variance) (dampens texture outliers; matches
        //            the AQ reference metric `frame_mean_log2var`).
        //   imp_max: max importance_q8 over the CTU (quality-steering weight).
        let ctu_cost = |cx: u32, cy: u32| -> (f64, f64, f64) {
            let (x0, y0) = (cx * ctb, cy * ctb);
            let (mut lin, mut log) = (0.0f64, 0.0f64);
            for dy in (0..ctb).step_by(32) {
                for dx in (0..ctb).step_by(32) {
                    let v = analysis.variance_at(x0 + dx, y0 + dy) as f64;
                    lin += v;
                    log += (1.0 + v).log2();
                }
            }
            let imp = analysis.importance_at(x0, y0, CTB_LOG2) as f64;
            (lin, log, imp)
        };
        let nt = tile_grid.num_tiles();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let maxmean = |sel: &dyn Fn((f64, f64, f64)) -> f64, name: &str| {
            let mut costs = Vec::with_capacity(nt as usize);
            for t in 0..nt {
                let (cx0, cy0, cx1, cy1) = tile_grid.tile_ctb_bounds(t);
                let mut c = 0.0f64;
                for cy in cy0..cy1 {
                    for cx in cx0..cx1 {
                        c += sel(ctu_cost(cx, cy));
                    }
                }
                costs.push(c);
            }
            let sum: f64 = costs.iter().sum();
            let mean = sum / nt.max(1) as f64;
            let max = costs.iter().cloned().fold(0.0f64, f64::max);
            let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "TILE_SKEW[{name}]: tiles={nt} max/mean={:.3} min/mean={:.3} \
                 (potential FAST speedup <= {:.1}%)",
                if mean > 0.0 { max / mean } else { 1.0 },
                if mean > 0.0 { min / mean } else { 1.0 },
                if mean > 0.0 {
                    (1.0 - mean / max) * 100.0
                } else {
                    0.0
                },
            );
        };
        eprintln!(
            "TILE_SKEW: {ctbs_x}x{ctbs_y} CTBs, grid={}x{}, cores={cores}",
            tile_grid.num_cols(),
            tile_grid.num_rows(),
        );
        maxmean(&|(lin, _, _)| lin, "var_lin");
        maxmean(&|(_, log, _)| log, "var_log");
        maxmean(&|(_, _, imp)| imp, "imp_max");
    }

    let mut state = Encoder {
        display_width: width,
        display_height: height,
        cat,
        qp_y,
        qp_c,
        bit_depth: bd,
        effort: config.effort,
        effort_template,
        tile_grid,
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
        best_angular_exclusion,
        search_scratch: rdo2::scratch::SearchScratch::default(),
        rdo2_tu,
        rdo2_luma,
        rdo2_luma_scratch,
        rdo2_luma_close_mult,
        rdo2_luma_legacy_escalate: !rdo2_best_luma_default
            && env_bool_or("BPG_RDO2_LUMA_LEGACY_ESCALATE", false),
        rdo2_chroma,
        rdo2_chroma_scratch,
        rdo2_nxn,
        rdo2_tt_native: rdo2_best_luma_default || env_bool_or("BPG_RDO2_TT_NATIVE", false),
        in_rdo2: false,
        elide_final_residual_pricing: false,
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
        best_trial_approx_bits,
        best_trial_rdoq_gate,
        best_tt_close_escalation,
        best_tt_escalation_margin,
        best_tt_cheap_trial: false,
        best_tt_exact_trial: false,
        sign_data_hiding: crate::sdh_active(config),
        best2_wpp,
        best_luma_leaf_screen,
        best_tu_neighbor_limit,
        smooth_tu_leaf_min_log2,
        best_aq: if config.effort == Effort::Best {
            crate::best_aq_params()
        } else {
            None
        },
        best2_tt_reuse: config.effort == Effort::Best
            && !aq_active
            && !best2_luma_fastrd
            && std::env::var("BPG_BEST2_TT_REUSE")
                .map(|v| v != "0")
                .unwrap_or(false),
        tt_screen_final: false,
        floorplus_budget: None,
        floorplus_repair_bits_bpp: std::env::var("BPG_FLOORPLUS_REPAIR_BITS_BPP")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.125),
        floorplus_repair_dist_px: std::env::var("BPG_FLOORPLUS_DIST_PX")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(4.0),
        slow_repair_thresh: std::env::var("BPG_SLOW_REPAIR_THRESH")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.01),
        slow_luma_cands: std::env::var("BPG_SLOW_LUMA_CANDS")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|v| *v >= 1 && *v <= 35)
            .unwrap_or(4),
        slowplus_luma_cands: std::env::var("BPG_SLOWPLUS_LUMA_CANDS")
            .or_else(|_| std::env::var("BPG_SLOW_LUMA_CANDS"))
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|v| *v >= 1 && *v <= 35)
            .unwrap_or(5),
        slowplus_skip_root_leaf: std::env::var("BPG_SLOWPLUS_SKIP_ROOT_LEAF")
            .map(|v| v.trim() == "1")
            .unwrap_or(false),
        slowplus_cu_escalate: std::env::var("BPG_SLOWPLUS_CU_ESCALATE")
            .map(|v| v.trim() == "1")
            .unwrap_or(false),
        slowplus_deep_repair: std::env::var("BPG_SLOWPLUS_DEEP_REPAIR")
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        floorplus_split: std::env::var("BPG_FLOORPLUS_SPLIT")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        floorplus_tu: std::env::var("BPG_FLOORPLUS_TU")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        floorplus_modes,
        floorplus2_repair_bits_bpp: std::env::var("BPG_FLOORPLUS2_REPAIR_BITS_BPP")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.125),
        floorplus2_repair_dist_px: std::env::var("BPG_FLOORPLUS2_DIST_PX")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(4.0),
        floorplus2_split: std::env::var("BPG_FLOORPLUS2_SPLIT")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        floorplus2_tu: std::env::var("BPG_FLOORPLUS2_TU")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        floorplus2_modes,
        floorplus2_mode_limit: std::env::var("BPG_FLOORPLUS2_MAX_EXACT_MODES")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .filter(|&v| (2..=8).contains(&v))
            .unwrap_or(5),
        floorplus2_max_bids_per_ctu: std::env::var("BPG_FLOORPLUS2_MAX_BIDS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(2),
        floorplus2_max_accepted_bids_per_ctu: std::env::var("BPG_FLOORPLUS2_MAX_ACCEPTED_BIDS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1),
        floorplus2_odds_bid_threshold: std::env::var("BPG_FLOORPLUS2_ODDS_BID")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.5),
        floorplus2_odds_mode_threshold: std::env::var("BPG_FLOORPLUS2_ODDS_MODE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(0.5),
        floorplus2_child_repair: std::env::var("BPG_FLOORPLUS2_CHILD_REPAIR")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
        best2_cu_reuse: matches!(
            config.effort,
            Effort::Best
                | Effort::Fastest
                | Effort::Floor
                | Effort::FloorPlus
                | Effort::FloorPlus2
                | Effort::FloorShallow
        ) && !aq_active
            && !best2_luma_fastrd,
        // PartNxN is a real coding tool, so it belongs on every quality-leaning
        // tier (each prunes the per-PU search via its own RD budget) — Best down
        // to Fast. Off for Fastest (speed floor) and the exact reference tiers.
        // Slow gets an env-gated diagnostic (BPG_SLOW_PARTNXN=1) so the
        // deep-island PartNxN path can be tested without cluttering the default
        // template.
        partnxn: std::env::var("BPG_PARTNXN").map(|v| v != "0").unwrap_or(
            matches!(
                config.effort,
                Effort::Best | Effort::Good | Effort::Balanced | Effort::Fast
            ) || slow_partnxn,
        ),
        partnxn_prune: match std::env::var("BPG_PARTNXN_PRUNE")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("0") | Some("off") | Some("false") => PartNxnPrune::Off,
            Some("1") | Some("on") | Some("true") | Some("conservative") => {
                PartNxnPrune::Conservative
            }
            Some("aggressive") => PartNxnPrune::Aggressive,
            _ if config.effort == Effort::Balanced || slow_partnxn => PartNxnPrune::Conservative,
            _ => PartNxnPrune::Off,
        },
        nxn_pu_rdoq_top: std::env::var("BPG_NXN_PU_RDOQ_TOP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .or((config.effort == Effort::SlowPlus).then_some(4)),
        nxn_close_mult: std::env::var("BPG_NXN_CLOSE_MULT")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0),
        nxn_exact: std::env::var_os("BPG_NXN_EXACT").is_some(),
        nxn_adaptive: std::env::var_os("BPG_NXN_EXACT").is_none()
            && (std::env::var_os("BPG_NXN_ADAPTIVE").is_some()
                || config.effort == Effort::SlowPlus),
        nxn_approx: std::env::var_os("BPG_NXN_APPROX").is_some(),
        nxn_carry: std::env::var_os("BPG_NXN_CARRY").is_some(),
        cu_early_diag: cu_early_diag_enabled,
        cabac_ctx_diag,
        best_cu_early_16: config.effort == Effort::Best,
        best_cu_force_split_edge_lowqp_min_log2,
        best_cu_no_64_leaf: config.effort == Effort::Best,
        best_cu_zero_leaf_32: config.effort == Effort::Best
            && std::env::var_os("BPG_BEST_CU_ZERO_LEAF_32").is_some(),
        // Stage-2 rough-skip-leaf gate. Best-only, default-off. `=32`/`=16` map to
        // the min log2_cb_size; `off`/`0`/unset/unparsable -> None.
        best_cu_rough_skip_min_log2: if config.effort == Effort::Best {
            match std::env::var("BPG_BEST_CU_ROUGH_SKIP")
                .ok()
                .as_deref()
                .map(str::trim)
            {
                Some("16") => Some(4),
                Some("32") => Some(5),
                Some("64") => Some(6),
                _ => None,
            }
        } else {
            None
        },
        best_cu_src_split_min_log2: if config.effort == Effort::Best {
            match std::env::var("BPG_BEST_CU_SRC_SPLIT")
                .ok()
                .as_deref()
                .map(str::trim)
            {
                Some("16") => Some(4),
                Some("32") => Some(5),
                Some("64") => Some(6),
                _ => None,
            }
        } else {
            None
        },
        // Resolution-gated default-on for Best: the cheap-leaf early-termination
        // buys ~24% encode time at >=8K for +1.4% bytes but is a pure BD loss at
        // low res (see rdo-plan-status-2026-06-20.md). Auto-enable k=8 only above
        // ~4 MP, where the resolution-scaling win dominates. `BPG_CU_EARLY_K`
        // overrides explicitly (including `=0` to force-disable).
        best_cu_early_term_k: if config.effort == Effort::Best {
            match std::env::var("BPG_CU_EARLY_K")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
            {
                Some(k) => k,
                None if (width as u64) * (height as u64) >= 4_000_000 => 8,
                None => 0,
            }
        } else {
            0
        },
        best_cu_early_term_min_log2: std::env::var("BPG_CU_EARLY_MIN_LOG2")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(5),
        best_cu_early_term_min_qp: std::env::var("BPG_CU_EARLY_MIN_QP")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .unwrap_or(0),
        cu_rough_m_leaf: std::env::var("BPG_CU_ROUGH_M_LEAF")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(100),
        cu_rough_m_split: std::env::var("BPG_CU_ROUGH_M_SPLIT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(96),
        cu_src_t_leaf: std::env::var("BPG_CU_SRC_T_LEAF")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(10),
        cu_src_t_split: std::env::var("BPG_CU_SRC_T_SPLIT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(40),
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

    let use_parallel_analysis = state.effort_template.parallel_analysis && !state.trace.enabled;
    if state.trace.enabled && state.effort_template.parallel_analysis {
        eprintln!("search-trace: forcing serial analysis so the per-decision ledger is complete");
    }

    let (slice_data, entry_sizes) = if config.sao == SaoMode::On {
        // Production SAO (docs/sao.md): build every CTU tree once (reconstructing
        // the frame), deblock, decide SAO on the deblocked frame, then *replay*
        // the cached trees with SAO syntax — no second RDO pass. Parallel tiers
        // build via the wavefront; serial tiers build interleaved (the throwaway
        // bitstream only evolves the running context their RD pricing needs).
        let phase_start = std::time::Instant::now();
        let trees = if use_parallel_analysis {
            build_slice_trees_parallel(&mut state, slice_qp_y)
        } else {
            build_slice_trees_serial(&mut state, slice_qp_y)
        };
        state.stats.phase_build_us += phase_start.elapsed().as_micros() as u64;
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
            state.stats.phase_deblock_us += phase_start.elapsed().as_micros() as u64;
        }
        let phase_start = std::time::Instant::now();
        let sao_map = state.decide_sao_map(ctb);
        state.stats.phase_sao_decide_us += phase_start.elapsed().as_micros() as u64;
        let phase_start = std::time::Instant::now();
        let (bytes, entries) =
            write_slice_from_trees(&mut state, &trees, Some(&sao_map), slice_qp_y);
        state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
        // The frame is already deblocked; SAO is the final in-loop step.
        let phase_start = std::time::Instant::now();
        apply_sao(&mut state.frame, &sao_map, ctb);
        state.stats.phase_sao_apply_us += phase_start.elapsed().as_micros() as u64;
        (bytes, entries)
    } else {
        let (bytes, entries) = if use_parallel_analysis {
            let phase_start = std::time::Instant::now();
            let trees = build_slice_trees_parallel(&mut state, slice_qp_y);
            state.stats.phase_build_us += phase_start.elapsed().as_micros() as u64;
            let phase_start = std::time::Instant::now();
            let (bytes, entries) = write_slice_from_trees(&mut state, &trees, None, slice_qp_y);
            state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
            (bytes, entries)
        } else {
            let phase_start = std::time::Instant::now();
            let bytes = encode_slice_data(&mut state, None, slice_qp_y);
            state.stats.phase_write_us += phase_start.elapsed().as_micros() as u64;
            (bytes, Vec::new())
        };
        if state.deblock {
            let phase_start = std::time::Instant::now();
            bpg_hevc_decode::hevc::deblock::apply_deblocking_filter(&mut state.frame, 0, 0, 0, 0);
            state.stats.phase_deblock_us += phase_start.elapsed().as_micros() as u64;
        }
        (bytes, entries)
    };

    state.stats.region_class_counts = analysis.class_counts();

    if state.trace.enabled {
        let total = state.stats.residual_bit_estimates;
        if let Err(e) = state.trace.write_reports(total) {
            eprintln!("search-trace: failed to write reports: {e}");
        }
    }

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
        eprintln!(
            "profile rdo2 eval (ms): predict={:.1}  fwd_transform={:.1}  \
             quant+rdoq={:.1}  inv_transform={:.1}  residual_price={:.1}  || rough_satd_search={:.1}",
            ms(p.eval_predict),
            ms(p.eval_transform),
            ms(p.eval_quant_rdoq),
            ms(p.eval_recon),
            ms(p.eval_residual_price),
            ms(p.rough_search),
        );

        // Per-size luma block-eval breakdown: is the cost VOLUME (too many
        // evals) or HEAVINESS (slow per eval) — and at which CU/TB size?
        let s = &state.stats;
        eprintln!("luma block-eval by size (calls / rdoq / ms / ns-per-call):");
        for li in 2..=5usize {
            let calls = s.eval_leaf_calls_by_log2[li];
            if calls == 0 {
                continue;
            }
            let rdoq = s.eval_leaf_rdoq_by_log2[li];
            let ns = s.eval_leaf_ns_by_log2[li];
            eprintln!(
                "  {:>2}x{:<2}: calls={:>9}  rdoq={:>9} ({:>4.1}%)  {:>7.1}ms  {:>5}ns/call  rough_scores={:>9}",
                1 << li,
                1 << li,
                calls,
                rdoq,
                100.0 * rdoq as f64 / calls as f64,
                ns as f64 / 1.0e6,
                if calls > 0 { ns / calls } else { 0 },
                s.rough_score_by_log2[li],
            );
        }
        eprintln!(
            "  rough: total_scores={}  via_permode_predict={} (slow path), via_batched={}",
            s.rough_score_by_log2.iter().sum::<u64>(),
            s.rough_permode_predicts,
            s.rough_score_by_log2
                .iter()
                .sum::<u64>()
                .saturating_sub(s.rough_permode_predicts),
        );
    }

    if std::env::var_os("BPG_RDO2_REPORT_LEGACY").is_some() {
        eprintln!("{}", state.stats.legacy.report());
    }

    if state.cu_early_diag {
        let s = &state.stats;
        eprintln!("cu-early-diag (per log2_cb_size 3..=6 = 8/16/32/64):");
        for i in 3..=6usize {
            let ev = s.cu_split_eval_by_log2[i];
            if ev == 0 {
                continue;
            }
            let won = s.cu_split_won_by_log2[i];
            let pred = s.cu_force_leaf_pred_by_log2[i];
            let mist = s.cu_force_leaf_mistake_by_log2[i];
            let pct = |n: u64| 100.0 * n as f64 / ev as f64;
            eprintln!(
                "  {:>2}x{:<2}: eval={:>7} split_won={:>6} ({:>4.1}%)  pred_force_leaf={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                1 << i,
                1 << i,
                ev,
                won,
                pct(won),
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            let pred_zr = s.cu_force_leaf_zero_resid_pred_by_log2[i];
            let mist_zr = s.cu_force_leaf_zero_resid_mistake_by_log2[i];
            eprintln!(
                "        pred_force_leaf_zero_resid ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                pred_zr,
                pct(pred_zr),
                mist_zr,
                if pred_zr > 0 { 100.0 * mist_zr as f64 / pred_zr as f64 } else { 0.0 },
            );
            let pred = s.cu_force_split_structured_pred_by_log2[i];
            let mist = s.cu_force_split_structured_mistake_by_log2[i];
            eprintln!(
                "        pred_force_split_structured={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            let pred = s.cu_force_split_edge_pred_by_log2[i];
            let mist = s.cu_force_split_edge_mistake_by_log2[i];
            eprintln!(
                "        pred_force_split_edge      ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            // Top-down rough-SATD predictor (probes BEFORE full leaf RDO).
            let pred = s.cu_rough_leaf_pred_by_log2[i];
            let mist = s.cu_rough_leaf_mistake_by_log2[i];
            eprintln!(
                "        cu-rough leaf  (m_leaf={:>3})  ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                state.cu_rough_m_leaf,
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            let pred = s.cu_rough_split_pred_by_log2[i];
            let mist = s.cu_rough_split_mistake_by_log2[i];
            eprintln!(
                "        cu-rough split (m_split={:>2}) ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                state.cu_rough_m_split,
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            // Source-activity predictor (faithful pre-decision; source-only).
            let pred = s.cu_src_leaf_pred_by_log2[i];
            let mist = s.cu_src_leaf_mistake_by_log2[i];
            eprintln!(
                "        cu-src  leaf  (t_leaf={:>3})  ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                state.cu_src_t_leaf,
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
            let pred = s.cu_src_split_pred_by_log2[i];
            let mist = s.cu_src_split_mistake_by_log2[i];
            eprintln!(
                "        cu-src  split (t_split={:>2}) ={:>6} ({:>4.1}%)  mistakes={:>5} ({:>4.1}% of pred)",
                state.cu_src_t_split,
                pred,
                pct(pred),
                mist,
                if pred > 0 { 100.0 * mist as f64 / pred as f64 } else { 0.0 },
            );
        }
    }

    if state.cabac_ctx_diag {
        let s = &state.stats;
        let compares = s.cabac_ctx_diag_cu_compares.max(1);
        eprintln!("cabac-context-diag:");
        eprintln!(
            "  compares={}  winner_flips={} ({:.3}%)",
            s.cabac_ctx_diag_cu_compares,
            s.cabac_ctx_diag_winner_flips,
            100.0 * s.cabac_ctx_diag_winner_flips as f64 / compares as f64,
        );
        eprintln!(
            "  frozen wins: leaf={} split={}  serial wins: leaf={} split={}",
            s.cabac_ctx_diag_frozen_leaf_wins,
            s.cabac_ctx_diag_frozen_split_wins,
            s.cabac_ctx_diag_serial_leaf_wins,
            s.cabac_ctx_diag_serial_split_wins,
        );
        eprintln!(
            "  leaf bits: frozen={} serial={} delta={}",
            s.cabac_ctx_diag_frozen_leaf_bits,
            s.cabac_ctx_diag_serial_leaf_bits,
            s.cabac_ctx_diag_serial_leaf_bits as i128 - s.cabac_ctx_diag_frozen_leaf_bits as i128,
        );
        eprintln!(
            "  split bits: frozen={} serial={} delta={}",
            s.cabac_ctx_diag_frozen_split_bits,
            s.cabac_ctx_diag_serial_split_bits,
            s.cabac_ctx_diag_serial_split_bits as i128 - s.cabac_ctx_diag_frozen_split_bits as i128,
        );
    }

    if std::env::var_os("BPG_TU_NEIGHBOR_DIAG").is_some() {
        let s = &state.stats;
        eprintln!("tu-neighbor-diag (per log2_size 2..=6 = 4/8/16/32/64):");
        for i in 2..=6usize {
            let calls = s.tu_neighbor_limit_calls_by_log2[i];
            if calls == 0 {
                continue;
            }
            let pct = |n: u64| 100.0 * n as f64 / calls as f64;
            eprintln!(
                "  {:>2}x{:<2}: calls={:>8} prior_none={:>8} ({:>4.1}%) prior_leaf={:>8} ({:>4.1}%) prior_split={:>8} ({:>4.1}%) residual_reject={:>8} ({:>4.1}%) region_reject={:>8} ({:>4.1}%) accept={:>8} ({:>4.1}%)",
                1 << i,
                1 << i,
                calls,
                s.tu_neighbor_limit_prior_none_by_log2[i],
                pct(s.tu_neighbor_limit_prior_none_by_log2[i]),
                s.tu_neighbor_limit_prior_leaf_by_log2[i],
                pct(s.tu_neighbor_limit_prior_leaf_by_log2[i]),
                s.tu_neighbor_limit_prior_split_by_log2[i],
                pct(s.tu_neighbor_limit_prior_split_by_log2[i]),
                s.tu_neighbor_limit_residual_reject_by_log2[i],
                pct(s.tu_neighbor_limit_residual_reject_by_log2[i]),
                s.tu_neighbor_limit_region_reject_by_log2[i],
                pct(s.tu_neighbor_limit_region_reject_by_log2[i]),
                s.tu_neighbor_limit_accept_by_log2[i],
                pct(s.tu_neighbor_limit_accept_by_log2[i]),
            );
            let labels = ["N", "L", "S"];
            let mut combos = Vec::new();
            for left in 0..3usize {
                for above in 0..3usize {
                    let count = s.tu_neighbor_limit_prior_combo_by_log2[i][left * 3 + above];
                    if count > 0 {
                        combos.push(format!(
                            "{}/{}={} ({:.1}%)",
                            labels[left],
                            labels[above],
                            count,
                            pct(count)
                        ));
                    }
                }
            }
            eprintln!("        prior_combo: {}", combos.join("  "));
            let mixed_pred = s.tu_neighbor_mixed_leaf_pred_by_log2[i];
            let mixed_mistake = s.tu_neighbor_mixed_leaf_mistake_by_log2[i];
            if mixed_pred > 0 {
                eprintln!(
                    "        mixed_leaf_candidate: pred={} ({:.1}%) mistakes={} ({:.1}% of pred)",
                    mixed_pred,
                    pct(mixed_pred),
                    mixed_mistake,
                    100.0 * mixed_mistake as f64 / mixed_pred as f64,
                );
            }
            let both_split_pred = s.tu_neighbor_both_split_pred_by_log2[i];
            let both_split_mistake = s.tu_neighbor_both_split_mistake_by_log2[i];
            if both_split_pred > 0 {
                eprintln!(
                    "        both_split_candidate: pred={} ({:.1}%) mistakes={} ({:.1}% of pred)",
                    both_split_pred,
                    pct(both_split_pred),
                    both_split_mistake,
                    100.0 * both_split_mistake as f64 / both_split_pred as f64,
                );
            }
        }
    }

    state.stats.phase_total_us = encode_start.elapsed().as_micros() as u64;
    (out, state.frame, state.stats)
}

pub fn encode(config: &StillHevcConfig, src: Source<'_>) -> (Vec<u8>, DecodedFrame) {
    let (bytes, recon, _) = encode_with_stats(config, src);
    (bytes, recon)
}
