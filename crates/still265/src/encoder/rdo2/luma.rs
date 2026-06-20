//! Staged luma candidate decision for `rdo2` (slice 3).
//!
//! This is the call target for `rdo.rs::build_cu_leaf`'s ranked-candidate path:
//! `build_cu_leaf` snapshots the CU region, decides the luma candidate list, and
//! then delegates the *decision* — cheap-screen the ranked candidates, escalate
//! only close calls to exact rechecks, and final-code the winner — to this
//! module. Keeping the decision here (rather than accreting in the old recursive
//! engine) is the architecture-guardrail boundary from
//! `docs/rdo2-refactor-progress.md`: `rdo.rs` stays a call-site interceptor.
//!
//! `BPG_RDO2_LUMA` (`self.rdo2_luma`, `Best` only) forces the cheap Best trial
//! path (no trial RDOQ + approximate residual bits) for the ranked screen; the
//! close-call escalation rechecks the top candidates exactly before the winning
//! CU is final-coded. With the gate off this method reproduces the legacy ranked
//! path (FastRd-trial escalation only), so it is behaviour-neutral by default.

use bpg_hevc_decode::hevc::intra::fill_mpm_candidates;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::contexts::Contexts;
use crate::effort::{BlockSearchBudget, TrialQuality};
use crate::plan::DecisionConfidence;
use crate::trace::WorkBucket;
use crate::Effort;

use super::super::types::{CuLeaf, FrameSnapshot, MapSnapshot, MAX_TB_LOG2};
use super::cost::{estimate_cu_leaf_bits, estimate_intra_luma_mode_bits};
use super::policy::{EvalKind, EvalPolicy};
use super::tu::LeafEval;

#[allow(dead_code)] // full cost record shape from plan.md; later slices consume more fields.
struct LumaCandidateCost {
    mode: u8,
    rank: usize,
    distortion: u64,
    frac_bits: u64,
    cost: f64,
    close_guard_cost: f64,
    trace: Option<crate::trace::CandRec>,
}

impl<'a> super::super::Encoder<'a> {
    fn take_luma_trace_cands(&mut self) -> Vec<crate::trace::CandRec> {
        let mut trace_cands = std::mem::take(&mut self.search_scratch.luma_trace_cands);
        trace_cands.clear();
        trace_cands
    }

    fn put_luma_trace_cands(&mut self, trace_cands: Vec<crate::trace::CandRec>) {
        self.search_scratch.luma_trace_cands = trace_cands;
    }

    fn take_luma_exact_trace_cands(&mut self) -> Vec<crate::trace::CandRec> {
        let mut trace_cands = std::mem::take(&mut self.search_scratch.luma_exact_trace_cands);
        trace_cands.clear();
        trace_cands
    }

    fn put_luma_exact_trace_cands(&mut self, trace_cands: Vec<crate::trace::CandRec>) {
        self.search_scratch.luma_exact_trace_cands = trace_cands;
    }

    fn rdo2_luma_candidate_bits(
        &self,
        ctxs: &Contexts,
        mpm: [IntraPredMode; 3],
        mode: u8,
        block_eval: &LeafEval,
        log2_cb_size: u8,
    ) -> u64 {
        let mut bits = 0u64;
        if log2_cb_size == 3 {
            let mut m = ctxs.models[ctx::PART_MODE];
            let mut est = CabacEstimator::new();
            est.encode_bin(1, &mut m);
            bits += est.frac_bits();
        }
        bits += estimate_intra_luma_mode_bits(ctxs, mpm, mode);
        let mut cbf_m = ctxs.models[ctx::CBF_LUMA + 1];
        let mut est = CabacEstimator::new();
        est.encode_bin(block_eval.coded.cbf as u8, &mut cbf_m);
        bits += est.frac_bits();
        if block_eval.coded.cbf {
            bits += block_eval.frac_bits;
        }
        bits
    }

    fn rdo2_eval_luma_candidate_scratch(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        luma_mode: u8,
        rank: usize,
        mpm: [IntraPredMode; 3],
        ctxs: &Contexts,
        kind: EvalKind,
    ) -> LumaCandidateCost {
        let bucket = match kind {
            EvalKind::CheapTrial => WorkBucket::LumaCandidateCheap,
            EvalKind::ExactTrial => WorkBucket::LumaCandidateExact,
            EvalKind::Final => WorkBucket::FinalReplay,
        };
        let policy = EvalPolicy::for_kind(kind).with_bucket(bucket);
        let eval = self.rdo2_eval_leaf_block(
            ctxs,
            x0,
            y0,
            log2_cb_size,
            0,
            luma_mode,
            self.cur_qp_y,
            policy,
        );
        let bits = self.rdo2_luma_candidate_bits(ctxs, mpm, luma_mode, &eval, log2_cb_size);
        let cost = self.rd_cost(eval.distortion, bits);
        let trace = self
            .trace
            .enabled
            .then(|| crate::trace::CandRec::with_flip(cost, cost, 0));
        match kind {
            EvalKind::CheapTrial => {
                self.stats.rdo2_luma_scratch_candidates += 1;
                self.stats.rdo2_luma_scratch_legacy_evals_skipped += 1;
                self.stats.rdo2_luma_scratch_snapshot_restores_saved += 3;
            }
            EvalKind::ExactTrial => {
                self.stats.rdo2_luma_scratch_exact_rechecks += 1;
            }
            EvalKind::Final => {}
        }
        LumaCandidateCost {
            mode: luma_mode,
            rank,
            distortion: eval.distortion,
            frac_bits: bits,
            cost,
            close_guard_cost: cost,
            trace,
        }
    }

    /// Screen a single luma candidate for a CU larger than the maximum transform
    /// block (e.g. 64x64): the luma "leaf" is itself a forced transform tree, so
    /// the single-block scratch evaluator cannot price it. Materialise the
    /// luma-only subtree (which commits its reconstruction into the frame, doing
    /// its own cheap-screen + close-call exact escalation per TU), price it, then
    /// restore the base reconstruction so the next candidate starts clean.
    #[allow(clippy::too_many_arguments)]
    fn rdo2_eval_luma_candidate_subtree(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        luma_mode: u8,
        rank: usize,
        mpm: [IntraPredMode; 3],
        base_frame: &FrameSnapshot,
        base_tu_depth: &MapSnapshot,
        ctxs: &Contexts,
    ) -> LumaCandidateCost {
        self.restore_frame_region(base_frame);
        self.restore_tu_depth_region(base_tu_depth);
        let (tt, distortion) = self.rdo2_luma_subtree(x0, y0, log2_cb_size, 0, luma_mode, ctxs);
        let bits =
            estimate_intra_luma_mode_bits(ctxs, mpm, luma_mode) + self.tt_bits_luma(ctxs, &tt);
        let cost = self.rd_cost(distortion, bits);
        self.restore_frame_region(base_frame);
        self.restore_tu_depth_region(base_tu_depth);
        let trace = self
            .trace
            .enabled
            .then(|| crate::trace::CandRec::with_flip(cost, cost, 0));
        self.stats.rdo2_luma_scratch_exact_rechecks += 1;
        LumaCandidateCost {
            mode: luma_mode,
            rank,
            distortion,
            frac_bits: bits,
            cost,
            close_guard_cost: cost,
            trace,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rdo2_exact_recheck_luma_candidates(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        modes: &[u8],
        mpm: [IntraPredMode; 3],
        ctxs: &Contexts,
    ) -> u8 {
        debug_assert!(!modes.is_empty());
        let mut best_mode = modes[0];
        let mut best_cost = f64::MAX;
        let mut trace_cands = self.take_luma_exact_trace_cands();
        for (rank0, &luma_mode) in modes.iter().enumerate() {
            let eval = self.rdo2_eval_luma_candidate_scratch(
                x0,
                y0,
                log2_cb_size,
                luma_mode,
                rank0 + 1,
                mpm,
                ctxs,
                EvalKind::ExactTrial,
            );
            if let Some(trace) = eval.trace {
                trace_cands.push(trace);
            }
            if eval.cost < best_cost {
                best_cost = eval.cost;
                best_mode = luma_mode;
            }
        }
        if self.trace.enabled {
            self.trace.note_decision(
                crate::trace::DecisionKind::Escalation,
                x0,
                y0,
                log2_cb_size.min(MAX_TB_LOG2),
                false,
                &trace_cands,
            );
        }
        self.put_luma_exact_trace_cands(trace_cands);
        self.stats
            .rdo2_luma_scratch_legacy_exact_escalations_avoided += 1;
        best_mode
    }

    /// Decide the luma mode for a CU from its ranked candidate list and return
    /// the final-coded [`CuLeaf`]. The caller has already snapshotted the CU
    /// region (`base_frame`/`base_mode_map`/`base_tu_depth`) and computed the
    /// MPM list and search budget.
    pub(in crate::encoder) fn build_cu_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuLeaf {
        self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let tt_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let luma_plan = self.decide_luma_modes(x0, y0, tt_log2, ct_depth, ctxs);
        let candidates = luma_plan.candidates;
        let cand_a = self.neighbor_left_mode(x0, y0);
        let cand_b = self.neighbor_above_mode(x0, y0);
        let mpm = fill_mpm_candidates(cand_a, cand_b);

        let base_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let base_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);

        let tt_budget = self.block_budget(x0, y0, tt_log2, 0);
        if candidates.len() > 1 && !self.effort_template.reference {
            return self.rdo2_cu_luma_ranked(
                x0,
                y0,
                log2_cb_size,
                tt_log2,
                &candidates,
                mpm,
                &base_frame,
                &base_mode_map,
                &base_tu_depth,
                tt_budget,
                ctxs,
            );
        }

        let mut best_leaf: Option<CuLeaf> = None;
        let mut best_cost = f64::MAX;
        let mut runner_up_cost = f64::MAX;
        let mut best_rank = 1usize;
        let mut best_frame: Option<FrameSnapshot> = None;
        let mut best_mode_map: Option<MapSnapshot> = None;
        let mut best_tu_depth: Option<MapSnapshot> = None;

        for (rank0, &luma_mode) in candidates.iter().enumerate() {
            self.stats.cu_trials += 1;
            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);
            self.restore_tu_depth_region(&base_tu_depth);

            let chroma = self.decide_chroma_mode(x0, y0, tt_log2, luma_mode, ctxs);
            self.store_mode(x0, y0, log2_cb_size, luma_mode);

            let tt =
                self.rdo2_analyze_tt(x0, y0, log2_cb_size, 0, luma_mode, chroma.plan.mode, ctxs);
            let leaf = CuLeaf {
                mpm,
                luma_mode,
                chroma_mode_idx: chroma.plan.mode_idx,
                confidence: DecisionConfidence::Clear,
                tt,
                nxn: None,
            };

            let distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
            let bits = estimate_cu_leaf_bits(ctxs, &leaf, log2_cb_size, self.cat);
            let cost = self.rd_cost(distortion, bits);

            if cost < best_cost {
                runner_up_cost = best_cost;
                best_cost = cost;
                best_rank = rank0 + 1;
                best_frame = Some(self.snapshot_frame_region(x0, y0, log2_cb_size));
                best_mode_map = Some(self.snapshot_mode_region(x0, y0, log2_cb_size));
                best_tu_depth = Some(self.snapshot_tu_depth_region(x0, y0, log2_cb_size));
                best_leaf = Some(leaf);
            } else if cost < runner_up_cost {
                runner_up_cost = cost;
            }
        }
        self.record_luma_winner_rank(best_rank);
        self.record_close_call(best_cost, runner_up_cost, tt_budget.close_call_margin);

        self.restore_frame_region(
            best_frame
                .as_ref()
                .expect("decide_luma_modes returns at least one candidate"),
        );
        self.restore_mode_region(
            best_mode_map
                .as_ref()
                .expect("decide_luma_modes returns at least one candidate"),
        );
        self.restore_tu_depth_region(
            best_tu_depth
                .as_ref()
                .expect("decide_luma_modes returns at least one candidate"),
        );
        best_leaf.expect("decide_luma_modes returns at least one candidate")
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::encoder) fn rdo2_cu_luma_ranked(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        tt_log2: u8,
        candidates: &[u8],
        mpm: [IntraPredMode; 3],
        base_frame: &FrameSnapshot,
        base_mode_map: &MapSnapshot,
        base_tu_depth: &MapSnapshot,
        tt_budget: BlockSearchBudget,
        ctxs: &Contexts,
    ) -> CuLeaf {
        let mut best_luma_mode = candidates[0];
        let mut runner_up_mode = candidates[0];
        let mut third_mode = candidates[0];
        let mut best_rank = 1usize;
        let mut best_cost = f64::MAX;
        let mut runner_up_cost = f64::MAX;
        let mut third_cost = f64::MAX;
        let mut trace_cands = self.take_luma_trace_cands();

        // rdo2 (slice 3): screen the luma candidate ranking cheaply (no trial
        // RDOQ / approximate bits, `Best` only); the close-call escalation
        // below rechecks the top two exactly, and the winning mode's CU is
        // always final-coded exactly afterwards.
        let rdo2_cheap_luma = self.rdo2_luma && self.effort == Effort::Best;
        if rdo2_cheap_luma {
            self.stats.rdo2_luma_cheap_cu_decisions += 1;
        }
        // A CU at or below the maximum transform size is screened with the
        // single-block scratch evaluator. A larger CU (e.g. 64x64) has no single
        // transform block, so each candidate is screened by materialising its
        // forced luma transform subtree instead.
        let use_scratch_screen = log2_cb_size <= MAX_TB_LOG2;
        let mut screen_candidates = |this: &mut Self| {
            for (rank0, &luma_mode) in candidates.iter().enumerate() {
                this.stats.cu_trials += 1;
                let eval = if use_scratch_screen {
                    let kind = if rdo2_cheap_luma {
                        EvalKind::CheapTrial
                    } else {
                        EvalKind::ExactTrial
                    };
                    this.rdo2_eval_luma_candidate_scratch(
                        x0,
                        y0,
                        log2_cb_size,
                        luma_mode,
                        rank0 + 1,
                        mpm,
                        ctxs,
                        kind,
                    )
                } else {
                    this.rdo2_eval_luma_candidate_subtree(
                        x0,
                        y0,
                        log2_cb_size,
                        luma_mode,
                        rank0 + 1,
                        mpm,
                        base_frame,
                        base_tu_depth,
                        ctxs,
                    )
                };
                let (cost, trace) = (eval.cost, eval.trace);
                if let Some(trace) = trace {
                    trace_cands.push(trace);
                }

                if cost < best_cost {
                    third_cost = runner_up_cost;
                    third_mode = runner_up_mode;
                    runner_up_cost = best_cost;
                    runner_up_mode = best_luma_mode;
                    best_cost = cost;
                    best_luma_mode = luma_mode;
                    best_rank = rank0 + 1;
                } else if cost < runner_up_cost {
                    third_cost = runner_up_cost;
                    third_mode = runner_up_mode;
                    runner_up_cost = cost;
                    runner_up_mode = luma_mode;
                } else if cost < third_cost {
                    third_cost = cost;
                    third_mode = luma_mode;
                }
            }
        };
        if rdo2_cheap_luma {
            self.with_tt_trial_flags(true, false, |this| screen_candidates(this));
        } else {
            screen_candidates(self);
        }
        self.record_luma_winner_rank(best_rank);
        let close_margin = tt_budget.close_call_margin * self.rdo2_luma_close_mult;
        self.record_close_call(best_cost, runner_up_cost, close_margin);
        if self.trace.enabled {
            self.trace.note_decision(
                crate::trace::DecisionKind::Luma,
                x0,
                y0,
                tt_log2,
                false,
                &trace_cands,
            );
        }
        self.put_luma_trace_cands(trace_cands);

        // The exact recheck re-prices candidates through the single-block scratch
        // evaluator, so it only applies to CUs at or below the max transform size.
        // Larger CUs are already screened by their materialised subtree, which
        // does its own per-TU exact escalation, so the screen winner stands.
        if use_scratch_screen
            && (self.luma_trial_quality(tt_budget) == TrialQuality::FastRd || rdo2_cheap_luma)
            && best_luma_mode != runner_up_mode
            && Self::is_close_call(best_cost, runner_up_cost, close_margin)
        {
            let cheap_winner = best_luma_mode;
            self.stats.rdo2_luma_exact_escalations += u64::from(rdo2_cheap_luma);
            self.stats.rdo2_luma_scratch_exact_escalations += u64::from(use_scratch_screen);
            let exact_modes = [best_luma_mode, runner_up_mode, third_mode];
            let exact_count = if rdo2_cheap_luma
                && third_mode != best_luma_mode
                && third_mode != runner_up_mode
                && Self::is_close_call(best_cost, third_cost, close_margin)
            {
                3
            } else {
                2
            };
            best_luma_mode = self.rdo2_exact_recheck_luma_candidates(
                x0,
                y0,
                log2_cb_size,
                &exact_modes[..exact_count],
                mpm,
                ctxs,
            );
            if rdo2_cheap_luma && best_luma_mode != cheap_winner {
                self.stats.rdo2_luma_exact_changed_winner += 1;
            }
            if use_scratch_screen && best_luma_mode != cheap_winner {
                self.stats.rdo2_luma_scratch_changed_winner += 1;
            }
        }

        self.restore_frame_region(base_frame);
        self.restore_mode_region(base_mode_map);
        self.restore_tu_depth_region(base_tu_depth);
        let chroma = self.decide_chroma_mode(x0, y0, tt_log2, best_luma_mode, ctxs);
        self.store_mode(x0, y0, log2_cb_size, best_luma_mode);
        let tt = self.rdo2_analyze_tt(
            x0,
            y0,
            log2_cb_size,
            0,
            best_luma_mode,
            chroma.plan.mode,
            ctxs,
        );
        CuLeaf {
            mpm,
            luma_mode: best_luma_mode,
            chroma_mode_idx: chroma.plan.mode_idx,
            confidence: DecisionConfidence::Clear,
            tt,
            nxn: None,
        }
    }
}
