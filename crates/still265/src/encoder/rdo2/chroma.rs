//! Staged chroma mode decision for `rdo2` (slice 4): "late and narrow".
//!
//! `decide_chroma_mode` (in `rdo.rs`) keeps the shared work — the rough SATD
//! screen across all chroma modes and the narrowing to the top-`n` RD
//! candidates — and then delegates the *RD decision* to this module:
//! `rdo2_chroma_rd_decision` codes each narrowed candidate, picks the best,
//! escalates close calls to an exact recheck, and returns the winner with an
//! **exact** reconstruction cache.
//!
//! `BPG_RDO2_CHROMA` (`self.rdo2_chroma`, `Best` only) forces a cheap Best
//! trial path (no trial RDOQ + approximate residual bits) for the candidate
//! screen. `BPG_RDO2_CHROMA_SCRATCH` routes the common one-block chroma geometry
//! through the non-committing rdo2 evaluator, so candidate ranking is cost-only:
//! it never builds a chroma cache and the final writer-compatible tree is still
//! materialized downstream by `final_code_tt`.

use bpg_hevc_decode::hevc::intra::predict_intra_into;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::Contexts;
use crate::plan::ChromaModePlan;
use crate::preanalysis::RegionClass;
use crate::trace::{WorkBucket, WorkSample};
use crate::Effort;

use super::super::types::{chroma_pred_mode, chroma_tb_geom, CHROMA_DM_IDX};
use super::cost::estimate_intra_chroma_mode_bits;
use super::policy::{EvalKind, EvalPolicy};

/// `decide_chroma_mode` returns this; defined alongside the legacy path.
use super::super::snapshot::ChromaDecision;

#[derive(Clone, Copy)]
struct ChromaCandidateCost {
    mode_idx: u8,
    rank: usize,
    cost: f64,
    exact_estimates: u32,
}

impl<'a> super::super::Encoder<'a> {
    pub(in crate::encoder) fn chroma_mode_from_idx(luma_mode: u8, mode_idx: u8) -> u8 {
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

    pub(in crate::encoder) fn decide_chroma_mode(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> ChromaDecision {
        let work_start = self.trace.enabled.then(std::time::Instant::now);
        let Some((cx, cy, clog2, count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) else {
            return ChromaDecision {
                plan: ChromaModePlan {
                    mode: luma_mode,
                    mode_idx: CHROMA_DM_IDX,
                },
            };
        };
        let budget = self.block_budget(x0, y0, log2_size, 1);
        if budget.chroma_rd_candidates == 0 {
            self.record_chroma_winner_rank(1);
            return ChromaDecision {
                plan: ChromaModePlan {
                    mode: luma_mode,
                    mode_idx: CHROMA_DM_IDX,
                },
            };
        }
        let size = 1usize << clog2;
        let cat = self.cat;

        let srcs: Vec<(Vec<u16>, Vec<u16>)> = (0..count)
            .map(|i| {
                let ty = cy + i as u32 * size as u32;
                (
                    self.source_block(1, cx, ty, size),
                    self.source_block(2, cx, ty, size),
                )
            })
            .collect();
        let src8s: Vec<(Option<Vec<u8>>, Option<Vec<u8>>)> = srcs
            .iter()
            .map(|(cb, cr)| {
                if self.bit_depth == 8 {
                    (
                        Some(cb.iter().map(|&v| v.min(255) as u8).collect()),
                        Some(cr.iter().map(|&v| v.min(255) as u8).collect()),
                    )
                } else {
                    (None, None)
                }
            })
            .collect();

        let mut scored: Vec<(u64, u8)> = Vec::with_capacity(5);
        let mut pred_buf = Vec::with_capacity(size * size);
        let mut pred8_buf = Vec::with_capacity(size * size);
        for mode_idx in 0..=CHROMA_DM_IDX {
            self.stats.chroma_rough_predictions += 1;
            let mode = Self::chroma_mode_from_idx(luma_mode, mode_idx);
            let pred = chroma_pred_mode(cat, mode);
            let pred_mode = IntraPredMode::from_u8(pred).unwrap_or(IntraPredMode::Dc);

            let mut cb_cost = 0u64;
            let mut cr_cost = 0u64;
            for (i, (cb_src, cr_src)) in srcs.iter().enumerate() {
                let ty = cy + i as u32 * size as u32;
                let (cb_src8, cr_src8) = &src8s[i];

                pred_buf.clear();
                pred_buf.resize(size * size, 0);
                predict_intra_into(
                    &self.frame,
                    cx,
                    ty,
                    clog2,
                    pred_mode,
                    1,
                    true,
                    &mut pred_buf,
                    size,
                );
                cb_cost += self.satd_block_cost(
                    cb_src,
                    cb_src8.as_deref(),
                    &pred_buf,
                    size,
                    &mut pred8_buf,
                ) as u64;

                pred_buf.clear();
                pred_buf.resize(size * size, 0);
                predict_intra_into(
                    &self.frame,
                    cx,
                    ty,
                    clog2,
                    pred_mode,
                    2,
                    true,
                    &mut pred_buf,
                    size,
                );
                cr_cost += self.satd_block_cost(
                    cr_src,
                    cr_src8.as_deref(),
                    &pred_buf,
                    size,
                    &mut pred8_buf,
                ) as u64;
            }

            let mode_bits = estimate_intra_chroma_mode_bits(ctxs, mode_idx);
            let cost = (cb_cost + cr_cost) * CabacEstimator::SCALE + mode_bits;
            scored.push((cost, mode_idx));
        }

        scored.sort_by_key(|&(cost, _)| cost);
        let base = budget.chroma_rd_candidates_base as usize;
        let mut n = budget.chroma_rd_candidates as usize;
        if n > base {
            self.stats.chroma_candidate_expansions += 1;
        }
        let chroma_critical = self.best2_chroma_protect
            && self.analysis.region_class_at(x0, y0, log2_size) == RegionClass::ChromaCritical;
        if let Some(margin) = self.best2_chroma_gate.filter(|_| !chroma_critical) {
            let best_rough = scored[0].0 as f64;
            let thresh = best_rough * (1.0 + margin);
            let within = scored
                .iter()
                .take(n)
                .filter(|&&(c, _)| (c as f64) <= thresh)
                .count();
            n = within.clamp(1, n);
        }
        if n <= 1 {
            if let Some(start) = work_start {
                self.trace.note_work(
                    WorkBucket::RoughChroma,
                    WorkSample {
                        wall_ns: start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                        log2_size: clog2,
                        c_idx: 1,
                        prediction_calls: scored.len() as u64 * u64::from(count) * 2,
                        source_block_calls: u64::from(count) * 2,
                        ..WorkSample::default()
                    },
                );
            }
            let idx = scored[0].1;
            self.record_chroma_winner_rank(1);
            return ChromaDecision {
                plan: ChromaModePlan {
                    mode: Self::chroma_mode_from_idx(luma_mode, idx),
                    mode_idx: idx,
                },
            };
        }

        let cand_idxs: Vec<u8> = scored.iter().take(n).map(|&(_, idx)| idx).collect();
        if let Some(start) = work_start {
            self.trace.note_work(
                WorkBucket::RoughChroma,
                WorkSample {
                    wall_ns: start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    log2_size: clog2,
                    c_idx: 1,
                    prediction_calls: scored.len() as u64 * u64::from(count) * 2,
                    source_block_calls: u64::from(count) * 2,
                    ..WorkSample::default()
                },
            );
        }
        self.rdo2_chroma_rd_decision(
            ctxs,
            x0,
            y0,
            log2_size,
            luma_mode,
            cx,
            cy,
            clog2,
            u32::from(count),
            size,
            &cand_idxs,
            budget.close_call_margin,
        )
    }

    fn take_chroma_trace_cands(&mut self) -> Vec<crate::trace::CandRec> {
        let mut trace_cands = std::mem::take(&mut self.search_scratch.chroma_trace_cands);
        trace_cands.clear();
        trace_cands
    }

    fn put_chroma_trace_cands(&mut self, trace_cands: Vec<crate::trace::CandRec>) {
        self.search_scratch.chroma_trace_cands = trace_cands;
    }

    #[allow(clippy::too_many_arguments)]
    fn rdo2_eval_chroma_candidate_scratch(
        &mut self,
        ctxs: &Contexts,
        mode_idx: u8,
        rank: usize,
        luma_mode: u8,
        cx: u32,
        cy: u32,
        clog2: u8,
        count: u32,
        kind: EvalKind,
    ) -> ChromaCandidateCost {
        let mode = Self::chroma_mode_from_idx(luma_mode, mode_idx);
        let pred = chroma_pred_mode(self.cat, mode);
        let bucket = match kind {
            EvalKind::CheapTrial => WorkBucket::ChromaCheap,
            EvalKind::ExactTrial => WorkBucket::ChromaExact,
            EvalKind::Final => WorkBucket::FinalReplay,
        };
        let mut policy = EvalPolicy::for_kind(kind).with_bucket(bucket);
        policy.commit = false;
        let exact0 = self.stats.residual_bit_estimates;

        let size = 1u32 << clog2;
        let mut distortion = 0u64;
        let mut frac_bits = 0u64;
        assert!(
            count <= 2,
            "chroma candidate scratch evaluator expects at most two stacked chroma blocks"
        );
        for i in 0..count {
            let ty = cy + i * size;
            let cb = self.rdo2_eval_leaf_block(ctxs, cx, ty, clog2, 1, pred, self.cur_qp_c, policy);
            let cr = self.rdo2_eval_leaf_block(ctxs, cx, ty, clog2, 2, pred, self.cur_qp_c, policy);
            distortion += cb.distortion + cr.distortion;
            frac_bits += cb.frac_bits + cr.frac_bits;
            match kind {
                EvalKind::CheapTrial => self.stats.rdo2_chroma_scratch_cheap_evals += 2,
                EvalKind::ExactTrial => self.stats.rdo2_chroma_scratch_exact_evals += 2,
                EvalKind::Final => {}
            }
            self.stats.rdo2_chroma_scratch_legacy_evals_skipped += 2;
        }

        let mode_bits = estimate_intra_chroma_mode_bits(ctxs, mode_idx);
        ChromaCandidateCost {
            mode_idx,
            rank,
            cost: self.rd_cost(distortion, frac_bits + mode_bits),
            exact_estimates: (self.stats.residual_bit_estimates - exact0) as u32,
        }
    }

    fn apply_chroma_cost_candidate(
        cand: ChromaCandidateCost,
        best_idx: &mut u8,
        best_mode: &mut u8,
        best_rank: &mut usize,
        best_cost: &mut f64,
        runner_up_idx: &mut u8,
        runner_up_rank: &mut usize,
        runner_up_cost: &mut f64,
        luma_mode: u8,
    ) {
        if cand.cost < *best_cost {
            *runner_up_cost = *best_cost;
            *runner_up_idx = *best_idx;
            *runner_up_rank = *best_rank;
            *best_cost = cand.cost;
            *best_idx = cand.mode_idx;
            *best_mode = Self::chroma_mode_from_idx(luma_mode, cand.mode_idx);
            *best_rank = cand.rank;
        } else if cand.cost < *runner_up_cost {
            *runner_up_cost = cand.cost;
            *runner_up_idx = cand.mode_idx;
            *runner_up_rank = cand.rank;
        }
    }

    /// RD-decide the chroma mode from the narrowed candidate list `cand_idxs`
    /// (rough-sorted chroma mode indices, `len >= 2`). Returns the winner and its
    /// **exact** reconstruction cache. The caller has already validated geometry
    /// and recorded none of the winner bookkeeping yet.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::encoder) fn rdo2_chroma_rd_decision(
        &mut self,
        ctxs: &Contexts,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        cx: u32,
        cy: u32,
        clog2: u8,
        count: u32,
        size: usize,
        cand_idxs: &[u8],
        close_call_margin: f64,
    ) -> ChromaDecision {
        let cheap = self.rdo2_chroma && self.effort == Effort::Best;
        if cheap {
            self.stats.rdo2_chroma_cheap_cu_decisions += 1;
        }
        // The scratch evaluator is the shared chroma coding primitive for every
        // effort tier (`chroma_tb_geom` guarantees `count <= 2` and a chroma TB no
        // larger than the max transform size). The `cheap` Best screen above is a
        // separate speed trick layered on top; it is not what makes scratch usable.
        let scratch = count <= 2;

        let mut best_idx = cand_idxs[0];
        let mut best_mode = Self::chroma_mode_from_idx(luma_mode, best_idx);
        let mut best_rank = 1usize;
        let mut best_cost = f64::MAX;
        let mut runner_up_idx = cand_idxs[0];
        let mut runner_up_rank = 1usize;
        let mut runner_up_cost = f64::MAX;
        let mut trace_cands = self.take_chroma_trace_cands();

        if scratch {
            for (rank0, &mode_idx) in cand_idxs.iter().enumerate() {
                self.stats.rdo2_chroma_scratch_candidates += 1;
                let kind = if cheap {
                    EvalKind::CheapTrial
                } else {
                    EvalKind::ExactTrial
                };
                let cand = self.rdo2_eval_chroma_candidate_scratch(
                    ctxs,
                    mode_idx,
                    rank0 + 1,
                    luma_mode,
                    cx,
                    cy,
                    clog2,
                    count,
                    kind,
                );
                if self.trace.enabled {
                    trace_cands.push(crate::trace::CandRec::new(cand.cost, cand.exact_estimates));
                }
                Self::apply_chroma_cost_candidate(
                    cand,
                    &mut best_idx,
                    &mut best_mode,
                    &mut best_rank,
                    &mut best_cost,
                    &mut runner_up_idx,
                    &mut runner_up_rank,
                    &mut runner_up_cost,
                    luma_mode,
                );
            }
        } else {
            panic!("unsupported chroma candidate geometry for rdo2 scratch evaluator");
        }
        let _ = size;

        // Cheap screen may misorder close decisions: re-score the top two exactly
        // (cost only — the winner's reconstruction is produced exactly downstream,
        // and `rdo2_analyze_tt` ignores any chroma cache) and keep the better mode.
        if cheap
            && runner_up_idx != best_idx
            && Self::is_close_call(best_cost, runner_up_cost, close_call_margin)
        {
            self.stats.rdo2_chroma_exact_escalations += 1;
            let (exact_best, exact_ru) = if scratch {
                self.stats.rdo2_chroma_scratch_exact_escalations += 1;
                let best = self.rdo2_eval_chroma_candidate_scratch(
                    ctxs,
                    best_idx,
                    best_rank,
                    luma_mode,
                    cx,
                    cy,
                    clog2,
                    count,
                    EvalKind::ExactTrial,
                );
                let ru = self.rdo2_eval_chroma_candidate_scratch(
                    ctxs,
                    runner_up_idx,
                    runner_up_rank,
                    luma_mode,
                    cx,
                    cy,
                    clog2,
                    count,
                    EvalKind::ExactTrial,
                );
                (best.cost, ru.cost)
            } else {
                panic!("unsupported chroma exact-recheck geometry for rdo2 scratch evaluator")
            };
            if exact_ru < exact_best {
                self.stats.rdo2_chroma_exact_changed_winner += 1;
                if scratch {
                    self.stats.rdo2_chroma_scratch_changed_winner += 1;
                }
                best_idx = runner_up_idx;
                best_mode = Self::chroma_mode_from_idx(luma_mode, runner_up_idx);
                best_rank = runner_up_rank;
            }
        }

        self.record_chroma_winner_rank(best_rank);
        self.record_close_call(best_cost, runner_up_cost, close_call_margin);
        if self.trace.enabled {
            self.trace.note_decision(
                crate::trace::DecisionKind::Chroma,
                x0,
                y0,
                log2_size,
                false,
                &trace_cands,
            );
        }
        self.put_chroma_trace_cands(trace_cands);

        ChromaDecision {
            plan: ChromaModePlan {
                mode: best_mode,
                mode_idx: best_idx,
            },
        }
    }
}
