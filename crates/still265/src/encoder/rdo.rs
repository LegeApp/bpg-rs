//! Rate-distortion optimisation: transform/quant/RDOQ, TU/CU mode and split
//! decision, CABAC bit estimation, and costing.

use bpg_hevc_decode::hevc::intra::{
    fill_mpm_candidates, predict_intra, predict_intra_into,
};
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::{ctx, Contexts};
use crate::preanalysis::RegionClass;
use crate::effort::{
    BlockSearchBudget, ComponentKind, RmdModeSet, SplitSearch, TrialQuality,
};
use crate::plan::{
    BlockEstimate, BlockPlan, ChromaModePlan, CuLeafPlan, CuPlan, DecisionConfidence, LumaModePlan,
    ParentChromaPlan, RdCost, TrialResult, TtPlan, WorkStage,
};
use crate::primitives;
use crate::rdoq;
use crate::residual::{estimate_residual_bits, get_scan_order, ResidualEstimateCache};
use crate::transform;
use crate::Effort;

use super::snapshot::{ChromaLeafCache, ChromaDecision};
use super::types::*;
use super::Encoder;

impl<'a> super::Encoder<'a> {
    pub(super) fn satd_block_cost(
        &self,
        src: &[u16],
        src8: Option<&[u8]>,
        pred: &[u16],
        size: usize,
        pred8_scratch: &mut Vec<u8>,
    ) -> u32 {
        match self.bit_depth {
            8 => {
                let src8 = src8.expect("8-bit source scratch must be present");
                pred8_scratch.clear();
                pred8_scratch.extend(pred.iter().map(|&v| v.min(255) as u8));
                primitives::satd_u8(src8, size, pred8_scratch, size, size)
            }
            10 | 12 => primitives::satd_u16(src, size, pred, size, size),
            _ => unreachable!("supported bit depths are checked at encode entry"),
        }
    }

    /// heindel2016 §3.1 Global Angular Mode Exclusion: decide whether to drop
    /// the angular rough-mode sweep for a homogeneous luma block (only Planar/DC
    /// and the MPM candidates remain). Uses the source-block variance as the
    /// structure signal (bit-depth normalized). `Fastest`/`Fast`/`Balanced`
    /// prune (at decreasing thresholds); `Good`/`Best` never do.
    pub(super) fn prune_angular_modes(&self, src: &[u16], size: usize, budget: BlockSearchBudget) -> bool {
        // Variance threshold in 8-bit² units, scaled to the working bit depth.
        let Some(th_8bit) = budget.angular_prune_var_threshold_8bit else {
            return false;
        };
        let scale = 1i64 << (2 * (self.bit_depth as i64 - 8));
        let threshold = th_8bit * scale;

        let n = (size * size) as i64;
        let sum: i64 = src.iter().map(|&v| v as i64).sum();
        let mean = sum / n;
        let var = src.iter().map(|&v| (v as i64 - mean).pow(2)).sum::<i64>() / n;
        var < threshold
    }

    pub(super) fn score_luma_rough_mode(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        mode: u8,
        ctxs: &Contexts,
        mpm: [IntraPredMode; 3],
        src: &[u16],
        src8: Option<&[u8]>,
        pred: &mut Vec<u16>,
        pred8: &mut Vec<u8>,
    ) -> u64 {
        let size = 1usize << log2_size;
        self.stats.luma_rough_predictions += 1;
        let mode_pred = IntraPredMode::from_u8(mode).unwrap();
        pred.clear();
        pred.resize(size * size, 0);
        predict_intra_into(
            &self.frame,
            x0,
            y0,
            log2_size,
            mode_pred,
            0,
            true,
            pred,
            size,
        );
        let satd = self.satd_block_cost(src, src8, pred, size, pred8) as u64;
        let mode_bits = estimate_intra_luma_mode_bits(ctxs, mpm, mode);
        if self.effort == Effort::Best {
            let cost = if self.best2_rough_lambda {
                // x265-parity rough cost: SATD + lambda_sad * bits.
                self.rough_cost(satd, mode_bits)
            } else {
                // Legacy: SSE-domain lambda applied to SATD (over-weights bits).
                self.rd_cost(satd, mode_bits)
            };
            return (cost * 1024.0).round() as u64;
        }
        satd * CabacEstimator::SCALE + mode_bits
    }

    pub(super) fn decide_luma_modes(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> LumaModePlan {
        let size = 1usize << log2_size;
        let src = self.source_block(0, x0, y0, size);
        let has_src8 = self.bit_depth == 8;
        let mut src8 = std::mem::take(&mut self.scratch_src8);
        src8.clear();
        if has_src8 {
            src8.extend(src.iter().map(|&v| v.min(255) as u8));
        }

        let cand_a = self.neighbor_left_mode(x0, y0);
        let cand_b = self.neighbor_above_mode(x0, y0);
        let mpm = fill_mpm_candidates(cand_a, cand_b);

        let budget = self.block_budget(x0, y0, log2_size, 0);
        let local_prune = self.prune_angular_modes(&src, size, budget);
        let prune_angular = match budget.angular_prune {
            Some(forced) => {
                if forced && !local_prune {
                    self.stats.policy_angular_forced += 1;
                } else if !forced && local_prune {
                    self.stats.policy_angular_guarded += 1;
                }
                forced
            }
            None => local_prune,
        };
        if prune_angular {
            self.stats.angular_exclusions += 1;
        }

        let mut scored = std::mem::take(&mut self.scratch_scored);
        scored.clear();
        let mut pred = std::mem::take(&mut self.scratch_pred);
        let mut pred8 = std::mem::take(&mut self.scratch_pred8);
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        if let (
            false,
            RmdModeSet::Progressive {
                coarse_step,
                top_regions,
                refine_radius,
            },
        ) = (prune_angular, budget.rmd_mode_set)
        {
            let mut seen = [false; 35];
            let mut coarse_scored = Vec::new();

            for m in RmdModeSet::MpmPlanarDcOnly.modes(mpm_u8) {
                if seen[m as usize] {
                    continue;
                }
                seen[m as usize] = true;
                let src8_ref = has_src8.then_some(src8.as_slice());
                let cost = self.score_luma_rough_mode(
                    x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                );
                scored.push((cost, m));
            }

            let step = coarse_step.max(1) as usize;
            for m in (2u8..=34).step_by(step) {
                if seen[m as usize] {
                    continue;
                }
                seen[m as usize] = true;
                let src8_ref = has_src8.then_some(src8.as_slice());
                let cost = self.score_luma_rough_mode(
                    x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                );
                scored.push((cost, m));
                coarse_scored.push((cost, m));
            }

            coarse_scored.sort_by_key(|&(cost, _)| cost);
            for &(_, center) in coarse_scored.iter().take(top_regions.max(1) as usize) {
                let lo = center.saturating_sub(refine_radius).max(2);
                let hi = center.saturating_add(refine_radius).min(34);
                for m in lo..=hi {
                    if seen[m as usize] {
                        continue;
                    }
                    seen[m as usize] = true;
                    let src8_ref = has_src8.then_some(src8.as_slice());
                    let cost = self.score_luma_rough_mode(
                        x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                    );
                    scored.push((cost, m));
                }
            }
        } else {
            for m in budget.rough_luma_modes(mpm_u8, prune_angular) {
                let src8_ref = has_src8.then_some(src8.as_slice());
                let cost = self.score_luma_rough_mode(
                    x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                );
                scored.push((cost, m));
            }
        }

        scored.sort_by_key(|&(cost, _)| cost);
        let base = budget.luma_rd_candidates_base as usize;
        let mut n = budget.luma_rd_candidates as usize;
        if n > base {
            self.stats.luma_candidate_expansions += 1;
        }
        n = match budget.rmd_prune_factor {
            Some(factor) if !scored.is_empty() && !self.best_luma_leaf_screen => {
                let best = scored[0].0 as f64;
                let mut keep = 1usize;
                while keep < n && keep < scored.len() && (scored[keep].0 as f64) <= best * factor {
                    keep += 1;
                }
                self.stats.rmd_prunes += (n.min(scored.len()) - keep) as u64;
                keep
            }
            _ => n,
        };
        let candidates = if self.best_luma_leaf_screen && !scored.is_empty() {
            let max_cands = (5 + ((ct_depth as usize) >> 1)).min(8);
            let best = scored[0].0;
            let padded_best = best.saturating_add(best >> 2);
            let mpm0 = mpm_u8[0];
            let mut selected: Vec<(u64, u8)> = scored
                .iter()
                .copied()
                .filter(|&(cost, mode)| cost <= padded_best || mode == mpm0)
                .collect();
            let mut mpm0_forced = selected
                .iter()
                .any(|&(cost, mode)| mode == mpm0 && cost > padded_best);

            selected.sort_by_key(|&(cost, _)| cost);
            if selected.len() > max_cands {
                selected.truncate(max_cands);
            }
            if !selected.iter().any(|&(_, mode)| mode == mpm0) {
                if let Some(&mpm0_rec) = scored.iter().find(|&&(_, mode)| mode == mpm0) {
                    mpm0_forced = true;
                    if selected.len() < max_cands {
                        selected.push(mpm0_rec);
                    } else if let Some(last) = selected.last_mut() {
                        *last = mpm0_rec;
                    }
                    selected.sort_by_key(|&(cost, _)| cost);
                }
            }
            self.stats.luma_candidate_expansions +=
                selected.len().saturating_sub(base.min(selected.len())) as u64;
            self.trace
                .note_luma_rmd_selection(scored.len(), selected.len(), mpm0_forced);
            selected.into_iter().map(|(_, mode)| mode).collect()
        } else {
            self.trace
                .note_luma_rmd_selection(scored.len(), n.min(scored.len()), false);
            scored.iter().take(n).map(|&(_, m)| m).collect()
        };
        scored.clear();
        self.scratch_scored = scored;
        self.scratch_src8 = src8;
        self.scratch_pred = pred;
        self.scratch_pred8 = pred8;
        LumaModePlan { candidates }
    }

    pub(super) fn chroma_mode_from_idx(luma_mode: u8, mode_idx: u8) -> u8 {
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

    pub(super) fn decide_chroma_mode(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> ChromaDecision {
        let Some((cx, cy, clog2, count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) else {
            return ChromaDecision {
                plan: ChromaModePlan {
                    mode: luma_mode,
                    mode_idx: CHROMA_DM_IDX,
                },
                leaf_cache: None,
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
                leaf_cache: None,
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
        // x265 high-RD intra runs real chroma coding for every allowed chroma
        // mode; our gate cuts obvious losers cheaply. Protect ChromaCritical
        // regions (high chroma detail) from the gate so salient chroma still
        // gets full RD — a scheduler prior, not lossy pruning (advisor gap #9).
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
            let idx = scored[0].1;
            self.record_chroma_winner_rank(1);
            return ChromaDecision {
                plan: ChromaModePlan {
                    mode: Self::chroma_mode_from_idx(luma_mode, idx),
                    mode_idx: idx,
                },
                leaf_cache: None,
            };
        }

        let mut best_idx = scored[0].1;
        let mut best_mode = Self::chroma_mode_from_idx(luma_mode, best_idx);
        let mut best_rank = 1usize;
        let mut best_cost = f64::MAX;
        let mut runner_up_cost = f64::MAX;
        let mut best_cache = None;
        let mut trace_cands: Vec<crate::trace::CandRec> = Vec::new();
        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);
        for (rank0, &(_, mode_idx)) in scored.iter().take(n).enumerate() {
            let mode = Self::chroma_mode_from_idx(luma_mode, mode_idx);
            let pred = chroma_pred_mode(cat, mode);
            let cand_e0 = self.stats.residual_bit_estimates;

            self.restore_frame_region(&base_frame);
            let mut distortion = 0u64;
            let mut frac_bits = 0u64;
            let mut coded = Vec::with_capacity(count as usize);
            for i in 0..count {
                let ty = cy + i as u32 * size as u32;
                let cb = self.trial_code_block(
                    ctxs,
                    cx,
                    ty,
                    clog2,
                    1,
                    pred,
                    self.cur_qp_c,
                    WorkStage::ChromaTrial,
                );
                distortion += cb.estimate.cost.distortion;
                let cr = self.trial_code_block(
                    ctxs,
                    cx,
                    ty,
                    clog2,
                    2,
                    pred,
                    self.cur_qp_c,
                    WorkStage::ChromaTrial,
                );
                distortion += cr.estimate.cost.distortion;
                frac_bits += cb.estimate.cost.frac_bits + cr.estimate.cost.frac_bits;
                coded.push((cb.coded, cr.coded));
            }

            let mode_bits = estimate_intra_chroma_mode_bits(ctxs, mode_idx);
            let cost = self.rd_cost(distortion, frac_bits + mode_bits);
            if self.trace.enabled {
                let cand_exact = (self.stats.residual_bit_estimates - cand_e0) as u32;
                let mut approx = 0u64;
                for (cbc, crc) in &coded {
                    if cbc.cbf {
                        approx += self.approx_residual_frac_bits(ctxs, &cbc.levels, clog2, 1);
                    }
                    if crc.cbf {
                        approx += self.approx_residual_frac_bits(ctxs, &crc.levels, clog2, 2);
                    }
                }
                let flip_cost = self.rd_cost(distortion, approx + mode_bits);
                trace_cands.push(crate::trace::CandRec::with_flip(
                    cost, flip_cost, cand_exact,
                ));
            }

            if cost < best_cost {
                runner_up_cost = best_cost;
                best_cost = cost;
                best_idx = mode_idx;
                best_mode = mode;
                best_rank = rank0 + 1;
                let mut iter = coded.into_iter();
                let (cb, cr) = iter
                    .next()
                    .unwrap_or((CodedBlock::empty(), CodedBlock::empty()));
                let (cb1, cr1) = iter
                    .next()
                    .unwrap_or((CodedBlock::empty(), CodedBlock::empty()));
                best_cache = Some(
                    self.snapshot_chroma_leaf_cache(x0, y0, log2_size, clog2, cb, cr, cb1, cr1),
                );
            } else if cost < runner_up_cost {
                runner_up_cost = cost;
            }
        }
        self.restore_frame_region(&base_frame);
        self.record_chroma_winner_rank(best_rank);
        self.record_close_call(best_cost, runner_up_cost, budget.close_call_margin);
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

        ChromaDecision {
            plan: ChromaModePlan {
                mode: best_mode,
                mode_idx: best_idx,
            },
            leaf_cache: best_cache,
        }
    }

    pub(super) fn code_block_internal(
        &mut self,
        ctxs: &Contexts,
        plan: &BlockPlan,
        stage: WorkStage,
    ) -> CodedBlock {
        debug_assert!(
            stage != WorkStage::FinalCode || plan.quality == TrialQuality::Final,
            "final block coding must use final-quality plans"
        );
        let tcb = self.prof.on.then(std::time::Instant::now);
        let x = plan.x;
        let y = plan.y;
        let log2_size = plan.log2_size;
        let c_idx = plan.c_idx;
        let mode = plan.mode;
        let qp = plan.qp;
        let budget = self.block_budget(x, y, log2_size, c_idx);
        self.stats.code_block_calls += 1;
        if stage == WorkStage::FinalCode {
            self.stats.final_coded_blocks += 1;
        } else {
            self.stats.trial_coded_blocks += 1;
        }
        let size = 1usize << log2_size;
        let pred_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);

        predict_intra(&mut self.frame, x, y, log2_size, pred_mode, c_idx, true);

        let mut residual = std::mem::take(&mut self.scratch_residual);
        residual.clear();
        residual.resize(size * size, 0);
        let (plane, stride) = self.frame.plane(c_idx);
        let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);
        if x + size as u32 <= sw && y + size as u32 <= sh {
            let pred = &plane[y as usize * stride + x as usize..];
            let src = &src_plane[y as usize * src_stride + x as usize..];
            primitives::sub_residual(src, src_stride, pred, stride, &mut residual, size);
        } else {
            for j in 0..size {
                for i in 0..size {
                    let pred = plane[(y as usize + j) * stride + x as usize + i] as i32;
                    let s = self.src_sample(c_idx, x + i as u32, y + j as u32) as i32;
                    residual[j * size + i] = (s - pred) as i16;
                }
            }
        }

        let is_dst = log2_size == 2 && c_idx == 0;
        self.stats.forward_transforms += 1;
        let mut coeffs = std::mem::take(&mut self.scratch_coeffs);
        let mut transform_tmp = std::mem::take(&mut self.scratch_transform_tmp);
        transform::forward_transform_into(
            &residual,
            log2_size,
            is_dst,
            self.bit_depth,
            &mut coeffs,
            &mut transform_tmp,
        );
        self.scratch_residual = residual;
        let do_rdoq = Self::rdoq_enabled_for_block(stage, plan.quality, budget);
        let (levels, nnz) = if self.single_scan_rdoq && do_rdoq {
            if stage == WorkStage::FinalCode {
                self.stats.final_rdoq_blocks += 1;
            } else {
                self.stats.trial_rdoq_blocks += 1;
            }
            let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
            let tr = self.prof.on.then(std::time::Instant::now);
            let r = rdoq::rdoq_single_scan(
                ctxs,
                &coeffs,
                log2_size,
                c_idx,
                qp,
                self.bit_depth,
                scan,
                self.lambda(),
            );
            if let Some(tr) = tr {
                self.prof.rdoq += tr.elapsed();
            }
            r
        } else {
            let (levels, _) = transform::quantize(&coeffs, log2_size, qp, self.bit_depth);
            let nnz = levels.iter().filter(|&&v| v != 0).count() as u32;
            let passes = if do_rdoq {
                budget.rdoq_passes(log2_size, ComponentKind::from_c_idx(c_idx), nnz)
            } else {
                0
            };
            if passes == 0 {
                (levels, nnz)
            } else {
                if stage == WorkStage::FinalCode {
                    self.stats.final_rdoq_blocks += 1;
                } else {
                    self.stats.trial_rdoq_blocks += 1;
                }
                self.refine_levels_rdoq_limited(
                    ctxs, &coeffs, levels, log2_size, c_idx, mode, qp, passes,
                )
            }
        };
        self.scratch_coeffs = coeffs;
        self.scratch_transform_tmp = transform_tmp;
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

        let exact = cbf && Self::exact_residual_bits_for_block(stage, plan.quality, budget);
        let frac_bits = if !cbf {
            0
        } else if exact {
            self.residual_frac_bits(ctxs, &levels, log2_size, c_idx, mode)
        } else {
            self.approx_residual_frac_bits(ctxs, &levels, log2_size, c_idx)
        };

        if self.trace.enabled {
            self.trace
                .note_code_block(stage, c_idx, x, y, log2_size, cbf, exact, frac_bits);
        }

        if let Some(t) = tcb {
            self.prof.code_block += t.elapsed();
        }
        CodedBlock {
            levels,
            cbf,
            frac_bits,
        }
    }

    pub(super) fn rdoq_enabled_for_block(
        stage: WorkStage,
        quality: TrialQuality,
        budget: BlockSearchBudget,
    ) -> bool {
        match stage {
            WorkStage::FinalCode => true,
            _ => match quality {
                TrialQuality::Rough | TrialQuality::FastRd => false,
                TrialQuality::FullRd => budget.rdoq_for_trials,
                TrialQuality::Final => true,
            },
        }
    }

    pub(super) fn exact_residual_bits_for_block(
        stage: WorkStage,
        quality: TrialQuality,
        budget: BlockSearchBudget,
    ) -> bool {
        match stage {
            WorkStage::FinalCode => true,
            _ => match quality {
                TrialQuality::Rough | TrialQuality::FastRd => false,
                TrialQuality::FullRd => true,
                TrialQuality::Final => budget.exact_residual_bits_for_trials,
            },
        }
    }

    pub(super) fn approx_residual_frac_bits(
        &self,
        ctxs: &Contexts,
        levels: &[i16],
        log2_size: u8,
        c_idx: u8,
    ) -> u64 {
        let nnz = levels.iter().filter(|&&v| v != 0).count() as u64;
        if nnz == 0 {
            return 0;
        }

        let scale = CabacEstimator::SCALE;
        let bits = |ci: usize, bin: u8| -> u64 { ctxs.models[ci].entropy_bits(bin) as u64 };

        let ctx_set = if c_idx > 0 { 0usize } else { 2 };
        let gt1_ci =
            ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + if c_idx > 0 { 16 } else { 0 } + ctx_set * 4 + 1;
        let gt2_ci = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + if c_idx > 0 { 4 } else { 0 } + ctx_set;
        let gt1_0 = bits(gt1_ci, 0);
        let gt1_1 = bits(gt1_ci, 1);
        let gt2_0 = bits(gt2_ci, 0);
        let gt2_1 = bits(gt2_ci, 1);

        let sig_ci = ctx::SIG_COEFF_FLAG + if c_idx > 0 { 27 } else { 0 };
        let sig_1 = bits(sig_ci, 1);
        let sig_0 = bits(sig_ci, 0);

        let area = 1u64 << (2 * log2_size as u64);
        let sig_bits = nnz * sig_1 + (area / 4) * sig_0;

        let level_bits = levels
            .iter()
            .filter_map(|&level| {
                let abs = level.unsigned_abs() as u32;
                (abs != 0).then(|| {
                    let mag = match abs {
                        1 => gt1_0,
                        2 => gt1_1 + gt2_0,
                        _ => gt1_1 + gt2_1 + rdoq::rice_bins(abs - 3, 0) as u64 * scale,
                    };
                    mag + scale
                })
            })
            .sum::<u64>();

        sig_bits + level_bits
    }

    pub(super) fn estimate_block(
        &mut self,
        ctxs: &Contexts,
        plan: BlockPlan,
        stage: WorkStage,
        budget: BlockSearchBudget,
    ) -> BlockTrial {
        debug_assert!(stage != WorkStage::FinalCode);
        debug_assert!(
            plan.quality == self.effective_trial_quality(plan.c_idx, budget),
            "trial block quality must come from the resolved block budget \
             (or the active luma close-call override)"
        );
        let coded = self.code_block_internal(ctxs, &plan, stage);
        let distortion = self.distortion_block(plan.c_idx, plan.x, plan.y, plan.log2_size);
        let cost = RdCost {
            distortion,
            frac_bits: coded.frac_bits,
            cost: self.rd_cost(distortion, coded.frac_bits),
        };
        BlockTrial {
            estimate: BlockEstimate {
                plan,
                cost,
                approx_frac_bits: coded.frac_bits,
                confidence: DecisionConfidence::Clear,
            },
            coded,
        }
    }

    pub(super) fn final_code_block(&mut self, ctxs: &Contexts, plan: &BlockPlan) -> CodedBlock {
        debug_assert_eq!(plan.quality, crate::effort::TrialQuality::Final);
        self.code_block_internal(ctxs, plan, WorkStage::FinalCode)
    }

    #[inline]
    pub(super) fn effective_trial_quality(&self, c_idx: u8, budget: BlockSearchBudget) -> TrialQuality {
        match c_idx {
            0 => self.luma_trial_quality(budget),
            _ => budget.chroma_trial_quality,
        }
    }

    #[inline]
    pub(super) fn luma_trial_quality(&self, budget: BlockSearchBudget) -> TrialQuality {
        if let Some(quality) = self.luma_trial_quality_override {
            quality
        } else if self.best2_luma_fastrd {
            TrialQuality::FastRd
        } else {
            budget.luma_trial_quality
        }
    }

    pub(super) fn trial_code_block(
        &mut self,
        ctxs: &Contexts,
        x: u32,
        y: u32,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        qp: i32,
        stage: WorkStage,
    ) -> BlockTrial {
        debug_assert!(stage != WorkStage::FinalCode);
        let budget = self.block_budget(x, y, log2_size, c_idx);
        let plan = BlockPlan {
            x,
            y,
            log2_size,
            c_idx,
            mode,
            qp,
            quality: self.effective_trial_quality(c_idx, budget),
        };
        self.estimate_block(ctxs, plan, stage, budget)
    }

    pub(super) fn distortion_block(&self, c_idx: u8, x: u32, y: u32, log2_size: u8) -> u64 {
        let size = 1usize << log2_size;
        let (plane, stride) = self.frame.plane(c_idx);
        let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);

        if x + size as u32 <= sw && y + size as u32 <= sh {
            let recon = &plane[y as usize * stride + x as usize..];
            let src = &src_plane[y as usize * src_stride + x as usize..];
            return primitives::ssd_u16(src, src_stride, recon, stride, size);
        }

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

    pub(super) fn distortion_tt_region(&self, x0: u32, y0: u32, log2_size: u8) -> u64 {
        let mut sse = self.distortion_block(0, x0, y0, log2_size);
        if let Some((cx, cy, clog2, count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) {
            let size = 1u32 << clog2;
            for i in 0..count {
                let ty = cy + i as u32 * size;
                sse += self.distortion_block(1, cx, ty, clog2);
                sse += self.distortion_block(2, cx, ty, clog2);
            }
        }
        sse
    }

    pub(super) fn distortion_cu_node(&self, node: &CuNode, x0: u32, y0: u32, log2_cb_size: u8) -> u64 {
        match node {
            CuNode::Leaf(_) => self.distortion_tt_region(x0, y0, log2_cb_size),
            CuNode::Split { kids } => {
                let half = (1u32 << log2_cb_size) / 2;
                let x1 = x0 + half;
                let y1 = y0 + half;
                let mut kids = kids.iter();
                let mut distortion = self.distortion_cu_node(
                    kids.next().expect("split CU has first child"),
                    x0,
                    y0,
                    log2_cb_size - 1,
                );
                if x1 < self.display_width {
                    distortion += self.distortion_cu_node(
                        kids.next().expect("split CU has right child"),
                        x1,
                        y0,
                        log2_cb_size - 1,
                    );
                }
                if y1 < self.display_height {
                    distortion += self.distortion_cu_node(
                        kids.next().expect("split CU has bottom child"),
                        x0,
                        y1,
                        log2_cb_size - 1,
                    );
                }
                if x1 < self.display_width && y1 < self.display_height {
                    distortion += self.distortion_cu_node(
                        kids.next().expect("split CU has bottom-right child"),
                        x1,
                        y1,
                        log2_cb_size - 1,
                    );
                }
                distortion
            }
        }
    }

    pub(super) fn rd_cost(&self, distortion: u64, frac_bits: u64) -> f64 {
        distortion as f64 + self.lambda() * (frac_bits as f64 / CabacEstimator::SCALE as f64)
    }

    pub(super) fn lambda(&self) -> f64 {
        0.57f64 * 2f64.powf((self.cur_qp_y as f64 - 12.0) / 3.0)
    }

    /// SAD/SATD-domain lambda for rough mode costing, ported from x265's
    /// `m_lambda` (`calcRdSADCost`) vs the SSE-domain `m_lambda2` (`calcRdCost`).
    /// HM/x265 weight rough SA8D costs with `sqrt(lambda_sse)` because SATD
    /// distortion grows ~linearly with error while SSE grows quadratically, so
    /// the bit term must use a different lambda to stay on the same RD curve.
    /// `lambda()` here is the HM SSE-domain formula, so the matching SAD-domain
    /// lambda is its square root.
    pub(super) fn lambda_sad(&self) -> f64 {
        self.lambda().sqrt()
    }

    /// Rough (SATD-domain) RD cost: `satd + lambda_sad * bits`, matching x265
    /// `calcRdSADCost(sad, bits)`. `frac_bits` is in `CabacEstimator::SCALE`
    /// fixed-point, like `rd_cost`.
    pub(super) fn rough_cost(&self, satd: u64, frac_bits: u64) -> f64 {
        satd as f64 + self.lambda_sad() * (frac_bits as f64 / CabacEstimator::SCALE as f64)
    }

    pub(super) fn residual_frac_bits(
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
        let t = self.prof.on.then(std::time::Instant::now);
        let mut ctxs = ctxs.clone();
        let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
        let r = estimate_residual_bits(&mut ctxs, levels, log2_size, c_idx, scan, false);
        if let Some(t) = t {
            self.prof.residual_bits += t.elapsed();
        }
        r
    }

    pub(super) fn refine_levels_rdoq_limited(
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
        let lambda = self.lambda();
        let scale = CabacEstimator::SCALE as f64;
        let dq = transform::DequantParams::new(log2_size, qp, self.bit_depth);
        let coeff_dist = |idx: usize, level: i16| -> u64 {
            let d = coeffs[idx] as i64 - dq.apply(level) as i64;
            (d * d) as u64
        };
        let mut cur_dist: u64 = (0..levels.len()).map(|i| coeff_dist(i, levels[i])).sum();

        let scan_order = get_scan_order(log2_size, mode, c_idx, self.cat);
        self.stats.cache_builds += 1;
        let mut cache =
            ResidualEstimateCache::build(ctxs, &levels, log2_size, c_idx, scan_order, false);

        for _ in 0..max_passes {
            let mut changed = false;
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
                    let est = cache.estimate_one_change(&levels, idx, candidate);
                    let bits = match est {
                        Some(b) => {
                            self.stats.cache_fast_hits += 1;
                            b
                        }
                        None => {
                            self.stats.cache_fallbacks += 1;
                            levels[idx] = candidate;
                            let b = self.residual_frac_bits(ctxs, &levels, log2_size, c_idx, mode);
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

    pub(super) fn can_split_tt(&self, log2_size: u8, trafo_depth: u8) -> bool {
        if !(log2_size <= MAX_TB_LOG2
            && log2_size > MIN_TB_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH)
        {
            return false;
        }
        !((self.cat == 1 || self.cat == 2) && log2_size == 3)
    }

    pub(super) fn build_parent_chroma_tu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        stage: WorkStage,
    ) -> Option<ParentChromaTu> {
        if self.cat != 1 || log2_size != 3 {
            return None;
        }
        let (cx, cy, clog2, _count) = chroma_tb_geom(self.cat, x0, y0, log2_size)?;
        let cb_plan = BlockPlan {
            x: cx,
            y: cy,
            log2_size: clog2,
            c_idx: 1,
            mode: chroma_mode,
            qp: self.cur_qp_c,
            quality: crate::effort::TrialQuality::Final,
        };
        let cr_plan = BlockPlan {
            c_idx: 2,
            ..cb_plan.clone()
        };
        let (cb, cr) = if stage == WorkStage::FinalCode {
            (
                self.final_code_block(ctxs, &cb_plan),
                self.final_code_block(ctxs, &cr_plan),
            )
        } else {
            let cb_budget =
                self.block_budget(cb_plan.x, cb_plan.y, cb_plan.log2_size, cb_plan.c_idx);
            let cr_budget =
                self.block_budget(cr_plan.x, cr_plan.y, cr_plan.log2_size, cr_plan.c_idx);
            (
                self.estimate_block(ctxs, cb_plan, stage, cb_budget).coded,
                self.estimate_block(ctxs, cr_plan, stage, cr_budget).coded,
            )
        };
        Some(ParentChromaTu {
            log2_size: clog2,
            chroma_mode,
            cb,
            cr,
        })
    }

    pub(super) fn build_tt_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        chroma_cache: Option<&ChromaLeafCache>,
    ) -> Tt {
        let luma = self
            .trial_code_block(
                ctxs,
                x0,
                y0,
                log2_size,
                0,
                luma_mode,
                self.cur_qp_y,
                WorkStage::TuDecision,
            )
            .coded;

        let (chroma_log2, cb, cr, cb1, cr1) = if let Some(cache) =
            chroma_cache.filter(|c| c.x0 == x0 && c.y0 == y0 && c.log2_size == log2_size)
        {
            self.restore_frame_region(&cache.frame);
            (
                cache.chroma_log2,
                cache.cb.clone(),
                cache.cr.clone(),
                cache.cb1.clone(),
                cache.cr1.clone(),
            )
        } else {
            let pred = chroma_pred_mode(self.cat, chroma_mode);
            match chroma_tb_geom(self.cat, x0, y0, log2_size) {
                Some((cx, cy, clog2, count)) => {
                    let size = 1u32 << clog2;
                    let cb = self
                        .trial_code_block(
                            ctxs,
                            cx,
                            cy,
                            clog2,
                            1,
                            pred,
                            self.cur_qp_c,
                            WorkStage::TuDecision,
                        )
                        .coded;
                    let cr = self
                        .trial_code_block(
                            ctxs,
                            cx,
                            cy,
                            clog2,
                            2,
                            pred,
                            self.cur_qp_c,
                            WorkStage::TuDecision,
                        )
                        .coded;
                    let (cb1, cr1) = if count > 1 {
                        let ty = cy + size;
                        (
                            self.trial_code_block(
                                ctxs,
                                cx,
                                ty,
                                clog2,
                                1,
                                pred,
                                self.cur_qp_c,
                                WorkStage::TuDecision,
                            )
                            .coded,
                            self.trial_code_block(
                                ctxs,
                                cx,
                                ty,
                                clog2,
                                2,
                                pred,
                                self.cur_qp_c,
                                WorkStage::TuDecision,
                            )
                            .coded,
                        )
                    } else {
                        (CodedBlock::empty(), CodedBlock::empty())
                    };
                    (clog2, cb, cr, cb1, cr1)
                }
                None => (
                    0,
                    CodedBlock::empty(),
                    CodedBlock::empty(),
                    CodedBlock::empty(),
                    CodedBlock::empty(),
                ),
            }
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
            cb1,
            cr1,
        })
    }

    pub(super) fn build_tt_split(
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
                None,
            ),
            self.build_tt(
                x0 + half,
                y0,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
                None,
            ),
            self.build_tt(
                x0,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
                None,
            ),
            self.build_tt(
                x0 + half,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
                None,
            ),
        ];
        let parent_chroma = self.build_parent_chroma_tu(
            x0,
            y0,
            log2_size,
            chroma_mode,
            ctxs,
            WorkStage::TuDecision,
        );
        let cbf_cb = parent_chroma
            .as_ref()
            .map(|c| c.cb.cbf)
            .unwrap_or_else(|| kids.iter().any(|k| k.cbf_cb() || k.cbf_cb1()));
        let cbf_cr = parent_chroma
            .as_ref()
            .map(|c| c.cr.cbf)
            .unwrap_or_else(|| kids.iter().any(|k| k.cbf_cr() || k.cbf_cr1()));
        let cbf_cb1 = kids.iter().any(|k| k.cbf_cb1());
        let cbf_cr1 = kids.iter().any(|k| k.cbf_cr1());
        Tt::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cb1,
            cbf_cr1,
            cbf_cr,
            parent_chroma,
            kids,
        }
    }

    pub(super) fn build_luma_tt_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let luma = self
            .trial_code_block(
                ctxs,
                x0,
                y0,
                log2_size,
                0,
                luma_mode,
                self.cur_qp_y,
                WorkStage::LumaTrial,
            )
            .coded;

        Tt::Leaf(LeafTu {
            log2_size,
            chroma_log2: 0,
            trafo_depth,
            luma_mode,
            chroma_mode: luma_mode,
            luma,
            cb: CodedBlock::empty(),
            cr: CodedBlock::empty(),
            cb1: CodedBlock::empty(),
            cr1: CodedBlock::empty(),
        })
    }

    pub(super) fn build_luma_tt_split(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let half = 1u32 << (log2_size - 1);
        let kids = vec![
            self.build_luma_tt(x0, y0, log2_size - 1, trafo_depth + 1, luma_mode, ctxs),
            self.build_luma_tt(
                x0 + half,
                y0,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                ctxs,
            ),
            self.build_luma_tt(
                x0,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                ctxs,
            ),
            self.build_luma_tt(
                x0 + half,
                y0 + half,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                ctxs,
            ),
        ];
        Tt::Split {
            log2_size,
            trafo_depth,
            cbf_cb: false,
            cbf_cr: false,
            cbf_cb1: false,
            cbf_cr1: false,
            parent_chroma: None,
            kids,
        }
    }

    pub(super) fn build_luma_tt_leaf_screen(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
            let half = 1u32 << (log2_size - 1);
            let kids = vec![
                self.build_luma_tt_leaf_screen(
                    x0,
                    y0,
                    log2_size - 1,
                    trafo_depth + 1,
                    luma_mode,
                    ctxs,
                ),
                self.build_luma_tt_leaf_screen(
                    x0 + half,
                    y0,
                    log2_size - 1,
                    trafo_depth + 1,
                    luma_mode,
                    ctxs,
                ),
                self.build_luma_tt_leaf_screen(
                    x0,
                    y0 + half,
                    log2_size - 1,
                    trafo_depth + 1,
                    luma_mode,
                    ctxs,
                ),
                self.build_luma_tt_leaf_screen(
                    x0 + half,
                    y0 + half,
                    log2_size - 1,
                    trafo_depth + 1,
                    luma_mode,
                    ctxs,
                ),
            ];
            return Tt::Split {
                log2_size,
                trafo_depth,
                cbf_cb: false,
                cbf_cr: false,
                cbf_cb1: false,
                cbf_cr1: false,
                parent_chroma: None,
                kids,
            };
        }

        self.record_tu_winner(x0, y0, log2_size, false);
        self.build_luma_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, ctxs)
    }

    pub(super) fn build_luma_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.build_luma_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        }

        if !self.can_split_tt(log2_size, trafo_depth) {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.build_luma_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        }

        let budget = self.block_budget(x0, y0, log2_size, 0);
        if budget.tu_split == SplitSearch::ForceLeaf {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.build_luma_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        }
        if budget.tu_split == SplitSearch::ForceSplit {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.build_luma_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        }

        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);

        if budget.tu_split == SplitSearch::PreferSplit {
            self.restore_frame_region(&base_frame);
            let split = self.build_luma_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
            let split_distortion = self.distortion_block(0, x0, y0, log2_size);
            let split_bits = estimate_tt_bits(ctxs, &split, 0, false, true, true);
            let split_cost = self.rd_cost(split_distortion, split_bits);
            let split_frame = self.snapshot_frame_region(x0, y0, log2_size);

            self.restore_frame_region(&base_frame);
            let leaf = self.build_luma_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
            let leaf_distortion = self.distortion_block(0, x0, y0, log2_size);
            let leaf_bits = estimate_tt_bits(ctxs, &leaf, 0, false, true, true);
            let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
            self.record_close_call(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );

            if split_cost < leaf_cost {
                self.restore_frame_region(&split_frame);
                self.record_tu_winner(x0, y0, log2_size, true);
                return split;
            }

            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }

        self.restore_frame_region(&base_frame);
        let leaf = self.build_luma_tt_leaf(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        if self.tu_split_early_terminate(&leaf, budget) {
            self.stats.tu_split_early_terminations += 1;
            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }
        let leaf_distortion = self.distortion_block(0, x0, y0, log2_size);
        let leaf_bits = estimate_tt_bits(ctxs, &leaf, 0, false, true, true);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        if self.should_limit_tu_to_neighbor_leaf(x0, y0, log2_size, &leaf) {
            self.record_tu_neighbor_leaf_skip(x0, y0, log2_size);
            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);

        self.restore_frame_region(&base_frame);
        let split = self.build_luma_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        let split_distortion = self.distortion_block(0, x0, y0, log2_size);
        let split_bits = estimate_tt_bits(ctxs, &split, 0, false, true, true);
        let split_cost = self.rd_cost(split_distortion, split_bits);
        self.record_close_call(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );

        if split_cost < leaf_cost {
            self.record_tu_winner(x0, y0, log2_size, true);
            split
        } else {
            self.restore_frame_region(&leaf_frame);
            self.record_tu_winner(x0, y0, log2_size, false);
            leaf
        }
    }

    pub(super) fn tu_split_early_terminate(&self, leaf: &Tt, budget: BlockSearchBudget) -> bool {
        let Tt::Leaf(leaf) = leaf else {
            return false;
        };
        budget.tu_split_early_terminate(leaf.luma.cbf)
    }

    pub(super) fn tt_to_plan(tt: &Tt) -> TtPlan {
        match tt {
            Tt::Leaf(leaf) => TtPlan::Leaf {
                log2_size: leaf.log2_size,
                trafo_depth: leaf.trafo_depth,
            },
            Tt::Split {
                log2_size,
                trafo_depth,
                parent_chroma,
                kids,
                ..
            } => TtPlan::Split {
                log2_size: *log2_size,
                trafo_depth: *trafo_depth,
                kids: kids.iter().map(Self::tt_to_plan).collect(),
                parent_chroma: parent_chroma.as_ref().map(|pc| ParentChromaPlan {
                    log2_size: pc.log2_size,
                }),
            },
        }
    }

    pub(super) fn final_code_tt_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let luma = self.final_code_block(
            ctxs,
            &BlockPlan {
                x: x0,
                y: y0,
                log2_size,
                c_idx: 0,
                mode: luma_mode,
                qp: self.cur_qp_y,
                quality: crate::effort::TrialQuality::Final,
            },
        );

        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let (chroma_log2, cb, cr, cb1, cr1) = match chroma_tb_geom(self.cat, x0, y0, log2_size) {
            Some((cx, cy, clog2, count)) => {
                let size = 1u32 << clog2;
                let cb = self.final_code_block(
                    ctxs,
                    &BlockPlan {
                        x: cx,
                        y: cy,
                        log2_size: clog2,
                        c_idx: 1,
                        mode: pred,
                        qp: self.cur_qp_c,
                        quality: crate::effort::TrialQuality::Final,
                    },
                );
                let cr = self.final_code_block(
                    ctxs,
                    &BlockPlan {
                        x: cx,
                        y: cy,
                        log2_size: clog2,
                        c_idx: 2,
                        mode: pred,
                        qp: self.cur_qp_c,
                        quality: crate::effort::TrialQuality::Final,
                    },
                );
                let (cb1, cr1) = if count > 1 {
                    let ty = cy + size;
                    (
                        self.final_code_block(
                            ctxs,
                            &BlockPlan {
                                x: cx,
                                y: ty,
                                log2_size: clog2,
                                c_idx: 1,
                                mode: pred,
                                qp: self.cur_qp_c,
                                quality: crate::effort::TrialQuality::Final,
                            },
                        ),
                        self.final_code_block(
                            ctxs,
                            &BlockPlan {
                                x: cx,
                                y: ty,
                                log2_size: clog2,
                                c_idx: 2,
                                mode: pred,
                                qp: self.cur_qp_c,
                                quality: crate::effort::TrialQuality::Final,
                            },
                        ),
                    )
                } else {
                    (CodedBlock::empty(), CodedBlock::empty())
                };
                (clog2, cb, cr, cb1, cr1)
            }
            None => (
                0,
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
            ),
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
            cb1,
            cr1,
        })
    }

    pub(super) fn final_code_tt(
        &mut self,
        plan: &TtPlan,
        x0: u32,
        y0: u32,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        match plan {
            TtPlan::Leaf {
                log2_size,
                trafo_depth,
            } => self.final_code_tt_leaf(
                x0,
                y0,
                *log2_size,
                *trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            ),
            TtPlan::Split {
                log2_size,
                trafo_depth,
                kids,
                parent_chroma,
            } => {
                let half = 1u32 << (*log2_size - 1);
                let child_pos = [
                    (x0, y0),
                    (x0 + half, y0),
                    (x0, y0 + half),
                    (x0 + half, y0 + half),
                ];
                let coded_kids: Vec<Tt> = kids
                    .iter()
                    .zip(child_pos)
                    .map(|(kid, (kx, ky))| {
                        self.final_code_tt(kid, kx, ky, luma_mode, chroma_mode, ctxs)
                    })
                    .collect();
                let coded_parent_chroma = parent_chroma.as_ref().and_then(|_| {
                    self.build_parent_chroma_tu(
                        x0,
                        y0,
                        *log2_size,
                        chroma_mode,
                        ctxs,
                        WorkStage::FinalCode,
                    )
                });
                let cbf_cb = coded_parent_chroma
                    .as_ref()
                    .map(|c| c.cb.cbf)
                    .unwrap_or_else(|| coded_kids.iter().any(|k| k.cbf_cb() || k.cbf_cb1()));
                let cbf_cr = coded_parent_chroma
                    .as_ref()
                    .map(|c| c.cr.cbf)
                    .unwrap_or_else(|| coded_kids.iter().any(|k| k.cbf_cr() || k.cbf_cr1()));
                let cbf_cb1 = coded_kids.iter().any(|k| k.cbf_cb1());
                let cbf_cr1 = coded_kids.iter().any(|k| k.cbf_cr1());
                Tt::Split {
                    log2_size: *log2_size,
                    trafo_depth: *trafo_depth,
                    cbf_cb,
                    cbf_cr,
                    cbf_cb1,
                    cbf_cr1,
                    parent_chroma: coded_parent_chroma,
                    kids: coded_kids,
                }
            }
        }
    }

    pub(super) fn decide_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        chroma_cache: Option<&ChromaLeafCache>,
    ) -> (TrialResult<TtPlan>, Tt) {
        let finish = |this: &mut Self, winner: Tt, distortion: u64, frac_bits: u64, confidence| {
            let plan = Self::tt_to_plan(&winner);
            let cost = RdCost {
                distortion,
                frac_bits,
                cost: this.rd_cost(distortion, frac_bits),
            };
            (
                TrialResult {
                    plan,
                    cost,
                    confidence,
                },
                winner,
            )
        };

        if log2_size > MAX_TB_LOG2 {
            let split =
                self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
            let distortion = self.distortion_tt_region(x0, y0, log2_size);
            let frac_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
            self.record_tu_winner(x0, y0, log2_size, true);
            return finish(
                self,
                split,
                distortion,
                frac_bits,
                DecisionConfidence::Clear,
            );
        }

        if !self.can_split_tt(log2_size, trafo_depth) {
            let leaf = self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
            let distortion = self.distortion_tt_region(x0, y0, log2_size);
            let frac_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
            self.record_tu_winner(x0, y0, log2_size, false);
            return finish(self, leaf, distortion, frac_bits, DecisionConfidence::Clear);
        }

        let budget = self.block_budget(x0, y0, log2_size, 0);
        if budget.tu_split == SplitSearch::ForceLeaf {
            let leaf = self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
            let distortion = self.distortion_tt_region(x0, y0, log2_size);
            let frac_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
            self.record_tu_winner(x0, y0, log2_size, false);
            return finish(self, leaf, distortion, frac_bits, DecisionConfidence::Clear);
        }
        if budget.tu_split == SplitSearch::ForceSplit {
            let split =
                self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
            let distortion = self.distortion_tt_region(x0, y0, log2_size);
            let frac_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
            self.record_tu_winner(x0, y0, log2_size, true);
            return finish(
                self,
                split,
                distortion,
                frac_bits,
                DecisionConfidence::Clear,
            );
        }

        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);

        if budget.tu_split == SplitSearch::PreferSplit {
            self.restore_frame_region(&base_frame);
            self.restore_tu_depth_region(&base_tu_depth);
            let split_e0 = self.stats.residual_bit_estimates;
            let split =
                self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
            let split_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);
            let split_exact = (self.stats.residual_bit_estimates - split_e0) as u32;
            let split_distortion = self.distortion_tt_region(x0, y0, log2_size);
            let split_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
            let split_cost = self.rd_cost(split_distortion, split_bits);

            self.restore_frame_region(&base_frame);
            self.restore_tu_depth_region(&base_tu_depth);
            let leaf = self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
            let leaf_distortion = self.distortion_tt_region(x0, y0, log2_size);
            let leaf_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
            let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
            self.record_close_call(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );
            let confidence = Self::decision_confidence(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );
            self.trace_split_decision(
                crate::trace::DecisionKind::Tu,
                x0,
                y0,
                log2_size,
                leaf_cost,
                split_cost,
                0,
                split_exact,
            );

            if split_cost < leaf_cost {
                self.restore_tu_depth_region(&split_tu_depth);
                self.record_tu_winner(x0, y0, log2_size, true);
                return finish(self, split, split_distortion, split_bits, confidence);
            }

            self.record_tu_winner(x0, y0, log2_size, false);
            return finish(self, leaf, leaf_distortion, leaf_bits, confidence);
        }

        self.restore_frame_region(&base_frame);
        self.restore_tu_depth_region(&base_tu_depth);
        let leaf = self.build_tt_leaf(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            chroma_cache,
        );
        let leaf_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let leaf_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        if self.tu_split_early_terminate(&leaf, budget) {
            self.stats.tu_split_early_terminations += 1;
            self.record_tu_winner(x0, y0, log2_size, false);
            return finish(
                self,
                leaf,
                leaf_distortion,
                leaf_bits,
                DecisionConfidence::Clear,
            );
        }
        if self.should_limit_tu_to_neighbor_leaf(x0, y0, log2_size, &leaf) {
            self.record_tu_neighbor_leaf_skip(x0, y0, log2_size);
            self.record_tu_winner(x0, y0, log2_size, false);
            return finish(
                self,
                leaf,
                leaf_distortion,
                leaf_bits,
                DecisionConfidence::Clear,
            );
        }
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let leaf_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);

        self.restore_frame_region(&base_frame);
        self.restore_tu_depth_region(&base_tu_depth);
        let split_e0 = self.stats.residual_bit_estimates;
        let split =
            self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
        let split_exact = (self.stats.residual_bit_estimates - split_e0) as u32;
        let split_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let split_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
        let split_cost = self.rd_cost(split_distortion, split_bits);
        self.record_close_call(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );
        let confidence = Self::decision_confidence(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );
        self.trace_split_decision(
            crate::trace::DecisionKind::Tu,
            x0,
            y0,
            log2_size,
            leaf_cost,
            split_cost,
            0,
            split_exact,
        );

        if split_cost < leaf_cost {
            self.record_tu_winner(x0, y0, log2_size, true);
            finish(self, split, split_distortion, split_bits, confidence)
        } else {
            self.restore_frame_region(&leaf_frame);
            self.restore_tu_depth_region(&leaf_tu_depth);
            self.record_tu_winner(x0, y0, log2_size, false);
            finish(self, leaf, leaf_distortion, leaf_bits, confidence)
        }
    }

    pub(super) fn decide_and_final_code_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        chroma_cache: Option<&ChromaLeafCache>,
    ) -> Tt {
        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);
        let (trial, winner) = self.decide_tt(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            chroma_cache,
        );
        if self.effort_template.reference || self.best2_tt_reuse {
            return winner;
        }
        self.restore_frame_region(&base_frame);
        self.restore_tu_depth_region(&base_tu_depth);
        self.final_code_tt(&trial.plan, x0, y0, luma_mode, chroma_mode, ctxs)
    }

    pub(super) fn build_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        chroma_cache: Option<&ChromaLeafCache>,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
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
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
        }

        let budget = self.block_budget(x0, y0, log2_size, 0);
        if budget.tu_split == SplitSearch::ForceLeaf {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
        }
        if budget.tu_split == SplitSearch::ForceSplit {
            self.record_tu_winner(x0, y0, log2_size, true);
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

        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);

        if budget.tu_split == SplitSearch::PreferSplit {
            self.restore_frame_region(&base_frame);
            self.restore_tu_depth_region(&base_tu_depth);
            let split =
                self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
            let split_distortion = self.distortion_tt_region(x0, y0, log2_size);
            let split_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
            let split_cost = self.rd_cost(split_distortion, split_bits);
            let split_frame = self.snapshot_frame_region(x0, y0, log2_size);
            let split_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);

            self.restore_frame_region(&base_frame);
            self.restore_tu_depth_region(&base_tu_depth);
            let leaf = self.build_tt_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                chroma_cache,
            );
            let leaf_distortion = self.distortion_tt_region(x0, y0, log2_size);
            let leaf_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
            let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
            self.record_close_call(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );

            if split_cost < leaf_cost {
                self.restore_frame_region(&split_frame);
                self.restore_tu_depth_region(&split_tu_depth);
                self.record_tu_winner(x0, y0, log2_size, true);
                return split;
            }

            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }

        self.restore_frame_region(&base_frame);
        self.restore_tu_depth_region(&base_tu_depth);
        let leaf = self.build_tt_leaf(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            chroma_cache,
        );
        if self.tu_split_early_terminate(&leaf, budget) {
            self.stats.tu_split_early_terminations += 1;
            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }
        let leaf_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let leaf_bits = estimate_tt_bits(ctxs, &leaf, self.cat, false, true, true);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        if self.should_limit_tu_to_neighbor_leaf(x0, y0, log2_size, &leaf) {
            self.record_tu_neighbor_leaf_skip(x0, y0, log2_size);
            self.record_tu_winner(x0, y0, log2_size, false);
            return leaf;
        }
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let leaf_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_size);

        self.restore_frame_region(&base_frame);
        self.restore_tu_depth_region(&base_tu_depth);
        let split =
            self.build_tt_split(x0, y0, log2_size, trafo_depth, luma_mode, chroma_mode, ctxs);
        let split_distortion = self.distortion_tt_region(x0, y0, log2_size);
        let split_bits = estimate_tt_bits(ctxs, &split, self.cat, false, true, true);
        let split_cost = self.rd_cost(split_distortion, split_bits);
        self.record_close_call(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );

        if split_cost < leaf_cost {
            self.record_tu_winner(x0, y0, log2_size, true);
            split
        } else {
            self.restore_frame_region(&leaf_frame);
            self.restore_tu_depth_region(&leaf_tu_depth);
            self.record_tu_winner(x0, y0, log2_size, false);
            leaf
        }
    }

    pub(super) fn build_cu_leaf(
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
            let mut best_luma_mode = candidates[0];
            let mut runner_up_mode = candidates[0];
            let mut best_rank = 1usize;
            let mut best_cost = f64::MAX;
            let mut runner_up_cost = f64::MAX;
            let mut trace_cands: Vec<crate::trace::CandRec> = Vec::new();

            for (rank0, &luma_mode) in candidates.iter().enumerate() {
                self.stats.cu_trials += 1;
                let cand_e0 = self.stats.residual_bit_estimates;
                self.restore_frame_region(&base_frame);
                self.restore_mode_region(&base_mode_map);
                self.restore_tu_depth_region(&base_tu_depth);

                self.store_mode(x0, y0, log2_cb_size, luma_mode);
                let tt = if self.best_luma_leaf_screen {
                    self.build_luma_tt_leaf_screen(x0, y0, log2_cb_size, 0, luma_mode, ctxs)
                } else {
                    self.build_luma_tt(x0, y0, log2_cb_size, 0, luma_mode, ctxs)
                };

                let distortion = self.distortion_block(0, x0, y0, log2_cb_size);
                let bits = estimate_luma_cu_trial_bits(ctxs, mpm, luma_mode, &tt, log2_cb_size);
                let cost = self.rd_cost(distortion, bits);
                if self.trace.enabled {
                    let cand_exact = (self.stats.residual_bit_estimates - cand_e0) as u32;
                    let flip_bits =
                        (bits as i64 + self.tt_luma_approx_delta(ctxs, &tt)).max(0) as u64;
                    let flip_cost = self.rd_cost(distortion, flip_bits);
                    trace_cands.push(crate::trace::CandRec::with_flip(
                        cost, flip_cost, cand_exact,
                    ));
                }

                if cost < best_cost {
                    runner_up_cost = best_cost;
                    runner_up_mode = best_luma_mode;
                    best_cost = cost;
                    best_luma_mode = luma_mode;
                    best_rank = rank0 + 1;
                } else if cost < runner_up_cost {
                    runner_up_cost = cost;
                    runner_up_mode = luma_mode;
                }
            }
            self.record_luma_winner_rank(best_rank);
            self.record_close_call(best_cost, runner_up_cost, tt_budget.close_call_margin);
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

            if self.luma_trial_quality(tt_budget) == TrialQuality::FastRd
                && best_luma_mode != runner_up_mode
                && Self::is_close_call(best_cost, runner_up_cost, tt_budget.close_call_margin)
            {
                best_luma_mode = self.escalate_luma_close_call(
                    x0,
                    y0,
                    log2_cb_size,
                    [best_luma_mode, runner_up_mode],
                    &base_frame,
                    &base_mode_map,
                    &base_tu_depth,
                    mpm,
                    ctxs,
                );
            }

            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);
            self.restore_tu_depth_region(&base_tu_depth);
            let chroma = self.decide_chroma_mode(x0, y0, tt_log2, best_luma_mode, ctxs);
            self.store_mode(x0, y0, log2_cb_size, best_luma_mode);
            let tt = self.decide_and_final_code_tt(
                x0,
                y0,
                log2_cb_size,
                0,
                best_luma_mode,
                chroma.plan.mode,
                ctxs,
                chroma.leaf_cache.as_ref(),
            );
            return CuLeaf {
                mpm,
                luma_mode: best_luma_mode,
                chroma_mode_idx: chroma.plan.mode_idx,
                confidence: DecisionConfidence::Clear,
                tt,
            };
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

            let tt = self.decide_and_final_code_tt(
                x0,
                y0,
                log2_cb_size,
                0,
                luma_mode,
                chroma.plan.mode,
                ctxs,
                chroma.leaf_cache.as_ref(),
            );
            let leaf = CuLeaf {
                mpm,
                luma_mode,
                chroma_mode_idx: chroma.plan.mode_idx,
                confidence: DecisionConfidence::Clear,
                tt,
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

    pub(super) fn escalate_luma_close_call(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        modes: [u8; 2],
        base_frame: &FrameSnapshot,
        base_mode_map: &MapSnapshot,
        base_tu_depth: &MapSnapshot,
        mpm: [IntraPredMode; 3],
        ctxs: &Contexts,
    ) -> u8 {
        self.stats.luma_close_call_escalations += 1;
        self.luma_trial_quality_override = Some(TrialQuality::FullRd);
        let mut best_mode = modes[0];
        let mut best_cost = f64::MAX;
        let mut runner_up_cost = f64::MAX;
        let mut best_rank = 1usize;
        let mut trace_cands: Vec<crate::trace::CandRec> = Vec::new();
        for (rank0, &luma_mode) in modes.iter().enumerate() {
            let cand_e0 = self.stats.residual_bit_estimates;
            self.restore_frame_region(base_frame);
            self.restore_mode_region(base_mode_map);
            self.restore_tu_depth_region(base_tu_depth);
            self.store_mode(x0, y0, log2_cb_size, luma_mode);
            let tt = self.build_luma_tt(x0, y0, log2_cb_size, 0, luma_mode, ctxs);
            let distortion = self.distortion_block(0, x0, y0, log2_cb_size);
            let bits = estimate_luma_cu_trial_bits(ctxs, mpm, luma_mode, &tt, log2_cb_size);
            let cost = self.rd_cost(distortion, bits);
            if self.trace.enabled {
                let cand_exact = (self.stats.residual_bit_estimates - cand_e0) as u32;
                let flip_bits = (bits as i64 + self.tt_luma_approx_delta(ctxs, &tt)).max(0) as u64;
                let flip_cost = self.rd_cost(distortion, flip_bits);
                trace_cands.push(crate::trace::CandRec::with_flip(
                    cost, flip_cost, cand_exact,
                ));
            }
            if cost < best_cost {
                runner_up_cost = best_cost;
                best_cost = cost;
                best_mode = luma_mode;
                best_rank = rank0 + 1;
            } else if cost < runner_up_cost {
                runner_up_cost = cost;
            }
        }
        self.luma_trial_quality_override = None;
        self.restore_frame_region(base_frame);
        self.restore_mode_region(base_mode_map);
        self.restore_tu_depth_region(base_tu_depth);
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
        let _ = best_rank;
        best_mode
    }

    pub(super) fn build_cu_kids(
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

    pub(super) fn cu_early_terminate(
        &self,
        budget: BlockSearchBudget,
        leaf: &CuLeaf,
        leaf_bits: u64,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
    ) -> bool {
        let qp = self.search_qp();
        let size = 1usize << log2_cb_size;
        budget.should_early_terminate_cu(
            cu_leaf_has_residual(leaf),
            leaf_bits / CabacEstimator::SCALE,
            log2_cb_size,
            qp,
            self.bit_depth,
            || self.source_luma_range(x0, y0, size),
        )
    }

    pub(super) fn cu_to_plan(cu: &CuNode) -> CuPlan {
        match cu {
            CuNode::Leaf(leaf) => CuPlan::Leaf(CuLeafPlan {
                mpm: leaf.mpm,
                luma_mode: leaf.luma_mode,
                chroma_mode_idx: leaf.chroma_mode_idx,
                chroma_mode: Self::chroma_mode_from_idx(leaf.luma_mode, leaf.chroma_mode_idx),
                tt: Self::tt_to_plan(&leaf.tt),
            }),
            CuNode::Split { kids } => CuPlan::Split {
                kids: kids.iter().map(Self::cu_to_plan).collect(),
            },
        }
    }

    pub(super) fn final_code_cu(
        &mut self,
        plan: &CuPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        if !self.aq.active {
            return self.final_code_cu_inner(plan, x0, y0, log2_cb_size, ct_depth, ctxs);
        }

        let saved_policy = self.cur_policy;
        let saved_qp = (self.cur_qp_y, self.cur_qp_c);
        self.cur_policy = Some(self.region_policy(x0, y0, log2_cb_size));
        self.aq_set_cu_qp(x0, y0);
        let node = self.final_code_cu_inner(plan, x0, y0, log2_cb_size, ct_depth, ctxs);
        self.cur_policy = saved_policy;
        (self.cur_qp_y, self.cur_qp_c) = saved_qp;
        node
    }

    pub(super) fn final_code_cu_inner(
        &mut self,
        plan: &CuPlan,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        match plan {
            CuPlan::Leaf(leaf) => {
                self.set_ct_depth(x0, y0, log2_cb_size, ct_depth);
                self.store_mode(x0, y0, log2_cb_size, leaf.luma_mode);
                let tt =
                    self.final_code_tt(&leaf.tt, x0, y0, leaf.luma_mode, leaf.chroma_mode, ctxs);
                CuNode::Leaf(CuLeaf {
                    mpm: leaf.mpm,
                    luma_mode: leaf.luma_mode,
                    chroma_mode_idx: leaf.chroma_mode_idx,
                    confidence: DecisionConfidence::Clear,
                    tt,
                })
            }
            CuPlan::Split { kids } => {
                let half = (1u32 << log2_cb_size) / 2;
                let x1 = x0 + half;
                let y1 = y0 + half;
                let mut kid_iter = kids.iter();
                let mut coded = Vec::with_capacity(kids.len());
                coded.push(self.final_code_cu(
                    kid_iter.next().expect("split CU has first child"),
                    x0,
                    y0,
                    log2_cb_size - 1,
                    ct_depth + 1,
                    ctxs,
                ));
                if x1 < self.display_width {
                    coded.push(self.final_code_cu(
                        kid_iter.next().expect("split CU has right child"),
                        x1,
                        y0,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                if y1 < self.display_height {
                    coded.push(self.final_code_cu(
                        kid_iter.next().expect("split CU has bottom child"),
                        x0,
                        y1,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                if x1 < self.display_width && y1 < self.display_height {
                    coded.push(self.final_code_cu(
                        kid_iter.next().expect("split CU has bottom-right child"),
                        x1,
                        y1,
                        log2_cb_size - 1,
                        ct_depth + 1,
                        ctxs,
                    ));
                }
                CuNode::Split { kids: coded }
            }
        }
    }

    pub(super) fn decide_cu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> (TrialResult<CuPlan>, CuNode) {
        let cb_size = 1u32 << log2_cb_size;
        let fully_inside =
            x0 + cb_size <= self.display_width && y0 + cb_size <= self.display_height;
        let can_split = log2_cb_size > 3;

        if !can_split {
            let leaf = self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs);
            let node = Self::cu_leaf_node(leaf, DecisionConfidence::Clear);
            self.record_cu_winner(x0, y0, log2_cb_size, false);
            let r = self.cu_trial_result(&node, x0, y0, log2_cb_size, ct_depth, ctxs);
            return (r, node);
        }
        if !fully_inside {
            let node = CuNode::Split {
                kids: self.build_cu_kids(x0, y0, log2_cb_size, ct_depth, ctxs),
            };
            self.record_cu_winner(x0, y0, log2_cb_size, true);
            let r = self.cu_trial_result(&node, x0, y0, log2_cb_size, ct_depth, ctxs);
            return (r, node);
        }

        let ctx_inc = self.split_ctx_inc(x0, y0, ct_depth);
        let budget = self.block_budget(x0, y0, log2_cb_size, 0);
        if budget.cu_split == SplitSearch::ForceLeaf {
            self.stats.cu_force_leaf += 1;
            let leaf = self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs);
            let node = Self::cu_leaf_node(leaf, DecisionConfidence::Clear);
            self.record_cu_winner(x0, y0, log2_cb_size, false);
            let r = self.cu_trial_result(&node, x0, y0, log2_cb_size, ct_depth, ctxs);
            return (r, node);
        }
        if budget.cu_split == SplitSearch::ForceSplit {
            let node = CuNode::Split {
                kids: self.build_cu_kids(x0, y0, log2_cb_size, ct_depth, ctxs),
            };
            self.record_cu_winner(x0, y0, log2_cb_size, true);
            let r = self.cu_trial_result(&node, x0, y0, log2_cb_size, ct_depth, ctxs);
            return (r, node);
        }

        let base_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let base_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let base_ct_depth_map = self.snapshot_ct_depth_region(x0, y0, log2_cb_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);

        if budget.cu_split == SplitSearch::PreferSplit {
            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);
            self.restore_ct_depth_region(&base_ct_depth_map);
            self.restore_tu_depth_region(&base_tu_depth);
            let split_e0 = self.stats.residual_bit_estimates;
            let split = CuNode::Split {
                kids: self.build_cu_kids(x0, y0, log2_cb_size, ct_depth, ctxs),
            };
            let split_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);
            let split_exact = (self.stats.residual_bit_estimates - split_e0) as u32;
            let split_distortion = self.distortion_cu_node(&split, x0, y0, log2_cb_size);
            let split_bits =
                estimate_cu_node_bits(ctxs, &split, x0, y0, log2_cb_size, ct_depth, self);
            let split_cost = self.rd_cost(split_distortion, split_bits);

            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);
            self.restore_ct_depth_region(&base_ct_depth_map);
            self.restore_tu_depth_region(&base_tu_depth);
            let leaf = self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs);
            let leaf_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);
            let leaf_distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
            let leaf_bits = estimate_cu_leaf_bits(ctxs, &leaf, log2_cb_size, self.cat)
                + estimate_split_cu_flag_bits(ctxs, ctx_inc, false);
            let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
            self.record_close_call(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );
            let confidence = Self::decision_confidence(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );
            self.trace_split_decision(
                crate::trace::DecisionKind::Cu,
                x0,
                y0,
                log2_cb_size,
                leaf_cost,
                split_cost,
                0,
                split_exact,
            );

            if split_cost < leaf_cost {
                self.restore_tu_depth_region(&split_tu_depth);
                self.record_cu_winner(x0, y0, log2_cb_size, true);
                let plan = Self::cu_to_plan(&split);
                return (
                    TrialResult {
                        plan,
                        cost: RdCost {
                            distortion: split_distortion,
                            frac_bits: split_bits,
                            cost: split_cost,
                        },
                        confidence,
                    },
                    split,
                );
            }

            let node = Self::cu_leaf_node(leaf, confidence);
            self.restore_tu_depth_region(&leaf_tu_depth);
            self.record_cu_winner(x0, y0, log2_cb_size, false);
            let plan = Self::cu_to_plan(&node);
            return (
                TrialResult {
                    plan,
                    cost: RdCost {
                        distortion: leaf_distortion,
                        frac_bits: leaf_bits,
                        cost: leaf_cost,
                    },
                    confidence,
                },
                node,
            );
        }

        let leaf = self.build_cu_leaf(x0, y0, log2_cb_size, ct_depth, ctxs);
        let leaf_distortion = self.distortion_tt_region(x0, y0, log2_cb_size);
        let leaf_bits = estimate_cu_leaf_bits(ctxs, &leaf, log2_cb_size, self.cat)
            + estimate_split_cu_flag_bits(ctxs, ctx_inc, false);

        if self.cu_early_terminate(budget, &leaf, leaf_bits, x0, y0, log2_cb_size) {
            if budget.allow_cu_early_terminate {
                self.stats.cu_early_terminations += 1;
                let node = Self::cu_leaf_node(leaf, DecisionConfidence::Clear);
                self.record_cu_winner(x0, y0, log2_cb_size, false);
                let plan = Self::cu_to_plan(&node);
                return (
                    TrialResult {
                        plan,
                        cost: RdCost {
                            distortion: leaf_distortion,
                            frac_bits: leaf_bits,
                            cost: self.rd_cost(leaf_distortion, leaf_bits),
                        },
                        confidence: DecisionConfidence::Clear,
                    },
                    node,
                );
            }
            self.stats.policy_early_term_suppressed += 1;
        }

        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        let leaf_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let leaf_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let leaf_ct_depth_map = self.snapshot_ct_depth_region(x0, y0, log2_cb_size);
        let leaf_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);

        self.restore_frame_region(&base_frame);
        self.restore_mode_region(&base_mode_map);
        self.restore_ct_depth_region(&base_ct_depth_map);
        self.restore_tu_depth_region(&base_tu_depth);

        let cb_half = cb_size / 2;
        let kid_pos = [
            (x0, y0),
            (x0 + cb_half, y0),
            (x0, y0 + cb_half),
            (x0 + cb_half, y0 + cb_half),
        ];
        let mut kids = Vec::with_capacity(4);
        let mut acc_distortion = 0u64;
        let mut acc_node_bits = 0u64;
        let split_flag_bits = estimate_split_cu_flag_bits(ctxs, ctx_inc, true);
        let mut bound_abort = false;
        let split_e0 = self.stats.residual_bit_estimates;
        for &(kx, ky) in &kid_pos {
            let kid = self.build_cu(kx, ky, log2_cb_size - 1, ct_depth + 1, ctxs);
            acc_distortion += self.distortion_tt_region(kx, ky, log2_cb_size - 1);
            acc_node_bits +=
                estimate_cu_node_bits(ctxs, &kid, kx, ky, log2_cb_size - 1, ct_depth + 1, self);
            kids.push(kid);
            let partial_split_cost = self.rd_cost(acc_distortion, split_flag_bits + acc_node_bits);
            if partial_split_cost >= leaf_cost {
                bound_abort = true;
                break;
            }
        }

        if bound_abort {
            self.stats.cu_split_bound_aborts += 1;
            let partial_split_cost = self.rd_cost(acc_distortion, split_flag_bits + acc_node_bits);
            self.trace.note_cu_split_bound_abort(
                x0,
                y0,
                log2_cb_size,
                kids.len(),
                leaf_cost,
                partial_split_cost,
                (self.stats.residual_bit_estimates - split_e0) as u32,
            );
            self.restore_frame_region(&leaf_frame);
            self.restore_mode_region(&leaf_mode_map);
            self.restore_ct_depth_region(&leaf_ct_depth_map);
            self.restore_tu_depth_region(&leaf_tu_depth);
            let node = Self::cu_leaf_node(leaf, DecisionConfidence::Clear);
            self.record_cu_winner(x0, y0, log2_cb_size, false);
            let plan = Self::cu_to_plan(&node);
            return (
                TrialResult {
                    plan,
                    cost: RdCost {
                        distortion: leaf_distortion,
                        frac_bits: leaf_bits,
                        cost: leaf_cost,
                    },
                    confidence: DecisionConfidence::Clear,
                },
                node,
            );
        }

        let split_distortion = acc_distortion;
        let split_bits = split_flag_bits + acc_node_bits;
        let split_cost = self.rd_cost(split_distortion, split_bits);
        self.record_close_call(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );
        let confidence = Self::decision_confidence(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );
        let split_exact = (self.stats.residual_bit_estimates - split_e0) as u32;
        self.trace_split_decision(
            crate::trace::DecisionKind::Cu,
            x0,
            y0,
            log2_cb_size,
            leaf_cost,
            split_cost,
            0,
            split_exact,
        );

        if split_cost < leaf_cost {
            let node = CuNode::Split { kids };
            self.record_cu_winner(x0, y0, log2_cb_size, true);
            let plan = Self::cu_to_plan(&node);
            (
                TrialResult {
                    plan,
                    cost: RdCost {
                        distortion: split_distortion,
                        frac_bits: split_bits,
                        cost: split_cost,
                    },
                    confidence,
                },
                node,
            )
        } else {
            self.restore_frame_region(&leaf_frame);
            self.restore_mode_region(&leaf_mode_map);
            self.restore_ct_depth_region(&leaf_ct_depth_map);
            self.restore_tu_depth_region(&leaf_tu_depth);
            let node = Self::cu_leaf_node(leaf, confidence);
            self.record_cu_winner(x0, y0, log2_cb_size, false);
            let plan = Self::cu_to_plan(&node);
            (
                TrialResult {
                    plan,
                    cost: RdCost {
                        distortion: leaf_distortion,
                        frac_bits: leaf_bits,
                        cost: leaf_cost,
                    },
                    confidence,
                },
                node,
            )
        }
    }

    pub(super) fn cu_trial_result(
        &self,
        node: &CuNode,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> TrialResult<CuPlan> {
        let distortion = self.distortion_cu_node(node, x0, y0, log2_cb_size);
        let frac_bits = estimate_cu_node_bits(ctxs, node, x0, y0, log2_cb_size, ct_depth, self);
        TrialResult {
            plan: Self::cu_to_plan(node),
            cost: RdCost {
                distortion,
                frac_bits,
                cost: self.rd_cost(distortion, frac_bits),
            },
            confidence: DecisionConfidence::Clear,
        }
    }

    pub(super) fn build_cu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> CuNode {
        let saved_policy = self.cur_policy;
        let saved_qp = (self.cur_qp_y, self.cur_qp_c);
        if self.aq.active {
            self.cur_policy = Some(self.region_policy(x0, y0, log2_cb_size));
            self.aq_set_cu_qp(x0, y0);
        }

        let base_frame = self.snapshot_frame_region(x0, y0, log2_cb_size);
        let base_mode_map = self.snapshot_mode_region(x0, y0, log2_cb_size);
        let base_ct_depth_map = self.snapshot_ct_depth_region(x0, y0, log2_cb_size);
        let base_tu_depth = self.snapshot_tu_depth_region(x0, y0, log2_cb_size);
        let (trial, winner) = self.decide_cu(x0, y0, log2_cb_size, ct_depth, ctxs);
        let node = if self.effort_template.reference || self.best2_cu_reuse {
            winner
        } else {
            self.restore_frame_region(&base_frame);
            self.restore_mode_region(&base_mode_map);
            self.restore_ct_depth_region(&base_ct_depth_map);
            self.restore_tu_depth_region(&base_tu_depth);
            let mut node = self.final_code_cu(&trial.plan, x0, y0, log2_cb_size, ct_depth, ctxs);
            Self::annotate_root_cu_confidence(&mut node, trial.confidence);
            node
        };
        self.cur_policy = saved_policy;
        (self.cur_qp_y, self.cur_qp_c) = saved_qp;
        node
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

    let decode_chroma_cbf = cat != 0 && (log2_size > 2 || cat == 3);
    let (cbf_cb, cbf_cr) = (node.cbf_cb(), node.cbf_cr());
    let second_cbf = decode_second_cbf(cat, log2_size, is_split);
    let (cbf_cb1, cbf_cr1) = (node.cbf_cb1(), node.cbf_cr1());
    if decode_chroma_cbf {
        let ci = ctx::CBF_CBCR + trafo_depth as usize;
        if trafo_depth == 0 || parent_cbf_cb {
            est.encode_bin(cbf_cb as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cb) {
            est.encode_bin(cbf_cb1 as u8, ctxs.get(ci));
        }
        if trafo_depth == 0 || parent_cbf_cr {
            est.encode_bin(cbf_cr as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cr) {
            est.encode_bin(cbf_cr1 as u8, ctxs.get(ci));
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
            let eff_cbf_cb1 = second_cbf && (trafo_depth == 0 || parent_cbf_cb) && cbf_cb1;
            let eff_cbf_cr1 = second_cbf && (trafo_depth == 0 || parent_cbf_cr) && cbf_cr1;

            let ctx_off = if trafo_depth == 0 { 1 } else { 0 };
            est.encode_bin(l.luma.cbf as u8, ctxs.get(ctx::CBF_LUMA + ctx_off));

            if l.luma.cbf {
                est.add_frac_bits(l.luma.frac_bits);
            }
            if has_chroma_tb(cat, l.log2_size) {
                if eff_cbf_cb {
                    est.add_frac_bits(l.cb.frac_bits);
                }
                if eff_cbf_cb1 {
                    est.add_frac_bits(l.cb1.frac_bits);
                }
                if eff_cbf_cr {
                    est.add_frac_bits(l.cr.frac_bits);
                }
                if eff_cbf_cr1 {
                    est.add_frac_bits(l.cr1.frac_bits);
                }
            }
        }
    }
}

fn estimate_intra_luma_mode_bits(ctxs: &Contexts, mpm: [IntraPredMode; 3], mode: u8) -> u64 {
    let mut m = ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let mut est = CabacEstimator::new();
    let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
    let in_mpm = mpm_u8.iter().position(|&m| m == mode);

    if let Some(idx) = in_mpm {
        est.encode_bin(1, &mut m);
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
        est.encode_bin(0, &mut m);
        for _ in 0..5 {
            est.encode_bin_ep(0);
        }
    }

    est.frac_bits()
}

fn estimate_intra_chroma_mode_bits(ctxs: &Contexts, mode_idx: u8) -> u64 {
    let mut m = ctxs.models[ctx::INTRA_CHROMA_PRED_MODE];
    let mut est = CabacEstimator::new();
    if mode_idx == CHROMA_DM_IDX {
        est.encode_bin(0, &mut m);
    } else {
        est.encode_bin(1, &mut m);
        est.encode_bin_ep((mode_idx >> 1) & 1);
        est.encode_bin_ep(mode_idx & 1);
    }
    est.frac_bits()
}

fn estimate_luma_cu_trial_bits(
    ctxs: &Contexts,
    mpm: [IntraPredMode; 3],
    luma_mode: u8,
    tt: &Tt,
    log2_cb_size: u8,
) -> u64 {
    let mut bits = 0u64;
    if log2_cb_size == 3 {
        let mut m = ctxs.models[ctx::PART_MODE];
        let mut est = CabacEstimator::new();
        est.encode_bin(1, &mut m);
        bits += est.frac_bits();
    }
    bits += estimate_intra_luma_mode_bits(ctxs, mpm, luma_mode);
    bits += estimate_tt_bits(ctxs, tt, 0, false, true, true);
    bits
}

fn estimate_split_cu_flag_bits(ctxs: &Contexts, ctx_inc: usize, value: bool) -> u64 {
    let mut m = ctxs.models[ctx::SPLIT_CU_FLAG + ctx_inc];
    let mut est = CabacEstimator::new();
    est.encode_bin(value as u8, &mut m);
    est.frac_bits()
}

fn estimate_cu_leaf_bits(ctxs: &Contexts, leaf: &CuLeaf, log2_cb_size: u8, cat: u8) -> u64 {
    let mut bits = 0u64;
    if log2_cb_size == 3 {
        let mut m = ctxs.models[ctx::PART_MODE];
        let mut est = CabacEstimator::new();
        est.encode_bin(1, &mut m);
        bits += est.frac_bits();
    }
    bits += estimate_intra_luma_mode_bits(ctxs, leaf.mpm, leaf.luma_mode);
    if cat != 0 {
        bits += estimate_intra_chroma_mode_bits(ctxs, leaf.chroma_mode_idx);
    }
    bits += estimate_tt_bits(ctxs, &leaf.tt, cat, false, true, true);
    bits
}

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
