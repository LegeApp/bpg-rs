//! Rate-distortion optimisation: transform/quant/RDOQ, TU/CU mode and split
//! decision, CABAC bit estimation, and costing.

use bpg_hevc_decode::hevc::intra::predict_intra;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::{ctx, Contexts};
use crate::effort::{BlockSearchBudget, ComponentKind, SplitSearch, TrialQuality};
use crate::plan::{BlockPlan, WorkStage};
use crate::primitives;
use crate::rdoq;
use crate::residual::{estimate_residual_bits, get_scan_order, ResidualEstimateCache};
use crate::transform;
use crate::Effort;

use super::snapshot::ChromaLeafCache;
use super::types::*;

impl<'a> super::Encoder<'a> {
    fn assert_not_best_rdo1(&self, path: &str) {
        assert_ne!(
            self.effort,
            Effort::Best,
            "Effort::Best must not enter legacy RDO path `{path}`"
        );
    }

    pub(super) fn code_block_internal(
        &mut self,
        ctxs: &Contexts,
        plan: &BlockPlan,
        stage: WorkStage,
    ) -> CodedBlock {
        self.assert_not_best_rdo1("code_block_internal");
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
        let do_rdoq = self.rdoq_enabled_for_block(stage, plan.quality, budget, c_idx, log2_size);
        let balanced_final_gate =
            budget.effort == Effort::Balanced && stage == WorkStage::FinalCode;
        let use_single_scan_rdoq = self.single_scan_rdoq
            && do_rdoq
            && (!balanced_final_gate || (c_idx == 0 && log2_size <= 3));
        let (mut levels, mut nnz) = if use_single_scan_rdoq {
            if stage == WorkStage::FinalCode {
                self.stats.final_rdoq_blocks += 1;
            } else {
                self.stats.trial_rdoq_blocks += 1;
            }
            self.trace.note_rdoq_block(stage, c_idx, log2_size);
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
                self.best2_rdoq2,
            );
            if let Some(tr) = tr {
                self.prof.rdoq += tr.elapsed();
            }
            r
        } else {
            let (levels, nnz) = transform::quantize(&coeffs, log2_size, qp, self.bit_depth);
            let passes = if do_rdoq {
                if balanced_final_gate {
                    0
                } else {
                    budget.rdoq_passes(log2_size, ComponentKind::from_c_idx(c_idx), nnz)
                }
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
                self.trace.note_rdoq_block(stage, c_idx, log2_size);
                self.refine_levels_rdoq_limited(
                    ctxs, &coeffs, levels, log2_size, c_idx, mode, qp, passes,
                )
            }
        };
        // Sign-data-hiding: make each coding group's hidden sign parity-consistent
        // before reconstruction so encoder recon == decoder recon, and before the
        // writer (which omits exactly those signs). Uses the pre-quant `coeffs`,
        // so it must run before they return to scratch.
        if self.sign_data_hiding && nnz > 0 {
            let (scale, qbits) = transform::quant_params(log2_size, qp, self.bit_depth);
            let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
            nnz = crate::residual::apply_sign_data_hiding(
                &mut levels,
                &coeffs,
                log2_size,
                scan,
                scale,
                qbits,
                nnz,
            );
        }
        let cbf = nnz > 0;

        if cbf {
            self.stats.inverse_transforms += 1;
            transform::reconstruct_residual_into(
                &levels,
                log2_size,
                qp,
                self.bit_depth,
                is_dst,
                &mut coeffs,
                &mut transform_tmp,
            );
            let max_val = (1i32 << self.bit_depth) - 1;
            let (plane, stride) = self.frame.plane_mut(c_idx);
            for j in 0..size {
                for i in 0..size {
                    let idx = (y as usize + j) * stride + x as usize + i;
                    let pred = plane[idx] as i32;
                    plane[idx] =
                        (pred + transform_tmp[j * size + i] as i32).clamp(0, max_val) as u16;
                }
            }
        }

        self.scratch_coeffs = coeffs;
        self.scratch_transform_tmp = transform_tmp;

        let elide_final_pricing =
            cbf && stage == WorkStage::FinalCode && self.elide_final_residual_pricing;
        let exact = cbf
            && !elide_final_pricing
            && self.exact_residual_bits_for_block(stage, plan.quality, budget);
        let frac_bits = if !cbf {
            0
        } else if elide_final_pricing {
            self.stats.rdo2_residual_final_pricings_elided += 1;
            0
        } else if exact {
            self.residual_frac_bits(ctxs, &levels, log2_size, c_idx, mode)
        } else {
            self.approx_residual_frac_bits(ctxs, &levels, log2_size, c_idx)
        };

        if self.trace.enabled {
            if elide_final_pricing {
                self.trace
                    .note_final_pricing_elided(c_idx, x, y, log2_size, cbf);
            } else {
                self.trace
                    .note_code_block(stage, c_idx, x, y, log2_size, cbf, exact, frac_bits);
            }
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
        &mut self,
        stage: WorkStage,
        quality: TrialQuality,
        budget: BlockSearchBudget,
        c_idx: u8,
        log2_size: u8,
    ) -> bool {
        if budget.effort == Effort::Best
            && stage != WorkStage::FinalCode
            && self.best_tt_cheap_trial
        {
            if self.full_trial_rdoq_enabled_for_block(stage, quality, budget) {
                self.stats.best_tt_full_trial_rdoq_blocks_saved += 1;
            }
            return false;
        }
        if budget.effort == Effort::Best
            && stage != WorkStage::FinalCode
            && self.best_tt_exact_trial
        {
            return self.full_trial_rdoq_enabled_for_block(stage, quality, budget);
        }
        if budget.effort == Effort::Best && stage != WorkStage::FinalCode {
            match self.best_trial_rdoq_gate {
                BestTrialRdoqGate::Off => {}
                BestTrialRdoqGate::Chroma => {
                    if c_idx != 0 {
                        return false;
                    }
                }
                BestTrialRdoqGate::Luma32 => {
                    if c_idx == 0 && log2_size >= 5 {
                        return false;
                    }
                }
                BestTrialRdoqGate::Large => {
                    if c_idx == 0 && log2_size > 3 {
                        return false;
                    }
                }
                BestTrialRdoqGate::SmallLuma => {
                    return c_idx == 0 && log2_size <= 3;
                }
            }
        }
        match stage {
            WorkStage::FinalCode => true,
            _ => match quality {
                TrialQuality::Rough | TrialQuality::FastRd => false,
                TrialQuality::FullRd => budget.rdoq_for_trials,
                TrialQuality::Final => true,
            },
        }
    }

    fn full_trial_rdoq_enabled_for_block(
        &self,
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
        &mut self,
        stage: WorkStage,
        quality: TrialQuality,
        budget: BlockSearchBudget,
    ) -> bool {
        if self.best_tt_cheap_trial
            && budget.effort == Effort::Best
            && stage != WorkStage::FinalCode
        {
            if self.full_trial_exact_residual_bits_for_block(stage, quality, budget) {
                self.stats.best_tt_exact_residual_estimates_saved += 1;
            }
            return false;
        }
        if self.best_tt_exact_trial
            && budget.effort == Effort::Best
            && stage != WorkStage::FinalCode
        {
            return self.full_trial_exact_residual_bits_for_block(stage, quality, budget);
        }
        if self.best_trial_approx_bits
            && budget.effort == Effort::Best
            && stage != WorkStage::FinalCode
        {
            return false;
        }
        match stage {
            WorkStage::FinalCode => budget.effort != Effort::Balanced,
            _ => match quality {
                TrialQuality::Rough | TrialQuality::FastRd => false,
                TrialQuality::FullRd => budget.exact_residual_bits_for_trials,
                TrialQuality::Final => budget.exact_residual_bits_for_trials,
            },
        }
    }

    fn full_trial_exact_residual_bits_for_block(
        &self,
        stage: WorkStage,
        quality: TrialQuality,
        budget: BlockSearchBudget,
    ) -> bool {
        match stage {
            WorkStage::FinalCode => budget.effort != Effort::Balanced,
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
        self.assert_not_best_rdo1("estimate_block");
        debug_assert!(stage != WorkStage::FinalCode);
        debug_assert!(
            plan.quality == self.effective_trial_quality(plan.c_idx, budget),
            "trial block quality must come from the resolved block budget \
             (or the active luma close-call override)"
        );
        let coded = self.code_block_internal(ctxs, &plan, stage);
        BlockTrial { coded }
    }

    pub(super) fn final_code_block(&mut self, ctxs: &Contexts, plan: &BlockPlan) -> CodedBlock {
        debug_assert_eq!(plan.quality, crate::effort::TrialQuality::Final);
        self.code_block_internal(ctxs, plan, WorkStage::FinalCode)
    }

    #[inline]
    pub(super) fn effective_trial_quality(
        &self,
        c_idx: u8,
        budget: BlockSearchBudget,
    ) -> TrialQuality {
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
        self.assert_not_best_rdo1("trial_code_block");
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
        let r = estimate_residual_bits(
            &mut ctxs,
            levels,
            log2_size,
            c_idx,
            scan,
            self.sign_data_hiding,
        );
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

    pub(super) fn build_parent_chroma_tu(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        stage: WorkStage,
    ) -> Option<ParentChromaTu> {
        self.assert_not_best_rdo1("build_parent_chroma_tu");
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
        self.assert_not_best_rdo1("build_tt_leaf");
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
        if self.rdo2_tu && !self.in_rdo2 {
            return self.rdo2_analyze_tt(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }
        self.assert_not_best_rdo1("build_tt");
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
