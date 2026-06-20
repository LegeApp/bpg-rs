//! Coding-unit recursion and split decision owned by rdo2.
//!
//! This module moves the default Best CU driver out of `rdo_legacy.rs`. Some
//! leaf construction and block-pricing helpers are still transitional legacy
//! calls; the CU recursion, split-vs-leaf comparison, bound abort, and final
//! replay orchestration now live under rdo2.

use bpg_hevc_decode::hevc::intra::fill_mpm_candidates;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::contexts::Contexts;
use crate::effort::SplitSearch;
use crate::plan::{CuPlan, DecisionConfidence, RdCost, TrialResult};
use crate::preanalysis::RegionClass;
use crate::trace::WorkBucket;
use crate::Effort;

use super::super::types::{CuLeaf, CuNode, NxnInfo, PartNxnPrune, Tt};
use super::cost::{
    estimate_cu_leaf_bits, estimate_cu_node_bits, estimate_intra_luma_mode_bits,
    estimate_split_cu_flag_bits,
};
use super::policy::{EvalKind, EvalPolicy, RdoqPolicy, ResidualBitPolicy};

impl<'a> super::super::Encoder<'a> {
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

    pub(in crate::encoder) fn cu_trial_result(
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

    /// RD cost of one 4x4 PartNxN sub-PU coded with `mode`, priced at `kind`
    /// (`CheapTrial` = plain quant + approx bits for the screen, `ExactTrial` =
    /// RDOQ + exact bits for the close-call recheck). Non-committing.
    fn rdo2_nxn_pu_cost(
        &mut self,
        px: u32,
        py: u32,
        mode: u8,
        mpm: [IntraPredMode; 3],
        ctxs: &Contexts,
        policy: EvalPolicy,
    ) -> f64 {
        let e = self.rdo2_eval_leaf_block(ctxs, px, py, 2, 0, mode, self.cur_qp_y, policy);
        let mode_bits = estimate_intra_luma_mode_bits(ctxs, mpm, mode);
        let mut cbf_m = ctxs.models[ctx::CBF_LUMA];
        let mut est = CabacEstimator::new();
        est.encode_bin(e.coded.cbf as u8, &mut cbf_m);
        let luma_bits = est.frac_bits() + if e.coded.cbf { e.frac_bits } else { 0 };
        self.rd_cost(e.distortion, mode_bits + luma_bits)
    }

    pub(super) fn build_cu_leaf_nxn(
        &mut self,
        x0: u32,
        y0: u32,
        ct_depth: u8,
        ctxs: &Contexts,
        _use_eval: bool,
    ) -> (CuLeaf, u64, u64) {
        self.set_ct_depth(x0, y0, 3, ct_depth);
        let positions = [(x0, y0), (x0 + 4, y0), (x0, y0 + 4), (x0 + 4, y0 + 4)];
        let mut luma_modes = [0u8; 4];
        let mut mpms = [[IntraPredMode::Dc; 3]; 4];
        let mut kids: Vec<Tt> = Vec::with_capacity(4);

        for (i, &(px, py)) in positions.iter().enumerate() {
            let cand_a = self.neighbor_left_mode(px, py);
            let cand_b = self.neighbor_above_mode(px, py);
            let mpm = fill_mpm_candidates(cand_a, cand_b);
            let plan = self.decide_luma_modes(px, py, 2, ct_depth, ctxs);

            // Diagnostic staged screen (rdo2 principle, mirrors the 2Nx2N luma
            // path): rank candidates cheaply, then exact-recheck close calls.
            // This stayed byte-identical in the high-res smoke, but did not
            // reduce RDOQ counts once measured against a clean full-RDOQ
            // baseline, so keep the original full-RDOQ ranking as the default.
            let nxn_close_mult = std::env::var("BPG_NXN_CLOSE_MULT")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(1.0);
            let close_margin = self.block_budget(px, py, 2, 0).close_call_margin
                * self.rdo2_luma_close_mult
                * nxn_close_mult;
            let nxn_exact = std::env::var("BPG_NXN_EXACT").is_ok();
            let nxn_adaptive = !nxn_exact && std::env::var("BPG_NXN_ADAPTIVE").is_ok();
            // Screen policy: plain quant (no RDOQ). Residual bits default to exact
            // (accurate ranking without the RDOQ cost); `BPG_NXN_APPROX` falls back
            // to approximate bits. Default/`BPG_NXN_EXACT` is full-RDOQ ranking.
            let exact_policy =
                EvalPolicy::for_kind(EvalKind::ExactTrial).with_bucket(WorkBucket::NxnPuExact);
            let screen_policy = if nxn_adaptive {
                EvalPolicy {
                    rdoq: RdoqPolicy::Off,
                    bits: if std::env::var("BPG_NXN_APPROX").is_ok() {
                        ResidualBitPolicy::Approx
                    } else {
                        ResidualBitPolicy::Exact
                    },
                    commit: false,
                    work_bucket: Some(crate::trace::WorkBucket::NxnPuCheap),
                }
            } else {
                exact_policy
            };
            let mut cheap: Vec<(u8, f64)> = Vec::with_capacity(plan.candidates.len());
            let mut best_cheap = f64::MAX;
            for &mode in &plan.candidates {
                self.stats.cu_trials += 1;
                let cost = self.rdo2_nxn_pu_cost(px, py, mode, mpm, ctxs, screen_policy);
                best_cheap = best_cheap.min(cost);
                cheap.push((mode, cost));
            }
            let best_mode = if nxn_adaptive {
                // Adaptive exact recheck (RDOQ): re-price every candidate whose
                // cheap cost is within the close-call margin of the cheap best,
                // and keep the lowest exact cost. The winner is always
                // RDOQ-evaluated.
                let mut best_mode = cheap[0].0;
                let mut best_exact = f64::MAX;
                for &(mode, cc) in &cheap {
                    if Self::is_close_call(cc, best_cheap, close_margin) {
                        let ec = self.rdo2_nxn_pu_cost(px, py, mode, mpm, ctxs, exact_policy);
                        if ec < best_exact {
                            best_exact = ec;
                            best_mode = mode;
                        }
                    }
                }
                best_mode
            } else {
                // Full-RDOQ reference path: the screen above already priced every
                // candidate exactly. Select the minimum without a second exact
                // pass, matching the original behavior and keeping diagnostics
                // honest.
                cheap
                    .iter()
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|&(mode, _)| mode)
                    .unwrap_or(cheap[0].0)
            };

            self.store_mode(px, py, 2, best_mode);
            let tt = self.rdo2_final_code_tt_leaf(px, py, 2, 1, best_mode, best_mode, ctxs);
            kids.push(tt);
            luma_modes[i] = best_mode;
            mpms[i] = mpm;
        }

        let chroma = self.decide_chroma_mode(x0, y0, 3, luma_modes[0], ctxs);
        let parent_chroma = self.rdo2_final_parent_chroma_tu(x0, y0, 3, chroma.plan.mode, ctxs);
        let cbf_cb = parent_chroma.as_ref().map(|c| c.cb.cbf).unwrap_or(false);
        let cbf_cr = parent_chroma.as_ref().map(|c| c.cr.cbf).unwrap_or(false);

        let tt = Tt::Split {
            log2_size: 3,
            trafo_depth: 0,
            cbf_cb,
            cbf_cr,
            cbf_cb1: false,
            cbf_cr1: false,
            parent_chroma,
            kids,
        };

        let leaf = CuLeaf {
            mpm: mpms[0],
            luma_mode: luma_modes[0],
            chroma_mode_idx: chroma.plan.mode_idx,
            confidence: DecisionConfidence::Clear,
            tt,
            nxn: Some(NxnInfo { luma_modes, mpms }),
        };
        let distortion = self.distortion_tt_region(x0, y0, 3);
        let bits = estimate_cu_leaf_bits(ctxs, &leaf, 3, self.cat);
        (leaf, distortion, bits)
    }

    fn rough_best_luma_mode(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ctxs: &Contexts,
    ) -> (u8, u64) {
        let size = 1usize << log2_size;
        let src = self.source_block(0, x0, y0, size);
        let mut src8 = std::mem::take(&mut self.scratch_src8);
        src8.clear();
        if self.bit_depth == 8 {
            src8.extend(src.iter().map(|&v| v.min(255) as u8));
        }
        let cand_a = self.neighbor_left_mode(x0, y0);
        let cand_b = self.neighbor_above_mode(x0, y0);
        let mpm = fill_mpm_candidates(cand_a, cand_b);
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let budget = self.block_budget(x0, y0, log2_size, 0);
        let modes = budget.rough_luma_modes(mpm_u8, false);

        let mut pred = std::mem::take(&mut self.scratch_pred);
        let mut pred8 = std::mem::take(&mut self.scratch_pred8);
        let mut best = (u64::MAX, mpm_u8[0]);
        for m in modes {
            let src8_ref = (self.bit_depth == 8).then_some(src8.as_slice());
            let cost = self.score_luma_rough_mode(
                x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
            );
            if cost < best.0 {
                best = (cost, m);
            }
        }
        self.scratch_src8 = src8;
        self.scratch_pred = pred;
        self.scratch_pred8 = pred8;
        (best.1, best.0)
    }

    #[inline]
    fn rough_mode_family(mode: u8) -> u8 {
        match mode {
            0 | 1 => mode,
            2..=9 => 2,
            10 => 3,
            11..=17 => 4,
            18..=25 => 5,
            26 => 6,
            27..=34 => 7,
            _ => 8,
        }
    }

    fn should_try_partnxn_8x8(&mut self, x0: u32, y0: u32, ctxs: &Contexts) -> bool {
        match self.partnxn_prune {
            PartNxnPrune::Off => return true,
            PartNxnPrune::Conservative => {
                let region = self.analysis.region_class_at(x0, y0, 3);
                if matches!(
                    region,
                    RegionClass::TextLike
                        | RegionClass::DirectionalEdge
                        | RegionClass::ChromaCritical
                ) {
                    return true;
                }
            }
            PartNxnPrune::Aggressive => {}
        }

        let base_frame = self.snapshot_frame_region(x0, y0, 3);
        let (_, cost_8x8) = self.rough_best_luma_mode(x0, y0, 3, ctxs);
        self.restore_frame_region(&base_frame);

        let positions = [(x0, y0), (x0 + 4, y0), (x0, y0 + 4), (x0 + 4, y0 + 4)];
        let mut cost_4x4 = 0u64;
        let mut families = [false; 9];
        for &(px, py) in &positions {
            let base = self.snapshot_frame_region(px, py, 2);
            let (mode, cost) = self.rough_best_luma_mode(px, py, 2, ctxs);
            self.restore_frame_region(&base);
            cost_4x4 = cost_4x4.saturating_add(cost);
            families[Self::rough_mode_family(mode) as usize] = true;
        }
        let diverse = families[2..].iter().filter(|&&v| v).count() >= 2;
        let rough_gain = (cost_4x4 as u128) * 100 <= (cost_8x8 as u128) * 96;
        match self.partnxn_prune {
            PartNxnPrune::Aggressive => rough_gain || diverse,
            PartNxnPrune::Conservative => {
                rough_gain || (diverse && (cost_4x4 as u128) * 100 <= (cost_8x8 as u128) * 125)
            }
            PartNxnPrune::Off => true,
        }
    }

    pub(super) fn decide_cu_8x8_part(
        &mut self,
        x0: u32,
        y0: u32,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> (TrialResult<CuPlan>, CuNode) {
        self.stats.partnxn_attempts += 1;
        let base_frame = self.snapshot_frame_region(x0, y0, 3);
        let base_mode = self.snapshot_mode_region(x0, y0, 3);
        let base_tu = self.snapshot_tu_depth_region(x0, y0, 3);
        let region_idx = self.analysis.region_class_at(x0, y0, 3).index();

        let leaf2 = self.build_cu_leaf(x0, y0, 3, ct_depth, ctxs);
        let d2 = self.distortion_tt_region(x0, y0, 3);
        let b2 = estimate_cu_leaf_bits(ctxs, &leaf2, 3, self.cat);
        let c2 = self.rd_cost(d2, b2);
        let frame2 = self.snapshot_frame_region(x0, y0, 3);
        let mode2 = self.snapshot_mode_region(x0, y0, 3);
        let tu2 = self.snapshot_tu_depth_region(x0, y0, 3);

        self.restore_frame_region(&base_frame);
        self.restore_mode_region(&base_mode);
        self.restore_tu_depth_region(&base_tu);
        if !self.should_try_partnxn_8x8(x0, y0, ctxs) {
            self.stats.partnxn_skips += 1;
            self.restore_frame_region(&frame2);
            self.restore_mode_region(&mode2);
            self.restore_tu_depth_region(&tu2);
            let node = Self::cu_leaf_node(leaf2, DecisionConfidence::Clear);
            self.record_cu_winner(x0, y0, 3, false);
            let plan = Self::cu_to_plan(&node);
            let r = TrialResult {
                plan,
                cost: RdCost {
                    distortion: d2,
                    frac_bits: b2,
                    cost: c2,
                },
                confidence: DecisionConfidence::Clear,
            };
            return (r, node);
        }

        self.restore_frame_region(&base_frame);
        self.restore_mode_region(&base_mode);
        self.restore_tu_depth_region(&base_tu);
        let use_eval = self.rdo2_nxn && self.effort == Effort::Best;
        if use_eval {
            self.stats.rdo2_nxn_bound_attempts += 1;
        }
        let cu_trials0 = self.stats.cu_trials;
        let code_blocks0 = self.stats.code_block_calls;
        let (leafn, dn, bn) = self.build_cu_leaf_nxn(x0, y0, ct_depth, ctxs, use_eval);
        self.stats.partnxn_cu_trials += self.stats.cu_trials - cu_trials0;
        self.stats.partnxn_code_block_calls += self.stats.code_block_calls - code_blocks0;
        let cn = self.rd_cost(dn, bn);

        if cn < c2 {
            self.stats.partnxn_wins += 1;
            self.stats.partnxn_wins_by_region[region_idx] += 1;
            let node = Self::cu_leaf_node(leafn, DecisionConfidence::Clear);
            self.record_cu_winner(x0, y0, 3, false);
            let plan = Self::cu_to_plan(&node);
            let r = TrialResult {
                plan,
                cost: RdCost {
                    distortion: dn,
                    frac_bits: bn,
                    cost: cn,
                },
                confidence: DecisionConfidence::Clear,
            };
            return (r, node);
        }

        self.stats.partnxn_losses += 1;
        self.restore_frame_region(&frame2);
        self.restore_mode_region(&mode2);
        self.restore_tu_depth_region(&tu2);
        let node = Self::cu_leaf_node(leaf2, DecisionConfidence::Clear);
        self.record_cu_winner(x0, y0, 3, false);
        let r = self.cu_trial_result(&node, x0, y0, 3, ct_depth, ctxs);
        (r, node)
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
            if log2_cb_size == 3 && self.partnxn && self.cat == 1 && fully_inside {
                return self.decide_cu_8x8_part(x0, y0, ct_depth, ctxs);
            }
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

    pub(in crate::encoder) fn build_cu(
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
            let saved_elide = self.elide_final_residual_pricing;
            self.elide_final_residual_pricing = self.rdo2_tu && ct_depth == 0;
            let mut node =
                self.rdo2_final_code_cu(&trial.plan, x0, y0, log2_cb_size, ct_depth, ctxs);
            self.elide_final_residual_pricing = saved_elide;
            Self::annotate_root_cu_confidence(&mut node, trial.confidence);
            node
        };
        self.cur_policy = saved_policy;
        (self.cur_qp_y, self.cur_qp_c) = saved_qp;
        node
    }
}
