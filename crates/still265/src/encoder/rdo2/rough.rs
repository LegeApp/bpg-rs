//! Rough luma mode selection for the rdo2 Best call graph.
//!
//! This preserves the existing SATD scoring and pruning policy, but moves the
//! candidate-list owner out of `rdo_legacy.rs` so Best luma mode selection is no
//! longer accounted as legacy control flow.

use bpg_hevc_decode::hevc::intra::fill_mpm_candidates;

use crate::contexts::Contexts;
use crate::effort::RmdModeSet;
use crate::plan::LumaModePlan;
use crate::trace::{WorkBucket, WorkSample};

impl<'a> super::super::Encoder<'a> {
    pub(in crate::encoder) fn decide_luma_modes(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> LumaModePlan {
        let trough = self.prof.on.then(std::time::Instant::now);
        let plan = self.rdo2_decide_luma_modes_inner(x0, y0, log2_size, ct_depth, ctxs);
        if let Some(trough) = trough {
            self.prof.rough_search += trough.elapsed();
        }
        plan
    }

    fn rdo2_decide_luma_modes_inner(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        ct_depth: u8,
        ctxs: &Contexts,
    ) -> LumaModePlan {
        let work_start = self.trace.enabled.then(std::time::Instant::now);
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
            let modes = budget.rough_luma_modes(mpm_u8, prune_angular);
            let angular_count = modes.iter().filter(|&&m| (2..=34).contains(&m)).count();
            if !prune_angular && angular_count >= 16 {
                let n = size * size;
                let (uf, ft, center, _) = bpg_hevc_decode::hevc::intra::build_reference_borders(
                    &self.frame,
                    x0,
                    y0,
                    log2_size,
                    0,
                    true,
                );
                let mut batch = std::mem::take(&mut self.scratch_allangs);
                batch.clear();
                batch.resize(crate::primitives::intra_angs::ANGULAR_MODES * n, 0);
                crate::primitives::intra_pred_allangs(
                    &mut batch,
                    &uf,
                    &ft,
                    center,
                    log2_size,
                    0,
                    self.bit_depth,
                );
                for m in modes {
                    let src8_ref = has_src8.then_some(src8.as_slice());
                    let cost = if (2..=34).contains(&m) {
                        let off = crate::primitives::intra_angs::slot_offset(m, log2_size);
                        self.luma_rough_score_from_pred(
                            &batch[off..off + n],
                            size,
                            m,
                            ctxs,
                            mpm,
                            &src,
                            src8_ref,
                            &mut pred8,
                        )
                    } else {
                        self.score_luma_rough_mode(
                            x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                        )
                    };
                    scored.push((cost, m));
                }
                self.scratch_allangs = batch;
            } else {
                for m in modes {
                    let src8_ref = has_src8.then_some(src8.as_slice());
                    let cost = self.score_luma_rough_mode(
                        x0, y0, log2_size, m, ctxs, mpm, &src, src8_ref, &mut pred, &mut pred8,
                    );
                    scored.push((cost, m));
                }
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
            let max_cands = 2 + 6 + ((ct_depth as usize) >> 1);
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
        let modes_scored = scored.len() as u64;
        if let Some(start) = work_start {
            self.trace.note_work(
                WorkBucket::RoughLumaAllAngles,
                WorkSample {
                    wall_ns: start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    log2_size,
                    c_idx: 0,
                    prediction_calls: modes_scored,
                    source_block_calls: 1,
                    ..WorkSample::default()
                },
            );
        }
        scored.clear();
        self.scratch_scored = scored;
        self.scratch_src8 = src8;
        self.scratch_pred = pred;
        self.scratch_pred8 = pred8;
        LumaModePlan { candidates }
    }
}
