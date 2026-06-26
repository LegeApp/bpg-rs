//! Rough and exact luma mode search.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::encoder::Encoder;
use crate::primitives;
use crate::primitives::intra_angs;
use crate::primitives::{satd_u8, satd_u16};

use super::depth::StillSearchDepth;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::overlay::OverlayCache;
use super::plan::TtPlan;
use super::price::luma_mode_bits;
use super::source::CtuSourceCache;
use crate::encoder::types::{MAX_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb};

use crate::effort::{
    AngularFamily, CheapMode, ExactModeWinner, ExactUsage, ModeClass, ModeCost, RoughBlockEvidence,
    TrialRdoqMode,
};

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// Choose the CU's luma intra mode through a four-stage pipeline:
    ///
    /// 1. **Rough** — SATD-based scoring of all relevant intra modes.
    /// 2. **Shortlist** — representation-aware subset (anchors + angular fills).
    /// 3. **Cheap** — hard-quant luma-only TU leaf evaluations for ranking.
    /// 4. **Exact** — full RD search (TU-split, luma+chroma) on promoted modes.
    ///
    /// The retained exact winner carries its TT plan and detached recon patches;
    /// callers reattach it instead of replaying the winning mode.
    pub(super) fn decide_cu_luma_mode(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm: [IntraPredMode; 3],
        lambda: f64,
    ) -> ExactModeWinner<TtPlan, O::Saved> {
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let scale = CabacEstimator::SCALE as f64;
        let rough_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let size = 1usize << rough_log2;

        // Stage 0: Rough SATD scoring.
        let evidence = self.compute_rough_scores(state, x0, y0, log2_cb_size, mpm, lambda, scale);

        // Store the best rough score for the NxN skip heuristic (consumed by
        // decide_cu_min_leaf_or_nxn for 8x8 CUs).
        if log2_cb_size == 3 {
            self.workspace.last_8x8_rough_satd = evidence.best_satd;
        }

        // Stage 1: Build representation-aware shortlist.
        let sl_policy = &state.effort_template.luma.shortlist;
        let shortlist = build_luma_shortlist(&evidence.rough_costs, mpm_u8, sl_policy);

        let cheap_policy = &state.effort_template.luma.cheap;
        let exact_policy = &state.effort_template.luma.exact;
        let cheap_enabled = super::env::luma_cheap_enabled(cheap_policy.enabled);

        let winner = match (cheap_enabled, exact_policy.exact_usage) {
            // Placebo: bypass cheap, run all shortlist through exact.
            (false, _) | (_, ExactUsage::AllShortlist) => self.evaluate_exact_modes(
                state,
                x0,
                y0,
                log2_cb_size,
                mpm_u8,
                lambda,
                scale,
                &shortlist,
            ),
            // Fast: use cheap winner directly, skip exact entirely.
            (true, ExactUsage::Disabled) => {
                let (_cheap_ranked, cheap_winner) = self.rank_cheap_modes(
                    state,
                    x0,
                    y0,
                    log2_cb_size,
                    mpm_u8,
                    lambda,
                    scale,
                    &shortlist,
                    &evidence,
                );
                let cheap_winner = cheap_winner
                    .expect("rank_cheap_modes must return a winner when cheap is enabled");
                self.materialize_cheap_winner(
                    state,
                    x0,
                    y0,
                    log2_cb_size,
                    mpm_u8,
                    lambda,
                    scale,
                    cheap_winner.mode,
                )
            }
            // Slow: cheap ranking → promote subset → exact.
            (true, ExactUsage::PromotedModes) => {
                let (cheap_ranked, _cheap_winner) = self.rank_cheap_modes(
                    state,
                    x0,
                    y0,
                    log2_cb_size,
                    mpm_u8,
                    lambda,
                    scale,
                    &shortlist,
                    &evidence,
                );
                let exact_set = build_exact_set(&shortlist, &cheap_ranked, &exact_policy.promote);
                let winner = self.evaluate_exact_modes(
                    state,
                    x0,
                    y0,
                    log2_cb_size,
                    mpm_u8,
                    lambda,
                    scale,
                    &exact_set,
                );
                if super::env::luma_oracle_enabled() {
                    self.run_luma_oracle(
                        state,
                        x0,
                        y0,
                        log2_cb_size,
                        mpm_u8,
                        lambda,
                        scale,
                        &evidence.rough_costs,
                        &shortlist,
                        Some(&cheap_ranked),
                        exact_set.len(),
                        winner.mode,
                        winner.cost,
                    );
                }
                winner
            }
        };

        #[cfg(test)]
        if winner.mode != IntraPredMode::Dc.as_u8() {
            super::api::LUMA_NONDC_PICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if super::env::rough_audit_enabled() {
            emit_rough_audit(&evidence, &shortlist, x0, y0, size, winner.mode);
        }
        if !cheap_enabled && super::env::luma_oracle_enabled() {
            let rough_vec: Vec<ModeCost> = evidence.rough_costs;
            self.run_luma_oracle(
                state,
                x0,
                y0,
                log2_cb_size,
                mpm_u8,
                lambda,
                scale,
                &rough_vec,
                &shortlist,
                None,
                0,
                winner.mode,
                winner.cost,
            );
        }
        winner
    }

    // ── Stage 0: Rough SATD scoring ──────────────────────────────

    /// Score intra modes by SATD (plus mode-bit cost) using the effort
    /// template's `RmdModeSet` to control angular density.
    fn compute_rough_scores(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm: [IntraPredMode; 3],
        lambda: f64,
        scale: f64,
    ) -> RoughBlockEvidence {
        let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
        let rough_log2 = log2_cb_size.min(MAX_TB_LOG2);
        let size = 1usize << rough_log2;
        let n = size * size;
        let lambda_sad = lambda.sqrt();
        let mbits_weight = super::env::rough_mode_bit_weight_override().unwrap_or(1.0);
        let scalar_angular = super::env::scalar_angular_rough_enabled();
        let mut rough_costs: Vec<ModeCost> = Vec::with_capacity(35);
        let rough_timer = StillSearchLedger::start_timer();

        // Build the mode list from the rough policy's RmdModeSet.
        let rough_policy = &state.effort_template.luma.rough;
        let mut mode_list = rough_policy.mode_set.modes(mpm_u8);
        // Ensure MPMs are always present (the policy may not include them).
        for &m in &mpm_u8 {
            if !mode_list.contains(&m) {
                mode_list.push(m);
            }
        }

        if state.bit_depth == 8 {
            let mut src = vec![0u8; n];
            self.source.sample_block_u8(0, x0, y0, size, &mut src);
            let mut pred = vec![0u8; n];
            let mut tmp_u16 = Vec::with_capacity(n);
            for &m in mode_list.iter().filter(|&&m| m <= 1) {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into_u8(state, x0, y0, rough_log2, 0, mode, &mut pred, &mut tmp_u16);
                let satd = satd_u8(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                let (class, family) = classify_mode(m);
                rough_costs.push(ModeCost {
                    mode: m,
                    cost,
                    satd,
                    class,
                    family,
                });
            }
            let angular_modes = mode_list.iter().filter(|&&m| (2..=34).contains(&m)).count();
            let has_angular = angular_modes > 0;
            let use_batched_angular = !scalar_angular && angular_modes >= 8;
            if has_angular && use_batched_angular {
                let mut batch8 = std::mem::take(&mut self.workspace.block_scratch.rough_pred_u8);
                batch8.resize(intra_angs::ANGULAR_MODES * n, 0);
                self.predict_all_angular_into_u8(state, x0, y0, rough_log2, &mut batch8);
                for &m in mode_list.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let off = intra_angs::slot_offset(m, rough_log2);
                    let satd = satd_u8(&src, size, &batch8[off..off + n], size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                    let (class, family) = classify_mode(m);
                    rough_costs.push(ModeCost {
                        mode: m,
                        cost,
                        satd,
                        class,
                        family,
                    });
                }
                self.workspace.block_scratch.rough_pred_u8 = batch8;
            } else if has_angular {
                for &m in mode_list.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let mode = IntraPredMode::from_u8(m).unwrap();
                    self.predict_into_u8(
                        state,
                        x0,
                        y0,
                        rough_log2,
                        0,
                        mode,
                        &mut pred,
                        &mut tmp_u16,
                    );
                    let satd = satd_u8(&src, size, &pred, size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                    let (class, family) = classify_mode(m);
                    rough_costs.push(ModeCost {
                        mode: m,
                        cost,
                        satd,
                        class,
                        family,
                    });
                }
            }
        } else {
            let mut src = vec![0u16; n];
            for y in 0..size {
                for x in 0..size {
                    src[y * size + x] = self.source.sample(0, x0 + x as u32, y0 + y as u32);
                }
            }
            let mut pred = vec![0u16; n];
            for &m in mode_list.iter().filter(|&&m| m <= 1) {
                let mode = IntraPredMode::from_u8(m).expect("0..=1 are valid intra modes");
                self.predict_into(state, x0, y0, rough_log2, 0, mode, &mut pred);
                let satd = satd_u16(&src, size, &pred, size, size);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                let (class, family) = classify_mode(m);
                rough_costs.push(ModeCost {
                    mode: m,
                    cost,
                    satd,
                    class,
                    family,
                });
            }
            let angular_modes = mode_list.iter().filter(|&&m| (2..=34).contains(&m)).count();
            let has_angular = angular_modes > 0;
            let use_batched_angular = !scalar_angular && angular_modes >= 8;
            if has_angular && use_batched_angular {
                let mut batch =
                    std::mem::take(&mut self.workspace.block_scratch.rough_angular_batch_u16);
                batch.resize(intra_angs::ANGULAR_MODES * n, 0);
                self.predict_all_angular_into_u16(state, x0, y0, rough_log2, &mut batch);
                for &m in mode_list.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let off = intra_angs::slot_offset(m, rough_log2);
                    let slot = &batch[off..off + n];
                    let satd = satd_u16(&src, size, slot, size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                    let (class, family) = classify_mode(m);
                    rough_costs.push(ModeCost {
                        mode: m,
                        cost,
                        satd,
                        class,
                        family,
                    });
                }
                self.workspace.block_scratch.rough_angular_batch_u16 = batch;
            } else if has_angular {
                for &m in mode_list.iter().filter(|&&m| (2..=34).contains(&m)) {
                    let mode = IntraPredMode::from_u8(m).unwrap();
                    self.predict_into(state, x0, y0, rough_log2, 0, mode, &mut pred);
                    let satd = satd_u16(&src, size, &pred, size, size);
                    let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, m);
                    let cost = satd as f64 + mbits_weight * lambda_sad * mbits as f64 / scale;
                    let (class, family) = classify_mode(m);
                    rough_costs.push(ModeCost {
                        mode: m,
                        cost,
                        satd,
                        class,
                        family,
                    });
                }
            }
        }
        self.workspace.ledger.bump(WorkBucket::RoughLuma);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::RoughLuma, rough_timer);

        // Cost-ascending, mode-ascending tiebreak.
        rough_costs.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.mode.cmp(&b.mode))
        });

        let best_global = rough_costs[0].mode;
        let best_satd = rough_costs[0].cost;
        let best_planar_dc = rough_costs.iter().find(|mc| mc.mode <= 1).map(|mc| mc.mode);
        let best_angular = rough_costs.iter().find(|mc| mc.mode >= 2).map(|mc| mc.mode);
        let best_angular_by_family = [
            AngularFamily::Horizontal,
            AngularFamily::Diagonal,
            AngularFamily::Vertical,
        ]
        .map(|f| {
            rough_costs
                .iter()
                .find(|mc| mc.family == Some(f))
                .map(|mc| mc.mode)
        });

        RoughBlockEvidence {
            best_global,
            best_planar_dc,
            best_angular,
            best_angular_by_family,
            rough_costs,
            best_satd,
            activity: 0,
            range: 0,
            directional_strength: 0,
        }
    }

    // ── Stage 2: Cheap ranking (with optional chroma SATD) ───────

    /// Rank shortlisted modes by TU-leaf cost with optional chroma SATD
    /// cost.  Also returns the cheapest mode's retained TT plan and recon
    /// so the caller can use it directly when the exact pass is disabled.
    fn rank_cheap_modes(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm_u8: [u8; 3],
        lambda: f64,
        scale: f64,
        shortlist: &[u8],
        evidence: &RoughBlockEvidence,
    ) -> (Vec<CheapMode>, Option<ExactModeWinner<TtPlan, O::Saved>>) {
        let cheap_tpl = &state.effort_template.luma.cheap;
        let chroma_satd_enabled =
            cheap_tpl.chroma_satd_in_cheap && has_chroma_tb(state.cat, log2_cb_size);
        let geoms = chroma_satd_enabled
            .then(|| chroma_tb_geom(state.cat, x0, y0, log2_cb_size))
            .flatten();
        let lambda_sad = lambda.sqrt();

        let mut best_cost = f64::MAX;
        let mut best_plan: Option<(TtPlan, O::Saved)> = None;
        let cheap_limit = if cheap_tpl.max_ranked_modes == 0 {
            shortlist.len()
        } else {
            shortlist.len().min(cheap_tpl.max_ranked_modes as usize)
        };
        let cheap_modes = &shortlist[..cheap_limit];
        let mut ranked: Vec<CheapMode> = Vec::with_capacity(cheap_modes.len());

        for &mode in cheap_modes {
            let cheap_timer = StillSearchLedger::start_timer();
            let mark = self.overlay.mark();
            let (tt, tt_cost) =
                self.decide_tt_luma_no_optional_split(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let mut total = tt_cost + lambda * mbits as f64 / scale;

            // Approximate chroma RD cost via SATD.
            let ch_satd = if let Some((cx, cy, clog2, count)) = geoms {
                self.cheap_chroma_satd_sum(state, cx, cy, clog2, count, mode, lambda_sad, scale)
            } else {
                0.0
            };
            total += ch_satd;

            if total < best_cost {
                drop(best_plan.take());
                best_cost = total;
                best_plan = Some((tt, self.overlay.detach_from(mark)));
            } else {
                self.overlay.truncate(mark);
            }

            self.workspace.ledger.bump(WorkBucket::LumaCheap);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::LumaCheap, cheap_timer);

            let rough_rank = evidence
                .rough_costs
                .iter()
                .position(|mc| mc.mode == mode)
                .unwrap_or(usize::MAX);
            ranked.push(CheapMode {
                mode,
                cost: total,
                luma_cbf: false,
                residual_bits: 0,
                distortion: 0,
                rough_rank,
            });
        }
        ranked.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.mode.cmp(&b.mode))
        });

        let cheap_winner = best_plan.map(|(tt, recon)| ExactModeWinner {
            mode: ranked[0].mode,
            cost: ranked[0].cost,
            tt,
            recon,
        });
        (ranked, cheap_winner)
    }

    /// Sum of SATD-based chroma costs for the full CU-area chroma
    /// prediction residual.  Iterates in sub-blocks of at most
    /// MAX_TB_LOG2 so the intra-prediction primitive never exceeds
    /// its 32×32 border array.
    fn cheap_chroma_satd_sum(
        &mut self,
        state: &Encoder<'_>,
        cx: u32,
        cy: u32,
        clog2: u8,
        count: u8,
        luma_mode: u8,
        lambda_sad: f64,
        scale: f64,
    ) -> f64 {
        let cmode = IntraPredMode::from_u8(chroma_pred_mode(state.cat, luma_mode))
            .unwrap_or(IntraPredMode::Dc);
        let block_log2 = clog2.min(MAX_TB_LOG2);
        let block_size = 1usize << block_log2;
        let block_pixels = block_size * block_size;
        let stride_full = 1usize << clog2;
        let step = 1u32 << block_log2;
        let bit_depth = state.bit_depth;

        // Number of sub-blocks needed to cover one full clog2-sized row
        // (applies to both X and Y for 4:4:4, or Y for 4:2:2).
        let tiles_x = if clog2 > block_log2 {
            1usize << (clog2 - block_log2)
        } else {
            1
        };
        let tiles_y = if clog2 > block_log2 {
            1usize << (clog2 - block_log2)
        } else {
            1
        };

        let mut total_satd: u32 = 0;

        if bit_depth == 8 {
            let mut src = std::mem::take(&mut self.workspace.block_scratch.component_src_u8);
            let mut pred = std::mem::take(&mut self.workspace.block_scratch.component_pred_u8);
            let mut tmp = std::mem::take(&mut self.workspace.block_scratch.component_pred_tmp_u16);
            src.resize(block_pixels, 0);
            pred.resize(block_pixels, 0);

            for row in 0..count {
                let y_base = cy + (row as u32) * stride_full as u32;
                for ti_y in 0..tiles_y {
                    let y_off = y_base + (ti_y as u32) * step;
                    for ti_x in 0..tiles_x {
                        let x_off = cx + (ti_x as u32) * step;
                        for c_idx in [1u8, 2] {
                            self.source
                                .sample_block_u8(c_idx, x_off, y_off, block_size, &mut src);
                            self.predict_into_u8(
                                state, x_off, y_off, block_log2, c_idx, cmode, &mut pred, &mut tmp,
                            );
                            total_satd += satd_u8(&src, block_size, &pred, block_size, block_size);
                        }
                    }
                }
            }

            self.workspace.block_scratch.component_src_u8 = src;
            self.workspace.block_scratch.component_pred_u8 = pred;
            self.workspace.block_scratch.component_pred_tmp_u16 = tmp;
        } else {
            for row in 0..count {
                let y_base = cy + (row as u32) * stride_full as u32;
                for ti_y in 0..tiles_y {
                    let y_off = y_base + (ti_y as u32) * step;
                    for ti_x in 0..tiles_x {
                        let x_off = cx + (ti_x as u32) * step;
                        for c_idx in [1u8, 2] {
                            let mut src = vec![0u16; block_pixels];
                            for y in 0..block_size {
                                for x in 0..block_size {
                                    src[y * block_size + x] = self.source.sample(
                                        c_idx,
                                        x_off + x as u32,
                                        y_off + y as u32,
                                    );
                                }
                            }
                            let mut pred = vec![0u16; block_pixels];
                            self.predict_into(
                                state, x_off, y_off, block_log2, c_idx, cmode, &mut pred,
                            );
                            total_satd += satd_u16(&src, block_size, &pred, block_size, block_size);
                        }
                    }
                }
            }
        }
        lambda_sad * total_satd as f64 / scale
    }

    // ── Stage 4: Full exact TU-split evaluation ───────────────────

    /// Evaluate every mode in `modes` with the full recursive `decide_tt`
    /// (TU-split, luma+chroma, exact residual pricing).  Returns the winner
    /// with its TT plan and detached recon.
    ///
    /// When the effort template has RDOQ trials enabled (`ExactCloseOnly`,
    /// `PlaceboAllExact`, etc.), a second pass selectively re-evaluates
    /// close candidates with [`QuantMode::RdoqTrial`] so that RDOQ
    /// influences the mode and TU-split ranking (matching x265's
    /// `--rdoq-level=1`).
    fn evaluate_exact_modes(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm_u8: [u8; 3],
        lambda: f64,
        scale: f64,
        modes: &[u8],
    ) -> ExactModeWinner<TtPlan, O::Saved> {
        // Phase 1: Hard-quant evaluation of every candidate mode.
        let mut trials: Vec<ExactModeWinner<TtPlan, O::Saved>> = Vec::with_capacity(modes.len());
        for &mode in modes {
            let exact_timer = StillSearchLedger::start_timer();
            let mark = self.overlay.mark();
            let (tt, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let cost = tt_cost + lambda * mbits as f64 / scale;
            record_exact_stats(state, mode, tt_cost);
            let recon = self.overlay.detach_from(mark);
            self.workspace.ledger.bump(WorkBucket::LumaExact);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::LumaExact, exact_timer);
            trials.push(ExactModeWinner {
                mode,
                tt,
                recon,
                cost,
            });
        }

        // Phase 2: Selective RDOQ trial for close candidates (Slow/Placebo).
        let rdoq_policy = state.effort_template.rdoq_trials;
        if rdoq_policy.mode != TrialRdoqMode::Off && rdoq_policy.max_rdoq_modes > 0 {
            trials.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());
            let best_cost = trials[0].cost;
            let n = trials.len().min(rdoq_policy.max_rdoq_modes as usize);
            for i in 0..n {
                let allowed = match rdoq_policy.mode {
                    TrialRdoqMode::Off => false,
                    TrialRdoqMode::ExactOnly => true,
                    TrialRdoqMode::ExactCloseOnly | TrialRdoqMode::CheapCloseAndExact => {
                        trials[i].cost <= best_cost * rdoq_policy.close_margin
                    }
                    TrialRdoqMode::PlaceboAllExact => true,
                };
                if !allowed {
                    continue;
                }

                let mode = trials[i].mode;
                let rdoq_timer = StillSearchLedger::start_timer();
                let mark = self.overlay.mark();
                let mut tu = state.effort_template.tu_exact;
                if !tu.rdoq_split_trials {
                    tu.split_mode = crate::effort::TuSplitMode::LeafFirstEarlyTerminate;
                    tu.max_extra_depth = 0;
                }
                let cfg = super::tu::ExactEvalConfig {
                    quant: super::eval::QuantMode::RdoqTrial {
                        level: rdoq_policy.level,
                    },
                    residual_pricing: super::eval::ResidualPricingMode::Exact,
                    // FullComponents ensures chroma recon is written to the
                    // overlay alongside luma, so the final pass has valid
                    // chroma reference samples.
                    scope: super::tu::TtEvalScope::FullComponents,
                    tu,
                    retain_coeff: true,
                };
                let (tt, tt_cost) =
                    self.decide_tt_with_config(state, x0, y0, log2_cb_size, 0, mode, lambda, cfg);
                let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
                let cost = tt_cost + lambda * mbits as f64 / scale;
                record_exact_stats(state, mode, tt_cost);
                let recon = self.overlay.detach_from(mark);
                self.workspace.ledger.bump(WorkBucket::LumaExact);
                self.workspace
                    .ledger
                    .finish_timer(WorkBucket::LumaExact, rdoq_timer);
                trials[i] = ExactModeWinner {
                    mode,
                    tt,
                    recon,
                    cost,
                };
            }
        }

        trials
            .into_iter()
            .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
            .unwrap()
    }

    /// Re-evaluate the cheap winner once with full components and retained
    /// coeffs, but keep optional TU split search disabled. This preserves the
    /// cheap-stage architecture while materializing valid chroma reconstruction
    /// for final writing.
    fn materialize_cheap_winner(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm_u8: [u8; 3],
        lambda: f64,
        scale: f64,
        mode: u8,
    ) -> ExactModeWinner<TtPlan, O::Saved> {
        let exact_timer = StillSearchLedger::start_timer();
        let mark = self.overlay.mark();
        let mut tu = state.effort_template.tu_exact;
        tu.split_mode = crate::effort::TuSplitMode::Disabled;
        tu.max_extra_depth = 0;
        let cfg = super::tu::ExactEvalConfig {
            quant: super::eval::QuantMode::HardQuantSearch,
            residual_pricing: super::eval::ResidualPricingMode::Exact,
            scope: super::tu::TtEvalScope::FullComponents,
            tu,
            retain_coeff: true,
        };
        let (tt, tt_cost) =
            self.decide_tt_with_config(state, x0, y0, log2_cb_size, 0, mode, lambda, cfg);
        let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
        let cost = tt_cost + lambda * mbits as f64 / scale;
        record_exact_stats(state, mode, tt_cost);
        let recon = self.overlay.detach_from(mark);
        self.workspace.ledger.bump(WorkBucket::LumaExact);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::LumaExact, exact_timer);
        ExactModeWinner {
            mode,
            tt,
            recon,
            cost,
        }
    }

    /// Sampled oracle diagnostic: for every N-th CU, run full `decide_tt()` on
    /// all 35 intra modes and report the true-best winner.  Does not change the
    /// current decision — purely observational.  Columns printed to stderr:
    ///
    ///   ORACLE: x,y,log2,cur,rough_best,sl_best,all_best,
    ///           all_is_ang,all_in_sl,all_in_cheap_topn,
    ///           all_rough_rank,all_cheap_rank,
    ///           cur_cost,all_cost,delta
    #[allow(clippy::too_many_arguments)]
    fn run_luma_oracle(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        mpm_u8: [u8; 3],
        lambda: f64,
        scale: f64,
        rough: &[ModeCost],
        shortlist: &[u8],
        cheap_ranked: Option<&[CheapMode]>,
        exact_top: usize,
        current_mode: u8,
        current_cost: f64,
    ) {
        const ALL_MODES: [u8; 35] = ALL_INTRA_MODES;
        static ORACLE_CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let ctr = ORACLE_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if ctr % super::env::luma_oracle_mod() != 0 {
            return;
        }

        let oracle_mark = self.overlay.mark();

        // Full decide_tt over the current shortlist.
        let mut best_sl_cost = f64::MAX;
        let mut best_sl_mode = 255u8;
        for &mode in shortlist {
            let m = self.overlay.mark();
            let (_, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let total = tt_cost + lambda * mbits as f64 / scale;
            if total < best_sl_cost {
                best_sl_cost = total;
                best_sl_mode = mode;
            }
            self.overlay.truncate(m);
        }

        // Full decide_tt over all 35 modes.
        let mut best_all_cost = f64::MAX;
        let mut best_all_mode = 255u8;
        for &mode in &ALL_MODES {
            let m = self.overlay.mark();
            let (_, tt_cost) = self.decide_tt(state, x0, y0, log2_cb_size, 0, mode, lambda);
            let mbits = luma_mode_bits(&self.workspace.price_base, mpm_u8, mode);
            let total = tt_cost + lambda * mbits as f64 / scale;
            if total < best_all_cost {
                best_all_cost = total;
                best_all_mode = mode;
            }
            self.overlay.truncate(m);
        }

        self.overlay.truncate(oracle_mark);

        let best_all_is_angular = (2..=34).contains(&best_all_mode);
        let best_all_in_shortlist = shortlist.contains(&best_all_mode);
        let best_all_rough_rank = rough
            .iter()
            .position(|mc| mc.mode == best_all_mode)
            .map_or(999, |r| r + 1);
        let (best_all_cheap_rank, best_all_in_cheap_topn) = match cheap_ranked {
            Some(cr) => {
                let rank = cr
                    .iter()
                    .position(|cm| cm.mode == best_all_mode)
                    .map_or(999, |r| r + 1);
                let in_topn = if exact_top > 0 {
                    cr.iter().take(exact_top).any(|cm| cm.mode == best_all_mode)
                } else {
                    false
                };
                (rank, in_topn)
            }
            None => (999, false),
        };
        let delta = current_cost - best_all_cost;

        eprintln!(
            "ORACLE: {},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1}",
            x0,
            y0,
            log2_cb_size,
            current_mode,
            rough.first().map_or(255, |mc| mc.mode),
            best_sl_mode,
            best_all_mode,
            best_all_is_angular,
            best_all_in_shortlist,
            best_all_in_cheap_topn,
            best_all_rough_rank,
            best_all_cheap_rank,
            current_cost,
            best_all_cost,
            delta,
        );
    }

    pub(super) fn predict_all_angular_into_u8(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        dst: &mut [u8],
    ) {
        let tile_bounds = state.tile_clamp_bounds(x0, y0, 0);
        let overlay = &self.overlay;
        let (uf, ft, center, bit_depth) =
            bpg_hevc_decode::hevc::intra::build_reference_borders_with_reader(
                &state.frame,
                x0,
                y0,
                log2_size,
                0,
                true,
                |c, rx, ry| {
                    if let Some((tx0, ty0, tx1, ty1)) = tile_bounds {
                        if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                            return Some(bpg_hevc_decode::hevc::UNINIT_SAMPLE);
                        }
                    }
                    overlay.sample(c, rx, ry)
                },
            );
        primitives::pred_allangs_u8(dst, &uf, &ft, center, log2_size, 0, bit_depth);
    }

    pub(super) fn predict_all_angular_into_u16(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        dst: &mut [u16],
    ) {
        let tile_bounds = state.tile_clamp_bounds(x0, y0, 0);
        let overlay = &self.overlay;
        let (uf, ft, center, bit_depth) =
            bpg_hevc_decode::hevc::intra::build_reference_borders_with_reader(
                &state.frame,
                x0,
                y0,
                log2_size,
                0,
                true,
                |c, rx, ry| {
                    if let Some((tx0, ty0, tx1, ty1)) = tile_bounds {
                        if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                            return Some(bpg_hevc_decode::hevc::UNINIT_SAMPLE);
                        }
                    }
                    overlay.sample(c, rx, ry)
                },
            );
        primitives::intra_pred_allangs(dst, &uf, &ft, center, log2_size, 0, bit_depth);
    }
}

/// Shortlist builder with representation guarantees.
///
/// Ensures the exact pass always sees a meaningful angular challenger
/// by mandating best-in-category anchors before filling optional slots
/// by rough cost.
struct CandidateSet {
    max_size: usize,
    modes: Vec<u8>,
}

impl CandidateSet {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            modes: Vec::with_capacity(max_size),
        }
    }

    fn insert_mandatory(&mut self, mode: u8) {
        if !self.modes.contains(&mode) {
            self.modes.push(mode);
        }
    }

    fn is_full(&self) -> bool {
        self.modes.len() >= self.max_size
    }

    fn insert_optional(&mut self, mode: u8) {
        if self.is_full() || self.modes.contains(&mode) {
            return;
        }
        self.modes.push(mode);
    }

    fn into_vec(self) -> Vec<u8> {
        self.modes
    }
}

fn classify_mode(mode: u8) -> (ModeClass, Option<AngularFamily>) {
    if mode == 0 {
        (ModeClass::Planar, None)
    } else if mode == 1 {
        (ModeClass::Dc, None)
    } else {
        (ModeClass::Angular, AngularFamily::classify(mode))
    }
}

/// Find the best-cost mode in `rough` matching `predicate`.
fn best_of_category(mc: &[ModeCost], predicate: impl Fn(u8) -> bool) -> Option<u8> {
    mc.iter().find(|mc| predicate(mc.mode)).map(|mc| mc.mode)
}

/// Return neighboring angular modes around `mode` within `radius` steps.
fn angular_neighbors(mode: u8, radius: u8) -> Vec<u8> {
    if !(2..=34).contains(&mode) || radius == 0 {
        return vec![];
    }
    let r = radius as i8;
    let mut out = Vec::new();
    for d in -r..=r {
        if d == 0 {
            continue;
        }
        let n = mode as i8 + d;
        if (2..=34).contains(&n) {
            out.push(n as u8);
        }
    }
    out
}

/// Build a representation-aware shortlist from rough-scored modes.
///
/// Anchors (best global, best planar/DC, best angular, MPMs) are inserted
/// as mandatory. Then angular family diversity and neighbor expansion fill
/// optional slots. Remaining capacity is filled by raw rough cost order.
pub(super) fn build_luma_shortlist(
    rough: &[ModeCost],
    mpm: [u8; 3],
    policy: &crate::effort::LumaShortlistPolicy,
) -> Vec<u8> {
    let mut out = CandidateSet::new(policy.max_modes as usize);

    // Mandatory anchors.
    if policy.include_best_global {
        if let Some(m) = rough.first().map(|mc| mc.mode) {
            out.insert_mandatory(m);
        }
    }
    if policy.include_best_planar_dc {
        if let Some(m) = best_of_category(rough, |m| m <= 1) {
            out.insert_mandatory(m);
        }
    }
    if policy.include_best_angular {
        if let Some(m) = best_of_category(rough, |m| (2..=34).contains(&m)) {
            out.insert_mandatory(m);
        }
    }
    if policy.include_mpm {
        for &mm in mpm.iter() {
            out.insert_mandatory(mm);
        }
    }

    // Angular family diversity (optional).
    if policy.angular_family_slots > 0 {
        let families = [
            AngularFamily::Horizontal,
            AngularFamily::Diagonal,
            AngularFamily::Vertical,
        ];
        let best_ang = best_of_category(rough, |m| (2..=34).contains(&m));
        let best_family = best_ang.and_then(AngularFamily::classify);
        let mut filled = 0u8;
        for family in &families {
            if filled >= policy.angular_family_slots {
                break;
            }
            if best_family == Some(*family) {
                continue;
            }
            if let Some(m) =
                best_of_category(rough, |m| AngularFamily::classify(m) == Some(*family))
            {
                out.insert_optional(m);
                filled += 1;
            }
        }
    }

    // Neighbor expansion around best angular.
    if policy.angular_neighbor_radius > 0 {
        if let Some(best_ang) = best_of_category(rough, |m| (2..=34).contains(&m)) {
            for n in angular_neighbors(best_ang, policy.angular_neighbor_radius) {
                out.insert_optional(n);
            }
        }
        for &mm in mpm.iter() {
            if (2..=34).contains(&mm) {
                for n in angular_neighbors(mm, policy.angular_neighbor_radius) {
                    out.insert_optional(n);
                }
            }
        }
    }

    // Fill remaining by raw rough cost.
    for mc in rough.iter() {
        out.insert_optional(mc.mode);
    }

    out.into_vec()
}

/// Build the exact-evaluation set from cheap-ranked modes.
///
/// Always includes the cheap winner. When the cheap winner is planar/DC
/// and a rough angular is within the close margin, promotes it. Similarly
/// promotes planar/DC when cheap winner is angular. Fills remaining slots
/// by cheap rank.
fn build_exact_set(
    shortlist: &[u8],
    cheap_ranked: &[CheapMode],
    policy: &crate::effort::ExactPromotionPolicy,
) -> Vec<u8> {
    let mut out = CandidateSet::new(policy.max_exact_modes as usize);

    if cheap_ranked.is_empty() {
        return out.into_vec();
    }

    // Always include cheap winner.
    if policy.include_cheap_winner {
        out.insert_mandatory(cheap_ranked[0].mode);
    }

    let cheap_best_cost = cheap_ranked[0].cost;
    let margin = policy.cheap_close_margin;

    // When cheap winner is planar/DC, promote best angular challenger.
    let cheap_winner = cheap_ranked[0].mode;
    if cheap_winner <= 1 && policy.include_best_rough_angular_if_pd_wins_cheap {
        let best_ang = shortlist
            .iter()
            .filter(|&&m| (2..=34).contains(&m))
            .filter(|&&m| {
                cheap_ranked
                    .iter()
                    .find(|cm| cm.mode == m)
                    .map_or(false, |cm| cm.cost <= cheap_best_cost * margin)
            })
            .min_by_key(|&&m| {
                cheap_ranked
                    .iter()
                    .find(|cm| cm.mode == m)
                    .map(|cm| (cm.cost * 1000.0) as u64)
                    .unwrap_or(u64::MAX)
            })
            .copied();
        if let Some(m) = best_ang {
            out.insert_mandatory(m);
        }
    }

    // When cheap winner is angular, promote best planar/DC.
    if (2..=34).contains(&cheap_winner) && policy.include_best_rough_pd_if_angular_wins_cheap {
        let best_pd = shortlist
            .iter()
            .filter(|&&m| m <= 1)
            .filter(|&&m| {
                cheap_ranked
                    .iter()
                    .find(|cm| cm.mode == m)
                    .map_or(false, |cm| cm.cost <= cheap_best_cost * margin)
            })
            .min_by_key(|&&m| {
                cheap_ranked
                    .iter()
                    .find(|cm| cm.mode == m)
                    .map(|cm| (cm.cost * 1000.0) as u64)
                    .unwrap_or(u64::MAX)
            })
            .copied();
        if let Some(m) = best_pd {
            out.insert_mandatory(m);
        }
    }

    // Fill remaining by cheap rank.
    for cm in cheap_ranked.iter() {
        out.insert_optional(cm.mode);
    }

    out.into_vec()
}

const ALL_INTRA_MODES: [u8; 35] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34,
];

fn angular_config() -> super::angular::AngularExclusionConfig {
    super::env::angular_exclusion_config()
}

fn angular_filtered_modes(
    state: &mut Encoder<'_>,
    x0: u32,
    y0: u32,
    log2_size: u8,
    src: &[u16],
    mpm: [IntraPredMode; 3],
) -> Option<Vec<u8>> {
    let cfg = angular_config();
    if !cfg.enabled() || log2_size < cfg.min_log2_size {
        return None;
    }

    // Diagnostic-only for now: this uses the committed-frame border builder
    // rather than StillSearch's overlay-aware prediction reader. Keep it
    // env-gated until an overlay-aware border builder and BD sweep prove it out.
    let (uf, ft, center, _) = bpg_hevc_decode::hevc::intra::build_reference_borders(
        &state.frame,
        x0,
        y0,
        log2_size,
        0,
        true,
    );
    let exclusion =
        super::angular::angular_exclusion_mask(cfg, src, &uf, &ft, center, log2_size, mpm);
    let mut modes: Vec<u8> = ALL_INTRA_MODES
        .iter()
        .copied()
        .filter(|&m| exclusion.mask.contains(m))
        .collect();
    let before_angular = 33usize;
    let after_angular = modes.iter().filter(|&&m| (2..=34).contains(&m)).count();
    if after_angular < before_angular {
        state.stats.angular_exclusions += 1;
        state.stats.rdo2_angular_exclusion_blocks += 1;
        state.stats.rdo2_angular_modes_before += before_angular as u64;
        state.stats.rdo2_angular_modes_after += after_angular as u64;
        state.stats.rdo2_angular_modes_removed += (before_angular - after_angular) as u64;
        state.stats.rdo2_angular_game_blocks += u64::from(exclusion.game_changed);
        state.stats.rdo2_angular_iame_blocks += u64::from(exclusion.iame_changed);
    }
    if modes.is_empty() {
        modes.extend([0, 1, mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()]);
        modes.sort_unstable();
        modes.dedup();
    }
    Some(modes)
}

/// Record exact-evaluation stats (cost tracking by mode category).
/// Called once per exact trial (hard-quant and RDOQ trial).
fn record_exact_stats(state: &mut Encoder<'_>, mode: u8, tt_cost: f64) {
    let cost_x1000 = (tt_cost * 1000.0) as u64;
    match mode {
        0 => {
            state.stats.luma_exact_planar_tt_cost_x1000 += cost_x1000;
            state.stats.luma_exact_planar_count += 1;
        }
        1 => {
            state.stats.luma_exact_dc_tt_cost_x1000 += cost_x1000;
            state.stats.luma_exact_dc_count += 1;
        }
        _ => {
            state.stats.luma_exact_angular_tt_cost_x1000 += cost_x1000;
            state.stats.luma_exact_angular_count += 1;
        }
    }
}

/// Emit a sampled rough-pass audit line to stderr.
fn emit_rough_audit(
    evidence: &RoughBlockEvidence,
    shortlist: &[u8],
    x0: u32,
    y0: u32,
    size: usize,
    winner_mode: u8,
) {
    const MOD: u64 = 64;
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let ctr = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if ctr % MOD != 0 {
        return;
    }
    let best_pd_cost = evidence
        .rough_costs
        .iter()
        .find(|mc| mc.mode <= 1)
        .map(|mc| mc.cost);
    let best_ang_cost = evidence
        .rough_costs
        .iter()
        .find(|mc| mc.mode >= 2)
        .map(|mc| mc.cost);
    let best_ang_mode = evidence.best_angular;
    let angular_in_shortlist = shortlist.iter().any(|&m| (2..=34).contains(&m));
    let sl_has_best_ang = evidence
        .best_angular
        .map(|bam| shortlist.contains(&bam))
        .unwrap_or(false);
    let angular_winner = (2..=34).contains(&winner_mode);
    let best_ang_mode_str = best_ang_mode.map_or("?".to_string(), |m| m.to_string());
    eprintln!(
        "AUDIT: cu=({},{}) size={} rough_cands={} shortlist={} \
         ang_in_sl={} best_ang={} sl_has_best_ang={} winner={}{} \
         best_pd={:.1} best_ang_cost={:.1}",
        x0,
        y0,
        size,
        evidence.rough_costs.len(),
        shortlist.len(),
        angular_in_shortlist,
        best_ang_mode_str,
        sl_has_best_ang,
        winner_mode,
        if angular_winner { "*" } else { "" },
        best_pd_cost.unwrap_or(f64::INFINITY),
        best_ang_cost.unwrap_or(f64::INFINITY),
    );
}
