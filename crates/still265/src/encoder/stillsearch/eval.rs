//! Component evaluation kernels.

use std::time::Instant;

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::encoder::Encoder;
use crate::primitives::{add_clip_u8, add_clip_u16, ssd_u8, ssd_u16, sub_residual_u8};
use crate::residual::{apply_sign_data_hiding, estimate_residual_bits_into, get_scan_order};
use crate::transform;

use super::arena::CoeffId;
use super::depth::StillSearchDepth;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::overlay::OverlayCache;
use super::plan::PlanBlock;
use super::price::entropy_bits;
use super::source::CtuSourceCache;
use super::workspace::BlockScratch;

/// Quantization mode for the shared component evaluator. Search/screening uses
/// [`QuantMode::HardQuantSearch`] (the green baseline); the winner-only
/// RDOQ finalizer uses [`QuantMode::RdoqFinal`]; and analysis-stage RDOQ
/// trials (for close candidates in Slow/Placebo) use [`QuantMode::RdoqTrial`].
/// All three modes share the same prediction, transform, sign-data-hiding, recon,
/// and residual-pricing path — only the quantization step differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QuantMode {
    HardQuantSearch,
    RdoqFinal,
    RdoqTrial { level: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResidualPricingMode {
    Exact,
    Skip,
}

/// Result of evaluating one transform block (luma or one chroma component).
/// Coefficients live in the CTU coeff arena; the trial holds only a `Copy`
/// handle, not an owned level buffer.
pub(super) struct BlockTrial {
    pub(super) coeff: Option<CoeffId>,
    pub(super) cbf: bool,
    pub(super) frac_bits: u64,
    pub(super) cost: f64,
}

impl BlockTrial {
    pub(super) fn into_plan_block(self) -> PlanBlock {
        PlanBlock {
            coeff: self.coeff,
            cbf: self.cbf,
            frac_bits: self.frac_bits,
        }
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// Evaluate a single transform block (one component) without mutating the
    /// shared frame. Predicts into local scratch (overlay-first reference
    /// reads), prices the coded and null-CBF candidates, and pushes the winning
    /// recon to the overlay. Returns the retained levels/cbf and RD cost.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_component(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: IntraPredMode,
        qp: i32,
        trafo_depth: u8,
        lambda: f64,
        quant_mode: QuantMode,
        residual_pricing: ResidualPricingMode,
        retain_coeff: bool,
    ) -> BlockTrial {
        // Chroma block evaluations (Cb/Cr) are otherwise invisible in the
        // ledger: luma evals are attributed to RoughLuma/LumaExact/Tu*/Nxn*,
        // but chroma has no rough/exact search of its own (DM-only). Count
        // every chroma component evaluation here so the profile accounts for
        // chroma transform/quant/pricing work. Counted once per call: the
        // 8-bit path below is only reached through this function.
        let chroma_timer = if c_idx != 0 {
            self.workspace.ledger.bump(WorkBucket::ChromaTrial);
            StillSearchLedger::start_timer()
        } else {
            None
        };
        if bit_depth_is_8(state) {
            let out = self.eval_component_8(
                state,
                x0,
                y0,
                log2_size,
                c_idx,
                mode,
                qp,
                trafo_depth,
                lambda,
                quant_mode,
                residual_pricing,
                retain_coeff,
            );
            self.workspace
                .ledger
                .finish_timer(WorkBucket::ChromaTrial, chroma_timer);
            return out;
        }

        let size = 1usize << log2_size;
        let n = size * size;
        let bit_depth = state.bit_depth;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;

        let mut src = vec![0u16; n];
        for y in 0..size {
            for x in 0..size {
                src[y * size + x] = self.source.sample(c_idx, x0 + x as u32, y0 + y as u32);
            }
        }

        let mut pred = vec![0u16; n];
        self.predict_into(state, x0, y0, log2_size, c_idx, mode, &mut pred);

        let mut levels_vec = Vec::new();
        let mut nnz;
        let mut recon_coded: Option<Vec<u16>> = None;
        let dist_zero = ssd_u16(&src, size, &pred, size, size);
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 0);
        let scale = CabacEstimator::SCALE as f64;
        let cost_zero = dist_zero as f64 + lambda * cbf0_bits as f64 / scale;
        if dist_zero == 0 {
            self.overlay
                .push_block(c_idx, x0, y0, size as u32, size as u32, &pred);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::ChromaTrial, chroma_timer);
            return BlockTrial {
                coeff: None,
                cbf: false,
                frac_bits: 0,
                cost: cost_zero,
            };
        }

        let mut dist_coded = dist_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);
            crate::primitives::sub_residual(&src, size, &pred, size, residual, size);

            transform::forward_transform_into(residual, log2_size, is_dst4, bit_depth, coeff, tmp);
            nnz = match quant_mode {
                QuantMode::HardQuantSearch => {
                    transform::quantize_into(coeff, log2_size, qp, bit_depth, levels)
                }
                QuantMode::RdoqFinal => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_base,
                        coeff,
                        log2_size,
                        c_idx,
                        qp,
                        bit_depth,
                        scan,
                        lambda,
                        true,
                        &mut self.workspace.rdoq_scratch,
                    );
                    levels.clear();
                    levels.extend_from_slice(rdoq.levels);
                    self.workspace.ledger.bump(WorkBucket::Rdoq);
                    self.workspace.ledger.finish_timer(WorkBucket::Rdoq, timer);
                    rdoq.nnz
                }
                QuantMode::RdoqTrial { level: _ } => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_base,
                        coeff,
                        log2_size,
                        c_idx,
                        qp,
                        bit_depth,
                        scan,
                        lambda,
                        true,
                        &mut self.workspace.rdoq_scratch,
                    );
                    levels.clear();
                    levels.extend_from_slice(rdoq.levels);
                    self.workspace.ledger.bump(WorkBucket::RdoqTrial);
                    self.workspace
                        .ledger
                        .finish_timer(WorkBucket::RdoqTrial, timer);
                    rdoq.nnz
                }
            };
            if sdh_enabled && nnz > 0 {
                let (scale, qbits) = transform::quant_params(log2_size, qp, bit_depth);
                nnz = apply_sign_data_hiding(levels, coeff, log2_size, scan, scale, qbits, nnz);
            }

            if nnz > 0 {
                transform::reconstruct_residual_into(
                    levels,
                    log2_size,
                    qp,
                    bit_depth,
                    is_dst4,
                    dequant,
                    recon_residual,
                );
                let max_sample = ((1i32 << bit_depth) - 1) as u16;
                let mut recon = vec![0u16; n];
                add_clip_u16(&pred, recon_residual, &mut recon, n, max_sample);
                dist_coded = ssd_u16(&src, size, &recon, size, size);
                levels_vec = levels.clone();
                recon_coded = Some(recon);
            }
        }

        let cbf1_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 1);

        let coded_min_cost = dist_coded as f64 + lambda * cbf1_bits as f64 / scale;
        let coded = if nnz > 0 && residual_pricing == ResidualPricingMode::Skip {
            Some((0, coded_min_cost))
        } else if nnz > 0 && coded_min_cost < cost_zero {
            let timer = StillSearchLedger::start_timer();
            let residual_bits = estimate_residual_bits_into(
                &self.workspace.price_base,
                &levels_vec,
                log2_size,
                c_idx,
                scan,
                sdh_enabled,
                &mut self.workspace.price_scratch,
            );
            self.workspace.ledger.bump(WorkBucket::ResidualPrice);
            self.workspace
                .ledger
                .finish_timer(WorkBucket::ResidualPrice, timer);
            let cost = dist_coded as f64 + lambda * (residual_bits + cbf1_bits) as f64 / scale;
            Some((residual_bits, cost))
        } else {
            None
        };

        let out = match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                let recon = recon_coded.expect("coded candidate has recon");
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &recon);
                let coeff = retain_coeff.then(|| self.workspace.coeffs.push(&levels_vec));
                BlockTrial {
                    coeff,
                    cbf: true,
                    frac_bits: residual_bits,
                    cost: cost_coded,
                }
            }
            _ => {
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &pred);
                BlockTrial {
                    coeff: None,
                    cbf: false,
                    frac_bits: 0,
                    cost: cost_zero,
                }
            }
        };
        self.workspace
            .ledger
            .finish_timer(WorkBucket::ChromaTrial, chroma_timer);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_component_8(
        &mut self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: IntraPredMode,
        qp: i32,
        trafo_depth: u8,
        lambda: f64,
        quant_mode: QuantMode,
        residual_pricing: ResidualPricingMode,
        retain_coeff: bool,
    ) -> BlockTrial {
        let profile = super::env::profile_enabled();
        self.workspace.substage.calls += u64::from(profile);

        let size = 1usize << log2_size;
        let n = size * size;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;

        let mut src = std::mem::take(&mut self.workspace.block_scratch.component_src_u8);
        src.resize(n, 0);
        self.source.sample_block_u8(c_idx, x0, y0, size, &mut src);

        let mut pred = std::mem::take(&mut self.workspace.block_scratch.component_pred_u8);
        pred.resize(n, 0);
        let mut pred_u16 = std::mem::take(&mut self.workspace.block_scratch.component_pred_tmp_u16);

        let t_predict = profile.then(Instant::now);
        self.predict_into_u8(
            state,
            x0,
            y0,
            log2_size,
            c_idx,
            mode,
            &mut pred,
            &mut pred_u16,
        );
        if let Some(t) = t_predict {
            let ns = t.elapsed().as_nanos() as u64;
            self.workspace.substage.predict_ns =
                self.workspace.substage.predict_ns.saturating_add(ns);
        }

        let mut levels_vec = Vec::new();
        let mut nnz;
        let mut recon_coded = false;
        let mut recon = std::mem::take(&mut self.workspace.block_scratch.component_recon_u8);
        let dist_zero = ssd_u8(&src, size, &pred, size, size);
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 0);
        let scale = CabacEstimator::SCALE as f64;
        let cost_zero = dist_zero as f64 + lambda * cbf0_bits as f64 / scale;
        if dist_zero == 0 {
            self.overlay
                .push_block_u8(c_idx, x0, y0, size as u32, size as u32, &pred);
            self.workspace.block_scratch.component_src_u8 = src;
            self.workspace.block_scratch.component_pred_u8 = pred;
            self.workspace.block_scratch.component_recon_u8 = recon;
            self.workspace.block_scratch.component_pred_tmp_u16 = pred_u16;
            return BlockTrial {
                coeff: None,
                cbf: false,
                frac_bits: 0,
                cost: cost_zero,
            };
        }

        let mut dist_coded = dist_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);

            let t_xform = profile.then(Instant::now);
            sub_residual_u8(&src, size, &pred, size, residual, size);
            transform::forward_transform_into(residual, log2_size, is_dst4, 8, coeff, tmp);
            if let Some(t) = t_xform {
                let ns = t.elapsed().as_nanos() as u64;
                self.workspace.substage.forward_xform_ns =
                    self.workspace.substage.forward_xform_ns.saturating_add(ns);
            }

            let t_quant = profile.then(Instant::now);
            nnz = match quant_mode {
                QuantMode::HardQuantSearch => {
                    transform::quantize_into(coeff, log2_size, qp, 8, levels)
                }
                QuantMode::RdoqFinal => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_base,
                        coeff,
                        log2_size,
                        c_idx,
                        qp,
                        8,
                        scan,
                        lambda,
                        true,
                        &mut self.workspace.rdoq_scratch,
                    );
                    levels.clear();
                    levels.extend_from_slice(rdoq.levels);
                    self.workspace.ledger.bump(WorkBucket::Rdoq);
                    self.workspace.ledger.finish_timer(WorkBucket::Rdoq, timer);
                    rdoq.nnz
                }
                QuantMode::RdoqTrial { level: _ } => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_base,
                        coeff,
                        log2_size,
                        c_idx,
                        qp,
                        8,
                        scan,
                        lambda,
                        true,
                        &mut self.workspace.rdoq_scratch,
                    );
                    levels.clear();
                    levels.extend_from_slice(rdoq.levels);
                    self.workspace.ledger.bump(WorkBucket::RdoqTrial);
                    self.workspace
                        .ledger
                        .finish_timer(WorkBucket::RdoqTrial, timer);
                    rdoq.nnz
                }
            };
            if sdh_enabled && nnz > 0 {
                let (scale, qbits) = transform::quant_params(log2_size, qp, 8);
                nnz = apply_sign_data_hiding(levels, coeff, log2_size, scan, scale, qbits, nnz);
            }
            if let Some(t) = t_quant {
                let ns = t.elapsed().as_nanos() as u64;
                self.workspace.substage.quant_ns =
                    self.workspace.substage.quant_ns.saturating_add(ns);
            }

            if nnz > 0 {
                let t_recon = profile.then(Instant::now);
                transform::reconstruct_residual_into(
                    levels,
                    log2_size,
                    qp,
                    8,
                    is_dst4,
                    dequant,
                    recon_residual,
                );
                recon.clear();
                recon.resize(n, 0);
                add_clip_u8(&pred, recon_residual, &mut recon, n);
                dist_coded = ssd_u8(&src, size, &recon, size, size);
                if let Some(t) = t_recon {
                    let ns = t.elapsed().as_nanos() as u64;
                    self.workspace.substage.recon_dist_ns =
                        self.workspace.substage.recon_dist_ns.saturating_add(ns);
                }
                // Clone levels only when we will actually use them (residual
                // pricing or coeff retention). Skip-pricing + discard covers
                // most cheap-pass evaluations.
                if retain_coeff || residual_pricing == ResidualPricingMode::Exact {
                    levels_vec = levels.clone();
                }
                recon_coded = true;
            }
        }

        let cbf1_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 1);

        let coded_min_cost = dist_coded as f64 + lambda * cbf1_bits as f64 / scale;
        let coded = if nnz > 0 && residual_pricing == ResidualPricingMode::Skip {
            Some((0, coded_min_cost))
        } else if nnz > 0 && coded_min_cost < cost_zero {
            let t_price = profile.then(Instant::now);
            let residual_bits = estimate_residual_bits_into(
                &self.workspace.price_base,
                &levels_vec,
                log2_size,
                c_idx,
                scan,
                sdh_enabled,
                &mut self.workspace.price_scratch,
            );
            if let Some(t) = t_price {
                let ns = t.elapsed().as_nanos() as u64;
                self.workspace.substage.residual_price_ns =
                    self.workspace.substage.residual_price_ns.saturating_add(ns);
            }
            self.workspace.ledger.bump(WorkBucket::ResidualPrice);
            let cost = dist_coded as f64 + lambda * (residual_bits + cbf1_bits) as f64 / scale;
            Some((residual_bits, cost))
        } else {
            None
        };

        let out = match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                debug_assert!(recon_coded, "coded candidate has recon");
                self.overlay
                    .push_block_u8(c_idx, x0, y0, size as u32, size as u32, &recon);
                let coeff = retain_coeff.then(|| self.workspace.coeffs.push(&levels_vec));
                BlockTrial {
                    coeff,
                    cbf: true,
                    frac_bits: residual_bits,
                    cost: cost_coded,
                }
            }
            _ => {
                self.overlay
                    .push_block_u8(c_idx, x0, y0, size as u32, size as u32, &pred);
                BlockTrial {
                    coeff: None,
                    cbf: false,
                    frac_bits: 0,
                    cost: cost_zero,
                }
            }
        };

        self.workspace.block_scratch.component_src_u8 = src;
        self.workspace.block_scratch.component_pred_u8 = pred;
        self.workspace.block_scratch.component_recon_u8 = recon;
        self.workspace.block_scratch.component_pred_tmp_u16 = pred_u16;
        out
    }
}

#[inline]
fn bit_depth_is_8(state: &Encoder<'_>) -> bool {
    state.bit_depth == 8
}

type BlockScratchParts<'a> = (
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
    &'a mut Vec<i16>,
);

fn scratch_parts_for_log2(scratch: &mut BlockScratch, log2_size: u8) -> BlockScratchParts<'_> {
    match log2_size {
        2 => (
            &mut scratch.residual_i16_4x4,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        3 => (
            &mut scratch.residual_i16_8x8,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        4 => (
            &mut scratch.residual_i16_16x16,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
        _ => (
            &mut scratch.residual_i16_32x32,
            &mut scratch.coeff_i16,
            &mut scratch.transform_tmp_i16,
            &mut scratch.levels_i16,
            &mut scratch.dequant_coeff_i16,
            &mut scratch.recon_residual_i16,
        ),
    }
}
