//! Component evaluation kernels.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::ctx;
use crate::encoder::Encoder;
use crate::primitives::{ssd_u8, ssd_u16};
use crate::residual::{apply_sign_data_hiding, estimate_residual_bits_into, get_scan_order};
use crate::transform;

use super::arena::CoeffId;
use super::depth::StillSearchDepth;
use super::ledger::WorkBucket;
use super::overlay::OverlayCache;
use super::plan::PlanBlock;
use super::price::entropy_bits;
use super::source::CtuSourceCache;
use super::workspace::BlockScratch;

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
    ) -> BlockTrial {
        if bit_depth_is_8(state) {
            return self.eval_component_8(
                state,
                x0,
                y0,
                log2_size,
                c_idx,
                mode,
                qp,
                trafo_depth,
                lambda,
            );
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
        let mut dist_coded = dist_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);
            for i in 0..n {
                residual[i] =
                    (src[i] as i32 - pred[i] as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }

            transform::forward_transform_into(residual, log2_size, is_dst4, bit_depth, coeff, tmp);
            nnz = transform::quantize_into(coeff, log2_size, qp, bit_depth, levels);
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
                let max_sample = (1i32 << bit_depth) - 1;
                let mut recon = vec![0u16; n];
                for i in 0..n {
                    recon[i] =
                        (pred[i] as i32 + recon_residual[i] as i32).clamp(0, max_sample) as u16;
                }
                dist_coded = ssd_u16(&src, size, &recon, size, size);
                levels_vec = levels.clone();
                recon_coded = Some(recon);
            }
        }

        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 0);
        let cbf1_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 1);

        let scale = CabacEstimator::SCALE as f64;
        let cost_zero = dist_zero as f64 + lambda * cbf0_bits as f64 / scale;

        let coded = if nnz > 0 {
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
            let cost = dist_coded as f64 + lambda * (residual_bits + cbf1_bits) as f64 / scale;
            Some((residual_bits, cost))
        } else {
            None
        };

        match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                let recon = recon_coded.expect("coded candidate has recon");
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &recon);
                let coeff = Some(self.workspace.coeffs.push(&levels_vec));
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
        }
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
    ) -> BlockTrial {
        let size = 1usize << log2_size;
        let n = size * size;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;

        let mut src = vec![0u8; n];
        for y in 0..size {
            for x in 0..size {
                src[y * size + x] = self
                    .source
                    .sample(c_idx, x0 + x as u32, y0 + y as u32)
                    .min(u8::MAX as u16) as u8;
            }
        }

        let mut pred = vec![0u8; n];
        let mut pred_u16 = Vec::with_capacity(n);
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

        let mut levels_vec = Vec::new();
        let mut nnz;
        let mut recon_coded: Option<Vec<u8>> = None;
        let dist_zero = ssd_u8(&src, size, &pred, size, size);
        let mut dist_coded = dist_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);
            for i in 0..n {
                residual[i] = src[i] as i16 - pred[i] as i16;
            }

            transform::forward_transform_into(residual, log2_size, is_dst4, 8, coeff, tmp);
            nnz = transform::quantize_into(coeff, log2_size, qp, 8, levels);
            if sdh_enabled && nnz > 0 {
                let (scale, qbits) = transform::quant_params(log2_size, qp, 8);
                nnz = apply_sign_data_hiding(levels, coeff, log2_size, scan, scale, qbits, nnz);
            }

            if nnz > 0 {
                transform::reconstruct_residual_into(
                    levels,
                    log2_size,
                    qp,
                    8,
                    is_dst4,
                    dequant,
                    recon_residual,
                );
                let mut recon = vec![0u8; n];
                for i in 0..n {
                    recon[i] = (pred[i] as i32 + recon_residual[i] as i32).clamp(0, 255) as u8;
                }
                dist_coded = ssd_u8(&src, size, &recon, size, size);
                levels_vec = levels.clone();
                recon_coded = Some(recon);
            }
        }

        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 0);
        let cbf1_bits = entropy_bits(&self.workspace.price_base.models[cbf_ci], 1);

        let scale = CabacEstimator::SCALE as f64;
        let cost_zero = dist_zero as f64 + lambda * cbf0_bits as f64 / scale;

        let coded = if nnz > 0 {
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
            let cost = dist_coded as f64 + lambda * (residual_bits + cbf1_bits) as f64 / scale;
            Some((residual_bits, cost))
        } else {
            None
        };

        match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                let recon = recon_coded.expect("coded candidate has recon");
                self.overlay
                    .push_block_u8(c_idx, x0, y0, size as u32, size as u32, &recon);
                let coeff = Some(self.workspace.coeffs.push(&levels_vec));
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
        }
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
