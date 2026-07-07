//! Component evaluation kernels.

use std::time::Instant;

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::Contexts;
use crate::contexts::ctx;
use crate::encoder::Encoder;
use crate::primitives::{add_clip_u8, add_clip_u16, ssd_u8, ssd_u16, sub_residual_u8};
use crate::residual::{
    ScanOrder, apply_sign_data_hiding, estimate_residual_bits, estimate_residual_bits_into_nnz,
    estimate_residual_bits_static, get_scan_order,
};
use crate::transform;

use super::arena::CoeffId;
use super::depth::StillSearchDepth;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::plan::PlanBlock;
use super::price::entropy_bits;
use super::source::CtuSourceCache;
use super::workspace::{BlockScratch, SsimRdNorm};

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

/// Quantization for search-stage RD decision sites (NxN PU modes, chroma mode
/// search, cheap luma ranking): x265 rdoq-level-2 applies RDOQ in every RD
/// evaluation. Follows the effort template's per-stage [`TrialQuant`] policy
/// unless `BPG_STILLSEARCH_RDOQ_SEARCH` overrides it globally.
#[inline]
pub(super) fn search_trial_quant(policy: crate::effort::TrialQuant) -> QuantMode {
    let rdoq =
        super::env::rdoq_search_override().unwrap_or(policy == crate::effort::TrialQuant::Rdoq);
    if rdoq {
        QuantMode::RdoqTrial { level: 2 }
    } else {
        QuantMode::HardQuantSearch
    }
}

/// Result of evaluating one transform block (luma or one chroma component).
/// Coefficients live in the CTU coeff arena; the trial holds only a `Copy`
/// handle, not an owned level buffer.
pub(super) struct BlockTrial {
    pub(super) coeff: Option<CoeffId>,
    pub(super) cbf: bool,
    pub(super) frac_bits: u64,
    pub(super) cost: f64,
    pub(super) dist: u64,
    pub(super) rd_frac_bits: u64,
}

impl BlockTrial {
    pub(super) fn into_plan_block(self) -> PlanBlock {
        PlanBlock {
            coeff: self.coeff,
            cbf: self.cbf,
            frac_bits: self.frac_bits,
            dist: self.dist,
            rd_frac_bits: self.rd_frac_bits,
        }
    }
}

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
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
        let chroma_timer = if c_idx != 0 {
            self.workspace.ledger.bump(WorkBucket::ChromaTrial);
            StillSearchLedger::start_timer()
        } else {
            None
        };
        let out = self.eval_component_impl(
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
            true,
        );
        self.workspace
            .ledger
            .finish_timer(WorkBucket::ChromaTrial, chroma_timer);
        out
    }

    /// Commit one selected block outcome's context updates into `price_cur`
    /// (x265 parity: the winning candidate's coded syntax evolves the trial
    /// contexts that later siblings price from). Updates the CBF bin and,
    /// for coded blocks, evolves the residual-syntax context range exactly as
    /// real coding would. Callers roll back via `price_cur` save/restore when
    /// the surrounding alternative loses.
    pub(super) fn commit_component_ctx(
        &mut self,
        cbf: bool,
        levels: &[i16],
        log2_size: u8,
        c_idx: u8,
        trafo_depth: u8,
        scan: ScanOrder,
        sdh_enabled: bool,
    ) {
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let ctxs = &mut self.workspace.price_cur;
        ctxs.models[cbf_ci].update(cbf as u8);
        if cbf {
            debug_assert!(
                !levels.is_empty(),
                "committed coded block must retain levels"
            );
            let _ = estimate_residual_bits(ctxs, levels, log2_size, c_idx, scan, sdh_enabled);
        }
    }

    /// Enter a committed-winner evaluation scope: subsequent `eval_component*`
    /// calls write their selected outcome's context updates into `price_cur`
    /// (with chroma deferred to the CU-level chroma-mode search when that is
    /// enabled). Returns the previous flags for [`Self::end_winner_commit_scope`].
    pub(super) fn begin_winner_commit_scope(&mut self) -> (bool, bool) {
        let prev = (
            self.workspace.commit_ctx,
            self.workspace.commit_ctx_skip_chroma,
        );
        self.workspace.commit_ctx = super::env::ctx_evolve_search();
        self.workspace.commit_ctx_skip_chroma = super::env::chroma_mode_search_enabled();
        prev
    }

    pub(super) fn end_winner_commit_scope(&mut self, prev: (bool, bool)) {
        self.workspace.commit_ctx = prev.0;
        self.workspace.commit_ctx_skip_chroma = prev.1;
    }

    /// Like [`eval_component`] but skips overlay push. Useful for cheap
    /// ranking passes where only the cost matters.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_component_no_overlay(
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
        self.eval_component_impl(
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
            false,
        )
    }

    /// Shared implementation for [`eval_component`] and
    /// [`eval_component_no_overlay`]. When `push_overlay` is false the winning
    /// reconstruction is not stored in the overlay cache.
    #[allow(clippy::too_many_arguments)]
    fn eval_component_impl(
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
        push_overlay: bool,
    ) -> BlockTrial {
        if bit_depth_is_8(state) {
            let out = self.eval_component_8_impl(
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
                push_overlay,
            );
            return out;
        }

        let size = 1usize << log2_size;
        let n = size * size;
        let bit_depth = state.bit_depth;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;
        let commit_ctx =
            self.workspace.commit_ctx && (c_idx == 0 || !self.workspace.commit_ctx_skip_chroma);

        let mut src = std::mem::take(&mut self.workspace.block_scratch.component_src_u16);
        src.resize(n, 0);
        for y in 0..size {
            for x in 0..size {
                src[y * size + x] = self.source.sample(c_idx, x0 + x as u32, y0 + y as u32);
            }
        }

        let mut pred = std::mem::take(&mut self.workspace.block_scratch.component_pred_u16);
        pred.resize(n, 0);
        self.predict_into(state, x0, y0, log2_size, c_idx, mode, &mut pred);

        let mut levels_vec = Vec::new();
        let mut nnz;
        let mut recon_coded: Option<()> = None;
        let dist_zero = ssd_u16(&src, size, &pred, size, size);
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_cur.models[cbf_ci], 0);
        let scale = CabacEstimator::SCALE as f64;
        let ssim_rd = super::env::ssim_rd_enabled();
        let ssim_weight = super::env::ssim_rd_weight();
        let ssim_norm = ssim_rd.then(|| self.ssim_rd_norm(state, c_idx, qp, x0, y0));
        let ssim_zero = if let Some(norm) = ssim_norm {
            ssim_rd_energy_u16(&src, &pred, size, norm, qp, state.bit_depth)
        } else {
            0
        };
        let cost_zero =
            component_rd_cost(dist_zero, ssim_zero, cbf0_bits, lambda, scale, ssim_weight);
        if dist_zero == 0 {
            if push_overlay {
                self.overlay
                    .push_block(c_idx, x0, y0, size as u32, size as u32, &pred);
            }
            if commit_ctx {
                let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
                self.commit_component_ctx(
                    false,
                    &[],
                    log2_size,
                    c_idx,
                    trafo_depth,
                    scan,
                    sdh_enabled,
                );
            }
            self.workspace.block_scratch.component_src_u16 = src;
            self.workspace.block_scratch.component_pred_u16 = pred;
            return BlockTrial {
                coeff: None,
                cbf: false,
                frac_bits: 0,
                cost: cost_zero,
                dist: dist_zero,
                rd_frac_bits: cbf0_bits,
            };
        }

        let mut dist_coded = dist_zero;
        let mut ssim_coded = ssim_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        let mut recon_buf = std::mem::take(&mut self.workspace.block_scratch.component_recon_u16);
        recon_buf.resize(n, 0);
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
                        &self.workspace.price_cur,
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
                    let mut nnz = rdoq.nnz;
                    self.workspace.ledger.bump(WorkBucket::Rdoq);
                    self.workspace.ledger.finish_timer(WorkBucket::Rdoq, timer);
                    let pricing_mode = super::env::rdoq_pricing_mode();
                    if pricing_mode != crate::rdoq::RdoqPricingMode::Frozen {
                        nnz = crate::rdoq::apply_rdoq_pricing_mode(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            bit_depth,
                            is_dst4,
                            scan,
                            lambda,
                            pricing_mode,
                            super::env::rdoq_trellis_beam(),
                            &mut self.workspace.rdoq_scratch,
                        );
                    }
                    if crate::rdoq::rdoq_refine_enabled() {
                        nnz = crate::rdoq::refine_rdoq_levels(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            bit_depth,
                            is_dst4,
                            scan,
                            lambda,
                        );
                    }
                    nnz
                }
                QuantMode::RdoqTrial { level: _ } => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_cur,
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
                    let mut nnz = rdoq.nnz;
                    self.workspace.ledger.bump(WorkBucket::RdoqTrial);
                    self.workspace
                        .ledger
                        .finish_timer(WorkBucket::RdoqTrial, timer);
                    let pricing_mode = super::env::rdoq_pricing_mode();
                    if pricing_mode != crate::rdoq::RdoqPricingMode::Frozen {
                        nnz = crate::rdoq::apply_rdoq_pricing_mode(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            bit_depth,
                            is_dst4,
                            scan,
                            lambda,
                            pricing_mode,
                            super::env::rdoq_trellis_beam(),
                            &mut self.workspace.rdoq_scratch,
                        );
                    }
                    nnz
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
                add_clip_u16(&pred, recon_residual, &mut recon_buf, n, max_sample);
                dist_coded = ssd_u16(&src, size, &recon_buf, size, size);
                if ssim_rd {
                    ssim_coded = ssim_rd_energy_u16(
                        &src,
                        &recon_buf,
                        size,
                        ssim_norm.expect("SSIM-RD norm"),
                        qp,
                        state.bit_depth,
                    );
                }
                levels_vec = levels.clone();
                recon_coded = Some(());
            }
        }

        let cbf1_bits = entropy_bits(&self.workspace.price_cur.models[cbf_ci], 1);

        let coded_min_cost = component_rd_cost(
            dist_coded,
            ssim_coded,
            cbf1_bits,
            lambda,
            scale,
            ssim_weight,
        );
        let coded = if nnz > 0 && residual_pricing == ResidualPricingMode::Skip {
            Some((0, coded_min_cost))
        } else if nnz > 0 && coded_min_cost < cost_zero {
            let timer = StillSearchLedger::start_timer();
            let residual_bits = estimate_residual_bits_into_nnz(
                &self.workspace.price_cur,
                &levels_vec,
                nnz,
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
            let cost = component_rd_cost(
                dist_coded,
                ssim_coded,
                residual_bits + cbf1_bits,
                lambda,
                scale,
                ssim_weight,
            );
            maybe_dump_rdoq_audit(
                &self.workspace.price_cur,
                &levels_vec,
                quant_mode,
                x0,
                y0,
                log2_size,
                c_idx,
                mode.as_u8(),
                qp,
                nnz,
                scan,
                sdh_enabled,
                dist_zero,
                dist_coded,
                residual_bits,
                cbf1_bits,
                cost_zero,
                cost,
            );
            Some((residual_bits, cost))
        } else {
            None
        };

        let out = match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                recon_coded.expect("coded candidate has recon");
                if push_overlay {
                    self.overlay
                        .push_block(c_idx, x0, y0, size as u32, size as u32, &recon_buf);
                }
                let coeff = retain_coeff.then(|| self.workspace.coeffs.push(&levels_vec));
                if commit_ctx {
                    self.commit_component_ctx(
                        true,
                        &levels_vec,
                        log2_size,
                        c_idx,
                        trafo_depth,
                        scan,
                        sdh_enabled,
                    );
                }
                BlockTrial {
                    coeff,
                    cbf: true,
                    frac_bits: residual_bits,
                    cost: cost_coded,
                    dist: dist_coded,
                    rd_frac_bits: residual_bits + cbf1_bits,
                }
            }
            _ => {
                if commit_ctx {
                    let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
                    self.commit_component_ctx(
                        false,
                        &[],
                        log2_size,
                        c_idx,
                        trafo_depth,
                        scan,
                        sdh_enabled,
                    );
                }
                if push_overlay {
                    self.overlay
                        .push_block(c_idx, x0, y0, size as u32, size as u32, &pred);
                }
                BlockTrial {
                    coeff: None,
                    cbf: false,
                    frac_bits: 0,
                    cost: cost_zero,
                    dist: dist_zero,
                    rd_frac_bits: cbf0_bits,
                }
            }
        };
        self.workspace.block_scratch.component_src_u16 = src;
        self.workspace.block_scratch.component_pred_u16 = pred;
        self.workspace.block_scratch.component_recon_u16 = recon_buf;
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_component_8_impl(
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
        push_overlay: bool,
    ) -> BlockTrial {
        let profile = super::env::profile_enabled();
        let size = 1usize << log2_size;
        let n = size * size;

        let mut src = std::mem::take(&mut self.workspace.block_scratch.component_src_u8);
        src.resize(n, 0);
        self.source.sample_block_u8(c_idx, x0, y0, size, &mut src);

        let mut pred = std::mem::take(&mut self.workspace.block_scratch.component_pred_u8);
        pred.resize(n, 0);
        let t_border = profile.then(Instant::now);
        let refs = self.build_intra_refs_for_block(state, x0, y0, log2_size, c_idx);
        if let Some(t) = t_border {
            let ns = t.elapsed().as_nanos() as u64;
            self.workspace.substage.border_ns =
                self.workspace.substage.border_ns.saturating_add(ns);
        }
        let t_predict = profile.then(Instant::now);
        self.predict_exact_from_refs_u8(&refs, log2_size, c_idx, mode, &mut pred);
        if let Some(t) = t_predict {
            let ns = t.elapsed().as_nanos() as u64;
            self.workspace.substage.predict_ns =
                self.workspace.substage.predict_ns.saturating_add(ns);
        }

        let out = self.eval_component_8_from_src_pred(
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
            push_overlay,
            &src,
            &pred,
        );

        self.workspace.block_scratch.component_src_u8 = src;
        self.workspace.block_scratch.component_pred_u8 = pred;
        out
    }

    /// Evaluate one 8-bit component from a precomputed source block and intra
    /// prediction. This is the shared body of [`eval_component_8_impl`] (which
    /// samples + predicts then calls this) and the batched cheap-RDO leaf path
    /// (which builds source + refs once per leaf and calls this once per mode).
    /// Byte-identical to the previous monolithic `eval_component_8_impl`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_component_8_from_src_pred(
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
        push_overlay: bool,
        src: &[u8],
        pred: &[u8],
    ) -> BlockTrial {
        let profile = super::env::profile_enabled();
        self.workspace.substage.calls += u64::from(profile);

        let size = 1usize << log2_size;
        let n = size * size;
        let cat = state.cat;
        let sdh_enabled = state.sign_data_hiding;
        let is_dst4 = c_idx == 0 && log2_size == 2;
        let commit_ctx =
            self.workspace.commit_ctx && (c_idx == 0 || !self.workspace.commit_ctx_skip_chroma);
        debug_assert!(src.len() >= n && pred.len() >= n);

        let mut levels_vec = Vec::new();
        let mut priced_residual_bits = None;
        let mut nnz;
        let mut recon_coded = false;
        let mut recon = std::mem::take(&mut self.workspace.block_scratch.component_recon_u8);
        let rdoq_audit = matches!(
            quant_mode,
            QuantMode::RdoqFinal | QuantMode::RdoqTrial { .. }
        ) && super::env::rdoq_audit_sample(x0, y0, log2_size, c_idx, mode.as_u8());
        let dist_zero = ssd_u8(src, size, pred, size, size);
        let cbf_ci = if c_idx == 0 {
            ctx::CBF_LUMA + if trafo_depth == 0 { 1 } else { 0 }
        } else {
            ctx::CBF_CBCR + trafo_depth as usize
        };
        let cbf0_bits = entropy_bits(&self.workspace.price_cur.models[cbf_ci], 0);
        let cbf1_bits = entropy_bits(&self.workspace.price_cur.models[cbf_ci], 1);
        let scale = CabacEstimator::SCALE as f64;
        let ssim_rd = super::env::ssim_rd_enabled();
        let ssim_weight = super::env::ssim_rd_weight();
        let ssim_norm = ssim_rd.then(|| self.ssim_rd_norm(state, c_idx, qp, x0, y0));
        let ssim_zero = if let Some(norm) = ssim_norm {
            ssim_rd_energy_u8(src, pred, size, norm, qp)
        } else {
            0
        };
        let cost_zero =
            component_rd_cost(dist_zero, ssim_zero, cbf0_bits, lambda, scale, ssim_weight);
        if dist_zero == 0 {
            if push_overlay {
                self.overlay
                    .push_block_u8(c_idx, x0, y0, size as u32, size as u32, pred);
            }
            self.workspace.block_scratch.component_recon_u8 = recon;
            if commit_ctx {
                let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
                self.commit_component_ctx(
                    false,
                    &[],
                    log2_size,
                    c_idx,
                    trafo_depth,
                    scan,
                    sdh_enabled,
                );
            }
            return BlockTrial {
                coeff: None,
                cbf: false,
                frac_bits: 0,
                cost: cost_zero,
                dist: dist_zero,
                rd_frac_bits: cbf0_bits,
            };
        }

        let mut dist_coded = dist_zero;
        let mut ssim_coded = ssim_zero;
        let scan = get_scan_order(log2_size, mode.as_u8(), c_idx, cat);
        {
            let bs: &mut BlockScratch = &mut self.workspace.block_scratch;
            let (residual, coeff, tmp, levels, dequant, recon_residual) =
                scratch_parts_for_log2(bs, log2_size);

            residual.clear();
            residual.resize(n, 0);

            let t_xform = profile.then(Instant::now);
            sub_residual_u8(src, size, pred, size, residual, size);
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
                        &self.workspace.price_cur,
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
                    let mut nnz = rdoq.nnz;
                    self.workspace.ledger.bump(WorkBucket::Rdoq);
                    self.workspace.ledger.finish_timer(WorkBucket::Rdoq, timer);
                    let pricing_mode = super::env::rdoq_pricing_mode();
                    if pricing_mode != crate::rdoq::RdoqPricingMode::Frozen {
                        nnz = crate::rdoq::apply_rdoq_pricing_mode(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            8,
                            is_dst4,
                            scan,
                            lambda,
                            pricing_mode,
                            super::env::rdoq_trellis_beam(),
                            &mut self.workspace.rdoq_scratch,
                        );
                    }
                    if crate::rdoq::rdoq_refine_enabled() {
                        nnz = crate::rdoq::refine_rdoq_levels(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            8,
                            is_dst4,
                            scan,
                            lambda,
                        );
                    }
                    nnz
                }
                QuantMode::RdoqTrial { level: _ } => {
                    let timer = StillSearchLedger::start_timer();
                    let rdoq = crate::rdoq::rdoq_single_scan_into(
                        &self.workspace.price_cur,
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
                    let mut nnz = rdoq.nnz;
                    self.workspace.ledger.bump(WorkBucket::RdoqTrial);
                    self.workspace
                        .ledger
                        .finish_timer(WorkBucket::RdoqTrial, timer);
                    let pricing_mode = super::env::rdoq_pricing_mode();
                    if pricing_mode != crate::rdoq::RdoqPricingMode::Frozen {
                        nnz = crate::rdoq::apply_rdoq_pricing_mode(
                            &self.workspace.price_cur,
                            coeff,
                            residual,
                            levels,
                            log2_size,
                            c_idx,
                            qp,
                            8,
                            is_dst4,
                            scan,
                            lambda,
                            pricing_mode,
                            super::env::rdoq_trellis_beam(),
                            &mut self.workspace.rdoq_scratch,
                        );
                    }
                    nnz
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
                add_clip_u8(pred, recon_residual, &mut recon, n);
                dist_coded = ssd_u8(src, size, &recon, size, size);
                if ssim_rd {
                    ssim_coded =
                        ssim_rd_energy_u8(src, &recon, size, ssim_norm.expect("SSIM-RD norm"), qp);
                }
                if let Some((dx, dy, dlog2, dci)) = super::env::dump_target() {
                    if x0 == dx && y0 == dy && log2_size == dlog2 && c_idx == dci {
                        dump_block_pipeline(
                            x0,
                            y0,
                            log2_size,
                            c_idx,
                            mode.as_u8(),
                            qp,
                            nnz,
                            dist_zero,
                            dist_coded,
                            src,
                            pred,
                            residual,
                            coeff,
                            levels,
                            dequant,
                            recon_residual,
                            &recon,
                            size,
                        );
                    }
                }
                if let Some(t) = t_recon {
                    let ns = t.elapsed().as_nanos() as u64;
                    self.workspace.substage.recon_dist_ns =
                        self.workspace.substage.recon_dist_ns.saturating_add(ns);
                }
                let coded_min_cost = component_rd_cost(
                    dist_coded,
                    ssim_coded,
                    cbf1_bits,
                    lambda,
                    scale,
                    ssim_weight,
                );
                if residual_pricing == ResidualPricingMode::Exact && coded_min_cost < cost_zero {
                    let t_price = profile.then(Instant::now);
                    let residual_bits = if super::env::cheap_bits_static() {
                        crate::residual::estimate_residual_bits_static(
                            &self.workspace.price_cur,
                            levels,
                            log2_size,
                            c_idx,
                            scan,
                            sdh_enabled,
                        )
                    } else {
                        estimate_residual_bits_into_nnz(
                            &self.workspace.price_cur,
                            levels,
                            nnz,
                            log2_size,
                            c_idx,
                            scan,
                            sdh_enabled,
                            &mut self.workspace.price_scratch,
                        )
                    };
                    if let Some(t) = t_price {
                        let ns = t.elapsed().as_nanos() as u64;
                        self.workspace.substage.residual_price_ns =
                            self.workspace.substage.residual_price_ns.saturating_add(ns);
                    }
                    self.workspace.ledger.bump(WorkBucket::ResidualPrice);
                    let rscale = super::env::residual_bits_search_scale();
                    let residual_bits = if rscale != 1.0 {
                        (residual_bits as f64 * rscale) as u64
                    } else {
                        residual_bits
                    };
                    priced_residual_bits = Some(residual_bits);
                }
                if (retain_coeff || rdoq_audit || commit_ctx) && coded_min_cost < cost_zero {
                    levels_vec = levels.clone();
                }
                recon_coded = true;
            }
        }

        let coded_min_cost = component_rd_cost(
            dist_coded,
            ssim_coded,
            cbf1_bits,
            lambda,
            scale,
            ssim_weight,
        );
        let coded = if nnz > 0 && residual_pricing == ResidualPricingMode::Skip {
            Some((0, coded_min_cost))
        } else if nnz > 0 && coded_min_cost < cost_zero {
            let residual_bits = priced_residual_bits.expect("exact residual bits were priced");
            let cost = component_rd_cost(
                dist_coded,
                ssim_coded,
                residual_bits + cbf1_bits,
                lambda,
                scale,
                ssim_weight,
            );
            maybe_dump_rdoq_audit(
                &self.workspace.price_cur,
                &levels_vec,
                quant_mode,
                x0,
                y0,
                log2_size,
                c_idx,
                mode.as_u8(),
                qp,
                nnz,
                scan,
                sdh_enabled,
                dist_zero,
                dist_coded,
                residual_bits,
                cbf1_bits,
                cost_zero,
                cost,
            );
            Some((residual_bits, cost))
        } else {
            None
        };

        let out = match coded {
            Some((residual_bits, cost_coded)) if cost_coded < cost_zero => {
                debug_assert!(recon_coded, "coded candidate has recon");
                if push_overlay {
                    self.overlay
                        .push_block_u8(c_idx, x0, y0, size as u32, size as u32, &recon);
                }
                let coeff = retain_coeff.then(|| self.workspace.coeffs.push(&levels_vec));
                if commit_ctx {
                    self.commit_component_ctx(
                        true,
                        &levels_vec,
                        log2_size,
                        c_idx,
                        trafo_depth,
                        scan,
                        sdh_enabled,
                    );
                }
                BlockTrial {
                    coeff,
                    cbf: true,
                    frac_bits: residual_bits,
                    cost: cost_coded,
                    dist: dist_coded,
                    rd_frac_bits: residual_bits + cbf1_bits,
                }
            }
            _ => {
                if push_overlay {
                    self.overlay
                        .push_block_u8(c_idx, x0, y0, size as u32, size as u32, pred);
                }
                if commit_ctx {
                    self.commit_component_ctx(
                        false,
                        &[],
                        log2_size,
                        c_idx,
                        trafo_depth,
                        scan,
                        sdh_enabled,
                    );
                }
                BlockTrial {
                    coeff: None,
                    cbf: false,
                    frac_bits: 0,
                    cost: cost_zero,
                    dist: dist_zero,
                    rd_frac_bits: cbf0_bits,
                }
            }
        };

        self.workspace.block_scratch.component_recon_u8 = recon;
        out
    }

    fn ssim_rd_norm(
        &mut self,
        state: &Encoder<'_>,
        c_idx: u8,
        qp: i32,
        x0: u32,
        y0: u32,
    ) -> SsimRdNorm {
        let slot = &mut self.workspace.ssim_rd_norm[c_idx as usize];
        if slot.valid && slot.qp == qp {
            return *slot;
        }

        let (sx, sy) = state.plane_shifts(c_idx);
        let comp_w = (64u32).div_ceil(1u32 << sx).max(4);
        let comp_h = (64u32).div_ceil(1u32 << sy).max(4);
        let ox = (x0 / comp_w) * comp_w;
        let oy = (y0 / comp_h) * comp_h;
        let (_, _, shift) = ssim_constants(state.bit_depth);

        let mut z_o = 0u64;
        let mut z_k = 0u64;
        for y in 0..comp_h {
            for x in 0..comp_w {
                let s = self.source.sample(c_idx, ox + x, oy + y) >> shift;
                let ss = u64::from(s).saturating_mul(u64::from(s));
                z_k = z_k.saturating_add(ss);
            }
        }
        for y in (0..comp_h).step_by(4) {
            for x in (0..comp_w).step_by(4) {
                let s = self.source.sample(c_idx, ox + x, oy + y) >> shift;
                z_o = z_o.saturating_add(u64::from(s).saturating_mul(u64::from(s)));
            }
        }
        let blocks4 = ((comp_w >> 2) * (comp_h >> 2)).max(1) as u64;
        let samples = u64::from(comp_w).saturating_mul(u64::from(comp_h));
        let (f_dc_den, f_ac_den) =
            ssim_denominators(z_o, z_k, samples, blocks4, qp, state.bit_depth);
        *slot = SsimRdNorm {
            valid: true,
            qp,
            f_dc_den,
            f_ac_den,
        };
        *slot
    }
}

#[inline]
fn component_rd_cost(
    dist: u64,
    ssim_energy: u64,
    frac_bits: u64,
    lambda: f64,
    scale: f64,
    ssim_weight: f64,
) -> f64 {
    dist as f64 + lambda * frac_bits as f64 / scale + lambda * ssim_weight * ssim_energy as f64
}

#[allow(clippy::too_many_arguments)]
fn maybe_dump_rdoq_audit(
    ctxs: &Contexts,
    levels: &[i16],
    quant_mode: QuantMode,
    x0: u32,
    y0: u32,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    qp: i32,
    nnz: u32,
    scan: crate::residual::ScanOrder,
    sign_data_hiding: bool,
    dist_zero: u64,
    dist_coded: u64,
    exact_bits: u64,
    cbf_bits: u64,
    cost_zero: f64,
    cost_coded: f64,
) {
    let stage = match quant_mode {
        QuantMode::HardQuantSearch => return,
        QuantMode::RdoqFinal => "final",
        QuantMode::RdoqTrial { .. } => "trial",
    };
    if !super::env::rdoq_audit_sample(x0, y0, log2_size, c_idx, mode) {
        return;
    }

    let static_bits =
        estimate_residual_bits_static(ctxs, levels, log2_size, c_idx, scan, sign_data_hiding);
    let abs_sum: u64 = levels.iter().map(|&v| u64::from(v.unsigned_abs())).sum();
    let level_energy: u64 = levels
        .iter()
        .map(|&v| {
            let a = u64::from(v.unsigned_abs());
            a * a
        })
        .sum();
    let ratio = if static_bits == 0 {
        0.0
    } else {
        exact_bits as f64 / static_bits as f64
    };

    eprintln!(
        "RDOQ_AUDIT stage={stage} x={x0} y={y0} size={} c_idx={c_idx} mode={mode} qp={qp} nnz={nnz} abs_sum={abs_sum} level_energy={level_energy} dist_zero={dist_zero} dist_coded={dist_coded} exact_bits={exact_bits} static_bits={static_bits} exact_over_static={ratio:.4} cbf_bits={cbf_bits} coded_wins={} cost_zero={cost_zero:.2} cost_coded={cost_coded:.2}",
        1usize << log2_size,
        cost_coded < cost_zero,
    );
}

#[inline]
fn ssim_constants(bit_depth: u8) -> (u64, u64, u32) {
    let shift = bit_depth.saturating_sub(8) as u32;
    let max_sample = ((1u64 << bit_depth.min(16)) - 1) as f64;
    let c1 = (0.01 * 0.01 * max_sample * max_sample * 64.0 + 0.5) as u64;
    let c2 = (0.03 * 0.03 * max_sample * max_sample * 64.0 * 63.0 + 0.5) as u64;
    (c1, c2, shift)
}

#[inline]
fn ssim_denominators(
    z_o: u64,
    z_k: u64,
    samples: u64,
    blocks4: u64,
    qp: i32,
    bit_depth: u8,
) -> (u64, u64) {
    let (c1, c2, _) = ssim_constants(bit_depth);
    let blocks4 = blocks4.max(1);
    let ac = z_k.saturating_sub(z_o);
    let s = 1.0 + 0.005 * qp as f64;
    let f_dc = (2u64
        .saturating_mul(z_o)
        .saturating_add(samples.saturating_mul(c1)))
        / blocks4;
    let f_ac = (ac.saturating_add((s * ac as f64) as u64).saturating_add(c2)) / blocks4;
    (f_dc.max(1), f_ac.max(1))
}

#[inline]
fn ssim_div(num: u64, den: u64) -> u64 {
    if den == 0 { num } else { num / den }
}

/// x265-shaped SSIM-RD residual energy for the 8-bit path. Unlike AQ, this
/// only changes candidate ranking. The CTU source denominators are cached in
/// the CTU workspace and mirror x265's `Analysis::normFactor`; each candidate
/// then mirrors `Quant::ssimDistortion` for its TU block.
fn ssim_rd_energy_u8(src: &[u8], recon: &[u8], size: usize, norm: SsimRdNorm, qp: i32) -> u64 {
    let n = size * size;
    let blocks4 = ((size >> 2) * (size >> 2)).max(1) as u64;
    let (c1, c2, _shift) = ssim_constants(8);
    let mut ss_dc = 0u64;
    let mut ss_block = 0u64;
    let mut dc_k = 0u64;
    let mut ac_k = 0u64;

    for y in 0..size {
        for x in 0..size {
            let s = src[y * size + x] as i32;
            let e = s - recon[y * size + x] as i32;
            ss_block = ss_block.saturating_add((e * e) as u64);
            ac_k = ac_k.saturating_add((s * s) as u64);
        }
    }
    for y in (0..size).step_by(4) {
        for x in (0..size).step_by(4) {
            let s = src[y * size + x] as i32;
            let e = s - recon[y * size + x] as i32;
            ss_dc = ss_dc.saturating_add((e * e) as u64);
            dc_k = dc_k.saturating_add((s * s) as u64);
        }
    }

    let f_dc_num = (2u64
        .saturating_mul(dc_k)
        .saturating_add((n as u64).saturating_mul(c1)))
        / blocks4;
    let ac = ac_k.saturating_sub(dc_k);
    let s = 1.0 + 0.005 * qp as f64;
    let f_ac_num = (ac.saturating_add((s * ac as f64) as u64).saturating_add(c2)) / blocks4;
    let ss_ac = ss_block.saturating_sub(ss_dc);
    ssim_div(ss_dc.saturating_mul(norm.f_dc_den), f_dc_num.max(1)).saturating_add(ssim_div(
        ss_ac.saturating_mul(norm.f_ac_den),
        f_ac_num.max(1),
    ))
}

fn ssim_rd_energy_u16(
    src: &[u16],
    recon: &[u16],
    size: usize,
    norm: SsimRdNorm,
    qp: i32,
    bit_depth: u8,
) -> u64 {
    let n = size * size;
    let blocks4 = ((size >> 2) * (size >> 2)).max(1) as u64;
    let (c1, c2, shift) = ssim_constants(bit_depth);
    let mut ss_dc = 0u64;
    let mut ss_block = 0u64;
    let mut dc_k = 0u64;
    let mut ac_k = 0u64;

    for y in 0..size {
        for x in 0..size {
            let s = src[y * size + x] as i64;
            let r = recon[y * size + x] as i64;
            let e = s - r;
            ss_block = ss_block.saturating_add((e * e) as u64);
            let sn = (src[y * size + x] as u64) >> shift;
            ac_k = ac_k.saturating_add(sn.saturating_mul(sn));
        }
    }
    for y in (0..size).step_by(4) {
        for x in (0..size).step_by(4) {
            let s = src[y * size + x] as i64;
            let r = recon[y * size + x] as i64;
            let e = s - r;
            ss_dc = ss_dc.saturating_add((e * e) as u64);
            let sn = (src[y * size + x] as u64) >> shift;
            dc_k = dc_k.saturating_add(sn.saturating_mul(sn));
        }
    }

    let f_dc_num = (2u64
        .saturating_mul(dc_k)
        .saturating_add((n as u64).saturating_mul(c1)))
        / blocks4;
    let ac = ac_k.saturating_sub(dc_k);
    let s = 1.0 + 0.005 * qp as f64;
    let f_ac_num = (ac.saturating_add((s * ac as f64) as u64).saturating_add(c2)) / blocks4;
    let ss_ac = ss_block.saturating_sub(ss_dc);
    ssim_div(ss_dc.saturating_mul(norm.f_dc_den), f_dc_num.max(1)).saturating_add(ssim_div(
        ss_ac.saturating_mul(norm.f_ac_den),
        f_ac_num.max(1),
    ))
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

/// Forensic single-block pipeline dump (env-gated by `BPG_STILL_DUMP`).
///
/// Prints, for one matched luma/chroma TU and one candidate mode, every stage
/// of the residual pipeline so it can be diffed coefficient-by-coefficient
/// against an x265 reference dump. All stages are in the transform's scan-
/// independent raster (row-major) layout. Fires once per matching candidate
/// mode (the search tries several per block), so each block produces several
/// labelled records; the winning mode is the one whose `cost` the search keeps.
#[allow(clippy::too_many_arguments)]
fn dump_block_pipeline(
    x0: u32,
    y0: u32,
    _log2_size: u8,
    c_idx: u8,
    mode: u8,
    qp: i32,
    nnz: u32,
    dist_zero: u64,
    dist_coded: u64,
    src: &[u8],
    pred: &[u8],
    residual: &[i16],
    coeff: &[i16],
    levels: &[i16],
    dequant: &[i16],
    inv_residual: &[i16],
    recon: &[u8],
    size: usize,
) {
    use std::fmt::Write as _;
    let n = size * size;
    let mut s = String::with_capacity(4096);
    let _ = writeln!(
        s,
        "=== STILL265 BLOCK DUMP x={x0} y={y0} size={size} c_idx={c_idx} mode={mode} qp={qp} nnz={nnz} dist_zero={dist_zero} dist_coded={dist_coded} ==="
    );
    let grid_i64 = |s: &mut String, label: &str, get: &dyn Fn(usize) -> i64| {
        let _ = writeln!(s, "[{label}]");
        for r in 0..size {
            for c in 0..size {
                let _ = write!(s, "{:6}", get(r * size + c));
            }
            let _ = writeln!(s);
        }
    };
    grid_i64(&mut s, "SRC", &|i| src[i] as i64);
    grid_i64(&mut s, "PRED", &|i| pred[i] as i64);
    grid_i64(&mut s, "RESIDUAL", &|i| {
        residual.get(i).copied().unwrap_or(0) as i64
    });
    grid_i64(&mut s, "FWD_COEFF", &|i| {
        coeff.get(i).copied().unwrap_or(0) as i64
    });
    grid_i64(&mut s, "LEVELS", &|i| {
        levels.get(i).copied().unwrap_or(0) as i64
    });
    grid_i64(&mut s, "DEQUANT", &|i| {
        dequant.get(i).copied().unwrap_or(0) as i64
    });
    grid_i64(&mut s, "INV_RESIDUAL", &|i| {
        inv_residual.get(i).copied().unwrap_or(0) as i64
    });
    grid_i64(&mut s, "RECON", &|i| recon[i] as i64);
    grid_i64(&mut s, "RECON_MINUS_SRC", &|i| {
        recon[i] as i64 - src[i] as i64
    });
    // Summary scalars for quick comparison.
    let res_energy: i64 = (0..n)
        .map(|i| {
            let v = residual.get(i).copied().unwrap_or(0) as i64;
            v * v
        })
        .sum();
    let coeff_energy: i64 = (0..n)
        .map(|i| {
            let v = coeff.get(i).copied().unwrap_or(0) as i64;
            v * v
        })
        .sum();
    let abs_level: i64 = (0..n)
        .map(|i| levels.get(i).copied().unwrap_or(0).unsigned_abs() as i64)
        .sum();
    let _ = writeln!(
        s,
        "[SUMMARY] residual_energy={res_energy} fwd_coeff_energy={coeff_energy} abs_level_sum={abs_level}"
    );
    eprint!("{s}");
}
