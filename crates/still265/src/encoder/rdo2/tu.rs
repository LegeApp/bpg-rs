//! Staged transform-tree (TU) decision for `rdo2` (slice 1).
//!
//! Under construction: this file currently provides the non-committing block
//! evaluator [`Encoder::rdo2_eval_leaf_block`], the central primitive the staged
//! `analyze_tt` will build on. It mirrors `code_block_internal` kernel-for-kernel
//! so a `Final`+commit evaluation is byte-identical to the legacy path, but a
//! cheap/exact *trial* computes clipped pixel-domain distortion from scratch
//! buffers and does not write the reconstructed frame.
#![allow(dead_code)]

use bpg_hevc_decode::hevc::intra::{predict_intra_into, predict_intra_into_with_reader};
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::contexts::Contexts;
use crate::primitives;
use crate::rdoq;
use crate::residual::get_scan_order;
use crate::trace::{WorkBucket, WorkSample};
use crate::transform;

use crate::effort::{BlockSearchBudget, SplitSearch};

use super::super::types::{
    chroma_pred_mode, chroma_tb_geom, CodedBlock, LeafTu, ParentChromaTu, Tt, MAX_TB_LOG2,
};
use super::policy::{close, EvalKind, EvalPolicy, RdoqPolicy, ResidualBitPolicy};
use super::residual::ResidualPricer;

/// Result of evaluating a single transform block.
pub(in crate::encoder) struct LeafEval {
    pub(in crate::encoder) distortion: u64,
    pub(in crate::encoder) frac_bits: u64,
    pub(in crate::encoder) cost: f64,
    pub(in crate::encoder) coded: CodedBlock,
}

#[derive(Clone)]
struct TrialReconPlane {
    c_idx: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<u16>,
}

#[derive(Clone)]
struct TrialReconRegion {
    planes: Vec<TrialReconPlane>,
}

impl TrialReconRegion {
    fn new(enc: &super::super::Encoder<'_>, x0: u32, y0: u32, log2_size: u8) -> Self {
        let size = 1usize << log2_size;
        let mut region = Self {
            planes: Vec::with_capacity(3),
        };
        region.push_plane(enc, 0, x0, y0, size, size);
        if let Some((cx, cy, clog2, count)) = chroma_tb_geom(enc.cat, x0, y0, log2_size) {
            let csize = 1usize << clog2;
            region.push_plane(enc, 1, cx, cy, csize, csize * count as usize);
            region.push_plane(enc, 2, cx, cy, csize, csize * count as usize);
        }
        region
    }

    fn push_plane(
        &mut self,
        enc: &super::super::Encoder<'_>,
        c_idx: u8,
        x: u32,
        y: u32,
        width: usize,
        height: usize,
    ) {
        let (plane_width, plane_height) = enc.frame_plane_dims(c_idx);
        let x = (x as usize).min(plane_width);
        let y = (y as usize).min(plane_height);
        let width = width.min(plane_width.saturating_sub(x));
        let height = height.min(plane_height.saturating_sub(y));
        let (plane, frame_stride) = enc.frame.plane(c_idx);
        let mut data = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = (y + row) * frame_stride + x;
            data.extend_from_slice(&plane[start..start + width]);
        }
        self.planes.push(TrialReconPlane {
            c_idx,
            x,
            y,
            width,
            height,
            stride: width,
            data,
        });
    }

    fn plane(&self, c_idx: u8) -> Option<&TrialReconPlane> {
        self.planes.iter().find(|p| p.c_idx == c_idx)
    }

    fn plane_mut(&mut self, c_idx: u8) -> Option<&mut TrialReconPlane> {
        self.planes.iter_mut().find(|p| p.c_idx == c_idx)
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        let x = x as usize;
        let y = y as usize;
        let p = self.plane(c_idx)?;
        if x < p.x || y < p.y || x >= p.x + p.width || y >= p.y + p.height {
            return None;
        }
        let idx = (y - p.y) * p.stride + (x - p.x);
        p.data.get(idx).copied()
    }

    fn write_block(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        log2_size: u8,
        pred: &[u16],
        recon_residual: &[i16],
        cbf: bool,
        max_val: i32,
    ) {
        let size = 1usize << log2_size;
        let Some(p) = self.plane_mut(c_idx) else {
            return;
        };
        let x = x as usize;
        let y = y as usize;
        for j in 0..size {
            for i in 0..size {
                let px = x + i;
                let py = y + j;
                if px < p.x || py < p.y || px >= p.x + p.width || py >= p.y + p.height {
                    continue;
                }
                let src_idx = j * size + i;
                let dst_idx = (py - p.y) * p.stride + (px - p.x);
                let residual = if cbf {
                    recon_residual[src_idx] as i32
                } else {
                    0
                };
                p.data[dst_idx] = (pred[src_idx] as i32 + residual).clamp(0, max_val) as u16;
            }
        }
    }
}

enum EvalLevels<'a> {
    Owned(Vec<i16>),
    Borrowed(&'a mut [i16]),
}

impl EvalLevels<'_> {
    fn as_slice(&self) -> &[i16] {
        match self {
            Self::Owned(levels) => levels,
            Self::Borrowed(levels) => levels,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [i16] {
        match self {
            Self::Owned(levels) => levels,
            Self::Borrowed(levels) => levels,
        }
    }
}

impl<'a> super::super::Encoder<'a> {
    /// Predict → residual → transform → quantise (per `policy`) → price, without
    /// writing the reconstructed frame unless `policy.commit`. Distortion is
    /// computed as clipped pixel-domain SSE from scratch buffers, which matches
    /// the legacy committed path without frame readback. When `policy.commit`,
    /// the chosen reconstruction `clamp(pred + recon_residual)` is written into
    /// the frame once.
    pub(in crate::encoder) fn rdo2_eval_leaf_block(
        &mut self,
        ctxs: &Contexts,
        x: u32,
        y: u32,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        qp: i32,
        policy: EvalPolicy,
    ) -> LeafEval {
        self.rdo2_eval_leaf_block_in_region(ctxs, x, y, log2_size, c_idx, mode, qp, policy, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn rdo2_eval_leaf_block_in_region(
        &mut self,
        ctxs: &Contexts,
        x: u32,
        y: u32,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        qp: i32,
        policy: EvalPolicy,
        mut trial_region: Option<&mut TrialReconRegion>,
    ) -> LeafEval {
        let work_start = self.trace.enabled.then(std::time::Instant::now);
        let size = 1usize << log2_size;
        let pred_mode = IntraPredMode::from_u8(mode).unwrap_or(IntraPredMode::Dc);
        let is_dst = log2_size == 2 && c_idx == 0;

        let mut sc = std::mem::take(&mut self.search_scratch);
        sc.pred.clear();
        sc.pred.resize(size * size, 0);
        sc.residual.clear();
        sc.residual.resize(size * size, 0);

        // Prediction reads committed neighbours from the frame, writes scratch.
        let tp = self.prof.on.then(std::time::Instant::now);
        if let Some(region) = trial_region.as_ref() {
            predict_intra_into_with_reader(
                &self.frame,
                x,
                y,
                log2_size,
                pred_mode,
                c_idx,
                true,
                &mut sc.pred,
                size,
                |rc_idx, rx, ry| region.sample(rc_idx, rx, ry),
            );
        } else {
            predict_intra_into(
                &self.frame,
                x,
                y,
                log2_size,
                pred_mode,
                c_idx,
                true,
                &mut sc.pred,
                size,
            );
        }
        if let Some(tp) = tp {
            self.prof.eval_predict += tp.elapsed();
        }

        // residual = src - pred
        {
            let (src_plane, src_stride, sw, sh) = self.src_plane(c_idx);
            if x + size as u32 <= sw && y + size as u32 <= sh {
                let src = &src_plane[y as usize * src_stride + x as usize..];
                primitives::sub_residual(src, src_stride, &sc.pred, size, &mut sc.residual, size);
            } else {
                for j in 0..size {
                    for i in 0..size {
                        let p = sc.pred[j * size + i] as i32;
                        let s = self.src_sample(c_idx, x + i as u32, y + j as u32) as i32;
                        sc.residual[j * size + i] = (s - p) as i16;
                    }
                }
            }
        }

        self.stats.forward_transforms += 1;
        let tt = self.prof.on.then(std::time::Instant::now);
        transform::forward_transform_into(
            &sc.residual,
            log2_size,
            is_dst,
            self.bit_depth,
            &mut sc.coeffs,
            &mut sc.transform_tmp,
        );
        if let Some(tt) = tt {
            self.prof.eval_transform += tt.elapsed();
        }

        let scan = get_scan_order(log2_size, mode, c_idx, self.cat);
        let tq = self.prof.on.then(std::time::Instant::now);
        let (mut levels, mut nnz): (EvalLevels<'_>, u32) = match policy.rdoq {
            RdoqPolicy::FullSingleScan => {
                self.stats.trial_rdoq_blocks += if policy.commit { 0 } else { 1 };
                let result = rdoq::rdoq_single_scan_into(
                    ctxs,
                    &sc.coeffs,
                    log2_size,
                    c_idx,
                    qp,
                    self.bit_depth,
                    scan,
                    self.lambda(),
                    self.best2_rdoq2,
                    &mut sc.rdoq,
                );
                (EvalLevels::Borrowed(result.levels), result.nnz)
            }
            RdoqPolicy::Off => {
                let mut levels = std::mem::take(&mut sc.levels);
                let nnz = transform::quantize_into(
                    &sc.coeffs,
                    log2_size,
                    qp,
                    self.bit_depth,
                    &mut levels,
                );
                (EvalLevels::Owned(levels), nnz)
            }
        };

        if self.sign_data_hiding && nnz > 0 {
            let (scale, qbits) = transform::quant_params(log2_size, qp, self.bit_depth);
            nnz = crate::residual::apply_sign_data_hiding(
                levels.as_mut_slice(),
                &sc.coeffs,
                log2_size,
                scan,
                scale,
                qbits,
                nnz,
            );
        }
        if let Some(tq) = tq {
            self.prof.eval_quant_rdoq += tq.elapsed();
        }
        let cbf = nnz > 0;

        // Reconstructed residual into transform_tmp (coeffs reused as scratch).
        // Distortion must match the legacy committed path: compare source
        // pixels to clipped reconstructed pixels, but do it from scratch buffers
        // instead of writing the frame and reading it back.
        let tr = self.prof.on.then(std::time::Instant::now);
        if cbf {
            self.stats.inverse_transforms += 1;
            transform::reconstruct_residual_into(
                levels.as_slice(),
                log2_size,
                qp,
                self.bit_depth,
                is_dst,
                &mut sc.coeffs,
                &mut sc.transform_tmp,
            );
        }
        if let Some(tr) = tr {
            self.prof.eval_recon += tr.elapsed();
        }
        let max_val = (1i32 << self.bit_depth) - 1;
        let mut distortion = 0u64;
        for i in 0..size * size {
            let pred = sc.pred[i] as i32;
            let src = pred + sc.residual[i] as i32;
            let recon_residual = if cbf { sc.transform_tmp[i] as i32 } else { 0 };
            let recon = (pred + recon_residual).clamp(0, max_val);
            let d = src as i64 - recon as i64;
            distortion += (d * d) as u64;
        }

        let trp = self.prof.on.then(std::time::Instant::now);
        let frac_bits = ResidualPricer::price(
            self,
            ctxs,
            levels.as_slice(),
            log2_size,
            c_idx,
            mode,
            policy,
            &mut sc.residual_pricing,
        );
        if let Some(trp) = trp {
            self.prof.eval_residual_price += trp.elapsed();
        }

        if policy.commit {
            if let Some(region) = trial_region.as_deref_mut() {
                region.write_block(
                    c_idx,
                    x,
                    y,
                    log2_size,
                    &sc.pred,
                    &sc.transform_tmp,
                    cbf,
                    max_val,
                );
            } else {
                let (plane, stride) = self.frame.plane_mut(c_idx);
                for j in 0..size {
                    for i in 0..size {
                        let idx = (y as usize + j) * stride + x as usize + i;
                        let pred = sc.pred[j * size + i] as i32;
                        let recon = if cbf {
                            sc.transform_tmp[j * size + i] as i32
                        } else {
                            0
                        };
                        plane[idx] = (pred + recon).clamp(0, max_val) as u16;
                    }
                }
            }
        }

        let coded_levels = match levels {
            EvalLevels::Owned(levels) => {
                let recycle_levels = !policy.commit
                    && policy.rdoq == RdoqPolicy::Off
                    && policy.bits == ResidualBitPolicy::Approx;
                if recycle_levels {
                    sc.levels = levels;
                    Vec::new()
                } else {
                    levels
                }
            }
            EvalLevels::Borrowed(levels) => {
                if policy.commit {
                    levels.to_vec()
                } else {
                    Vec::new()
                }
            }
        };

        let cost = self.rd_cost(distortion, frac_bits);
        if let Some(start) = work_start {
            let bucket = policy
                .work_bucket
                .unwrap_or(match (policy.commit, policy.rdoq) {
                    (true, _) => WorkBucket::FinalReplay,
                    (false, RdoqPolicy::Off) => WorkBucket::TtLeafCheap,
                    (false, RdoqPolicy::FullSingleScan) => WorkBucket::TtLeafExact,
                });
            self.trace.note_work(
                bucket,
                WorkSample {
                    wall_ns: start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
                    log2_size,
                    c_idx,
                    prediction_calls: 1,
                    forward_transforms: 1,
                    quant_calls: 1,
                    rdoq_calls: u64::from(policy.rdoq == RdoqPolicy::FullSingleScan),
                    inverse_transforms: u64::from(cbf),
                    exact_residual_pricings: u64::from(policy.bits == ResidualBitPolicy::Exact),
                    approx_residual_pricings: u64::from(policy.bits == ResidualBitPolicy::Approx),
                    source_block_calls: 1,
                    ..WorkSample::default()
                },
            );
        }
        self.search_scratch = sc;
        LeafEval {
            distortion,
            frac_bits,
            cost,
            coded: CodedBlock {
                levels: coded_levels,
                cbf,
                frac_bits,
            },
        }
    }

    /// Evaluate a luma-only transform-block leaf at `kind`, committing its
    /// reconstruction into the frame, and wrap it as a luma-only [`Tt::Leaf`]
    /// (empty chroma) — the shape `build_luma_tt_leaf` produces. Returns the
    /// tree and its clipped pixel-domain distortion.
    fn rdo2_luma_leaf(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
        kind: EvalKind,
    ) -> (Tt, u64) {
        let bucket = match kind {
            EvalKind::CheapTrial => WorkBucket::TtLeafCheap,
            EvalKind::ExactTrial => WorkBucket::TtLeafExact,
            EvalKind::Final => WorkBucket::FinalReplay,
        };
        let mut policy = EvalPolicy::for_kind(kind).with_bucket(bucket);
        policy.commit = true;
        let e =
            self.rdo2_eval_leaf_block(ctxs, x0, y0, log2_size, 0, luma_mode, self.cur_qp_y, policy);
        let tt = Tt::Leaf(LeafTu {
            log2_size,
            chroma_log2: 0,
            trafo_depth,
            luma_mode,
            chroma_mode: luma_mode,
            luma: e.coded,
            cb: CodedBlock::empty(),
            cr: CodedBlock::empty(),
            cb1: CodedBlock::empty(),
            cr1: CodedBlock::empty(),
        });
        (tt, e.distortion)
    }

    /// Build a luma-only split node by recursively analysing the four quadrants
    /// (each committing in z-order so siblings see prior reconstruction).
    fn rdo2_luma_split(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> (Tt, u64) {
        let half = 1u32 << (log2_size - 1);
        let mut dist = 0u64;
        let mut kids = Vec::with_capacity(4);
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let (kt, kd) = self.rdo2_luma_subtree(
                x0 + dx,
                y0 + dy,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                ctxs,
            );
            dist += kd;
            kids.push(kt);
        }
        let tt = Tt::Split {
            log2_size,
            trafo_depth,
            cbf_cb: false,
            cbf_cr: false,
            cbf_cb1: false,
            cbf_cr1: false,
            parent_chroma: None,
            kids,
        };
        (tt, dist)
    }

    /// Staged luma transform-tree decision (slice 1), mirroring
    /// `build_luma_tt`'s contract: returns the chosen `Tt` and leaves the
    /// winner's luma reconstruction committed in the frame. The leaf is screened
    /// with a cheap (no-RDOQ, approximate-bits) non-committing evaluation; the
    /// leaf-vs-split decision escalates to an exact leaf recheck only when the
    /// costs are close.
    pub(in crate::encoder) fn rdo2_luma_subtree(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> (Tt, u64) {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.rdo2_luma_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        }
        if !self.can_split_tt(log2_size, trafo_depth) {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.rdo2_luma_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                ctxs,
                EvalKind::ExactTrial,
            );
        }
        let budget = self.block_budget(x0, y0, log2_size, 0);
        match budget.tu_split {
            SplitSearch::ForceLeaf => {
                self.record_tu_winner(x0, y0, log2_size, false);
                return self.rdo2_luma_leaf(
                    x0,
                    y0,
                    log2_size,
                    trafo_depth,
                    luma_mode,
                    ctxs,
                    EvalKind::ExactTrial,
                );
            }
            SplitSearch::ForceSplit => {
                self.record_tu_winner(x0, y0, log2_size, true);
                return self.rdo2_luma_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
            }
            _ => {}
        }

        let base = self.snapshot_frame_region(x0, y0, log2_size);

        // Cheap leaf screen (commits into the frame).
        let (leaf_tt, leaf_dist) = self.rdo2_luma_leaf(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            ctxs,
            EvalKind::CheapTrial,
        );
        if self.tu_split_early_terminate(&leaf_tt, budget) {
            self.stats.tu_split_early_terminations += 1;
            self.record_tu_winner(x0, y0, log2_size, false);
            return (leaf_tt, leaf_dist);
        }
        if self.should_limit_tu_to_neighbor_leaf(x0, y0, log2_size, &leaf_tt) {
            self.record_tu_neighbor_leaf_skip(x0, y0, log2_size);
            self.record_tu_winner(x0, y0, log2_size, false);
            return (leaf_tt, leaf_dist);
        }
        let mut leaf_cost = self.rd_cost(leaf_dist, self.tt_bits_luma(ctxs, &leaf_tt));
        let mut leaf_tt = leaf_tt;
        let mut leaf_dist = leaf_dist;
        let mut leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);

        // Split screen (commits into the frame).
        self.restore_frame_region(&base);
        let (split_tt, split_dist) =
            self.rdo2_luma_split(x0, y0, log2_size, trafo_depth, luma_mode, ctxs);
        let split_cost = self.rd_cost(split_dist, self.tt_bits_luma(ctxs, &split_tt));

        let margin = budget.close_call_margin;
        if close(leaf_cost, split_cost, margin) {
            // Escalate: re-evaluate the leaf exactly. Frame currently holds the
            // split reconstruction; save it, recheck the leaf, then restore split
            // as the comparison baseline.
            self.stats.best_tt_escalated_tu_decisions += 1;
            let split_frame = self.snapshot_frame_region(x0, y0, log2_size);
            self.restore_frame_region(&base);
            let (l2, ld2) = self.rdo2_luma_leaf(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                ctxs,
                EvalKind::ExactTrial,
            );
            leaf_cost = self.rd_cost(ld2, self.tt_bits_luma(ctxs, &l2));
            leaf_tt = l2;
            leaf_dist = ld2;
            leaf_frame = self.snapshot_frame_region(x0, y0, log2_size);
            self.restore_frame_region(&split_frame);
        } else {
            self.stats.best_tt_cheap_tu_decisions += 1;
        }

        self.record_close_call(leaf_cost.min(split_cost), leaf_cost.max(split_cost), margin);
        if split_cost < leaf_cost {
            self.record_tu_winner(x0, y0, log2_size, true);
            (split_tt, split_dist)
        } else {
            self.restore_frame_region(&leaf_frame);
            self.record_tu_winner(x0, y0, log2_size, false);
            (leaf_tt, leaf_dist)
        }
    }

    /// Staged luma+chroma transform-tree decision (slice 2), matching
    /// `build_tt`'s contract. Strategy (refactor-rdo.md §13, "plan first, replay
    /// once"): screen the whole subtree cheaply via the legacy `build_tt` with
    /// `best_tt_cheap_trial` (no trial RDOQ, approximate bits) under the
    /// `in_rdo2` re-entry guard, take the chosen structure as a `TtPlan`, restore
    /// the frame, and replay it exactly via `final_code_tt`. The cheap screen
    /// already recorded the winner's tu-depth map, so it is kept; only the
    /// reconstruction is recomputed exactly. Reuses the validated chroma /
    /// parent-chroma / final-coding code unchanged.
    pub(in crate::encoder) fn rdo2_analyze_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let base_frame = self.snapshot_frame_region(x0, y0, log2_size);
        let cheap = if self.rdo2_tt_native {
            // Phase 8: screen the structure with a native rdo2 recursion whose
            // leaves are costed through `rdo2_eval_leaf_block` at an explicit
            // `EvalKind::CheapTrial` policy, rather than re-entering the legacy
            // recursive `build_tt` under the implicit `best_tt_cheap_trial` flag.
            self.rdo2_tt_subtree_native(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            )
        } else {
            assert_ne!(
                self.effort,
                crate::Effort::Best,
                "Best TT analysis must use the native rdo2 screen"
            );
            self.in_rdo2 = true;
            let cheap = self.with_tt_trial_flags(true, false, |this| {
                this.build_tt(
                    x0,
                    y0,
                    log2_size,
                    trafo_depth,
                    luma_mode,
                    chroma_mode,
                    ctxs,
                    None,
                )
            });
            self.in_rdo2 = false;
            cheap
        };

        let plan = Self::tt_to_plan(&cheap);
        // Keep the screen's tu-depth map (already the winner's); recompute exact
        // reconstruction from the chosen structure.
        self.restore_frame_region(&base_frame);
        self.rdo2_final_code_tt(&plan, x0, y0, luma_mode, chroma_mode, ctxs)
    }

    /// Cheap, committing luma+chroma transform-block leaf for the native Phase-8
    /// TU screen. Mirrors `build_tt_leaf`'s geometry (luma + chroma Cb/Cr, with
    /// stacked 4:2:2 sub-blocks) but codes every block through
    /// `rdo2_eval_leaf_block` at an explicit `EvalKind::CheapTrial` policy
    /// (plain quant, approximate bits) committed into the frame, so the
    /// leaf-vs-split decision can read `distortion_tt_region` exactly as the
    /// legacy screen does. There is no chroma-cache path: the native screen is
    /// only the structure decision; the winner is exact-replayed by
    /// `final_code_tt`.
    fn rdo2_tt_leaf_cheap(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let mut policy =
            EvalPolicy::for_kind(EvalKind::CheapTrial).with_bucket(WorkBucket::TtLeafCheap);
        policy.commit = true;
        let luma = self
            .rdo2_eval_leaf_block(ctxs, x0, y0, log2_size, 0, luma_mode, self.cur_qp_y, policy)
            .coded;

        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let (chroma_log2, cb, cr, cb1, cr1) = match chroma_tb_geom(self.cat, x0, y0, log2_size) {
            Some((cx, cy, clog2, count)) => {
                let size = 1u32 << clog2;
                let cb = self
                    .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 1, pred, self.cur_qp_c, policy)
                    .coded;
                let cr = self
                    .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 2, pred, self.cur_qp_c, policy)
                    .coded;
                let (cb1, cr1) = if count > 1 {
                    let ty = cy + size;
                    (
                        self.rdo2_eval_leaf_block(
                            ctxs,
                            cx,
                            ty,
                            clog2,
                            1,
                            pred,
                            self.cur_qp_c,
                            policy,
                        )
                        .coded,
                        self.rdo2_eval_leaf_block(
                            ctxs,
                            cx,
                            ty,
                            clog2,
                            2,
                            pred,
                            self.cur_qp_c,
                            policy,
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

    fn rdo2_tt_leaf_cheap_trial(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        region: &mut TrialReconRegion,
    ) -> (Tt, u64) {
        let mut policy =
            EvalPolicy::for_kind(EvalKind::CheapTrial).with_bucket(WorkBucket::TtLeafCheap);
        policy.commit = true;
        let luma = self.rdo2_eval_leaf_block_in_region(
            ctxs,
            x0,
            y0,
            log2_size,
            0,
            luma_mode,
            self.cur_qp_y,
            policy,
            Some(region),
        );
        let mut distortion = luma.distortion;

        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let (chroma_log2, cb, cr, cb1, cr1) = match chroma_tb_geom(self.cat, x0, y0, log2_size) {
            Some((cx, cy, clog2, count)) => {
                let size = 1u32 << clog2;
                let cb = self.rdo2_eval_leaf_block_in_region(
                    ctxs,
                    cx,
                    cy,
                    clog2,
                    1,
                    pred,
                    self.cur_qp_c,
                    policy,
                    Some(region),
                );
                distortion += cb.distortion;
                let cr = self.rdo2_eval_leaf_block_in_region(
                    ctxs,
                    cx,
                    cy,
                    clog2,
                    2,
                    pred,
                    self.cur_qp_c,
                    policy,
                    Some(region),
                );
                distortion += cr.distortion;
                let (cb1, cr1) = if count > 1 {
                    let ty = cy + size;
                    let cb1 = self.rdo2_eval_leaf_block_in_region(
                        ctxs,
                        cx,
                        ty,
                        clog2,
                        1,
                        pred,
                        self.cur_qp_c,
                        policy,
                        Some(region),
                    );
                    distortion += cb1.distortion;
                    let cr1 = self.rdo2_eval_leaf_block_in_region(
                        ctxs,
                        cx,
                        ty,
                        clog2,
                        2,
                        pred,
                        self.cur_qp_c,
                        policy,
                        Some(region),
                    );
                    distortion += cr1.distortion;
                    (cb1.coded, cr1.coded)
                } else {
                    (CodedBlock::empty(), CodedBlock::empty())
                };
                (clog2, cb.coded, cr.coded, cb1, cr1)
            }
            None => (
                0,
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
                CodedBlock::empty(),
            ),
        };

        (
            Tt::Leaf(LeafTu {
                log2_size,
                chroma_log2,
                trafo_depth,
                luma_mode,
                chroma_mode,
                luma: luma.coded,
                cb,
                cr,
                cb1,
                cr1,
            }),
            distortion,
        )
    }

    fn rdo2_parent_chroma_tu_cheap(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Option<ParentChromaTu> {
        if self.cat != 1 || log2_size != 3 {
            return None;
        }
        let (cx, cy, clog2, _count) = chroma_tb_geom(self.cat, x0, y0, log2_size)?;
        let mut policy =
            EvalPolicy::for_kind(EvalKind::CheapTrial).with_bucket(WorkBucket::TtLeafCheap);
        policy.commit = true;
        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let cb = self
            .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 1, pred, self.cur_qp_c, policy)
            .coded;
        let cr = self
            .rdo2_eval_leaf_block(ctxs, cx, cy, clog2, 2, pred, self.cur_qp_c, policy)
            .coded;
        Some(ParentChromaTu {
            log2_size: clog2,
            chroma_mode,
            cb,
            cr,
        })
    }

    fn rdo2_parent_chroma_tu_cheap_trial(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        region: &mut TrialReconRegion,
    ) -> (Option<ParentChromaTu>, u64) {
        if self.cat != 1 || log2_size != 3 {
            return (None, 0);
        }
        let Some((cx, cy, clog2, _count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) else {
            return (None, 0);
        };
        let mut policy =
            EvalPolicy::for_kind(EvalKind::CheapTrial).with_bucket(WorkBucket::TtLeafCheap);
        policy.commit = true;
        let pred = chroma_pred_mode(self.cat, chroma_mode);
        let cb = self.rdo2_eval_leaf_block_in_region(
            ctxs,
            cx,
            cy,
            clog2,
            1,
            pred,
            self.cur_qp_c,
            policy,
            Some(region),
        );
        let cr = self.rdo2_eval_leaf_block_in_region(
            ctxs,
            cx,
            cy,
            clog2,
            2,
            pred,
            self.cur_qp_c,
            policy,
            Some(region),
        );
        let distortion = cb.distortion + cr.distortion;
        (
            Some(ParentChromaTu {
                log2_size: clog2,
                chroma_mode,
                cb: cb.coded,
                cr: cr.coded,
            }),
            distortion,
        )
    }

    /// Native split node for the Phase-8 TU screen: recurse into the four
    /// quadrants (committing in z-order so siblings see prior reconstruction),
    /// then code the parent chroma TU through rdo2's explicit cheap policy.
    fn rdo2_tt_split_native(
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
        let mut kids = Vec::with_capacity(4);
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            kids.push(self.rdo2_tt_subtree_native(
                x0 + dx,
                y0 + dy,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
            ));
        }
        let parent_chroma = self.rdo2_parent_chroma_tu_cheap(x0, y0, log2_size, chroma_mode, ctxs);
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

    fn rdo2_tt_split_native_trial(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        region: &mut TrialReconRegion,
    ) -> (Tt, u64) {
        let half = 1u32 << (log2_size - 1);
        let mut kids = Vec::with_capacity(4);
        let mut distortion = 0u64;
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            let (kid, kid_dist) = self.rdo2_tt_subtree_native_trial(
                x0 + dx,
                y0 + dy,
                log2_size - 1,
                trafo_depth + 1,
                luma_mode,
                chroma_mode,
                ctxs,
                region,
            );
            distortion += kid_dist;
            kids.push(kid);
        }
        let (parent_chroma, parent_dist) =
            self.rdo2_parent_chroma_tu_cheap_trial(x0, y0, log2_size, chroma_mode, ctxs, region);
        distortion += parent_dist;
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
        (
            Tt::Split {
                log2_size,
                trafo_depth,
                cbf_cb,
                cbf_cb1,
                cbf_cr1,
                cbf_cr,
                parent_chroma,
                kids,
            },
            distortion,
        )
    }

    /// Native luma+chroma transform-tree structure decision for Phase 8,
    /// replacing `build_tt`'s recursive cheap screen. The leaf-vs-split control
    /// flow (forced leaf/split, prefer-split, early-terminate, neighbour-leaf
    /// limit, close-call recording, tu-depth winner map) is identical to
    /// `build_tt`; only the block coder changed — every cheap trial block is now
    /// an explicit `EvalKind::CheapTrial` `rdo2_eval_leaf_block` call instead of
    /// a `trial_code_block` riding the implicit `best_tt_cheap_trial` flag.
    fn rdo2_tt_subtree_native(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.rdo2_tt_split_native(
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
            return self.rdo2_tt_leaf_cheap(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }

        let budget = self.block_budget(x0, y0, log2_size, 0);
        if budget.tu_split == SplitSearch::ForceLeaf {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.rdo2_tt_leaf_cheap(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }
        if budget.tu_split == SplitSearch::ForceSplit {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.rdo2_tt_split_native(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
            );
        }

        self.rdo2_tt_compare_native(
            budget,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn rdo2_tt_subtree_native_trial(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        region: &mut TrialReconRegion,
    ) -> (Tt, u64) {
        if log2_size > MAX_TB_LOG2 {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.rdo2_tt_split_native_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                region,
            );
        }
        if !self.can_split_tt(log2_size, trafo_depth) {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.rdo2_tt_leaf_cheap_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                region,
            );
        }

        let budget = self.block_budget(x0, y0, log2_size, 0);
        if budget.tu_split == SplitSearch::ForceLeaf {
            self.record_tu_winner(x0, y0, log2_size, false);
            return self.rdo2_tt_leaf_cheap_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                region,
            );
        }
        if budget.tu_split == SplitSearch::ForceSplit {
            self.record_tu_winner(x0, y0, log2_size, true);
            return self.rdo2_tt_split_native_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                region,
            );
        }

        self.rdo2_tt_compare_native_trial(
            budget,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            region,
        )
    }

    fn commit_trial_recon_region(&mut self, region: &TrialReconRegion) {
        for src_plane in &region.planes {
            let (dst, dst_stride) = self.frame.plane_mut(src_plane.c_idx);
            for row in 0..src_plane.height {
                let src = row * src_plane.stride;
                let dst_idx = (src_plane.y + row) * dst_stride + src_plane.x;
                dst[dst_idx..dst_idx + src_plane.width]
                    .copy_from_slice(&src_plane.data[src..src + src_plane.width]);
            }
        }
    }

    /// Leaf-vs-split structure decision for the native TU screen. Trials are
    /// evaluated against cloned local recon regions, and only the selected
    /// region is committed to `self.frame`.
    #[allow(clippy::too_many_arguments)]
    fn rdo2_tt_compare_native(
        &mut self,
        budget: BlockSearchBudget,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        let mut region = TrialReconRegion::new(self, x0, y0, log2_size);
        let (tt, _distortion) = self.rdo2_tt_compare_native_trial(
            budget,
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            &mut region,
        );
        self.commit_trial_recon_region(&region);
        tt
    }

    #[allow(clippy::too_many_arguments)]
    fn rdo2_tt_compare_native_trial(
        &mut self,
        budget: BlockSearchBudget,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        chroma_mode: u8,
        ctxs: &Contexts,
        region: &mut TrialReconRegion,
    ) -> (Tt, u64) {
        if budget.tu_split == SplitSearch::PreferSplit {
            let mut split_region = region.clone();
            let (split, split_distortion) = self.rdo2_tt_split_native_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                &mut split_region,
            );
            let split_bits = self.tt_bits_full(ctxs, &split);
            let split_cost = self.rd_cost(split_distortion, split_bits);

            let mut leaf_region = region.clone();
            let (leaf, leaf_distortion) = self.rdo2_tt_leaf_cheap_trial(
                x0,
                y0,
                log2_size,
                trafo_depth,
                luma_mode,
                chroma_mode,
                ctxs,
                &mut leaf_region,
            );
            let leaf_bits = self.tt_bits_full(ctxs, &leaf);
            let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
            self.record_close_call(
                leaf_cost.min(split_cost),
                leaf_cost.max(split_cost),
                budget.close_call_margin,
            );

            if split_cost < leaf_cost {
                *region = split_region;
                self.record_tu_winner(x0, y0, log2_size, true);
                return (split, split_distortion);
            }
            *region = leaf_region;
            self.record_tu_winner(x0, y0, log2_size, false);
            return (leaf, leaf_distortion);
        }

        let mut leaf_region = region.clone();
        let (leaf, leaf_distortion) = self.rdo2_tt_leaf_cheap_trial(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            &mut leaf_region,
        );
        if self.tu_split_early_terminate(&leaf, budget) {
            self.stats.tu_split_early_terminations += 1;
            *region = leaf_region;
            self.record_tu_winner(x0, y0, log2_size, false);
            return (leaf, leaf_distortion);
        }
        let leaf_bits = self.tt_bits_full(ctxs, &leaf);
        let leaf_cost = self.rd_cost(leaf_distortion, leaf_bits);
        if self.should_limit_tu_to_neighbor_leaf(x0, y0, log2_size, &leaf) {
            self.record_tu_neighbor_leaf_skip(x0, y0, log2_size);
            *region = leaf_region;
            self.record_tu_winner(x0, y0, log2_size, false);
            return (leaf, leaf_distortion);
        }

        let mut split_region = region.clone();
        let (split, split_distortion) = self.rdo2_tt_split_native_trial(
            x0,
            y0,
            log2_size,
            trafo_depth,
            luma_mode,
            chroma_mode,
            ctxs,
            &mut split_region,
        );
        let split_bits = self.tt_bits_full(ctxs, &split);
        let split_cost = self.rd_cost(split_distortion, split_bits);
        self.record_close_call(
            leaf_cost.min(split_cost),
            leaf_cost.max(split_cost),
            budget.close_call_margin,
        );

        if split_cost < leaf_cost {
            *region = split_region;
            self.record_tu_winner(x0, y0, log2_size, true);
            (split, split_distortion)
        } else {
            *region = leaf_region;
            self.record_tu_winner(x0, y0, log2_size, false);
            (leaf, leaf_distortion)
        }
    }

    /// Entry point matching `build_luma_tt`'s signature/contract.
    pub(in crate::encoder) fn rdo2_analyze_luma_tt(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
        ctxs: &Contexts,
    ) -> Tt {
        self.rdo2_luma_subtree(x0, y0, log2_size, trafo_depth, luma_mode, ctxs)
            .0
    }
}
