//! StillSearch public-in-crate entry points.

use bpg_hevc_decode::hevc::intra;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::plan::DecisionConfidence;

use super::emit;
use super::ledger::WorkBucket;
use super::overlay::{ReconOverlay8, ReconOverlay16};
use super::workspace::{BlockScratch, CtuWorkspace};
use crate::encoder::Encoder;
use crate::encoder::syntax::CodedBlock;
use crate::encoder::syntax::{CuNode, Tt};
use crate::encoder::types::{MAX_TB_LOG2, chroma_pred_mode, chroma_tb_geom, has_chroma_tb};
use crate::residual::{apply_sign_data_hiding, get_scan_order};
use crate::transform;

pub(in crate::encoder) struct StillSearch {
    imp: StillSearchImpl,
}

enum StillSearchImpl {
    EightBit(StillSearchDepth<CtuSource8, ReconOverlay8>),
    HighBit(StillSearchDepth<CtuSource16, ReconOverlay16>),
}

struct StillSearchDepth<S, O> {
    workspace: CtuWorkspace,
    source: S,
    overlay: O,
}

impl StillSearch {
    pub(in crate::encoder) fn new(bit_depth: u8) -> Self {
        let imp = if bit_depth == 8 {
            StillSearchImpl::EightBit(StillSearchDepth::default())
        } else {
            StillSearchImpl::HighBit(StillSearchDepth::default())
        };
        Self { imp }
    }

    pub(in crate::encoder) fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        match &mut self.imp {
            StillSearchImpl::EightBit(search) => {
                search.build_ctu(state, x0, y0, log2_cb_size, ct_depth)
            }
            StillSearchImpl::HighBit(search) => {
                search.build_ctu(state, x0, y0, log2_cb_size, ct_depth)
            }
        }
    }
}

impl<S, O> Default for StillSearchDepth<S, O>
where
    S: Default,
    O: Default,
{
    fn default() -> Self {
        Self {
            workspace: CtuWorkspace::default(),
            source: S::default(),
            overlay: O::default(),
        }
    }
}

trait OverlayCache {
    fn clear(&mut self);
}

impl OverlayCache for ReconOverlay8 {
    fn clear(&mut self) {
        ReconOverlay8::clear(self);
    }
}

impl OverlayCache for ReconOverlay16 {
    fn clear(&mut self) {
        ReconOverlay16::clear(self);
    }
}

trait CtuSourceCache {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8);
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16;
}

#[derive(Default)]
struct CtuSource8 {
    planes: [CachedPlane8; 3],
}

#[derive(Default)]
struct CtuSource16 {
    planes: [CachedPlane16; 3],
}

#[derive(Default)]
struct CachedPlane8 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

#[derive(Default)]
struct CachedPlane16 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u16>,
}

impl CtuSourceCache for CtuSource8 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane.samples.push(
                        state
                            .src_sample(c_idx, px + dx, py + dy)
                            .min(u8::MAX as u16) as u8,
                    );
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y) as u16)
            .unwrap_or(128)
    }
}

impl CtuSourceCache for CtuSource16 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane
                        .samples
                        .push(state.src_sample(c_idx, px + dx, py + dy));
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y))
            .unwrap_or(128)
    }
}

impl CachedPlane8 {
    fn sample(&self, x: u32, y: u32) -> u8 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}

impl CachedPlane16 {
    fn sample(&self, x: u32, y: u32) -> u16 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        self.workspace.reset();
        self.source.reset_from_ctu(state, x0, y0, log2_cb_size);
        self.overlay.clear();
        self.build_cu(state, x0, y0, log2_cb_size, ct_depth)
    }

    fn build_cu(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        let cb_size = 1u32 << log2_cb_size;
        let fully_inside =
            x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
        if !fully_inside && log2_cb_size > 3 {
            let half = cb_size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            // Final syntax only. Candidate search must use PlanId/TtPlanId arenas.
            let mut kids = Vec::with_capacity(4);
            kids.push(self.build_cu(state, x0, y0, log2_cb_size - 1, ct_depth + 1));
            if x1 < state.display_width {
                kids.push(self.build_cu(state, x1, y0, log2_cb_size - 1, ct_depth + 1));
            }
            if y1 < state.display_height {
                kids.push(self.build_cu(state, x0, y1, log2_cb_size - 1, ct_depth + 1));
            }
            if x1 < state.display_width && y1 < state.display_height {
                kids.push(self.build_cu(state, x1, y1, log2_cb_size - 1, ct_depth + 1));
            }
            return CuNode::Split { kids };
        }

        self.workspace.ledger.bump(WorkBucket::FinalCommit);
        state.stats.cu_trials += 1;
        state.set_ct_depth(x0, y0, log2_cb_size, ct_depth);

        let mpm = intra::fill_mpm_candidates(
            state.neighbor_left_mode(x0, y0),
            state.neighbor_above_mode(x0, y0),
        );
        let luma_mode = IntraPredMode::Dc.as_u8();
        state.store_mode(x0, y0, log2_cb_size, luma_mode);

        if state.aq.active {
            state.aq_set_cu_qp(x0, y0);
        }
        let tt = self.build_tt(state, x0, y0, log2_cb_size, 0, luma_mode);
        state.stats.final_coded_blocks += 1;
        CuNode::Leaf(emit::cu_leaf(mpm, luma_mode, tt)).tap_confidence(DecisionConfidence::Clear)
    }

    fn build_tt(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        trafo_depth: u8,
        luma_mode: u8,
    ) -> Tt {
        if log2_size > MAX_TB_LOG2 {
            self.workspace.ledger.bump(WorkBucket::TuSplit);
            let half = 1u32 << (log2_size - 1);
            let kid_log2 = log2_size - 1;
            // Final syntax only. Candidate search must use arena plan IDs.
            let kids = vec![
                self.build_tt(state, x0, y0, kid_log2, trafo_depth + 1, luma_mode),
                self.build_tt(state, x0 + half, y0, kid_log2, trafo_depth + 1, luma_mode),
                self.build_tt(state, x0, y0 + half, kid_log2, trafo_depth + 1, luma_mode),
                self.build_tt(
                    state,
                    x0 + half,
                    y0 + half,
                    kid_log2,
                    trafo_depth + 1,
                    luma_mode,
                ),
            ];
            let cbf_cb = kids.iter().any(Tt::cbf_cb);
            let cbf_cr = kids.iter().any(Tt::cbf_cr);
            let cbf_cb1 = kids.iter().any(Tt::cbf_cb1);
            let cbf_cr1 = kids.iter().any(Tt::cbf_cr1);
            return Tt::Split {
                log2_size,
                trafo_depth,
                cbf_cb,
                cbf_cr,
                cbf_cb1,
                cbf_cr1,
                parent_chroma: None,
                kids,
            };
        }

        self.workspace.ledger.bump(WorkBucket::TuLeaf);
        let (luma, cb, cr, cb1, cr1, chroma_log2, chroma_mode) =
            self.commit_fixed_dc_leaf_bridge(state, x0, y0, log2_size, luma_mode);
        state.store_tu_depth(x0, y0, log2_size, false);
        emit::leaf_tu(
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
        )
    }

    fn commit_fixed_dc_leaf_bridge(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
    ) -> (
        CodedBlock,
        CodedBlock,
        CodedBlock,
        CodedBlock,
        CodedBlock,
        u8,
        u8,
    ) {
        state.predict_intra_tiled(x0, y0, log2_size, IntraPredMode::Dc, 0);
        let luma =
            self.commit_fixed_dc_predicted_block_bridge(state, x0, y0, log2_size, 0, luma_mode);

        let mut cb = emit::empty_block();
        let mut cr = emit::empty_block();
        let mut cb1 = emit::empty_block();
        let mut cr1 = emit::empty_block();
        let mut chroma_log2 = 0;
        let chroma_mode_for_syntax = luma_mode;

        if has_chroma_tb(state.cat, log2_size) {
            if let Some((cx, cy, clog2, count)) = chroma_tb_geom(state.cat, x0, y0, log2_size) {
                chroma_log2 = clog2;
                let cmode =
                    IntraPredMode::from_u8(chroma_pred_mode(state.cat, chroma_mode_for_syntax))
                        .unwrap_or(IntraPredMode::Dc);
                let step = 1u32 << clog2;
                state.predict_intra_tiled(cx, cy, clog2, cmode, 1);
                cb = self.commit_fixed_dc_predicted_block_bridge(
                    state,
                    cx,
                    cy,
                    clog2,
                    1,
                    cmode.as_u8(),
                );
                state.predict_intra_tiled(cx, cy, clog2, cmode, 2);
                cr = self.commit_fixed_dc_predicted_block_bridge(
                    state,
                    cx,
                    cy,
                    clog2,
                    2,
                    cmode.as_u8(),
                );
                if count > 1 {
                    let y1 = cy + step;
                    state.predict_intra_tiled(cx, y1, clog2, cmode, 1);
                    cb1 = self.commit_fixed_dc_predicted_block_bridge(
                        state,
                        cx,
                        y1,
                        clog2,
                        1,
                        cmode.as_u8(),
                    );
                    state.predict_intra_tiled(cx, y1, clog2, cmode, 2);
                    cr1 = self.commit_fixed_dc_predicted_block_bridge(
                        state,
                        cx,
                        y1,
                        clog2,
                        2,
                        cmode.as_u8(),
                    );
                }
            }
        }

        (luma, cb, cr, cb1, cr1, chroma_log2, chroma_mode_for_syntax)
    }

    /// Final-commit bridge for the fixed-DC skeleton. This function may mutate
    /// `state.frame`; trial/eval functions must not. `frac_bits` is deliberately
    /// zero because this bridge performs no RD comparison yet.
    fn commit_fixed_dc_predicted_block_bridge(
        &mut self,
        state: &mut Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        pred_mode: u8,
    ) -> CodedBlock {
        let size = 1usize << log2_size;
        let qp = if c_idx == 0 {
            state.cur_qp_y
        } else {
            state.cur_qp_c
        };
        let (plane, stride) = state.frame.plane(c_idx);
        let source = &self.source;
        let bit_depth = state.bit_depth;
        let neutral = 1u16 << bit_depth.saturating_sub(1);
        let scratch = &mut self.workspace.block_scratch;
        let (
            residual_i16,
            coeff_i16,
            transform_tmp_i16,
            levels_i16,
            dequant_coeff_i16,
            recon_residual_i16,
        ) = scratch_parts_for_log2(scratch, log2_size);

        residual_i16.clear();
        residual_i16.reserve(size * size);
        for y in 0..size {
            for x in 0..size {
                let px = x0 + x as u32;
                let py = y0 + y as u32;
                let pred = plane
                    .get(py as usize * stride + px as usize)
                    .copied()
                    .unwrap_or(neutral);
                let src = source.sample(c_idx, px, py);
                residual_i16.push(
                    (src as i32 - pred as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                );
            }
        }

        let is_dst4 = c_idx == 0 && log2_size == 2;
        transform::forward_transform_into(
            residual_i16,
            log2_size,
            is_dst4,
            bit_depth,
            coeff_i16,
            transform_tmp_i16,
        );
        let mut nnz = transform::quantize_into(coeff_i16, log2_size, qp, bit_depth, levels_i16);
        if state.sign_data_hiding && nnz > 0 {
            let scan = get_scan_order(log2_size, pred_mode, c_idx, state.cat);
            let (scale, qbits) = transform::quant_params(log2_size, qp, bit_depth);
            nnz = apply_sign_data_hiding(levels_i16, coeff_i16, log2_size, scan, scale, qbits, nnz);
        }

        if nnz > 0 {
            transform::reconstruct_residual_into(
                levels_i16,
                log2_size,
                qp,
                bit_depth,
                is_dst4,
                dequant_coeff_i16,
                recon_residual_i16,
            );
            let max_sample = (1i32 << bit_depth) - 1;
            let (plane, stride) = state.frame.plane_mut(c_idx);
            for y in 0..size {
                for x in 0..size {
                    let idx = (y0 as usize + y) * stride + x0 as usize + x;
                    if let Some(sample) = plane.get_mut(idx) {
                        let v = (*sample as i32 + recon_residual_i16[y * size + x] as i32)
                            .clamp(0, max_sample) as u16;
                        *sample = v;
                    }
                }
            }
            state.stats.final_rdoq_blocks += 1;
        }
        state.stats.final_coded_blocks += 1;
        CodedBlock {
            levels: if nnz > 0 {
                levels_i16.clone()
            } else {
                Vec::new()
            },
            cbf: nnz > 0,
            frac_bits: 0,
        }
    }
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

trait TapConfidence {
    fn tap_confidence(self, confidence: DecisionConfidence) -> Self;
}

impl TapConfidence for CuNode {
    fn tap_confidence(mut self, confidence: DecisionConfidence) -> Self {
        if let CuNode::Leaf(leaf) = &mut self {
            leaf.confidence = confidence;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    fn scan_eval_function_bodies(src: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let bytes = src.as_bytes();
        let mut i = 0;
        while let Some(pos) = src[i..].find("fn eval_") {
            let fn_start = i + pos;
            let Some(open_rel) = src[fn_start..].find('{') else {
                break;
            };
            let open = fn_start + open_rel;
            let mut depth = 0i32;
            let mut end = open;
            for (off, b) in bytes[open..].iter().enumerate() {
                match *b {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + off + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            out.push(&src[fn_start..end]);
            i = end;
        }
        out
    }

    #[test]
    fn search_eval_helpers_do_not_mutate_shared_frame() {
        for body in scan_eval_function_bodies(include_str!("api.rs")) {
            assert!(
                !body.contains("state.frame.plane_mut") && !body.contains("predict_intra_tiled"),
                "eval_* functions must write to overlays/scratch, not mutate state.frame directly:\n{body}"
            );
        }
    }
}
