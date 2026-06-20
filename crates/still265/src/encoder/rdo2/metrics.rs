//! Shared RDO metrics and tree helpers owned by rdo2.

use bpg_hevc_decode::hevc::intra::predict_intra_into;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::Contexts;
use crate::effort::BlockSearchBudget;
use crate::plan::{CuLeafPlan, CuPlan, ParentChromaPlan, TtPlan};
use crate::primitives;
use crate::Effort;

use super::super::types::{
    chroma_tb_geom, cu_leaf_has_residual, CuLeaf, CuNode, Tt, MAX_INTRA_TT_DEPTH, MAX_TB_LOG2,
    MIN_TB_LOG2,
};
use super::cost::{estimate_intra_luma_mode_bits, estimate_tt_bits};

impl<'a> super::super::Encoder<'a> {
    pub(in crate::encoder) fn satd_block_cost(
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
    pub(in crate::encoder) fn prune_angular_modes(
        &self,
        src: &[u16],
        size: usize,
        budget: BlockSearchBudget,
    ) -> bool {
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

    #[allow(clippy::too_many_arguments)]
    pub(in crate::encoder) fn score_luma_rough_mode(
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
        self.luma_rough_score_from_pred(pred, size, mode, ctxs, mpm, src, src8, pred8)
    }

    /// Score a luma mode from an already-predicted block (the post-prediction
    /// half of [`Self::score_luma_rough_mode`]). Lets the batched
    /// `intra_pred_allangs` path score angular modes from precomputed slots.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::encoder) fn luma_rough_score_from_pred(
        &mut self,
        pred: &[u16],
        size: usize,
        mode: u8,
        ctxs: &Contexts,
        mpm: [IntraPredMode; 3],
        src: &[u16],
        src8: Option<&[u8]>,
        pred8: &mut Vec<u8>,
    ) -> u64 {
        self.stats.luma_rough_predictions += 1;
        let satd = self.satd_block_cost(src, src8, pred, size, pred8) as u64;
        let mode_bits = estimate_intra_luma_mode_bits(ctxs, mpm, mode);
        if self.effort == Effort::Best {
            let cost = if self.best2_rough_lambda {
                self.rough_cost(satd, mode_bits)
            } else {
                self.rd_cost(satd, mode_bits)
            };
            return (cost * 1024.0).round() as u64;
        }
        satd * CabacEstimator::SCALE + mode_bits
    }

    pub(in crate::encoder) fn distortion_block(
        &self,
        c_idx: u8,
        x: u32,
        y: u32,
        log2_size: u8,
    ) -> u64 {
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

    pub(in crate::encoder) fn distortion_tt_region(&self, x0: u32, y0: u32, log2_size: u8) -> u64 {
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

    pub(in crate::encoder) fn distortion_cu_node(
        &self,
        node: &CuNode,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
    ) -> u64 {
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

    pub(in crate::encoder) fn rd_cost(&self, distortion: u64, frac_bits: u64) -> f64 {
        distortion as f64 + self.lambda() * (frac_bits as f64 / CabacEstimator::SCALE as f64)
    }

    pub(in crate::encoder) fn lambda(&self) -> f64 {
        if self.best2_rd_lambda {
            0.038f64 * (0.234f64 * self.cur_qp_y as f64).exp()
        } else {
            Self::legacy_lambda(self.cur_qp_y)
        }
    }

    #[inline]
    fn legacy_lambda(qp: i32) -> f64 {
        0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0)
    }

    pub(in crate::encoder) fn lambda_sad(&self) -> f64 {
        Self::legacy_lambda(self.cur_qp_y).sqrt()
    }

    pub(in crate::encoder) fn rough_cost(&self, satd: u64, frac_bits: u64) -> f64 {
        satd as f64 + self.lambda_sad() * (frac_bits as f64 / CabacEstimator::SCALE as f64)
    }

    pub(in crate::encoder) fn can_split_tt(&self, log2_size: u8, trafo_depth: u8) -> bool {
        if !(log2_size <= MAX_TB_LOG2
            && log2_size > MIN_TB_LOG2
            && trafo_depth < MAX_INTRA_TT_DEPTH)
        {
            return false;
        }
        !((self.cat == 1 || self.cat == 2) && log2_size == 3)
    }

    pub(in crate::encoder) fn tt_bits_luma(&self, ctxs: &Contexts, tt: &Tt) -> u64 {
        estimate_tt_bits(ctxs, tt, 0, false, true, true)
    }

    pub(in crate::encoder) fn tt_bits_full(&self, ctxs: &Contexts, tt: &Tt) -> u64 {
        estimate_tt_bits(ctxs, tt, self.cat, false, true, true)
    }

    pub(in crate::encoder) fn tu_split_early_terminate(
        &self,
        leaf: &Tt,
        budget: BlockSearchBudget,
    ) -> bool {
        let Tt::Leaf(leaf) = leaf else {
            return false;
        };
        budget.tu_split_early_terminate(leaf.luma.cbf)
    }

    pub(in crate::encoder) fn tt_to_plan(tt: &Tt) -> TtPlan {
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

    pub(in crate::encoder) fn cu_early_terminate(
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

    pub(in crate::encoder) fn cu_to_plan(cu: &CuNode) -> CuPlan {
        match cu {
            CuNode::Leaf(leaf) => CuPlan::Leaf(CuLeafPlan {
                mpm: leaf.mpm,
                luma_mode: leaf.luma_mode,
                chroma_mode_idx: leaf.chroma_mode_idx,
                chroma_mode: Self::chroma_mode_from_idx(leaf.luma_mode, leaf.chroma_mode_idx),
                tt: Self::tt_to_plan(&leaf.tt),
                nxn: leaf.nxn.as_ref().map(|n| n.luma_modes),
            }),
            CuNode::Split { kids } => CuPlan::Split {
                kids: kids.iter().map(Self::cu_to_plan).collect(),
            },
        }
    }
}
