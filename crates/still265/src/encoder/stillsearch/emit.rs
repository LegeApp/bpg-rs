//! Plan-to-final-syntax bridge helpers.
//!
//! This is the only StillSearch module allowed to materialize writer-facing
//! `CuNode`/`Tt`/`CodedBlock` trees. Search code should pass internal plans.

use crate::plan::DecisionConfidence;

use super::super::syntax::{CodedBlock, CuLeaf, CuNode, LeafTu, NxnInfo, ParentChromaTu, Tt};
use super::super::types::CHROMA_DM_IDX;
use super::arena::CoeffArena;
use super::plan::{
    CuLeafPlan, CuPlan, LeafTuPlan, NxnInfoPlan, ParentChromaPlan, PlanBlock, TtPlan,
};

pub(super) fn empty_block() -> PlanBlock {
    PlanBlock::empty()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn leaf_tu(
    log2_size: u8,
    chroma_log2: u8,
    trafo_depth: u8,
    luma_mode: u8,
    chroma_mode: u8,
    luma: PlanBlock,
    cb: PlanBlock,
    cr: PlanBlock,
    cb1: PlanBlock,
    cr1: PlanBlock,
) -> TtPlan {
    TtPlan::Leaf(LeafTuPlan {
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

pub(super) fn cu_leaf(
    mpm: [bpg_hevc_decode::hevc::slice::IntraPredMode; 3],
    luma_mode: u8,
    tt: TtPlan,
) -> CuLeafPlan {
    CuLeafPlan {
        mpm,
        luma_mode,
        chroma_mode_idx: CHROMA_DM_IDX,
        confidence: DecisionConfidence::Clear,
        tt,
        nxn: None,
    }
}

pub(super) fn cu_node(plan: CuPlan, coeffs: &CoeffArena) -> CuNode {
    match plan {
        CuPlan::Split { kids } => CuNode::Split {
            kids: kids.into_iter().map(|k| cu_node(k, coeffs)).collect(),
        },
        CuPlan::Leaf(leaf) => CuNode::Leaf(cu_leaf_syntax(leaf, coeffs)),
    }
}

fn plan_block(block: PlanBlock, coeffs: &CoeffArena) -> CodedBlock {
    CodedBlock {
        levels: block
            .coeff
            .map(|id| coeffs.get(id).to_vec())
            .unwrap_or_default(),
        cbf: block.cbf,
        frac_bits: block.frac_bits,
    }
}

fn parent_chroma(plan: ParentChromaPlan, coeffs: &CoeffArena) -> ParentChromaTu {
    ParentChromaTu {
        log2_size: plan.log2_size,
        chroma_mode: plan.chroma_mode,
        cb: plan_block(plan.cb, coeffs),
        cb1: plan_block(plan.cb1, coeffs),
        cr: plan_block(plan.cr, coeffs),
        cr1: plan_block(plan.cr1, coeffs),
    }
}

fn tt(plan: TtPlan, coeffs: &CoeffArena) -> Tt {
    match plan {
        TtPlan::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cr,
            cbf_cb1,
            cbf_cr1,
            parent_chroma: pc,
            kids,
        } => Tt::Split {
            log2_size,
            trafo_depth,
            cbf_cb,
            cbf_cr,
            cbf_cb1,
            cbf_cr1,
            parent_chroma: pc.map(|c| parent_chroma(c, coeffs)),
            kids: kids.into_iter().map(|k| tt(k, coeffs)).collect(),
        },
        TtPlan::Leaf(LeafTuPlan {
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
        }) => Tt::Leaf(LeafTu {
            log2_size,
            chroma_log2,
            trafo_depth,
            luma_mode,
            chroma_mode,
            luma: plan_block(luma, coeffs),
            cb: plan_block(cb, coeffs),
            cr: plan_block(cr, coeffs),
            cb1: plan_block(cb1, coeffs),
            cr1: plan_block(cr1, coeffs),
        }),
    }
}

fn nxn(plan: NxnInfoPlan) -> NxnInfo {
    NxnInfo {
        luma_modes: plan.luma_modes,
        mpms: plan.mpms,
        chroma_mode_idx: plan.chroma_mode_idx,
    }
}

fn cu_leaf_syntax(plan: CuLeafPlan, coeffs: &CoeffArena) -> CuLeaf {
    CuLeaf {
        mpm: plan.mpm,
        luma_mode: plan.luma_mode,
        chroma_mode_idx: plan.chroma_mode_idx,
        confidence: plan.confidence,
        tt: tt(plan.tt, coeffs),
        nxn: plan.nxn.map(nxn),
    }
}
