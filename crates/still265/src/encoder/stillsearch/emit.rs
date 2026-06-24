//! Plan-to-final-syntax bridge helpers.

use crate::plan::DecisionConfidence;

use super::super::syntax::{CodedBlock, CuLeaf, LeafTu, Tt};
use super::super::types::CHROMA_DM_IDX;

pub(super) fn empty_block() -> CodedBlock {
    CodedBlock::empty()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn leaf_tu(
    log2_size: u8,
    chroma_log2: u8,
    trafo_depth: u8,
    luma_mode: u8,
    chroma_mode: u8,
    luma: CodedBlock,
    cb: CodedBlock,
    cr: CodedBlock,
    cb1: CodedBlock,
    cr1: CodedBlock,
) -> Tt {
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

pub(super) fn cu_leaf(
    mpm: [bpg_hevc_decode::hevc::slice::IntraPredMode; 3],
    luma_mode: u8,
    tt: Tt,
) -> CuLeaf {
    CuLeaf {
        mpm,
        luma_mode,
        chroma_mode_idx: CHROMA_DM_IDX,
        confidence: DecisionConfidence::Clear,
        tt,
        nxn: None,
    }
}
