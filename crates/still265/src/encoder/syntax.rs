//! Final writer-facing syntax tree types.
//!
//! These structures are the narrow bridge into `write.rs`: StillSearch may build
//! them only for the selected CTU winner. They are not search-candidate storage.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::plan::DecisionConfidence;

/// Quantized residual data for one transform block in the final syntax tree.
#[derive(Clone, Debug)]
pub(super) struct CodedBlock {
    pub(super) levels: Vec<i16>,
    pub(super) cbf: bool,
    pub(super) frac_bits: u64,
}

impl CodedBlock {
    pub(super) fn empty() -> Self {
        Self {
            levels: Vec::new(),
            cbf: false,
            frac_bits: 0,
        }
    }
}

/// A reconstructed-and-recorded leaf transform unit's coded data.
#[derive(Debug)]
pub(super) struct LeafTu {
    pub(super) log2_size: u8,
    /// `log2` size of chroma TBs at this leaf, if any.
    pub(super) chroma_log2: u8,
    pub(super) trafo_depth: u8,
    pub(super) luma_mode: u8,
    /// Actual chroma prediction mode used for residual scan derivation.
    pub(super) chroma_mode: u8,
    pub(super) luma: CodedBlock,
    pub(super) cb: CodedBlock,
    pub(super) cr: CodedBlock,
    /// Second stacked chroma TB (4:2:2 only).
    pub(super) cb1: CodedBlock,
    pub(super) cr1: CodedBlock,
}

/// Chroma transform data carried by a split node in the H.265 special case
/// where subsampled chroma is coded at the parent after split luma children.
#[derive(Debug)]
pub(super) struct ParentChromaTu {
    pub(super) log2_size: u8,
    pub(super) chroma_mode: u8,
    pub(super) cb: CodedBlock,
    /// Second stacked chroma TB (4:2:2 only).
    pub(super) cb1: CodedBlock,
    pub(super) cr: CodedBlock,
    /// Second stacked chroma TB (4:2:2 only).
    pub(super) cr1: CodedBlock,
}

/// Final transform tree syntax for one coded CU.
#[derive(Debug)]
pub(super) enum Tt {
    Split {
        log2_size: u8,
        trafo_depth: u8,
        cbf_cb: bool,
        cbf_cr: bool,
        /// Second stacked-chroma-TB cbf (4:2:2 only).
        cbf_cb1: bool,
        cbf_cr1: bool,
        parent_chroma: Option<ParentChromaTu>,
        kids: Vec<Tt>,
    },
    Leaf(LeafTu),
}

impl Tt {
    pub(super) fn cbf_cb(&self) -> bool {
        match self {
            Tt::Split { cbf_cb, .. } => *cbf_cb,
            Tt::Leaf(l) => l.cb.cbf,
        }
    }

    pub(super) fn cbf_cr(&self) -> bool {
        match self {
            Tt::Split { cbf_cr, .. } => *cbf_cr,
            Tt::Leaf(l) => l.cr.cbf,
        }
    }

    pub(super) fn cbf_cb1(&self) -> bool {
        match self {
            Tt::Split { cbf_cb1, .. } => *cbf_cb1,
            Tt::Leaf(l) => l.cb1.cbf,
        }
    }

    pub(super) fn cbf_cr1(&self) -> bool {
        match self {
            Tt::Split { cbf_cr1, .. } => *cbf_cr1,
            Tt::Leaf(l) => l.cr1.cbf,
        }
    }
}

/// Per-PU data for an 8x8 CU coded as `PartNxN`.
#[derive(Debug)]
pub(super) struct NxnInfo {
    pub(super) luma_modes: [u8; 4],
    pub(super) mpms: [[IntraPredMode; 3]; 4],
    /// Per-PU `intra_chroma_pred_mode` indexes for 4:4:4 PartNxN. Subsampled
    /// formats use the CU-level `CuLeaf::chroma_mode_idx` instead.
    pub(super) chroma_mode_idx: [u8; 4],
}

/// A coding unit, fully coded (intra mode + transform tree).
#[derive(Debug)]
pub(super) struct CuLeaf {
    pub(super) mpm: [IntraPredMode; 3],
    pub(super) luma_mode: u8,
    pub(super) chroma_mode_idx: u8,
    pub(super) confidence: DecisionConfidence,
    pub(super) tt: Tt,
    pub(super) nxn: Option<NxnInfo>,
}

/// A final coding-quadtree node.
#[derive(Debug)]
pub(super) enum CuNode {
    Split { kids: Vec<CuNode> },
    Leaf(CuLeaf),
}
