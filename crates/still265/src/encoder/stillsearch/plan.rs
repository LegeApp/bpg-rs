//! Internal StillSearch plans.
//!
//! These are search-owned descriptors. They deliberately mirror only the data
//! needed to emit final syntax, but they are not writer-facing `CuNode`/`Tt`
//! values. Loser candidates may allocate and discard these freely; `emit.rs` is
//! the only bridge into final syntax trees.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::plan::DecisionConfidence;

use super::arena::CoeffId;

#[derive(Debug)]
pub(super) struct PlanBlock {
    /// Handle into the CTU coeff arena, or `None` for a null-CBF block.
    pub(super) coeff: Option<CoeffId>,
    pub(super) cbf: bool,
    pub(super) frac_bits: u64,
}

impl PlanBlock {
    pub(super) fn empty() -> Self {
        Self {
            coeff: None,
            cbf: false,
            frac_bits: 0,
        }
    }
}

#[derive(Debug)]
pub(super) struct LeafTuPlan {
    pub(super) log2_size: u8,
    pub(super) chroma_log2: u8,
    pub(super) trafo_depth: u8,
    pub(super) luma_mode: u8,
    pub(super) chroma_mode: u8,
    pub(super) luma: PlanBlock,
    pub(super) cb: PlanBlock,
    pub(super) cr: PlanBlock,
    pub(super) cb1: PlanBlock,
    pub(super) cr1: PlanBlock,
}

#[derive(Debug)]
pub(super) struct ParentChromaPlan {
    pub(super) log2_size: u8,
    pub(super) chroma_mode: u8,
    pub(super) cb: PlanBlock,
    pub(super) cb1: PlanBlock,
    pub(super) cr: PlanBlock,
    pub(super) cr1: PlanBlock,
}

#[derive(Debug)]
pub(super) enum TtPlan {
    Split {
        log2_size: u8,
        trafo_depth: u8,
        cbf_cb: bool,
        cbf_cr: bool,
        cbf_cb1: bool,
        cbf_cr1: bool,
        parent_chroma: Option<ParentChromaPlan>,
        kids: Vec<TtPlan>,
    },
    Leaf(LeafTuPlan),
}

impl TtPlan {
    pub(super) fn cbf_cb(&self) -> bool {
        match self {
            TtPlan::Split { cbf_cb, .. } => *cbf_cb,
            TtPlan::Leaf(l) => l.cb.cbf,
        }
    }

    pub(super) fn cbf_cr(&self) -> bool {
        match self {
            TtPlan::Split { cbf_cr, .. } => *cbf_cr,
            TtPlan::Leaf(l) => l.cr.cbf,
        }
    }

    pub(super) fn cbf_cb1(&self) -> bool {
        match self {
            TtPlan::Split { cbf_cb1, .. } => *cbf_cb1,
            TtPlan::Leaf(l) => l.cb1.cbf,
        }
    }

    pub(super) fn cbf_luma(&self) -> bool {
        match self {
            TtPlan::Split { kids, .. } => kids.iter().any(TtPlan::cbf_luma),
            TtPlan::Leaf(l) => l.luma.cbf,
        }
    }

    pub(super) fn cbf_cr1(&self) -> bool {
        match self {
            TtPlan::Split { cbf_cr1, .. } => *cbf_cr1,
            TtPlan::Leaf(l) => l.cr1.cbf,
        }
    }
    pub(super) fn has_parent_second_chroma_cbf(&self) -> bool {
        match self {
            TtPlan::Split {
                parent_chroma,
                kids,
                ..
            } => {
                parent_chroma
                    .as_ref()
                    .is_some_and(|p| p.cb1.cbf || p.cr1.cbf)
                    || kids.iter().any(TtPlan::has_parent_second_chroma_cbf)
            }
            TtPlan::Leaf(_) => false,
        }
    }
}

#[derive(Debug)]
pub(super) struct NxnInfoPlan {
    pub(super) luma_modes: [u8; 4],
    pub(super) mpms: [[IntraPredMode; 3]; 4],
    pub(super) chroma_mode_idx: [u8; 4],
}

#[derive(Debug)]
pub(super) struct CuLeafPlan {
    pub(super) mpm: [IntraPredMode; 3],
    pub(super) luma_mode: u8,
    pub(super) chroma_mode_idx: u8,
    pub(super) confidence: DecisionConfidence,
    pub(super) tt: TtPlan,
    pub(super) nxn: Option<NxnInfoPlan>,
}

#[derive(Debug)]
pub(super) enum CuPlan {
    Split { kids: Vec<CuPlan> },
    Leaf(CuLeafPlan),
}

impl CuPlan {
    pub(super) fn has_parent_second_chroma_cbf(&self) -> bool {
        match self {
            CuPlan::Split { kids } => kids.iter().any(CuPlan::has_parent_second_chroma_cbf),
            CuPlan::Leaf(leaf) => leaf.tt.has_parent_second_chroma_cbf(),
        }
    }
}
