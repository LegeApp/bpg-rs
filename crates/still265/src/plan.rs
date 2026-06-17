//! Decision-plan data for intra CU/TU search.
//!
//! Plans describe the selected syntax shape and modes without carrying coded
//! residual output. Final coding walks these plans and emits the concrete
//! `CuNode`/`Tt` structures used by the CABAC writer.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::effort::TrialQuality;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RdCost {
    pub distortion: u64,
    pub frac_bits: u64,
    pub cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrialResult<P> {
    pub plan: P,
    pub cost: RdCost,
    pub confidence: DecisionConfidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionConfidence {
    Clear,
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkStage {
    RoughModeSearch,
    LumaTrial,
    ChromaTrial,
    TuDecision,
    CuDecision,
    FinalCode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockPlan {
    pub x: u32,
    pub y: u32,
    pub log2_size: u8,
    pub c_idx: u8,
    pub mode: u8,
    pub qp: i32,
    pub quality: TrialQuality,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlockEstimate {
    pub plan: BlockPlan,
    pub cost: RdCost,
    pub approx_frac_bits: u64,
    pub confidence: DecisionConfidence,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LumaModePlan {
    pub candidates: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaModePlan {
    pub mode: u8,
    pub mode_idx: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParentChromaPlan {
    pub log2_size: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TtPlan {
    Leaf {
        log2_size: u8,
        trafo_depth: u8,
    },
    Split {
        log2_size: u8,
        trafo_depth: u8,
        kids: Vec<TtPlan>,
        parent_chroma: Option<ParentChromaPlan>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum CuPlan {
    Leaf(CuLeafPlan),
    Split { kids: Vec<CuPlan> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CuLeafPlan {
    pub mpm: [IntraPredMode; 3],
    pub luma_mode: u8,
    pub chroma_mode_idx: u8,
    pub chroma_mode: u8,
    pub tt: TtPlan,
    /// `Some([m0,m1,m2,m3])` when this 8x8 CU is `PartNxN`: the four 4x4-PU luma
    /// modes (z-order). Final coding re-derives each PU's MPM from neighbours and
    /// rebuilds the forced-split TT; `luma_mode` mirrors PU0.
    pub nxn: Option<[u8; 4]>,
}
