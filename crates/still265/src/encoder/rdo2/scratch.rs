//! Allocation-free working buffers for the staged search (refactor-rdo.md §9).
//!
//! Owned by the [`super::super::Encoder`] and borrowed via `mem::take` on the
//! hot path (mirroring the existing `scratch_*` fields), so the new TU path
//! reuses capacity across candidates instead of heap-allocating per block.
//! [`Default`] yields *empty* vecs (the cheap `mem::take` placeholder); callers
//! `clear()` + `resize()` to the block size, which retains the backing capacity.

#[derive(Default)]
pub(in crate::encoder) struct SearchScratch {
    /// Predicted samples for the block under evaluation (non-committing).
    pub(in crate::encoder) pred: Vec<u16>,
    /// Source-minus-prediction residual.
    pub(in crate::encoder) residual: Vec<i16>,
    /// Forward-transform coefficients.
    pub(in crate::encoder) coeffs: Vec<i16>,
    /// Quantized levels for cheap/plain-quant non-committing trials. Exact or
    /// committed evaluations still move their levels into `CodedBlock`.
    pub(in crate::encoder) levels: Vec<i16>,
    /// Scratch-owned RDOQ working buffers for exact rdo2 trials.
    pub(in crate::encoder) rdoq: crate::rdoq::RdoqScratch,
    /// Scratch-owned exact residual pricing state for rdo2 trials.
    pub(in crate::encoder) residual_pricing: crate::residual::ResidualPricingScratch,
    /// Transform scratch, and after `reconstruct_residual_into` it holds the
    /// reconstructed residual (used for clipped scratch distortion / commit).
    pub(in crate::encoder) transform_tmp: Vec<i16>,
    /// Reused trace candidate records for the luma ranked-candidate screen.
    pub(in crate::encoder) luma_trace_cands: Vec<crate::trace::CandRec>,
    /// Reused trace candidate records for nested luma exact escalation.
    pub(in crate::encoder) luma_exact_trace_cands: Vec<crate::trace::CandRec>,
    /// Reused trace candidate records for the chroma RD decision.
    pub(in crate::encoder) chroma_trace_cands: Vec<crate::trace::CandRec>,
}
