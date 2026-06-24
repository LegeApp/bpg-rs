//! RD pricing helpers.

use crate::cabac::{CabacEstimator, ContextModel};
use crate::contexts::{Contexts, ctx};

/// HEVC luma RD lambda (HM/x265 `0.57 * 2^((qp-12)/3)`), in pixel-SSE units.
pub(super) fn rd_lambda(qp: i32) -> f64 {
    0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0)
}

/// Estimated `split_transform_flag` entropy bits at this node (1/32768-bit
/// units), mirroring the context selection in `write.rs`.
pub(super) fn split_flag_bits(ctxs: &Contexts, log2_size: u8, is_split: bool) -> u64 {
    let ci = ctx::SPLIT_TRANSFORM_FLAG + (5usize.saturating_sub(log2_size as usize)).min(2);
    ctxs.models[ci].entropy_bits(is_split as u8) as u64
}

#[inline]
pub(super) fn entropy_bits(model: &ContextModel, bin: u8) -> u64 {
    model.entropy_bits(bin) as u64
}

/// Max rough-search candidates (within the x265 25% threshold) carried into the
/// exact RD luma-mode pass, before MPMs are unioned in. Tunable per effort tier.
pub(super) const ROUGH_RD_CANDS: usize = 4;

/// Estimated `prev_intra_luma_pred_flag` + `mpm_idx`/`rem_intra_luma_pred_mode`
/// bits (1/32768-bit units) for signaling `mode` given the MPM list, mirroring
/// `write_intra_luma_mode`.
pub(super) fn luma_mode_bits(ctxs: &Contexts, mpm_u8: [u8; 3], mode: u8) -> u64 {
    let prev = &ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let bypass = CabacEstimator::SCALE;
    match mpm_u8.iter().position(|&m| m == mode) {
        Some(0) => entropy_bits(prev, 1) + bypass,
        Some(_) => entropy_bits(prev, 1) + 2 * bypass,
        None => entropy_bits(prev, 0) + 5 * bypass,
    }
}

/// Estimated `part_mode` bin bits at the minimum CU size. For intra 8x8 CUs,
/// bin 1 means `2Nx2N` and bin 0 means `PartNxN`.
pub(super) fn part_mode_bits(ctxs: &Contexts, part_nxn: bool) -> u64 {
    entropy_bits(&ctxs.models[ctx::PART_MODE], (!part_nxn) as u8)
}

/// Estimated `intra_chroma_pred_mode` bits for the DM/default path used by the
/// current StillSearch chroma search.
pub(super) fn chroma_dm_bits(ctxs: &Contexts, cat: u8) -> u64 {
    if cat == 0 {
        0
    } else {
        entropy_bits(&ctxs.models[ctx::INTRA_CHROMA_PRED_MODE], 0)
    }
}
