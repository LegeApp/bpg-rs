//! RD pricing helpers.

use crate::cabac::{CabacEstimator, ContextModel};
use crate::contexts::{Contexts, ctx};

/// HEVC luma RD lambda (HM/x265 `0.57 * 2^((qp-12)/3)`), in pixel-SSE units.
///
/// `BPG_STILLSEARCH_LAMBDA_SCALE` is an explicit quality/bitrate diagnostic
/// knob: values below 1.0 spend more bits for lower distortion, values above
/// 1.0 save bits. It is deliberately process-global/cached like the other
/// StillSearch env gates; set it before starting an encode.
pub(super) fn rd_lambda(qp: i32) -> f64 {
    let base = 0.57f64 * 2f64.powf((qp as f64 - 12.0) / 3.0);
    base * super::env::lambda_scale()
}

/// x265 SAD-domain lambda used by `calcRdSADCost` during RMD/SA8D ranking.
#[inline]
pub(super) fn x265_sad_lambda(qp: i32, bit_depth: u8) -> f64 {
    let depth_scale = 1u32 << bit_depth.saturating_sub(8);
    2f64.powf(qp as f64 / 6.0 - 2.0) * f64::from(depth_scale)
}

/// x265's integer SAD-domain lambda (`floor(256 * x265_lambda_tab[qp])`).
#[inline]
pub(super) fn x265_sad_lambda_q8(qp: i32, bit_depth: u8) -> u64 {
    (256.0 * x265_sad_lambda(qp, bit_depth)).floor() as u64
}

/// x265 `RDCost::calcRdSADCost`: `sad + ((bits * lambda + 128) >> 8)`.
#[inline]
pub(super) fn x265_rd_sad_cost(sad: u32, bits: u32, qp: i32, bit_depth: u8) -> u64 {
    sad as u64 + ((bits as u64 * x265_sad_lambda_q8(qp, bit_depth) + 128) >> 8)
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

/// Diagnostic approximation of x265's RMD luma-mode bit estimate.
///
/// x265's `bitsCodeBin()` returns whole bits from the current CABAC fractional
/// carry: `(m_fracBits & 32767) + entropy_bits(ctx, bin) >> 15`, then adds
/// bypass bits for `mpm_idx` / `rem_intra_luma_pred_mode`. Still265's rough
/// search normally uses full fractional precision. This helper intentionally
/// assumes zero carry so audits can separate the effect of integer-bit RMD
/// shaping from the rest of the search without changing canonical scoring.
pub(super) fn x265_rmd_luma_mode_bits_no_carry(ctxs: &Contexts, mpm_u8: [u8; 3], mode: u8) -> u32 {
    x265_rmd_luma_mode_bits_with_carry(ctxs, mpm_u8, mode, 0)
}

/// Same as [`x265_rmd_luma_mode_bits_no_carry`], with an explicit
/// `m_fracBits & 32767` carry from x265's `Entropy::bitsCodeBin()`.
pub(super) fn x265_rmd_luma_mode_bits_with_carry(
    ctxs: &Contexts,
    mpm_u8: [u8; 3],
    mode: u8,
    carry: u32,
) -> u32 {
    let prev = &ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let carry = carry & 32767;
    match mpm_u8.iter().position(|&m| m == mode) {
        Some(0) => ((entropy_bits(prev, 1) as u32 + carry) >> 15) + 1,
        Some(_) => ((entropy_bits(prev, 1) as u32 + carry) >> 15) + 2,
        None => ((entropy_bits(prev, 0) as u32 + carry) >> 15) + 5,
    }
}

/// Carry values where x265's whole-bit RMD estimate can change for the current
/// `prev_intra_luma_pred_flag` context. Testing just these representatives
/// brackets every possible `m_fracBits & 32767` state.
pub(super) fn x265_rmd_luma_mode_carry_representatives(ctxs: &Contexts) -> [u32; 4] {
    let prev = &ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    [
        0,
        carry_transition(entropy_bits(prev, 0) as u32),
        carry_transition(entropy_bits(prev, 1) as u32),
        32767,
    ]
}

#[inline]
fn carry_transition(bits: u32) -> u32 {
    let low = bits & 32767;
    if low == 0 { 0 } else { 32768 - low }
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
