//! Forward transform + quantization, and the matching dequant + inverse
//! transform used to reconstruct a transform block.
//!
//! The forward path (residual -> coefficients -> quantized levels) uses a
//! Rust scalar HEVC transform and x265's quantizer formula. The inverse path
//! (levels -> dequantized coefficients -> residual) is written to be
//! **bit-identical to `bpg-hevc-decode`'s** `transform::dequantize` +
//! `transform::inverse_transform`: the dequant scale/shift/rounding mirror
//! H.265 8.6.3 exactly, and reconstruction dispatches through the decoder's
//! Rust inverse transform implementation (verified equal in
//! `tests/transform_recon.rs`). This guarantees that a coefficient block the
//! encoder reconstructs to feed intra prediction matches what the decoder will
//! reconstruct from the same coded levels.
//!
//! Quantization here provides the plain x265 quantizer as the starting point;
//! the encoder layers CABAC-estimated coefficient RD refinement on top before
//! reconstruction. Sign data hiding is not applied (the caller emits residual
//! with `sign_data_hiding = false`).

use crate::primitives;

/// Inverse quant level scale, H.265 Table 8-8 (`g_invQuantScales`).
const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
/// Forward quant scale, x265 `g_quantScales`.
const QUANT_SCALE: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
/// x265 `MAX_TR_DYNAMIC_RANGE`.
const MAX_TR_DYNAMIC_RANGE: i32 = 15;
/// x265 `QUANT_SHIFT`.
const QUANT_SHIFT: i32 = 14;

/// Forward transform a residual block (row-major, stride `1 << log2_size`).
///
/// Uses the 4x4 DST for intra luma 4x4 blocks (H.265 8.6.4.1), DCT otherwise.
/// `bit_depth` selects the matching x265 primitive (8 or 10), whose internal
/// transform shifts are bit-depth specialized.
pub fn forward_transform(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
) -> Vec<i16> {
    primitives::forward_transform(residual, log2_size, is_intra_4x4_luma, bit_depth)
}

/// Forward transform into caller-owned buffers. `out` receives the coefficient
/// block and `tmp` is used only for the intermediate DCT transpose, allowing the
/// encoder to pool both allocations across transform blocks.
pub fn forward_transform_into(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
    out: &mut Vec<i16>,
    tmp: &mut Vec<i16>,
) {
    primitives::forward_transform_into(residual, log2_size, is_intra_4x4_luma, bit_depth, out, tmp);
}

/// Quantize transform coefficients to initial levels using x265's plain
/// quantizer. The encoder may refine these levels with CABAC-estimated RDOQ
/// before reconstruction. Returns `(levels, num_nonzero)`; `levels` is
/// row-major with the same layout as the input coefficients.
pub fn quantize(coeffs: &[i16], log2_size: u8, qp: i32, bit_depth: u8) -> (Vec<i16>, u32) {
    let mut levels = Vec::new();
    let nnz = quantize_into(coeffs, log2_size, qp, bit_depth, &mut levels);
    (levels, nnz)
}

pub fn quantize_into(
    coeffs: &[i16],
    log2_size: u8,
    qp: i32,
    bit_depth: u8,
    levels: &mut Vec<i16>,
) -> u32 {
    let per = qp / 6;
    let rem = (qp % 6) as usize;
    let transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth as i32 - log2_size as i32;
    let qbits = QUANT_SHIFT + per + transform_shift;
    // Intra rounding offset (x265: 171 for I, 85 for P/B), scaled into qbits.
    // `qbits >= 9` (max TU 32x32, max bit depth 12) and `scale <= 26214`, so both
    // fit i32; see [`primitives::quantize`] for the overflow analysis.
    let add = (171i64 << (qbits - 9)) as i32;
    let scale = QUANT_SCALE[rem];

    levels.clear();
    levels.resize(coeffs.len(), 0);
    // Dispatched (scalar or `wide`-SIMD) forward-quant kernel; see [`primitives`].
    primitives::quantize(coeffs, levels, scale, add, qbits)
}

/// Forward-quantizer scale and `qbits` shift for `(log2_size, qp, bit_depth)`,
/// matching [`quantize`]. Exposed so the sign-data-hiding pass can recompute
/// each coefficient's rounding remainder (x265 `deltaU`) from the pre-quant
/// coefficient and its level: `deltaU = (|coeff|*scale - (|level|<<qbits)) >> (qbits-8)`.
pub fn quant_params(log2_size: u8, qp: i32, bit_depth: u8) -> (i64, i32) {
    let per = qp / 6;
    let rem = (qp % 6) as usize;
    let transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth as i32 - log2_size as i32;
    let qbits = QUANT_SHIFT + per + transform_shift;
    (QUANT_SCALE[rem] as i64, qbits)
}

/// Dequantize levels in place (flat scaling list), bit-identical to
/// `bpg-hevc-decode::hevc::transform::dequantize` (H.265 8.6.3).
pub fn dequantize(levels: &mut [i16], log2_size: u8, qp: i32, bit_depth: u8) {
    let per = qp / 6;
    let rem = (qp % 6) as usize;
    let combined = LEVEL_SCALE[rem] * (1 << per);
    let shift = bit_depth as i32 - 9 + log2_size as i32;
    // Dispatched (scalar or `wide`-SIMD) inverse-quant kernel; see [`primitives`].
    primitives::dequantize(levels, combined, shift);
}

/// Per-block dequant constants (combined scale, right-shift, rounding add),
/// derived once from `(log2_size, qp, bit_depth)` so a single level can be
/// dequantized without allocating. `apply` is bit-identical to one iteration
/// of [`dequantize`] — the RDOQ inner loop uses it to evaluate one
/// coefficient's distortion without rebuilding/dequantizing the whole block.
#[derive(Clone, Copy)]
pub struct DequantParams {
    combined: i32,
    shift: i32,
    add: i32,
}

impl DequantParams {
    pub fn new(log2_size: u8, qp: i32, bit_depth: u8) -> Self {
        let per = qp / 6;
        let rem = (qp % 6) as usize;
        let combined = LEVEL_SCALE[rem] * (1 << per);
        let shift = bit_depth as i32 - 9 + log2_size as i32;
        let add = if shift > 0 { 1 << (shift - 1) } else { 0 };
        Self {
            combined,
            shift,
            add,
        }
    }

    /// Dequantize a single level, matching [`dequantize`]'s clamp.
    #[inline]
    pub fn apply(&self, level: i16) -> i32 {
        let value = if self.shift >= 0 {
            (level as i32 * self.combined + self.add) >> self.shift
        } else {
            (level as i32 * self.combined) << (-self.shift)
        };
        value.clamp(-32768, 32767)
    }
}

/// Inverse transform dequantized coefficients to a residual block, using
/// x265's scalar inverse DCT/DST (bit-matched to the decoder for `bit_depth`).
pub fn inverse_transform(
    coeffs: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
) -> Vec<i16> {
    primitives::inverse_transform(coeffs, log2_size, is_intra_4x4_luma, bit_depth)
}

pub fn inverse_transform_into(
    coeffs: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
    out: &mut Vec<i16>,
) {
    primitives::inverse_transform_into(coeffs, log2_size, is_intra_4x4_luma, bit_depth, out);
}

/// Reconstruct the residual a decoder would produce from coded `levels`:
/// dequant (8.6.3) then inverse transform (8.6.4). The encoder adds this to
/// its prediction and clips, exactly as the decoder does, so reconstructed
/// neighbors match.
pub fn reconstruct_residual(
    levels: &[i16],
    log2_size: u8,
    qp: i32,
    bit_depth: u8,
    is_intra_4x4_luma: bool,
) -> Vec<i16> {
    let mut coeffs = levels.to_vec();
    dequantize(&mut coeffs, log2_size, qp, bit_depth);
    inverse_transform(&coeffs, log2_size, is_intra_4x4_luma, bit_depth)
}

pub fn reconstruct_residual_into(
    levels: &[i16],
    log2_size: u8,
    qp: i32,
    bit_depth: u8,
    is_intra_4x4_luma: bool,
    coeffs: &mut Vec<i16>,
    residual: &mut Vec<i16>,
) {
    coeffs.clear();
    coeffs.extend_from_slice(levels);
    dequantize(coeffs, log2_size, qp, bit_depth);
    inverse_transform_into(coeffs, log2_size, is_intra_4x4_luma, bit_depth, residual);
}
