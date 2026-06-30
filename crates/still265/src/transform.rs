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

/// x265's RDOQ measures distortion in the transform-coefficient domain, not
/// directly in pixel SSE. In `Quant::rdoQuant` it scales each coefficient
/// squared error by:
///
/// ```text
/// scaleBits = SCALE_BITS - 2 * transformShift
/// transformShift = MAX_TR_DYNAMIC_RANGE - bit_depth - log2TrSize
/// SCALE_BITS = 15
/// ```
///
/// before adding `lambda2 * CABAC_bits`. Our RDOQ code keeps the coefficient
/// squared error unscaled (`(coeff - dequant(level))^2`), so the equivalent
/// operation is to divide the pixel-domain lambda by `2^scaleBits`.
///
/// Missing this conversion makes the bit term 32×/128×/512×/2048× too large for
/// 4×4/8×8/16×16/32×32 8-bit transforms, aggressively zeroing coefficients and
/// shifting the effective QP axis coarser than x265.
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

#[cfg(test)]
mod fwd_ref_tests {
    //! Independent reference check of the forward transform's two-stage integer
    //! shift/rounding against `forward_transform`. A per-coefficient mismatch
    //! here would read as the flat, bitrate-independent luma gap to x265.
    use super::*;

    // Canonical HEVC integer DCT matrices (rows = frequencies).
    const DCT4: [[i32; 4]; 4] = [
        [64, 64, 64, 64],
        [83, 36, -36, -83],
        [64, -64, -64, 64],
        [36, -83, 83, -36],
    ];
    // HEVC DST-VII (intra 4x4 luma).
    const DST4: [[i32; 4]; 4] = [
        [29, 55, 74, 84],
        [74, 74, 0, -74],
        [84, -29, -74, 55],
        [55, -84, 74, -29],
    ];
    const DCT8: [[i32; 8]; 8] = [
        [64, 64, 64, 64, 64, 64, 64, 64],
        [89, 75, 50, 18, -18, -50, -75, -89],
        [83, 36, -36, -83, -83, -36, 36, 83],
        [75, -18, -89, -50, 50, 89, 18, -75],
        [64, -64, -64, 64, 64, -64, -64, 64],
        [50, -89, 18, 75, -75, -18, 89, -50],
        [36, -83, 83, -36, -36, 83, -83, 36],
        [18, -50, 75, -89, 89, -75, 50, -18],
    ];

    /// Reference 2-D forward transform with HEVC shifts:
    /// pass1 shift = log2 - 1 + (bd-8); pass2 shift = log2 + 6. Output is
    /// row-major coeff[freq_y * n + freq_x].
    fn reference_fwd(
        residual: &[i16],
        mat: &dyn Fn(usize, usize) -> i32,
        n: usize,
        bd: i32,
    ) -> Vec<i32> {
        let log2 = (n.trailing_zeros()) as i32;
        let shift1 = log2 - 1 + (bd - 8);
        let shift2 = log2 + 6;
        let add1 = if shift1 > 0 { 1i64 << (shift1 - 1) } else { 0 };
        let add2 = 1i64 << (shift2 - 1);
        // Pass 1: transform rows -> tmp[y][kx]
        let mut tmp = vec![0i64; n * n];
        for y in 0..n {
            for kx in 0..n {
                let mut acc = 0i64;
                for x in 0..n {
                    acc += mat(kx, x) as i64 * residual[y * n + x] as i64;
                }
                tmp[y * n + kx] = (acc + add1) >> shift1;
            }
        }
        // Pass 2: transform columns -> out[ky][kx]
        let mut out = vec![0i32; n * n];
        for kx in 0..n {
            for ky in 0..n {
                let mut acc = 0i64;
                for y in 0..n {
                    acc += mat(ky, y) as i64 * tmp[y * n + kx];
                }
                out[ky * n + kx] = ((acc + add2) >> shift2) as i32;
            }
        }
        out
    }

    fn run_case(n: usize, is_dst: bool, seed: u64) {
        let log2 = n.trailing_zeros() as u8;
        // Deterministic pseudo-random residual in [-255,255].
        let mut st = seed;
        let mut res = vec![0i16; n * n];
        for v in res.iter_mut() {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((st >> 33) as i32 % 511 - 255) as i16;
        }
        let rust = forward_transform(&res, log2, is_dst, 8);
        let getm = |k: usize, x: usize| -> i32 {
            if is_dst {
                DST4[k][x]
            } else if n == 4 {
                DCT4[k][x]
            } else {
                DCT8[k][x]
            }
        };
        let reference = reference_fwd(&res, &getm, n, 8);
        let mut max_diff = 0i32;
        let mut n_diff = 0usize;
        for i in 0..n * n {
            let d = (rust[i] as i32 - reference[i]).abs();
            if d != 0 {
                n_diff += 1;
            }
            max_diff = max_diff.max(d);
        }
        eprintln!(
            "fwd_ref n={n} dst={is_dst} seed={seed}: max_diff={max_diff} n_diff={n_diff}/{}",
            n * n
        );
        if max_diff > 0 {
            eprintln!("  rust[0..8]={:?}", &rust[..8.min(rust.len())]);
            eprintln!("  ref [0..8]={:?}", &reference[..8.min(reference.len())]);
        }
        assert_eq!(
            max_diff, 0,
            "rust forward transform != HEVC reference (n={n}, dst={is_dst})"
        );
    }

    #[test]
    fn forward_transform_matches_hevc_reference() {
        for seed in [1u64, 42, 12345] {
            run_case(4, false, seed); // 4x4 DCT
            run_case(4, true, seed); // 4x4 DST (intra luma)
            run_case(8, false, seed); // 8x8 DCT
        }
    }
}
