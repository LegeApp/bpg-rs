//! YCbCr→RGB color conversion and chroma upsampling.
//!
//! Coefficients and the Lanczos upsampling filter are derived to be bit-exact
//! with libbpg/bpgdec (see [`get_coefficients`], [`upsample_chroma_420`], and
//! [`upsample_chroma_422`]).

fn lrint(x: f64) -> i32 {
    x.round_ties_even() as i32
}

/// Apply `f(row_index, row_slice)` to each `row_len`-element row of `buf`. With
/// the `rayon` feature the rows are processed in parallel across cores;
/// otherwise sequentially. Rows are independent, so the result is identical
/// either way. This is the data-parallel lever for the color-output kernels:
/// for cross-channel per-pixel conversions (YCbCr→RGB) lane-wise SIMD helps
/// little, but distributing whole rows across cores does.
#[inline]
pub(crate) fn for_each_row<T, F>(buf: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    if row_len == 0 {
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, r)| f(i, r));
    }
    #[cfg(not(feature = "rayon"))]
    {
        buf.chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, r)| f(i, r));
    }
}

#[inline]
fn scale_component_to_u8(v: i32, bit_depth: u8, full_range: bool) -> u8 {
    if bit_depth == 8 && full_range {
        return v.clamp(0, 255) as u8;
    }
    let out_pixel_max = 255.0;
    let c_shift = 30 - 8;
    let mult = out_pixel_max * (1u32 << c_shift) as f64 / ((1u32 << bit_depth) - 1) as f64;
    let (y_one, y_offset) = if full_range {
        (lrint(mult), 1 << (c_shift - 1))
    } else {
        let mult_y = out_pixel_max * (1u32 << c_shift) as f64 / (219u32 << (bit_depth - 8)) as f64;
        (
            lrint(mult_y),
            -(16i32 << (bit_depth - 8)) * lrint(mult_y) + (1 << (c_shift - 1)),
        )
    };
    ((v * y_one + y_offset) >> c_shift).clamp(0, 255) as u8
}

/// Convert one decoded BPG pixel to RGB.
///
/// `matrix_coeffs` follows BPG/HEVC semantics: 0 = RGB stored as G/B/R planes,
/// 8 = YCgCo, 1/6/9 = YCbCr matrices.
pub fn pixel_to_rgb(
    y_val: i32,
    cb_val: i32,
    cr_val: i32,
    bit_depth: u8,
    full_range: bool,
    matrix_coeffs: u8,
) -> (u8, u8, u8) {
    match matrix_coeffs {
        0 => (
            scale_component_to_u8(cr_val, bit_depth, full_range),
            scale_component_to_u8(y_val, bit_depth, full_range),
            scale_component_to_u8(cb_val, bit_depth, full_range),
        ),
        8 => {
            let center = 1i32 << (bit_depth - 1);
            let cg = cb_val - center;
            let co = cr_val - center;
            (
                scale_component_to_u8(y_val - cg + co, bit_depth, full_range),
                scale_component_to_u8(y_val + cg, bit_depth, full_range),
                scale_component_to_u8(y_val - cg - co, bit_depth, full_range),
            )
        }
        _ => ycbcr_pixel_to_rgb(y_val, cb_val, cr_val, full_range, matrix_coeffs),
    }
}

#[inline(always)]
pub fn ycbcr_pixel_to_rgb(
    y_val: i32,
    cb_val: i32,
    cr_val: i32,
    full_range: bool,
    matrix_coeffs: u8,
) -> (u8, u8, u8) {
    let (cr_r, cb_g, cr_g, cb_b, y_bias, y_scale, rnd, shr) =
        get_coefficients(full_range, matrix_coeffs);
    let cb = cb_val - 128;
    let cr = cr_val - 128;
    let yv = (y_val - y_bias) * y_scale;
    let r = (yv + cr_r * cr + rnd) >> shr;
    let g = (yv + cb_g * cb + cr_g * cr + rnd) >> shr;
    let b = (yv + cb_b * cb + rnd) >> shr;
    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

/// libbpg's 7-tap phase-0.5 Lanczos interpolation kernel (`IP1C0..IP1C6`), used
/// for 4:2:0 chroma upsampling on the normal `BPG_FORMAT_420` path (`c_h_phase=1`,
/// chroma sited at the centre of the 2×2 luma quad). Coefficients sum to 64. The
/// *even* output sample uses the reversed pattern (C6..C0) and the *odd* output
/// uses the forward pattern (C0..C6) — see libbpg `interp2p1_simple`.
const LANCZOS_P1: [i32; 7] = [-1, 4, -10, 57, 18, -6, 2];

/// `LANCZOS_P1` reversed (C6..C0). The *even* output phase uses the reversed
/// pattern; hoisting this choice out of the 7-tap inner loop (instead of a
/// per-tap `if even`) removes a branch from the hot path and lets the compiler
/// autovectorize the accumulation. Bit-identical to the per-tap selection.
const LANCZOS_P1_REV: [i32; 7] = [2, -6, 18, 57, -10, 4, -1];

/// Upsample a half-resolution 4:2:0 chroma plane to full luma resolution using
/// libbpg's separable 7-tap phase-0.5 Lanczos filter — bit-exact with bpgdec.
///
/// `src` holds `h2` rows of stride `c_stride` (only the first `w2` columns are
/// meaningful); samples are downshifted by `shift` to 8-bit during the gather, so
/// the returned `out_w * out_h` plane (row-major, stride `out_w`) is 8-bit and
/// downstream conversion must not shift it again. Edges are clamped (replicated),
/// matching libbpg's `interp2_h`/`interp2_vh` padding.
pub fn upsample_chroma_420(
    src: &[u16],
    w2: usize,
    h2: usize,
    c_stride: usize,
    out_w: usize,
    out_h: usize,
    shift: u32,
) -> Vec<u16> {
    // Vertical pass → (w2 × out_h) intermediate, held at the ~×64 filter scale
    // (8-bit input means no rounding/downshift here, exactly as interp2_vh).
    // Output rows are independent → parallelized by `for_each_row`.
    let mut vtmp = vec![0i32; w2 * out_h];
    for_each_row(&mut vtmp, w2, |y, vrow| {
        let y2 = (y >> 1) as isize;
        let even = (y & 1) == 0;
        let mut rows = [0usize; 7];
        for (k, r) in rows.iter_mut().enumerate() {
            let ry = (y2 + k as isize - 3).clamp(0, h2 as isize - 1) as usize;
            *r = ry * c_stride;
        }
        let coeffs = if even { &LANCZOS_P1_REV } else { &LANCZOS_P1 };
        vertical_pass_row(src, &rows, coeffs, shift, vrow);
    });
    // Horizontal pass → (out_w × out_h). The second ×64 stage combines with the
    // vertical one for an overall /4096 (>>12, +2048 rounding), then clamp to 8-bit.
    let mut out = vec![0u16; out_w * out_h];
    for_each_row(&mut out, out_w, |y, orow| {
        let vrow = y * w2;
        for x in 0..out_w {
            let x2 = (x >> 1) as isize;
            let coeffs = if (x & 1) == 0 {
                &LANCZOS_P1_REV
            } else {
                &LANCZOS_P1
            };
            let mut acc = 0i32;
            for k in 0..7 {
                let cx = (x2 + k as isize - 3).clamp(0, w2 as isize - 1) as usize;
                acc += vtmp[vrow + cx] * coeffs[k];
            }
            orow[x] = ((acc + 2048) >> 12).clamp(0, 255) as u16;
        }
    });
    out
}

/// One row of the 4:2:0 separable-Lanczos **vertical** pass: for each output
/// column `x`, accumulate `sum_k (src[rows[k] + x] >> shift) * coeffs[k]`. The
/// columns are contiguous in `src`, so with the `wide-simd` feature this is the
/// one chroma kernel that vectorizes cleanly (gather-free); the scalar path is
/// the bit-exact reference and the SIMD path performs the identical integer
/// multiply-accumulate, lane for lane.
#[inline]
fn vertical_pass_row(
    src: &[u16],
    rows: &[usize; 7],
    coeffs: &[i32; 7],
    shift: u32,
    out: &mut [i32],
) {
    let w2 = out.len();
    #[cfg(feature = "wide-simd")]
    {
        use wide::i32x8;
        let cvec: [i32x8; 7] = core::array::from_fn(|k| i32x8::splat(coeffs[k]));
        let mut x = 0;
        while x + 8 <= w2 {
            let mut acc = i32x8::ZERO;
            for k in 0..7 {
                let base = rows[k] + x;
                let lanes: [i32; 8] = core::array::from_fn(|i| (src[base + i] >> shift) as i32);
                acc += i32x8::from(lanes) * cvec[k];
            }
            out[x..x + 8].copy_from_slice(&acc.to_array());
            x += 8;
        }
        while x < w2 {
            let mut acc = 0i32;
            for k in 0..7 {
                acc += (src[rows[k] + x] >> shift) as i32 * coeffs[k];
            }
            out[x] = acc;
            x += 1;
        }
    }
    #[cfg(not(feature = "wide-simd"))]
    {
        for x in 0..w2 {
            let mut acc = 0i32;
            for k in 0..7 {
                acc += (src[rows[k] + x] >> shift) as i32 * coeffs[k];
            }
            out[x] = acc;
        }
    }
}

/// Upsample a half-horizontal-resolution 4:2:2 chroma plane to full luma width
/// using libbpg's 7-tap phase-0.5 Lanczos filter (`interp2_h` with
/// `c_h_phase=1`, the normal JPEG/BPG 4:2:2 siting).
///
/// This is the horizontal half of [`upsample_chroma_420`]: rows are copied
/// one-for-one, only columns are interpolated. Samples are downshifted to 8-bit
/// during the gather so the returned `out_w * out_h` plane is ready for the
/// fixed 8-bit RGB conversion path.
pub fn upsample_chroma_422(
    src: &[u16],
    w2: usize,
    h: usize,
    c_stride: usize,
    out_w: usize,
    out_h: usize,
    shift: u32,
) -> Vec<u16> {
    let mut out = vec![0u16; out_w * out_h];
    if w2 == 0 || h == 0 {
        return out;
    }
    for_each_row(&mut out, out_w, |y, orow| {
        let sy = y.min(h - 1);
        let srow = sy * c_stride;
        for x in 0..out_w {
            let x2 = (x >> 1) as isize;
            let coeffs = if (x & 1) == 0 {
                &LANCZOS_P1_REV
            } else {
                &LANCZOS_P1
            };
            let mut acc = 0i32;
            for k in 0..7 {
                let cx = (x2 + k as isize - 3).clamp(0, w2 as isize - 1) as usize;
                acc += (src[srow + cx] >> shift) as i32 * coeffs[k];
            }
            orow[x] = ((acc + 32) >> 6).clamp(0, 255) as u16;
        }
    });
    out
}

/// Get color matrix coefficients for YCbCr→RGB conversion.
///
/// Returns (cr_r, cb_g, cr_g, cb_b, y_bias, y_scale, rounding, shift_bits) for
/// the fixed-point pixel formula
///   yv = (y - y_bias) * y_scale
///   r  = (yv + cr_r*(cr-128) + rnd) >> shr
///   g  = (yv + cb_g*(cb-128) + cr_g*(cr-128) + rnd) >> shr
///   b  = (yv + cb_b*(cb-128) + rnd) >> shr
///
/// Coefficients are derived exactly as libbpg's `convert_init`
/// (30 - out_bit_depth = 22-bit fixed point for 8-bit output) so our output is
/// bit-exact with bpgdec. `cb_g`/`cr_g` are returned negated (libbpg stores them
/// positive and subtracts; we add). Computed for 8-bit in/out — chroma/luma are
/// downshifted to 8-bit before this call.
#[inline]
fn get_coefficients(
    full_range: bool,
    matrix_coeffs: u8,
) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
    const C_SHIFT: i32 = 22; // 30 - out_bit_depth(8)
    const PIXEL_MAX: f64 = 255.0; // (1 << 8) - 1, in == out == 8 bit
    let scale = (1i64 << C_SHIFT) as f64;
    let mult = PIXEL_MAX * scale / PIXEL_MAX; // == scale
    let (mult_y, mult_c) = if full_range {
        (mult, mult)
    } else {
        (PIXEL_MAX * scale / 219.0, PIXEL_MAX * scale / 224.0)
    };
    let (k_r, k_b) = match matrix_coeffs {
        1 => (0.2126, 0.0722), // BT.709
        9 => (0.2627, 0.0593), // BT.2020
        _ => (0.299, 0.114),   // BT.601
    };
    let cr_r = lrint(2.0 * (1.0 - k_r) * mult_c);
    let cb_g = lrint(2.0 * k_b * (1.0 - k_b) / (1.0 - k_b - k_r) * mult_c);
    let cr_g = lrint(2.0 * k_r * (1.0 - k_r) / (1.0 - k_b - k_r) * mult_c);
    let cb_b = lrint(2.0 * (1.0 - k_b) * mult_c);
    let c_one = lrint(mult);
    let c_rnd = 1 << (C_SHIFT - 1);
    let (y_bias, y_scale) = if full_range {
        (0, c_one)
    } else {
        (16, lrint(mult_y))
    };
    (cr_r, -cb_g, -cr_g, cb_b, y_bias, y_scale, c_rnd, C_SHIFT)
}

/// True for the `matrix_coefficients` values that [`transcode_to_jfif_ycbcr`]
/// can convert directly in the YCbCr domain (BT.601 525/625, BT.709, BT.2020
/// non-constant-luma). RGB-stored (0) and YCgCo (8) are not YCbCr matrices and
/// must use the RGB output path.
pub fn is_ycbcr_matrix(matrix_coeffs: u8) -> bool {
    matches!(matrix_coeffs, 1 | 4 | 5 | 6 | 7 | 9)
}

/// Source YCbCr (8-bit samples) → RGB at 8-bit scale **without** the [0,255]
/// clamp, using the same tested fixed-point inverse as [`ycbcr_pixel_to_rgb`].
/// Skipping the clamp means a subsequent re-encode to a different YCbCr matrix
/// loses nothing to gamut clipping. Valid only for [`is_ycbcr_matrix`] inputs.
#[inline]
fn ycbcr_to_rgb_unclamped(
    y8: i32,
    cb8: i32,
    cr8: i32,
    full_range: bool,
    matrix_coeffs: u8,
) -> (i32, i32, i32) {
    let (cr_r, cb_g, cr_g, cb_b, y_bias, y_scale, rnd, shr) =
        get_coefficients(full_range, matrix_coeffs);
    let cb = cb8 - 128;
    let cr = cr8 - 128;
    let yv = (y8 - y_bias) * y_scale;
    let r = (yv + cr_r * cr + rnd) >> shr;
    let g = (yv + cb_g * cb + cr_g * cr + rnd) >> shr;
    let b = (yv + cb_b * cb + rnd) >> shr;
    (r, g, b)
}

/// RGB (8-bit scale, may fall outside [0,255]) → JFIF (BT.601 full-range) 8-bit
/// YCbCr — the color space baseline JPEG/JFIF mandates. Uses the standard JFIF
/// forward matrix (identical coefficients to TooJpeg's `rgb2y/rgb2cb/rgb2cr`).
#[inline]
fn rgb_to_jfif_ycbcr(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as f32, g as f32, b as f32);
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = -0.168_736 * rf - 0.331_264 * gf + 0.5 * bf + 128.0;
    let cr = 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf + 128.0;
    let clamp8 = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    (clamp8(y), clamp8(cb), clamp8(cr))
}

/// Transcode one 8-bit source YCbCr sample to JFIF (BT.601 full-range) 8-bit
/// YCbCr, the JPEG baseline. This is the color-space-preserving alternative to a
/// full RGB output buffer for re-encoding BT.709/BT.2020/limited-range frames to
/// JPEG: the RGB triple exists only transiently and unclamped, so the only loss
/// versus a perfect transform is final 8-bit YCbCr rounding (no gamut clipping,
/// no RGB byte round-trip). Equivalent in result to the RGB path for in-gamut
/// pixels; never materializes a clamped RGB image.
#[inline]
pub fn transcode_to_jfif_ycbcr(
    y8: i32,
    cb8: i32,
    cr8: i32,
    full_range: bool,
    matrix_coeffs: u8,
) -> (u8, u8, u8) {
    let (r, g, b) = ycbcr_to_rgb_unclamped(y8, cb8, cr8, full_range, matrix_coeffs);
    rgb_to_jfif_ycbcr(r, g, b)
}

/// Scale a single luma sample at the frame's native `bit_depth` to a full-range
/// 8-bit value (JFIF expects full-range luma). For full-range 8-bit input this
/// is a clamp; for limited range it expands 16..235 → 0..255. Used for the
/// grayscale JPEG path.
pub fn luma_to_jfif_8bit(v: i32, bit_depth: u8, full_range: bool) -> u8 {
    scale_component_to_u8(v, bit_depth, full_range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_flat_plane_is_constant() {
        // A flat chroma plane must upsample to the same constant everywhere: the
        // 7 taps sum to 64, so two stages (/64 each, +2048 rounding >>12) recover
        // the input value exactly, including the replicated edges.
        let (w2, h2) = (5usize, 4usize);
        let src = vec![137u16; w2 * h2];
        let out = upsample_chroma_420(&src, w2, h2, w2, w2 * 2, h2 * 2, 0);
        assert!(out.iter().all(|&v| v == 137), "flat plane changed value");
        assert_eq!(out.len(), (w2 * 2) * (h2 * 2));
    }

    #[test]
    fn upsample_matches_libbpg_kernel_on_a_ramp() {
        // 1-D horizontal check against a hand-evaluated libbpg interp2p1 step.
        // Single row [0, 64, 128, 192] → 8 output samples. Interior sample x=2
        // (even, centred on src col 1) uses reversed taps C6..C0 over clamped
        // cols [-2..4]→[0,0,0,1,2,3,3]; x=3 (odd) uses C0..C6 over the same.
        let src = [0u16, 64, 128, 192];
        let out = upsample_chroma_420(&src, 4, 1, 4, 8, 2, 0);
        let c = LANCZOS_P1; // [-1,4,-10,57,18,-6,2]
        let col = |i: isize| src[i.clamp(0, 3) as usize] as i32;
        // x=2: x2=1, even → reversed taps over cols (1-3..1+3)
        let even = (0..7)
            .map(|k| col(1 + k as isize - 3) * c[6 - k])
            .sum::<i32>();
        let exp2 = (((even * 64) + 2048) >> 12).clamp(0, 255) as u16;
        // x=3: x2=1, odd → forward taps
        let odd = (0..7).map(|k| col(1 + k as isize - 3) * c[k]).sum::<i32>();
        let exp3 = (((odd * 64) + 2048) >> 12).clamp(0, 255) as u16;
        assert_eq!(out[2], exp2);
        assert_eq!(out[3], exp3);
        // Endpoints stay within range and near the clamped edges.
        assert!(out[0] <= 16 && out[7] >= 176);
    }

    #[test]
    fn upsample_422_matches_horizontal_libbpg_kernel_on_a_ramp() {
        let src = [0u16, 64, 128, 192];
        let out = upsample_chroma_422(&src, 4, 1, 4, 8, 1, 0);
        let c = LANCZOS_P1;
        let col = |i: isize| src[i.clamp(0, 3) as usize] as i32;
        let even = (0..7)
            .map(|k| col(1 + k as isize - 3) * c[6 - k])
            .sum::<i32>();
        let odd = (0..7).map(|k| col(1 + k as isize - 3) * c[k]).sum::<i32>();
        assert_eq!(out[2], ((even + 32) >> 6).clamp(0, 255) as u16);
        assert_eq!(out[3], ((odd + 32) >> 6).clamp(0, 255) as u16);
    }

    /// Independent scalar reference for the full 4:2:0 separable filter, written
    /// without the parity-hoist or SIMD, used to prove the production kernel is
    /// bit-identical. Under `--features wide-simd` this exercises the `wide`
    /// vertical pass (input is wide enough to fill 8-lane vectors).
    fn reference_420(src: &[u16], w2: usize, h2: usize, shift: u32) -> Vec<u16> {
        let c = LANCZOS_P1;
        let out_w = w2 * 2;
        let out_h = h2 * 2;
        let mut vtmp = vec![0i32; w2 * out_h];
        for y in 0..out_h {
            let y2 = (y >> 1) as isize;
            let even = (y & 1) == 0;
            for x in 0..w2 {
                let mut acc = 0i32;
                for k in 0..7 {
                    let ry = (y2 + k as isize - 3).clamp(0, h2 as isize - 1) as usize;
                    let s = (src[ry * w2 + x] >> shift) as i32;
                    acc += s * if even { c[6 - k] } else { c[k] };
                }
                vtmp[y * w2 + x] = acc;
            }
        }
        let mut out = vec![0u16; out_w * out_h];
        for y in 0..out_h {
            for x in 0..out_w {
                let x2 = (x >> 1) as isize;
                let even = (x & 1) == 0;
                let mut acc = 0i32;
                for k in 0..7 {
                    let cx = (x2 + k as isize - 3).clamp(0, w2 as isize - 1) as usize;
                    acc += vtmp[y * w2 + cx] * if even { c[6 - k] } else { c[k] };
                }
                out[y * out_w + x] = ((acc + 2048) >> 12).clamp(0, 255) as u16;
            }
        }
        out
    }

    #[test]
    fn transcode_bt601_full_range_is_near_identity() {
        // BT.601 full-range → RGB → JFIF(BT.601 full-range) round-trips to itself
        // (within float rounding) since source and dest matrices are identical.
        for y in (0..=255).step_by(17) {
            for cb in (0..=255).step_by(51) {
                for cr in (0..=255).step_by(51) {
                    let (yy, ccb, ccr) = transcode_to_jfif_ycbcr(y, cb, cr, true, 6);
                    assert!((yy as i32 - y).abs() <= 1, "Y {y}->{yy}");
                    assert!((ccb as i32 - cb).abs() <= 1, "Cb {cb}->{ccb}");
                    assert!((ccr as i32 - cr).abs() <= 1, "Cr {cr}->{ccr}");
                }
            }
        }
    }

    #[test]
    fn transcode_matches_explicit_rgb_path() {
        // The YCbCr-domain transcode must agree with the explicit
        // YCbCr→RGB(clamped)→JFIF path for in-gamut samples (≤1 LSB), proving it
        // is a faithful, lossless-where-it-counts replacement for the RGB route.
        let jfif = |r: i32, g: i32, b: i32| {
            let (rf, gf, bf) = (r as f32, g as f32, b as f32);
            let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
            let cb = -0.168_736 * rf - 0.331_264 * gf + 0.5 * bf + 128.0;
            let cr = 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf + 128.0;
            (
                y.round().clamp(0.0, 255.0) as i32,
                cb.round().clamp(0.0, 255.0) as i32,
                cr.round().clamp(0.0, 255.0) as i32,
            )
        };
        let mut checked = 0;
        for &mc in &[1u8, 6, 9] {
            for &fr in &[true, false] {
                for y in (16..=240).step_by(16) {
                    for cb in (16..=240).step_by(16) {
                        for cr in (16..=240).step_by(16) {
                            // Only compare where RGB is in-gamut: out-of-[0,255]
                            // is exactly where the clamping RGB path is *lossy*
                            // and the transcode (intentionally) preserves more.
                            let (ur, ug, ub) = ycbcr_to_rgb_unclamped(y, cb, cr, fr, mc);
                            if [ur, ug, ub].iter().any(|&v| !(0..=255).contains(&v)) {
                                continue;
                            }
                            let (er, eg, eb) = jfif(ur, ug, ub);
                            let (ty, tcb, tcr) = transcode_to_jfif_ycbcr(y, cb, cr, fr, mc);
                            assert!((ty as i32 - er).abs() <= 1, "Y mc={mc} fr={fr}");
                            assert!((tcb as i32 - eg).abs() <= 1, "Cb mc={mc} fr={fr}");
                            assert!((tcr as i32 - eb).abs() <= 1, "Cr mc={mc} fr={fr}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(
            checked > 1000,
            "too few in-gamut samples checked: {checked}"
        );
    }

    #[test]
    fn upsample_420_bit_identical_to_reference_on_wide_block() {
        // 40×6 chroma block (w2=40 > 8 lanes) with a deterministic pattern, so
        // the SIMD vertical pass and its scalar tail are both exercised.
        let (w2, h2) = (40usize, 6usize);
        let mut src = vec![0u16; w2 * h2];
        for (i, s) in src.iter_mut().enumerate() {
            *s = ((i * 37 + 11) % 256) as u16;
        }
        let got = upsample_chroma_420(&src, w2, h2, w2, w2 * 2, h2 * 2, 0);
        let want = reference_420(&src, w2, h2, 0);
        assert_eq!(got, want, "production 4:2:0 kernel diverged from reference");
    }
}
