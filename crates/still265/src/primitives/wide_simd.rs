//! Portable-SIMD primitive kernels built on the `wide` crate (pure Rust, no
//! C/asm). Each kernel here is **bit-identical** to its scalar reference in the
//! parent module — enforced by the `tests` below — and is only selected when
//! the `wide-simd` feature is enabled and `BPG_PRIMITIVES` does not force the
//! scalar path. These mirror the kernel families x265 hand-vectorizes for the
//! still-intra path:
//!
//! - [`ssd_u16`]  ↔ x265 `pixel_ssd`        (`ssd-a.asm`)        — RD distortion.
//! - [`sub_residual`] ↔ x265 `pixel_sub_ps` (`pixel-util8.asm`)  — residual gen.
//! - [`forward_dct_1d`] ↔ x265 `dct4/8/16/32` (`dct8.asm`)       — forward DCT.
//! - [`satd_u16`] ↔ x265 high-bit-depth `pixel_satd` (`pixel-a.asm`) — 10/12-bit
//!   mode-decision cost.
//!
//! All arithmetic stays integer and within proven bounds (see the per-kernel
//! comments), so the SIMD reduction order does not change the result.

use super::round_shift;
use wide::{i16x8, i32x4, i32x8};

/// Reinterpret a `u16` sample slice as `i16`. Callers only pass non-negative
/// samples `< 1 << bit_depth <= 4096`, so the reinterpreted value is
/// numerically identical to the original (and `u16`/`i16` share layout).
#[inline]
fn as_i16(s: &[u16]) -> &[i16] {
    // SAFETY: identical size/alignment; the values are < 4096 so the bit
    // pattern is a valid non-negative `i16`.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const i16, s.len()) }
}

/// SSD over a `size`x`size` block. Bit-identical to
/// [`super::ssd_u16_scalar`].
///
/// Bounds: per-element squared diff `<= 4095^2 ≈ 16.7M`; a single row
/// (`size <= 32`) sums to `<= 535M`, well within `i32`, so the per-row `i32`
/// reduction never overflows and matches the scalar element-order sum exactly.
pub fn ssd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u64 {
    debug_assert!(stride_a >= size && stride_b >= size);
    let mut sse = 0u64;
    for j in 0..size {
        let ra = as_i16(&a[j * stride_a..j * stride_a + size]);
        let rb = as_i16(&b[j * stride_b..j * stride_b + size]);
        let mut acc = i32x8::new([0; 8]);
        let mut i = 0;
        while i + 8 <= size {
            let va = i32x8::from(i16x8::from_slice_unaligned(&ra[i..]));
            let vb = i32x8::from(i16x8::from_slice_unaligned(&rb[i..]));
            let d = va - vb;
            acc = acc + d * d;
            i += 8;
        }
        let mut row = acc.reduce_add() as i64;
        while i < size {
            let d = ra[i] as i64 - rb[i] as i64;
            row += d * d;
            i += 1;
        }
        sse += row as u64;
    }
    sse
}

/// Residual generation (`out = src - pred`, `out` contiguous). Bit-identical to
/// [`super::sub_residual_scalar`].
///
/// Bounds: both samples are in `[0, 4095]`, so each difference fits `i16`
/// without overflow, and the lane-wise `i16` subtraction equals the scalar one.
pub fn sub_residual(
    src: &[u16],
    src_stride: usize,
    pred: &[u16],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    debug_assert!(src_stride >= size && pred_stride >= size && out.len() >= size * size);
    for j in 0..size {
        let s = as_i16(&src[j * src_stride..j * src_stride + size]);
        let p = as_i16(&pred[j * pred_stride..j * pred_stride + size]);
        let o = &mut out[j * size..j * size + size];
        let mut i = 0;
        while i + 8 <= size {
            let d = i16x8::from_slice_unaligned(&s[i..]) - i16x8::from_slice_unaligned(&p[i..]);
            o[i..i + 8].copy_from_slice(&d.to_array());
            i += 8;
        }
        while i < size {
            o[i] = s[i] - p[i];
            i += 1;
        }
    }
}

/// One-dimensional forward DCT row (dot products against `matrix`), then
/// `round_shift`. Bit-identical to [`super::forward_dct_1d`].
///
/// Bounds: `|matrix[k]| <= 90` and `|src[i]| <= 32767`, so each product is
/// `<= ~3M` and the length-`n` (`<= 32`) accumulation is `<= ~94M`, within
/// `i32`. Integer addition is associative with no overflow, so the SIMD
/// tree-reduction equals the scalar left-to-right sum.
pub fn forward_dct_1d(src: &[i16], dst: &mut [i16], matrix: &[i32], n: usize, shift: i32) {
    let mut s = [0i32; 32];
    for i in 0..n {
        s[i] = src[i] as i32;
    }
    for k in 0..n {
        let row = &matrix[k * n..k * n + n];
        let mut acc = i32x8::new([0; 8]);
        let mut i = 0;
        while i + 8 <= n {
            let m = i32x8::new(<[i32; 8]>::try_from(&row[i..i + 8]).unwrap());
            let v = i32x8::new(<[i32; 8]>::try_from(&s[i..i + 8]).unwrap());
            acc = acc + m * v;
            i += 8;
        }
        let mut sum = acc.reduce_add();
        while i < n {
            sum += row[i] * s[i];
            i += 1;
        }
        dst[k] = round_shift(sum, shift);
    }
}

/// One pass of the HEVC 4x4 Hadamard butterfly applied lane-wise across the
/// four row vectors (i.e. `M * D` where `M` is the order-4 Walsh-Hadamard
/// matrix). Matches the per-axis butterfly in [`super::hadamard4_satd`].
#[inline]
fn butterfly4(r0: i32x4, r1: i32x4, r2: i32x4, r3: i32x4) -> [i32x4; 4] {
    let a0 = r0 + r3;
    let a1 = r1 + r2;
    let a2 = r1 - r2;
    let a3 = r0 - r3;
    [a0 + a1, a3 + a2, a0 - a1, a3 - a2]
}

/// SATD of one 4x4 tile. Computes `(M·Dᵀ·Mᵀ)`, whose 16 entries are the
/// transpose of the scalar's `M·D·Mᵀ` (so the absolute-value sum is identical),
/// using a SIMD vertical butterfly → 4x4 transpose → vertical butterfly.
#[inline]
fn satd_4x4(a: &[u16], sa: usize, b: &[u16], sb: usize, by: usize, bx: usize) -> u32 {
    let load = |y: usize| -> i32x4 {
        let oa = (by + y) * sa + bx;
        let ob = (by + y) * sb + bx;
        i32x4::new([
            a[oa] as i32 - b[ob] as i32,
            a[oa + 1] as i32 - b[ob + 1] as i32,
            a[oa + 2] as i32 - b[ob + 2] as i32,
            a[oa + 3] as i32 - b[ob + 3] as i32,
        ])
    };

    // Pass 1: vertical butterfly across the four diff rows → M·D.
    let s = butterfly4(load(0), load(1), load(2), load(3));
    // Transpose, then a second vertical butterfly applies M along the other
    // axis → M·Dᵀ·Mᵀ = (M·D·Mᵀ)ᵀ.
    let t = i32x4::transpose(s);
    let f = butterfly4(t[0], t[1], t[2], t[3]);

    let total = f[0].abs().reduce_add()
        + f[1].abs().reduce_add()
        + f[2].abs().reduce_add()
        + f[3].abs().reduce_add();
    ((total as u32) + 1) >> 1
}

/// SATD for a 10/12-bit block (tiled 4x4 Hadamard). Bit-identical to
/// [`super::satd_u16_scalar`].
///
/// Bounds: a 12-bit diff is `<= 4095`; a 4x4 2-D Hadamard coefficient is at
/// most `16*4095 = 65520`, so the 16-entry per-tile sum (`<= ~1.05M`) and the
/// per-vector `i32` reductions never overflow.
pub fn satd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    debug_assert!(stride_a >= size && stride_b >= size);
    let mut sum = 0u32;
    for by in (0..size).step_by(4) {
        for bx in (0..size).step_by(4) {
            sum += satd_4x4(a, stride_a, b, stride_b, by, bx);
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::super::{
        dct_matrix, forward_dct_1d as scalar_dct_1d, satd_u16_scalar, ssd_u16_scalar,
        sub_residual_scalar,
    };

    /// Deterministic xorshift32 PRNG (no external dep).
    fn rng(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    #[test]
    fn ssd_matches_scalar() {
        let mut st = 0x1234_5678u32;
        for &bd in &[8u32, 10, 12] {
            let max = (1u32 << bd) - 1;
            for &size in &[4usize, 8, 16, 32] {
                for _ in 0..40 {
                    let sa = size + (rng(&mut st) as usize & 7);
                    let sb = size + (rng(&mut st) as usize & 7);
                    let mut a = vec![0u16; sa * size];
                    let mut b = vec![0u16; sb * size];
                    for v in a.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    for v in b.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    let want = ssd_u16_scalar(&a, sa, &b, sb, size);
                    let got = super::ssd_u16(&a, sa, &b, sb, size);
                    assert_eq!(got, want, "bd={bd} size={size}");
                }
            }
        }
    }

    #[test]
    fn sub_residual_matches_scalar() {
        let mut st = 0x9e37_79b9u32;
        for &bd in &[8u32, 10, 12] {
            let max = (1u32 << bd) - 1;
            for &size in &[4usize, 8, 16, 32] {
                for _ in 0..40 {
                    let ss = size + (rng(&mut st) as usize & 7);
                    let ps = size + (rng(&mut st) as usize & 7);
                    let mut src = vec![0u16; ss * size];
                    let mut pred = vec![0u16; ps * size];
                    for v in src.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    for v in pred.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    let mut want = vec![0i16; size * size];
                    let mut got = vec![0i16; size * size];
                    sub_residual_scalar(&src, ss, &pred, ps, &mut want, size);
                    super::sub_residual(&src, ss, &pred, ps, &mut got, size);
                    assert_eq!(got, want, "bd={bd} size={size}");
                }
            }
        }
    }

    #[test]
    fn forward_dct_1d_matches_scalar() {
        let mut st = 0x0bad_f00du32;
        for &n in &[4usize, 8, 16, 32] {
            let matrix = dct_matrix(n);
            for &shift in &[1i32, 4, 8, 11] {
                for _ in 0..50 {
                    let mut line = [0i16; 32];
                    for v in line[..n].iter_mut() {
                        // Cover the full i16 intermediate range, signed.
                        *v = (rng(&mut st) as i32 as i16).clamp(-32767, 32767);
                    }
                    let mut want = [0i16; 32];
                    let mut got = [0i16; 32];
                    scalar_dct_1d(&line[..n], &mut want[..n], matrix, n, shift);
                    super::forward_dct_1d(&line[..n], &mut got[..n], matrix, n, shift);
                    assert_eq!(got[..n], want[..n], "n={n} shift={shift}");
                }
            }
        }
    }

    #[test]
    fn satd_u16_matches_scalar() {
        let mut st = 0xfeed_face_u32;
        for &bd in &[10u32, 12] {
            let max = (1u32 << bd) - 1;
            for &size in &[4usize, 8, 16, 32] {
                for _ in 0..40 {
                    let sa = size + (rng(&mut st) as usize & 7);
                    let sb = size + (rng(&mut st) as usize & 7);
                    let mut a = vec![0u16; sa * size];
                    let mut b = vec![0u16; sb * size];
                    for v in a.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    for v in b.iter_mut() {
                        *v = (rng(&mut st) & max) as u16;
                    }
                    let want = satd_u16_scalar(&a, sa, &b, sb, size);
                    let got = super::satd_u16(&a, sa, &b, sb, size);
                    assert_eq!(got, want, "bd={bd} size={size}");
                }
            }
        }
    }

    /// Extreme/structured tiles that maximise the Hadamard coefficient
    /// magnitudes, guarding the per-tile `i32` accumulation bound.
    #[test]
    fn satd_u16_extreme_patterns() {
        let max = (1u16 << 12) - 1;
        for &size in &[4usize, 8, 16, 32] {
            let mut a = vec![0u16; size * size];
            let mut b = vec![0u16; size * size];
            // Worst case for one 4x4: full-swing checkerboard against zero.
            for (i, (pa, pb)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
                let checker = ((i / size) ^ i) & 1 == 0;
                *pa = if checker { max } else { 0 };
                *pb = if checker { 0 } else { max };
            }
            assert_eq!(
                super::satd_u16(&a, size, &b, size, size),
                satd_u16_scalar(&a, size, &b, size, size),
                "size={size}"
            );
        }
    }
}
