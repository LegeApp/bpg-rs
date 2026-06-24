//! `x86_64` SSE2 SATD kernels.
//!
//! These must be byte-identical to the scalar reference in the parent module
//! ([`super::satd_u8_scalar`]); the equivalence tests in `mod.rs`'s `tests`
//! enforce that across sizes, bit depths, and pixel patterns.
//!
//! Correctness note: the scalar reference applies the 4x4 Hadamard to rows
//! first, then columns. A separable 2D Hadamard `H * D * H^T` yields the same
//! 16 coefficients regardless of which dimension is transformed first
//! (`(H*D)*H^T == H*(D*H^T)`), so doing the column (cross-register) butterfly
//! first, transposing, then the row butterfly produces the identical set of
//! coefficients — hence the identical sum of absolute values. SSE2 is part of
//! the `x86_64` baseline, so these compile and run on every `x86_64` target.

use std::arch::x86_64::*;

/// Tiled 8-bit SATD: identical structure to the scalar `satd_by`, summing the
/// per-4x4 Hadamard results, so the total is bit-exact with the reference.
pub fn satd_u8_sse2(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    assert!(matches!(size, 4 | 8 | 16 | 32));
    assert!(stride_a >= size && stride_b >= size);
    assert!(a.len() >= stride_a * (size - 1) + size);
    assert!(b.len() >= stride_b * (size - 1) + size);

    let mut sum = 0u32;
    for by in (0..size).step_by(4) {
        for bx in (0..size).step_by(4) {
            let a0 = by * stride_a + bx;
            let b0 = by * stride_b + bx;
            // SAFETY: the bounds asserts above guarantee every byte read in
            // `satd_4x4` (rows `a0..a0+3*stride`, 4 bytes each) is in range.
            sum += unsafe { satd_4x4(&a[a0..], stride_a, &b[b0..], stride_b) };
        }
    }
    sum
}

/// One 4x4 8-bit SATD, bit-exact with `super::hadamard4_satd` on the diff.
#[inline]
unsafe fn satd_4x4(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize) -> u32 {
    // SAFETY: caller validates the four 4-byte rows are in bounds, and this
    // module is compiled only on x86_64 where SSE2 is baseline.
    unsafe {
        let zero = _mm_setzero_si128();

        // Load each 4-pixel row, zero-extend bytes to i16 (lanes 0..3), subtract.
        let load_diff = |row: usize| -> __m128i {
            let pa = _mm_cvtsi32_si128(read_u32(a.as_ptr().add(row * stride_a)) as i32);
            let pb = _mm_cvtsi32_si128(read_u32(b.as_ptr().add(row * stride_b)) as i32);
            let za = _mm_unpacklo_epi8(pa, zero);
            let zb = _mm_unpacklo_epi8(pb, zero);
            _mm_sub_epi16(za, zb)
        };
        let d0 = load_diff(0);
        let d1 = load_diff(1);
        let d2 = load_diff(2);
        let d3 = load_diff(3);

        // Pass 1: cross-register (column) butterfly.
        let (t0, t1, t2, t3) = butterfly4(d0, d1, d2, d3);

        // Transpose the 4x4 i16 block (valid data in low 64 bits of each row).
        let m0 = _mm_unpacklo_epi16(t0, t1);
        let m1 = _mm_unpacklo_epi16(t2, t3);
        let n0 = _mm_unpacklo_epi32(m0, m1);
        let n1 = _mm_unpackhi_epi32(m0, m1);
        let c0 = n0;
        let c1 = _mm_unpackhi_epi64(n0, n0);
        let c2 = n1;
        let c3 = _mm_unpackhi_epi64(n1, n1);

        // Pass 2: butterfly on the transposed rows.
        let (f0, f1, f2, f3) = butterfly4(c0, c1, c2, c3);

        // |coeff| via max(x, -x) (safe in i16: |coeff| <= 255*16 = 4080), summed.
        let abs = |x: __m128i| _mm_max_epi16(x, _mm_sub_epi16(zero, x));
        let acc = _mm_add_epi16(
            _mm_add_epi16(abs(f0), abs(f1)),
            _mm_add_epi16(abs(f2), abs(f3)),
        );

        // Horizontal-sum the 4 valid (non-negative) i16 lanes via i32 widening.
        let lo = _mm_unpacklo_epi16(acc, zero);
        let h = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0b01_00_11_10));
        let h2 = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b00_00_00_01));
        let sum = _mm_cvtsi128_si32(h2) as u32;

        (sum + 1) >> 1
    }
}

/// The shared 4-point Hadamard butterfly used by both passes. Each input holds
/// 4 valid i16 lanes (low 64 bits).
#[inline]
unsafe fn butterfly4(
    x0: __m128i,
    x1: __m128i,
    x2: __m128i,
    x3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // SAFETY: caller runs only under the x86_64 SSE2 baseline.
    unsafe {
        let a0 = _mm_add_epi16(x0, x3);
        let a1 = _mm_add_epi16(x1, x2);
        let a2 = _mm_sub_epi16(x1, x2);
        let a3 = _mm_sub_epi16(x0, x3);
        (
            _mm_add_epi16(a0, a1),
            _mm_add_epi16(a3, a2),
            _mm_sub_epi16(a0, a1),
            _mm_sub_epi16(a3, a2),
        )
    }
}

#[inline]
unsafe fn read_u32(p: *const u8) -> u32 {
    // SAFETY: caller guarantees at least four readable bytes at `p`; unaligned
    // loads are intentional for row starts.
    unsafe { (p as *const u32).read_unaligned() }
}
