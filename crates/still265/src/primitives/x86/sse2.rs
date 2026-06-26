//! `x86_64` SSE2 primitive kernels.
//!
//! These must be byte-identical to the scalar references in [`crate::primitives::scalar`];
//! the equivalence tests in `primitives/mod.rs` enforce that across sizes, bit
//! depths, and pixel patterns.
//!
//! SSE2 is part of the `x86_64` baseline so these compile and run on every
//! `x86_64` target without a runtime guard.

use std::arch::x86_64::*;

/// Tiled 8-bit SATD. Byte-exact with the scalar `satd_by` reference.
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
            // SAFETY: bounds asserts above guarantee every byte read in
            // `satd_4x4` (rows `a0..a0+3*stride`, 4 bytes each) is in range.
            sum += unsafe { satd_4x4(&a[a0..], stride_a, &b[b0..], stride_b) };
        }
    }
    sum
}

/// One 4x4 8-bit SATD, bit-exact with `hadamard4_satd` on the diff.
#[inline]
unsafe fn satd_4x4(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize) -> u32 {
    // SAFETY: caller validates the four 4-byte rows are in bounds, and this
    // module is compiled only on x86_64 where SSE2 is baseline.
    unsafe {
        let zero = _mm_setzero_si128();

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

        let (t0, t1, t2, t3) = butterfly4(d0, d1, d2, d3);

        let m0 = _mm_unpacklo_epi16(t0, t1);
        let m1 = _mm_unpacklo_epi16(t2, t3);
        let n0 = _mm_unpacklo_epi32(m0, m1);
        let n1 = _mm_unpackhi_epi32(m0, m1);
        let c0 = n0;
        let c1 = _mm_unpackhi_epi64(n0, n0);
        let c2 = n1;
        let c3 = _mm_unpackhi_epi64(n1, n1);

        let (f0, f1, f2, f3) = butterfly4(c0, c1, c2, c3);

        let abs = |x: __m128i| _mm_max_epi16(x, _mm_sub_epi16(zero, x));
        let acc = _mm_add_epi16(
            _mm_add_epi16(abs(f0), abs(f1)),
            _mm_add_epi16(abs(f2), abs(f3)),
        );

        let lo = _mm_unpacklo_epi16(acc, zero);
        let h = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0b01_00_11_10));
        let h2 = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0b00_00_00_01));
        let sum = _mm_cvtsi128_si32(h2) as u32;

        (sum + 1) >> 1
    }
}

#[inline]
unsafe fn butterfly4(
    x0: __m128i,
    x1: __m128i,
    x2: __m128i,
    x3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
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
    unsafe { (p as *const u32).read_unaligned() }
}

/// 8-bit SSD over a `size`×`size` block. Uses `_mm_madd_epi16` for efficient
/// multiply-add; bit-identical to `scalar::ssd_u8_scalar`.
pub fn ssd_u8_sse2(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u64 {
    assert!(stride_a >= size && stride_b >= size);
    let mut sse = 0u64;
    unsafe {
        let zero = _mm_setzero_si128();
        for j in 0..size {
            let mut acc = _mm_setzero_si128();
            let row_a = &a[j * stride_a..];
            let row_b = &b[j * stride_b..];
            let mut i = 0;
            while i + 8 <= size {
                // Load 8 bytes, zero-extend to i16.
                let va =
                    _mm_unpacklo_epi8(_mm_loadl_epi64(row_a[i..].as_ptr() as *const __m128i), zero);
                let vb =
                    _mm_unpacklo_epi8(_mm_loadl_epi64(row_b[i..].as_ptr() as *const __m128i), zero);
                let d = _mm_sub_epi16(va, vb);
                // _mm_madd_epi16(d, d): multiplies pairs and sums adjacent → 4 i32 lanes.
                acc = _mm_add_epi32(acc, _mm_madd_epi16(d, d));
                i += 8;
            }
            // Horizontal sum of 4 i32 lanes.
            let s = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
            sse += _mm_cvtsi128_si32(s) as u64;
            // Scalar tail.
            while i < size {
                let d = row_a[i] as i32 - row_b[i] as i32;
                sse += (d * d) as u64;
                i += 1;
            }
        }
    }
    sse
}

/// 8-bit residual subtraction. Bit-identical to `scalar::sub_residual_u8_scalar`.
pub fn sub_residual_u8_sse2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    assert!(src_stride >= size && pred_stride >= size && out.len() >= size * size);
    unsafe {
        let zero = _mm_setzero_si128();
        for j in 0..size {
            let rs = &src[j * src_stride..];
            let rp = &pred[j * pred_stride..];
            let ro = &mut out[j * size..j * size + size];
            let mut i = 0;
            while i + 8 <= size {
                let vs =
                    _mm_unpacklo_epi8(_mm_loadl_epi64(rs[i..].as_ptr() as *const __m128i), zero);
                let vp =
                    _mm_unpacklo_epi8(_mm_loadl_epi64(rp[i..].as_ptr() as *const __m128i), zero);
                let d = _mm_sub_epi16(vs, vp);
                _mm_storeu_si128(ro[i..].as_mut_ptr() as *mut __m128i, d);
                i += 8;
            }
            while i < size {
                ro[i] = rs[i] as i16 - rp[i] as i16;
                i += 1;
            }
        }
    }
}

/// Install SSE2 kernels into the primitive table.
pub(super) fn setup(p: &mut crate::primitives::Primitives) {
    p.pixel.satd_u8 = satd_u8_sse2;
    p.pixel.ssd_u8 = ssd_u8_sse2;
    p.pixel.sub_residual_u8 = sub_residual_u8_sse2;
    p.backend = "sse2";
}
