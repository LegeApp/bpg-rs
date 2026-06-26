//! AVX2 primitive kernels.
//!
//! Tagged `#[target_feature(enable = "avx2")]`, called only after
//! `is_x86_feature_detected!("avx2")` in `select_primitives`.

use std::arch::x86_64::*;

// ─── quantize ──────────────────────────────────────────────────────────────

/// Forward quantization — AVX2, 16 coefficients/iteration.
///
/// Bit-identical to `scalar::quantize_scalar`. Processes two 8-lane i32
/// registers per iteration, packs results to i16 with `_mm256_packs_epi32`
/// + `_mm256_permute4x64_epi64` to fix the AVX2 lane-interleave.
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_avx2(
    coeffs: &[i16],
    levels: &mut [i16],
    scale: i32,
    add: i32,
    qbits: i32,
) -> u32 {
    let n = coeffs.len().min(levels.len());
    let vscale = _mm256_set1_epi32(scale);
    let vadd = _mm256_set1_epi32(add);
    let vmax = _mm256_set1_epi32(32767);
    let vzero = _mm256_setzero_si256();
    let vshift = _mm256_set1_epi32(qbits);
    let mut nnz = 0u32;

    let mut i = 0;
    while i + 16 <= n {
        // Load 16 i16 coefficients.
        let raw = _mm256_loadu_si256(coeffs.as_ptr().add(i) as *const __m256i);

        // Widen each 8-lane half to i32.
        let lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw));
        let hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw, 1));

        // |coeff|
        let lo_abs = _mm256_abs_epi32(lo_i32);
        let hi_abs = _mm256_abs_epi32(hi_i32);

        // level = (|coeff| * scale + add) >> qbits
        let lo_lvl = _mm256_srav_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(lo_abs, vscale), vadd),
            vshift,
        );
        let hi_lvl = _mm256_srav_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(hi_abs, vscale), vadd),
            vshift,
        );

        // clamp to 32767
        let lo_lvl = _mm256_min_epi32(lo_lvl, vmax);
        let hi_lvl = _mm256_min_epi32(hi_lvl, vmax);

        // Re-sign: negate where original coeff < 0.
        let lo_neg = _mm256_cmpgt_epi32(vzero, lo_i32);
        let hi_neg = _mm256_cmpgt_epi32(vzero, hi_i32);
        let lo_s = _mm256_blendv_epi8(lo_lvl, _mm256_sub_epi32(vzero, lo_lvl), lo_neg);
        let hi_s = _mm256_blendv_epi8(hi_lvl, _mm256_sub_epi32(vzero, hi_lvl), hi_neg);

        // Pack two i32x8 → i16x16.
        // _mm256_packs_epi32 interleaves 128-bit halves, so permute to fix order:
        // before: [lo[0..3], hi[0..3], lo[4..7], hi[4..7]]
        // after : [lo[0..7], hi[0..7]]   (imm8 = 0b11_01_10_00 = 0xD8)
        let packed = _mm256_packs_epi32(lo_s, hi_s);
        let fixed = _mm256_permute4x64_epi64(packed, 0xD8);

        _mm256_storeu_si256(levels.as_mut_ptr().add(i) as *mut __m256i, fixed);

        // Count non-zeros: cmpeq returns 0xFFFF per zero lane; movemask sees MSBs.
        // A zero i16 contributes 2 set bits; count_ones()/2 = zero count.
        let zeromask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(fixed, vzero));
        nnz += 16 - (zeromask.count_ones() >> 1);

        i += 16;
    }

    // Scalar tail.
    while i < n {
        let c = coeffs[i];
        let level = (c.unsigned_abs() as i64 * scale as i64 + add as i64) >> qbits;
        let level = level.min(32767) as i32;
        if level != 0 {
            levels[i] = if c < 0 { -level } else { level } as i16;
            nnz += 1;
        } else {
            levels[i] = 0;
        }
        i += 1;
    }
    nnz
}

pub fn quantize_avx2_dispatch(
    coeffs: &[i16],
    levels: &mut [i16],
    scale: i32,
    add: i32,
    qbits: i32,
) -> u32 {
    unsafe { quantize_avx2(coeffs, levels, scale, add, qbits) }
}

// ─── ssd_u8 ────────────────────────────────────────────────────────────────

/// 8-bit SSD — AVX2, 16 bytes/iteration. Bit-identical to `scalar::ssd_u8_scalar`.
#[target_feature(enable = "avx2")]
pub unsafe fn ssd_u8_avx2(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u64 {
    let mut sse = 0u64;
    let zero = _mm256_setzero_si256();

    for j in 0..size {
        let ra = &a[j * stride_a..];
        let rb = &b[j * stride_b..];
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;

        while i + 16 <= size {
            // Load 16 bytes, zero-extend to i16 via 128-bit load + _mm256_cvtepu8_epi16.
            let va16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(ra[i..].as_ptr() as *const __m128i));
            let vb16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(rb[i..].as_ptr() as *const __m128i));
            let d = _mm256_sub_epi16(va16, vb16);
            // madd_epi16(d, d): multiply adjacent pairs and add → 8 i32 lanes.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
            i += 16;
        }

        // Horizontal sum of 8 i32 lanes.
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1),
        );
        let s = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        sse += _mm_cvtsi128_si32(s) as u64;

        // SSE2 scalar tail (remaining < 16 bytes).
        while i < size {
            let d = ra[i] as i32 - rb[i] as i32;
            sse += (d * d) as u64;
            i += 1;
        }
    }
    sse
}

pub fn ssd_u8_avx2_dispatch(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u64 {
    unsafe { ssd_u8_avx2(a, stride_a, b, stride_b, size) }
}

// ─── sub_residual_u8 ───────────────────────────────────────────────────────

/// 8-bit residual subtraction — AVX2, 16 bytes/iteration.
/// Bit-identical to `scalar::sub_residual_u8_scalar`.
#[target_feature(enable = "avx2")]
pub unsafe fn sub_residual_u8_avx2(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    for j in 0..size {
        let rs = &src[j * src_stride..];
        let rp = &pred[j * pred_stride..];
        let ro = &mut out[j * size..];
        let mut i = 0;
        while i + 16 <= size {
            let vs = _mm256_cvtepu8_epi16(_mm_loadu_si128(rs[i..].as_ptr() as *const __m128i));
            let vp = _mm256_cvtepu8_epi16(_mm_loadu_si128(rp[i..].as_ptr() as *const __m128i));
            let d = _mm256_sub_epi16(vs, vp);
            _mm256_storeu_si256(ro[i..].as_mut_ptr() as *mut __m256i, d);
            i += 16;
        }
        while i < size {
            ro[i] = rs[i] as i16 - rp[i] as i16;
            i += 1;
        }
    }
}

pub fn sub_residual_u8_avx2_dispatch(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    unsafe { sub_residual_u8_avx2(src, src_stride, pred, pred_stride, out, size) }
}

// ─── satd_u8 ───────────────────────────────────────────────────────────────

/// 8-bit SATD — AVX2. Processes two 4×4 blocks simultaneously in the two
/// 128-bit lanes of a 256-bit register, doubling throughput vs SSE2.
#[target_feature(enable = "avx2")]
pub unsafe fn satd_u8_avx2(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u32 {
    assert!(matches!(size, 4 | 8 | 16 | 32));
    assert!(stride_a >= size && stride_b >= size);

    let mut sum = 0u32;

    // Tile over 4×4 blocks, pairing horizontally adjacent blocks in AVX2 lanes.
    let mut by = 0;
    while by < size {
        let mut bx = 0;
        while bx + 8 <= size {
            // Two adjacent 4×4 blocks share a single AVX2 register.
            // Block 0: columns [bx..bx+4],  Block 1: columns [bx+4..bx+8].
            let a0 = by * stride_a + bx;
            let b0 = by * stride_b + bx;
            sum += unsafe { satd_two_4x4(&a[a0..], stride_a, &b[b0..], stride_b) };
            bx += 8;
        }
        // Odd column leftover (only when size is not a multiple of 8 in x-dimension,
        // which doesn't happen for HEVC block sizes, but guard with SSE2 anyway).
        while bx < size {
            sum += super::sse2::satd_u8_sse2(
                &a[by * stride_a + bx..],
                stride_a,
                &b[by * stride_b + bx..],
                stride_b,
                4,
            );
            bx += 4;
        }
        by += 4;
    }
    sum
}

/// Compute SATD for two horizontally adjacent 4×4 blocks simultaneously.
/// Block 0 columns: `[bx..bx+4]`, block 1 columns: `[bx+4..bx+8]`.
/// Both are placed in separate 128-bit lanes of 256-bit registers and
/// processed independently through the Hadamard butterfly.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn satd_two_4x4(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize) -> u32 {
    unsafe {
        let zero128 = _mm_setzero_si128();

        // Load 4 bytes as u32, reinterpret as i32 (bytes unchanged) for _mm_cvtsi32_si128.
        let read4 = |p: *const u8| -> i32 { (p as *const u32).read_unaligned() as i32 };

        // Build one row: block 0 diff in lower 128-bit lane, block 1 diff in upper lane.
        let load_row = |row: usize| -> __m256i {
            let ap0 = a.as_ptr().add(row * stride_a);
            let bp0 = b.as_ptr().add(row * stride_b);
            // Zero-extend 4 u8s to 4 i16s in lower 64 bits of a 128-bit reg.
            let za0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(read4(ap0)), zero128);
            let zb0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(read4(bp0)), zero128);
            let za1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(read4(ap0.add(4))), zero128);
            let zb1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(read4(bp0.add(4))), zero128);
            let d0 = _mm_sub_epi16(za0, zb0);
            let d1 = _mm_sub_epi16(za1, zb1);
            _mm256_set_m128i(d1, d0) // block0 in lower lane, block1 in upper
        };

        let d0 = load_row(0);
        let d1 = load_row(1);
        let d2 = load_row(2);
        let d3 = load_row(3);

        // Horizontal Hadamard butterfly — operates independently on each 128-bit lane.
        let (t0, t1, t2, t3) = hadamard4_rows(d0, d1, d2, d3);

        // Transpose 4×4 within each 128-bit lane independently.
        let m0 = _mm256_unpacklo_epi16(t0, t1);
        let m1 = _mm256_unpacklo_epi16(t2, t3);
        let n0 = _mm256_unpacklo_epi32(m0, m1);
        let n1 = _mm256_unpackhi_epi32(m0, m1);
        // _mm256_srli_si256 shifts within each 128-bit lane (does not cross lane boundary).
        let c0 = n0;
        let c1 = _mm256_srli_si256(n0, 8);
        let c2 = n1;
        let c3 = _mm256_srli_si256(n1, 8);

        let (f0, f1, f2, f3) = hadamard4_rows(c0, c1, c2, c3);

        // Absolute value sum.
        let zero256 = _mm256_setzero_si256();
        let abs256 = |x: __m256i| _mm256_max_epi16(x, _mm256_sub_epi16(zero256, x));
        let acc = _mm256_add_epi16(
            _mm256_add_epi16(abs256(f0), abs256(f1)),
            _mm256_add_epi16(abs256(f2), abs256(f3)),
        );

        // Horizontal sum per 128-bit lane separately to preserve per-block rounding.
        // Block 0 is in lower lane, block 1 in upper lane.
        let acc0 = _mm256_castsi256_si128(acc);
        let acc1 = _mm256_extracti128_si256(acc, 1);

        let hsum = |v: __m128i| -> u32 {
            let wide = _mm_unpacklo_epi16(v, zero128); // 4 i32 (positions 4..7 are zeros)
            let s = _mm_add_epi32(wide, _mm_shuffle_epi32(wide, 0b01_00_11_10));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
            _mm_cvtsi128_si32(s) as u32
        };

        let s0 = hsum(acc0);
        let s1 = hsum(acc1);
        ((s0 + 1) >> 1) + ((s1 + 1) >> 1)
    }
}

/// Row-wise Hadamard butterfly on 4 rows (independent in each 128-bit lane).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hadamard4_rows(
    r0: __m256i,
    r1: __m256i,
    r2: __m256i,
    r3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    unsafe {
        let a0 = _mm256_add_epi16(r0, r3);
        let a1 = _mm256_add_epi16(r1, r2);
        let a2 = _mm256_sub_epi16(r1, r2);
        let a3 = _mm256_sub_epi16(r0, r3);
        (
            _mm256_add_epi16(a0, a1),
            _mm256_add_epi16(a3, a2),
            _mm256_sub_epi16(a0, a1),
            _mm256_sub_epi16(a3, a2),
        )
    }
}

pub fn satd_u8_avx2_dispatch(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u32 {
    unsafe { satd_u8_avx2(a, stride_a, b, stride_b, size) }
}

// ─── setup ─────────────────────────────────────────────────────────────────

/// Install AVX2 kernels into the primitive table.
pub(super) fn setup(p: &mut crate::primitives::Primitives) {
    p.pixel.satd_u8 = satd_u8_avx2_dispatch;
    p.pixel.ssd_u8 = ssd_u8_avx2_dispatch;
    p.pixel.sub_residual_u8 = sub_residual_u8_avx2_dispatch;
    p.quant.quantize = quantize_avx2_dispatch;
    p.backend = if p.backend.contains("wide") {
        "avx2+wide"
    } else {
        "avx2"
    };
}
