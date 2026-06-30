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

// ─── add_clip_u8 (pack-store) ────────────────────────────────────────────────

/// Reconstruct 8-bit samples: `out = clip(pred + residual, 0, 255)`, 16/iter.
///
/// Bit-identical to `scalar::add_clip_u8_scalar`. Unlike the `wide` path, which
/// widens to `i32x8` and stores the 8 lanes element-by-element, this stays in
/// i16 and uses saturating add + `packus` — the x265-style narrow-type pipeline
/// that `wide` cannot express. `_mm256_adds_epi16` saturates the signed sum (a
/// `u8` pred plus an i16 residual can exceed i16 range), then `_mm_packus_epi16`
/// clamps to `[0, 255]`; the two together reproduce the scalar `clamp(0, 255)`
/// exactly.
#[target_feature(enable = "avx2")]
pub unsafe fn add_clip_u8_avx2(pred: &[u8], residual: &[i16], out: &mut [u8], n: usize) {
    debug_assert!(pred.len() >= n && residual.len() >= n && out.len() >= n);
    let mut i = 0;
    while i + 16 <= n {
        // 16 u8 pred → 16 i16, 16 i16 residual, saturating add.
        let p = _mm256_cvtepu8_epi16(_mm_loadu_si128(pred.as_ptr().add(i) as *const __m128i));
        let r = _mm256_loadu_si256(residual.as_ptr().add(i) as *const __m256i);
        let s = _mm256_adds_epi16(p, r);
        // Pack 16 i16 → 16 u8 (clamps to [0,255]); packus interleaves the two
        // 128-bit lanes, so permute the qwords back into source order before the
        // 16-byte store: result low 128 = [lane0.lo, lane1.lo].
        let packed = _mm256_packus_epi16(s, s);
        let perm = _mm256_permute4x64_epi64(packed, 0b0000_1000);
        _mm_storeu_si128(
            out.as_mut_ptr().add(i) as *mut __m128i,
            _mm256_castsi256_si128(perm),
        );
        i += 16;
    }
    while i < n {
        out[i] = (pred[i] as i32 + residual[i] as i32).clamp(0, 255) as u8;
        i += 1;
    }
}

pub fn add_clip_u8_avx2_dispatch(pred: &[u8], residual: &[i16], out: &mut [u8], n: usize) {
    unsafe { add_clip_u8_avx2(pred, residual, out, n) }
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

// ─── sa8d_u8 (8x8 Hadamard) ──────────────────────────────────────────────────

/// 8-bit SA8D — AVX2. Bit-identical to `scalar::sa8d_u8_scalar`: size 4 is a
/// single 4x4 SATD, larger sizes tile into 8x8 Hadamard blocks each scaled by
/// `(sum + 2) >> 2`, summed over tiles.
///
/// The 8x8 block uses an i16 separable Walsh-Hadamard (row pass, transpose, row
/// pass). The scalar `hadamard8_1d` emits its outputs in a different lane order
/// than this kernel's `hadamard8_rows_i16`, but the final sum-of-absolutes is
/// invariant to that permutation as long as both passes share one butterfly
/// (the permutation P turns the 2D result T into P·T·Pᵀ, leaving Σ|·| unchanged).
///
/// For 8-bit input the butterflies stay within i16 (worst-case coefficient
/// 255·64 = 16320 < 32767), but the per-lane absolute-value accumulation across
/// 8 rows would overflow i16, so it widens to i32 before summing.
#[target_feature(enable = "avx2")]
pub unsafe fn sa8d_u8_avx2(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u32 {
    assert!(matches!(size, 4 | 8 | 16 | 32));
    assert!(stride_a >= size && stride_b >= size);

    if size == 4 {
        return unsafe { satd_u8_avx2(a, stride_a, b, stride_b, 4) };
    }

    let mut sum = 0u32;
    let mut by = 0;
    while by < size {
        let mut bx = 0;
        while bx < size {
            sum += unsafe {
                sa8d_8x8_u8_avx2(
                    &a[by * stride_a + bx..],
                    stride_a,
                    &b[by * stride_b + bx..],
                    stride_b,
                )
            };
            bx += 8;
        }
        by += 8;
    }
    sum
}

/// One 8x8 SA8D tile. Returns `(Σ|coeff| + 2) >> 2`, matching `sa8d_8x8_by`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sa8d_8x8_u8_avx2(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    unsafe {
        let zero = _mm_setzero_si128();

        // Load row `row` as eight i16 differences (a - b).
        let load_diff8 = |row: usize| -> __m128i {
            let va = _mm_loadl_epi64(a.as_ptr().add(row * sa) as *const __m128i);
            let vb = _mm_loadl_epi64(b.as_ptr().add(row * sb) as *const __m128i);
            _mm_sub_epi16(_mm_unpacklo_epi8(va, zero), _mm_unpacklo_epi8(vb, zero))
        };

        let mut r = [
            load_diff8(0),
            load_diff8(1),
            load_diff8(2),
            load_diff8(3),
            load_diff8(4),
            load_diff8(5),
            load_diff8(6),
            load_diff8(7),
        ];

        hadamard8_rows_i16(&mut r);
        transpose8x8_i16(&mut r);
        hadamard8_rows_i16(&mut r);

        // Σ|coeff|, widened to i32 per lane to avoid i16 overflow across 8 rows.
        let mut acc = _mm_setzero_si128(); // 4 × i32
        for v in r {
            let abs = _mm_max_epi16(v, _mm_sub_epi16(zero, v));
            let lo = _mm_unpacklo_epi16(abs, zero); // 4 × i32 (abs ≥ 0)
            let hi = _mm_unpackhi_epi16(abs, zero); // 4 × i32
            acc = _mm_add_epi32(acc, _mm_add_epi32(lo, hi));
        }

        // Horizontal sum of the four i32 lanes.
        let s = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        let sum = _mm_cvtsi128_si32(s) as u32;
        (sum + 2) >> 2
    }
}

/// In-place 8-point Hadamard butterfly applied to each of the 8 lanes across the
/// eight `__m128i` rows (one i16 per lane per row).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hadamard8_rows_i16(r: &mut [__m128i; 8]) {
    unsafe {
        let a0 = _mm_add_epi16(r[0], r[4]);
        let a1 = _mm_add_epi16(r[1], r[5]);
        let a2 = _mm_add_epi16(r[2], r[6]);
        let a3 = _mm_add_epi16(r[3], r[7]);
        let a4 = _mm_sub_epi16(r[0], r[4]);
        let a5 = _mm_sub_epi16(r[1], r[5]);
        let a6 = _mm_sub_epi16(r[2], r[6]);
        let a7 = _mm_sub_epi16(r[3], r[7]);

        let b0 = _mm_add_epi16(a0, a2);
        let b1 = _mm_add_epi16(a1, a3);
        let b2 = _mm_sub_epi16(a0, a2);
        let b3 = _mm_sub_epi16(a1, a3);
        let b4 = _mm_add_epi16(a4, a6);
        let b5 = _mm_add_epi16(a5, a7);
        let b6 = _mm_sub_epi16(a4, a6);
        let b7 = _mm_sub_epi16(a5, a7);

        r[0] = _mm_add_epi16(b0, b1);
        r[1] = _mm_sub_epi16(b0, b1);
        r[2] = _mm_add_epi16(b2, b3);
        r[3] = _mm_sub_epi16(b2, b3);
        r[4] = _mm_add_epi16(b4, b5);
        r[5] = _mm_sub_epi16(b4, b5);
        r[6] = _mm_add_epi16(b6, b7);
        r[7] = _mm_sub_epi16(b6, b7);
    }
}

/// Transpose eight `__m128i` rows of eight i16 each.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn transpose8x8_i16(r: &mut [__m128i; 8]) {
    unsafe {
        let t0 = _mm_unpacklo_epi16(r[0], r[1]);
        let t1 = _mm_unpackhi_epi16(r[0], r[1]);
        let t2 = _mm_unpacklo_epi16(r[2], r[3]);
        let t3 = _mm_unpackhi_epi16(r[2], r[3]);
        let t4 = _mm_unpacklo_epi16(r[4], r[5]);
        let t5 = _mm_unpackhi_epi16(r[4], r[5]);
        let t6 = _mm_unpacklo_epi16(r[6], r[7]);
        let t7 = _mm_unpackhi_epi16(r[6], r[7]);

        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);

        r[0] = _mm_unpacklo_epi64(u0, u4);
        r[1] = _mm_unpackhi_epi64(u0, u4);
        r[2] = _mm_unpacklo_epi64(u1, u5);
        r[3] = _mm_unpackhi_epi64(u1, u5);
        r[4] = _mm_unpacklo_epi64(u2, u6);
        r[5] = _mm_unpackhi_epi64(u2, u6);
        r[6] = _mm_unpacklo_epi64(u3, u7);
        r[7] = _mm_unpackhi_epi64(u3, u7);
    }
}

pub fn sa8d_u8_avx2_dispatch(
    a: &[u8],
    stride_a: usize,
    b: &[u8],
    stride_b: usize,
    size: usize,
) -> u32 {
    unsafe { sa8d_u8_avx2(a, stride_a, b, stride_b, size) }
}

// ─── Intra prediction (8-bit) ────────────────────────────────────────────────
//
// All prediction functions implement the HEVC spec formulas exactly.
// They are bit-identical to the scalar reference in exact.rs.

/// HEVC planar intra prediction — AVX2, 8-bit output.
///
/// For each (px, py):
///   pred = ((n-1-py)*top[px] + (py+1)*bottom + (n-1-px)*left[py] + (px+1)*right + n)
///          >> (log2_size+1)
///
/// Reformulated per-row: pred[px] = (wy*top[px] + C1 + px*C2) >> shift
/// where wy=n-1-py, C1=(n-1)*left_y+right+(py+1)*bottom+n, C2=right-left_y.
/// The inner loop over px is vectorised 8-wide with i32 lanes.
#[target_feature(enable = "avx2")]
unsafe fn pred_planar_u8_avx2(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    _c_idx: u8,
    _bit_depth: u8,
) {
    let n = 1usize << log2_size;
    let shift = (log2_size + 1) as u32;
    let vshift = unsafe { _mm_cvtsi32_si128(shift as i32) };

    let right = border[center + 1 + n];
    let bottom = border[center - 1 - n];
    let top_ptr = border[center + 1..].as_ptr();

    // Lane index vector [0,1,2,3,4,5,6,7] used to compute px offsets within chunks.
    // _mm256_set_epi32(e7,e6,...,e0): lane 0 = e0, lane 7 = e7.
    let vlane = unsafe { _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0) };

    for py in 0..n {
        let left_y = border[center - 1 - py];
        let wy = n as i32 - 1 - py as i32;
        let c1 = (n as i32 - 1) * left_y + right + (py as i32 + 1) * bottom + n as i32;
        let c2 = right - left_y;

        let vc1 = unsafe { _mm256_set1_epi32(c1) };
        let vwy = unsafe { _mm256_set1_epi32(wy) };
        let vc2 = unsafe { _mm256_set1_epi32(c2) };
        let dst_row = &mut dst[py * n..];

        let mut px = 0usize;
        while px + 8 <= n {
            // Load top[px..px+8] (8 consecutive i32 values).
            let vtop = unsafe { _mm256_loadu_si256(top_ptr.add(px) as *const __m256i) };
            // Absolute column index for this chunk.
            let vpx = unsafe { _mm256_add_epi32(vlane, _mm256_set1_epi32(px as i32)) };
            // sum = wy*top + c1 + px*c2
            let sum = unsafe {
                _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(vwy, vtop), vc1),
                    _mm256_mullo_epi32(vpx, vc2),
                )
            };
            // Arithmetic right shift by `shift`.
            let shifted = unsafe { _mm256_sra_epi32(sum, vshift) };
            // Pack i32×8 → i16×8 → u8×8 with saturation.
            let lo128 = unsafe { _mm256_castsi256_si128(shifted) };
            let hi128 = unsafe { _mm256_extracti128_si256(shifted, 1) };
            let packed16 = unsafe { _mm_packs_epi32(lo128, hi128) };
            let packed8 = unsafe { _mm_packus_epi16(packed16, packed16) };
            unsafe { _mm_storel_epi64(dst_row.as_mut_ptr().add(px) as *mut __m128i, packed8) };
            px += 8;
        }
        // Scalar tail for n=4 or any leftover.
        while px < n {
            let top_px = unsafe { *top_ptr.add(px) };
            let pred = (wy * top_px + c1 + px as i32 * c2) >> shift;
            dst_row[px] = pred.clamp(0, 255) as u8;
            px += 1;
        }
    }
}

pub fn pred_planar_u8_avx2_dispatch(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    unsafe { pred_planar_u8_avx2(dst, border, center, log2_size, c_idx, bit_depth) }
}

/// HEVC DC intra prediction — AVX2, 8-bit output.
///
/// dc_val = (sum(top[0..n]) + sum(left[0..n]) + n) >> (log2_size+1)
/// Then fills the block, with a luma-only border filter for c_idx==0 and size<32.
#[target_feature(enable = "avx2")]
unsafe fn pred_dc_u8_avx2(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    _bit_depth: u8,
) {
    let n = 1usize << log2_size;

    // Horizontal reduction: sum top[0..n] and left[0..n].
    let mut sum = n as i32; // round bias
    for i in 0..n {
        sum += border[center + 1 + i]; // top
        sum += border[center - 1 - i]; // left
    }
    let dc_val = (sum >> (log2_size + 1)).clamp(0, 255) as u8;

    if c_idx == 0 && n < 32 {
        let top0 = border[center + 1] as i32;
        let left0 = border[center - 1] as i32;
        let dc = dc_val as i32;

        // Top-left corner
        dst[0] = ((left0 + 2 * dc + top0 + 2) >> 2).clamp(0, 255) as u8;

        // First row (excluding corner): blend with top
        let vtop_blend = unsafe { _mm256_set1_epi32((3 * dc + 2) as i32) };
        let vshift1 = unsafe { _mm_cvtsi32_si128(2) };
        let mut px = 1usize;
        while px + 8 <= n {
            let vtop =
                unsafe { _mm256_loadu_si256(border[center + 1 + px..].as_ptr() as *const __m256i) };
            let sum_v = unsafe { _mm256_add_epi32(vtop, vtop_blend) };
            let shifted = unsafe { _mm256_sra_epi32(sum_v, vshift1) };
            let lo128 = unsafe { _mm256_castsi256_si128(shifted) };
            let hi128 = unsafe { _mm256_extracti128_si256(shifted, 1) };
            let packed16 = unsafe { _mm_packs_epi32(lo128, hi128) };
            let packed8 = unsafe { _mm_packus_epi16(packed16, packed16) };
            unsafe { _mm_storel_epi64(dst.as_mut_ptr().add(px) as *mut __m128i, packed8) };
            px += 8;
        }
        while px < n {
            dst[px] = ((border[center + 1 + px] as i32 + 3 * dc + 2) >> 2).clamp(0, 255) as u8;
            px += 1;
        }

        // Remaining rows: first column blended with left; rest filled with dc_val.
        let vfill = unsafe { _mm256_set1_epi8(dc_val as i8) };
        let vfill128 = unsafe { _mm_set1_epi8(dc_val as i8) };
        for py in 1..n {
            let left_py = border[center - 1 - py] as i32;
            dst[py * n] = ((left_py + 3 * dc + 2) >> 2).clamp(0, 255) as u8;
            // Fill rest of row with dc_val using AVX2 store.
            let row = &mut dst[py * n + 1..py * n + n];
            let mut i = 0;
            while i + 32 <= row.len() {
                unsafe { _mm256_storeu_si256(row.as_mut_ptr().add(i) as *mut __m256i, vfill) };
                i += 32;
            }
            while i + 16 <= row.len() {
                unsafe { _mm_storeu_si128(row.as_mut_ptr().add(i) as *mut __m128i, vfill128) };
                i += 16;
            }
            row[i..].fill(dc_val);
        }
    } else {
        // Simple fill: entire block = dc_val.
        let n2 = n * n;
        let mut i = 0;
        while i + 32 <= n2 {
            let v = unsafe { _mm256_set1_epi8(dc_val as i8) };
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v) };
            i += 32;
        }
        while i + 16 <= n2 {
            let v = unsafe { _mm_set1_epi8(dc_val as i8) };
            unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, v) };
            i += 16;
        }
        dst[i..n2].fill(dc_val);
    }
}

pub fn pred_dc_u8_avx2_dispatch(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    unsafe { pred_dc_u8_avx2(dst, border, center, log2_size, c_idx, bit_depth) }
}

/// HEVC angular intra prediction — AVX2, 8-bit output.
///
/// For vertical modes (mode >= 18), the inner per-row fill/blend loop is
/// vectorised 8-wide: each output pixel is a 2-point interpolation between
/// consecutive reference samples with a per-row blend weight (constant per row).
///
/// For horizontal modes (mode < 18), each column has a different blend weight,
/// so we use a per-column precomputed gather; the fill loop is also 8-wide using
/// AVX2 gather intrinsics.
#[target_feature(enable = "avx2")]
unsafe fn pred_angular_u8_avx2(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    _bit_depth: u8,
) {
    use bpg_hevc_decode::hevc::intra::{INTRA_PRED_ANGLE, INV_ANGLE, MAX_INTRA_PRED_BLOCK_SIZE};

    let size = 1usize << log2_size;
    let n = size as i32;
    let intra_pred_angle = INTRA_PRED_ANGLE[mode as usize] as i32;

    let mut ref_arr = [0i32; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
    let ref_center = 2 * MAX_INTRA_PRED_BLOCK_SIZE;

    // Helper: clip to [0,255] for 8-bit.
    #[inline(always)]
    fn clip8(v: i32) -> u8 {
        v.clamp(0, 255) as u8
    }

    // Helper: compute i16 inv_angle for modes 11..=25.
    #[inline(always)]
    fn inv_angle_for(mode: u8) -> i32 {
        if (11..=25).contains(&mode) {
            INV_ANGLE[(mode - 11) as usize]
        } else {
            0
        }
    }

    if mode >= 18 {
        // ── Vertical modes ──────────────────────────────────────────────────
        ref_arr[ref_center..ref_center + size + 1]
            .copy_from_slice(&border[center..center + size + 1]);

        if intra_pred_angle < 0 {
            let inv_ang = inv_angle_for(mode);
            let ext = (n * intra_pred_angle) >> 5;
            if ext < -1 {
                for xx in ext..=-1 {
                    let idx = (xx * inv_ang + 128) >> 8;
                    if idx >= 0 && idx <= 2 * n {
                        ref_arr[(ref_center as i32 + xx) as usize] =
                            border[(center as i32 - idx) as usize];
                    }
                }
            }
        } else {
            let src_start = center + size + 1;
            let dst_start = ref_center + size + 1;
            ref_arr[dst_start..dst_start + size]
                .copy_from_slice(&border[src_start..src_start + size]);
        }

        let vround = unsafe { _mm256_set1_epi32(16) };
        let vshift5 = unsafe { _mm_cvtsi32_si128(5) };

        for py in 0..n {
            let i_idx = ((py + 1) * intra_pred_angle) >> 5;
            let i_fact = ((py + 1) * intra_pred_angle) & 31;
            let base_idx = (ref_center as i32 + i_idx + 1) as usize;
            let row = &mut dst[py as usize * size..py as usize * size + size];

            if i_fact != 0 {
                let w0 = 32 - i_fact;
                let vw0 = unsafe { _mm256_set1_epi32(w0) };
                let vw1 = unsafe { _mm256_set1_epi32(i_fact) };
                let mut px = 0usize;
                while px + 8 <= size {
                    let v0 = unsafe {
                        _mm256_loadu_si256(ref_arr[base_idx + px..].as_ptr() as *const __m256i)
                    };
                    let v1 = unsafe {
                        _mm256_loadu_si256(ref_arr[base_idx + px + 1..].as_ptr() as *const __m256i)
                    };
                    // (w0*v0 + w1*v1 + 16) >> 5
                    let sum = unsafe {
                        _mm256_add_epi32(
                            _mm256_add_epi32(
                                _mm256_mullo_epi32(vw0, v0),
                                _mm256_mullo_epi32(vw1, v1),
                            ),
                            vround,
                        )
                    };
                    let shifted = unsafe { _mm256_sra_epi32(sum, vshift5) };
                    let lo128 = unsafe { _mm256_castsi256_si128(shifted) };
                    let hi128 = unsafe { _mm256_extracti128_si256(shifted, 1) };
                    let packed16 = unsafe { _mm_packs_epi32(lo128, hi128) };
                    let packed8 = unsafe { _mm_packus_epi16(packed16, packed16) };
                    unsafe { _mm_storel_epi64(row.as_mut_ptr().add(px) as *mut __m128i, packed8) };
                    px += 8;
                }
                while px < size {
                    let idx = base_idx + px;
                    row[px] = clip8((w0 * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5);
                    px += 1;
                }
            } else {
                // Exact hit: no blend, just copy reference samples.
                let mut px = 0usize;
                while px + 8 <= size {
                    let v0 = unsafe {
                        _mm256_loadu_si256(ref_arr[base_idx + px..].as_ptr() as *const __m256i)
                    };
                    let lo128 = unsafe { _mm256_castsi256_si128(v0) };
                    let hi128 = unsafe { _mm256_extracti128_si256(v0, 1) };
                    let packed16 = unsafe { _mm_packs_epi32(lo128, hi128) };
                    let packed8 = unsafe { _mm_packus_epi16(packed16, packed16) };
                    unsafe { _mm_storel_epi64(row.as_mut_ptr().add(px) as *mut __m128i, packed8) };
                    px += 8;
                }
                while px < size {
                    row[px] = clip8(ref_arr[base_idx + px]);
                    px += 1;
                }
            }
        }

        // Luma horizontal mode-26 border correction.
        if mode == 26 && c_idx == 0 && size < 32 {
            for py in 0..n {
                let pred =
                    border[center + 1] + ((border[center - 1 - py as usize] - border[center]) >> 1);
                dst[py as usize * size] = clip8(pred);
            }
        }
    } else {
        // ── Horizontal modes ────────────────────────────────────────────────
        for i in 0..=n {
            ref_arr[ref_center + i as usize] = border[center - i as usize];
        }

        if intra_pred_angle < 0 {
            let inv_ang = inv_angle_for(mode);
            let ext = (n * intra_pred_angle) >> 5;
            if ext < -1 {
                for xx in ext..=-1 {
                    let idx = (xx * inv_ang + 128) >> 8;
                    if idx >= 0 && idx <= 2 * n {
                        ref_arr[(ref_center as i32 + xx) as usize] =
                            border[(center as i32 + idx) as usize];
                    }
                }
            }
        } else {
            for xx in (n + 1)..=(2 * n) {
                ref_arr[ref_center + xx as usize] = border[center - xx as usize];
            }
        }

        // Precompute per-column i_idx and i_fact (constant across rows).
        let mut col_idx = [0i32; 32];
        let mut col_fact = [0i32; 32];
        for px in 0..size {
            col_idx[px] = ((px as i32 + 1) * intra_pred_angle) >> 5;
            col_fact[px] = ((px as i32 + 1) * intra_pred_angle) & 31;
        }

        let vround = unsafe { _mm256_set1_epi32(16) };
        let vshift5 = unsafe { _mm_cvtsi32_si128(5) };

        for py in 0..n {
            let row_base = (ref_center as i32 + py + 1) as usize;
            let row = &mut dst[py as usize * size..py as usize * size + size];
            let ref_base_ptr = ref_arr[row_base..].as_ptr();

            let mut px = 0usize;
            while px + 8 <= size {
                // Gather: for each of 8 columns, load ref_arr[row_base + col_idx[px+k]].
                // scale=4 because ref_arr elements are i32 (4 bytes each).
                let vidx0 = unsafe { _mm256_loadu_si256(col_idx[px..].as_ptr() as *const __m256i) };
                let vidx1 = unsafe { _mm256_add_epi32(vidx0, _mm256_set1_epi32(1)) };
                let v0 = unsafe { _mm256_i32gather_epi32::<4>(ref_base_ptr, vidx0) };
                let v1 = unsafe { _mm256_i32gather_epi32::<4>(ref_base_ptr, vidx1) };
                let vfact =
                    unsafe { _mm256_loadu_si256(col_fact[px..].as_ptr() as *const __m256i) };
                let vw0 = unsafe { _mm256_sub_epi32(_mm256_set1_epi32(32), vfact) };

                // Blend: when i_fact == 0, w0*v0 + 0*v1 = w0*v0 = 32*v0 → >> 5 = v0. Correct.
                let sum = unsafe {
                    _mm256_add_epi32(
                        _mm256_add_epi32(
                            _mm256_mullo_epi32(vw0, v0),
                            _mm256_mullo_epi32(vfact, v1),
                        ),
                        vround,
                    )
                };
                let shifted = unsafe { _mm256_sra_epi32(sum, vshift5) };
                let lo128 = unsafe { _mm256_castsi256_si128(shifted) };
                let hi128 = unsafe { _mm256_extracti128_si256(shifted, 1) };
                let packed16 = unsafe { _mm_packs_epi32(lo128, hi128) };
                let packed8 = unsafe { _mm_packus_epi16(packed16, packed16) };
                unsafe { _mm_storel_epi64(row.as_mut_ptr().add(px) as *mut __m128i, packed8) };
                px += 8;
            }
            while px < size {
                let i_idx = col_idx[px];
                let i_fact = col_fact[px];
                let idx = (row_base as i32 + i_idx) as usize;
                let pred = if i_fact != 0 {
                    ((32 - i_fact) * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5
                } else {
                    ref_arr[idx]
                };
                row[px] = clip8(pred);
                px += 1;
            }
        }

        // Luma vertical mode-10 border correction.
        if mode == 10 && c_idx == 0 && size < 32 {
            for px in 0..n {
                let pred =
                    border[center - 1] + ((border[center + 1 + px as usize] - border[center]) >> 1);
                dst[px as usize] = clip8(pred);
            }
        }
    }
}

pub fn pred_angular_u8_avx2_dispatch(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    unsafe { pred_angular_u8_avx2(dst, border, center, log2_size, c_idx, mode, bit_depth) }
}

// ─── forward DCT ───────────────────────────────────────────────────────────

/// Transpose an 8×8 block of i16 stored row-major in 8 xmm registers.
///
/// Input:  r0..r7 each hold one row of 8 i16.
/// Output: c0..c7 each hold one *column* of 8 i16 (col k across all rows).
#[inline(always)]
unsafe fn transpose_8x8_i16(
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
    r4: __m128i,
    r5: __m128i,
    r6: __m128i,
    r7: __m128i,
) -> (
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
    __m128i,
) {
    // Phase 1: interleave adjacent pairs of rows (i16 granularity)
    let t00 = _mm_unpacklo_epi16(r0, r1);
    let t01 = _mm_unpackhi_epi16(r0, r1);
    let t02 = _mm_unpacklo_epi16(r2, r3);
    let t03 = _mm_unpackhi_epi16(r2, r3);
    let t04 = _mm_unpacklo_epi16(r4, r5);
    let t05 = _mm_unpackhi_epi16(r4, r5);
    let t06 = _mm_unpacklo_epi16(r6, r7);
    let t07 = _mm_unpackhi_epi16(r6, r7);
    // Phase 2: interleave groups of 2 (i32 granularity)
    let t10 = _mm_unpacklo_epi32(t00, t02);
    let t11 = _mm_unpackhi_epi32(t00, t02);
    let t12 = _mm_unpacklo_epi32(t01, t03);
    let t13 = _mm_unpackhi_epi32(t01, t03);
    let t14 = _mm_unpacklo_epi32(t04, t06);
    let t15 = _mm_unpackhi_epi32(t04, t06);
    let t16 = _mm_unpacklo_epi32(t05, t07);
    let t17 = _mm_unpackhi_epi32(t05, t07);
    // Phase 3: interleave groups of 4 (i64 granularity) → full columns
    let c0 = _mm_unpacklo_epi64(t10, t14);
    let c1 = _mm_unpackhi_epi64(t10, t14);
    let c2 = _mm_unpacklo_epi64(t11, t15);
    let c3 = _mm_unpackhi_epi64(t11, t15);
    let c4 = _mm_unpacklo_epi64(t12, t16);
    let c5 = _mm_unpackhi_epi64(t12, t16);
    let c6 = _mm_unpacklo_epi64(t13, t17);
    let c7 = _mm_unpackhi_epi64(t13, t17);
    (c0, c1, c2, c3, c4, c5, c6, c7)
}

/// Pack 8 i32 in a ymm to 8 i16 (saturating) and store at `dst[offset..]`.
#[inline(always)]
unsafe fn pack_store_i16(dst: &mut [i16], offset: usize, v: __m256i) {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let packed = _mm_packs_epi32(lo, hi);
    _mm_storeu_si128(dst.as_mut_ptr().add(offset) as *mut __m128i, packed);
}

/// One DCT-8 pass: 8 rows × 8 columns, all 8 rows processed in parallel.
///
/// Reads `src` (row-major, 8×8 i16), writes `dst` in transposed column-major
/// layout (freq*8+j) matching the scalar `butterfly8_1d` output convention.
#[target_feature(enable = "avx2")]
unsafe fn dct8_pass_avx2(src: &[i16], dst: &mut [i16], shift: u32) {
    let round = _mm256_set1_epi32(1i32 << (shift - 1));
    let vshift = _mm_cvtsi32_si128(shift as i32);

    // Load 8 rows (each 8×i16 = 128 bits)
    let r0 = _mm_loadu_si128(src.as_ptr().add(0) as *const __m128i);
    let r1 = _mm_loadu_si128(src.as_ptr().add(8) as *const __m128i);
    let r2 = _mm_loadu_si128(src.as_ptr().add(16) as *const __m128i);
    let r3 = _mm_loadu_si128(src.as_ptr().add(24) as *const __m128i);
    let r4 = _mm_loadu_si128(src.as_ptr().add(32) as *const __m128i);
    let r5 = _mm_loadu_si128(src.as_ptr().add(40) as *const __m128i);
    let r6 = _mm_loadu_si128(src.as_ptr().add(48) as *const __m128i);
    let r7 = _mm_loadu_si128(src.as_ptr().add(56) as *const __m128i);

    // Transpose: after this, col_k[j] = src[j*8 + k] (col k across all rows)
    let (col0_x, col1_x, col2_x, col3_x, col4_x, col5_x, col6_x, col7_x) =
        transpose_8x8_i16(r0, r1, r2, r3, r4, r5, r6, r7);

    // Sign-extend 8×i16 columns to 8×i32 ymm
    let c0 = _mm256_cvtepi16_epi32(col0_x);
    let c7 = _mm256_cvtepi16_epi32(col7_x);
    let e0 = _mm256_add_epi32(c0, c7);
    let o0 = _mm256_sub_epi32(c0, c7);

    let c1 = _mm256_cvtepi16_epi32(col1_x);
    let c6 = _mm256_cvtepi16_epi32(col6_x);
    let e1 = _mm256_add_epi32(c1, c6);
    let o1 = _mm256_sub_epi32(c1, c6);

    let c2 = _mm256_cvtepi16_epi32(col2_x);
    let c5 = _mm256_cvtepi16_epi32(col5_x);
    let e2 = _mm256_add_epi32(c2, c5);
    let o2 = _mm256_sub_epi32(c2, c5);

    let c3 = _mm256_cvtepi16_epi32(col3_x);
    let c4 = _mm256_cvtepi16_epi32(col4_x);
    let e3 = _mm256_add_epi32(c3, c4);
    let o3 = _mm256_sub_epi32(c3, c4);

    // EE / EO sub-butterfly
    let ee0 = _mm256_add_epi32(e0, e3);
    let eo0 = _mm256_sub_epi32(e0, e3);
    let ee1 = _mm256_add_epi32(e1, e2);
    let eo1 = _mm256_sub_epi32(e1, e2);

    // Output freq 0: 64*(ee0+ee1) >> shift  (shift-left-6 = mul-64)
    let out0 = _mm256_sra_epi32(
        _mm256_add_epi32(_mm256_slli_epi32(_mm256_add_epi32(ee0, ee1), 6), round),
        vshift,
    );
    // Output freq 4: 64*(ee0-ee1) >> shift
    let out4 = _mm256_sra_epi32(
        _mm256_add_epi32(_mm256_slli_epi32(_mm256_sub_epi32(ee0, ee1), 6), round),
        vshift,
    );
    // Output freq 2: (83*eo0 + 36*eo1 + round) >> shift
    let v83 = _mm256_set1_epi32(83);
    let v36 = _mm256_set1_epi32(36);
    let out2 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(v83, eo0), _mm256_mullo_epi32(v36, eo1)),
            round,
        ),
        vshift,
    );
    // Output freq 6: (36*eo0 - 83*eo1 + round) >> shift
    let out6 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_sub_epi32(_mm256_mullo_epi32(v36, eo0), _mm256_mullo_epi32(v83, eo1)),
            round,
        ),
        vshift,
    );

    // Odd outputs: coefficients from HEVC DCT-8 table
    let v89 = _mm256_set1_epi32(89);
    let v75 = _mm256_set1_epi32(75);
    let v50 = _mm256_set1_epi32(50);
    let v18 = _mm256_set1_epi32(18);

    // out1 = 89*o0 + 75*o1 + 50*o2 + 18*o3
    let out1 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(v89, o0), _mm256_mullo_epi32(v75, o1)),
                    _mm256_mullo_epi32(v50, o2),
                ),
                _mm256_mullo_epi32(v18, o3),
            ),
            round,
        ),
        vshift,
    );
    // out3 = 75*o0 - 18*o1 - 89*o2 - 50*o3
    let out3 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_sub_epi32(
                    _mm256_sub_epi32(_mm256_mullo_epi32(v75, o0), _mm256_mullo_epi32(v18, o1)),
                    _mm256_mullo_epi32(v89, o2),
                ),
                _mm256_mullo_epi32(v50, o3),
            ),
            round,
        ),
        vshift,
    );
    // out5 = 50*o0 - 89*o1 + 18*o2 + 75*o3
    let out5 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_sub_epi32(_mm256_mullo_epi32(v50, o0), _mm256_mullo_epi32(v89, o1)),
                    _mm256_mullo_epi32(v18, o2),
                ),
                _mm256_mullo_epi32(v75, o3),
            ),
            round,
        ),
        vshift,
    );
    // out7 = 18*o0 - 50*o1 + 75*o2 - 89*o3
    let out7 = _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_add_epi32(
                    _mm256_sub_epi32(_mm256_mullo_epi32(v18, o0), _mm256_mullo_epi32(v50, o1)),
                    _mm256_mullo_epi32(v75, o2),
                ),
                _mm256_mullo_epi32(v89, o3),
            ),
            round,
        ),
        vshift,
    );

    // Pack each output ymm (8×i32) to 8 i16 (saturating) and store.
    // dst[freq*8 .. freq*8+8] matches the scalar butterfly8_1d layout.
    pack_store_i16(dst, 0, out0);
    pack_store_i16(dst, 8, out1);
    pack_store_i16(dst, 16, out2);
    pack_store_i16(dst, 24, out3);
    pack_store_i16(dst, 32, out4);
    pack_store_i16(dst, 40, out5);
    pack_store_i16(dst, 48, out6);
    pack_store_i16(dst, 56, out7);
}

/// 2-D forward DCT-8 using AVX2. Bit-identical to `fwd_dct8_butterfly`.
#[target_feature(enable = "avx2")]
unsafe fn fwd_dct8_avx2_inner(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    let shift1 = (2u32 + bit_depth as u32).saturating_sub(8);
    let mut tmp = [0i16; 64];
    dct8_pass_avx2(residual, &mut tmp, shift1);
    dct8_pass_avx2(&tmp, out, 9);
}

pub fn fwd_dct8_avx2_dispatch(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    unsafe { fwd_dct8_avx2_inner(residual, out, bit_depth) }
}

// ─── DCT-16 and DCT-32 AVX2 ────────────────────────────────────────────────
//
// Strategy: keep scalar for the E/O sub-butterfly hierarchy; vectorize the
// expensive O-term dot products (dot8 for DCT-16, dot16 for DCT-32) using a
// single ymm mullo + horizontal reduction per coefficient row.

/// Horizontal sum of 8 × i32 lanes in a ymm register.
#[inline(always)]
unsafe fn hsum8_i32(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum4 = _mm_add_epi32(lo, hi);
    let sum2 = _mm_add_epi32(sum4, _mm_srli_si128(sum4, 8));
    let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
    _mm_cvtsi128_si32(sum1)
}

// DCT-16 O-term coefficient rows (8 coefficients each, 8 odd output freqs).
// Identical to DCT-32's EO table; defined locally to avoid cross-module deps.
const DCT16_O: [[i32; 8]; 8] = [
    [90, 87, 80, 70, 57, 43, 25, 9],      // freq=1
    [87, 57, 9, -43, -80, -90, -70, -25], // freq=3
    [80, 9, -70, -87, -25, 57, 90, 43],   // freq=5
    [70, -43, -87, 9, 90, 25, -80, -57],  // freq=7
    [57, -80, -25, 90, -9, -87, 43, 70],  // freq=9
    [43, -90, 57, 25, -87, 70, 9, -80],   // freq=11
    [25, -70, 90, -80, 43, 9, -57, 87],   // freq=13
    [9, -25, 43, -57, 70, -80, 87, -90],  // freq=15
];
const DCT32_O_AVX2: [[i32; 16]; 16] = [
    [
        90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4,
    ],
    [
        90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
    ],
    [
        88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22,
    ],
    [
        85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31,
    ],
    [
        82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38,
    ],
    [
        78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46,
    ],
    [
        73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54,
    ],
    [
        67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61,
    ],
    [
        61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67,
    ],
    [
        54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73,
    ],
    [
        46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78,
    ],
    [
        38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82,
    ],
    [
        31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85,
    ],
    [
        22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88,
    ],
    [
        13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90,
    ],
    [
        4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
    ],
];
const DCT32_EO_AVX2: [[i32; 8]; 8] = [
    [90, 87, 80, 70, 57, 43, 25, 9],
    [87, 57, 9, -43, -80, -90, -70, -25],
    [80, 9, -70, -87, -25, 57, 90, 43],
    [70, -43, -87, 9, 90, 25, -80, -57],
    [57, -80, -25, 90, -9, -87, 43, 70],
    [43, -90, 57, 25, -87, 70, 9, -80],
    [25, -70, 90, -80, 43, 9, -57, 87],
    [9, -25, 43, -57, 70, -80, 87, -90],
];
const DCT32_EEO_AVX2: [[i32; 4]; 4] = [
    [89, 75, 50, 18],
    [75, -18, -89, -50],
    [50, -89, 18, 75],
    [18, -50, 75, -89],
];

/// One DCT-16 pass (16 rows × 16 cols). Scalar butterfly for E/O hierarchy;
/// AVX2 for the 8 O-term dot8 products.
#[target_feature(enable = "avx2")]
unsafe fn dct16_pass_avx2(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    let rs = |v: i32| super::super::round_shift(v, shift);
    for j in 0..line {
        let s = &src[j * 16..j * 16 + 16];
        let e0 = s[0] as i32 + s[15] as i32;
        let o0 = s[0] as i32 - s[15] as i32;
        let e1 = s[1] as i32 + s[14] as i32;
        let o1 = s[1] as i32 - s[14] as i32;
        let e2 = s[2] as i32 + s[13] as i32;
        let o2 = s[2] as i32 - s[13] as i32;
        let e3 = s[3] as i32 + s[12] as i32;
        let o3 = s[3] as i32 - s[12] as i32;
        let e4 = s[4] as i32 + s[11] as i32;
        let o4 = s[4] as i32 - s[11] as i32;
        let e5 = s[5] as i32 + s[10] as i32;
        let o5 = s[5] as i32 - s[10] as i32;
        let e6 = s[6] as i32 + s[9] as i32;
        let o6 = s[6] as i32 - s[9] as i32;
        let e7 = s[7] as i32 + s[8] as i32;
        let o7 = s[7] as i32 - s[8] as i32;
        let ee0 = e0 + e7;
        let ee1 = e1 + e6;
        let ee2 = e2 + e5;
        let ee3 = e3 + e4;
        let eo0 = e0 - e7;
        let eo1 = e1 - e6;
        let eo2 = e2 - e5;
        let eo3 = e3 - e4;
        let eee0 = ee0 + ee3;
        let eee1 = ee1 + ee2;
        let eeo0 = ee0 - ee3;
        let eeo1 = ee1 - ee2;
        // EEE → freqs 0,8
        dst[0 * line + j] = rs(64 * eee0 + 64 * eee1);
        dst[8 * line + j] = rs(64 * eee0 - 64 * eee1);
        // EEO → freqs 4,12
        dst[4 * line + j] = rs(83 * eeo0 + 36 * eeo1);
        dst[12 * line + j] = rs(36 * eeo0 - 83 * eeo1);
        // EO → freqs 2,6,10,14 (dot4, scalar)
        dst[2 * line + j] = rs(89 * eo0 + 75 * eo1 + 50 * eo2 + 18 * eo3);
        dst[6 * line + j] = rs(75 * eo0 - 18 * eo1 - 89 * eo2 - 50 * eo3);
        dst[10 * line + j] = rs(50 * eo0 - 89 * eo1 + 18 * eo2 + 75 * eo3);
        dst[14 * line + j] = rs(18 * eo0 - 50 * eo1 + 75 * eo2 - 89 * eo3);
        // O → freqs 1,3,5,7,9,11,13,15 (8 × dot8 with AVX2)
        let o_ymm = _mm256_set_epi32(o7, o6, o5, o4, o3, o2, o1, o0);
        for (k, coeff) in DCT16_O.iter().enumerate() {
            let c = _mm256_loadu_si256(coeff.as_ptr() as *const __m256i);
            let dot = hsum8_i32(_mm256_mullo_epi32(o_ymm, c));
            dst[(2 * k + 1) * line + j] = rs(dot);
        }
    }
}

/// 2-D forward DCT-16 using AVX2 O-term dot products.
#[target_feature(enable = "avx2")]
unsafe fn fwd_dct16_avx2_inner(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    let shift1 = 3i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 256];
    dct16_pass_avx2(residual, &mut tmp, 16, shift1);
    dct16_pass_avx2(&tmp, out, 16, 10);
}

pub fn fwd_dct16_avx2_dispatch(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    unsafe { fwd_dct16_avx2_inner(residual, out, bit_depth) }
}

/// One DCT-32 pass (32 rows × 32 cols). Scalar for E/O/EE/EO/EEE/EEO
/// hierarchy; AVX2 for the 8 EO-term dot8 and 16 O-term dot16 products.
#[target_feature(enable = "avx2")]
unsafe fn dct32_pass_avx2(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    let rs = |v: i32| super::super::round_shift(v, shift);
    for j in 0..line {
        let s = &src[j * 32..j * 32 + 32];
        // E/O split: 16 pairs
        let o0 = s[0] as i32 - s[31] as i32;
        let e0 = s[0] as i32 + s[31] as i32;
        let o1 = s[1] as i32 - s[30] as i32;
        let e1 = s[1] as i32 + s[30] as i32;
        let o2 = s[2] as i32 - s[29] as i32;
        let e2 = s[2] as i32 + s[29] as i32;
        let o3 = s[3] as i32 - s[28] as i32;
        let e3 = s[3] as i32 + s[28] as i32;
        let o4 = s[4] as i32 - s[27] as i32;
        let e4 = s[4] as i32 + s[27] as i32;
        let o5 = s[5] as i32 - s[26] as i32;
        let e5 = s[5] as i32 + s[26] as i32;
        let o6 = s[6] as i32 - s[25] as i32;
        let e6 = s[6] as i32 + s[25] as i32;
        let o7 = s[7] as i32 - s[24] as i32;
        let e7 = s[7] as i32 + s[24] as i32;
        let o8 = s[8] as i32 - s[23] as i32;
        let e8 = s[8] as i32 + s[23] as i32;
        let o9 = s[9] as i32 - s[22] as i32;
        let e9 = s[9] as i32 + s[22] as i32;
        let o10 = s[10] as i32 - s[21] as i32;
        let e10 = s[10] as i32 + s[21] as i32;
        let o11 = s[11] as i32 - s[20] as i32;
        let e11 = s[11] as i32 + s[20] as i32;
        let o12 = s[12] as i32 - s[19] as i32;
        let e12 = s[12] as i32 + s[19] as i32;
        let o13 = s[13] as i32 - s[18] as i32;
        let e13 = s[13] as i32 + s[18] as i32;
        let o14 = s[14] as i32 - s[17] as i32;
        let e14 = s[14] as i32 + s[17] as i32;
        let o15 = s[15] as i32 - s[16] as i32;
        let e15 = s[15] as i32 + s[16] as i32;
        // EE/EO split: 8 pairs
        let ee0 = e0 + e15;
        let ee1 = e1 + e14;
        let ee2 = e2 + e13;
        let ee3 = e3 + e12;
        let ee4 = e4 + e11;
        let ee5 = e5 + e10;
        let ee6 = e6 + e9;
        let ee7 = e7 + e8;
        let eo0 = e0 - e15;
        let eo1 = e1 - e14;
        let eo2 = e2 - e13;
        let eo3 = e3 - e12;
        let eo4 = e4 - e11;
        let eo5 = e5 - e10;
        let eo6 = e6 - e9;
        let eo7 = e7 - e8;
        // EEE/EEO split: 4 pairs
        let eee0 = ee0 + ee7;
        let eee1 = ee1 + ee6;
        let eee2 = ee2 + ee5;
        let eee3 = ee3 + ee4;
        let eeo0 = ee0 - ee7;
        let eeo1 = ee1 - ee6;
        let eeo2 = ee2 - ee5;
        let eeo3 = ee3 - ee4;
        // EEEE/EEEO split: 2 pairs
        let eeee0 = eee0 + eee3;
        let eeee1 = eee1 + eee2;
        let eeeo0 = eee0 - eee3;
        let eeeo1 = eee1 - eee2;
        // DC outputs: freqs 0,16,8,24
        dst[0 * line + j] = rs(64 * eeee0 + 64 * eeee1);
        dst[16 * line + j] = rs(64 * eeee0 - 64 * eeee1);
        dst[8 * line + j] = rs(83 * eeeo0 + 36 * eeeo1);
        dst[24 * line + j] = rs(36 * eeeo0 - 83 * eeeo1);
        // EEO dot4 outputs: freqs 4,12,20,28 (scalar, only 16 mults total)
        for (k, c) in DCT32_EEO_AVX2.iter().enumerate() {
            let dot = eeo0 * c[0] + eeo1 * c[1] + eeo2 * c[2] + eeo3 * c[3];
            dst[(4 + k * 8) * line + j] = rs(dot);
        }
        // EO dot8 outputs: freqs 2,6,10,...,30 (AVX2 dot8)
        let eo_ymm = _mm256_set_epi32(eo7, eo6, eo5, eo4, eo3, eo2, eo1, eo0);
        for (k, coeff) in DCT32_EO_AVX2.iter().enumerate() {
            let c = _mm256_loadu_si256(coeff.as_ptr() as *const __m256i);
            let dot = hsum8_i32(_mm256_mullo_epi32(eo_ymm, c));
            dst[(2 + k * 4) * line + j] = rs(dot);
        }
        // O dot16 outputs: freqs 1,3,5,...,31 (AVX2 dot16 using 2 ymm)
        let o_lo = _mm256_set_epi32(o7, o6, o5, o4, o3, o2, o1, o0);
        let o_hi = _mm256_set_epi32(o15, o14, o13, o12, o11, o10, o9, o8);
        for (k, c) in DCT32_O_AVX2.iter().enumerate() {
            let c_lo = _mm256_loadu_si256(c[0..].as_ptr() as *const __m256i);
            let c_hi = _mm256_loadu_si256(c[8..].as_ptr() as *const __m256i);
            let dot = hsum8_i32(_mm256_add_epi32(
                _mm256_mullo_epi32(o_lo, c_lo),
                _mm256_mullo_epi32(o_hi, c_hi),
            ));
            dst[(1 + k * 2) * line + j] = rs(dot);
        }
    }
}

/// 2-D forward DCT-32 using AVX2 dot products.
#[target_feature(enable = "avx2")]
unsafe fn fwd_dct32_avx2_inner(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    let shift1 = 4i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 1024];
    dct32_pass_avx2(residual, &mut tmp, 32, shift1);
    dct32_pass_avx2(&tmp, out, 32, 11);
}

pub fn fwd_dct32_avx2_dispatch(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    unsafe { fwd_dct32_avx2_inner(residual, out, bit_depth) }
}

// ─── setup ─────────────────────────────────────────────────────────────────

/// Install AVX2 kernels into the primitive table.
pub(super) fn setup(p: &mut crate::primitives::Primitives) {
    p.pixel.satd_u8 = satd_u8_avx2_dispatch;
    p.pixel.sa8d_u8 = sa8d_u8_avx2_dispatch;
    p.pixel.ssd_u8 = ssd_u8_avx2_dispatch;
    p.pixel.sub_residual_u8 = sub_residual_u8_avx2_dispatch;
    p.pixel.add_clip_u8 = add_clip_u8_avx2_dispatch;
    p.quant.quantize = quantize_avx2_dispatch;
    p.intra.pred_planar_u8 = pred_planar_u8_avx2_dispatch;
    p.intra.pred_dc_u8 = pred_dc_u8_avx2_dispatch;
    p.intra.pred_angular_u8 = pred_angular_u8_avx2_dispatch;
    p.transform.fwd_dct8 = fwd_dct8_avx2_dispatch;
    p.transform.fwd_dct16 = fwd_dct16_avx2_dispatch;
    p.transform.fwd_dct32 = fwd_dct32_avx2_dispatch;
    p.backend = if p.backend.contains("wide") {
        "avx2+wide"
    } else {
        "avx2"
    };
}
