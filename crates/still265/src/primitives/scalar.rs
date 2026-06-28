//! Canonical scalar primitive implementations.
//!
//! These are the **reference** implementations that all optimised (SIMD/ASM)
//! kernels must produce byte-identical output to. They are selected as the
//! default table entries and are always available via `BPG_PRIMITIVES=scalar`.

use super::round_shift;

// ─── SATD ──────────────────────────────────────────────────────────────────

fn hadamard4_satd(diff: &[i32; 16]) -> u32 {
    let mut tmp = [0i32; 16];
    for y in 0..4 {
        let a0 = diff[y * 4] + diff[y * 4 + 3];
        let a1 = diff[y * 4 + 1] + diff[y * 4 + 2];
        let a2 = diff[y * 4 + 1] - diff[y * 4 + 2];
        let a3 = diff[y * 4] - diff[y * 4 + 3];
        tmp[y * 4] = a0 + a1;
        tmp[y * 4 + 1] = a3 + a2;
        tmp[y * 4 + 2] = a0 - a1;
        tmp[y * 4 + 3] = a3 - a2;
    }

    let mut sum = 0u32;
    for x in 0..4 {
        let a0 = tmp[x] + tmp[12 + x];
        let a1 = tmp[4 + x] + tmp[8 + x];
        let a2 = tmp[4 + x] - tmp[8 + x];
        let a3 = tmp[x] - tmp[12 + x];
        sum += (a0 + a1).unsigned_abs();
        sum += (a3 + a2).unsigned_abs();
        sum += (a0 - a1).unsigned_abs();
        sum += (a3 - a2).unsigned_abs();
    }

    (sum + 1) >> 1
}

fn satd_by<T: Copy>(
    a: &[T],
    stride_a: usize,
    b: &[T],
    stride_b: usize,
    size: usize,
    to_i32: impl Fn(T) -> i32,
) -> u32 {
    assert!(matches!(size, 4 | 8 | 16 | 32));
    assert!(stride_a >= size && stride_b >= size);
    assert!(a.len() >= stride_a * (size - 1) + size);
    assert!(b.len() >= stride_b * (size - 1) + size);

    let mut sum = 0u32;
    let mut diff = [0i32; 16];
    for by in (0..size).step_by(4) {
        for bx in (0..size).step_by(4) {
            for y in 0..4 {
                for x in 0..4 {
                    diff[y * 4 + x] = to_i32(a[(by + y) * stride_a + bx + x])
                        - to_i32(b[(by + y) * stride_b + bx + x]);
                }
            }
            sum += hadamard4_satd(&diff);
        }
    }
    sum
}

/// Canonical scalar 8-bit SATD. Reference for all optimised backends.
pub fn satd_u8_scalar(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

/// Canonical scalar 10/12-bit SATD. Reference for all optimised backends.
pub fn satd_u16_scalar(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

fn hadamard8_1d(v: &mut [i32; 8]) {
    let a0 = v[0] + v[4];
    let a1 = v[1] + v[5];
    let a2 = v[2] + v[6];
    let a3 = v[3] + v[7];
    let a4 = v[0] - v[4];
    let a5 = v[1] - v[5];
    let a6 = v[2] - v[6];
    let a7 = v[3] - v[7];

    let b0 = a0 + a2;
    let b1 = a1 + a3;
    let b2 = a0 - a2;
    let b3 = a1 - a3;
    let b4 = a4 + a6;
    let b5 = a5 + a7;
    let b6 = a4 - a6;
    let b7 = a5 - a7;

    v[0] = b0 + b1;
    v[1] = b4 + b5;
    v[2] = b2 + b3;
    v[3] = b6 + b7;
    v[4] = b0 - b1;
    v[5] = b4 - b5;
    v[6] = b2 - b3;
    v[7] = b6 - b7;
}

fn sa8d_8x8_by<T: Copy>(
    a: &[T],
    stride_a: usize,
    b: &[T],
    stride_b: usize,
    by: usize,
    bx: usize,
    to_i32: &impl Fn(T) -> i32,
) -> u32 {
    let mut tmp = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            tmp[y][x] =
                to_i32(a[(by + y) * stride_a + bx + x]) - to_i32(b[(by + y) * stride_b + bx + x]);
        }
        hadamard8_1d(&mut tmp[y]);
    }

    let mut sum = 0u32;
    for x in 0..8 {
        let mut col = [0i32; 8];
        for y in 0..8 {
            col[y] = tmp[y][x];
        }
        hadamard8_1d(&mut col);
        for v in col {
            sum = sum.saturating_add(v.unsigned_abs());
        }
    }
    (sum + 2) >> 2
}

fn sa8d_by<T: Copy>(
    a: &[T],
    stride_a: usize,
    b: &[T],
    stride_b: usize,
    size: usize,
    to_i32: impl Fn(T) -> i32,
) -> u32 {
    assert!(matches!(size, 4 | 8 | 16 | 32));
    assert!(stride_a >= size && stride_b >= size);
    assert!(a.len() >= stride_a * (size - 1) + size);
    assert!(b.len() >= stride_b * (size - 1) + size);

    if size == 4 {
        return satd_by(a, stride_a, b, stride_b, size, to_i32);
    }

    let mut sum = 0u32;
    for by in (0..size).step_by(8) {
        for bx in (0..size).step_by(8) {
            sum += sa8d_8x8_by(a, stride_a, b, stride_b, by, bx, &to_i32);
        }
    }
    sum
}

/// x265-style CU SA8D: 4x4 SATD for 4x4, 8x8 Hadamard tiles otherwise.
pub fn sa8d_u8_scalar(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    sa8d_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

/// x265-style CU SA8D for 10/12-bit samples.
pub fn sa8d_u16_scalar(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    sa8d_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

// ─── SSD ───────────────────────────────────────────────────────────────────

/// Sum of squared differences over an 8-bit `size`×`size` block.
/// Canonical scalar reference. Dispatched through [`PRIMITIVES`] after Phase 2.
pub fn ssd_u8_scalar(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u64 {
    debug_assert!(stride_a >= size && stride_b >= size);
    let mut sse = 0u64;
    for j in 0..size {
        let ra = &a[j * stride_a..j * stride_a + size];
        let rb = &b[j * stride_b..j * stride_b + size];
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            let d = x as i32 - y as i32;
            sse += (d * d) as u64;
        }
    }
    sse
}

/// 8-bit residual subtraction. Canonical scalar reference.
pub fn sub_residual_u8_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    debug_assert!(src_stride >= size && pred_stride >= size && out.len() >= size * size);
    for j in 0..size {
        let s = &src[j * src_stride..j * src_stride + size];
        let p = &pred[j * pred_stride..j * pred_stride + size];
        let o = &mut out[j * size..j * size + size];
        for i in 0..size {
            o[i] = s[i] as i16 - p[i] as i16;
        }
    }
}

/// Add reconstructed residual to 8-bit prediction and clip to [0,255]. Canonical scalar.
pub fn add_clip_u8_scalar(pred: &[u8], residual: &[i16], out: &mut [u8], n: usize) {
    debug_assert!(pred.len() >= n && residual.len() >= n && out.len() >= n);
    for i in 0..n {
        out[i] = (pred[i] as i32 + residual[i] as i32).clamp(0, 255) as u8;
    }
}

/// Add reconstructed residual to 10/12-bit prediction and clip to [0, max]. Canonical scalar.
pub fn add_clip_u16_scalar(pred: &[u16], residual: &[i16], out: &mut [u16], n: usize, max: u16) {
    debug_assert!(pred.len() >= n && residual.len() >= n && out.len() >= n);
    for i in 0..n {
        out[i] = (pred[i] as i32 + residual[i] as i32).clamp(0, max as i32) as u16;
    }
}

/// Narrow a u16 prediction block to u8 (clip to [0,255]). Canonical scalar.
pub fn narrow_u16_to_u8_scalar(src: &[u16], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    for i in 0..n {
        dst[i] = src[i].min(255) as u8;
    }
}

/// Canonical scalar SSD (u16 path). Reference for all optimised backends.
pub fn ssd_u16_scalar(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u64 {
    debug_assert!(stride_a >= size && stride_b >= size);
    let mut sse = 0u64;
    for j in 0..size {
        let ra = &a[j * stride_a..j * stride_a + size];
        let rb = &b[j * stride_b..j * stride_b + size];
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            let d = x as i64 - y as i64;
            sse += (d * d) as u64;
        }
    }
    sse
}

// ─── Residual ──────────────────────────────────────────────────────────────

/// Canonical scalar residual subtraction (u16 path). Reference for all optimised backends.
pub fn sub_residual_scalar(
    src: &[u16],
    src_stride: usize,
    pred: &[u16],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    debug_assert!(src_stride >= size && pred_stride >= size && out.len() >= size * size);
    for j in 0..size {
        let s = &src[j * src_stride..j * src_stride + size];
        let p = &pred[j * pred_stride..j * pred_stride + size];
        let o = &mut out[j * size..j * size + size];
        for i in 0..size {
            o[i] = s[i] as i16 - p[i] as i16;
        }
    }
}

// ─── Quantization ──────────────────────────────────────────────────────────

/// Canonical scalar forward quantization. Reference for the SIMD backend.
pub fn quantize_scalar(
    coeffs: &[i16],
    levels: &mut [i16],
    scale: i32,
    add: i32,
    qbits: i32,
) -> u32 {
    let mut nnz = 0u32;
    for (i, &c) in coeffs.iter().enumerate() {
        let level = ((c.unsigned_abs() as i64 * scale as i64 + add as i64) >> qbits) as i32;
        let level = level.min(32767);
        if level != 0 {
            levels[i] = if c < 0 { -level } else { level } as i16;
            nnz += 1;
        }
    }
    nnz
}

// ─── Coefficient/residual tools ────────────────────────────────────────────

/// Count nonzero i16 values in a flat coefficient slice. Canonical scalar.
pub fn count_nonzero_scalar(levels: &[i16]) -> u32 {
    levels.iter().filter(|&&v| v != 0).count() as u32
}

/// Absolute sum of all i16 values in a flat slice. Canonical scalar.
pub fn abs_sum_i16_scalar(levels: &[i16]) -> u64 {
    levels.iter().map(|&v| v.unsigned_abs() as u64).sum()
}

/// Index of the last nonzero element in linear order, or `None` if all zero.
pub fn last_nonzero_scalar(levels: &[i16]) -> Option<usize> {
    levels.iter().rposition(|&v| v != 0)
}

/// Canonical scalar inverse quantization. Reference for all optimised backends.
pub fn dequantize_scalar(levels: &mut [i16], combined: i32, shift: i32) {
    if shift >= 0 {
        let add = if shift > 0 { 1 << (shift - 1) } else { 0 };
        for v in levels.iter_mut() {
            let value = (*v as i32 * combined + add) >> shift;
            *v = value.clamp(-32768, 32767) as i16;
        }
    } else {
        let neg = -shift;
        for v in levels.iter_mut() {
            *v = ((*v as i32 * combined) << neg).clamp(-32768, 32767) as i16;
        }
    }
}

// ─── SAO ───────────────────────────────────────────────────────────────────

// ─── SAO stats ─────────────────────────────────────────────────────────────

// Inner: accumulate one EO-class stat for an interior row.
#[inline]
fn eo_accum(recon: i32, n0: i32, n1: i32, src: i64, sum: &mut [i64; 5], count: &mut [u32; 5]) {
    let edge = (2 + (recon - n0).signum() + (recon - n1).signum()) as usize;
    if edge != 2 {
        sum[edge] += src - recon as i64;
        count[edge] += 1;
    }
}

/// Canonical scalar SAO horizontal-EO stats. Reference for the SIMD backend.
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_e0_scalar(
    rec: &[u16],
    rec_stride: usize,
    src: &[u16],
    src_stride: usize,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    sum: &mut [i64; 5],
    count: &mut [u32; 5],
) {
    for y in y0..y0 + h {
        let rrow = y as usize * rec_stride;
        let srow = y as usize * src_stride;
        for x in x0..x0 + w {
            let xi = x as usize;
            let r = rec[rrow + xi] as i32;
            let left = rec[rrow + xi - 1] as i32;
            let right = rec[rrow + xi + 1] as i32;
            let edge = (2 + (r - left).signum() + (r - right).signum()) as usize;
            if edge == 2 {
                continue;
            }
            sum[edge] += src[srow + xi] as i64 - r as i64;
            count[edge] += 1;
        }
    }
}

/// SAO vertical EO stats (EO1). Interior region: `y_start >= 1`, `y_end + 1 <= plane_h`.
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_e1_scalar(
    rec: &[u16],
    rec_stride: usize,
    src: &[u16],
    src_stride: usize,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    sum: &mut [i64; 5],
    count: &mut [u32; 5],
) {
    for y in y0..y0 + h {
        let yi = y as usize;
        let rrow = yi * rec_stride;
        let srow = yi * src_stride;
        for x in x0..x0 + w {
            let xi = x as usize;
            let recon = rec[rrow + xi] as i32;
            let above = rec[(yi - 1) * rec_stride + xi] as i32;
            let below = rec[(yi + 1) * rec_stride + xi] as i32;
            eo_accum(recon, above, below, src[srow + xi] as i64, sum, count);
        }
    }
}

/// SAO 135° EO stats (EO2). Interior: `x_start >= 1`, `y_start >= 1`, both ends+1 within plane.
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_e2_scalar(
    rec: &[u16],
    rec_stride: usize,
    src: &[u16],
    src_stride: usize,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    sum: &mut [i64; 5],
    count: &mut [u32; 5],
) {
    for y in y0..y0 + h {
        let yi = y as usize;
        let rrow = yi * rec_stride;
        let srow = yi * src_stride;
        for x in x0..x0 + w {
            let xi = x as usize;
            let recon = rec[rrow + xi] as i32;
            let nw = rec[(yi - 1) * rec_stride + xi - 1] as i32;
            let se = rec[(yi + 1) * rec_stride + xi + 1] as i32;
            eo_accum(recon, nw, se, src[srow + xi] as i64, sum, count);
        }
    }
}

/// SAO 45° EO stats (EO3). Interior: `x_end + 1 <= plane_w`, `y_start >= 1`, etc.
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_e3_scalar(
    rec: &[u16],
    rec_stride: usize,
    src: &[u16],
    src_stride: usize,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    sum: &mut [i64; 5],
    count: &mut [u32; 5],
) {
    for y in y0..y0 + h {
        let yi = y as usize;
        let rrow = yi * rec_stride;
        let srow = yi * src_stride;
        for x in x0..x0 + w {
            let xi = x as usize;
            let recon = rec[rrow + xi] as i32;
            let ne = rec[(yi - 1) * rec_stride + xi + 1] as i32;
            let sw = rec[(yi + 1) * rec_stride + xi - 1] as i32;
            eo_accum(recon, ne, sw, src[srow + xi] as i64, sum, count);
        }
    }
}

/// SAO Band Offset stats. Accumulates into 32 bands keyed by `rec >> band_shift`.
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_bo_scalar(
    rec: &[u16],
    rec_stride: usize,
    src: &[u16],
    src_stride: usize,
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    band_shift: u8,
    sum: &mut [i64; 32],
    count: &mut [u32; 32],
) {
    for y in y0..y0 + h {
        let rrow = y as usize * rec_stride;
        let srow = y as usize * src_stride;
        for x in x0..x0 + w {
            let xi = x as usize;
            let recon = rec[rrow + xi] as i32;
            let band = ((rec[rrow + xi] >> band_shift) & 31) as usize;
            sum[band] += src[srow + xi] as i64 - recon as i64;
            count[band] += 1;
        }
    }
}

// ─── Forward DCT 1-D ───────────────────────────────────────────────────────

/// Canonical scalar 1-D forward DCT row. Reference for the SIMD backend;
/// dot-products `matrix` against `src` and round-shifts the result.
pub fn forward_dct_1d(src: &[i16], dst: &mut [i16], matrix: &[i32], n: usize, shift: i32) {
    for (k, out) in dst.iter_mut().enumerate().take(n) {
        let mut sum = 0i32;
        for (i, &sample) in src.iter().enumerate().take(n) {
            sum += matrix[k * n + i] * sample as i32;
        }
        *out = round_shift(sum, shift);
    }
}

// ─── 2-D forward transforms ────────────────────────────────────────────────

// Private inner 1-D kernel for DST-4.
fn dst4_1d(src: &[i16], out: &mut [i16; 4], shift: i32) {
    let c0 = src[0] as i32 + src[3] as i32;
    let c1 = src[1] as i32 + src[3] as i32;
    let c2 = src[0] as i32 - src[1] as i32;
    let c3 = 74 * src[2] as i32;
    out[0] = round_shift(29 * c0 + 55 * c1 + c3, shift);
    out[1] = round_shift(74 * (src[0] as i32 + src[1] as i32 - src[3] as i32), shift);
    out[2] = round_shift(29 * c2 + 55 * c0 - c3, shift);
    out[3] = round_shift(55 * c2 - 29 * c1 + c3, shift);
}

/// Canonical scalar 2-D DST-4 (4×4 intra luma). Bit-identical to `forward_dst4_into`.
pub fn fwd_dst4_scalar(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    debug_assert_eq!(residual.len(), 16);
    debug_assert!(out.len() >= 16);
    let shift1 = 1 + bit_depth as i32 - 8;
    let shift2 = 8;
    let mut tmp = [0i16; 16];
    let mut row_out = [0i16; 4];
    for row in 0..4 {
        dst4_1d(&residual[row * 4..row * 4 + 4], &mut row_out, shift1);
        for k in 0..4 {
            tmp[k * 4 + row] = row_out[k];
        }
    }
    for row in 0..4 {
        dst4_1d(&tmp[row * 4..row * 4 + 4], &mut row_out, shift2);
        for k in 0..4 {
            out[k * 4 + row] = row_out[k];
        }
    }
}

// Generic 2-D DCT-N using the dispatched 1-D kernel. Stack workspace up to 32×32=1024 samples.
fn fwd_dct_nxn(residual: &[i16], out: &mut [i16], n: usize, bit_depth: u8) {
    debug_assert_eq!(residual.len(), n * n);
    debug_assert!(out.len() >= n * n);
    debug_assert!(n <= 32);
    let log2_n = n.trailing_zeros() as i32;
    let shift1 = log2_n - 1 + bit_depth as i32 - 8;
    let shift2 = log2_n + 6;
    let matrix = super::dct_matrix(n);
    let mut tmp = [0i16; 1024];
    let mut line = [0i16; 32];
    let mut transformed = [0i16; 32];
    for row in 0..n {
        line[..n].copy_from_slice(&residual[row * n..row * n + n]);
        forward_dct_1d(&line[..n], &mut transformed[..n], matrix, n, shift1);
        for k in 0..n {
            tmp[k * n + row] = transformed[k];
        }
    }
    for row in 0..n {
        line[..n].copy_from_slice(&tmp[row * n..row * n + n]);
        forward_dct_1d(&line[..n], &mut transformed[..n], matrix, n, shift2);
        for k in 0..n {
            out[k * n + row] = transformed[k];
        }
    }
}

/// Canonical scalar 2-D DCT-4 (4×4).
pub fn fwd_dct4_scalar(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    fwd_dct_nxn(residual, out, 4, bit_depth);
}

/// Canonical scalar 2-D DCT-8 (8×8).
pub fn fwd_dct8_scalar(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    fwd_dct_nxn(residual, out, 8, bit_depth);
}

/// Canonical scalar 2-D DCT-16 (16×16).
pub fn fwd_dct16_scalar(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    fwd_dct_nxn(residual, out, 16, bit_depth);
}

/// Canonical scalar 2-D DCT-32 (32×32).
pub fn fwd_dct32_scalar(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    fwd_dct_nxn(residual, out, 32, bit_depth);
}

// ─── Butterfly forward DCT ────────────────────────────────────────────────
// Algebraically equivalent to fwd_dctN_scalar (same coefficients, same
// round_shift), but factored through the DCT butterfly to reduce multiply-add
// count: ~O(N·log N) vs O(N²) for the matrix multiply.
//
// Coefficient correctness: build_dct_matrix uses HEVC_DCT_BASE, which exactly
// matches x265 g_t4/8/16/32 (verified). The butterfly is a pure algebraic
// refactoring of the same dot products, so output is bit-identical.

#[inline(always)]
fn butterfly4_1d(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    for j in 0..line {
        let s = &src[j * 4..j * 4 + 4];
        let e0 = s[0] as i32 + s[3] as i32;
        let e1 = s[1] as i32 + s[2] as i32;
        let o0 = s[0] as i32 - s[3] as i32;
        let o1 = s[1] as i32 - s[2] as i32;
        dst[0 * line + j] = round_shift(64 * e0 + 64 * e1, shift);
        dst[2 * line + j] = round_shift(64 * e0 - 64 * e1, shift);
        dst[1 * line + j] = round_shift(83 * o0 + 36 * o1, shift);
        dst[3 * line + j] = round_shift(36 * o0 - 83 * o1, shift);
    }
}

/// Butterfly 2-D DCT-4. Bit-identical to `fwd_dct4_scalar`.
pub fn fwd_dct4_butterfly(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    debug_assert_eq!(residual.len(), 16);
    debug_assert!(out.len() >= 16);
    let shift1 = 1i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 16];
    butterfly4_1d(residual, &mut tmp, 4, shift1);
    butterfly4_1d(&tmp, out, 4, 8);
}

#[inline(always)]
fn butterfly8_1d(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    for j in 0..line {
        let s = &src[j * 8..j * 8 + 8];
        let e0 = s[0] as i32 + s[7] as i32;
        let e1 = s[1] as i32 + s[6] as i32;
        let e2 = s[2] as i32 + s[5] as i32;
        let e3 = s[3] as i32 + s[4] as i32;
        let o0 = s[0] as i32 - s[7] as i32;
        let o1 = s[1] as i32 - s[6] as i32;
        let o2 = s[2] as i32 - s[5] as i32;
        let o3 = s[3] as i32 - s[4] as i32;
        let ee0 = e0 + e3;
        let ee1 = e1 + e2;
        let eo0 = e0 - e3;
        let eo1 = e1 - e2;
        dst[0 * line + j] = round_shift(64 * ee0 + 64 * ee1, shift);
        dst[4 * line + j] = round_shift(64 * ee0 - 64 * ee1, shift);
        dst[2 * line + j] = round_shift(83 * eo0 + 36 * eo1, shift);
        dst[6 * line + j] = round_shift(36 * eo0 - 83 * eo1, shift);
        dst[1 * line + j] = round_shift(89 * o0 + 75 * o1 + 50 * o2 + 18 * o3, shift);
        dst[3 * line + j] = round_shift(75 * o0 - 18 * o1 - 89 * o2 - 50 * o3, shift);
        dst[5 * line + j] = round_shift(50 * o0 - 89 * o1 + 18 * o2 + 75 * o3, shift);
        dst[7 * line + j] = round_shift(18 * o0 - 50 * o1 + 75 * o2 - 89 * o3, shift);
    }
}

/// Butterfly 2-D DCT-8. Bit-identical to `fwd_dct8_scalar`.
pub fn fwd_dct8_butterfly(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    debug_assert_eq!(residual.len(), 64);
    debug_assert!(out.len() >= 64);
    let shift1 = 2i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 64];
    butterfly8_1d(residual, &mut tmp, 8, shift1);
    butterfly8_1d(&tmp, out, 8, 9);
}

#[allow(clippy::too_many_lines)]
#[inline(always)]
fn butterfly16_1d(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    for j in 0..line {
        let s = &src[j * 16..j * 16 + 16];
        let e0 = s[0] as i32 + s[15] as i32;
        let e1 = s[1] as i32 + s[14] as i32;
        let e2 = s[2] as i32 + s[13] as i32;
        let e3 = s[3] as i32 + s[12] as i32;
        let e4 = s[4] as i32 + s[11] as i32;
        let e5 = s[5] as i32 + s[10] as i32;
        let e6 = s[6] as i32 + s[9] as i32;
        let e7 = s[7] as i32 + s[8] as i32;
        let o0 = s[0] as i32 - s[15] as i32;
        let o1 = s[1] as i32 - s[14] as i32;
        let o2 = s[2] as i32 - s[13] as i32;
        let o3 = s[3] as i32 - s[12] as i32;
        let o4 = s[4] as i32 - s[11] as i32;
        let o5 = s[5] as i32 - s[10] as i32;
        let o6 = s[6] as i32 - s[9] as i32;
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
        // EEE: k=0,8
        dst[ 0 * line + j] = round_shift(64 * eee0 + 64 * eee1, shift);
        dst[ 8 * line + j] = round_shift(64 * eee0 - 64 * eee1, shift);
        // EEO: k=4,12
        dst[ 4 * line + j] = round_shift(83 * eeo0 + 36 * eeo1, shift);
        dst[12 * line + j] = round_shift(36 * eeo0 - 83 * eeo1, shift);
        // EO: k=2,6,10,14
        dst[ 2 * line + j] = round_shift(89*eo0 + 75*eo1 + 50*eo2 + 18*eo3, shift);
        dst[ 6 * line + j] = round_shift(75*eo0 - 18*eo1 - 89*eo2 - 50*eo3, shift);
        dst[10 * line + j] = round_shift(50*eo0 - 89*eo1 + 18*eo2 + 75*eo3, shift);
        dst[14 * line + j] = round_shift(18*eo0 - 50*eo1 + 75*eo2 - 89*eo3, shift);
        // O: k=1,3,5,7,9,11,13,15
        dst[ 1 * line + j] = round_shift(90*o0 + 87*o1 + 80*o2 + 70*o3 + 57*o4 + 43*o5 + 25*o6 +  9*o7, shift);
        dst[ 3 * line + j] = round_shift(87*o0 + 57*o1 +  9*o2 - 43*o3 - 80*o4 - 90*o5 - 70*o6 - 25*o7, shift);
        dst[ 5 * line + j] = round_shift(80*o0 +  9*o1 - 70*o2 - 87*o3 - 25*o4 + 57*o5 + 90*o6 + 43*o7, shift);
        dst[ 7 * line + j] = round_shift(70*o0 - 43*o1 - 87*o2 +  9*o3 + 90*o4 + 25*o5 - 80*o6 - 57*o7, shift);
        dst[ 9 * line + j] = round_shift(57*o0 - 80*o1 - 25*o2 + 90*o3 -  9*o4 - 87*o5 + 43*o6 + 70*o7, shift);
        dst[11 * line + j] = round_shift(43*o0 - 90*o1 + 57*o2 + 25*o3 - 87*o4 + 70*o5 +  9*o6 - 80*o7, shift);
        dst[13 * line + j] = round_shift(25*o0 - 70*o1 + 90*o2 - 80*o3 + 43*o4 +  9*o5 - 57*o6 + 87*o7, shift);
        dst[15 * line + j] = round_shift( 9*o0 - 25*o1 + 43*o2 - 57*o3 + 70*o4 - 80*o5 + 87*o6 - 90*o7, shift);
    }
}

/// Butterfly 2-D DCT-16. Bit-identical to `fwd_dct16_scalar`.
pub fn fwd_dct16_butterfly(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    debug_assert_eq!(residual.len(), 256);
    debug_assert!(out.len() >= 256);
    let shift1 = 3i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 256];
    butterfly16_1d(residual, &mut tmp, 16, shift1);
    butterfly16_1d(&tmp, out, 16, 10);
}

// DCT-32 butterfly coefficient tables (first N/2 elements per row only; the
// second half is antisymmetric and absorbed into the O[] butterfly values).
// O[k] rows (k=1,3,…,31 → indices 0..16 in table), 16 coefficients each.
const DCT32_O: [[i32; 16]; 16] = [
    [ 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13,  4], // k=1
    [ 90, 82, 67, 46, 22, -4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13], // k=3
    [ 88, 67, 31,-13,-54,-82,-90,-78,-46, -4, 38, 73, 90, 85, 61, 22], // k=5
    [ 85, 46,-13,-67,-90,-73,-22, 38, 82, 88, 54, -4,-61,-90,-78,-31], // k=7
    [ 82, 22,-54,-90,-61, 13, 78, 85, 31,-46,-90,-67,  4, 73, 88, 38], // k=9
    [ 78, -4,-82,-73, 13, 85, 67,-22,-88,-61, 31, 90, 54,-38,-90,-46], // k=11
    [ 73,-31,-90,-22, 78, 67,-38,-90,-13, 82, 61,-46,-88, -4, 85, 54], // k=13
    [ 67,-54,-78, 38, 85,-22,-90,  4, 90, 13,-88,-31, 82, 46,-73,-61], // k=15
    [ 61,-73,-46, 82, 31,-88,-13, 90, -4,-90, 22, 85,-38,-78, 54, 67], // k=17
    [ 54,-85, -4, 88,-46,-61, 82, 13,-90, 38, 67,-78,-22, 90,-31,-73], // k=19
    [ 46,-90, 38, 54,-90, 31, 61,-88, 22, 67,-85, 13, 73,-82,  4, 78], // k=21
    [ 38,-88, 73, -4,-67, 90,-46,-31, 85,-78, 13, 61,-90, 54, 22,-82], // k=23
    [ 31,-78, 90,-61,  4, 54,-88, 82,-38,-22, 73,-90, 67,-13,-46, 85], // k=25
    [ 22,-61, 85,-90, 73,-38, -4, 46,-78, 90,-82, 54,-13,-31, 67,-88], // k=27
    [ 13,-38, 61,-78, 88,-90, 85,-73, 54,-31,  4, 22,-46, 67,-82, 90], // k=29
    [  4,-13, 22,-31, 38,-46, 54,-61, 67,-73, 78,-82, 85,-88, 90,-90], // k=31
];
// EO[k] rows (k=2,6,…,30 → indices 0..8 in table), 8 coefficients each.
const DCT32_EO: [[i32; 8]; 8] = [
    [90, 87, 80, 70, 57, 43, 25,  9], // k=2
    [87, 57,  9,-43,-80,-90,-70,-25], // k=6
    [80,  9,-70,-87,-25, 57, 90, 43], // k=10
    [70,-43,-87,  9, 90, 25,-80,-57], // k=14
    [57,-80,-25, 90, -9,-87, 43, 70], // k=18
    [43,-90, 57, 25,-87, 70,  9,-80], // k=22
    [25,-70, 90,-80, 43,  9,-57, 87], // k=26
    [ 9,-25, 43,-57, 70,-80, 87,-90], // k=30
];
// EEO[k] rows (k=4,12,20,28 → indices 0..4), 4 coefficients each.
const DCT32_EEO: [[i32; 4]; 4] = [
    [89, 75, 50, 18], // k=4
    [75,-18,-89,-50], // k=12
    [50,-89, 18, 75], // k=20
    [18,-50, 75,-89], // k=28
];

#[inline(always)]
fn dot16(a: [i32; 16], c: &[i32; 16]) -> i32 {
    let mut s = 0i32;
    for i in 0..16 {
        s += a[i] * c[i];
    }
    s
}

#[inline(always)]
fn dot8(a: [i32; 8], c: &[i32; 8]) -> i32 {
    let mut s = 0i32;
    for i in 0..8 {
        s += a[i] * c[i];
    }
    s
}

#[inline(always)]
fn dot4(a: [i32; 4], c: &[i32; 4]) -> i32 {
    a[0] * c[0] + a[1] * c[1] + a[2] * c[2] + a[3] * c[3]
}

fn butterfly32_1d(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    for j in 0..line {
        let s = &src[j * 32..j * 32 + 32];
        let o: [i32; 16] = std::array::from_fn(|k| s[k] as i32 - s[31 - k] as i32);
        let e: [i32; 16] = std::array::from_fn(|k| s[k] as i32 + s[31 - k] as i32);
        let eo: [i32; 8] = std::array::from_fn(|k| e[k] - e[15 - k]);
        let ee: [i32; 8] = std::array::from_fn(|k| e[k] + e[15 - k]);
        let eeo: [i32; 4] = std::array::from_fn(|k| ee[k] - ee[7 - k]);
        let eee: [i32; 4] = std::array::from_fn(|k| ee[k] + ee[7 - k]);
        let eeeo0 = eee[0] - eee[3];
        let eeeo1 = eee[1] - eee[2];
        let eeee0 = eee[0] + eee[3];
        let eeee1 = eee[1] + eee[2];

        // EEEE: k=0,16
        dst[ 0 * line + j] = round_shift(64 * eeee0 + 64 * eeee1, shift);
        dst[16 * line + j] = round_shift(64 * eeee0 - 64 * eeee1, shift);
        // EEEO: k=8,24
        dst[ 8 * line + j] = round_shift(83 * eeeo0 + 36 * eeeo1, shift);
        dst[24 * line + j] = round_shift(36 * eeeo0 - 83 * eeeo1, shift);
        // EEO: k=4,12,20,28
        let eeo4 = [eeo[0], eeo[1], eeo[2], eeo[3]];
        for (i, c) in DCT32_EEO.iter().enumerate() {
            dst[(4 + i * 8) * line + j] = round_shift(dot4(eeo4, c), shift);
        }
        // EO: k=2,6,10,14,18,22,26,30
        let eo8 = [eo[0], eo[1], eo[2], eo[3], eo[4], eo[5], eo[6], eo[7]];
        for (i, c) in DCT32_EO.iter().enumerate() {
            dst[(2 + i * 4) * line + j] = round_shift(dot8(eo8, c), shift);
        }
        // O: k=1,3,5,...,31
        for (i, c) in DCT32_O.iter().enumerate() {
            dst[(1 + i * 2) * line + j] = round_shift(dot16(o, c), shift);
        }
    }
}

/// Butterfly 2-D DCT-32. Bit-identical to `fwd_dct32_scalar`.
pub fn fwd_dct32_butterfly(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    debug_assert_eq!(residual.len(), 1024);
    debug_assert!(out.len() >= 1024);
    let shift1 = 4i32 + bit_depth as i32 - 8;
    let mut tmp = [0i16; 1024];
    butterfly32_1d(residual, &mut tmp, 32, shift1);
    butterfly32_1d(&tmp, out, 32, 11);
}
