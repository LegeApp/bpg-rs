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
