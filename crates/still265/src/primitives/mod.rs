//! Rust-local primitive kernels used by the still encoder.
//!
//! The scalar implementations in this module are the **canonical reference**:
//! every optimized (SIMD/ASM) kernel must produce byte-identical output to
//! them. Hot kernels are dispatched through [`PRIMITIVES`], a function-pointer
//! table selected once at startup from the runtime CPU features and the
//! `BPG_PRIMITIVES` environment variable (`scalar`/`simd`/`asm`/`auto`).
//!
//! Dispatched kernels: 8-bit SATD (SSE2 on `x86_64`), plus 10/12-bit SATD, SSD,
//! residual subtraction, and the forward 1-D DCT (portable `wide`-SIMD, behind
//! the optional `wide-simd` feature). See `docs/remaining-gaps.md` for the
//! not-yet-vectorized kernels.

use bpg_hevc_decode::hevc::transform as dec_transform;
use std::sync::LazyLock;

pub mod intra_angs;

#[cfg(target_arch = "x86_64")]
mod simd_x86;

#[cfg(feature = "wide-simd")]
mod wide_simd;

static DCT4: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(4));
static DCT8: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(8));
static DCT16: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(16));
static DCT32: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(32));

#[inline]
fn round_shift(value: i32, shift: i32) -> i16 {
    let add = 1i32 << (shift - 1);
    ((value + add) >> shift).clamp(-32768, 32767) as i16
}

fn build_dct_matrix(n: usize) -> Vec<i32> {
    let mut matrix = vec![0i32; n * n];
    for k in 0..n {
        for i in 0..n {
            matrix[k * n + i] = if k == 0 {
                64
            } else {
                let scale = 64.0f64 * 2.0f64.sqrt();
                let angle = std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64;
                (scale * angle.cos()).round() as i32
            };
        }
    }
    matrix
}

fn dct_matrix(n: usize) -> &'static [i32] {
    match n {
        4 => &DCT4,
        8 => &DCT8,
        16 => &DCT16,
        32 => &DCT32,
        _ => unreachable!("unsupported DCT size"),
    }
}

fn forward_dct_1d(src: &[i16], dst: &mut [i16], matrix: &[i32], n: usize, shift: i32) {
    for (k, out) in dst.iter_mut().enumerate().take(n) {
        let mut sum = 0i32;
        for (i, &sample) in src.iter().enumerate().take(n) {
            sum += matrix[k * n + i] * sample as i32;
        }
        *out = round_shift(sum, shift);
    }
}

fn forward_dct_into(
    residual: &[i16],
    log2_size: u8,
    bit_depth: u8,
    out: &mut Vec<i16>,
    tmp: &mut Vec<i16>,
) {
    let n = 1usize << log2_size;
    assert_eq!(residual.len(), n * n);

    let shift1 = log2_size as i32 - 1 + bit_depth as i32 - 8;
    let shift2 = log2_size as i32 + 6;
    tmp.resize(n * n, 0);
    out.resize(n * n, 0);
    let matrix = dct_matrix(n);

    // Dispatched 1-D transform kernel (scalar or `wide`-SIMD); see [`PRIMITIVES`].
    let dct1d = PRIMITIVES.forward_dct_1d;

    let mut line = [0i16; 32];
    let mut transformed = [0i16; 32];

    for row in 0..n {
        line[..n].copy_from_slice(&residual[row * n..row * n + n]);
        dct1d(&line[..n], &mut transformed[..n], matrix, n, shift1);
        for k in 0..n {
            tmp[k * n + row] = transformed[k];
        }
    }

    for row in 0..n {
        line[..n].copy_from_slice(&tmp[row * n..row * n + n]);
        dct1d(&line[..n], &mut transformed[..n], matrix, n, shift2);
        for k in 0..n {
            out[k * n + row] = transformed[k];
        }
    }
}

fn forward_dct(residual: &[i16], log2_size: u8, bit_depth: u8) -> Vec<i16> {
    let mut out = Vec::new();
    let mut tmp = Vec::new();
    forward_dct_into(residual, log2_size, bit_depth, &mut out, &mut tmp);
    out
}

fn forward_dst4_into(residual: &[i16], bit_depth: u8, out: &mut Vec<i16>) {
    assert_eq!(residual.len(), 16);

    let shift1 = 1 + bit_depth as i32 - 8;
    let shift2 = 8;
    let mut tmp = [0i16; 16];
    let mut local = [0i16; 16];

    for row in 0..4 {
        forward_dst4_1d(&residual[row * 4..row * 4 + 4], &mut tmp[row..], 4, shift1);
    }

    for row in 0..4 {
        forward_dst4_1d(&tmp[row * 4..row * 4 + 4], &mut local[row..], 4, shift2);
    }

    out.clear();
    out.extend_from_slice(&local);
}

fn forward_dst4(residual: &[i16], bit_depth: u8) -> Vec<i16> {
    let mut out = Vec::with_capacity(16);
    forward_dst4_into(residual, bit_depth, &mut out);
    out
}

fn forward_dst4_1d(src: &[i16], dst: &mut [i16], line: usize, shift: i32) {
    let c0 = src[0] as i32 + src[3] as i32;
    let c1 = src[1] as i32 + src[3] as i32;
    let c2 = src[0] as i32 - src[1] as i32;
    let c3 = 74 * src[2] as i32;

    dst[0] = round_shift(29 * c0 + 55 * c1 + c3, shift);
    dst[line] = round_shift(74 * (src[0] as i32 + src[1] as i32 - src[3] as i32), shift);
    dst[2 * line] = round_shift(29 * c2 + 55 * c0 - c3, shift);
    dst[3 * line] = round_shift(55 * c2 - 29 * c1 + c3, shift);
}

pub fn forward_transform(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
) -> Vec<i16> {
    match (log2_size, is_intra_4x4_luma) {
        (2, true) => forward_dst4(residual, bit_depth),
        (2..=5, _) => forward_dct(residual, log2_size, bit_depth),
        _ => panic!("unsupported transform size log2={log2_size}"),
    }
}

pub fn forward_transform_into(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
    out: &mut Vec<i16>,
    tmp: &mut Vec<i16>,
) {
    match (log2_size, is_intra_4x4_luma) {
        (2, true) => forward_dst4_into(residual, bit_depth, out),
        (2..=5, _) => forward_dct_into(residual, log2_size, bit_depth, out, tmp),
        _ => panic!("unsupported transform size log2={log2_size}"),
    }
}

pub fn inverse_transform(
    coeffs: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
) -> Vec<i16> {
    let n = 1usize << log2_size;
    let mut out = vec![0i16; n * n];
    dec_transform::inverse_transform(coeffs, &mut out, n, bit_depth, is_intra_4x4_luma);
    out
}

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

/// Sum of absolute Hadamard-transformed differences for an 8-bit block.
///
/// Dispatched through [`PRIMITIVES`] (SSE2 on `x86_64`, scalar elsewhere or
/// when `BPG_PRIMITIVES=scalar`). All backends are byte-identical to
/// [`satd_u8_scalar`].
pub fn satd_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.satd_u8)(a, stride_a, b, stride_b, size)
}

/// Sum of absolute Hadamard-transformed differences for a 10/12-bit block.
/// Dispatched through [`PRIMITIVES`] (portable `wide` SIMD under the
/// `wide-simd` feature, else scalar); byte-identical to [`satd_u16_scalar`].
pub fn satd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.satd_u16)(a, stride_a, b, stride_b, size)
}

/// Canonical scalar 8-bit SATD. Reference for all optimized backends.
pub fn satd_u8_scalar(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

/// Canonical scalar 10/12-bit SATD. Reference for all optimized backends.
pub fn satd_u16_scalar(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

/// Sum of squared differences over a `size`x`size` block (HEVC distortion).
/// Dispatched through [`PRIMITIVES`]; byte-identical to [`ssd_u16_scalar`].
/// Maps to x265's `pixel_ssd` (`ssd-a.asm`).
pub fn ssd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u64 {
    (PRIMITIVES.ssd_u16)(a, stride_a, b, stride_b, size)
}

/// Canonical scalar SSD. Reference for all optimized backends.
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

/// Residual generation: `out[j*size+i] = src[..] - pred[..]` for a `size`x`size`
/// block, where `out` is contiguous (stride `size`). Dispatched through
/// [`PRIMITIVES`]; byte-identical to [`sub_residual_scalar`]. Maps to x265's
/// `pixel_sub_ps` / `getResidual*` (`pixel-util8.asm`).
pub fn sub_residual(
    src: &[u16],
    src_stride: usize,
    pred: &[u16],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    (PRIMITIVES.sub_residual)(src, src_stride, pred, pred_stride, out, size)
}

/// Canonical scalar residual subtraction. Reference for all optimized backends.
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

/// Inverse quantization (H.265 8.6.3, flat scaling list) of a coefficient
/// block in place: `level = clip((level * combined + add) >> shift)`, where
/// `add = 1 << (shift-1)` for `shift > 0`. Dispatched through [`PRIMITIVES`];
/// byte-identical to [`dequantize_scalar`]. Maps to x265's `dequant_normal`
/// (`quant-a.asm`) — an RDO/reconstruction hot path (every TU trial + final).
pub fn dequantize(levels: &mut [i16], combined: i32, shift: i32) {
    (PRIMITIVES.dequantize)(levels, combined, shift)
}

/// Canonical scalar inverse quantization. Reference for all optimized backends;
/// kept bit-identical to `bpg-hevc-decode`'s decoder dequant.
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

/// Sample-Adaptive-Offset horizontal edge-offset (`eo_class == 0`) statistics
/// over an interior region whose every pixel has valid left/right neighbours
/// and an unclamped co-located source sample. Accumulates `sum[edgeIdx] +=
/// src - rec` and `count[edgeIdx] += 1` for `edgeIdx` in `0,1,3,4` (the "no
/// edge" category 2 is skipped). Dispatched through [`PRIMITIVES`];
/// byte-identical to [`sao_stats_e0_scalar`]. Maps to x265's `saoCuStatsE0`
/// (`loopfilter.asm`).
#[allow(clippy::too_many_arguments)]
pub fn sao_stats_e0(
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
    (PRIMITIVES.sao_stats_e0)(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

/// Canonical scalar SAO horizontal-EO stats. Reference for the SIMD backend;
/// matches the per-pixel logic in the encoder's `sao_eo_stats` for `eo_class 0`
/// over a region with valid neighbours (no plane-edge / source clamping).
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

/// Function-pointer table of the dispatchable hot kernels. Selected once,
/// lazily, by [`select_primitives`] from the CPU features and the
/// `BPG_PRIMITIVES` environment variable.
pub struct Primitives {
    pub satd_u8: fn(&[u8], usize, &[u8], usize, usize) -> u32,
    pub satd_u16: fn(&[u16], usize, &[u16], usize, usize) -> u32,
    pub ssd_u16: fn(&[u16], usize, &[u16], usize, usize) -> u64,
    pub sub_residual: fn(&[u16], usize, &[u16], usize, &mut [i16], usize),
    pub dequantize: fn(&mut [i16], i32, i32),
    #[allow(clippy::type_complexity)]
    pub sao_stats_e0:
        fn(&[u16], usize, &[u16], usize, u32, u32, u32, u32, &mut [i64; 5], &mut [u32; 5]),
    pub forward_dct_1d: fn(&[i16], &mut [i16], &[i32], usize, i32),
    /// Batched angular intra prediction (modes 2..=34) for the luma rough search.
    /// `(dst, unfiltered_border, filtered_border, center, log2_size, c_idx, bit_depth)`.
    pub intra_pred_allangs: fn(&mut [u16], &[i32], &[i32], usize, u8, u8, u8),
    /// Human-readable name of the selected backend, for `--debug-stats`.
    pub backend: &'static str,
}

/// Batched angular intra prediction, dispatched through [`PRIMITIVES`].
/// Byte-identical to per-mode `predict_intra_into` for modes 2..=34.
pub fn intra_pred_allangs(
    dst: &mut [u16],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra_pred_allangs)(
        dst, unfiltered, filtered, center, log2_size, c_idx, bit_depth,
    )
}

/// The active primitive backend, chosen once on first use.
pub static PRIMITIVES: LazyLock<Primitives> = LazyLock::new(select_primitives);

/// Which optimized tier the build requested, after resolving the
/// `BPG_PRIMITIVES` override against what is actually available.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Scalar,
    /// Best available SIMD (and, in future, ASM) kernels.
    Simd,
}

fn select_primitives() -> Primitives {
    let choice = match std::env::var("BPG_PRIMITIVES").ok().as_deref() {
        Some("scalar") => BackendChoice::Scalar,
        // `asm` has no kernels yet, so it resolves to the best SIMD path.
        Some("simd") | Some("asm") | Some("auto") | None => BackendChoice::Simd,
        Some(other) => {
            eprintln!("warning: unknown BPG_PRIMITIVES={other:?}, using scalar");
            BackendChoice::Scalar
        }
    };

    // Scalar reference table.
    let mut p = Primitives {
        satd_u8: satd_u8_scalar,
        satd_u16: satd_u16_scalar,
        ssd_u16: ssd_u16_scalar,
        sub_residual: sub_residual_scalar,
        dequantize: dequantize_scalar,
        sao_stats_e0: sao_stats_e0_scalar,
        forward_dct_1d,
        intra_pred_allangs: intra_angs::intra_pred_allangs_scalar,
        backend: "scalar",
    };

    #[cfg(target_arch = "x86_64")]
    if choice == BackendChoice::Simd && std::arch::is_x86_feature_detected!("sse2") {
        p.satd_u8 = simd_x86::satd_u8_sse2;
        p.backend = "sse2";
    }

    // Portable-SIMD (`wide`) kernels: SSD, residual subtraction, forward DCT.
    // Bit-identical to the scalar references (enforced by `wide_simd::tests`).
    #[cfg(feature = "wide-simd")]
    if choice == BackendChoice::Simd {
        p.satd_u16 = wide_simd::satd_u16;
        p.ssd_u16 = wide_simd::ssd_u16;
        p.sub_residual = wide_simd::sub_residual;
        p.dequantize = wide_simd::dequantize;
        p.sao_stats_e0 = wide_simd::sao_stats_e0;
        p.forward_dct_1d = wide_simd::forward_dct_1d;
        p.intra_pred_allangs = wide_simd::intra_pred_allangs;
        p.backend = if p.backend == "sse2" {
            "sse2+wide"
        } else {
            "wide"
        };
    }

    let _ = choice; // used on non-x86_64 only to silence unused warnings
    p
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct4_constant_block_is_pure_dc() {
        let coeffs = forward_transform(&[8i16; 16], 2, false, 8);
        assert_eq!(coeffs[0], 128 * 8);
        assert!(coeffs[1..].iter().all(|&c| c == 0), "{coeffs:?}");
    }

    #[test]
    fn forward_inverse_round_trips_fixed_vectors() {
        for log2_size in 2..=5 {
            let n = 1usize << (2 * log2_size);
            let block: Vec<i16> = (0..n).map(|i| ((i as i16 * 3) % 17) - 8).collect();
            let coeffs = forward_transform(&block, log2_size, false, 8);
            let back = inverse_transform(&coeffs, log2_size, false, 8);
            assert_eq!(back, block, "DCT round-trip mismatch log2={log2_size}");
        }
    }

    /// Deterministic xorshift so patterns are reproducible without a dep.
    fn fill_pattern(buf: &mut [u8], kind: u32, seed: &mut u32) {
        for (i, p) in buf.iter_mut().enumerate() {
            *p = match kind {
                0 => 0,                          // flat
                1 => 255,                        // saturated
                2 => (i as u8).wrapping_mul(17), // gradient
                3 => {
                    if i % 2 == 0 {
                        0
                    } else {
                        255
                    }
                } // alternating
                _ => {
                    // xorshift32 pseudo-random
                    let mut x = *seed;
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    *seed = x;
                    x as u8
                }
            };
        }
    }

    #[test]
    fn sse2_satd_u8_matches_scalar() {
        // Exercise the active dispatched kernel (SSE2 on this host) against the
        // canonical scalar reference, with a non-trivial stride.
        let mut seed = 0x1234_5678u32;
        for &size in &[4usize, 8, 16, 32] {
            let stride = size + 3;
            let mut a = vec![0u8; stride * size];
            let mut b = vec![0u8; stride * size];
            for ka in 0..5 {
                for kb in 0..5 {
                    fill_pattern(&mut a, ka, &mut seed);
                    fill_pattern(&mut b, kb, &mut seed);
                    let want = satd_u8_scalar(&a, stride, &b, stride, size);
                    let got = (PRIMITIVES.satd_u8)(&a, stride, &b, stride, size);
                    assert_eq!(got, want, "size={size} pattern a={ka} b={kb}");
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sse2_satd_u8_randomized_matches_scalar() {
        let mut seed = 0x9e37_79b9u32;
        for _ in 0..200 {
            for &size in &[4usize, 8, 16, 32] {
                let stride = size + (seed as usize & 7);
                let mut a = vec![0u8; stride * size];
                let mut b = vec![0u8; stride * size];
                fill_pattern(&mut a, 99, &mut seed);
                fill_pattern(&mut b, 99, &mut seed);
                let want = satd_u8_scalar(&a, stride, &b, stride, size);
                let got = super::simd_x86::satd_u8_sse2(&a, stride, &b, stride, size);
                assert_eq!(got, want, "size={size} stride={stride}");
            }
        }
    }
}
