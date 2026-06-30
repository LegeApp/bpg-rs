//! Primitive kernel dispatch table for the still265 encoder.
//!
//! The encoder calls thin public wrappers (`satd_u8`, `ssd_u16`, …) which
//! dispatch through [`PRIMITIVES`], a function-pointer table selected once at
//! startup from the runtime CPU features and the `BPG_PRIMITIVES` environment
//! variable (`scalar`/`simd`/`asm`/`auto`).
//!
//! ## Backend stack (best first, each overwrites table entries from the one before)
//!
//! 1. `scalar` — portable canonical Rust (always available; `BPG_PRIMITIVES=scalar` stops here)
//! 2. `wide`   — portable SIMD via the `wide` crate (no ISA-specific intrinsics)
//! 3. `x86::sse2`  — x86_64 SSE2 intrinsics (unconditional on `x86_64`)
//! 4. `x86::avx2`  — x86_64 AVX2 intrinsics (runtime-detected)
//!
//! All optimised backends are **bit-identical** to their scalar references,
//! enforced by tests in each sub-module.

use bpg_hevc_decode::hevc::transform as dec_transform;
use std::sync::LazyLock;

pub mod intra;
pub mod scalar;
pub mod wide;

#[cfg(target_arch = "x86_64")]
mod x86;

// ─── Backward-compatibility re-exports ────────────────────────────────────

/// Keep the old module path `crate::primitives::intra_angs` working.
pub use intra::angs as intra_angs;

// Re-export scalar functions so sub-module tests can reach them via
// `super::super::*` (e.g. `wide.rs` tests). Also re-exported for use by
// the public caller of `ssd_u8`.
#[allow(unused_imports)]
pub use scalar::{
    abs_sum_i16_scalar, add_clip_u8_scalar, add_clip_u16_scalar, count_nonzero_scalar,
    dequantize_scalar, forward_dct_1d, fwd_dct4_butterfly, fwd_dct4_scalar, fwd_dct8_butterfly,
    fwd_dct8_scalar, fwd_dct16_butterfly, fwd_dct16_scalar, fwd_dct32_butterfly, fwd_dct32_scalar,
    fwd_dst4_scalar, last_nonzero_scalar, narrow_u16_to_u8_scalar, quantize_scalar, sa8d_u8_scalar,
    sa8d_u16_scalar, sao_stats_bo_scalar, sao_stats_e0_scalar, sao_stats_e1_scalar,
    sao_stats_e2_scalar, sao_stats_e3_scalar, satd_u8_scalar, satd_u16_scalar, ssd_u8_scalar,
    ssd_u16_scalar, sub_residual_scalar, sub_residual_u8_scalar,
};

// ─── DCT matrix infrastructure ────────────────────────────────────────────
// Shared by scalar.rs and the dispatch functions below.

static DCT4: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(4));
static DCT8: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(8));
static DCT16: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(16));
static DCT32: LazyLock<Vec<i32>> = LazyLock::new(|| build_dct_matrix(32));

#[inline]
pub(crate) fn round_shift(value: i32, shift: i32) -> i16 {
    let add = 1i32 << (shift - 1);
    ((value + add) >> shift).clamp(-32768, 32767) as i16
}

/// Canonical HEVC first-quadrant cosine constants (H.265 §8.6.4.2) — the master
/// table every DCT-II transform matrix is sub-sampled from. These are hand-tuned
/// integers, **not** analytic roundings: the π/8 entry is `83`, whereas
/// `round(64·√2·cos(π/8)) = 84`. Using the analytic value desynchronises the
/// encoder's forward (analysis) basis from the decoder's spec inverse (synthesis)
/// basis — the two stop being an exact transform pair, so the encoder quantises
/// coefficients for one basis while the decoder reconstructs with another,
/// injecting ~1% coherent inter-band leakage (a ~52 dB reconstruction ceiling)
/// into every block at all bitrates. This is the same table the decoder's inverse
/// transform is built on, so forward and inverse now round-trip losslessly.
const HEVC_DCT_BASE: [i32; 32] = [
    64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46, 43, 38,
    36, 31, 25, 22, 18, 13, 9, 4,
];

/// `cos(i·π/64)` on the HEVC integer scale (±[`HEVC_DCT_BASE`]).
#[inline]
fn hevc_cos(i: usize) -> i32 {
    match i % 128 {
        i @ 0..=31 => HEVC_DCT_BASE[i],
        32 | 96 => 0,
        i @ 33..=63 => -HEVC_DCT_BASE[64 - i],
        i @ 64..=95 => -HEVC_DCT_BASE[i - 64],
        i => HEVC_DCT_BASE[128 - i],
    }
}

fn build_dct_matrix(n: usize) -> Vec<i32> {
    let step = 32 / n;
    let mut matrix = vec![0i32; n * n];
    for k in 0..n {
        for col in 0..n {
            matrix[k * n + col] = hevc_cos((2 * col + 1) * k * step);
        }
    }
    matrix
}

pub(crate) fn dct_matrix(n: usize) -> &'static [i32] {
    match n {
        4 => &DCT4,
        8 => &DCT8,
        16 => &DCT16,
        32 => &DCT32,
        _ => unreachable!("unsupported DCT size"),
    }
}

// ─── High-level transform functions ───────────────────────────────────────

pub fn forward_transform(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
) -> Vec<i16> {
    let n = 1usize << log2_size;
    let mut out = vec![0i16; n * n];
    let p = &*PRIMITIVES;
    match (log2_size, is_intra_4x4_luma) {
        (2, true) => (p.transform.fwd_dst4)(residual, &mut out, bit_depth),
        (2, false) => (p.transform.fwd_dct4)(residual, &mut out, bit_depth),
        (3, _) => (p.transform.fwd_dct8)(residual, &mut out, bit_depth),
        (4, _) => (p.transform.fwd_dct16)(residual, &mut out, bit_depth),
        (5, _) => (p.transform.fwd_dct32)(residual, &mut out, bit_depth),
        _ => panic!("unsupported transform size log2={log2_size}"),
    }
    out
}

pub fn forward_transform_into(
    residual: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
    out: &mut Vec<i16>,
    _tmp: &mut Vec<i16>,
) {
    let n = 1usize << log2_size;
    out.resize(n * n, 0);
    let p = &*PRIMITIVES;
    match (log2_size, is_intra_4x4_luma) {
        (2, true) => (p.transform.fwd_dst4)(residual, out, bit_depth),
        (2, false) => (p.transform.fwd_dct4)(residual, out, bit_depth),
        (3, _) => (p.transform.fwd_dct8)(residual, out, bit_depth),
        (4, _) => (p.transform.fwd_dct16)(residual, out, bit_depth),
        (5, _) => (p.transform.fwd_dct32)(residual, out, bit_depth),
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

pub fn inverse_transform_into(
    coeffs: &[i16],
    log2_size: u8,
    is_intra_4x4_luma: bool,
    bit_depth: u8,
    out: &mut Vec<i16>,
) {
    let n = 1usize << log2_size;
    out.clear();
    out.resize(n * n, 0);
    dec_transform::inverse_transform(coeffs, out, n, bit_depth, is_intra_4x4_luma);
}

// ─── Public kernel wrappers ────────────────────────────────────────────────

/// 8-bit SATD (Sum of Absolute Hadamard-Transformed Differences).
/// Dispatched through [`PRIMITIVES`]; byte-identical to [`satd_u8_scalar`].
pub fn satd_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.pixel.satd_u8)(a, stride_a, b, stride_b, size)
}

/// 10/12-bit SATD. Dispatched through [`PRIMITIVES`]; byte-identical to [`satd_u16_scalar`].
pub fn satd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.pixel.satd_u16)(a, stride_a, b, stride_b, size)
}

/// 8-bit x265-style CU SA8D.
pub fn sa8d_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.pixel.sa8d_u8)(a, stride_a, b, stride_b, size)
}

/// 10/12-bit x265-style CU SA8D.
pub fn sa8d_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    (PRIMITIVES.pixel.sa8d_u16)(a, stride_a, b, stride_b, size)
}

/// Sum of squared differences (8-bit). Dispatched; byte-identical to [`ssd_u8_scalar`].
pub fn ssd_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u64 {
    (PRIMITIVES.pixel.ssd_u8)(a, stride_a, b, stride_b, size)
}

/// Sum of squared differences (u16 path). Dispatched; byte-identical to [`ssd_u16_scalar`].
pub fn ssd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u64 {
    (PRIMITIVES.pixel.ssd_u16)(a, stride_a, b, stride_b, size)
}

/// 8-bit residual subtraction. Dispatched; byte-identical to [`sub_residual_u8_scalar`].
pub fn sub_residual_u8(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    (PRIMITIVES.pixel.sub_residual_u8)(src, src_stride, pred, pred_stride, out, size)
}

/// Add residual to 8-bit prediction and clip. Dispatched; byte-identical to [`add_clip_u8_scalar`].
pub fn add_clip_u8(pred: &[u8], residual: &[i16], out: &mut [u8], n: usize) {
    (PRIMITIVES.pixel.add_clip_u8)(pred, residual, out, n)
}

/// Add residual to u16 prediction and clip. Dispatched; byte-identical to [`add_clip_u16_scalar`].
pub fn add_clip_u16(pred: &[u16], residual: &[i16], out: &mut [u16], n: usize, max: u16) {
    (PRIMITIVES.pixel.add_clip_u16)(pred, residual, out, n, max)
}

/// Narrow u16 prediction to u8 (clip to [0,255]). Dispatched; byte-identical to [`narrow_u16_to_u8_scalar`].
pub fn narrow_u16_to_u8(src: &[u16], dst: &mut [u8], n: usize) {
    (PRIMITIVES.pixel.narrow_u16_to_u8)(src, dst, n)
}

/// Residual generation (u16 path). Dispatched; byte-identical to [`sub_residual_scalar`].
#[allow(dead_code)]
pub fn sub_residual(
    src: &[u16],
    src_stride: usize,
    pred: &[u16],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    (PRIMITIVES.residual.sub_residual)(src, src_stride, pred, pred_stride, out, size)
}

/// Inverse quantization in place. Dispatched; byte-identical to [`dequantize_scalar`].
pub fn dequantize(levels: &mut [i16], combined: i32, shift: i32) {
    (PRIMITIVES.quant.dequantize)(levels, combined, shift)
}

/// Forward quantization. Dispatched; byte-identical to [`quantize_scalar`].
pub fn quantize(coeffs: &[i16], levels: &mut [i16], scale: i32, add: i32, qbits: i32) -> u32 {
    (PRIMITIVES.quant.quantize)(coeffs, levels, scale, add, qbits)
}

/// SAO vertical EO stats (EO1). Dispatched.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn sao_stats_e1(
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
    (PRIMITIVES.sao.stats_e1)(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

/// SAO 135° EO stats (EO2). Dispatched.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn sao_stats_e2(
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
    (PRIMITIVES.sao.stats_e2)(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

/// SAO 45° EO stats (EO3). Dispatched.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn sao_stats_e3(
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
    (PRIMITIVES.sao.stats_e3)(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

/// SAO Band Offset stats. Dispatched.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn sao_stats_bo(
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
    (PRIMITIVES.sao.stats_bo)(
        rec, rec_stride, src, src_stride, x0, y0, w, h, band_shift, sum, count,
    )
}

/// SAO horizontal edge-offset stats. Dispatched; byte-identical to [`sao_stats_e0_scalar`].
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
    (PRIMITIVES.sao.stats_e0)(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

/// Count nonzero i16 elements in a flat slice. Dispatched; byte-identical to scalar.
#[allow(dead_code)]
pub fn count_nonzero(levels: &[i16]) -> u32 {
    (PRIMITIVES.residual.count_nonzero)(levels)
}

/// Absolute sum of i16 values in a flat slice. Dispatched; byte-identical to scalar.
#[allow(dead_code)]
pub fn abs_sum_i16(levels: &[i16]) -> u64 {
    (PRIMITIVES.residual.abs_sum_i16)(levels)
}

/// Index of the last nonzero element in linear order. Dispatched; byte-identical to scalar.
#[allow(dead_code)]
pub fn last_nonzero(levels: &[i16]) -> Option<usize> {
    (PRIMITIVES.residual.last_nonzero)(levels)
}

/// Batched all-angular intra prediction to u8 (8-bit content only). Dispatched.
pub fn pred_allangs_u8(
    dst: &mut [u8],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_allangs_u8)(
        dst, unfiltered, filtered, center, log2_size, c_idx, bit_depth,
    )
}

/// Batched all-angular intra prediction (modes 2..=34). Dispatched;
/// byte-identical to [`intra::angs::intra_pred_allangs_scalar`].
pub fn intra_pred_allangs(
    dst: &mut [u16],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_allangs)(
        dst, unfiltered, filtered, center, log2_size, c_idx, bit_depth,
    )
}

/// Exact planar intra prediction to u16 from a prepared border. Dispatched.
#[allow(dead_code)]
pub fn pred_planar_u16(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_planar_u16)(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Exact DC intra prediction to u16 from a prepared border. Dispatched.
#[allow(dead_code)]
pub fn pred_dc_u16(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_dc_u16)(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Exact angular intra prediction to u16 from a prepared border. Dispatched.
#[allow(dead_code)]
pub fn pred_angular_u16(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_angular_u16)(dst, border, center, log2_size, c_idx, mode, bit_depth)
}

/// Backward-compatible alias for the exact u16 planar primitive.
#[allow(dead_code)]
pub fn pred_planar(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    pred_planar_u16(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Backward-compatible alias for the exact u16 DC primitive.
#[allow(dead_code)]
pub fn pred_dc(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    pred_dc_u16(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Backward-compatible alias for the exact u16 angular primitive.
#[allow(dead_code)]
pub fn pred_angular(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    pred_angular_u16(dst, border, center, log2_size, c_idx, mode, bit_depth)
}

/// Exact planar intra prediction to u8 from a prepared border. Dispatched.
pub fn pred_planar_u8(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_planar_u8)(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Exact DC intra prediction to u8 from a prepared border. Dispatched.
pub fn pred_dc_u8(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_dc_u8)(dst, border, center, log2_size, c_idx, bit_depth)
}

/// Exact angular intra prediction to u8 from a prepared border. Dispatched.
pub fn pred_angular_u8(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    (PRIMITIVES.intra.pred_angular_u8)(dst, border, center, log2_size, c_idx, mode, bit_depth)
}

// ─── Sub-struct definitions ────────────────────────────────────────────────

pub struct PixelPrimitives {
    pub satd_u8: fn(&[u8], usize, &[u8], usize, usize) -> u32,
    pub satd_u16: fn(&[u16], usize, &[u16], usize, usize) -> u32,
    pub sa8d_u8: fn(&[u8], usize, &[u8], usize, usize) -> u32,
    pub sa8d_u16: fn(&[u16], usize, &[u16], usize, usize) -> u32,
    pub ssd_u8: fn(&[u8], usize, &[u8], usize, usize) -> u64,
    pub ssd_u16: fn(&[u16], usize, &[u16], usize, usize) -> u64,
    pub sub_residual_u8: fn(&[u8], usize, &[u8], usize, &mut [i16], usize),
    pub add_clip_u8: fn(&[u8], &[i16], &mut [u8], usize),
    pub add_clip_u16: fn(&[u16], &[i16], &mut [u16], usize, u16),
    pub narrow_u16_to_u8: fn(&[u16], &mut [u8], usize),
}

pub struct TransformPrimitives {
    pub fwd_dst4: fn(&[i16], &mut [i16], u8),
    pub fwd_dct4: fn(&[i16], &mut [i16], u8),
    pub fwd_dct8: fn(&[i16], &mut [i16], u8),
    pub fwd_dct16: fn(&[i16], &mut [i16], u8),
    pub fwd_dct32: fn(&[i16], &mut [i16], u8),
}

pub struct QuantPrimitives {
    pub dequantize: fn(&mut [i16], i32, i32),
    pub quantize: fn(&[i16], &mut [i16], i32, i32, i32) -> u32,
}

pub struct ResidualPrimitives {
    pub sub_residual: fn(&[u16], usize, &[u16], usize, &mut [i16], usize),
    pub count_nonzero: fn(&[i16]) -> u32,
    pub abs_sum_i16: fn(&[i16]) -> u64,
    pub last_nonzero: fn(&[i16]) -> Option<usize>,
}

type SaoEoFn = fn(&[u16], usize, &[u16], usize, u32, u32, u32, u32, &mut [i64; 5], &mut [u32; 5]);
type SaoBoFn =
    fn(&[u16], usize, &[u16], usize, u32, u32, u32, u32, u8, &mut [i64; 32], &mut [u32; 32]);

pub struct SaoPrimitives {
    pub stats_e0: SaoEoFn,
    pub stats_e1: SaoEoFn,
    pub stats_e2: SaoEoFn,
    pub stats_e3: SaoEoFn,
    pub stats_bo: SaoBoFn,
}

/// Function-pointer table of the dispatchable hot kernels. Selected once,
/// lazily, by [`select_primitives`] from the CPU features and the
/// `BPG_PRIMITIVES` environment variable.
pub struct Primitives {
    pub pixel: PixelPrimitives,
    pub transform: TransformPrimitives,
    pub quant: QuantPrimitives,
    pub residual: ResidualPrimitives,
    pub intra: intra::IntraPrimitives,
    pub sao: SaoPrimitives,
    /// Human-readable name of the selected backend, for `--debug-stats`.
    pub backend: &'static str,
}

// ─── Dispatch selection ────────────────────────────────────────────────────

/// The active primitive backend, chosen once on first use.
pub static PRIMITIVES: LazyLock<Primitives> = LazyLock::new(select_primitives);

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Scalar,
    Simd,
}

fn select_primitives() -> Primitives {
    let choice = match std::env::var("BPG_PRIMITIVES").ok().as_deref() {
        Some("scalar") => BackendChoice::Scalar,
        Some("simd") | Some("asm") | Some("auto") | None => BackendChoice::Simd,
        Some(other) => {
            eprintln!("warning: unknown BPG_PRIMITIVES={other:?}, using scalar");
            BackendChoice::Scalar
        }
    };

    // Scalar reference table.
    let mut p = Primitives {
        pixel: PixelPrimitives {
            satd_u8: scalar::satd_u8_scalar,
            satd_u16: scalar::satd_u16_scalar,
            sa8d_u8: scalar::sa8d_u8_scalar,
            sa8d_u16: scalar::sa8d_u16_scalar,
            ssd_u8: scalar::ssd_u8_scalar,
            ssd_u16: scalar::ssd_u16_scalar,
            sub_residual_u8: scalar::sub_residual_u8_scalar,
            add_clip_u8: scalar::add_clip_u8_scalar,
            add_clip_u16: scalar::add_clip_u16_scalar,
            narrow_u16_to_u8: scalar::narrow_u16_to_u8_scalar,
        },
        transform: TransformPrimitives {
            fwd_dst4: scalar::fwd_dst4_scalar,
            fwd_dct4: scalar::fwd_dct4_scalar,
            fwd_dct8: scalar::fwd_dct8_scalar,
            fwd_dct16: scalar::fwd_dct16_scalar,
            fwd_dct32: scalar::fwd_dct32_scalar,
        },
        quant: QuantPrimitives {
            dequantize: scalar::dequantize_scalar,
            quantize: scalar::quantize_scalar,
        },
        residual: ResidualPrimitives {
            sub_residual: scalar::sub_residual_scalar,
            count_nonzero: scalar::count_nonzero_scalar,
            abs_sum_i16: scalar::abs_sum_i16_scalar,
            last_nonzero: scalar::last_nonzero_scalar,
        },
        intra: intra::IntraPrimitives {
            pred_allangs: intra::angs::intra_pred_allangs_scalar,
            pred_allangs_u8: intra::angs::intra_pred_allangs_u8_scalar,
            pred_planar_u16: intra::exact::pred_planar_scalar,
            pred_dc_u16: intra::exact::pred_dc_scalar,
            pred_angular_u16: intra::exact::pred_angular_scalar,
            pred_planar_u8: intra::exact::pred_planar_u8_scalar,
            pred_dc_u8: intra::exact::pred_dc_u8_scalar,
            pred_angular_u8: intra::exact::pred_angular_u8_scalar,
        },
        sao: SaoPrimitives {
            stats_e0: scalar::sao_stats_e0_scalar,
            stats_e1: scalar::sao_stats_e1_scalar,
            stats_e2: scalar::sao_stats_e2_scalar,
            stats_e3: scalar::sao_stats_e3_scalar,
            stats_bo: scalar::sao_stats_bo_scalar,
        },
        backend: "scalar",
    };

    if choice == BackendChoice::Simd {
        // Portable wide-SIMD layer: bit-identical to scalar, no ISA-specific
        // intrinsics. Overwrites every kernel it covers.
        p.pixel.satd_u16 = wide::satd_u16;
        p.pixel.ssd_u8 = wide::ssd_u8;
        p.pixel.ssd_u16 = wide::ssd_u16;
        p.pixel.sub_residual_u8 = wide::sub_residual_u8;
        p.pixel.add_clip_u8 = wide::add_clip_u8;
        p.pixel.add_clip_u16 = wide::add_clip_u16;
        p.pixel.narrow_u16_to_u8 = wide::narrow_u16_to_u8;
        p.residual.sub_residual = wide::sub_residual;
        p.residual.count_nonzero = wide::count_nonzero;
        p.residual.abs_sum_i16 = wide::abs_sum_i16;
        p.residual.last_nonzero = wide::last_nonzero;
        p.quant.dequantize = wide::dequantize;
        p.quant.quantize = wide::quantize;
        p.sao.stats_e0 = wide::sao_stats_e0;
        p.sao.stats_e1 = wide::sao_stats_e1;
        p.sao.stats_e2 = wide::sao_stats_e2;
        p.sao.stats_e3 = wide::sao_stats_e3;
        p.sao.stats_bo = wide::sao_stats_bo;
        p.transform.fwd_dst4 = wide::fwd_dst4;
        p.transform.fwd_dct4 = wide::fwd_dct4;
        p.transform.fwd_dct8 = wide::fwd_dct8;
        p.transform.fwd_dct16 = wide::fwd_dct16;
        p.transform.fwd_dct32 = wide::fwd_dct32;
        p.intra.pred_allangs = wide::intra_pred_allangs;
        p.intra.pred_allangs_u8 = wide::pred_allangs_u8;
        p.backend = "wide";

        // ISA-specific layers overwrite individual entries as detected.
        #[cfg(target_arch = "x86_64")]
        x86::setup(&mut p);
    }

    p
}

// ─── Tests ─────────────────────────────────────────────────────────────────

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

    fn fill_pattern_i16(buf: &mut [i16], seed: &mut u32) {
        for p in buf.iter_mut() {
            let mut x = *seed;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *seed = x;
            // Map to range [-256, 255] (typical residual range).
            *p = ((x >> 16) as i16).wrapping_mul(2);
        }
    }

    fn fill_pattern(buf: &mut [u8], kind: u32, seed: &mut u32) {
        for (i, p) in buf.iter_mut().enumerate() {
            *p = match kind {
                0 => 0,
                1 => 255,
                2 => (i as u8).wrapping_mul(17),
                3 => {
                    if i % 2 == 0 {
                        0
                    } else {
                        255
                    }
                }
                _ => {
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
    fn ssd_u8_matches_widened_u16_reference() {
        let mut seed = 0x243f_6a88u32;
        for &size in &[4usize, 8, 16, 32] {
            let stride = size + 5;
            let mut a = vec![0u8; stride * size];
            let mut b = vec![0u8; stride * size];
            fill_pattern(&mut a, 99, &mut seed);
            fill_pattern(&mut b, 99, &mut seed);
            let aw: Vec<u16> = a.iter().map(|&v| u16::from(v)).collect();
            let bw: Vec<u16> = b.iter().map(|&v| u16::from(v)).collect();
            assert_eq!(
                ssd_u8_scalar(&a, stride, &b, stride, size),
                ssd_u16_scalar(&aw, stride, &bw, stride, size),
                "size={size}"
            );
        }
    }

    #[test]
    fn dispatched_satd_u8_matches_scalar() {
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
                    let got = (PRIMITIVES.pixel.satd_u8)(&a, stride, &b, stride, size);
                    assert_eq!(got, want, "size={size} pattern a={ka} b={kb}");
                }
            }
        }
    }

    #[test]
    fn sa8d_u8_matches_x265_shape_sanity() {
        let zero = vec![0u8; 32 * 32];
        let full = vec![7u8; 32 * 32];
        assert_eq!(sa8d_u8_scalar(&zero, 4, &full, 4, 4), 56);
        assert_eq!(sa8d_u8_scalar(&zero, 8, &full, 8, 8), 112);
        assert_eq!(sa8d_u8_scalar(&zero, 16, &full, 16, 16), 448);
        assert_eq!(sa8d_u8_scalar(&zero, 32, &full, 32, 32), 1792);
    }

    fn transpose_square_u8(src: &[u8], stride: usize, size: usize) -> Vec<u8> {
        let mut dst = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                dst[x * size + y] = src[y * stride + x];
            }
        }
        dst
    }

    #[test]
    fn sa8d_u8_is_transpose_invariant_for_x265_allangs_layout() {
        let mut seed = 0x3141_5926u32;
        for &size in &[4usize, 8, 16, 32] {
            let stride = size + 7;
            let mut a = vec![0u8; stride * size];
            let mut b = vec![0u8; stride * size];
            for ka in 0..5 {
                for kb in 0..5 {
                    fill_pattern(&mut a, ka, &mut seed);
                    fill_pattern(&mut b, kb, &mut seed);
                    let at = transpose_square_u8(&a, stride, size);
                    let bt = transpose_square_u8(&b, stride, size);
                    assert_eq!(
                        sa8d_u8_scalar(&a, stride, &b, stride, size),
                        sa8d_u8_scalar(&at, size, &bt, size, size),
                        "size={size} pattern a={ka} b={kb}"
                    );
                }
            }
        }
    }

    #[test]
    fn dispatched_sa8d_u8_matches_scalar() {
        let mut seed = 0x2468_ace0u32;
        for &size in &[4usize, 8, 16, 32] {
            let stride = size + 5;
            let mut a = vec![0u8; stride * size];
            let mut b = vec![0u8; stride * size];
            for ka in 0..5 {
                for kb in 0..5 {
                    fill_pattern(&mut a, ka, &mut seed);
                    fill_pattern(&mut b, kb, &mut seed);
                    let want = sa8d_u8_scalar(&a, stride, &b, stride, size);
                    let got = (PRIMITIVES.pixel.sa8d_u8)(&a, stride, &b, stride, size);
                    assert_eq!(got, want, "size={size} pattern a={ka} b={kb}");
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_sa8d_u8_matches_scalar() {
        let mut seed = 0x51a8_d000u32;
        for _ in 0..50 {
            for &size in &[4usize, 8, 16, 32] {
                let stride = size + (seed as usize & 7);
                let mut a = vec![0u8; stride * size];
                let mut b = vec![0u8; stride * size];
                fill_pattern(&mut a, 99, &mut seed);
                fill_pattern(&mut b, 99, &mut seed);
                let want = sa8d_u8_scalar(&a, stride, &b, stride, size);
                let got = x86::avx2::sa8d_u8_avx2_dispatch(&a, stride, &b, stride, size);
                assert_eq!(got, want, "size={size} stride={stride}");
            }
        }
        // Extreme contrast (all-255 vs all-0) maximizes Hadamard coefficient
        // magnitude — guards the i32 accumulation against i16 overflow.
        for &size in &[8usize, 16, 32] {
            let stride = size;
            let zero = vec![0u8; stride * size];
            let full = vec![255u8; stride * size];
            assert_eq!(
                x86::avx2::sa8d_u8_avx2_dispatch(&zero, stride, &full, stride, size),
                sa8d_u8_scalar(&zero, stride, &full, stride, size),
                "extreme size={size}",
            );
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
                let got = x86::sse2::satd_u8_sse2(&a, stride, &b, stride, size);
                assert_eq!(got, want, "size={size} stride={stride}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_satd_u8_stub_matches_sse2() {
        let mut seed = 0xdeadbeef_u32;
        for &size in &[4usize, 8, 16, 32] {
            let stride = size + 2;
            let mut a = vec![0u8; stride * size];
            let mut b = vec![0u8; stride * size];
            fill_pattern(&mut a, 99, &mut seed);
            fill_pattern(&mut b, 99, &mut seed);
            let want = x86::sse2::satd_u8_sse2(&a, stride, &b, stride, size);
            let got = x86::avx2::satd_u8_avx2_dispatch(&a, stride, &b, stride, size);
            assert_eq!(got, want, "size={size}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_quantize_matches_scalar() {
        let mut seed = 0xabcd1234_u32;
        for _ in 0..100 {
            for n in [16, 32, 64, 128, 1024, 1025] {
                let mut coeffs = vec![0i16; n];
                fill_pattern_i16(&mut coeffs, &mut seed);
                let mut want_out = vec![0i16; n];
                let mut got_out = vec![0i16; n];
                let scale = 13762;
                let add = 1 << 14;
                let qbits = 15;
                let want_nnz = quantize_scalar(&coeffs, &mut want_out, scale, add, qbits);
                let got_nnz =
                    x86::avx2::quantize_avx2_dispatch(&coeffs, &mut got_out, scale, add, qbits);
                assert_eq!(got_out, want_out, "n={n}");
                assert_eq!(got_nnz, want_nnz, "nnz n={n}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_ssd_u8_matches_scalar() {
        let mut seed = 0x5555aaaa_u32;
        for _ in 0..100 {
            for &size in &[4usize, 8, 16, 32] {
                let stride = size + (seed as usize & 3);
                let mut a = vec![0u8; stride * size];
                let mut b = vec![0u8; stride * size];
                fill_pattern(&mut a, 99, &mut seed);
                fill_pattern(&mut b, 99, &mut seed);
                let want = ssd_u8_scalar(&a, stride, &b, stride, size);
                let got = x86::avx2::ssd_u8_avx2_dispatch(&a, stride, &b, stride, size);
                assert_eq!(got, want, "size={size} stride={stride}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_sub_residual_u8_matches_scalar() {
        let mut seed = 0x13571357_u32;
        for _ in 0..100 {
            for &size in &[4usize, 8, 16, 32] {
                let stride = size + (seed as usize & 3);
                let mut src = vec![0u8; stride * size];
                let mut pred = vec![0u8; stride * size];
                fill_pattern(&mut src, 99, &mut seed);
                fill_pattern(&mut pred, 99, &mut seed);
                let mut want = vec![0i16; size * size];
                let mut got = vec![0i16; size * size];
                sub_residual_u8_scalar(&src, stride, &pred, stride, &mut want, size);
                x86::avx2::sub_residual_u8_avx2_dispatch(
                    &src, stride, &pred, stride, &mut got, size,
                );
                assert_eq!(got, want, "size={size} stride={stride}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_add_clip_u8_matches_scalar() {
        use super::scalar::add_clip_u8_scalar;
        let mut seed = 0x0badf00d_u32;
        for _ in 0..100 {
            // Lengths cross the 16-wide boundary so both the vector body and the
            // scalar tail are exercised.
            for n in [1usize, 7, 16, 17, 31, 64, 256, 1024, 1025] {
                let mut pred = vec![0u8; n];
                fill_pattern(&mut pred, 255, &mut seed);
                // Full-range i16 residual so the saturating add + packus clamp is
                // covered on both the +overflow and -overflow sides.
                let mut residual = vec![0i16; n];
                fill_pattern_i16(&mut residual, &mut seed);
                let mut want = vec![0u8; n];
                let mut got = vec![0u8; n];
                add_clip_u8_scalar(&pred, &residual, &mut want, n);
                x86::avx2::add_clip_u8_avx2_dispatch(&pred, &residual, &mut got, n);
                assert_eq!(got, want, "n={n}");
            }
        }
    }
}
