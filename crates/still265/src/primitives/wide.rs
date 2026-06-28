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

#[cfg(feature = "dct-histogram")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::round_shift;
use bpg_hevc_decode::hevc::intra::{
    INTRA_BORDER_CENTER, INTRA_BORDER_LEN, INTRA_PRED_ANGLE, INV_ANGLE, reference_filter_applies,
};
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

/// Inverse quantization in place (H.265 8.6.3, flat list). Bit-identical to
/// [`super::dequantize_scalar`].
///
/// Bounds: `|level| <= 32767` and `combined = LEVEL_SCALE[rem] * (1<<per) <=
/// 72 * 256 = 18432`, so `level * combined <= ~6.04e8` fits `i32`; adding the
/// rounding offset (`< 1<<shift`) stays within `i32`, and the signed lane-wise
/// arithmetic shift + `clamp` match the scalar i32 path exactly. The (unused
/// for `bit_depth >= 8`, `log2 >= 2`) negative-shift case defers to scalar.
/// Forward quantization, bit-identical to [`super::quantize_scalar`].
///
/// `|coeff| <= 32767` and `scale <= 26214`, so `|coeff|*scale <= ~8.6e8`; the
/// rounding offset `add` is `< 1<<qbits` with `qbits <= 27`, so `add <= ~4.5e7`
/// and the sum stays well within `i32` (`< 2.1e9`). The lane-wise arithmetic
/// shift, `min(32767)`, sign reapplication, and non-zero count therefore match
/// the scalar `i64` reference exactly. `qbits >= 9` always (max TU is 32x32,
/// max bit depth 12), so the shift is always positive.
pub fn quantize(coeffs: &[i16], levels: &mut [i16], scale: i32, add: i32, qbits: i32) -> u32 {
    let vscale = i32x8::new([scale; 8]);
    let vadd = i32x8::new([add; 8]);
    let vmax = i32x8::new([32767; 8]);
    let zero = i32x8::new([0; 8]);
    let mut nnz_acc = i32x8::new([0; 8]);

    let mut i = 0;
    while i + 8 <= coeffs.len() {
        let c = i32x8::from(i16x8::from_slice_unaligned(&coeffs[i..]));
        let absc = c.max(zero - c);
        let level = ((absc * vscale + vadd) >> qbits).min(vmax);
        // Re-sign: -level where c<0, level otherwise.
        let neg = c.is_negative();
        let signed = neg.blend(zero - level, level);
        let arr = signed.to_array();
        for (k, &x) in arr.iter().enumerate() {
            levels[i + k] = x as i16;
        }
        // is_positive() is -1 per non-zero (level>=0) lane.
        nnz_acc = nnz_acc + level.is_positive();
        i += 8;
    }
    let mut nnz = (-nnz_acc.reduce_add()) as u32;

    while i < coeffs.len() {
        let c = coeffs[i];
        let level = ((c.unsigned_abs() as i64 * scale as i64 + add as i64) >> qbits) as i32;
        let level = level.min(32767);
        if level != 0 {
            levels[i] = if c < 0 { -level } else { level } as i16;
            nnz += 1;
        }
        i += 1;
    }
    nnz
}

pub fn dequantize(levels: &mut [i16], combined: i32, shift: i32) {
    if shift < 0 {
        super::dequantize_scalar(levels, combined, shift);
        return;
    }
    let vcomb = i32x8::new([combined; 8]);
    let add = if shift > 0 { 1i32 << (shift - 1) } else { 0 };
    let vadd = i32x8::new([add; 8]);
    let lo = i32x8::new([-32768; 8]);
    let hi = i32x8::new([32767; 8]);

    let mut i = 0;
    while i + 8 <= levels.len() {
        let v = i32x8::from(i16x8::from_slice_unaligned(&levels[i..]));
        let scaled = ((v * vcomb + vadd) >> shift).max(lo).min(hi);
        let arr = scaled.to_array();
        for (k, &x) in arr.iter().enumerate() {
            levels[i + k] = x as i16;
        }
        i += 8;
    }
    if i < levels.len() {
        let add0 = if shift > 0 { 1 << (shift - 1) } else { 0 };
        for v in &mut levels[i..] {
            let value = (*v as i32 * combined + add0) >> shift;
            *v = value.clamp(-32768, 32767) as i16;
        }
    }
}

/// SAO horizontal edge-offset (`eo_class 0`) statistics over an interior
/// region. Bit-identical to [`super::sao_stats_e0_scalar`].
///
/// Each pixel's edge category is `2 + sign(rec-left) + sign(rec-right)`; we
/// compute the two signs lane-wise as `cmp_lt - cmp_gt` (masks are `-1` where
/// true) and accumulate per category by masked-blend. Per-row `i32`
/// accumulators stay tiny (`|src-rec| <= 4095`, `<= 64` pixels/row), so the
/// reduction is exact; counts are `-sum(mask)`.
///
/// Bounds/safety: the caller guarantees `x0 >= 1`, `x0+w <= rec_w-1` (right
/// neighbour valid) and the region lies inside the unclamped source, so no
/// edge handling is needed here.
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
    let zero = i32x8::new([0; 8]);
    // (edgeIdx, edge value = sign0+sign1) for the four coded categories.
    let cats: [(usize, i32); 4] = [(0, -2), (1, -1), (3, 1), (4, 2)];
    let cat_vec: [i32x8; 4] = [
        i32x8::new([-2; 8]),
        i32x8::new([-1; 8]),
        i32x8::new([1; 8]),
        i32x8::new([2; 8]),
    ];
    let x0 = x0 as usize;
    let w = w as usize;

    for y in y0..y0 + h {
        let rrow = y as usize * rec_stride;
        let srow = y as usize * src_stride;
        let rec_row = as_i16(&rec[rrow..rrow + x0 + w + 1]);
        let src_row = as_i16(&src[srow..srow + x0 + w]);

        let mut sum_acc = [zero; 4];
        let mut cnt_acc = [zero; 4];
        let mut xi = 0usize;
        while xi + 8 <= w {
            let x = x0 + xi;
            let r = i32x8::from(i16x8::from_slice_unaligned(&rec_row[x..]));
            let l = i32x8::from(i16x8::from_slice_unaligned(&rec_row[x - 1..]));
            let rt = i32x8::from(i16x8::from_slice_unaligned(&rec_row[x + 1..]));
            let s = i32x8::from(i16x8::from_slice_unaligned(&src_row[x..]));

            let d0 = r - l;
            let d1 = r - rt;
            // sign(d) via inherent masks (-1 where true): is_negative = d<0,
            // is_positive = d>0. sign = is_negative - is_positive gives
            // d<0 -> -1, d>0 -> +1, d==0 -> 0.
            let edge =
                (d0.is_negative() - d0.is_positive()) + (d1.is_negative() - d1.is_positive());
            let diff = s - r;

            for k in 0..4 {
                // edge == ev  <=>  (edge-ev) is neither negative nor positive.
                let dd = edge - cat_vec[k];
                let mask = !(dd.is_negative() | dd.is_positive());
                sum_acc[k] = sum_acc[k] + mask.blend(diff, zero);
                cnt_acc[k] = cnt_acc[k] + mask; // adds -1 per matching lane
            }
            xi += 8;
        }
        for (k, &(cat, _)) in cats.iter().enumerate() {
            sum[cat] += sum_acc[k].reduce_add() as i64;
            count[cat] += (-cnt_acc[k].reduce_add()) as u32;
        }

        while xi < w {
            let x = x0 + xi;
            let r = rec_row[x] as i32;
            let left = rec_row[x - 1] as i32;
            let right = rec_row[x + 1] as i32;
            let edge = (2 + (r - left).signum() + (r - right).signum()) as usize;
            if edge != 2 {
                sum[edge] += src_row[x] as i64 - r as i64;
                count[edge] += 1;
            }
            xi += 1;
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

#[inline]
fn inv_angle(mode: u8) -> i32 {
    if (11..=25).contains(&mode) {
        INV_ANGLE[(mode - 11) as usize]
    } else {
        0
    }
}

/// SIMD batched all-angular intra prediction (`intra_pred_allangs`).
///
/// Vertical modes (>=18) vectorize the per-row reference convolution.
/// Horizontal modes (<18) vectorize the transposed per-column convolution.
/// Byte-identical to `intra_angs::intra_pred_allangs_scalar` (enforced by
/// tests).
pub fn intra_pred_allangs(
    dst: &mut [u16],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    allangs_into::<u16>(dst, unfiltered, filtered, center, log2_size, c_idx, bit_depth);
}

/// Output sample type for the angular predictors. Values are always clamped to
/// `[0, max_val]` (and thus to `[0, 255]` for 8-bit) before storing, so the
/// narrowing cast is lossless — `u8` output is byte-identical to computing `u16`
/// and then narrowing with [`narrow_u16_to_u8`].
pub trait PredSample: Copy {
    /// Store a value already clamped into the sample's range.
    fn from_clamped(v: i32) -> Self;
}
impl PredSample for u16 {
    #[inline(always)]
    fn from_clamped(v: i32) -> u16 {
        v as u16
    }
}
impl PredSample for u8 {
    #[inline(always)]
    fn from_clamped(v: i32) -> u8 {
        v as u8
    }
}

/// Generic core of [`intra_pred_allangs`] / [`pred_allangs_u8`]. Writes all 33
/// angular modes (2..=34) into `dst` with the sample type `T`, avoiding any
/// intermediate buffer for the `u8` path.
fn allangs_into<T: PredSample>(
    dst: &mut [T],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    let n = 1usize << log2_size;
    let area = n * n;
    let max_val = (1i32 << bit_depth) - 1;
    for mode in 2u8..=34 {
        let border: &[i32] = if reference_filter_applies(n, mode) {
            filtered
        } else {
            unfiltered
        };
        let off = ((mode - 2) as usize) << (2 * log2_size as usize);
        let slot = &mut dst[off..off + area];
        if mode >= 18 {
            predict_vertical_simd(slot, n, mode, max_val, border, center);
        } else {
            let _ = c_idx;
            predict_horizontal_simd(slot, n, mode, max_val, border, center);
        }
    }
}

/// Vertical-mode (>=18) angular prediction into a `n`-stride slot, mirroring the
/// `mode >= 18` branch of the decoder's `predict_angular` with the inner row
/// convolution vectorized.
fn predict_vertical_simd<T: PredSample>(
    slot: &mut [T],
    n: usize,
    mode: u8,
    max_val: i32,
    border: &[i32],
    center: usize,
) {
    let angle = INTRA_PRED_ANGLE[mode as usize] as i32;
    let rc = INTRA_BORDER_CENTER;
    let ni = n as i32;
    let mut ref_arr = [0i32; INTRA_BORDER_LEN];

    // Build the (top-based) reference exactly as predict_angular does.
    ref_arr[rc..rc + n + 1].copy_from_slice(&border[center..center + n + 1]);
    if angle < 0 {
        let inv = inv_angle(mode);
        let ext = (ni * angle) >> 5;
        if ext < -1 {
            for xx in ext..=-1 {
                let idx = (xx * inv + 128) >> 8;
                if idx >= 0 && idx <= 2 * ni {
                    ref_arr[(rc as i32 + xx) as usize] = border[(center as i32 - idx) as usize];
                }
            }
        }
    } else {
        let src = center + n + 1;
        let dst0 = rc + n + 1;
        ref_arr[dst0..dst0 + n].copy_from_slice(&border[src..src + n]);
    }

    let zero = i32x8::new([0; 8]);
    let max_v = i32x8::new([max_val; 8]);
    let bias = i32x8::new([16; 8]);
    for py in 0..n {
        let i_idx = (((py as i32) + 1) * angle) >> 5;
        let i_fact = (((py as i32) + 1) * angle) & 31;
        let base = (rc as i32 + i_idx + 1) as usize;
        let row_off = py * n;
        if i_fact != 0 {
            let w0 = i32x8::new([32 - i_fact; 8]);
            let f = i32x8::new([i_fact; 8]);
            let mut px = 0usize;
            while px + 8 <= n {
                let a =
                    i32x8::new(<[i32; 8]>::try_from(&ref_arr[base + px..base + px + 8]).unwrap());
                let b = i32x8::new(
                    <[i32; 8]>::try_from(&ref_arr[base + px + 1..base + px + 9]).unwrap(),
                );
                let r = ((a * w0 + b * f + bias) >> 5u32).max(zero).min(max_v);
                let arr = r.to_array();
                for k in 0..8 {
                    slot[row_off + px + k] = T::from_clamped(arr[k]);
                }
                px += 8;
            }
            while px < n {
                let v = ((32 - i_fact) * ref_arr[base + px] + i_fact * ref_arr[base + px + 1] + 16)
                    >> 5;
                slot[row_off + px] = T::from_clamped(v.clamp(0, max_val));
                px += 1;
            }
        } else {
            for px in 0..n {
                slot[row_off + px] = T::from_clamped(ref_arr[base + px].clamp(0, max_val));
            }
        }
    }

    // Boundary filter for mode 26 (luma, size < 32) — column 0 of each row.
    if mode == 26 && n < 32 {
        for py in 0..n {
            let pred = border[center + 1] + ((border[center - 1 - py] - border[center]) >> 1);
            slot[py * n] = T::from_clamped(pred.clamp(0, max_val));
        }
    }
}

/// Horizontal-mode (<18) angular prediction into a `n`-stride slot, mirroring
/// the transposed branch of the decoder's `predict_angular`. The natural row
/// loop gathers from the reference array, so this implementation flips the
/// loops: for one output column, rows are contiguous in the reference array and
/// can be processed with `i32x8`, then scattered into the column.
fn predict_horizontal_simd<T: PredSample>(
    slot: &mut [T],
    n: usize,
    mode: u8,
    max_val: i32,
    border: &[i32],
    center: usize,
) {
    let angle = INTRA_PRED_ANGLE[mode as usize] as i32;
    let rc = INTRA_BORDER_CENTER;
    let ni = n as i32;
    let mut ref_arr = [0i32; INTRA_BORDER_LEN];

    for i in 0..=n {
        ref_arr[rc + i] = border[center - i];
    }
    if angle < 0 {
        let inv = inv_angle(mode);
        let ext = (ni * angle) >> 5;
        if ext < -1 {
            for xx in ext..=-1 {
                let idx = (xx * inv + 128) >> 8;
                if idx >= 0 && idx <= 2 * ni {
                    ref_arr[(rc as i32 + xx) as usize] = border[(center as i32 + idx) as usize];
                }
            }
        }
    } else {
        for xx in (ni + 1)..=(2 * ni) {
            ref_arr[rc + xx as usize] = border[center - xx as usize];
        }
    }

    let zero = i32x8::new([0; 8]);
    let max_v = i32x8::new([max_val; 8]);
    let bias = i32x8::new([16; 8]);
    for px in 0..n {
        let i_idx = (((px as i32) + 1) * angle) >> 5;
        let i_fact = (((px as i32) + 1) * angle) & 31;
        let base = (rc as i32 + i_idx + 1) as usize;
        let mut py = 0usize;
        if i_fact != 0 {
            let w0 = i32x8::new([32 - i_fact; 8]);
            let f = i32x8::new([i_fact; 8]);
            while py + 8 <= n {
                let a =
                    i32x8::new(<[i32; 8]>::try_from(&ref_arr[base + py..base + py + 8]).unwrap());
                let b = i32x8::new(
                    <[i32; 8]>::try_from(&ref_arr[base + py + 1..base + py + 9]).unwrap(),
                );
                let r = ((a * w0 + b * f + bias) >> 5u32).max(zero).min(max_v);
                let arr = r.to_array();
                for k in 0..8 {
                    slot[(py + k) * n + px] = T::from_clamped(arr[k]);
                }
                py += 8;
            }
            while py < n {
                let v = ((32 - i_fact) * ref_arr[base + py] + i_fact * ref_arr[base + py + 1] + 16)
                    >> 5;
                slot[py * n + px] = T::from_clamped(v.clamp(0, max_val));
                py += 1;
            }
        } else {
            while py + 8 <= n {
                let r =
                    i32x8::new(<[i32; 8]>::try_from(&ref_arr[base + py..base + py + 8]).unwrap())
                        .max(zero)
                        .min(max_v);
                let arr = r.to_array();
                for k in 0..8 {
                    slot[(py + k) * n + px] = T::from_clamped(arr[k]);
                }
                py += 8;
            }
            while py < n {
                slot[py * n + px] = T::from_clamped(ref_arr[base + py].clamp(0, max_val));
                py += 1;
            }
        }
    }

    // Boundary filter for mode 10 (luma, size < 32) — row 0 of each column.
    if mode == 10 && n < 32 {
        for px in 0..n {
            let pred = border[center - 1] + ((border[center + 1 + px] - border[center]) >> 1);
            slot[px] = T::from_clamped(pred.clamp(0, max_val));
        }
    }
}

// ─── Phase 5: native u8 rough intra prediction ────────────────────────────

/// Batched all-angular prediction to u8 for the 8-bit rough path.
///
/// Writes u8 samples directly via [`allangs_into`] — no intermediate u16 buffer
/// and no separate narrowing pass. Byte-identical to the previous
/// `intra_pred_allangs` + [`narrow_u16_to_u8`] sequence because every sample is
/// clamped to `[0, 255]` before the store (enforced by `allangs_simd_matches_scalar`).
pub fn pred_allangs_u8(
    dst: &mut [u8],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    debug_assert_eq!(bit_depth, 8, "pred_allangs_u8 is only for 8-bit content");
    let n = 1usize << log2_size;
    let total = super::intra::angs::ANGULAR_MODES * n * n;
    debug_assert!(dst.len() >= total);
    allangs_into::<u8>(dst, unfiltered, filtered, center, log2_size, c_idx, bit_depth);
}

// ─── Phase 4: coefficient/residual tools ──────────────────────────────────

/// Count nonzero i16 elements in a flat slice.
/// Uses `abs` + i32 accumulation to avoid per-lane comparisons.
pub fn count_nonzero(levels: &[i16]) -> u32 {
    // Strategy: widen to i32, take abs, clamp to 1 by min(v, 1) — impossible with
    // i32x8::max(1). Instead, simplest accurate approach: use the scalar iterator
    // for small slices, but batch with i16x8 abs+reduce for larger.
    // abs(v) reduces to 0 or positive; we accumulate then compare at the end.
    // However we can't easily "clamp to 1" in wide. Use scalar fallback for now;
    // the wide path here is abs_sum_i16 which is more naturally vectorized.
    levels.iter().filter(|&&v| v != 0).count() as u32
}

/// Absolute sum of i16 values. Wide SIMD via i32 accumulators (avoids i16 overflow).
pub fn abs_sum_i16(levels: &[i16]) -> u64 {
    let zero = i32x8::splat(0);
    let mut acc = zero;
    let mut i = 0;
    while i + 8 <= levels.len() {
        let v = i32x8::from(i16x8::from_slice_unaligned(&levels[i..]));
        // abs via max(v, -v) in i32 domain
        let neg = i32x8::splat(0) - v;
        acc = acc + v.max(neg);
        i += 8;
    }
    let mut sum = acc.reduce_add() as i64 as u64;
    while i < levels.len() {
        sum += levels[i].unsigned_abs() as u64;
        i += 1;
    }
    sum
}

/// Index of the last nonzero element in linear order. Scalar fallback (reverse scan
/// is hard to vectorize profitably for small blocks).
pub fn last_nonzero(levels: &[i16]) -> Option<usize> {
    levels.iter().rposition(|&v| v != 0)
}

// ─── Phase 6: SAO stats completeness ──────────────────────────────────────

// EO1/EO2/EO3 neighbour-access patterns are not row-adjacent, so the wide
// vectorization benefit is marginal. Delegate to scalar for now; AVX2 gathers
// can accelerate these later if profiling shows they're hot.

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
    super::scalar::sao_stats_e1_scalar(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

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
    super::scalar::sao_stats_e2_scalar(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

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
    super::scalar::sao_stats_e3_scalar(rec, rec_stride, src, src_stride, x0, y0, w, h, sum, count)
}

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
    super::scalar::sao_stats_bo_scalar(
        rec, rec_stride, src, src_stride, x0, y0, w, h, band_shift, sum, count,
    )
}

// ─── Phase 3: per-size 2-D forward transforms ──────────────────────────────

fn fwd_dct_nxn_wide(residual: &[i16], out: &mut [i16], n: usize, bit_depth: u8) {
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

#[cfg(feature = "dct-histogram")]
static DCT_COUNT_4: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "dct-histogram")]
static DCT_COUNT_8: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "dct-histogram")]
static DCT_COUNT_16: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "dct-histogram")]
static DCT_COUNT_32: AtomicU64 = AtomicU64::new(0);

/// Returns [count4, count8, count16, count32] accumulated since process start.
///
/// Always returns zeros unless built with the `dct-histogram` feature; callers
/// treat an all-zero result as "instrumentation disabled".
#[cfg(feature = "dct-histogram")]
pub fn dct_size_histogram() -> [u64; 4] {
    [
        DCT_COUNT_4.load(Ordering::Relaxed),
        DCT_COUNT_8.load(Ordering::Relaxed),
        DCT_COUNT_16.load(Ordering::Relaxed),
        DCT_COUNT_32.load(Ordering::Relaxed),
    ]
}

/// See the `dct-histogram`-gated variant above; returns zeros when disabled.
#[cfg(not(feature = "dct-histogram"))]
pub fn dct_size_histogram() -> [u64; 4] {
    [0; 4]
}

// DST-4 is not hot enough to warrant vectorization beyond the scalar path.
pub fn fwd_dst4(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    super::scalar::fwd_dst4_scalar(residual, out, bit_depth);
}

pub fn fwd_dct4(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    #[cfg(feature = "dct-histogram")]
    DCT_COUNT_4.fetch_add(1, Ordering::Relaxed);
    super::scalar::fwd_dct4_butterfly(residual, out, bit_depth);
}

pub fn fwd_dct8(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    #[cfg(feature = "dct-histogram")]
    DCT_COUNT_8.fetch_add(1, Ordering::Relaxed);
    super::scalar::fwd_dct8_butterfly(residual, out, bit_depth);
}

pub fn fwd_dct16(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    #[cfg(feature = "dct-histogram")]
    DCT_COUNT_16.fetch_add(1, Ordering::Relaxed);
    super::scalar::fwd_dct16_butterfly(residual, out, bit_depth);
}

pub fn fwd_dct32(residual: &[i16], out: &mut [i16], bit_depth: u8) {
    #[cfg(feature = "dct-histogram")]
    DCT_COUNT_32.fetch_add(1, Ordering::Relaxed);
    super::scalar::fwd_dct32_butterfly(residual, out, bit_depth);
}

// ─── Phase 2: u8 coverage ─────────────────────────────────────────────────

/// Load 8 `u8` values from `s[i..]` as signed i16 lanes.
#[inline]
fn load_u8x8_as_i16(s: &[u8], i: usize) -> i16x8 {
    i16x8::new([
        s[i] as i16,
        s[i + 1] as i16,
        s[i + 2] as i16,
        s[i + 3] as i16,
        s[i + 4] as i16,
        s[i + 5] as i16,
        s[i + 6] as i16,
        s[i + 7] as i16,
    ])
}

/// 8-bit SSD. Bit-identical to [`super::ssd_u8_scalar`].
///
/// Differences fit i16 (range ±255). Squares fit i32 (max 65025).
/// A 32-wide row accumulates ≤ 32×65025 < 2^21, well within i32.
pub fn ssd_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u64 {
    debug_assert!(stride_a >= size && stride_b >= size);
    let mut sse = 0u64;
    for j in 0..size {
        let ra = &a[j * stride_a..];
        let rb = &b[j * stride_b..];
        let mut acc = i32x8::new([0; 8]);
        let mut i = 0;
        while i + 8 <= size {
            let d = i32x8::from(load_u8x8_as_i16(ra, i) - load_u8x8_as_i16(rb, i));
            acc = acc + d * d;
            i += 8;
        }
        let mut row = acc.reduce_add() as i64;
        while i < size {
            let d = ra[i] as i32 - rb[i] as i32;
            row += (d * d) as i64;
            i += 1;
        }
        sse += row as u64;
    }
    sse
}

/// 8-bit residual subtraction. Bit-identical to [`super::sub_residual_u8_scalar`].
pub fn sub_residual_u8(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    out: &mut [i16],
    size: usize,
) {
    debug_assert!(src_stride >= size && pred_stride >= size && out.len() >= size * size);
    for j in 0..size {
        let s = &src[j * src_stride..];
        let p = &pred[j * pred_stride..];
        let o = &mut out[j * size..j * size + size];
        let mut i = 0;
        while i + 8 <= size {
            let d = load_u8x8_as_i16(s, i) - load_u8x8_as_i16(p, i);
            o[i..i + 8].copy_from_slice(&d.to_array());
            i += 8;
        }
        while i < size {
            o[i] = s[i] as i16 - p[i] as i16;
            i += 1;
        }
    }
}

/// Add residual to 8-bit prediction and clip to [0,255]. Bit-identical to [`super::add_clip_u8_scalar`].
pub fn add_clip_u8(pred: &[u8], residual: &[i16], out: &mut [u8], n: usize) {
    debug_assert!(pred.len() >= n && residual.len() >= n && out.len() >= n);
    let zero = i32x8::new([0; 8]);
    let max_v = i32x8::new([255; 8]);
    let mut i = 0;
    while i + 8 <= n {
        let p = i32x8::from(load_u8x8_as_i16(pred, i));
        let r = i32x8::from(i16x8::from_slice_unaligned(&residual[i..]));
        let v = (p + r).max(zero).min(max_v);
        let arr = v.to_array();
        for k in 0..8 {
            out[i + k] = arr[k] as u8;
        }
        i += 8;
    }
    while i < n {
        out[i] = (pred[i] as i32 + residual[i] as i32).clamp(0, 255) as u8;
        i += 1;
    }
}

/// Add residual to u16 prediction and clip to [0, max]. Bit-identical to [`super::add_clip_u16_scalar`].
pub fn add_clip_u16(pred: &[u16], residual: &[i16], out: &mut [u16], n: usize, max: u16) {
    debug_assert!(pred.len() >= n && residual.len() >= n && out.len() >= n);
    let zero = i32x8::new([0; 8]);
    let max_v = i32x8::new([max as i32; 8]);
    let mut i = 0;
    while i + 8 <= n {
        let p = i32x8::from(i16x8::from_slice_unaligned(as_i16(&pred[i..])));
        let r = i32x8::from(i16x8::from_slice_unaligned(&residual[i..]));
        let v = (p + r).max(zero).min(max_v);
        let arr = v.to_array();
        for k in 0..8 {
            out[i + k] = arr[k] as u16;
        }
        i += 8;
    }
    while i < n {
        out[i] = (pred[i] as i32 + residual[i] as i32).clamp(0, max as i32) as u16;
        i += 1;
    }
}

/// Narrow u16 prediction to u8 (clip to [0,255]). Bit-identical to [`super::narrow_u16_to_u8_scalar`].
pub fn narrow_u16_to_u8(src: &[u16], dst: &mut [u8], n: usize) {
    debug_assert!(src.len() >= n && dst.len() >= n);
    let zero = i32x8::new([0; 8]);
    let max_v = i32x8::new([255; 8]);
    let mut i = 0;
    while i + 8 <= n {
        let v = i32x8::from(i16x8::from_slice_unaligned(as_i16(&src[i..])))
            .max(zero)
            .min(max_v);
        let arr = v.to_array();
        for k in 0..8 {
            dst[i + k] = arr[k] as u8;
        }
        i += 8;
    }
    while i < n {
        dst[i] = src[i].min(255) as u8;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        abs_sum_i16_scalar, add_clip_u8_scalar, add_clip_u16_scalar, count_nonzero_scalar,
        dct_matrix, dequantize_scalar, forward_dct_1d as scalar_dct_1d, fwd_dct4_scalar,
        fwd_dct8_scalar, fwd_dct16_scalar, fwd_dct32_scalar, fwd_dst4_scalar, last_nonzero_scalar,
        narrow_u16_to_u8_scalar, quantize_scalar, sao_stats_e0_scalar, satd_u16_scalar,
        ssd_u8_scalar, ssd_u16_scalar, sub_residual_scalar, sub_residual_u8_scalar,
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
    fn allangs_simd_matches_scalar() {
        use crate::primitives::intra_angs::{ANGULAR_MODES, intra_pred_allangs_scalar};
        use bpg_hevc_decode::hevc::intra::{INTRA_BORDER_CENTER, INTRA_BORDER_LEN};
        let mut st = 0xC0FF_EE11u32;
        for &bd in &[8u8, 10, 12] {
            let max = (1u32 << bd) - 1;
            for log2 in 2u8..=5 {
                let n = 1usize << log2;
                let mut uf = [0i32; INTRA_BORDER_LEN];
                let mut ft = [0i32; INTRA_BORDER_LEN];
                for i in 0..INTRA_BORDER_LEN {
                    uf[i] = (rng(&mut st) % (max + 1)) as i32;
                    ft[i] = (rng(&mut st) % (max + 1)) as i32;
                }
                let area = n * n;
                let mut a = vec![0u16; ANGULAR_MODES * area];
                let mut b = vec![0u16; ANGULAR_MODES * area];
                intra_pred_allangs_scalar(&mut a, &uf, &ft, INTRA_BORDER_CENTER, log2, 0, bd);
                super::intra_pred_allangs(&mut b, &uf, &ft, INTRA_BORDER_CENTER, log2, 0, bd);
                assert_eq!(a, b, "size {n}, bd {bd}");
            }
        }
    }

    #[test]
    fn sao_stats_e0_matches_scalar() {
        let mut st = 0xfeed_5eedu32;
        for &bd in &[8u32, 10, 12] {
            let max = (1u32 << bd) - 1;
            // A plane wider than the region so left/right neighbours are valid.
            let stride = 80usize;
            let height = 70usize;
            let mut rec = vec![0u16; stride * height];
            let mut src = vec![0u16; stride * height];
            for v in rec.iter_mut() {
                *v = (rng(&mut st) & max) as u16;
            }
            for v in src.iter_mut() {
                *v = (rng(&mut st) & max) as u16;
            }
            // Interior regions of assorted widths (incl. non-multiples of 8).
            for &(x0, y0, w, h) in &[
                (1u32, 0u32, 64u32, 64u32),
                (1, 0, 17, 9),
                (3, 5, 23, 31),
                (1, 0, 78, 70),
            ] {
                let mut want_sum = [0i64; 5];
                let mut want_cnt = [0u32; 5];
                let mut got_sum = [0i64; 5];
                let mut got_cnt = [0u32; 5];
                sao_stats_e0_scalar(
                    &rec,
                    stride,
                    &src,
                    stride,
                    x0,
                    y0,
                    w,
                    h,
                    &mut want_sum,
                    &mut want_cnt,
                );
                super::sao_stats_e0(
                    &rec,
                    stride,
                    &src,
                    stride,
                    x0,
                    y0,
                    w,
                    h,
                    &mut got_sum,
                    &mut got_cnt,
                );
                assert_eq!(got_sum, want_sum, "sum bd={bd} region=({x0},{y0},{w},{h})");
                assert_eq!(
                    got_cnt, want_cnt,
                    "count bd={bd} region=({x0},{y0},{w},{h})"
                );
            }
        }
    }

    #[test]
    fn quantize_matches_scalar() {
        // HEVC QUANT_SCALE and qbits ranges spanning all valid
        // (qp, log2_size, bit_depth) combinations, plus a non-multiple-of-8 tail.
        const QUANT_SCALE: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
        let mut st = 0x1234_abcdu32;
        for &bd in &[8i32, 10, 12] {
            for log2 in 2i32..=5 {
                for qp in 0i32..=51 {
                    let per = qp / 6;
                    let rem = (qp % 6) as usize;
                    let transform_shift = 15 - bd - log2;
                    let qbits = 14 + per + transform_shift;
                    let add = (171i64 << (qbits - 9)) as i32;
                    let scale = QUANT_SCALE[rem];
                    for &len in &[16usize, 17, 64, 1024] {
                        let mut coeffs = vec![0i16; len];
                        for v in coeffs.iter_mut() {
                            *v = (rng(&mut st) as i32 % 65536 - 32768) as i16;
                        }
                        let mut want = vec![0i16; len];
                        let mut got = vec![0i16; len];
                        let nnz_w = quantize_scalar(&coeffs, &mut want, scale, add, qbits);
                        let nnz_g = super::quantize(&coeffs, &mut got, scale, add, qbits);
                        assert_eq!(got, want, "levels bd={bd} log2={log2} qp={qp} len={len}");
                        assert_eq!(nnz_g, nnz_w, "nnz bd={bd} log2={log2} qp={qp} len={len}");
                    }
                }
            }
        }
    }

    #[test]
    fn dequantize_matches_scalar() {
        // HEVC LEVEL_SCALE * (1<<per) and shift ranges spanning all valid
        // (qp, log2_size, bit_depth) combinations, plus a non-multiple-of-8 tail.
        const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];
        let mut st = 0x0bad_f00du32;
        for &bd in &[8i32, 10, 12] {
            for log2 in 2i32..=5 {
                for qp in 0i32..=51 {
                    let per = qp / 6;
                    let rem = (qp % 6) as usize;
                    let combined = LEVEL_SCALE[rem] * (1 << per);
                    let shift = bd - 9 + log2;
                    for &len in &[16usize, 17, 64, 1024] {
                        let mut a = vec![0i16; len];
                        for v in a.iter_mut() {
                            *v = (rng(&mut st) as i32 % 65536 - 32768) as i16;
                        }
                        let mut want = a.clone();
                        let mut got = a;
                        dequantize_scalar(&mut want, combined, shift);
                        super::dequantize(&mut got, combined, shift);
                        assert_eq!(got, want, "bd={bd} log2={log2} qp={qp} len={len}");
                    }
                }
            }
        }
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

    #[test]
    fn ssd_u8_matches_scalar() {
        let mut st = 0xc0c0_a0a0_u32;
        for &size in &[4usize, 8, 16, 32] {
            for _ in 0..40 {
                let sa = size + (rng(&mut st) as usize & 7);
                let sb = size + (rng(&mut st) as usize & 7);
                let mut a = vec![0u8; sa * size];
                let mut b = vec![0u8; sb * size];
                for v in a.iter_mut() {
                    *v = (rng(&mut st) & 0xff) as u8;
                }
                for v in b.iter_mut() {
                    *v = (rng(&mut st) & 0xff) as u8;
                }
                let want = ssd_u8_scalar(&a, sa, &b, sb, size);
                let got = super::ssd_u8(&a, sa, &b, sb, size);
                assert_eq!(got, want, "size={size}");
            }
        }
    }

    #[test]
    fn sub_residual_u8_matches_scalar() {
        let mut st = 0xb33f_1234_u32;
        for &size in &[4usize, 8, 16, 32] {
            for _ in 0..40 {
                let ss = size + (rng(&mut st) as usize & 7);
                let ps = size + (rng(&mut st) as usize & 7);
                let mut src = vec![0u8; ss * size];
                let mut pred = vec![0u8; ps * size];
                for v in src.iter_mut() {
                    *v = (rng(&mut st) & 0xff) as u8;
                }
                for v in pred.iter_mut() {
                    *v = (rng(&mut st) & 0xff) as u8;
                }
                let mut want = vec![0i16; size * size];
                let mut got = vec![0i16; size * size];
                sub_residual_u8_scalar(&src, ss, &pred, ps, &mut want, size);
                super::sub_residual_u8(&src, ss, &pred, ps, &mut got, size);
                assert_eq!(got, want, "size={size}");
            }
        }
    }

    #[test]
    fn add_clip_u8_matches_scalar() {
        let mut st = 0x1234_abcd_u32;
        for &n in &[16usize, 64, 256, 1024] {
            for _ in 0..20 {
                let mut pred = vec![0u8; n];
                let mut residual = vec![0i16; n];
                for v in pred.iter_mut() {
                    *v = (rng(&mut st) & 0xff) as u8;
                }
                for v in residual.iter_mut() {
                    *v = (rng(&mut st) as i32 as i16).clamp(-300, 300);
                }
                let mut want = vec![0u8; n];
                let mut got = vec![0u8; n];
                add_clip_u8_scalar(&pred, &residual, &mut want, n);
                super::add_clip_u8(&pred, &residual, &mut got, n);
                assert_eq!(got, want, "n={n}");
            }
        }
    }

    #[test]
    fn add_clip_u16_matches_scalar() {
        let mut st = 0xfeed_cafe_u32;
        for &bd in &[8u16, 10, 12] {
            let max = (1u16 << bd) - 1;
            for &n in &[16usize, 64, 256, 1024] {
                let mut pred = vec![0u16; n];
                let mut residual = vec![0i16; n];
                for v in pred.iter_mut() {
                    *v = (rng(&mut st) & max as u32) as u16;
                }
                for v in residual.iter_mut() {
                    *v = (rng(&mut st) as i32 as i16).clamp(-4096, 4096);
                }
                let mut want = vec![0u16; n];
                let mut got = vec![0u16; n];
                add_clip_u16_scalar(&pred, &residual, &mut want, n, max);
                super::add_clip_u16(&pred, &residual, &mut got, n, max);
                assert_eq!(got, want, "bd={bd} n={n}");
            }
        }
    }

    #[test]
    fn narrow_u16_to_u8_matches_scalar() {
        let mut st = 0x0a0b_0c0d_u32;
        for &n in &[16usize, 64, 256, 1024] {
            let mut src = vec![0u16; n];
            for v in src.iter_mut() {
                // include values > 255 to exercise clipping
                *v = (rng(&mut st) & 0x1ff) as u16;
            }
            let mut want = vec![0u8; n];
            let mut got = vec![0u8; n];
            narrow_u16_to_u8_scalar(&src, &mut want, n);
            super::narrow_u16_to_u8(&src, &mut got, n);
            assert_eq!(got, want, "n={n}");
        }
    }

    #[test]
    fn count_nonzero_matches_scalar() {
        let mut st = 0xcafe_babe_u32;
        for &n in &[8usize, 64, 128, 1024] {
            for _ in 0..20 {
                let levels: Vec<i16> = (0..n)
                    .map(|_| {
                        if rng(&mut st) & 3 == 0 {
                            0
                        } else {
                            rng(&mut st) as i16
                        }
                    })
                    .collect();
                assert_eq!(
                    super::count_nonzero(&levels),
                    count_nonzero_scalar(&levels),
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn abs_sum_i16_matches_scalar() {
        let mut st = 0xabcd_1234_u32;
        for &n in &[8usize, 64, 128, 1024] {
            for _ in 0..20 {
                let levels: Vec<i16> = (0..n).map(|_| rng(&mut st) as i16).collect();
                assert_eq!(
                    super::abs_sum_i16(&levels),
                    abs_sum_i16_scalar(&levels),
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn last_nonzero_matches_scalar() {
        let mut st = 0x1357_2468_u32;
        for &n in &[8usize, 64, 128, 1024] {
            for _ in 0..20 {
                let levels: Vec<i16> = (0..n)
                    .map(|_| {
                        if rng(&mut st) & 7 == 0 {
                            0
                        } else {
                            rng(&mut st) as i16
                        }
                    })
                    .collect();
                assert_eq!(
                    super::last_nonzero(&levels),
                    last_nonzero_scalar(&levels),
                    "n={n}"
                );
            }
        }
        // All-zero case
        assert_eq!(super::last_nonzero(&[0i16; 64]), None);
    }

    #[test]
    fn fwd_dst4_wide_matches_scalar() {
        let mut st = 0xdead_beef_u32;
        for &bd in &[8u8, 10, 12] {
            for _ in 0..40 {
                let res: Vec<i16> = (0..16).map(|_| rng(&mut st) as i16 % 256).collect();
                let mut want = vec![0i16; 16];
                let mut got = vec![0i16; 16];
                fwd_dst4_scalar(&res, &mut want, bd);
                super::fwd_dst4(&res, &mut got, bd);
                assert_eq!(got, want, "bd={bd}");
            }
        }
    }

    #[test]
    fn fwd_dct_wide_matches_scalar() {
        let mut st = 0xfade_1234_u32;
        for &bd in &[8u8, 10, 12] {
            for &(n, func_scalar, func_wide) in &[
                (
                    4usize,
                    fwd_dct4_scalar as fn(&[i16], &mut [i16], u8),
                    super::fwd_dct4 as fn(&[i16], &mut [i16], u8),
                ),
                (8, fwd_dct8_scalar, super::fwd_dct8),
                (16, fwd_dct16_scalar, super::fwd_dct16),
                (32, fwd_dct32_scalar, super::fwd_dct32),
            ] {
                for _ in 0..10 {
                    let res: Vec<i16> = (0..n * n).map(|_| rng(&mut st) as i16 % 256).collect();
                    let mut want = vec![0i16; n * n];
                    let mut got = vec![0i16; n * n];
                    func_scalar(&res, &mut want, bd);
                    func_wide(&res, &mut got, bd);
                    assert_eq!(got, want, "n={n} bd={bd}");
                }
            }
        }
    }
}
