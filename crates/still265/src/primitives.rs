//! Rust-local primitive kernels used by the still encoder.
//!
//! These are intentionally scalar reference implementations. They keep the
//! production Rust encoder independent from the x265 C++ scalar shim while the
//! optimized assembly/SIMD split is still being designed.

use bpg_hevc_decode::hevc::transform as dec_transform;
use std::sync::LazyLock;

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

fn forward_dct(residual: &[i16], log2_size: u8, bit_depth: u8) -> Vec<i16> {
    let n = 1usize << log2_size;
    assert_eq!(residual.len(), n * n);

    let shift1 = log2_size as i32 - 1 + bit_depth as i32 - 8;
    let shift2 = log2_size as i32 + 6;
    let mut tmp = vec![0i16; n * n];
    let mut out = vec![0i16; n * n];
    let matrix = dct_matrix(n);

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

    out
}

fn forward_dst4(residual: &[i16], bit_depth: u8) -> Vec<i16> {
    assert_eq!(residual.len(), 16);

    let shift1 = 1 + bit_depth as i32 - 8;
    let shift2 = 8;
    let mut tmp = [0i16; 16];
    let mut out = [0i16; 16];

    for row in 0..4 {
        forward_dst4_1d(&residual[row * 4..row * 4 + 4], &mut tmp[row..], 4, shift1);
    }

    for row in 0..4 {
        forward_dst4_1d(&tmp[row * 4..row * 4 + 4], &mut out[row..], 4, shift2);
    }

    out.to_vec()
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

pub fn satd_u8(a: &[u8], stride_a: usize, b: &[u8], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
}

pub fn satd_u16(a: &[u16], stride_a: usize, b: &[u16], stride_b: usize, size: usize) -> u32 {
    satd_by(a, stride_a, b, stride_b, size, |v| v as i32)
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
}
