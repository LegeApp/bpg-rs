//! psy-rd energy kernel and weight (x265 `psyCost_pp` / `calcPsyRdCost` port).
//!
//! Plain RDO prefers blurring — zeroing AC coefficients barely costs SSE — so
//! psy-rd charges each candidate for the AC energy its reconstruction loses
//! relative to the source. The AC energy of a block is measured tile-wise
//! against a zero reference: `sa8d(tile, 0)` captures AC+DC and
//! `sad(tile, 0) >> 2` (the sample sum, scaled to sa8d's normalization)
//! removes the DC part (x265 pixel.cpp:736 `psyCost_pp`).

use crate::primitives::{sa8d_u8, sa8d_u16, satd_u8, satd_u16};

use super::price::x265_sad_lambda;

/// Zero reference tile (8x8, stride 8) shared by every energy measurement.
const ZERO_U8: [u8; 64] = [0; 64];
const ZERO_U16: [u16; 64] = [0; 64];

/// AC energy of one 8x8 tile: `sa8d(tile, 0) - (sad(tile, 0) >> 2)`, where
/// `sad(tile, 0)` is just the sample sum.
#[inline]
fn tile_energy_u8(tile: &[u8], stride: usize) -> i64 {
    let sa8d = i64::from(sa8d_u8(tile, stride, &ZERO_U8, 8, 8));
    let mut sad = 0u64;
    for row in tile.chunks(stride).take(8) {
        sad += row[..8].iter().map(|&s| u64::from(s)).sum::<u64>();
    }
    sa8d - (sad >> 2) as i64
}

#[inline]
fn tile_energy_u16(tile: &[u16], stride: usize) -> i64 {
    let sa8d = i64::from(sa8d_u16(tile, stride, &ZERO_U16, 8, 8));
    let mut sad = 0u64;
    for row in tile.chunks(stride).take(8) {
        sad += row[..8].iter().map(|&s| u64::from(s)).sum::<u64>();
    }
    sa8d - (sad >> 2) as i64
}

/// AC energy of a 4x4 block (`satd_4x4` instead of sa8d — 4x4 is too small
/// for the 8x8 Hadamard).
#[inline]
fn energy_4x4_u8(block: &[u8], stride: usize) -> i64 {
    let satd = i64::from(satd_u8(block, stride, &ZERO_U8, 8, 4));
    let mut sad = 0u64;
    for row in block.chunks(stride).take(4) {
        sad += row[..4].iter().map(|&s| u64::from(s)).sum::<u64>();
    }
    satd - (sad >> 2) as i64
}

#[inline]
fn energy_4x4_u16(block: &[u16], stride: usize) -> i64 {
    let satd = i64::from(satd_u16(block, stride, &ZERO_U16, 8, 4));
    let mut sad = 0u64;
    for row in block.chunks(stride).take(4) {
        sad += row[..4].iter().map(|&s| u64::from(s)).sum::<u64>();
    }
    satd - (sad >> 2) as i64
}

/// x265 `psyCost_pp`: total AC-energy difference between a source block and
/// its reconstruction, compared per 8x8 tile (`|e_src - e_rec|` summed) with a
/// single 4x4 SATD tile for 4x4 blocks. `size` must be 4/8/16/32.
pub(super) fn psy_energy_diff_u8(
    src: &[u8],
    s_stride: usize,
    rec: &[u8],
    r_stride: usize,
    size: usize,
) -> u64 {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    if size == 4 {
        return energy_4x4_u8(src, s_stride).abs_diff(energy_4x4_u8(rec, r_stride));
    }
    let mut energy = 0u64;
    for y in (0..size).step_by(8) {
        for x in (0..size).step_by(8) {
            let e_src = tile_energy_u8(&src[y * s_stride + x..], s_stride);
            let e_rec = tile_energy_u8(&rec[y * r_stride + x..], r_stride);
            energy += e_src.abs_diff(e_rec);
        }
    }
    energy
}

/// 10/12-bit variant of [`psy_energy_diff_u8`].
pub(super) fn psy_energy_diff_u16(
    src: &[u16],
    s_stride: usize,
    rec: &[u16],
    r_stride: usize,
    size: usize,
) -> u64 {
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    if size == 4 {
        return energy_4x4_u16(src, s_stride).abs_diff(energy_4x4_u16(rec, r_stride));
    }
    let mut energy = 0u64;
    for y in (0..size).step_by(8) {
        for x in (0..size).step_by(8) {
            let e_src = tile_energy_u16(&src[y * s_stride + x..], s_stride);
            let e_rec = tile_energy_u16(&rec[y * r_stride + x..], r_stride);
            energy += e_src.abs_diff(e_rec);
        }
    }
    energy
}

/// SSE-domain psy-rd weight: multiply by the [`psy_energy_diff_u8`] energy to
/// get the additive RD term. x265 fixed point translated to f64 (rdcost.h:47
/// `setPsyRdScale` × the I-slice `psyScaleFix8[2] = 96` factor and the
/// high-QP taper from `setQP`, then `calcPsyRdCost`'s
/// `(m_lambda * m_psyRd * psycost) >> 24` with the sqrt-domain `m_lambda` =
/// [`x265_sad_lambda`]).
pub(super) fn psy_rd_weight(psy_rd: f32, qp: i32, bit_depth: u8) -> f64 {
    let mut factor = f64::from(psy_rd) * 0.33 * (96.0 / 256.0);
    // At high QP psy-rd causes artifacts; x265 tapers it to 0 by QP 51.
    if qp >= 40 {
        factor *= if qp >= 51 {
            0.0
        } else {
            (51 - qp) as f64 * 23.0 / 256.0
        };
    }
    x265_sad_lambda(qp, bit_depth) * factor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat blocks carry no AC energy, so flat->flat pairs score 0. The 8x8
    /// sa8d tiles remove DC exactly (two flats of *different* levels still
    /// cancel); the 4x4 SATD tile keeps a DC remainder (SATD DC gain 8 vs
    /// `sad >> 2` gain 4, matching x265), so 4x4 needs equal-level flats.
    #[test]
    fn flat_blocks_have_zero_energy() {
        for size in [8usize, 16, 32] {
            let src = vec![128u8; size * size];
            let rec = vec![37u8; size * size];
            assert_eq!(psy_energy_diff_u8(&src, size, &rec, size, size), 0);
            let src16 = vec![512u16; size * size];
            let rec16 = vec![100u16; size * size];
            assert_eq!(psy_energy_diff_u16(&src16, size, &rec16, size, size), 0);
        }
        let flat4 = vec![128u8; 16];
        assert_eq!(psy_energy_diff_u8(&flat4, 4, &flat4, 4, 4), 0);
        let flat4_16 = vec![512u16; 16];
        assert_eq!(psy_energy_diff_u16(&flat4_16, 4, &flat4_16, 4, 4), 0);
    }

    /// Identical source and recon score 0 at every size, textured or not.
    #[test]
    fn identical_blocks_have_zero_energy() {
        for size in [4usize, 8, 16, 32] {
            let src: Vec<u8> = (0..size * size).map(|i| (i * 7 % 251) as u8).collect();
            assert_eq!(psy_energy_diff_u8(&src, size, &src, size, size), 0);
        }
    }

    /// A blurred (flat) recon of a sharp source loses AC energy; the kernel
    /// must charge for it, and equally for the mirrored (sharpened) case.
    #[test]
    fn blurred_recon_of_sharp_source_scores_positive() {
        for size in [4usize, 8, 16, 32] {
            // Checkerboard source: maximal AC, mean 125.
            let src: Vec<u8> = (0..size * size)
                .map(|i| {
                    let (x, y) = (i % size, i / size);
                    if (x ^ y) & 1 == 0 { 50 } else { 200 }
                })
                .collect();
            let flat = vec![125u8; size * size];
            let e = psy_energy_diff_u8(&src, size, &flat, size, size);
            assert!(e > 0, "size {size}: blur scored 0");
            // Energy difference is symmetric per tile.
            assert_eq!(e, psy_energy_diff_u8(&flat, size, &src, size, size));
        }
    }

    /// u8 and u16 kernels agree on identical 8-bit content.
    #[test]
    fn u8_and_u16_kernels_match_on_8bit_content() {
        for size in [4usize, 8, 16, 32] {
            let src: Vec<u8> = (0..size * size).map(|i| (i * 13 % 256) as u8).collect();
            let rec: Vec<u8> = (0..size * size).map(|i| (i * 5 % 256) as u8).collect();
            let src16: Vec<u16> = src.iter().map(|&v| u16::from(v)).collect();
            let rec16: Vec<u16> = rec.iter().map(|&v| u16::from(v)).collect();
            assert_eq!(
                psy_energy_diff_u8(&src, size, &rec, size, size),
                psy_energy_diff_u16(&src16, size, &rec16, size, size),
            );
        }
    }

    /// Tiles are compared independently (abs per 8x8 tile, then summed): a
    /// 16x16 block whose recon swaps energy between tiles must not cancel.
    #[test]
    fn tile_wise_abs_does_not_cancel_across_tiles() {
        let size = 16usize;
        let flat = vec![128u8; size * size];
        // Source: texture only in the top-left tile.
        let mut src = flat.clone();
        for y in 0..8 {
            for x in 0..8 {
                src[y * size + x] = if (x ^ y) & 1 == 0 { 50 } else { 200 };
            }
        }
        // Recon: the same texture moved to the bottom-right tile.
        let mut rec = flat.clone();
        for y in 8..16 {
            for x in 8..16 {
                rec[y * size + x] = if (x ^ y) & 1 == 0 { 50 } else { 200 };
            }
        }
        let e_src = psy_energy_diff_u8(&src, size, &flat, size, size);
        let e_rec = psy_energy_diff_u8(&rec, size, &flat, size, size);
        // Both blocks carry the same total AC energy, yet the swapped pair
        // scores double (each tile differs), not zero.
        assert_eq!(
            psy_energy_diff_u8(&src, size, &rec, size, size),
            e_src + e_rec
        );
    }

    /// Weight formula: linear in psy_rd below QP 40, tapered above, 0 at 51+.
    #[test]
    fn psy_rd_weight_matches_x265_shape() {
        let w = psy_rd_weight(2.0, 29, 8);
        let expected = x265_sad_lambda(29, 8) * 2.0 * 0.33 * (96.0 / 256.0);
        assert!((w - expected).abs() < 1e-12);
        // Linear in psy_rd.
        assert!((psy_rd_weight(4.0, 29, 8) - 2.0 * w).abs() < 1e-12);
        // High-QP taper: QP 45 keeps (51-45)*23/256 of the base factor.
        let w45 = psy_rd_weight(2.0, 45, 8);
        let base45 = x265_sad_lambda(45, 8) * 2.0 * 0.33 * (96.0 / 256.0);
        assert!((w45 - base45 * (6.0 * 23.0 / 256.0)).abs() < 1e-9);
        // Zero at QP >= 51 and for psy_rd == 0.
        assert_eq!(psy_rd_weight(2.0, 51, 8), 0.0);
        assert_eq!(psy_rd_weight(0.0, 29, 8), 0.0);
    }
}
