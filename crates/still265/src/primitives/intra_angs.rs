//! Batched all-angular intra prediction (`intra_pred_allangs`), ported from
//! x265's `all_angs_pred_c` (`x265_4.1/source/common/intrapred.cpp`).
//!
//! The luma rough-mode search scores angular modes 2..=34. Instead of rebuilding
//! the reference border + smoothing per mode (as repeated
//! `predict_intra_into` calls do), the caller builds the unfiltered + smoothed
//! borders **once** (`bpg_hevc_decode::hevc::intra::build_reference_borders`) and
//! this kernel predicts all 33 modes into one batch buffer, selecting the
//! smoothed border per mode via `reference_filter_applies` (x265's
//! `g_intraFilterFlags[mode] & size`).
//!
//! The scalar reference reuses the decoder's `predict_angular` verbatim, so it
//! is bit-identical to per-mode `predict_intra_into`; the SIMD variant is
//! validated byte-for-byte against it.

use bpg_hevc_decode::hevc::intra::{predict_angular, reference_filter_applies};

/// Number of angular modes (2..=34) produced by the batch.
pub const ANGULAR_MODES: usize = 33;

/// Byte offset (in samples) of `mode`'s slot within a batch buffer.
#[inline]
pub fn slot_offset(mode: u8, log2_size: u8) -> usize {
    ((mode - 2) as usize) << (2 * log2_size as usize)
}

/// Scalar reference: predict angular modes 2..=34 into `dst` (33 contiguous
/// `size*size` slots). `unfiltered`/`filtered` are the two reference borders
/// from `build_reference_borders`; `center` is their top-left index.
///
/// Bit-identical to per-mode `predict_intra_into` because it calls the decoder's
/// `predict_angular` into each slot.
pub fn intra_pred_allangs_scalar(
    dst: &mut [u16],
    unfiltered: &[i32],
    filtered: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    let size = 1u32 << log2_size;
    let n = (size * size) as usize;
    let max_val = (1i32 << bit_depth) - 1;
    debug_assert!(dst.len() >= ANGULAR_MODES * n);

    for mode in 2u8..=34 {
        let border: &[i32] = if reference_filter_applies(size as usize, mode) {
            filtered
        } else {
            unfiltered
        };
        let off = slot_offset(mode, log2_size);
        let slot = &mut dst[off..off + n];
        predict_angular(
            slot,
            size as usize,
            0,
            0,
            size,
            c_idx,
            mode,
            max_val,
            border,
            center,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bpg_hevc_decode::hevc::intra::{build_reference_borders, predict_intra_into};
    use bpg_hevc_decode::hevc::slice::IntraPredMode;
    use bpg_hevc_decode::DecodedFrame;

    fn fill_pseudo_random(frame: &mut DecodedFrame, bit_depth: u8) {
        let max = (1u32 << bit_depth) - 1;
        let mut s: u32 = 0x9e3779b9;
        let (plane, _) = frame.plane_mut(0);
        for px in plane.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = (s % (max + 1)) as u16;
        }
    }

    /// Each batch slot must equal the decoder's per-mode `predict_intra_into`,
    /// across sizes / bit depths / all angular modes (incl. the mode-10/26
    /// boundary filters and 32x32 strong-smoothing path).
    #[test]
    fn allangs_scalar_matches_predict_intra_into() {
        for &bit_depth in &[8u8, 10, 12] {
            for log2_size in 2u8..=5 {
                let size = 1u32 << log2_size;
                let dim = 4 * size;
                let mut frame = DecodedFrame::with_params(dim, dim, bit_depth, 1);
                fill_pseudo_random(&mut frame, bit_depth);

                // Block with fully-available left/top/corner neighbours.
                let (bx, by) = (size, size);
                let (uf, ft, center, _) =
                    build_reference_borders(&frame, bx, by, log2_size, 0, true);

                let n = (size * size) as usize;
                let mut batch = vec![0u16; ANGULAR_MODES * n];
                intra_pred_allangs_scalar(&mut batch, &uf, &ft, center, log2_size, 0, bit_depth);

                let mut expected = vec![0u16; n];
                for mode in 2u8..=34 {
                    predict_intra_into(
                        &frame,
                        bx,
                        by,
                        log2_size,
                        IntraPredMode::from_u8(mode).unwrap(),
                        0,
                        true,
                        &mut expected,
                        size as usize,
                    );
                    let off = slot_offset(mode, log2_size);
                    assert_eq!(
                        &batch[off..off + n],
                        &expected[..],
                        "mode {mode}, size {size}, bd {bit_depth}"
                    );
                }
            }
        }
    }
}
