//! Single-mode exact intra-prediction primitive references.
//!
//! These wrappers make exact RDO prediction kernel-shaped: callers build one
//! reference border, select the filtered/unfiltered version for the mode, and
//! then call a mode-specific primitive.

use bpg_hevc_decode::hevc::intra::{
    INTRA_PRED_ANGLE, INV_ANGLE, MAX_INTRA_PRED_BLOCK_SIZE, predict_angular,
    predict_dc_from_border, predict_planar_from_border,
};

#[inline]
fn max_val(bit_depth: u8) -> i32 {
    (1i32 << bit_depth) - 1
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn inv_angle(mode: u8) -> i32 {
    if (11..=25).contains(&mode) {
        INV_ANGLE[(mode - 11) as usize]
    } else {
        0
    }
}

pub fn pred_planar_scalar(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    _c_idx: u8,
    bit_depth: u8,
) {
    let size = 1usize << log2_size;
    predict_planar_from_border(dst, size, log2_size, bit_depth, border, center);
}

pub fn pred_dc_scalar(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    let size = 1usize << log2_size;
    predict_dc_from_border(dst, size, log2_size, c_idx, bit_depth, border, center);
}

pub fn pred_angular_scalar(
    dst: &mut [u16],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    let size = 1usize << log2_size;
    predict_angular(
        dst,
        size,
        0,
        0,
        size as u32,
        c_idx,
        mode,
        max_val(bit_depth),
        border,
        center,
    );
}

pub fn pred_planar_u8_scalar(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    _c_idx: u8,
    bit_depth: u8,
) {
    debug_assert_eq!(bit_depth, 8, "pred_planar_u8 is only for 8-bit content");
    let size = 1usize << log2_size;
    let n_i = size as i32;
    let right = border[center + 1 + size];
    let bottom = border[center - 1 - size];
    debug_assert!(dst.len() >= size * size);

    for py in 0..size {
        let py_i = py as i32;
        let left = border[center - 1 - py];
        let row = &mut dst[py * size..py * size + size];
        for (px, out) in row.iter_mut().enumerate() {
            let px_i = px as i32;
            let top = border[center + 1 + px];
            let pred = ((n_i - 1 - px_i) * left
                + (px_i + 1) * right
                + (n_i - 1 - py_i) * top
                + (py_i + 1) * bottom
                + n_i)
                >> (log2_size + 1);
            *out = clip_u8(pred);
        }
    }
}

pub fn pred_dc_u8_scalar(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    debug_assert_eq!(bit_depth, 8, "pred_dc_u8 is only for 8-bit content");
    let size = 1usize << log2_size;
    debug_assert!(dst.len() >= size * size);

    let mut dc_val = 0i32;
    for i in 0..size {
        dc_val += border[center + 1 + i];
        dc_val += border[center - 1 - i];
    }
    dc_val = (dc_val + size as i32) >> (log2_size + 1);

    if c_idx == 0 && size < 32 {
        dst[0] = clip_u8((border[center - 1] + 2 * dc_val + border[center + 1] + 2) >> 2);

        for px in 1..size {
            let pred = (border[center + 1 + px] + 3 * dc_val + 2) >> 2;
            dst[px] = clip_u8(pred);
        }

        for py in 1..size {
            let row_start = py * size;
            let pred = (border[center - 1 - py] + 3 * dc_val + 2) >> 2;
            dst[row_start] = clip_u8(pred);
            dst[row_start + 1..row_start + size].fill(clip_u8(dc_val));
        }
    } else {
        let dc = clip_u8(dc_val);
        for row in dst[..size * size].chunks_exact_mut(size) {
            row.fill(dc);
        }
    }
}

pub fn pred_angular_u8_scalar(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    debug_assert_eq!(bit_depth, 8, "pred_angular_u8 is only for 8-bit content");
    let size = 1usize << log2_size;
    let n = size as i32;
    let intra_pred_angle = INTRA_PRED_ANGLE[mode as usize] as i32;
    debug_assert!(dst.len() >= size * size);

    let mut ref_arr = [0i32; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
    let ref_center = 2 * MAX_INTRA_PRED_BLOCK_SIZE;

    if mode >= 18 {
        ref_arr[ref_center..ref_center + size + 1]
            .copy_from_slice(&border[center..center + size + 1]);

        if intra_pred_angle < 0 {
            let inv_angle = inv_angle(mode);
            let ext = (n * intra_pred_angle) >> 5;
            if ext < -1 {
                for xx in ext..=-1 {
                    let idx = (xx * inv_angle + 128) >> 8;
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

        for py in 0..n {
            let i_idx = ((py + 1) * intra_pred_angle) >> 5;
            let i_fact = ((py + 1) * intra_pred_angle) & 31;
            let base_idx = (ref_center as i32 + i_idx + 1) as usize;
            let row = &mut dst[py as usize * size..py as usize * size + size];
            if i_fact != 0 {
                let w0 = 32 - i_fact;
                for (px, out) in row.iter_mut().enumerate() {
                    let idx = base_idx + px;
                    *out = clip_u8((w0 * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5);
                }
            } else {
                for (px, out) in row.iter_mut().enumerate() {
                    *out = clip_u8(ref_arr[base_idx + px]);
                }
            }
        }

        if mode == 26 && c_idx == 0 && size < 32 {
            for py in 0..n {
                let pred =
                    border[center + 1] + ((border[center - 1 - py as usize] - border[center]) >> 1);
                dst[py as usize * size] = clip_u8(pred);
            }
        }
    } else {
        for i in 0..=n {
            ref_arr[ref_center + i as usize] = border[center - i as usize];
        }

        if intra_pred_angle < 0 {
            let inv_angle = inv_angle(mode);
            let ext = (n * intra_pred_angle) >> 5;
            if ext < -1 {
                for xx in ext..=-1 {
                    let idx = (xx * inv_angle + 128) >> 8;
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

        for py in 0..n {
            let row_base = (ref_center as i32 + py + 1) as usize;
            let row = &mut dst[py as usize * size..py as usize * size + size];
            for (px, out) in row.iter_mut().enumerate() {
                let i_idx = ((px as i32 + 1) * intra_pred_angle) >> 5;
                let i_fact = ((px as i32 + 1) * intra_pred_angle) & 31;
                let idx = (row_base as i32 + i_idx) as usize;
                let pred = if i_fact != 0 {
                    ((32 - i_fact) * ref_arr[idx] + i_fact * ref_arr[idx + 1] + 16) >> 5
                } else {
                    ref_arr[idx]
                };
                *out = clip_u8(pred);
            }
        }

        if mode == 10 && c_idx == 0 && size < 32 {
            for px in 0..n {
                let pred =
                    border[center - 1] + ((border[center + 1 + px as usize] - border[center]) >> 1);
                dst[px as usize] = clip_u8(pred);
            }
        }
    }
}

/// `wide` i32x8 planar prediction; bit-identical to [`pred_planar_u8_scalar`]
/// (same i32 arithmetic, vectorized 8 pixels per step).
pub fn pred_planar_u8_wide(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    bit_depth: u8,
) {
    let size = 1usize << log2_size;
    if size < 8 {
        return pred_planar_u8_scalar(dst, border, center, log2_size, c_idx, bit_depth);
    }
    debug_assert_eq!(bit_depth, 8, "pred_planar_u8 is only for 8-bit content");
    debug_assert!(dst.len() >= size * size);
    use wide::i32x8;
    let n_i = size as i32;
    let right = i32x8::splat(border[center + 1 + size]);
    let bottom = border[center - 1 - size];
    let round = i32x8::splat(n_i);
    let shift = log2_size as i32 + 1;
    let zero = i32x8::splat(0);
    let maxv = i32x8::splat(255);
    let iota = i32x8::new([0, 1, 2, 3, 4, 5, 6, 7]);
    for py in 0..size {
        let py_i = py as i32;
        let left = i32x8::splat(border[center - 1 - py]);
        let wy_top = i32x8::splat(n_i - 1 - py_i);
        let y_bottom = i32x8::splat((py_i + 1) * bottom);
        let row = &mut dst[py * size..py * size + size];
        for px in (0..size).step_by(8) {
            let px_v = i32x8::splat(px as i32) + iota;
            let wx_left = i32x8::splat(n_i - 1) - px_v;
            let wx_right = px_v + i32x8::splat(1);
            let top = i32x8::new(
                <[i32; 8]>::try_from(&border[center + 1 + px..center + 9 + px]).unwrap(),
            );
            let pred =
                (wx_left * left + wx_right * right + wy_top * top + y_bottom + round) >> shift;
            let arr = pred.max(zero).min(maxv).to_array();
            for k in 0..8 {
                row[px + k] = arr[k] as u8;
            }
        }
    }
}

/// `wide` i32x8 angular prediction for the vertical family (mode >= 18);
/// bit-identical to [`pred_angular_u8_scalar`]. The horizontal family and
/// sub-8 sizes fall back to the scalar kernel.
pub fn pred_angular_u8_wide(
    dst: &mut [u8],
    border: &[i32],
    center: usize,
    log2_size: u8,
    c_idx: u8,
    mode: u8,
    bit_depth: u8,
) {
    let size = 1usize << log2_size;
    if size < 8 || mode < 18 {
        return pred_angular_u8_scalar(dst, border, center, log2_size, c_idx, mode, bit_depth);
    }
    debug_assert_eq!(bit_depth, 8, "pred_angular_u8 is only for 8-bit content");
    debug_assert!(dst.len() >= size * size);
    use wide::i32x8;
    let n = size as i32;
    let intra_pred_angle = INTRA_PRED_ANGLE[mode as usize] as i32;

    let mut ref_arr = [0i32; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
    let ref_center = 2 * MAX_INTRA_PRED_BLOCK_SIZE;
    ref_arr[ref_center..ref_center + size + 1].copy_from_slice(&border[center..center + size + 1]);
    if intra_pred_angle < 0 {
        let inv_angle = inv_angle(mode);
        let ext = (n * intra_pred_angle) >> 5;
        if ext < -1 {
            for xx in ext..=-1 {
                let idx = (xx * inv_angle + 128) >> 8;
                if idx >= 0 && idx <= 2 * n {
                    ref_arr[(ref_center as i32 + xx) as usize] =
                        border[(center as i32 - idx) as usize];
                }
            }
        }
    } else {
        let src_start = center + size + 1;
        let dst_start = ref_center + size + 1;
        ref_arr[dst_start..dst_start + size].copy_from_slice(&border[src_start..src_start + size]);
    }

    let zero = i32x8::splat(0);
    let maxv = i32x8::splat(255);
    for py in 0..n {
        let i_idx = ((py + 1) * intra_pred_angle) >> 5;
        let i_fact = ((py + 1) * intra_pred_angle) & 31;
        let base_idx = (ref_center as i32 + i_idx + 1) as usize;
        let row = &mut dst[py as usize * size..py as usize * size + size];
        if i_fact != 0 {
            let w0 = i32x8::splat(32 - i_fact);
            let w1 = i32x8::splat(i_fact);
            let rnd = i32x8::splat(16);
            for px in (0..size).step_by(8) {
                let a = i32x8::new(
                    <[i32; 8]>::try_from(&ref_arr[base_idx + px..base_idx + px + 8]).unwrap(),
                );
                let b = i32x8::new(
                    <[i32; 8]>::try_from(&ref_arr[base_idx + px + 1..base_idx + px + 9]).unwrap(),
                );
                let v: i32x8 = (w0 * a + w1 * b + rnd) >> 5;
                let arr = v.max(zero).min(maxv).to_array();
                for k in 0..8 {
                    row[px + k] = arr[k] as u8;
                }
            }
        } else {
            for px in (0..size).step_by(8) {
                let a = i32x8::new(
                    <[i32; 8]>::try_from(&ref_arr[base_idx + px..base_idx + px + 8]).unwrap(),
                );
                let arr = a.max(zero).min(maxv).to_array();
                for k in 0..8 {
                    row[px + k] = arr[k] as u8;
                }
            }
        }
    }

    if mode == 26 && c_idx == 0 && size < 32 {
        for py in 0..n {
            let pred =
                border[center + 1] + ((border[center - 1 - py as usize] - border[center]) >> 1);
            dst[py as usize * size] = clip_u8(pred);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{
        pred_angular_u8, pred_angular_u16, pred_dc_u8, pred_dc_u16, pred_planar_u8, pred_planar_u16,
    };
    use bpg_hevc_decode::DecodedFrame;
    use bpg_hevc_decode::hevc::UNINIT_SAMPLE;
    use bpg_hevc_decode::hevc::intra::{
        build_reference_borders, build_reference_borders_with_reader, predict_intra_into,
        predict_intra_into_with_reader, reference_filter_applies,
    };
    use bpg_hevc_decode::hevc::slice::IntraPredMode;

    fn fill_pseudo_random(frame: &mut DecodedFrame, bit_depth: u8) {
        let max = (1u32 << bit_depth) - 1;
        let mut s: u32 = 0x7f4a_7c15;
        for c_idx in 0..=2 {
            let (w, h) = frame.component_dims(c_idx);
            if w == 0 || h == 0 {
                continue;
            }
            let (plane, _) = frame.plane_mut(c_idx);
            for px in plane.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *px = (s % (max + 1)) as u16;
            }
        }
    }

    fn selected_border<'a>(uf: &'a [i32], ft: &'a [i32], size: usize, mode: u8) -> &'a [i32] {
        if reference_filter_applies(size, mode) {
            ft
        } else {
            uf
        }
    }

    fn component_indices(chroma_format: u8) -> &'static [u8] {
        match chroma_format {
            0 => &[0],
            _ => &[0, 1, 2],
        }
    }

    fn positions_for(size: u32) -> [(u32, u32); 4] {
        [(size, size), (0, 0), (0, size), (size, 0)]
    }

    fn predict_slot_u16(
        border: &[i32],
        center: usize,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        bit_depth: u8,
    ) -> Vec<u16> {
        let size = 1usize << log2_size;
        let n = size * size;
        let mut got = vec![0u16; n];
        match IntraPredMode::from_u8(mode).unwrap() {
            IntraPredMode::Planar => {
                pred_planar_u16(&mut got, border, center, log2_size, c_idx, bit_depth)
            }
            IntraPredMode::Dc => pred_dc_u16(&mut got, border, center, log2_size, c_idx, bit_depth),
            _ => pred_angular_u16(&mut got, border, center, log2_size, c_idx, mode, bit_depth),
        }
        got
    }

    fn predict_slot_u8(
        border: &[i32],
        center: usize,
        log2_size: u8,
        c_idx: u8,
        mode: u8,
        bit_depth: u8,
    ) -> Vec<u8> {
        let size = 1usize << log2_size;
        let n = size * size;
        let mut got = vec![0u8; n];
        match IntraPredMode::from_u8(mode).unwrap() {
            IntraPredMode::Planar => {
                pred_planar_u8(&mut got, border, center, log2_size, c_idx, bit_depth)
            }
            IntraPredMode::Dc => pred_dc_u8(&mut got, border, center, log2_size, c_idx, bit_depth),
            _ => pred_angular_u8(&mut got, border, center, log2_size, c_idx, mode, bit_depth),
        }
        got
    }

    #[test]
    fn exact_u16_slots_match_decoder_prediction() {
        for &bit_depth in &[8u8, 10, 12] {
            for &chroma_format in &[1u8, 2, 3] {
                let mut frame = DecodedFrame::with_params(128, 128, bit_depth, chroma_format);
                fill_pseudo_random(&mut frame, bit_depth);
                for &c_idx in component_indices(chroma_format) {
                    for log2_size in 2u8..=5 {
                        let size = 1u32 << log2_size;
                        let n = (size * size) as usize;
                        let size_usize = size as usize;
                        let (cw, ch) = frame.component_dims(c_idx);
                        if cw < size * 2 || ch < size * 2 {
                            continue;
                        }
                        for (bx, by) in positions_for(size) {
                            if bx + size > cw || by + size > ch {
                                continue;
                            }
                            let (uf, ft, center, bd) =
                                build_reference_borders(&frame, bx, by, log2_size, c_idx, true);
                            for mode in 0u8..=34 {
                                let pred_mode = IntraPredMode::from_u8(mode).unwrap();
                                let border = selected_border(&uf, &ft, size_usize, mode);
                                let got =
                                    predict_slot_u16(border, center, log2_size, c_idx, mode, bd);

                                let mut want = vec![0u16; n];
                                predict_intra_into(
                                    &frame, bx, by, log2_size, pred_mode, c_idx, true, &mut want,
                                    size_usize,
                                );
                                assert_eq!(
                                    got, want,
                                    "mode {mode}, size {size}, bd {bit_depth}, cf {chroma_format}, c{c_idx}, pos=({bx},{by})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_u8_slots_match_decoder_prediction() {
        let bit_depth = 8u8;
        for &chroma_format in &[1u8, 2, 3] {
            let mut frame = DecodedFrame::with_params(128, 128, bit_depth, chroma_format);
            fill_pseudo_random(&mut frame, bit_depth);
            for &c_idx in component_indices(chroma_format) {
                for log2_size in 2u8..=5 {
                    let size = 1u32 << log2_size;
                    let n = (size * size) as usize;
                    let size_usize = size as usize;
                    let (cw, ch) = frame.component_dims(c_idx);
                    if cw < size * 2 || ch < size * 2 {
                        continue;
                    }
                    for (bx, by) in positions_for(size) {
                        if bx + size > cw || by + size > ch {
                            continue;
                        }
                        let (uf, ft, center, bd) =
                            build_reference_borders(&frame, bx, by, log2_size, c_idx, true);
                        for mode in 0u8..=34 {
                            let pred_mode = IntraPredMode::from_u8(mode).unwrap();
                            let border = selected_border(&uf, &ft, size_usize, mode);
                            let got = predict_slot_u8(border, center, log2_size, c_idx, mode, bd);

                            let mut want_u16 = vec![0u16; n];
                            predict_intra_into(
                                &frame,
                                bx,
                                by,
                                log2_size,
                                pred_mode,
                                c_idx,
                                true,
                                &mut want_u16,
                                size_usize,
                            );
                            let want: Vec<u8> =
                                want_u16.into_iter().map(|v| v.min(255) as u8).collect();
                            assert_eq!(
                                got, want,
                                "mode {mode}, size {size}, cf {chroma_format}, c{c_idx}, pos=({bx},{by})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_slots_match_decoder_with_tile_boundary_unavailable_refs() {
        let bit_depth = 8u8;
        let chroma_format = 1u8;
        let mut frame = DecodedFrame::with_params(128, 128, bit_depth, chroma_format);
        fill_pseudo_random(&mut frame, bit_depth);

        for &(c_idx, log2_size, bx, by) in &[(0u8, 4u8, 32u32, 32u32), (1, 3, 16, 16)] {
            let size = 1u32 << log2_size;
            let size_usize = size as usize;
            let n = (size * size) as usize;
            let tile_x0 = bx;
            let tile_y0 = by;
            let (uf, ft, center, bd) = build_reference_borders_with_reader(
                &frame,
                bx,
                by,
                log2_size,
                c_idx,
                true,
                |_, rx, ry| {
                    if rx < tile_x0 || ry < tile_y0 {
                        Some(UNINIT_SAMPLE)
                    } else {
                        None
                    }
                },
            );

            for mode in 0u8..=34 {
                let pred_mode = IntraPredMode::from_u8(mode).unwrap();
                let border = selected_border(&uf, &ft, size_usize, mode);
                let got = predict_slot_u16(border, center, log2_size, c_idx, mode, bd);

                let mut want = vec![0u16; n];
                predict_intra_into_with_reader(
                    &frame,
                    bx,
                    by,
                    log2_size,
                    pred_mode,
                    c_idx,
                    true,
                    &mut want,
                    size_usize,
                    |_, rx, ry| {
                        if rx < tile_x0 || ry < tile_y0 {
                            Some(UNINIT_SAMPLE)
                        } else {
                            None
                        }
                    },
                );
                assert_eq!(got, want, "mode {mode}, c{c_idx}, tile-origin refs");
            }
        }
    }
}
