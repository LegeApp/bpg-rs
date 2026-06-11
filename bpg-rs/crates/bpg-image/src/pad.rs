//! CTU-alignment padding, ported from `image_pad` in `libbpg-0.9.8/bpgenc.c`:
//! duplicates the right and bottom edge samples so each plane's dimensions
//! become a multiple of `cb_size`.

use crate::Plane;

fn round_up(v: u32, cb_size: u32) -> u32 {
    (v + cb_size - 1) & !(cb_size - 1)
}

/// Pad a single plane so its width/height become `(w1, h1)`, replicating the
/// right/bottom edge samples. `(w1, h1)` must be >= the plane's current
/// dimensions.
pub fn pad_plane(plane: &Plane<u8>, w1: u32, h1: u32) -> Plane<u8> {
    let c_w = plane.width as usize;
    let c_h = plane.height as usize;
    let c_w1 = w1 as usize;
    let c_h1 = h1 as usize;

    let mut data = vec![0u8; c_w1 * c_h1];

    // Copy existing rows and pad horizontally.
    for y in 0..c_h {
        let src_row = &plane.data[y * plane.stride..y * plane.stride + c_w];
        let dst_row = &mut data[y * c_w1..y * c_w1 + c_w1];
        dst_row[..c_w].copy_from_slice(src_row);
        let edge = src_row[c_w - 1];
        for x in c_w..c_w1 {
            dst_row[x] = edge;
        }
    }

    // Pad vertically by duplicating the (already horizontally-padded) last row.
    let (last_rows, rest) = data.split_at_mut(c_h * c_w1);
    let last_row = &last_rows[(c_h - 1) * c_w1..c_h * c_w1];
    for chunk in rest.chunks_mut(c_w1) {
        chunk.copy_from_slice(last_row);
    }

    Plane {
        data,
        width: w1,
        height: h1,
        stride: c_w1,
    }
}

/// Compute the padded luma dimensions for a given `cb_size` (a power of two).
pub fn padded_dims(w: u32, h: u32, cb_size: u32) -> (u32, u32) {
    if cb_size <= 1 {
        (w, h)
    } else {
        (round_up(w, cb_size), round_up(h, cb_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_with_edge_replication() {
        // 3x2 plane:
        // 1 2 3
        // 4 5 6
        let plane = Plane {
            data: vec![1, 2, 3, 4, 5, 6],
            width: 3,
            height: 2,
            stride: 3,
        };
        let padded = pad_plane(&plane, 4, 4);
        assert_eq!(padded.width, 4);
        assert_eq!(padded.height, 4);
        #[rustfmt::skip]
        let expected = vec![
            1, 2, 3, 3,
            4, 5, 6, 6,
            4, 5, 6, 6,
            4, 5, 6, 6,
        ];
        assert_eq!(padded.data, expected);
    }

    #[test]
    fn padded_dims_rounds_up_to_power_of_two() {
        assert_eq!(padded_dims(3200, 2528, 8), (3200, 2528));
        assert_eq!(padded_dims(3201, 2528, 8), (3208, 2528));
        assert_eq!(padded_dims(10, 10, 16), (16, 16));
        assert_eq!(padded_dims(16, 16, 16), (16, 16));
    }
}
