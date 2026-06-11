//! 4:4:4 -> 4:2:0 chroma decimation, ported from `decimate2_hv` /
//! `decimate2_h16` / `decimate2_v` in `libbpg-0.9.8/bpgenc.c`, restricted to
//! `h_phase == 1` ("chroma half way between luma samples"), which is the
//! phase `bpgenc` uses when the preferred chroma format is 4:2:0
//! (`c_h_phase = (preferred_chroma_format == BPG_FORMAT_420)`).

use crate::Plane;

// phase = 0.5 filter taps (DP1*)
const DP1C0: i64 = 57;
const DP1C1: i64 = 17;
const DP1C2: i64 = -8;
const DP1C3: i64 = -4;
const DP1C4: i64 = 2;

fn clamp_pix(a: i64, pixel_max: i64) -> i64 {
    a.clamp(0, pixel_max)
}

fn clamp_idx(idx: i64, len: usize) -> usize {
    idx.clamp(0, len as i64 - 1) as usize
}

/// Horizontally decimate one row of width `w` to width `w2 = (w+1)/2` using
/// the phase-0.5 filter, with edge samples replicated past the row bounds.
/// Output values are intermediate (unsaturated) `int16`-range values, per
/// `decimate2p1_simple16`.
fn decimate_row_h(row: &[u16], w: usize, bit_depth: u32) -> Vec<i32> {
    let shift = bit_depth as i64 - 7;
    let rnd = 1i64 << (shift - 1);
    let w2 = (w + 1) / 2;
    let mut out = Vec::with_capacity(w2);
    for i in 0..w2 {
        let center = 2 * i as i64;
        let s = |off: i64| row[clamp_idx(center + off, w)] as i64;
        let v = (s(-4) + s(5)) * DP1C4
            + (s(-3) + s(4)) * DP1C3
            + (s(-2) + s(3)) * DP1C2
            + (s(-1) + s(2)) * DP1C1
            + (s(0) + s(1)) * DP1C0
            + rnd;
        out.push((v >> shift) as i32);
    }
    out
}

/// Decimate a 4:4:4 chroma plane to 4:2:0 (half resolution in both
/// dimensions), matching `image_ycc444_to_ycc420` with `h_phase == 1`.
pub fn decimate_to_420(src: &Plane<u16>, bit_depth: u32) -> Plane<u16> {
    let w = src.width as usize;
    let h = src.height as usize;
    let w2 = (w + 1) / 2;
    let h2 = (h + 1) / 2;

    // Horizontally decimate every row first.
    let hrows: Vec<Vec<i32>> = (0..h)
        .map(|y| {
            let row = &src.data[y * src.stride..y * src.stride + w];
            decimate_row_h(row, w, bit_depth)
        })
        .collect();

    let v_shift = 21 - bit_depth as i64;
    let v_offset = 1i64 << (v_shift - 1);
    let pixel_max = (1i64 << bit_depth) - 1;

    let mut data = vec![0u16; w2 * h2];
    for y in (0..h).step_by(2) {
        let y2 = y / 2;
        let rows: [&[i32]; 10] = std::array::from_fn(|k| {
            let idx = clamp_idx(y as i64 - 4 + k as i64, h);
            hrows[idx].as_slice()
        });
        let dst_row = &mut data[y2 * w2..y2 * w2 + w2];
        for x in 0..w2 {
            let v = (rows[0][x] as i64 + rows[9][x] as i64) * DP1C4
                + (rows[1][x] as i64 + rows[8][x] as i64) * DP1C3
                + (rows[2][x] as i64 + rows[7][x] as i64) * DP1C2
                + (rows[3][x] as i64 + rows[6][x] as i64) * DP1C1
                + (rows[4][x] as i64 + rows[5][x] as i64) * DP1C0
                + v_offset;
            dst_row[x] = clamp_pix(v >> v_shift, pixel_max) as u16;
        }
    }

    Plane {
        data,
        width: w2 as u32,
        height: h2 as u32,
        stride: w2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_color_is_unchanged() {
        // A flat plane should decimate to the same flat value everywhere.
        let w = 16;
        let h = 16;
        let src = Plane {
            data: vec![200u16; w * h],
            width: w as u32,
            height: h as u32,
            stride: w,
        };
        let dst = decimate_to_420(&src, 8);
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
        assert!(dst.data.iter().all(|&v| v == 200));
    }

    #[test]
    fn odd_dimensions_round_up() {
        let w = 5;
        let h = 3;
        let src = Plane {
            data: vec![100u16; w * h],
            width: w as u32,
            height: h as u32,
            stride: w,
        };
        let dst = decimate_to_420(&src, 8);
        assert_eq!(dst.width, 3); // (5+1)/2
        assert_eq!(dst.height, 2); // (3+1)/2
    }

    #[test]
    fn solid_color_is_unchanged_10bit() {
        let w = 16;
        let h = 16;
        let src = Plane {
            data: vec![800u16; w * h],
            width: w as u32,
            height: h as u32,
            stride: w,
        };
        let dst = decimate_to_420(&src, 10);
        assert_eq!(dst.width, 8);
        assert_eq!(dst.height, 8);
        assert!(dst.data.iter().all(|&v| v == 800));
    }
}
