//! RGB -> YCbCr color conversion, ported from `convert_init`/`rgb24_to_ycc`
//! in `libbpg-0.9.8/bpgenc.c`.

use crate::ColorSpace;

fn lrint(x: f64) -> i64 {
    // C's lrint() with the default rounding mode (round-to-nearest, ties to
    // even) matches Rust's `f64::round_ties_even`.
    x.round_ties_even() as i64
}

fn clamp_pix(a: i64, pixel_max: i64) -> i64 {
    a.clamp(0, pixel_max)
}

/// Precomputed integer coefficients for 8-bit RGB -> YCbCr conversion.
pub struct ColorConvertState {
    /// rgb_to_ycc[0..3] = Y coefficients (R,G,B); [3..6] = Cb; [6..9] = Cr.
    coeffs: [i64; 9],
    c_shift: u32,
    c_rnd: i64,
    y_offset: i64,
    c_center: i64,
    pixel_max: i64,
}

impl ColorConvertState {
    /// `in_bit_depth`/`out_bit_depth`: 8 for the M1 (8-bit PNG -> 8-bit BPG)
    /// path. `color_space` selects the (k_r, k_b) luma coefficients for the
    /// `YCbCr*` family; `Rgb`/`YCgCo` are not yet ported.
    pub fn new(in_bit_depth: u32, out_bit_depth: u32, color_space: ColorSpace, limited_range: bool) -> Self {
        let (k_r, k_b) = match color_space {
            ColorSpace::YCbCr => (0.299, 0.114),
            ColorSpace::YCbCrBt709 => (0.2126, 0.0722),
            ColorSpace::YCbCrBt2020 => (0.2627, 0.0593),
            ColorSpace::Rgb | ColorSpace::YCgCo => {
                unimplemented!("only YCbCr* color spaces are implemented in bpg-rs M1")
            }
        };

        let c_shift = 31 - out_bit_depth;
        let in_pixel_max = ((1u64 << in_bit_depth) - 1) as f64;
        let out_pixel_max = (1u64 << out_bit_depth) - 1;
        let mult = out_pixel_max as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
        let (mult_y, mult_c) = if limited_range {
            let mult_y = (219u64 << (out_bit_depth - 8)) as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
            let mult_c = (224u64 << (out_bit_depth - 8)) as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
            (mult_y, mult_c)
        } else {
            (mult, mult)
        };

        let rgb_to_ycc_f = [
            k_r,
            1.0 - k_r - k_b,
            k_b,
            -0.5 * k_r / (1.0 - k_b),
            -0.5 * (1.0 - k_r - k_b) / (1.0 - k_b),
            0.5,
            0.5,
            -0.5 * (1.0 - k_r - k_b) / (1.0 - k_r),
            -0.5 * k_b / (1.0 - k_r),
        ];

        let mut coeffs = [0i64; 9];
        for i in 0..3 {
            coeffs[i] = lrint(rgb_to_ycc_f[i] * mult_y);
        }
        for i in 3..9 {
            coeffs[i] = lrint(rgb_to_ycc_f[i] * mult_c);
        }

        let c_one = lrint(mult);
        let c_rnd = 1i64 << (c_shift - 1);
        let (y_offset, _y_one) = if limited_range {
            (c_rnd + (16i64 << (c_shift + out_bit_depth - 8)), lrint(mult_y))
        } else {
            (c_rnd, c_one)
        };

        Self {
            coeffs,
            c_shift,
            c_rnd,
            y_offset,
            c_center: 1i64 << (out_bit_depth - 1),
            pixel_max: out_pixel_max as i64,
        }
    }

    /// Convert one 8-bit RGB triple to (Y, Cb, Cr).
    pub fn rgb_to_ycc(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let (r, g, b) = (r as i64, g as i64, b as i64);
        let [c0, c1, c2, c3, c4, c5, c6, c7, c8] = self.coeffs;
        let shift = self.c_shift;
        let y = clamp_pix((c0 * r + c1 * g + c2 * b + self.y_offset) >> shift, self.pixel_max);
        let cb = clamp_pix(((c3 * r + c4 * g + c5 * b + self.c_rnd) >> shift) + self.c_center, self.pixel_max);
        let cr = clamp_pix(((c6 * r + c7 * g + c8 * b + self.c_rnd) >> shift) + self.c_center, self.pixel_max);
        (y as u8, cb as u8, cr as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_full_range_known_values() {
        let cvt = ColorConvertState::new(8, 8, ColorSpace::YCbCr, false);
        let cases: [((u8, u8, u8), (u8, u8, u8)); 7] = [
            ((0, 0, 0), (0, 128, 128)),
            ((255, 255, 255), (255, 128, 128)),
            ((255, 0, 0), (76, 85, 255)),
            ((0, 255, 0), (150, 44, 21)),
            ((0, 0, 255), (29, 255, 107)),
            ((128, 128, 128), (128, 128, 128)),
            ((255, 255, 0), (226, 1, 149)),
        ];
        for ((r, g, b), expected) in cases {
            assert_eq!(cvt.rgb_to_ycc(r, g, b), expected, "rgb=({r},{g},{b})");
        }
    }

    #[test]
    fn ycbcr_limited_range_known_values() {
        let cvt = ColorConvertState::new(8, 8, ColorSpace::YCbCr, true);
        let cases: [((u8, u8, u8), (u8, u8, u8)); 7] = [
            ((0, 0, 0), (16, 128, 128)),
            ((255, 255, 255), (235, 128, 128)),
            ((255, 0, 0), (81, 90, 240)),
            ((0, 255, 0), (145, 54, 34)),
            ((0, 0, 255), (41, 240, 110)),
            ((128, 128, 128), (126, 128, 128)),
            ((255, 255, 0), (210, 16, 146)),
        ];
        for ((r, g, b), expected) in cases {
            assert_eq!(cvt.rgb_to_ycc(r, g, b), expected, "rgb=({r},{g},{b})");
        }
    }
}
