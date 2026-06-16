//! RGB -> BPG color-space conversion, ported from `convert_init` and the
//! `rgb*_to_*` helpers in `libbpg-0.9.8/bpgenc.c`.

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
    color_space: ColorSpace,
    c_0_25: i64,
    c_0_5: i64,
    y_one: i64,
    c_shift: u32,
    c_rnd: i64,
    y_offset: i64,
    c_center: i64,
    pixel_max: i64,
}

impl ColorConvertState {
    /// `color_space` selects the destination BPG component transform. RGB uses
    /// BPG's GBR plane order (`Y=G, Cb=B, Cr=R`); YCgCo uses HEVC matrix 8.
    pub fn new(
        in_bit_depth: u32,
        out_bit_depth: u32,
        color_space: ColorSpace,
        limited_range: bool,
    ) -> Self {
        let c_shift = 31 - out_bit_depth;
        let in_pixel_max = ((1u64 << in_bit_depth) - 1) as f64;
        let out_pixel_max = (1u64 << out_bit_depth) - 1;
        let mult = out_pixel_max as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
        let (mult_y, mult_c) = if limited_range {
            let mult_y =
                (219u64 << (out_bit_depth - 8)) as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
            let mult_c =
                (224u64 << (out_bit_depth - 8)) as f64 * (1u64 << c_shift) as f64 / in_pixel_max;
            (mult_y, mult_c)
        } else {
            (mult, mult)
        };

        let mut coeffs = [0i64; 9];
        let mut c_0_25 = 0;
        let mut c_0_5 = 0;
        match color_space {
            ColorSpace::YCbCr | ColorSpace::YCbCrBt709 | ColorSpace::YCbCrBt2020 => {
                let (k_r, k_b) = match color_space {
                    ColorSpace::YCbCr => (0.299, 0.114),
                    ColorSpace::YCbCrBt709 => (0.2126, 0.0722),
                    ColorSpace::YCbCrBt2020 => (0.2627, 0.0593),
                    ColorSpace::Rgb | ColorSpace::YCgCo => unreachable!(),
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
                for i in 0..3 {
                    coeffs[i] = lrint(rgb_to_ycc_f[i] * mult_y);
                }
                for i in 3..9 {
                    coeffs[i] = lrint(rgb_to_ycc_f[i] * mult_c);
                }
            }
            ColorSpace::Rgb => {}
            ColorSpace::YCgCo => {
                c_0_25 = lrint(0.25 * mult_y);
                c_0_5 = lrint(0.5 * mult_y);
            }
        }

        let c_one = lrint(mult);
        let c_rnd = 1i64 << (c_shift - 1);
        let (y_offset, y_one) = if limited_range {
            (
                c_rnd + (16i64 << (c_shift + out_bit_depth - 8)),
                lrint(mult_y),
            )
        } else {
            (c_rnd, c_one)
        };

        Self {
            coeffs,
            color_space,
            c_0_25,
            c_0_5,
            y_one,
            c_shift,
            c_rnd,
            y_offset,
            c_center: 1i64 << (out_bit_depth - 1),
            pixel_max: out_pixel_max as i64,
        }
    }

    /// Convert one RGB triple (`in_bit_depth`-wide samples) to BPG planes
    /// (`out_bit_depth`-wide samples, returned widened to `u16` so callers
    /// don't need to special-case 8 vs 10/12-bit storage).
    pub fn rgb_to_planes<P: Into<i64>>(&self, r: P, g: P, b: P) -> (u16, u16, u16) {
        let (r, g, b) = (r.into(), g.into(), b.into());
        let shift = self.c_shift;
        match self.color_space {
            ColorSpace::YCbCr | ColorSpace::YCbCrBt709 | ColorSpace::YCbCrBt2020 => {
                let [c0, c1, c2, c3, c4, c5, c6, c7, c8] = self.coeffs;
                let y = clamp_pix(
                    (c0 * r + c1 * g + c2 * b + self.y_offset) >> shift,
                    self.pixel_max,
                );
                let cb = clamp_pix(
                    ((c3 * r + c4 * g + c5 * b + self.c_rnd) >> shift) + self.c_center,
                    self.pixel_max,
                );
                let cr = clamp_pix(
                    ((c6 * r + c7 * g + c8 * b + self.c_rnd) >> shift) + self.c_center,
                    self.pixel_max,
                );
                (y as u16, cb as u16, cr as u16)
            }
            ColorSpace::Rgb => {
                let y = clamp_pix((self.y_one * g + self.y_offset) >> shift, self.pixel_max);
                let cb = clamp_pix((self.y_one * b + self.y_offset) >> shift, self.pixel_max);
                let cr = clamp_pix((self.y_one * r + self.y_offset) >> shift, self.pixel_max);
                (y as u16, cb as u16, cr as u16)
            }
            ColorSpace::YCgCo => {
                let t1 = self.c_0_5 * g;
                let t2 = self.c_0_25 * (r + b);
                let y = clamp_pix((t1 + t2 + self.y_offset) >> shift, self.pixel_max);
                let cb = clamp_pix(
                    ((t1 - t2 + self.c_rnd) >> shift) + self.c_center,
                    self.pixel_max,
                );
                let cr = clamp_pix(
                    ((self.c_0_5 * (r - b) + self.c_rnd) >> shift) + self.c_center,
                    self.pixel_max,
                );
                (y as u16, cb as u16, cr as u16)
            }
        }
    }

    pub fn rgb_to_ycc<P: Into<i64>>(&self, r: P, g: P, b: P) -> (u16, u16, u16) {
        self.rgb_to_planes(r, g, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_full_range_known_values() {
        let cvt = ColorConvertState::new(8, 8, ColorSpace::YCbCr, false);
        let cases: [((u8, u8, u8), (u16, u16, u16)); 7] = [
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
        let cases: [((u8, u8, u8), (u16, u16, u16)); 7] = [
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

    #[test]
    fn ycbcr_10bit_output_scales_up() {
        // 8-bit input -> 10-bit output: full-scale white maps to 1023, mid
        // values scale roughly by 4x relative to the 8-bit table above.
        let cvt = ColorConvertState::new(8, 10, ColorSpace::YCbCr, false);
        assert_eq!(cvt.rgb_to_ycc(0u8, 0, 0), (0, 512, 512));
        assert_eq!(cvt.rgb_to_ycc(255u8, 255, 255), (1023, 512, 512));
    }

    #[test]
    fn rgb_color_space_uses_gbr_plane_order() {
        let cvt = ColorConvertState::new(8, 8, ColorSpace::Rgb, false);
        assert_eq!(cvt.rgb_to_planes(10u8, 20, 30), (20, 30, 10));

        let cvt = ColorConvertState::new(8, 8, ColorSpace::Rgb, true);
        assert_eq!(cvt.rgb_to_planes(0u8, 0, 0), (16, 16, 16));
        assert_eq!(cvt.rgb_to_planes(255u8, 255, 255), (235, 235, 235));
    }

    #[test]
    fn ycgco_known_values() {
        let cvt = ColorConvertState::new(8, 8, ColorSpace::YCgCo, false);
        assert_eq!(cvt.rgb_to_planes(0u8, 0, 0), (0, 128, 128));
        assert_eq!(cvt.rgb_to_planes(255u8, 255, 255), (255, 128, 128));
        assert_eq!(cvt.rgb_to_planes(255u8, 0, 0), (64, 64, 255));
        assert_eq!(cvt.rgb_to_planes(0u8, 255, 0), (128, 255, 128));
        assert_eq!(cvt.rgb_to_planes(0u8, 0, 255), (64, 64, 1));
    }
}
