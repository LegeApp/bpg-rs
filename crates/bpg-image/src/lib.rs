//! Image types, RGB->BPG color conversion, chroma subsampling and CTU
//! padding for the bpg-rs encoder pipeline. Ported from the `Image`
//! handling in `libbpg-0.9.8/bpgenc.c`.

pub mod chroma;
pub mod convert;
pub mod pad;

pub use convert::ColorConvertState;

/// A single image plane: `height` rows of `width` samples, each row
/// occupying `stride` elements (`stride >= width`).
#[derive(Debug, Clone)]
pub struct Plane<T> {
    pub data: Vec<T>,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
}

/// Chroma subsampling format. Only `Yuv444` and `Yuv420` are exercised by
/// the M1 pipeline; `Gray`/`Yuv422` are included so the type doesn't need to
/// change when those are implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    Gray,
    Yuv420,
    Yuv422,
    Yuv444,
}

/// Output color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    YCbCr,
    Rgb,
    YCgCo,
    YCbCrBt709,
    YCbCrBt2020,
}

/// A decoded/converted image ready for encoding: `width`/`height` are the
/// *luma* (pre-padding) dimensions; chroma plane dimensions follow from
/// `chroma_format`. Samples are stored as `u16` regardless of `bit_depth`,
/// matching `bpgenc.c`'s internal `typedef uint16_t PIXEL` representation;
/// values are bounded by `(1 << bit_depth) - 1`.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma_format: ChromaFormat,
    pub color_space: ColorSpace,
    pub limited_range: bool,
    pub planes: Vec<Plane<u16>>,
    /// TODO(extension): alpha plane support.
    pub has_alpha: bool,
}

impl Image {
    /// Build a monochrome image from 8-bit luma samples (`width * height` bytes,
    /// row-major), optionally scaling up to a higher output `bit_depth`.
    pub fn from_luma8(
        pixels: &[u8],
        width: u32,
        height: u32,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        assert!(
            matches!(
                color_space,
                ColorSpace::YCbCr | ColorSpace::YCbCrBt709 | ColorSpace::YCbCrBt2020
            ),
            "gray BPG images use a YCbCr-family color space"
        );
        Self::from_luma_pixels(
            width,
            height,
            pixels.iter().map(|&v| v as u16),
            8,
            color_space,
            limited_range,
            bit_depth,
        )
    }

    /// Build a monochrome image from 16-bit luma samples (`width * height` u16
    /// values, row-major, native endian), converting to the requested output
    /// `bit_depth`.
    pub fn from_luma16(
        pixels: &[u16],
        width: u32,
        height: u32,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        assert!(
            matches!(
                color_space,
                ColorSpace::YCbCr | ColorSpace::YCbCrBt709 | ColorSpace::YCbCrBt2020
            ),
            "gray BPG images use a YCbCr-family color space"
        );
        Self::from_luma_pixels(
            width,
            height,
            pixels.iter().copied(),
            16,
            color_space,
            limited_range,
            bit_depth,
        )
    }

    fn from_luma_pixels(
        width: u32,
        height: u32,
        pixels: impl Iterator<Item = u16>,
        in_bit_depth: u8,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        let in_max = (1u32 << in_bit_depth) - 1;
        let out_max = (1u32 << bit_depth) - 1;
        let y_data = pixels
            .map(|v| ((v as u32 * out_max + in_max / 2) / in_max) as u16)
            .collect();

        Image {
            width,
            height,
            bit_depth,
            chroma_format: ChromaFormat::Gray,
            color_space,
            limited_range,
            planes: vec![Plane {
                data: y_data,
                width,
                height,
                stride: width as usize,
            }],
            has_alpha: false,
        }
    }

    /// Convert an 8-bit RGB image (`width * height * 3` bytes, row-major, R G B
    /// per pixel) to a 4:4:4 YCbCr `Image`, optionally scaling up to a higher
    /// `bit_depth` during color conversion.
    pub fn from_rgb8(
        pixels: &[u8],
        width: u32,
        height: u32,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        let cvt = ColorConvertState::new(8, bit_depth as u32, color_space, limited_range);
        Self::from_rgb_pixels(
            width,
            height,
            pixels.chunks_exact(3).map(|p| (p[0], p[1], p[2])),
            &cvt,
            color_space,
            limited_range,
            bit_depth,
        )
    }

    /// Convert a 16-bit RGB image (`width * height * 3` u16 values, row-major,
    /// native endian) to a 4:4:4 YCbCr `Image`, converting to the requested
    /// output `bit_depth`.
    pub fn from_rgb16(
        pixels: &[u16],
        width: u32,
        height: u32,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        let cvt = ColorConvertState::new(16, bit_depth as u32, color_space, limited_range);
        Self::from_rgb_pixels(
            width,
            height,
            pixels.chunks_exact(3).map(|p| (p[0], p[1], p[2])),
            &cvt,
            color_space,
            limited_range,
            bit_depth,
        )
    }

    fn from_rgb_pixels<P: Into<i64> + Copy>(
        width: u32,
        height: u32,
        pixels: impl Iterator<Item = (P, P, P)>,
        cvt: &ColorConvertState,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        let mut y_data = Vec::with_capacity((width * height) as usize);
        let mut cb_data = Vec::with_capacity((width * height) as usize);
        let mut cr_data = Vec::with_capacity((width * height) as usize);

        for (r, g, b) in pixels {
            let (y, cb, cr) = cvt.rgb_to_planes(r, g, b);
            y_data.push(y);
            cb_data.push(cb);
            cr_data.push(cr);
        }

        let plane = |data: Vec<u16>| Plane {
            data,
            width,
            height,
            stride: width as usize,
        };

        Image {
            width,
            height,
            bit_depth,
            chroma_format: ChromaFormat::Yuv444,
            color_space,
            limited_range,
            planes: vec![plane(y_data), plane(cb_data), plane(cr_data)],
            has_alpha: false,
        }
    }

    /// Build an `Image` directly from pre-separated 8-bit YCbCr planes.
    ///
    /// This bypasses RGB→YCbCr conversion entirely — use this when input is
    /// already decoded YCbCr (e.g. from HEIC/HEVC, H.264, or any YCbCr source).
    ///
    /// Plane layouts by `chroma`:
    /// - `Gray`: only `y` is used; `cb`/`cr` must be empty.
    /// - `Yuv420`: `cb`/`cr` are `ceil(width/2) × ceil(height/2)`.
    /// - `Yuv422`: `cb`/`cr` are `ceil(width/2) × height`.
    /// - `Yuv444`: all planes are `width × height`.
    ///
    /// Samples are upscaled from 8-bit to `out_bit_depth` (pass 8 for no scaling).
    pub fn from_ycbcr_planes_u8(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: u32,
        height: u32,
        chroma: ChromaFormat,
        color_space: ColorSpace,
        limited_range: bool,
        out_bit_depth: u8,
    ) -> Self {
        let out_max = (1u32 << out_bit_depth) - 1;
        let scale = move |v: u8| -> u16 {
            if out_bit_depth == 8 {
                v as u16
            } else {
                ((v as u32 * out_max + 127) / 255) as u16
            }
        };
        let make_plane = |data: &[u8], pw: u32, ph: u32| Plane {
            data: data.iter().map(|&v| scale(v)).collect(),
            width: pw,
            height: ph,
            stride: pw as usize,
        };
        let y_plane = make_plane(y, width, height);
        let planes = if chroma == ChromaFormat::Gray {
            vec![y_plane]
        } else {
            let (cw, ch) = chroma_plane_dims(width, height, chroma);
            vec![y_plane, make_plane(cb, cw, ch), make_plane(cr, cw, ch)]
        };
        Image {
            width,
            height,
            bit_depth: out_bit_depth,
            chroma_format: chroma,
            color_space,
            limited_range,
            planes,
            has_alpha: false,
        }
    }

    /// Build an `Image` directly from pre-separated 16-bit YCbCr planes
    /// (`in_bit_depth`-wide samples, native endian).
    ///
    /// See [`from_ycbcr_planes_u8`] for plane layout details.
    /// Samples are rescaled from `in_bit_depth` to `out_bit_depth`.
    pub fn from_ycbcr_planes_u16(
        y: &[u16],
        cb: &[u16],
        cr: &[u16],
        in_bit_depth: u8,
        width: u32,
        height: u32,
        chroma: ChromaFormat,
        color_space: ColorSpace,
        limited_range: bool,
        out_bit_depth: u8,
    ) -> Self {
        let in_max = (1u32 << in_bit_depth) - 1;
        let out_max = (1u32 << out_bit_depth) - 1;
        let scale = move |v: u16| -> u16 {
            if in_bit_depth == out_bit_depth {
                v
            } else {
                ((v as u32 * out_max + in_max / 2) / in_max) as u16
            }
        };
        let make_plane = |data: &[u16], pw: u32, ph: u32| Plane {
            data: data.iter().map(|&v| scale(v)).collect(),
            width: pw,
            height: ph,
            stride: pw as usize,
        };
        let y_plane = make_plane(y, width, height);
        let planes = if chroma == ChromaFormat::Gray {
            vec![y_plane]
        } else {
            let (cw, ch) = chroma_plane_dims(width, height, chroma);
            vec![y_plane, make_plane(cb, cw, ch), make_plane(cr, cw, ch)]
        };
        Image {
            width,
            height,
            bit_depth: out_bit_depth,
            chroma_format: chroma,
            color_space,
            limited_range,
            planes,
            has_alpha: false,
        }
    }

    /// Subsample the chroma planes from 4:4:4 to 4:2:0, matching
    /// `image_ycc444_to_ycc420(img, h_phase)`. Only `h_phase == 1` ("chroma
    /// half way between luma samples") is implemented, which is the value
    /// `bpgenc` uses when 4:2:0 is the preferred chroma format.
    pub fn subsample_to_420(&mut self, h_phase: u8) {
        assert_eq!(
            self.chroma_format,
            ChromaFormat::Yuv444,
            "subsample_to_420 requires a 4:4:4 image"
        );
        assert_eq!(h_phase, 1, "only h_phase == 1 is implemented");

        self.planes[1] = chroma::decimate_to_420(&self.planes[1], self.bit_depth as u32);
        self.planes[2] = chroma::decimate_to_420(&self.planes[2], self.bit_depth as u32);
        self.chroma_format = ChromaFormat::Yuv420;
    }

    /// Subsample the chroma planes from 4:4:4 to 4:2:2, matching
    /// `image_ycc444_to_ycc422(img, h_phase)`. Only `h_phase == 1` ("chroma
    /// half way between luma samples") is implemented, which is the value
    /// `bpgenc` uses when 4:2:2 is the preferred chroma format.
    pub fn subsample_to_422(&mut self, h_phase: u8) {
        assert_eq!(
            self.chroma_format,
            ChromaFormat::Yuv444,
            "subsample_to_422 requires a 4:4:4 image"
        );
        assert_eq!(h_phase, 1, "only h_phase == 1 is implemented");

        self.planes[1] = chroma::decimate_to_422(&self.planes[1], self.bit_depth as u32);
        self.planes[2] = chroma::decimate_to_422(&self.planes[2], self.bit_depth as u32);
        self.chroma_format = ChromaFormat::Yuv422;
    }

    /// Downscale an image so neither luma dimension exceeds `max_edge`.
    ///
    /// A separable, scale-aware Lanczos-3 filter is used for thumbnail
    /// generation. Images already within the bound are cloned unchanged.
    pub fn resize_to_fit(&self, max_edge: u32) -> Self {
        assert!(max_edge > 0, "max_edge must be non-zero");
        let longest = self.width.max(self.height);
        if longest <= max_edge {
            return self.clone();
        }
        let scale = max_edge as f64 / longest as f64;
        let width = ((self.width as f64 * scale).round() as u32).max(1);
        let height = ((self.height as f64 * scale).round() as u32).max(1);
        let mut planes = Vec::with_capacity(self.planes.len());
        for (index, plane) in self.planes.iter().enumerate() {
            let (pw, ph) = if index == 0 {
                (width, height)
            } else {
                chroma_plane_dims(width, height, self.chroma_format)
            };
            planes.push(resize_plane_lanczos3(plane, pw, ph, self.bit_depth));
        }
        Self {
            width,
            height,
            bit_depth: self.bit_depth,
            chroma_format: self.chroma_format,
            color_space: self.color_space,
            limited_range: self.limited_range,
            planes,
            has_alpha: false,
        }
    }

    /// Returns the (h_shift, v_shift) of plane `idx` relative to the luma
    /// plane: chroma plane dimensions are `(w + h_shift) >> h_shift` etc.
    fn plane_shifts(&self, idx: usize) -> (u32, u32) {
        if idx == 0 {
            return (0, 0);
        }
        match self.chroma_format {
            ChromaFormat::Yuv420 => (1, 1),
            ChromaFormat::Yuv422 => (1, 0),
            ChromaFormat::Yuv444 | ChromaFormat::Gray => (0, 0),
        }
    }

    /// Pad the image so its dimensions become a multiple of `cb_size`
    /// (a power of two), replicating right/bottom edge samples. Matches
    /// `image_pad`. `width`/`height` are updated to the padded (luma)
    /// dimensions; callers that need the original dimensions (e.g. for the
    /// BPG container header) must capture them beforehand.
    pub fn pad_to_cb_size(&mut self, cb_size: u32) {
        let (w1, h1) = pad::padded_dims(self.width, self.height, cb_size);
        if (w1, h1) == (self.width, self.height) {
            return;
        }
        for idx in 0..self.planes.len() {
            let (h_shift, v_shift) = self.plane_shifts(idx);
            let pw1 = w1 >> h_shift;
            let ph1 = h1 >> v_shift;
            self.planes[idx] = pad::pad_plane(&self.planes[idx], pw1, ph1);
        }
        self.width = w1;
        self.height = h1;
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

fn lanczos_weight(distance: f64, downscale: f64) -> f64 {
    let x = distance / downscale;
    if x.abs() >= 3.0 {
        0.0
    } else {
        sinc(x) * sinc(x / 3.0)
    }
}

fn resize_plane_lanczos3(
    source: &Plane<u16>,
    width: u32,
    height: u32,
    bit_depth: u8,
) -> Plane<u16> {
    let src_w = source.width as usize;
    let src_h = source.height as usize;
    let dst_w = width as usize;
    let dst_h = height as usize;
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;
    let support_x = 3.0 * sx.max(1.0);
    let support_y = 3.0 * sy.max(1.0);

    let mut horizontal = vec![0.0f64; dst_w * src_h];
    for y in 0..src_h {
        for dx in 0..dst_w {
            let center = (dx as f64 + 0.5) * sx - 0.5;
            let first = (center - support_x).floor() as i64;
            let last = (center + support_x).ceil() as i64;
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for ix in first..=last {
                let clamped = ix.clamp(0, src_w as i64 - 1) as usize;
                let weight = lanczos_weight(center - ix as f64, sx.max(1.0));
                sum += source.data[y * source.stride + clamped] as f64 * weight;
                weight_sum += weight;
            }
            horizontal[y * dst_w + dx] = if weight_sum.abs() > 1e-12 {
                sum / weight_sum
            } else {
                source.data
                    [y * source.stride + center.round().clamp(0.0, src_w as f64 - 1.0) as usize]
                    as f64
            };
        }
    }

    let pixel_max = ((1u32 << bit_depth) - 1) as f64;
    let mut data = vec![0u16; dst_w * dst_h];
    for dy in 0..dst_h {
        let center = (dy as f64 + 0.5) * sy - 0.5;
        let first = (center - support_y).floor() as i64;
        let last = (center + support_y).ceil() as i64;
        for x in 0..dst_w {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for iy in first..=last {
                let clamped = iy.clamp(0, src_h as i64 - 1) as usize;
                let weight = lanczos_weight(center - iy as f64, sy.max(1.0));
                sum += horizontal[clamped * dst_w + x] * weight;
                weight_sum += weight;
            }
            let value = if weight_sum.abs() > 1e-12 {
                sum / weight_sum
            } else {
                horizontal[center.round().clamp(0.0, src_h as f64 - 1.0) as usize * dst_w + x]
            };
            data[dy * dst_w + x] = value.round().clamp(0.0, pixel_max) as u16;
        }
    }
    Plane {
        data,
        width,
        height,
        stride: dst_w,
    }
}

/// Chroma plane dimensions for a given luma size and subsampling format.
fn chroma_plane_dims(width: u32, height: u32, chroma: ChromaFormat) -> (u32, u32) {
    match chroma {
        ChromaFormat::Gray | ChromaFormat::Yuv444 => (width, height),
        ChromaFormat::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        ChromaFormat::Yuv422 => (width.div_ceil(2), height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rgb8_produces_444_planes() {
        // 2×2 image: red, green, blue, white
        let pixels: &[u8] = &[
            255, 0, 0, // (0,0): red
            0, 255, 0, // (1,0): green
            0, 0, 255, // (0,1): blue
            255, 255, 255, // (1,1): white
        ];

        let img = Image::from_rgb8(pixels, 2, 2, ColorSpace::YCbCr, false, 8);
        assert_eq!(img.chroma_format, ChromaFormat::Yuv444);
        assert_eq!(img.planes.len(), 3);
        for p in &img.planes {
            assert_eq!((p.width, p.height), (2, 2));
        }
        // top-left = pure red -> Y=76 (per convert.rs known values)
        assert_eq!(img.planes[0].data[0], 76);
    }

    #[test]
    fn pad_to_cb_size_updates_chroma_dims_for_420() {
        let pixels = vec![100u8; 3 * 3 * 3]; // 3×3 RGB image
        let mut img = Image::from_rgb8(&pixels, 3, 3, ColorSpace::YCbCr, false, 8);
        img.subsample_to_420(1);
        assert_eq!((img.planes[1].width, img.planes[1].height), (2, 2));

        img.pad_to_cb_size(8);
        assert_eq!((img.width, img.height), (8, 8));
        assert_eq!((img.planes[0].width, img.planes[0].height), (8, 8));
        // chroma is half-resolution: (8+1)>>1 padded dims = 8>>1 = 4
        assert_eq!((img.planes[1].width, img.planes[1].height), (4, 4));
        assert_eq!((img.planes[2].width, img.planes[2].height), (4, 4));
    }

    #[test]
    fn subsample_to_422_halves_chroma_width_only() {
        let pixels = vec![100u8; 3 * 3 * 3]; // 3×3 RGB image
        let mut img = Image::from_rgb8(&pixels, 3, 3, ColorSpace::YCbCr, false, 8);
        img.subsample_to_422(1);
        assert_eq!(img.chroma_format, ChromaFormat::Yuv422);
        // (3+1)/2 = 2 wide, height unchanged at 3
        assert_eq!((img.planes[1].width, img.planes[1].height), (2, 3));
        assert_eq!((img.planes[2].width, img.planes[2].height), (2, 3));
    }

    #[test]
    fn pad_to_cb_size_updates_chroma_dims_for_422() {
        let pixels = vec![100u8; 3 * 3 * 3]; // 3×3 RGB image
        let mut img = Image::from_rgb8(&pixels, 3, 3, ColorSpace::YCbCr, false, 8);
        img.subsample_to_422(1);

        img.pad_to_cb_size(8);
        assert_eq!((img.width, img.height), (8, 8));
        assert_eq!((img.planes[0].width, img.planes[0].height), (8, 8));
        // chroma is half-resolution horizontally only: (8>>1, 8)
        assert_eq!((img.planes[1].width, img.planes[1].height), (4, 8));
        assert_eq!((img.planes[2].width, img.planes[2].height), (4, 8));
    }

    #[test]
    fn from_ycbcr_planes_u8_420_roundtrip() {
        // Y=128 plane (gray), Cb/Cr=128 (neutral chroma) — should stay near-neutral
        let w = 4u32;
        let h = 4u32;
        let y = vec![128u8; (w * h) as usize];
        let cb = vec![128u8; (w.div_ceil(2) * h.div_ceil(2)) as usize];
        let cr = vec![128u8; (w.div_ceil(2) * h.div_ceil(2)) as usize];
        let img = Image::from_ycbcr_planes_u8(
            &y,
            &cb,
            &cr,
            w,
            h,
            ChromaFormat::Yuv420,
            ColorSpace::YCbCr,
            false,
            8,
        );
        assert_eq!(img.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(img.planes[0].data[0], 128);
        assert_eq!(img.planes[1].data[0], 128);
        assert_eq!((img.planes[1].width, img.planes[1].height), (2, 2));
    }

    #[test]
    fn from_ycbcr_planes_u16_scales_bit_depth() {
        let w = 2u32;
        let h = 2u32;
        // 10-bit max Y (1023)
        let y = vec![1023u16; (w * h) as usize];
        let cb = vec![512u16; (w * h) as usize];
        let cr = vec![512u16; (w * h) as usize];
        let img = Image::from_ycbcr_planes_u16(
            &y,
            &cb,
            &cr,
            10,
            w,
            h,
            ChromaFormat::Yuv444,
            ColorSpace::YCbCr,
            false,
            8,
        );
        // 1023/1023 * 255 = 255
        assert_eq!(img.planes[0].data[0], 255);
        // 512/1023 * 255 ≈ 127
        assert!((img.planes[1].data[0] as i32 - 127).abs() <= 1);
    }

    #[test]
    fn resize_to_fit_preserves_format_and_chroma_geometry() {
        let pixels = vec![96u8; 17 * 9 * 3];
        let mut image = Image::from_rgb8(&pixels, 17, 9, ColorSpace::YCbCrBt709, true, 10);
        image.subsample_to_420(1);

        let thumbnail = image.resize_to_fit(8);

        assert_eq!((thumbnail.width, thumbnail.height), (8, 4));
        assert_eq!(thumbnail.bit_depth, 10);
        assert_eq!(thumbnail.chroma_format, ChromaFormat::Yuv420);
        assert_eq!(thumbnail.color_space, ColorSpace::YCbCrBt709);
        assert!(thumbnail.limited_range);
        assert_eq!(
            (thumbnail.planes[0].width, thumbnail.planes[0].height),
            (8, 4)
        );
        assert_eq!(
            (thumbnail.planes[1].width, thumbnail.planes[1].height),
            (4, 2)
        );
        assert_eq!(
            (thumbnail.planes[2].width, thumbnail.planes[2].height),
            (4, 2)
        );
    }
}
