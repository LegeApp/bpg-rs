//! Image types, RGB->YCbCr color conversion, chroma subsampling and CTU
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

/// Output color space. Only the `YCbCr*` family is implemented in M1; `Rgb`
/// and `YCgCo` are accepted by the type but `ColorConvertState::new` will
/// panic if selected.
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
    /// Convert an 8-bit RGB image to a 4:4:4 YCbCr `Image`, optionally
    /// scaling up to a higher `bit_depth` (10/12-bit) during color
    /// conversion.
    pub fn from_rgb8(rgb: &image::RgbImage, color_space: ColorSpace, limited_range: bool, bit_depth: u8) -> Self {
        let cvt = ColorConvertState::new(8, bit_depth as u32, color_space, limited_range);
        Self::from_rgb_pixels(
            rgb.width(),
            rgb.height(),
            rgb.pixels().map(|px| (px[0], px[1], px[2])),
            &cvt,
            color_space,
            limited_range,
            bit_depth,
        )
    }

    /// Convert a 16-bit RGB image (e.g. a 16-bit PNG) to a 4:4:4 YCbCr
    /// `Image`, converting to the requested `bit_depth` (10/12-bit).
    pub fn from_rgb16(
        rgb: &image::ImageBuffer<image::Rgb<u16>, Vec<u16>>,
        color_space: ColorSpace,
        limited_range: bool,
        bit_depth: u8,
    ) -> Self {
        let cvt = ColorConvertState::new(16, bit_depth as u32, color_space, limited_range);
        Self::from_rgb_pixels(
            rgb.width(),
            rgb.height(),
            rgb.pixels().map(|px| (px[0], px[1], px[2])),
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
            let (y, cb, cr) = cvt.rgb_to_ycc(r, g, b);
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

    /// Subsample the chroma planes from 4:4:4 to 4:2:0, matching
    /// `image_ycc444_to_ycc420(img, h_phase)`. Only `h_phase == 1` ("chroma
    /// half way between luma samples") is implemented, which is the value
    /// `bpgenc` uses when 4:2:0 is the preferred chroma format.
    pub fn subsample_to_420(&mut self, h_phase: u8) {
        assert_eq!(self.chroma_format, ChromaFormat::Yuv444, "subsample_to_420 requires a 4:4:4 image");
        assert_eq!(h_phase, 1, "only h_phase == 1 is implemented");

        self.planes[1] = chroma::decimate_to_420(&self.planes[1], self.bit_depth as u32);
        self.planes[2] = chroma::decimate_to_420(&self.planes[2], self.bit_depth as u32);
        self.chroma_format = ChromaFormat::Yuv420;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rgb8_produces_444_planes() {
        let mut rgb = image::RgbImage::new(2, 2);
        rgb.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        rgb.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        rgb.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        rgb.put_pixel(1, 1, image::Rgb([255, 255, 255]));

        let img = Image::from_rgb8(&rgb, ColorSpace::YCbCr, false, 8);
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
        let mut rgb = image::RgbImage::new(3, 3);
        for p in rgb.pixels_mut() {
            *p = image::Rgb([100, 100, 100]);
        }
        let mut img = Image::from_rgb8(&rgb, ColorSpace::YCbCr, false, 8);
        img.subsample_to_420(1);
        assert_eq!((img.planes[1].width, img.planes[1].height), (2, 2));

        img.pad_to_cb_size(8);
        assert_eq!((img.width, img.height), (8, 8));
        assert_eq!((img.planes[0].width, img.planes[0].height), (8, 8));
        // chroma is half-resolution: (8+1)>>1 padded dims = 8>>1 = 4
        assert_eq!((img.planes[1].width, img.planes[1].height), (4, 4));
        assert_eq!((img.planes[2].width, img.planes[2].height), (4, 4));
    }
}
