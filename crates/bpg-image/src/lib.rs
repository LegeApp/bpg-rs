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
/// `chroma_format`.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma_format: ChromaFormat,
    pub color_space: ColorSpace,
    pub limited_range: bool,
    pub planes: Vec<Plane<u8>>,
    /// TODO(extension): alpha plane support.
    pub has_alpha: bool,
}

impl Image {
    /// Convert an 8-bit RGB image to a 4:4:4 YCbCr `Image` (M1: 8-bit only).
    pub fn from_rgb8(rgb: &image::RgbImage, color_space: ColorSpace, limited_range: bool) -> Self {
        let width = rgb.width();
        let height = rgb.height();
        let cvt = ColorConvertState::new(8, 8, color_space, limited_range);

        let mut y_data = Vec::with_capacity((width * height) as usize);
        let mut cb_data = Vec::with_capacity((width * height) as usize);
        let mut cr_data = Vec::with_capacity((width * height) as usize);

        for px in rgb.pixels() {
            let (y, cb, cr) = cvt.rgb_to_ycc(px[0], px[1], px[2]);
            y_data.push(y);
            cb_data.push(cb);
            cr_data.push(cr);
        }

        let plane = |data: Vec<u8>| Plane {
            data,
            width,
            height,
            stride: width as usize,
        };

        Image {
            width,
            height,
            bit_depth: 8,
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

        let img = Image::from_rgb8(&rgb, ColorSpace::YCbCr, false);
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
        let mut img = Image::from_rgb8(&rgb, ColorSpace::YCbCr, false);
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
