//! Deterministic regression corpus for the x265 oracle.
//!
//! Every image is generated from code with a fixed seed (no randomness that
//! varies run-to-run), so the corpus — and therefore the oracle manifest — is
//! exactly reproducible. The categories cover the content classes the BPG
//! tuning roadmap cares about: flat color, smooth gradients, high-frequency
//! checkerboards, line art, sensor-style noise, and a real natural-photo crop.

use image::{Rgb, RgbImage};

/// The bit depth of a corpus source image.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// 8-bit source -> exercises the 8-bit x265 lib.
    Eight,
    /// 16-bit source -> exercises the linked 10-bit x265 lib.
    Sixteen,
}

/// One corpus entry: a name, a content category, and the pixel data.
pub struct CorpusImage {
    pub name: &'static str,
    pub category: &'static str,
    pub width: u32,
    pub height: u32,
    pub depth: Depth,
    /// 8-bit RGB (present when `depth == Eight`).
    pub rgb8: Option<RgbImage>,
    /// 16-bit RGB as a flat `R,G,B,...` buffer (present when `depth == Sixteen`).
    pub rgb16: Option<Vec<u16>>,
}

/// Edge length of every synthetic corpus image (4x4 CTUs at 64px).
pub const SIZE: u32 = 256;

/// A tiny deterministic xorshift32 PRNG so "noise" is reproducible.
struct XorShift(u32);
impl XorShift {
    fn next_u8(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 24) as u8
    }
}

/// Build the full corpus. `dusk_png` is the repo's natural-photo source; when
/// present a center crop is included, otherwise the photo entries are skipped.
pub fn build(dusk_png: Option<&std::path::Path>) -> Vec<CorpusImage> {
    let mut out = vec![
        flat(),
        gradient8(),
        checkerboard(),
        lineart(),
        noise(),
        gradient16(),
    ];
    if let Some(path) = dusk_png {
        if let Ok(dyn_img) = image::open(path) {
            out.push(photo8(&dyn_img));
            out.push(photo16(&dyn_img));
        }
    }
    out
}

fn flat() -> CorpusImage {
    let img = RgbImage::from_pixel(SIZE, SIZE, Rgb([120, 90, 160]));
    eight("flat", "flat-color", img)
}

fn gradient8() -> CorpusImage {
    let mut img = RgbImage::new(SIZE, SIZE);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let r = (x * 255 / (SIZE - 1)) as u8;
        let g = (y * 255 / (SIZE - 1)) as u8;
        let b = ((x + y) * 255 / (2 * (SIZE - 1))) as u8;
        *p = Rgb([r, g, b]);
    }
    eight("gradient", "gradient", img)
}

fn checkerboard() -> CorpusImage {
    let mut img = RgbImage::new(SIZE, SIZE);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let on = ((x / 8) + (y / 8)) % 2 == 0;
        *p = if on { Rgb([255, 255, 255]) } else { Rgb([0, 0, 0]) };
    }
    eight("checkerboard", "high-frequency", img)
}

fn lineart() -> CorpusImage {
    let mut img = RgbImage::from_pixel(SIZE, SIZE, Rgb([255, 255, 255]));
    let c = (SIZE / 2) as i32;
    let r = (SIZE as i32 / 2) - 8;
    for (x, y, p) in img.enumerate_pixels_mut() {
        let (xi, yi) = (x as i32, y as i32);
        // grid lines every 32px, the two diagonals, and a circle outline
        let on_grid = x % 32 == 0 || y % 32 == 0;
        let on_diag = xi == yi || xi == (SIZE as i32 - 1 - yi);
        let dr = (((xi - c) * (xi - c) + (yi - c) * (yi - c)) as f64).sqrt();
        let on_circle = (dr - r as f64).abs() < 1.0;
        if on_grid || on_diag || on_circle {
            *p = Rgb([0, 0, 0]);
        }
    }
    eight("lineart", "line-art", img)
}

fn noise() -> CorpusImage {
    let mut rng = XorShift(0x1234_5678);
    let mut img = RgbImage::new(SIZE, SIZE);
    for p in img.pixels_mut() {
        // Noise around mid-gray so it reads like sensor grain, not pure static.
        let n = |rng: &mut XorShift| 128i32 + (rng.next_u8() as i32 - 128) / 2;
        *p = Rgb([n(&mut rng) as u8, n(&mut rng) as u8, n(&mut rng) as u8]);
    }
    eight("noise", "noise", img)
}

fn gradient16() -> CorpusImage {
    // Full 16-bit-range smooth ramp: the kind of content where 10-bit coding
    // earns its keep (no 8-bit banding in the source).
    let mut buf = Vec::with_capacity((SIZE * SIZE * 3) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let r = (x * 65535 / (SIZE - 1)) as u16;
            let g = (y * 65535 / (SIZE - 1)) as u16;
            let b = ((x + y) * 65535 / (2 * (SIZE - 1))) as u16;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    CorpusImage {
        name: "gradient16",
        category: "gradient-hdr",
        width: SIZE,
        height: SIZE,
        depth: Depth::Sixteen,
        rgb8: None,
        rgb16: Some(buf),
    }
}

/// Center crop of the natural test photo at `SIZE`x`SIZE`, 8-bit.
fn photo8(src: &image::DynamicImage) -> CorpusImage {
    let img = center_crop_rgb8(src);
    eight("photo", "natural-photo", img)
}

/// Same crop promoted to 16-bit (`<<8`) for the 10-bit path.
fn photo16(src: &image::DynamicImage) -> CorpusImage {
    let img = center_crop_rgb8(src);
    let buf: Vec<u16> = img.as_raw().iter().map(|&v| (v as u16) << 8).collect();
    CorpusImage {
        name: "photo16",
        category: "natural-photo-hdr",
        width: SIZE,
        height: SIZE,
        depth: Depth::Sixteen,
        rgb8: None,
        rgb16: Some(buf),
    }
}

fn center_crop_rgb8(src: &image::DynamicImage) -> RgbImage {
    let rgb = src.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let x0 = w.saturating_sub(SIZE) / 2;
    let y0 = h.saturating_sub(SIZE) / 2;
    let mut out = RgbImage::new(SIZE, SIZE);
    for (dx, dy, p) in out.enumerate_pixels_mut() {
        let sx = (x0 + dx).min(w - 1);
        let sy = (y0 + dy).min(h - 1);
        *p = *rgb.get_pixel(sx, sy);
    }
    out
}

fn eight(name: &'static str, category: &'static str, img: RgbImage) -> CorpusImage {
    CorpusImage {
        name,
        category,
        width: img.width(),
        height: img.height(),
        depth: Depth::Eight,
        rgb8: Some(img),
        rgb16: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic() {
        // Same generation twice must be byte-identical (the manifest depends
        // on this). Compare the synthetic (non-photo) entries.
        let a = build(None);
        let b = build(None);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.name, y.name);
            match x.depth {
                Depth::Eight => assert_eq!(
                    x.rgb8.as_ref().unwrap().as_raw(),
                    y.rgb8.as_ref().unwrap().as_raw(),
                    "{} differs between runs",
                    x.name
                ),
                Depth::Sixteen => {
                    assert_eq!(x.rgb16, y.rgb16, "{} differs between runs", x.name)
                }
            }
        }
    }

    #[test]
    fn synthetic_corpus_shape() {
        let imgs = build(None);
        // 5 eight-bit + 1 sixteen-bit synthetic entries, no photo.
        assert_eq!(imgs.len(), 6);
        for img in &imgs {
            assert_eq!((img.width, img.height), (SIZE, SIZE));
        }
    }
}
