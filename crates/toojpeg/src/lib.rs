//! A Rust port of the TooJpeg JPEG encoder with performance optimizations.
//!
//! This library provides a simple interface for encoding RGB(A) images to JPEG format
//! with various quality and optimization settings.
//!
//! Vendored into bpg-rs with one addition over upstream TooJpeg: a native
//! **YCbCr-plane** entry point ([`encode_jpeg_ycbcr`]) that writes pre-computed
//! BT.601 full-range YCbCr samples straight into the JPEG, skipping the
//! RGB→YCbCr step. This lets a decoded BPG/HEIC frame (already YCbCr) be
//! re-encoded to JPEG without an intermediate RGB conversion. Upstream:
//! <https://create.stephan-brumme.com/toojpeg/>.

#![allow(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

mod toojpeg;

pub use toojpeg::{write_jpeg, BitCode, BitWriter, I16, I32, U16, U8};

/// Image format options for the JPEG encoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// RGB format (3 bytes per pixel)
    RGB,
    /// RGBA format (4 bytes per pixel, alpha is ignored)
    RGBA,
    /// Grayscale format (1 byte per pixel)
    Gray,
}

/// JPEG encoding options
#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Image format (RGB, RGBA, or Grayscale)
    pub format: ImageFormat,
    /// Quality from 1 (worst) to 100 (best)
    pub quality: u8,
    /// Whether to use baseline DCT encoding (true) or progressive (false)
    pub baseline: bool,
    /// Whether to use optimized Huffman tables
    pub optimized: bool,
    /// Whether to downsample chroma channels (4:2:0 subsampling)
    pub downsample: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: ImageFormat::RGB,
            quality: 90,
            baseline: true,
            optimized: true,
            downsample: true,
        }
    }
}

/// Encode an image to JPEG format
///
/// # Arguments
/// * `pixels` - The image pixel data in the format specified by `options.format`
/// * `options` - Encoding options including dimensions, format, and quality
/// * `output` - A writer that implements `std::io::Write` to receive the JPEG data
///
/// # Returns
/// `Result<(), &'static str>` indicating success or an error message
pub fn encode_jpeg<W: std::io::Write>(
    pixels: &[u8],
    options: EncodeOptions,
    output: &mut W,
) -> Result<(), &'static str> {
    // Input validation
    let bytes_per_pixel = match options.format {
        ImageFormat::RGB => 3,
        ImageFormat::RGBA => 4,
        ImageFormat::Gray => 1,
    };

    let expected_len = (options.width * options.height * bytes_per_pixel) as usize;
    if pixels.len() < expected_len {
        return Err("Input buffer too small for specified dimensions and format");
    }

    // Input validation
    let bytes_per_pixel = match options.format {
        ImageFormat::RGB => 3,
        ImageFormat::RGBA => 4,
        ImageFormat::Gray => 1,
    };

    let expected_len = (options.width as usize)
        .checked_mul(options.height as usize)
        .and_then(|x| x.checked_mul(bytes_per_pixel));

    if expected_len.map_or(true, |len| pixels.len() < len) {
        return Err("Input buffer too small for specified dimensions and format");
    }

    // Convert to the format expected by write_jpeg
    let is_rgb = matches!(options.format, ImageFormat::RGB | ImageFormat::RGBA);
    let quality = options.quality.clamp(1, 100) as u8;

    // Create a BitWriter for the output
    let mut writer = BitWriter::new(|byte| {
        output
            .write_all(&[byte])
            .map_err(|_| "Failed to write output")
    });

    // Call the low-level write_jpeg function
    write_jpeg(
        &mut writer,
        pixels,
        options.width as u16,
        options.height as u16,
        is_rgb,
        false, // input_is_ycbcr: this path takes RGB(A)/Gray
        quality,
        options.downsample,
        None, // comment
    )
}

/// Encode pre-computed BT.601 full-range YCbCr samples to JPEG without an
/// intermediate RGB conversion.
///
/// `pixels` must be interleaved `[Y, Cb, Cr, Y, Cb, Cr, ...]` (3 bytes per
/// pixel, full-resolution 4:4:4), with Y/Cb/Cr each in `0..=255` and chroma
/// centered at 128 — exactly the layout `zune-jpeg` produces for native YCbCr
/// output and the layout a decoded BPG frame yields. When `downsample` is true
/// the encoder writes 4:2:0 (averaging the supplied chroma 2×2); otherwise 4:4:4.
///
/// For grayscale, pass the single-channel luma via [`encode_jpeg`] with
/// [`ImageFormat::Gray`] instead.
pub fn encode_jpeg_ycbcr<W: std::io::Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    downsample: bool,
    output: &mut W,
) -> Result<(), &'static str> {
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|x| x.checked_mul(3));
    if expected_len.map_or(true, |len| pixels.len() < len) {
        return Err("Input buffer too small for specified dimensions (need width*height*3 YCbCr)");
    }
    let quality = quality.clamp(1, 100);
    let mut writer = BitWriter::new(|byte| {
        output
            .write_all(&[byte])
            .map_err(|_| "Failed to write output")
    });
    write_jpeg(
        &mut writer,
        pixels,
        width as u16,
        height as u16,
        false, // is_rgb: input is YCbCr, not RGB
        true,  // input_is_ycbcr
        quality,
        downsample,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_rgb() {
        // Create a simple 2x2 RGB image (red, green, blue, white)
        let pixels = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 255, // White
        ];

        let options = EncodeOptions {
            width: 2,
            height: 2,
            format: ImageFormat::RGB,
            quality: 90,
            ..Default::default()
        };

        let mut output = Vec::new();
        encode_jpeg(&pixels, options, &mut output).unwrap();

        // Basic validation of JPEG output
        assert!(output.len() > 100); // Should be at least 100 bytes
        assert_eq!(&output[0..2], [0xFF, 0xD8]); // JPEG SOI marker
    }

    #[test]
    fn test_encode_ycbcr_produces_valid_jpeg() {
        // 8x8 mid-gray, neutral chroma (Y=128, Cb=Cr=128) -> valid baseline JPEG.
        let w = 8u32;
        let h = 8u32;
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            px.extend_from_slice(&[128, 128, 128]);
        }
        for downsample in [false, true] {
            let mut out = Vec::new();
            encode_jpeg_ycbcr(&px, w, h, 90, downsample, &mut out).unwrap();
            assert!(out.len() > 100, "jpeg too small (downsample={downsample})");
            assert_eq!(&out[0..2], [0xFF, 0xD8], "missing SOI");
            assert_eq!(&out[out.len() - 2..], [0xFF, 0xD9], "missing EOI");
        }
    }

    #[test]
    fn test_encode_ycbcr_rejects_short_buffer() {
        let px = vec![128u8; 8 * 8 * 3 - 1];
        assert!(encode_jpeg_ycbcr(&px, 8, 8, 90, false, &mut Vec::new()).is_err());
    }
}
