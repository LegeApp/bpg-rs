//! BPG container header, ported from the `img_header` construction in
//! `bpg_encoder_encode` (`libbpg-0.9.8/bpgenc.c:2515-2573`), per
//! `bpg_spec.txt`'s `heic_file()` syntax.

use bpg_bitstream::write_ue7;

/// `pixel_format` field (3 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Gray = 0,
    Yuv420 = 1,
    Yuv422 = 2,
    Yuv444 = 3,
    /// 4:2:0 with chroma aligned to luma samples (MPEG2-style), used when
    /// `c_h_phase == 0`. TODO(extension): not produced by the M1 pipeline.
    Yuv420Video = 4,
    /// TODO(extension): not produced by the M1 pipeline.
    Yuv422Video = 5,
}

/// `color_space` field (4 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpaceCode {
    YCbCr = 0,
    Rgb = 1,
    YCgCo = 2,
    YCbCrBt709 = 3,
    YCbCrBt2020 = 4,
}

/// The fixed-size portion of a BPG file header (everything before the
/// optional extension data and the picture data itself).
#[derive(Debug, Clone)]
pub struct BpgHeader {
    pub pixel_format: PixelFormat,
    /// TODO(extension): alpha channel support.
    pub alpha1_flag: bool,
    pub bit_depth_minus_8: u8,
    pub color_space: ColorSpaceCode,
    /// TODO(extension): premultiplied-alpha / extra alpha plane support.
    pub alpha2_flag: bool,
    pub limited_range: bool,
    /// TODO(extension): animation support.
    pub animation_flag: bool,
    /// Pre-padding picture width.
    pub width: u32,
    /// Pre-padding picture height.
    pub height: u32,
}

impl BpgHeader {
    pub const MAGIC: [u8; 4] = [0x42, 0x50, 0x47, 0xfb];

    /// Serialize the header into `out`. `picture_data_length == 0` means
    /// "up to the end of the file" (used for single still images).
    /// `extension_data`, if `Some` and non-empty, is written after the
    /// header with its length prefix.
    pub fn write(&self, out: &mut Vec<u8>, picture_data_length: u32, extension_data: Option<&[u8]>) {
        let extension_data = extension_data.filter(|d| !d.is_empty());
        let has_extension = extension_data.is_some();

        out.extend_from_slice(&Self::MAGIC);

        let byte1 = ((self.pixel_format as u8) << 5)
            | ((self.alpha1_flag as u8) << 4)
            | (self.bit_depth_minus_8 & 0xf);
        out.push(byte1);

        let byte2 = ((self.color_space as u8) << 4)
            | ((has_extension as u8) << 3)
            | ((self.alpha2_flag as u8) << 2)
            | ((self.limited_range as u8) << 1)
            | (self.animation_flag as u8);
        out.push(byte2);

        write_ue7(out, self.width);
        write_ue7(out, self.height);
        write_ue7(out, picture_data_length);

        if let Some(ext) = extension_data {
            write_ue7(out, ext.len() as u32);
            out.extend_from_slice(ext);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches the header bytes of a real `bpgenc -e x265 -q 28 -f 420`
    /// output for an RGB (no-alpha), 3200x2528, 8-bit, full-range image.
    #[test]
    fn matches_reference_header_bytes() {
        let header = BpgHeader {
            pixel_format: PixelFormat::Yuv420,
            alpha1_flag: false,
            bit_depth_minus_8: 0,
            color_space: ColorSpaceCode::YCbCr,
            alpha2_flag: false,
            limited_range: false,
            animation_flag: false,
            width: 3200,
            height: 2528,
        };
        let mut out = Vec::new();
        header.write(&mut out, 0, None);
        assert_eq!(out, vec![0x42, 0x50, 0x47, 0xfb, 0x20, 0x00, 0x99, 0x00, 0x93, 0x60, 0x00]);
    }

    #[test]
    fn extension_data_is_length_prefixed() {
        let header = BpgHeader {
            pixel_format: PixelFormat::Yuv444,
            alpha1_flag: false,
            bit_depth_minus_8: 0,
            color_space: ColorSpaceCode::YCbCr,
            alpha2_flag: false,
            limited_range: true,
            animation_flag: false,
            width: 1,
            height: 1,
        };
        let mut out = Vec::new();
        header.write(&mut out, 0, Some(&[0xaa, 0xbb, 0xcc]));
        // magic(4) + byte1 + byte2 + ue7(1) + ue7(1) + ue7(0) + ue7(3) + 3 bytes
        assert_eq!(out.len(), 4 + 1 + 1 + 1 + 1 + 1 + 1 + 3);
        // has_extension bit (bit 3 of byte2) must be set
        assert_eq!(out[5] & 0x08, 0x08);
        assert_eq!(&out[out.len() - 4..], &[0x03, 0xaa, 0xbb, 0xcc]);
    }
}
