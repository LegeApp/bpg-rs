//! BPG container header, ported from the `img_header` construction in
//! `bpg_encoder_encode` (`libbpg-0.9.8/bpgenc.c:2515-2573`), per
//! `bpg_spec.txt`'s `heic_file()` syntax.

use bpg_bitstream::{read_ue7, write_ue7};

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

/// Parsed still-image BPG input. `payload` points at the modified-HEVC data
/// after the container header and optional extension data.
#[derive(Debug, Clone)]
pub struct BpgFile<'a> {
    pub header: BpgHeader,
    pub payload: &'a [u8],
    pub extensions: Vec<BpgExtension<'a>>,
    pub has_alpha: bool,
    pub premultiplied_alpha: bool,
    pub has_w_plane: bool,
    pub chroma_phase: ChromaPhase,
}

#[derive(Debug, Clone)]
pub struct BpgExtension<'a> {
    pub tag: u32,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaPhase {
    Jpeg,
    Mpeg2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    Truncated,
    InvalidMagic,
    InvalidFlags(&'static str),
    InvalidUe7,
    InvalidDimensions,
    Unsupported(&'static str),
}

impl BpgHeader {
    pub const MAGIC: [u8; 4] = [0x42, 0x50, 0x47, 0xfb];

    /// Serialize the header into `out`. `picture_data_length == 0` means
    /// "up to the end of the file" (used for single still images).
    /// `extension_data`, if `Some` and non-empty, is written after the
    /// header with its length prefix.
    pub fn write(
        &self,
        out: &mut Vec<u8>,
        picture_data_length: u32,
        extension_data: Option<&[u8]>,
    ) {
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

    pub fn read(data: &[u8]) -> Result<BpgFile<'_>, FormatError> {
        if data.len() < 6 {
            return Err(FormatError::Truncated);
        }
        if data[..4] != Self::MAGIC {
            return Err(FormatError::InvalidMagic);
        }

        let mut pos = 4;
        let flags1 = data[pos];
        pos += 1;
        let flags2 = data[pos];
        pos += 1;

        let raw_format = flags1 >> 5;
        let pixel_format = match raw_format {
            0 => PixelFormat::Gray,
            1 => PixelFormat::Yuv420,
            2 => PixelFormat::Yuv422,
            3 => PixelFormat::Yuv444,
            4 => PixelFormat::Yuv420Video,
            5 => PixelFormat::Yuv422Video,
            _ => return Err(FormatError::InvalidFlags("unknown pixel format")),
        };
        let alpha1_flag = ((flags1 >> 4) & 1) != 0;
        let bit_depth_minus_8 = flags1 & 0x0f;
        if bit_depth_minus_8 > 6 {
            return Err(FormatError::InvalidFlags("bit depth > 14"));
        }

        let raw_color_space = flags2 >> 4;
        let color_space = match raw_color_space {
            0 => ColorSpaceCode::YCbCr,
            1 => ColorSpaceCode::Rgb,
            2 => ColorSpaceCode::YCgCo,
            3 => ColorSpaceCode::YCbCrBt709,
            4 => ColorSpaceCode::YCbCrBt2020,
            _ => return Err(FormatError::InvalidFlags("unknown color space")),
        };
        let has_extension = ((flags2 >> 3) & 1) != 0;
        let alpha2_flag = ((flags2 >> 2) & 1) != 0;
        let limited_range = ((flags2 >> 1) & 1) != 0;
        let animation_flag = (flags2 & 1) != 0;

        let mut has_alpha = false;
        let mut premultiplied_alpha = false;
        let mut has_w_plane = false;
        if alpha1_flag {
            has_alpha = true;
            premultiplied_alpha = alpha2_flag;
        } else if alpha2_flag {
            has_alpha = true;
            has_w_plane = true;
        }
        if pixel_format == PixelFormat::Gray && color_space != ColorSpaceCode::YCbCr {
            return Err(FormatError::InvalidFlags(
                "gray images must use YCbCr color space",
            ));
        }
        if has_w_plane && pixel_format == PixelFormat::Gray {
            return Err(FormatError::InvalidFlags(
                "gray images cannot have a W plane",
            ));
        }

        let width = read_ue7_strict(data, &mut pos)?;
        let height = read_ue7_strict(data, &mut pos)?;
        if width == 0 || height == 0 {
            return Err(FormatError::InvalidDimensions);
        }
        let picture_data_length = read_ue7_strict(data, &mut pos)?;

        let mut extensions = Vec::new();
        if has_extension {
            let extension_len = read_ue7_strict(data, &mut pos)? as usize;
            let extension_end = pos
                .checked_add(extension_len)
                .ok_or(FormatError::Truncated)?;
            if extension_end > data.len() {
                return Err(FormatError::Truncated);
            }
            let mut ext_pos = pos;
            while ext_pos < extension_end {
                let tag = read_ue7_strict(data, &mut ext_pos)?;
                let len = read_ue7_strict(data, &mut ext_pos)? as usize;
                let end = ext_pos.checked_add(len).ok_or(FormatError::Truncated)?;
                if end > extension_end {
                    return Err(FormatError::Truncated);
                }
                extensions.push(BpgExtension {
                    tag,
                    data: &data[ext_pos..end],
                });
                ext_pos = end;
            }
            pos = extension_end;
        }

        let payload_len = if picture_data_length == 0 {
            data.len().checked_sub(pos).ok_or(FormatError::Truncated)?
        } else {
            picture_data_length as usize
        };
        let payload_end = pos.checked_add(payload_len).ok_or(FormatError::Truncated)?;
        if payload_end > data.len() {
            return Err(FormatError::Truncated);
        }

        let (pixel_format, chroma_phase) = match pixel_format {
            PixelFormat::Yuv420Video => (PixelFormat::Yuv420, ChromaPhase::Mpeg2),
            PixelFormat::Yuv422Video => (PixelFormat::Yuv422, ChromaPhase::Mpeg2),
            other => (other, ChromaPhase::Jpeg),
        };

        Ok(BpgFile {
            header: BpgHeader {
                pixel_format,
                alpha1_flag,
                bit_depth_minus_8,
                color_space,
                alpha2_flag,
                limited_range,
                animation_flag,
                width,
                height,
            },
            payload: &data[pos..payload_end],
            extensions,
            has_alpha,
            premultiplied_alpha,
            has_w_plane,
            chroma_phase,
        })
    }
}

fn read_ue7_strict(data: &[u8], pos: &mut usize) -> Result<u32, FormatError> {
    let start = *pos;
    let value = read_ue7(data, pos).ok_or(FormatError::Truncated)?;
    let mut canonical = Vec::new();
    write_ue7(&mut canonical, value);
    if data.get(start..*pos) != Some(canonical.as_slice()) {
        return Err(FormatError::InvalidUe7);
    }
    Ok(value)
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
        assert_eq!(
            out,
            vec![0x42, 0x50, 0x47, 0xfb, 0x20, 0x00, 0x99, 0x00, 0x93, 0x60, 0x00]
        );
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
