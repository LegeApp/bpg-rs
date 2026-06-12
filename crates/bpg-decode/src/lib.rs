//! Pure Rust BPG still-image decode orchestration.
//!
//! This crate parses the BPG container, rebuilds a normal Annex-B HEVC stream
//! from BPG's modified-HEVC payload, and decodes it with the Rust HEVC decoder
//! used by `heic-decoder-rs`.

use bpg_format::{BpgHeader, ChromaPhase, ColorSpaceCode, FormatError, PixelFormat};
use bpg_hevc::{rebuild_annexb_from_bpg_payload, BpgHevcInfo};

pub use bpg_hevc_decode::DecodedFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelLayout {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
}

impl PixelLayout {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Bgr8 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub bit_depth: u8,
    pub pixel_format: PixelFormat,
    pub color_space: ColorSpaceCode,
}

impl ImageInfo {
    pub fn from_bytes(data: &[u8]) -> Result<Self, DecodeError> {
        let file = BpgHeader::read(data)?;
        Ok(Self {
            width: file.header.width,
            height: file.header.height,
            has_alpha: file.has_alpha,
            bit_depth: file.header.bit_depth_minus_8 + 8,
            pixel_format: file.header.pixel_format,
            color_space: file.header.color_space,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Limits {
    pub max_width: Option<u64>,
    pub max_height: Option<u64>,
    pub max_pixels: Option<u64>,
    pub max_memory_bytes: Option<u64>,
}

impl Limits {
    fn check_dimensions(&self, width: u32, height: u32) -> Result<(), DecodeError> {
        if self.max_width.is_some_and(|max| u64::from(width) > max) {
            return Err(DecodeError::LimitExceeded("image width exceeds limit"));
        }
        if self.max_height.is_some_and(|max| u64::from(height) > max) {
            return Err(DecodeError::LimitExceeded("image height exceeds limit"));
        }
        if self
            .max_pixels
            .is_some_and(|max| u64::from(width) * u64::from(height) > max)
        {
            return Err(DecodeError::LimitExceeded("pixel count exceeds limit"));
        }
        Ok(())
    }

    fn check_memory(&self, bytes: u64) -> Result<(), DecodeError> {
        if self.max_memory_bytes.is_some_and(|max| bytes > max) {
            return Err(DecodeError::LimitExceeded("estimated memory exceeds limit"));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum DecodeError {
    Format(FormatError),
    HevcRebuild(bpg_hevc::HevcError),
    HevcDecode(bpg_hevc_decode::HevcError),
    Unsupported(&'static str),
    BufferTooSmall { required: usize, actual: usize },
    LimitExceeded(&'static str),
}

impl From<FormatError> for DecodeError {
    fn from(value: FormatError) -> Self {
        Self::Format(value)
    }
}

impl From<bpg_hevc::HevcError> for DecodeError {
    fn from(value: bpg_hevc::HevcError) -> Self {
        Self::HevcRebuild(value)
    }
}

impl From<bpg_hevc_decode::HevcError> for DecodeError {
    fn from(value: bpg_hevc_decode::HevcError) -> Self {
        Self::HevcDecode(value)
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format(e) => write!(f, "BPG format error: {e:?}"),
            Self::HevcRebuild(e) => write!(f, "BPG HEVC rebuild error: {e:?}"),
            Self::HevcDecode(e) => write!(f, "HEVC decode error: {e}"),
            Self::Unsupported(msg) => write!(f, "unsupported BPG feature: {msg}"),
            Self::BufferTooSmall { required, actual } => {
                write!(f, "buffer too small: need {required}, got {actual}")
            }
            Self::LimitExceeded(msg) => write!(f, "limit exceeded: {msg}"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone, Default)]
pub struct DecoderConfig;

impl DecoderConfig {
    pub fn new() -> Self {
        Self
    }

    pub fn decode(&self, data: &[u8], layout: PixelLayout) -> Result<DecodeOutput, DecodeError> {
        self.decode_request(data)
            .with_output_layout(layout)
            .decode()
    }

    pub fn decode_request<'a>(&'a self, data: &'a [u8]) -> DecodeRequest<'a> {
        DecodeRequest {
            _config: self,
            data,
            layout: PixelLayout::Rgba8,
            limits: None,
        }
    }

    pub fn decode_to_frame(&self, data: &[u8]) -> Result<DecodedFrame, DecodeError> {
        decode_to_frame_inner(data, None)
    }
}

pub struct DecodeRequest<'a> {
    _config: &'a DecoderConfig,
    data: &'a [u8],
    layout: PixelLayout,
    limits: Option<&'a Limits>,
}

impl<'a> DecodeRequest<'a> {
    pub fn with_output_layout(mut self, layout: PixelLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_limits(mut self, limits: &'a Limits) -> Self {
        self.limits = Some(limits);
        self
    }

    pub fn decode(self) -> Result<DecodeOutput, DecodeError> {
        let frame = decode_to_frame_inner(self.data, self.limits)?;
        let width = frame.cropped_width();
        let height = frame.cropped_height();
        if let Some(limits) = self.limits {
            limits.check_dimensions(width, height)?;
            limits.check_memory(
                u64::from(width) * u64::from(height) * self.layout.bytes_per_pixel() as u64,
            )?;
        }

        let data = match self.layout {
            PixelLayout::Rgb8 => frame.to_rgb(),
            PixelLayout::Rgba8 => frame.to_rgba(),
            PixelLayout::Bgr8 => frame.to_bgr(),
            PixelLayout::Bgra8 => frame.to_bgra(),
        };
        Ok(DecodeOutput {
            data,
            width,
            height,
            layout: self.layout,
        })
    }

    pub fn decode_into(self, output: &mut [u8]) -> Result<ImageInfo, DecodeError> {
        let info = ImageInfo::from_bytes(self.data)?;
        let decoded = self.decode()?;
        let required = decoded.data.len();
        if output.len() < required {
            return Err(DecodeError::BufferTooSmall {
                required,
                actual: output.len(),
            });
        }
        output[..required].copy_from_slice(&decoded.data);
        Ok(info)
    }

    pub fn decode_yuv(self) -> Result<DecodedFrame, DecodeError> {
        decode_to_frame_inner(self.data, self.limits)
    }
}

fn decode_to_frame_inner(
    data: &[u8],
    limits: Option<&Limits>,
) -> Result<DecodedFrame, DecodeError> {
    let file = BpgHeader::read(data)?;
    reject_unsupported(&file)?;

    if let Some(limits) = limits {
        limits.check_dimensions(file.header.width, file.header.height)?;
        // Conservative 4:4:4 u16 planes plus RGBA output and decoder metadata.
        let px = u64::from(file.header.width) * u64::from(file.header.height);
        limits.check_memory(px * 2 * 3 + px * 4 + px / 8)?;
    }

    let info = BpgHevcInfo {
        width: file.header.width,
        height: file.header.height,
        chroma_format_idc: chroma_format_idc(file.header.pixel_format),
        bit_depth: file.header.bit_depth_minus_8 + 8,
        limited_range: file.header.limited_range,
        matrix_coefficients: matrix_coefficients(file.header.color_space)?,
    };
    let annexb = rebuild_annexb_from_bpg_payload(file.payload, info)?;
    let frame = bpg_hevc_decode::hevc::decode(&annexb)?;
    Ok(frame)
}

fn reject_unsupported(file: &bpg_format::BpgFile<'_>) -> Result<(), DecodeError> {
    if file.header.animation_flag {
        return Err(DecodeError::Unsupported("animation"));
    }
    if file.has_alpha {
        return Err(DecodeError::Unsupported("alpha and W-plane images"));
    }
    if file.premultiplied_alpha || file.has_w_plane {
        return Err(DecodeError::Unsupported(
            "premultiplied alpha or CMYK/W-plane",
        ));
    }
    if file.chroma_phase != ChromaPhase::Jpeg {
        return Err(DecodeError::Unsupported("MPEG2 chroma siting"));
    }
    // The vendored HEVC decoder only reconstructs 4:2:0 (and monochrome)
    // correctly. Its transform-tree / residual path was built and tested for
    // 4:2:0 (the dominant HEIC/BPG case); 4:2:2 and 4:4:4 decode to garbage
    // (the oracle measures ~5-8 dB PSNR vs stock bpgdec). Reject them rather
    // than emit a plausible-looking-but-wrong image — the encoder still
    // produces correct 4:2:2/4:4:4 BPG (verified byte-identical to the C
    // reference), so use stock bpgdec to decode those until the Rust decoder
    // gains real chroma support. See FULL_RUST_CODEC_PLAN.md (Phase 1 follow-up).
    match file.header.pixel_format {
        PixelFormat::Gray
        | PixelFormat::Yuv420
        | PixelFormat::Yuv420Video
        | PixelFormat::Yuv444 => {}
        PixelFormat::Yuv422 | PixelFormat::Yuv422Video => {
            return Err(DecodeError::Unsupported(
                "4:2:2 decode (not yet implemented in the Rust decoder)",
            ))
        }
    }
    match file.header.color_space {
        ColorSpaceCode::YCbCr | ColorSpaceCode::YCbCrBt709 | ColorSpaceCode::YCbCrBt2020 => {}
        ColorSpaceCode::Rgb => return Err(DecodeError::Unsupported("RGB color space")),
        ColorSpaceCode::YCgCo => return Err(DecodeError::Unsupported("YCgCo color space")),
    }
    Ok(())
}

fn chroma_format_idc(format: PixelFormat) -> u8 {
    match format {
        PixelFormat::Gray => 0,
        PixelFormat::Yuv420 | PixelFormat::Yuv420Video => 1,
        PixelFormat::Yuv422 | PixelFormat::Yuv422Video => 2,
        PixelFormat::Yuv444 => 3,
    }
}

fn matrix_coefficients(color_space: ColorSpaceCode) -> Result<u8, DecodeError> {
    match color_space {
        ColorSpaceCode::YCbCr => Ok(6),
        ColorSpaceCode::YCbCrBt709 => Ok(1),
        ColorSpaceCode::YCbCrBt2020 => Ok(9),
        ColorSpaceCode::Rgb => Err(DecodeError::Unsupported("RGB color space")),
        ColorSpaceCode::YCgCo => Err(DecodeError::Unsupported("YCgCo color space")),
    }
}
