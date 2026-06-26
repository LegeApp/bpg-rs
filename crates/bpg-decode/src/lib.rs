//! Pure Rust BPG/HEIC still-image decode orchestration.
//!
//! * BPG: parses the BPG container, rebuilds an Annex-B HEVC stream from the
//!   modified-HEVC payload, and decodes with `bpg-hevc-decode`.
//! * HEIC/HEIF: full ISOBMFF container parser + same HEVC decoder.
//!
//! The HEIC path is exposed via the [`heic`] submodule.

pub mod heic;

use bpg_format::{BpgHeader, ChromaPhase, ColorSpaceCode, FormatError, PixelFormat};
use bpg_hevc::{rebuild_annexb_from_bpg_payload, BpgHevcInfo};
use std::path::Path;

pub use bpg_hevc_decode::hevc::is_ycbcr_matrix;
pub use bpg_hevc_decode::DecodedFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerKind {
    Bpg,
    Heif,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EstimateCompleteness {
    Complete,
    NeedsContainerMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerMemoryEstimate {
    pub kind: ContainerKind,
    pub completeness: EstimateCompleteness,
    pub encoded_bytes: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bit_depth: Option<u8>,
    pub yuv_decode_bytes: Option<u64>,
    pub rgba8_output_bytes: Option<u64>,
    pub transient_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
}

#[derive(Debug)]
pub enum EstimateError {
    Format(FormatError),
    Io(std::io::Error),
    Unsupported(&'static str),
}

impl From<FormatError> for EstimateError {
    fn from(value: FormatError) -> Self {
        Self::Format(value)
    }
}

impl From<std::io::Error> for EstimateError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl std::fmt::Display for EstimateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Format(e) => write!(f, "BPG format error: {e:?}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Unsupported(msg) => write!(f, "unsupported container estimate: {msg}"),
        }
    }
}

impl std::error::Error for EstimateError {}

pub fn detect_container_kind(data: &[u8]) -> ContainerKind {
    if data.len() >= 4 && &data[0..4] == b"BPG\xFB" {
        return ContainerKind::Bpg;
    }
    if parse_ftyp_major_brand(data).is_some_and(is_heif_brand) {
        return ContainerKind::Heif;
    }
    ContainerKind::Unknown
}

pub fn estimate_container_memory_from_bytes(
    data: &[u8],
) -> Result<ContainerMemoryEstimate, EstimateError> {
    match detect_container_kind(data) {
        ContainerKind::Bpg => estimate_bpg_container_memory_from_bytes(data),
        ContainerKind::Heif => Ok(estimate_heif_container_placeholder(data.len() as u64)),
        ContainerKind::Unknown => Err(EstimateError::Unsupported(
            "unknown container (expected BPG or HEIF-like ftyp brand)",
        )),
    }
}

pub fn estimate_container_memory_from_path(
    path: &Path,
) -> Result<ContainerMemoryEstimate, EstimateError> {
    let data = std::fs::read(path)?;
    estimate_container_memory_from_bytes(&data)
}

pub fn estimate_bpg_container_memory_from_bytes(
    data: &[u8],
) -> Result<ContainerMemoryEstimate, EstimateError> {
    let file = BpgHeader::read(data)?;
    let width = file.header.width;
    let height = file.header.height;
    let bit_depth = file.header.bit_depth_minus_8 + 8;
    let px = u64::from(width) * u64::from(height);

    let yuv_samples = bpg_plane_samples(width, height, file.header.pixel_format);
    // Keep this conservative and aligned with the existing decoder-side limit model:
    // use u16 plane storage regardless of nominal bit depth.
    let yuv_decode_bytes = yuv_samples * 2;
    let rgba8_output_bytes = px.saturating_mul(4);
    let transient_bytes = px / 8;
    let peak_bytes = yuv_decode_bytes
        .saturating_add(rgba8_output_bytes)
        .saturating_add(transient_bytes);

    Ok(ContainerMemoryEstimate {
        kind: ContainerKind::Bpg,
        completeness: EstimateCompleteness::Complete,
        encoded_bytes: data.len() as u64,
        width: Some(width),
        height: Some(height),
        bit_depth: Some(bit_depth),
        yuv_decode_bytes: Some(yuv_decode_bytes),
        rgba8_output_bytes: Some(rgba8_output_bytes),
        transient_bytes: Some(transient_bytes),
        peak_bytes: Some(peak_bytes),
    })
}

fn estimate_heif_container_placeholder(encoded_bytes: u64) -> ContainerMemoryEstimate {
    ContainerMemoryEstimate {
        kind: ContainerKind::Heif,
        completeness: EstimateCompleteness::NeedsContainerMetadata,
        encoded_bytes,
        width: None,
        height: None,
        bit_depth: None,
        yuv_decode_bytes: None,
        rgba8_output_bytes: None,
        transient_bytes: None,
        peak_bytes: None,
    }
}

fn parse_ftyp_major_brand(data: &[u8]) -> Option<[u8; 4]> {
    if data.len() < 12 {
        return None;
    }
    if &data[4..8] != b"ftyp" {
        return None;
    }
    Some([data[8], data[9], data[10], data[11]])
}

fn is_heif_brand(brand: [u8; 4]) -> bool {
    matches!(
        &brand,
        b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1" | b"avif" | b"avis"
    )
}

fn bpg_plane_samples(width: u32, height: u32, format: PixelFormat) -> u64 {
    let y = u64::from(width) * u64::from(height);
    match format {
        PixelFormat::Gray => y,
        PixelFormat::Yuv420 | PixelFormat::Yuv420Video => {
            let cw = u64::from(width.div_ceil(2));
            let ch = u64::from(height.div_ceil(2));
            y + 2 * cw * ch
        }
        PixelFormat::Yuv422 | PixelFormat::Yuv422Video => {
            let cw = u64::from(width.div_ceil(2));
            let ch = u64::from(height);
            y + 2 * cw * ch
        }
        PixelFormat::Yuv444 => y.saturating_mul(3),
    }
}

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
    BufferTooSmall {
        required: usize,
        actual: usize,
    },
    LimitExceeded(&'static str),
    /// HEIF/ISOBMFF container parse error.
    Container(&'static str),
    /// Invalid data encountered during HEIC decode.
    InvalidData(&'static str),
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
            Self::Container(msg) => write!(f, "HEIF container error: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid HEIC data: {msg}"),
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
    // All BPG chroma formats (4:2:0, 4:2:2, 4:4:4) and monochrome are
    // supported by the vendored HEVC decoder.
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
        ColorSpaceCode::Rgb => Ok(0),
        ColorSpaceCode::YCgCo => Ok(8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_bpg_container() {
        let mut bytes = Vec::new();
        let hdr = bpg_format::BpgHeader {
            pixel_format: bpg_format::PixelFormat::Yuv420,
            alpha1_flag: false,
            bit_depth_minus_8: 0,
            color_space: bpg_format::ColorSpaceCode::YCbCr,
            alpha2_flag: false,
            limited_range: false,
            animation_flag: false,
            width: 16,
            height: 8,
        };
        hdr.write(&mut bytes, 0, None);
        bytes.extend_from_slice(&[0, 0, 0, 1]);

        assert_eq!(detect_container_kind(&bytes), ContainerKind::Bpg);
    }

    #[test]
    fn detects_heif_container_brand() {
        let bytes: [u8; 24] = [
            0, 0, 0, 24, b'f', b't', b'y', b'p', b'h', b'e', b'i', b'c', 0, 0, 0, 0, b'm', b'i',
            b'f', b'1', b'h', b'e', b'i', b'c',
        ];
        assert_eq!(detect_container_kind(&bytes), ContainerKind::Heif);
    }

    #[test]
    fn estimates_bpg_memory_from_header() {
        let mut bytes = Vec::new();
        let hdr = bpg_format::BpgHeader {
            pixel_format: bpg_format::PixelFormat::Yuv420,
            alpha1_flag: false,
            bit_depth_minus_8: 0,
            color_space: bpg_format::ColorSpaceCode::YCbCr,
            alpha2_flag: false,
            limited_range: false,
            animation_flag: false,
            width: 64,
            height: 32,
        };
        hdr.write(&mut bytes, 0, None);
        bytes.extend_from_slice(&[0, 1, 2, 3, 4]);

        let est = estimate_container_memory_from_bytes(&bytes).unwrap();
        assert_eq!(est.kind, ContainerKind::Bpg);
        assert_eq!(est.completeness, EstimateCompleteness::Complete);
        assert_eq!(est.width, Some(64));
        assert_eq!(est.height, Some(32));
        assert!(est.peak_bytes.unwrap() > 0);
    }

    #[test]
    fn heif_estimate_is_placeholder_for_future_item_parsing() {
        let bytes: [u8; 24] = [
            0, 0, 0, 24, b'f', b't', b'y', b'p', b'h', b'e', b'i', b'c', 0, 0, 0, 0, b'm', b'i',
            b'f', b'1', b'h', b'e', b'i', b'c',
        ];
        let est = estimate_container_memory_from_bytes(&bytes).unwrap();
        assert_eq!(est.kind, ContainerKind::Heif);
        assert_eq!(
            est.completeness,
            EstimateCompleteness::NeedsContainerMetadata
        );
        assert_eq!(est.peak_bytes, None);
    }
}
