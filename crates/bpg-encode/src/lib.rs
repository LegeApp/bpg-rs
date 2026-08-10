//! Still-image BPG encode orchestration: ties together the image pipeline
//! (`bpg-image`), the HEVC backend (via the [`HevcEncoder`] trait), the HEVC
//! bitstream rewriting (`bpg-hevc`), and the container header
//! (`bpg-format`).
//!
//! Ported from the still-image (`!animated`, no-alpha) path of
//! `bpg_encoder_encode` in `libbpg-0.9.8/bpgenc.c:2360-2601`. Alpha,
//! animation, and lossless are `TODO(extension)`. 8/10/12-bit are supported.

use bpg_format::{BpgHeader, ColorSpaceCode, PixelFormat};
use bpg_hevc::{build_modified_hevc, HevcError};
use bpg_image::{ChromaFormat, ColorSpace, Image};

pub mod heic;

/// Encoder-tuning knobs resolved from a `bpg-tune::EncodePlan`. These are the
/// *soft* parameters a tune is allowed to set; the mandatory BPG/structural
/// params (intra-only, AMP, CQP, csp, bit depth, repeat-headers) are always
/// re-asserted by the backend AFTER these, so a tune can never violate the
/// modified-SPS preconditions in `bpg-hevc`. A failure (bad x265 param name or
/// value) surfaces as `EncodeError::Backend`, never a corrupt bitstream.
#[derive(Debug, Clone, Default)]
pub struct EncoderTuning {
    /// x265's own `--tune` (psnr/ssim/grain/...), passed to
    /// `param_default_preset`. `None` keeps x265's preset default. Note
    /// `Default` gives `None`, which is *not* the historical `"ssim"` the M1
    /// reference path uses — see [`EncoderTuning::neutral`].
    pub x265_tune: Option<String>,
    /// Ordered `x265_param_parse(name, value)` overrides (e.g.
    /// `("psy-rd", "2.0")`, `("aq-mode", "3")`). Applied after preset+tune but
    /// before the mandatory structural fields, so later entries win and the
    /// container invariants always override. Order is preserved.
    pub x265_params: Vec<(String, String)>,
}

impl EncoderTuning {
    /// The historical M1 default: x265 `--tune ssim`, no extra overrides. This
    /// reproduces the byte-for-byte-identical-to-C-reference behavior and is
    /// what `Tune::Neutral` should resolve to.
    pub fn neutral() -> Self {
        Self {
            x265_tune: Some("ssim".to_string()),
            x265_params: Vec::new(),
        }
    }
}

/// Parameters for one HEVC encode, mirroring the subset of
/// `HEVCEncodeParams` (`bpgenc.h`) that the M1 x265 path uses. The backend is
/// responsible for translating these into x265 params (`x265_glue.c`).
#[derive(Debug, Clone)]
pub struct HevcEncodeParams {
    /// Padded (CTU-aligned) luma width fed to the encoder.
    pub width: u32,
    /// Padded (CTU-aligned) luma height fed to the encoder.
    pub height: u32,
    /// HEVC chroma_format_idc (0=gray,1=420,2=422,3=444).
    pub chroma_format: ChromaFormat,
    /// 8, 10, or 12. The backend must support encoding at this depth.
    pub bit_depth: u8,
    /// Quantizer, 0-51 (CQP rate-control mode).
    pub qp: u8,
    /// x265 preset index, 1 (fast) - 9 (placebo). Maps to compress_level.
    pub compress_level: u8,
    /// Emit a decoded-picture-hash SEI (0 = none).
    pub sei_decoded_picture_hash: bool,
    pub verbose: bool,
    /// Soft encoder-tuning knobs (x265 `--tune` + extra param overrides),
    /// resolved from a `bpg-tune::EncodePlan`. The backend applies these
    /// before re-asserting the mandatory container/structural params.
    pub tuning: EncoderTuning,
}

/// Declared capabilities of a [`HevcEncoder`] backend, used by callers (e.g.
/// `bpg-tools`) to validate a request *before* doing any expensive work
/// (image loading, color conversion, CTU padding) and to keep CLI help/README
/// claims honest. A backend must not silently downgrade or ignore an
/// unsupported request: [`HevcEncoder::encode_still`] still returns
/// [`EncodeError::Unsupported`] for anything outside these bounds, but
/// `caps()` lets callers fail fast with a clear message.
#[derive(Debug, Clone, Copy)]
pub struct HevcBackendCaps {
    /// Sample bit depths this backend can encode (e.g. `&[8, 10, 12]`).
    pub bit_depths: &'static [u8],
    /// Chroma subsampling formats this backend can encode.
    pub chroma_formats: &'static [ChromaFormat],
    /// Whether the backend can apply Sample Adaptive Offset for extra RD
    /// gain (the decoder can always *decode* SAO; this is about the encoder
    /// choosing to use it).
    pub supports_sao: bool,
    /// Whether the backend can apply in-loop deblocking for extra RD gain.
    pub supports_deblock: bool,
    /// Whether the backend can produce a lossless (transquant-bypass)
    /// bitstream.
    pub supports_lossless: bool,
    /// Whether the backend can encode an alpha plane.
    pub supports_alpha: bool,
}

impl HevcBackendCaps {
    /// Whether `bit_depth` is one of [`Self::bit_depths`].
    pub fn supports_bit_depth(&self, bit_depth: u8) -> bool {
        self.bit_depths.contains(&bit_depth)
    }

    /// Whether `chroma_format` is one of [`Self::chroma_formats`].
    pub fn supports_chroma_format(&self, chroma_format: ChromaFormat) -> bool {
        self.chroma_formats.contains(&chroma_format)
    }
}

/// A HEVC backend that encodes a single intra still frame and returns its raw
/// Annex-B elementary stream. Implemented by `still265`.
///
/// This is the Rust analogue of the C `HEVCEncoder` vtable
/// (`open`/`encode`/`close`) collapsed into one call, since the M1 path
/// encodes exactly one frame.
pub trait HevcEncoder {
    /// Encode the planar `image` (already color-converted, subsampled, and
    /// CTU-padded) as a single intra frame, returning the raw Annex-B HEVC
    /// stream (with `bRepeatHeaders` so VPS/SPS/PPS precede the slice).
    fn encode_still(
        &self,
        image: &Image,
        params: &HevcEncodeParams,
    ) -> Result<Vec<u8>, EncodeError>;

    /// Report this backend's capabilities (supported bit depths, chroma
    /// formats, and in-loop filter / lossless / alpha support), so callers
    /// can validate a request before doing expensive work.
    fn caps(&self) -> HevcBackendCaps;
}

/// Errors from the encode pipeline.
#[derive(Debug)]
pub enum EncodeError {
    /// The HEVC backend failed. `msg` carries backend-specific detail.
    Backend(String),
    /// The HEVC bitstream rewriting (`bpg-hevc`) failed.
    Hevc(HevcError),
    /// An input the M1 pipeline does not support.
    Unsupported(&'static str),
    /// HEIF/ISOBMFF container construction failed.
    Container(String),
}

impl From<HevcError> for EncodeError {
    fn from(e: HevcError) -> Self {
        EncodeError::Hevc(e)
    }
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::Backend(m) => write!(f, "HEVC backend error: {m}"),
            EncodeError::Hevc(e) => write!(f, "HEVC rewrite error: {e:?}"),
            EncodeError::Unsupported(w) => write!(f, "unsupported: {w}"),
            EncodeError::Container(m) => write!(f, "container error: {m}"),
        }
    }
}

impl std::error::Error for EncodeError {}

/// CTU size assumed by the encoder. `bpgenc.c` hard-codes `cb_size = 8` and
/// requires the HEVC encoder to use the same value.
pub const CB_SIZE: u32 = 8;

/// Raw single-picture HEVC plus the image metadata needed by a container.
///
/// The encoder consumes a padded [`Image`], but BPG and HEIF both need the
/// pre-padding display dimensions. Keeping the two dimensions together here
/// prevents a container from accidentally exposing replicated edge pixels.
#[derive(Debug)]
pub(crate) struct EncodedHevcStill {
    pub annexb: Vec<u8>,
    pub display_width: u32,
    pub display_height: u32,
    pub coded_width: u32,
    pub coded_height: u32,
    pub chroma_format: ChromaFormat,
    pub bit_depth: u8,
    pub color_space: ColorSpace,
    pub limited_range: bool,
}

fn encode_annexb_still(
    mut image: Image,
    backend: &dyn HevcEncoder,
    qp: u8,
    compress_level: u8,
    tuning: EncoderTuning,
) -> Result<EncodedHevcStill, EncodeError> {
    if ![8, 10, 12].contains(&image.bit_depth) {
        return Err(EncodeError::Unsupported("bit depth must be 8, 10, or 12"));
    }
    if image.has_alpha {
        return Err(EncodeError::Unsupported("alpha not supported"));
    }

    let display_width = image.width;
    let display_height = image.height;
    image.pad_to_cb_size(CB_SIZE);

    let params = HevcEncodeParams {
        width: image.width,
        height: image.height,
        chroma_format: image.chroma_format,
        bit_depth: image.bit_depth,
        qp,
        compress_level,
        sei_decoded_picture_hash: false,
        verbose: false,
        tuning,
    };
    let annexb = backend.encode_still(&image, &params)?;

    Ok(EncodedHevcStill {
        annexb,
        display_width,
        display_height,
        coded_width: image.width,
        coded_height: image.height,
        chroma_format: image.chroma_format,
        bit_depth: image.bit_depth,
        color_space: image.color_space,
        limited_range: image.limited_range,
    })
}

/// Encode an in-memory `image` (4:4:4 or already-subsampled YCbCr) to a
/// complete `.bpg` byte stream using the given HEVC `backend`.
///
/// The `image` is consumed/mutated: it is CTU-padded in place. The original
/// (pre-padding) dimensions are captured for the container header before
/// padding, matching the C code's `width = img->w; height = img->h;` taken
/// before `image_pad`.
pub fn encode_still_image(
    image: Image,
    backend: &dyn HevcEncoder,
    qp: u8,
    compress_level: u8,
    tuning: EncoderTuning,
) -> Result<Vec<u8>, EncodeError> {
    let encoded = encode_annexb_still(image, backend, qp, compress_level, tuning)?;
    let payload = build_modified_hevc(&encoded.annexb)?;

    // Container header (pre-padding dimensions) + payload.
    let header = BpgHeader {
        pixel_format: pixel_format_of(encoded.chroma_format),
        alpha1_flag: false,
        bit_depth_minus_8: encoded.bit_depth - 8,
        color_space: color_space_code_of(encoded.color_space),
        alpha2_flag: false,
        limited_range: encoded.limited_range,
        animation_flag: false,
        width: encoded.display_width,
        height: encoded.display_height,
    };

    let mut out = Vec::with_capacity(16 + payload.len());
    header.write(&mut out, 0, None);
    out.extend_from_slice(&payload);
    Ok(out)
}

fn pixel_format_of(cf: ChromaFormat) -> PixelFormat {
    // M1 always uses JPEG chroma siting (c_h_phase == 1), so the *_VIDEO
    // formats are never produced. TODO(extension): MPEG2 siting.
    match cf {
        ChromaFormat::Gray => PixelFormat::Gray,
        ChromaFormat::Yuv420 => PixelFormat::Yuv420,
        ChromaFormat::Yuv422 => PixelFormat::Yuv422,
        ChromaFormat::Yuv444 => PixelFormat::Yuv444,
    }
}

fn color_space_code_of(cs: ColorSpace) -> ColorSpaceCode {
    match cs {
        ColorSpace::YCbCr => ColorSpaceCode::YCbCr,
        ColorSpace::Rgb => ColorSpaceCode::Rgb,
        ColorSpace::YCgCo => ColorSpaceCode::YCgCo,
        ColorSpace::YCbCrBt709 => ColorSpaceCode::YCbCrBt709,
        ColorSpace::YCbCrBt2020 => ColorSpaceCode::YCbCrBt2020,
    }
}
