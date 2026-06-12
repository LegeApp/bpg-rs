//! Rust-native still-picture HEVC intra encoder (Phase 3 syntax skeleton).
//!
//! This crate is a line-by-line port of the *syntax* layer of x265's still
//! (single-frame, all-intra) encode path: the CABAC arithmetic engine
//! (`cabac`), the NAL/RBSP writer (`nal`), the VPS/SPS/PPS parameter-set
//! writers (`params`), and the slice segment header writer (`slice`). Motion,
//! lookahead, rate control, B/P frames, and the DPB are explicitly out of
//! scope (see `bpg-rs/FULL_RUST_CODEC_PLAN.md` Phase 3/5).
//!
//! The pixel-domain encoder (CTU/CU/TU recursion, intra prediction,
//! transform/quant, residual CABAC syntax) is Phase 5 and not implemented
//! here; [`StillHevcEncoder`] currently only assembles parameter-set NALs and
//! a minimal slice header.

pub mod cabac;
pub mod nal;
pub mod params;
pub mod slice;

pub use bpg_image::ChromaFormat;

/// How much effort to spend searching for a good encode (placeholder for the
/// Phase 5 mode-decision search; unused by the Phase 3 skeleton).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Effort {
    Fast,
    #[default]
    Balanced,
    Best,
}

/// Sample Adaptive Offset mode (placeholder; Phase 3 emits `sao_enabled_flag
/// = 0` regardless).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaoMode {
    #[default]
    Off,
    On,
}

/// In-loop deblocking filter mode (placeholder; Phase 3 emits
/// `slice_deblocking_filter_disabled_flag = 1` regardless).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblockMode {
    #[default]
    Off,
    On,
}

/// Still-picture HEVC encoder configuration.
#[derive(Debug, Clone)]
pub struct StillHevcConfig {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
    pub qp: u8,
    pub effort: Effort,
    pub sao: SaoMode,
    pub deblock: DeblockMode,
}

/// Rust-native still-picture HEVC intra encoder.
///
/// The Phase 3 skeleton only implements [`StillHevcEncoder::syntax_skeleton`],
/// which emits VPS+SPS+PPS+slice-segment-header NALs with no slice data (CTU
/// coding is Phase 5).
pub struct StillHevcEncoder;

impl StillHevcEncoder {
    /// Emit an Annex-B byte stream containing VPS, SPS, PPS, and an IDR
    /// slice NAL whose header is complete but whose slice data is empty.
    ///
    /// This is the Phase 3 acceptance artifact: parameter sets and the slice
    /// header are syntactically valid and round-trip through
    /// `bpg-hevc-decode`'s parsers, but the slice contains no coding-tree
    /// data (Phase 5).
    pub fn syntax_skeleton(config: &StillHevcConfig) -> Vec<u8> {
        let mut out = Vec::new();
        nal::write_annexb_nal(&mut out, nal::NalType::Vps, &params::write_vps());
        nal::write_annexb_nal(&mut out, nal::NalType::Sps, &params::write_sps(config));
        nal::write_annexb_nal(&mut out, nal::NalType::Pps, &params::write_pps());
        nal::write_annexb_nal(
            &mut out,
            nal::NalType::IdrWRadl,
            &slice::write_slice_segment_header(config),
        );
        out
    }
}
