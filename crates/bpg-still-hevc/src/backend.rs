//! [`bpg_encode::HevcEncoder`] implementation backed by this crate's
//! Rust-native still-picture HEVC encoder ([`crate::encoder::encode`]).
//!
//! This is the Rust-backend analogue of `bpg-x265::X265Encoder`: it converts
//! a CTU-padded [`Image`] into [`Source`] slices, builds a
//! [`StillHevcConfig`], and returns the resulting Annex-B stream for
//! `bpg_encode::encode_still_image` to pipe through
//! `bpg_hevc::build_modified_hevc`.

use bpg_encode::{EncodeError, HevcEncodeParams, HevcEncoder};
use bpg_image::{ChromaFormat, Image};

use crate::encoder::{encode_with_stats, Source};
use crate::{DeblockMode, Effort, SaoMode, StillHevcConfig};

/// The Rust-native still-picture HEVC backend.
pub struct RustStillHevcEncoder {
    effort: Effort,
    debug_stats: bool,
}

impl RustStillHevcEncoder {
    pub fn new(effort: Effort) -> Self {
        Self {
            effort,
            debug_stats: false,
        }
    }

    pub fn with_debug_stats(mut self, debug_stats: bool) -> Self {
        self.debug_stats = debug_stats;
        self
    }
}

impl Default for RustStillHevcEncoder {
    fn default() -> Self {
        Self::new(Effort::default())
    }
}

impl HevcEncoder for RustStillHevcEncoder {
    fn encode_still(
        &self,
        image: &Image,
        params: &HevcEncodeParams,
    ) -> Result<Vec<u8>, EncodeError> {
        if !matches!(
            params.chroma_format,
            ChromaFormat::Yuv420 | ChromaFormat::Yuv444
        ) {
            return Err(EncodeError::Unsupported(
                "bpg-still-hevc backend supports only 4:2:0/4:4:4 chroma",
            ));
        }
        if params.bit_depth != 8 && params.bit_depth != 10 {
            return Err(EncodeError::Unsupported(
                "bpg-still-hevc backend supports only 8/10-bit",
            ));
        }

        let config = StillHevcConfig {
            width: params.width,
            height: params.height,
            bit_depth: params.bit_depth,
            chroma: params.chroma_format,
            qp: params.qp,
            effort: self.effort,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
        };

        let src = Source {
            y: &image.planes[0].data,
            cb: &image.planes[1].data,
            cr: &image.planes[2].data,
        };

        let (annexb, _decoded, stats) = encode_with_stats(&config, src);
        if self.debug_stats {
            eprintln!("{stats}");
        }
        Ok(annexb)
    }
}
