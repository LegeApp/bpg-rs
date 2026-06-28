//! [`bpg_encode::HevcEncoder`] implementation backed by this crate's
//! Rust-native still-picture HEVC encoder ([`crate::encoder::encode`]).
//!
//! It converts a CTU-padded [`Image`] into [`Source`] slices, builds a
//! [`StillHevcConfig`], and returns the resulting Annex-B stream for
//! `bpg_encode::encode_still_image` to pipe through
//! `bpg_hevc::build_modified_hevc`.

use bpg_encode::{EncodeError, HevcBackendCaps, HevcEncodeParams, HevcEncoder};
use bpg_hevc_decode::DecodedFrame;
use bpg_image::{ChromaFormat, Image};

use crate::effort::describe_effort;
use crate::encoder::{EncodeStats, Source, encode_with_stats};
use crate::{DeblockMode, Effort, SaoMode, StillHevcConfig};
use std::sync::Mutex;

/// Counters captured from the most recent [`RustStillHevcEncoder`] encode.
#[derive(Debug, Clone)]
pub struct LastEncodeStats {
    pub annexb_bytes: usize,
    pub stats: EncodeStats,
}

/// The Rust-native still-picture HEVC backend.
pub struct RustStillHevcEncoder {
    effort: Effort,
    debug_stats: bool,
    sao: SaoMode,
    deblock: DeblockMode,
    adaptive_qp: bool,
    last_stats: Mutex<Option<LastEncodeStats>>,
    last_recon: Mutex<Option<DecodedFrame>>,
}

impl RustStillHevcEncoder {
    pub fn new(effort: Effort) -> Self {
        Self {
            effort,
            debug_stats: false,
            sao: SaoMode::Off,
            deblock: DeblockMode::On,
            adaptive_qp: false,
            last_stats: Mutex::new(None),
            last_recon: Mutex::new(None),
        }
    }

    /// Enable per-CU adaptive quantization on the speed tiers (off by default;
    /// the ladder is uniform-QP). No effect on `Best`/reference tiers.
    pub fn with_adaptive_qp(mut self, adaptive_qp: bool) -> Self {
        self.adaptive_qp = adaptive_qp;
        self
    }

    pub fn with_debug_stats(mut self, debug_stats: bool) -> Self {
        self.debug_stats = debug_stats;
        self
    }

    /// Enable the conservative SAO encoder (see `still265::sao`). Off by default.
    pub fn with_sao(mut self, sao: SaoMode) -> Self {
        self.sao = sao;
        self
    }

    /// Toggle the in-loop deblocking filter (on by default).
    pub fn with_deblock(mut self, deblock: DeblockMode) -> Self {
        self.deblock = deblock;
        self
    }

    /// Return the counters captured from the most recent encode through this
    /// backend. The value is primarily used by `bpg-tools --debug-stats-csv`,
    /// where the final BPG container size is only known outside this backend.
    pub fn last_encode_stats(&self) -> Option<LastEncodeStats> {
        self.last_stats
            .lock()
            .expect("still265 last-stats mutex poisoned")
            .clone()
    }

    /// Return the reconstruction planes from the most recent encode.
    pub fn last_reconstruction(&self) -> Option<DecodedFrame> {
        self.last_recon
            .lock()
            .expect("still265 last-recon mutex poisoned")
            .clone()
    }
}

impl Default for RustStillHevcEncoder {
    fn default() -> Self {
        Self::new(Effort::default())
    }
}

/// `still265`'s declared capabilities (see [`HevcBackendCaps`]).
/// Encoder-side lossless/alpha are not yet implemented; the decoder
/// (`bpg-hevc-decode`) can still decode third-party streams that use them.
/// Deblocking is implemented and enabled by default (see
/// [`crate::DeblockMode`]). SAO is implemented as a conservative
/// band-offset/edge-offset encoder (see [`crate::sao`]), off by default
/// ([`RustStillHevcEncoder::with_sao`]).
const CAPS: HevcBackendCaps = HevcBackendCaps {
    bit_depths: &[8, 10, 12],
    chroma_formats: &[
        ChromaFormat::Gray,
        ChromaFormat::Yuv420,
        ChromaFormat::Yuv422,
        ChromaFormat::Yuv444,
    ],
    supports_sao: true,
    supports_deblock: true,
    supports_lossless: false,
    supports_alpha: false,
};

impl HevcEncoder for RustStillHevcEncoder {
    fn caps(&self) -> HevcBackendCaps {
        CAPS
    }

    fn encode_still(
        &self,
        image: &Image,
        params: &HevcEncodeParams,
    ) -> Result<Vec<u8>, EncodeError> {
        if !CAPS.supports_chroma_format(params.chroma_format) {
            return Err(EncodeError::Unsupported(
                "still265 backend supports only 4:2:0/4:4:4 chroma",
            ));
        }
        if !CAPS.supports_bit_depth(params.bit_depth) {
            return Err(EncodeError::Unsupported(
                "still265 backend supports only 8/10/12-bit",
            ));
        }

        let config = StillHevcConfig {
            width: params.width,
            height: params.height,
            bit_depth: params.bit_depth,
            chroma: params.chroma_format,
            qp: params.qp,
            effort: self.effort,
            sao: self.sao,
            deblock: self.deblock,
            adaptive_qp: self.adaptive_qp,
        };

        let src = Source {
            y: &image.planes[0].data,
            cb: image.planes.get(1).map_or(&[][..], |p| p.data.as_slice()),
            cr: image.planes.get(2).map_or(&[][..], |p| p.data.as_slice()),
        };

        let (annexb, decoded, stats) = encode_with_stats(&config, src);
        *self
            .last_stats
            .lock()
            .expect("still265 last-stats mutex poisoned") = Some(LastEncodeStats {
            annexb_bytes: annexb.len(),
            stats: stats.clone(),
        });
        *self
            .last_recon
            .lock()
            .expect("still265 last-recon mutex poisoned") = Some(decoded);
        if self.debug_stats {
            eprintln!("{}", describe_effort(self.effort, params.qp as i32));
            eprintln!("{stats}");
            eprintln!(
                "  primitive_backend: {}",
                crate::primitives::PRIMITIVES.backend
            );
            let dct_hist = crate::primitives::wide::dct_size_histogram();
            let dct_total: u64 = dct_hist.iter().sum();
            if dct_total > 0 {
                eprintln!(
                    "  dct_size_histogram: 4x4={} ({:.1}%)  8x8={} ({:.1}%)  16x16={} ({:.1}%)  32x32={} ({:.1}%)",
                    dct_hist[0], 100.0 * dct_hist[0] as f64 / dct_total as f64,
                    dct_hist[1], 100.0 * dct_hist[1] as f64 / dct_total as f64,
                    dct_hist[2], 100.0 * dct_hist[2] as f64 / dct_total as f64,
                    dct_hist[3], 100.0 * dct_hist[3] as f64 / dct_total as f64,
                );
            }
        }
        Ok(annexb)
    }
}
