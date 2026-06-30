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
    aq_mode: crate::AqMode,
    aq_strength: f32,
    aq_clamp: u8,
    two_pass_gate: bool,
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
            aq_mode: crate::AqMode::Off,
            aq_strength: 0.35,
            aq_clamp: 2,
            two_pass_gate: true,
            last_stats: Mutex::new(None),
            last_recon: Mutex::new(None),
        }
    }

    /// Enable per-CU adaptive quantization on production tiers (off by default;
    /// Fast/Slow stay uniform-QP unless explicitly opted in).
    pub fn with_adaptive_qp(mut self, adaptive_qp: bool) -> Self {
        self.adaptive_qp = adaptive_qp;
        self
    }

    /// Select the adaptive-quantization strategy, strength, and clamp (the
    /// `--aq` flag). `AqMode::Off` (the default) keeps uniform QP.
    pub fn with_aq(mut self, mode: crate::AqMode, strength: f32, clamp: u8) -> Self {
        self.aq_mode = mode;
        self.aq_strength = strength;
        self.aq_clamp = clamp;
        self
    }

    /// Toggle the two-pass AQ candidate-compare gate (default on). When on,
    /// [`crate::AqMode::TwoPassMeasured`] keeps its AQ candidate only if it is a
    /// perceptual rate-distortion win over the uniform pass. Turning it off
    /// always applies the AQ candidate (for A/B measurement).
    pub fn with_two_pass_gate(mut self, enabled: bool) -> Self {
        self.two_pass_gate = enabled;
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
            aq_mode: self.aq_mode,
            aq_strength: self.aq_strength,
            aq_clamp: self.aq_clamp,
            two_pass_gate: self.two_pass_gate,
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
            // Per-depth CU/TU evaluation work, normalized per CTU, in x265's
            // depth order (64/32/16/8 for CU) so it lines up with x265's
            // DETAILED_CU_STATS "Intra RDO calls per depth". TU shown 32/16/8/4.
            let ctus = stats.ctu_count.max(1) as f64;
            let cu_at = |l: usize| stats.cu_trials_by_log2[l] as f64 / ctus;
            let cusplit_at = |l: usize| stats.cu_splits_taken_by_log2[l] as f64 / ctus;
            let cuterm_at = |l: usize| stats.cu_early_term_by_log2[l] as f64 / ctus;
            eprintln!(
                "  cu_trials/ctu by depth (64/32/16/8): {:.2} {:.2} {:.2} {:.2}",
                cu_at(6),
                cu_at(5),
                cu_at(4),
                cu_at(3),
            );
            eprintln!(
                "  cu_splits_taken/ctu  (64/32/16/8): {:.2} {:.2} {:.2} {:.2}   early_term/ctu: {:.2} {:.2} {:.2} {:.2}",
                cusplit_at(6),
                cusplit_at(5),
                cusplit_at(4),
                cusplit_at(3),
                cuterm_at(6),
                cuterm_at(5),
                cuterm_at(4),
                cuterm_at(3),
            );
            let tuleaf_at = |l: usize| stats.tu_leaf_by_log2[l] as f64 / ctus;
            let tusplit_at = |l: usize| stats.tu_split_by_log2[l] as f64 / ctus;
            eprintln!(
                "  tu_leaf/ctu by depth (32/16/8/4): {:.2} {:.2} {:.2} {:.2}   tu_split/ctu: {:.2} {:.2} {:.2} {:.2}",
                tuleaf_at(5),
                tuleaf_at(4),
                tuleaf_at(3),
                tuleaf_at(2),
                tusplit_at(5),
                tusplit_at(4),
                tusplit_at(3),
                tusplit_at(2),
            );
            #[cfg(feature = "overlay-probe")]
            {
                let (calls, iters, chroma) = crate::encoder::overlay_probe_counts();
                eprintln!(
                    "  overlay_probe: sample_calls={calls} patch_iters={iters} avg_patches={:.2} chroma_calls={chroma} ({:.1}%)",
                    iters as f64 / calls.max(1) as f64,
                    100.0 * chroma as f64 / calls.max(1) as f64,
                );
            }
            let dct_hist = crate::primitives::wide::dct_size_histogram();
            let dct_total: u64 = dct_hist.iter().sum();
            if dct_total > 0 {
                eprintln!(
                    "  dct_size_histogram: 4x4={} ({:.1}%)  8x8={} ({:.1}%)  16x16={} ({:.1}%)  32x32={} ({:.1}%)",
                    dct_hist[0],
                    100.0 * dct_hist[0] as f64 / dct_total as f64,
                    dct_hist[1],
                    100.0 * dct_hist[1] as f64 / dct_total as f64,
                    dct_hist[2],
                    100.0 * dct_hist[2] as f64 / dct_total as f64,
                    dct_hist[3],
                    100.0 * dct_hist[3] as f64 / dct_total as f64,
                );
            }
        }
        Ok(annexb)
    }
}
