//! Minimal HEVC decoder-configuration shim.
//!
//! The full `heic-decoder-rs` parses an `hvcC` box out of a HEIF container into
//! a `HevcDecoderConfig`. BPG has no HEIF container — it carries a modified-HEVC
//! payload that `bpg-hevc` rebuilds into an Annex-B stream — so the only decode
//! path BPG uses is `hevc::decode(annexb)`.
//!
//! This shim keeps just the fields the vendored `hevc` module references
//! (`nal_units`, `length_size_minus_one`) so the length-prefixed
//! `decode_with_config`/`get_info_from_config` helpers still compile, in case a
//! future caller wants the hvcC-style entry point.

use alloc::vec::Vec;

/// HEVC decoder configuration (subset of the HEIF `hvcC` box).
#[derive(Debug, Clone, Default)]
pub struct HevcDecoderConfig {
    /// `length_size_minus_one` from the hvcC box: NAL length-prefix size − 1.
    pub length_size_minus_one: u8,
    /// Parameter-set NAL units (VPS, SPS, PPS, …), each as a raw NAL payload.
    pub nal_units: Vec<Vec<u8>>,
}
