//! Pure-Rust HEVC still-picture intra decoder for BPG.
//!
//! This crate is vendored from the HEVC decode module of `heic-decoder-rs`
//! (`src/hevc/*`). Only the HEVC bitstream decoder is brought over — the HEIF
//! container parsing, color-conversion entry points beyond `DecodedFrame`, and
//! the `enough`/`whereat` cancellation/location-tracking machinery are left
//! behind. See `FULL_RUST_CODEC_PLAN.md` (Phase 1) for the rationale.
//!
//! The public surface BPG decode uses is [`hevc::decode`] (Annex-B in,
//! [`DecodedFrame`] out) plus the [`HevcError`] error type.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod error;
pub mod heif;
pub mod hevc;

pub use error::HevcError;
pub use hevc::DecodedFrame;
