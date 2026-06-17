//! BPG decode via the pure-Rust `bpg-decode` crate (the bpg-rs workspace).
//!
//! This replaces the previous C `libbpgdec` FFI path: the viewer no longer
//! needs `libbpgdec.so` at runtime, and it inherits bpg-rs's large-image
//! support (e.g. 18432x9216 / ~170 MP heightmaps that crash the C tooling).
//! `bpg-decode` performs the YCbCr->RGB conversion (BT.601/709/2020) internally
//! and returns display-ready RGBA8.

use anyhow::{anyhow, Result};
use bpg_decode::{DecoderConfig, PixelLayout};

/// A decoded BPG image, as RGBA8 (`width * height * 4` bytes in `data`).
pub struct DecodedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Decode a BPG file at `input_path` to RGBA8.
pub fn decode_file(input_path: &str) -> Result<DecodedImage> {
    let data = std::fs::read(input_path)?;
    decode_memory(&data)
}

/// Decode an in-memory BPG byte stream to RGBA8.
pub fn decode_memory(input_data: &[u8]) -> Result<DecodedImage> {
    let out = DecoderConfig::new()
        .decode(input_data, PixelLayout::Rgba8)
        .map_err(|e| anyhow!("{e}"))?;
    Ok(DecodedImage {
        data: out.data,
        width: out.width,
        height: out.height,
    })
}
