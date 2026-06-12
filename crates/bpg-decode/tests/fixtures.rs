//! Fixture decode tests for the self-contained BPG decoder.
//!
//! These exercise the vendored `bpg-hevc-decode` path end-to-end (BPG container
//! -> Annex-B rebuild -> HEVC decode -> RGBA), with no external dependencies.

use bpg_decode::{DecodeError, DecoderConfig, ImageInfo, PixelLayout};

/// Repo-root `html/` BPG demo fixtures (shared with the upstream libbpg demo).
fn html_fixture(name: &str) -> Vec<u8> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../html/");
    std::fs::read(format!("{path}{name}")).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

/// Crate-local fixtures generated for these tests.
fn local_fixture(name: &str) -> Vec<u8> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
    std::fs::read(format!("{path}{name}")).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

#[test]
fn lena_decodes_to_rgba() {
    let data = html_fixture("lena512color.bpg");

    let info = ImageInfo::from_bytes(&data).expect("probe lena");
    assert_eq!((info.width, info.height), (512, 512));
    assert!(!info.has_alpha);
    assert_eq!(info.bit_depth, 8);

    let out = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgba8)
        .expect("decode lena");
    assert_eq!((out.width, out.height), (512, 512));
    assert_eq!(out.layout, PixelLayout::Rgba8);
    assert_eq!(out.data.len(), 512 * 512 * 4);
    // RGBA output from an opaque image: alpha channel is fully opaque.
    assert!(out.data.chunks_exact(4).all(|px| px[3] == 255));
}

#[test]
fn lena_decodes_to_rgb() {
    let data = html_fixture("lena512color.bpg");
    let out = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgb8)
        .expect("decode lena rgb");
    assert_eq!(out.data.len(), 512 * 512 * 3);
}

#[test]
fn clock_animation_is_unsupported() {
    let data = html_fixture("clock.bpg");
    let err = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgba8)
        .expect_err("clock is animated and must be rejected");
    assert!(
        matches!(err, DecodeError::Unsupported("animation")),
        "expected unsupported animation, got {err:?}"
    );
}

#[test]
fn non_420_chroma_is_unsupported() {
    // The vendored decoder only reconstructs 4:2:0 correctly; 4:2:2 and 4:4:4
    // must be rejected rather than decoded to garbage. The encoder still
    // produces valid 4:2:2/4:4:4 BPG (these fixtures came from stock bpgenc).
    for name in ["solid422.bpg", "solid444.bpg"] {
        let data = local_fixture(name);
        let err = DecoderConfig::new()
            .decode(&data, PixelLayout::Rgb8)
            .expect_err(&format!("{name} must be rejected, not decoded"));
        assert!(matches!(err, DecodeError::Unsupported(_)), "got {err:?}");
    }
}

#[test]
fn non_420_chroma_error_message() {
    let data = local_fixture("solid444.bpg");
    let err = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgb8)
        .expect_err("4:4:4 must be rejected");
    assert!(
        matches!(err, DecodeError::Unsupported(msg) if msg.contains("4:4:4")),
        "expected 4:4:4 unsupported, got {err:?}"
    );
}

#[test]
fn alpha_image_is_unsupported() {
    let data = local_fixture("alpha64.bpg");

    // Probing still reports the alpha flag without decoding.
    let info = ImageInfo::from_bytes(&data).expect("probe alpha64");
    assert!(info.has_alpha);

    let err = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgba8)
        .expect_err("alpha images are not yet supported");
    assert!(
        matches!(err, DecodeError::Unsupported(msg) if msg.contains("alpha")),
        "expected unsupported alpha, got {err:?}"
    );
}
