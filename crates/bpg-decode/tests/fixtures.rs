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
fn yuv444_decodes() {
    // 4:4:4 is supported (chroma TBs mirror luma). Fixture is a 64x64 solid
    // teal from stock bpgenc -f 444.
    let data = local_fixture("solid444.bpg");
    let info = ImageInfo::from_bytes(&data).expect("probe solid444");
    assert!(matches!(
        info.pixel_format,
        bpg_format::PixelFormat::Yuv444
    ));
    let out = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgb8)
        .expect("4:4:4 must decode");
    assert_eq!((out.width, out.height), (64, 64));
    assert_eq!(out.data.len(), 64 * 64 * 3);
}

#[test]
fn yuv422_decodes() {
    // 4:2:2 is supported (two vertically-stacked chroma TBs per luma TU).
    // Fixture is a 64x64 solid teal from stock bpgenc -f 422.
    let data = local_fixture("solid422.bpg");
    let info = ImageInfo::from_bytes(&data).expect("probe solid422");
    assert!(matches!(
        info.pixel_format,
        bpg_format::PixelFormat::Yuv422
    ));
    let out = DecoderConfig::new()
        .decode(&data, PixelLayout::Rgb8)
        .expect("4:2:2 must decode");
    assert_eq!((out.width, out.height), (64, 64));
    assert_eq!(out.data.len(), 64 * 64 * 3);
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
