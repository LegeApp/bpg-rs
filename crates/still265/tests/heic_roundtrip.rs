use bpg_decode::heic::{decode_heic_to_frame, get_heic_image_info};
use bpg_encode::EncoderTuning;
use bpg_encode::heic::{HeicEncodeOptions, ImageOrientation, encode_heic_still_image};
use bpg_image::{ChromaFormat, ColorSpace, Image};
use still265::backend::RustStillHevcEncoder;
use still265::{DeblockMode, Effort, SaoMode};

fn image(chroma: ChromaFormat, depth: u8) -> Image {
    let (w, h) = (17u32, 19u32);
    if chroma == ChromaFormat::Gray {
        let pixels: Vec<u8> = (0..w * h)
            .map(|i| ((i % w) as u8).wrapping_mul(11).wrapping_add((i / w) as u8))
            .collect();
        return Image::from_luma8(&pixels, w, h, ColorSpace::YCbCrBt709, false, depth);
    }
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let x = (i % w) as u8;
            let y = (i / w) as u8;
            [
                x.wrapping_mul(11),
                y.wrapping_mul(9),
                x.wrapping_add(y).wrapping_mul(7),
            ]
        })
        .collect();
    let mut image = Image::from_rgb8(&pixels, w, h, ColorSpace::YCbCrBt709, false, depth);
    match chroma {
        ChromaFormat::Yuv420 => image.subsample_to_420(1),
        ChromaFormat::Yuv422 => image.subsample_to_422(1),
        ChromaFormat::Yuv444 => {}
        ChromaFormat::Gray => unreachable!(),
    }
    image
}

#[test]
fn heic_round_trips_supported_color_matrix_and_crops_padding() {
    let backend = RustStillHevcEncoder::new(Effort::Fast)
        .with_sao(SaoMode::Off)
        .with_deblock(DeblockMode::On);
    for depth in [8, 10, 12] {
        for chroma in [
            ChromaFormat::Gray,
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            let bytes = encode_heic_still_image(
                image(chroma, depth),
                &backend,
                28,
                1,
                EncoderTuning::default(),
                HeicEncodeOptions::default(),
            )
            .unwrap();
            let info = get_heic_image_info(&bytes).unwrap();
            let frame = decode_heic_to_frame(&bytes).unwrap();
            assert_eq!((info.width, info.height), (17, 19));
            assert_eq!((frame.cropped_width(), frame.cropped_height()), (17, 19));
            assert_eq!(info.bit_depth, depth);
            assert_eq!(
                info.chroma_format,
                match chroma {
                    ChromaFormat::Yuv420 => 1,
                    ChromaFormat::Yuv422 => 2,
                    ChromaFormat::Yuv444 => 3,
                    ChromaFormat::Gray => 0,
                }
            );
        }
    }
}

#[test]
fn heic_reports_thumbnail_and_metadata_items() {
    let backend = RustStillHevcEncoder::new(Effort::Fast);
    let bytes = encode_heic_still_image(
        image(ChromaFormat::Yuv420, 8),
        &backend,
        30,
        1,
        EncoderTuning::default(),
        HeicEncodeOptions {
            thumbnail: Some(image(ChromaFormat::Yuv420, 8)),
            exif: Some(b"II*\0test".to_vec()),
            xmp: Some(b"<x:xmpmeta/>".to_vec()),
            orientation: ImageOrientation::Rotate90,
            ..HeicEncodeOptions::default()
        },
    )
    .unwrap();
    let info = get_heic_image_info(&bytes).unwrap();
    assert!(info.has_thumbnail);
    assert!(info.has_exif);
    assert!(info.has_xmp);
    assert_eq!((info.width, info.height), (19, 17));
    let frame = decode_heic_to_frame(&bytes).unwrap();
    assert_eq!((frame.cropped_width(), frame.cropped_height()), (19, 17));
}
