use bpg_decode::{DecoderConfig, ImageInfo, PixelLayout};
use bpg_encode::{encode_still_image, EncoderTuning};
use bpg_format::{ColorSpaceCode, PixelFormat};
use bpg_image::{ColorSpace, Image};
use still265::backend::RustStillHevcEncoder;
use still265::{DeblockMode, Effort, SaoMode};

fn sample_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let base = ((y * w + x) * 3) as usize;
            pixels[base] = (20 + x * 3 + y * 2) as u8;
            pixels[base + 1] = (16 + x * 2 + y * 3) as u8;
            pixels[base + 2] = (30 + x + y * 2) as u8;
        }
    }
    pixels
}

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    let sse = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>();
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn round_trip_color_space(color_space: ColorSpace, expected_code: ColorSpaceCode) {
    let rgb = sample_rgb(41, 35);
    let image = Image::from_rgb8(&rgb, 41, 35, color_space, false, 8);
    let backend = RustStillHevcEncoder::new(Effort::Balanced)
        .with_deblock(DeblockMode::On)
        .with_sao(SaoMode::Off);

    let bytes = encode_still_image(image, &backend, 0, 5, EncoderTuning::default())
        .expect("encode must accept 4:4:4 RGB-family color spaces");
    let info = ImageInfo::from_bytes(&bytes).expect("container header must parse");
    assert_eq!(info.pixel_format, PixelFormat::Yuv444);
    assert_eq!(info.color_space, expected_code);

    let out = DecoderConfig::new()
        .decode(&bytes, PixelLayout::Rgb8)
        .expect("decoder must convert back to RGB");
    let frame = DecoderConfig::new()
        .decode_request(&bytes)
        .decode_yuv()
        .expect("decoder must expose reconstructed frame");
    let expected_matrix = match expected_code {
        ColorSpaceCode::Rgb => 0,
        ColorSpaceCode::YCgCo => 8,
        _ => unreachable!(),
    };
    assert_eq!(frame.matrix_coeffs, expected_matrix);
    assert!(frame.full_range);
    assert_eq!((out.width, out.height), (41u32, 35u32));
    let psnr = psnr_rgb(&rgb, &out.data);
    assert!(
        psnr > 20.0,
        "{color_space:?} RGB round-trip PSNR too low: {psnr:.2}"
    );
}

#[test]
fn rgb_color_space_bpg_round_trip() {
    round_trip_color_space(ColorSpace::Rgb, ColorSpaceCode::Rgb);
}

#[test]
fn ycgco_color_space_bpg_round_trip() {
    round_trip_color_space(ColorSpace::YCgCo, ColorSpaceCode::YCgCo);
}
