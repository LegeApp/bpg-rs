//! `bpg-tools` — command-line front-end for the bpg-rs encoder.
//!
//! M1 supports `encode` and still-image `decode` subcommands. See
//! `bpg-rs/PLAN.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use bpg_decode::{DecoderConfig, PixelLayout};
use bpg_encode::{encode_still_image, EncoderTuning};
use bpg_image::{ColorSpace, Image};
use bpg_x265::X265Encoder;

#[derive(Parser)]
#[command(name = "bpg-tools", about = "Rust BPG encoder (x265 backend)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a PNG image to BPG.
    Encode(EncodeArgs),
    /// Decode a still-image BPG to PNG.
    Decode(DecodeArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    X265,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "422")]
    Yuv422,
    #[value(name = "444")]
    Yuv444,
}

#[derive(Clone, Copy, ValueEnum)]
enum CsArg {
    Ycbcr,
    Bt709,
    Bt2020,
}

#[derive(Clone, Copy, ValueEnum)]
enum DecodeFormat {
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

#[derive(clap::Args)]
struct EncodeArgs {
    /// Input PNG file.
    input: PathBuf,

    /// Output BPG file.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// HEVC backend.
    #[arg(long, value_enum, default_value_t = Backend::X265)]
    backend: Backend,

    /// Quantizer parameter (0-51).
    #[arg(short = 'q', long, default_value_t = 28)]
    qp: u8,

    /// Output bit depth (8, 10, or 12). 16-bit PNG input is required to
    /// benefit from 10/12-bit output; 8-bit input is upscaled.
    #[arg(short = 'b', long = "bit-depth", default_value_t = 8)]
    bit_depth: u8,

    /// Chroma format.
    #[arg(short = 'f', long, value_enum, default_value_t = Format::Yuv420)]
    format: Format,

    /// Compression level / x265 preset (1 = fast .. 9 = slowest).
    #[arg(short = 'm', long = "compress-level", default_value_t = 8)]
    compress_level: u8,

    /// Color space.
    #[arg(long = "color-space", value_enum, default_value_t = CsArg::Ycbcr)]
    color_space: CsArg,

    /// Use limited (TV) range instead of full range.
    #[arg(long)]
    limited_range: bool,
}

#[derive(clap::Args)]
struct DecodeArgs {
    /// Input BPG file.
    input: PathBuf,

    /// Output PNG file.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Decoded pixel layout before PNG encoding.
    #[arg(long = "format", value_enum, default_value_t = DecodeFormat::Rgba)]
    format: DecodeFormat,
}

fn run_encode(args: &EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let Backend::X265 = args.backend;

    if ![8, 10, 12].contains(&args.bit_depth) {
        return Err(format!("--bit-depth must be 8, 10, or 12 (got {})", args.bit_depth).into());
    }

    // Load PNG, drop alpha (M1 ignores alpha).
    let dyn_img = image::open(&args.input)?;

    let color_space = match args.color_space {
        CsArg::Ycbcr => ColorSpace::YCbCr,
        CsArg::Bt709 => ColorSpace::YCbCrBt709,
        CsArg::Bt2020 => ColorSpace::YCbCrBt2020,
    };

    // 16-bit PNGs are read as Rgb16 to preserve full input precision; all
    // other inputs go through the 8-bit path (and may be upscaled to a
    // higher output bit_depth during color conversion).
    let mut image = match dyn_img {
        image::DynamicImage::ImageRgb16(_) | image::DynamicImage::ImageRgba16(_) => {
            let rgb16 = dyn_img.to_rgb16();
            Image::from_rgb16(&rgb16, color_space, args.limited_range, args.bit_depth)
        }
        _ => {
            let rgb = dyn_img.to_rgb8();
            Image::from_rgb8(&rgb, color_space, args.limited_range, args.bit_depth)
        }
    };
    match args.format {
        Format::Yuv420 => image.subsample_to_420(1),
        Format::Yuv422 => image.subsample_to_422(1),
        Format::Yuv444 => {}
    }

    let backend = X265Encoder::new();
    // M1 CLI uses the neutral tune (x265 --tune ssim, no overrides), which
    // reproduces the byte-for-byte-identical-to-C-reference output. The
    // tuning system (bpg-tune) will supply a richer EncoderTuning here later.
    let bpg = encode_still_image(
        image,
        &backend,
        args.qp,
        args.compress_level,
        EncoderTuning::neutral(),
    )?;

    std::fs::write(&args.output, &bpg)?;
    eprintln!("wrote {} ({} bytes)", args.output.display(), bpg.len());
    Ok(())
}

fn run_decode(args: &DecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(&args.input)?;
    let layout = match args.format {
        DecodeFormat::Rgb => PixelLayout::Rgb8,
        DecodeFormat::Rgba => PixelLayout::Rgba8,
        DecodeFormat::Bgr => PixelLayout::Bgr8,
        DecodeFormat::Bgra => PixelLayout::Bgra8,
    };
    let decoded = DecoderConfig::new().decode(&data, layout)?;

    match decoded.layout {
        PixelLayout::Rgb8 => {
            let img = image::RgbImage::from_raw(decoded.width, decoded.height, decoded.data)
                .ok_or("decoded RGB buffer has invalid length")?;
            img.save(&args.output)?;
        }
        PixelLayout::Rgba8 => {
            let img = image::RgbaImage::from_raw(decoded.width, decoded.height, decoded.data)
                .ok_or("decoded RGBA buffer has invalid length")?;
            img.save(&args.output)?;
        }
        PixelLayout::Bgr8 => {
            let mut data = decoded.data;
            for px in data.chunks_exact_mut(3) {
                px.swap(0, 2);
            }
            let img = image::RgbImage::from_raw(decoded.width, decoded.height, data)
                .ok_or("decoded BGR buffer has invalid length")?;
            img.save(&args.output)?;
        }
        PixelLayout::Bgra8 => {
            let mut data = decoded.data;
            for px in data.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            let img = image::RgbaImage::from_raw(decoded.width, decoded.height, data)
                .ok_or("decoded BGRA buffer has invalid length")?;
            img.save(&args.output)?;
        }
    }

    eprintln!(
        "wrote {} ({}x{})",
        args.output.display(),
        decoded.width,
        decoded.height
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Encode(args) => run_encode(args),
        Command::Decode(args) => run_decode(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
