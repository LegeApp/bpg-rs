//! `bpg-tools` — command-line front-end for the bpg-rs encoder.
//!
//! M1 supports a single `encode` subcommand: 8-bit PNG in, BPG out, x265
//! backend, lossy CQP, YCbCr, 4:2:0 or 4:4:4. See `bpg-rs/PLAN.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use bpg_encode::encode_still_image;
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
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    X265,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "444")]
    Yuv444,
}

#[derive(Clone, Copy, ValueEnum)]
enum CsArg {
    Ycbcr,
    Bt709,
    Bt2020,
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

fn run_encode(args: &EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let Backend::X265 = args.backend;

    // Load PNG, drop alpha (M1 ignores alpha).
    let dyn_img = image::open(&args.input)?;
    let rgb = dyn_img.to_rgb8();

    let color_space = match args.color_space {
        CsArg::Ycbcr => ColorSpace::YCbCr,
        CsArg::Bt709 => ColorSpace::YCbCrBt709,
        CsArg::Bt2020 => ColorSpace::YCbCrBt2020,
    };

    let mut image = Image::from_rgb8(&rgb, color_space, args.limited_range);
    if let Format::Yuv420 = args.format {
        image.subsample_to_420(1);
    }

    let backend = X265Encoder::new();
    let bpg = encode_still_image(image, &backend, args.qp, args.compress_level)?;

    std::fs::write(&args.output, &bpg)?;
    eprintln!(
        "wrote {} ({} bytes)",
        args.output.display(),
        bpg.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Encode(args) => run_encode(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
