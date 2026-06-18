//! `bpg-tools` — command-line front-end for the bpg-rs encoder.
//!
//! M1 supports `encode` and still-image `decode` subcommands. See
//! `bpg-rs/PLAN.md`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::{fs::OpenOptions, io::Write};

use clap::{Parser, Subcommand, ValueEnum};

use bpg_decode::{DecoderConfig, PixelLayout};
use bpg_encode::{encode_still_image, EncoderTuning, HevcEncoder};
use bpg_image::{ChromaFormat, ColorSpace, Image};
use still265::backend::RustStillHevcEncoder;
use still265::{DeblockMode, Effort, SaoMode};

#[derive(Parser)]
#[command(name = "bpg-tools", about = "Rust BPG encoder")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a PNG or JPEG image to BPG.
    Encode(EncodeArgs),
    /// Decode a still-image BPG to PNG.
    Decode(DecodeArgs),
}

/// CLI mirror of `still265::Effort` (HandBrake/x265-style ladder).
#[derive(Clone, Copy, Debug, ValueEnum)]
enum EffortArg {
    Fastest,
    Fast,
    Balanced,
    Good,
    Best,
    Placebo,
    Reference,
}

impl From<EffortArg> for Effort {
    fn from(e: EffortArg) -> Self {
        match e {
            EffortArg::Fastest => Effort::Fastest,
            EffortArg::Fast => Effort::Fast,
            EffortArg::Balanced => Effort::Balanced,
            EffortArg::Good => Effort::Good,
            EffortArg::Best => Effort::Best,
            EffortArg::Placebo => Effort::Placebo,
            EffortArg::Reference => Effort::Reference,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    #[value(name = "gray")]
    Gray,
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
    Rgb,
    Ycgco,
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
    /// Input image file (PNG or JPEG).
    input: PathBuf,

    /// Output BPG file.
    #[arg(short = 'o', long = "output")]
    output: PathBuf,

    /// Quantizer parameter (0-51).
    #[arg(short = 'q', long, default_value_t = 28)]
    qp: u8,

    /// Output bit depth (8, 10, or 12). 16-bit PNG input is required to
    /// benefit from 10/12-bit output; 8-bit input is upscaled. The still265
    /// backend supports all three bit depths and 4:2:0/4:2:2/4:4:4 chroma.
    #[arg(short = 'b', long = "bit-depth", default_value_t = 8)]
    bit_depth: u8,

    /// Chroma format. Omit to auto-detect grayscale input, otherwise default to 4:2:0.
    #[arg(short = 'f', long, value_enum)]
    format: Option<Format>,

    /// Compression level / x265 preset (1 = fast .. 9 = slowest). Ignored
    /// (use `--effort` instead).
    #[arg(short = 'm', long = "compress-level", default_value_t = 8)]
    compress_level: u8,

    /// RD-search effort.
    #[arg(long, value_enum, default_value_t = EffortArg::Balanced)]
    effort: EffortArg,

    /// Print Rust-backend analysis counters after encoding.
    #[arg(long)]
    debug_stats: bool,

    /// Append one machine-readable Rust-backend stats row after encoding.
    #[arg(long = "debug-stats-csv")]
    debug_stats_csv: Option<PathBuf>,

    /// Force-enable Sample Adaptive Offset (on by default for all efforts; the
    /// single-pass replay path makes it ~free). Kept for explicitness.
    #[arg(long, conflicts_with = "no_sao")]
    sao: bool,

    /// Disable Sample Adaptive Offset (SAO is on by default for every effort).
    #[arg(long = "no-sao", conflicts_with = "sao")]
    no_sao: bool,

    /// Disable the in-loop deblocking filter (on by default).
    #[arg(long = "no-deblock")]
    no_deblock: bool,

    /// Color space.
    #[arg(long = "color-space", value_enum, default_value_t = CsArg::Ycbcr)]
    color_space: CsArg,

    /// Use limited (TV) range instead of full range.
    #[arg(long)]
    limited_range: bool,

    /// Target output size in bytes. When set, the encoder searches QP for the
    /// best quality (lowest QP) whose output does not exceed this budget,
    /// overriding `--qp`. Each probe is a full encode (see x265-parity-plan.md
    /// gap #8 — analysis-reuse warm-start across probes is a future refinement).
    #[arg(long = "target-size", conflicts_with = "target_bpp")]
    target_size: Option<u64>,

    /// Target bits per pixel (same QP search as `--target-size`, expressed as
    /// bpp of the source resolution). Overrides `--qp`.
    #[arg(long = "target-bpp")]
    target_bpp: Option<f64>,
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

// ---------------------------------------------------------------------------
// Image I/O
// ---------------------------------------------------------------------------

/// Pixel color type for decoded input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorType {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

/// Raw decoded pixel data from any supported input format.
struct LoadedImage {
    width: u32,
    height: u32,
    color_type: ColorType,
    bit_depth: u8,
    pixels: Vec<u8>,
}

/// Read a PNG file using zune-png. Accepts Gray, GrayAlpha, RGB, RGBA at
/// 8 or 16 bits. Indexed/palette PNGs are expanded automatically.
fn open_png(path: &std::path::Path) -> Result<LoadedImage, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut decoder = zune_png::PngDecoder::new(&data);
    let pixels = decoder
        .decode_raw()
        .map_err(|e| format!("PNG decode error: {e}"))?;

    let (w, h) = decoder.get_dimensions().ok_or("PNG: missing dimensions")?;
    let width = u32::try_from(w).map_err(|_| format!("PNG width {w} exceeds u32"))?;
    let height = u32::try_from(h).map_err(|_| format!("PNG height {h} exceeds u32"))?;

    let bit_depth = match decoder.get_depth().ok_or("PNG: missing bit depth")? {
        zune_core::bit_depth::BitDepth::Eight => 8u8,
        zune_core::bit_depth::BitDepth::Sixteen => 16u8,
        _ => {
            return Err("PNG bit depth is not supported; use 8 or 16-bit PNG".into())
        }
    };

    let color_type = match decoder.get_colorspace().ok_or("PNG: missing colorspace")? {
        zune_core::colorspace::ColorSpace::Luma => ColorType::Gray,
        zune_core::colorspace::ColorSpace::LumaA => ColorType::GrayAlpha,
        zune_core::colorspace::ColorSpace::RGB => ColorType::Rgb,
        zune_core::colorspace::ColorSpace::RGBA => ColorType::Rgba,
        other => {
            return Err(format!(
                "PNG color type {other:?} is not supported; use grayscale, RGB, or RGBA PNG",
            )
            .into())
        }
    };

    Ok(LoadedImage { width, height, color_type, bit_depth, pixels })
}

/// Read a JPEG file using zune-jpeg. Always outputs 8-bit RGB.
fn open_jpeg(path: &std::path::Path) -> Result<LoadedImage, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        &data,
        zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB),
    );
    let pixels = decoder
        .decode()
        .map_err(|e| format!("JPEG decode error: {e}"))?;
    let (w, h) = decoder.dimensions().ok_or("JPEG: missing dimensions")?;
    let width = u32::try_from(w).map_err(|_| format!("JPEG width {w} exceeds u32"))?;
    let height = u32::try_from(h).map_err(|_| format!("JPEG height {h} exceeds u32"))?;

    Ok(LoadedImage { width, height, color_type: ColorType::Rgb, bit_depth: 8, pixels })
}

/// BT.709 luma coefficients (same as the image crate's to_luma8/16).
fn rgb_to_luma8(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 2126 + g as u32 * 7152 + b as u32 * 722 + 5000) / 10000) as u8
}

fn rgb_to_luma16(r: u16, g: u16, b: u16) -> u16 {
    ((r as u64 * 2126 + g as u64 * 7152 + b as u64 * 722 + 5000) / 10000) as u16
}

/// Returns true if all pixels appear achromatic (R==G==B).
fn is_gray_image(img: &LoadedImage) -> bool {
    match (img.color_type, img.bit_depth) {
        (ColorType::Gray, _) | (ColorType::GrayAlpha, _) => true,
        (ColorType::Rgb, 8) => img.pixels.chunks_exact(3).all(|p| p[0] == p[1] && p[1] == p[2]),
        (ColorType::Rgba, 8) => img.pixels.chunks_exact(4).all(|p| p[0] == p[1] && p[1] == p[2]),
        (ColorType::Rgb, _) => img.pixels.chunks_exact(6).all(|p| {
            u16::from_be_bytes([p[0], p[1]]) == u16::from_be_bytes([p[2], p[3]])
                && u16::from_be_bytes([p[2], p[3]]) == u16::from_be_bytes([p[4], p[5]])
        }),
        (ColorType::Rgba, _) => img.pixels.chunks_exact(8).all(|p| {
            u16::from_be_bytes([p[0], p[1]]) == u16::from_be_bytes([p[2], p[3]])
                && u16::from_be_bytes([p[2], p[3]]) == u16::from_be_bytes([p[4], p[5]])
        }),
    }
}

/// Extract gray 8-bit pixels regardless of source color type.
/// 16-bit sources are downscaled by taking the high byte.
fn extract_gray8(img: &LoadedImage) -> Vec<u8> {
    match (img.color_type, img.bit_depth) {
        (ColorType::Gray, 8) => img.pixels.clone(),
        (ColorType::GrayAlpha, 8) => img.pixels.chunks_exact(2).map(|c| c[0]).collect(),
        (ColorType::Gray, _) => img.pixels.chunks_exact(2).map(|c| c[0]).collect(),
        (ColorType::GrayAlpha, _) => img.pixels.chunks_exact(4).map(|c| c[0]).collect(),
        (ColorType::Rgb, 8) => img
            .pixels
            .chunks_exact(3)
            .map(|p| rgb_to_luma8(p[0], p[1], p[2]))
            .collect(),
        (ColorType::Rgba, 8) => img
            .pixels
            .chunks_exact(4)
            .map(|p| rgb_to_luma8(p[0], p[1], p[2]))
            .collect(),
        (ColorType::Rgb, _) => img
            .pixels
            .chunks_exact(6)
            .map(|p| rgb_to_luma8(p[0], p[2], p[4]))
            .collect(),
        (ColorType::Rgba, _) => img
            .pixels
            .chunks_exact(8)
            .map(|p| rgb_to_luma8(p[0], p[2], p[4]))
            .collect(),
    }
}

/// Extract gray 16-bit pixels (native endian) regardless of source color type.
/// 8-bit sources are upscaled.
fn extract_gray16(img: &LoadedImage) -> Vec<u16> {
    let be16 = |a: u8, b: u8| u16::from_be_bytes([a, b]);
    match (img.color_type, img.bit_depth) {
        (ColorType::Gray, 16) => img.pixels.chunks_exact(2).map(|c| be16(c[0], c[1])).collect(),
        (ColorType::GrayAlpha, 16) => img.pixels.chunks_exact(4).map(|c| be16(c[0], c[1])).collect(),
        (ColorType::Rgb, 16) => img
            .pixels
            .chunks_exact(6)
            .map(|p| rgb_to_luma16(be16(p[0], p[1]), be16(p[2], p[3]), be16(p[4], p[5])))
            .collect(),
        (ColorType::Rgba, 16) => img
            .pixels
            .chunks_exact(8)
            .map(|p| rgb_to_luma16(be16(p[0], p[1]), be16(p[2], p[3]), be16(p[4], p[5])))
            .collect(),
        _ => extract_gray8(img)
            .into_iter()
            .map(|v| ((v as u32 * 65535 + 127) / 255) as u16)
            .collect(),
    }
}

/// Extract RGB 8-bit pixels (3 bytes per pixel). 16-bit sources use the high byte.
fn extract_rgb8(img: &LoadedImage) -> Vec<u8> {
    match (img.color_type, img.bit_depth) {
        (ColorType::Rgb, 8) => img.pixels.clone(),
        (ColorType::Rgba, 8) => img.pixels.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
        (ColorType::Gray, 8) => img.pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        (ColorType::GrayAlpha, 8) => img.pixels.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        (ColorType::Rgb, _) => img.pixels.chunks_exact(6).flat_map(|p| [p[0], p[2], p[4]]).collect(),
        (ColorType::Rgba, _) => img.pixels.chunks_exact(8).flat_map(|p| [p[0], p[2], p[4]]).collect(),
        (ColorType::Gray, _) => img.pixels.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        (ColorType::GrayAlpha, _) => img.pixels.chunks_exact(4).flat_map(|c| [c[0], c[0], c[0]]).collect(),
    }
}

/// Extract RGB 16-bit pixels (3 u16 per pixel, native endian). 8-bit sources
/// are upscaled.
fn extract_rgb16(img: &LoadedImage) -> Vec<u16> {
    let be16 = |a: u8, b: u8| u16::from_be_bytes([a, b]);
    match (img.color_type, img.bit_depth) {
        (ColorType::Rgb, 16) => img
            .pixels
            .chunks_exact(6)
            .flat_map(|p| [be16(p[0], p[1]), be16(p[2], p[3]), be16(p[4], p[5])])
            .collect(),
        (ColorType::Rgba, 16) => img
            .pixels
            .chunks_exact(8)
            .flat_map(|p| [be16(p[0], p[1]), be16(p[2], p[3]), be16(p[4], p[5])])
            .collect(),
        (ColorType::Gray, 16) => img
            .pixels
            .chunks_exact(2)
            .flat_map(|c| { let v = be16(c[0], c[1]); [v, v, v] })
            .collect(),
        (ColorType::GrayAlpha, 16) => img
            .pixels
            .chunks_exact(4)
            .flat_map(|c| { let v = be16(c[0], c[1]); [v, v, v] })
            .collect(),
        _ => extract_rgb8(img)
            .into_iter()
            .map(|v| ((v as u32 * 65535 + 127) / 255) as u16)
            .collect(),
    }
}

/// Write raw pixels to a PNG file (8-bit, RGB or RGBA).
fn save_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    color_type: ColorType,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let png_color = match color_type {
        ColorType::Rgb => png::ColorType::Rgb,
        ColorType::Rgba => png::ColorType::Rgba,
        _ => return Err("only RGB/RGBA output is supported for PNG saving".into()),
    };
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png_color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Encode command
// ---------------------------------------------------------------------------

fn run_encode(args: &EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if ![8, 10, 12].contains(&args.bit_depth) {
        return Err(format!("--bit-depth must be 8, 10, or 12 (got {})", args.bit_depth).into());
    }

    // SAO is on by default for every effort (the single-pass replay path makes
    // it ~free); --no-sao opts out.
    let sao = if args.no_sao { SaoMode::Off } else { SaoMode::On };
    let deblock = if args.no_deblock {
        DeblockMode::Off
    } else {
        DeblockMode::On
    };
    let backend = RustStillHevcEncoder::new(args.effort.into())
        .with_debug_stats(args.debug_stats)
        .with_sao(sao)
        .with_deblock(deblock);
    let caps = backend.caps();
    if !caps.supports_bit_depth(args.bit_depth) {
        return Err(format!(
            "--bit-depth {} is not supported by the still265 backend (supported: {:?})",
            args.bit_depth, caps.bit_depths
        )
        .into());
    }

    let ext = args.input.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    let img = match ext.as_str() {
        "jpg" | "jpeg" => open_jpeg(&args.input)?,
        _ => open_png(&args.input)?,
    };
    let color_space = match args.color_space {
        CsArg::Ycbcr => ColorSpace::YCbCr,
        CsArg::Rgb => ColorSpace::Rgb,
        CsArg::Ycgco => ColorSpace::YCgCo,
        CsArg::Bt709 => ColorSpace::YCbCrBt709,
        CsArg::Bt2020 => ColorSpace::YCbCrBt2020,
    };
    let auto_gray = args.format.is_none() && is_gray_image(&img);
    let requested_format = args.format.unwrap_or(if auto_gray {
        Format::Gray
    } else if matches!(color_space, ColorSpace::Rgb | ColorSpace::YCgCo) {
        Format::Yuv444
    } else {
        Format::Yuv420
    });
    if requested_format == Format::Gray
        && !matches!(
            color_space,
            ColorSpace::YCbCr | ColorSpace::YCbCrBt709 | ColorSpace::YCbCrBt2020
        )
    {
        return Err("gray images must use a YCbCr-family color space".into());
    }
    if matches!(color_space, ColorSpace::Rgb | ColorSpace::YCgCo)
        && requested_format != Format::Yuv444
    {
        return Err("--color-space rgb/ycgco currently requires --format 444".into());
    }

    let requested_chroma = match requested_format {
        Format::Gray => ChromaFormat::Gray,
        Format::Yuv420 => ChromaFormat::Yuv420,
        Format::Yuv422 => ChromaFormat::Yuv422,
        Format::Yuv444 => ChromaFormat::Yuv444,
    };
    if !caps.supports_chroma_format(requested_chroma) {
        return Err(format!(
            "--format {} is not supported by the still265 backend (supported: {:?})",
            match requested_format {
                Format::Gray => "gray",
                Format::Yuv420 => "420",
                Format::Yuv422 => "422",
                Format::Yuv444 => "444",
            },
            caps.chroma_formats
        )
        .into());
    }

    let is_16bit = img.bit_depth == 16;

    let mut image = match requested_format {
        Format::Gray => {
            if is_16bit {
                let gray16 = extract_gray16(&img);
                Image::from_luma16(
                    &gray16,
                    img.width,
                    img.height,
                    color_space,
                    args.limited_range,
                    args.bit_depth,
                )
            } else {
                let gray8 = extract_gray8(&img);
                Image::from_luma8(
                    &gray8,
                    img.width,
                    img.height,
                    color_space,
                    args.limited_range,
                    args.bit_depth,
                )
            }
        }
        _ => {
            if is_16bit {
                let rgb16 = extract_rgb16(&img);
                Image::from_rgb16(
                    &rgb16,
                    img.width,
                    img.height,
                    color_space,
                    args.limited_range,
                    args.bit_depth,
                )
            } else {
                let rgb8 = extract_rgb8(&img);
                Image::from_rgb8(
                    &rgb8,
                    img.width,
                    img.height,
                    color_space,
                    args.limited_range,
                    args.bit_depth,
                )
            }
        }
    };
    match requested_format {
        Format::Gray => {}
        Format::Yuv420 => image.subsample_to_420(1),
        Format::Yuv422 => image.subsample_to_422(1),
        Format::Yuv444 => {}
    }
    let encoded_width = image.width;
    let encoded_height = image.height;
    let encoded_chroma = image.chroma_format;
    let encoded_bit_depth = image.bit_depth;

    // Resolve an optional target budget (bytes). `--target-bpp` is converted
    // against the *source* pixel count (pre-subsampling).
    let target_bytes = match (args.target_size, args.target_bpp) {
        (Some(_), _) => args.target_size,
        (None, Some(bpp)) => {
            let pixels = encoded_width as u64 * encoded_height as u64;
            Some(((bpp * pixels as f64) / 8.0).round().max(1.0) as u64)
        }
        (None, None) => None,
    };

    let t0 = std::time::Instant::now();
    let (bpg, chosen_qp) = if let Some(target) = target_bytes {
        encode_to_target(&image, &backend, args, target)?
    } else {
        let bpg = encode_still_image(
            image,
            &backend,
            args.qp,
            args.compress_level,
            EncoderTuning::neutral(),
        )?;
        (bpg, args.qp)
    };
    let elapsed = t0.elapsed();

    if let Some(path) = &args.debug_stats_csv {
        if let Some(last) = backend.last_encode_stats() {
            append_debug_stats_csv(
                path,
                args,
                encoded_width,
                encoded_height,
                encoded_chroma,
                encoded_bit_depth,
                bpg.len(),
                elapsed.as_secs_f64(),
                &last,
            )?;
        }
    }

    std::fs::write(&args.output, &bpg)?;
    if let Some(target) = target_bytes {
        eprintln!(
            "wrote {} ({} bytes, effort {:?}, qp {} [target {} B], {:.2}s)",
            args.output.display(),
            bpg.len(),
            args.effort,
            chosen_qp,
            target,
            elapsed.as_secs_f64()
        );
    } else {
        eprintln!(
            "wrote {} ({} bytes, effort {:?}, {:.2}s)",
            args.output.display(),
            bpg.len(),
            args.effort,
            elapsed.as_secs_f64()
        );
    }
    Ok(())
}

/// QP search for `--target-size`/`--target-bpp`: find the lowest QP (best
/// quality) whose encoded output does not exceed `target_bytes`. Output size is
/// monotone-decreasing in QP, so the predicate "size <= target" is monotone and
/// a binary search over QP 1..=51 finds the boundary. Encoded payloads are
/// memoized so the winning QP is not re-encoded. Each probe is a full encode;
/// analysis-reuse warm-start across probes is a documented future refinement
/// (x265-parity-plan.md gap #8). Returns the chosen `(bpg, qp)`.
fn encode_to_target(
    image: &Image,
    backend: &RustStillHevcEncoder,
    args: &EncodeArgs,
    target_bytes: u64,
) -> Result<(Vec<u8>, u8), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    let mut cache: HashMap<u8, Vec<u8>> = HashMap::new();
    let mut encode_at = |qp: u8| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if let Some(b) = cache.get(&qp) {
            return Ok(b.clone());
        }
        let bpg = encode_still_image(
            image.clone(),
            backend,
            qp,
            args.compress_level,
            EncoderTuning::neutral(),
        )?;
        cache.insert(qp, bpg.clone());
        Ok(bpg)
    };

    let (mut lo, mut hi) = (1u8, 51u8);
    let mut ans: Option<u8> = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let size = encode_at(mid)?.len() as u64;
        eprintln!("  target-search: qp {mid} -> {size} B (budget {target_bytes} B)");
        if size <= target_bytes {
            ans = Some(mid);
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    // If even qp 51 exceeds the budget, fall back to the smallest output (qp 51).
    let qp = ans.unwrap_or(51);
    let bpg = encode_at(qp)?;
    Ok((bpg, qp))
}

// ---------------------------------------------------------------------------
// Decode command
// ---------------------------------------------------------------------------

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
            save_png(
                &args.output,
                decoded.width,
                decoded.height,
                ColorType::Rgb,
                &decoded.data,
            )?;
        }
        PixelLayout::Rgba8 => {
            save_png(
                &args.output,
                decoded.width,
                decoded.height,
                ColorType::Rgba,
                &decoded.data,
            )?;
        }
        PixelLayout::Bgr8 => {
            let mut data = decoded.data;
            for px in data.chunks_exact_mut(3) {
                px.swap(0, 2);
            }
            save_png(
                &args.output,
                decoded.width,
                decoded.height,
                ColorType::Rgb,
                &data,
            )?;
        }
        PixelLayout::Bgra8 => {
            let mut data = decoded.data;
            for px in data.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            save_png(
                &args.output,
                decoded.width,
                decoded.height,
                ColorType::Rgba,
                &data,
            )?;
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

// ---------------------------------------------------------------------------
// CSV stats helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn append_debug_stats_csv(
    path: &PathBuf,
    args: &EncodeArgs,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bit_depth: u8,
    bpg_bytes: usize,
    encode_s: f64,
    last: &still265::backend::LastEncodeStats,
) -> Result<(), Box<dyn std::error::Error>> {
    let write_header = match std::fs::metadata(path) {
        Ok(m) => m.len() == 0,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => return Err(e.into()),
    };
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    if write_header {
        writeln!(
            file,
            "input,output,effort,qp,format,bit_depth,width,height,pixels,bpg_bytes,annexb_bytes,bpp,bytes_per_mp,encode_s,ctu_count,cu_trials,cu_early_terminations,cu_split_bound_aborts,cu_force_leaf,tu_split_early_terminations,rmd_prunes,luma_candidate_expansions,chroma_candidate_expansions,final_coded_blocks,trial_coded_blocks,trial_final_ratio,final_rdoq_blocks,trial_rdoq_blocks,rdoq_trial_final_ratio,full_rd_close_calls,luma_close_call_escalations,luma_rough_predictions,chroma_rough_predictions,code_block_calls,code_block_per_final,forward_transforms,inverse_transforms,residual_bit_estimates,residual_estimates_per_final,cache_builds,cache_fast_hits,cache_fallbacks,frame_snapshots,frame_restores,map_snapshots,map_restores,bytes_snapshotted,angular_exclusions,policy_angular_forced,policy_angular_guarded,policy_early_term_suppressed,region_class_counts,luma_winner_rank_counts,chroma_winner_rank_counts,cu_leaf_wins_by_region,cu_split_wins_by_region,tu_leaf_wins_by_region,tu_split_wins_by_region"
        )?;
    }

    let stats = &last.stats;
    let pixels = u64::from(width) * u64::from(height);
    let pixels_f = pixels.max(1) as f64;
    let final_blocks_f = stats.final_coded_blocks.max(1) as f64;
    let final_rdoq_f = stats.final_rdoq_blocks.max(1) as f64;
    let bpp = (bpg_bytes as f64 * 8.0) / pixels_f;
    let bytes_per_mp = bpg_bytes as f64 * 1_000_000.0 / pixels_f;

    let fields = vec![
        csv_field(&args.input.display().to_string()),
        csv_field(&args.output.display().to_string()),
        effort_name(args.effort).to_string(),
        args.qp.to_string(),
        chroma_name(chroma).to_string(),
        bit_depth.to_string(),
        width.to_string(),
        height.to_string(),
        pixels.to_string(),
        bpg_bytes.to_string(),
        last.annexb_bytes.to_string(),
        format!("{bpp:.6}"),
        format!("{bytes_per_mp:.3}"),
        format!("{encode_s:.6}"),
        stats.ctu_count.to_string(),
        stats.cu_trials.to_string(),
        stats.cu_early_terminations.to_string(),
        stats.cu_split_bound_aborts.to_string(),
        stats.cu_force_leaf.to_string(),
        stats.tu_split_early_terminations.to_string(),
        stats.rmd_prunes.to_string(),
        stats.luma_candidate_expansions.to_string(),
        stats.chroma_candidate_expansions.to_string(),
        stats.final_coded_blocks.to_string(),
        stats.trial_coded_blocks.to_string(),
        format!("{:.6}", stats.trial_coded_blocks as f64 / final_blocks_f),
        stats.final_rdoq_blocks.to_string(),
        stats.trial_rdoq_blocks.to_string(),
        format!("{:.6}", stats.trial_rdoq_blocks as f64 / final_rdoq_f),
        stats.full_rd_close_calls.to_string(),
        stats.luma_close_call_escalations.to_string(),
        stats.luma_rough_predictions.to_string(),
        stats.chroma_rough_predictions.to_string(),
        stats.code_block_calls.to_string(),
        format!("{:.6}", stats.code_block_calls as f64 / final_blocks_f),
        stats.forward_transforms.to_string(),
        stats.inverse_transforms.to_string(),
        stats.residual_bit_estimates.to_string(),
        format!(
            "{:.6}",
            stats.residual_bit_estimates as f64 / final_blocks_f
        ),
        stats.cache_builds.to_string(),
        stats.cache_fast_hits.to_string(),
        stats.cache_fallbacks.to_string(),
        stats.frame_snapshots.to_string(),
        stats.frame_restores.to_string(),
        stats.map_snapshots.to_string(),
        stats.map_restores.to_string(),
        stats.bytes_snapshotted.to_string(),
        stats.angular_exclusions.to_string(),
        stats.policy_angular_forced.to_string(),
        stats.policy_angular_guarded.to_string(),
        stats.policy_early_term_suppressed.to_string(),
        csv_field(&join_u64s(&stats.region_class_counts)),
        csv_field(&join_u64s(&stats.luma_winner_rank_counts)),
        csv_field(&join_u64s(&stats.chroma_winner_rank_counts)),
        csv_field(&join_u64s(&stats.cu_leaf_wins_by_region)),
        csv_field(&join_u64s(&stats.cu_split_wins_by_region)),
        csv_field(&join_u64s(&stats.tu_leaf_wins_by_region)),
        csv_field(&join_u64s(&stats.tu_split_wins_by_region)),
    ];
    writeln!(file, "{}", fields.join(","))?;
    Ok(())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        let escaped = value.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        value.to_string()
    }
}

fn join_u64s(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn effort_name(effort: EffortArg) -> &'static str {
    match effort {
        EffortArg::Fastest => "fastest",
        EffortArg::Fast => "fast",
        EffortArg::Balanced => "balanced",
        EffortArg::Good => "good",
        EffortArg::Best => "best",
        EffortArg::Placebo => "placebo",
        EffortArg::Reference => "reference",
    }
}

fn chroma_name(chroma: ChromaFormat) -> &'static str {
    match chroma {
        ChromaFormat::Gray => "gray",
        ChromaFormat::Yuv420 => "420",
        ChromaFormat::Yuv422 => "422",
        ChromaFormat::Yuv444 => "444",
    }
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
