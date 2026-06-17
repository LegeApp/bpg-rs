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
    /// Encode a PNG image to BPG.
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
    /// Input PNG file.
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
    let dyn_img = image::open(&args.input)?;
    let color_space = match args.color_space {
        CsArg::Ycbcr => ColorSpace::YCbCr,
        CsArg::Rgb => ColorSpace::Rgb,
        CsArg::Ycgco => ColorSpace::YCgCo,
        CsArg::Bt709 => ColorSpace::YCbCrBt709,
        CsArg::Bt2020 => ColorSpace::YCbCrBt2020,
    };
    let auto_gray = args.format.is_none() && is_grayscale_input(&dyn_img);
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

    // 16-bit PNGs are read as Rgb16 to preserve full input precision; all
    // other inputs go through the 8-bit path (and may be upscaled to a
    // higher output bit_depth during color conversion).
    let mut image = match requested_format {
        Format::Gray => match dyn_img {
            image::DynamicImage::ImageLuma16(_) | image::DynamicImage::ImageLumaA16(_) => {
                let gray16 = dyn_img.to_luma16();
                Image::from_luma16(&gray16, color_space, args.limited_range, args.bit_depth)
            }
            _ => {
                let gray8 = dyn_img.to_luma8();
                Image::from_luma8(&gray8, color_space, args.limited_range, args.bit_depth)
            }
        },
        _ => match dyn_img {
            image::DynamicImage::ImageRgb16(_) | image::DynamicImage::ImageRgba16(_) => {
                let rgb16 = dyn_img.to_rgb16();
                Image::from_rgb16(&rgb16, color_space, args.limited_range, args.bit_depth)
            }
            _ => {
                let rgb = dyn_img.to_rgb8();
                Image::from_rgb8(&rgb, color_space, args.limited_range, args.bit_depth)
            }
        },
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

fn is_grayscale_input(img: &image::DynamicImage) -> bool {
    match img {
        image::DynamicImage::ImageLuma8(_)
        | image::DynamicImage::ImageLumaA8(_)
        | image::DynamicImage::ImageLuma16(_)
        | image::DynamicImage::ImageLumaA16(_) => true,
        image::DynamicImage::ImageRgb16(_) | image::DynamicImage::ImageRgba16(_) => img
            .to_rgb16()
            .pixels()
            .all(|p| p[0] == p[1] && p[1] == p[2]),
        _ => img.to_rgb8().pixels().all(|p| p[0] == p[1] && p[1] == p[2]),
    }
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
