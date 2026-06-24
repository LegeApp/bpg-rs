//! High-resolution encode timing harness comparing the still265 StillSearch
//! core against C `bpgenc`, with quality metrics and the StillSearch work
//! ledger.
//!
//! Generates deterministic high-res PNGs from `test-set` (or encodes sources
//! natively with `--native`), then compares process-level `bpgenc` time with
//! in-process still265 encode timing.
//!
//! ⚠️ EFFORT IS CURRENTLY INERT. The StillSearch v2 core does not yet consult
//! `cfg.effort` — every tier runs the same single search, so passing multiple
//! `--efforts` just produces identical rows. The default is therefore a single
//! tier. The legacy preset ladder (best/slow/fastadaptive) and its README
//! timings belong to the removed rdo2 core; compare those by hand until
//! StillSearch reintroduces effort policies (plan Phase 15).
//!
//! ⚠️ QP-AXIS OFFSET — READ BEFORE INTERPRETING QP-MATCHED NUMBERS. still265's
//! effective quantiser is roughly **2 QP steps COARSER** than x265/bpgenc at the
//! same nominal `-q`/`qp`. At equal nominal QP still265 makes a smaller file and
//! retains fewer/weaker coefficients; rust QP24 ≈ bpgenc QP26 in coefficient
//! count/bytes (measured 4 MP 2026-06-23, `bpg-decision-diff`). So an equal-QP
//! C-vs-rust row is NOT an equal-rate comparison and OVERSTATES the quality gap
//! by most of ~1 dB — always compare at matched BYTES (the real equal-bitrate
//! deficit in textured regions is ~0.6 dB, an RDOQ coefficient-strength gap, not
//! the full equal-QP gap). This fact keeps getting rediscovered; hence this note.

use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use bpg_encode::{encode_still_image, EncoderTuning};
use bpg_image::{ColorSpace, Image};
use clap::{Parser, ValueEnum};
use still265::backend::RustStillHevcEncoder;
use still265::{DeblockMode, Effort, SaoMode};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[derive(Parser)]
#[command(
    name = "bpg-highres-compare",
    about = "Generate high-res synthetic PNGs and compare C bpgenc vs Rust still265 timing"
)]
struct Args {
    /// Source folder containing PNG/JPEG test images.
    #[arg(long, default_value = "test-set")]
    input_dir: PathBuf,

    /// Work/output directory for generated PNGs, BPGs, CSV, and summary.
    #[arg(long, default_value = "target/highres-compare")]
    work_dir: PathBuf,

    /// C bpgenc executable to compare against. Use --skip-c if unavailable.
    #[arg(long, default_value = "bpgenc_native.exe")]
    bpgenc: PathBuf,

    /// Comma-separated output sizes, e.g. 1000x750,2000x1500,4000x3000.
    #[arg(long, default_value = "1000x750")]
    sizes: String,

    /// Number of timing runs per case.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// Maximum number of source images to derive synthetic cases from.
    #[arg(long, default_value_t = 1)]
    max_sources: usize,

    /// Quantizer parameter.
    #[arg(short = 'q', long, default_value_t = 28)]
    qp: u8,

    /// bpgenc compression level.
    #[arg(short = 'm', long = "compress-level", default_value_t = 8)]
    compress_level: u8,

    /// Rust RD-search effort(s). Comma-separated. NOTE: the StillSearch core
    /// currently ignores effort (single behavior), so multiple values just
    /// repeat the same encode; the default is one tier.
    #[arg(long, default_value = "balanced")]
    efforts: String,

    /// Chroma format.
    #[arg(short = 'f', long, default_value_t = FormatArg::Yuv420)]
    format: FormatArg,

    /// Rust still265 SAO mode.
    #[arg(long, default_value_t = SaoArg::On)]
    sao: SaoArg,

    /// Extra argument passed through to C bpgenc. Repeat for multiple args.
    #[arg(long = "c-extra-arg")]
    c_extra_args: Vec<String>,

    /// x265 param override for rebuilt C bpgenc wrappers. Repeat as
    /// `--c-x265-param name=value`.
    #[arg(long = "c-x265-param")]
    c_x265_params: Vec<String>,

    /// Run C bpgenc with true x265 single-thread controls when supported by the
    /// wrapper (`frame-threads=1,pools=none`), plus one-CPU process affinity as
    /// a compatibility fallback for stock bpgenc builds.
    #[arg(long)]
    c_single_thread: bool,

    /// Run the in-process Rust encoder with BPG_ENC_THREADS=1.
    #[arg(long)]
    rust_single_thread: bool,

    /// Output bit depth.
    #[arg(short = 'b', long = "bit-depth", default_value_t = 8)]
    bit_depth: u8,

    /// Disable C bpgenc runs and only time Rust still265.
    #[arg(long)]
    skip_c: bool,

    /// Recreate synthetic PNGs even if they already exist.
    #[arg(long)]
    regenerate: bool,

    /// Encode each source image directly at its native resolution (no
    /// synthetic resampling/grain). Ignores --sizes.
    #[arg(long)]
    native: bool,

    /// Enable per-CU adaptive quantization (default: off = uniform QP).
    #[arg(long)]
    adaptive_qp: bool,

    /// Print still265 debug stats during Rust encodes.
    #[arg(long)]
    debug_stats: bool,
}

fn parse_efforts(s: &str) -> Result<Vec<EffortArg>, String> {
    s.split(',')
        .map(|name| {
            let trimmed = name.trim().to_lowercase();
            match trimmed.as_str() {
                "floor" => Ok(EffortArg::Floor),
                "floorplus" => Ok(EffortArg::FloorPlus),
                "floorplus2" => Ok(EffortArg::FloorPlus2),
                "floorshallow" => Ok(EffortArg::FloorShallow),
                "slow" => Ok(EffortArg::Slow),
                "slowplus" => Ok(EffortArg::SlowPlus),
                "fastest" => Ok(EffortArg::Fastest),
                "fastadaptive" => Ok(EffortArg::FastAdaptive),
                "fast" => Ok(EffortArg::Fast),
                "balanced" => Ok(EffortArg::Balanced),
                "good" => Ok(EffortArg::Good),
                "best" => Ok(EffortArg::Best),
                "placebo" => Ok(EffortArg::Placebo),
                "reference" => Ok(EffortArg::Reference),
                _ => Err(format!("unknown effort '{trimmed}'; valid: floor, floorplus, floorplus2, floorshallow, slow, slowplus, fastest, fastadaptive, fast, balanced, good, best, placebo, reference")),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Size {
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug)]
enum EffortArg {
    Floor,
    FloorPlus,
    FloorPlus2,
    FloorShallow,
    Slow,
    SlowPlus,
    Fastest,
    FastAdaptive,
    Fast,
    Balanced,
    Good,
    Best,
    Placebo,
    Reference,
}

impl EffortArg {
    fn as_str(self) -> &'static str {
        match self {
            EffortArg::Floor => "floor",
            EffortArg::FloorPlus => "floorplus",
            EffortArg::FloorPlus2 => "floorplus2",
            EffortArg::FloorShallow => "floorshallow",
            EffortArg::Slow => "slow",
            EffortArg::SlowPlus => "slowplus",
            EffortArg::Fastest => "fastest",
            EffortArg::FastAdaptive => "fastadaptive",
            EffortArg::Fast => "fast",
            EffortArg::Balanced => "balanced",
            EffortArg::Good => "good",
            EffortArg::Best => "best",
            EffortArg::Placebo => "placebo",
            EffortArg::Reference => "reference",
        }
    }
}

impl From<EffortArg> for Effort {
    fn from(value: EffortArg) -> Self {
        match value {
            EffortArg::Floor => Effort::Floor,
            EffortArg::FloorPlus => Effort::FloorPlus,
            EffortArg::FloorPlus2 => Effort::FloorPlus2,
            EffortArg::FloorShallow => Effort::FloorShallow,
            EffortArg::Slow => Effort::Slow,
            EffortArg::SlowPlus => Effort::SlowPlus,
            EffortArg::Fastest => Effort::Fastest,
            EffortArg::FastAdaptive => Effort::FastAdaptive,
            EffortArg::Fast => Effort::Fast,
            EffortArg::Balanced => Effort::Balanced,
            EffortArg::Good => Effort::Good,
            EffortArg::Best => Effort::Best,
            EffortArg::Placebo => Effort::Placebo,
            EffortArg::Reference => Effort::Reference,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "422")]
    Yuv422,
    #[value(name = "444")]
    Yuv444,
}

impl std::fmt::Display for FormatArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatArg::Yuv420 => write!(f, "420"),
            FormatArg::Yuv422 => write!(f, "422"),
            FormatArg::Yuv444 => write!(f, "444"),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SaoArg {
    On,
    Off,
}

impl std::fmt::Display for SaoArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaoArg::On => write!(f, "on"),
            SaoArg::Off => write!(f, "off"),
        }
    }
}

impl From<SaoArg> for SaoMode {
    fn from(value: SaoArg) -> Self {
        match value {
            SaoArg::On => SaoMode::On,
            SaoArg::Off => SaoMode::Off,
        }
    }
}

impl FormatArg {
    fn bpgenc_arg(self) -> &'static str {
        match self {
            FormatArg::Yuv420 => "420",
            FormatArg::Yuv422 => "422",
            FormatArg::Yuv444 => "444",
        }
    }
    fn chroma_subsampling(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            FormatArg::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
            FormatArg::Yuv422 => (w.div_ceil(2), h),
            FormatArg::Yuv444 => (w, h),
        }
    }
}

struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone)]
struct QualityMetrics {
    psnr_y: f64,
    psnr_cb: f64,
    psnr_cr: f64,
    psnr_rgb: f64,
}

#[derive(Clone)]
struct Row {
    effort: String,
    source: String,
    width: u32,
    height: u32,
    pixels: u64,
    run: usize,
    c_total: Option<Duration>,
    c_bytes: Option<u64>,
    /// Decoded RGB PSNR of the C bpgenc output vs source (same metric as `rust_rgb_psnr`).
    c_rgb_psnr: Option<f64>,
    /// Decoded luma-Y PSNR of the C output vs source Y (comparable to rust `psnr_y`).
    c_y_psnr: Option<f64>,
    /// Decoded RGB PSNR of the still265 output vs source.
    rust_rgb_psnr: Option<f64>,
    rust_total: Duration,
    rust_prepare: Duration,
    rust_encode: Duration,
    rust_bpg_bytes: usize,
    rust_annexb_bytes: usize,
    quality: Option<QualityMetrics>,
    ctu_count: u64,
    cu_trials: u64,
    cu_early_terminations: u64,
    cu_split_bound_aborts: u64,
    cu_force_leaf: u64,
    code_block_calls: u64,
    forward_transforms: u64,
    trial_coded_blocks: u64,
    final_coded_blocks: u64,
    trial_rdoq_blocks: u64,
    final_rdoq_blocks: u64,
    phase_build_us: u64,
    phase_write_us: u64,
    bytes_restored: u64,
    partnxn_attempts: u64,
    partnxn_wins: u64,
    stillsearch_ledger: [u64; 15],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    if ![8, 10, 12].contains(&args.bit_depth) {
        return Err("--bit-depth must be 8, 10, or 12".into());
    }
    let efforts = parse_efforts(&args.efforts)?;
    if efforts.is_empty() {
        return Err("at least one effort required".into());
    }
    let sizes = parse_sizes(&args.sizes)?;

    let repo_root = std::env::current_dir()?;
    let input_dir = resolve(&repo_root, &args.input_dir);
    let work_dir = resolve(&repo_root, &args.work_dir);
    let generated_dir = work_dir.join("generated");
    let rust_dir = work_dir.join("rust");
    let c_dir = work_dir.join("c");
    fs::create_dir_all(&generated_dir)?;
    fs::create_dir_all(&rust_dir)?;
    fs::create_dir_all(&c_dir)?;

    let sources = discover_sources(&input_dir, args.max_sources)?;
    if sources.is_empty() {
        return Err(format!("no PNG/JPEG sources found under {}", input_dir.display()).into());
    }

    let bpgenc = resolve(&repo_root, &args.bpgenc);
    if !args.skip_c && !bpgenc.exists() {
        return Err(format!(
            "C encoder not found at {}; pass --bpgenc or --skip-c",
            bpgenc.display()
        )
        .into());
    }

    let csv_path = work_dir.join("results.csv");
    let mut csv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&csv_path)?;
    writeln!(
        csv,
        "effort,source,width,height,pixels,run,c_total_s,c_bpg_bytes,rust_total_s,rust_prepare_s,rust_encode_s,rust_bpg_bytes,rust_annexb_bytes,psnr_y,psnr_cb,psnr_cr,psnr_rgb,ctu_count,cu_trials,cu_early_terminations,cu_split_bound_aborts,cu_force_leaf,code_block_calls,forward_transforms,trial_coded_blocks,final_coded_blocks,trial_rdoq_blocks,final_rdoq_blocks,phase_build_us,phase_write_us,bytes_restored,partnxn_attempts,partnxn_wins,c_rgb_psnr,rust_rgb_psnr,c_y_psnr,ss_rough_luma,ss_luma_cheap,ss_luma_exact,ss_tu_leaf,ss_tu_split,ss_nxn_rough,ss_nxn_batch,ss_chroma_rough,ss_chroma_trial,ss_rdoq,ss_residual_price,ss_final_commit,ss_writer,ss_deblock,ss_sao"
    )?;

    let mut rows: Vec<Row> = Vec::new();
    for source in sources {
        let base = source
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("source")
            .to_string();

        // Build the list of (png_path, size) cases. In --native mode each source
        // is encoded directly at its own resolution; otherwise we synthesize a
        // resampled PNG for every requested --sizes entry.
        let cases: Vec<(PathBuf, Size)> = if args.native {
            let dims = png_dimensions(&source)?;
            vec![(
                source.clone(),
                Size {
                    width: dims.0,
                    height: dims.1,
                },
            )]
        } else {
            let src_img = load_rgb(&source)?;
            let mut v = Vec::new();
            for &size in &sizes {
                let png_path =
                    generated_dir.join(format!("{base}_{}x{}.png", size.width, size.height));
                if args.regenerate || !png_path.exists() {
                    let synthetic = synthesize_rgb(&src_img, size);
                    save_rgb_png(&png_path, &synthetic)?;
                }
                v.push((png_path, size));
            }
            v
        };

        for (png_path, size) in cases {
            for &effort in &efforts {
                let effort_dir = rust_dir.join(effort.as_str());
                fs::create_dir_all(&effort_dir)?;

                for run in 1..=args.runs {
                    let c_result = if args.skip_c {
                        None
                    } else {
                        let c_out = c_dir.join(format!(
                            "{base}_{}x{}_run{run}.bpg",
                            size.width, size.height
                        ));
                        Some(run_c_bpgenc(&bpgenc, &png_path, &c_out, &args)?)
                    };

                    let rust_out = effort_dir.join(format!(
                        "{base}_{}x{}_run{run}.bpg",
                        size.width, size.height
                    ));
                    let rust_result = run_rust_encode(&png_path, &rust_out, effort, &args)?;

                    let row = Row {
                        effort: effort.as_str().to_string(),
                        source: base.clone(),
                        width: size.width,
                        height: size.height,
                        pixels: u64::from(size.width) * u64::from(size.height),
                        run,
                        c_total: c_result.as_ref().map(|r| r.time),
                        c_bytes: c_result.as_ref().map(|r| r.bytes),
                        c_rgb_psnr: c_result.as_ref().and_then(|r| r.rgb_psnr),
                        c_y_psnr: c_result.as_ref().and_then(|r| r.y_psnr),
                        rust_rgb_psnr: rust_result.rgb_psnr,
                        rust_total: rust_result.total,
                        rust_prepare: rust_result.prepare,
                        rust_encode: rust_result.encode,
                        rust_bpg_bytes: rust_result.bpg_bytes,
                        rust_annexb_bytes: rust_result.annexb_bytes,
                        quality: rust_result.quality,
                        ctu_count: rust_result.ctu_count,
                        cu_trials: rust_result.cu_trials,
                        cu_early_terminations: rust_result.cu_early_terminations,
                        cu_split_bound_aborts: rust_result.cu_split_bound_aborts,
                        cu_force_leaf: rust_result.cu_force_leaf,
                        code_block_calls: rust_result.code_block_calls,
                        forward_transforms: rust_result.forward_transforms,
                        trial_coded_blocks: rust_result.trial_coded_blocks,
                        final_coded_blocks: rust_result.final_coded_blocks,
                        trial_rdoq_blocks: rust_result.trial_rdoq_blocks,
                        final_rdoq_blocks: rust_result.final_rdoq_blocks,
                        phase_build_us: rust_result.phase_build_us,
                        phase_write_us: rust_result.phase_write_us,
                        bytes_restored: rust_result.bytes_restored,
                        partnxn_attempts: rust_result.partnxn_attempts,
                        partnxn_wins: rust_result.partnxn_wins,
                        stillsearch_ledger: rust_result.stillsearch_ledger,
                    };
                    write_csv_row(&mut csv, &row)?;
                    let q_str = row
                        .quality
                        .as_ref()
                        .map_or(String::new(), |q| format!(" psnr_y={:.2}", q.psnr_y));
                    let rgb_str = row
                        .rust_rgb_psnr
                        .map_or(String::new(), |p| format!(" rgb={p:.2}"));
                    let c_str = match (row.c_bytes, row.c_y_psnr, row.c_rgb_psnr) {
                        (Some(b), Some(y), Some(p)) => format!("  [C {b}B y={y:.2} rgb={p:.2}]"),
                        (Some(b), _, _) => format!("  [C {b}B]"),
                        _ => String::new(),
                    };
                    println!(
                        "{} {} {}x{} run {}: encode={:.3}s bpg={}B{}{}{}",
                        row.effort,
                        row.source,
                        row.width,
                        row.height,
                        row.run,
                        row.rust_encode.as_secs_f64(),
                        row.rust_bpg_bytes,
                        q_str,
                        rgb_str,
                        c_str,
                    );
                    rows.push(row);
                }
            }
        }
    }

    write_combined_summary(&work_dir.join("summary.md"), &rows, &efforts, &args)?;
    println!("wrote {}", csv_path.display());
    println!("wrote {}", work_dir.join("summary.md").display());
    Ok(())
}

fn parse_sizes(s: &str) -> Result<Vec<Size>, String> {
    s.split(',')
        .map(|part| {
            let (w, h) = part
                .split_once('x')
                .or_else(|| part.split_once('X'))
                .ok_or_else(|| format!("size must be WIDTHxHEIGHT: {part}"))?;
            let width = w
                .parse::<u32>()
                .map_err(|e| format!("bad width {w}: {e}"))?;
            let height = h
                .parse::<u32>()
                .map_err(|e| format!("bad height {h}: {e}"))?;
            if width == 0 || height == 0 {
                return Err("width and height must be non-zero".to_string());
            }
            Ok(Size { width, height })
        })
        .collect()
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn discover_sources(
    dir: &Path,
    max_sources: usize,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_image(&path) {
                out.push(path);
            }
        }
    }
    out.sort();
    out.truncate(max_sources);
    Ok(out)
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(OsStr::to_str)
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg")
    )
}

/// Read just the pixel dimensions of a PNG/JPEG without keeping the decoded buffer.
fn png_dimensions(path: &Path) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    let img = load_rgb(path)?;
    Ok((img.width, img.height))
}

fn load_rgb(path: &Path) -> Result<RgbImage, Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(OsStr::to_str)
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => load_jpeg_rgb(path),
        _ => load_png_rgb(path),
    }
}

fn load_jpeg_rgb(path: &Path) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
        &data,
        zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB),
    );
    let pixels = decoder
        .decode()
        .map_err(|e| format!("JPEG decode error: {e}"))?;
    let (width, height) = decoder.dimensions().ok_or("JPEG: missing dimensions")?;
    Ok(RgbImage {
        width: u32::try_from(width)?,
        height: u32::try_from(height)?,
        pixels,
    })
}

fn load_png_rgb(path: &Path) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut decoder = zune_png::PngDecoder::new(&data);
    let pixels = decoder
        .decode_raw()
        .map_err(|e| format!("PNG decode error: {e}"))?;
    let (width, height) = decoder.get_dimensions().ok_or("PNG: missing dimensions")?;
    let bit_depth = decoder.get_depth().ok_or("PNG: missing bit depth")?;
    let color_space = decoder.get_colorspace().ok_or("PNG: missing colorspace")?;
    if bit_depth != zune_core::bit_depth::BitDepth::Eight {
        return Err("highres harness currently expects 8-bit PNG inputs".into());
    }
    let rgb = match color_space {
        zune_core::colorspace::ColorSpace::RGB => pixels,
        zune_core::colorspace::ColorSpace::RGBA => pixels
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        zune_core::colorspace::ColorSpace::Luma => pixels.iter().flat_map(|&v| [v, v, v]).collect(),
        zune_core::colorspace::ColorSpace::LumaA => pixels
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0]])
            .collect(),
        other => return Err(format!("unsupported PNG colorspace {other:?}").into()),
    };
    Ok(RgbImage {
        width: u32::try_from(width)?,
        height: u32::try_from(height)?,
        pixels: rgb,
    })
}

fn synthesize_rgb(src: &RgbImage, size: Size) -> RgbImage {
    let mut pixels = vec![0u8; (size.width as usize) * (size.height as usize) * 3];
    for y in 0..size.height {
        let sy = ((u64::from(y) * u64::from(src.height)) / u64::from(size.height)) as u32;
        for x in 0..size.width {
            let sx = ((u64::from(x) * u64::from(src.width)) / u64::from(size.width)) as u32;
            let src_i = ((sy * src.width + sx) as usize) * 3;
            let dst_i = ((y * size.width + x) as usize) * 3;
            let grain =
                (((x.wrapping_mul(13)) ^ (y.wrapping_mul(17)) ^ ((x + y) * 3)) & 15) as i16 - 8;
            for c in 0..3 {
                let channel_bias = match c {
                    0 => ((x / 32) & 7) as i16,
                    1 => ((y / 32) & 7) as i16,
                    _ => (((x + y) / 64) & 7) as i16,
                };
                let v = src.pixels[src_i + c] as i16 + grain + channel_bias - 3;
                pixels[dst_i + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    RgbImage {
        width: size.width,
        height: size.height,
        pixels,
    }
}

fn save_rgb_png(path: &Path, image: &RgbImage) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image.pixels)?;
    Ok(())
}

struct CResult {
    time: Duration,
    bytes: u64,
    /// Decoded RGB PSNR vs source (end-to-end, comparable across encoders).
    rgb_psnr: Option<f64>,
    /// Decoded luma-Y PSNR vs the source Y, in the same YCbCr space as the rust
    /// `psnr_y` (luma-sensitive; not capped by 4:2:0 chroma subsampling).
    y_psnr: Option<f64>,
}

fn run_c_bpgenc(
    bpgenc: &Path,
    png: &Path,
    out: &Path,
    args: &Args,
) -> Result<CResult, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut cmd = Command::new(bpgenc);
    cmd.arg("-o")
        .arg(out)
        .arg("-q")
        .arg(args.qp.to_string())
        .arg("-f")
        .arg(args.format.bpgenc_arg())
        .arg("-b")
        .arg(args.bit_depth.to_string())
        .arg("-m")
        .arg(args.compress_level.to_string())
        .args(&args.c_extra_args)
        .arg(png);
    if args.c_single_thread {
        cmd.env("BPG_X265_SINGLE_THREAD", "1");
    }
    let mut x265_params = Vec::new();
    if args.c_single_thread {
        x265_params.extend([
            "frame-threads=1".to_string(),
            "pools=none".to_string(),
            "wpp=0".to_string(),
            "pmode=0".to_string(),
        ]);
    }
    x265_params.extend(args.c_x265_params.iter().cloned());
    if !x265_params.is_empty() {
        cmd.env("BPG_X265_PARAMS", x265_params.join(","));
    }
    let output = if args.c_single_thread {
        run_command_single_cpu(cmd)?
    } else {
        cmd.output()?
    };
    let elapsed = start.elapsed();
    if !output.status.success() {
        return Err(format!(
            "bpgenc failed for {}: {}{}",
            png.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let bytes = fs::metadata(out)?.len();
    // Decode the C output and measure quality the same way the rust path is, so
    // the numbers are directly comparable / BD-rateable:
    //   - rgb_psnr: end-to-end decode→RGB vs source RGB (4:2:0 chroma-ceiling'd).
    //   - y_psnr:   decoded luma vs the source luma plane (same conversion the
    //               rust encoder uses), luma-sensitive — the right metric for
    //               coefficient-coding work.
    let (rgb_psnr, y_psnr) = match (load_rgb(png), fs::read(out)) {
        (Ok(src), Ok(c_bpg)) => {
            let rgb = decoded_rgb_psnr(&c_bpg, &src);
            let y = c_decoded_y_psnr(&c_bpg, &src, args);
            (rgb, y)
        }
        _ => (None, None),
    };
    Ok(CResult {
        time: elapsed,
        bytes,
        rgb_psnr,
        y_psnr,
    })
}

/// Luma-Y PSNR of a decoded `.bpg` against the source luma, where the source Y is
/// produced by the *same* RGB→YCbCr conversion the rust encoder uses. Both C and
/// rust are decoded through the same (bpgdec-exact) decoder, so this is a fair,
/// luma-sensitive comparison provided both encoders share libbpg's color matrix —
/// which a near-lossless (QP1) sanity check confirms (C y_psnr there is ~60+ dB,
/// not floored). Unlike RGB PSNR it is not capped by 4:2:0 chroma subsampling.
fn c_decoded_y_psnr(bpg: &[u8], src: &RgbImage, args: &Args) -> Option<f64> {
    let mut image = Image::from_rgb8(
        &src.pixels,
        src.width,
        src.height,
        ColorSpace::YCbCr,
        false,
        args.bit_depth,
    );
    match args.format {
        FormatArg::Yuv420 => image.subsample_to_420(1),
        FormatArg::Yuv422 => image.subsample_to_422(1),
        FormatArg::Yuv444 => {}
    }
    let display_w = image.width as usize;
    let display_h = image.height as usize;
    let src_y = &image.planes[0].data;
    let peak = ((1u32 << args.bit_depth) - 1) as f64;

    let frame = bpg_decode::DecoderConfig::new().decode_to_frame(bpg).ok()?;
    let dec_y = crop_plane(&frame.y_plane, frame.width as usize, display_w, display_h);
    if dec_y.len() != src_y.len() {
        return None;
    }
    Some(psnr(src_y, &dec_y, peak))
}

#[cfg(windows)]
fn run_command_single_cpu(mut cmd: Command) -> Result<Output, Box<dyn std::error::Error>> {
    use std::process::Stdio;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let handle = child.as_raw_handle();
    let ok = unsafe { windows_sys::Win32::System::Threading::SetProcessAffinityMask(handle, 1) };
    if ok == 0 {
        let _ = child.kill();
        let _ = child.wait();
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(child.wait_with_output()?)
}

#[cfg(not(windows))]
fn run_command_single_cpu(_cmd: Command) -> Result<Output, Box<dyn std::error::Error>> {
    Err("--c-single-thread currently uses Windows process affinity".into())
}

struct RustResult {
    total: Duration,
    prepare: Duration,
    encode: Duration,
    bpg_bytes: usize,
    annexb_bytes: usize,
    quality: Option<QualityMetrics>,
    ctu_count: u64,
    cu_trials: u64,
    cu_early_terminations: u64,
    cu_split_bound_aborts: u64,
    cu_force_leaf: u64,
    code_block_calls: u64,
    forward_transforms: u64,
    trial_coded_blocks: u64,
    final_coded_blocks: u64,
    trial_rdoq_blocks: u64,
    final_rdoq_blocks: u64,
    phase_build_us: u64,
    phase_write_us: u64,
    bytes_restored: u64,
    partnxn_attempts: u64,
    partnxn_wins: u64,
    /// StillSearch per-CTU work-ledger bucket call counts (summed over the
    /// frame). Indexed by `WorkBucket` order: RoughLuma, LumaCheap, LumaExact,
    /// TuLeaf, TuSplit, NxnRough, NxnBatch, ChromaRough, ChromaTrial, Rdoq,
    /// ResidualPrice, FinalCommit, Writer, Deblock, Sao.
    stillsearch_ledger: [u64; 15],
    /// True end-to-end RGB PSNR (decode `.bpg` → RGB vs source), comparable to the
    /// C number from [`decoded_rgb_psnr`].
    rgb_psnr: Option<f64>,
}

fn run_rust_encode(
    png: &Path,
    out: &Path,
    effort: EffortArg,
    args: &Args,
) -> Result<RustResult, Box<dyn std::error::Error>> {
    if args.rust_single_thread {
        return with_env_var("BPG_ENC_THREADS", "1", || {
            run_rust_encode_inner(png, out, effort, args)
        });
    }
    run_rust_encode_inner(png, out, effort, args)
}

fn run_rust_encode_inner(
    png: &Path,
    out: &Path,
    effort: EffortArg,
    args: &Args,
) -> Result<RustResult, Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let img = load_rgb(png)?;
    let prep_start = Instant::now();
    let mut image = Image::from_rgb8(
        &img.pixels,
        img.width,
        img.height,
        ColorSpace::YCbCr,
        false,
        args.bit_depth,
    );
    match args.format {
        FormatArg::Yuv420 => image.subsample_to_420(1),
        FormatArg::Yuv422 => image.subsample_to_422(1),
        FormatArg::Yuv444 => {}
    }

    // Save source YCbCr planes before CTU padding for quality measurement.
    let display_w = image.width as usize;
    let display_h = image.height as usize;
    let (cw, ch) = args.format.chroma_subsampling(display_w, display_h);
    let peak = ((1u32 << args.bit_depth) - 1) as f64;
    let src_y = image.planes[0].data.clone();
    let src_cb = image
        .planes
        .get(1)
        .map_or(vec![0u16; cw * ch], |p| p.data.clone());
    let src_cr = image
        .planes
        .get(2)
        .map_or(vec![0u16; cw * ch], |p| p.data.clone());

    let prepare = prep_start.elapsed();
    let backend = RustStillHevcEncoder::new(effort.into())
        .with_adaptive_qp(args.adaptive_qp)
        .with_debug_stats(args.debug_stats)
        .with_sao(args.sao.into())
        .with_deblock(DeblockMode::On);
    let encode_start = Instant::now();
    let bpg = encode_still_image(
        image,
        &backend,
        args.qp,
        args.compress_level,
        EncoderTuning::neutral(),
    )?;
    let encode = encode_start.elapsed();
    fs::write(out, &bpg)?;
    let total = total_start.elapsed();

    // True end-to-end RGB PSNR, measured identically to the C path (decode the
    // written stream back to RGB and compare to the source) so they BD-rate.
    let rgb_psnr = decoded_rgb_psnr(&bpg, &img);

    let last = backend
        .last_encode_stats()
        .ok_or("still265 did not publish last encode stats")?;

    // Compute quality from reconstruction vs source.
    let quality = backend.last_reconstruction().map(|recon| {
        let y_stride = recon.width as usize;
        let c_stride = (recon.width as usize).div_ceil(if args.format == FormatArg::Yuv444 {
            1
        } else {
            2
        });

        // Crop reconstruction to display size.
        let cropped_y = crop_plane(&recon.y_plane, y_stride, display_w, display_h);
        let cropped_cb = crop_plane(&recon.cb_plane, c_stride, cw, ch);
        let cropped_cr = crop_plane(&recon.cr_plane, c_stride, cw, ch);

        let py = psnr(&src_y, &cropped_y, peak);
        let pcb = psnr(&src_cb, &cropped_cb, peak);
        let pcr = psnr(&src_cr, &cropped_cr, peak);

        // Weighted PSNR-RGB: 6:1:1 for Y:Cb:Cr (industry convention for 4:2:0)
        let psnr_rgb = if py.is_finite() && pcb.is_finite() && pcr.is_finite() {
            -(10.0
                * ((6.0 * 10.0_f64.powf(-py / 10.0)
                    + 10.0_f64.powf(-pcb / 10.0)
                    + 10.0_f64.powf(-pcr / 10.0))
                    / 8.0)
                    .log10())
        } else {
            py
        };

        QualityMetrics {
            psnr_y: py,
            psnr_cb: pcb,
            psnr_cr: pcr,
            psnr_rgb,
        }
    });

    Ok(RustResult {
        total,
        prepare,
        encode,
        bpg_bytes: bpg.len(),
        annexb_bytes: last.annexb_bytes,
        quality,
        ctu_count: last.stats.ctu_count,
        cu_trials: last.stats.cu_trials,
        cu_early_terminations: last.stats.cu_early_terminations,
        cu_split_bound_aborts: last.stats.cu_split_bound_aborts,
        cu_force_leaf: last.stats.cu_force_leaf,
        code_block_calls: last.stats.code_block_calls,
        forward_transforms: last.stats.forward_transforms,
        trial_coded_blocks: last.stats.trial_coded_blocks,
        final_coded_blocks: last.stats.final_coded_blocks,
        trial_rdoq_blocks: last.stats.trial_rdoq_blocks,
        final_rdoq_blocks: last.stats.final_rdoq_blocks,
        phase_build_us: last.stats.phase_build_us,
        phase_write_us: last.stats.phase_write_us,
        bytes_restored: last.stats.bytes_restored,
        partnxn_attempts: last.stats.partnxn_attempts,
        partnxn_wins: last.stats.partnxn_wins,
        stillsearch_ledger: last.stats.stillsearch_ledger,
        rgb_psnr,
    })
}

fn crop_plane(src: &[u16], stride: usize, width: usize, height: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let row = y * stride;
        out.extend_from_slice(&src[row..row + width]);
    }
    out
}

/// Compute PSNR between two equal-length plane buffers.
/// True end-to-end RGB PSNR: decode the encoded `.bpg` with our (bpgdec-exact)
/// decoder to RGB8 and compare against the source PNG/JPEG RGB.
///
/// This is the fair cross-encoder metric: C bpgenc and still265 each pick their
/// own internal RGB↔YCbCr conversion, so comparing reconstructed YCbCr against an
/// encoder-internal source plane penalises whichever encoder's convention differs
/// from the harness's. Measuring after a full decode back to display RGB — the
/// pixels the user actually receives — removes that bias and lets the C and rust
/// numbers be BD-rated against each other directly.
fn decoded_rgb_psnr(bpg: &[u8], src: &RgbImage) -> Option<f64> {
    let out = bpg_decode::DecoderConfig::new()
        .decode(bpg, bpg_decode::PixelLayout::Rgb8)
        .ok()?;
    if out.width != src.width || out.height != src.height || out.data.len() != src.pixels.len() {
        return None;
    }
    let mut sse = 0f64;
    for (&a, &b) in out.data.iter().zip(src.pixels.iter()) {
        let d = a as f64 - b as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return Some(f64::INFINITY);
    }
    let mse = sse / out.data.len() as f64;
    Some(10.0 * (255.0 * 255.0 / mse).log10())
}

fn psnr(a: &[u16], b: &[u16], peak: f64) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut sse = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / a.len() as f64;
    10.0 * (peak * peak / mse).log10()
}

fn with_env_var<T, F>(key: &str, value: &str, f: F) -> Result<T, Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
{
    let old = std::env::var_os(key);
    std::env::set_var(key, value);
    let result = f();
    if let Some(old) = old {
        std::env::set_var(key, old);
    } else {
        std::env::remove_var(key);
    }
    result
}

fn write_csv_row(out: &mut File, row: &Row) -> Result<(), Box<dyn std::error::Error>> {
    let q = row.quality.as_ref();
    let qy = q.map_or(String::new(), |q| format!("{:.4}", q.psnr_y));
    let qcb = q.map_or(String::new(), |q| format!("{:.4}", q.psnr_cb));
    let qcr = q.map_or(String::new(), |q| format!("{:.4}", q.psnr_cr));
    let qrgb = q.map_or(String::new(), |q| format!("{:.4}", q.psnr_rgb));
    write!(
        out,
        "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        csv_field(&row.effort),
        csv_field(&row.source),
        row.width,
        row.height,
        row.pixels,
        row.run,
        opt_secs(row.c_total),
        row.c_bytes.map_or(String::new(), |v| v.to_string()),
        row.rust_total.as_secs_f64(),
        row.rust_prepare.as_secs_f64(),
        row.rust_encode.as_secs_f64(),
        row.rust_bpg_bytes,
        row.rust_annexb_bytes,
        qy,
        qcb,
        qcr,
        qrgb,
        row.ctu_count,
        row.cu_trials,
        row.cu_early_terminations,
        row.cu_split_bound_aborts,
        row.cu_force_leaf,
        row.code_block_calls,
        row.forward_transforms,
        row.trial_coded_blocks,
        row.final_coded_blocks,
        row.trial_rdoq_blocks,
        row.final_rdoq_blocks,
        row.phase_build_us,
        row.phase_write_us,
        row.bytes_restored,
        row.partnxn_attempts,
        row.partnxn_wins,
        row.c_rgb_psnr.map_or(String::new(), |p| format!("{p:.4}")),
        row.rust_rgb_psnr.map_or(String::new(), |p| format!("{p:.4}")),
        row.c_y_psnr.map_or(String::new(), |p| format!("{p:.4}")),
    )?;
    let ledger = row
        .stillsearch_ledger
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    writeln!(out, ",{ledger}")?;
    Ok(())
}

fn opt_secs(value: Option<Duration>) -> String {
    value.map_or(String::new(), |d| format!("{:.6}", d.as_secs_f64()))
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Write a combined multi-effort summary table sorted by size then effort.
fn write_combined_summary(
    path: &Path,
    rows: &[Row],
    efforts: &[EffortArg],
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;

    let mut out = File::create(path)?;
    writeln!(out, "# Multi-effort BPG comparison summary")?;
    writeln!(out)?;
    writeln!(
        out,
        "Generated at QP={}, format={}, bit-depth={}, C -m {}, {} run(s) per case{}.",
        args.qp,
        args.format,
        args.bit_depth,
        args.compress_level,
        args.runs,
        if args.native {
            ", native resolution"
        } else {
            ""
        },
    )?;
    writeln!(out)?;

    // Group rows by size.
    let mut by_size: BTreeMap<(u32, u32), Vec<&Row>> = BTreeMap::new();
    for row in rows {
        by_size
            .entry((row.width, row.height))
            .or_default()
            .push(row);
    }

    for ((w, h), group) in &by_size {
        let mp = (*w as f64 * *h as f64) / 1_000_000.0;
        writeln!(out, "## {w}x{h} ({mp:.1} MP)")?;
        writeln!(out)?;

        // C bpgenc baseline (same for every effort row in this size group).
        let c_row = group.iter().find(|r| r.c_total.is_some());
        let c_rgb = c_row.and_then(|c| c.c_rgb_psnr);
        let c_y = c_row.and_then(|c| c.c_y_psnr);
        if let Some(c) = c_row {
            if let (Some(t), Some(b)) = (c.c_total, c.c_bytes) {
                let c_q = match (c_y, c_rgb) {
                    (Some(y), Some(r)) => format!(", {y:.2} dB psnr_y / {r:.2} dB rgb(dec)"),
                    _ => String::new(),
                };
                writeln!(
                    out,
                    "C bpgenc (-m {}): {:.3} s ({:.3} s/MP), {} bytes{}.",
                    args.compress_level,
                    t.as_secs_f64(),
                    t.as_secs_f64() / mp,
                    b,
                    c_q,
                )?;
                writeln!(out)?;
            }
        }

        // Both PSNR columns are decode-side and directly comparable to C:
        // `psnr_y` is luma (decoded recon vs source Y — the metric for
        // coefficient-coding work, not capped by 4:2:0 chroma); `rgb(dec)` is the
        // end-to-end decode→RGB PSNR (chroma-ceiling'd at 4:2:0). `Δ.. vs C` is
        // rust − C in each: positive = rust higher quality at this QP. Read with
        // `vs C bytes` for RD position (true BD-rate needs a QP sweep).
        writeln!(
            out,
            "| effort | encode s | vs C time | bpg bytes | vs C bytes | psnr_y | Δpy vs C | rgb(dec) | Δrgb vs C | cu_trials |"
        )?;
        writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")?;

        for effort in efforts {
            let name = effort.as_str();
            if let Some(row) = group.iter().find(|r| r.effort == name) {
                let vs_c_time = row.c_total.map_or(String::new(), |c| {
                    format!("{:.2}x", row.rust_encode.as_secs_f64() / c.as_secs_f64())
                });
                let vs_c_bytes = row.c_bytes.map_or(String::new(), |c| {
                    if c > 0 {
                        format!(
                            "{:+.1}%",
                            (row.rust_bpg_bytes as f64 / c as f64 - 1.0) * 100.0
                        )
                    } else {
                        String::new()
                    }
                });

                let q = row.quality.as_ref();
                let py = q.map(|q| q.psnr_y);
                let py_str = py.map_or(String::new(), |p| format!("{p:.2}"));
                let dpy = match (py, c_y) {
                    (Some(r), Some(c)) => format!("{:+.2}", r - c),
                    _ => String::new(),
                };
                let rgb_dec = row
                    .rust_rgb_psnr
                    .map_or(String::new(), |p| format!("{p:.2}"));
                let drgb = match (row.rust_rgb_psnr, c_rgb) {
                    (Some(r), Some(c)) => format!("{:+.2}", r - c),
                    _ => String::new(),
                };
                writeln!(
                    out,
                    "| {} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    name,
                    row.rust_encode.as_secs_f64(),
                    vs_c_time,
                    row.rust_bpg_bytes,
                    vs_c_bytes,
                    py_str,
                    dpy,
                    rgb_dec,
                    drgb,
                    row.cu_trials,
                )?;
            }
        }
        writeln!(out)?;
    }

    writeln!(out, "---")?;
    writeln!(out)?;
    writeln!(out, "## StillSearch work ledger (per-bucket call counts)")?;
    writeln!(out)?;
    writeln!(
        out,
        "Summed over the frame, for the first size group. These are the live \
         StillSearch instrumentation buckets (`WorkBucket`); the old rdo2 \
         work-volume columns are retired with that core. `cu_trials` and \
         `partnxn_*` are the surviving CU-level counters.",
    )?;
    writeln!(out)?;

    const BUCKETS: [&str; 15] = [
        "RoughLuma",
        "LumaCheap",
        "LumaExact",
        "TuLeaf",
        "TuSplit",
        "NxnRough",
        "NxnBatch",
        "ChromaRough",
        "ChromaTrial",
        "Rdoq",
        "ResidualPrice",
        "FinalCommit",
        "Writer",
        "Deblock",
        "Sao",
    ];

    let first_size_key = by_size.keys().next();
    if let Some(size_key) = first_size_key {
        let group = &by_size[size_key];
        writeln!(
            out,
            "| effort | cu_trials | partnxn_att | partnxn_win | {} |",
            BUCKETS.join(" | "),
        )?;
        writeln!(
            out,
            "|---|---:|---:|---:|{}|",
            "---:|".repeat(BUCKETS.len()),
        )?;
        for effort in efforts {
            let name = effort.as_str();
            if let Some(row) = group.iter().find(|r| r.effort == name) {
                let ledger = row
                    .stillsearch_ledger
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(" | ");
                writeln!(
                    out,
                    "| {} | {} | {} | {} | {} |",
                    name, row.cu_trials, row.partnxn_attempts, row.partnxn_wins, ledger,
                )?;
            }
        }
    }
    writeln!(out)?;
    Ok(())
}
