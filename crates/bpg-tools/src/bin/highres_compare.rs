//! Rust-only high-resolution encode timing harness.
//!
//! Generates deterministic high-res PNGs from `test-set`, then compares
//! process-level `bpgenc` time with in-process still265 encode timing.

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
    #[arg(long, default_value = "1000x750,2000x1500,4000x3000")]
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

    /// Rust RD-search effort.
    #[arg(long, value_enum, default_value_t = EffortArg::Balanced)]
    effort: EffortArg,

    /// Chroma format.
    #[arg(short = 'f', long, value_enum, default_value_t = FormatArg::Yuv420)]
    format: FormatArg,

    /// Rust still265 SAO mode.
    #[arg(long, value_enum, default_value_t = SaoArg::On)]
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
    ///
    /// Stock libbpg's x265 wrapper does not expose x265_param_parse overrides.
    /// x265's native single-thread controls are frame-threads=1 and pools=none;
    /// this flag enforces the equivalent benchmark constraint for the bundled
    /// executable by applying one-CPU process affinity.
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

    /// Print still265 debug stats during Rust encodes.
    #[arg(long)]
    debug_stats: bool,
}

#[derive(Clone, Copy, Debug)]
struct Size {
    width: u32,
    height: u32,
}

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
    fn from(value: EffortArg) -> Self {
        match value {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    #[value(name = "420")]
    Yuv420,
    #[value(name = "422")]
    Yuv422,
    #[value(name = "444")]
    Yuv444,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SaoArg {
    On,
    Off,
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
}

struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone)]
struct Row {
    source: String,
    width: u32,
    height: u32,
    pixels: u64,
    run: usize,
    c_total: Option<Duration>,
    c_bytes: Option<u64>,
    rust_total: Duration,
    rust_prepare: Duration,
    rust_encode: Duration,
    rust_bpg_bytes: usize,
    rust_annexb_bytes: usize,
    ctu_count: u64,
    cu_trials: u64,
    cu_early_terminations: u64,
    cu_split_bound_aborts: u64,
    cu_force_leaf: u64,
    tu_split_early_terminations: u64,
    rmd_prunes: u64,
    luma_candidate_expansions: u64,
    chroma_candidate_expansions: u64,
    partnxn_attempts: u64,
    partnxn_skips: u64,
    partnxn_wins: u64,
    partnxn_losses: u64,
    partnxn_cu_trials: u64,
    partnxn_code_block_calls: u64,
    trial_coded_blocks: u64,
    final_coded_blocks: u64,
    final_rdoq_blocks: u64,
    trial_rdoq_blocks: u64,
    best_tt_cheap_tu_decisions: u64,
    best_tt_escalated_tu_decisions: u64,
    best_tt_escalation_changed_winner: u64,
    best_tt_full_trial_rdoq_blocks_saved: u64,
    best_tt_exact_residual_estimates_saved: u64,
    rdo2_chroma_scratch_candidates: u64,
    rdo2_chroma_scratch_cheap_evals: u64,
    rdo2_chroma_scratch_exact_evals: u64,
    rdo2_chroma_scratch_exact_escalations: u64,
    rdo2_chroma_scratch_changed_winner: u64,
    rdo2_chroma_scratch_legacy_evals_skipped: u64,
    rdo2_chroma_scratch_stacked_fallbacks: u64,
    full_rd_close_calls: u64,
    luma_close_call_escalations: u64,
    luma_rough_predictions: u64,
    chroma_rough_predictions: u64,
    code_block_calls: u64,
    forward_transforms: u64,
    inverse_transforms: u64,
    residual_bit_estimates: u64,
    cache_builds: u64,
    cache_fast_hits: u64,
    cache_fallbacks: u64,
    frame_snapshots: u64,
    frame_restores: u64,
    map_snapshots: u64,
    map_restores: u64,
    bytes_snapshotted: u64,
    bytes_restored: u64,
    phase_total_us: u64,
    phase_build_us: u64,
    phase_parallel_restore_us: u64,
    phase_deblock_us: u64,
    phase_sao_decide_us: u64,
    phase_sao_apply_us: u64,
    phase_write_us: u64,
    angular_exclusions: u64,
    policy_angular_forced: u64,
    policy_angular_guarded: u64,
    policy_early_term_suppressed: u64,
    region_class_counts: String,
    luma_winner_rank_counts: String,
    chroma_winner_rank_counts: String,
    cu_leaf_wins_by_region: String,
    cu_split_wins_by_region: String,
    tu_leaf_wins_by_region: String,
    tu_split_wins_by_region: String,
    partnxn_wins_by_region: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.runs == 0 {
        return Err("--runs must be at least 1".into());
    }
    if ![8, 10, 12].contains(&args.bit_depth) {
        return Err("--bit-depth must be 8, 10, or 12".into());
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
        "source,width,height,pixels,run,c_total_s,c_bpg_bytes,rust_total_s,rust_prepare_s,rust_encode_s,rust_bpg_bytes,rust_annexb_bytes,ctu_count,cu_trials,cu_early_terminations,cu_split_bound_aborts,cu_force_leaf,tu_split_early_terminations,rmd_prunes,luma_candidate_expansions,chroma_candidate_expansions,partnxn_attempts,partnxn_skips,partnxn_wins,partnxn_losses,partnxn_cu_trials,partnxn_code_block_calls,trial_coded_blocks,final_coded_blocks,final_rdoq_blocks,trial_rdoq_blocks,best_tt_cheap_tu_decisions,best_tt_escalated_tu_decisions,best_tt_escalation_changed_winner,best_tt_full_trial_rdoq_blocks_saved,best_tt_exact_residual_estimates_saved,rdo2_chroma_scratch_candidates,rdo2_chroma_scratch_cheap_evals,rdo2_chroma_scratch_exact_evals,rdo2_chroma_scratch_exact_escalations,rdo2_chroma_scratch_changed_winner,rdo2_chroma_scratch_legacy_evals_skipped,rdo2_chroma_scratch_stacked_fallbacks,full_rd_close_calls,luma_close_call_escalations,luma_rough_predictions,chroma_rough_predictions,code_block_calls,forward_transforms,inverse_transforms,residual_bit_estimates,cache_builds,cache_fast_hits,cache_fallbacks,frame_snapshots,frame_restores,map_snapshots,map_restores,bytes_snapshotted,bytes_restored,phase_total_us,phase_build_us,phase_parallel_restore_us,phase_deblock_us,phase_sao_decide_us,phase_sao_apply_us,phase_write_us,angular_exclusions,policy_angular_forced,policy_angular_guarded,policy_early_term_suppressed,region_class_counts,luma_winner_rank_counts,chroma_winner_rank_counts,cu_leaf_wins_by_region,cu_split_wins_by_region,tu_leaf_wins_by_region,tu_split_wins_by_region,partnxn_wins_by_region"
    )?;

    let mut rows = Vec::new();
    for source in sources {
        let base = source
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("source")
            .to_string();
        let src_img = load_rgb(&source)?;
        for &size in &sizes {
            let png_path = generated_dir.join(format!("{base}_{}x{}.png", size.width, size.height));
            if args.regenerate || !png_path.exists() {
                let synthetic = synthesize_rgb(&src_img, size);
                save_rgb_png(&png_path, &synthetic)?;
            }
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
                let rust_out = rust_dir.join(format!(
                    "{base}_{}x{}_run{run}.bpg",
                    size.width, size.height
                ));
                let rust_result = run_rust_encode(&png_path, &rust_out, &args)?;
                let row = Row {
                    source: base.clone(),
                    width: size.width,
                    height: size.height,
                    pixels: u64::from(size.width) * u64::from(size.height),
                    run,
                    c_total: c_result.as_ref().map(|r| r.0),
                    c_bytes: c_result.as_ref().map(|r| r.1),
                    rust_total: rust_result.total,
                    rust_prepare: rust_result.prepare,
                    rust_encode: rust_result.encode,
                    rust_bpg_bytes: rust_result.bpg_bytes,
                    rust_annexb_bytes: rust_result.annexb_bytes,
                    ctu_count: rust_result.ctu_count,
                    cu_trials: rust_result.cu_trials,
                    cu_early_terminations: rust_result.cu_early_terminations,
                    cu_split_bound_aborts: rust_result.cu_split_bound_aborts,
                    cu_force_leaf: rust_result.cu_force_leaf,
                    tu_split_early_terminations: rust_result.tu_split_early_terminations,
                    rmd_prunes: rust_result.rmd_prunes,
                    luma_candidate_expansions: rust_result.luma_candidate_expansions,
                    chroma_candidate_expansions: rust_result.chroma_candidate_expansions,
                    partnxn_attempts: rust_result.partnxn_attempts,
                    partnxn_skips: rust_result.partnxn_skips,
                    partnxn_wins: rust_result.partnxn_wins,
                    partnxn_losses: rust_result.partnxn_losses,
                    partnxn_cu_trials: rust_result.partnxn_cu_trials,
                    partnxn_code_block_calls: rust_result.partnxn_code_block_calls,
                    trial_coded_blocks: rust_result.trial_coded_blocks,
                    final_coded_blocks: rust_result.final_coded_blocks,
                    final_rdoq_blocks: rust_result.final_rdoq_blocks,
                    trial_rdoq_blocks: rust_result.trial_rdoq_blocks,
                    best_tt_cheap_tu_decisions: rust_result.best_tt_cheap_tu_decisions,
                    best_tt_escalated_tu_decisions: rust_result.best_tt_escalated_tu_decisions,
                    best_tt_escalation_changed_winner: rust_result
                        .best_tt_escalation_changed_winner,
                    best_tt_full_trial_rdoq_blocks_saved: rust_result
                        .best_tt_full_trial_rdoq_blocks_saved,
                    best_tt_exact_residual_estimates_saved: rust_result
                        .best_tt_exact_residual_estimates_saved,
                    rdo2_chroma_scratch_candidates: rust_result.rdo2_chroma_scratch_candidates,
                    rdo2_chroma_scratch_cheap_evals: rust_result.rdo2_chroma_scratch_cheap_evals,
                    rdo2_chroma_scratch_exact_evals: rust_result.rdo2_chroma_scratch_exact_evals,
                    rdo2_chroma_scratch_exact_escalations: rust_result
                        .rdo2_chroma_scratch_exact_escalations,
                    rdo2_chroma_scratch_changed_winner: rust_result
                        .rdo2_chroma_scratch_changed_winner,
                    rdo2_chroma_scratch_legacy_evals_skipped: rust_result
                        .rdo2_chroma_scratch_legacy_evals_skipped,
                    rdo2_chroma_scratch_stacked_fallbacks: rust_result
                        .rdo2_chroma_scratch_stacked_fallbacks,
                    full_rd_close_calls: rust_result.full_rd_close_calls,
                    luma_close_call_escalations: rust_result.luma_close_call_escalations,
                    luma_rough_predictions: rust_result.luma_rough_predictions,
                    chroma_rough_predictions: rust_result.chroma_rough_predictions,
                    code_block_calls: rust_result.code_block_calls,
                    forward_transforms: rust_result.forward_transforms,
                    inverse_transforms: rust_result.inverse_transforms,
                    residual_bit_estimates: rust_result.residual_bit_estimates,
                    cache_builds: rust_result.cache_builds,
                    cache_fast_hits: rust_result.cache_fast_hits,
                    cache_fallbacks: rust_result.cache_fallbacks,
                    frame_snapshots: rust_result.frame_snapshots,
                    frame_restores: rust_result.frame_restores,
                    map_snapshots: rust_result.map_snapshots,
                    map_restores: rust_result.map_restores,
                    bytes_snapshotted: rust_result.bytes_snapshotted,
                    bytes_restored: rust_result.bytes_restored,
                    phase_total_us: rust_result.phase_total_us,
                    phase_build_us: rust_result.phase_build_us,
                    phase_parallel_restore_us: rust_result.phase_parallel_restore_us,
                    phase_deblock_us: rust_result.phase_deblock_us,
                    phase_sao_decide_us: rust_result.phase_sao_decide_us,
                    phase_sao_apply_us: rust_result.phase_sao_apply_us,
                    phase_write_us: rust_result.phase_write_us,
                    angular_exclusions: rust_result.angular_exclusions,
                    policy_angular_forced: rust_result.policy_angular_forced,
                    policy_angular_guarded: rust_result.policy_angular_guarded,
                    policy_early_term_suppressed: rust_result.policy_early_term_suppressed,
                    region_class_counts: rust_result.region_class_counts,
                    luma_winner_rank_counts: rust_result.luma_winner_rank_counts,
                    chroma_winner_rank_counts: rust_result.chroma_winner_rank_counts,
                    cu_leaf_wins_by_region: rust_result.cu_leaf_wins_by_region,
                    cu_split_wins_by_region: rust_result.cu_split_wins_by_region,
                    tu_leaf_wins_by_region: rust_result.tu_leaf_wins_by_region,
                    tu_split_wins_by_region: rust_result.tu_split_wins_by_region,
                    partnxn_wins_by_region: rust_result.partnxn_wins_by_region,
                };
                write_csv_row(&mut csv, &row)?;
                println!(
                    "{} {}x{} run {}: c={} rust_total={:.3}s rust_encode={:.3}s cu_trials={}",
                    row.source,
                    row.width,
                    row.height,
                    row.run,
                    row.c_total.map_or("skipped".to_string(), |d| format!(
                        "{:.3}s",
                        d.as_secs_f64()
                    )),
                    row.rust_total.as_secs_f64(),
                    row.rust_encode.as_secs_f64(),
                    row.cu_trials,
                );
                rows.push(row);
            }
        }
    }

    write_summary(&work_dir.join("summary.md"), &rows)?;
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

fn run_c_bpgenc(
    bpgenc: &Path,
    png: &Path,
    out: &Path,
    args: &Args,
) -> Result<(Duration, u64), Box<dyn std::error::Error>> {
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
    Ok((elapsed, fs::metadata(out)?.len()))
}

#[cfg(windows)]
fn run_command_single_cpu(mut cmd: Command) -> Result<Output, Box<dyn std::error::Error>> {
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
    ctu_count: u64,
    cu_trials: u64,
    cu_early_terminations: u64,
    cu_split_bound_aborts: u64,
    cu_force_leaf: u64,
    tu_split_early_terminations: u64,
    rmd_prunes: u64,
    luma_candidate_expansions: u64,
    chroma_candidate_expansions: u64,
    partnxn_attempts: u64,
    partnxn_skips: u64,
    partnxn_wins: u64,
    partnxn_losses: u64,
    partnxn_cu_trials: u64,
    partnxn_code_block_calls: u64,
    trial_coded_blocks: u64,
    final_coded_blocks: u64,
    final_rdoq_blocks: u64,
    trial_rdoq_blocks: u64,
    best_tt_cheap_tu_decisions: u64,
    best_tt_escalated_tu_decisions: u64,
    best_tt_escalation_changed_winner: u64,
    best_tt_full_trial_rdoq_blocks_saved: u64,
    best_tt_exact_residual_estimates_saved: u64,
    rdo2_chroma_scratch_candidates: u64,
    rdo2_chroma_scratch_cheap_evals: u64,
    rdo2_chroma_scratch_exact_evals: u64,
    rdo2_chroma_scratch_exact_escalations: u64,
    rdo2_chroma_scratch_changed_winner: u64,
    rdo2_chroma_scratch_legacy_evals_skipped: u64,
    rdo2_chroma_scratch_stacked_fallbacks: u64,
    full_rd_close_calls: u64,
    luma_close_call_escalations: u64,
    luma_rough_predictions: u64,
    chroma_rough_predictions: u64,
    code_block_calls: u64,
    forward_transforms: u64,
    inverse_transforms: u64,
    residual_bit_estimates: u64,
    cache_builds: u64,
    cache_fast_hits: u64,
    cache_fallbacks: u64,
    frame_snapshots: u64,
    frame_restores: u64,
    map_snapshots: u64,
    map_restores: u64,
    bytes_snapshotted: u64,
    bytes_restored: u64,
    phase_total_us: u64,
    phase_build_us: u64,
    phase_parallel_restore_us: u64,
    phase_deblock_us: u64,
    phase_sao_decide_us: u64,
    phase_sao_apply_us: u64,
    phase_write_us: u64,
    angular_exclusions: u64,
    policy_angular_forced: u64,
    policy_angular_guarded: u64,
    policy_early_term_suppressed: u64,
    region_class_counts: String,
    luma_winner_rank_counts: String,
    chroma_winner_rank_counts: String,
    cu_leaf_wins_by_region: String,
    cu_split_wins_by_region: String,
    tu_leaf_wins_by_region: String,
    tu_split_wins_by_region: String,
    partnxn_wins_by_region: String,
}

fn run_rust_encode(
    png: &Path,
    out: &Path,
    args: &Args,
) -> Result<RustResult, Box<dyn std::error::Error>> {
    if args.rust_single_thread {
        return with_env_var("BPG_ENC_THREADS", "1", || {
            run_rust_encode_inner(png, out, args)
        });
    }
    run_rust_encode_inner(png, out, args)
}

fn run_rust_encode_inner(
    png: &Path,
    out: &Path,
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
    let prepare = prep_start.elapsed();
    let backend = RustStillHevcEncoder::new(args.effort.into())
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
    let last = backend
        .last_encode_stats()
        .ok_or("still265 did not publish last encode stats")?;
    Ok(RustResult {
        total,
        prepare,
        encode,
        bpg_bytes: bpg.len(),
        annexb_bytes: last.annexb_bytes,
        ctu_count: last.stats.ctu_count,
        cu_trials: last.stats.cu_trials,
        cu_early_terminations: last.stats.cu_early_terminations,
        cu_split_bound_aborts: last.stats.cu_split_bound_aborts,
        cu_force_leaf: last.stats.cu_force_leaf,
        tu_split_early_terminations: last.stats.tu_split_early_terminations,
        rmd_prunes: last.stats.rmd_prunes,
        luma_candidate_expansions: last.stats.luma_candidate_expansions,
        chroma_candidate_expansions: last.stats.chroma_candidate_expansions,
        partnxn_attempts: last.stats.partnxn_attempts,
        partnxn_skips: last.stats.partnxn_skips,
        partnxn_wins: last.stats.partnxn_wins,
        partnxn_losses: last.stats.partnxn_losses,
        partnxn_cu_trials: last.stats.partnxn_cu_trials,
        partnxn_code_block_calls: last.stats.partnxn_code_block_calls,
        trial_coded_blocks: last.stats.trial_coded_blocks,
        final_coded_blocks: last.stats.final_coded_blocks,
        final_rdoq_blocks: last.stats.final_rdoq_blocks,
        trial_rdoq_blocks: last.stats.trial_rdoq_blocks,
        best_tt_cheap_tu_decisions: last.stats.best_tt_cheap_tu_decisions,
        best_tt_escalated_tu_decisions: last.stats.best_tt_escalated_tu_decisions,
        best_tt_escalation_changed_winner: last.stats.best_tt_escalation_changed_winner,
        best_tt_full_trial_rdoq_blocks_saved: last.stats.best_tt_full_trial_rdoq_blocks_saved,
        best_tt_exact_residual_estimates_saved: last.stats.best_tt_exact_residual_estimates_saved,
        rdo2_chroma_scratch_candidates: last.stats.rdo2_chroma_scratch_candidates,
        rdo2_chroma_scratch_cheap_evals: last.stats.rdo2_chroma_scratch_cheap_evals,
        rdo2_chroma_scratch_exact_evals: last.stats.rdo2_chroma_scratch_exact_evals,
        rdo2_chroma_scratch_exact_escalations: last.stats.rdo2_chroma_scratch_exact_escalations,
        rdo2_chroma_scratch_changed_winner: last.stats.rdo2_chroma_scratch_changed_winner,
        rdo2_chroma_scratch_legacy_evals_skipped: last
            .stats
            .rdo2_chroma_scratch_legacy_evals_skipped,
        rdo2_chroma_scratch_stacked_fallbacks: last.stats.rdo2_chroma_scratch_stacked_fallbacks,
        full_rd_close_calls: last.stats.full_rd_close_calls,
        luma_close_call_escalations: last.stats.luma_close_call_escalations,
        luma_rough_predictions: last.stats.luma_rough_predictions,
        chroma_rough_predictions: last.stats.chroma_rough_predictions,
        code_block_calls: last.stats.code_block_calls,
        forward_transforms: last.stats.forward_transforms,
        inverse_transforms: last.stats.inverse_transforms,
        residual_bit_estimates: last.stats.residual_bit_estimates,
        cache_builds: last.stats.cache_builds,
        cache_fast_hits: last.stats.cache_fast_hits,
        cache_fallbacks: last.stats.cache_fallbacks,
        frame_snapshots: last.stats.frame_snapshots,
        frame_restores: last.stats.frame_restores,
        map_snapshots: last.stats.map_snapshots,
        map_restores: last.stats.map_restores,
        bytes_snapshotted: last.stats.bytes_snapshotted,
        bytes_restored: last.stats.bytes_restored,
        phase_total_us: last.stats.phase_total_us,
        phase_build_us: last.stats.phase_build_us,
        phase_parallel_restore_us: last.stats.phase_parallel_restore_us,
        phase_deblock_us: last.stats.phase_deblock_us,
        phase_sao_decide_us: last.stats.phase_sao_decide_us,
        phase_sao_apply_us: last.stats.phase_sao_apply_us,
        phase_write_us: last.stats.phase_write_us,
        angular_exclusions: last.stats.angular_exclusions,
        policy_angular_forced: last.stats.policy_angular_forced,
        policy_angular_guarded: last.stats.policy_angular_guarded,
        policy_early_term_suppressed: last.stats.policy_early_term_suppressed,
        region_class_counts: join_u64s(&last.stats.region_class_counts),
        luma_winner_rank_counts: join_u64s(&last.stats.luma_winner_rank_counts),
        chroma_winner_rank_counts: join_u64s(&last.stats.chroma_winner_rank_counts),
        cu_leaf_wins_by_region: join_u64s(&last.stats.cu_leaf_wins_by_region),
        cu_split_wins_by_region: join_u64s(&last.stats.cu_split_wins_by_region),
        tu_leaf_wins_by_region: join_u64s(&last.stats.tu_leaf_wins_by_region),
        tu_split_wins_by_region: join_u64s(&last.stats.tu_split_wins_by_region),
        partnxn_wins_by_region: join_u64s(&last.stats.partnxn_wins_by_region),
    })
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
    let fields = vec![
        csv_field(&row.source),
        row.width.to_string(),
        row.height.to_string(),
        row.pixels.to_string(),
        row.run.to_string(),
        opt_secs(row.c_total),
        row.c_bytes.map_or(String::new(), |v| v.to_string()),
        format!("{:.6}", row.rust_total.as_secs_f64()),
        format!("{:.6}", row.rust_prepare.as_secs_f64()),
        format!("{:.6}", row.rust_encode.as_secs_f64()),
        row.rust_bpg_bytes.to_string(),
        row.rust_annexb_bytes.to_string(),
        row.ctu_count.to_string(),
        row.cu_trials.to_string(),
        row.cu_early_terminations.to_string(),
        row.cu_split_bound_aborts.to_string(),
        row.cu_force_leaf.to_string(),
        row.tu_split_early_terminations.to_string(),
        row.rmd_prunes.to_string(),
        row.luma_candidate_expansions.to_string(),
        row.chroma_candidate_expansions.to_string(),
        row.partnxn_attempts.to_string(),
        row.partnxn_skips.to_string(),
        row.partnxn_wins.to_string(),
        row.partnxn_losses.to_string(),
        row.partnxn_cu_trials.to_string(),
        row.partnxn_code_block_calls.to_string(),
        row.trial_coded_blocks.to_string(),
        row.final_coded_blocks.to_string(),
        row.final_rdoq_blocks.to_string(),
        row.trial_rdoq_blocks.to_string(),
        row.best_tt_cheap_tu_decisions.to_string(),
        row.best_tt_escalated_tu_decisions.to_string(),
        row.best_tt_escalation_changed_winner.to_string(),
        row.best_tt_full_trial_rdoq_blocks_saved.to_string(),
        row.best_tt_exact_residual_estimates_saved.to_string(),
        row.rdo2_chroma_scratch_candidates.to_string(),
        row.rdo2_chroma_scratch_cheap_evals.to_string(),
        row.rdo2_chroma_scratch_exact_evals.to_string(),
        row.rdo2_chroma_scratch_exact_escalations.to_string(),
        row.rdo2_chroma_scratch_changed_winner.to_string(),
        row.rdo2_chroma_scratch_legacy_evals_skipped.to_string(),
        row.rdo2_chroma_scratch_stacked_fallbacks.to_string(),
        row.full_rd_close_calls.to_string(),
        row.luma_close_call_escalations.to_string(),
        row.luma_rough_predictions.to_string(),
        row.chroma_rough_predictions.to_string(),
        row.code_block_calls.to_string(),
        row.forward_transforms.to_string(),
        row.inverse_transforms.to_string(),
        row.residual_bit_estimates.to_string(),
        row.cache_builds.to_string(),
        row.cache_fast_hits.to_string(),
        row.cache_fallbacks.to_string(),
        row.frame_snapshots.to_string(),
        row.frame_restores.to_string(),
        row.map_snapshots.to_string(),
        row.map_restores.to_string(),
        row.bytes_snapshotted.to_string(),
        row.bytes_restored.to_string(),
        row.phase_total_us.to_string(),
        row.phase_build_us.to_string(),
        row.phase_parallel_restore_us.to_string(),
        row.phase_deblock_us.to_string(),
        row.phase_sao_decide_us.to_string(),
        row.phase_sao_apply_us.to_string(),
        row.phase_write_us.to_string(),
        row.angular_exclusions.to_string(),
        row.policy_angular_forced.to_string(),
        row.policy_angular_guarded.to_string(),
        row.policy_early_term_suppressed.to_string(),
        csv_field(&row.region_class_counts),
        csv_field(&row.luma_winner_rank_counts),
        csv_field(&row.chroma_winner_rank_counts),
        csv_field(&row.cu_leaf_wins_by_region),
        csv_field(&row.cu_split_wins_by_region),
        csv_field(&row.tu_leaf_wins_by_region),
        csv_field(&row.tu_split_wins_by_region),
        csv_field(&row.partnxn_wins_by_region),
    ];
    writeln!(out, "{}", fields.join(","))?;
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

fn join_u64s(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn write_summary(path: &Path, rows: &[Row]) -> Result<(), Box<dyn std::error::Error>> {
    let mut out = File::create(path)?;
    writeln!(out, "# High-resolution BPG timing summary")?;
    writeln!(out)?;
    writeln!(
        out,
        "| source | size | c total s | rust total s | rust encode s | rust/C total | build s | write s | restore GB | restore fanout s | encode s/MP | cu trials/MP | code blocks/MP |"
    )?;
    writeln!(
        out,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for row in rows {
        let mp = row.pixels as f64 / 1_000_000.0;
        let ratio = match row.c_total {
            Some(c) if c.as_secs_f64() > 0.0 => {
                format!("{:.2}", row.rust_total.as_secs_f64() / c.as_secs_f64())
            }
            _ => String::new(),
        };
        let c_total = row
            .c_total
            .map_or(String::new(), |d| format!("{:.3}", d.as_secs_f64()));
        writeln!(
            out,
            "| {} | {}x{} | {} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {:.2} | {:.3} | {:.3} | {:.0} | {:.0} |",
            row.source,
            row.width,
            row.height,
            c_total,
            row.rust_total.as_secs_f64(),
            row.rust_encode.as_secs_f64(),
            ratio,
            row.phase_build_us as f64 / 1_000_000.0,
            row.phase_write_us as f64 / 1_000_000.0,
            row.bytes_restored as f64 / (1024.0 * 1024.0 * 1024.0),
            row.phase_parallel_restore_us as f64 / 1_000_000.0,
            row.rust_encode.as_secs_f64() / mp,
            row.cu_trials as f64 / mp,
            row.code_block_calls as f64 / mp,
        )?;
    }
    Ok(())
}
