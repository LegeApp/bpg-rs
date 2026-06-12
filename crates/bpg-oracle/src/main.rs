//! x265 reference oracle for the BPG Rust-encoder port.
//!
//! This binary builds a deterministic regression corpus (see [`corpus`]),
//! encodes each image to BPG through the **locked still-image x265 settings**
//! (the same path `bpg-tools encode` uses: intra-only, CQP, repeat-headers — see
//! `SETTINGS.md`, written by `gen`), and for every encode records:
//!
//! - the source image metadata and the resolved x265 params,
//! - the rebuilt Annex-B HEVC the decoder actually consumes (`*.hevc`),
//! - the final BPG bytes (`*.bpg`) and their size,
//! - the stock-bpgdec reference decode (`*.decoded.png`) — the ground truth,
//! - size + quality metrics: bytes, bits/pixel, encode-quality PSNR (bpgdec vs
//!   source), and the in-repo Rust decoder's status/fidelity as a second axis.
//!
//! `gen` writes the whole tree + `manifest.csv`. `check` regenerates into a temp
//! dir and diffs the metrics against a committed `manifest.csv`, so the corpus
//! is a regression baseline: when the future Rust encoder (`bpg-still-hevc`)
//! lands, its output is compared against these same rows. See
//! `FULL_RUST_CODEC_PLAN.md` Phase 2.

mod corpus;

use bpg_encode::{encode_still_image, EncoderTuning};
use bpg_format::BpgHeader;
use bpg_hevc::{rebuild_annexb_from_bpg_payload, BpgHevcInfo};
use bpg_image::{ChromaFormat, ColorSpace, Image};
use bpg_x265::X265Encoder;
use clap::{Parser, Subcommand};
use corpus::{CorpusImage, Depth};
use std::path::{Path, PathBuf};

/// The fixed x265 preset index (compress level) the oracle encodes at. Matches
/// the `bpg-tools encode` default so the oracle reflects real output.
const COMPRESS_LEVEL: u8 = 8;

#[derive(Parser)]
#[command(about = "x265 reference oracle: regenerable BPG corpus + metrics")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate the corpus, encode every config, and write manifest.csv.
    Gen {
        /// Output root (corpus/, out/, manifest.csv land here).
        #[arg(short, long, default_value = "oracle")]
        out: PathBuf,
        /// Natural-photo source (center-cropped into the corpus).
        #[arg(long, default_value = "../dusk.png")]
        dusk: PathBuf,
        /// Stock bpgdec binary (the ground-truth reference decoder).
        #[arg(long, default_value = "../libbpg-0.9.8/bpgdec")]
        bpgdec: PathBuf,
    },
    /// Regenerate into a temp dir and compare metrics against a committed manifest.
    Check {
        /// The committed manifest.csv to compare against.
        #[arg(short, long, default_value = "oracle/manifest.csv")]
        manifest: PathBuf,
        /// Natural-photo source (must match what `gen` used).
        #[arg(long, default_value = "../dusk.png")]
        dusk: PathBuf,
        /// Stock bpgdec binary (the ground-truth reference decoder).
        #[arg(long, default_value = "../libbpg-0.9.8/bpgdec")]
        bpgdec: PathBuf,
    },
}

/// One (chroma, bit-depth, qp) encode configuration.
#[derive(Clone, Copy)]
struct Config {
    chroma: ChromaFormat,
    bit_depth: u8,
    qp: u8,
}

impl Config {
    /// Tag used in filenames and the manifest, e.g. `444_8bit_qp24`.
    fn tag(&self) -> String {
        let cf = match self.chroma {
            ChromaFormat::Yuv420 => "420",
            ChromaFormat::Yuv422 => "422",
            ChromaFormat::Yuv444 => "444",
            ChromaFormat::Gray => "gray",
        };
        format!("{cf}_{}bit_qp{}", self.bit_depth, self.qp)
    }
}

/// 8-bit sources: all three chroma formats, two QPs (a small RD spread).
const CONFIGS_8: &[Config] = &[
    cfg(ChromaFormat::Yuv420, 8, 24),
    cfg(ChromaFormat::Yuv420, 8, 32),
    cfg(ChromaFormat::Yuv422, 8, 24),
    cfg(ChromaFormat::Yuv422, 8, 32),
    cfg(ChromaFormat::Yuv444, 8, 24),
    cfg(ChromaFormat::Yuv444, 8, 32),
];

/// 16-bit sources: exercise the linked 10-bit lib on 4:2:0 and 4:4:4.
const CONFIGS_16: &[Config] = &[
    cfg(ChromaFormat::Yuv420, 10, 28),
    cfg(ChromaFormat::Yuv444, 10, 28),
];

const fn cfg(chroma: ChromaFormat, bit_depth: u8, qp: u8) -> Config {
    Config {
        chroma,
        bit_depth,
        qp,
    }
}

/// One manifest row: identity + the metrics `check` compares.
///
/// `psnr` is the *encode-quality* metric — stock bpgdec (the ground-truth
/// reference decoder) decoded vs the source. `rust_decode` is the Rust
/// decoder's status on the same BPG (`ok` / `unsupported` / `fail`); when it
/// succeeds, `rust_psnr` is its fidelity vs the bpgdec reference. This second
/// axis explicitly tracks the Rust decoder's 4:2:0-only limitation.
struct Row {
    image: String,
    category: String,
    config: String,
    width: u32,
    height: u32,
    bpg_bytes: usize,
    hevc_bytes: usize,
    bpp: f64,
    psnr: f64,
    rust_decode: String,
    rust_psnr: Option<f64>,
}

impl Row {
    const HEADER: &'static str =
        "image,category,config,width,height,bpg_bytes,hevc_bytes,bpp,psnr,rust_decode,rust_psnr";

    fn to_csv(&self) -> String {
        let rust_psnr = self
            .rust_psnr
            .map(|v| format!("{v:.3}"))
            .unwrap_or_default();
        format!(
            "{},{},{},{},{},{},{},{:.4},{:.3},{},{}",
            self.image,
            self.category,
            self.config,
            self.width,
            self.height,
            self.bpg_bytes,
            self.hevc_bytes,
            self.bpp,
            self.psnr,
            self.rust_decode,
            rust_psnr
        )
    }
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Gen { out, dusk, bpgdec } => run_gen(&out, &dusk, &bpgdec),
        Cmd::Check {
            manifest,
            dusk,
            bpgdec,
        } => run_check(&manifest, &dusk, &bpgdec),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Encode one corpus image under one config, write the artifacts under
/// `out_dir`, and return the manifest row. When `out_dir` is `None`, nothing is
/// written (used by `check`).
fn encode_one(
    img: &CorpusImage,
    config: Config,
    bpgdec: &Path,
    out_dir: Option<&Path>,
) -> Result<Row, Box<dyn std::error::Error>> {
    // Build the 4:4:4 YCbCr image, then subsample per the config.
    let mut image = match img.depth {
        Depth::Eight => Image::from_rgb8(
            img.rgb8.as_ref().unwrap(),
            ColorSpace::YCbCr,
            false,
            config.bit_depth,
        ),
        Depth::Sixteen => {
            let buf = img.rgb16.as_ref().unwrap().clone();
            let rgb16 = image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(img.width, img.height, buf)
                .ok_or("rgb16 buffer size mismatch")?;
            Image::from_rgb16(&rgb16, ColorSpace::YCbCr, false, config.bit_depth)
        }
    };
    match config.chroma {
        ChromaFormat::Yuv420 => image.subsample_to_420(1),
        ChromaFormat::Yuv422 => image.subsample_to_422(1),
        ChromaFormat::Yuv444 => {}
        ChromaFormat::Gray => return Err("gray not in corpus".into()),
    }

    let backend = X265Encoder::new();
    let bpg = encode_still_image(image, &backend, config.qp, COMPRESS_LEVEL, EncoderTuning::neutral())?;

    // Rebuild the Annex-B the decoder consumes (the Phase-3 comparison target).
    let file = BpgHeader::read(&bpg).map_err(|e| format!("BPG header read: {e:?}"))?;
    let info = BpgHevcInfo {
        width: file.header.width,
        height: file.header.height,
        chroma_format_idc: chroma_idc(config.chroma),
        bit_depth: config.bit_depth,
        limited_range: file.header.limited_range,
        matrix_coefficients: 6, // BT.601 == ColorSpace::YCbCr
    };
    let annexb = rebuild_annexb_from_bpg_payload(file.payload, info)
        .map_err(|e| format!("annexb rebuild: {e:?}"))?;

    // Ground-truth decode with stock bpgdec -> encode-quality PSNR vs source.
    let ref_rgb = bpgdec_decode(bpgdec, &bpg)?;
    let psnr = rgb_psnr_vs(img, &ref_rgb);

    // Rust-decoder fidelity (second axis): ok/unsupported/fail + PSNR vs bpgdec.
    let (rust_decode, rust_psnr) =
        match bpg_decode::DecoderConfig::new().decode(&bpg, bpg_decode::PixelLayout::Rgb8) {
            Ok(out) => ("ok".to_string(), Some(psnr_bytes(&ref_rgb, &out.data))),
            Err(bpg_decode::DecodeError::Unsupported(_)) => ("unsupported".to_string(), None),
            Err(_) => ("fail".to_string(), None),
        };

    let pixels = (img.width as u64) * (img.height as u64);
    let bpp = (bpg.len() as f64) * 8.0 / pixels as f64;

    if let Some(dir) = out_dir {
        let stem = format!("{}__{}", img.name, config.tag());
        std::fs::write(dir.join(format!("{stem}.bpg")), &bpg)?;
        std::fs::write(dir.join(format!("{stem}.hevc")), &annexb)?;
        let buf = image::RgbImage::from_raw(img.width, img.height, ref_rgb)
            .ok_or("reference buffer size mismatch")?;
        buf.save(dir.join(format!("{stem}.decoded.png")))?;
    }

    Ok(Row {
        image: img.name.to_string(),
        category: img.category.to_string(),
        config: config.tag(),
        width: img.width,
        height: img.height,
        bpg_bytes: bpg.len(),
        hevc_bytes: annexb.len(),
        bpp,
        psnr,
        rust_decode,
        rust_psnr,
    })
}

/// Decode a BPG with stock bpgdec, returning 8-bit RGB. Uses temp files since
/// bpgdec is file-in/file-out.
fn bpgdec_decode(bpgdec: &Path, bpg: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let tmp_bpg = std::env::temp_dir().join(format!("oracle_{nonce}.bpg"));
    let tmp_png = std::env::temp_dir().join(format!("oracle_{nonce}.png"));
    std::fs::write(&tmp_bpg, bpg)?;
    let status = std::process::Command::new(bpgdec)
        .arg("-o")
        .arg(&tmp_png)
        .arg(&tmp_bpg)
        .status()
        .map_err(|e| format!("running {}: {e}", bpgdec.display()))?;
    let rgb = if status.success() {
        image::open(&tmp_png)?.to_rgb8().into_raw()
    } else {
        return Err(format!("bpgdec exited with {status}").into());
    };
    let _ = std::fs::remove_file(&tmp_bpg);
    let _ = std::fs::remove_file(&tmp_png);
    Ok(rgb)
}

/// Build every (image, config) row, in deterministic order.
fn build_rows(
    dusk: &Path,
    bpgdec: &Path,
    out_dir: Option<&Path>,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    if !bpgdec.exists() {
        return Err(format!(
            "bpgdec not found at {} (build it with `make bpgdec` in libbpg-0.9.8/, or pass --bpgdec)",
            bpgdec.display()
        )
        .into());
    }
    let dusk_opt = dusk.exists().then_some(dusk);
    if dusk_opt.is_none() {
        eprintln!("note: {} not found; photo entries skipped", dusk.display());
    }
    let images = corpus::build(dusk_opt);
    let mut rows = Vec::new();
    for img in &images {
        let configs = match img.depth {
            Depth::Eight => CONFIGS_8,
            Depth::Sixteen => CONFIGS_16,
        };
        for &config in configs {
            let row = encode_one(img, config, bpgdec, out_dir)?;
            rows.push(row);
        }
    }
    Ok(rows)
}

fn run_gen(out: &Path, dusk: &Path, bpgdec: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_dir = out.join("corpus");
    let out_files = out.join("out");
    std::fs::create_dir_all(&corpus_dir)?;
    std::fs::create_dir_all(&out_files)?;

    // Save the source corpus PNGs (regenerable, but handy for inspection).
    let dusk_opt = dusk.exists().then_some(dusk);
    for img in corpus::build(dusk_opt) {
        save_source(&corpus_dir, &img)?;
    }

    let rows = build_rows(dusk, bpgdec, Some(&out_files))?;

    // manifest.csv
    let mut csv = String::from(Row::HEADER);
    csv.push('\n');
    for r in &rows {
        csv.push_str(&r.to_csv());
        csv.push('\n');
    }
    std::fs::write(out.join("manifest.csv"), &csv)?;
    std::fs::write(out.join("SETTINGS.md"), settings_doc())?;

    let n_images = rows
        .iter()
        .map(|r| r.image.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let rust_ok = rows.iter().filter(|r| r.rust_decode == "ok").count();
    println!(
        "generated {} encodes across {} images -> {} (rust decoder: {}/{} ok)",
        rows.len(),
        n_images,
        out.display(),
        rust_ok,
        rows.len()
    );
    Ok(())
}

fn run_check(manifest: &Path, dusk: &Path, bpgdec: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let want = std::fs::read_to_string(manifest)
        .map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let rows = build_rows(dusk, bpgdec, None)?;

    // Index committed rows by image+config.
    let mut expected = std::collections::HashMap::new();
    for line in want.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() >= 10 {
            expected.insert((f[0].to_string(), f[2].to_string()), line.to_string());
        }
    }

    let mut problems = Vec::new();
    for r in &rows {
        let Some(exp_line) = expected.get(&(r.image.clone(), r.config.clone())) else {
            problems.push(format!("{} [{}]: not in committed manifest", r.image, r.config));
            continue;
        };
        let f: Vec<&str> = exp_line.split(',').collect();
        let exp_bytes: usize = f[5].parse().unwrap_or(0);
        let exp_psnr: f64 = f[8].parse().unwrap_or(0.0);
        // Byte size is exact for a fixed x265 build/config; PSNR within rounding.
        if r.bpg_bytes != exp_bytes {
            problems.push(format!(
                "{} [{}]: bpg_bytes {} != expected {}",
                r.image, r.config, r.bpg_bytes, exp_bytes
            ));
        }
        if (r.psnr - exp_psnr).abs() > 0.01 {
            problems.push(format!(
                "{} [{}]: psnr {:.3} != expected {:.3}",
                r.image, r.config, r.psnr, exp_psnr
            ));
        }
    }

    if problems.is_empty() {
        println!("check OK: {} encodes match {}", rows.len(), manifest.display());
        Ok(())
    } else {
        for p in &problems {
            eprintln!("MISMATCH {p}");
        }
        Err(format!("{} mismatch(es) vs committed manifest", problems.len()).into())
    }
}

fn save_source(dir: &Path, img: &CorpusImage) -> Result<(), Box<dyn std::error::Error>> {
    match img.depth {
        Depth::Eight => {
            img.rgb8.as_ref().unwrap().save(dir.join(format!("{}.png", img.name)))?;
        }
        Depth::Sixteen => {
            let buf = img.rgb16.as_ref().unwrap().clone();
            let b: image::ImageBuffer<image::Rgb<u16>, _> =
                image::ImageBuffer::from_raw(img.width, img.height, buf)
                    .ok_or("rgb16 buffer size mismatch")?;
            b.save(dir.join(format!("{}.png", img.name)))?;
        }
    }
    Ok(())
}

fn chroma_idc(cf: ChromaFormat) -> u8 {
    match cf {
        ChromaFormat::Gray => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// RGB PSNR (dB) of a decoded 8-bit RGB buffer against the source. 16-bit
/// sources are compared in their 8-bit-reduced form (bpgdec emits 8-bit RGB
/// here), so PSNR for the 10-bit configs is an 8-bit-domain approximation —
/// adequate as a regression signal, not an absolute fidelity number.
fn rgb_psnr_vs(img: &CorpusImage, decoded_rgb: &[u8]) -> f64 {
    let src: Vec<u8> = match img.depth {
        Depth::Eight => img.rgb8.as_ref().unwrap().as_raw().clone(),
        Depth::Sixteen => img
            .rgb16
            .as_ref()
            .unwrap()
            .iter()
            .map(|&v| (v >> 8) as u8)
            .collect(),
    };
    psnr_bytes(&src, decoded_rgb)
}

/// PSNR (dB) between two equal-length 8-bit buffers.
fn psnr_bytes(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sse = 0u64;
    for i in 0..n {
        let d = a[i] as i64 - b[i] as i64;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 99.0; // capped "lossless" sentinel
    }
    let mse = sse as f64 / n as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

fn settings_doc() -> String {
    format!(
        "# Oracle x265 still-image settings (locked)\n\
\n\
These are asserted by `bpg-x265`'s `X265Encoder` AFTER any tune overrides, so\n\
every oracle encode uses exactly this configuration:\n\
\n\
- preset: index {COMPRESS_LEVEL} (`bpg-tools encode` default)\n\
- tune: `ssim` (EncoderTuning::neutral — the byte-identical-to-C-reference path)\n\
- rate control: CQP (`rc.rateControlMode = X265_RC_CQP`, `rc.qp = <qp>`)\n\
- intra only / still: `keyframeMax = 1`, `totalFrames = 1`\n\
- `bRepeatHeaders = 1` (VPS/SPS/PPS precede the slice)\n\
- `bEnableAMP = 1`, `bEnableRectInter = 1` (BPG modified-SPS requires AMP)\n\
- `bEmitInfoSEI = 0`, `decodedPictureHashSEI = 0`\n\
- color: ColorSpace::YCbCr (BT.601, matrix_coefficients = 6), full range\n\
- chroma siting: JPEG (c_h_phase = 1)\n\
\n\
## Metrics\n\
\n\
- `psnr`: **encode quality** — stock `bpgdec` (the ground-truth reference\n\
  decoder) decoded vs the source. High across all chroma formats confirms the\n\
  encoder produces correct 4:2:0/4:2:2/4:4:4 BPG.\n\
- `rust_decode` / `rust_psnr`: the in-repo `bpg-decode` status on the same BPG\n\
  (`ok`/`unsupported`/`fail`) and, when it decodes, its fidelity vs the bpgdec\n\
  reference. The Rust decoder currently supports 4:2:0 only, so 4:2:2/4:4:4\n\
  rows read `unsupported` (see FULL_RUST_CODEC_PLAN.md Phase 1 follow-up).\n\
\n\
The corpus is generated deterministically by `bpg-oracle gen`; `bpg-oracle\n\
check` regenerates and compares bpg_bytes (exact) and encode-quality PSNR\n\
(±0.01 dB) against the committed manifest.csv. Byte-exactness assumes the same\n\
vendored x265 build (x265 4.1, ENABLE_ASSEMBLY=OFF in the dev env) and the same\n\
bpgdec.\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_identical_is_capped() {
        let a = vec![10u8, 20, 30, 40];
        assert_eq!(psnr_bytes(&a, &a), 99.0);
    }

    #[test]
    fn psnr_decreases_with_error() {
        let a = vec![100u8; 64];
        let mut b = a.clone();
        for v in b.iter_mut() {
            *v = 110;
        }
        let near = psnr_bytes(&a, &b);
        for v in b.iter_mut() {
            *v = 160;
        }
        let far = psnr_bytes(&a, &b);
        assert!(near > far, "more error must lower PSNR ({near} vs {far})");
    }
}
