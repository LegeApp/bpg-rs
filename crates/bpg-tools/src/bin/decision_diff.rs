//! `bpg-decision-diff` — diff two encoders' per-block decisions for the same image.
//!
//! The decoder is bit-exact, so decoding a `.bpg` recovers every decision the
//! producing encoder made (intra mode, TU partition, quantised levels, residual
//! energy). This tool decodes a still265 `.bpg` and a reference (`bpgenc`) `.bpg`
//! of the *same* source at matched QP, classifies each block by its source-luma
//! local variance (smooth / mid / textured), and compares — without
//! instrumenting the reference encoder — *where* the two diverge in prediction
//! (intra mode), transform-unit structure, and coded residual energy.
//!
//! Usage:
//!   bpg-decision-diff --source img.png --rust rust.bpg --c c.bpg
//!
//! Motivation: the ~1 dB equal-bitrate luma gap to the reference is uniform-QP
//! vs uniform-QP (AQ is inert in bpgenc's intra CQP path), broadband, and
//! texture-concentrated. "Reference keeps more coefficients but reconstructs no
//! better" points at the predicted block — this tool tests it directly.

use std::path::PathBuf;

use bpg_hevc_decode::hevc::debug::{self, DecisionLog};

/// Variance thresholds for region classification (match the texture-gap docs).
const SMOOTH_VAR: f64 = 20.0;
const TEXTURED_VAR: f64 = 200.0;

#[derive(Clone, Copy, PartialEq)]
enum Class {
    Smooth,
    Mid,
    Textured,
}
const CLASSES: [Class; 3] = [Class::Smooth, Class::Mid, Class::Textured];

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Smooth => "smooth (var<20)",
            Class::Mid => "mid",
            Class::Textured => "textured (var>200)",
        }
    }
}

fn classify(var: f64) -> Class {
    if var < SMOOTH_VAR {
        Class::Smooth
    } else if var > TEXTURED_VAR {
        Class::Textured
    } else {
        Class::Mid
    }
}

struct Luma {
    w: usize,
    h: usize,
    /// Full-resolution luma (BT.601), one f64 per pixel.
    y: Vec<f64>,
}

impl Luma {
    /// Population variance of source luma over a block, clamped to image bounds.
    fn block_variance(&self, x0: u32, y0: u32, size: usize) -> f64 {
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let mut n = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        for yy in y0..(y0 + size).min(self.h) {
            let row = yy * self.w;
            for xx in x0..(x0 + size).min(self.w) {
                let v = self.y[row + xx];
                n += 1.0;
                sum += v;
                sum2 += v * v;
            }
        }
        if n < 1.0 {
            return 0.0;
        }
        (sum2 / n) - (sum / n) * (sum / n)
    }
}

fn load_luma(path: &PathBuf) -> Result<Luma, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let mut decoder = zune_png::PngDecoder::new(&data);
    let pixels = decoder
        .decode_raw()
        .map_err(|e| format!("PNG decode error: {e}"))?;
    let (w, h) = decoder.get_dimensions().ok_or("PNG: missing dimensions")?;
    let cs = decoder.get_colorspace().ok_or("PNG: missing colorspace")?;
    let chans = cs.num_components();
    let y: Vec<f64> = match chans {
        1 => pixels.iter().map(|&v| v as f64).collect(),
        _ => pixels
            .chunks_exact(chans)
            .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
            .collect(),
    };
    Ok(Luma { w, h, y })
}

fn decode_with_log(path: &PathBuf) -> Result<DecisionLog, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    debug::enable_decision_log();
    let _ = bpg_decode::DecoderConfig::new()
        .decode_to_frame(&bytes)
        .map_err(|e| format!("bpg decode error: {e:?}"))?;
    debug::take_decision_log().ok_or_else(|| "decision log was not collected".into())
}

#[derive(Default)]
struct ClassStats {
    // Luma PU statistics, area-weighted by pixels.
    pu_area: u64,
    planar_area: u64,
    dc_area: u64,
    angular_area: u64,
    // Luma TU statistics.
    tu_count: u64,
    tu_area: u64,
    tu4_area: u64,
    tu8_area: u64,
    tu16_area: u64,
    tu32_area: u64,
    nz: u64,
    abs_level_sum: u64,
    residual_energy: u64,
    /// Per-TU-size coefficient stats (index by log2_size 2..=5 → 0..=3):
    /// (tu_area, nz, abs_level_sum, residual_energy). Lets us compare
    /// coefficient strength at *matched* TU size (RDOQ/lambda vs TU-shape).
    by_size: [[u64; 4]; 4],
}

fn aggregate(log: &DecisionLog, luma: &Luma) -> [ClassStats; 3] {
    let mut stats: [ClassStats; 3] = Default::default();
    let idx = |c: Class| match c {
        Class::Smooth => 0,
        Class::Mid => 1,
        Class::Textured => 2,
    };

    for pu in &log.pus {
        let size = 1usize << pu.log2_size;
        let area = (size * size) as u64;
        let var = luma.block_variance(pu.x, pu.y, size);
        let s = &mut stats[idx(classify(var))];
        s.pu_area += area;
        match pu.luma_mode {
            0 => s.planar_area += area,
            1 => s.dc_area += area,
            _ => s.angular_area += area,
        }
    }

    for tu in &log.tus {
        if tu.c_idx != 0 {
            continue; // luma only
        }
        let size = 1usize << tu.log2_size;
        let area = (size * size) as u64;
        let var = luma.block_variance(tu.x, tu.y, size);
        let s = &mut stats[idx(classify(var))];
        s.tu_count += 1;
        s.tu_area += area;
        match tu.log2_size {
            2 => s.tu4_area += area,
            3 => s.tu8_area += area,
            4 => s.tu16_area += area,
            _ => s.tu32_area += area,
        }
        s.nz += tu.nz as u64;
        s.abs_level_sum += tu.abs_level_sum;
        s.residual_energy += tu.residual_energy;
        let sz = (tu.log2_size as usize).clamp(2, 5) - 2;
        s.by_size[sz][0] += area;
        s.by_size[sz][1] += tu.nz as u64;
        s.by_size[sz][2] += tu.abs_level_sum;
        s.by_size[sz][3] += tu.residual_energy;
    }

    stats
}

/// Build a per-pixel map of the luma-TU log2 size covering each pixel.
fn tu_log2_map(log: &DecisionLog, w: usize, h: usize) -> Vec<u8> {
    let mut map = vec![0u8; w * h];
    for tu in &log.tus {
        if tu.c_idx != 0 {
            continue;
        }
        let size = 1usize << tu.log2_size;
        for yy in (tu.y as usize)..(tu.y as usize + size).min(h) {
            let row = yy * w;
            for xx in (tu.x as usize)..(tu.x as usize + size).min(w) {
                map[row + xx] = tu.log2_size;
            }
        }
    }
    map
}

/// Compare rust vs C luma-TU size on a partition-independent 16×16 grid,
/// classified by source variance. Answers "where does rust split (smaller TU)
/// while C keeps a larger TU?" — directly actionable for a TU-split policy fix.
fn print_tu_map_diff(rust: &DecisionLog, c: &DecisionLog, luma: &Luma) {
    let (w, h) = (luma.w, luma.h);
    let rmap = tu_log2_map(rust, w, h);
    let cmap = tu_log2_map(c, w, h);
    // Per class: cells where rust TU smaller / equal / larger than C, plus mean log2.
    let mut cells = [[0u64; 3]; 3]; // [class][rust<c, ==, rust>c]
    let mut sum_r = [0f64; 3];
    let mut sum_c = [0f64; 3];
    let mut n = [0u64; 3];
    let step = 16usize;
    for cy in (0..h).step_by(step) {
        for cx in (0..w).step_by(step) {
            let idx = (cy + step / 2).min(h - 1) * w + (cx + step / 2).min(w - 1);
            let (r, cc) = (rmap[idx], cmap[idx]);
            if r == 0 || cc == 0 {
                continue; // uncoded (cbf=0) cell in one of them
            }
            let var = luma.block_variance(cx as u32, cy as u32, step);
            let k = match classify(var) {
                Class::Smooth => 0,
                Class::Mid => 1,
                Class::Textured => 2,
            };
            match r.cmp(&cc) {
                std::cmp::Ordering::Less => cells[k][0] += 1,
                std::cmp::Ordering::Equal => cells[k][1] += 1,
                std::cmp::Ordering::Greater => cells[k][2] += 1,
            }
            sum_r[k] += r as f64;
            sum_c[k] += cc as f64;
            n[k] += 1;
        }
    }
    println!(
        "\n================ luma TU size: rust vs C (16×16 grid, coded cells) ================"
    );
    println!(
        "  {:<20} {:>10} {:>12} {:>10} {:>12} {:>12}",
        "region", "cells", "rust splits%", "equal%", "C splits%", "mean log2 r/C"
    );
    for (k, class) in CLASSES.iter().enumerate() {
        let tot = n[k].max(1);
        println!(
            "  {:<20} {:>10} {:>11.1}% {:>9.1}% {:>11.1}% {:>6.2}/{:.2}",
            class.name(),
            n[k],
            pct(cells[k][0], tot),
            pct(cells[k][1], tot),
            pct(cells[k][2], tot),
            sum_r[k] / tot as f64,
            sum_c[k] / tot as f64,
        );
    }
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        100.0 * num as f64 / den as f64
    }
}

fn print_class(class: Class, r: &ClassStats, c: &ClassStats) {
    println!("\n=== {} ===", class.name());
    println!(
        "  {:<26} {:>14} {:>14}",
        "metric", "rust(still265)", "c(bpgenc)"
    );
    let row = |label: &str, rv: f64, cv: f64| {
        println!("  {label:<26} {rv:>14.2} {cv:>14.2}");
    };
    // Prediction: luma-mode usage, area-weighted.
    row(
        "PU planar %",
        pct(r.planar_area, r.pu_area),
        pct(c.planar_area, c.pu_area),
    );
    row(
        "PU dc %",
        pct(r.dc_area, r.pu_area),
        pct(c.dc_area, c.pu_area),
    );
    row(
        "PU angular %",
        pct(r.angular_area, r.pu_area),
        pct(c.angular_area, c.pu_area),
    );
    // Transform-unit structure, area-weighted.
    row(
        "TU 4x4 %",
        pct(r.tu4_area, r.tu_area),
        pct(c.tu4_area, c.tu_area),
    );
    row(
        "TU 8x8 %",
        pct(r.tu8_area, r.tu_area),
        pct(c.tu8_area, c.tu_area),
    );
    row(
        "TU 16x16 %",
        pct(r.tu16_area, r.tu_area),
        pct(c.tu16_area, c.tu_area),
    );
    row(
        "TU 32x32 %",
        pct(r.tu32_area, r.tu_area),
        pct(c.tu32_area, c.tu_area),
    );
    // Coded residual: energy and levels per coded luma pixel.
    let rpx = r.tu_area.max(1) as f64;
    let cpx = c.tu_area.max(1) as f64;
    row(
        "residual energy / px",
        r.residual_energy as f64 / rpx,
        c.residual_energy as f64 / cpx,
    );
    row("nz coeffs / px", r.nz as f64 / rpx, c.nz as f64 / cpx);
    row(
        "abs level sum / px",
        r.abs_level_sum as f64 / rpx,
        c.abs_level_sum as f64 / cpx,
    );
    row("coded luma area (px)", r.tu_area as f64, c.tu_area as f64);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut source: Option<PathBuf> = None;
    let mut rust: Option<PathBuf> = None;
    let mut cref: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--source" => source = args.next().map(PathBuf::from),
            "--rust" => rust = args.next().map(PathBuf::from),
            "--c" => cref = args.next().map(PathBuf::from),
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let source = source.ok_or("missing --source <png>")?;
    let rust = rust.ok_or("missing --rust <bpg>")?;
    let cref = cref.ok_or("missing --c <bpg>")?;

    let luma = load_luma(&source)?;
    println!("source {}x{} ({} px)", luma.w, luma.h, luma.w * luma.h);

    let rust_log = decode_with_log(&rust)?;
    let c_log = decode_with_log(&cref)?;
    println!(
        "rust: {} PUs, {} luma TUs | c: {} PUs, {} luma TUs",
        rust_log.pus.len(),
        rust_log.tus.iter().filter(|t| t.c_idx == 0).count(),
        c_log.pus.len(),
        c_log.tus.iter().filter(|t| t.c_idx == 0).count(),
    );

    // Luma vs chroma coefficient share — tests whether still265 over-spends on
    // chroma vs the reference (the chroma-QP-offset hypothesis). nz/abs-level are
    // bit-cost proxies; if rust's chroma share >> C's, chroma is a real lever.
    let plane_tot = |log: &DecisionLog, c_idx_pred: fn(u8) -> bool| -> (u64, u64) {
        log.tus
            .iter()
            .filter(|t| c_idx_pred(t.c_idx))
            .fold((0u64, 0u64), |(nz, lvl), t| {
                (nz + t.nz as u64, lvl + t.abs_level_sum)
            })
    };
    println!("\n================ luma vs chroma coefficient share ================");
    println!("  {:<8} {:>14} {:>14}", "", "rust(still265)", "c(bpgenc)");
    for (label, pred) in [
        ("luma nz", (|c| c == 0) as fn(u8) -> bool),
        ("chroma nz", (|c| c != 0) as fn(u8) -> bool),
    ] {
        let (rn, _) = plane_tot(&rust_log, pred);
        let (cn, _) = plane_tot(&c_log, pred);
        println!("  {label:<8} {rn:>14} {cn:>14}");
    }
    let (rl_nz, rl_lvl) = plane_tot(&rust_log, |c| c == 0);
    let (rc_nz, rc_lvl) = plane_tot(&rust_log, |c| c != 0);
    let (cl_nz, cl_lvl) = plane_tot(&c_log, |c| c == 0);
    let (cc_nz, cc_lvl) = plane_tot(&c_log, |c| c != 0);
    println!(
        "  chroma nz share:   rust {:.1}%   c {:.1}%",
        pct(rc_nz, rl_nz + rc_nz),
        pct(cc_nz, cl_nz + cc_nz)
    );
    println!(
        "  chroma level share: rust {:.1}%   c {:.1}%",
        pct(rc_lvl, rl_lvl + rc_lvl),
        pct(cc_lvl, cl_lvl + cc_lvl)
    );

    print_tu_map_diff(&rust_log, &c_log, &luma);

    let rust_stats = aggregate(&rust_log, &luma);
    let c_stats = aggregate(&c_log, &luma);

    for (i, class) in CLASSES.iter().enumerate() {
        print_class(*class, &rust_stats[i], &c_stats[i]);
    }

    // Coefficient strength at MATCHED TU size — isolates RDOQ/level-selection
    // from TU-shape. If C keeps higher abs-level / more nz at the *same* TU
    // size, the difference is coefficient behavior, not partitioning.
    println!("\n================ coefficient strength at matched TU size ================");
    println!("(per coded luma px, within each region × TU size)\n");
    for (i, class) in CLASSES.iter().enumerate() {
        println!("--- {} ---", class.name());
        println!(
            "  {:<8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "TU", "rust nz/px", "c nz/px", "rust lvl/px", "c lvl/px", "rust E/px", "c E/px"
        );
        for sz in 0..4 {
            let r = &rust_stats[i].by_size[sz];
            let c = &c_stats[i].by_size[sz];
            if r[0] == 0 && c[0] == 0 {
                continue;
            }
            let rpx = r[0].max(1) as f64;
            let cpx = c[0].max(1) as f64;
            println!(
                "  {:<8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.2} {:>12.2}",
                format!("{}x{}", 1 << (sz + 2), 1 << (sz + 2)),
                r[1] as f64 / rpx,
                c[1] as f64 / cpx,
                r[2] as f64 / rpx,
                c[2] as f64 / cpx,
                r[3] as f64 / rpx,
                c[3] as f64 / cpx,
            );
        }
    }
    Ok(())
}
