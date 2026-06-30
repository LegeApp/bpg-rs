//! Per-CTU partition/mode oracle dump. Decodes a `.bpg` and prints the luma
//! PU partition (position, size, intra mode) and luma TU partition (position,
//! size, nonzeros, abs-level-sum, residual energy) inside one 64x64 CTU. Run on
//! both a still265 and a C `bpgenc` stream of the same source to compare exactly
//! which fine partitions + modes each encoder landed on in a given block.
//!
//!   bpg-ctu-dump <file.bpg> <ctu_px_x> <ctu_px_y> [size=64]

use bpg_hevc_decode::hevc::debug::{self, DecisionLog};
use std::path::PathBuf;

const MODE_NAMES: [&str; 2] = ["PLANAR", "DC"];

fn mode_name(m: u8) -> String {
    match m {
        0 | 1 => MODE_NAMES[m as usize].to_string(),
        2..=34 => format!("ANG{m}"),
        _ => format!("?{m}"),
    }
}

fn decode_with_log(path: &PathBuf) -> Result<DecisionLog, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    debug::enable_decision_log();
    let _ = bpg_decode::DecoderConfig::new()
        .decode_to_frame(&bytes)
        .map_err(|e| format!("bpg decode error: {e:?}"))?;
    debug::take_decision_log().ok_or_else(|| "decision log was not collected".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: bpg-ctu-dump <file.bpg> <x> <y> [size]")?,
    );
    let cx: u32 = args.next().ok_or("missing ctu_px_x")?.parse()?;
    let cy: u32 = args.next().ok_or("missing ctu_px_y")?.parse()?;
    let size: u32 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(64);

    let log = decode_with_log(&path)?;
    let in_region = |x: u32, y: u32| x >= cx && x < cx + size && y >= cy && y < cy + size;

    let mut pus: Vec<_> = log.pus.iter().filter(|p| in_region(p.x, p.y)).collect();
    pus.sort_by_key(|p| (p.y, p.x));
    let mut tus: Vec<_> = log
        .tus
        .iter()
        .filter(|t| t.c_idx == 0 && in_region(t.x, t.y))
        .collect();
    tus.sort_by_key(|t| (t.y, t.x));

    println!("== {} CTU @ ({cx},{cy}) {size}x{size} ==", path.display());

    // PU partition: count by size, then list with modes.
    let mut pu_by_sz = [0u32; 7];
    for p in &pus {
        pu_by_sz[p.log2_size as usize] += 1;
    }
    print!("  luma PUs: {} total [", pus.len());
    for log2 in (2..=6).rev() {
        if pu_by_sz[log2] > 0 {
            print!("{}x{}:{} ", 1 << log2, 1 << log2, pu_by_sz[log2]);
        }
    }
    println!("]");
    for p in &pus {
        let n = 1u32 << p.log2_size;
        println!(
            "    PU ({:4},{:4}) {:2}x{:<2} {}",
            p.x,
            p.y,
            n,
            n,
            mode_name(p.luma_mode)
        );
    }

    // TU partition: count by size + aggregate residual.
    let mut tu_by_sz = [0u32; 7];
    let (mut tot_nz, mut tot_lvl, mut tot_e, mut tot_px) = (0u64, 0u64, 0u64, 0u64);
    for t in &tus {
        tu_by_sz[t.log2_size as usize] += 1;
        let px = 1u64 << (2 * t.log2_size);
        tot_nz += t.nz as u64;
        tot_lvl += t.abs_level_sum;
        tot_e += t.residual_energy;
        tot_px += px;
    }
    print!("  luma TUs: {} total [", tus.len());
    for log2 in (2..=5).rev() {
        if tu_by_sz[log2] > 0 {
            print!("{}x{}:{} ", 1 << log2, 1 << log2, tu_by_sz[log2]);
        }
    }
    println!("]");
    let px = tot_px.max(1) as f64;
    println!(
        "  residual: nz/px={:.3}  lvl/px={:.3}  E/px={:.1}  (coded {} px)",
        tot_nz as f64 / px,
        tot_lvl as f64 / px,
        tot_e as f64 / px,
        tot_px
    );
    Ok(())
}
