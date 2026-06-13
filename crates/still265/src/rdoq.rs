//! Experimental single-scan rate-distortion-optimized quantization (RDOQ).
//!
//! This is an *alternative* to the exact greedy per-coefficient refinement in
//! `encoder::refine_levels_rdoq_limited`. Instead of mutating one coefficient
//! at a time and re-pricing the whole transform block, it makes each
//! coefficient's level decision once, in a single reverse (high-frequency ->
//! DC) scan, pricing the candidate levels with CABAC entropy-bit estimates
//! taken from the block's entry context state (the standard HM/x265 RDOQ
//! approximation: contexts are frozen for pricing rather than re-coded).
//!
//! It reuses the *same* RD framework as the greedy path — coefficient-domain
//! SSE distortion, the same `lambda`, the same 1/32768-bit rate units — so its
//! decisions are directly comparable. Selected per [`Effort`] tier
//! (`Best` keeps exact greedy; `Balanced`/`Fast` use this) and overridable via
//! `BPG_RDOQ_SINGLESCAN`. It is *not* byte-identical to the greedy path.
//!
//! Stages implemented:
//! - **(a)** per-coefficient level decision (reverse scan, local RD).
//! - **(b)** last-significant-coefficient-position optimization: after the
//!   level decision, choose the scan position to terminate coding at (zeroing
//!   the higher-frequency tail, or the whole block) by minimizing total RD —
//!   this is what recovers the low-bitrate efficiency stage (a) left behind.
//!
//! Still TODO **(c)**: per-sub-block `coded_sub_block_flag` all-zero forcing.

use crate::cabac::CabacEstimator;
use crate::contexts::{ctx, Contexts};
use crate::residual::{calc_sig_coeff_flag_ctx, get_scan_4x4, get_scan_sub_block, ScanOrder};
use crate::transform::DequantParams;

/// x265 `g_quantScales` (forward quant scale per `qp % 6`).
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
const MAX_TR_DYNAMIC_RANGE: i32 = 15;
const QUANT_SHIFT: i32 = 14;

/// One bypass bin in 1/32768-bit units.
const BYPASS_BITS: u64 = CabacEstimator::SCALE;

/// Number of `coeff_abs_level_remaining` bins for `value` at Rice parameter
/// `rice` (truncated-Rice prefix + EGk), matching `encode_coeff_abs_level_remaining`.
fn rice_bins(value: u32, rice: u32) -> u32 {
    if value < (4u32 << rice) {
        (value >> rice) + 1 + rice
    } else {
        let mut p = 4u32;
        loop {
            let base = ((1u32 << (p - 3)) + 2) << rice;
            let width = 1u32 << (p - 3 + rice);
            if value < base + width {
                return p + 1 + (p - 3 + rice);
            }
            p += 1;
        }
    }
}

/// Estimated bits for one axis of `last_sig_coeff_{x,y}` (context-coded
/// truncated-unary prefix + bypass suffix), mirroring `encode_last_prefix` /
/// `encode_bypass_bits` but counting entropy bits instead of coding.
fn last_axis_bits(ctxs: &Contexts, value: u32, log2_size: u8, c_idx: u8, is_x: bool) -> u64 {
    let ctx_base = if is_x {
        ctx::LAST_SIG_COEFF_X_PREFIX
    } else {
        ctx::LAST_SIG_COEFF_Y_PREFIX
    };
    let (ctx_offset, ctx_shift) = if c_idx == 0 {
        let offset = 3 * (log2_size as usize - 2) + ((log2_size as usize - 1) >> 2);
        let shift = (log2_size + 1) >> 2;
        (offset, shift)
    } else {
        (15usize, log2_size - 2)
    };
    let max_prefix = ((log2_size << 1) - 1) as u32;
    let (prefix, n_bits) = if value <= 3 {
        (value, 0u8)
    } else {
        let group = 31 - value.leading_zeros();
        (2 * group + ((value >> (group - 1)) & 1), (group - 1) as u8)
    };

    let mut bits = 0u64;
    for j in 0..prefix {
        let ci = ctx_base + ctx_offset + (j as usize >> ctx_shift as usize);
        bits += ctxs.models[ci].entropy_bits(1) as u64;
    }
    if prefix < max_prefix {
        let ci = ctx_base + ctx_offset + (prefix as usize >> ctx_shift as usize);
        bits += ctxs.models[ci].entropy_bits(0) as u64;
    }
    if value > 3 {
        bits += n_bits as u64 * BYPASS_BITS;
    }
    bits
}

/// Total estimated `last_sig_coeff_x/y` bits for a last position at `(x, y)`.
fn last_pos_bits(ctxs: &Contexts, x: u32, y: u32, log2_size: u8, c_idx: u8, scan: ScanOrder) -> u64 {
    let (raw_x, raw_y) = if scan == ScanOrder::Vertical {
        (y, x)
    } else {
        (x, y)
    };
    last_axis_bits(ctxs, raw_x, log2_size, c_idx, true)
        + last_axis_bits(ctxs, raw_y, log2_size, c_idx, false)
}

/// Per-coefficient state captured during the level-decision scan, indexed by
/// forward (DC -> high-frequency) combined scan rank.
#[derive(Clone, Copy, Default)]
struct CoeffRec {
    idx: usize, // row-major coefficient index
    x: u32,
    y: u32,
    level: i16,   // chosen signed level (0 if quantized away)
    dist0: f64,   // distortion if zeroed
    dist_l: f64,  // distortion at chosen level
    rd_normal: f64, // RD cost coded as a normal coefficient (sig flag + level)
    rd_last: f64,   // RD cost coded as the last significant coeff (no sig flag)
}

/// Single-scan RDOQ: choose quantized levels for `coeffs` directly from the
/// transform coefficients. Returns `(levels, nnz)` with the same layout as
/// `transform::quantize`.
#[allow(clippy::too_many_arguments)]
pub fn rdoq_single_scan(
    ctxs: &Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    scan_order: ScanOrder,
    lambda: f64,
) -> (Vec<i16>, u32) {
    let size = 1usize << log2_size;
    let scan_sub = get_scan_sub_block(log2_size, scan_order);
    let scan_pos = get_scan_4x4(scan_order);
    let scan_idx = scan_order as u8;
    let sb_width = size / 4;

    // Forward-quant scaling (matches `transform::quantize`).
    let per = qp / 6;
    let rem = (qp % 6) as usize;
    let transform_shift = MAX_TR_DYNAMIC_RANGE - bit_depth as i32 - log2_size as i32;
    let qbits = QUANT_SHIFT + per + transform_shift;
    let scale = QUANT_SCALE[rem];
    let round = 1i64 << (qbits - 1); // round-to-nearest center for the candidate level

    let dq = DequantParams::new(log2_size, qp, bit_depth);
    let lam = lambda / CabacEstimator::SCALE as f64;
    let bits = |ci: usize, bin: u8| -> u64 { ctxs.models[ci].entropy_bits(bin) as u64 };

    // Frozen representative greater1/greater2 contexts (greater1_ctx = 1, first
    // significant coefficient; prev_subblock_had_gt1 = 0).
    let ctx_set = if c_idx > 0 { 0usize } else { 2 };
    let gt1_ci = ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + if c_idx > 0 { 16 } else { 0 } + ctx_set * 4 + 1;
    let gt2_ci = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + if c_idx > 0 { 4 } else { 0 } + ctx_set;
    // coeff_abs_level bits (greater1 + greater2 + remaining + sign) for abs level l>=1.
    let magnitude_bits = |l: u32| -> u64 {
        let mag = match l {
            1 => bits(gt1_ci, 0),
            2 => bits(gt1_ci, 1) + bits(gt2_ci, 0),
            _ => bits(gt1_ci, 1) + bits(gt2_ci, 1) + rice_bins(l - 3, 0) as u64 * BYPASS_BITS,
        };
        mag + BYPASS_BITS // sign
    };

    // Forward combined scan rank for every in-bounds position.
    let mut recs: Vec<CoeffRec> = Vec::with_capacity(size * size);
    let mut rank_of = vec![usize::MAX; size * size];
    for &(sbx, sby) in scan_sub.iter() {
        for &(px, py) in scan_pos.iter() {
            let x = sbx as usize * 4 + px as usize;
            let y = sby as usize * 4 + py as usize;
            if x < size && y < size {
                rank_of[y * size + x] = recs.len();
                recs.push(CoeffRec {
                    idx: y * size + x,
                    x: x as u32,
                    y: y as u32,
                    ..Default::default()
                });
            }
        }
    }

    // --- Stage (a): per-coefficient level decision, reverse sub-block scan. ---
    let mut coded_sb_flags = [[false; 8]; 8];
    for &(sbx, sby) in scan_sub.iter().rev() {
        let sbx = sbx as usize;
        let sby = sby as usize;
        let right = sbx + 1 < sb_width && coded_sb_flags[sby][sbx + 1];
        let below = sby + 1 < sb_width && coded_sb_flags[sby + 1][sbx];
        let prev_csbf = (right as u8) | ((below as u8) << 1);

        let mut sb_has_sig = false;
        for pos in (0..16).rev() {
            let (px, py) = scan_pos[pos];
            let x = sbx * 4 + px as usize;
            let y = sby * 4 + py as usize;
            if x >= size || y >= size {
                continue;
            }
            let rank = rank_of[y * size + x];
            let coeff = coeffs[y * size + x];
            let abs = coeff.unsigned_abs() as i64;
            let q = ((abs * scale + round) >> qbits).min(32767) as i32;

            let sig_ci =
                calc_sig_coeff_flag_ctx(x as u8, y as u8, log2_size, c_idx, scan_idx, prev_csbf);
            let sig0 = bits(sig_ci, 0);
            let sig1 = bits(sig_ci, 1);

            let dist0 = (abs * abs) as f64;
            let mut best_l = 0i64;
            let mut best_levbits = 0u64;
            let mut best_distl = dist0;
            let mut best_cost = dist0 + lam * sig0 as f64;
            for l in (q - 1).max(1)..=(q + 1) {
                let dl = abs - dq.apply(l as i16) as i64;
                let dist = (dl * dl) as f64;
                let mb = magnitude_bits(l as u32);
                let cost = dist + lam * (sig1 + mb) as f64;
                if cost < best_cost {
                    best_cost = cost;
                    best_l = l as i64;
                    best_levbits = mb;
                    best_distl = dist;
                }
            }

            let rec = &mut recs[rank];
            rec.dist0 = dist0;
            rec.dist_l = best_distl;
            if best_l != 0 {
                rec.level = if coeff < 0 { -best_l } else { best_l } as i16;
                rec.rd_normal = best_distl + lam * (sig1 + best_levbits) as f64;
                rec.rd_last = best_distl + lam * best_levbits as f64;
                sb_has_sig = true;
            } else {
                rec.level = 0;
                rec.rd_normal = dist0 + lam * sig0 as f64;
                rec.rd_last = f64::INFINITY; // a zero coeff can never be the last
            }
        }
        if sb_has_sig {
            coded_sb_flags[sby][sbx] = true;
        }
    }

    // --- Stage (b): last-significant-position optimization. ---
    // Choose the forward scan rank `p` (a significant coefficient) to be the
    // last coded coefficient: ranks < p are coded normally, p is coded as last
    // (its sig flag inferred), ranks > p are zeroed (distortion only). Also
    // consider zeroing the whole block (cbf = 0).
    let total = recs.len();
    // Suffix distortion of zeroing rank..end.
    let mut suff_zero = vec![0f64; total + 1];
    for k in (0..total).rev() {
        suff_zero[k] = suff_zero[k + 1] + recs[k].dist0;
    }
    // Prefix RD of coding ranks 0..k as normal coefficients.
    let mut pref_normal = vec![0f64; total + 1];
    for k in 0..total {
        pref_normal[k + 1] = pref_normal[k] + recs[k].rd_normal;
    }

    // Baseline: all-zero block (cbf = 0), no residual bits.
    let mut best_cost = suff_zero[0];
    let mut best_last: isize = -1;
    for p in 0..total {
        if recs[p].level == 0 {
            continue;
        }
        let lp = lam * last_pos_bits(ctxs, recs[p].x, recs[p].y, log2_size, c_idx, scan_order) as f64;
        let cost = pref_normal[p] + recs[p].rd_last + suff_zero[p + 1] + lp;
        if cost < best_cost {
            best_cost = cost;
            best_last = p as isize;
        }
    }

    // Emit levels, zeroing everything after the chosen last position.
    let mut levels = vec![0i16; size * size];
    let mut nnz = 0u32;
    if best_last >= 0 {
        let last = best_last as usize;
        for (rank, rec) in recs.iter().enumerate().take(last + 1) {
            if rec.level != 0 {
                levels[rec.idx] = rec.level;
                nnz += 1;
            }
            let _ = rank;
        }
    }

    (levels, nnz)
}
