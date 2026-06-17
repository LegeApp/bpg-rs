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
//! (`Placebo`/`Reference` keep exact greedy; the practical tiers use this) and
//! overridable via `BPG_RDOQ_SINGLESCAN`. It is *not* byte-identical to the
//! greedy path.
//!
//! Stages implemented:
//! - **(a)** per-coefficient level decision (reverse scan, local RD).
//! - **(b)** last-significant-coefficient-position optimization: after the
//!   level decision, choose the scan position to terminate coding at (zeroing
//!   the higher-frequency tail, or the whole block) by minimizing total RD —
//!   this is what recovers the low-bitrate efficiency stage (a) left behind.
//! - **(c)** per-sub-block `coded_sub_block_flag` all-zero forcing: for each
//!   middle sub-block (explicit-CSBF, i.e. strictly between the DC and the
//!   last-significant sub-block), zero the whole sub-block when CSBF=0 + the
//!   dropped-coefficient distortion beats coding it. Net RD win on the test
//!   corpus (occasional sub-1% local loss from the frozen-context estimate).

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
pub(crate) fn rice_bins(value: u32, rice: u32) -> u32 {
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

/// Magnitude (`coeff_abs_level_greater1/2` + `coeff_abs_level_remaining`) bits
/// for an absolute `level >= 1`, given the *tracked* coding-group state. Faithful
/// port of x265 `getICRateCost` (common/quant.cpp): `gt1`/`gt2` are the
/// greater1/greater2 flag entropy bits (`[0]`/`[1]`) for the current
/// `4*ctxSet+c1` / `ctxSet+c2` contexts, `c1c2idx` selects the `{1,2,1,3}`
/// `base_level`, and `go_rice` is the adapted Rice parameter. Excludes the sign
/// bypass bit (the caller adds it).
#[inline]
fn ic_level_bits(
    level: u32,
    base_level: u32,
    gt1: [u64; 2],
    gt2: [u64; 2],
    c1c2idx: u32,
    go_rice: u32,
) -> u64 {
    let diff = level as i64 - base_level as i64;
    if diff < 0 {
        // level is 1 or 2 (below base_level): just the greater1 flag, plus the
        // greater2 flag when level == 2.
        gt1[(level == 2) as usize] + if level == 2 { gt2[0] } else { 0 }
    } else {
        // level >= base_level: greater1/greater2 "=1" flags (when still within
        // their per-group budgets) plus the remaining truncated-Rice/EGk code.
        let c1c2_rate =
            (if c1c2idx & 1 != 0 { gt1[1] } else { 0 }) + (if c1c2idx == 3 { gt2[1] } else { 0 });
        rice_bins(diff as u32, go_rice) as u64 * BYPASS_BITS + c1c2_rate
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
fn last_pos_bits(
    ctxs: &Contexts,
    x: u32,
    y: u32,
    log2_size: u8,
    c_idx: u8,
    scan: ScanOrder,
) -> u64 {
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
    level: i16,     // chosen signed level (0 if quantized away)
    dist0: f64,     // distortion if zeroed
    dist_l: f64,    // distortion at chosen level
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
    level2: bool,
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
    // significant coefficient; prev_subblock_had_gt1 = 0). Used when `level2` is
    // off (the legacy single-scan rate model).
    let ctx_set = if c_idx > 0 { 0usize } else { 2 };
    let gt1_ci =
        ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + if c_idx > 0 { 16 } else { 0 } + ctx_set * 4 + 1;
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

    // `level2` (x265 `rdoqQuant`) greater1/greater2 context banks. Per-coefficient
    // indices are `gt1_base + 4*ctxSet + c1` and `gt2_base + ctxSet` (the `+c2`
    // term in x265 is only read when c2 == 0, so it folds into `ctxSet`).
    let gt1_base = ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + if c_idx > 0 { 16 } else { 0 };
    let gt2_base = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + if c_idx > 0 { 4 } else { 0 };

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
    // `level2` coding-group state carried across groups (x265 `rdoqQuant`):
    // `prev_cg_c1` is the previous coded group's final greater1 context, used to
    // bump `ctxSet` for the next group.
    let mut prev_cg_c1 = 1u32;
    for &(sbx, sby) in scan_sub.iter().rev() {
        let sbx = sbx as usize;
        let sby = sby as usize;
        let right = sbx + 1 < sb_width && coded_sb_flags[sby][sbx + 1];
        let below = sby + 1 < sb_width && coded_sb_flags[sby + 1][sbx];
        let prev_csbf = (right as u8) | ((below as u8) << 1);

        // Per-group greater1/greater2 state (x265 quant.cpp:778-866). `ctx_set`
        // is 2 for non-DC luma groups, 0 otherwise, +1 if the previous group
        // ended with c1 == 0. The greater1/greater2 flag budgets (`c1_idx < 8`,
        // `c2_idx == 0`) and the adapted Rice parameter (`go_rice`) reset here.
        let cg_is_not_dc = !(sbx == 0 && sby == 0);
        let mut ctx_set2 = if cg_is_not_dc && c_idx == 0 { 2u32 } else { 0 };
        if prev_cg_c1 == 0 {
            ctx_set2 += 1;
        }
        let mut c1 = 1u32;
        let mut c2 = 0u32;
        let mut go_rice = 0u32;
        let mut level_threshold = 3u32;
        let mut c1_idx = 0u32;
        let mut c2_idx = 0u32;

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

            // Per-coefficient `level2` rate parameters from the tracked state.
            let (base_level, gt1, gt2, c1c2idx) = if level2 {
                let c1c2idx =
                    (if c1_idx < 8 { 1u32 } else { 0 }) + (if c2_idx == 0 { 2 } else { 0 });
                let base_level = if c1_idx < 8 {
                    if c2_idx == 0 {
                        3
                    } else {
                        2
                    }
                } else {
                    1
                };
                let gt1_ci = gt1_base + 4 * ctx_set2 as usize + c1 as usize;
                let gt2_ci = gt2_base + ctx_set2 as usize;
                (
                    base_level,
                    [bits(gt1_ci, 0), bits(gt1_ci, 1)],
                    [bits(gt2_ci, 0), bits(gt2_ci, 1)],
                    c1c2idx,
                )
            } else {
                (0, [0, 0], [0, 0], 0)
            };

            let dist0 = (abs * abs) as f64;
            let mut best_l = 0i64;
            let mut best_levbits = 0u64;
            let mut best_distl = dist0;
            let mut best_cost = dist0 + lam * sig0 as f64;
            for l in (q - 1).max(1)..=(q + 1) {
                let dl = abs - dq.apply(l as i16) as i64;
                let dist = (dl * dl) as f64;
                let mb = if level2 {
                    ic_level_bits(l as u32, base_level, gt1, gt2, c1c2idx, go_rice) + BYPASS_BITS
                } else {
                    magnitude_bits(l as u32)
                };
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

            // Advance the `level2` greater1/greater2/Rice state (x265 quant.cpp:1074-1091).
            if level2 {
                let l = best_l as u32; // chosen absolute level (0 if zeroed)
                if l >= base_level && go_rice < 4 && l > level_threshold {
                    go_rice += 1;
                    level_threshold <<= 1;
                }
                if best_l != 0 {
                    c1_idx += 1;
                }
                if l > 1 {
                    c1 = 0;
                    if c2 < 2 {
                        c2 += 1;
                    }
                    c2_idx += 1;
                } else if (c1 == 1 || c1 == 2) && best_l != 0 {
                    c1 += 1;
                }
            }
        }
        // Carry this group's final greater1 context only when it coded a
        // coefficient (empty groups leave c1 == 1 and don't shift the next
        // group's ctxSet), matching x265 skipping all-zero groups.
        if level2 && sb_has_sig {
            prev_cg_c1 = c1;
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
        let lp =
            lam * last_pos_bits(ctxs, recs[p].x, recs[p].y, log2_size, c_idx, scan_order) as f64;
        let cost = pref_normal[p] + recs[p].rd_last + suff_zero[p + 1] + lp;
        if cost < best_cost {
            best_cost = cost;
            best_last = p as isize;
        }
    }

    // --- Stage (c): middle sub-block `coded_sub_block_flag` all-zero forcing. ---
    // For each sub-block coded with an explicit CSBF (strictly between the DC
    // sub-block and the last-significant sub-block, in forward scan order),
    // compare keeping it (CSBF=1: pay each coefficient's sig/level bits) against
    // zeroing it (CSBF=0: pay only the flag, take the distortion of dropping
    // every coefficient). Zeroing is always a bitstream-valid choice — the
    // residual encoder re-derives CSBF from the levels — so this can only change
    // RD, never validity. Sub-blocks are visited high->low frequency so a
    // neighbour's final coded state is known when its CSBF context is priced.
    if best_last >= 0 {
        let last = best_last as usize;
        let last_sbx = recs[last].x as usize / 4;
        let last_sby = recs[last].y as usize / 4;
        let last_sb_scan = scan_sub
            .iter()
            .position(|&(sx, sy)| sx as usize == last_sbx && sy as usize == last_sby)
            .unwrap_or(0);

        // Final coded-sub-block state from the post-stage-(b) levels.
        let mut coded = [[false; 8]; 8];
        for rec in recs.iter().take(last + 1) {
            if rec.level != 0 {
                coded[rec.y as usize / 4][rec.x as usize / 4] = true;
            }
        }

        for sb_scan in (1..last_sb_scan).rev() {
            let (sbx, sby) = scan_sub[sb_scan];
            let (sbx, sby) = (sbx as usize, sby as usize);
            if !coded[sby][sbx] {
                continue; // already empty: CSBF=0 either way
            }
            let right = sbx + 1 < sb_width && coded[sby][sbx + 1];
            let below = sby + 1 < sb_width && coded[sby + 1][sbx];
            let csbf_neighbors = (right as u8) | ((below as u8) << 1);
            let csbf_ci = ctx::CODED_SUB_BLOCK_FLAG
                + (csbf_neighbors != 0) as usize
                + if c_idx > 0 { 2 } else { 0 };

            let mut keep = lam * bits(csbf_ci, 1) as f64;
            let mut zero = lam * bits(csbf_ci, 0) as f64;
            for &(px, py) in scan_pos.iter() {
                let x = sbx * 4 + px as usize;
                let y = sby * 4 + py as usize;
                if x >= size || y >= size {
                    continue;
                }
                let rank = rank_of[y * size + x];
                if rank == usize::MAX || rank > last {
                    continue;
                }
                keep += recs[rank].rd_normal;
                zero += recs[rank].dist0;
            }

            // Keep/zero uses frozen contexts and ignores the prev_csbf context
            // shift zeroing imposes on lower-frequency neighbours, so this
            // greedy decision (like HM/x265's sub-block RDOQ) is a net RD win
            // but can lose a fraction of a percent on the odd block. A slack
            // margin was tried and made it worse (it blocked genuine wins
            // without fixing the directional misestimates), so decide bare.
            if zero < keep {
                for &(px, py) in scan_pos.iter() {
                    let x = sbx * 4 + px as usize;
                    let y = sby * 4 + py as usize;
                    if x < size && y < size {
                        let rank = rank_of[y * size + x];
                        if rank != usize::MAX && rank <= last {
                            recs[rank].level = 0;
                        }
                    }
                }
                coded[sby][sbx] = false;
            }
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
