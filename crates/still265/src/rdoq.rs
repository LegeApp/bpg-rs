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
//! (`Placebo` keeps exact greedy; the practical tiers use this) and
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
use crate::contexts::{Contexts, ctx};
use crate::residual::{ScanOrder, calc_sig_coeff_flag_ctx, get_scan_4x4, get_scan_sub_block};
use crate::transform::DequantParams;

/// x265 `g_quantScales` (forward quant scale per `qp % 6`).
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
const MAX_TR_DYNAMIC_RANGE: i32 = 15;
const QUANT_SHIFT: i32 = 14;
/// x265 `SCALE_BITS`: the bit-shift used to scale coefficient-domain
/// distortion into pixel-domain SSE (see `transform.rs` for derivation).
/// Without this scaling, the CABAC-bit term dominates the RD cost and
/// RDOQ aggressively zeroes coefficients (32×–2048× too few bits for
/// 8-bit 4×4–32×32 transforms).
const SCALE_BITS: i32 = 15;

/// One bypass bin in 1/32768-bit units.
const BYPASS_BITS: u64 = CabacEstimator::SCALE;

/// Diagnostic multiplier on the RDOQ rate/distortion lambda
/// (`BPG_RDOQ_LAMBDA_SCALE`, default 1.0). >1 penalizes bits more, <1 keeps
/// more. This is intentionally separate from StillSearch's pixel-domain lambda
/// gate so RDOQ unit changes can be swept without perturbing mode decisions.
fn rdoq_lambda_scale() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BPG_RDOQ_LAMBDA_SCALE")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0)
    })
}

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

#[derive(Default)]
pub struct RdoqScratch {
    recs: Vec<CoeffRec>,
    rank_of: Vec<usize>,
    suff_zero: Vec<f64>,
    pref_normal: Vec<f64>,
    levels: Vec<i16>,
}

pub struct RdoqResult<'a> {
    pub levels: &'a mut [i16],
    pub nnz: u32,
}

/// `BPG_RDOQ_REFINE` gate (default off). Post-RDOQ coordinate-descent that
/// minimizes pixel-domain SSE + lambda * exact-CABAC residual bits (priced from
/// the frozen `price_base` contexts).
///
/// DIAGNOSTIC / RD-NEGATIVE — DO NOT ENABLE BY DEFAULT. The `rdoq_optimality_tests`
/// self-test shows RDOQ sits ~1.5% above this objective's coordinate-descent
/// optimum, but wiring the refinement into a real encode is **RD-negative**
/// (4MP q28/30/32: bigger files, equal/worse decoded PSNR, ≈−0.03 dB at equal
/// bytes). recon==decode still holds, and SDH is off in the Slow path, so the
/// loss is not a round-trip or sign-hiding bug: the frozen-context
/// `estimate_residual_bits` diverges from the real evolving-context coded cost,
/// so aggressively optimizing it over-fits. Conclusion: RDOQ coefficient
/// *selection* is effectively well-calibrated for real coding; the 1.5% is an
/// artifact of the offline objective, not recoverable real RD. Kept as a probe.
pub fn rdoq_refine_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BPG_RDOQ_REFINE")
            .ok()
            .map(|s| matches!(s.trim(), "1" | "true" | "on"))
            .unwrap_or(false)
    })
}

/// Coordinate-descent refinement of RDOQ `levels` toward the true-RD optimum.
/// Candidates per position: `{0, |L|-1, |L|, |L|+1}` with the level's (or
/// coefficient's) sign — the same RD-relevant neighborhood the trellis chooses
/// among. Returns the refined nnz. `residual` is the pixel-domain residual the
/// transform was taken of, used for exact SSE distortion via `reconstruct_residual`.
#[allow(clippy::too_many_arguments)]
pub fn refine_rdoq_levels(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    residual: &[i16],
    levels: &mut [i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
) -> u32 {
    use crate::residual::{ResidualPricingScratch, estimate_residual_bits_into};
    use crate::transform::reconstruct_residual;
    let mut scratch = ResidualPricingScratch::default();
    let cost = |levels: &[i16], scratch: &mut ResidualPricingScratch| -> f64 {
        let recon = reconstruct_residual(levels, log2_size, qp, bit_depth, is_dst4);
        let dist: f64 = residual
            .iter()
            .zip(&recon)
            .map(|(&r, &c)| {
                let d = r as i64 - c as i64;
                (d * d) as f64
            })
            .sum();
        let bits =
            estimate_residual_bits_into(base_ctxs, levels, log2_size, c_idx, scan, false, scratch);
        dist + lambda * bits as f64 / CabacEstimator::SCALE as f64
    };
    let mut best = cost(levels, &mut scratch);
    loop {
        let mut improved = false;
        for i in 0..levels.len() {
            let sign: i32 = if levels[i] != 0 {
                levels[i].signum() as i32
            } else if coeffs[i] != 0 {
                coeffs[i].signum() as i32
            } else {
                continue; // zero coeff stays zero
            };
            let m = levels[i].unsigned_abs() as i32;
            let cands = [
                0i16,
                (sign * (m - 1).max(0)) as i16,
                (sign * (m + 1)) as i16,
            ];
            for &cand in &cands {
                if cand == levels[i] {
                    continue;
                }
                let save = levels[i];
                levels[i] = cand;
                let c = cost(levels, &mut scratch);
                if c + 1e-6 < best {
                    best = c;
                    improved = true;
                } else {
                    levels[i] = save;
                }
            }
        }
        if !improved {
            break;
        }
    }
    levels.iter().filter(|&&l| l != 0).count() as u32
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
    let mut scratch = RdoqScratch::default();
    let result = rdoq_single_scan_into(
        ctxs,
        coeffs,
        log2_size,
        c_idx,
        qp,
        bit_depth,
        scan_order,
        lambda,
        level2,
        &mut scratch,
    );
    (result.levels.to_vec(), result.nnz)
}

/// Scratch-backed single-scan RDOQ. The returned levels borrow from `scratch`;
/// callers should copy them only when a materialized coded block is needed.
#[allow(clippy::too_many_arguments)]
pub fn rdoq_single_scan_into<'scratch>(
    ctxs: &Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    scan_order: ScanOrder,
    lambda: f64,
    level2: bool,
    scratch: &'scratch mut RdoqScratch,
) -> RdoqResult<'scratch> {
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
    // Scale lambda from pixel-domain SSE to coefficient-domain SSE.
    //
    // x265 computes RDOQ as:
    //   ((coeff - dequant(level))^2 << scaleBits) + lambda2 * cabac_frac_bits
    // where `cabac_frac_bits` is already in 1/32768-bit units. Since this code
    // keeps the coefficient squared error unshifted, the equivalent multiplier
    // for those same fractional-bit units is `lambda / 2^scaleBits`. Do not
    // divide by CabacEstimator::SCALE here; that conversion is only for costs
    // expressed as `lambda * real_bits` against pixel-domain SSE.
    let scale_bits = SCALE_BITS - 2 * transform_shift;
    let lam = rdoq_lambda_scale() * lambda / (1u64 << scale_bits as u32) as f64;
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
    let recs = &mut scratch.recs;
    recs.clear();
    recs.reserve(size * size);
    let rank_of = &mut scratch.rank_of;
    rank_of.clear();
    rank_of.resize(size * size, usize::MAX);
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
                    if c2_idx == 0 { 3 } else { 2 }
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
            // Match x265's RDOQ refinement boundary: start from the
            // round-to-nearest quantized level, never resurrect an initial
            // zero, and only compare the kept level with one step down.
            if q == 0 {
                let rec = &mut recs[rank];
                rec.dist0 = dist0;
                rec.dist_l = dist0;
                rec.level = 0;
                rec.rd_normal = best_cost;
                rec.rd_last = f64::INFINITY;
                continue;
            }
            for l in (q - 1).max(1)..=q {
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
    let suff_zero = &mut scratch.suff_zero;
    suff_zero.clear();
    suff_zero.resize(total + 1, 0.0);
    for k in (0..total).rev() {
        suff_zero[k] = suff_zero[k + 1] + recs[k].dist0;
    }
    // Prefix RD of coding ranks 0..k as normal coefficients.
    let pref_normal = &mut scratch.pref_normal;
    pref_normal.clear();
    pref_normal.resize(total + 1, 0.0);
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
    let levels = &mut scratch.levels;
    levels.clear();
    levels.resize(size * size, 0);
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

    RdoqResult {
        levels: &mut scratch.levels,
        nnz,
    }
}

#[cfg(test)]
mod rdoq_optimality_tests {
    //! Self-contained RDOQ selection-quality probe. Block distortion is
    //! `SSE(residual, reconstruct_residual(levels))` — prediction/reference
    //! independent — so we can feed a known residual through still265's RDOQ and
    //! compare its TRUE rate-distortion cost against a coordinate-descent optimum
    //! (candidates {0, |hq|-1, |hq|, |hq|+1} per coefficient, the RD-relevant
    //! neighborhood x265's trellis also chooses among). If a cheap local search
    //! consistently beats RDOQ on TRUE RD, the trellis is leaving bits on the
    //! table; if not, RDOQ selection is near-optimal and the still265-vs-x265
    //! gap is NOT in coefficient selection.
    use super::rdoq_single_scan;
    use crate::cabac::CabacEstimator;
    use crate::contexts::Contexts;
    use crate::residual::{ScanOrder, estimate_residual_bits};
    use crate::transform::{forward_transform, quantize, reconstruct_residual};

    fn lambda(qp: i32) -> f64 {
        0.57 * 2f64.powf((qp as f64 - 12.0) / 3.0)
    }

    fn true_rd(
        levels: &[i16],
        residual: &[i16],
        log2: u8,
        qp: i32,
        bd: u8,
        scan: ScanOrder,
        lam: f64,
    ) -> (f64, f64, u64) {
        let recon = reconstruct_residual(levels, log2, qp, bd, false);
        let dist: f64 = residual
            .iter()
            .zip(&recon)
            .map(|(&r, &c)| {
                let d = r as i64 - c as i64;
                (d * d) as f64
            })
            .sum();
        let mut ctx = Contexts::new(qp);
        let bits = estimate_residual_bits(&mut ctx, levels, log2, 0, scan, false);
        (
            dist + lam * bits as f64 / CabacEstimator::SCALE as f64,
            dist,
            bits,
        )
    }

    /// Coordinate-descent over the per-coefficient RD-relevant candidate set,
    /// starting from `start`, minimizing true RD.
    fn optimize(
        start: &[i16],
        hq: &[i16],
        coeffs: &[i16],
        residual: &[i16],
        log2: u8,
        qp: i32,
        bd: u8,
        scan: ScanOrder,
        lam: f64,
    ) -> Vec<i16> {
        let mut cur = start.to_vec();
        let mut best = true_rd(&cur, residual, log2, qp, bd, scan, lam).0;
        loop {
            let mut improved = false;
            for i in 0..cur.len() {
                let center = hq[i];
                let sign: i16 = if center != 0 {
                    center.signum()
                } else if coeffs[i] != 0 {
                    coeffs[i].signum()
                } else {
                    1
                };
                let m = center.unsigned_abs() as i32;
                let cands = [
                    0i16,
                    (sign as i32 * (m - 1).max(0)) as i16,
                    (sign as i32 * m) as i16,
                    (sign as i32 * (m + 1)) as i16,
                ];
                for &cand in &cands {
                    if cand == cur[i] {
                        continue;
                    }
                    let save = cur[i];
                    cur[i] = cand;
                    let c = true_rd(&cur, residual, log2, qp, bd, scan, lam).0;
                    if c + 1e-6 < best {
                        best = c;
                        improved = true;
                    } else {
                        cur[i] = save;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        cur
    }

    /// Deterministic Laplacian-ish residual generator (seeded LCG).
    fn make_residual(n: usize, scale: f64, seed: u64) -> Vec<i16> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64
        };
        (0..n)
            .map(|_| {
                let u = next().max(1e-6);
                let sign = if next() < 0.5 { -1.0 } else { 1.0 };
                (sign * -scale * u.ln()).round().clamp(-90.0, 90.0) as i16
            })
            .collect()
    }

    #[test]
    fn rdoq_vs_bruteforce_optimum() {
        let log2: u8 = 3; // 8x8 luma
        let n = 1usize << (2 * log2);
        let bd = 8u8;
        let scan = ScanOrder::Diagonal;
        let mut agg_rdoq_excess = 0.0f64; // % over optimum
        let mut agg_hq_excess = 0.0f64;
        let mut count = 0;
        let mut rdoq_beaten = 0;
        println!(
            "\n  qp scale seed | nnz(hq/rdoq/opt) | RD hq / rdoq / opt | rdoq-excess% hq-excess%"
        );
        for qp in [22i32, 28, 34] {
            let lam = lambda(qp);
            for &scale in &[6.0f64, 12.0, 22.0] {
                for seed in 0..6u64 {
                    let residual = make_residual(n, scale, seed * 131 + qp as u64);
                    let coeffs = forward_transform(&residual, log2, false, bd);
                    let (hq, _) = quantize(&coeffs, log2, qp, bd);
                    let ctx = Contexts::new(qp);
                    let (rdoq, _) =
                        rdoq_single_scan(&ctx, &coeffs, log2, 0, qp, bd, scan, lam, true);
                    if hq.iter().all(|&l| l == 0) && rdoq.iter().all(|&l| l == 0) {
                        continue;
                    }
                    let opt_from_hq =
                        optimize(&hq, &hq, &coeffs, &residual, log2, qp, bd, scan, lam);
                    let opt_from_rdoq =
                        optimize(&rdoq, &hq, &coeffs, &residual, log2, qp, bd, scan, lam);
                    let rd_hq = true_rd(&hq, &residual, log2, qp, bd, scan, lam).0;
                    let rd_rdoq = true_rd(&rdoq, &residual, log2, qp, bd, scan, lam).0;
                    let rd_o1 = true_rd(&opt_from_hq, &residual, log2, qp, bd, scan, lam).0;
                    let rd_o2 = true_rd(&opt_from_rdoq, &residual, log2, qp, bd, scan, lam).0;
                    let rd_opt = rd_o1.min(rd_o2);
                    let nz = |v: &[i16]| v.iter().filter(|&&l| l != 0).count();
                    let rdoq_excess = 100.0 * (rd_rdoq - rd_opt) / rd_opt.max(1.0);
                    let hq_excess = 100.0 * (rd_hq - rd_opt) / rd_opt.max(1.0);
                    agg_rdoq_excess += rdoq_excess;
                    agg_hq_excess += hq_excess;
                    count += 1;
                    if rd_opt + 1.0 < rd_rdoq {
                        rdoq_beaten += 1;
                    }
                    if seed < 2 {
                        println!(
                            "  {qp:2} {scale:4.0} {seed:4} | {:2}/{:2}/{:2} | {:9.0} {:9.0} {:9.0} | {:+6.2}% {:+6.2}%",
                            nz(&hq),
                            nz(&rdoq),
                            nz(&opt_from_rdoq),
                            rd_hq,
                            rd_rdoq,
                            rd_opt,
                            rdoq_excess,
                            hq_excess
                        );
                    }
                }
            }
        }
        println!(
            "\n  AGGREGATE over {count} blocks: mean RDOQ excess over local-opt = {:.3}%, \
             mean hard-quant excess = {:.3}%, RDOQ beaten in {rdoq_beaten}/{count} blocks",
            agg_rdoq_excess / count as f64,
            agg_hq_excess / count as f64
        );
    }
}

#[cfg(test)]
mod context_drift_sizing {
    //! Sizes the agent's "rate-context fidelity" suspect: still265 prices all
    //! in-CTU trial decisions from `price_base` frozen at CTU entry, but real
    //! coded bits use the CABAC context evolving block-by-block. This measures
    //! how much a textured block's exact CABAC bit cost moves between a fresh
    //! CTU-entry context and one drifted by ~30 prior textured blocks (mid-CTU).
    //! If the per-block drift is tiny, the frozen-context approximation cannot
    //! account for the ~0.3 dB texture gap and the running-context RD build is
    //! not worth it.
    use crate::contexts::Contexts;
    use crate::residual::{ScanOrder, estimate_residual_bits};
    use crate::transform::{forward_transform, quantize};

    fn make_residual(n: usize, scale: f64, seed: u64) -> Vec<i16> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64
        };
        (0..n)
            .map(|_| {
                let u = next().max(1e-6);
                let sign = if next() < 0.5 { -1.0 } else { 1.0 };
                (sign * -scale * u.ln()).round().clamp(-90.0, 90.0) as i16
            })
            .collect()
    }

    #[test]
    fn context_drift_pricing_magnitude() {
        let log2 = 3u8;
        let n = 1usize << (2 * log2);
        let bd = 8u8;
        let scan = ScanOrder::Diagonal;
        let drift_blocks = 30; // ~mid-CTU position (64 8x8 blocks per 64x64 CTU)
        for qp in [22i32, 28, 34] {
            for &scale in &[8.0f64, 18.0] {
                let mut sum_fresh = 0i64;
                let mut sum_drift = 0i64;
                let mut sum_abs_delta = 0i64;
                let trials = 40;
                for t in 0..trials {
                    let coeffs = forward_transform(
                        &make_residual(n, scale, t * 7 + qp as u64),
                        log2,
                        false,
                        bd,
                    );
                    let (levels, _) = quantize(&coeffs, log2, qp, bd);
                    // (a) fresh CTU-entry context
                    let mut cf = Contexts::new(qp);
                    let bf = estimate_residual_bits(&mut cf, &levels, log2, 0, scan, false) as i64;
                    // (b) context drifted by coding `drift_blocks` prior textured blocks
                    let mut cd = Contexts::new(qp);
                    for k in 0..drift_blocks {
                        let pc = forward_transform(
                            &make_residual(n, scale, 9000 + t * 131 + k * 17 + qp as u64),
                            log2,
                            false,
                            bd,
                        );
                        let (pl, _) = quantize(&pc, log2, qp, bd);
                        let _ = estimate_residual_bits(&mut cd, &pl, log2, 0, scan, false);
                    }
                    let bd2 = estimate_residual_bits(&mut cd, &levels, log2, 0, scan, false) as i64;
                    sum_fresh += bf;
                    sum_drift += bd2;
                    sum_abs_delta += (bf - bd2).abs();
                }
                let scale_u = crate::cabac::CabacEstimator::SCALE as f64;
                let mf = sum_fresh as f64 / trials as f64 / scale_u;
                let md = sum_drift as f64 / trials as f64 / scale_u;
                println!(
                    "  qp{qp} scale{scale:4.0}: fresh={mf:7.1}b drift={md:7.1}b  mean_signed_delta={:+.2}b ({:+.2}%)  mean_abs_delta={:.2}b",
                    md - mf,
                    100.0 * (md - mf) / mf,
                    sum_abs_delta as f64 / trials as f64 / scale_u
                );
            }
        }
    }
}
