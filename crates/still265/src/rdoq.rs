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
use crate::residual::{
    ResidualPricingScratch, ScanOrder, estimate_residual_bits_into, get_scan_4x4,
    get_scan_sub_block,
};
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

/// A/B gate for x265's coding-group carry shape. In x265 `rdoQuant`, the
/// greater1 carry (`c1`) is reset to 1 at the start of every coefficient group;
/// an empty group therefore clears any previous `c1 == 0` carry before the next
/// lower-frequency group. The historical still265 single-scan path left the
/// carry unchanged across empty groups.
fn x265_empty_cg_carry_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BPG_RDOQ_X265_EMPTY_CG_CARRY")
            .ok()
            .map(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
            .unwrap_or(false)
    })
}

/// A/B gate for x265's last-significant refinement envelope. x265 evaluates
/// candidate last positions walking high-frequency -> low-frequency, but stops
/// after the first coded coefficient whose selected absolute level is > 1
/// (`rdoqLevel > 1`). The historical still265 path considered every non-zero
/// coefficient as a possible last.
fn x265_last_stop_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BPG_RDOQ_X265_LAST_STOP")
            .ok()
            .map(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"))
            .unwrap_or(false)
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

/// Per-coefficient state captured during the level-decision scan, indexed by
/// forward (DC -> high-frequency) combined scan rank.
#[derive(Clone, Copy, Default)]
struct CoeffRec {
    idx: usize, // row-major coefficient index
    x: u32,
    y: u32,
    level: i16,     // chosen signed level (0 if quantized away)
    dist0: f64,     // distortion if zeroed
    rd_normal: f64, // RD cost coded as a normal coefficient (sig flag + level)
    rd_last: f64,   // RD cost coded as the last significant coeff (no sig flag)
}

#[derive(Default)]
pub struct RdoqScratch {
    recs: Vec<CoeffRec>,
    suff_zero: Vec<f64>,
    pref_normal: Vec<f64>,
    levels: Vec<i16>,
    price_scratch: ResidualPricingScratch,
    dequant: Vec<i16>,
    recon: Vec<i16>,
    trial_levels: Vec<i16>,
}

pub struct RdoqResult<'a> {
    pub levels: &'a mut [i16],
    pub nnz: u32,
}

/// Experimental post-RDOQ residual-pricing models selected by
/// `BPG_RDOQ_PRICING`. `Frozen` is the existing single-scan behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RdoqPricingMode {
    #[default]
    Frozen,
    Running,
    Hybrid,
    ExactCleanup,
    BoundedTrellis,
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

#[allow(clippy::too_many_arguments)]
fn exact_rd_cost(
    base_ctxs: &Contexts,
    residual: &[i16],
    levels: &[i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
) -> f64 {
    crate::transform::reconstruct_residual_into(
        levels,
        log2_size,
        qp,
        bit_depth,
        is_dst4,
        &mut scratch.dequant,
        &mut scratch.recon,
    );
    let dist: f64 = residual
        .iter()
        .zip(&scratch.recon)
        .map(|(&r, &c)| {
            let d = r as i64 - c as i64;
            (d * d) as f64
        })
        .sum();
    let bits = estimate_residual_bits_into(
        base_ctxs,
        levels,
        log2_size,
        c_idx,
        scan,
        false,
        &mut scratch.price_scratch,
    );
    dist + lambda * bits as f64 / CabacEstimator::SCALE as f64
}

fn level_sign(coeff: i16, level: i16) -> Option<i32> {
    if level != 0 {
        Some(level.signum() as i32)
    } else if coeff != 0 {
        Some(coeff.signum() as i32)
    } else {
        None
    }
}

fn signed_level(sign: i32, mag: i32) -> i16 {
    let mag = mag.clamp(0, i16::MAX as i32);
    (sign * mag) as i16
}

fn nnz(levels: &[i16]) -> u32 {
    levels.iter().filter(|&&l| l != 0).count() as u32
}

fn scan_indices(log2_size: u8, scan: ScanOrder) -> Vec<usize> {
    let size = 1usize << log2_size;
    let mut out = Vec::with_capacity(size * size);
    for &(sbx, sby) in get_scan_sub_block(log2_size, scan).iter() {
        for &(px, py) in get_scan_4x4(scan).iter() {
            let x = sbx as usize * 4 + px as usize;
            let y = sby as usize * 4 + py as usize;
            if x < size && y < size {
                out.push(y * size + x);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn keep_if_better(
    base_ctxs: &Contexts,
    residual: &[i16],
    levels: &mut [i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) -> bool {
    let cost = exact_rd_cost(
        base_ctxs, residual, levels, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
        scratch,
    );
    if cost + 1e-6 < *best {
        *best = cost;
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn try_set_level(
    base_ctxs: &Contexts,
    residual: &[i16],
    levels: &mut [i16],
    idx: usize,
    cand: i16,
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) -> bool {
    if levels[idx] == cand {
        return false;
    }
    let save = levels[idx];
    levels[idx] = cand;
    if keep_if_better(
        base_ctxs, residual, levels, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
        scratch, best,
    ) {
        true
    } else {
        levels[idx] = save;
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_drop_sweep(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    residual: &[i16],
    levels: &mut [i16],
    order: &[usize],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) -> bool {
    let mut improved = false;
    for &idx in order.iter().rev() {
        let cur = levels[idx];
        if cur == 0 {
            continue;
        }
        let sign = level_sign(coeffs[idx], cur).unwrap_or(1);
        let mag = cur.unsigned_abs() as i32;
        let down = signed_level(sign, mag - 1);
        if try_set_level(
            base_ctxs, residual, levels, idx, down, log2_size, c_idx, qp, bit_depth, is_dst4, scan,
            lambda, scratch, best,
        ) {
            improved = true;
        }
        if levels[idx] != 0
            && try_set_level(
                base_ctxs, residual, levels, idx, 0, log2_size, c_idx, qp, bit_depth, is_dst4,
                scan, lambda, scratch, best,
            )
        {
            improved = true;
        }
    }
    improved
}

#[allow(clippy::too_many_arguments)]
fn exact_tail_cleanup(
    base_ctxs: &Contexts,
    residual: &[i16],
    levels: &mut [i16],
    order: &[usize],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) -> bool {
    let mut improved = false;
    scratch.trial_levels.clear();
    scratch.trial_levels.extend_from_slice(levels);
    if scratch.trial_levels.iter().any(|&l| l != 0) {
        for v in &mut scratch.trial_levels {
            *v = 0;
        }
        let trial = scratch.trial_levels.clone();
        let cost = exact_rd_cost(
            base_ctxs, residual, &trial, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
            scratch,
        );
        if cost + 1e-6 < *best {
            *best = cost;
            levels.copy_from_slice(&trial);
            improved = true;
        }
    }

    scratch.trial_levels.clear();
    scratch.trial_levels.extend_from_slice(levels);
    for last_rank in (0..order.len()).rev() {
        let last_idx = order[last_rank];
        if levels[last_idx] == 0 {
            continue;
        }
        scratch.trial_levels.copy_from_slice(levels);
        for &idx in order.iter().skip(last_rank + 1) {
            scratch.trial_levels[idx] = 0;
        }
        let trial = scratch.trial_levels.clone();
        let cost = exact_rd_cost(
            base_ctxs, residual, &trial, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
            scratch,
        );
        if cost + 1e-6 < *best {
            *best = cost;
            levels.copy_from_slice(&trial);
            improved = true;
        }
    }
    improved
}

#[allow(clippy::too_many_arguments)]
fn exact_subblock_cleanup(
    base_ctxs: &Contexts,
    residual: &[i16],
    levels: &mut [i16],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) -> bool {
    let size = 1usize << log2_size;
    let sb_width = size / 4;
    let mut improved = false;
    for &(sbx, sby) in get_scan_sub_block(log2_size, scan).iter().rev() {
        let (sbx, sby) = (sbx as usize, sby as usize);
        if sbx >= sb_width || sby >= sb_width {
            continue;
        }
        let mut has_nz = false;
        scratch.trial_levels.clear();
        scratch.trial_levels.extend_from_slice(levels);
        for y in sby * 4..((sby + 1) * 4).min(size) {
            for x in sbx * 4..((sbx + 1) * 4).min(size) {
                let idx = y * size + x;
                has_nz |= scratch.trial_levels[idx] != 0;
                scratch.trial_levels[idx] = 0;
            }
        }
        if !has_nz {
            continue;
        }
        let trial = scratch.trial_levels.clone();
        let cost = exact_rd_cost(
            base_ctxs, residual, &trial, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
            scratch,
        );
        if cost + 1e-6 < *best {
            *best = cost;
            levels.copy_from_slice(&trial);
            improved = true;
        }
    }
    improved
}

#[allow(clippy::too_many_arguments)]
fn exact_coordinate_descent(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    residual: &[i16],
    levels: &mut [i16],
    order: &[usize],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) {
    loop {
        let mut improved = false;
        for &idx in order.iter().rev() {
            let Some(sign) = level_sign(coeffs[idx], levels[idx]) else {
                continue;
            };
            let mag = levels[idx].unsigned_abs() as i32;
            let cands = [0, signed_level(sign, mag - 1), signed_level(sign, mag + 1)];
            for cand in cands {
                if try_set_level(
                    base_ctxs, residual, levels, idx, cand, log2_size, c_idx, qp, bit_depth,
                    is_dst4, scan, lambda, scratch, best,
                ) {
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bounded_trellis(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    residual: &[i16],
    levels: &mut [i16],
    order: &[usize],
    log2_size: u8,
    c_idx: u8,
    qp: i32,
    bit_depth: u8,
    is_dst4: bool,
    scan: ScanOrder,
    lambda: f64,
    beam_width: usize,
    scratch: &mut RdoqScratch,
    best: &mut f64,
) {
    const MAX_TRELLIS_POSITIONS: usize = 32;
    let beam_width = beam_width.clamp(1, 16);
    let mut positions: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .filter(|&idx| levels[idx] != 0 || coeffs[idx] != 0)
        .take(MAX_TRELLIS_POSITIONS)
        .collect();
    if positions.is_empty() {
        return;
    }
    positions.reverse();

    let mut beam = vec![(levels.to_vec(), *best)];
    for idx in positions {
        let mut next: Vec<(Vec<i16>, f64)> = Vec::new();
        for (state, _) in beam.iter() {
            let Some(sign) = level_sign(coeffs[idx], state[idx]) else {
                continue;
            };
            let mag = state[idx].unsigned_abs() as i32;
            let cands = [
                state[idx],
                0,
                signed_level(sign, mag - 1),
                signed_level(sign, mag + 1),
            ];
            for cand in cands {
                let mut candidate = state.clone();
                candidate[idx] = cand;
                if next.iter().any(|(seen, _)| seen == &candidate) {
                    continue;
                }
                let cost = exact_rd_cost(
                    base_ctxs, residual, &candidate, log2_size, c_idx, qp, bit_depth, is_dst4,
                    scan, lambda, scratch,
                );
                next.push((candidate, cost));
            }
        }
        next.sort_by(|a, b| a.1.total_cmp(&b.1));
        next.truncate(beam_width);
        if next.is_empty() {
            break;
        }
        beam = next;
    }
    if let Some((state, cost)) = beam.first() {
        if *cost + 1e-6 < *best {
            *best = *cost;
            levels.copy_from_slice(state);
        }
    }
}

/// Applies opt-in residual-pricing experiments after the normal frozen
/// single-scan RDOQ decision. Returns the resulting non-zero coefficient count.
#[allow(clippy::too_many_arguments)]
pub fn apply_rdoq_pricing_mode(
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
    mode: RdoqPricingMode,
    trellis_beam: usize,
    scratch: &mut RdoqScratch,
) -> u32 {
    if mode == RdoqPricingMode::Frozen {
        return nnz(levels);
    }

    let order = scan_indices(log2_size, scan);
    let mut best = exact_rd_cost(
        base_ctxs, residual, levels, log2_size, c_idx, qp, bit_depth, is_dst4, scan, lambda,
        scratch,
    );

    match mode {
        RdoqPricingMode::Frozen => {}
        RdoqPricingMode::Running => exact_coordinate_descent(
            base_ctxs, coeffs, residual, levels, &order, log2_size, c_idx, qp, bit_depth, is_dst4,
            scan, lambda, scratch, &mut best,
        ),
        RdoqPricingMode::Hybrid => {
            exact_tail_cleanup(
                base_ctxs, residual, levels, &order, log2_size, c_idx, qp, bit_depth, is_dst4,
                scan, lambda, scratch, &mut best,
            );
            exact_drop_sweep(
                base_ctxs, coeffs, residual, levels, &order, log2_size, c_idx, qp, bit_depth,
                is_dst4, scan, lambda, scratch, &mut best,
            );
        }
        RdoqPricingMode::ExactCleanup => loop {
            let changed_tail = exact_tail_cleanup(
                base_ctxs, residual, levels, &order, log2_size, c_idx, qp, bit_depth, is_dst4,
                scan, lambda, scratch, &mut best,
            );
            let changed_sb = exact_subblock_cleanup(
                base_ctxs, residual, levels, log2_size, c_idx, qp, bit_depth, is_dst4, scan,
                lambda, scratch, &mut best,
            );
            let changed_drop = exact_drop_sweep(
                base_ctxs, coeffs, residual, levels, &order, log2_size, c_idx, qp, bit_depth,
                is_dst4, scan, lambda, scratch, &mut best,
            );
            if !(changed_tail || changed_sb || changed_drop) {
                break;
            }
        },
        RdoqPricingMode::BoundedTrellis => bounded_trellis(
            base_ctxs,
            coeffs,
            residual,
            levels,
            &order,
            log2_size,
            c_idx,
            qp,
            bit_depth,
            is_dst4,
            scan,
            lambda,
            trellis_beam,
            scratch,
            &mut best,
        ),
    }
    nnz(levels)
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

    // Pre-pass: reverse-scan for the last significant coefficient under
    // round-to-nearest quant.
    // `q >= 1  ⟺  abs*scale + round >= 1<<qbits  ⟺  abs >= ceil(round/scale)`
    // (since `round` is exactly half of `1<<qbits`). Positions after the last
    // significant rank can never be coded (stage (b) only considers significant
    // `last` candidates and never prices ranks beyond it), so all work past it
    // is trimmed to a pure zeroing-distortion suffix, and an all-insignificant
    // block returns immediately. Scanning high->low frequency stops at the
    // first significant coefficient instead of touching the whole block.
    debug_assert_eq!(scan_sub.len() * 16, size * size);
    let sig_threshold = ((round + scale - 1) / scale) as u32;
    let mut last_sig_rank: isize = -1;
    'find_last: for (sb_i, &(sbx, sby)) in scan_sub.iter().enumerate().rev() {
        let row0 = sby as usize * 4 * size + sbx as usize * 4;
        for pos in (0..16).rev() {
            let (px, py) = scan_pos[pos];
            let abs = coeffs[row0 + py as usize * size + px as usize].unsigned_abs() as u32;
            if abs >= sig_threshold {
                last_sig_rank = (sb_i * 16 + pos) as isize;
                break 'find_last;
            }
        }
    }
    if last_sig_rank < 0 {
        scratch.levels.clear();
        scratch.levels.resize(size * size, 0);
        return RdoqResult {
            levels: &mut scratch.levels,
            nnz: 0,
        };
    }
    let last_cg = last_sig_rank as usize / 16;
    let last_cg_top = last_sig_rank as usize % 16;
    let kept_cgs = last_cg + 1;
    let kept = last_sig_rank as usize + 1;

    // Records are indexed by forward combined scan rank (rank = sb*16 + pos;
    // the scan tables cover the block exactly, so ranks are arithmetic).
    // Fields are filled during the stage-(a) scan.
    let recs = &mut scratch.recs;
    recs.clear();
    recs.resize(kept, CoeffRec::default());

    // Zeroing distortion of the trimmed tail (ranks kept..size*size), summed in
    // reverse scan order — the identical float accumulation the full suffix
    // pass used, so downstream costs stay bit-exact. Full coding groups beyond
    // the last significant one first, then the last group's tail positions.
    let mut tail_zero_dist = 0.0f64;
    for &(sbx, sby) in scan_sub[kept_cgs..].iter().rev() {
        let row0 = sby as usize * 4 * size + sbx as usize * 4;
        for &(px, py) in scan_pos.iter().rev() {
            let abs = coeffs[row0 + py as usize * size + px as usize].unsigned_abs() as i64;
            tail_zero_dist += (abs * abs) as f64;
        }
    }
    {
        let (sbx, sby) = scan_sub[last_cg];
        let row0 = sby as usize * 4 * size + sbx as usize * 4;
        for pos in (last_cg_top + 1..16).rev() {
            let (px, py) = scan_pos[pos];
            let abs = coeffs[row0 + py as usize * size + px as usize].unsigned_abs() as i64;
            tail_zero_dist += (abs * abs) as f64;
        }
    }

    // --- Stage (a): per-coefficient level decision, reverse sub-block scan. ---
    let mut coded_sb_flags = [0u8; 8];
    // `level2` coding-group state carried across groups (x265 `rdoqQuant`):
    // `prev_cg_c1` is the previous coded group's final greater1 context, used to
    // bump `ctxSet` for the next group.
    let mut prev_cg_c1 = 1u32;
    for (sb_i, &(sbx, sby)) in scan_sub[..kept_cgs].iter().enumerate().rev() {
        let sbx = sbx as usize;
        let sby = sby as usize;
        let right = sbx + 1 < sb_width && (coded_sb_flags[sby] & (1u8 << (sbx + 1))) != 0;
        let below = sby + 1 < sb_width && (coded_sb_flags[sby + 1] & (1u8 << sbx)) != 0;
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

        // The greater2 context index only depends on `ctx_set2`, fixed for the
        // whole group — hoist its entropy bits out of the position loop.
        let gt2 = if level2 {
            let gt2_ci = gt2_base + ctx_set2 as usize;
            [bits(gt2_ci, 0), bits(gt2_ci, 1)]
        } else {
            [0, 0]
        };

        let sig_row =
            crate::residual::sig_ctx_row(log2_size, c_idx, scan_idx, !cg_is_not_dc, prev_csbf);
        let mut sb_has_sig = false;
        // Positions above the last significant rank in the last group carry no
        // record (their zeroing distortion lives in `tail_zero_dist`).
        let top = if sb_i == last_cg { last_cg_top } else { 15 };
        for pos in (0..=top).rev() {
            let (px, py) = scan_pos[pos];
            let x = sbx * 4 + px as usize;
            let y = sby * 4 + py as usize;
            let rank = sb_i * 16 + pos;
            let coeff = coeffs[y * size + x];
            let abs = coeff.unsigned_abs() as i64;
            let q = ((abs * scale + round) >> qbits).min(32767) as i32;

            let sig_ci = sig_row[pos] as usize;
            let sig0 = bits(sig_ci, 0);
            let dist0 = (abs * abs) as f64;

            let rec = &mut recs[rank];
            rec.idx = y * size + x;
            rec.x = x as u32;
            rec.y = y as u32;
            rec.dist0 = dist0;

            // Match x265's RDOQ refinement boundary: start from the
            // round-to-nearest quantized level, never resurrect an initial
            // zero, and only compare the kept level with one step down. An
            // initial zero needs no level-rate parameters and no state advance,
            // so it exits before the greater1/greater2 lookups.
            if q == 0 {
                rec.level = 0;
                rec.rd_normal = dist0 + lam * sig0 as f64;
                rec.rd_last = f64::INFINITY;
                continue;
            }

            let sig1 = bits(sig_ci, 1);
            // Per-coefficient `level2` rate parameters from the tracked state
            // (`gt2` was hoisted per group above).
            let (base_level, gt1, c1c2idx) = if level2 {
                let c1c2idx =
                    (if c1_idx < 8 { 1u32 } else { 0 }) + (if c2_idx == 0 { 2 } else { 0 });
                let base_level = if c1_idx < 8 {
                    if c2_idx == 0 { 3 } else { 2 }
                } else {
                    1
                };
                let gt1_ci = gt1_base + 4 * ctx_set2 as usize + c1 as usize;
                (base_level, [bits(gt1_ci, 0), bits(gt1_ci, 1)], c1c2idx)
            } else {
                (0, [0, 0], 0)
            };

            let mut best_l = 0i64;
            let mut best_levbits = 0u64;
            let mut best_distl = dist0;
            let mut best_cost = dist0 + lam * sig0 as f64;
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
        if level2 {
            if sb_has_sig || x265_empty_cg_carry_enabled() {
                prev_cg_c1 = c1;
            }
        }
        if sb_has_sig {
            coded_sb_flags[sby] |= 1u8 << sbx;
        }
    }

    // --- Stage (b): last-significant-position optimization. ---
    // Choose the forward scan rank `p` (a significant coefficient) to be the
    // last coded coefficient: ranks < p are coded normally, p is coded as last
    // (its sig flag inferred), ranks > p are zeroed (distortion only). Also
    // consider zeroing the whole block (cbf = 0).
    let total = recs.len();
    // Suffix distortion of zeroing rank..end. `suff_zero[total]` seeds the
    // trimmed tail's zeroing distortion so costs match the untrimmed pass.
    let suff_zero = &mut scratch.suff_zero;
    suff_zero.clear();
    suff_zero.resize(total + 1, 0.0);
    suff_zero[total] = tail_zero_dist;
    for k in (0..total).rev() {
        suff_zero[k] = suff_zero[k + 1] + recs[k].dist0;
    }
    // Baseline: all-zero block (cbf = 0), no residual bits.
    let mut best_cost = suff_zero[0];
    let mut best_last: isize = -1;

    // `last_sig_coeff_x/y` prices depend only on the axis value for this block.
    // Cache the two tiny axis tables once per RDOQ call instead of walking the
    // truncated-unary contexts for every non-zero last-position candidate.
    let mut last_x_bits_by_val = [0u64; 32];
    let mut last_y_bits_by_val = [0u64; 32];
    for v in 0..size {
        last_x_bits_by_val[v] = last_axis_bits(ctxs, v as u32, log2_size, c_idx, true);
        last_y_bits_by_val[v] = last_axis_bits(ctxs, v as u32, log2_size, c_idx, false);
    }
    let last_bits_for_rec = |rec: &CoeffRec| -> u64 {
        let (raw_x, raw_y) = if scan_order == ScanOrder::Vertical {
            (rec.y as usize, rec.x as usize)
        } else {
            (rec.x as usize, rec.y as usize)
        };
        last_x_bits_by_val[raw_x] + last_y_bits_by_val[raw_y]
    };

    let last_stop = x265_last_stop_enabled();
    if last_stop {
        // Prefix RD of coding ranks 0..k as normal coefficients. Only the
        // diagnostic reverse-scan envelope needs random prefix lookup; the
        // default forward scan below carries the prefix in a scalar and avoids
        // one Vec fill per RDOQ call.
        let pref_normal = &mut scratch.pref_normal;
        pref_normal.clear();
        pref_normal.resize(total + 1, 0.0);
        for k in 0..total {
            pref_normal[k + 1] = pref_normal[k] + recs[k].rd_normal;
        }
        for p in (0..total).rev() {
            if recs[p].level == 0 {
                continue;
            }
            let lp = lam * last_bits_for_rec(&recs[p]) as f64;
            let cost = pref_normal[p] + recs[p].rd_last + suff_zero[p + 1] + lp;
            if cost < best_cost {
                best_cost = cost;
                best_last = p as isize;
            }
            if recs[p].level.unsigned_abs() > 1 {
                break;
            }
        }
    } else {
        let mut pref_normal = 0.0f64;
        for p in 0..total {
            if recs[p].level != 0 {
                let lp = lam * last_bits_for_rec(&recs[p]) as f64;
                let cost = pref_normal + recs[p].rd_last + suff_zero[p + 1] + lp;
                if cost < best_cost {
                    best_cost = cost;
                    best_last = p as isize;
                }
            }
            pref_normal += recs[p].rd_normal;
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
    // Ranks are arithmetic, so the last coefficient's sub-block scan index is
    // `last / 16`; with fewer than two groups before it the loop below is
    // empty, so skip the coded-state build entirely.
    let last_sb_scan = if best_last >= 0 {
        best_last as usize / 16
    } else {
        0
    };
    if last_sb_scan > 1 {
        let last = best_last as usize;

        // Final coded-sub-block state from the post-stage-(b) levels.
        let mut coded = [0u8; 8];
        for rec in recs.iter().take(last + 1) {
            if rec.level != 0 {
                coded[rec.y as usize / 4] |= 1u8 << (rec.x as usize / 4);
            }
        }

        for sb_scan in (1..last_sb_scan).rev() {
            let (sbx, sby) = scan_sub[sb_scan];
            let (sbx, sby) = (sbx as usize, sby as usize);
            if (coded[sby] & (1u8 << sbx)) == 0 {
                continue; // already empty: CSBF=0 either way
            }
            let right = sbx + 1 < sb_width && (coded[sby] & (1u8 << (sbx + 1))) != 0;
            let below = sby + 1 < sb_width && (coded[sby + 1] & (1u8 << sbx)) != 0;
            let csbf_neighbors = (right as u8) | ((below as u8) << 1);
            let csbf_ci = ctx::CODED_SUB_BLOCK_FLAG
                + (csbf_neighbors != 0) as usize
                + if c_idx > 0 { 2 } else { 0 };

            let mut keep = lam * bits(csbf_ci, 1) as f64;
            let mut zero = lam * bits(csbf_ci, 0) as f64;
            for rank in sb_scan * 16..(sb_scan + 1) * 16 {
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
                for rank in sb_scan * 16..(sb_scan + 1) * 16 {
                    recs[rank].level = 0;
                }
                coded[sby] &= !(1u8 << sbx);
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
        for rec in recs.iter().take(last + 1) {
            if rec.level != 0 {
                levels[rec.idx] = rec.level;
                nnz += 1;
            }
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
    use super::{RdoqPricingMode, RdoqScratch, apply_rdoq_pricing_mode, rdoq_single_scan};
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

    #[test]
    fn pricing_modes_do_not_worsen_exact_rd_objective() {
        let log2 = 3;
        let qp = 28;
        let bd = 8;
        let scan = ScanOrder::Diagonal;
        let lam = lambda(qp);
        let residual: Vec<i16> = (0..64)
            .map(|i| {
                let x = (i & 7) as i16;
                let y = (i >> 3) as i16;
                (x * 9 - y * 5 + ((i % 5) as i16 - 2) * 7).clamp(-127, 127)
            })
            .collect();
        let coeffs = forward_transform(&residual, log2, false, bd);
        let ctx = Contexts::new(qp);
        let (start, _) = rdoq_single_scan(&ctx, &coeffs, log2, 0, qp, bd, scan, lam, true);

        for mode in [
            RdoqPricingMode::Running,
            RdoqPricingMode::Hybrid,
            RdoqPricingMode::ExactCleanup,
            RdoqPricingMode::BoundedTrellis,
        ] {
            let mut levels = start.clone();
            let before = true_rd(&levels, &residual, log2, qp, bd, scan, lam).0;
            let mut scratch = RdoqScratch::default();
            let returned_nnz = apply_rdoq_pricing_mode(
                &ctx,
                &coeffs,
                &residual,
                &mut levels,
                log2,
                0,
                qp,
                bd,
                false,
                scan,
                lam,
                mode,
                4,
                &mut scratch,
            );
            let after = true_rd(&levels, &residual, log2, qp, bd, scan, lam).0;
            assert!(
                after <= before + 1e-6,
                "{mode:?} worsened exact RD: before={before}, after={after}"
            );
            assert_eq!(
                returned_nnz,
                levels.iter().filter(|&&l| l != 0).count() as u32
            );
        }
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
