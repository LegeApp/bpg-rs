//! `residual_coding()` CABAC writer (H.265 7.3.8.11 / 9.3.4.2), the exact
//! inverse of `bpg-hevc-decode::hevc::residual::decode_residual`.
//!
//! Given a transform block's quantized coefficients (row-major, stride =
//! `1 << log2_size`), this emits the CABAC bins a conforming decoder reads
//! back to reconstruct those exact coefficients: `last_sig_coeff_x/y`
//! prefix+suffix, `coded_sub_block_flag`, `sig_coeff_flag`,
//! `coeff_abs_level_greater1/2_flag`, the bypass sign bits (with sign data
//! hiding), and `coeff_abs_level_remaining` (Golomb-Rice / EGk).
//!
//! Scan tables, context-index derivation, the greater1 context evolution, and
//! the Rice-parameter update are mirrored verbatim from the decoder so the two
//! sides stay bit-for-bit in lockstep (validated by `tests/residual_roundtrip`).
//!
//! `transform_skip` and `cu_transquant_bypass` are not emitted here (the
//! still-image PPS sets `transform_skip_enabled_flag = 0` and
//! `transquant_bypass_enabled_flag = 0`); the decoder only reads
//! `transform_skip_flag` when those are enabled.

use bpg_bitstream::BitWriter;

use crate::cabac::{CabacEncoder, CabacEstimator, ContextModel};
use crate::contexts::{ctx, Contexts};

trait CabacSyntax {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel);
    fn encode_bin_ep(&mut self, bin_value: u8);
}

struct CabacWriter<'a, 'b> {
    enc: &'a mut CabacEncoder,
    w: &'a mut BitWriter,
    _marker: core::marker::PhantomData<&'b ()>,
}

impl CabacSyntax for CabacWriter<'_, '_> {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel) {
        self.enc.encode_bin(self.w, bin_value, ctx);
    }

    fn encode_bin_ep(&mut self, bin_value: u8) {
        self.enc.encode_bin_ep(self.w, bin_value);
    }
}

impl CabacSyntax for CabacEstimator {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel) {
        CabacEstimator::encode_bin(self, bin_value, ctx);
    }

    fn encode_bin_ep(&mut self, bin_value: u8) {
        CabacEstimator::encode_bin_ep(self, bin_value);
    }
}

/// Coefficient scan order (H.265 6.5.3), matching the decoder's `ScanOrder`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOrder {
    Diagonal = 0,
    Horizontal = 1,
    Vertical = 2,
}

/// H.265 Table 6-5 / decoder `get_scan_order`: directional scan for small
/// luma/chroma transform blocks under (near-)horizontal/vertical intra modes.
pub fn get_scan_order(
    log2_size: u8,
    intra_mode: u8,
    c_idx: u8,
    chroma_array_type: u8,
) -> ScanOrder {
    let use_directional = if c_idx == 0 {
        log2_size == 2 || log2_size == 3
    } else if chroma_array_type == 3 {
        log2_size == 2 || log2_size == 3
    } else {
        log2_size == 2
    };

    if use_directional {
        if (6..=14).contains(&intra_mode) {
            ScanOrder::Vertical
        } else if (22..=30).contains(&intra_mode) {
            ScanOrder::Horizontal
        } else {
            ScanOrder::Diagonal
        }
    } else {
        ScanOrder::Diagonal
    }
}

static SCAN_ORDER_4X4_DIAG: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (1, 3),
    (2, 2),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
];
static SCAN_ORDER_4X4_HORIZ: [(u8, u8); 16] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
];
static SCAN_ORDER_4X4_VERT: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
];

pub(crate) fn get_scan_4x4(order: ScanOrder) -> &'static [(u8, u8); 16] {
    match order {
        ScanOrder::Diagonal => &SCAN_ORDER_4X4_DIAG,
        ScanOrder::Horizontal => &SCAN_ORDER_4X4_HORIZ,
        ScanOrder::Vertical => &SCAN_ORDER_4X4_VERT,
    }
}

#[rustfmt::skip]
static SCAN_2X2_DIAG: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
#[rustfmt::skip]
static SCAN_2X2_HORIZ: [(u8, u8); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];
static SCAN_1X1: [(u8, u8); 1] = [(0, 0)];
#[rustfmt::skip]
static SCAN_4X4_SB_DIAG: [(u8, u8); 16] = [
    (0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2),
    (2, 1), (3, 0), (1, 3), (2, 2), (3, 1), (2, 3), (3, 2), (3, 3),
];
#[rustfmt::skip]
static SCAN_8X8_SB_DIAG: [(u8, u8); 64] = [
    (0,0),(0,1),(1,0),(0,2),(1,1),(2,0),(0,3),(1,2),(2,1),(3,0),(0,4),(1,3),(2,2),(3,1),(4,0),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(0,6),(1,5),(2,4),(3,3),(4,2),(5,1),(6,0),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(1,7),(2,6),(3,5),(4,4),(5,3),(6,2),(7,1),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(3,7),(4,6),(5,5),(6,4),(7,3),(4,7),(5,6),(6,5),(7,4),(5,7),(6,6),(7,5),(6,7),(7,6),(7,7),
];

/// Sub-block scan order for a `2^log2_size` TU (grid of `2^(log2_size-2)`
/// sub-blocks). Mirrors the decoder's `get_scan_sub_block`.
pub(crate) fn get_scan_sub_block(log2_size: u8, order: ScanOrder) -> &'static [(u8, u8)] {
    match log2_size {
        2 => &SCAN_1X1,
        3 => match order {
            ScanOrder::Horizontal => &SCAN_2X2_HORIZ,
            _ => &SCAN_2X2_DIAG,
        },
        4 => &SCAN_4X4_SB_DIAG,
        5 => &SCAN_8X8_SB_DIAG,
        _ => &SCAN_1X1,
    }
}

/// H.265 Table 9-39: sig_coeff_flag context map for a 4x4 TU.
static CTX_IDX_MAP_4X4: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// sig_coeff_flag context index (H.265 9.3.4.2.5), identical to the decoder's
/// `calc_sig_coeff_flag_ctx`.
pub(crate) fn calc_sig_coeff_flag_ctx(
    x_c: u8,
    y_c: u8,
    log2_size: u8,
    c_idx: u8,
    scan_idx: u8,
    prev_csbf: u8,
) -> usize {
    let sb_width = 1u8 << (log2_size - 2);

    let sig_ctx = if sb_width == 1 {
        CTX_IDX_MAP_4X4[(y_c as usize * 4 + x_c as usize).min(15)]
    } else if x_c == 0 && y_c == 0 {
        0
    } else {
        let x_s = x_c >> 2;
        let y_s = y_c >> 2;
        let x_p = x_c & 3;
        let y_p = y_c & 3;

        let mut ctx = match prev_csbf {
            0 => {
                if x_p + y_p >= 3 {
                    0
                } else if x_p + y_p > 0 {
                    1
                } else {
                    2
                }
            }
            1 => {
                if y_p == 0 {
                    2
                } else if y_p == 1 {
                    1
                } else {
                    0
                }
            }
            2 => {
                if x_p == 0 {
                    2
                } else if x_p == 1 {
                    1
                } else {
                    0
                }
            }
            _ => 2,
        };

        if c_idx == 0 {
            if x_s + y_s > 0 {
                ctx += 3;
            }
            if sb_width == 2 {
                ctx += if scan_idx == 0 { 9 } else { 15 };
            } else {
                ctx += 21;
            }
        } else if sb_width == 2 {
            ctx += 9;
        } else {
            ctx += 12;
        }
        ctx
    };

    ctx::SIG_COEFF_FLAG + if c_idx > 0 { 27 } else { 0 } + sig_ctx as usize
}

/// Encode one transform block's quantized coefficients as `residual_coding()`.
///
/// `coeffs` is row-major with stride `1 << log2_size`. There must be at least
/// one non-zero coefficient (the caller only invokes this when `cbf == 1`).
/// When `sign_data_hiding` triggers for a sub-block, the hidden coefficient's
/// sign must already be parity-consistent (the quantizer's responsibility);
/// this writer omits that sign bit exactly as the decoder infers it.
#[allow(clippy::too_many_arguments)]
pub fn encode_residual(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
) {
    let mut sink = CabacWriter {
        enc,
        w,
        _marker: core::marker::PhantomData,
    };
    residual_syntax(
        &mut sink,
        ctxs,
        coeffs,
        log2_size,
        c_idx,
        scan_order,
        sign_data_hiding,
    );
}

/// Estimate `residual_coding()` cost in x265 fixed-point 1/32768-bit units,
/// updating `ctxs` exactly as real coding would.
pub fn estimate_residual_bits(
    ctxs: &mut Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
) -> u64 {
    let mut sink = CabacEstimator::new();
    residual_syntax(
        &mut sink,
        ctxs,
        coeffs,
        log2_size,
        c_idx,
        scan_order,
        sign_data_hiding,
    );
    sink.frac_bits()
}

/// HEVC sign-data-hiding: make each qualifying coding group's first significant
/// coefficient sign parity-consistent with the group's |level| sum, so the
/// decoder can infer that sign from the parity instead of coding it. Faithful
/// port of x265 `Quant::signBitHidingHDQ` (common/quant.cpp) operating on the
/// final quantized `levels` in place. For every 4x4 group whose significant
/// span is `>= 4`, if the first coeff's sign disagrees with `sum & 1`, the
/// minimum-distortion +/-1 level change — ranked by each coefficient's
/// quantization remainder `deltaU` (recomputed from `coeffs`, the pre-quant
/// transform coefficients) — flips the group parity. The residual writer omits
/// exactly the hidden signs this pass makes consistent. Returns the updated
/// non-zero coefficient count.
pub fn apply_sign_data_hiding(
    levels: &mut [i16],
    coeffs: &[i16],
    log2_size: u8,
    scan_order: ScanOrder,
    scale: i64,
    qbits: i32,
    mut nnz: u32,
) -> u32 {
    const SBH_THRESHOLD: usize = 4;
    let size = 1usize << log2_size;
    let scan_sub = get_scan_sub_block(log2_size, scan_order);
    let scan_pos = get_scan_4x4(scan_order);
    let (last_sb_idx, last_pos_in_sb) = find_last_sig(levels, size, scan_sub, scan_pos);

    let blk_pos = |sbx: usize, sby: usize, n: usize| -> Option<usize> {
        let (px, py) = scan_pos[n];
        let x = sbx * 4 + px as usize;
        let y = sby * 4 + py as usize;
        (x < size && y < size).then_some(y * size + x)
    };
    // x265 `deltaU`: signed rounding remainder at 8-bit fractional precision.
    // > 0 means the coeff was truncated down (incrementing the level is cheap).
    let delta_u = |levels: &[i16], blk: usize| -> i64 {
        let tmp = (coeffs[blk].unsigned_abs() as i64) * scale;
        let lvl = levels[blk].unsigned_abs() as i64;
        (tmp - (lvl << qbits)) >> (qbits - 8)
    };

    for (sb_idx, &(sbx, sby)) in scan_sub.iter().enumerate() {
        if sb_idx > last_sb_idx {
            break;
        }
        let sbx = sbx as usize;
        let sby = sby as usize;

        let mut first_nz = None;
        let mut last_nz = 0usize;
        for n in 0..16 {
            if let Some(blk) = blk_pos(sbx, sby, n) {
                if levels[blk] != 0 {
                    if first_nz.is_none() {
                        first_nz = Some(n);
                    }
                    last_nz = n;
                }
            }
        }
        let first_nz = match first_nz {
            Some(f) => f,
            None => continue,
        };
        if last_nz - first_nz < SBH_THRESHOLD {
            continue;
        }

        let first_blk = blk_pos(sbx, sby, first_nz).unwrap();
        let signbit = if levels[first_blk] > 0 { 0u32 } else { 1 };
        let mut abs_sum: i64 = 0;
        for n in first_nz..=last_nz {
            if let Some(blk) = blk_pos(sbx, sby, n) {
                abs_sum += levels[blk].unsigned_abs() as i64;
            }
        }
        if signbit == (abs_sum as u32 & 1) {
            continue; // parity already encodes the hidden sign correctly
        }

        // Find the cheapest +/-1 level change that flips the group parity.
        let cand_max = if sb_idx == last_sb_idx {
            last_pos_in_sb
        } else {
            15
        };
        let mut min_cost = i64::MAX;
        let mut final_change = 0i32;
        let mut min_blk = usize::MAX;
        for n in (0..=cand_max).rev() {
            let blk = match blk_pos(sbx, sby, n) {
                Some(b) => b,
                None => continue,
            };
            let cur_sig = levels[blk] != 0;
            let lower_sig = (0..n).any(|m| blk_pos(sbx, sby, m).is_some_and(|b| levels[b] != 0));
            let du = delta_u(levels, blk);
            let (cost, change) = if cur_sig {
                if du > 0 {
                    (-du, 1)
                } else if !lower_sig && levels[blk].unsigned_abs() == 1 {
                    // sole/first coeff: don't shrink it to zero (would move firstNZ)
                    (i64::MAX, 0)
                } else {
                    (du, -1)
                }
            } else if !lower_sig {
                // leading-zero region: a new coeff here becomes the first
                // significant one, so only allow it if its sign matches signbit.
                let this_sign = if coeffs[blk] >= 0 { 0u32 } else { 1 };
                if this_sign != signbit {
                    (i64::MAX, 0)
                } else {
                    (-du, 1)
                }
            } else {
                (-du, 1)
            };
            if cost < min_cost {
                min_cost = cost;
                final_change = change;
                min_blk = blk;
            }
        }
        // Guaranteed fallback: incrementing the first coeff's magnitude always
        // flips parity and preserves its sign (only reached if every ranked
        // candidate was disallowed, which the >=4 span makes unreachable).
        if min_blk == usize::MAX || final_change == 0 {
            min_blk = first_blk;
            final_change = 1;
        }

        // Honor the coeff clamp (mirror x265).
        if levels[min_blk] == 32767 || levels[min_blk] == -32768 {
            final_change = -1;
        }

        if levels[min_blk] == 0 {
            nnz += 1;
            levels[min_blk] = if coeffs[min_blk] >= 0 { 1 } else { -1 };
        } else {
            let sign = levels[min_blk].signum() as i32;
            let new_abs = levels[min_blk].unsigned_abs() as i32 + final_change;
            if new_abs == 0 {
                nnz -= 1;
                levels[min_blk] = 0;
            } else {
                levels[min_blk] = (sign * new_abs) as i16;
            }
        }
    }
    nnz
}

#[allow(clippy::too_many_arguments)]
fn residual_syntax<S: CabacSyntax>(
    sink: &mut S,
    ctxs: &mut Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
) {
    let size = 1usize << log2_size;
    let scan_sub = get_scan_sub_block(log2_size, scan_order);
    let scan_pos = get_scan_4x4(scan_order);
    let scan_idx = scan_order as u8;
    let sb_width = size / 4;

    let (last_sb_idx, last_pos_in_sb) = find_last_sig(coeffs, size, scan_sub, scan_pos);
    encode_last_position(
        sink,
        ctxs,
        scan_order,
        scan_sub,
        scan_pos,
        last_sb_idx,
        last_pos_in_sb,
        log2_size,
        c_idx,
    );

    let cfg = SubBlockCfg {
        coeffs,
        override_coeff: None,
        size,
        sb_width,
        log2_size,
        c_idx,
        scan_idx,
        scan_sub,
        scan_pos,
        last_sb_idx,
        last_pos_in_sb,
        sign_data_hiding,
    };
    let mut coded_sb_flags = [[false; 8]; 8];
    let mut prev_subblock_had_gt1 = false;
    for sb_idx in (0..=last_sb_idx).rev() {
        cfg.code_sub_block(
            sink,
            ctxs,
            sb_idx,
            &mut prev_subblock_had_gt1,
            &mut coded_sb_flags,
        );
    }
}

/// Find the last significant coefficient (in combined sub-block + intra-block
/// scan order). Returns `(last_sb_idx, last_pos_in_sb)`.
fn find_last_sig(
    coeffs: &[i16],
    size: usize,
    scan_sub: &[(u8, u8)],
    scan_pos: &[(u8, u8); 16],
) -> (usize, usize) {
    let mut last_sb_idx = 0usize;
    let mut last_pos_in_sb = 0usize;
    for (sbi, &(sbx, sby)) in scan_sub.iter().enumerate() {
        for (n, &(px, py)) in scan_pos.iter().enumerate() {
            let x = sbx as usize * 4 + px as usize;
            let y = sby as usize * 4 + py as usize;
            if x < size && y < size && coeffs[y * size + x] != 0 {
                last_sb_idx = sbi;
                last_pos_in_sb = n;
            }
        }
    }
    (last_sb_idx, last_pos_in_sb)
}

/// Emit `last_sig_coeff_x/y` prefix + bypass suffix for the given last position.
#[allow(clippy::too_many_arguments)]
fn encode_last_position<S: CabacSyntax>(
    sink: &mut S,
    ctxs: &mut Contexts,
    scan_order: ScanOrder,
    scan_sub: &[(u8, u8)],
    scan_pos: &[(u8, u8); 16],
    last_sb_idx: usize,
    last_pos_in_sb: usize,
    log2_size: u8,
    c_idx: u8,
) {
    let (lsb_x, lsb_y) = scan_sub[last_sb_idx];
    let (lp_x, lp_y) = scan_pos[last_pos_in_sb];
    let last_x = lsb_x as u32 * 4 + lp_x as u32;
    let last_y = lsb_y as u32 * 4 + lp_y as u32;

    // The encoded prefix coordinates are swapped relative to actual coords for
    // a vertical scan (decoder swaps back after reading).
    let (raw_x, raw_y) = if scan_order == ScanOrder::Vertical {
        (last_y, last_x)
    } else {
        (last_x, last_y)
    };

    let (xnb, xsuf) = encode_last_prefix(sink, ctxs, raw_x, log2_size, c_idx, true);
    let (ynb, ysuf) = encode_last_prefix(sink, ctxs, raw_y, log2_size, c_idx, false);
    if raw_x > 3 {
        encode_bypass_bits(sink, xsuf, xnb);
    }
    if raw_y > 3 {
        encode_bypass_bits(sink, ysuf, ynb);
    }
}

/// Immutable per-block configuration shared by `residual_syntax` (full encode)
/// and the RDOQ bit-cost cache, so both walk the residual sub-blocks through
/// the exact same code. `override_coeff` lets the cache evaluate one candidate
/// coefficient value without mutating the caller's level array.
struct SubBlockCfg<'a> {
    coeffs: &'a [i16],
    override_coeff: Option<(usize, i16)>,
    size: usize,
    sb_width: usize,
    log2_size: u8,
    c_idx: u8,
    scan_idx: u8,
    scan_sub: &'a [(u8, u8)],
    scan_pos: &'a [(u8, u8); 16],
    last_sb_idx: usize,
    last_pos_in_sb: usize,
    sign_data_hiding: bool,
}

impl SubBlockCfg<'_> {
    /// Code (or estimate) one 4x4 coefficient sub-block at scan index `sb_idx`,
    /// advancing the carried state (`prev_subblock_had_gt1`, `coded_sb_flags`)
    /// and the sink/contexts exactly as `residual_coding()` does. Verbatim port
    /// of the former inline sub-block loop body.
    fn code_sub_block<S: CabacSyntax>(
        &self,
        sink: &mut S,
        ctxs: &mut Contexts,
        sb_idx: usize,
        prev_subblock_had_gt1: &mut bool,
        coded_sb_flags: &mut [[bool; 8]; 8],
    ) {
        let size = self.size;
        let sb_width = self.sb_width;
        let log2_size = self.log2_size;
        let c_idx = self.c_idx;
        let scan_idx = self.scan_idx;
        let scan_sub = self.scan_sub;
        let scan_pos = self.scan_pos;
        let last_sb_idx = self.last_sb_idx;
        let last_pos_in_sb = self.last_pos_in_sb;
        let sign_data_hiding = self.sign_data_hiding;
        let coeffs = self.coeffs;
        let override_coeff = self.override_coeff;
        let at = |x: usize, y: usize| -> i16 {
            let idx = y * size + x;
            match override_coeff {
                Some((oi, ov)) if oi == idx => ov,
                _ => coeffs[idx],
            }
        };

        let (sb_x, sb_y) = scan_sub[sb_idx];
        let sb_x = sb_x as usize;
        let sb_y = sb_y as usize;

        let right_coded = sb_x + 1 < sb_width && coded_sb_flags[sb_y][sb_x + 1];
        let below_coded = sb_y + 1 < sb_width && coded_sb_flags[sb_y + 1][sb_x];
        let csbf_neighbors = (right_coded as u8) | ((below_coded as u8) << 1);

        // Does this sub-block contain any significant coefficient?
        let sb_has_sig = scan_pos.iter().any(|&(px, py)| {
            let x = sb_x * 4 + px as usize;
            let y = sb_y * 4 + py as usize;
            x < size && y < size && at(x, y) != 0
        });

        let (sb_coded, infer_sb_dc_sig) = if sb_idx > 0 && sb_idx < last_sb_idx {
            encode_coded_sub_block_flag(sink, ctxs, c_idx, csbf_neighbors, sb_has_sig);
            (sb_has_sig, sb_has_sig)
        } else {
            (true, false)
        };

        if sb_coded {
            coded_sb_flags[sb_y][sb_x] = true;
        }
        let prev_csbf = csbf_neighbors;
        if !sb_coded {
            return;
        }

        let start_pos = if sb_idx == last_sb_idx {
            last_pos_in_sb
        } else {
            15
        };

        // Significance of each scan position within the sub-block.
        let mut sig = [false; 16];
        let coeff_at = |n: usize| -> i16 {
            let (px, py) = scan_pos[n];
            let x = sb_x * 4 + px as usize;
            let y = sb_y * 4 + py as usize;
            if x < size && y < size {
                at(x, y)
            } else {
                0
            }
        };
        for n in 0..=start_pos {
            sig[n] = coeff_at(n) != 0;
        }

        // last_coeff: positions whose sig_coeff_flag is explicitly coded.
        let last_coeff = if sb_idx == last_sb_idx {
            // start_pos is the known last significant coeff (no bin).
            start_pos.saturating_sub(1)
        } else {
            15
        };

        for n in (1..=last_coeff).rev() {
            let (px, py) = scan_pos[n];
            let xc = sb_x as u8 * 4 + px;
            let yc = sb_y as u8 * 4 + py;
            let ci = calc_sig_coeff_flag_ctx(xc, yc, log2_size, c_idx, scan_idx, prev_csbf);
            sink.encode_bin(sig[n] as u8, ctxs.get(ci));
        }

        // DC (position 0): coded unless it is the known last coeff, or unless
        // the decoder infers it (middle coded sub-block with no other sig).
        if start_pos > 0 {
            let others_sig = (1..=last_coeff).any(|n| sig[n]);
            let can_infer_dc = infer_sb_dc_sig && !others_sig;
            if !can_infer_dc {
                let ci = calc_sig_coeff_flag_ctx(
                    sb_x as u8 * 4,
                    sb_y as u8 * 4,
                    log2_size,
                    c_idx,
                    scan_idx,
                    prev_csbf,
                );
                sink.encode_bin(sig[0] as u8, ctxs.get(ci));
            }
        }

        // Collect significant positions (high scan pos -> low).
        let mut sig_positions = [0usize; 16];
        let mut n_sig = 0usize;
        for n in (0..=start_pos).rev() {
            if sig[n] {
                sig_positions[n_sig] = n;
                n_sig += 1;
            }
        }
        if n_sig == 0 {
            return;
        }

        let base = if sb_idx == 0 || c_idx > 0 { 0u8 } else { 2 };
        let ctx_set = base + *prev_subblock_had_gt1 as u8;

        // greater1 / greater2 flags.
        let max_g1 = n_sig.min(8);
        let mut greater1_ctx = 1u8;
        let mut last_greater1_flag = false;
        let mut this_subblock_had_gt1 = false;
        let mut first_g1_idx: Option<usize> = None;
        // base level used as the starting point for coeff_abs_level_remaining.
        let mut rem_base = [0i16; 16];
        let mut needs_remaining = [false; 16];

        for (g1_count, &n) in sig_positions[..n_sig].iter().enumerate() {
            let abs_level = coeff_at(n).unsigned_abs();
            if g1_count >= max_g1 {
                // Beyond the first 8: base level 1, always send remaining.
                rem_base[n] = 1;
                needs_remaining[n] = true;
                continue;
            }

            if g1_count > 0 && greater1_ctx > 0 {
                greater1_ctx = if last_greater1_flag {
                    0
                } else {
                    greater1_ctx + 1
                };
            }

            let g1 = abs_level > 1;
            let ci = ctx::COEFF_ABS_LEVEL_GREATER1_FLAG
                + if c_idx > 0 { 16 } else { 0 }
                + ctx_set as usize * 4
                + (greater1_ctx as usize).min(3);
            sink.encode_bin(g1 as u8, ctxs.get(ci));
            last_greater1_flag = g1;

            if g1 {
                this_subblock_had_gt1 = true;
                if first_g1_idx.is_none() {
                    first_g1_idx = Some(n);
                    rem_base[n] = 2; // refined below by greater2
                } else {
                    rem_base[n] = 2;
                    needs_remaining[n] = true;
                }
            } else {
                rem_base[n] = 1; // level exactly 1
            }
        }

        if let Some(g1_idx) = first_g1_idx {
            let abs_level = coeff_at(g1_idx).unsigned_abs();
            let g2 = abs_level > 2;
            let ci = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG
                + if c_idx > 0 { 4 } else { 0 }
                + ctx_set as usize;
            sink.encode_bin(g2 as u8, ctxs.get(ci));
            if g2 {
                rem_base[g1_idx] = 3;
                needs_remaining[g1_idx] = true;
            }
        }

        // Sign data hiding decision (H.265 9.3.4.3).
        let first_sig_pos = (0..=start_pos).find(|&n| sig[n]).unwrap();
        let last_sig_pos = (0..=start_pos).rev().find(|&n| sig[n]).unwrap();
        let sign_hidden = sign_data_hiding && (last_sig_pos - first_sig_pos) > 3;

        // Signs (bypass), high scan pos -> low; the lowest-pos sign is hidden.
        for i in 0..n_sig.saturating_sub(1) {
            let neg = coeff_at(sig_positions[i]) < 0;
            sink.encode_bin_ep(neg as u8);
        }
        if !sign_hidden {
            let neg = coeff_at(sig_positions[n_sig - 1]) < 0;
            sink.encode_bin_ep(neg as u8);
        }

        // coeff_abs_level_remaining, high scan pos -> low, adaptive Rice.
        let mut rice_param = 0u8;
        for &n in sig_positions[..n_sig].iter() {
            if !needs_remaining[n] {
                continue;
            }
            let base_level = rem_base[n];
            let value = (coeff_at(n).unsigned_abs() as i32 - base_level as i32) as u32;
            encode_coeff_abs_level_remaining(sink, value, rice_param);
            let full = base_level.unsigned_abs() as u32 + value;
            let threshold = 3u32 * (1 << rice_param);
            if full > threshold {
                rice_param = (rice_param + 1).min(4);
            }
        }

        *prev_subblock_had_gt1 = this_subblock_had_gt1;
    }
}

/// Row-major coefficient position -> its (sub-block scan index, position within
/// the sub-block's 4x4 scan).
#[derive(Clone, Copy)]
struct CoeffLoc {
    sb_idx: usize,
    pos_in_sb: usize,
}

fn build_coeff_locs(log2_size: u8, scan_order: ScanOrder) -> Vec<CoeffLoc> {
    let size = 1usize << log2_size;
    let scan_sub = get_scan_sub_block(log2_size, scan_order);
    let scan_pos = get_scan_4x4(scan_order);
    let mut locs = vec![
        CoeffLoc {
            sb_idx: 0,
            pos_in_sb: 0,
        };
        size * size
    ];
    for (sb_idx, &(sbx, sby)) in scan_sub.iter().enumerate() {
        for (pos, &(px, py)) in scan_pos.iter().enumerate() {
            let x = sbx as usize * 4 + px as usize;
            let y = sby as usize * 4 + py as usize;
            if x < size && y < size {
                locs[y * size + x] = CoeffLoc {
                    sb_idx,
                    pos_in_sb: pos,
                };
            }
        }
    }
    locs
}

/// CABAC state captured immediately before a residual sub-block is coded, so a
/// one-coefficient RDOQ candidate can resume coding from here instead of
/// re-encoding the whole block.
#[derive(Clone)]
struct ResidualBoundary {
    bits_before: u64,
    ctxs: Contexts,
    coded_sb_flags: [[bool; 8]; 8],
    prev_subblock_had_gt1: bool,
}

/// Exact sub-block-boundary cache for RDOQ candidate bit-cost evaluation.
///
/// Built once from the current `levels`, it records the CABAC state before each
/// residual sub-block. [`Self::estimate_one_change`] then returns the exact
/// `residual_coding()` bit cost of changing a single coefficient by replaying
/// only that coefficient's sub-block plus the lower-frequency tail from the
/// cached boundary — never the whole block. The result is bit-identical to a
/// full [`estimate_residual_bits`]; the cache must be rebuilt after any
/// coefficient is actually changed.
pub struct ResidualEstimateCache {
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
    size: usize,
    sb_width: usize,
    scan_sub: &'static [(u8, u8)],
    scan_pos: &'static [(u8, u8); 16],
    last_sb_idx: usize,
    last_pos_in_sb: usize,
    coeff_locs: Vec<CoeffLoc>,
    boundaries: Vec<Option<ResidualBoundary>>,
    total_bits: u64,
}

impl ResidualEstimateCache {
    pub fn build(
        base_ctxs: &Contexts,
        levels: &[i16],
        log2_size: u8,
        c_idx: u8,
        scan_order: ScanOrder,
        sign_data_hiding: bool,
    ) -> Self {
        let size = 1usize << log2_size;
        let scan_sub = get_scan_sub_block(log2_size, scan_order);
        let scan_pos = get_scan_4x4(scan_order);
        let scan_idx = scan_order as u8;
        let sb_width = size / 4;
        let (last_sb_idx, last_pos_in_sb) = find_last_sig(levels, size, scan_sub, scan_pos);

        let mut sink = CabacEstimator::new();
        let mut ctxs = base_ctxs.clone();
        encode_last_position(
            &mut sink,
            &mut ctxs,
            scan_order,
            scan_sub,
            scan_pos,
            last_sb_idx,
            last_pos_in_sb,
            log2_size,
            c_idx,
        );

        let cfg = SubBlockCfg {
            coeffs: levels,
            override_coeff: None,
            size,
            sb_width,
            log2_size,
            c_idx,
            scan_idx,
            scan_sub,
            scan_pos,
            last_sb_idx,
            last_pos_in_sb,
            sign_data_hiding,
        };
        let mut boundaries: Vec<Option<ResidualBoundary>> = vec![None; scan_sub.len()];
        let mut coded_sb_flags = [[false; 8]; 8];
        let mut prev_subblock_had_gt1 = false;
        for sb_idx in (0..=last_sb_idx).rev() {
            boundaries[sb_idx] = Some(ResidualBoundary {
                bits_before: sink.frac_bits(),
                ctxs: ctxs.clone(),
                coded_sb_flags,
                prev_subblock_had_gt1,
            });
            cfg.code_sub_block(
                &mut sink,
                &mut ctxs,
                sb_idx,
                &mut prev_subblock_had_gt1,
                &mut coded_sb_flags,
            );
        }
        let total_bits = sink.frac_bits();

        Self {
            log2_size,
            c_idx,
            scan_order,
            sign_data_hiding,
            size,
            sb_width,
            scan_sub,
            scan_pos,
            last_sb_idx,
            last_pos_in_sb,
            coeff_locs: build_coeff_locs(log2_size, scan_order),
            boundaries,
            total_bits,
        }
    }

    /// Exact `residual_coding()` bit cost of the levels the cache was built from.
    pub fn total_bits(&self) -> u64 {
        self.total_bits
    }

    /// Incrementally fold an accepted one-coefficient change into the cache.
    /// `levels` must already hold the new value at `changed_idx`. Only the
    /// changed coefficient's sub-block and the lower-frequency tail are
    /// recomputed (boundaries above are unaffected, since they are coded
    /// first). Returns `false` when a full [`Self::build`] is required: a 4x4
    /// block, or a change that zeroed the last significant coefficient (which
    /// moves the coded last position).
    pub fn apply_change(&mut self, levels: &[i16], changed_idx: usize) -> bool {
        if self.log2_size < 3 {
            return false;
        }
        let loc = self.coeff_locs[changed_idx];
        if levels[changed_idx] == 0
            && loc.sb_idx == self.last_sb_idx
            && loc.pos_in_sb == self.last_pos_in_sb
        {
            return false;
        }
        let boundary = match &self.boundaries[loc.sb_idx] {
            Some(b) => b.clone(),
            None => return false,
        };

        let mut sink = CabacEstimator::new();
        sink.add_frac_bits(boundary.bits_before);
        let mut ctxs = boundary.ctxs.clone();
        let mut coded_sb_flags = boundary.coded_sb_flags;
        let mut prev_subblock_had_gt1 = boundary.prev_subblock_had_gt1;
        let cfg = SubBlockCfg {
            coeffs: levels,
            override_coeff: None,
            size: self.size,
            sb_width: self.sb_width,
            log2_size: self.log2_size,
            c_idx: self.c_idx,
            scan_idx: self.scan_order as u8,
            scan_sub: self.scan_sub,
            scan_pos: self.scan_pos,
            last_sb_idx: self.last_sb_idx,
            last_pos_in_sb: self.last_pos_in_sb,
            sign_data_hiding: self.sign_data_hiding,
        };
        for sb_idx in (0..=loc.sb_idx).rev() {
            self.boundaries[sb_idx] = Some(ResidualBoundary {
                bits_before: sink.frac_bits(),
                ctxs: ctxs.clone(),
                coded_sb_flags,
                prev_subblock_had_gt1,
            });
            cfg.code_sub_block(
                &mut sink,
                &mut ctxs,
                sb_idx,
                &mut prev_subblock_had_gt1,
                &mut coded_sb_flags,
            );
        }
        self.total_bits = sink.frac_bits();
        true
    }

    /// Exact bit cost of setting `base_levels[changed_idx] = new_level`, reusing
    /// the cached prefix. Returns `None` when the fast path does not apply and
    /// the caller should fall back to a full [`estimate_residual_bits`]:
    /// single-sub-block (4x4) blocks (no tail to save), or a change that zeroes
    /// the current last significant coefficient (which moves the coded last
    /// position, invalidating the cached prefix). `base_levels` must equal the
    /// build-time levels except possibly at `changed_idx`.
    pub fn estimate_one_change(
        &self,
        base_levels: &[i16],
        changed_idx: usize,
        new_level: i16,
    ) -> Option<u64> {
        if self.log2_size < 3 {
            return None;
        }
        let loc = self.coeff_locs[changed_idx];
        if new_level == 0 && loc.sb_idx == self.last_sb_idx && loc.pos_in_sb == self.last_pos_in_sb
        {
            return None;
        }
        let boundary = self.boundaries[loc.sb_idx].as_ref()?;

        let mut sink = CabacEstimator::new();
        let mut ctxs = boundary.ctxs.clone();
        let mut coded_sb_flags = boundary.coded_sb_flags;
        let mut prev_subblock_had_gt1 = boundary.prev_subblock_had_gt1;
        let cfg = SubBlockCfg {
            coeffs: base_levels,
            override_coeff: Some((changed_idx, new_level)),
            size: self.size,
            sb_width: self.sb_width,
            log2_size: self.log2_size,
            c_idx: self.c_idx,
            scan_idx: self.scan_order as u8,
            scan_sub: self.scan_sub,
            scan_pos: self.scan_pos,
            last_sb_idx: self.last_sb_idx,
            last_pos_in_sb: self.last_pos_in_sb,
            sign_data_hiding: self.sign_data_hiding,
        };
        for sb_idx in (0..=loc.sb_idx).rev() {
            cfg.code_sub_block(
                &mut sink,
                &mut ctxs,
                sb_idx,
                &mut prev_subblock_had_gt1,
                &mut coded_sb_flags,
            );
        }
        Some(boundary.bits_before + sink.frac_bits())
    }
}

/// Encode `coded_sub_block_flag` for a middle sub-block (H.265 9.3.4.2.4).
fn encode_coded_sub_block_flag<S: CabacSyntax>(
    sink: &mut S,
    ctxs: &mut Contexts,
    c_idx: u8,
    csbf_neighbors: u8,
    coded: bool,
) {
    let csbf_ctx = (csbf_neighbors != 0) as usize;
    let ci = ctx::CODED_SUB_BLOCK_FLAG + csbf_ctx + if c_idx > 0 { 2 } else { 0 };
    sink.encode_bin(coded as u8, ctxs.get(ci));
}

/// Encode `last_sig_coeff_{x,y}_prefix` (context-coded truncated unary).
/// Returns `(suffix_bits, suffix_value)` for the bypass suffix the caller
/// emits afterwards (only when `value > 3`).
fn encode_last_prefix<S: CabacSyntax>(
    sink: &mut S,
    ctxs: &mut Contexts,
    value: u32,
    log2_size: u8,
    c_idx: u8,
    is_x: bool,
) -> (u8, u32) {
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

    let (prefix, n_bits, suffix) = if value <= 3 {
        (value, 0u8, 0u32)
    } else {
        let group = 31 - value.leading_zeros(); // floor(log2(value)), value >= 4
        let prefix = 2 * group + ((value >> (group - 1)) & 1);
        let n_bits = (group - 1) as u8;
        let base = (2 + (prefix & 1)) << n_bits;
        (prefix, n_bits, value - base)
    };

    for j in 0..prefix {
        let ci = ctx_base + ctx_offset + (j as usize >> ctx_shift as usize);
        sink.encode_bin(1, ctxs.get(ci));
    }
    if prefix < max_prefix {
        let ci = ctx_base + ctx_offset + (prefix as usize >> ctx_shift as usize);
        sink.encode_bin(0, ctxs.get(ci));
    }

    (n_bits, suffix)
}

/// Encode `coeff_abs_level_remaining` (H.265 9.3.3.2): truncated-Rice prefix
/// (0..=3) then EGk, all in bypass, matching the decoder's parse.
fn encode_coeff_abs_level_remaining<S: CabacSyntax>(sink: &mut S, value: u32, rice: u8) {
    let rice = rice as u32;
    if value < (4u32 << rice) {
        let prefix = value >> rice;
        let suffix = value & ((1u32 << rice) - 1);
        for _ in 0..prefix {
            sink.encode_bin_ep(1);
        }
        sink.encode_bin_ep(0);
        if rice > 0 {
            encode_bypass_bits(sink, suffix, rice as u8);
        }
    } else {
        // Find prefix p >= 4 whose EGk bucket contains `value`.
        let mut p = 4u32;
        loop {
            let base = ((1u32 << (p - 3)) + 2) << rice;
            let width = 1u32 << (p - 3 + rice);
            if value < base + width {
                let suffix = value - base;
                let n_bits = (p - 3 + rice) as u8;
                for _ in 0..p {
                    sink.encode_bin_ep(1);
                }
                sink.encode_bin_ep(0);
                encode_bypass_bits(sink, suffix, n_bits);
                return;
            }
            p += 1;
        }
    }
}

/// Emit `n_bits` of `value`, MSB first, as bypass bins (the order the decoder's
/// `decode_bypass_bits` reads).
fn encode_bypass_bits<S: CabacSyntax>(sink: &mut S, value: u32, n_bits: u8) {
    for i in (0..n_bits).rev() {
        sink.encode_bin_ep(((value >> i) & 1) as u8);
    }
}
