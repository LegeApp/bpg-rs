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

use std::sync::OnceLock;

use bpg_bitstream::BitWriter;

use crate::cabac::{CabacEncoder, CabacEstimator, ContextModel};
use crate::contexts::{Contexts, ctx};

trait CabacSyntax {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel);
    fn encode_bin_ep(&mut self, bin_value: u8);
    fn encode_bins_ep(&mut self, bin_values: u32, num_bins: u32) {
        for i in (0..num_bins).rev() {
            self.encode_bin_ep(((bin_values >> i) & 1) as u8);
        }
    }
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

    fn encode_bins_ep(&mut self, bin_values: u32, num_bins: u32) {
        self.enc.encode_bins_ep(self.w, bin_values, num_bins);
    }
}

impl CabacSyntax for CabacEstimator {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel) {
        CabacEstimator::encode_bin(self, bin_value, ctx);
    }

    fn encode_bin_ep(&mut self, bin_value: u8) {
        CabacEstimator::encode_bin_ep(self, bin_value);
    }

    fn encode_bins_ep(&mut self, bin_values: u32, num_bins: u32) {
        CabacEstimator::encode_bins_ep(self, bin_values, num_bins);
    }
}

/// Static-context fractional-bit estimator: counts the same bins as
/// [`CabacEstimator`] but never advances context state (`ctx.update`). This is
/// the x265 `EstBitsSbac` / RDOQ rate model (`g_entropyBits` against the frozen
/// snapshot) — the bin costs are read from the base context states only. Used to
/// price the cheap intra-mode ranking pass without the per-bin context-evolution
/// write; the base `Contexts` is therefore left unmodified, so no per-call copy
/// is needed.
#[derive(Default)]
struct StaticEstimator {
    frac_bits: u64,
}

impl CabacSyntax for StaticEstimator {
    fn encode_bin(&mut self, bin_value: u8, ctx: &mut ContextModel) {
        // Read-only: cost from the current (frozen) state, no transition.
        self.frac_bits += ctx.entropy_bits(bin_value) as u64;
    }

    fn encode_bin_ep(&mut self, _bin_value: u8) {
        self.frac_bits += CabacEstimator::SCALE;
    }

    fn encode_bins_ep(&mut self, _bin_values: u32, num_bins: u32) {
        self.frac_bits += CabacEstimator::SCALE * num_bins as u64;
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

/// Precomputed `sig_coeff_flag` context index for every scan position of a 4×4
/// sub-block, keyed by everything [`calc_sig_coeff_flag_ctx`] actually depends
/// on: `[log2_size-2][c_idx>0][scan_idx][is_dc_subblock][prev_csbf][scan_pos]`.
/// The result is independent of the exact `(sb_x, sb_y)` — only whether the
/// sub-block is the DC sub-block matters (`x_s + y_s > 0`) — and of which
/// non-DC sub-block, so the table is small. Built once from the reference
/// function and proven equivalent by `sig_ctx_table_matches_function`.
type SigCtxTable = [[[[[[u16; 16]; 4]; 2]; 3]; 2]; 4];

fn scan_order_from_idx(idx: u8) -> ScanOrder {
    match idx {
        1 => ScanOrder::Horizontal,
        2 => ScanOrder::Vertical,
        _ => ScanOrder::Diagonal,
    }
}

fn build_sig_ctx_table() -> Box<SigCtxTable> {
    let mut t: Box<SigCtxTable> = Box::new([[[[[[0u16; 16]; 4]; 2]; 3]; 2]; 4]);
    for sbw_idx in 0..4usize {
        let log2_size = sbw_idx as u8 + 2;
        let sb_width = 1u8 << (log2_size - 2);
        for c_class in 0..2usize {
            let c_idx = c_class as u8; // 0 = luma, 1 = any chroma (c_idx > 0)
            for scan_idx in 0..3usize {
                let scan_pos = get_scan_4x4(scan_order_from_idx(scan_idx as u8));
                for is_dc in 0..2usize {
                    // Any non-DC sub-block yields the same contexts (`x_s+y_s>0`,
                    // `x_p/y_p` come from the within-4×4 position); pick (1,0).
                    let (sb_x, sb_y) = if is_dc == 1 || sb_width == 1 {
                        (0u8, 0u8)
                    } else {
                        (1u8, 0u8)
                    };
                    for prev_csbf in 0..4usize {
                        for (n, &(px, py)) in scan_pos.iter().enumerate() {
                            let x_c = sb_x * 4 + px;
                            let y_c = sb_y * 4 + py;
                            t[sbw_idx][c_class][scan_idx][is_dc][prev_csbf][n] =
                                calc_sig_coeff_flag_ctx(
                                    x_c,
                                    y_c,
                                    log2_size,
                                    c_idx,
                                    scan_idx as u8,
                                    prev_csbf as u8,
                                ) as u16;
                        }
                    }
                }
            }
        }
    }
    t
}

/// `&[u16; 16]` of `sig_coeff_flag` context indices for the current sub-block's
/// scan positions (index 0 = DC). Replaces the per-bin `calc_sig_coeff_flag_ctx`
/// in the residual traversal hot loop.
#[inline]
fn sig_ctx_row(
    log2_size: u8,
    c_idx: u8,
    scan_idx: u8,
    is_dc_sb: bool,
    prev_csbf: u8,
) -> &'static [u16; 16] {
    static TABLE: OnceLock<Box<SigCtxTable>> = OnceLock::new();
    let t = TABLE.get_or_init(build_sig_ctx_table);
    &t[(log2_size - 2) as usize][(c_idx > 0) as usize][scan_idx as usize][is_dc_sb as usize]
        [prev_csbf as usize]
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

/// Reusable exact residual-pricing state for non-writing analysis paths.
///
/// The estimator must preserve the caller's CABAC contexts, so each pricing
/// starts from a copied context state and a reset fractional-bit accumulator.
pub struct ResidualPricingScratch {
    ctxs: Contexts,
    sink: CabacEstimator,
}

impl Default for ResidualPricingScratch {
    fn default() -> Self {
        Self {
            ctxs: Contexts::new(0),
            sink: CabacEstimator::new(),
        }
    }
}

/// Estimate `residual_coding()` cost using caller-owned scratch state.
#[allow(clippy::too_many_arguments)]
pub fn estimate_residual_bits_into(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
    scratch: &mut ResidualPricingScratch,
) -> u64 {
    // `residual_coding()` only touches the residual syntax context range
    // [LAST_SIG_COEFF_X_PREFIX, SAO_MERGE_FLAG). Copying just that range avoids
    // rewriting the whole 170-context array for every trial-priced block while
    // preserving exact context evolution within the residual estimator.
    scratch.ctxs.models[ctx::LAST_SIG_COEFF_X_PREFIX..ctx::SAO_MERGE_FLAG]
        .copy_from_slice(&base_ctxs.models[ctx::LAST_SIG_COEFF_X_PREFIX..ctx::SAO_MERGE_FLAG]);
    scratch.sink = CabacEstimator::new();
    residual_syntax(
        &mut scratch.sink,
        &mut scratch.ctxs,
        coeffs,
        log2_size,
        c_idx,
        scan_order,
        sign_data_hiding,
    );
    scratch.sink.frac_bits()
}

/// Estimate `residual_coding()` cost when the caller already knows `nnz`.
///
/// The `nnz == 1 && coeffs[0] != 0` case has a fixed syntax shape: last-x/y at
/// zero, no sub-block/significance scan, one greater1 flag, optional greater2,
/// one sign bit, and optional Rice remainder. Pricing it directly avoids the
/// full residual traversal and even the residual context-range copy.
#[allow(clippy::too_many_arguments)]
pub fn estimate_residual_bits_into_nnz(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    nnz: u32,
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
    scratch: &mut ResidualPricingScratch,
) -> u64 {
    if nnz == 1 {
        if let Some(bits) = estimate_dc_only_residual_bits(base_ctxs, coeffs, log2_size, c_idx) {
            return bits;
        }
    }

    estimate_residual_bits_into(
        base_ctxs,
        coeffs,
        log2_size,
        c_idx,
        scan_order,
        sign_data_hiding,
        scratch,
    )
}

fn estimate_dc_only_residual_bits(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
) -> Option<u64> {
    let level = *coeffs.first()?;
    if level == 0 {
        return None;
    }

    let mut bits = 0u64;
    bits += dc_last_prefix_zero_bits(base_ctxs, log2_size, c_idx, true);
    bits += dc_last_prefix_zero_bits(base_ctxs, log2_size, c_idx, false);

    let abs_level = level.unsigned_abs() as u32;
    let gt1 = abs_level > 1;
    let gt1_ci = ctx::COEFF_ABS_LEVEL_GREATER1_FLAG + if c_idx > 0 { 16 } else { 0 } + 1;
    bits += base_ctxs.models[gt1_ci].entropy_bits(gt1 as u8) as u64;

    if gt1 {
        let gt2 = abs_level > 2;
        let gt2_ci = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + if c_idx > 0 { 4 } else { 0 };
        bits += base_ctxs.models[gt2_ci].entropy_bits(gt2 as u8) as u64;
        if gt2 {
            bits += CabacEstimator::SCALE as u64
                * coeff_abs_level_remaining_ep_bins(abs_level - 3, 0) as u64;
        }
    }

    bits += CabacEstimator::SCALE as u64; // sign_coeff_flag
    Some(bits)
}

fn dc_last_prefix_zero_bits(base_ctxs: &Contexts, log2_size: u8, c_idx: u8, is_x: bool) -> u64 {
    let ctx_base = if is_x {
        ctx::LAST_SIG_COEFF_X_PREFIX
    } else {
        ctx::LAST_SIG_COEFF_Y_PREFIX
    };
    let ctx_offset = if c_idx == 0 {
        3 * (log2_size as usize - 2) + ((log2_size as usize - 1) >> 2)
    } else {
        15usize
    };
    base_ctxs.models[ctx_base + ctx_offset].entropy_bits(0) as u64
}

fn coeff_abs_level_remaining_ep_bins(value: u32, rice: u8) -> u32 {
    let rice = rice as u32;
    if value < (4u32 << rice) {
        let prefix = value >> rice;
        prefix + 1 + rice
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

/// Static-context residual bit estimate (x265 `EstBitsSbac` / RDOQ rate model):
/// same residual-syntax traversal as [`estimate_residual_bits_into`] but counts
/// bins against the *frozen* base context states (no per-bin context evolution),
/// so it needs no per-call context-range copy.
///
/// MEASURED (2026-06-29, 4000×3000 q28 x265shape, env `BPG_STILLSEARCH_CHEAP_BITS_STATIC=1`):
/// ~1.2 s faster (12.8→11.5 s) but **BD-rate worse** — −0.24 dB PSNR_y for only
/// −0.9 % bytes (1296789→1285065 B). The frozen-context count is biased enough to
/// shift RD decisions; not shippable. Kept as an off-by-default A/B knob: the
/// evolving-context exact count is what holds still265's RD operating point.
pub fn estimate_residual_bits_static(
    base_ctxs: &Contexts,
    coeffs: &[i16],
    log2_size: u8,
    c_idx: u8,
    scan_order: ScanOrder,
    sign_data_hiding: bool,
) -> u64 {
    // `residual_syntax` requires `&mut Contexts` because the exact path mutates
    // context state. `StaticEstimator` never mutates, so a shadow copy that we
    // discard preserves the signature while leaving `base_ctxs` untouched.
    // The copy is the full context array but happens without the
    // `LAST_SIG..SAO` range restriction; benchmark shows it is negligible. If it
    // shows up, narrow it to the residual range like `estimate_residual_bits_into`.
    let mut shadow = base_ctxs.clone();
    let mut sink = StaticEstimator::default();
    residual_syntax(
        &mut sink,
        &mut shadow,
        coeffs,
        log2_size,
        c_idx,
        scan_order,
        sign_data_hiding,
    );
    sink.frac_bits
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

        let mut blk_pos = [usize::MAX; 16];
        let mut sig_mask = 0u16;
        for (n, slot) in blk_pos.iter_mut().enumerate() {
            let (px, py) = scan_pos[n];
            let x = sbx * 4 + px as usize;
            let y = sby * 4 + py as usize;
            if x < size && y < size {
                let blk = y * size + x;
                *slot = blk;
                sig_mask |= ((levels[blk] != 0) as u16) << n;
            }
        }

        if sig_mask == 0 {
            continue;
        }
        let first_nz = sig_mask.trailing_zeros() as usize;
        let last_nz = 15 - sig_mask.leading_zeros() as usize;
        if last_nz - first_nz < SBH_THRESHOLD {
            continue;
        }

        let first_blk = blk_pos[first_nz];
        let signbit = if levels[first_blk] > 0 { 0u32 } else { 1 };
        let mut abs_sum: i64 = 0;
        for n in first_nz..=last_nz {
            let blk = blk_pos[n];
            if blk != usize::MAX {
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
            let blk = blk_pos[n];
            if blk == usize::MAX {
                continue;
            }
            let cur_sig = levels[blk] != 0;
            let lower_sig = (sig_mask & ((1u16 << n) - 1)) != 0;
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
    if c_idx == 0 {
        match log2_size {
            2 => {
                return residual_syntax_luma::<S, 2>(
                    sink,
                    ctxs,
                    coeffs,
                    scan_order,
                    sign_data_hiding,
                );
            }
            3 => {
                return residual_syntax_luma::<S, 3>(
                    sink,
                    ctxs,
                    coeffs,
                    scan_order,
                    sign_data_hiding,
                );
            }
            4 => {
                return residual_syntax_luma::<S, 4>(
                    sink,
                    ctxs,
                    coeffs,
                    scan_order,
                    sign_data_hiding,
                );
            }
            5 => {
                return residual_syntax_luma::<S, 5>(
                    sink,
                    ctxs,
                    coeffs,
                    scan_order,
                    sign_data_hiding,
                );
            }
            _ => {}
        }
    }

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

#[allow(clippy::too_many_arguments)]
fn residual_syntax_luma<S: CabacSyntax, const LOG2_SIZE: u8>(
    sink: &mut S,
    ctxs: &mut Contexts,
    coeffs: &[i16],
    scan_order: ScanOrder,
    sign_data_hiding: bool,
) {
    let scan_sub = get_scan_sub_block(LOG2_SIZE, scan_order);
    let scan_pos = get_scan_4x4(scan_order);
    let scan_idx = scan_order as u8;

    let (last_sb_idx, last_pos_in_sb) = find_last_sig_luma::<LOG2_SIZE>(coeffs, scan_sub, scan_pos);
    encode_last_position(
        sink,
        ctxs,
        scan_order,
        scan_sub,
        scan_pos,
        last_sb_idx,
        last_pos_in_sb,
        LOG2_SIZE,
        0,
    );

    let mut coded_sb_flags = [[false; 8]; 8];
    let mut prev_subblock_had_gt1 = false;
    for sb_idx in (0..=last_sb_idx).rev() {
        code_sub_block_luma::<S, LOG2_SIZE>(
            sink,
            ctxs,
            coeffs,
            scan_idx,
            scan_sub,
            scan_pos,
            sb_idx,
            last_sb_idx,
            last_pos_in_sb,
            sign_data_hiding,
            &mut prev_subblock_had_gt1,
            &mut coded_sb_flags,
        );
    }
}

/// Find the last significant luma coefficient for a power-of-two HEVC TU.
/// Luma 4/8/16/32 sub-block grids tile exactly, so the scan positions are all
/// in-bounds and the hot search can avoid per-position geometry checks.
fn find_last_sig_luma<const LOG2_SIZE: u8>(
    coeffs: &[i16],
    scan_sub: &[(u8, u8)],
    scan_pos: &[(u8, u8); 16],
) -> (usize, usize) {
    let size = 1usize << LOG2_SIZE;

    for (sbi, &(sbx, sby)) in scan_sub.iter().enumerate().rev() {
        let base_x = sbx as usize * 4;
        let base_y = sby as usize * 4;
        for (n, &(px, py)) in scan_pos.iter().enumerate().rev() {
            let idx = (base_y + py as usize) * size + base_x + px as usize;
            if coeffs[idx] != 0 {
                return (sbi, n);
            }
        }
    }
    (0, 0)
}

#[allow(clippy::too_many_arguments)]
fn code_sub_block_luma<S: CabacSyntax, const LOG2_SIZE: u8>(
    sink: &mut S,
    ctxs: &mut Contexts,
    coeffs: &[i16],
    scan_idx: u8,
    scan_sub: &[(u8, u8)],
    scan_pos: &[(u8, u8); 16],
    sb_idx: usize,
    last_sb_idx: usize,
    last_pos_in_sb: usize,
    sign_data_hiding: bool,
    prev_subblock_had_gt1: &mut bool,
    coded_sb_flags: &mut [[bool; 8]; 8],
) {
    let size = 1usize << LOG2_SIZE;
    let sb_width = 1usize << (LOG2_SIZE - 2);

    let (sb_x, sb_y) = scan_sub[sb_idx];
    let sb_x = sb_x as usize;
    let sb_y = sb_y as usize;

    let mut sb_coeff = [0i16; 16];
    let mut sig_mask: u16 = 0;
    let base_x = sb_x * 4;
    let base_y = sb_y * 4;
    for (n, slot) in sb_coeff.iter_mut().enumerate() {
        let (px, py) = scan_pos[n];
        let idx = (base_y + py as usize) * size + base_x + px as usize;
        let v = coeffs[idx];
        *slot = v;
        sig_mask |= ((v != 0) as u16) << n;
    }

    let right_coded = sb_x + 1 < sb_width && coded_sb_flags[sb_y][sb_x + 1];
    let below_coded = sb_y + 1 < sb_width && coded_sb_flags[sb_y + 1][sb_x];
    let csbf_neighbors = (right_coded as u8) | ((below_coded as u8) << 1);

    let sb_has_sig = sig_mask != 0;
    let (sb_coded, infer_sb_dc_sig) = if sb_idx > 0 && sb_idx < last_sb_idx {
        encode_coded_sub_block_flag(sink, ctxs, 0, csbf_neighbors, sb_has_sig);
        (sb_has_sig, sb_has_sig)
    } else {
        (true, false)
    };

    if sb_coded {
        coded_sb_flags[sb_y][sb_x] = true;
    }
    if !sb_coded {
        return;
    }

    let start_pos = if sb_idx == last_sb_idx {
        last_pos_in_sb
    } else {
        15
    };
    let sig = |n: usize| -> bool { (sig_mask >> n) & 1 != 0 };

    let last_coeff = if sb_idx == last_sb_idx {
        start_pos.saturating_sub(1)
    } else {
        15
    };

    let sig_ctx = sig_ctx_row(
        LOG2_SIZE,
        0,
        scan_idx,
        sb_x == 0 && sb_y == 0,
        csbf_neighbors,
    );

    for n in (1..=last_coeff).rev() {
        let ci = sig_ctx[n] as usize;
        sink.encode_bin(sig(n) as u8, ctxs.get(ci));
    }

    if start_pos > 0 {
        let others_mask = (((1u32 << (last_coeff + 1)) - 1) as u16) & !1u16;
        let others_sig = sig_mask & others_mask != 0;
        let can_infer_dc = infer_sb_dc_sig && !others_sig;
        if !can_infer_dc {
            let ci = sig_ctx[0] as usize;
            sink.encode_bin(sig(0) as u8, ctxs.get(ci));
        }
    }

    let mut sig_positions = [0usize; 16];
    let mut n_sig = 0usize;
    let mut m = sig_mask & (((1u32 << (start_pos + 1)) - 1) as u16);
    while m != 0 {
        let n = 15 - m.leading_zeros() as usize;
        sig_positions[n_sig] = n;
        n_sig += 1;
        m &= !(1u16 << n);
    }
    if n_sig == 0 {
        return;
    }

    let base = if sb_idx == 0 { 0u8 } else { 2 };
    let ctx_set = base + *prev_subblock_had_gt1 as u8;

    let max_g1 = n_sig.min(8);
    let mut greater1_ctx = 1u8;
    let mut last_greater1_flag = false;
    let mut this_subblock_had_gt1 = false;
    let mut first_g1_idx: Option<usize> = None;
    let mut rem_base = [0i16; 16];
    let mut needs_remaining = [false; 16];

    for (g1_count, &n) in sig_positions[..n_sig].iter().enumerate() {
        let abs_level = sb_coeff[n].unsigned_abs();
        if g1_count >= max_g1 {
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
            + ctx_set as usize * 4
            + (greater1_ctx as usize).min(3);
        sink.encode_bin(g1 as u8, ctxs.get(ci));
        last_greater1_flag = g1;

        if g1 {
            this_subblock_had_gt1 = true;
            if first_g1_idx.is_none() {
                first_g1_idx = Some(n);
                rem_base[n] = 2;
            } else {
                rem_base[n] = 2;
                needs_remaining[n] = true;
            }
        } else {
            rem_base[n] = 1;
        }
    }

    if let Some(g1_idx) = first_g1_idx {
        let abs_level = sb_coeff[g1_idx].unsigned_abs();
        let g2 = abs_level > 2;
        let ci = ctx::COEFF_ABS_LEVEL_GREATER2_FLAG + ctx_set as usize;
        sink.encode_bin(g2 as u8, ctxs.get(ci));
        if g2 {
            rem_base[g1_idx] = 3;
            needs_remaining[g1_idx] = true;
        }
    }

    let first_sig_pos = sig_mask.trailing_zeros() as usize;
    let last_sig_pos = 15 - sig_mask.leading_zeros() as usize;
    let sign_hidden = sign_data_hiding && (last_sig_pos - first_sig_pos) > 3;

    let sign_bins = if sign_hidden { n_sig - 1 } else { n_sig };
    if sign_bins > 0 {
        let mut signs = 0u32;
        for &n in &sig_positions[..sign_bins] {
            signs = (signs << 1) | (sb_coeff[n] < 0) as u32;
        }
        sink.encode_bins_ep(signs, sign_bins as u32);
    }

    let mut rice_param = 0u8;
    for &n in sig_positions[..n_sig].iter() {
        if !needs_remaining[n] {
            continue;
        }
        let base_level = rem_base[n];
        let value = (sb_coeff[n].unsigned_abs() as i32 - base_level as i32) as u32;
        encode_coeff_abs_level_remaining(sink, value, rice_param);
        let full = base_level.unsigned_abs() as u32 + value;
        let threshold = 3u32 * (1 << rice_param);
        if full > threshold {
            rice_param = (rice_param + 1).min(4);
        }
    }

    *prev_subblock_had_gt1 = this_subblock_had_gt1;
}

/// Find the last significant coefficient (in combined sub-block + intra-block
/// scan order). Returns `(last_sb_idx, last_pos_in_sb)`.
fn find_last_sig(
    coeffs: &[i16],
    size: usize,
    scan_sub: &[(u8, u8)],
    scan_pos: &[(u8, u8); 16],
) -> (usize, usize) {
    // The last significant coefficient is the highest position in combined
    // sub-block + intra-sub-block scan order. Search from the end and return
    // immediately; trial residual pricing touches this path for every coded
    // candidate and sparse blocks are common after quantization.
    for (sbi, &(sbx, sby)) in scan_sub.iter().enumerate().rev() {
        for (n, &(px, py)) in scan_pos.iter().enumerate().rev() {
            let x = sbx as usize * 4 + px as usize;
            let y = sby as usize * 4 + py as usize;
            if x < size && y < size && coeffs[y * size + x] != 0 {
                return (sbi, n);
            }
        }
    }
    (0, 0)
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

        let (sb_x, sb_y) = scan_sub[sb_idx];
        let sb_x = sb_x as usize;
        let sb_y = sb_y as usize;

        // Load this sub-block's 16 coefficients once, in intra-sub-block scan
        // order, applying any `override_coeff`. Every significance/level loop
        // below then indexes this hot local array instead of re-reading the
        // (bounds-checked) coefficient plane and re-evaluating the override
        // branch on each of the ~3-4 passes — `code_sub_block` is the single
        // hottest encoder function (perf: ~26% self-time, dominated by these
        // repeated per-position reads). Out-of-bounds scan positions stay 0,
        // matching the previous `coeff_at` fallback.
        let mut sb_coeff = [0i16; 16];
        // `sig_mask` bit `n` set iff scan position `n` is significant. Built in
        // the same single pass so every significance query below is a bit op
        // instead of another 4×4 rescan (sb_has_sig / others_sig / first-last /
        // sig-position collection). Out-of-bounds positions stay 0/unset.
        let mut sig_mask: u16 = 0;
        for (n, slot) in sb_coeff.iter_mut().enumerate() {
            let (px, py) = scan_pos[n];
            let x = sb_x * 4 + px as usize;
            let y = sb_y * 4 + py as usize;
            if x < size && y < size {
                let idx = y * size + x;
                let v = match override_coeff {
                    Some((oi, ov)) if oi == idx => ov,
                    _ => coeffs[idx],
                };
                *slot = v;
                sig_mask |= ((v != 0) as u16) << n;
            }
        }

        let right_coded = sb_x + 1 < sb_width && coded_sb_flags[sb_y][sb_x + 1];
        let below_coded = sb_y + 1 < sb_width && coded_sb_flags[sb_y + 1][sb_x];
        let csbf_neighbors = (right_coded as u8) | ((below_coded as u8) << 1);

        // Does this sub-block contain any significant coefficient?
        let sb_has_sig = sig_mask != 0;

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

        // Significance test for a scan position, read from `sig_mask` (bits
        // above `start_pos` are 0 since those coefficients are zero).
        let sig = |n: usize| -> bool { (sig_mask >> n) & 1 != 0 };

        // last_coeff: positions whose sig_coeff_flag is explicitly coded.
        let last_coeff = if sb_idx == last_sb_idx {
            // start_pos is the known last significant coeff (no bin).
            start_pos.saturating_sub(1)
        } else {
            15
        };

        // Precomputed sig_coeff_flag context index per scan position (index 0 =
        // DC), replacing the per-bin `calc_sig_coeff_flag_ctx`.
        let sig_ctx = sig_ctx_row(
            log2_size,
            c_idx,
            scan_idx,
            sb_x == 0 && sb_y == 0,
            prev_csbf,
        );

        for n in (1..=last_coeff).rev() {
            let ci = sig_ctx[n] as usize;
            sink.encode_bin(sig(n) as u8, ctxs.get(ci));
        }

        // DC (position 0): coded unless it is the known last coeff, or unless
        // the decoder infers it (middle coded sub-block with no other sig).
        if start_pos > 0 {
            // Any explicitly-coded significant coeff above DC (positions 1..=last_coeff).
            let others_mask = (((1u32 << (last_coeff + 1)) - 1) as u16) & !1u16;
            let others_sig = sig_mask & others_mask != 0;
            let can_infer_dc = infer_sb_dc_sig && !others_sig;
            if !can_infer_dc {
                let ci = sig_ctx[0] as usize;
                sink.encode_bin(sig(0) as u8, ctxs.get(ci));
            }
        }

        // Collect significant positions (high scan pos -> low) straight from the
        // mask: clear the highest set bit each iteration.
        let mut sig_positions = [0usize; 16];
        let mut n_sig = 0usize;
        let mut m = sig_mask & (((1u32 << (start_pos + 1)) - 1) as u16);
        while m != 0 {
            let n = 15 - m.leading_zeros() as usize;
            sig_positions[n_sig] = n;
            n_sig += 1;
            m &= !(1u16 << n);
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
            let abs_level = sb_coeff[n].unsigned_abs();
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
            let abs_level = sb_coeff[g1_idx].unsigned_abs();
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

        // Sign data hiding decision (H.265 9.3.4.3). `sig_mask` only has bits in
        // 0..=start_pos and is non-zero here (n_sig > 0), so first/last
        // significant positions are its lowest/highest set bit.
        let first_sig_pos = sig_mask.trailing_zeros() as usize;
        let last_sig_pos = 15 - sig_mask.leading_zeros() as usize;
        let sign_hidden = sign_data_hiding && (last_sig_pos - first_sig_pos) > 3;

        // Signs (bypass), high scan pos -> low; the lowest-pos sign is hidden.
        // x265's entropy path batches bypass bins (`encodeBinsEP`); doing the
        // same here avoids a call/context dispatch per sign in the exact
        // residual-pricing hot loop while preserving the exact bit order.
        let sign_bins = if sign_hidden { n_sig - 1 } else { n_sig };
        if sign_bins > 0 {
            let mut signs = 0u32;
            for &n in &sig_positions[..sign_bins] {
                signs = (signs << 1) | (sb_coeff[n] < 0) as u32;
            }
            sink.encode_bins_ep(signs, sign_bins as u32);
        }

        // coeff_abs_level_remaining, high scan pos -> low, adaptive Rice.
        let mut rice_param = 0u8;
        for &n in sig_positions[..n_sig].iter() {
            if !needs_remaining[n] {
                continue;
            }
            let base_level = rem_base[n];
            let value = (sb_coeff[n].unsigned_abs() as i32 - base_level as i32) as u32;
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
        encode_bypass_prefix_suffix(sink, prefix, suffix, rice);
    } else {
        // Find prefix p >= 4 whose EGk bucket contains `value`.
        let mut p = 4u32;
        loop {
            let base = ((1u32 << (p - 3)) + 2) << rice;
            let width = 1u32 << (p - 3 + rice);
            if value < base + width {
                let suffix = value - base;
                let n_bits = p - 3 + rice;
                encode_bypass_prefix_suffix(sink, p, suffix, n_bits);
                return;
            }
            p += 1;
        }
    }
}

/// Emit `ones` one-bits, one zero terminator, then `suffix_bits` bypass suffix
/// bits. This is the common bypass shape for coeff_abs_level_remaining. It is
/// equivalent to repeated `encode_bin_ep` calls but lets both the real CABAC
/// writer and the fractional-bit estimator use their batched bypass path.
fn encode_bypass_prefix_suffix<S: CabacSyntax>(
    sink: &mut S,
    ones: u32,
    suffix: u32,
    suffix_bits: u32,
) {
    let total = ones + 1 + suffix_bits;
    if total <= u32::BITS {
        let prefix = if ones == 0 { 0 } else { (1u64 << ones) - 1 };
        let pattern = (prefix << (1 + suffix_bits)) | suffix as u64;
        sink.encode_bins_ep(pattern as u32, total);
        return;
    }

    // Extremely large escaped levels are not expected for still265's i16
    // coefficient range, but keep the helper total for correctness.
    for _ in 0..ones {
        sink.encode_bin_ep(1);
    }
    sink.encode_bin_ep(0);
    encode_bypass_bits(sink, suffix, suffix_bits as u8);
}

/// Emit `n_bits` of `value`, MSB first, as bypass bins (the order the decoder's
/// `decode_bypass_bits` reads).
fn encode_bypass_bits<S: CabacSyntax>(sink: &mut S, value: u32, n_bits: u8) {
    if n_bits > 0 {
        sink.encode_bins_ep(value, n_bits as u32);
    }
}

#[cfg(test)]
mod sig_ctx_tests {
    use super::*;

    /// The generated `SIG_CTX` table must equal the reference
    /// `calc_sig_coeff_flag_ctx` for every legal input the residual traversal
    /// can produce (all transform sizes, luma+both chroma planes, all scan
    /// orders, every real sub-block position, and every neighbour-coded state).
    /// This proves the `(sb_x,sb_y) -> is_dc_subblock` key reduction is exact.
    #[test]
    fn sig_ctx_table_matches_function() {
        for log2_size in 2u8..=5 {
            let sb_width = 1u8 << (log2_size - 2);
            for c_idx in 0u8..=2 {
                for scan_idx in 0u8..3 {
                    let scan_pos = get_scan_4x4(scan_order_from_idx(scan_idx));
                    for sb_x in 0..sb_width {
                        for sb_y in 0..sb_width {
                            let is_dc = sb_x == 0 && sb_y == 0;
                            for prev_csbf in 0u8..4 {
                                let row = sig_ctx_row(log2_size, c_idx, scan_idx, is_dc, prev_csbf);
                                for (n, &(px, py)) in scan_pos.iter().enumerate() {
                                    let x_c = sb_x * 4 + px;
                                    let y_c = sb_y * 4 + py;
                                    let expect = calc_sig_coeff_flag_ctx(
                                        x_c, y_c, log2_size, c_idx, scan_idx, prev_csbf,
                                    );
                                    assert_eq!(
                                        row[n] as usize, expect,
                                        "log2={log2_size} c={c_idx} scan={scan_idx} \
                                         sb=({sb_x},{sb_y}) prev={prev_csbf} n={n}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn dc_only_shortcut_matches_full_estimator() {
        for qp in [0, 22, 37] {
            let base_ctxs = Contexts::new(qp);
            for log2_size in 2u8..=5 {
                let size = 1usize << log2_size;
                for c_idx in 0u8..=2 {
                    for scan_order in [
                        ScanOrder::Diagonal,
                        ScanOrder::Horizontal,
                        ScanOrder::Vertical,
                    ] {
                        for sign_data_hiding in [false, true] {
                            for level in [-9i16, -3, -2, -1, 1, 2, 3, 9] {
                                let mut levels = vec![0i16; size * size];
                                levels[0] = level;

                                let mut full_scratch = ResidualPricingScratch::default();
                                let full = estimate_residual_bits_into(
                                    &base_ctxs,
                                    &levels,
                                    log2_size,
                                    c_idx,
                                    scan_order,
                                    sign_data_hiding,
                                    &mut full_scratch,
                                );

                                let direct = estimate_dc_only_residual_bits(
                                    &base_ctxs, &levels, log2_size, c_idx,
                                )
                                .expect("DC-only level should use shortcut");
                                assert_eq!(
                                    direct, full,
                                    "qp={qp} log2={log2_size} c={c_idx} scan={scan_order:?} \
                                     sdh={sign_data_hiding} level={level}"
                                );

                                let mut nnz_scratch = ResidualPricingScratch::default();
                                let via_public = estimate_residual_bits_into_nnz(
                                    &base_ctxs,
                                    &levels,
                                    1,
                                    log2_size,
                                    c_idx,
                                    scan_order,
                                    sign_data_hiding,
                                    &mut nnz_scratch,
                                );
                                assert_eq!(via_public, full);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sign_data_hiding_mask_path_matches_reference() {
        for log2_size in 2u8..=5 {
            let size = 1usize << log2_size;
            let n = size * size;
            let (scale, qbits) = crate::transform::quant_params(log2_size, 28, 8);
            for scan_order in [
                ScanOrder::Diagonal,
                ScanOrder::Horizontal,
                ScanOrder::Vertical,
            ] {
                for seed in 0usize..12 {
                    let mut levels = vec![0i16; n];
                    let mut coeffs = vec![0i16; n];
                    for i in 0..n {
                        let raw = ((i * 37 + seed * 53 + 11) % 2047) as i16 - 1023;
                        coeffs[i] = if raw == 0 { 1 } else { raw };

                        let hit = ((i * 13 + seed * 7) % 19) < 4 || (i + seed) % (size + 3) == 0;
                        if hit {
                            let mag = (((i * 5 + seed * 3) % 5) + 1) as i16;
                            levels[i] = if (i + seed) & 1 == 0 { mag } else { -mag };
                        }
                    }

                    if seed % 3 == 0 {
                        levels[0] = 1;
                        levels[(n - 1).min(15)] = -2;
                    }
                    if seed % 4 == 0 && n > 32 {
                        levels[n - 1] = 3;
                    }

                    let nnz = levels.iter().filter(|&&v| v != 0).count() as u32;
                    let mut got = levels.clone();
                    let mut want = levels;

                    let got_nnz = apply_sign_data_hiding(
                        &mut got, &coeffs, log2_size, scan_order, scale, qbits, nnz,
                    );
                    let want_nnz = apply_sign_data_hiding_reference(
                        &mut want, &coeffs, log2_size, scan_order, scale, qbits, nnz,
                    );

                    assert_eq!(
                        got_nnz, want_nnz,
                        "nnz log2={log2_size} scan={scan_order:?} seed={seed}"
                    );
                    assert_eq!(
                        got, want,
                        "levels log2={log2_size} scan={scan_order:?} seed={seed}"
                    );
                }
            }
        }
    }

    fn apply_sign_data_hiding_reference(
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
                continue;
            }

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
                let lower_sig =
                    (0..n).any(|m| blk_pos(sbx, sby, m).is_some_and(|b| levels[b] != 0));
                let du = delta_u(levels, blk);
                let (cost, change) = if cur_sig {
                    if du > 0 {
                        (-du, 1)
                    } else if !lower_sig && levels[blk].unsigned_abs() == 1 {
                        (i64::MAX, 0)
                    } else {
                        (du, -1)
                    }
                } else if !lower_sig {
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
            if min_blk == usize::MAX || final_change == 0 {
                min_blk = first_blk;
                final_change = 1;
            }

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
}
