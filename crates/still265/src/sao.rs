//! Encoder-side Sample Adaptive Offset (SAO) syntax writer and parameter search.
//!
//! Field order/conditions mirror `decode_sao`/`decode_sao_type_idx`/
//! `decode_cabac_tu_bypass` in `bpg-hevc-decode::hevc::ctu`, specialized to:
//! - `slice_addr_rs == 0` (single slice), so `sao_merge_left_flag`/
//!   `sao_merge_up_flag` are written when `x_ctb > 0` / `y_ctb > 0`
//!   respectively. The writer merges when a CTB's final parameters are
//!   identical to its left or upper neighbour.
//! - Monochrome skips chroma components when `slice_sao_chroma == false`.
//! - `sao_type_idx` 0 (off), 1 (band offset), and 2 (edge offset) are supported.
//! - `sao_type_idx` for Cb is decoded once and reused for Cr (per H.265
//!   `decode_sao`), so `SaoInfo::sao_type_idx[1] == SaoInfo::sao_type_idx[2]`
//!   and `SaoInfo::sao_eo_class[1] == SaoInfo::sao_eo_class[2]` are
//!   invariants maintained by [`crate::encoder`]'s SAO decision.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::hevc::sao::SaoMap;

use crate::cabac::CabacEncoder;
use crate::contexts::{ctx, Contexts};

/// Edge-offset neighbour direction pairs `(dx0, dy0, dx1, dy1)` for each
/// `eo_class`, duplicated from the private `EO_OFFSETS` table in
/// `bpg-hevc-decode::hevc::sao` (eo_class 0=horizontal, 1=vertical,
/// 2=135° diagonal, 3=45° diagonal).
pub const EO_OFFSETS: [(i32, i32, i32, i32); 4] =
    [(-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (1, -1, -1, 1)];

/// `sao_offset_abs`'s truncated-unary bound, H.265 7.4.9.3:
/// `(1 << (Min(bitDepth, 10) - 5)) - 1`.
pub fn sao_c_max(bit_depth: u8) -> u32 {
    (1u32 << (bit_depth.min(10) - 5)) - 1
}

/// `1 << (bitDepth - Min(bitDepth, 10))`: the scale factor between a coded
/// `sao_offset_abs` and the actual sample-domain offset.
pub fn sao_offset_scale(bit_depth: u8) -> i32 {
    1i32 << bit_depth.saturating_sub(bit_depth.min(10))
}

/// Per-`edgeIdx` (0..=4) accumulators for one CTB/component/`eo_class`:
/// `sum[edgeIdx]` is `sum(source - reconstruction)` and `count[edgeIdx]` is
/// the number of pixels classified into that category. `edgeIdx == 2`
/// ("no edge") is never populated since its offset is always 0.
#[derive(Default, Clone, Copy)]
pub struct EoStats {
    pub sum: [i64; 5],
    pub count: [u32; 5],
}

/// Per-band accumulators for Band Offset search.
#[derive(Default, Clone, Copy)]
pub struct BoStats {
    pub sum: [i64; 32],
    pub count: [u32; 32],
}

fn offset_reduction(sum: i64, count: u32, applied_offset: i32) -> i64 {
    let off = applied_offset as i64;
    2 * off * sum - count as i64 * off * off
}

/// Given the per-`edgeIdx` statistics for one `eo_class`, choose the offset
/// for each of the four coded categories (`edgeIdx` 0, 1, 3, 4) that
/// minimizes SSE against the source, and return the total SSE reduction
/// across all four categories.
///
/// `sao_offset_val` layout matches `bpg_hevc_decode::hevc::sao::SaoInfo`:
/// `[0]`/`[1]` (categories 1/2, `edgeIdx` 0/1) are applied as positive
/// offsets; `[2]`/`[3]` (categories 3/4, `edgeIdx` 3/4) are stored as
/// non-negative magnitudes but applied as *negative* offsets by
/// `apply_sao_edge`. A category whose best offset doesn't reduce SSE is left
/// at 0 (`sao_offset_abs == 0`, a no-op for that category).
pub fn eo_offsets_and_reduction(stats: &EoStats, bit_depth: u8) -> ([i8; 4], i64) {
    let c_max = sao_c_max(bit_depth) as i32;
    let scale = sao_offset_scale(bit_depth);
    let max_offset = c_max * scale;

    let mut offsets = [0i8; 4];
    let mut reduction = 0i64;

    // (output index into sao_offset_val, edgeIdx, sign applied to the offset)
    const CATS: [(usize, usize, i64); 4] = [(0, 0, 1), (1, 1, 1), (2, 3, -1), (3, 4, -1)];

    for (out_idx, edge_idx, sign) in CATS {
        let count = stats.count[edge_idx];
        if count == 0 {
            continue;
        }
        let mean = stats.sum[edge_idx] as f64 / count as f64;
        let raw = sign as f64 * mean;
        if raw <= 0.0 {
            continue;
        }
        let mut off = raw.round() as i32;
        off = off.clamp(0, max_offset);
        if scale > 1 {
            off = ((off + scale / 2) / scale) * scale;
            off = off.clamp(0, max_offset);
        }
        if off == 0 {
            continue;
        }

        let applied = sign * off as i64;
        let sum = stats.sum[edge_idx];
        let r = offset_reduction(sum, count, applied as i32);
        if r > 0 {
            reduction += r;
            offsets[out_idx] = off as i8;
        }
    }

    (offsets, reduction)
}

/// Choose the best four consecutive SAO bands and signed offsets.
pub fn bo_offsets_and_reduction(stats: &BoStats, bit_depth: u8) -> (u8, [i8; 4], i64) {
    let c_max = sao_c_max(bit_depth) as i32;
    let scale = sao_offset_scale(bit_depth);
    let max_offset = c_max * scale;

    let mut best_pos = 0u8;
    let mut best_offsets = [0i8; 4];
    let mut best_reduction = 0i64;

    for start in 0..32usize {
        let mut offsets = [0i8; 4];
        let mut reduction = 0i64;
        for (i, off_slot) in offsets.iter_mut().enumerate() {
            let band = (start + i) & 31;
            let count = stats.count[band];
            if count == 0 {
                continue;
            }
            let mean = stats.sum[band] as f64 / count as f64;
            let mut off = mean.round() as i32;
            off = off.clamp(-max_offset, max_offset);
            if scale > 1 {
                off = if off >= 0 {
                    ((off + scale / 2) / scale) * scale
                } else {
                    -(((-off + scale / 2) / scale) * scale)
                };
                off = off.clamp(-max_offset, max_offset);
            }
            if off == 0 {
                continue;
            }
            let r = offset_reduction(stats.sum[band], count, off);
            if r > 0 {
                *off_slot = off as i8;
                reduction += r;
            }
        }
        if reduction > best_reduction {
            best_pos = start as u8;
            best_offsets = offsets;
            best_reduction = reduction;
        }
    }

    (best_pos, best_offsets, best_reduction)
}

/// `sao_offset_abs`: truncated unary, bypass-coded (H.265 9.3.3.x via
/// `decode_cabac_tu_bypass`'s encode-side mirror).
fn encode_tu_bypass(enc: &mut CabacEncoder, w: &mut BitWriter, value: u32, c_max: u32) {
    let value = value.min(c_max);
    for _ in 0..value {
        enc.encode_bin_ep(w, 1);
    }
    if value < c_max {
        enc.encode_bin_ep(w, 0);
    }
}

/// `sao_type_idx_luma`/`sao_type_idx_chroma`: one context-coded bin
/// (0 => off), then for non-zero one bypass bin (0 => band offset (1),
/// 1 => edge offset (2)), mirroring `decode_sao_type_idx`.
fn encode_sao_type_idx(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    type_idx: u8,
) {
    if type_idx == 0 {
        enc.encode_bin(w, 0, ctxs.get(ctx::SAO_TYPE_IDX));
    } else {
        enc.encode_bin(w, 1, ctxs.get(ctx::SAO_TYPE_IDX));
        enc.encode_bin_ep(w, (type_idx == 2) as u8);
    }
}

/// Write one CTU's `sao()` syntax (H.265 7.3.8.3), mirroring `decode_sao`.
///
/// `sao_map` must already hold the SAO decision for `(x_ctb, y_ctb)`
/// (produced by [`crate::encoder`]'s SAO search), with the invariants
/// documented in the module docs (`sao_type_idx[1] == sao_type_idx[2]`,
/// `sao_eo_class[1] == sao_eo_class[2]`, types limited to 0/1/2).
#[allow(clippy::too_many_arguments)]
pub fn write_sao(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    sao_map: &SaoMap,
    x_ctb: u32,
    y_ctb: u32,
    slice_sao_luma: bool,
    slice_sao_chroma: bool,
    bit_depth: u8,
) {
    let info = sao_map.get(x_ctb, y_ctb);

    // sao_merge_left_flag / sao_merge_up_flag. Merge is legal when the final
    // parameters are identical to the neighbour; no component syntax follows.
    if x_ctb > 0 {
        let merge_left = sao_map.get(x_ctb - 1, y_ctb) == info;
        enc.encode_bin(w, merge_left as u8, ctxs.get(ctx::SAO_MERGE_FLAG));
        if merge_left {
            return;
        }
    }
    if y_ctb > 0 {
        let merge_up = sao_map.get(x_ctb, y_ctb - 1) == info;
        enc.encode_bin(w, merge_up as u8, ctxs.get(ctx::SAO_MERGE_FLAG));
        if merge_up {
            return;
        }
    }

    let c_max = sao_c_max(bit_depth);
    let offset_scale = sao_offset_scale(bit_depth) as u32;

    for c_idx in 0..3usize {
        let should_write = (slice_sao_luma && c_idx == 0) || (slice_sao_chroma && c_idx > 0);
        if !should_write {
            continue;
        }

        // sao_type_idx_chroma is decoded once (c_idx == 1) and reused for
        // c_idx == 2; SaoInfo::sao_type_idx[1] == sao_type_idx[2] by
        // construction, so only c_idx <= 1 codes the type bins.
        if c_idx <= 1 {
            encode_sao_type_idx(enc, w, ctxs, info.sao_type_idx[c_idx]);
        }

        let sao_type_idx = info.sao_type_idx[c_idx];
        if sao_type_idx == 0 {
            continue;
        }
        for &v in &info.sao_offset_val[c_idx] {
            let abs = v.unsigned_abs() as u32 / offset_scale;
            encode_tu_bypass(enc, w, abs, c_max);
        }

        if sao_type_idx == 1 {
            for &v in &info.sao_offset_val[c_idx] {
                if v != 0 {
                    enc.encode_bin_ep(w, (v < 0) as u8);
                }
            }
            enc.encode_bins_ep(w, info.sao_band_position[c_idx] as u32, 5);
        } else {
            // sao_eo_class_chroma is decoded once (c_idx == 1) and reused for
            // c_idx == 2; SaoInfo::sao_eo_class[1] == sao_eo_class[2] by
            // construction, so only c_idx <= 1 codes the class bits.
            if c_idx <= 1 {
                enc.encode_bins_ep(w, info.sao_eo_class[c_idx] as u32, 2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bo_search_picks_best_consecutive_band_run() {
        let mut stats = BoStats::default();
        stats.sum[9] = 30;
        stats.count[9] = 10;
        stats.sum[10] = -40;
        stats.count[10] = 10;

        let (band_pos, offsets, reduction) = bo_offsets_and_reduction(&stats, 8);

        assert_eq!(band_pos, 7);
        assert_eq!(offsets, [0, 0, 3, -4]);
        assert!(reduction > 0);
    }
}
