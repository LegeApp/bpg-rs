//! Adaptive quantization (AQ) writer state, per-CU QP derivation, and QP delta
//! CABAC encoding.
//!
//! Ported from `bpgenc.c`'s `x265_glue.c` AQ integration (`aq_qg_target`,
//! `aq_cu_begin`, `aq_take_cu_qp_delta`, `encode_cu_qp_delta`). The prediction
//! mirror matches `bpg-hevc-decode::hevc::ctu::decode_quantization_parameters`.

use bpg_bitstream::BitWriter;
use crate::cabac::CabacEncoder;
use crate::contexts::{ctx, Contexts};

use super::types::{chroma_qp_from_luma, QG_LOG2};

/// Adaptive-quantization writer state — the encoder-side mirror of the decoder's
/// `decode_quantization_parameters` QP-prediction machine
/// (`bpg-hevc-decode::hevc::ctu`). Inert (`active == false`) on the byte-exact
/// reference tiers; otherwise every field tracks the decoder's
/// per-quantization-group state in decoding order so that the `cu_qp_delta` the
/// encoder emits resolves, on the decoder, back to the exact QP the encoder
/// quantized with.
///
/// The "decouple property": the encoder quantizes each TU with its target QP
/// **and** emits `cu_qp_delta = target − pred`. The decoder lands on `pred +
/// delta`, which equals `target` iff this prediction (`pred`) matches the
/// decoder's — so the prediction mirror is the only correctness-critical part,
/// verified by reconstruction equality. The QP a quantization group is *constant*
/// across all its coded CUs because the prediction uses QG-boundary neighbours
/// (not CU-boundary), exactly as the decoder does.
#[derive(Clone)]
pub(super) struct AqState {
    /// Adaptive QP active for this encode (every non-reference tier).
    pub(super) active: bool,
    /// `6 * (bit_depth − 8)` — the QP bit-depth offset (H.265 8.6.1). All
    /// prediction arithmetic is in the *without-offset* `QpY` domain `[0,51]`
    /// (matching the decoder's `current_qpy`); quantization QPs add it back.
    pub(super) qp_bd_offset: i32,
    /// Slice luma QP (`QpY`, without offset) — the first-QG-in-slice prediction.
    pub(super) slice_qp_y: i32,
    /// `current_qpy` — last resolved `QpY` (without offset).
    pub(super) current_qpy: i32,
    /// `lastQpYinPrevQG` — `current_qpy` as of the previous quantization group.
    pub(super) last_qpy_prev_qg: i32,
    /// Top-left luma position of the current quantization group (or −1 / unset).
    pub(super) qg_x: i32,
    pub(super) qg_y: i32,
    /// `IsCuQpDeltaCoded` — whether this QG has already carried its one delta.
    pub(super) coded: bool,
    /// `CuQpDeltaVal` — the delta resolved for the current QG (0 until coded).
    pub(super) cu_qp_delta: i32,
    /// `qPY_PRED` (the averaged neighbour prediction, without offset) for the CU
    /// currently being written — set in [`Encoder::aq_cu_begin`], consumed when
    /// the CU's first coded TU emits the delta.
    pub(super) pred: i32,
    /// Target `QpY` (without offset) for the CU currently being written.
    pub(super) cu_target: i32,
    /// Memo for [`Encoder::aq_qg_target`]: the QG origin (`& !31`) the cached
    /// `target` was computed for, or `(-1, -1)` when unset. The QG target is a
    /// pure function of QG position (slice QP + importance), so it is computed
    /// once per QG and reused across all of that QG's CUs (every recursion
    /// level) and the writer pass, instead of re-folding the importance map
    /// ~5–10× per QG.
    pub(super) target_qg: (i32, i32),
    pub(super) target: i32,
}

impl AqState {
    /// The inert state used off the adaptive-QP path (the reference tiers, and
    /// parallel workers, which only run on the `Placebo` tier).
    pub(super) fn inert() -> Self {
        AqState {
            active: false,
            qp_bd_offset: 0,
            slice_qp_y: 0,
            current_qpy: 0,
            last_qpy_prev_qg: 0,
            qg_x: -1,
            qg_y: -1,
            coded: false,
            cu_qp_delta: 0,
            pred: 0,
            cu_target: 0,
            target_qg: (-1, -1),
            target: 0,
        }
    }

    /// Reset the per-picture QG-prediction state to the slice start (first QG
    /// predicts from the slice QP). Called at each slice-data pass so the SAO
    /// two-pass re-entry doesn't inherit the previous pass's trailing QG state.
    /// The `target`/`target_qg` memo needs no reset — it is a pure function of QG
    /// position, so a stale entry self-corrects on the next position mismatch.
    pub(super) fn reset_for_slice(&mut self) {
        self.current_qpy = self.slice_qp_y;
        self.last_qpy_prev_qg = self.slice_qp_y;
        self.qg_x = -1;
        self.qg_y = -1;
        self.coded = false;
        self.cu_qp_delta = 0;
    }
}

impl<'a> super::Encoder<'a> {
    // --- Adaptive quantization (Phase B) -----------------------------------
    //
    // These mirror the decoder's `decode_quantization_parameters`
    // (`bpg-hevc-decode::hevc::ctu`) so the per-CU QP the encoder quantizes with
    // is exactly the QP the decoder reconstructs. All arithmetic is in the
    // *without-offset* `QpY` domain (matching the decoder's `current_qpy`);
    // quantization QPs add `qp_bd_offset` back.

    /// Target `QpY` (without offset, clamped `[0,51]`) for the quantization group
    /// containing `(x0, y0)`. Snapped to the QG grid (`& !(QG-1)`) so every CU in
    /// a QG resolves to the identical QP — the invariant the decoder enforces by
    /// predicting from QG-boundary neighbours. Driven by the preanalysis
    /// `importance` of the QG cell (low importance ⇒ higher QP ⇒ fewer bits).
    ///
    /// Memoized on [`AqState`]: the target is a pure function of QG position, so
    /// it is folded from the importance map once per QG and reused across all of
    /// that QG's CUs (every recursion level) and the writer pass.
    pub(super) fn aq_qg_target(&mut self, x0: u32, y0: u32) -> i32 {
        let mask = (1u32 << QG_LOG2) - 1;
        let (qx, qy) = ((x0 & !mask) as i32, (y0 & !mask) as i32);
        if (qx, qy) != self.aq.target_qg {
            let imp = self.analysis.importance_at(qx as u32, qy as u32, QG_LOG2);
            self.aq.target =
                (self.aq.slice_qp_y + crate::preanalysis::aq_qp_offset(imp)).clamp(0, 51);
            self.aq.target_qg = (qx, qy);
        }
        self.aq.target
    }

    /// Set the per-CU quantization QPs (`cur_qp_y`/`cur_qp_c`, *with* offset) from
    /// the QG target, mirroring the decoder's luma + chroma QP derivation
    /// (`decode_quantization_parameters`, chroma via [`chroma_qp_from_luma`]).
    pub(super) fn aq_set_cu_qp(&mut self, x0: u32, y0: u32) {
        let target = self.aq_qg_target(x0, y0);
        let off = self.aq.qp_bd_offset;
        self.cur_qp_y = target + off;
        let qpi_c = target.clamp(-off, 57);
        self.cur_qp_c = chroma_qp_from_luma(qpi_c) + off;
    }

    /// Read a neighbour CU's resolved `QpY` (without offset) from the per-4x4 QP
    /// map — the encoder analogue of the decoder's `get_qpy_at`.
    pub(super) fn aq_get_qpy_at(&self, x: u32, y: u32) -> i32 {
        let stride = self.frame.deblock_stride;
        let idx = ((y / 4) * stride + (x / 4)) as usize;
        self.frame
            .qp_map
            .get(idx)
            .map(|&q| q as i32)
            .unwrap_or(self.aq.slice_qp_y)
    }

    /// H.265 8.6.1 final-QP wrap (`((pred + delta + 52 + 2*off) % (52+off)) - off`).
    /// A no-op for in-range 8-bit QPs; matters only for the 10/12-bit offset.
    pub(super) fn aq_resolve_qpy(&self, pred: i32, delta: i32) -> i32 {
        let off = self.aq.qp_bd_offset;
        ((pred + delta + 52 + 2 * off) % (52 + off)) - off
    }

    /// Reset the per-quantization-group delta state, mirroring the decoder's
    /// `coding_quadtree` reset at every node `>=` the QG size (and at CTU start).
    /// Called from the writer at each `write_cu` node with `log2_cb_size >= 5`.
    pub(super) fn aq_qg_reset(&mut self) {
        self.aq.coded = false;
        self.aq.cu_qp_delta = 0;
    }

    /// At a coding-unit (leaf) start in the writer: compute `qPY_PRED` from the
    /// QG-boundary neighbours and the current (possibly already-coded) QG delta,
    /// exactly as `decode_quantization_parameters`, and record the CU's target.
    pub(super) fn aq_cu_begin(&mut self, cu_x: u32, cu_y: u32) {
        let qg_mask = (1u32 << QG_LOG2) - 1;
        let x_qg = (cu_x & !qg_mask) as i32;
        let y_qg = (cu_y & !qg_mask) as i32;
        if x_qg != self.aq.qg_x || y_qg != self.aq.qg_y {
            self.aq.last_qpy_prev_qg = self.aq.current_qpy;
            self.aq.qg_x = x_qg;
            self.aq.qg_y = y_qg;
        }

        // qPY_PRED initial: slice QP at the first QG of the slice (entropy-sync /
        // WPP is disabled here, so the first-CTB-row case never applies), else the
        // last QP of the previous QG.
        let first_qg_in_slice = x_qg == 0 && y_qg == 0;
        let pred0 = if first_qg_in_slice {
            self.aq.slice_qp_y
        } else {
            self.aq.last_qpy_prev_qg
        };

        let ctb_size = 1u32 << super::types::CTB_LOG2;
        let ctb_x = cu_x / ctb_size;
        let ctb_y = cu_y / ctb_size;

        // qpYA (left): available only when the QG-left neighbour is in the same CTB.
        let qp_y_a = if x_qg > 0 {
            let left_x = (x_qg as u32) - 1;
            let left_ctb_x = left_x / ctb_size;
            let our_ctb_x = cu_x / ctb_size;
            if (our_ctb_x == left_ctb_x || (x_qg as u32) < ctb_size) && left_ctb_x == ctb_x {
                self.aq_get_qpy_at(left_x, y_qg as u32)
            } else {
                pred0
            }
        } else {
            pred0
        };

        // qpYB (above): available only when the QG-above neighbour is in the same CTB.
        let qp_y_b = if y_qg > 0 {
            let above_y = (y_qg as u32) - 1;
            if above_y / ctb_size == ctb_y {
                self.aq_get_qpy_at(x_qg as u32, above_y)
            } else {
                pred0
            }
        } else {
            pred0
        };

        let pred = (qp_y_a + qp_y_b + 1) >> 1;
        self.aq.pred = pred;
        self.aq.current_qpy = self.aq_resolve_qpy(pred, self.aq.cu_qp_delta);
        self.aq.cu_target = self.aq_qg_target(cu_x, cu_y);
    }

    /// At the CU's first coded TU: resolve and record the QG's `cu_qp_delta`
    /// (`target − qPY_PRED`), marking the QG coded, and return the delta for the
    /// caller to emit into CABAC. Mirrors the decoder's
    /// `decode_transform_unit` → re-run of `decode_quantization_parameters`.
    pub(super) fn aq_take_cu_qp_delta(&mut self) -> i32 {
        let delta = self.aq.cu_target - self.aq.pred;
        self.aq.coded = true;
        self.aq.cu_qp_delta = delta;
        self.aq.current_qpy = self.aq_resolve_qpy(self.aq.pred, delta);
        delta
    }
}

/// Emit `cu_qp_delta_abs` + sign for adaptive QP (H.265 7.3.8.11), the exact
/// inverse of the decoder's `decode_cu_qp_delta_abs` + sign bypass: a
/// truncated-unary prefix (<=5 bins; ctx `CU_QP_DELTA_ABS` for bin 0, `+1` for
/// the rest), an EGk(0) bypass suffix when the prefix saturates at 5, then one
/// bypass sign bin when the magnitude is non-zero.
pub(super) fn encode_cu_qp_delta(enc: &mut CabacEncoder, w: &mut BitWriter, ctxs: &mut Contexts, delta: i32) {
    let abs = delta.unsigned_abs();
    let prefix = abs.min(5);
    enc.encode_bin(w, (prefix >= 1) as u8, ctxs.get(ctx::CU_QP_DELTA_ABS));
    if prefix >= 1 {
        for _ in 1..prefix {
            enc.encode_bin(w, 1, ctxs.get(ctx::CU_QP_DELTA_ABS + 1));
        }
        if prefix < 5 {
            enc.encode_bin(w, 0, ctxs.get(ctx::CU_QP_DELTA_ABS + 1));
        }
    }
    if abs >= 5 {
        encode_egk0_bypass(enc, w, abs - 5);
    }
    if abs >= 1 {
        enc.encode_bin_ep(w, (delta < 0) as u8);
    }
}

/// Emit `value` as an order-0 Exp-Golomb code in bypass mode — the inverse of the
/// decoder's `decode_egk_bypass(0)`: `m` unary one-bits (`m = floor(log2(value+1))`)
/// terminated by a zero, then the `m`-bit binary remainder, all MSB-first.
pub(super) fn encode_egk0_bypass(enc: &mut CabacEncoder, w: &mut BitWriter, value: u32) {
    let mut m = 0u32;
    while (1u32 << (m + 1)) - 1 <= value {
        m += 1;
    }
    for _ in 0..m {
        enc.encode_bin_ep(w, 1);
    }
    enc.encode_bin_ep(w, 0);
    if m > 0 {
        let suffix = value - ((1u32 << m) - 1);
        enc.encode_bins_ep(w, suffix, m);
    }
}
