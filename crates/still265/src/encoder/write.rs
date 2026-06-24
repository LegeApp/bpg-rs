//! CABAC bitstream emission for the H.265 still-picture intra slice-data
//! syntax: coding quadtree, intra modes, transform tree, residual coefficients,
//! SAO, and deblock-edge marking.
//!
//! Ported from `bpgenc.c`'s `put_*` / `*_coding_quadtree` / `*_transform_tree`
//! emit functions, matching the decode order in `bpg-hevc-decode::hevc::ctu`.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::sao::SaoMap;

use crate::cabac::CabacEncoder;
use crate::contexts::{Contexts, ctx};
use crate::residual::{encode_residual, get_scan_order};
use crate::sao;

use super::Encoder;
use super::aq::encode_cu_qp_delta;
use super::syntax::{CuLeaf, CuNode, NxnInfo, Tt};
use super::types::{
    CHROMA_DM_IDX, CTB_LOG2, MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, MIN_TB_LOG2, chroma_pred_mode,
    decode_second_cbf, has_chroma_tb,
};

/// Emit `prev_intra_luma_pred_flag` + `mpm_idx`/`rem_intra_luma_pred_mode` for
/// a chosen luma `mode`, mirroring the decoder's MPM derivation.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_intra_luma_mode(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    mpm: [bpg_hevc_decode::hevc::slice::IntraPredMode; 3],
    mode: u8,
) {
    let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
    let in_mpm = mpm_u8.iter().position(|&m| m == mode);

    if let Some(idx) = in_mpm {
        enc.encode_bin(w, 1, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        match idx {
            0 => enc.encode_bin_ep(w, 0),
            1 => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 0);
            }
            _ => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 1);
            }
        }
    } else {
        enc.encode_bin(w, 0, ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG));
        let mut sorted = mpm_u8;
        sorted.sort_unstable();
        let mut rem = mode as i32;
        for &v in sorted.iter().rev() {
            if rem >= v as i32 {
                rem -= 1;
            }
        }
        for b in (0..5).rev() {
            enc.encode_bin_ep(w, ((rem >> b) & 1) as u8);
        }
    }
}

/// Emit an 8x8 `PartNxN` CU: `part_mode=0`, four `prev_intra_luma_pred_flag`
/// (all context bins first), then four `mpm_idx`/`rem` (bypass), one chroma
/// mode, and the forced-split transform tree. Mirrors the decoder's
/// `decode_coding_unit` `PartNxN` branch exactly.
#[allow(clippy::too_many_arguments)]
fn write_cu_nxn(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    leaf: &CuLeaf,
    nxn: &NxnInfo,
    x0: u32,
    y0: u32,
) {
    enc.encode_bin(w, 0, ctxs.get(ctx::PART_MODE));

    // First pass: four prev_intra_luma_pred_flag (context-coded).
    let mut in_mpm = [None; 4];
    for pu in 0..4 {
        let mode = nxn.luma_modes[pu];
        let mpm_u8 = [
            nxn.mpms[pu][0].as_u8(),
            nxn.mpms[pu][1].as_u8(),
            nxn.mpms[pu][2].as_u8(),
        ];
        let idx = mpm_u8.iter().position(|&m| m == mode);
        enc.encode_bin(
            w,
            idx.is_some() as u8,
            ctxs.get(ctx::PREV_INTRA_LUMA_PRED_FLAG),
        );
        in_mpm[pu] = idx;
    }
    // Second pass: mpm_idx (1-2 bypass) or rem_intra_luma_pred_mode (5 bypass).
    for pu in 0..4 {
        let mode = nxn.luma_modes[pu];
        match in_mpm[pu] {
            Some(0) => enc.encode_bin_ep(w, 0),
            Some(1) => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 0);
            }
            Some(_) => {
                enc.encode_bin_ep(w, 1);
                enc.encode_bin_ep(w, 1);
            }
            None => {
                let mut sorted = [
                    nxn.mpms[pu][0].as_u8(),
                    nxn.mpms[pu][1].as_u8(),
                    nxn.mpms[pu][2].as_u8(),
                ];
                sorted.sort_unstable();
                let mut rem = mode as i32;
                for &v in sorted.iter().rev() {
                    if rem >= v as i32 {
                        rem -= 1;
                    }
                }
                for b in (0..5).rev() {
                    enc.encode_bin_ep(w, ((rem >> b) & 1) as u8);
                }
            }
        }
    }

    if state.cat != 0 {
        write_intra_chroma_mode(enc, w, ctxs, leaf.chroma_mode_idx);
    }
    let cat = state.cat;
    if state.aq.active {
        state.aq_cu_begin(x0, y0);
    }
    write_tt(state, enc, w, ctxs, &leaf.tt, x0, y0, cat, true, true, true);
}

/// Emit `intra_chroma_pred_mode` for H.265 Table 8-2 mode indexes:
/// 0=Planar, 1=Vertical, 2=Horizontal, 3=DC, 4=DM.
pub(super) fn write_intra_chroma_mode(
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    mode_idx: u8,
) {
    if mode_idx == CHROMA_DM_IDX {
        enc.encode_bin(w, 0, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
        return;
    }

    enc.encode_bin(w, 1, ctxs.get(ctx::INTRA_CHROMA_PRED_MODE));
    enc.encode_bin_ep(w, (mode_idx >> 1) & 1);
    enc.encode_bin_ep(w, mode_idx & 1);
}

/// Write the transform tree's CABAC syntax (mirrors `decode_transform_tree_inner`).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_tt(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    node: &Tt,
    x0: u32,
    y0: u32,
    cat: u8,
    intra_split_flag: bool,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
) {
    let (log2_size, trafo_depth) = match node {
        Tt::Split {
            log2_size,
            trafo_depth,
            ..
        } => (*log2_size, *trafo_depth),
        Tt::Leaf(l) => (l.log2_size, l.trafo_depth),
    };
    let is_split = matches!(node, Tt::Split { .. });

    let max_trafo_depth = MAX_INTRA_TT_DEPTH + intra_split_flag as u8;
    let split_coded = log2_size <= MAX_TB_LOG2
        && log2_size > MIN_TB_LOG2
        && trafo_depth < max_trafo_depth
        && !(intra_split_flag && trafo_depth == 0);
    if split_coded {
        let ci = ctx::SPLIT_TRANSFORM_FLAG + (5 - log2_size as usize).min(2);
        enc.encode_bin(w, is_split as u8, ctxs.get(ci));
    }

    let sdh = state.sign_data_hiding;
    let decode_chroma_cbf = cat != 0 && (log2_size > 2 || cat == 3);
    let (cbf_cb, cbf_cr) = (node.cbf_cb(), node.cbf_cr());
    let second_cbf = decode_second_cbf(cat, log2_size, is_split);
    let (cbf_cb1, cbf_cr1) = (node.cbf_cb1(), node.cbf_cr1());
    if decode_chroma_cbf {
        let ci = ctx::CBF_CBCR + trafo_depth as usize;
        if trafo_depth == 0 || parent_cbf_cb {
            enc.encode_bin(w, cbf_cb as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cb) {
            enc.encode_bin(w, cbf_cb1 as u8, ctxs.get(ci));
        }
        if trafo_depth == 0 || parent_cbf_cr {
            enc.encode_bin(w, cbf_cr as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cr) {
            enc.encode_bin(w, cbf_cr1 as u8, ctxs.get(ci));
        }
    }

    match node {
        Tt::Split {
            kids,
            parent_chroma,
            ..
        } => {
            let half = 1u32 << (log2_size - 1);
            for (i, kid) in kids.iter().enumerate() {
                let kx = x0 + (i as u32 & 1) * half;
                let ky = y0 + (i as u32 >> 1) * half;
                write_tt(
                    state,
                    enc,
                    w,
                    ctxs,
                    kid,
                    kx,
                    ky,
                    cat,
                    intra_split_flag,
                    cbf_cb,
                    cbf_cr,
                );
            }
            if let Some(c) = parent_chroma {
                if c.cb.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 1, cat);
                    encode_residual(enc, w, ctxs, &c.cb.levels, c.log2_size, 1, scan, sdh);
                }
                if c.cr.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &c.cr.levels, c.log2_size, 2, scan, sdh);
                }
            }
        }
        Tt::Leaf(l) => {
            let eff_cbf_cb = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cb) {
                cbf_cb
            } else {
                parent_cbf_cb
            };
            let eff_cbf_cr = if decode_chroma_cbf && (trafo_depth == 0 || parent_cbf_cr) {
                cbf_cr
            } else {
                parent_cbf_cr
            };
            let eff_cbf_cb1 = second_cbf && (trafo_depth == 0 || parent_cbf_cb) && cbf_cb1;
            let eff_cbf_cr1 = second_cbf && (trafo_depth == 0 || parent_cbf_cr) && cbf_cr1;

            let ctx_off = if trafo_depth == 0 { 1 } else { 0 };
            enc.encode_bin(w, l.luma.cbf as u8, ctxs.get(ctx::CBF_LUMA + ctx_off));

            if state.aq.active
                && !state.aq.coded
                && (l.luma.cbf || eff_cbf_cb || eff_cbf_cr || eff_cbf_cb1 || eff_cbf_cr1)
            {
                let delta = state.aq_take_cu_qp_delta();
                encode_cu_qp_delta(enc, w, ctxs, delta);
            }
            if state.aq.active {
                let tu = 1u32 << l.log2_size;
                state
                    .frame
                    .store_block_qp(x0, y0, tu, state.aq.current_qpy as i8);
            }

            if l.luma.cbf {
                let scan = get_scan_order(l.log2_size, l.luma_mode, 0, cat);
                encode_residual(enc, w, ctxs, &l.luma.levels, l.log2_size, 0, scan, sdh);
            }
            if has_chroma_tb(cat, l.log2_size) {
                let clog2 = l.chroma_log2;
                let cmode = chroma_pred_mode(cat, l.chroma_mode);
                if eff_cbf_cb {
                    let scan = get_scan_order(clog2, cmode, 1, cat);
                    encode_residual(enc, w, ctxs, &l.cb.levels, clog2, 1, scan, sdh);
                }
                if eff_cbf_cb1 {
                    let scan = get_scan_order(clog2, cmode, 1, cat);
                    encode_residual(enc, w, ctxs, &l.cb1.levels, clog2, 1, scan, sdh);
                }
                if eff_cbf_cr {
                    let scan = get_scan_order(clog2, cmode, 2, cat);
                    encode_residual(enc, w, ctxs, &l.cr.levels, clog2, 2, scan, sdh);
                }
                if eff_cbf_cr1 {
                    let scan = get_scan_order(clog2, cmode, 2, cat);
                    encode_residual(enc, w, ctxs, &l.cr1.levels, clog2, 2, scan, sdh);
                }
            }
        }
    }
}

/// Write a coding quadtree node's CABAC syntax: `split_cu_flag` (when
/// coded), then either the four children or the leaf CU's `coding_unit`
/// (intra mode signalling + transform tree).
#[allow(clippy::too_many_arguments)]
pub(super) fn write_cu(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    node: &CuNode,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
) {
    let cb_size = 1u32 << log2_cb_size;
    let fully_inside = x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
    let can_split = log2_cb_size > 3;
    let split_coded = fully_inside && can_split;

    if state.aq.active && log2_cb_size >= 5 {
        state.aq_qg_reset();
    }

    if split_coded {
        let ci = ctx::SPLIT_CU_FLAG + state.split_ctx_inc(x0, y0, ct_depth);
        enc.encode_bin(w, matches!(node, CuNode::Split { .. }) as u8, ctxs.get(ci));
    }

    match node {
        CuNode::Split { kids } => {
            let half = cb_size / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            let mut kids = kids.iter();

            write_cu(
                state,
                enc,
                w,
                ctxs,
                kids.next().unwrap(),
                x0,
                y0,
                log2_cb_size - 1,
                ct_depth + 1,
            );
            if x1 < state.display_width {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x1,
                    y0,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
            if y1 < state.display_height {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x0,
                    y1,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
            if x1 < state.display_width && y1 < state.display_height {
                write_cu(
                    state,
                    enc,
                    w,
                    ctxs,
                    kids.next().unwrap(),
                    x1,
                    y1,
                    log2_cb_size - 1,
                    ct_depth + 1,
                );
            }
        }
        CuNode::Leaf(leaf) => {
            if let Some(nxn) = &leaf.nxn {
                write_cu_nxn(state, enc, w, ctxs, leaf, nxn, x0, y0);
                return;
            }
            if log2_cb_size == 3 {
                enc.encode_bin(w, 1, ctxs.get(ctx::PART_MODE));
            }
            write_intra_luma_mode(enc, w, ctxs, leaf.mpm, leaf.luma_mode);
            if state.cat != 0 {
                write_intra_chroma_mode(enc, w, ctxs, leaf.chroma_mode_idx);
            }
            let cat = state.cat;
            if state.aq.active {
                state.aq_cu_begin(x0, y0);
            }
            write_tt(
                state, enc, w, ctxs, &leaf.tt, x0, y0, cat, false, true, true,
            );
        }
    }
}

/// Build the CU via RD analysis, mark deblock edges, and write its CABAC
/// syntax for one CTB.
pub(super) fn write_coding_quadtree(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
) -> CuNode {
    let _ = ctxs;
    let cu = super::stillsearch::StillSearch::new(state.bit_depth).build_ctu(
        state,
        x0,
        y0,
        log2_cb_size,
        ct_depth,
    );
    state.record_analysis_cache_cu_node(&cu, x0, y0, log2_cb_size, ct_depth);
    if state.deblock {
        let (display_width, display_height) = (state.display_width, state.display_height);
        mark_cu_deblock(
            &mut state.frame,
            &cu,
            x0,
            y0,
            log2_cb_size,
            display_width,
            display_height,
        );
    }
    write_cu(state, enc, w, ctxs, &cu, x0, y0, log2_cb_size, ct_depth);
    cu
}

/// Mark the deblocking-filter edge flags (`DecodedFrame::mark_tu_boundary`)
/// for every transform-unit leaf in a coded coding-quadtree node.
pub(super) fn mark_cu_deblock(
    frame: &mut DecodedFrame,
    node: &CuNode,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    display_width: u32,
    display_height: u32,
) {
    match node {
        CuNode::Split { kids } => {
            let half = (1u32 << log2_cb_size) / 2;
            let x1 = x0 + half;
            let y1 = y0 + half;
            let mut kids = kids.iter();
            let log2_kid = log2_cb_size - 1;

            mark_cu_deblock(
                frame,
                kids.next().unwrap(),
                x0,
                y0,
                log2_kid,
                display_width,
                display_height,
            );
            if x1 < display_width {
                mark_cu_deblock(
                    frame,
                    kids.next().unwrap(),
                    x1,
                    y0,
                    log2_kid,
                    display_width,
                    display_height,
                );
            }
            if y1 < display_height {
                mark_cu_deblock(
                    frame,
                    kids.next().unwrap(),
                    x0,
                    y1,
                    log2_kid,
                    display_width,
                    display_height,
                );
            }
            if x1 < display_width && y1 < display_height {
                mark_cu_deblock(
                    frame,
                    kids.next().unwrap(),
                    x1,
                    y1,
                    log2_kid,
                    display_width,
                    display_height,
                );
            }
        }
        CuNode::Leaf(leaf) => mark_tt_deblock(frame, &leaf.tt, x0, y0, log2_cb_size),
    }
}

/// Mark TU-leaf boundaries within a transform tree, mirroring
/// `build_tt_split`'s kid geometry (4 equal quadrants).
pub(super) fn mark_tt_deblock(
    frame: &mut DecodedFrame,
    node: &Tt,
    x0: u32,
    y0: u32,
    log2_size: u8,
) {
    match node {
        Tt::Split { kids, .. } => {
            let half = 1u32 << (log2_size - 1);
            let log2_kid = log2_size - 1;
            mark_tt_deblock(frame, &kids[0], x0, y0, log2_kid);
            mark_tt_deblock(frame, &kids[1], x0 + half, y0, log2_kid);
            mark_tt_deblock(frame, &kids[2], x0, y0 + half, log2_kid);
            mark_tt_deblock(frame, &kids[3], x0 + half, y0 + half, log2_kid);
        }
        Tt::Leaf(_) => frame.mark_tu_boundary(x0, y0, 1u32 << log2_size),
    }
}

/// Worker-thread budget retained for CLI compatibility; StillSearch skeleton
/// forces serial CTU build until overlay worker-merge exists.
pub(super) fn parallel_thread_count() -> usize {
    1
}

/// StillSearch skeleton: parallel snapshot/restore analysis is disabled. Build
/// serially so the shared reconstructed frame is committed exactly once in CTU
/// order.
pub(super) fn build_slice_trees_parallel(
    state: &mut Encoder<'_>,
    slice_qp_y: i32,
) -> Vec<Option<CuNode>> {
    build_slice_trees_serial(state, slice_qp_y)
}

/// Serial CABAC write of pre-built CTU trees, optionally prefixing each CTU
/// with `sao()` syntax. This is a pure bitstream pass: it reads the cached
/// `CuNode` coefficients/modes and evolves the CABAC contexts but does **not**
/// touch `state.frame`, so the caller may deblock + decide SAO on the
/// reconstruction before calling this.
/// Serialize the per-CTU trees into the slice's CABAC `slice_data()`. Returns
/// the bytes and the per-tile entry-point byte sizes (`entry_point_offset[i]`
/// for substreams `0..num_tiles-1`; empty when tiles are disabled).
///
/// When tiles are active, each tile is coded as an independent byte-aligned
/// substream (fresh CABAC engine + slice-init contexts, matching the decoder's
/// per-tile reset), then the substreams are concatenated — the HM/x265 method,
/// which guarantees byte-aligned substream boundaries.
pub(super) fn write_slice_from_trees(
    state: &mut Encoder<'_>,
    trees: &[Option<CuNode>],
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
) -> (Vec<u8>, Vec<u32>) {
    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let total = (ctbs_x * ctbs_y) as u32;

    if state.tile_grid.is_single() {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ctxs = Contexts::new(slice_qp_y);
        let mut done = 0u32;
        for cy in 0..ctbs_y {
            for cx in 0..ctbs_x {
                write_one_ctu_from_trees(
                    state, &mut enc, &mut w, &mut ctxs, trees, sao_map, cx, cy, ctbs_x, ctb,
                );
                done += 1;
                enc.encode_bin_trm(&mut w, (done == total) as u8);
            }
        }
        enc.finish(&mut w);
        w.write_bit(1);
        w.byte_align();
        return (w.into_bytes(), Vec::new());
    }

    // Tiled: one independent byte-aligned substream per tile.
    let num_tiles = state.tile_grid.num_tiles();
    let bounds: Vec<(u32, u32, u32, u32)> = (0..num_tiles)
        .map(|t| state.tile_grid.tile_ctb_bounds(t))
        .collect();
    let mut substreams: Vec<Vec<u8>> = Vec::with_capacity(num_tiles as usize);
    let mut done = 0u32;
    for (t, &(cx0, cy0, cx1, cy1)) in bounds.iter().enumerate() {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ctxs = Contexts::new(slice_qp_y);
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                write_one_ctu_from_trees(
                    state, &mut enc, &mut w, &mut ctxs, trees, sao_map, cx, cy, ctbs_x, ctb,
                );
                done += 1;
                // end_of_slice_segment_flag: 1 only for the final CTU of the slice.
                enc.encode_bin_trm(&mut w, (done == total) as u8);
            }
        }
        if t + 1 < num_tiles as usize {
            // end_of_sub_stream_one_bit (= 1) then byte_alignment(): an
            // alignment_bit_equal_to_one followed by zero bits to the byte
            // boundary. The leading 1 bit is mandatory — omitting it loses a
            // whole byte whenever `finish()` ends byte-aligned, desyncing the
            // next substream.
            enc.encode_bin_trm(&mut w, 1);
            enc.finish(&mut w);
            w.write_bit(1);
            w.byte_align();
        } else {
            // Final substream: rbsp_slice_segment_trailing_bits.
            enc.finish(&mut w);
            w.write_bit(1);
            w.byte_align();
        }
        substreams.push(w.into_bytes());
    }

    let entry_sizes = tile_entry_offsets(&substreams);
    if std::env::var_os("BPG_TILE_DEBUG").is_some() {
        let sizes: Vec<usize> = substreams.iter().map(|s| s.len()).collect();
        eprintln!(
            "TILE_DEBUG: num_tiles={num_tiles} bounds={bounds:?} rbsp_sizes={sizes:?} entry_offsets={entry_sizes:?}"
        );
    }
    let mut out = Vec::new();
    for s in &substreams {
        out.extend_from_slice(s);
    }
    (out, entry_sizes)
}

/// Compute `entry_point_offset[i]` for tile substreams. Per H.265, the offset is
/// the byte length of each substream **in the NAL unit** — i.e. including the
/// emulation_prevention_three_bytes (0x03) that `write_annexb_nal` will insert,
/// not the raw RBSP length. The zero-run is carried continuously across
/// substreams (matching the single de-emulation pass the decoder performs); the
/// run starts at 0 because the slice header always ends in a non-zero byte
/// (`alignment_bit_equal_to_one`). Offsets are returned for substreams
/// `0..n-1` (the last substream's size is implicit).
fn tile_entry_offsets(substreams: &[Vec<u8>]) -> Vec<u32> {
    let n = substreams.len();
    if n <= 1 {
        return Vec::new();
    }
    let mut offsets = Vec::with_capacity(n - 1);
    let mut zeros = 0u32;
    for s in &substreams[..n - 1] {
        let mut nal_len = 0u32;
        for &b in s {
            if zeros >= 2 && b <= 0x03 {
                nal_len += 1; // emulation_prevention_three_byte
                zeros = 0;
            }
            nal_len += 1;
            if b == 0 {
                zeros += 1;
            } else {
                zeros = 0;
            }
        }
        offsets.push(nal_len);
    }
    offsets
}

/// Write SAO syntax (if any) and the CU quadtree for one CTU from the prebuilt
/// trees. Shared by the single-substream and tiled paths.
#[allow(clippy::too_many_arguments)]
fn write_one_ctu_from_trees(
    state: &mut Encoder<'_>,
    enc: &mut CabacEncoder,
    w: &mut BitWriter,
    ctxs: &mut Contexts,
    trees: &[Option<CuNode>],
    sao_map: Option<&SaoMap>,
    cx: u32,
    cy: u32,
    ctbs_x: u32,
    ctb: u32,
) {
    state.stats.ctu_count += 1;
    if let Some(sao_map) = sao_map {
        let left_merge_avail = cx > 0 && state.tile_grid.same_tile_ctb(cx - 1, cy, cx, cy);
        let up_merge_avail = cy > 0 && state.tile_grid.same_tile_ctb(cx, cy - 1, cx, cy);
        sao::write_sao(
            enc,
            w,
            ctxs,
            sao_map,
            cx,
            cy,
            left_merge_avail,
            up_merge_avail,
            true,
            state.cat != 0,
            state.bit_depth,
        );
    }
    let cu = trees[(cy * ctbs_x + cx) as usize]
        .as_ref()
        .expect("every CTU built");
    write_cu(state, enc, w, ctxs, cu, cx * ctb, cy * ctb, CTB_LOG2, 0);
}

/// Encode `slice_segment_data()` for every CTU, optionally with SAO syntax.
pub(super) fn encode_slice_data(
    state: &mut Encoder<'_>,
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
) -> Vec<u8> {
    encode_slice_data_capture(state, sao_map, slice_qp_y, None)
}

/// Serial (interleaved build+write) slice encode. When `capture` is `Some`, the
/// per-CTU `CuNode` trees are stored (raster order) so the SAO path can replay
/// them with SAO syntax instead of re-running RDO — see [`build_slice_trees_serial`].
pub(super) fn encode_slice_data_capture(
    state: &mut Encoder<'_>,
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
    mut capture: Option<&mut Vec<Option<CuNode>>>,
) -> Vec<u8> {
    // This serial path emits a single CABAC substream in raster order with no
    // per-tile context reset or entry-point offsets. It is only ever selected
    // when tiles are disabled (see `params::tile_capable`: tiles require SAO or
    // parallel analysis, both of which route through `write_slice_from_trees`).
    // Assert that invariant so a future routing change can't silently produce a
    // PPS that declares tiles the slice data doesn't carry.
    debug_assert!(
        state.tile_grid.is_single(),
        "serial slice encode cannot emit a multi-tile partition ({} tiles); \
         tiled encodes must use write_slice_from_trees",
        state.tile_grid.num_tiles()
    );

    let mut w = BitWriter::new();
    let mut enc = CabacEncoder::new();
    let mut ctxs = Contexts::new(slice_qp_y);

    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let total = ctbs_x * ctbs_y;
    let mut done = 0u32;

    if state.aq.active {
        state.aq.reset_for_slice();
    }

    for cy in 0..ctbs_y {
        for cx in 0..ctbs_x {
            let x0 = cx * ctb;
            let y0 = cy * ctb;
            state.stats.ctu_count += 1;

            if let Some(sao_map) = sao_map {
                // Single-tile path (see the `is_single` assert above), so these
                // reduce to `cx > 0` / `cy > 0`; derived via the grid anyway so
                // the merge-availability rule stays correct in one place.
                let left_merge_avail = cx > 0 && state.tile_grid.same_tile_ctb(cx - 1, cy, cx, cy);
                let up_merge_avail = cy > 0 && state.tile_grid.same_tile_ctb(cx, cy - 1, cx, cy);
                sao::write_sao(
                    &mut enc,
                    &mut w,
                    &mut ctxs,
                    sao_map,
                    cx,
                    cy,
                    left_merge_avail,
                    up_merge_avail,
                    true,
                    state.cat != 0,
                    state.bit_depth,
                );
            }

            let cu = write_coding_quadtree(state, &mut enc, &mut w, &mut ctxs, x0, y0, CTB_LOG2, 0);
            if let Some(trees) = capture.as_deref_mut() {
                trees[(cy * ctbs_x + cx) as usize] = Some(cu);
            }

            done += 1;
            enc.encode_bin_trm(&mut w, (done == total) as u8);
        }
    }

    enc.finish(&mut w);
    w.write_bit(1);
    w.byte_align();
    w.into_bytes()
}

/// Serial build pass that captures every CTU's `CuNode` tree (and leaves the
/// reconstruction + deblock marks in `state.frame`), for the SAO replay path on
/// non-parallel tiers. The throwaway bitstream it writes is only there to evolve
/// the running CABAC context that serial RD pricing depends on.
pub(super) fn build_slice_trees_serial(
    state: &mut Encoder<'_>,
    slice_qp_y: i32,
) -> Vec<Option<CuNode>> {
    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let mut trees: Vec<Option<CuNode>> = (0..(ctbs_x * ctbs_y) as usize).map(|_| None).collect();
    let _ = encode_slice_data_capture(state, None, slice_qp_y, Some(&mut trees));
    trees
}
