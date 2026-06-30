//! CABAC bitstream emission for the H.265 still-picture intra slice-data
//! syntax: coding quadtree, intra modes, transform tree, residual coefficients,
//! SAO, and deblock-edge marking.
//!
//! Ported from `bpgenc.c`'s `put_*` / `*_coding_quadtree` / `*_transform_tree`
//! emit functions, matching the decode order in `bpg-hevc-decode::hevc::ctu`.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::sao::SaoMap;
use std::sync::{Arc, Condvar, Mutex};

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
/// mode (or four per-PU chroma modes for 4:4:4), and the forced-split transform
/// tree. Mirrors the decoder's `decode_coding_unit` `PartNxN` branch exactly.
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

    if state.cat == 3 {
        for pu in 0..4 {
            write_intra_chroma_mode(enc, w, ctxs, nxn.chroma_mode_idx[pu]);
        }
    } else if state.cat != 0 {
        write_intra_chroma_mode(enc, w, ctxs, leaf.chroma_mode_idx);
    }
    let cat = state.cat;
    if state.aq.active {
        state.aq_cu_begin(x0, y0, 3);
    }
    write_tt(state, enc, w, ctxs, &leaf.tt, x0, y0, cat, true, true, true);
    if state.aq.active {
        state.aq_node_end(x0, y0, 3);
    }
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
                if c.cb1.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 1, cat);
                    encode_residual(enc, w, ctxs, &c.cb1.levels, c.log2_size, 1, scan, sdh);
                }
                if c.cr.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &c.cr.levels, c.log2_size, 2, scan, sdh);
                }
                if c.cr1.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &c.cr1.levels, c.log2_size, 2, scan, sdh);
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
                state.aq_cu_begin(x0, y0, log2_cb_size);
            }
            write_tt(
                state, enc, w, ctxs, &leaf.tt, x0, y0, cat, false, true, true,
            );
            if state.aq.active {
                state.aq_node_end(x0, y0, log2_cb_size);
            }
        }
    }

    if state.aq.active && matches!(node, CuNode::Split { .. }) {
        state.aq_node_end(x0, y0, log2_cb_size);
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
    let price_ctx = ctxs.clone();
    let cu = super::stillsearch::StillSearch::new(state.bit_depth).build_ctu(
        state,
        &price_ctx,
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

/// Worker-thread budget for row-wavefront CTU analysis. `BPG_ENC_THREADS`
/// limits active row workers; otherwise we use host parallelism.
pub(super) fn parallel_thread_count() -> usize {
    std::env::var("BPG_ENC_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1)
}

/// Build CTU trees with WPP row-wavefront dependencies. Workers keep private
/// encoder state and publish deterministic CTU snapshots; AQ-active configs do
/// not enter this path until QP targets/syntax commits are split.
pub(super) fn build_slice_trees_parallel(
    state: &mut Encoder<'_>,
    slice_qp_y: i32,
) -> Vec<Option<CuNode>> {
    debug_assert!(
        !state.aq.active,
        "WPP v1 keeps AQ serial until QP targets and syntax prediction are split"
    );
    debug_assert!(
        state.tile_grid.is_single(),
        "WPP v1 is not combined with tiles"
    );

    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let total = (ctbs_x * ctbs_y) as usize;
    let worker_limit = parallel_thread_count().min(ctbs_y as usize).max(1);
    if worker_limit <= 1 || ctbs_x <= 1 || ctbs_y <= 1 {
        return build_slice_trees_serial(state, slice_qp_y);
    }

    let shared = Arc::new((
        Mutex::new(WppBuildShared {
            row_progress: vec![0; ctbs_y as usize],
            row_start_ctx: {
                let mut v = vec![None; ctbs_y as usize];
                v[0] = Some(Contexts::new(slice_qp_y));
                v
            },
            next_row: 0,
            snapshots: (0..total).map(|_| None).collect(),
            trees: (0..total).map(|_| None).collect(),
            stats: Default::default(),
        }),
        Condvar::new(),
    ));

    // Keep at most `worker_limit` live row states. The first WPP prototype
    // spawned one thread (and cloned one full `Encoder`/reconstruction frame)
    // per CTB row, then limited only the number of CTUs actively running. That
    // is fine for 12MP smoke tests but scales badly on tall/50MP stills: dozens
    // of parked row threads each held a full-frame clone. A fixed worker pool
    // claims rows in wavefront order as soon as their row-start context is
    // available, preserving the same CTU dependency graph while bounding peak
    // memory to O(threads), not O(rows).
    let master_state: &Encoder<'_> = &*state;
    std::thread::scope(|scope| {
        for _ in 0..worker_limit {
            let shared = Arc::clone(&shared);
            scope.spawn(move || {
                loop {
                    let cy = claim_wpp_row(&shared, ctbs_y);
                    let Some(cy) = cy else {
                        break;
                    };
                    let mut worker_state = clone_worker_encoder(master_state);
                    build_wpp_row(
                        &mut worker_state,
                        shared.clone(),
                        slice_qp_y,
                        cy,
                        ctbs_x,
                        ctbs_y,
                        ctb,
                    );
                }
            });
        }
    });

    let (lock, _) = &*shared;
    let mut guard = lock.lock().expect("WPP build mutex poisoned");
    let mut trees: Vec<Option<CuNode>> = (0..total).map(|_| None).collect();
    std::mem::swap(&mut trees, &mut guard.trees);

    for snapshot in guard.snapshots.iter().filter_map(|s| s.as_ref()) {
        apply_ctu_snapshot(state, snapshot);
    }
    state.stats.merge(&guard.stats);
    trees
}

struct WppBuildShared {
    row_progress: Vec<u32>,
    row_start_ctx: Vec<Option<Contexts>>,
    next_row: u32,
    snapshots: Vec<Option<CtuSnapshot>>,
    trees: Vec<Option<CuNode>>,
    stats: super::types::EncodeStats,
}

#[derive(Clone)]
struct CtuSnapshot {
    planes: Vec<PlanePatch>,
    mode: U8MapPatch,
    ct_depth: U8MapPatch,
    deblock: U8MapPatch,
    qp: I8MapPatch,
}

#[derive(Clone)]
struct PlanePatch {
    c_idx: u8,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    data: Vec<u16>,
}

#[derive(Clone)]
struct U8MapPatch {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    data: Vec<u8>,
}

#[derive(Clone)]
struct I8MapPatch {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    data: Vec<i8>,
}

fn claim_wpp_row(shared: &Arc<(Mutex<WppBuildShared>, Condvar)>, ctbs_y: u32) -> Option<u32> {
    let (lock, cvar) = &**shared;
    let mut guard = lock.lock().expect("WPP build mutex poisoned");
    loop {
        let cy = guard.next_row;
        if cy >= ctbs_y {
            return None;
        }
        if guard.row_start_ctx[cy as usize].is_some() {
            guard.next_row += 1;
            return Some(cy);
        }
        guard = cvar.wait(guard).expect("WPP build mutex poisoned");
    }
}

fn clone_worker_encoder<'a>(state: &Encoder<'a>) -> Encoder<'a> {
    Encoder {
        display_width: state.display_width,
        display_height: state.display_height,
        cat: state.cat,
        bit_depth: state.bit_depth,
        tile_grid: state.tile_grid.clone(),
        src: state.src,
        frame: state.frame.clone(),
        mode_map: state.mode_map.clone(),
        mode_stride: state.mode_stride,
        ct_depth_map: state.ct_depth_map.clone(),
        ct_depth_stride: state.ct_depth_stride,
        deblock: state.deblock,
        sign_data_hiding: state.sign_data_hiding,
        aq: state.aq.clone(),
        cur_qp_y: state.cur_qp_y,
        cur_qp_c: state.cur_qp_c,
        aq_mode: state.aq_mode,
        aq_strength: state.aq_strength,
        aq_clamp: state.aq_clamp,
        aq_offset_map: state.aq_offset_map.as_ref().map(Arc::clone),
        part_nxn_enabled: state.part_nxn_enabled,
        analysis: Arc::clone(&state.analysis),
        stats: Default::default(),
        effort_template: state.effort_template,
    }
}

fn build_wpp_row(
    state: &mut Encoder<'_>,
    shared: Arc<(Mutex<WppBuildShared>, Condvar)>,
    _slice_qp_y: i32,
    cy: u32,
    ctbs_x: u32,
    ctbs_y: u32,
    ctb: u32,
) {
    let (lock, cvar) = &*shared;
    let mut search = super::stillsearch::StillSearch::new(state.bit_depth);
    let mut prev_applied = 0u32;

    let mut ctxs = {
        let mut guard = lock.lock().expect("WPP build mutex poisoned");
        loop {
            if let Some(ctxs) = guard.row_start_ctx[cy as usize].clone() {
                break ctxs;
            }
            guard = cvar.wait(guard).expect("WPP build mutex poisoned");
        }
    };
    let mut enc = CabacEncoder::new();
    let mut w = BitWriter::new();
    let total = ctbs_x * ctbs_y;

    for cx in 0..ctbs_x {
        let need_prev = if cy == 0 { 0 } else { (cx + 2).min(ctbs_x) };
        let dependency_snapshots = {
            let mut guard = lock.lock().expect("WPP build mutex poisoned");
            loop {
                let prev_ready = cy == 0 || guard.row_progress[cy as usize - 1] >= need_prev;
                let left_ready = cx == 0 || guard.row_progress[cy as usize] >= cx;
                if prev_ready && left_ready {
                    let mut snapshots = Vec::new();
                    if cy > 0 {
                        for px in prev_applied..need_prev {
                            let idx = (cy - 1) as usize * ctbs_x as usize + px as usize;
                            snapshots.push(
                                guard.snapshots[idx]
                                    .as_ref()
                                    .expect("previous-row CTU snapshot missing")
                                    .clone(),
                            );
                        }
                    }
                    prev_applied = need_prev;
                    break snapshots;
                }
                guard = cvar.wait(guard).expect("WPP build mutex poisoned");
            }
        };

        for snapshot in &dependency_snapshots {
            apply_ctu_snapshot(state, snapshot);
        }

        let x0 = cx * ctb;
        let y0 = cy * ctb;
        let price_ctx = ctxs.clone();
        let cu = search.build_ctu(state, &price_ctx, x0, y0, CTB_LOG2, 0);
        state.record_analysis_cache_cu_node(&cu, x0, y0, CTB_LOG2, 0);
        if state.deblock {
            let (display_width, display_height) = (state.display_width, state.display_height);
            mark_cu_deblock(
                &mut state.frame,
                &cu,
                x0,
                y0,
                CTB_LOG2,
                display_width,
                display_height,
            );
        }
        write_cu(state, &mut enc, &mut w, &mut ctxs, &cu, x0, y0, CTB_LOG2, 0);
        let done = cy * ctbs_x + cx + 1;
        enc.encode_bin_trm(&mut w, (done == total) as u8);

        let snapshot = snapshot_ctu(state, cx, cy, ctb);
        let stats = std::mem::take(&mut state.stats);
        let row_start_for_next = if cx == 1 && cy + 1 < ctbs_y {
            Some(ctxs.clone())
        } else {
            None
        };

        let mut guard = lock.lock().expect("WPP build mutex poisoned");
        let idx = cy as usize * ctbs_x as usize + cx as usize;
        guard.snapshots[idx] = Some(snapshot);
        guard.trees[idx] = Some(cu);
        guard.row_progress[cy as usize] = cx + 1;
        if let Some(ctx) = row_start_for_next {
            guard.row_start_ctx[cy as usize + 1] = Some(ctx);
        }
        guard.stats.merge(&stats);
        cvar.notify_all();
    }
}

fn snapshot_ctu(state: &Encoder<'_>, cx: u32, cy: u32, ctb: u32) -> CtuSnapshot {
    let mut planes = Vec::new();
    let plane_count = if state.cat == 0 { 1 } else { 3 };
    for c_idx in 0..plane_count {
        let (plane, stride) = state.frame.plane(c_idx);
        let plane_h = if stride == 0 { 0 } else { plane.len() / stride };
        let (sx, sy) = state.plane_shifts(c_idx);
        let x0 = ((cx * ctb) >> sx).min(stride as u32) as usize;
        let y0 = ((cy * ctb) >> sy).min(plane_h as u32) as usize;
        let x1 = ((cx + 1) * ctb).div_ceil(1u32 << sx).min(stride as u32) as usize;
        let y1 = ((cy + 1) * ctb).div_ceil(1u32 << sy).min(plane_h as u32) as usize;
        planes.push(snapshot_plane(c_idx, plane, stride, x0, y0, x1, y1));
    }

    let mode = snapshot_u8_map(
        &state.mode_map,
        state.mode_stride,
        cx as usize * 16,
        cy as usize * 16,
        (cx as usize + 1) * 16,
        (cy as usize + 1) * 16,
    );
    let ct_depth = snapshot_u8_map(
        &state.ct_depth_map,
        state.ct_depth_stride,
        cx as usize * 8,
        cy as usize * 8,
        (cx as usize + 1) * 8,
        (cy as usize + 1) * 8,
    );
    let deblock = snapshot_u8_map(
        &state.frame.deblock_flags,
        state.frame.deblock_stride as usize,
        cx as usize * 16,
        cy as usize * 16,
        (cx as usize + 1) * 16,
        (cy as usize + 1) * 16,
    );
    let qp = snapshot_i8_map(
        &state.frame.qp_map,
        state.frame.deblock_stride as usize,
        cx as usize * 16,
        cy as usize * 16,
        (cx as usize + 1) * 16,
        (cy as usize + 1) * 16,
    );

    CtuSnapshot {
        planes,
        mode,
        ct_depth,
        deblock,
        qp,
    }
}

fn snapshot_plane(
    c_idx: u8,
    plane: &[u16],
    stride: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> PlanePatch {
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    let mut data = Vec::with_capacity(w * h);
    for y in y0..y1 {
        data.extend_from_slice(&plane[y * stride + x0..][..w]);
    }
    PlanePatch {
        c_idx,
        x: x0,
        y: y0,
        w,
        h,
        data,
    }
}

fn snapshot_u8_map(
    buf: &[u8],
    stride: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> U8MapPatch {
    let height = if stride == 0 { 0 } else { buf.len() / stride };
    let x0 = x0.min(stride);
    let x1 = x1.min(stride);
    let y0 = y0.min(height);
    let y1 = y1.min(height);
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    let mut data = Vec::with_capacity(w * h);
    for y in y0..y1 {
        data.extend_from_slice(&buf[y * stride + x0..][..w]);
    }
    U8MapPatch {
        x: x0,
        y: y0,
        w,
        h,
        data,
    }
}

fn snapshot_i8_map(
    buf: &[i8],
    stride: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> I8MapPatch {
    let height = if stride == 0 { 0 } else { buf.len() / stride };
    let x0 = x0.min(stride);
    let x1 = x1.min(stride);
    let y0 = y0.min(height);
    let y1 = y1.min(height);
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    let mut data = Vec::with_capacity(w * h);
    for y in y0..y1 {
        data.extend_from_slice(&buf[y * stride + x0..][..w]);
    }
    I8MapPatch {
        x: x0,
        y: y0,
        w,
        h,
        data,
    }
}

fn apply_ctu_snapshot(state: &mut Encoder<'_>, snapshot: &CtuSnapshot) {
    for patch in &snapshot.planes {
        let (plane, stride) = state.frame.plane_mut(patch.c_idx);
        apply_u16_rect(
            plane,
            stride,
            patch.x,
            patch.y,
            patch.w,
            patch.h,
            &patch.data,
        );
    }
    apply_u8_rect(&mut state.mode_map, state.mode_stride, &snapshot.mode);
    apply_u8_rect(
        &mut state.ct_depth_map,
        state.ct_depth_stride,
        &snapshot.ct_depth,
    );
    apply_u8_rect(
        &mut state.frame.deblock_flags,
        state.frame.deblock_stride as usize,
        &snapshot.deblock,
    );
    apply_i8_rect(
        &mut state.frame.qp_map,
        state.frame.deblock_stride as usize,
        &snapshot.qp,
    );
}

fn apply_u16_rect(
    buf: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    data: &[u16],
) {
    for row in 0..h {
        let dst = (y + row) * stride + x;
        let src = row * w;
        buf[dst..][..w].copy_from_slice(&data[src..][..w]);
    }
}

fn apply_u8_rect(buf: &mut [u8], stride: usize, patch: &U8MapPatch) {
    for row in 0..patch.h {
        let dst = (patch.y + row) * stride + patch.x;
        let src = row * patch.w;
        buf[dst..][..patch.w].copy_from_slice(&patch.data[src..][..patch.w]);
    }
}

fn apply_i8_rect(buf: &mut [i8], stride: usize, patch: &I8MapPatch) {
    for row in 0..patch.h {
        let dst = (patch.y + row) * stride + patch.x;
        let src = row * patch.w;
        buf[dst..][..patch.w].copy_from_slice(&patch.data[src..][..patch.w]);
    }
}

/// Serial CABAC write of pre-built CTU trees, optionally prefixing each CTU
/// with `sao()` syntax. This is a pure bitstream pass: it reads the cached
/// `CuNode` coefficients/modes and evolves the CABAC contexts but does **not**
/// touch `state.frame`, so the caller may deblock + decide SAO on the
/// reconstruction before calling this.
/// Serialize the per-CTU trees into the slice's CABAC `slice_data()`. Returns
/// the bytes and the entry-point byte sizes (`entry_point_offset[i]` for
/// substreams `0..n-1`; empty when a single substream is emitted).
///
/// When tiles are active, each tile is coded as an independent byte-aligned
/// substream (fresh CABAC engine + slice-init contexts, matching the decoder's
/// per-tile reset), then the substreams are concatenated — the HM/x265 method,
/// which guarantees byte-aligned substream boundaries.
///
/// When WPP is active, each CTB row is a byte-aligned substream. Row contexts
/// start from the previous row's context snapshot after CTB column 1; this
/// matches H.265 entropy-coding-sync and keeps AQ out of WPP v1 by construction
/// (AQ-active configs do not resolve to WPP).
pub(super) fn write_slice_from_trees(
    state: &mut Encoder<'_>,
    trees: &[Option<CuNode>],
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
    wpp: bool,
) -> (Vec<u8>, Vec<u32>) {
    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let total = (ctbs_x * ctbs_y) as u32;

    if wpp {
        return write_slice_from_trees_wpp(state, trees, sao_map, slice_qp_y, ctbs_x, ctbs_y, ctb);
    }

    if state.aq.active {
        state.aq.reset_for_slice();
    }

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
        if state.aq.active {
            state.aq.reset_for_slice();
        }
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

    let entry_sizes = substream_entry_offsets(&substreams);
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
fn write_slice_from_trees_wpp(
    state: &mut Encoder<'_>,
    trees: &[Option<CuNode>],
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
    ctbs_x: u32,
    ctbs_y: u32,
    ctb: u32,
) -> (Vec<u8>, Vec<u32>) {
    debug_assert!(
        state.tile_grid.is_single(),
        "WPP v1 is not combined with tiles"
    );
    debug_assert!(ctbs_x > 1 && ctbs_y > 1, "WPP requires at least 2x2 CTBs");

    let total = ctbs_x * ctbs_y;
    let mut substreams: Vec<Vec<u8>> = Vec::with_capacity(ctbs_y as usize);
    let mut next_row_ctx = Contexts::new(slice_qp_y);
    let mut done = 0u32;

    for cy in 0..ctbs_y {
        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        let mut ctxs = next_row_ctx.clone();
        let mut row_saved_ctx: Option<Contexts> = None;

        for cx in 0..ctbs_x {
            write_one_ctu_from_trees(
                state, &mut enc, &mut w, &mut ctxs, trees, sao_map, cx, cy, ctbs_x, ctb,
            );
            done += 1;
            enc.encode_bin_trm(&mut w, (done == total) as u8);
            if cx == 1 && cy + 1 < ctbs_y {
                row_saved_ctx = Some(ctxs.clone());
            }
        }

        if cy + 1 < ctbs_y {
            enc.encode_bin_trm(&mut w, 1); // end_of_sub_stream_one_bit
            enc.finish(&mut w);
            w.write_bit(1);
            w.byte_align();
            next_row_ctx = row_saved_ctx.expect("WPP requires at least two CTBs per row");
        } else {
            enc.finish(&mut w);
            w.write_bit(1);
            w.byte_align();
        }
        substreams.push(w.into_bytes());
    }

    let entry_sizes = substream_entry_offsets(&substreams);
    if std::env::var_os("BPG_WPP_DEBUG").is_some() {
        let sizes: Vec<usize> = substreams.iter().map(|s| s.len()).collect();
        eprintln!("WPP_DEBUG: rows={ctbs_y} rbsp_sizes={sizes:?} entry_offsets={entry_sizes:?}");
    }

    let mut out = Vec::new();
    for s in &substreams {
        out.extend_from_slice(s);
    }
    (out, entry_sizes)
}

fn substream_entry_offsets(substreams: &[Vec<u8>]) -> Vec<u32> {
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
