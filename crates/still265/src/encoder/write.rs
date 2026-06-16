//! CABAC bitstream emission for the H.265 still-picture intra slice-data
//! syntax: coding quadtree, intra modes, transform tree, residual coefficients,
//! SAO, and deblock-edge marking.
//!
//! Ported from `bpgenc.c`'s `put_*` / `*_coding_quadtree` / `*_transform_tree`
//! emit functions, matching the decode order in `bpg-hevc-decode::hevc::ctu`.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::hevc::sao::SaoMap;
use bpg_hevc_decode::DecodedFrame;

use crate::cabac::CabacEncoder;
use crate::contexts::{ctx, Contexts};
use crate::residual::{encode_residual, get_scan_order};
use crate::sao;

use super::aq::encode_cu_qp_delta;
use super::types::{
    chroma_pred_mode, decode_second_cbf, has_chroma_tb, CtuBuild, CuNode, Tt,
    CHROMA_DM_IDX, CTB_LOG2, MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, MIN_TB_LOG2,
};
use super::Encoder;

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
                    state, enc, w, ctxs, kid, kx, ky, cat,
                    intra_split_flag, cbf_cb, cbf_cr,
                );
            }
            if let Some(c) = parent_chroma {
                if c.cb.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 1, cat);
                    encode_residual(enc, w, ctxs, &c.cb.levels, c.log2_size, 1, scan, false);
                }
                if c.cr.cbf {
                    let scan = get_scan_order(c.log2_size, c.chroma_mode, 2, cat);
                    encode_residual(enc, w, ctxs, &c.cr.levels, c.log2_size, 2, scan, false);
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
                encode_residual(enc, w, ctxs, &l.luma.levels, l.log2_size, 0, scan, false);
            }
            if has_chroma_tb(cat, l.log2_size) {
                let clog2 = l.chroma_log2;
                let cmode = chroma_pred_mode(cat, l.chroma_mode);
                if eff_cbf_cb {
                    let scan = get_scan_order(clog2, cmode, 1, cat);
                    encode_residual(enc, w, ctxs, &l.cb.levels, clog2, 1, scan, false);
                }
                if eff_cbf_cb1 {
                    let scan = get_scan_order(clog2, cmode, 1, cat);
                    encode_residual(enc, w, ctxs, &l.cb1.levels, clog2, 1, scan, false);
                }
                if eff_cbf_cr {
                    let scan = get_scan_order(clog2, cmode, 2, cat);
                    encode_residual(enc, w, ctxs, &l.cr.levels, clog2, 2, scan, false);
                }
                if eff_cbf_cr1 {
                    let scan = get_scan_order(clog2, cmode, 2, cat);
                    encode_residual(enc, w, ctxs, &l.cr1.levels, clog2, 2, scan, false);
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

            write_cu(state, enc, w, ctxs, kids.next().unwrap(), x0, y0, log2_cb_size - 1, ct_depth + 1);
            if x1 < state.display_width {
                write_cu(state, enc, w, ctxs, kids.next().unwrap(), x1, y0, log2_cb_size - 1, ct_depth + 1);
            }
            if y1 < state.display_height {
                write_cu(state, enc, w, ctxs, kids.next().unwrap(), x0, y1, log2_cb_size - 1, ct_depth + 1);
            }
            if x1 < state.display_width && y1 < state.display_height {
                write_cu(state, enc, w, ctxs, kids.next().unwrap(), x1, y1, log2_cb_size - 1, ct_depth + 1);
            }
        }
        CuNode::Leaf(leaf) => {
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
            write_tt(state, enc, w, ctxs, &leaf.tt, x0, y0, cat, false, true, true);
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
) {
    let cu = state.build_cu(x0, y0, log2_cb_size, ct_depth, ctxs);
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

            mark_cu_deblock(frame, kids.next().unwrap(), x0, y0, log2_kid, display_width, display_height);
            if x1 < display_width {
                mark_cu_deblock(frame, kids.next().unwrap(), x1, y0, log2_kid, display_width, display_height);
            }
            if y1 < display_height {
                mark_cu_deblock(frame, kids.next().unwrap(), x0, y1, log2_kid, display_width, display_height);
            }
            if x1 < display_width && y1 < display_height {
                mark_cu_deblock(frame, kids.next().unwrap(), x1, y1, log2_kid, display_width, display_height);
            }
        }
        CuNode::Leaf(leaf) => mark_tt_deblock(frame, &leaf.tt, x0, y0, log2_cb_size),
    }
}

/// Mark TU-leaf boundaries within a transform tree, mirroring
/// `build_tt_split`'s kid geometry (4 equal quadrants).
pub(super) fn mark_tt_deblock(frame: &mut DecodedFrame, node: &Tt, x0: u32, y0: u32, log2_size: u8) {
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

/// Worker-thread budget for the parallel `Placebo` analysis. `BPG_ENC_THREADS`
/// overrides it (set to 1 for the determinism oracle).
pub(super) fn parallel_thread_count() -> usize {
    if let Ok(v) = std::env::var("BPG_ENC_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// `Placebo`-tier encode: CTU-wavefront-**parallel** analysis followed by a serial
/// CABAC write.
pub(super) fn encode_slice_data_parallel(state: &mut Encoder<'_>, slice_qp_y: i32) -> Vec<u8> {
    let ctb = 1u32 << CTB_LOG2;
    let ctbs_x = state.display_width.div_ceil(ctb);
    let ctbs_y = state.display_height.div_ceil(ctb);
    let n_ctus = (ctbs_x * ctbs_y) as usize;

    let slice_init = Contexts::new(slice_qp_y);
    let wpp = state.best2_wpp;
    // WPP analysis-context propagation: `row_ctx[cy]` is the context the next
    // (left-to-right) CTU of row `cy` is priced against. Row 0 starts from the
    // slice-init context; lower rows are seeded from the row above's CTU at
    // `sync_cx` (HEVC WPP: 2nd CTU, index 1). When `wpp` is off (Placebo), every
    // CTU is priced against the cold slice-init context (the original frozen
    // behavior; byte-identical because the priced context is the same value).
    let sync_cx = if ctbs_x > 1 { 1 } else { 0 };
    let mut row_ctx: Vec<Contexts> = vec![slice_init.clone(); ctbs_y.max(1) as usize];

    let n_threads = parallel_thread_count().min(ctbs_y.max(1) as usize).max(1);
    let mut workers: Vec<Encoder<'_>> = (0..n_threads).map(|_| state.fork_worker()).collect();
    let mut trees: Vec<Option<CuNode>> = (0..n_ctus).map(|_| None).collect();

    let max_t = (ctbs_x - 1) + 2 * (ctbs_y - 1);
    for t in 0..=max_t {
        let mut step: Vec<(u32, u32)> = Vec::new();
        for cy in 0..ctbs_y {
            if 2 * cy > t {
                break;
            }
            let cx = t - 2 * cy;
            if cx < ctbs_x {
                step.push((cx, cy));
            }
        }
        if step.is_empty() {
            continue;
        }

        // Snapshot the pricing context for each CTU in this diagonal *before* the
        // parallel build (the merge below mutates `row_ctx`).
        let step_ctx: Vec<Contexts> = step
            .iter()
            .map(|&(_, cy)| {
                if wpp {
                    row_ctx[cy as usize].clone()
                } else {
                    slice_init.clone()
                }
            })
            .collect();

        let n_used = n_threads.min(step.len());
        let chunk_len = step.len().div_ceil(n_used);

        let results: Vec<Vec<CtuBuild>> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (worker, (chunk, cchunk)) in workers
                .iter_mut()
                .zip(step.chunks(chunk_len).zip(step_ctx.chunks(chunk_len)))
            {
                handles.push(s.spawn(move || {
                    let mut out = Vec::with_capacity(chunk.len());
                    for (i, &(cx, cy)) in chunk.iter().enumerate() {
                        let x0 = cx * ctb;
                        let y0 = cy * ctb;
                        let cu = worker.build_cu(x0, y0, CTB_LOG2, 0, &cchunk[i]);
                        let fsnap = worker.snapshot_frame_region(x0, y0, CTB_LOG2);
                        let msnap = worker.snapshot_mode_region(x0, y0, CTB_LOG2);
                        let csnap = worker.snapshot_ct_depth_region(x0, y0, CTB_LOG2);
                        out.push(CtuBuild { cx, cy, cu, fsnap, msnap, csnap });
                    }
                    out
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for builds in results {
            for b in builds {
                state.restore_frame_region(&b.fsnap);
                state.restore_mode_region(&b.msnap);
                state.restore_ct_depth_region(&b.csnap);
                for w in workers.iter_mut() {
                    w.restore_frame_region(&b.fsnap);
                    w.restore_mode_region(&b.msnap);
                    w.restore_ct_depth_region(&b.csnap);
                }
                trees[(b.cy * ctbs_x + b.cx) as usize] = Some(b.cu);
            }
        }

        // WPP: evolve each row's pricing context through the CTU just built (in
        // raster-within-row order — one CTU per row per diagonal), and seed the
        // row below at the sync column. Done serially after the barrier so the
        // propagation is deterministic regardless of thread count.
        if wpp {
            let mut sw = BitWriter::new();
            let mut se = CabacEncoder::new();
            for &(cx, cy) in &step {
                let cu = trees[(cy * ctbs_x + cx) as usize]
                    .as_ref()
                    .expect("CTU built this step");
                write_cu(
                    state,
                    &mut se,
                    &mut sw,
                    &mut row_ctx[cy as usize],
                    cu,
                    cx * ctb,
                    cy * ctb,
                    CTB_LOG2,
                    0,
                );
                if cx == sync_cx && cy + 1 < ctbs_y {
                    row_ctx[(cy + 1) as usize] = row_ctx[cy as usize].clone();
                }
            }
        }
    }

    for w in &workers {
        state.stats.merge(&w.stats);
    }

    if state.deblock {
        let (dw, dh) = (state.display_width, state.display_height);
        for cy in 0..ctbs_y {
            for cx in 0..ctbs_x {
                if let Some(cu) = &trees[(cy * ctbs_x + cx) as usize] {
                    mark_cu_deblock(&mut state.frame, cu, cx * ctb, cy * ctb, CTB_LOG2, dw, dh);
                }
            }
        }
    }

    let mut w = BitWriter::new();
    let mut enc = CabacEncoder::new();
    let mut ctxs = Contexts::new(slice_qp_y);
    let total = n_ctus as u32;
    let mut done = 0u32;
    for cy in 0..ctbs_y {
        for cx in 0..ctbs_x {
            state.stats.ctu_count += 1;
            let cu = trees[(cy * ctbs_x + cx) as usize]
                .as_ref()
                .expect("every CTU built");
            write_cu(state, &mut enc, &mut w, &mut ctxs, cu, cx * ctb, cy * ctb, CTB_LOG2, 0);
            done += 1;
            enc.encode_bin_trm(&mut w, (done == total) as u8);
        }
    }
    enc.finish(&mut w);
    w.write_bit(1);
    w.byte_align();
    w.into_bytes()
}

/// Encode `slice_segment_data()` for every CTU, optionally with SAO syntax.
pub(super) fn encode_slice_data(
    state: &mut Encoder<'_>,
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
) -> Vec<u8> {
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
                sao::write_sao(
                    &mut enc, &mut w, &mut ctxs, sao_map, cx, cy, true, state.cat != 0, state.bit_depth,
                );
            }

            write_coding_quadtree(state, &mut enc, &mut w, &mut ctxs, x0, y0, CTB_LOG2, 0);

            done += 1;
            enc.encode_bin_trm(&mut w, (done == total) as u8);
        }
    }

    enc.finish(&mut w);
    w.write_bit(1);
    w.byte_align();
    w.into_bytes()
}
