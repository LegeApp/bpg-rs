//! CABAC syntax cost helpers owned by rdo2.

use bpg_hevc_decode::hevc::slice::IntraPredMode;

use crate::cabac::CabacEstimator;
use crate::contexts::{ctx, Contexts};

use super::super::types::{
    decode_second_cbf, has_chroma_tb, CuLeaf, CuNode, NxnInfo, Tt, CHROMA_DM_IDX,
    MAX_INTRA_TT_DEPTH, MAX_TB_LOG2, MIN_TB_LOG2,
};
use super::super::Encoder;

pub(in crate::encoder) fn estimate_tt_bits(
    ctxs: &Contexts,
    node: &Tt,
    cat: u8,
    intra_split_flag: bool,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
) -> u64 {
    let mut ctxs = ctxs.clone();
    let mut est = CabacEstimator::new();
    estimate_tt_inner(
        &mut est,
        &mut ctxs,
        node,
        cat,
        intra_split_flag,
        parent_cbf_cb,
        parent_cbf_cr,
    );
    est.frac_bits()
}

fn estimate_tt_inner(
    est: &mut CabacEstimator,
    ctxs: &mut Contexts,
    node: &Tt,
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
        est.encode_bin(is_split as u8, ctxs.get(ci));
    }

    let decode_chroma_cbf = cat != 0 && (log2_size > 2 || cat == 3);
    let (cbf_cb, cbf_cr) = (node.cbf_cb(), node.cbf_cr());
    let second_cbf = decode_second_cbf(cat, log2_size, is_split);
    let (cbf_cb1, cbf_cr1) = (node.cbf_cb1(), node.cbf_cr1());
    if decode_chroma_cbf {
        let ci = ctx::CBF_CBCR + trafo_depth as usize;
        if trafo_depth == 0 || parent_cbf_cb {
            est.encode_bin(cbf_cb as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cb) {
            est.encode_bin(cbf_cb1 as u8, ctxs.get(ci));
        }
        if trafo_depth == 0 || parent_cbf_cr {
            est.encode_bin(cbf_cr as u8, ctxs.get(ci));
        }
        if second_cbf && (trafo_depth == 0 || parent_cbf_cr) {
            est.encode_bin(cbf_cr1 as u8, ctxs.get(ci));
        }
    }

    match node {
        Tt::Split {
            kids,
            parent_chroma,
            ..
        } => {
            for kid in kids {
                estimate_tt_inner(est, ctxs, kid, cat, intra_split_flag, cbf_cb, cbf_cr);
            }
            if let Some(c) = parent_chroma {
                if c.cb.cbf {
                    est.add_frac_bits(c.cb.frac_bits);
                }
                if c.cr.cbf {
                    est.add_frac_bits(c.cr.frac_bits);
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
            est.encode_bin(l.luma.cbf as u8, ctxs.get(ctx::CBF_LUMA + ctx_off));

            if l.luma.cbf {
                est.add_frac_bits(l.luma.frac_bits);
            }
            if has_chroma_tb(cat, l.log2_size) {
                if eff_cbf_cb {
                    est.add_frac_bits(l.cb.frac_bits);
                }
                if eff_cbf_cb1 {
                    est.add_frac_bits(l.cb1.frac_bits);
                }
                if eff_cbf_cr {
                    est.add_frac_bits(l.cr.frac_bits);
                }
                if eff_cbf_cr1 {
                    est.add_frac_bits(l.cr1.frac_bits);
                }
            }
        }
    }
}

pub(in crate::encoder) fn estimate_intra_luma_mode_bits(
    ctxs: &Contexts,
    mpm: [IntraPredMode; 3],
    mode: u8,
) -> u64 {
    let mut m = ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let mut est = CabacEstimator::new();
    let mpm_u8 = [mpm[0].as_u8(), mpm[1].as_u8(), mpm[2].as_u8()];
    let in_mpm = mpm_u8.iter().position(|&m| m == mode);

    if let Some(idx) = in_mpm {
        est.encode_bin(1, &mut m);
        match idx {
            0 => est.encode_bin_ep(0),
            1 => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(0);
            }
            _ => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(1);
            }
        }
    } else {
        est.encode_bin(0, &mut m);
        for _ in 0..5 {
            est.encode_bin_ep(0);
        }
    }

    est.frac_bits()
}

pub(in crate::encoder) fn estimate_intra_chroma_mode_bits(ctxs: &Contexts, mode_idx: u8) -> u64 {
    let mut m = ctxs.models[ctx::INTRA_CHROMA_PRED_MODE];
    let mut est = CabacEstimator::new();
    if mode_idx == CHROMA_DM_IDX {
        est.encode_bin(0, &mut m);
    } else {
        est.encode_bin(1, &mut m);
        est.encode_bin_ep((mode_idx >> 1) & 1);
        est.encode_bin_ep(mode_idx & 1);
    }
    est.frac_bits()
}

pub(in crate::encoder) fn estimate_split_cu_flag_bits(
    ctxs: &Contexts,
    ctx_inc: usize,
    value: bool,
) -> u64 {
    let mut m = ctxs.models[ctx::SPLIT_CU_FLAG + ctx_inc];
    let mut est = CabacEstimator::new();
    est.encode_bin(value as u8, &mut m);
    est.frac_bits()
}

pub(in crate::encoder) fn estimate_cu_leaf_bits(
    ctxs: &Contexts,
    leaf: &CuLeaf,
    log2_cb_size: u8,
    cat: u8,
) -> u64 {
    if let Some(nxn) = &leaf.nxn {
        return estimate_cu_leaf_nxn_bits(ctxs, leaf, nxn, cat);
    }
    let mut bits = 0u64;
    if log2_cb_size == 3 {
        let mut m = ctxs.models[ctx::PART_MODE];
        let mut est = CabacEstimator::new();
        est.encode_bin(1, &mut m);
        bits += est.frac_bits();
    }
    bits += estimate_intra_luma_mode_bits(ctxs, leaf.mpm, leaf.luma_mode);
    if cat != 0 {
        bits += estimate_intra_chroma_mode_bits(ctxs, leaf.chroma_mode_idx);
    }
    bits += estimate_tt_bits(ctxs, &leaf.tt, cat, false, true, true);
    bits
}

fn estimate_cu_leaf_nxn_bits(ctxs: &Contexts, leaf: &CuLeaf, nxn: &NxnInfo, cat: u8) -> u64 {
    let mut est = CabacEstimator::new();
    let mut part_m = ctxs.models[ctx::PART_MODE];
    est.encode_bin(0, &mut part_m);

    let mut flag_m = ctxs.models[ctx::PREV_INTRA_LUMA_PRED_FLAG];
    let mut in_mpm = [None; 4];
    for (pu, slot) in in_mpm.iter_mut().enumerate() {
        let mode = nxn.luma_modes[pu];
        let mpm_u8 = [
            nxn.mpms[pu][0].as_u8(),
            nxn.mpms[pu][1].as_u8(),
            nxn.mpms[pu][2].as_u8(),
        ];
        let idx = mpm_u8.iter().position(|&m| m == mode);
        est.encode_bin(idx.is_some() as u8, &mut flag_m);
        *slot = idx;
    }
    for &idx in &in_mpm {
        match idx {
            Some(0) => est.encode_bin_ep(0),
            Some(1) => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(0);
            }
            Some(_) => {
                est.encode_bin_ep(1);
                est.encode_bin_ep(1);
            }
            None => {
                for _ in 0..5 {
                    est.encode_bin_ep(0);
                }
            }
        }
    }
    let mut bits = est.frac_bits();
    if cat != 0 {
        bits += estimate_intra_chroma_mode_bits(ctxs, leaf.chroma_mode_idx);
    }
    bits += estimate_tt_bits(ctxs, &leaf.tt, cat, true, true, true);
    bits
}

pub(in crate::encoder) fn estimate_cu_node_bits(
    ctxs: &Contexts,
    node: &CuNode,
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
    state: &Encoder<'_>,
) -> u64 {
    let cb_size = 1u32 << log2_cb_size;
    let fully_inside = x0 + cb_size <= state.display_width && y0 + cb_size <= state.display_height;
    let can_split = log2_cb_size > 3;
    let split_coded = fully_inside && can_split;

    let mut bits = 0u64;
    if split_coded {
        let ctx_inc = state.split_ctx_inc(x0, y0, ct_depth);
        bits += estimate_split_cu_flag_bits(ctxs, ctx_inc, matches!(node, CuNode::Split { .. }));
    }
    match node {
        CuNode::Split { kids } => {
            bits += estimate_cu_kids_bits(ctxs, kids, x0, y0, log2_cb_size, ct_depth, state);
        }
        CuNode::Leaf(leaf) => {
            bits += estimate_cu_leaf_bits(ctxs, leaf, log2_cb_size, state.cat);
        }
    }
    bits
}

fn estimate_cu_kids_bits(
    ctxs: &Contexts,
    kids: &[CuNode],
    x0: u32,
    y0: u32,
    log2_cb_size: u8,
    ct_depth: u8,
    state: &Encoder<'_>,
) -> u64 {
    let half = (1u32 << log2_cb_size) / 2;
    let x1 = x0 + half;
    let y1 = y0 + half;
    let mut bits = 0u64;
    let mut kids = kids.iter();
    bits += estimate_cu_node_bits(
        ctxs,
        kids.next().unwrap(),
        x0,
        y0,
        log2_cb_size - 1,
        ct_depth + 1,
        state,
    );
    if x1 < state.display_width {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x1,
            y0,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    if y1 < state.display_height {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x0,
            y1,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    if x1 < state.display_width && y1 < state.display_height {
        bits += estimate_cu_node_bits(
            ctxs,
            kids.next().unwrap(),
            x1,
            y1,
            log2_cb_size - 1,
            ct_depth + 1,
            state,
        );
    }
    bits
}
