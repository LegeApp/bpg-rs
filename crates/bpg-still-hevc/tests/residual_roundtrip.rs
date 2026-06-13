//! Round-trip test for `residual_coding()`: coefficient blocks written by
//! `bpg_still_hevc::residual::encode_residual` must decode back identically
//! through `bpg_hevc_decode::hevc::residual::decode_residual`, with both sides
//! using the shared 170-context CABAC state initialized from the same H.265
//! init values + slice QP. This isolates the residual-coding port (Phase 5)
//! from the CTU/CU/TU syntax layers above it.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::hevc::cabac::{
    context::NUM_CONTEXTS, CabacDecoder, ContextModel as DecCtx, INIT_VALUES,
};
use bpg_hevc_decode::hevc::residual::{decode_residual, ScanOrder as DecScan};
use bpg_still_hevc::cabac::CabacEncoder;
use bpg_still_hevc::contexts::Contexts;
use bpg_still_hevc::residual::{encode_residual, estimate_residual_bits, ScanOrder};

const SLICE_QP: i32 = 32;

fn dec_contexts(slice_qp: i32) -> Vec<DecCtx> {
    (0..NUM_CONTEXTS)
        .map(|i| {
            let mut c = DecCtx::new(0);
            c.init(INIT_VALUES[i], slice_qp);
            c
        })
        .collect()
}

fn dec_scan(order: ScanOrder) -> DecScan {
    match order {
        ScanOrder::Diagonal => DecScan::Diagonal,
        ScanOrder::Horizontal => DecScan::Horizontal,
        ScanOrder::Vertical => DecScan::Vertical,
    }
}

/// Encode `coeffs`, decode them back, assert exact equality.
fn round_trip(coeffs: &[i16], log2_size: u8, c_idx: u8, order: ScanOrder, sdh: bool) {
    let size = 1usize << log2_size;
    assert_eq!(coeffs.len(), size * size);
    assert!(coeffs.iter().any(|&c| c != 0), "need at least one nonzero");

    // --- Encode ---
    let mut w = BitWriter::new();
    let mut enc = CabacEncoder::new();
    let mut enc_ctx = Contexts::new(SLICE_QP);
    let mut est_ctx = enc_ctx.clone();
    let estimated_bits = estimate_residual_bits(&mut est_ctx, coeffs, log2_size, c_idx, order, sdh);
    encode_residual(
        &mut enc,
        &mut w,
        &mut enc_ctx,
        coeffs,
        log2_size,
        c_idx,
        order,
        sdh,
    );
    enc.encode_bin_trm(&mut w, 1);
    enc.finish(&mut w);
    w.write_bit(1);
    w.byte_align();
    let bytes = w.into_bytes();
    assert!(estimated_bits > 0, "residual estimate must be non-zero");
    for (i, (enc_model, est_model)) in enc_ctx.models.iter().zip(est_ctx.models.iter()).enumerate()
    {
        assert_eq!(
            enc_model.get_state(),
            est_model.get_state(),
            "estimated context state mismatch at context {i}"
        );
    }

    // --- Decode ---
    let mut dec = CabacDecoder::new(&bytes).expect("decoder init");
    let mut dec_ctx = dec_contexts(SLICE_QP);
    let (buf, _tskip) = decode_residual(
        &mut dec,
        &mut dec_ctx,
        log2_size,
        c_idx,
        dec_scan(order),
        sdh,
        false,
        false,
        0,
        0,
    )
    .expect("decode_residual");

    assert_eq!(
        &buf.coeffs[..size * size],
        coeffs,
        "coefficient mismatch (log2_size={log2_size}, c_idx={c_idx}, order={order:?}, sdh={sdh})"
    );
}

#[test]
fn dc_only_4x4() {
    let mut c = [0i16; 16];
    c[0] = 7;
    round_trip(&c, 2, 0, ScanOrder::Diagonal, false);
    c[0] = -7;
    round_trip(&c, 2, 0, ScanOrder::Diagonal, false);
}

#[test]
fn sparse_4x4_levels() {
    // mix of level-1, level-2, level-3, and large (EGk) coefficients
    let mut c = [0i16; 16];
    c[0] = 1;
    c[1] = -2;
    c[2] = 3;
    c[5] = -1;
    c[10] = 42; // forces coeff_abs_level_remaining EGk
    round_trip(&c, 2, 0, ScanOrder::Diagonal, false);
}

#[test]
fn scan_orders_4x4() {
    let mut c = [0i16; 16];
    c[0] = 4;
    c[3] = -5;
    c[12] = 2;
    c[15] = -1;
    round_trip(&c, 2, 0, ScanOrder::Diagonal, false);
    round_trip(&c, 2, 0, ScanOrder::Horizontal, false);
    round_trip(&c, 2, 0, ScanOrder::Vertical, false);
}

#[test]
fn chroma_4x4() {
    let mut c = [0i16; 16];
    c[0] = -3;
    c[4] = 6;
    c[9] = 1;
    round_trip(&c, 2, 1, ScanOrder::Diagonal, false);
    round_trip(&c, 2, 2, ScanOrder::Diagonal, false);
}

#[test]
fn block_8x8_multi_subblock() {
    let mut c = [0i16; 64];
    c[0] = 9;
    c[1] = -1;
    c[8] = 2;
    c[9] = -3;
    c[27] = 5; // a coefficient in a later sub-block
    c[63] = -2; // bottom-right: last sub-block
    round_trip(&c, 3, 0, ScanOrder::Diagonal, false);
}

#[test]
fn block_16x16_sparse() {
    let mut c = [0i16; 256];
    c[0] = 12;
    c[17] = -4;
    c[34] = 7;
    c[200] = -1;
    c[255] = 3;
    round_trip(&c, 4, 0, ScanOrder::Diagonal, false);
}

#[test]
fn block_32x32_sparse() {
    let mut c = [0i16; 1024];
    c[0] = 20;
    c[33] = -8;
    c[100] = 2;
    c[1023] = -5;
    round_trip(&c, 5, 0, ScanOrder::Diagonal, false);
}

#[test]
fn sign_data_hiding_parity_consistent() {
    // A 4x4 sub-block with span(last-first) > 3 so SDH triggers. The hidden
    // (lowest-scan-position) coefficient's sign must agree with the parity of
    // the sum of absolute levels: positions in diag scan, sig at n=0,1,2,9.
    // Sum of abs = 3+1+2+4 = 10 (even) -> hidden sign must be positive.
    let mut c = [0i16; 16];
    // diag scan: n0=(0,0), n1=(0,1), n2=(1,0), n9=(3,0)
    c[0 * 4 + 0] = 3; // n0 (hidden, positive since sum even)
    c[1 * 4 + 0] = -1; // n1 -> (0,1)
    c[0 * 4 + 1] = 2; // n2 -> (1,0)
    c[0 * 4 + 3] = 4; // n9 -> (3,0)
    round_trip(&c, 2, 0, ScanOrder::Diagonal, true);
}
