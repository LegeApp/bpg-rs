//! Standalone round-trip test for the CABAC engine: bins written by
//! `still265::cabac::CabacEncoder` must be read back identically by
//! `bpg_hevc_decode::hevc::cabac::CabacDecoder`, with both sides' context
//! models tracking the same (state, mps) after every bin. This isolates the
//! arithmetic-coder/state-table port from residual-coding syntax (Phase 5),
//! per `FULL_RUST_CODEC_PLAN.md` Phase 3.

use bpg_bitstream::BitWriter;
use bpg_hevc_decode::hevc::cabac::{CabacDecoder, ContextModel as DecCtx};
use still265::cabac::{CabacEncoder, CabacEstimator, ContextModel as EncCtx};

const INIT_VALUE: u8 = 154; // SAO_MERGE_FLAG init value, H.265 Table 9-31
const SLICE_QP: i32 = 28;

#[test]
fn regular_and_bypass_bins_round_trip() {
    let regular_bins = [1u8, 0, 1, 1, 0, 0, 1, 1];
    let bypass_bins = [1u8, 0, 1, 0, 1, 1, 0];

    // --- Encode ---
    let mut w = BitWriter::new();
    let mut enc = CabacEncoder::new();
    let mut enc_ctx = EncCtx::new(INIT_VALUE, SLICE_QP);

    // Decoder-side context, initialized identically, checked against the
    // encoder's context *before* either side processes any bins.
    let mut dec_ctx = DecCtx::new(0);
    dec_ctx.init(INIT_VALUE, SLICE_QP);
    assert_eq!(
        enc_ctx.get_state(),
        dec_ctx.get_state(),
        "context models must start in the same (state, mps)"
    );

    // Encoder context state after each regular bin, for per-step comparison
    // against the decoder below.
    let mut enc_states_after = Vec::with_capacity(regular_bins.len());
    for &bin in &regular_bins {
        enc.encode_bin(&mut w, bin, &mut enc_ctx);
        enc_states_after.push(enc_ctx.get_state());
    }
    for &bin in &bypass_bins {
        enc.encode_bin_ep(&mut w, bin);
    }
    enc.encode_bin_trm(&mut w, 1); // end_of_slice_segment_flag == 1
    enc.finish(&mut w);
    w.write_bit(1); // rbsp_trailing_bits: rbsp_stop_one_bit
    w.byte_align(); // rbsp_alignment_zero_bit*

    let bytes = w.into_bytes();
    assert!(bytes.len() >= 2, "CabacDecoder::new requires >= 2 bytes");

    // --- Decode ---
    let mut dec = CabacDecoder::new(&bytes).expect("decoder init");

    for (i, &bin) in regular_bins.iter().enumerate() {
        let decoded = dec.decode_bin(&mut dec_ctx).expect("decode_bin");
        assert_eq!(decoded, bin, "regular bin {i} mismatch");
        assert_eq!(
            enc_states_after[i],
            dec_ctx.get_state(),
            "context state diverged after regular bin {i}"
        );
    }

    for (i, &bin) in bypass_bins.iter().enumerate() {
        let decoded = dec.decode_bypass().expect("decode_bypass");
        assert_eq!(decoded, bin, "bypass bin {i} mismatch");
    }

    let end_of_slice = dec.decode_terminate().expect("decode_terminate");
    assert_eq!(
        end_of_slice, 1,
        "encode_bin_trm(1) must decode as terminate"
    );
}

#[test]
fn cabac_estimator_matches_x265_entropy_bits_and_context_updates() {
    let bins = [1u8, 0, 1, 1, 0, 0, 1, 1];
    let expected_costs = [
        0x07b23u64, 0x08cbcu64, 0x07b23u64, 0x074a0u64, 0x09354u64, 0x08cbcu64, 0x07b23u64,
        0x074a0u64,
    ];

    let mut est = CabacEstimator::new();
    let mut est_ctx = EncCtx::new(INIT_VALUE, SLICE_QP);
    let mut writer_ctx = EncCtx::new(INIT_VALUE, SLICE_QP);
    let mut expected_total = 0u64;

    for (i, (&bin, &cost)) in bins.iter().zip(expected_costs.iter()).enumerate() {
        expected_total += cost;
        est.encode_bin(bin, &mut est_ctx);

        let mut w = BitWriter::new();
        let mut enc = CabacEncoder::new();
        enc.encode_bin(&mut w, bin, &mut writer_ctx);

        assert_eq!(
            est.frac_bits(),
            expected_total,
            "estimated cost mismatch after bin {i}"
        );
        assert_eq!(
            est_ctx.get_state(),
            writer_ctx.get_state(),
            "estimated context state mismatch after bin {i}"
        );
    }

    est.encode_bin_ep(0);
    est.encode_bins_ep(0b101, 3);
    assert_eq!(
        est.frac_bits(),
        expected_total + CabacEstimator::SCALE * 4,
        "bypass bins must cost exactly one fixed-point bit each"
    );
}
