//! Phase 3 acceptance test: `StillHevcEncoder::syntax_skeleton` must emit a
//! VPS/SPS/PPS + slice segment header that `bpg-hevc-decode`'s parameter-set
//! and slice-header parsers accept. The skeleton emits no slice data, so
//! full CTU/picture decode is out of scope here.

use bpg_hevc_decode::hevc::bitstream::{NalType, parse_nal_units};
use bpg_hevc_decode::hevc::params::{parse_pps, parse_sps, parse_vps};
use bpg_hevc_decode::hevc::slice::SliceHeader;
use still265::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig, StillHevcEncoder};

#[test]
fn syntax_skeleton_headers_parse() {
    let config = StillHevcConfig {
        width: 64,
        height: 64,
        bit_depth: 8,
        chroma: ChromaFormat::Yuv420,
        qp: 28,
        effort: Effort::Slow,
        sao: SaoMode::Off,
        deblock: DeblockMode::Off,
        adaptive_qp: false,
        aq_mode: still265::AqMode::Off,
        aq_strength: 0.35,
        aq_clamp: 2,
        two_pass_gate: true,
    };

    let bytes = StillHevcEncoder::syntax_skeleton(&config);
    let nals = parse_nal_units(&bytes).expect("parse NAL units");

    let vps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::VpsNut)
        .expect("VPS NAL present");
    parse_vps(&vps_nal.payload).expect("VPS parses");

    let sps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::SpsNut)
        .expect("SPS NAL present");
    let sps = parse_sps(&sps_nal.payload).expect("SPS parses");
    assert_eq!(sps.pic_width_in_luma_samples, config.width);
    assert_eq!(sps.pic_height_in_luma_samples, config.height);
    assert_eq!(sps.chroma_format_idc, 1);
    assert!(!sps.sample_adaptive_offset_enabled_flag);

    let pps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::PpsNut)
        .expect("PPS NAL present");
    let pps = parse_pps(&pps_nal.payload).expect("PPS parses");

    let slice_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::IdrWRadl)
        .expect("IDR slice NAL present");
    let result = SliceHeader::parse(slice_nal, &sps, &pps).expect("slice header parses");
    assert!(result.header.slice_type.is_intra());
    assert_eq!(result.header.slice_qp_y, config.qp as i32);
    // The skeleton emits no slice data, so the header should consume the
    // whole (tiny) payload.
    assert_eq!(result.data_offset, slice_nal.payload.len());
}

/// The other half of the Phase 3 acceptance criterion: feeding the skeleton
/// to the full decoder must fail *because slice data is missing*, not
/// because any header is malformed. Parameter-set and slice-header parsing
/// (exercised directly above) must therefore succeed first, and only the
/// CABAC slice-data decode — which needs at least 2 bytes and gets zero —
/// should report the error.
#[test]
fn syntax_skeleton_decode_fails_on_missing_slice_data() {
    let config = StillHevcConfig {
        width: 64,
        height: 64,
        bit_depth: 8,
        chroma: ChromaFormat::Yuv420,
        qp: 28,
        effort: Effort::Slow,
        sao: SaoMode::Off,
        deblock: DeblockMode::Off,
        adaptive_qp: false,
        aq_mode: still265::AqMode::Off,
        aq_strength: 0.35,
        aq_clamp: 2,
        two_pass_gate: true,
    };

    let bytes = StillHevcEncoder::syntax_skeleton(&config);

    let err = bpg_hevc_decode::hevc::decode(&bytes).expect_err("decode must fail: no slice data");
    match err {
        bpg_hevc_decode::HevcError::CabacError(msg) => {
            assert!(
                msg.contains("short"),
                "expected a CABAC 'data too short' error, got: {msg}"
            );
        }
        other => panic!("expected CabacError (missing slice data), got: {other:?}"),
    }
}
