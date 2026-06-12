//! Scratch test: dump VPS/SPS/PPS field values from a real x265-produced
//! oracle stream, to anchor the Phase 3 writers' constant choices.
//! Not a real assertion test -- prints to stdout with `--nocapture`.

use bpg_hevc_decode::hevc::bitstream::{parse_nal_units, NalType};
use bpg_hevc_decode::hevc::params::{parse_pps, parse_sps, parse_vps};

#[test]
fn dump_checkerboard_420_qp24() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../oracle/out/checkerboard__420_8bit_qp24.hevc"
    );
    let data = std::fs::read(path).expect("read oracle hevc");
    let nals = parse_nal_units(&data).expect("parse nals");
    for nal in &nals {
        match nal.nal_type {
            NalType::VpsNut => {
                let vps = parse_vps(&nal.payload).expect("parse vps");
                println!("VPS: {vps:#?}");
            }
            NalType::SpsNut => {
                let sps = parse_sps(&nal.payload).expect("parse sps");
                println!("SPS: {sps:#?}");
            }
            NalType::PpsNut => {
                let pps = parse_pps(&nal.payload).expect("parse pps");
                println!("PPS: {pps:#?}");
            }
            _ => println!("NAL type {:?}, {} bytes", nal.nal_type, nal.payload.len()),
        }
    }
}
