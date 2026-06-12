//! Annex-B NAL unit writer: start code + 2-byte NAL header + emulation
//! prevention.
//!
//! Ported from `make_nal` in `crates/bpg-hevc/src/lib.rs` (itself ported from
//! the NAL-assembly half of `build_msps`/`hevc_decode_frame_internal` in
//! `libbpg-0.9.8/libbpg.c`), generalized to take any `nal_unit_type` and
//! always emit a 4-byte start code.

/// HEVC `nal_unit_type` values used by the Phase 3 skeleton (H.265 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalType {
    /// IDR picture, may have RADL pictures present.
    IdrWRadl,
    Vps,
    Sps,
    Pps,
}

impl NalType {
    fn nal_unit_type(self) -> u8 {
        match self {
            NalType::IdrWRadl => 19,
            NalType::Vps => 32,
            NalType::Sps => 33,
            NalType::Pps => 34,
        }
    }
}

/// Append `00 00 00 01` + a 2-byte NAL header + `rbsp` (with emulation
/// prevention `00 00 03` inserted before any `00 00 0{0,1,2,3}` byte) to
/// `out`.
pub fn write_annexb_nal(out: &mut Vec<u8>, nal_type: NalType, rbsp: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);

    let nuh_layer_id = 0u8;
    let nuh_temporal_id_plus1 = 1u8;
    let header = [
        (nal_type.nal_unit_type() << 1) | (nuh_layer_id >> 5),
        (nuh_layer_id << 5) | nuh_temporal_id_plus1,
    ];

    let mut zeros = 0usize;
    for &b in header.iter().chain(rbsp.iter()) {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bytes_for_sps() {
        let mut out = Vec::new();
        write_annexb_nal(&mut out, NalType::Sps, &[]);
        // start code + nal_unit_header (nut=33 -> 0b0100_0010, 0b0000_0001)
        assert_eq!(out, vec![0, 0, 0, 1, 33 << 1, 1]);
    }

    #[test]
    fn emulation_prevention_inserted() {
        let mut out = Vec::new();
        // RBSP containing 00 00 00, 00 00 01, 00 00 02, 00 00 03; a 0x03
        // must be inserted before any byte <= 0x03 that follows two
        // consecutive zero bytes (the zero run carries across groups).
        write_annexb_nal(
            &mut out,
            NalType::Sps,
            &[0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03],
        );
        // Skip start code (4) + header (2).
        let rbsp_out = &out[6..];
        assert_eq!(
            rbsp_out,
            &[
                0x00, 0x00, 0x03, 0x00, // 00 00 00 -> 00 00 [03] 00 (zeros now 1)
                0x00, 0x03, 0x00, 0x01, // 00 00 01 -> 00 [03] 00 01 (carried zero run)
                0x00, 0x00, 0x03, 0x02, // 00 00 02 -> 00 00 [03] 02
                0x00, 0x00, 0x03, 0x03, // 00 00 03 -> 00 00 [03] 03
            ]
        );
    }

    #[test]
    fn no_false_positive_epb() {
        let mut out = Vec::new();
        // 00 00 04 must NOT get an inserted 0x03 (only <= 0x03 triggers it).
        write_annexb_nal(&mut out, NalType::Sps, &[0x00, 0x00, 0x04]);
        let rbsp_out = &out[6..];
        assert_eq!(rbsp_out, &[0x00, 0x00, 0x04]);
    }
}
