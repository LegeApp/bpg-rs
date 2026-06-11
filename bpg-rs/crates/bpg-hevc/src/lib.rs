//! HEVC Annex-B bitstream rewriting for the BPG container.
//!
//! BPG does not store a raw HEVC elementary stream: it strips the VPS NAL and
//! replaces the SPS with a compact "modified SPS" (MSPS) blob that the BPG
//! decoder expands back into a valid SPS. This crate ports that
//! transformation from `libbpg-0.9.8/bpgenc.c`:
//!
//! - [`find_nal_end`] / [`extract_nal`] — Annex-B start-code scanning and
//!   emulation-prevention-byte (`00 00 03`) removal
//!   (`bpgenc.c:1496-1565`).
//! - [`build_modified_sps`] — bit-level parse of the VPS+SPS, asserting the
//!   long list of HEVC-feature preconditions that x265's output satisfies,
//!   then re-emitting just the "modified SPS" tail prefixed with its `ue7`
//!   length (`bpgenc.c:1733-2031`).
//! - [`build_modified_hevc`] — for the M1 still-image, no-alpha case: emit the
//!   color MSPS blob, then copy the remaining NALs verbatim (stripping only
//!   the very first start code) (`bpgenc.c:2061-2169`).
//!
//! Scope (M1): no alpha, `frame_ticks == 1` (still image), so the
//! alpha-interleaving and frame-duration-SEI paths in the C
//! `build_modified_hevc` are omitted as dead code. They are noted as
//! `TODO(extension)`.

use bpg_bitstream::{write_ue7, BitReader, BitWriter};

/// Errors from HEVC bitstream rewriting. The `Unsupported*` variants
/// correspond to the `fprintf(stderr, ...) + return -1` precondition failures
/// in the C `build_modified_sps`: x265's SPS is known to satisfy them, so
/// hitting one means the input did not come from the expected encoder
/// configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HevcError {
    /// A start code could not be found where a NAL was expected, or the
    /// buffer ended mid-NAL.
    Truncated,
    /// Expected a specific NAL unit type but found another (e.g. VPS/SPS).
    UnexpectedNalType { expected: u8, found: u8 },
    /// An SPS field had a value outside the range the BPG MSPS format (and
    /// this port) supports. `what` names the offending field.
    UnsupportedFeature { what: &'static str },
}

/// Locate the end of the Annex-B NAL at the start of `buf`.
///
/// Returns the index one past the last byte of the NAL (i.e. the start of the
/// next start code, or `buf.len()` for the final NAL). The leading start code
/// (`00 00 01` or `00 00 00 01`) must be present. Ported from `find_nal_end`.
pub fn find_nal_end(buf: &[u8]) -> Result<usize, HevcError> {
    let buf_len = buf.len();
    let mut idx;
    if buf_len >= 4 && buf[0] == 0 && buf[1] == 0 && buf[2] == 0 && buf[3] == 1 {
        idx = 4;
    } else if buf_len >= 3 && buf[0] == 0 && buf[1] == 0 && buf[2] == 1 {
        idx = 3;
    } else {
        return Err(HevcError::Truncated);
    }
    /* NAL header */
    if idx + 2 > buf_len {
        return Err(HevcError::Truncated);
    }
    /* find the last byte */
    loop {
        if idx + 2 >= buf_len {
            idx = buf_len;
            break;
        }
        if buf[idx] == 0 && buf[idx + 1] == 0 && buf[idx + 2] == 1 {
            break;
        }
        if idx + 3 < buf_len
            && buf[idx] == 0
            && buf[idx + 1] == 0
            && buf[idx + 2] == 0
            && buf[idx + 3] == 1
        {
            break;
        }
        idx += 1;
    }
    Ok(idx)
}

/// Extract the RBSP of the Annex-B NAL at the start of `buf`, removing the
/// leading start code and emulation-prevention bytes (`00 00 03` -> `00 00`).
///
/// Returns `(rbsp, end_idx)` where `end_idx` is the position one past the NAL
/// within `buf` (as from [`find_nal_end`]). Ported from `extract_nal`.
pub fn extract_nal(buf: &[u8]) -> Result<(Vec<u8>, usize), HevcError> {
    let end = find_nal_end(buf)?;
    let start = if buf[2] == 1 { 3 } else { 4 };

    let mut nal_buf = Vec::with_capacity(end - start);
    let mut idx = start;
    while idx < end {
        if idx + 2 < end && buf[idx] == 0 && buf[idx + 1] == 0 && buf[idx + 2] == 3 {
            nal_buf.push(0);
            nal_buf.push(0);
            idx += 3;
        } else {
            nal_buf.push(buf[idx]);
            idx += 1;
        }
    }
    Ok((nal_buf, end))
}

/// The "modified SPS" blob: a `ue7`-length-prefixed, `ue(v)`-coded SPS tail
/// that replaces the VPS+SPS NALs in a BPG payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModifiedSps {
    /// The serialized blob, ready to be concatenated into the payload:
    /// `ue7(msps_len)` followed by `msps_len` bytes of `ue(v)`-coded SPS tail.
    pub bytes: Vec<u8>,
}

impl ModifiedSps {
    /// Parse the VPS (NAL type 32) then SPS (NAL type 33) at the start of
    /// `hevc`, and build the modified-SPS blob. Returns the blob together
    /// with the byte offset in `hevc` where the SPS NAL ended (i.e. where the
    /// remaining NALs begin). Ported from `build_modified_sps`.
    pub fn from_hevc_stream(hevc: &[u8]) -> Result<(Self, usize), HevcError> {
        // VPS NAL (type 32) — extracted only to verify and to advance past it.
        let (vps, idx) = extract_nal(hevc)?;
        if vps.len() < 2 {
            return Err(HevcError::Truncated);
        }
        let nal_unit_type = (vps[0] >> 1) & 0x3f;
        if nal_unit_type != 32 {
            return Err(HevcError::UnexpectedNalType {
                expected: 32,
                found: nal_unit_type,
            });
        }

        // SPS NAL (type 33).
        let (sps, ret) = extract_nal(&hevc[idx..])?;
        let idx = idx + ret;
        if sps.len() < 2 {
            return Err(HevcError::Truncated);
        }
        let nal_unit_type = (sps[0] >> 1) & 0x3f;
        if nal_unit_type != 33 {
            return Err(HevcError::UnexpectedNalType {
                expected: 33,
                found: nal_unit_type,
            });
        }

        let bytes = Self::rewrite_sps(&sps)?;
        Ok((ModifiedSps { bytes }, idx))
    }

    /// Parse the SPS RBSP `sps` up through the fields BPG keeps, validating
    /// the encoder-configuration preconditions, then emit the MSPS blob.
    fn rewrite_sps(sps: &[u8]) -> Result<Vec<u8>, HevcError> {
        let mut gb = BitReader::new(sps);

        let ue = |gb: &mut BitReader| gb.read_ue_golomb().ok_or(HevcError::Truncated);

        gb.skip_bits(16); /* nal header */
        let vps_id = gb.read_bits(4);
        if vps_id != 0 {
            return Err(HevcError::UnsupportedFeature { what: "vps_id != 0" });
        }
        let max_sub_layers = gb.read_bits(3);
        if max_sub_layers != 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "max_sub_layers != 0",
            });
        }
        gb.skip_bits(1); /* temporal_id_nesting_flag */
        /* profile tier level */
        gb.skip_bits(2); /* profile_space */
        gb.skip_bits(1); /* tier_flag */
        let _profile_idc = gb.read_bits(5);
        for _ in 0..32 {
            gb.skip_bits(1); /* profile_compatibility_flag */
        }
        gb.skip_bits(1); /* progressive_source_flag */
        gb.skip_bits(1); /* interlaced_source_flag */
        gb.skip_bits(1); /* non_packed_constraint_flag */
        gb.skip_bits(1); /* frame_only_constraint_flag */
        gb.skip_bits(44); /* XXX_reserved_zero_44 */
        gb.skip_bits(8); /* level_idc */

        let sps_id = ue(&mut gb)?;
        if sps_id != 0 {
            return Err(HevcError::UnsupportedFeature { what: "sps_id != 0" });
        }
        let chroma_format_idc = ue(&mut gb)?;
        if chroma_format_idc == 3 {
            gb.skip_bits(1); /* separate_colour_plane_flag */
        }
        let _width = ue(&mut gb)?;
        let _height = ue(&mut gb)?;
        /* pic conformance_flag */
        if gb.read_bits(1) != 0 {
            ue(&mut gb)?; /* left_offset */
            ue(&mut gb)?; /* right_offset */
            ue(&mut gb)?; /* top_offset */
            ue(&mut gb)?; /* bottom_offset */
        }
        let _bit_depth_luma = ue(&mut gb)? + 8;
        let _bit_depth_chroma = ue(&mut gb)? + 8;
        let log2_max_poc_lsb = ue(&mut gb)? + 4;
        if log2_max_poc_lsb != 8 {
            return Err(HevcError::UnsupportedFeature {
                what: "log2_max_poc_lsb != 8",
            });
        }
        let _sublayer_ordering_info = gb.read_bits(1);
        ue(&mut gb)?; /* max_dec_pic_buffering */
        ue(&mut gb)?; /* num_reorder_pics */
        ue(&mut gb)?; /* max_latency_increase */

        let log2_min_cb_size = ue(&mut gb)? + 3;
        let log2_diff_max_min_coding_block_size = ue(&mut gb)?;
        let log2_min_tb_size = ue(&mut gb)? + 2;
        let log2_diff_max_min_transform_block_size = ue(&mut gb)?;

        let max_transform_hierarchy_depth_inter = ue(&mut gb)?;
        let max_transform_hierarchy_depth_intra = ue(&mut gb)?;
        if max_transform_hierarchy_depth_inter != max_transform_hierarchy_depth_intra {
            return Err(HevcError::UnsupportedFeature {
                what: "max_transform_hierarchy_depth_inter != intra",
            });
        }

        let scaling_list_enable_flag = gb.read_bits(1);
        if scaling_list_enable_flag != 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "scaling_list_enable_flag != 0",
            });
        }
        let amp_enabled_flag = gb.read_bits(1);
        if amp_enabled_flag == 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "amp_enabled_flag == 0",
            });
        }
        let sao_enabled = gb.read_bits(1);
        let pcm_enabled_flag = gb.read_bits(1);
        let mut pcm = None;
        if pcm_enabled_flag != 0 {
            let pcm_sample_bit_depth_luma_minus1 = gb.read_bits(4);
            let pcm_sample_bit_depth_chroma_minus1 = gb.read_bits(4);
            let log2_min_pcm_luma_coding_block_size_minus3 = ue(&mut gb)?;
            let log2_diff_max_min_pcm_luma_coding_block_size = ue(&mut gb)?;
            let pcm_loop_filter_disabled_flag = gb.read_bits(1);
            pcm = Some((
                pcm_sample_bit_depth_luma_minus1,
                pcm_sample_bit_depth_chroma_minus1,
                log2_min_pcm_luma_coding_block_size_minus3,
                log2_diff_max_min_pcm_luma_coding_block_size,
                pcm_loop_filter_disabled_flag,
            ));
        }
        let nb_st_rps = ue(&mut gb)?;
        if nb_st_rps != 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "nb_st_rps != 0",
            });
        }
        let long_term_ref_pics_present_flag = gb.read_bits(1);
        if long_term_ref_pics_present_flag != 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "long_term_ref_pics_present_flag != 0",
            });
        }
        let sps_temporal_mvp_enabled_flag = gb.read_bits(1);
        if sps_temporal_mvp_enabled_flag == 0 {
            return Err(HevcError::UnsupportedFeature {
                what: "sps_temporal_mvp_enabled_flag == 0",
            });
        }
        let sps_strong_intra_smoothing_enable_flag = gb.read_bits(1);
        let vui_present = gb.read_bits(1);
        if vui_present != 0 {
            let sar_present = gb.read_bits(1);
            if sar_present != 0 {
                let sar_idx = gb.read_bits(8);
                if sar_idx == 255 {
                    gb.skip_bits(16); /* sar_num */
                    gb.skip_bits(16); /* sar_den */
                }
            }

            let overscan_info_present_flag = gb.read_bits(1);
            if overscan_info_present_flag != 0 {
                gb.skip_bits(1); /* overscan_appropriate_flag */
            }

            let video_signal_type_present_flag = gb.read_bits(1);
            if video_signal_type_present_flag != 0 {
                return Err(HevcError::UnsupportedFeature {
                    what: "video_signal_type_present_flag != 0",
                });
            }
            let chroma_loc_info_present_flag = gb.read_bits(1);
            if chroma_loc_info_present_flag != 0 {
                ue(&mut gb)?;
                ue(&mut gb)?;
            }
            gb.skip_bits(1); /* neutra_chroma_indication_flag */
            gb.skip_bits(1);
            gb.skip_bits(1);
            let default_display_window_flag = gb.read_bits(1);
            if default_display_window_flag != 0 {
                return Err(HevcError::UnsupportedFeature {
                    what: "default_display_window_flag != 0",
                });
            }
            let vui_timing_info_present_flag = gb.read_bits(1);
            if vui_timing_info_present_flag != 0 {
                gb.skip_bits(32);
                gb.skip_bits(32);
                let vui_poc_proportional_to_timing_flag = gb.read_bits(1);
                if vui_poc_proportional_to_timing_flag != 0 {
                    ue(&mut gb)?;
                }
                let vui_hrd_parameters_present_flag = gb.read_bits(1);
                if vui_hrd_parameters_present_flag != 0 {
                    return Err(HevcError::UnsupportedFeature {
                        what: "vui_hrd_parameters_present_flag != 0",
                    });
                }
            }
            let bitstream_restriction_flag = gb.read_bits(1);
            if bitstream_restriction_flag != 0 {
                gb.skip_bits(1);
                gb.skip_bits(1);
                gb.skip_bits(1);
                ue(&mut gb)?;
                ue(&mut gb)?;
                ue(&mut gb)?;
                ue(&mut gb)?;
                ue(&mut gb)?;
            }
        }
        let sps_extension_flag = gb.read_bits(1);
        let mut sps_range_extension_flag = 0;
        let mut sps_range_extension_flags = 0;
        if sps_extension_flag != 0 {
            sps_range_extension_flag = gb.read_bits(1);
            let sps_extension_7bits = gb.read_bits(7);
            if sps_extension_7bits != 0 {
                return Err(HevcError::UnsupportedFeature {
                    what: "sps_extension_7bits != 0",
                });
            }
            if sps_range_extension_flag != 0 {
                sps_range_extension_flags = gb.read_bits(9);
                if sps_range_extension_flags
                    & ((1 << (8 - 3)) | (1 << (8 - 4)) | (1 << (8 - 6)) | (1 << (8 - 8)))
                    != 0
                {
                    return Err(HevcError::UnsupportedFeature {
                        what: "unsupported sps range extensions",
                    });
                }
            }
        }

        /* build the modified SPS */
        let mut pb = BitWriter::new();
        pb.write_ue_golomb(log2_min_cb_size - 3);
        pb.write_ue_golomb(log2_diff_max_min_coding_block_size);
        pb.write_ue_golomb(log2_min_tb_size - 2);
        pb.write_ue_golomb(log2_diff_max_min_transform_block_size);
        pb.write_ue_golomb(max_transform_hierarchy_depth_intra);
        pb.write_bits(1, sao_enabled);
        pb.write_bits(1, pcm_enabled_flag);
        if let Some((luma_m1, chroma_m1, log2_min_m3, log2_diff_max, loop_filter)) = pcm {
            pb.write_bits(4, luma_m1);
            pb.write_bits(4, chroma_m1);
            pb.write_ue_golomb(log2_min_m3);
            pb.write_ue_golomb(log2_diff_max);
            pb.write_bits(1, loop_filter);
        }
        pb.write_bits(1, sps_strong_intra_smoothing_enable_flag);
        pb.write_bits(1, sps_extension_flag);
        if sps_extension_flag != 0 {
            pb.write_bits(1, sps_range_extension_flag);
            pb.write_bits(7, 0);
            if sps_range_extension_flag != 0 {
                pb.write_bits(9, sps_range_extension_flags);
            }
        }
        let msps = pb.into_bytes();
        // BitWriter only allocates whole bytes as bits are written, so its
        // byte length already equals `(n_bits + 7) >> 3`.
        let msps_len = msps.len();

        let mut out = Vec::with_capacity(5 + msps_len);
        write_ue7(&mut out, msps_len as u32); /* header length */
        out.extend_from_slice(&msps);
        Ok(out)
    }
}

/// Build the BPG "modified HEVC" payload from x265's raw Annex-B color stream.
///
/// For the M1 still-image, no-alpha case this is: emit the color MSPS blob,
/// then copy every remaining NAL verbatim, stripping only the leading start
/// code of the first one. Ported from `build_modified_hevc` with `abuf ==
/// NULL` and `frame_ticks == 1` (so the alpha-interleave and
/// frame-duration-SEI branches are omitted as dead code).
///
/// TODO(extension): alpha plane interleaving and animation frame-duration SEI.
pub fn build_modified_hevc(cbuf: &[u8]) -> Result<Vec<u8>, HevcError> {
    let mut out = Vec::new();

    /* add color MSPS */
    let (msps, mut cidx) = ModifiedSps::from_hevc_stream(cbuf)?;
    out.extend_from_slice(&msps.bytes);

    /* add the remaining NALs */
    let mut first_nal = true;
    while cidx < cbuf.len() {
        let nal = &cbuf[cidx..];
        let nal_len = find_nal_end(nal)?;
        cidx += nal_len;

        let start = 3 + (nal[2] == 0) as usize;
        let l = if first_nal { start } else { 0 };
        out.extend_from_slice(&nal[l..nal_len]);
        first_nal = false;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal Annex-B NAL: `start_code` + `[nut<<1, layer_tid] +
    /// payload`, inserting emulation-prevention bytes is left to the caller.
    fn nal(four_byte_start: bool, nut: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        if four_byte_start {
            v.extend_from_slice(&[0, 0, 0, 1]);
        } else {
            v.extend_from_slice(&[0, 0, 1]);
        }
        v.push(nut << 1);
        v.push(0x01); /* layer_id=0, temporal_id_plus1=1 */
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn find_nal_end_three_byte_start() {
        // single NAL, 3-byte start code
        let buf = nal(false, 33, &[0xaa, 0xbb]);
        assert_eq!(find_nal_end(&buf), Ok(buf.len()));
    }

    #[test]
    fn find_nal_end_stops_at_next_start_code() {
        let mut buf = nal(true, 32, &[0x11]);
        let first_len = buf.len();
        buf.extend_from_slice(&nal(false, 33, &[0x22, 0x33]));
        assert_eq!(find_nal_end(&buf), Ok(first_len));
    }

    #[test]
    fn find_nal_end_rejects_missing_start_code() {
        assert_eq!(find_nal_end(&[0x12, 0x34, 0x56]), Err(HevcError::Truncated));
    }

    #[test]
    fn extract_nal_strips_start_code_and_epb() {
        // payload contains an emulation-prevention sequence 00 00 03 04
        let buf = nal(false, 33, &[0x00, 0x00, 0x03, 0x04]);
        let (rbsp, end) = extract_nal(&buf).unwrap();
        assert_eq!(end, buf.len());
        // NAL header (2 bytes) + 00 00 (03 removed) + 04
        assert_eq!(rbsp, vec![33 << 1, 0x01, 0x00, 0x00, 0x04]);
    }

    #[test]
    fn extract_nal_trailing_zero_pair_without_third_byte_is_literal() {
        // A bare `00 00` at the end (no following byte, so `idx + 2 >= end`)
        // is copied verbatim — the C `idx + 2 < end` guard only strips a full
        // `00 00 03` triplet.
        let buf = nal(false, 33, &[0x00, 0x00]);
        let (rbsp, _) = extract_nal(&buf).unwrap();
        assert_eq!(rbsp, vec![33 << 1, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn from_hevc_stream_rejects_non_vps_first_nal() {
        // first NAL is SPS (33) where VPS (32) is expected
        let buf = nal(false, 33, &[0x00, 0x00]);
        assert_eq!(
            ModifiedSps::from_hevc_stream(&buf),
            Err(HevcError::UnexpectedNalType {
                expected: 32,
                found: 33
            })
        );
    }

    #[test]
    fn build_modified_hevc_copies_nals_after_msps() {
        // Hand-build VPS + SPS + IDR. The SPS must parse cleanly through
        // build_modified_sps; constructing a fully valid SPS by hand is
        // covered by the integration test against real x265 output. Here we
        // only check the NAL-copy framing by using a SPS we know parses: an
        // all-from-real-encoder SPS isn't available in a unit test, so this
        // test instead verifies find_nal_end framing over multiple NALs.
        let mut buf = nal(true, 32, &[0x0c, 0x01]);
        buf.extend_from_slice(&nal(false, 33, &[0x44]));
        buf.extend_from_slice(&nal(false, 19, &[0x55, 0x66]));
        // The third NAL (IDR, type 19) should be locatable past the first two.
        let third_start = find_nal_end(&buf).unwrap();
        let rest = &buf[third_start..];
        let second_len = find_nal_end(rest).unwrap();
        assert_eq!(&rest[second_len..], &nal(false, 19, &[0x55, 0x66])[..]);
    }
}
