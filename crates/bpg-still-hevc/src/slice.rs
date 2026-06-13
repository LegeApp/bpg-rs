//! Slice segment header writer for a single first-slice IDR I-slice.
//!
//! Field order/conditions mirror `SliceHeader::parse` in
//! `bpg-hevc-decode::hevc::slice`, specialized to the parameter sets written
//! by `params.rs`:
//! - `first_slice_segment_in_pic_flag = 1` (single slice per picture), so
//!   `dependent_slice_segment_flag`/`slice_segment_address` are absent.
//! - `pps.num_extra_slice_header_bits == 0`, `pps.output_flag_present_flag
//!   == false`, `sps.separate_colour_plane_flag == false`.
//! - NAL type is IDR (`IdrWRadl`/`IdrNLp`), so POC LSB and ref-pic-set
//!   syntax are absent.
//! - `sps.sample_adaptive_offset_enabled_flag == false`, so
//!   `slice_sao_luma_flag`/`slice_sao_chroma_flag` are absent.
//! - `pps.pps_slice_chroma_qp_offsets_present_flag == false` and
//!   `pps.deblocking_filter_override_enabled_flag == false`, so the chroma
//!   QP offset and deblocking-override fields are absent.
//! - `pps.pps_loop_filter_across_slices_enabled_flag == true` and
//!   `slice_deblocking_filter_disabled_flag == false` (inherited from
//!   `pps_deblocking_filter_disabled_flag == false`), so
//!   `slice_loop_filter_across_slices_enabled_flag` IS present.
//! - `pps.tiles_enabled_flag == false` and
//!   `pps.entropy_coding_sync_enabled_flag == false`, so
//!   `num_entry_point_offsets` is absent (implicitly 0).
//! - `pps.slice_segment_header_extension_present_flag == false`.
//!
//! After these fields comes `byte_alignment()`: `alignment_bit_equal_to_one`
//! (1) followed by zero `alignment_bit_equal_to_zero`s up to the next byte
//! boundary, after which `slice_data()` (the CABAC-coded CTU data) begins.

use bpg_bitstream::BitWriter;

use crate::StillHevcConfig;

/// `slice_segment_header()` (H.265 7.3.6.1) for a single first-slice IDR
/// I-slice, ending with `byte_alignment()`. The returned bytes are the
/// header only; CABAC-coded slice data (not yet implemented) follows.
pub fn write_slice_segment_header(config: &StillHevcConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bit(1); // first_slice_segment_in_pic_flag
    w.write_bit(0); // no_output_of_prior_pics_flag (IRAP)
    w.write_ue_golomb(0); // slice_pic_parameter_set_id

    w.write_ue_golomb(2); // slice_type (2 == I)

    // slice_qp_delta: SliceQPY = 26 + pps.init_qp_minus26 (0) + slice_qp_delta
    w.write_se_golomb(config.qp as i32 - 26);

    // With SAO off and deblocking disabled in the PPS,
    // slice_loop_filter_across_slices_enabled_flag is absent (its presence
    // condition `slice_sao_* || !slice_deblocking_filter_disabled` is false).

    // byte_alignment()
    w.write_bit(1); // alignment_bit_equal_to_one
    w.byte_align(); // alignment_bit_equal_to_zero*

    w.into_bytes()
}
