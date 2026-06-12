//! VPS/SPS/PPS (parameter set) writers for the Phase 3 syntax skeleton.
//!
//! Field order and presence conditions are the mirror image of
//! `parse_vps`/`parse_sps`/`parse_pps`/`parse_profile_tier_level` in
//! `bpg-hevc-decode::hevc::params`. Constant choices (profile/level, AMP,
//! VUI colour info, etc.) are anchored to a real x265 BPG-still-image
//! encode, dumped via `tests/dump_oracle_params.rs` from
//! `oracle/out/checkerboard__420_8bit_qp24.hevc`.
//!
//! Phase 3 simplifications relative to that oracle stream (each chosen so
//! the slice-segment-header writer in `slice.rs` doesn't need to handle the
//! corresponding optional fields):
//! - `sample_adaptive_offset_enabled_flag = false` (oracle: true) — matches
//!   `StillHevcConfig::sao` defaulting to `SaoMode::Off`; avoids
//!   `slice_sao_luma_flag`/`slice_sao_chroma_flag` in the slice header.
//! - `entropy_coding_sync_enabled_flag = false` (oracle: true) — avoids
//!   `num_entry_point_offsets` parsing in the slice header.
//! - `deblocking_filter_control_present_flag = false` — avoids
//!   `deblocking_filter_override_flag` and friends in the slice header.
//!
//! The SPS VUI is written out to the end of `vui_parameters()` (H.265
//! E.2.1), not just the fields `parse_sps` reads — `parse_sps` stops after
//! `matrix_coeffs`, but a conformant decoder (stock `bpgdec`/libde265, used
//! for Phase 5 verification) keeps parsing, and would otherwise misread
//! `rbsp_trailing_bits()` as further VUI flags.

use bpg_bitstream::BitWriter;

use crate::{ChromaFormat, StillHevcConfig};

/// `rbsp_trailing_bits()`: `rbsp_stop_one_bit` (1) followed by zero
/// `rbsp_alignment_zero_bit`s up to the next byte boundary.
fn write_rbsp_trailing_bits(w: &mut BitWriter) {
    w.write_bit(1);
    w.byte_align();
}

/// `profile_tier_level(profilePresentFlag=1, maxNumSubLayersMinus1=0)`
/// (H.265 7.3.3), with the general profile/level constants from the oracle
/// dump (Main profile, level 4.0 == `general_level_idc=120`).
///
/// `max_sub_layers_minus1` must be 0: the per-sub-layer profile/level loop
/// and the `reserved_zero_2bits` padding loop (present only when
/// `max_sub_layers_minus1 > 0`) are not implemented.
fn write_profile_tier_level(w: &mut BitWriter, max_sub_layers_minus1: u8) {
    assert_eq!(
        max_sub_layers_minus1, 0,
        "write_profile_tier_level: only max_sub_layers_minus1 == 0 is implemented"
    );

    w.write_bits(2, 0); // general_profile_space
    w.write_bit(0); // general_tier_flag
    w.write_bits(5, 1); // general_profile_idc (Main)

    for _ in 0..32 {
        w.write_bit(0); // general_profile_compatibility_flag[i]
    }

    w.write_bit(1); // general_progressive_source_flag
    w.write_bit(0); // general_interlaced_source_flag
    w.write_bit(0); // general_non_packed_constraint_flag
    w.write_bit(1); // general_frame_only_constraint_flag

    // 44 reserved bits (general_reserved_zero_44bits, here split as the
    // decoder reads it: 32 bits then 12 bits).
    w.write_bits(32, 0);
    w.write_bits(12, 0);

    w.write_bits(8, 120); // general_level_idc
                          // max_sub_layers_minus1 == 0: no sub-layer profile/level info, and no
                          // reserved_zero_2bits padding loop.
}

/// `video_parameter_set_rbsp()` (H.265 7.3.2.1), trimmed to the fields
/// `parse_vps` reads (id/layer/sub-layer counts, `vps_reserved_0xffff_16bits`,
/// `profile_tier_level`) plus `rbsp_trailing_bits`. The remaining VPS syntax
/// (operation points, HRD, extensions, ...) is not written; `parse_vps`
/// doesn't read it and the decoder discards the parsed VPS entirely.
pub fn write_vps() -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bits(4, 0); // vps_video_parameter_set_id
    w.write_bit(1); // vps_base_layer_internal_flag
    w.write_bit(1); // vps_base_layer_available_flag
    w.write_bits(6, 0); // vps_max_layers_minus1
    w.write_bits(3, 0); // vps_max_sub_layers_minus1
    w.write_bit(1); // vps_temporal_id_nesting_flag
    w.write_bits(16, 0xFFFF); // vps_reserved_0xffff_16bits

    write_profile_tier_level(&mut w, 0);

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}

/// `chroma_format_idc` per H.265 Table 6-1.
fn chroma_format_idc(chroma: ChromaFormat) -> u32 {
    match chroma {
        ChromaFormat::Gray => 0,
        ChromaFormat::Yuv420 => 1,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 3,
    }
}

/// `seq_parameter_set_rbsp()` (H.265 7.3.2.2), single-layer / single
/// sub-layer, no scaling lists / PCM / long-term ref pics / short-term ref
/// pic sets. Field order mirrors `parse_sps` exactly.
pub fn write_sps(config: &StillHevcConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bits(4, 0); // sps_video_parameter_set_id
    w.write_bits(3, 0); // sps_max_sub_layers_minus1
    w.write_bit(1); // sps_temporal_id_nesting_flag

    write_profile_tier_level(&mut w, 0);

    w.write_ue_golomb(0); // sps_seq_parameter_set_id

    let chroma_idc = chroma_format_idc(config.chroma);
    w.write_ue_golomb(chroma_idc);
    if chroma_idc == 3 {
        w.write_bit(0); // separate_colour_plane_flag
    }

    w.write_ue_golomb(config.width); // pic_width_in_luma_samples
    w.write_ue_golomb(config.height); // pic_height_in_luma_samples

    w.write_bit(0); // conformance_window_flag

    let bit_depth_minus8 = (config.bit_depth - 8) as u32;
    w.write_ue_golomb(bit_depth_minus8); // bit_depth_luma_minus8
    w.write_ue_golomb(bit_depth_minus8); // bit_depth_chroma_minus8

    w.write_ue_golomb(4); // log2_max_pic_order_cnt_lsb_minus4

    w.write_bit(0); // sps_sub_layer_ordering_info_present_flag
    // sub_layer_ordering_info_present_flag == 0 -> the loop still runs once
    // (for i == sps_max_sub_layers_minus1 == 0).
    w.write_ue_golomb(0); // sps_max_dec_pic_buffering_minus1
    w.write_ue_golomb(0); // sps_max_num_reorder_pics
    w.write_ue_golomb(0); // sps_max_latency_increase_plus1

    w.write_ue_golomb(0); // log2_min_luma_coding_block_size_minus3
    w.write_ue_golomb(3); // log2_diff_max_min_luma_coding_block_size (CTB = 64x64)
    w.write_ue_golomb(0); // log2_min_luma_transform_block_size_minus2
    w.write_ue_golomb(3); // log2_diff_max_min_luma_transform_block_size
    w.write_ue_golomb(2); // max_transform_hierarchy_depth_inter
    w.write_ue_golomb(2); // max_transform_hierarchy_depth_intra

    w.write_bit(0); // scaling_list_enabled_flag

    w.write_bit(1); // amp_enabled_flag
    w.write_bit(0); // sample_adaptive_offset_enabled_flag (see module docs)

    w.write_bit(0); // pcm_enabled_flag

    w.write_ue_golomb(0); // num_short_term_ref_pic_sets

    w.write_bit(0); // long_term_ref_pics_present_flag

    w.write_bit(1); // sps_temporal_mvp_enabled_flag
    w.write_bit(1); // strong_intra_smoothing_enabled_flag

    // VUI: signal full-range BT.601 (matrix_coeffs == 6), matching the BPG
    // colour convention. Only the fields `parse_sps` reads are written.
    w.write_bit(1); // vui_parameters_present_flag
    w.write_bit(0); // aspect_ratio_info_present_flag
    w.write_bit(0); // overscan_info_present_flag
    w.write_bit(1); // video_signal_type_present_flag
    w.write_bits(3, 5); // video_format (5 == Unspecified)
    w.write_bit(1); // video_full_range_flag
    w.write_bit(1); // colour_description_present_flag
    w.write_bits(8, 2); // colour_primaries (2 == Unspecified)
    w.write_bits(8, 2); // transfer_characteristics (2 == Unspecified)
    w.write_bits(8, 6); // matrix_coefficients (6 == BT.601)

    // Close out vui_parameters() per H.265 E.2.1 so a conformant decoder
    // (e.g. stock `bpgdec`/libde265, which reads the full VUI rather than
    // stopping after matrix_coeffs like `parse_sps` does) doesn't misread
    // rbsp_trailing_bits() as further VUI fields.
    w.write_bit(0); // chroma_loc_info_present_flag
    w.write_bit(0); // neutral_chroma_indication_flag
    w.write_bit(0); // field_seq_flag
    w.write_bit(0); // frame_field_info_present_flag
    w.write_bit(0); // default_display_window_flag
    w.write_bit(0); // vui_timing_info_present_flag
    w.write_bit(0); // bitstream_restriction_flag

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}

/// `pic_parameter_set_rbsp()` (H.265 7.3.2.3), no tiles / scaling lists /
/// deblocking overrides. Field order mirrors `parse_pps` exactly.
pub fn write_pps() -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_ue_golomb(0); // pps_pic_parameter_set_id
    w.write_ue_golomb(0); // pps_seq_parameter_set_id
    w.write_bit(0); // dependent_slice_segments_enabled_flag
    w.write_bit(0); // output_flag_present_flag
    w.write_bits(3, 0); // num_extra_slice_header_bits
    w.write_bit(1); // sign_data_hiding_enabled_flag
    w.write_bit(0); // cabac_init_present_flag
    w.write_ue_golomb(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue_golomb(0); // num_ref_idx_l1_default_active_minus1
    w.write_se_golomb(0); // init_qp_minus26
    w.write_bit(0); // constrained_intra_pred_flag
    w.write_bit(0); // transform_skip_enabled_flag
    w.write_bit(0); // cu_qp_delta_enabled_flag
    w.write_se_golomb(0); // pps_cb_qp_offset
    w.write_se_golomb(0); // pps_cr_qp_offset
    w.write_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.write_bit(0); // weighted_pred_flag
    w.write_bit(0); // weighted_bipred_flag
    w.write_bit(0); // transquant_bypass_enabled_flag
    w.write_bit(0); // tiles_enabled_flag
    w.write_bit(0); // entropy_coding_sync_enabled_flag (see module docs)
    w.write_bit(1); // pps_loop_filter_across_slices_enabled_flag
    w.write_bit(0); // deblocking_filter_control_present_flag
    w.write_bit(0); // pps_scaling_list_data_present_flag
    w.write_bit(0); // lists_modification_present_flag
    w.write_ue_golomb(0); // log2_parallel_merge_level_minus2
    w.write_bit(0); // slice_segment_header_extension_present_flag

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}
