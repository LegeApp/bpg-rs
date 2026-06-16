//! VPS/SPS/PPS (parameter set) writers for the Phase 3 syntax skeleton.
//!
//! Field order and presence conditions are the mirror image of
//! `parse_vps`/`parse_sps`/`parse_pps`/`parse_profile_tier_level` in
//! `bpg-hevc-decode::hevc::params`. Constant choices (profile/level, AMP,
//! etc.) are anchored to a real x265 BPG-still-image encode, dumped via
//! `tests/dump_oracle_params.rs` from
//! `oracle/out/checkerboard__420_8bit_qp24.hevc`. The SPS VUI omits
//! `video_signal_type_present_flag` (colour info travels in the BPG
//! container header instead), matching that oracle stream and satisfying
//! `bpg_hevc::ModifiedSps::rewrite_sps`'s preconditions.
//!
//! Phase 3 simplifications relative to that oracle stream (each chosen so
//! the slice-segment-header writer in `slice.rs` doesn't need to handle the
//! corresponding optional fields):
//! - `sample_adaptive_offset_enabled_flag = false` (oracle: true) when
//!   `StillHevcConfig::sao == SaoMode::Off` (the default) — avoids
//!   `slice_sao_luma_flag`/`slice_sao_chroma_flag` in the slice header.
//!   `SaoMode::On` sets this flag, and `slice.rs` writes those two flags
//!   (see `crate::sao` for the per-CTU SAO syntax this then requires).
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

/// `diff_cu_qp_delta_depth` written in the PPS when adaptive quantization is
/// active. `1` ⇒ `Log2MinCuQpDeltaSize = CtbLog2SizeY(6) − 1 = 5`, i.e. a 32x32
/// quantization group, aligned 1:1 with the 32x32 preanalysis cell grid. The
/// adaptive-QP writer in `encoder.rs` reads the same constant so the PPS and the
/// per-CU QP plan agree on the QG size.
pub const AQ_DIFF_CU_QP_DELTA_DEPTH: u32 = 1;

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

/// `video_parameter_set_rbsp()` (H.265 7.3.2.1). BPG drops the VPS entirely
/// (`build_modified_hevc` only checks its NAL type before discarding it), but
/// `encode()`'s raw Annex-B output is still a complete, self-contained
/// bitstream that conformant decoders parse in full — so the VPS body must be
/// well-formed (not just `profile_tier_level` + `rbsp_trailing_bits`) or a
/// strict decoder misaligns on everything that follows.
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

    w.write_bit(1); // vps_sub_layer_ordering_info_present_flag
    w.write_ue_golomb(0); // vps_max_dec_pic_buffering_minus1[0]
    w.write_ue_golomb(0); // vps_max_num_reorder_pics[0]
    w.write_ue_golomb(0); // vps_max_latency_increase_plus1[0]

    w.write_bits(6, 0); // vps_max_layer_id
    w.write_ue_golomb(0); // vps_num_layer_sets_minus1 (no layer_id_included_flag loop)

    w.write_bit(0); // vps_timing_info_present_flag
    w.write_bit(0); // vps_extension_flag

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bpg_bitstream::BitReader;

    fn test_config(deblock: crate::DeblockMode) -> StillHevcConfig {
        StillHevcConfig {
            width: 64,
            height: 64,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 28,
            effort: crate::Effort::Balanced,
            sao: crate::SaoMode::Off,
            deblock,
        }
    }

    /// Parse `write_pps()`'s RBSP the way a *conformant* decoder does — through
    /// `pps_extension_present_flag`, the field stock `bpgdec`'s bundled ffmpeg
    /// reads but the vendored `parse_pps` stops short of — and assert the
    /// stream terminates cleanly (`rbsp_stop_one_bit` then only zero padding to
    /// the byte boundary). Guards against a missing/extra PPS field, which the
    /// internal roundtrip cannot catch because the encoder and the vendored
    /// decoder share the same field list.
    fn check_pps_rbsp_terminates(deblock: crate::DeblockMode) {
        check_pps_rbsp_terminates_config(&test_config(deblock));
    }

    fn check_pps_rbsp_terminates_config(config: &StillHevcConfig) {
        let pps = write_pps(config);
        let mut r = BitReader::new(&pps);
        let ue = |r: &mut BitReader| r.read_ue_golomb().unwrap();

        ue(&mut r); // pps_pic_parameter_set_id
        ue(&mut r); // pps_seq_parameter_set_id
        r.read_bits(1); // dependent_slice_segments_enabled_flag
        r.read_bits(1); // output_flag_present_flag
        r.read_bits(3); // num_extra_slice_header_bits
        r.read_bits(1); // sign_data_hiding_enabled_flag
        r.read_bits(1); // cabac_init_present_flag
        ue(&mut r); // num_ref_idx_l0_default_active_minus1
        ue(&mut r); // num_ref_idx_l1_default_active_minus1
        r.read_se_golomb().unwrap(); // init_qp_minus26
        r.read_bits(1); // constrained_intra_pred_flag
        r.read_bits(1); // transform_skip_enabled_flag
        let cu_qp_delta_enabled = r.read_bits(1); // cu_qp_delta_enabled_flag
        assert_eq!(cu_qp_delta_enabled, crate::aq_active(config) as u32);
        if cu_qp_delta_enabled == 1 {
            // diff_cu_qp_delta_depth — present only when the flag is set.
            assert_eq!(ue(&mut r), AQ_DIFF_CU_QP_DELTA_DEPTH);
        }
        r.read_se_golomb().unwrap(); // pps_cb_qp_offset
        r.read_se_golomb().unwrap(); // pps_cr_qp_offset
        r.read_bits(1); // pps_slice_chroma_qp_offsets_present_flag
        r.read_bits(1); // weighted_pred_flag
        r.read_bits(1); // weighted_bipred_flag
        r.read_bits(1); // transquant_bypass_enabled_flag
        assert_eq!(r.read_bits(1), 0, "tiles_enabled_flag"); // no tile cols/rows loop
        r.read_bits(1); // entropy_coding_sync_enabled_flag
        r.read_bits(1); // pps_loop_filter_across_slices_enabled_flag
        let deblock_ctrl = r.read_bits(1); // deblocking_filter_control_present_flag
        if deblock_ctrl == 1 {
            let override_enabled = r.read_bits(1); // deblocking_filter_override_enabled_flag
            assert_eq!(override_enabled, 0, "override loop not written");
            let disabled = r.read_bits(1); // pps_deblocking_filter_disabled_flag
            if disabled == 0 {
                r.read_se_golomb().unwrap(); // pps_beta_offset_div2
                r.read_se_golomb().unwrap(); // pps_tc_offset_div2
            }
        }
        assert_eq!(r.read_bits(1), 0, "pps_scaling_list_data_present_flag");
        r.read_bits(1); // lists_modification_present_flag
        ue(&mut r); // log2_parallel_merge_level_minus2
        r.read_bits(1); // slice_segment_header_extension_present_flag
        assert_eq!(r.read_bits(1), 0, "pps_extension_present_flag"); // no range-ext bits

        // rbsp_trailing_bits(): stop bit then zero padding to the byte boundary.
        assert_eq!(r.read_bits(1), 1, "rbsp_stop_one_bit");
        let left = pps.len() * 8 - r.bit_pos();
        assert!(left < 8, "more than a byte of padding left: {left} bits");
        for _ in 0..left {
            assert_eq!(r.read_bits(1), 0, "rbsp_alignment_zero_bit");
        }
    }

    #[test]
    fn pps_rbsp_terminates_for_conformant_decoder_deblock_off() {
        check_pps_rbsp_terminates(crate::DeblockMode::Off);
    }

    #[test]
    fn pps_rbsp_terminates_for_conformant_decoder_deblock_on() {
        check_pps_rbsp_terminates(crate::DeblockMode::On);
    }

    /// With adaptive QP active (any non-reference tier, every chroma format), the
    /// PPS must set `cu_qp_delta_enabled_flag = 1`, emit `diff_cu_qp_delta_depth`,
    /// and still terminate cleanly for a conformant decoder.
    #[test]
    fn pps_aq_active_emits_cu_qp_delta_depth_and_terminates() {
        for chroma in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ] {
            for deblock in [crate::DeblockMode::Off, crate::DeblockMode::On] {
                let mut config = test_config(deblock); // Balanced ⇒ AQ active
                config.chroma = chroma;
                assert!(
                    crate::aq_active(&config),
                    "AQ should be active for {chroma:?}"
                );
                check_pps_rbsp_terminates_config(&config);
            }
        }
    }

    /// Monochrome is excluded from adaptive QP (see [`crate::aq_active`]), so its
    /// PPS carries `cu_qp_delta_enabled_flag = 0` and still terminates cleanly.
    #[test]
    fn pps_aq_off_for_monochrome() {
        let mut config = test_config(crate::DeblockMode::On); // Balanced
        config.chroma = ChromaFormat::Gray;
        assert!(!crate::aq_active(&config));
        check_pps_rbsp_terminates_config(&config);
    }

    /// The high-quality uniform-QP tiers never enable adaptive QP, so their PPS
    /// carries `cu_qp_delta_enabled_flag = 0`.
    #[test]
    fn pps_aq_off_for_high_quality_tiers() {
        for effort in [
            crate::Effort::Best,
            crate::Effort::Placebo,
            crate::Effort::Reference,
        ] {
            let mut config = test_config(crate::DeblockMode::On);
            config.effort = effort;
            assert!(!crate::aq_active(&config));
            // `check_pps_rbsp_terminates_config` asserts the flag matches
            // `aq_active` (i.e. 0 here) and the RBSP still terminates cleanly.
            check_pps_rbsp_terminates_config(&config);
        }
    }
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
                    // sample_adaptive_offset_enabled_flag: SaoMode::Off (default) keeps the
                    // Phase 3 simplification documented in the module docs (no
                    // slice_sao_luma_flag/slice_sao_chroma_flag in the slice header).
                    // SaoMode::On enables it; slice.rs then writes those two flags.
    w.write_bit((config.sao != crate::SaoMode::Off) as u32);

    w.write_bit(0); // pcm_enabled_flag

    w.write_ue_golomb(0); // num_short_term_ref_pic_sets

    w.write_bit(0); // long_term_ref_pics_present_flag

    w.write_bit(1); // sps_temporal_mvp_enabled_flag
    w.write_bit(1); // strong_intra_smoothing_enabled_flag

    // VUI: no video_signal_type (colour info travels in the BPG container
    // header, not the SPS) — this matches x265's BPG-still-image SPS, which
    // is the profile `bpg_hevc::ModifiedSps::rewrite_sps` accepts.
    w.write_bit(1); // vui_parameters_present_flag
    w.write_bit(0); // aspect_ratio_info_present_flag
    w.write_bit(0); // overscan_info_present_flag
    w.write_bit(0); // video_signal_type_present_flag

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

    w.write_bit(0); // sps_extension_flag

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}

/// `pic_parameter_set_rbsp()` (H.265 7.3.2.3), no tiles / scaling lists /
/// deblocking overrides. Field order mirrors `parse_pps` exactly.
///
/// `pps_deblocking_filter_disabled_flag` follows `config.deblock`:
/// `DeblockMode::Off` (default) disables the in-loop filter, matching the
/// slice-header simplifications documented in the module docs above (no
/// `slice_loop_filter_across_slices_enabled_flag`). `DeblockMode::On` enables
/// it; `slice.rs` then writes the now-present
/// `slice_loop_filter_across_slices_enabled_flag` bit.
pub fn write_pps(config: &StillHevcConfig) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_ue_golomb(0); // pps_pic_parameter_set_id
    w.write_ue_golomb(0); // pps_seq_parameter_set_id
    w.write_bit(0); // dependent_slice_segments_enabled_flag
    w.write_bit(0); // output_flag_present_flag
    w.write_bits(3, 0); // num_extra_slice_header_bits
    w.write_bit(0); // sign_data_hiding_enabled_flag (milestone 1: SDH not applied)
    w.write_bit(0); // cabac_init_present_flag
    w.write_ue_golomb(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue_golomb(0); // num_ref_idx_l1_default_active_minus1
    w.write_se_golomb(0); // init_qp_minus26
    w.write_bit(0); // constrained_intra_pred_flag
    w.write_bit(0); // transform_skip_enabled_flag
                    // Adaptive quantization (per-CU QP) is gated by [`crate::aq_active`] (the
                    // single gate the encoder also consults, so the flag and the per-CU QP plan
                    // can't disagree). It is `0` (and the stream is the exact reference bitstream)
                    // only on the high-quality uniform-QP tiers. `diff_cu_qp_delta_depth = 1` ⇒
                    // Log2MinCuQpDeltaSize = CtbLog2(6) − 1 = 5, i.e. a 32x32 quantization group
                    // aligned 1:1 with the preanalysis cell grid (H.265 7.3.2.3; decoder
                    // `bpg-hevc-decode::hevc::ctu::decode_quantization_parameters`).
    let aq_active = crate::aq_active(config);
    w.write_bit(aq_active as u32); // cu_qp_delta_enabled_flag
    if aq_active {
        w.write_ue_golomb(AQ_DIFF_CU_QP_DELTA_DEPTH); // diff_cu_qp_delta_depth (QG = 32x32)
    }
    w.write_se_golomb(0); // pps_cb_qp_offset
    w.write_se_golomb(0); // pps_cr_qp_offset
    w.write_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.write_bit(0); // weighted_pred_flag
    w.write_bit(0); // weighted_bipred_flag
    w.write_bit(0); // transquant_bypass_enabled_flag
    w.write_bit(0); // tiles_enabled_flag
    w.write_bit(0); // entropy_coding_sync_enabled_flag (see module docs)
    w.write_bit(1); // pps_loop_filter_across_slices_enabled_flag
    let deblock_enabled = config.deblock == crate::DeblockMode::On;
    // control present + override disabled. With the filter disabled
    // (`DeblockMode::Off`), the slice inherits a disabled filter and omits
    // the override/loop-filter-across fields. With the filter enabled
    // (`DeblockMode::On`), `pps_beta_offset_div2`/`pps_tc_offset_div2` (both
    // 0, i.e. no slice-level offset) become present, and the slice header
    // gains `slice_loop_filter_across_slices_enabled_flag`.
    w.write_bit(1); // deblocking_filter_control_present_flag
    w.write_bit(0); // deblocking_filter_override_enabled_flag
    w.write_bit((!deblock_enabled) as u32); // pps_deblocking_filter_disabled_flag
    if deblock_enabled {
        w.write_se_golomb(0); // pps_beta_offset_div2
        w.write_se_golomb(0); // pps_tc_offset_div2
    }
    w.write_bit(0); // pps_scaling_list_data_present_flag
    w.write_bit(0); // lists_modification_present_flag
    w.write_ue_golomb(0); // log2_parallel_merge_level_minus2
    w.write_bit(0); // slice_segment_header_extension_present_flag

    // `pps_extension_present_flag`: read by conformant decoders (stock
    // `bpgdec`'s bundled ffmpeg `ff_hevc_decode_nal_pps`) immediately after
    // `slice_segment_header_extension_present_flag`. `parse_pps` in the
    // vendored Rust decoder stops one field earlier, so omitting this passed
    // the internal roundtrip but made stock `bpgdec` misread
    // `rbsp_trailing_bits()` as `pps_extension_present_flag = 1` and overread
    // the PPS by a byte.
    w.write_bit(0); // pps_extension_present_flag

    write_rbsp_trailing_bits(&mut w);
    w.into_bytes()
}
