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
//! - `entropy_coding_sync_enabled_flag` follows the resolved encoder WPP mode.
//!   It defaults off for serial/tile streams and on only when the encoder emits
//!   row substreams with entry-point offsets.
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
            effort: crate::Effort::Slow,
            sao: crate::SaoMode::Off,
            deblock,
            adaptive_qp: false,
            aq_mode: crate::AqMode::Off,
            aq_strength: 0.35,
            aq_clamp: 2,
            two_pass_gate: true,
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
        let wpp = effective_wpp_enabled(config);
        let pps = write_pps(config, effective_tile_dims(config), wpp);
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
        assert_eq!(
            r.read_bits(1),
            wpp as u32,
            "entropy_coding_sync_enabled_flag"
        );
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

    /// With adaptive QP active (every chroma format except gray), the
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
                let mut config = test_config(deblock); // Balanced
                config.chroma = chroma;
                config.adaptive_qp = true; // opt into AQ (uniform-QP is the default)
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
        config.adaptive_qp = true; // even opted-in, monochrome stays uniform-QP
        assert!(!crate::aq_active(&config));
        check_pps_rbsp_terminates_config(&config);
    }

    /// The high-quality uniform-QP tiers never enable adaptive QP, so their PPS
    /// carries `cu_qp_delta_enabled_flag = 0`.
    #[test]
    fn pps_aq_off_for_high_quality_tiers() {
        for effort in [crate::Effort::Slow, crate::Effort::Placebo] {
            let mut config = test_config(crate::DeblockMode::On);
            config.effort = effort;
            assert!(!crate::aq_active(&config));
            // `check_pps_rbsp_terminates_config` asserts the flag matches
            // `aq_active` (i.e. 0 here) and the RBSP still terminates cleanly.
            check_pps_rbsp_terminates_config(&config);
        }
    }

    #[test]
    fn auto_grid_small_image_stays_single() {
        // 16x12 CTBs (1000x750), min 6 CTBs/tile → at most 2x2; with few cores
        // it must not over-tile, and 1 core disables tiles entirely.
        assert_eq!(super::auto_tile_dims(16, 12, 1, 6), None);
        // A picture smaller than two min-tiles in both axes can't tile.
        assert_eq!(super::auto_tile_dims(6, 6, 20, 6), None);
        assert_eq!(super::auto_tile_dims(11, 11, 20, 6), None);
    }

    #[test]
    fn auto_grid_targets_cores_without_overshoot() {
        // 38x29 CTBs (2400x1800), 20 cores, min 6 → 5x4 = 20 tiles (gap 0).
        assert_eq!(super::auto_tile_dims(38, 29, 20, 6), Some((5, 4)));
        // Min-size floor caps tile count below the core budget on the same
        // picture: min 8 CTBs → at most 4x3 = 12.
        assert_eq!(super::auto_tile_dims(38, 29, 20, 8), Some((4, 3)));
        // Never exceed the core count even when the picture could host more.
        let (c, r) = super::auto_tile_dims(63, 47, 20, 6).unwrap();
        assert!(c * r <= 20, "tile count {} overshot cores", c * r);
        assert!(c * r >= 16, "tile count {} undershot badly", c * r);
    }

    #[test]
    fn auto_grid_prefers_square_tiles() {
        // A square CTB field with 4 cores should pick 2x2, not 4x1/1x4.
        assert_eq!(super::auto_tile_dims(24, 24, 4, 6), Some((2, 2)));
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
/// Core budget the tile auto-grid targets. Mirrors the encoder's
/// `parallel_thread_count` (`BPG_ENC_THREADS` override, else
/// `available_parallelism`) so the grid is sized for the same pool that builds
/// it.
fn tile_core_count() -> u32 {
    if let Ok(v) = std::env::var("BPG_ENC_THREADS") {
        if let Ok(n) = v.trim().parse::<u32>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn explicit_tile_request() -> bool {
    std::env::var("BPG_TILES")
        .ok()
        .map(|v| {
            let s = v.trim();
            !(s.is_empty()
                || s.eq_ignore_ascii_case("auto")
                || s == "0"
                || s.eq_ignore_ascii_case("off")
                || s.eq_ignore_ascii_case("none"))
        })
        .unwrap_or(false)
}

fn wpp_env_allows_auto() -> bool {
    std::env::var("BPG_WPP")
        .ok()
        .map(|v| {
            let s = v.trim();
            !(s == "0" || s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("none"))
        })
        .unwrap_or(true)
}

fn aq_wpp_build_env_allows_auto() -> bool {
    std::env::var("BPG_AQ_WPP")
        .ok()
        .map(|v| {
            let s = v.trim();
            !(s == "0" || s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("none"))
        })
        .unwrap_or(true)
}

fn wpp_shape_enabled(config: &StillHevcConfig) -> bool {
    if !wpp_env_allows_auto() || explicit_tile_request() {
        return false;
    }
    if !crate::effort::template(config.effort).parallel_analysis {
        return false;
    }
    if tile_core_count() <= 1 {
        return false;
    }
    let ctb = 1u32 << 6;
    config.width.div_ceil(ctb) > 1 && config.height.div_ceil(ctb) > 1
}

/// Resolved encoder WPP mode for emitted bitstreams. AQ streams intentionally
/// keep serial slice syntax because HEVC QP prediction is not reset at WPP row
/// boundaries.
pub fn effective_wpp_enabled(config: &StillHevcConfig) -> bool {
    if crate::aq_active(config) {
        return false;
    }
    wpp_shape_enabled(config)
}

/// Resolved WPP row-worker mode for CU tree construction. AQ can use this as a
/// build-only acceleration path while the final syntax writer remains serial.
pub fn effective_wpp_build_enabled(config: &StillHevcConfig) -> bool {
    if crate::aq_active(config) && !aq_wpp_build_env_allows_auto() {
        return false;
    }
    wpp_shape_enabled(config)
}

/// Uniform auto-grid (Phase 5a): choose `cols x rows` so the tile count is as
/// close to `cores` as possible *without overshooting* (extra tiles past the
/// core count only cost bytes from broken cross-tile prediction with no added
/// parallelism), while keeping every tile at least `min_ctbs` CTBs per
/// dimension. The min-size floor is what stops small images from over-tiling:
/// when the picture can't host more than one such tile the grid collapses to
/// `None` (single tile = tiles disabled). Among equal-tile-count candidates the
/// squarest tiling wins (least cross-tile boundary, best load balance).
fn auto_tile_dims(ctbs_x: u32, ctbs_y: u32, cores: u32, min_ctbs: u32) -> Option<(u32, u32)> {
    if cores <= 1 {
        return None;
    }
    let max_cols = (ctbs_x / min_ctbs).max(1);
    let max_rows = (ctbs_y / min_ctbs).max(1);
    if max_cols * max_rows <= 1 {
        return None;
    }
    let mut best: Option<(u32, u32)> = None;
    let mut best_score = f64::INFINITY;
    for c in 1..=max_cols {
        for r in 1..=max_rows {
            let t = c * r;
            if t <= 1 || t > cores {
                continue;
            }
            // Primary: get the tile count to `cores` (more parallelism).
            // Secondary tie-break: square-ish tiles (boundary-minimizing).
            let tile_aspect = (ctbs_x as f64 / c as f64) / (ctbs_y as f64 / r as f64);
            let aspect_dev = tile_aspect.ln().abs();
            let score = (cores - t) as f64 * 2.0 + aspect_dev;
            if score < best_score {
                best_score = score;
                best = Some((c, r));
            }
        }
    }
    best.filter(|&(c, r)| c * r > 1)
}

/// Whether this encode will route through the tile-capable writer
/// (`write_slice_from_trees`). The serial single-substream path
/// (`encode_slice_data`) ignores the tile grid entirely, so declaring tiles
/// there would desync the stream (PPS says N tiles, slice data is one raster
/// substream). That tree-replay writer runs whenever SAO is on, or whenever the
/// effort uses parallel analysis.
fn tile_capable(config: &StillHevcConfig) -> bool {
    config.sao == crate::SaoMode::On || crate::effort::template(config.effort).parallel_analysis
}

/// Whether `effort` opts into the auto tile-grid (when `BPG_TILES` is unset or
/// `auto`). Tiling itself is orthogonal to the quality ladder — an explicit
/// `BPG_TILES=COLSxROWS` works for any tile-capable effort — but the auto-grid
/// default is currently scoped to the Slow tier, the validated tiled config.
fn auto_tile_effort(effort: crate::Effort) -> bool {
    matches!(effort, crate::Effort::Slow)
}

/// Effective tile grid `(columns, rows)` for this encode, or `None` when tiles
/// are disabled. Controlled by `BPG_TILES`:
/// * unset or `auto` → uniform auto-grid sized for the core count (Phase 5a),
///   for efforts that opt in via [`auto_tile_effort`];
/// * `0`/`off`/`none` → tiles disabled;
/// * `COLSxROWS` → an explicit uniform grid (diagnostic; any tile-capable effort).
///
/// Returns `None` unconditionally when the encode path can't emit tiles (see
/// [`tile_capable`]) — otherwise the PPS would declare a partition the slice data
/// doesn't carry. `BPG_TILE_MIN_CTBS` (default 6 = 384px) tunes the auto-grid's
/// minimum tile dimension. The request is validated against the picture's CTB
/// dimensions; an out-of-range or trivial (1x1) request disables tiles. This is
/// the single source of truth shared by [`write_pps`], the slice header, and the
/// encoder, so they always agree on the partition.
pub fn effective_tile_dims(config: &StillHevcConfig) -> Option<(u32, u32)> {
    if effective_wpp_enabled(config) || effective_wpp_build_enabled(config) {
        return None;
    }
    if !tile_capable(config) {
        return None;
    }
    let ctb = 1u32 << 6; // 64x64 CTBs
    let ctbs_x = config.width.div_ceil(ctb);
    let ctbs_y = config.height.div_ceil(ctb);

    let spec = std::env::var("BPG_TILES").ok();
    let spec = spec.as_deref().map(str::trim);
    match spec {
        // Default and explicit `auto`: uniform auto-grid, for opted-in efforts.
        None | Some("") | Some("auto") | Some("AUTO") => {
            if !auto_tile_effort(config.effort) {
                return None;
            }
            let min_ctbs = std::env::var("BPG_TILE_MIN_CTBS")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .filter(|&v| v >= 1)
                .unwrap_or(6);
            // `BPG_TILE_OVERSUB` (default 1.0) over-partitions the grid beyond the
            // core count. With work-stealing tile scheduling, more tiles than
            // threads lets cheap-tile workers absorb heavy-tile slack so the
            // parallel makespan approaches the *mean* tile cost rather than the
            // heaviest tile's — trading a small byte cost (more cross-tile
            // boundaries) for the ~18-22% makespan skew measured on real images.
            let oversub = std::env::var("BPG_TILE_OVERSUB")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|&v| v >= 1.0)
                .unwrap_or(1.0);
            let budget = ((tile_core_count() as f64 * oversub).round() as u32).max(1);
            auto_tile_dims(ctbs_x, ctbs_y, budget, min_ctbs)
        }
        Some("0") | Some("off") | Some("OFF") | Some("none") | Some("NONE") => None,
        // Explicit `COLSxROWS`.
        Some(s) => {
            let (c, r) = s.split_once(|ch| ch == 'x' || ch == 'X' || ch == ',')?;
            let cols: u32 = c.trim().parse().ok()?;
            let rows: u32 = r.trim().parse().ok()?;
            if cols < 1 || rows < 1 || cols > ctbs_x || rows > ctbs_y || (cols == 1 && rows == 1) {
                return None;
            }
            Some((cols, rows))
        }
    }
}

/// `tiles` and `wpp` are the resolved partition mode, passed in by the caller so
/// the PPS, slice header, and encoder share one computation.
pub fn write_pps(config: &StillHevcConfig, tiles: Option<(u32, u32)>, wpp: bool) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_ue_golomb(0); // pps_pic_parameter_set_id
    w.write_ue_golomb(0); // pps_seq_parameter_set_id
    w.write_bit(0); // dependent_slice_segments_enabled_flag
    w.write_bit(0); // output_flag_present_flag
    w.write_bits(3, 0); // num_extra_slice_header_bits
    w.write_bit(crate::sdh_active(config) as u32); // sign_data_hiding_enabled_flag
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
    let chroma_qp_offset = crate::chroma_qp_offset() as i32;
    w.write_se_golomb(chroma_qp_offset); // pps_cb_qp_offset
    w.write_se_golomb(chroma_qp_offset); // pps_cr_qp_offset
    w.write_bit(0); // pps_slice_chroma_qp_offsets_present_flag
    w.write_bit(0); // weighted_pred_flag
    w.write_bit(0); // weighted_bipred_flag
    w.write_bit(0); // transquant_bypass_enabled_flag
    w.write_bit(tiles.is_some() as u32); // tiles_enabled_flag
    w.write_bit(wpp as u32); // entropy_coding_sync_enabled_flag
    if let Some((cols, rows)) = tiles {
        w.write_ue_golomb(cols - 1); // num_tile_columns_minus1
        w.write_ue_golomb(rows - 1); // num_tile_rows_minus1
        w.write_bit(1); // uniform_spacing_flag
        w.write_bit(1); // loop_filter_across_tiles_enabled_flag (whole-frame deblock)
    }
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
