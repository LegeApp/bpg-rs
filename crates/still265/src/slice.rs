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
//! - `sps.sample_adaptive_offset_enabled_flag` follows `config.sao` (see
//!   `params.rs`): with `SaoMode::Off` (default),
//!   `slice_sao_luma_flag`/`slice_sao_chroma_flag` are absent; with
//!   `SaoMode::On` both are present and written as 1 (per-CTU SAO syntax —
//!   see `crate::sao` — then determines whether SAO is actually applied).
//! - `pps.pps_slice_chroma_qp_offsets_present_flag == false` and
//!   `pps.deblocking_filter_override_enabled_flag == false`, so the chroma
//!   QP offset and deblocking-override fields are absent.
//! - `pps.pps_loop_filter_across_slices_enabled_flag == true`. With
//!   `config.deblock == DeblockMode::On`, `pps_deblocking_filter_disabled_flag
//!   == false`, so `slice_loop_filter_across_slices_enabled_flag` IS present
//!   (written as 1). With `DeblockMode::Off` (default),
//!   `pps_deblocking_filter_disabled_flag == true`, so the presence condition
//!   `slice_sao_* || !slice_deblocking_filter_disabled` reduces to
//!   `slice_sao_*` — present (and written as 1) iff `config.sao !=
//!   SaoMode::Off`.
//! - `num_entry_point_offsets` is present when tiles or WPP are enabled.
//! - `pps.slice_segment_header_extension_present_flag == false`.
//!
//! After these fields comes `byte_alignment()`: `alignment_bit_equal_to_one`
//! (1) followed by zero `alignment_bit_equal_to_zero`s up to the next byte
//! boundary, after which `slice_data()` (the CABAC-coded CTU data) begins.

use bpg_bitstream::BitWriter;

use crate::{ChromaFormat, StillHevcConfig};

/// `slice_segment_header()` (H.265 7.3.6.1) for a single first-slice IDR
/// I-slice, ending with `byte_alignment()`. The returned bytes are the
/// header only; CABAC-coded slice data (not yet implemented) follows.
pub fn write_slice_segment_header(
    config: &StillHevcConfig,
    tiles: Option<(u32, u32)>,
    wpp: bool,
    entry_sizes: &[u32],
) -> Vec<u8> {
    let mut w = BitWriter::new();

    w.write_bit(1); // first_slice_segment_in_pic_flag
    w.write_bit(0); // no_output_of_prior_pics_flag (IRAP)
    w.write_ue_golomb(0); // slice_pic_parameter_set_id

    w.write_ue_golomb(2); // slice_type (2 == I)

    // SAO flags (H.265 7.3.6.1), present iff
    // sps.sample_adaptive_offset_enabled_flag (== `config.sao !=
    // SaoMode::Off`, see params.rs). For monochrome, `slice_sao_chroma_flag`
    // is absent because `chroma_array_type() == 0`.
    let sao_enabled = config.sao != crate::SaoMode::Off;
    if sao_enabled {
        w.write_bit(1); // slice_sao_luma_flag
        if config.chroma != ChromaFormat::Gray {
            w.write_bit(1); // slice_sao_chroma_flag
        }
    }

    // slice_qp_delta: SliceQPY = 26 + pps.init_qp_minus26 (0) + slice_qp_delta
    w.write_se_golomb(config.qp as i32 - 26);

    if sao_enabled || config.deblock == crate::DeblockMode::On {
        // pps_deblocking_filter_disabled_flag == false (deblocking enabled),
        // deblocking_filter_override_enabled_flag == false, so
        // deblocking_filter_override_flag/slice_deblocking_filter_disabled_flag
        // are absent and inherited as false/false. The presence condition for
        // slice_loop_filter_across_slices_enabled_flag (`slice_sao_* ||
        // !slice_deblocking_filter_disabled`) is then true whenever SAO is on
        // (regardless of deblocking) or deblocking is on; write 1, matching
        // `pps_loop_filter_across_slices_enabled_flag`.
        w.write_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    // With SAO off and DeblockMode::Off, deblocking is disabled in the PPS,
    // so slice_loop_filter_across_slices_enabled_flag is absent (its
    // presence condition `slice_sao_* || !slice_deblocking_filter_disabled`
    // is false).

    // Entry point offsets (present iff tiles or WPP are enabled in the PPS).
    // One offset per substream except the last.
    if tiles.is_some() || wpp {
        w.write_ue_golomb(entry_sizes.len() as u32); // num_entry_point_offsets
        if !entry_sizes.is_empty() {
            // Each entry size is a NAL substream byte length and must be >= 1
            // (every substream carries at least its trailing bits); guard the
            // `- 1` so a degenerate empty substream can't underflow to u32::MAX
            // and corrupt offset_len.
            debug_assert!(
                entry_sizes.iter().all(|&s| s >= 1),
                "tile entry-point substream length must be >= 1, got {entry_sizes:?}"
            );
            let max_minus1 = entry_sizes
                .iter()
                .map(|&s| s.saturating_sub(1))
                .max()
                .unwrap();
            // offset_len = bits needed to hold the largest entry_point_offset_minus1.
            let offset_len = (32 - max_minus1.leading_zeros()).max(1);
            w.write_ue_golomb(offset_len - 1); // offset_len_minus1
            for &s in entry_sizes {
                w.write_bits(offset_len, s.saturating_sub(1)); // entry_point_offset_minus1[i]
            }
        }
    }

    // byte_alignment()
    w.write_bit(1); // alignment_bit_equal_to_one
    w.byte_align(); // alignment_bit_equal_to_zero*

    w.into_bytes()
}
