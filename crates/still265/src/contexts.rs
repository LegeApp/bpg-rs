//! CABAC context-model storage and the H.265 Table 9-4..9-37 initialization
//! values for the still-image intra encode path.
//!
//! The context layout (the `ctx` index constants and `INIT_VALUES` table) is a
//! verbatim mirror of `bpg-hevc-decode::hevc::cabac`'s `context` module and
//! `INIT_VALUES`, so the encoder and decoder address the identical 170-context
//! array. `contexts_match_decoder` (a test) cross-checks the two tables. Each
//! [`crate::cabac::ContextModel`] is initialized from its `init_value` and the
//! slice QP per H.265 9.3.2.2 (`ContextModel::new`).

use crate::cabac::ContextModel;

/// Context-index bases for each syntax element, H.265 9.3.2.2 / Table 9-4.
/// Mirrors `bpg-hevc-decode::hevc::cabac::context`.
#[allow(dead_code)]
pub mod ctx {
    pub const SPLIT_CU_FLAG: usize = 0;
    pub const CU_TRANSQUANT_BYPASS_FLAG: usize = 3;
    pub const CU_SKIP_FLAG: usize = 4;
    pub const PALETTE_MODE_FLAG: usize = 7;
    pub const PRED_MODE_FLAG: usize = 8;
    pub const PART_MODE: usize = 9;
    pub const PREV_INTRA_LUMA_PRED_FLAG: usize = 13;
    pub const INTRA_CHROMA_PRED_MODE: usize = 14;
    pub const INTER_PRED_IDC: usize = 15;
    pub const MERGE_FLAG: usize = 20;
    pub const MERGE_IDX: usize = 21;
    pub const MVP_LX_FLAG: usize = 22;
    pub const REF_IDX: usize = 23;
    pub const ABS_MVD_GREATER0_FLAG: usize = 25;
    pub const ABS_MVD_GREATER1_FLAG: usize = 27;
    pub const SPLIT_TRANSFORM_FLAG: usize = 28;
    pub const CBF_LUMA: usize = 31;
    pub const CBF_CBCR: usize = 33;
    pub const TRANSFORM_SKIP_FLAG: usize = 38;
    pub const LAST_SIG_COEFF_X_PREFIX: usize = 40;
    pub const LAST_SIG_COEFF_Y_PREFIX: usize = 58;
    pub const CODED_SUB_BLOCK_FLAG: usize = 76;
    pub const SIG_COEFF_FLAG: usize = 80;
    pub const COEFF_ABS_LEVEL_GREATER1_FLAG: usize = 124;
    pub const COEFF_ABS_LEVEL_GREATER2_FLAG: usize = 148;
    pub const SAO_MERGE_FLAG: usize = 154;
    pub const SAO_TYPE_IDX: usize = 155;
    pub const CU_QP_DELTA_ABS: usize = 156;
    pub const CU_CHROMA_QP_OFFSET_FLAG: usize = 158;
    pub const CU_CHROMA_QP_OFFSET_IDX: usize = 159;
    pub const LOG2_RES_SCALE_ABS_PLUS1: usize = 160;
    pub const RES_SCALE_SIGN_FLAG: usize = 168;
    pub const NUM_CONTEXTS: usize = 170;
}

/// Initial context values from H.265 (Table 9-4 et seq.), I-slice column.
/// Mirrors `bpg-hevc-decode::hevc::cabac::INIT_VALUES`.
pub static INIT_VALUES: [u8; ctx::NUM_CONTEXTS] = [
    // SPLIT_CU_FLAG (3)
    139, 141, 157, // CU_TRANSQUANT_BYPASS_FLAG (1)
    154, // CU_SKIP_FLAG (3)
    197, 185, 201, // PALETTE_MODE_FLAG (1)
    154, // PRED_MODE_FLAG (1)
    149, // PART_MODE (4)
    184, 154, 139, 154, // PREV_INTRA_LUMA_PRED_FLAG (1)
    184, // INTRA_CHROMA_PRED_MODE (1)
    63,  // INTER_PRED_IDC (5)
    95, 79, 63, 31, 31,  // MERGE_FLAG (1)
    110, // MERGE_IDX (1)
    122, // MVP_LX_FLAG (1)
    168, // REF_IDX (2)
    153, 153, // ABS_MVD_GREATER0_FLAG (2)
    140, 198, // ABS_MVD_GREATER1_FLAG (1)
    140, // SPLIT_TRANSFORM_FLAG (3)
    153, 138, 138, // CBF_LUMA (2)
    111, 141, // CBF_CBCR (5)
    94, 138, 182, 154, 154, // TRANSFORM_SKIP_FLAG (2)
    139, 139, // LAST_SIG_COEFF_X_PREFIX (18)
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    // LAST_SIG_COEFF_Y_PREFIX (18)
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
    // CODED_SUB_BLOCK_FLAG (4)
    91, 171, 134, 141, // SIG_COEFF_FLAG (44)
    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179,
    153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139,
    111, 136, 139, 111, 155, 154, // COEFF_ABS_LEVEL_GREATER1_FLAG (24)
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166,
    182, 140, 227, 122, 197, // COEFF_ABS_LEVEL_GREATER2_FLAG (6)
    138, 153, 136, 167, 152, 152, // SAO_MERGE_FLAG (1)
    153, // SAO_TYPE_IDX (1)
    200, // CU_QP_DELTA_ABS (2)
    154, 154, // CU_CHROMA_QP_OFFSET_FLAG (1)
    154, // CU_CHROMA_QP_OFFSET_IDX (1)
    154, // LOG2_RES_SCALE_ABS_PLUS1 (8)
    154, 154, 154, 154, 154, 154, 154, 154, // RES_SCALE_SIGN_FLAG (2)
    154, 154,
];

/// The full 170-entry CABAC context state for one slice, initialized from
/// [`INIT_VALUES`] and the slice QP.
#[derive(Clone)]
pub struct Contexts {
    pub models: [ContextModel; ctx::NUM_CONTEXTS],
}

impl Contexts {
    /// Initialize every context for `slice_qp` per H.265 9.3.2.2.
    pub fn new(slice_qp: i32) -> Self {
        let models = core::array::from_fn(|i| ContextModel::new(INIT_VALUES[i], slice_qp));
        Self { models }
    }

    #[inline]
    pub fn get(&mut self, idx: usize) -> &mut ContextModel {
        &mut self.models[idx]
    }
}
