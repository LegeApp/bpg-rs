//! CTU-local workspace and scratch buffers.

use super::arena::CtuArenas;
use super::ledger::StillSearchLedger;
use crate::contexts::Contexts;
use crate::residual::ResidualPricingScratch;

pub(super) struct CtuWorkspace {
    pub(super) arenas: CtuArenas,
    pub(super) ledger: StillSearchLedger,
    pub(super) block_scratch: BlockScratch,
    /// Frozen entry context for trial residual/flag pricing (the standard
    /// HM/x265 RDOQ approximation: contexts are not re-coded across candidates).
    /// Re-seeded per CTU with the slice QP.
    pub(super) price_base: Contexts,
    pub(super) price_scratch: ResidualPricingScratch,
}

impl Default for CtuWorkspace {
    fn default() -> Self {
        Self {
            arenas: CtuArenas::default(),
            ledger: StillSearchLedger::default(),
            block_scratch: BlockScratch::default(),
            price_base: Contexts::new(0),
            price_scratch: ResidualPricingScratch::default(),
        }
    }
}

impl CtuWorkspace {
    pub(super) fn reset(&mut self) {
        self.arenas.clear();
        self.ledger.clear_ctu();
        self.block_scratch.clear_ctu();
    }

    /// Seed the trial-pricing entry context with the slice QP.
    pub(super) fn set_price_qp(&mut self, slice_qp: i32) {
        self.price_base = Contexts::new(slice_qp.clamp(0, 51));
    }
}

/// Fixed CTU-local scratch for final bridge block coding and future trial
/// evaluators. Search paths must reuse these buffers rather than allocate per
/// candidate.
#[derive(Default, Debug)]
pub(super) struct BlockScratch {
    pub(super) residual_i16_4x4: Vec<i16>,
    pub(super) residual_i16_8x8: Vec<i16>,
    pub(super) residual_i16_16x16: Vec<i16>,
    pub(super) residual_i16_32x32: Vec<i16>,
    pub(super) coeff_i16: Vec<i16>,
    pub(super) transform_tmp_i16: Vec<i16>,
    pub(super) levels_i16: Vec<i16>,
    pub(super) dequant_coeff_i16: Vec<i16>,
    pub(super) recon_residual_i16: Vec<i16>,
}

impl BlockScratch {
    pub(super) fn clear_ctu(&mut self) {
        self.residual_i16_4x4.clear();
        self.residual_i16_8x8.clear();
        self.residual_i16_16x16.clear();
        self.residual_i16_32x32.clear();
        self.coeff_i16.clear();
        self.transform_tmp_i16.clear();
        self.levels_i16.clear();
        self.dequant_coeff_i16.clear();
        self.recon_residual_i16.clear();
    }

    pub(super) fn residual_i16_for_log2(&mut self, log2_size: u8) -> &mut Vec<i16> {
        match log2_size {
            2 => &mut self.residual_i16_4x4,
            3 => &mut self.residual_i16_8x8,
            4 => &mut self.residual_i16_16x16,
            5 => &mut self.residual_i16_32x32,
            _ => &mut self.residual_i16_32x32,
        }
    }
}
