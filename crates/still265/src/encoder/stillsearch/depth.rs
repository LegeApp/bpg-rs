//! StillSearch depth dispatch and CTU entrypoint.

use crate::contexts::Contexts;
use crate::encoder::Encoder;
use crate::encoder::syntax::CuNode;

use super::emit;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::overlay::{OverlayCache, ReconOverlay8, ReconOverlay16};
use super::source::{CtuSource8, CtuSource16, CtuSourceCache};
use super::workspace::CtuWorkspace;

pub(in crate::encoder) struct StillSearch {
    imp: StillSearchImpl,
}

enum StillSearchImpl {
    EightBit(StillSearchDepth<CtuSource8, ReconOverlay8>),
    HighBit(StillSearchDepth<CtuSource16, ReconOverlay16>),
}

pub(super) struct StillSearchDepth<S, O> {
    pub(super) workspace: CtuWorkspace,
    pub(super) source: S,
    pub(super) overlay: O,
}

impl StillSearch {
    pub(in crate::encoder) fn new(bit_depth: u8) -> Self {
        let imp = if bit_depth == 8 {
            StillSearchImpl::EightBit(StillSearchDepth::default())
        } else {
            StillSearchImpl::HighBit(StillSearchDepth::default())
        };
        Self { imp }
    }

    pub(in crate::encoder) fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        price_ctx: &Contexts,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        match &mut self.imp {
            StillSearchImpl::EightBit(search) => {
                search.build_ctu(state, price_ctx, x0, y0, log2_cb_size, ct_depth)
            }
            StillSearchImpl::HighBit(search) => {
                search.build_ctu(state, price_ctx, x0, y0, log2_cb_size, ct_depth)
            }
        }
    }
}

impl<S, O> Default for StillSearchDepth<S, O>
where
    S: Default,
    O: Default,
{
    fn default() -> Self {
        Self {
            workspace: CtuWorkspace::default(),
            source: S::default(),
            overlay: O::default(),
        }
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    pub(super) fn build_ctu(
        &mut self,
        state: &mut Encoder<'_>,
        price_ctx: &Contexts,
        x0: u32,
        y0: u32,
        log2_cb_size: u8,
        ct_depth: u8,
    ) -> CuNode {
        self.workspace.reset();
        self.workspace.set_price_context(price_ctx);
        self.source.reset_from_ctu(state, x0, y0, log2_cb_size);
        self.overlay.clear();

        let lambda = super::price::rd_lambda(state.cur_qp_y);
        let (plan, _cost) = self.decide_cu(state, x0, y0, log2_cb_size, ct_depth);

        // Winner-only RDOQ finalize: discard the hard-quant trial recon, then
        // re-code the chosen plan in decoder order with RDOQ, rebuilding coeffs/
        // recon/CBFs. Search decided the structure; this only refines coding.
        self.overlay.clear();
        let plan = self.finalize_cu(state, plan, x0, y0, log2_cb_size, lambda);

        let final_commit_timer = StillSearchLedger::start_timer();
        self.overlay.commit_to_frame(&mut state.frame);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::FinalCommit, final_commit_timer);
        self.overlay.clear();
        self.workspace.ledger.bump(WorkBucket::FinalCommit);
        // One plan->syntax materialization per CTU (emit::cu_node below).
        self.workspace.ledger.bump(WorkBucket::Writer);
        state.stats.final_rdoq_blocks += self.workspace.ledger.calls(WorkBucket::Rdoq);
        let writer_timer = StillSearchLedger::start_timer();
        let cu = emit::cu_node(plan, &self.workspace.coeffs);
        self.workspace
            .ledger
            .finish_timer(WorkBucket::Writer, writer_timer);
        self.workspace
            .ledger
            .merge_into(&mut state.stats.stillsearch_ledger);
        self.workspace
            .ledger
            .merge_wall_ns_into(&mut state.stats.stillsearch_ledger_ns);
        cu
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// Predict a block into caller-owned `dst` (`size*size`, stride `size`),
    /// reading the committed frame immutably with overlay-first reference
    /// samples and tile-boundary clamping. Never mutates the shared frame.
    pub(super) fn predict_into(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: bpg_hevc_decode::hevc::slice::IntraPredMode,
        dst: &mut [u16],
    ) {
        let size = 1usize << log2_size;
        let tile_bounds = state.tile_clamp_bounds(x0, y0, c_idx);
        let overlay = &self.overlay;
        bpg_hevc_decode::hevc::intra::predict_intra_into_with_reader(
            &state.frame,
            x0,
            y0,
            log2_size,
            mode,
            c_idx,
            true,
            dst,
            size,
            |c, rx, ry| {
                if let Some((tx0, ty0, tx1, ty1)) = tile_bounds {
                    if rx < tx0 || rx >= tx1 || ry < ty0 || ry >= ty1 {
                        return Some(bpg_hevc_decode::hevc::UNINIT_SAMPLE);
                    }
                }
                overlay.sample(c, rx, ry)
            },
        );
    }
}

impl<S, O> StillSearchDepth<S, O>
where
    S: CtuSourceCache,
    O: OverlayCache,
{
    /// 8-bit prediction wrapper. The decoder primitive is still u16-oriented;
    /// this narrows immediately into caller-owned u8 storage so the rest of the
    /// 8-bit analysis path (distortion, residual, recon, overlay) stays u8.
    pub(super) fn predict_into_u8(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        mode: bpg_hevc_decode::hevc::slice::IntraPredMode,
        dst: &mut [u8],
        tmp_u16: &mut Vec<u16>,
    ) {
        let size = 1usize << log2_size;
        let n = size * size;
        tmp_u16.clear();
        tmp_u16.resize(n, 0);
        self.predict_into(state, x0, y0, log2_size, c_idx, mode, tmp_u16);
        debug_assert!(dst.len() >= n);
        for i in 0..n {
            dst[i] = tmp_u16[i].min(u8::MAX as u16) as u8;
        }
    }
}
