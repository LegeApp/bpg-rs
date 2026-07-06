//! StillSearch depth dispatch and CTU entrypoint.

use crate::contexts::Contexts;
use crate::encoder::Encoder;
use crate::encoder::syntax::CuNode;

use super::canvas::CanvasOverlay;
use super::emit;
use super::ledger::{StillSearchLedger, WorkBucket};
use super::source::{CtuSource8, CtuSource16, CtuSourceCache};
use super::workspace::CtuWorkspace;

pub(in crate::encoder) struct StillSearch {
    imp: StillSearchImpl,
}

enum StillSearchImpl {
    EightBit(StillSearchDepth<CtuSource8>),
    HighBit(StillSearchDepth<CtuSource16>),
}

pub(super) struct StillSearchDepth<S> {
    pub(super) workspace: CtuWorkspace,
    pub(super) source: S,
    pub(super) overlay: CanvasOverlay,
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

impl<S> Default for StillSearchDepth<S>
where
    S: Default,
{
    fn default() -> Self {
        Self {
            workspace: CtuWorkspace::default(),
            source: S::default(),
            overlay: CanvasOverlay::default(),
        }
    }
}

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
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
        self.overlay.reset_from_ctu(&state.frame, x0, y0);

        let (plan, _cost) = self.decide_cu(state, x0, y0, log2_cb_size, ct_depth);

        // Winner-only RDOQ finalize: discard the hard-quant trial recon, then
        // re-code the chosen plan in decoder order with RDOQ, rebuilding coeffs/
        // recon/CBFs. Search decided the structure; this only refines coding.
        self.overlay.clear();
        // Re-seed the evolving trial context from the CTU-entry writer state
        // for the decoder-order walk. With `ctx_evolve_finalize`, each
        // finalized block commits its syntax so the final RDOQ rate model
        // prices from per-block contexts close to the real writer's (x265
        // parity) instead of frozen CTU-entry states.
        let prev_commit = self.workspace.commit_ctx;
        let prev_skip = self.workspace.commit_ctx_skip_chroma;
        self.workspace.price_cur = self.workspace.price_base.clone();
        self.workspace.commit_ctx = super::env::ctx_evolve_finalize();
        self.workspace.commit_ctx_skip_chroma = false;
        let plan = self.finalize_cu(state, plan, x0, y0, log2_cb_size);
        self.workspace.commit_ctx = prev_commit;
        self.workspace.commit_ctx_skip_chroma = prev_skip;

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
        state.stats.substage_border_ns = state
            .stats
            .substage_border_ns
            .saturating_add(self.workspace.substage.border_ns);
        state.stats.substage_predict_ns = state
            .stats
            .substage_predict_ns
            .saturating_add(self.workspace.substage.predict_ns);
        state.stats.substage_forward_xform_ns = state
            .stats
            .substage_forward_xform_ns
            .saturating_add(self.workspace.substage.forward_xform_ns);
        state.stats.substage_quant_ns = state
            .stats
            .substage_quant_ns
            .saturating_add(self.workspace.substage.quant_ns);
        state.stats.substage_recon_dist_ns = state
            .stats
            .substage_recon_dist_ns
            .saturating_add(self.workspace.substage.recon_dist_ns);
        state.stats.substage_residual_price_ns = state
            .stats
            .substage_residual_price_ns
            .saturating_add(self.workspace.substage.residual_price_ns);
        state.stats.substage_calls += self.workspace.substage.calls;
        state.stats.tu_split_early_terminations += self.workspace.tu_split_early_terminations;
        state.stats.tu_split_bound_aborts += self.workspace.tu_split_bound_aborts;
        for i in 0..7 {
            state.stats.tu_leaf_by_log2[i] += self.workspace.tu_leaf_by_log2[i];
            state.stats.tu_split_by_log2[i] += self.workspace.tu_split_by_log2[i];
        }
        cu
    }
}

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
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

impl<S> StillSearchDepth<S>
where
    S: CtuSourceCache,
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
        crate::primitives::narrow_u16_to_u8(tmp_u16, dst, n);
    }

    /// Build the intra reference borders (unfiltered + filtered, plus the center
    /// index and bit depth) for one block, reading overlay-first with
    /// tile-boundary clamping. These borders are *identical* across all candidate
    /// intra modes for a block as long as the overlay is unchanged, so callers
    /// that try many modes (cheap/exact ranking) should build once via this and
    /// reuse with [`predict_exact_from_refs_u8`] instead of rebuilding per mode.
    pub(super) fn build_intra_refs_for_block(
        &self,
        state: &Encoder<'_>,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
    ) -> IntraRefs {
        let tile_bounds = state.tile_clamp_bounds(x0, y0, c_idx);
        let (uf, ft, center, bit_depth) =
            self.build_intra_refs_bulk_overlay(&state.frame, x0, y0, log2_size, c_idx, tile_bounds);
        IntraRefs {
            uf,
            ft,
            center,
            bit_depth,
        }
    }

    /// Predict one block from prebuilt [`IntraRefs`] into `dst`. Pure function of
    /// the refs + mode; does no overlay reads.
    #[inline]
    pub(super) fn predict_exact_from_refs_u8(
        &self,
        refs: &IntraRefs,
        log2_size: u8,
        c_idx: u8,
        mode: bpg_hevc_decode::hevc::slice::IntraPredMode,
        dst: &mut [u8],
    ) {
        let mode_u8 = mode.as_u8();
        let size = 1usize << log2_size;
        let border = if bpg_hevc_decode::hevc::intra::reference_filter_applies(size, mode_u8) {
            &refs.ft
        } else {
            &refs.uf
        };
        match mode {
            bpg_hevc_decode::hevc::slice::IntraPredMode::Planar => {
                crate::primitives::pred_planar_u8(
                    dst,
                    border,
                    refs.center,
                    log2_size,
                    c_idx,
                    refs.bit_depth,
                )
            }
            bpg_hevc_decode::hevc::slice::IntraPredMode::Dc => crate::primitives::pred_dc_u8(
                dst,
                border,
                refs.center,
                log2_size,
                c_idx,
                refs.bit_depth,
            ),
            _ => crate::primitives::pred_angular_u8(
                dst,
                border,
                refs.center,
                log2_size,
                c_idx,
                mode_u8,
                refs.bit_depth,
            ),
        }
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn build_intra_refs_bulk_overlay(
        &self,
        frame: &bpg_hevc_decode::DecodedFrame,
        x0: u32,
        y0: u32,
        log2_size: u8,
        c_idx: u8,
        tile_bounds: Option<(u32, u32, u32, u32)>,
    ) -> (
        [i32; bpg_hevc_decode::hevc::intra::INTRA_BORDER_LEN],
        [i32; bpg_hevc_decode::hevc::intra::INTRA_BORDER_LEN],
        usize,
        u8,
    ) {
        use bpg_hevc_decode::hevc::UNINIT_SAMPLE;
        use bpg_hevc_decode::hevc::intra::{
            INTRA_BORDER_CENTER, INTRA_BORDER_LEN, MAX_INTRA_PRED_BLOCK_SIZE,
            apply_reference_filter,
        };

        let size = 1u32 << log2_size;
        let bit_depth = frame.bit_depth;
        let center = INTRA_BORDER_CENTER;
        let default_val = 1i32 << (bit_depth - 1);
        let (plane, stride) = frame.plane(c_idx);
        let (frame_w, frame_h) = frame.component_dims(c_idx);
        let avail_left = x0 > 0;
        let avail_top = y0 > 0;
        let mut unfiltered = [0i32; INTRA_BORDER_LEN];
        let mut avail = [false; 4 * MAX_INTRA_PRED_BLOCK_SIZE + 1];
        let mut avail_count = 0u32;
        let total_samples = 4 * size + 1;

        #[inline]
        fn in_tile(tile: Option<(u32, u32, u32, u32)>, x: u32, y: u32) -> bool {
            tile.is_none_or(|(tx0, ty0, tx1, ty1)| x >= tx0 && x < tx1 && y >= ty0 && y < ty1)
        }

        #[inline]
        fn read_plane_sample(plane: &[u16], stride: usize, x: u32, y: u32) -> u16 {
            plane
                .get(y as usize * stride + x as usize)
                .copied()
                .unwrap_or(0)
        }

        #[inline]
        fn maybe_set(
            border: &mut [i32],
            avail: &mut [bool],
            avail_count: &mut u32,
            idx: usize,
            raw: u16,
        ) -> bool {
            if raw != UNINIT_SAMPLE {
                border[idx] = raw as i32;
                avail[idx] = true;
                *avail_count += 1;
                true
            } else {
                false
            }
        }

        let read_one = |rx: u32, ry: u32| -> u16 {
            if !in_tile(tile_bounds, rx, ry) {
                return UNINIT_SAMPLE;
            }
            self.overlay
                .sample(c_idx, rx, ry)
                .unwrap_or_else(|| read_plane_sample(plane, stride, rx, ry))
        };

        // Top-left corner, with the same top/left fallback order as the decoder
        // helper. Tile-clamped samples are treated as unavailable.
        let mut corner_avail = false;
        if avail_left && avail_top {
            corner_avail = maybe_set(
                &mut unfiltered,
                &mut avail,
                &mut avail_count,
                center,
                read_one(x0 - 1, y0 - 1),
            );
        }
        if !corner_avail && avail_top {
            corner_avail = maybe_set(
                &mut unfiltered,
                &mut avail,
                &mut avail_count,
                center,
                read_one(x0, y0 - 1),
            );
        }
        if !corner_avail && avail_left {
            corner_avail = maybe_set(
                &mut unfiltered,
                &mut avail,
                &mut avail_count,
                center,
                read_one(x0 - 1, y0),
            );
        }
        if !corner_avail {
            unfiltered[center] = default_val;
        }

        // Top + above-right: start from the committed frame row in one pass,
        // then overlay all intersecting patches with a single patch-stack scan.
        if avail_top {
            let top_count = (2 * size).min(frame_w.saturating_sub(x0)) as usize;
            let mut row = [UNINIT_SAMPLE; 2 * MAX_INTRA_PRED_BLOCK_SIZE];
            if top_count > 0 {
                // Canvas fast path: one contiguous merged read. Fallback:
                // frame-initialize, then overlay the patch stack.
                if !self
                    .overlay
                    .read_row_merged(c_idx, y0 - 1, x0, &mut row[..top_count])
                {
                    let row_base = (y0 - 1) as usize * stride + x0 as usize;
                    for (i, dst) in row[..top_count].iter_mut().enumerate() {
                        *dst = plane.get(row_base + i).copied().unwrap_or(0);
                    }
                    self.overlay
                        .overlay_row_samples(c_idx, y0 - 1, x0, &mut row[..top_count]);
                }
                for (i, &raw) in row[..top_count].iter().enumerate() {
                    let rx = x0 + i as u32;
                    if in_tile(tile_bounds, rx, y0 - 1) {
                        maybe_set(
                            &mut unfiltered,
                            &mut avail,
                            &mut avail_count,
                            center + 1 + i,
                            raw,
                        );
                    }
                }
            }
        }

        // Left + below-left: same bulk-overlay path for the reference column.
        if avail_left {
            let left_count = (2 * size).min(frame_h.saturating_sub(y0)) as usize;
            let mut col = [UNINIT_SAMPLE; 2 * MAX_INTRA_PRED_BLOCK_SIZE];
            if left_count > 0 {
                if !self
                    .overlay
                    .read_col_merged(c_idx, x0 - 1, y0, &mut col[..left_count])
                {
                    let left_x = (x0 - 1) as usize;
                    for (i, dst) in col[..left_count].iter_mut().enumerate() {
                        let plane_idx = (y0 as usize + i) * stride + left_x;
                        *dst = plane.get(plane_idx).copied().unwrap_or(0);
                    }
                    self.overlay
                        .overlay_col_samples(c_idx, x0 - 1, y0, &mut col[..left_count]);
                }
                for (i, &raw) in col[..left_count].iter().enumerate() {
                    let ry = y0 + i as u32;
                    if in_tile(tile_bounds, x0 - 1, ry) {
                        maybe_set(
                            &mut unfiltered,
                            &mut avail,
                            &mut avail_count,
                            center - 1 - i,
                            raw,
                        );
                    }
                }
            }
        }

        if avail_count < total_samples {
            substitute_unavailable_refs(
                &mut unfiltered,
                &avail,
                center,
                size as usize,
                default_val,
            );
        }

        let mut filtered = unfiltered;
        if (c_idx == 0 || frame.chroma_format == 3) && size > 4 {
            apply_reference_filter(
                &mut filtered,
                center,
                size as usize,
                c_idx,
                true,
                bit_depth as usize,
            );
        }

        (unfiltered, filtered, center, bit_depth)
    }
}

fn substitute_unavailable_refs(
    border: &mut [i32],
    avail: &[bool],
    center: usize,
    size: usize,
    default_val: i32,
) {
    let mut first_avail_val = None;
    for i in (0..(2 * size)).rev() {
        let idx = center - 1 - i;
        if avail[idx] {
            first_avail_val = Some(border[idx]);
            break;
        }
    }
    if first_avail_val.is_none() && avail[center] {
        first_avail_val = Some(border[center]);
    }
    if first_avail_val.is_none() {
        for i in 0..(2 * size) {
            let idx = center + 1 + i;
            if avail[idx] {
                first_avail_val = Some(border[idx]);
                break;
            }
        }
    }

    let mut current = first_avail_val.unwrap_or(default_val);
    for i in (0..(2 * size)).rev() {
        let idx = center - 1 - i;
        if avail[idx] {
            current = border[idx];
        } else {
            border[idx] = current;
        }
    }
    if avail[center] {
        current = border[center];
    } else {
        border[center] = current;
    }
    for i in 0..(2 * size) {
        let idx = center + 1 + i;
        if avail[idx] {
            current = border[idx];
        } else {
            border[idx] = current;
        }
    }
}

/// Prebuilt intra reference borders for one block, reusable across all candidate
/// modes while the overlay is unchanged. The borders are fixed-size stack arrays
/// (no heap), so copying/holding one is cheap.
#[derive(Clone, Copy)]
pub(super) struct IntraRefs {
    pub(super) uf: [i32; bpg_hevc_decode::hevc::intra::INTRA_BORDER_LEN],
    pub(super) ft: [i32; bpg_hevc_decode::hevc::intra::INTRA_BORDER_LEN],
    pub(super) center: usize,
    pub(super) bit_depth: u8,
}
