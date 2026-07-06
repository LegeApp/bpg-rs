//! Contiguous per-CTU reconstruction canvas (speed-parity audit #6,
//! `canvas-design.md` phases 1+2).
//!
//! The trial-reconstruction store for the CTU search: one contiguous `u16`
//! working buffer per component covering
//! `[x0c-1, x0c+2*CTB) × [y0c-1, y0c+2*CTB)` (chroma scaled by subsampling).
//! On [`reset_from_ctu`](CanvasOverlay::reset_from_ctu) the canvas is filled
//! with `UNINIT_SAMPLE` — which **is** the reference-availability sentinel
//! (see `depth.rs::maybe_set`) — and the committed background actually
//! reachable by intra reads is loaded: the top border row (`y0c-1`, full
//! canvas width) and the left border column (`x0c-1`, CTU height).
//!
//! Trials push candidate recon rectangles; the winning branch commits once
//! to `state.frame`. Loser rollback uses an undo journal (this replaced the
//! retired patch-stack overlays, whose contract it reproduces exactly):
//!
//! - `push_block` journals the rectangle's previous canvas content, then
//!   writes the new samples;
//! - `truncate(mark)` walks the journal backwards restoring `old`;
//! - `detach_from(mark)` additionally captures each rectangle's current
//!   content (redo) into [`CanvasSaved`];
//! - `reattach(saved)` re-applies the redo rectangles.
//!
//! Equivalence to the committed frame: the canvas state after any call
//! sequence equals the frame-plus-journal merge at every in-canvas
//! coordinate, and `sample()` returning `Some(background)` for never-pushed
//! positions is value-identical to a frame read because the background was
//! loaded from that very frame content (including `UNINIT`).

use bpg_hevc_decode::DecodedFrame;
use bpg_hevc_decode::hevc::UNINIT_SAMPLE;

const CTB: usize = 64;

#[derive(Default)]
struct CanvasPlane {
    buf: Vec<u16>,
    stride: usize,
    rows: usize,
    /// Plane coordinates of `buf[0]` (the CTU origin minus one in each
    /// axis; may be -1 at the picture edge, where that row/column simply
    /// stays UNINIT and is never read).
    origin_x: i64,
    origin_y: i64,
    /// Background copies of canvas row 0 and column 0 (the only non-UNINIT
    /// background), re-applied by `clear()`.
    bg_top: Vec<u16>,
    bg_left: Vec<u16>,
}

impl CanvasPlane {
    #[inline]
    fn coords(&self, x: u32, y: u32) -> Option<(usize, usize)> {
        let cx = x as i64 - self.origin_x;
        let cy = y as i64 - self.origin_y;
        if cx >= 0 && (cx as usize) < self.stride && cy >= 0 && (cy as usize) < self.rows {
            Some((cx as usize, cy as usize))
        } else {
            None
        }
    }

    fn reset(&mut self, frame: &DecodedFrame, c_idx: u8, ctu_x: u32, ctu_y: u32) {
        let (sx, sy) = bpg_hevc_decode::hevc::tile::plane_shifts(c_idx, frame.chroma_format);
        let cw = CTB >> sx;
        let ch = CTB >> sy;
        let stride = 1 + 2 * cw;
        let rows = 1 + 2 * ch;
        self.stride = stride;
        self.rows = rows;
        self.origin_x = ((ctu_x >> sx) as i64) - 1;
        self.origin_y = ((ctu_y >> sy) as i64) - 1;
        self.buf.clear();
        self.buf.resize(stride * rows, UNINIT_SAMPLE);

        // Load the committed background reachable by intra reads: the top
        // border row (canvas row 0, spanning the whole canvas width so
        // top-right references of row-top blocks resolve) and the left
        // border column (canvas column 0, CTU height; below-CTU rows are
        // UNINIT in the frame too).
        let (plane, pstride) = frame.plane(c_idx);
        let plane_h = if pstride == 0 {
            0
        } else {
            plane.len() / pstride
        };
        if self.origin_y >= 0 && (self.origin_y as usize) < plane_h {
            let py = self.origin_y as usize;
            let px0 = self.origin_x.max(0) as usize;
            let px1 = ((self.origin_x + stride as i64).min(pstride as i64)).max(0) as usize;
            if px0 < px1 {
                let dst0 = (px0 as i64 - self.origin_x) as usize;
                self.buf[dst0..dst0 + (px1 - px0)]
                    .copy_from_slice(&plane[py * pstride + px0..py * pstride + px1]);
            }
        }
        if self.origin_x >= 0 && (self.origin_x as usize) < pstride {
            let px = self.origin_x as usize;
            let py0 = (self.origin_y + 1).max(0) as usize;
            let py1 = ((self.origin_y + 1 + ch as i64).min(plane_h as i64)).max(0) as usize;
            for py in py0..py1 {
                let cy = (py as i64 - self.origin_y) as usize;
                self.buf[cy * stride] = plane[py * pstride + px];
            }
        }

        self.bg_top.clear();
        self.bg_top.extend_from_slice(&self.buf[..stride]);
        self.bg_left.clear();
        self.bg_left
            .extend((0..rows).map(|cy| self.buf[cy * stride]));
    }

    /// Restore the background state (UNINIT + saved border strips) without
    /// touching the frame. No-op geometry until the first `reset`.
    fn clear_to_background(&mut self) {
        if self.stride == 0 {
            return;
        }
        for v in self.buf.iter_mut() {
            *v = UNINIT_SAMPLE;
        }
        self.buf[..self.stride].copy_from_slice(&self.bg_top);
        for (cy, &v) in self.bg_left.iter().enumerate() {
            self.buf[cy * self.stride] = v;
        }
    }
}

struct JournalEntry {
    c_idx: u8,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    /// Canvas content of the rectangle before this push (row-major).
    old: Vec<u16>,
}

/// Owned buffer detached by [`CanvasOverlay::detach_from`]: the losing/
/// kept-aside branch's redo rectangles, re-applied by
/// [`CanvasOverlay::reattach`]. May be dropped without reattaching.
pub(super) type CanvasSaved = Vec<RedoPatch>;

/// A detached (kept-aside) rectangle: the content the push had written,
/// re-applied by `reattach`.
pub(super) struct RedoPatch {
    c_idx: u8,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u16>,
}

#[derive(Default)]
pub(super) struct CanvasOverlay {
    planes: [CanvasPlane; 3],
    n_comp: usize,
    journal: Vec<JournalEntry>,
}

impl CanvasOverlay {
    /// Copy a journal/redo rectangle between the canvas and a side buffer.
    /// `to_canvas`: write `data` into the canvas; else fill `data` from it.
    fn rect_copy(
        plane: &mut CanvasPlane,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        data: &mut [u16],
        to_canvas: bool,
    ) {
        let (cx, cy) = plane
            .coords(x, y)
            .expect("canvas rect outside canvas bounds");
        debug_assert!(cx + w as usize <= plane.stride && cy + h as usize <= plane.rows);
        for row in 0..h as usize {
            let off = (cy + row) * plane.stride + cx;
            let line = &mut data[row * w as usize..(row + 1) * w as usize];
            if to_canvas {
                plane.buf[off..off + w as usize].copy_from_slice(line);
            } else {
                line.copy_from_slice(&plane.buf[off..off + w as usize]);
            }
        }
    }

    fn push_rect(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32) -> usize {
        let mut old = vec![0u16; (width * height) as usize];
        Self::rect_copy(
            &mut self.planes[c_idx as usize],
            x,
            y,
            width,
            height,
            &mut old,
            false,
        );
        self.journal.push(JournalEntry {
            c_idx,
            x,
            y,
            width,
            height,
            old,
        });
        self.journal.len() - 1
    }
}

impl CanvasOverlay {
    /// Rebase the canvas onto a new CTU: reload the committed background
    /// strips from the frame and clear the journal.
    pub(super) fn reset_from_ctu(&mut self, frame: &DecodedFrame, x0: u32, y0: u32) {
        self.n_comp = if frame.chroma_format == 0 { 1 } else { 3 };
        for c_idx in 0..self.n_comp {
            self.planes[c_idx].reset(frame, c_idx as u8, x0, y0);
        }
        self.journal.clear();
    }

    /// Restore the background state (drop all pushed trial content) without
    /// touching the frame.
    pub(super) fn clear(&mut self) {
        for plane in self.planes[..self.n_comp].iter_mut() {
            plane.clear_to_background();
        }
        self.journal.clear();
    }

    /// One merged frame+trials sample, or `None` outside the canvas (the
    /// caller falls back to a committed-frame read).
    pub(super) fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        let plane = &self.planes[c_idx as usize];
        plane
            .coords(x, y)
            .map(|(cx, cy)| plane.buf[cy * plane.stride + cx])
    }

    /// Overlay `dst.len()` samples from row `y`, starting at `x0`, onto an
    /// already frame-initialized destination buffer.
    pub(super) fn overlay_row_samples(&self, c_idx: u8, y: u32, x0: u32, dst: &mut [u16]) {
        // Every in-canvas position equals the frame+trials merge, so a
        // blanket overwrite of the in-canvas span is value-identical to
        // overlaying trial content onto a frame-initialized destination.
        let plane = &self.planes[c_idx as usize];
        for (i, sample) in dst.iter_mut().enumerate() {
            if let Some((cx, cy)) = plane.coords(x0 + i as u32, y) {
                *sample = plane.buf[cy * plane.stride + cx];
            }
        }
    }

    /// Column variant of [`overlay_row_samples`](Self::overlay_row_samples).
    pub(super) fn overlay_col_samples(&self, c_idx: u8, x: u32, y0: u32, dst: &mut [u16]) {
        let plane = &self.planes[c_idx as usize];
        for (i, sample) in dst.iter_mut().enumerate() {
            if let Some((cx, cy)) = plane.coords(x, y0 + i as u32) {
                *sample = plane.buf[cy * plane.stride + cx];
            }
        }
    }

    /// Fast path: read `dst.len()` frame+trials *merged* samples from row `y`
    /// starting at `x0` in one contiguous copy. Returns `false` when the span
    /// leaves the canvas (the caller then frame-initializes `dst` and runs
    /// [`overlay_row_samples`](Self::overlay_row_samples)).
    pub(super) fn read_row_merged(&self, c_idx: u8, y: u32, x0: u32, dst: &mut [u16]) -> bool {
        let plane = &self.planes[c_idx as usize];
        let Some((cx, cy)) = plane.coords(x0, y) else {
            return false;
        };
        if cx + dst.len() > plane.stride {
            return false;
        }
        let off = cy * plane.stride + cx;
        dst.copy_from_slice(&plane.buf[off..off + dst.len()]);
        true
    }

    /// Column variant of [`read_row_merged`](Self::read_row_merged).
    pub(super) fn read_col_merged(&self, c_idx: u8, x: u32, y0: u32, dst: &mut [u16]) -> bool {
        let plane = &self.planes[c_idx as usize];
        let Some((cx, cy)) = plane.coords(x, y0) else {
            return false;
        };
        if cy + dst.len() > plane.rows {
            return false;
        }
        for (i, sample) in dst.iter_mut().enumerate() {
            *sample = plane.buf[(cy + i) * plane.stride + cx];
        }
        true
    }

    /// Record the `width * height` reconstructed `samples` (row-major) as a
    /// new trial rectangle. Only winner materialization writes recon; cheap
    /// trials are non-committing scratch evaluations.
    pub(super) fn push_block(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u16],
    ) {
        debug_assert_eq!(samples.len(), (width * height) as usize);
        self.push_rect(c_idx, x, y, width, height);
        let mut data = samples.to_vec();
        Self::rect_copy(
            &mut self.planes[c_idx as usize],
            x,
            y,
            width,
            height,
            &mut data,
            true,
        );
    }

    /// [`push_block`](Self::push_block) from 8-bit samples, widening in place.
    pub(super) fn push_block_u8(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u8],
    ) {
        debug_assert_eq!(samples.len(), (width * height) as usize);
        self.push_rect(c_idx, x, y, width, height);
        let plane = &mut self.planes[c_idx as usize];
        let (cx, cy) = plane
            .coords(x, y)
            .expect("canvas rect outside canvas bounds");
        for row in 0..height as usize {
            let off = (cy + row) * plane.stride + cx;
            let src = &samples[row * width as usize..(row + 1) * width as usize];
            for (dst, &s) in plane.buf[off..off + width as usize].iter_mut().zip(src) {
                *dst = u16::from(s);
            }
        }
    }

    /// Journal position token for [`truncate`](Self::truncate) /
    /// [`detach_from`](Self::detach_from) rollback.
    pub(super) fn mark(&self) -> usize {
        self.journal.len()
    }

    /// Drop every trial rectangle pushed after `mark` (loser rollback),
    /// restoring the canvas content beneath them.
    pub(super) fn truncate(&mut self, mark: usize) {
        while self.journal.len() > mark {
            let mut e = self.journal.pop().unwrap();
            Self::rect_copy(
                &mut self.planes[e.c_idx as usize],
                e.x,
                e.y,
                e.width,
                e.height,
                &mut e.old,
                true,
            );
        }
    }

    /// Detach every rectangle pushed after `mark` into an owned buffer
    /// (keep-aside rollback), leaving the canvas at `mark`. The result is
    /// either dropped or [`reattach`](Self::reattach)ed at the same `mark`.
    pub(super) fn detach_from(&mut self, mark: usize) -> CanvasSaved {
        let mut redo: Vec<RedoPatch> = Vec::with_capacity(self.journal.len().saturating_sub(mark));
        while self.journal.len() > mark {
            let mut e = self.journal.pop().unwrap();
            let mut samples = vec![0u16; (e.width * e.height) as usize];
            Self::rect_copy(
                &mut self.planes[e.c_idx as usize],
                e.x,
                e.y,
                e.width,
                e.height,
                &mut samples,
                false,
            );
            Self::rect_copy(
                &mut self.planes[e.c_idx as usize],
                e.x,
                e.y,
                e.width,
                e.height,
                &mut e.old,
                true,
            );
            redo.push(RedoPatch {
                c_idx: e.c_idx,
                x: e.x,
                y: e.y,
                width: e.width,
                height: e.height,
                samples,
            });
        }
        redo.reverse();
        redo
    }

    /// Re-apply rectangles previously [`detach_from`](Self::detach_from)ed,
    /// journaling fresh undo content.
    pub(super) fn reattach(&mut self, saved: CanvasSaved) {
        for mut patch in saved {
            let mut old = vec![0u16; (patch.width * patch.height) as usize];
            Self::rect_copy(
                &mut self.planes[patch.c_idx as usize],
                patch.x,
                patch.y,
                patch.width,
                patch.height,
                &mut old,
                false,
            );
            self.journal.push(JournalEntry {
                c_idx: patch.c_idx,
                x: patch.x,
                y: patch.y,
                width: patch.width,
                height: patch.height,
                old,
            });
            Self::rect_copy(
                &mut self.planes[patch.c_idx as usize],
                patch.x,
                patch.y,
                patch.width,
                patch.height,
                &mut patch.samples,
                true,
            );
        }
    }

    /// Copy the coded CTU rectangle into the committed frame — the single
    /// point where trial reconstruction reaches shared state.
    pub(super) fn commit_to_frame(&self, frame: &mut DecodedFrame) {
        for c_idx in 0..self.n_comp {
            let plane = &self.planes[c_idx];
            let (dst, pstride) = frame.plane_mut(c_idx as u8);
            let plane_h = if pstride == 0 { 0 } else { dst.len() / pstride };
            // The coded CTU rectangle is canvas rows/cols 1..=CTU size;
            // clamp to the plane. Never-coded positions inside the rect hold
            // UNINIT on both sides, so the full-rect copy is value-safe.
            let ctu_w = (plane.stride - 1) / 2;
            let ctu_h = (plane.rows - 1) / 2;
            let px0 = plane.origin_x + 1;
            let py0 = plane.origin_y + 1;
            let px1 = (px0 + ctu_w as i64).min(pstride as i64);
            let py1 = (py0 + ctu_h as i64).min(plane_h as i64);
            for py in py0..py1 {
                let cy = (py - plane.origin_y) as usize;
                let cx0 = (px0 - plane.origin_x) as usize;
                let n = (px1 - px0) as usize;
                let src = &plane.buf[cy * plane.stride + cx0..cy * plane.stride + cx0 + n];
                dst[py as usize * pstride + px0 as usize..py as usize * pstride + px0 as usize + n]
                    .copy_from_slice(src);
            }
        }
    }
}
