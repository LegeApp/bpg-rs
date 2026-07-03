//! Read-through reconstruction overlay model.
//!
//! Prediction reads must consult local candidate overlay first, then the
//! committed frame, then let the HEVC unavailable-edge fallback resolve missing
//! references. Trial blocks write their reconstructed samples into the overlay
//! (never the shared frame); the winning branch's patches are committed to the
//! frame exactly once via [`ReconOverlay16::commit_to_frame`] /
//! [`ReconOverlay8::commit_to_frame`].
//!
//! Loser branches are discarded by rewinding the patch stack: record a
//! [`mark`](ReconOverlay16::mark) before a candidate, then either
//! [`truncate`](ReconOverlay16::truncate) (drop a later candidate's patches) or
//! [`drain_range`](ReconOverlay16::drain_range) (drop an earlier candidate's
//! patches) so only the winner's recon remains visible to siblings.

use bpg_hevc_decode::DecodedFrame;

#[derive(Clone, Debug)]
pub(super) struct ReconPatch8 {
    /// Identity of the push event that created this patch (see
    /// [`OverlayCache::push_id_at`]). Preserved across detach/reattach.
    pub(super) push_id: u64,
    pub(super) c_idx: u8,
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(super) struct ReconPatch16 {
    pub(super) push_id: u64,
    pub(super) c_idx: u8,
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: Vec<u16>,
}

#[derive(Default, Debug)]
pub(super) struct ReconOverlay8 {
    patches: Vec<ReconPatch8>,
    /// Monotone id assigned to each pushed patch. Together with the patch
    /// stack's length and top-of-prefix id this identifies the stack content
    /// exactly (patches below the top only ever change by being popped and
    /// re-pushed, which mints a fresh id).
    next_push_id: u64,
    /// Bumped only by [`drain_range`](Self::drain_range), the one mutation
    /// that removes mid-stack elements and breaks the stack-identity model.
    drain_epoch: u64,
}

#[derive(Default, Debug)]
pub(super) struct ReconOverlay16 {
    patches: Vec<ReconPatch16>,
    next_push_id: u64,
    drain_epoch: u64,
}

impl ReconOverlay8 {
    pub(super) fn clear(&mut self) {
        self.patches.clear();
    }

    pub(super) fn push(&mut self, mut patch: ReconPatch8) {
        patch.push_id = self.mint_push_id();
        self.patches.push(patch);
    }

    fn mint_push_id(&mut self) -> u64 {
        self.next_push_id += 1;
        self.next_push_id
    }

    /// Record the `width * height` reconstructed `samples` (row-major, given as
    /// `u16` for a uniform interface) as a new overlay patch, narrowing to the
    /// 8-bit native storage. Losslessly exact for 8-bit reconstruction.
    pub(super) fn push_block(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u16],
    ) {
        let push_id = self.mint_push_id();
        self.patches.push(ReconPatch8 {
            push_id,
            c_idx,
            x,
            y,
            width,
            height,
            samples: samples
                .iter()
                .map(|&s| s.min(u8::MAX as u16) as u8)
                .collect(),
        });
    }

    pub(super) fn push_block_u8(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u8],
    ) {
        let push_id = self.mint_push_id();
        self.patches.push(ReconPatch8 {
            push_id,
            c_idx,
            x,
            y,
            width,
            height,
            samples: samples.to_vec(),
        });
    }

    pub(super) fn mark(&self) -> usize {
        self.patches.len()
    }

    pub(super) fn truncate(&mut self, mark: usize) {
        self.patches.truncate(mark);
    }

    pub(super) fn drain_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.patches.len());
        if start < end {
            self.patches.drain(start..end);
            self.drain_epoch += 1;
        }
    }

    /// Detach all patches from `mark` onward into an owned buffer, leaving the
    /// overlay at `mark`. Used to evaluate an alternative candidate on a clean
    /// overlay, then [`reattach`](Self::reattach) the winner if it was the first.
    pub(super) fn split_off(&mut self, mark: usize) -> Vec<ReconPatch8> {
        let mark = mark.min(self.patches.len());
        self.patches.split_off(mark)
    }

    pub(super) fn reattach(&mut self, patches: Vec<ReconPatch8>) {
        self.patches.extend(patches);
    }

    pub(super) fn commit_to_frame(&self, frame: &mut DecodedFrame) {
        for p in &self.patches {
            let (plane, stride) = frame.plane_mut(p.c_idx);
            for ly in 0..p.height as usize {
                for lx in 0..p.width as usize {
                    let idx = (p.y as usize + ly) * stride + p.x as usize + lx;
                    if let Some(dst) = plane.get_mut(idx) {
                        *dst = p.samples[ly * p.width as usize + lx] as u16;
                    }
                }
            }
        }
    }

    pub(super) fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u8> {
        #[cfg(feature = "overlay-probe")]
        {
            OVL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            OVL_ITERS.fetch_add(
                self.patches.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            if c_idx != 0 {
                OVL_CALLS_CHROMA.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        for p in self.patches.iter().rev() {
            if p.c_idx == c_idx && x >= p.x && y >= p.y && x < p.x + p.width && y < p.y + p.height {
                let lx = (x - p.x) as usize;
                let ly = (y - p.y) as usize;
                return p.samples.get(ly * p.width as usize + lx).copied();
            }
        }
        None
    }
}

#[cfg(feature = "overlay-probe")]
pub static OVL_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "overlay-probe")]
pub static OVL_ITERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "overlay-probe")]
pub static OVL_CALLS_CHROMA: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl ReconOverlay16 {
    pub(super) fn clear(&mut self) {
        self.patches.clear();
    }

    pub(super) fn push(&mut self, mut patch: ReconPatch16) {
        patch.push_id = self.mint_push_id();
        self.patches.push(patch);
    }

    fn mint_push_id(&mut self) -> u64 {
        self.next_push_id += 1;
        self.next_push_id
    }

    pub(super) fn push_block(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u16],
    ) {
        let push_id = self.mint_push_id();
        self.patches.push(ReconPatch16 {
            push_id,
            c_idx,
            x,
            y,
            width,
            height,
            samples: samples.to_vec(),
        });
    }

    pub(super) fn mark(&self) -> usize {
        self.patches.len()
    }

    pub(super) fn truncate(&mut self, mark: usize) {
        self.patches.truncate(mark);
    }

    pub(super) fn drain_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.patches.len());
        if start < end {
            self.patches.drain(start..end);
            self.drain_epoch += 1;
        }
    }

    pub(super) fn split_off(&mut self, mark: usize) -> Vec<ReconPatch16> {
        let mark = mark.min(self.patches.len());
        self.patches.split_off(mark)
    }

    pub(super) fn reattach(&mut self, patches: Vec<ReconPatch16>) {
        self.patches.extend(patches);
    }

    pub(super) fn commit_to_frame(&self, frame: &mut DecodedFrame) {
        for p in &self.patches {
            let (plane, stride) = frame.plane_mut(p.c_idx);
            for ly in 0..p.height as usize {
                for lx in 0..p.width as usize {
                    let idx = (p.y as usize + ly) * stride + p.x as usize + lx;
                    if let Some(dst) = plane.get_mut(idx) {
                        *dst = p.samples[ly * p.width as usize + lx];
                    }
                }
            }
        }
    }

    pub(super) fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        for p in self.patches.iter().rev() {
            if p.c_idx == c_idx && x >= p.x && y >= p.y && x < p.x + p.width && y < p.y + p.height {
                let lx = (x - p.x) as usize;
                let ly = (y - p.y) as usize;
                return p.samples.get(ly * p.width as usize + lx).copied();
            }
        }
        None
    }
}

/// CTU-local reconstruction overlay used for non-mutating trials. Trials push
/// candidate recon patches; the winning branch commits once to `state.frame`.
pub(super) trait OverlayCache {
    /// Owned patch buffer detached by [`detach_from`](Self::detach_from).
    type Saved;
    fn clear(&mut self);
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16>;
    /// Overlay `dst.len()` samples from row `y`, starting at `x0`, onto an
    /// already frame-initialized destination buffer. Later patches win, matching
    /// [`sample`](Self::sample)'s reverse-stack lookup, but this scans the patch
    /// stack once instead of once per reference sample.
    fn overlay_row_samples(&self, c_idx: u8, y: u32, x0: u32, dst: &mut [u16]) {
        for (i, sample) in dst.iter_mut().enumerate() {
            if let Some(v) = self.sample(c_idx, x0 + i as u32, y) {
                *sample = v;
            }
        }
    }
    /// Overlay `dst.len()` samples from column `x`, starting at `y0`, onto an
    /// already frame-initialized destination buffer.
    fn overlay_col_samples(&self, c_idx: u8, x: u32, y0: u32, dst: &mut [u16]) {
        for (i, sample) in dst.iter_mut().enumerate() {
            if let Some(v) = self.sample(c_idx, x, y0 + i as u32) {
                *sample = v;
            }
        }
    }
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]);
    fn push_block_u8(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u8],
    ) {
        let widened: Vec<u16> = samples.iter().map(|&s| u16::from(s)).collect();
        self.push_block(c_idx, x, y, width, height, &widened);
    }
    fn mark(&self) -> usize;
    fn truncate(&mut self, mark: usize);
    fn drain_range(&mut self, start: usize, end: usize);
    fn detach_from(&mut self, mark: usize) -> Self::Saved;
    fn reattach(&mut self, saved: Self::Saved);
    fn commit_to_frame(&self, frame: &mut bpg_hevc_decode::DecodedFrame);
    /// Bumped whenever mid-stack patches were removed ([`drain_range`]
    /// invalidates the push-id stack-identity model).
    fn drain_epoch(&self) -> u64;
    /// Push id of the patch at stack index `idx`, or `None` when out of range.
    /// `(drain_epoch, len, push_id_at(len-1))` identifies the stack content
    /// exactly: an element below the top can only change by first popping
    /// everything above it, and any re-push mints a fresh id.
    fn push_id_at(&self, idx: usize) -> Option<u64>;
    /// Whether any patch at stack index `from` or above intersects the intra
    /// reference-border read region of the `size`-wide block at `(x0, y0)` in
    /// component `c_idx`: row `y0-1` over `x ∈ [x0-1, x0+2*size-1]` and column
    /// `x0-1` over `y ∈ [y0-1, y0+2*size-1]` (matching
    /// `fill_border_samples_with_reader`'s reads).
    fn any_border_overlap_since(&self, from: usize, c_idx: u8, x0: u32, y0: u32, size: u32)
    -> bool;
}

/// Shared rectangle-vs-border-strips intersection test for
/// [`OverlayCache::any_border_overlap_since`], in i64 to sidestep edge
/// underflow. Conservative only in never under-reporting overlap.
#[inline]
fn patch_overlaps_border(
    (px, py, pw, ph): (u32, u32, u32, u32),
    x0: u32,
    y0: u32,
    size: u32,
) -> bool {
    let (px, py) = (px as i64, py as i64);
    let (pw, ph) = (pw as i64, ph as i64);
    let (x0, y0, size) = (x0 as i64, y0 as i64, size as i64);
    // Top strip: y == y0-1, x in [x0-1, x0+2*size-1].
    if y0 > 0 && py <= y0 - 1 && y0 - 1 < py + ph && px <= x0 + 2 * size - 1 && px + pw > x0 - 1 {
        return true;
    }
    // Left strip: x == x0-1, y in [y0-1, y0+2*size-1].
    if x0 > 0 && px <= x0 - 1 && x0 - 1 < px + pw && py <= y0 + 2 * size - 1 && py + ph > y0 - 1 {
        return true;
    }
    false
}

impl OverlayCache for ReconOverlay8 {
    type Saved = Vec<ReconPatch8>;
    fn clear(&mut self) {
        ReconOverlay8::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay8::sample(self, c_idx, x, y).map(u16::from)
    }
    fn overlay_row_samples(&self, c_idx: u8, y: u32, x0: u32, dst: &mut [u16]) {
        let x1 = x0.saturating_add(dst.len() as u32);
        for p in &self.patches {
            if p.c_idx != c_idx || y < p.y || y >= p.y.saturating_add(p.height) {
                continue;
            }
            let ix0 = x0.max(p.x);
            let ix1 = x1.min(p.x.saturating_add(p.width));
            if ix0 >= ix1 {
                continue;
            }
            let src_y = (y - p.y) as usize * p.width as usize;
            let src_x = (ix0 - p.x) as usize;
            let dst_x = (ix0 - x0) as usize;
            let n = (ix1 - ix0) as usize;
            for i in 0..n {
                dst[dst_x + i] = u16::from(p.samples[src_y + src_x + i]);
            }
        }
    }
    fn overlay_col_samples(&self, c_idx: u8, x: u32, y0: u32, dst: &mut [u16]) {
        let y1 = y0.saturating_add(dst.len() as u32);
        for p in &self.patches {
            if p.c_idx != c_idx || x < p.x || x >= p.x.saturating_add(p.width) {
                continue;
            }
            let iy0 = y0.max(p.y);
            let iy1 = y1.min(p.y.saturating_add(p.height));
            if iy0 >= iy1 {
                continue;
            }
            let src_x = (x - p.x) as usize;
            let dst_y = (iy0 - y0) as usize;
            let n = (iy1 - iy0) as usize;
            for i in 0..n {
                let src_y = (iy0 - p.y) as usize + i;
                dst[dst_y + i] = u16::from(p.samples[src_y * p.width as usize + src_x]);
            }
        }
    }
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]) {
        ReconOverlay8::push_block(self, c_idx, x, y, width, height, samples);
    }
    fn push_block_u8(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        samples: &[u8],
    ) {
        ReconOverlay8::push_block_u8(self, c_idx, x, y, width, height, samples);
    }
    fn mark(&self) -> usize {
        ReconOverlay8::mark(self)
    }
    fn truncate(&mut self, mark: usize) {
        ReconOverlay8::truncate(self, mark);
    }
    fn drain_range(&mut self, start: usize, end: usize) {
        ReconOverlay8::drain_range(self, start, end);
    }
    fn detach_from(&mut self, mark: usize) -> Self::Saved {
        ReconOverlay8::split_off(self, mark)
    }
    fn reattach(&mut self, saved: Self::Saved) {
        ReconOverlay8::reattach(self, saved);
    }
    fn commit_to_frame(&self, frame: &mut bpg_hevc_decode::DecodedFrame) {
        ReconOverlay8::commit_to_frame(self, frame);
    }
    fn drain_epoch(&self) -> u64 {
        self.drain_epoch
    }
    fn push_id_at(&self, idx: usize) -> Option<u64> {
        self.patches.get(idx).map(|p| p.push_id)
    }
    fn any_border_overlap_since(
        &self,
        from: usize,
        c_idx: u8,
        x0: u32,
        y0: u32,
        size: u32,
    ) -> bool {
        self.patches[from.min(self.patches.len())..]
            .iter()
            .any(|p| {
                p.c_idx == c_idx
                    && patch_overlaps_border((p.x, p.y, p.width, p.height), x0, y0, size)
            })
    }
}

impl OverlayCache for ReconOverlay16 {
    type Saved = Vec<ReconPatch16>;
    fn clear(&mut self) {
        ReconOverlay16::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay16::sample(self, c_idx, x, y)
    }
    fn overlay_row_samples(&self, c_idx: u8, y: u32, x0: u32, dst: &mut [u16]) {
        let x1 = x0.saturating_add(dst.len() as u32);
        for p in &self.patches {
            if p.c_idx != c_idx || y < p.y || y >= p.y.saturating_add(p.height) {
                continue;
            }
            let ix0 = x0.max(p.x);
            let ix1 = x1.min(p.x.saturating_add(p.width));
            if ix0 >= ix1 {
                continue;
            }
            let src_y = (y - p.y) as usize * p.width as usize;
            let src_x = (ix0 - p.x) as usize;
            let dst_x = (ix0 - x0) as usize;
            let n = (ix1 - ix0) as usize;
            dst[dst_x..dst_x + n].copy_from_slice(&p.samples[src_y + src_x..src_y + src_x + n]);
        }
    }
    fn overlay_col_samples(&self, c_idx: u8, x: u32, y0: u32, dst: &mut [u16]) {
        let y1 = y0.saturating_add(dst.len() as u32);
        for p in &self.patches {
            if p.c_idx != c_idx || x < p.x || x >= p.x.saturating_add(p.width) {
                continue;
            }
            let iy0 = y0.max(p.y);
            let iy1 = y1.min(p.y.saturating_add(p.height));
            if iy0 >= iy1 {
                continue;
            }
            let src_x = (x - p.x) as usize;
            let dst_y = (iy0 - y0) as usize;
            let n = (iy1 - iy0) as usize;
            for i in 0..n {
                let src_y = (iy0 - p.y) as usize + i;
                dst[dst_y + i] = p.samples[src_y * p.width as usize + src_x];
            }
        }
    }
    fn push_block(&mut self, c_idx: u8, x: u32, y: u32, width: u32, height: u32, samples: &[u16]) {
        ReconOverlay16::push_block(self, c_idx, x, y, width, height, samples);
    }
    fn mark(&self) -> usize {
        ReconOverlay16::mark(self)
    }
    fn truncate(&mut self, mark: usize) {
        ReconOverlay16::truncate(self, mark);
    }
    fn drain_range(&mut self, start: usize, end: usize) {
        ReconOverlay16::drain_range(self, start, end);
    }
    fn detach_from(&mut self, mark: usize) -> Self::Saved {
        ReconOverlay16::split_off(self, mark)
    }
    fn reattach(&mut self, saved: Self::Saved) {
        ReconOverlay16::reattach(self, saved);
    }
    fn commit_to_frame(&self, frame: &mut bpg_hevc_decode::DecodedFrame) {
        ReconOverlay16::commit_to_frame(self, frame);
    }
    fn drain_epoch(&self) -> u64 {
        self.drain_epoch
    }
    fn push_id_at(&self, idx: usize) -> Option<u64> {
        self.patches.get(idx).map(|p| p.push_id)
    }
    fn any_border_overlap_since(
        &self,
        from: usize,
        c_idx: u8,
        x0: u32,
        y0: u32,
        size: u32,
    ) -> bool {
        self.patches[from.min(self.patches.len())..]
            .iter()
            .any(|p| {
                p.c_idx == c_idx
                    && patch_overlaps_border((p.x, p.y, p.width, p.height), x0, y0, size)
            })
    }
}
