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
    pub(super) c_idx: u8,
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(super) struct ReconPatch16 {
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
}

#[derive(Default, Debug)]
pub(super) struct ReconOverlay16 {
    patches: Vec<ReconPatch16>,
}

impl ReconOverlay8 {
    pub(super) fn clear(&mut self) {
        self.patches.clear();
    }

    pub(super) fn push(&mut self, patch: ReconPatch8) {
        self.patches.push(patch);
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
        self.patches.push(ReconPatch8 {
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
        self.patches.push(ReconPatch8 {
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

impl ReconOverlay16 {
    pub(super) fn clear(&mut self) {
        self.patches.clear();
    }

    pub(super) fn push(&mut self, patch: ReconPatch16) {
        self.patches.push(patch);
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
        self.patches.push(ReconPatch16 {
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
}

impl OverlayCache for ReconOverlay8 {
    type Saved = Vec<ReconPatch8>;
    fn clear(&mut self) {
        ReconOverlay8::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay8::sample(self, c_idx, x, y).map(u16::from)
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
}

impl OverlayCache for ReconOverlay16 {
    type Saved = Vec<ReconPatch16>;
    fn clear(&mut self) {
        ReconOverlay16::clear(self);
    }
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> Option<u16> {
        ReconOverlay16::sample(self, c_idx, x, y)
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
}
