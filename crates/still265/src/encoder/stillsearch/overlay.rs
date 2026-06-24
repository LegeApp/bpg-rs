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
            samples: samples.iter().map(|&s| s.min(u8::MAX as u16) as u8).collect(),
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
