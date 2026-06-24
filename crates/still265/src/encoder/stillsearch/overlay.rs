//! Read-through reconstruction overlay model.
//!
//! Prediction reads must consult local candidate overlay first, then the
//! committed frame, then let the HEVC unavailable-edge fallback resolve missing
//! references. The current fixed-DC bridge commits directly because it has no
//! loser branches; future trials use these depth-specialized overlays for
//! non-mutating search.

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
