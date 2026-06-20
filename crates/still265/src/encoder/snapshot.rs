//! Frame and neighbour-map snapshot/restore for the CU/TU search-trial undo
//! mechanism and parallel-CTU analysis.
//!
//! Ported from `bpgenc.c`'s snapshot/restore patterns, used by `build_cu` /
//! `build_tt` to roll back the reconstruction after a trial that did not win.

use std::mem::size_of;

use super::types::{chroma_tb_geom, CodedBlock, FrameSnapshot, MapSnapshot, PlaneSnapshot};
use super::Encoder;

pub(super) fn snapshot_map(
    map: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> MapSnapshot {
    let rows = map.len().div_ceil(stride);
    let x = x.min(stride);
    let y = y.min(rows);
    let width = width.min(stride.saturating_sub(x));
    let height = height.min(rows.saturating_sub(y));
    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = (y + row) * stride + x;
        data.extend_from_slice(&map[start..start + width]);
    }
    MapSnapshot {
        x,
        y,
        width,
        height,
        stride,
        data,
    }
}

pub(super) fn restore_map(map: &mut [u8], snapshot: &MapSnapshot) {
    for row in 0..snapshot.height {
        let src = row * snapshot.width;
        let dst = (snapshot.y + row) * snapshot.stride + snapshot.x;
        map[dst..dst + snapshot.width].copy_from_slice(&snapshot.data[src..src + snapshot.width]);
    }
}

/// Snapshot of the reconstructed-frame samples for a chroma-leaf-cache entry,
/// alongside the per-component coded block data used to restore the chroma mode's
/// transform-tree branch after a non-winning trial.
pub(super) struct ChromaLeafCache {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) log2_size: u8,
    pub(super) chroma_log2: u8,
    pub(super) cb: CodedBlock,
    pub(super) cr: CodedBlock,
    pub(super) cb1: CodedBlock,
    pub(super) cr1: CodedBlock,
    pub(super) frame: FrameSnapshot,
}

/// Result of a full RD chroma-mode evaluation.
pub(super) struct ChromaDecision {
    pub(super) plan: crate::plan::ChromaModePlan,
}

impl<'a> Encoder<'a> {
    pub(super) fn snapshot_plane(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        width: usize,
        height: usize,
    ) -> PlaneSnapshot {
        let (plane_width, plane_height) = self.frame_plane_dims(c_idx);
        let x = (x as usize).min(plane_width);
        let y = (y as usize).min(plane_height);
        let width = width.min(plane_width.saturating_sub(x));
        let height = height.min(plane_height.saturating_sub(y));
        let (plane, stride) = self.frame.plane(c_idx);
        let mut data = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = (y + row) * stride + x;
            data.extend_from_slice(&plane[start..start + width]);
        }
        let snapshot = PlaneSnapshot {
            c_idx,
            x,
            y,
            width,
            height,
            data,
        };
        self.stats.frame_snapshots += 1;
        self.stats.bytes_snapshotted += (snapshot.data.len() * size_of::<u16>()) as u64;
        snapshot
    }

    pub(super) fn restore_plane(&mut self, snapshot: &PlaneSnapshot) {
        self.stats.frame_restores += 1;
        self.stats.bytes_restored += (snapshot.data.len() * size_of::<u16>()) as u64;
        let (plane, stride) = self.frame.plane_mut(snapshot.c_idx);
        for row in 0..snapshot.height {
            let src = row * snapshot.width;
            let dst = (snapshot.y + row) * stride + snapshot.x;
            plane[dst..dst + snapshot.width]
                .copy_from_slice(&snapshot.data[src..src + snapshot.width]);
        }
    }

    pub(super) fn snapshot_frame_region(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
    ) -> FrameSnapshot {
        let t = self.prof.on.then(std::time::Instant::now);
        let size = 1usize << log2_size;
        let mut planes = Vec::with_capacity(3);
        planes.push(self.snapshot_plane(0, x0, y0, size, size));
        if let Some((cx, cy, clog2, count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) {
            let csize = 1usize << clog2;
            let cheight = csize * count as usize;
            planes.push(self.snapshot_plane(1, cx, cy, csize, cheight));
            planes.push(self.snapshot_plane(2, cx, cy, csize, cheight));
        }
        if let Some(t) = t {
            self.prof.snapshot += t.elapsed();
        }
        FrameSnapshot { planes }
    }

    /// Like [`Self::snapshot_frame_region`] but reuses a recycled
    /// [`FrameSnapshot`] (and its per-plane `data` capacity) from the search
    /// scratch pool instead of allocating fresh `Vec`s. Byte-identical content;
    /// the caller should return the snapshot via [`Self::recycle_frame_snapshot`]
    /// once it is no longer needed. The plane count is fixed by `self.cat`, so a
    /// recycled snapshot always has compatible plane slots.
    pub(super) fn snapshot_frame_region_pooled(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
    ) -> FrameSnapshot {
        let t = self.prof.on.then(std::time::Instant::now);
        let mut snap = self
            .search_scratch
            .frame_snapshot_pool
            .pop()
            .unwrap_or_else(|| FrameSnapshot { planes: Vec::new() });
        let size = 1usize << log2_size;
        let mut slot = 0;
        self.fill_pooled_plane(&mut snap, slot, 0, x0, y0, size, size);
        slot += 1;
        if let Some((cx, cy, clog2, count)) = chroma_tb_geom(self.cat, x0, y0, log2_size) {
            let csize = 1usize << clog2;
            let cheight = csize * count as usize;
            self.fill_pooled_plane(&mut snap, slot, 1, cx, cy, csize, cheight);
            slot += 1;
            self.fill_pooled_plane(&mut snap, slot, 2, cx, cy, csize, cheight);
            slot += 1;
        }
        snap.planes.truncate(slot);
        if let Some(t) = t {
            self.prof.snapshot += t.elapsed();
        }
        snap
    }

    /// Fill plane slot `slot` of `snap` with the current frame region, reusing
    /// the slot's existing `data` buffer when present (clearing retains its
    /// capacity). Mirrors [`Self::snapshot_plane`]'s clamping and accounting.
    fn fill_pooled_plane(
        &mut self,
        snap: &mut FrameSnapshot,
        slot: usize,
        c_idx: u8,
        x: u32,
        y: u32,
        width: usize,
        height: usize,
    ) {
        let (plane_width, plane_height) = self.frame_plane_dims(c_idx);
        let x = (x as usize).min(plane_width);
        let y = (y as usize).min(plane_height);
        let width = width.min(plane_width.saturating_sub(x));
        let height = height.min(plane_height.saturating_sub(y));
        if slot >= snap.planes.len() {
            snap.planes.push(PlaneSnapshot {
                c_idx,
                x,
                y,
                width,
                height,
                data: Vec::new(),
            });
        }
        let dst = &mut snap.planes[slot];
        dst.c_idx = c_idx;
        dst.x = x;
        dst.y = y;
        dst.width = width;
        dst.height = height;
        dst.data.clear();
        let (plane, stride) = self.frame.plane(c_idx);
        for row in 0..height {
            let start = (y + row) * stride + x;
            dst.data.extend_from_slice(&plane[start..start + width]);
        }
        self.stats.frame_snapshots += 1;
        self.stats.bytes_snapshotted += (dst.data.len() * size_of::<u16>()) as u64;
    }

    /// Return a pooled snapshot to the free-list for reuse. Capped so an idle
    /// encoder does not retain an unbounded number of region buffers.
    pub(super) fn recycle_frame_snapshot(&mut self, snap: FrameSnapshot) {
        if self.search_scratch.frame_snapshot_pool.len() < 8 {
            self.search_scratch.frame_snapshot_pool.push(snap);
        }
    }

    pub(super) fn restore_frame_region(&mut self, snapshot: &FrameSnapshot) {
        let t = self.prof.on.then(std::time::Instant::now);
        for plane in &snapshot.planes {
            self.restore_plane(plane);
        }
        if let Some(t) = t {
            self.prof.snapshot += t.elapsed();
        }
    }

    pub(super) fn snapshot_mode_region(&mut self, x0: u32, y0: u32, log2_size: u8) -> MapSnapshot {
        let x = (x0 / 4) as usize;
        let y = (y0 / 4) as usize;
        let width = ((1u32 << log2_size) / 4) as usize;
        let height = width;
        let snapshot = snapshot_map(&self.mode_map, self.mode_stride, x, y, width, height);
        self.stats.map_snapshots += 1;
        self.stats.bytes_snapshotted += snapshot.data.len() as u64;
        snapshot
    }

    pub(super) fn restore_mode_region(&mut self, snapshot: &MapSnapshot) {
        self.stats.map_restores += 1;
        self.stats.bytes_restored += snapshot.data.len() as u64;
        restore_map(&mut self.mode_map, snapshot);
    }

    pub(super) fn snapshot_ct_depth_region(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
    ) -> MapSnapshot {
        let x = (x0 / 8) as usize;
        let y = (y0 / 8) as usize;
        let width = (1u32 << log2_size).div_ceil(8) as usize;
        let height = width;
        let snapshot = snapshot_map(
            &self.ct_depth_map,
            self.ct_depth_stride,
            x,
            y,
            width,
            height,
        );
        self.stats.map_snapshots += 1;
        self.stats.bytes_snapshotted += snapshot.data.len() as u64;
        snapshot
    }

    pub(super) fn restore_ct_depth_region(&mut self, snapshot: &MapSnapshot) {
        self.stats.map_restores += 1;
        self.stats.bytes_restored += snapshot.data.len() as u64;
        restore_map(&mut self.ct_depth_map, snapshot);
    }

    pub(super) fn snapshot_tu_depth_region(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
    ) -> MapSnapshot {
        let x = (x0 / 4) as usize;
        let y = (y0 / 4) as usize;
        let width = ((1u32 << log2_size) / 4) as usize;
        let height = width;
        let snapshot = snapshot_map(
            &self.tu_depth_map,
            self.tu_depth_stride,
            x,
            y,
            width,
            height,
        );
        self.stats.map_snapshots += 1;
        self.stats.bytes_snapshotted += snapshot.data.len() as u64;
        snapshot
    }

    pub(super) fn restore_tu_depth_region(&mut self, snapshot: &MapSnapshot) {
        self.stats.map_restores += 1;
        self.stats.bytes_restored += snapshot.data.len() as u64;
        restore_map(&mut self.tu_depth_map, snapshot);
    }
}
