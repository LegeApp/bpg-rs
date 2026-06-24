//! CTU-local source caches.

use crate::encoder::Encoder;

pub(super) trait CtuSourceCache {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8);
    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16;
}

#[derive(Default)]
pub(super) struct CtuSource8 {
    planes: [CachedPlane8; 3],
}

#[derive(Default)]
pub(super) struct CtuSource16 {
    planes: [CachedPlane16; 3],
}

#[derive(Default)]
struct CachedPlane8 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

#[derive(Default)]
struct CachedPlane16 {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    samples: Vec<u16>,
}

impl CtuSourceCache for CtuSource8 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane.samples.push(
                        state
                            .src_sample(c_idx, px + dx, py + dy)
                            .min(u8::MAX as u16) as u8,
                    );
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y) as u16)
            .unwrap_or(128)
    }
}

impl CtuSourceCache for CtuSource16 {
    fn reset_from_ctu(&mut self, state: &Encoder<'_>, x0: u32, y0: u32, log2_cb_size: u8) {
        for c_idx in 0..3u8 {
            let (sx, sy) = state.plane_shifts(c_idx);
            let px = x0 >> sx;
            let py = y0 >> sy;
            let width = (1u32 << log2_cb_size).div_ceil(1u32 << sx);
            let height = (1u32 << log2_cb_size).div_ceil(1u32 << sy);
            let plane = &mut self.planes[c_idx as usize];
            plane.x = px;
            plane.y = py;
            plane.width = width;
            plane.height = height;
            plane.samples.clear();
            plane.samples.reserve((width * height) as usize);
            for dy in 0..height {
                for dx in 0..width {
                    plane
                        .samples
                        .push(state.src_sample(c_idx, px + dx, py + dy));
                }
            }
        }
    }

    fn sample(&self, c_idx: u8, x: u32, y: u32) -> u16 {
        self.planes
            .get(c_idx as usize)
            .map(|p| p.sample(x, y))
            .unwrap_or(128)
    }
}

impl CachedPlane8 {
    fn sample(&self, x: u32, y: u32) -> u8 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}

impl CachedPlane16 {
    fn sample(&self, x: u32, y: u32) -> u16 {
        if self.width == 0 || self.height == 0 || self.samples.is_empty() {
            return 128;
        }
        let lx = x.saturating_sub(self.x).min(self.width - 1);
        let ly = y.saturating_sub(self.y).min(self.height - 1);
        self.samples[(ly * self.width + lx) as usize]
    }
}
