//! Intra analysis reuse cache.
//!
//! This is the storage boundary for x265-style `--analysis-save` /
//! `--analysis-load` work. The current encoder only records selected decisions;
//! it does not yet load or trust them during search. Keeping this as a separate
//! data model prevents future target-QP/target-size passes from scraping the
//! encoder's transient candidate maps directly.

use crate::{ChromaFormat, Effort};

pub const UNKNOWN_U8: u8 = 0xFF;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CacheDecisionConfidence {
    #[default]
    Unknown = 0,
    Clear = 1,
    Close = 2,
}

impl CacheDecisionConfidence {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisCacheMeta {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
    pub effort: Effort,
    pub qp: u8,
    pub source_hash: u64,
}

#[derive(Clone, Debug)]
pub struct AnalysisCache {
    pub meta: AnalysisCacheMeta,
    pub cu_stride: usize,
    pub cu_height: usize,
    pub pu_stride: usize,
    pub pu_height: usize,
    /// Selected CU depth on the 8x8 min-CB grid.
    pub cu_depth: Vec<u8>,
    /// Selected CU decision confidence on the 8x8 min-CB grid.
    pub cu_confidence: Vec<u8>,
    /// Selected luma intra mode on the 4x4 min-PU grid.
    pub luma_mode: Vec<u8>,
    /// Selected chroma syntax mode index on the 4x4 min-PU grid.
    pub chroma_mode_idx: Vec<u8>,
    /// Selected TU split depth on the 4x4 grid.
    pub tu_depth: Vec<u8>,
}

impl AnalysisCache {
    pub fn new(
        width: u32,
        height: u32,
        bit_depth: u8,
        chroma: ChromaFormat,
        effort: Effort,
        qp: u8,
        source_hash: u64,
    ) -> Self {
        let cu_stride = width.div_ceil(8) as usize;
        let cu_height = height.div_ceil(8) as usize;
        let pu_stride = width.div_ceil(4) as usize;
        let pu_height = height.div_ceil(4) as usize;
        AnalysisCache {
            meta: AnalysisCacheMeta {
                width,
                height,
                bit_depth,
                chroma,
                effort,
                qp,
                source_hash,
            },
            cu_stride,
            cu_height,
            pu_stride,
            pu_height,
            cu_depth: vec![UNKNOWN_U8; cu_stride * cu_height],
            cu_confidence: vec![CacheDecisionConfidence::Unknown.as_u8(); cu_stride * cu_height],
            luma_mode: vec![UNKNOWN_U8; pu_stride * pu_height],
            chroma_mode_idx: vec![UNKNOWN_U8; pu_stride * pu_height],
            tu_depth: vec![UNKNOWN_U8; pu_stride * pu_height],
        }
    }

    pub fn empty_like(&self) -> Self {
        Self::new(
            self.meta.width,
            self.meta.height,
            self.meta.bit_depth,
            self.meta.chroma,
            self.meta.effort,
            self.meta.qp,
            self.meta.source_hash,
        )
    }

    pub fn clear(&mut self) {
        self.cu_depth.fill(UNKNOWN_U8);
        self.cu_confidence
            .fill(CacheDecisionConfidence::Unknown.as_u8());
        self.luma_mode.fill(UNKNOWN_U8);
        self.chroma_mode_idx.fill(UNKNOWN_U8);
        self.tu_depth.fill(UNKNOWN_U8);
    }

    pub fn record_cu_region(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        depth: u8,
        confidence: CacheDecisionConfidence,
    ) {
        let n = (1u32 << log2_size).div_ceil(8);
        let sx = x0 / 8;
        let sy = y0 / 8;
        for dy in 0..n {
            for dx in 0..n {
                let idx = (sy + dy) as usize * self.cu_stride + (sx + dx) as usize;
                if idx < self.cu_depth.len() {
                    self.cu_depth[idx] = depth;
                    self.cu_confidence[idx] = confidence.as_u8();
                }
            }
        }
    }

    pub fn record_leaf_modes(
        &mut self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        luma_mode: u8,
        chroma_mode_idx: u8,
    ) {
        Self::fill_pu_region(
            &mut self.chroma_mode_idx,
            self.pu_stride,
            x0,
            y0,
            log2_size,
            chroma_mode_idx,
        );
        Self::fill_pu_region(
            &mut self.luma_mode,
            self.pu_stride,
            x0,
            y0,
            log2_size,
            luma_mode,
        );
    }

    pub fn record_tu_region(&mut self, x0: u32, y0: u32, log2_size: u8, depth: u8) {
        Self::fill_pu_region(&mut self.tu_depth, self.pu_stride, x0, y0, log2_size, depth);
    }

    fn fill_pu_region(map: &mut [u8], stride: usize, x0: u32, y0: u32, log2_size: u8, value: u8) {
        let n = (1u32 << log2_size) / 4;
        let sx = x0 / 4;
        let sy = y0 / 4;
        for dy in 0..n {
            for dx in 0..n {
                let idx = (sy + dy) as usize * stride + (sx + dx) as usize;
                if idx < map.len() {
                    map[idx] = value;
                }
            }
        }
    }
}
