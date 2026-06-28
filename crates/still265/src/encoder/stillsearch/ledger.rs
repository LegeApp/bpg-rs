//! StillSearch work ledger buckets.
//!
//! Buckets read 0 in the current core for one of two reasons:
//! - **Search stage not active for this preset/path.** `ChromaRough` remains
//!   reserved because chroma is DM-only, but `LumaCheap` and `ChromaTrial` are
//!   live buckets: cheap luma ranking and chroma component trials bump them
//!   whenever those paths execute.
//! - **Measured elsewhere.** `Deblock` and `Sao` are frame-level post-passes
//!   timed by `EncodeStats::phase_deblock_us` / `phase_sao_*_us`, not counted
//!   here.
//!
//! `Rdoq` counts winner-only final RDOQ blocks. `RdoqTrial` counts analysis-stage
//! RDOQ trials (close candidate re-evaluation in Slow/Placebo).
//!
//! Broad search/trial screening remains on hard quantization and must not bump
//! these buckets.

use std::time::Instant;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkBucket {
    RoughLuma,
    /// Cheap luma ranking evaluations.
    LumaCheap,
    LumaExact,
    TuLeaf,
    TuSplit,
    NxnRough,
    NxnBatch,
    /// Reserved: chroma is DM-only (no chroma mode search) in the current core.
    ChromaRough,
    /// Chroma component evaluations for luma-mode/TU candidates.
    ChromaTrial,
    Rdoq,
    /// Analysis-stage RDOQ trials (separate from winner-only final RDOQ).
    RdoqTrial,
    ResidualPrice,
    FinalCommit,
    Writer,
    /// Reserved: deblock is a frame-level pass timed by `phase_deblock_us`.
    Deblock,
    /// Reserved: SAO is a frame-level pass timed by `phase_sao_*_us`.
    Sao,
}

impl WorkBucket {
    pub(super) const COUNT: usize = 16;

    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct StillSearchLedger {
    calls: [u64; WorkBucket::COUNT],
    wall_ns: [u64; WorkBucket::COUNT],
}

impl StillSearchLedger {
    pub(super) fn clear_ctu(&mut self) {
        self.calls = [0; WorkBucket::COUNT];
        self.wall_ns = [0; WorkBucket::COUNT];
    }

    pub(super) fn bump(&mut self, bucket: WorkBucket) {
        self.calls[bucket.idx()] += 1;
    }

    pub(super) fn calls(&self, bucket: WorkBucket) -> u64 {
        self.calls[bucket.idx()]
    }

    #[inline]
    pub(super) fn start_timer() -> Option<Instant> {
        super::env::profile_enabled().then(Instant::now)
    }

    #[inline]
    pub(super) fn finish_timer(&mut self, bucket: WorkBucket, start: Option<Instant>) {
        if let Some(start) = start {
            let ns = start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
            self.wall_ns[bucket.idx()] = self.wall_ns[bucket.idx()].saturating_add(ns);
        }
    }

    pub(super) fn merge_into(&self, dst: &mut [u64; WorkBucket::COUNT]) {
        for (d, s) in dst.iter_mut().zip(self.calls.iter()) {
            *d += *s;
        }
    }

    pub(super) fn merge_wall_ns_into(&self, dst: &mut [u64; WorkBucket::COUNT]) {
        for (d, s) in dst.iter_mut().zip(self.wall_ns.iter()) {
            *d = d.saturating_add(*s);
        }
    }
}
