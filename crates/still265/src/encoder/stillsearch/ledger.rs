//! StillSearch work ledger buckets.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkBucket {
    RoughLuma,
    LumaCheap,
    LumaExact,
    TuLeaf,
    TuSplit,
    NxnRough,
    NxnBatch,
    ChromaRough,
    ChromaTrial,
    Rdoq,
    ResidualPrice,
    FinalCommit,
    Writer,
    Deblock,
    Sao,
}

impl WorkBucket {
    pub(super) const COUNT: usize = 15;

    fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct StillSearchLedger {
    calls: [u64; WorkBucket::COUNT],
}

impl StillSearchLedger {
    pub(super) fn clear_ctu(&mut self) {
        self.calls = [0; WorkBucket::COUNT];
    }

    pub(super) fn bump(&mut self, bucket: WorkBucket) {
        self.calls[bucket.idx()] += 1;
    }

    pub(super) fn calls(&self, bucket: WorkBucket) -> u64 {
        self.calls[bucket.idx()]
    }

    pub(super) fn merge_into(&self, dst: &mut [u64; WorkBucket::COUNT]) {
        for (d, s) in dst.iter_mut().zip(self.calls.iter()) {
            *d += *s;
        }
    }
}
