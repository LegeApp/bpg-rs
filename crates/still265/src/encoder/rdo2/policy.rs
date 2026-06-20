//! Resolved, call-site-explicit evaluation policy for the staged search.
//!
//! The core invariant (refactor-rdo.md §15): approximate search may reject
//! obvious losers, but may not *decide* a close case without an exact recheck.
#![allow(dead_code)] // staged rdo2 engine under construction (slice 1)

use crate::trace::WorkBucket;

/// How a candidate block is evaluated. Strong/obvious candidates stay
/// [`EvalKind::CheapTrial`]; only close decisions escalate to
/// [`EvalKind::ExactTrial`]; the chosen path is coded once at [`EvalKind::Final`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::encoder) enum EvalKind {
    /// Cheap RD trial: plain quantisation, approximate residual bits,
    /// non-committing (does not write the reconstructed frame).
    CheapTrial,
    /// Exact close-call recheck: full single-scan RDOQ + exact residual bits,
    /// non-committing.
    ExactTrial,
    /// Final coding for the chosen path: full RDOQ + exact bits, committed into
    /// the frame.
    Final,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::encoder) enum RdoqPolicy {
    /// Plain quantisation, no rate-distortion optimised quantisation.
    Off,
    /// Full single-scan RDOQ (`rdoq::rdoq_single_scan`).
    FullSingleScan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::encoder) enum ResidualBitPolicy {
    /// Frozen-context approximate residual bit estimate.
    Approx,
    /// Exact CABAC residual syntax bit count.
    Exact,
}

/// A resolved policy for a single `eval_block` call.
#[derive(Clone, Copy, Debug)]
pub(in crate::encoder) struct EvalPolicy {
    pub(in crate::encoder) rdoq: RdoqPolicy,
    pub(in crate::encoder) bits: ResidualBitPolicy,
    /// When true the reconstructed samples are written into the frame (the
    /// chosen path, or a split child whose siblings depend on its recon).
    pub(in crate::encoder) commit: bool,
    pub(in crate::encoder) work_bucket: Option<WorkBucket>,
}

impl EvalPolicy {
    pub(in crate::encoder) fn for_kind(kind: EvalKind) -> Self {
        match kind {
            EvalKind::CheapTrial => Self {
                rdoq: RdoqPolicy::Off,
                bits: ResidualBitPolicy::Approx,
                commit: false,
                work_bucket: None,
            },
            EvalKind::ExactTrial => Self {
                rdoq: RdoqPolicy::FullSingleScan,
                bits: ResidualBitPolicy::Exact,
                commit: false,
                work_bucket: None,
            },
            EvalKind::Final => Self {
                rdoq: RdoqPolicy::FullSingleScan,
                bits: ResidualBitPolicy::Exact,
                commit: true,
                work_bucket: Some(WorkBucket::FinalReplay),
            },
        }
    }

    pub(in crate::encoder) fn with_bucket(mut self, bucket: WorkBucket) -> Self {
        self.work_bucket = Some(bucket);
        self
    }
}

/// Whether two RD costs are close enough that the cheap ranking can't be
/// trusted and both branches must be re-evaluated exactly. `margin` is a
/// relative fraction of the better (smaller) cost.
#[inline]
pub(in crate::encoder) fn close(a: f64, b: f64, margin: f64) -> bool {
    let lo = a.min(b);
    let hi = a.max(b);
    hi - lo <= margin * lo.max(1.0)
}
