//! Cost and decision descriptors used by StillSearch trials.

use super::arena::CoeffId;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub(super) struct RdCost(pub f64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Confidence {
    Clear,
    Close,
    Exact,
}

impl Default for Confidence {
    fn default() -> Self {
        Self::Clear
    }
}

/// A search decision descriptor. The coeff handle is optional so a default value
/// can never silently point at arena slot zero before a candidate exists.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Decision {
    pub(super) cost: RdCost,
    pub(super) coeff: Option<CoeffId>,
    pub(super) confidence: Confidence,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BlockEval {
    pub(super) distortion: u64,
    pub(super) frac_bits: u64,
    pub(super) cost: RdCost,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_decision_has_no_valid_handle() {
        let d = Decision::default();
        assert!(d.coeff.is_none());
    }
}
