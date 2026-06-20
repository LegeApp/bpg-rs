//! `rdo2`: a staged, scratch-based, close-call-escalating intra RDO search
//! engine, replacing the monolithic recursion in `super::rdo_legacy` one slice at a
//! time (see `crates/still265/refactor-rdo.md`).
//!
//! The decision shape is `rough screen -> cheap RD trial -> exact close-call
//! recheck -> final replay/commit`, rather than funnelling every candidate
//! through the same heavy block path. Weak candidates are screened cheaply and
//! never pay exact RDOQ / exact residual pricing; only genuinely close decisions
//! escalate to exact evaluation before the winner is committed.
//!
//! `BPG_RDO2_TU` gates the transform-tree leaf-vs-split decision. `BPG_RDO2_LUMA`
//! gates cheap luma-mode candidate screening plus exact close-call recheck.
//! `BPG_RDO2_LUMA_SCRATCH` replaces the Best leaf-screen candidate cost loop with
//! non-committing scratch `LeafEval` records. `BPG_RDO2_CHROMA_SCRATCH` does the
//! same for one-block chroma RD candidate costs when `BPG_RDO2_CHROMA` is active.
//! The CABAC writer is unchanged: this module still materialises the existing
//! [`super::types::Tt`] structures (with retained coefficient `levels`) and
//! leaves the winning TU's reconstruction committed in `state.frame`, so
//! `write.rs` and the parallel replay path keep working without modification.

pub(super) mod chroma;
pub(super) mod cost;
pub(super) mod cu;
pub(super) mod finalize;
pub(super) mod luma;
pub(super) mod metrics;
pub(super) mod policy;
pub(super) mod residual;
pub(super) mod rough;
pub(super) mod scratch;
pub(super) mod tu;
