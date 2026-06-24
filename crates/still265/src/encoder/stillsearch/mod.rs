//! StillSearch v2: plan-first, arena-backed intra search/split core.
//!
//! Guardrails:
//! - Search candidates return small decision/arena IDs; large coeff/recon data
//!   lives in CTU-local arenas.
//! - Losers are discarded by dropping IDs or rewinding arena marks.
//! - `CuNode`/`Tt`/`CodedBlock` final syntax trees are emitted only for the
//!   selected CTU winner.
//! - The shared reconstructed frame is mutated only by final commit.
//! - No frame snapshot/restore is used as a trial mechanism.
//!
//! The first compile milestone implements the narrowest final bridge: an
//! all-leaf, all-DC, zero-residual plan that reconstructs by prediction and then
//! emits final syntax structs. Real RDO phases are added under this module only.

mod api;
mod arena;
mod cost;
mod emit;
mod geom;
mod ledger;
mod overlay;
mod workspace;

pub(super) use api::StillSearch;
