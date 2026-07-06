//! StillSearch v2: plan-first, arena-backed intra search/split core.
//!
//! Guardrails:
//! - Search candidates return small decision/arena IDs or internal plans; large
//!   coeff/recon data should move into CTU-local arenas as phases mature.
//! - Losers are discarded by dropping IDs/plans or rewinding arena/overlay marks.
//! - `CuNode`/`Tt`/`CodedBlock` final syntax trees are emitted only for the
//!   selected CTU winner, through `emit.rs`.
//! - The shared reconstructed frame is mutated only by final commit.
//! - No frame snapshot/restore is used as a trial mechanism.

mod angular;
mod api;
mod arena;
mod canvas;
mod chroma;
mod cost;
mod cu;
mod depth;
mod emit;
mod env;
mod eval;
mod finalize;
mod geom;
mod ledger;
mod nxn;
mod plan;
mod price;
mod rough;
mod source;
mod tu;
mod workspace;

use bpg_hevc_decode::hevc::intra;
use bpg_hevc_decode::hevc::slice::IntraPredMode;

use super::Encoder;
pub(super) use api::StillSearch;
pub(super) use canvas::{CanvasOverlay, ReconBackground, RowStrips};

fn intra_mpm(state: &Encoder<'_>, x0: u32, y0: u32) -> [IntraPredMode; 3] {
    intra::fill_mpm_candidates(
        state.neighbor_left_mode(x0, y0),
        state.neighbor_above_mode(x0, y0),
    )
}
