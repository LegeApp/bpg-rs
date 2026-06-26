//! TU leaf-vs-split decision diagnostic (encoder side).
//!
//! Gated entirely by the `BPG_TU_DIAG=<path>` environment variable: when set,
//! every luma transform-tree leaf-vs-split decision in `rdo2_luma_subtree`
//! appends one CSV row with the rate-distortion costs, the margin, and the
//! winner. This exposes the *why* of the TU partition (how close each decision
//! was, and how large a leaf-bias would flip a split win) to complement the
//! decoder-side `bpg-decision-diff` TU-map, which only sees the final structure.
//!
//! Note: `rdo2_luma_subtree` fires during both trial search and the chosen path,
//! so rows include trial evaluations (same granularity as the existing
//! `tu_split_wins_by_region` stats). Analyse the distribution, or join against
//! the decoder-derived final TU map for committed-only decisions.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static SINK: Mutex<Option<std::io::BufWriter<std::fs::File>>> = Mutex::new(None);
static INIT: OnceLock<()> = OnceLock::new();

/// One captured leaf-vs-split decision.
pub struct TuDecision {
    pub x: u32,
    pub y: u32,
    pub log2_size: u8,
    pub trafo_depth: u8,
    pub region_idx: usize,
    pub leaf_dist: u64,
    pub leaf_bits: u64,
    pub leaf_cost: f64,
    pub split_dist: u64,
    pub split_bits: u64,
    pub split_cost: f64,
    pub margin: f64,
    pub winner_split: bool,
    pub escalated: bool,
}

fn init() {
    INIT.get_or_init(|| {
        if let Ok(path) = std::env::var("BPG_TU_DIAG") {
            if !path.trim().is_empty() {
                if let Ok(f) = std::fs::File::create(path.trim()) {
                    let mut w = std::io::BufWriter::new(f);
                    let _ = writeln!(
                        w,
                        "x,y,log2,depth,region,leaf_dist,leaf_bits,leaf_cost,split_dist,split_bits,split_cost,margin,winner_split,escalated,split_won_by"
                    );
                    *SINK.lock().unwrap() = Some(w);
                    ACTIVE.store(true, Ordering::Relaxed);
                }
            }
        }
    });
}

/// True when `BPG_TU_DIAG` is set and the sink opened. Cheap after first call.
#[inline]
pub fn active() -> bool {
    init();
    ACTIVE.load(Ordering::Relaxed)
}

/// Append one decision row (call only when [`active`] returned true).
pub fn record(d: &TuDecision) {
    if let Some(w) = SINK.lock().unwrap().as_mut() {
        // `split_won_by` is `leaf_cost - split_cost`: positive ⇒ split won, and
        // a leaf-bias above this magnitude would flip it to a (larger) leaf.
        let split_won_by = d.leaf_cost - d.split_cost;
        let _ = writeln!(
            w,
            "{},{},{},{},{},{},{},{:.3},{},{},{:.3},{:.4},{},{},{:.4}",
            d.x,
            d.y,
            d.log2_size,
            d.trafo_depth,
            d.region_idx,
            d.leaf_dist,
            d.leaf_bits,
            d.leaf_cost,
            d.split_dist,
            d.split_bits,
            d.split_cost,
            d.margin,
            d.winner_split as u8,
            d.escalated as u8,
            split_won_by,
        );
    }
}

/// Flush the sink. Call once after encoding (the static is never dropped).
pub fn flush() {
    if let Some(w) = SINK.lock().unwrap().as_mut() {
        let _ = w.flush();
    }
}
