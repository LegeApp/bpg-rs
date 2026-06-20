//! Iterative intra-search test harness — a per-image **CTU intra-search
//! ledger** for the still265 RD search.
//!
//! Spec: `bpg-rs/docs/iterative-intra-search-test-harness.md`. A normal
//! profiler answers "where did CPU time go"; this answers the harder question
//! "which search *decisions* caused expensive discarded work, and did that work
//! change the final decision?". It is the measurement loop that should precede
//! any further intra-search pruning.
//!
//! Two complementary tracks, both **output-neutral** and inert unless
//! `BPG_TRACE_SEARCH=<dir>` is set:
//!
//!  * **Track 1 — residual-estimate accounting** ([`SearchTrace::note_code_block`],
//!    from `code_block_internal`). Every coded transform block reports its
//!    [`WorkStage`], component, position, size and whether it triggered an
//!    *exact* residual-bit estimate (the dominant cost unit). This is the
//!    non-overlapping partition of `EncodeStats::residual_bit_estimates` —
//!    bucketed by stage, component, region class, **block size** and CTU.
//!
//!  * **Track 2 — decision ledger** ([`SearchTrace::note_decision`], from the
//!    luma/chroma/TU/CU RD decisions). Each decision reports its candidate set
//!    as [`CandRec`]s (RD cost, an optional *flip cost* under the opposite
//!    residual-bit model, and the exact residual estimates that candidate
//!    consumed). From those the harness derives — per **region class** and
//!    **block size** — winning-rank histograms, loser-margin histograms (both
//!    **counted and weighted by discarded exact estimates**), split-built-but-
//!    lost waste, and "did exact bits change the winner?".
//!
//! Region classification uses a trace-private
//! [`AnalysisMaps`], never the encoder's `self.analysis`, so a tier that does
//! not normally build a map (e.g. `Best`, whose `force_split` path keys off
//! `importance_q8`) is not perturbed into a different bitstream.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use crate::cabac::CabacEstimator;
use crate::plan::WorkStage;
use crate::preanalysis::{AnalysisMaps, RegionClass, NUM_CLASSES};

const NUM_STAGES: usize = 6;
const NUM_KINDS: usize = 5;
const NUM_COMPONENTS: usize = 3;
/// Block-size buckets, indexed by `log2_size - 2`: 4x4, 8x8, 16x16, 32x32,
/// 64x64. Transform blocks span 2..=5; CU decisions reach 6.
const NUM_SIZES: usize = 5;
const SIZE_NAMES: [&str; NUM_SIZES] = ["4x4", "8x8", "16x16", "32x32", "64x64"];
/// Loser RD-cost margin buckets, in percent over the winning cost:
/// `[0,2) [2,5) [5,10) [10,25) [25,50) [50,inf)`.
const MARGIN_EDGES: [f64; 5] = [0.02, 0.05, 0.10, 0.25, 0.50];
const NUM_MARGIN_BUCKETS: usize = MARGIN_EDGES.len() + 1;
const MARGIN_LABELS: [&str; NUM_MARGIN_BUCKETS] = [
    "0-2pct", "2-5pct", "5-10pct", "10-25pct", "25-50pct", "50pct+",
];
/// Winner-rank histogram width (rank 1..=MAX_RANK, last bucket a catch-all).
/// Index 0 is unused.
const MAX_RANK: usize = 6;

#[inline]
fn stage_index(stage: WorkStage) -> usize {
    match stage {
        WorkStage::RoughModeSearch => 0,
        WorkStage::LumaTrial => 1,
        WorkStage::ChromaTrial => 2,
        WorkStage::TuDecision => 3,
        WorkStage::CuDecision => 4,
        WorkStage::FinalCode => 5,
    }
}

const STAGE_NAMES: [&str; NUM_STAGES] = [
    "rough_mode",
    "luma_trial",
    "chroma_trial",
    "tu_decision",
    "cu_decision",
    "final_code",
];

const COMPONENT_NAMES: [&str; NUM_COMPONENTS] = ["y", "cb", "cr"];

/// Search-decision kind, indexing the per-kind tables.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    /// Luma intra-mode candidate ranking (`build_cu_leaf`).
    Luma = 0,
    /// Chroma intra-mode candidate ranking (`decide_chroma_mode`).
    Chroma = 1,
    /// TU leaf-vs-split RD decision (`decide_tt`).
    Tu = 2,
    /// CU leaf-vs-split RD decision (`decide_cu`).
    Cu = 3,
    /// Close-call luma re-evaluation at `FullRd` (`escalate_luma_close_call`).
    Escalation = 4,
}

const KIND_NAMES: [&str; NUM_KINDS] = ["luma", "chroma", "tu", "cu", "escalation"];

const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "Flat",
    "Gradient",
    "ChromaCritical",
    "Noisy",
    "Texture",
    "DirectionalEdge",
    "TextLike",
];

#[inline]
fn size_bucket(log2_size: u8) -> usize {
    (log2_size as usize).saturating_sub(2).min(NUM_SIZES - 1)
}

#[inline]
fn component_index(c_idx: u8) -> usize {
    (c_idx as usize).min(NUM_COMPONENTS - 1)
}

/// One evaluated candidate in a Track-2 decision. The winner is `argmin(cost)`.
#[derive(Clone, Copy)]
pub struct CandRec {
    /// Exact RD cost (`D + λ·R`). For luma/chroma the candidates are passed in
    /// rough-rank (SATD) order, so the winner's index is its rough rank − 1.
    pub cost: f64,
    /// RD cost of the *same* coded candidate priced under the opposite residual-
    /// bit model (approx vs exact). `NaN` when not computed. If `argmin(cost)`
    /// and `argmin(flip_cost)` disagree, exact bits changed the winner.
    pub flip_cost: f64,
    /// Exact residual-bit estimates this candidate consumed — the cost weight
    /// for the discarded-work histograms.
    pub exact_estimates: u32,
}

impl CandRec {
    pub fn new(cost: f64, exact_estimates: u32) -> Self {
        CandRec {
            cost,
            flip_cost: f64::NAN,
            exact_estimates,
        }
    }
    pub fn with_flip(cost: f64, flip_cost: f64, exact_estimates: u32) -> Self {
        CandRec {
            cost,
            flip_cost,
            exact_estimates,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct StageTally {
    code_blocks: u64,
    nonzero_blocks: u64,
    exact_estimates: u64,
    luma_exact_estimates: u64,
    chroma_exact_estimates: u64,
    final_pricings_elided: u64,
}

#[derive(Clone, Copy, Default)]
struct CodeBlockVolume {
    code_blocks: u64,
    nonzero_blocks: u64,
    exact_estimates: u64,
    approx_priced_blocks: u64,
    final_pricings_elided: u64,
}

#[derive(Clone, Copy, Default)]
struct CtuTally {
    code_blocks: u64,
    exact_estimates: u64,
    final_exact_estimates: u64,
    trial_exact_estimates: u64,
    final_pricings_elided: u64,
    luma_decisions: u64,
    luma_losers: u64,
    chroma_decisions: u64,
    close_calls: u64,
    split_built_but_lost: u64,
    final_frac_bits: u64,
}

#[derive(Clone, Copy)]
struct DecisionEvent {
    kind: u8,
    ctu: u32,
    x: u16,
    y: u16,
    log2_size: u8,
    region_class: u8,
    n_candidates: u8,
    winner_rank: u8,
    best_cost: f64,
    runner_up_cost: f64,
    loser_exact_estimates: u32,
    exact_changed_winner: bool,
    split_built_but_lost: bool,
}

/// Per-(kind) decision aggregates, sliced by region class and block size.
#[derive(Clone)]
struct KindTables {
    /// Winner-rank histogram (overall and per region / per size).
    rank: [u64; MAX_RANK + 1],
    rank_by_region: [[u64; MAX_RANK + 1]; NUM_CLASSES],
    rank_by_size: [[u64; MAX_RANK + 1]; NUM_SIZES],
    /// Loser-margin histogram, counted and weighted by discarded exact estimates.
    loss_count: [u64; NUM_MARGIN_BUCKETS],
    loss_weight: [u64; NUM_MARGIN_BUCKETS],
    loss_count_by_region: [[u64; NUM_MARGIN_BUCKETS]; NUM_CLASSES],
    /// "Did exact bits change the winner" — decisions with both costs available,
    /// and of those how many flipped.
    flip_total: u64,
    flip_changed: u64,
    /// Split decisions where the split branch was fully built and still lost,
    /// by region (count + discarded exact estimates).
    split_lost_count_by_region: [u64; NUM_CLASSES],
    split_lost_weight_by_region: [u64; NUM_CLASSES],
}

impl Default for KindTables {
    fn default() -> Self {
        KindTables {
            rank: [0; MAX_RANK + 1],
            rank_by_region: [[0; MAX_RANK + 1]; NUM_CLASSES],
            rank_by_size: [[0; MAX_RANK + 1]; NUM_SIZES],
            loss_count: [0; NUM_MARGIN_BUCKETS],
            loss_weight: [0; NUM_MARGIN_BUCKETS],
            loss_count_by_region: [[0; NUM_MARGIN_BUCKETS]; NUM_CLASSES],
            flip_total: 0,
            flip_changed: 0,
            split_lost_count_by_region: [0; NUM_CLASSES],
            split_lost_weight_by_region: [0; NUM_CLASSES],
        }
    }
}

/// Per-image search ledger. Disabled (zero-overhead) unless
/// `BPG_TRACE_SEARCH=<dir>` was set when the encode began.
pub struct SearchTrace {
    pub enabled: bool,
    dir: PathBuf,
    dump_events: bool,
    ctb_cols: u32,
    ctb_log2: u8,
    analysis: Option<Arc<AnalysisMaps>>,
    stage: [StageTally; NUM_STAGES],
    code_block_volume: [[[CodeBlockVolume; NUM_SIZES]; NUM_COMPONENTS]; NUM_STAGES],
    rdoq_volume: [[[u64; NUM_SIZES]; NUM_COMPONENTS]; NUM_STAGES],
    region_exact_estimates: [u64; NUM_CLASSES],
    /// Exact estimates by region class × block size (from `note_code_block`).
    region_size_exact: [[u64; NUM_SIZES]; NUM_CLASSES],
    ctus: Vec<CtuTally>,
    kinds: [KindTables; NUM_KINDS],
    luma_rmd_selections: u64,
    luma_rmd_modes_scored: u64,
    luma_rmd_candidates_retained: u64,
    luma_rmd_mpm0_forced: u64,
    rdo2_residual_approx_pricings: u64,
    rdo2_residual_exact_pricings: u64,
    tu_neighbor_prior_seen: u64,
    tu_neighbor_prior_leaf: u64,
    tu_neighbor_prior_split: u64,
    tu_neighbor_prior_matches: u64,
    tu_neighbor_leaf_skips: u64,
    tu_neighbor_leaf_skips_by_region: [u64; NUM_CLASSES],
    tu_neighbor_leaf_skips_by_size: [u64; NUM_SIZES],
    cu_split_bound_aborts: u64,
    cu_split_bound_abort_exact_estimates: u64,
    cu_split_bound_abort_by_child: [u64; 5],
    cu_split_bound_abort_exact_by_child: [u64; 5],
    cu_split_bound_abort_margin: [u64; NUM_MARGIN_BUCKETS],
    cu_split_bound_abort_by_region: [u64; NUM_CLASSES],
    cu_split_bound_abort_by_size: [u64; NUM_SIZES],
    decisions: Vec<DecisionEvent>,
    events_cap: usize,
    events_truncated: u64,
    meta: TraceMeta,
}

/// Default cap on the opt-in per-decision event dump. The aggregate reports are
/// never capped — only the raw drill-down is.
const DEFAULT_EVENTS_CAP: usize = 1_000_000;

#[derive(Clone, Default)]
struct TraceMeta {
    image: String,
    effort: String,
    qp: i32,
    width: u32,
    height: u32,
    chroma: String,
}

impl Default for SearchTrace {
    fn default() -> Self {
        SearchTrace {
            enabled: false,
            dir: PathBuf::new(),
            dump_events: false,
            ctb_cols: 0,
            ctb_log2: 6,
            analysis: None,
            stage: [StageTally::default(); NUM_STAGES],
            code_block_volume: [[[CodeBlockVolume::default(); NUM_SIZES]; NUM_COMPONENTS];
                NUM_STAGES],
            rdoq_volume: [[[0; NUM_SIZES]; NUM_COMPONENTS]; NUM_STAGES],
            region_exact_estimates: [0; NUM_CLASSES],
            region_size_exact: [[0; NUM_SIZES]; NUM_CLASSES],
            ctus: Vec::new(),
            kinds: Default::default(),
            luma_rmd_selections: 0,
            luma_rmd_modes_scored: 0,
            luma_rmd_candidates_retained: 0,
            luma_rmd_mpm0_forced: 0,
            rdo2_residual_approx_pricings: 0,
            rdo2_residual_exact_pricings: 0,
            tu_neighbor_prior_seen: 0,
            tu_neighbor_prior_leaf: 0,
            tu_neighbor_prior_split: 0,
            tu_neighbor_prior_matches: 0,
            tu_neighbor_leaf_skips: 0,
            tu_neighbor_leaf_skips_by_region: [0; NUM_CLASSES],
            tu_neighbor_leaf_skips_by_size: [0; NUM_SIZES],
            cu_split_bound_aborts: 0,
            cu_split_bound_abort_exact_estimates: 0,
            cu_split_bound_abort_by_child: [0; 5],
            cu_split_bound_abort_exact_by_child: [0; 5],
            cu_split_bound_abort_margin: [0; NUM_MARGIN_BUCKETS],
            cu_split_bound_abort_by_region: [0; NUM_CLASSES],
            cu_split_bound_abort_by_size: [0; NUM_SIZES],
            decisions: Vec::new(),
            events_cap: 0,
            events_truncated: 0,
            meta: TraceMeta::default(),
        }
    }
}

impl SearchTrace {
    /// Build an enabled trace if `BPG_TRACE_SEARCH` is set, else a disabled stub.
    #[allow(clippy::too_many_arguments)]
    pub fn from_env(
        ctb_log2: u8,
        width: u32,
        height: u32,
        ctb_cols: u32,
        ctb_rows: u32,
        analysis: Option<Arc<AnalysisMaps>>,
        effort: &str,
        qp: i32,
        chroma: &str,
    ) -> Self {
        let Some(dir) = std::env::var_os("BPG_TRACE_SEARCH") else {
            return SearchTrace::default();
        };
        let image = std::env::var("BPG_TRACE_IMAGE").unwrap_or_else(|_| "image".to_string());
        let events = std::env::var("BPG_TRACE_EVENTS").ok();
        let dump_events = events.is_some();
        let events_cap = events
            .as_deref()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_EVENTS_CAP);
        SearchTrace {
            enabled: true,
            dir: PathBuf::from(dir),
            dump_events,
            events_cap,
            ctb_cols,
            ctb_log2,
            analysis,
            ctus: vec![CtuTally::default(); (ctb_cols * ctb_rows) as usize],
            meta: TraceMeta {
                image,
                effort: effort.to_string(),
                qp,
                width,
                height,
                chroma: chroma.to_string(),
            },
            ..SearchTrace::default()
        }
    }

    #[inline]
    fn ctu_index(&self, x: u32, y: u32) -> usize {
        let cx = x >> self.ctb_log2;
        let cy = y >> self.ctb_log2;
        (cy * self.ctb_cols + cx) as usize
    }

    #[inline]
    fn region_at(&self, x: u32, y: u32, log2_size: u8) -> RegionClass {
        match &self.analysis {
            Some(a) => a.region_class_at(x, y, log2_size),
            None => RegionClass::Flat,
        }
    }

    /// Track 1: one coded transform block.
    pub fn note_code_block(
        &mut self,
        stage: WorkStage,
        c_idx: u8,
        x: u32,
        y: u32,
        log2_size: u8,
        cbf: bool,
        exact: bool,
        frac_bits: u64,
    ) {
        let si = stage_index(stage);
        let ci = component_index(c_idx);
        let sz = size_bucket(log2_size);
        let st = &mut self.stage[si];
        st.code_blocks += 1;
        if cbf {
            st.nonzero_blocks += 1;
        }
        if exact {
            st.exact_estimates += 1;
            if c_idx == 0 {
                st.luma_exact_estimates += 1;
            } else {
                st.chroma_exact_estimates += 1;
            }
        }
        let volume = &mut self.code_block_volume[si][ci][sz];
        volume.code_blocks += 1;
        if cbf {
            volume.nonzero_blocks += 1;
            if exact {
                volume.exact_estimates += 1;
            } else {
                volume.approx_priced_blocks += 1;
            }
        }
        let region = self.region_at(x, y, log2_size).index();
        if exact {
            self.region_exact_estimates[region] += 1;
            self.region_size_exact[region][sz] += 1;
        }
        let ctu_idx = self.ctu_index(x, y);
        let ctu = &mut self.ctus[ctu_idx];
        ctu.code_blocks += 1;
        if exact {
            ctu.exact_estimates += 1;
            if stage == WorkStage::FinalCode {
                ctu.final_exact_estimates += 1;
            } else {
                ctu.trial_exact_estimates += 1;
            }
        }
        if stage == WorkStage::FinalCode {
            ctu.final_frac_bits += frac_bits;
        }
    }

    /// Track a final-coded transform block whose residual syntax bits are not
    /// estimated during analysis because the writer will immediately emit the
    /// same syntax from the materialized levels.
    pub fn note_final_pricing_elided(
        &mut self,
        c_idx: u8,
        x: u32,
        y: u32,
        log2_size: u8,
        cbf: bool,
    ) {
        if !self.enabled {
            return;
        }
        let si = stage_index(WorkStage::FinalCode);
        let ci = component_index(c_idx);
        let sz = size_bucket(log2_size);
        let st = &mut self.stage[si];
        st.code_blocks += 1;
        if cbf {
            st.nonzero_blocks += 1;
            st.final_pricings_elided += 1;
        }
        let volume = &mut self.code_block_volume[si][ci][sz];
        volume.code_blocks += 1;
        if cbf {
            volume.nonzero_blocks += 1;
            volume.final_pricings_elided += 1;
        }
        let ctu_idx = self.ctu_index(x, y);
        let ctu = &mut self.ctus[ctu_idx];
        ctu.code_blocks += 1;
        if cbf {
            ctu.final_pricings_elided += 1;
        }
    }

    /// Track RDOQ blocks by the same stage/component/size dimensions as
    /// `note_code_block`. This is a block count, not a pass count.
    pub fn note_rdoq_block(&mut self, stage: WorkStage, c_idx: u8, log2_size: u8) {
        if !self.enabled {
            return;
        }
        self.rdoq_volume[stage_index(stage)][component_index(c_idx)][size_bucket(log2_size)] += 1;
    }

    /// Track 0: rough luma mode scheduler selection before expensive luma RD.
    pub fn note_luma_rmd_selection(&mut self, scored: usize, retained: usize, mpm0_forced: bool) {
        if !self.enabled {
            return;
        }
        self.luma_rmd_selections += 1;
        self.luma_rmd_modes_scored += scored as u64;
        self.luma_rmd_candidates_retained += retained as u64;
        if mpm0_forced {
            self.luma_rmd_mpm0_forced += 1;
        }
    }

    pub fn note_rdo2_residual_pricing(&mut self, exact: bool) {
        if !self.enabled {
            return;
        }
        if exact {
            self.rdo2_residual_exact_pricings += 1;
        } else {
            self.rdo2_residual_approx_pricings += 1;
        }
    }

    pub fn note_tu_neighbor_prior(&mut self, prior_depth: u8, winner_split: bool) {
        if !self.enabled {
            return;
        }
        self.tu_neighbor_prior_seen += 1;
        if prior_depth == 0 {
            self.tu_neighbor_prior_leaf += 1;
        } else {
            self.tu_neighbor_prior_split += 1;
        }
        if (prior_depth > 0) == winner_split {
            self.tu_neighbor_prior_matches += 1;
        }
    }

    pub fn note_tu_neighbor_leaf_skip(&mut self, x: u32, y: u32, log2_size: u8) {
        if !self.enabled {
            return;
        }
        let ri = self.region_at(x, y, log2_size).index();
        let sz = size_bucket(log2_size);
        self.tu_neighbor_leaf_skips += 1;
        self.tu_neighbor_leaf_skips_by_region[ri] += 1;
        self.tu_neighbor_leaf_skips_by_size[sz] += 1;
    }

    /// Track exact CU split branch-and-bound aborts. This captures whether the
    /// leaf-first bound stopped after child 1/2/3/4 and how much partial split
    /// work had already been spent before the abort.
    #[allow(clippy::too_many_arguments)]
    pub fn note_cu_split_bound_abort(
        &mut self,
        x: u32,
        y: u32,
        log2_size: u8,
        children_built: usize,
        leaf_cost: f64,
        partial_split_cost: f64,
        exact_estimates: u32,
    ) {
        if !self.enabled {
            return;
        }
        let child = children_built.min(4);
        let ri = self.region_at(x, y, log2_size).index();
        let sz = size_bucket(log2_size);
        self.cu_split_bound_aborts += 1;
        self.cu_split_bound_abort_exact_estimates += exact_estimates as u64;
        self.cu_split_bound_abort_by_child[child] += 1;
        self.cu_split_bound_abort_exact_by_child[child] += exact_estimates as u64;
        self.cu_split_bound_abort_by_region[ri] += 1;
        self.cu_split_bound_abort_by_size[sz] += 1;

        if leaf_cost.is_finite() && leaf_cost > 0.0 && partial_split_cost.is_finite() {
            let margin = (partial_split_cost - leaf_cost) / leaf_cost;
            let mut b = NUM_MARGIN_BUCKETS - 1;
            for (j, &edge) in MARGIN_EDGES.iter().enumerate() {
                if margin < edge {
                    b = j;
                    break;
                }
            }
            self.cu_split_bound_abort_margin[b] += 1;
        }
    }

    /// Track 2: record a resolved RD decision from its candidate set. Winner is
    /// `argmin(cost)`; for luma/chroma the candidates are in rough-rank order so
    /// the winner index is its rough rank. `is_split` marks leaf-vs-split
    /// decisions (cands `[leaf, split]`).
    pub fn note_decision(
        &mut self,
        kind: DecisionKind,
        x: u32,
        y: u32,
        log2_size: u8,
        is_split: bool,
        cands: &[CandRec],
    ) {
        if cands.is_empty() {
            return;
        }
        let ki = kind as usize;
        let region = self.region_at(x, y, log2_size);
        let ri = region.index();
        let sz = size_bucket(log2_size);

        // Winner = lowest exact RD cost.
        let mut win = 0usize;
        for (i, c) in cands.iter().enumerate() {
            if c.cost < cands[win].cost {
                win = i;
            }
        }
        let best = cands[win].cost;
        let winner_rank = (win + 1).min(MAX_RANK);

        let t = &mut self.kinds[ki];
        t.rank[winner_rank] += 1;
        t.rank_by_region[ri][winner_rank] += 1;
        t.rank_by_size[sz][winner_rank] += 1;

        // Losers: margin histograms (counted + exact-estimate weighted).
        let mut runner_up = f64::INFINITY;
        let mut loser_exact_total = 0u32;
        if best.is_finite() && best > 0.0 {
            for (i, c) in cands.iter().enumerate() {
                if i == win || !c.cost.is_finite() {
                    continue;
                }
                runner_up = runner_up.min(c.cost);
                loser_exact_total = loser_exact_total.saturating_add(c.exact_estimates);
                let margin = (c.cost - best) / best;
                let mut b = NUM_MARGIN_BUCKETS - 1;
                for (j, &edge) in MARGIN_EDGES.iter().enumerate() {
                    if margin < edge {
                        b = j;
                        break;
                    }
                }
                t.loss_count[b] += 1;
                t.loss_weight[b] += c.exact_estimates as u64;
                t.loss_count_by_region[ri][b] += 1;
            }
        }

        // "Did exact bits change the winner?" — needs flip costs on every cand.
        if cands.iter().all(|c| c.flip_cost.is_finite()) {
            let mut fwin = 0usize;
            for (i, c) in cands.iter().enumerate() {
                if c.flip_cost < cands[fwin].flip_cost {
                    fwin = i;
                }
            }
            t.flip_total += 1;
            if fwin != win {
                t.flip_changed += 1;
            }
        }

        // Split-built-but-lost waste (leaf-vs-split kinds): cands == [leaf, split].
        let split_lost = is_split && cands.len() == 2 && win == 0;
        if split_lost {
            t.split_lost_count_by_region[ri] += 1;
            t.split_lost_weight_by_region[ri] += cands[1].exact_estimates as u64;
        }

        // CTU rollups.
        let ci = self.ctu_index(x, y);
        let ctu = &mut self.ctus[ci];
        match kind {
            DecisionKind::Luma => {
                ctu.luma_decisions += 1;
                ctu.luma_losers += (cands.len() - 1) as u64;
            }
            DecisionKind::Chroma => ctu.chroma_decisions += 1,
            _ => {}
        }
        if split_lost {
            ctu.split_built_but_lost += 1;
        }

        if self.dump_events {
            if self.decisions.len() >= self.events_cap {
                self.events_truncated += 1;
                return;
            }
            let changed = cands.iter().all(|c| c.flip_cost.is_finite()) && {
                let mut fwin = 0usize;
                for (i, c) in cands.iter().enumerate() {
                    if c.flip_cost < cands[fwin].flip_cost {
                        fwin = i;
                    }
                }
                fwin != win
            };
            self.decisions.push(DecisionEvent {
                kind: ki as u8,
                ctu: ci as u32,
                x: x.min(u16::MAX as u32) as u16,
                y: y.min(u16::MAX as u32) as u16,
                log2_size,
                region_class: ri as u8,
                n_candidates: cands.len().min(u8::MAX as usize) as u8,
                winner_rank: winner_rank as u8,
                best_cost: best,
                runner_up_cost: if runner_up.is_finite() {
                    runner_up
                } else {
                    best
                },
                loser_exact_estimates: loser_exact_total,
                exact_changed_winner: changed,
                split_built_but_lost: split_lost,
            });
        }
    }

    /// Write all report files. Called once after the slice is encoded.
    pub fn write_reports(&self, total_residual_estimates: u64) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        self.write_summary(total_residual_estimates)?;
        self.write_waste_buckets()?;
        self.write_stage_table()?;
        self.write_code_block_volume()?;
        self.write_rdoq_volume()?;
        self.write_loss_hist()?;
        self.write_rank_hist()?;
        self.write_region_table()?;
        self.write_exact_by_region_size()?;
        self.write_rank_cross_tabs()?;
        self.write_exact_changed_winner()?;
        self.write_split_waste_by_region()?;
        self.write_split_abort_summary()?;
        self.write_tu_neighbor_summary()?;
        self.write_ctu_heatmap()?;
        if self.dump_events {
            self.write_events()?;
        }
        eprintln!(
            "search-trace: wrote ledger for {} ({}, qp{}) to {} \
             [aggregates always complete; {} decision rows{}]",
            self.meta.image,
            self.meta.effort,
            self.meta.qp,
            self.dir.display(),
            self.decisions.len(),
            if self.events_truncated > 0 {
                format!(
                    ", {} truncated (raise BPG_TRACE_EVENTS=N)",
                    self.events_truncated
                )
            } else if !self.dump_events {
                " (set BPG_TRACE_EVENTS for drill-down)".to_string()
            } else {
                String::new()
            },
        );
        Ok(())
    }

    fn stage_sum(&self, field: impl Fn(&StageTally) -> u64) -> u64 {
        self.stage.iter().map(field).sum()
    }

    fn code_block_volume_sum(&self, field: impl Fn(&CodeBlockVolume) -> u64) -> u64 {
        self.code_block_volume
            .iter()
            .flat_map(|by_component| by_component.iter())
            .flat_map(|by_size| by_size.iter())
            .map(field)
            .sum()
    }

    fn rdoq_volume_sum(&self) -> u64 {
        self.rdoq_volume
            .iter()
            .flat_map(|by_component| by_component.iter())
            .flat_map(|by_size| by_size.iter())
            .copied()
            .sum()
    }

    fn write_summary(&self, total_residual_estimates: u64) -> std::io::Result<()> {
        let mut w = self.create("search_summary.csv")?;
        writeln!(w, "metric,value")?;
        let m = &self.meta;
        writeln!(w, "image,{}", m.image)?;
        writeln!(w, "effort,{}", m.effort)?;
        writeln!(w, "qp,{}", m.qp)?;
        writeln!(w, "width,{}", m.width)?;
        writeln!(w, "height,{}", m.height)?;
        writeln!(w, "chroma,{}", m.chroma)?;

        let mpix = (m.width as f64 * m.height as f64) / 1e6;
        let code_blocks = self.stage_sum(|s| s.code_blocks);
        let final_blocks = self.stage[stage_index(WorkStage::FinalCode)].code_blocks;
        let trial_blocks = code_blocks - final_blocks;
        let exact = self.stage_sum(|s| s.exact_estimates);
        let final_pricings_elided = self.stage_sum(|s| s.final_pricings_elided);
        let approx_priced = self.code_block_volume_sum(|v| v.approx_priced_blocks);
        let final_exact = self.stage[stage_index(WorkStage::FinalCode)].exact_estimates;
        let trial_exact = exact - final_exact;

        writeln!(w, "megapixels,{mpix:.3}")?;
        writeln!(w, "code_blocks,{code_blocks}")?;
        writeln!(w, "final_coded_blocks,{final_blocks}")?;
        writeln!(w, "trial_coded_blocks,{trial_blocks}")?;
        writeln!(
            w,
            "trial_final_ratio,{:.3}",
            ratio(trial_blocks, final_blocks)
        )?;
        writeln!(w, "exact_residual_estimates,{exact}")?;
        writeln!(
            w,
            "exact_residual_estimates_counter,{total_residual_estimates}"
        )?;
        writeln!(w, "exact_residual_estimates_final,{final_exact}")?;
        writeln!(w, "final_residual_pricings_elided,{final_pricings_elided}")?;
        writeln!(w, "exact_residual_estimates_trial,{trial_exact}")?;
        writeln!(w, "approx_residual_priced_blocks,{approx_priced}")?;
        writeln!(w, "rdoq_blocks,{}", self.rdoq_volume_sum())?;
        writeln!(
            w,
            "exact_residual_per_final_block,{:.3}",
            ratio(exact, final_blocks)
        )?;
        writeln!(
            w,
            "trial_exact_per_final_exact,{:.3}",
            ratio(trial_exact, final_exact)
        )?;
        writeln!(
            w,
            "exact_residual_per_megapixel,{:.0}",
            exact as f64 / mpix.max(1e-9)
        )?;

        for (ki, name) in KIND_NAMES.iter().enumerate() {
            let dec: u64 = self.kinds[ki].rank.iter().sum();
            writeln!(w, "{name}_decisions,{dec}")?;
        }
        writeln!(w, "luma_rmd_selections,{}", self.luma_rmd_selections)?;
        writeln!(w, "luma_rmd_modes_scored,{}", self.luma_rmd_modes_scored)?;
        writeln!(
            w,
            "luma_rmd_candidates_retained,{}",
            self.luma_rmd_candidates_retained
        )?;
        writeln!(
            w,
            "luma_rmd_avg_scored,{:.3}",
            ratio(self.luma_rmd_modes_scored, self.luma_rmd_selections)
        )?;
        writeln!(
            w,
            "luma_rmd_avg_retained,{:.3}",
            ratio(self.luma_rmd_candidates_retained, self.luma_rmd_selections)
        )?;
        writeln!(w, "luma_rmd_mpm0_forced,{}", self.luma_rmd_mpm0_forced)?;
        writeln!(
            w,
            "rdo2_residual_approx_pricings,{}",
            self.rdo2_residual_approx_pricings
        )?;
        writeln!(
            w,
            "rdo2_residual_exact_pricings,{}",
            self.rdo2_residual_exact_pricings
        )?;
        writeln!(w, "tu_neighbor_prior_seen,{}", self.tu_neighbor_prior_seen)?;
        writeln!(w, "tu_neighbor_prior_leaf,{}", self.tu_neighbor_prior_leaf)?;
        writeln!(
            w,
            "tu_neighbor_prior_split,{}",
            self.tu_neighbor_prior_split
        )?;
        writeln!(
            w,
            "tu_neighbor_prior_matches,{}",
            self.tu_neighbor_prior_matches
        )?;
        writeln!(
            w,
            "tu_neighbor_prior_match_rate,{:.3}",
            ratio(self.tu_neighbor_prior_matches, self.tu_neighbor_prior_seen)
        )?;
        writeln!(w, "tu_neighbor_leaf_skips,{}", self.tu_neighbor_leaf_skips)?;
        writeln!(
            w,
            "cu_split_bound_aborts_trace,{}",
            self.cu_split_bound_aborts
        )?;
        writeln!(
            w,
            "cu_split_bound_abort_exact_estimates,{}",
            self.cu_split_bound_abort_exact_estimates
        )?;
        let split_built_but_lost: u64 = self.ctus.iter().map(|c| c.split_built_but_lost).sum();
        let close_calls: u64 = self.ctus.iter().map(|c| c.close_calls).sum();
        writeln!(w, "split_built_but_lost,{split_built_but_lost}")?;
        writeln!(w, "close_calls,{close_calls}")?;
        // CU-attribution note: decide_cu/decide_tt themselves cost no exact
        // estimates — they reuse the cached frac_bits of their already-coded
        // children, so a CU split branch's exact cost lands under luma/chroma/
        // final. The split waste is captured by split_lost weight, not stage.
        writeln!(
            w,
            "note_cu_tu_stage_exact,attributed_to_child_luma_chroma_final_stages"
        )?;
        Ok(())
    }

    fn write_code_block_volume(&self) -> std::io::Result<()> {
        let mut w = self.create("code_block_volume.csv")?;
        writeln!(
            w,
            "stage,component,block_size,code_blocks,nonzero_blocks,exact_residual_estimates,approx_residual_priced_blocks,final_pricings_elided"
        )?;
        for (si, stage) in STAGE_NAMES.iter().enumerate() {
            for (ci, component) in COMPONENT_NAMES.iter().enumerate() {
                for (zi, size) in SIZE_NAMES.iter().enumerate() {
                    let v = self.code_block_volume[si][ci][zi];
                    if v.code_blocks == 0 {
                        continue;
                    }
                    writeln!(
                        w,
                        "{stage},{component},{size},{},{},{},{},{}",
                        v.code_blocks,
                        v.nonzero_blocks,
                        v.exact_estimates,
                        v.approx_priced_blocks,
                        v.final_pricings_elided
                    )?;
                }
            }
        }
        Ok(())
    }

    fn write_rdoq_volume(&self) -> std::io::Result<()> {
        let mut w = self.create("rdoq_volume.csv")?;
        writeln!(w, "stage,component,block_size,rdoq_blocks")?;
        for (si, stage) in STAGE_NAMES.iter().enumerate() {
            for (ci, component) in COMPONENT_NAMES.iter().enumerate() {
                for (zi, size) in SIZE_NAMES.iter().enumerate() {
                    let blocks = self.rdoq_volume[si][ci][zi];
                    if blocks == 0 {
                        continue;
                    }
                    writeln!(w, "{stage},{component},{size},{blocks}")?;
                }
            }
        }
        Ok(())
    }

    fn write_split_abort_summary(&self) -> std::io::Result<()> {
        let mut w = self.create("split_abort_summary.csv")?;
        writeln!(w, "section,bucket,count,exact_estimates")?;
        for child in 1..=4 {
            writeln!(
                w,
                "children_built,{child},{},{}",
                self.cu_split_bound_abort_by_child[child],
                self.cu_split_bound_abort_exact_by_child[child]
            )?;
        }
        for (i, label) in MARGIN_LABELS.iter().enumerate() {
            writeln!(
                w,
                "abort_margin,{label},{},",
                self.cu_split_bound_abort_margin[i]
            )?;
        }
        for (i, name) in CLASS_NAMES.iter().enumerate() {
            writeln!(
                w,
                "region,{name},{},",
                self.cu_split_bound_abort_by_region[i]
            )?;
        }
        for (i, name) in SIZE_NAMES.iter().enumerate() {
            writeln!(
                w,
                "block_size,{name},{},",
                self.cu_split_bound_abort_by_size[i]
            )?;
        }
        Ok(())
    }

    fn write_tu_neighbor_summary(&self) -> std::io::Result<()> {
        let mut w = self.create("tu_neighbor_summary.csv")?;
        writeln!(w, "section,bucket,count")?;
        writeln!(w, "overall,prior_seen,{}", self.tu_neighbor_prior_seen)?;
        writeln!(
            w,
            "overall,prior_matches,{}",
            self.tu_neighbor_prior_matches
        )?;
        writeln!(w, "overall,leaf_skips,{}", self.tu_neighbor_leaf_skips)?;
        for (i, name) in CLASS_NAMES.iter().enumerate() {
            writeln!(
                w,
                "leaf_skip_region,{name},{}",
                self.tu_neighbor_leaf_skips_by_region[i]
            )?;
        }
        for (i, name) in SIZE_NAMES.iter().enumerate() {
            writeln!(
                w,
                "leaf_skip_block_size,{name},{}",
                self.tu_neighbor_leaf_skips_by_size[i]
            )?;
        }
        Ok(())
    }

    fn write_stage_table(&self) -> std::io::Result<()> {
        let mut w = self.create("stage_table.csv")?;
        writeln!(
            w,
            "stage,code_blocks,nonzero_blocks,exact_estimates,luma_exact,chroma_exact,final_pricings_elided,exact_share_pct"
        )?;
        let total_exact = self.stage_sum(|s| s.exact_estimates).max(1);
        for (i, st) in self.stage.iter().enumerate() {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{:.1}",
                STAGE_NAMES[i],
                st.code_blocks,
                st.nonzero_blocks,
                st.exact_estimates,
                st.luma_exact_estimates,
                st.chroma_exact_estimates,
                st.final_pricings_elided,
                100.0 * st.exact_estimates as f64 / total_exact as f64,
            )?;
        }
        Ok(())
    }

    fn write_waste_buckets(&self) -> std::io::Result<()> {
        let mut w = self.create("waste_buckets.csv")?;
        writeln!(w, "bucket,exact_estimates,share_pct")?;
        let total = self.stage_sum(|s| s.exact_estimates).max(1) as f64;
        let row = |w: &mut std::fs::File, name: &str, st: WorkStage| -> std::io::Result<()> {
            let n = self.stage[stage_index(st)].exact_estimates;
            writeln!(w, "{name},{n},{:.1}", 100.0 * n as f64 / total)
        };
        row(&mut w, "final_code", WorkStage::FinalCode)?;
        row(&mut w, "luma_trial", WorkStage::LumaTrial)?;
        row(&mut w, "chroma_trial", WorkStage::ChromaTrial)?;
        row(&mut w, "tu_decision", WorkStage::TuDecision)?;
        row(&mut w, "cu_decision", WorkStage::CuDecision)?;
        row(&mut w, "rough_mode", WorkStage::RoughModeSearch)?;
        Ok(())
    }

    fn write_loss_hist(&self) -> std::io::Result<()> {
        let mut w = self.create("loss_margin_hist.csv")?;
        write!(w, "kind,weight")?;
        for l in MARGIN_LABELS {
            write!(w, ",{l}")?;
        }
        writeln!(w, ",total")?;
        for (ki, name) in KIND_NAMES.iter().enumerate() {
            let t = &self.kinds[ki];
            let total_c: u64 = t.loss_count.iter().sum();
            if total_c == 0 {
                continue;
            }
            // counted losers
            write!(w, "{name},count")?;
            for &v in &t.loss_count {
                write!(w, ",{v}")?;
            }
            writeln!(w, ",{total_c}")?;
            // exact-estimate-weighted losers (the doc's "weighted loss histogram")
            let total_w: u64 = t.loss_weight.iter().sum();
            write!(w, "{name},exact_estimates")?;
            for &v in &t.loss_weight {
                write!(w, ",{v}")?;
            }
            writeln!(w, ",{total_w}")?;
        }
        Ok(())
    }

    fn write_rank_hist(&self) -> std::io::Result<()> {
        let mut w = self.create("winner_rank_hist.csv")?;
        self.rank_header(&mut w)?;
        for (ki, name) in KIND_NAMES.iter().enumerate() {
            self.rank_row(&mut w, name, &self.kinds[ki].rank)?;
        }
        Ok(())
    }

    fn rank_header(&self, w: &mut std::fs::File) -> std::io::Result<()> {
        write!(w, "kind")?;
        for r in 1..=MAX_RANK {
            if r == MAX_RANK {
                write!(w, ",rank{r}plus")?;
            } else {
                write!(w, ",rank{r}")?;
            }
        }
        writeln!(w, ",total")
    }

    fn rank_row(
        &self,
        w: &mut std::fs::File,
        label: &str,
        rank: &[u64; MAX_RANK + 1],
    ) -> std::io::Result<()> {
        let total: u64 = rank.iter().sum();
        if total == 0 {
            return Ok(());
        }
        write!(w, "{label}")?;
        for r in 1..=MAX_RANK {
            write!(w, ",{}", rank[r])?;
        }
        writeln!(w, ",{total}")
    }

    fn write_region_table(&self) -> std::io::Result<()> {
        let mut w = self.create("region_table.csv")?;
        writeln!(w, "region_class,exact_estimates,share_pct,cells")?;
        let total = self.region_exact_estimates.iter().sum::<u64>().max(1) as f64;
        let counts = self.analysis.as_ref().map(|a| a.class_counts());
        for (i, name) in CLASS_NAMES.iter().enumerate() {
            let cells = counts.map_or(0, |c| c[i]);
            writeln!(
                w,
                "{name},{},{:.1},{cells}",
                self.region_exact_estimates[i],
                100.0 * self.region_exact_estimates[i] as f64 / total,
            )?;
        }
        Ok(())
    }

    fn write_exact_by_region_size(&self) -> std::io::Result<()> {
        let mut w = self.create("exact_by_region_size.csv")?;
        write!(w, "region_class")?;
        for s in SIZE_NAMES {
            write!(w, ",{s}")?;
        }
        writeln!(w, ",total")?;
        for (i, name) in CLASS_NAMES.iter().enumerate() {
            let total: u64 = self.region_size_exact[i].iter().sum();
            if total == 0 {
                continue;
            }
            write!(w, "{name}")?;
            for &v in &self.region_size_exact[i] {
                write!(w, ",{v}")?;
            }
            writeln!(w, ",{total}")?;
        }
        Ok(())
    }

    /// Winner-rank distributions sliced by region class and by block size — the
    /// cross-tab the doc asks for so luma rank-3 gating can be made conditional
    /// (e.g. "rank 3 wins often in DirectionalEdge/TextLike, rarely in Flat").
    fn write_rank_cross_tabs(&self) -> std::io::Result<()> {
        let mut w = self.create("rank_by_region.csv")?;
        write!(w, "kind,region_class")?;
        for r in 1..=MAX_RANK {
            write!(
                w,
                ",rank{}",
                if r == MAX_RANK {
                    format!("{r}plus")
                } else {
                    r.to_string()
                }
            )?;
        }
        writeln!(w, ",total")?;
        for (ki, kname) in KIND_NAMES.iter().enumerate() {
            for (ri, rname) in CLASS_NAMES.iter().enumerate() {
                let rank = &self.kinds[ki].rank_by_region[ri];
                let total: u64 = rank.iter().sum();
                if total == 0 {
                    continue;
                }
                write!(w, "{kname},{rname}")?;
                for r in 1..=MAX_RANK {
                    write!(w, ",{}", rank[r])?;
                }
                writeln!(w, ",{total}")?;
            }
        }

        let mut w = self.create("rank_by_size.csv")?;
        write!(w, "kind,block_size")?;
        for r in 1..=MAX_RANK {
            write!(
                w,
                ",rank{}",
                if r == MAX_RANK {
                    format!("{r}plus")
                } else {
                    r.to_string()
                }
            )?;
        }
        writeln!(w, ",total")?;
        for (ki, kname) in KIND_NAMES.iter().enumerate() {
            for (si, sname) in SIZE_NAMES.iter().enumerate() {
                let rank = &self.kinds[ki].rank_by_size[si];
                let total: u64 = rank.iter().sum();
                if total == 0 {
                    continue;
                }
                write!(w, "{kname},{sname}")?;
                for r in 1..=MAX_RANK {
                    write!(w, ",{}", rank[r])?;
                }
                writeln!(w, ",{total}")?;
            }
        }
        Ok(())
    }

    /// How often exact residual bits flipped the winner vs the approx-bit model
    /// — the justification (or not) for staged FastRd→FullRd search.
    fn write_exact_changed_winner(&self) -> std::io::Result<()> {
        let mut w = self.create("exact_changed_winner.csv")?;
        writeln!(
            w,
            "kind,decisions_with_both_costs,exact_changed_winner,changed_pct"
        )?;
        for (ki, name) in KIND_NAMES.iter().enumerate() {
            let t = &self.kinds[ki];
            if t.flip_total == 0 {
                continue;
            }
            writeln!(
                w,
                "{name},{},{},{:.1}",
                t.flip_total,
                t.flip_changed,
                100.0 * t.flip_changed as f64 / t.flip_total as f64,
            )?;
        }
        Ok(())
    }

    /// Split-built-but-lost waste by region class, counted and weighted by the
    /// discarded exact estimates the lost split branch consumed.
    fn write_split_waste_by_region(&self) -> std::io::Result<()> {
        let mut w = self.create("split_waste_by_region.csv")?;
        writeln!(
            w,
            "kind,region_class,split_decisions,split_built_but_lost,lost_exact_estimates"
        )?;
        for ki in [DecisionKind::Tu as usize, DecisionKind::Cu as usize] {
            let t = &self.kinds[ki];
            for (ri, rname) in CLASS_NAMES.iter().enumerate() {
                let decisions: u64 = t.rank_by_region[ri].iter().sum();
                if decisions == 0 {
                    continue;
                }
                writeln!(
                    w,
                    "{},{rname},{decisions},{},{}",
                    KIND_NAMES[ki],
                    t.split_lost_count_by_region[ri],
                    t.split_lost_weight_by_region[ri],
                )?;
            }
        }
        Ok(())
    }

    fn write_ctu_heatmap(&self) -> std::io::Result<()> {
        let mut w = self.create("ctu_heatmap.csv")?;
        writeln!(
            w,
            "ctu,cx,cy,code_blocks,exact_estimates,trial_exact,final_exact,\
             luma_decisions,luma_losers,chroma_decisions,close_calls,\
             split_built_but_lost,final_residual_bytes"
        )?;
        for (i, c) in self.ctus.iter().enumerate() {
            let cx = i as u32 % self.ctb_cols;
            let cy = i as u32 / self.ctb_cols;
            let bytes = c.final_frac_bits as f64 / CabacEstimator::SCALE as f64 / 8.0;
            writeln!(
                w,
                "{i},{cx},{cy},{},{},{},{},{},{},{},{},{},{:.0}",
                c.code_blocks,
                c.exact_estimates,
                c.trial_exact_estimates,
                c.final_exact_estimates,
                c.luma_decisions,
                c.luma_losers,
                c.chroma_decisions,
                c.close_calls,
                c.split_built_but_lost,
                bytes,
            )?;
        }
        Ok(())
    }

    fn write_events(&self) -> std::io::Result<()> {
        let mut w = self.create("search_events.csv")?;
        writeln!(
            w,
            "kind,ctu,x,y,log2_size,region_class,n_candidates,winner_rank,\
             best_cost,runner_up_cost,loser_exact_estimates,exact_changed_winner,\
             split_built_but_lost"
        )?;
        for e in &self.decisions {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{:.1},{:.1},{},{},{}",
                KIND_NAMES[e.kind as usize],
                e.ctu,
                e.x,
                e.y,
                e.log2_size,
                CLASS_NAMES[e.region_class as usize],
                e.n_candidates,
                e.winner_rank,
                e.best_cost,
                e.runner_up_cost,
                e.loser_exact_estimates,
                e.exact_changed_winner as u8,
                e.split_built_but_lost as u8,
            )?;
        }
        Ok(())
    }

    fn create(&self, name: &str) -> std::io::Result<std::fs::File> {
        std::fs::File::create(self.dir.join(name))
    }
}

#[inline]
fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}
