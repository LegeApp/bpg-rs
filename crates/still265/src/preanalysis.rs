//! Source-derived spatial preanalysis: a cheap, read-only, per-32x32-cell
//! complexity/structure map used to steer the encoder's search budget
//! non-uniformly (spend more RD effort on structured regions, prune harder on
//! flat ones).
//!
//! This is the in-encoder, hot-path-friendly cousin of the offline Python/
//! OpenCV profiler in `psnrtune-dev/psnrtune_characteristics.py`: it mirrors
//! the same feature *semantics* (local variance / flat share, Sobel gradient +
//! edge density, gradient-orientation entropy + dominance, high-pass noise
//! proxy, chroma activity) but reimplements them as small integer kernels over
//! the `u16` source planes — no `image`/OpenCV deps, no whole-image
//! connected-component / Canny / Otsu passes. The Python profiler's per-image
//! corpus profiles serve as the offline oracle for tuning the thresholds here.
//!
//! The map is built once per picture (`analyze`) and consulted read-only during
//! mode decision (`AnalysisMaps::policy_at`) and adaptive-QP planning
//! (`importance_at`). The map itself is read-only; its `importance` scalar drives
//! both search-budget steering (no syntax change) and the encoder's per-CU
//! adaptive QP. Steering applies to production tiers whose templates enable it;
//! `Placebo` keeps inert search steering and uniform QP.

use crate::Effort;
use crate::encoder::Source;

/// log2 of the analysis cell size in luma samples (32x32 cells, four per CTB).
pub const CELL_LOG2: u8 = 5;
const CELL: u32 = 1 << CELL_LOG2;

/// Number of [`RegionClass`] variants (for the stats histogram).
pub const NUM_CLASSES: usize = 7;

/// Coarse content class of a 32x32 source cell. Ordered by "protect priority"
/// for large-block aggregation (see [`AnalysisMaps::policy_at`]); higher-index
/// structured classes win when a 64x64 CU spans several cells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegionClass {
    /// Smooth, near-constant — prune angular hard, exit split early.
    #[default]
    Flat,
    /// Low-detail ramp — few hard edges; mild pruning.
    Gradient,
    /// Strong chroma activity over otherwise unstructured luma — spend more
    /// chroma RD.
    ChromaCritical,
    /// Random/grainy high-frequency without coherent edges — don't chase
    /// angular detail that quantizes away.
    Noisy,
    /// Natural texture — normal search.
    Texture,
    /// One dominant gradient direction (a clean edge) — keep angular search.
    DirectionalEdge,
    /// Dense, axis-aligned, low-entropy edges (text/line-art proxy) — protect
    /// search hardest.
    TextLike,
}

impl RegionClass {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Per-cell extracted features (8-bit-normalized units; `*_q8` are
/// fixed-point shares in 0..=256).
#[derive(Clone, Copy, Debug, Default)]
pub struct CellAnalysis {
    pub class: RegionClass,
    pub variance: u32,
    pub edge_density_q8: u16,
    pub flat_ratio_q8: u16,
    pub orient_entropy_q8: u16,
    pub dir_dominance_q8: u16,
    pub noise: u16,
    pub chroma_activity: u16,
    // --- Per-cell means (8-bit-normalized). Local features above are computed
    // in the per-cell pass; the *contextual* features below are filled by a
    // cheap second pass over the whole cell grid, so a cell's distinctiveness
    // relative to the rest of the picture can be judged. Ported in spirit from
    // the global-contrast/center-prior saliency
    // features of Xu & Wu (2014) and the center-surround static saliency of Wei
    // et al. (2018) — see `saliency/`.
    pub luma_mean: u16,
    pub cb_mean: u16,
    pub cr_mean: u16,
    /// |cell mean − image mean| in (luma, Cb, Cr), L1, scaled to q8 (0..=256):
    /// how distinct this cell's colour/brightness is from the whole picture.
    pub global_contrast_q8: u16,
    /// |cell mean − neighbour-cell mean| (luma+chroma L1), scaled to q8: local
    /// center-surround pop-out, more robust than global contrast for grading
    /// complexity among photographic regions.
    pub center_surround_q8: u16,
    /// Resolved 0..=256 visual-importance weight: high for structured/distinct
    /// cells (edges, text, high-contrast subjects), low for flat/noisy
    /// background. The single scalar consumed by rough-mode pruning and adaptive
    /// QP. Zero unless the contextual pass ran.
    pub importance_q8: u16,
}

/// Read-only per-picture analysis map on a 32x32 cell grid.
pub struct AnalysisMaps {
    cells_x: u32,
    cells_y: u32,
    cells: Vec<CellAnalysis>,
    /// Picture mean of `log2(1 + cell.variance)`. The reference point for the
    /// bidirectional variance AQ ([`aq_qp_offset_variance`]): a cell's QP offset
    /// is proportional to how its log-variance deviates from this mean. 0 when
    /// the map is empty / the analysis pass didn't run.
    frame_mean_log2var: f32,
    /// Picture min/max of `log2(1 + cell.variance)` — the dynamic range the
    /// rank-based `PositiveProbe` control map ([`aq_qp_offset_positive`]) spreads
    /// its shrink offset across. Both 0 when the map is empty.
    frame_min_log2var: f32,
    frame_max_log2var: f32,
    /// Picture mean of the chroma-aware activity
    /// (`log2(1 + luma_var) + CHROMA_AQ_WEIGHT * log2(1 + chroma_activity)`),
    /// the reference point for [`aq_qp_offset_chroma`] (Prangnell 2017-style
    /// luma+chroma QG activity). 0 when the map is empty.
    frame_mean_activity: f32,
    /// Weight applied to `cell.chroma_activity` when reconstructing an x265
    /// `acEnergyCu`-style 16x16-block AC energy for the auto-variance AQ modes
    /// (per-cell `qp_adj0` is recomputed on demand from this;
    /// see [`Self::autovar_qp_adj0_at`]). Depends on the chroma format: 0 for
    /// monochrome, 64 for 4:2:0 (8x8 chroma block), 128 for 4:2:2, 256 for
    /// 4:4:4.
    autovar_chroma_weight: f32,
    /// Picture mean of the auto-variance `qp_adj0 = (energy + 1)^0.1` — x265
    /// `AUTO_VARIANCE`'s `avg_adj`. 0 when the map is empty.
    autovar_avg_adj: f32,
    /// x265's bias-corrected reference point
    /// `avg_adj − 0.5·(avg_adj_pow2 − modeTwoConst)/avg_adj` (modeTwoConst = 11
    /// for the 16x16 quant-group size). The value each cell's `qp_adj0` is
    /// compared against in [`aq_qp_offset_autovar`]. 0 when the map is empty.
    autovar_avg_adj_final: f32,
}

/// Optional external per-QG QP-offset map for two-pass AQ experiments.
///
/// Loaded from `BPG_AQ_OFFSET_MAP` when present. The file is intentionally
/// simple CSV/text: each useful row starts with `x,y,offset`, where `x`/`y` are
/// luma sample coordinates (normally QG origins — 32x32 by default, 16x16 with
/// `aq_qg = 16`) and `offset` is added to the slice QP. Extra columns are
/// ignored, so diagnostic tools can append labels after the first three fields.
pub struct AqOffsetMap {
    cells_x: u32,
    cells_y: u32,
    /// log2 of the map's cell size in luma samples — the active QG size (5 =
    /// 32x32 default, 4 = 16x16 opt-in).
    cell_log2: u8,
    offsets: Vec<i8>,
}

/// Encoder-side search-budget steering derived from a cell's [`RegionClass`].
///
/// The first four fields only widen or narrow *search* and never change the
/// emitted syntax. [`Self::force_leaf`] and [`Self::force_split`] change which
/// partition is chosen — still a valid, decodable bitstream, but deliberately
/// lossy search shortcuts. They are `false` (no-op) on the byte-exact reference
/// tiers.
#[derive(Clone, Copy, Debug)]
pub struct SearchPolicy {
    /// `Some(true)`: force angular-mode pruning; `Some(false)`: guard — never
    /// prune angular even if the local block looks flat; `None`: defer to the
    /// local per-block variance heuristic.
    pub angular_prune: Option<bool>,
    /// `false`: suppress the lossy CU-split early-termination on this CU (guard
    /// for structured regions).
    pub allow_early_term: bool,
    /// Added to [`crate::encoder`]'s luma RD-candidate count (clamped >= 1).
    pub luma_rd_bias: i8,
    /// Added to the chroma RD-candidate count (clamped >= 1).
    pub chroma_rd_bias: i8,
    /// Force this CU to be coded as a leaf (skip the split RD trial) on
    /// very-low-importance flat regions. `false` defers to the normal split RD
    /// comparison. Always `false` when the selected template disables steering.
    pub force_leaf: bool,
    /// Prefer building split structure before the leaf path on structured
    /// regions. This is a search-order hint unless upgraded to
    /// [`Self::force_split`] by the effort resolver.
    pub prefer_split: bool,
    /// Force this CU/TU to split on high-importance structured regions. `false`
    /// defers to the normal split RD comparison. Always `false` when the
    /// selected template disables steering.
    pub force_split: bool,
    /// RMD cost-curve pruning factor for luma mode candidates. `Some(f)` drops
    /// rough-mode candidates whose cost exceeds `f ×` the best (shi2013 knee /
    /// xiong2016 average-cost); `None` keeps the full effort-derived candidate
    /// set.
    pub rmd_prune_factor: Option<f64>,
}

/// No-op policy: defer entirely to the existing per-tier/per-block heuristics.
pub const INERT: SearchPolicy = SearchPolicy {
    angular_prune: None,
    allow_early_term: true,
    luma_rd_bias: 0,
    chroma_rd_bias: 0,
    force_leaf: false,
    prefer_split: false,
    force_split: false,
    rmd_prune_factor: None,
};

/// Whether an external AQ offset map was requested. This is the syntax gate:
/// the PPS writer and encoder must both agree that `cu_qp_delta` syntax can
/// appear before the map is loaded with picture dimensions.
pub fn external_aq_map_requested() -> bool {
    std::env::var("BPG_AQ_OFFSET_MAP")
        .ok()
        .map(|s| !s.trim().is_empty() && s.trim() != "0")
        .unwrap_or(false)
}

/// Load `BPG_AQ_OFFSET_MAP` for the current picture. `qg_log2` is the active
/// quantization-group size (5 = 32x32, 4 = 16x16) — the map's cell grid follows
/// it, so row coordinates snap to QG origins of that size. Bad rows are
/// ignored; if no valid row lands inside the image's QG grid, the caller gets
/// `None`.
pub fn load_external_aq_offset_map(width: u32, height: u32, qg_log2: u8) -> Option<AqOffsetMap> {
    let path = std::env::var("BPG_AQ_OFFSET_MAP").ok()?;
    let path = path.trim();
    if path.is_empty() || path == "0" {
        return None;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("BPG_AQ_OFFSET_MAP: could not read {path}: {err}");
            return None;
        }
    };
    let cell = 1u32 << qg_log2;
    let cells_x = width.div_ceil(cell).max(1);
    let cells_y = height.div_ceil(cell).max(1);
    let mut offsets = vec![0i8; (cells_x * cells_y) as usize];
    let mut used = 0u32;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|s| !s.is_empty());
        let Some(x) = fields.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(y) = fields.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(offset) = fields.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        if x < 0 || y < 0 {
            continue;
        }
        let cx = ((x as u32) >> qg_log2).min(cells_x.saturating_sub(1));
        let cy = ((y as u32) >> qg_log2).min(cells_y.saturating_sub(1));
        offsets[(cy * cells_x + cx) as usize] = offset.clamp(-12, 12) as i8;
        used += 1;
    }
    if used == 0 {
        eprintln!("BPG_AQ_OFFSET_MAP: no usable x,y,offset rows in {path}");
        None
    } else {
        Some(AqOffsetMap {
            cells_x,
            cells_y,
            cell_log2: qg_log2,
            offsets,
        })
    }
}

// --- Classification thresholds (8-bit units / q8 shares). Seeded from the
// psnrtune-dev feature distributions; tuned empirically on the test-set. ---
const FLAT_VAR: u32 = 12; // near-constant cell
const FLAT_EDGE_Q8: u16 = 13; // ~5% edge pixels
const GRAD_EDGE_Q8: u16 = 26; // ~10%
const EDGE_DENSE_Q8: u16 = 38; // ~15%
const TEXT_EDGE_Q8: u16 = 64; // ~25%
const DIR_DOMINANT_Q8: u16 = 128; // one direction >= 50% of edges
const AXIS_ALIGNED_Q8: u16 = 160; // H+V directions >= ~62% of edges
const LOW_ENTROPY_Q8: u16 = 110; // aligned/structured orientations
const HIGH_ENTROPY_Q8: u16 = 150; // isotropic texture
const TEXTURE_VAR: u32 = 64;
const NOISE_HI: u16 = 7;
const CHROMA_HI: u16 = 96; // combined Cb+Cr 8-bit variance — genuinely busy chroma

/// Build the per-cell analysis map from the source picture.
///
/// Run when search steering or AQ needs source analysis. A cheap second
/// cell-grid-level pass fills the contextual features (`global_contrast_q8`,
/// `center_surround_q8`,
/// `importance_q8`) that drive adaptive QP and rough-mode pruning, on top of the
/// per-cell local features that drive `class`.
pub fn analyze(width: u32, height: u32, bit_depth: u8, cat: u8, src: Source<'_>) -> AnalysisMaps {
    let cells_x = width.div_ceil(CELL).max(1);
    let cells_y = height.div_ceil(CELL).max(1);
    let mut cells = Vec::with_capacity((cells_x * cells_y) as usize);

    let shift = (bit_depth - 8) as u32;
    let luma_stride = width as usize;
    // Clamped 8-bit luma reader.
    let lum = |x: i64, y: i64| -> i32 {
        let sx = x.clamp(0, width as i64 - 1) as usize;
        let sy = y.clamp(0, height as i64 - 1) as usize;
        (src.y[sy * luma_stride + sx] >> shift) as i32
    };

    let (cw, ch) = chroma_dims(width, height, cat);
    let (cell_cw, cell_ch) = chroma_cell_size(cat); // chroma samples per luma cell

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let x0 = (cx * CELL) as i64;
            let y0 = (cy * CELL) as i64;
            cells.push(analyze_cell(
                x0, y0, width, height, &lum, src, cat, cw, ch, cx, cy, cell_cw, cell_ch, shift,
            ));
        }
    }

    fill_contextual_features(&mut cells, cells_x, cells_y);

    let autovar_chroma_weight = autovar_chroma_weight(cat);
    let (
        frame_mean_log2var,
        frame_min_log2var,
        frame_max_log2var,
        frame_mean_activity,
        autovar_avg_adj,
        autovar_avg_adj_final,
    ) = if cells.is_empty() {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let n = cells.len() as f32;
        let mut sum_lv = 0.0f32;
        let mut sum_act = 0.0f32;
        let mut min_lv = f32::INFINITY;
        let mut max_lv = f32::NEG_INFINITY;
        let mut sum_adj = 0.0f32;
        let mut sum_adj_pow2 = 0.0f32;
        for c in &cells {
            let lv = ((1 + c.variance) as f32).log2();
            sum_lv += lv;
            sum_act += combined_activity(c.variance, c.chroma_activity);
            min_lv = min_lv.min(lv);
            max_lv = max_lv.max(lv);
            let adj = autovar_qp_adj0(c.variance, c.chroma_activity, autovar_chroma_weight);
            sum_adj += adj;
            sum_adj_pow2 += adj * adj;
        }
        // x265 slicetype.cpp AUTO_VARIANCE picture stats: avg_adj is the plain
        // mean of qp_adj0; the *final* reference point is bias-corrected by the
        // second moment around modeTwoConst (= 11 for qgSize 16):
        // `avg_adj − 0.5·(avg_adj_pow2 − 11)/avg_adj`. qp_adj0 >= 1 always
        // (energy >= 0), so avg_adj >= 1 and the division is safe.
        let avg_adj = sum_adj / n;
        let avg_adj_pow2 = sum_adj_pow2 / n;
        let avg_adj_final = avg_adj - 0.5 * (avg_adj_pow2 - AUTOVAR_MODE_TWO_CONST) / avg_adj;
        (
            sum_lv / n,
            min_lv,
            max_lv,
            sum_act / n,
            avg_adj,
            avg_adj_final,
        )
    };

    AnalysisMaps {
        cells_x,
        cells_y,
        cells,
        frame_mean_log2var,
        frame_min_log2var,
        frame_max_log2var,
        frame_mean_activity,
        autovar_chroma_weight,
        autovar_avg_adj,
        autovar_avg_adj_final,
    }
}

/// Second pass over the finished cell grid: derive each cell's distinctiveness
/// relative to the whole picture
/// (`global_contrast`) and to its immediate neighbours (`center_surround`),
/// then fold those plus the local structure features into a single
/// `importance_q8` weight. O(cells) — the per-pixel work is already done.
fn fill_contextual_features(cells: &mut [CellAnalysis], cells_x: u32, cells_y: u32) {
    let n = cells.len().max(1) as i64;
    // Image-global means (average of per-cell means — cheap and good enough at
    // 32x32 granularity).
    let (mut sy, mut scb, mut scr) = (0i64, 0i64, 0i64);
    for c in cells.iter() {
        sy += c.luma_mean as i64;
        scb += c.cb_mean as i64;
        scr += c.cr_mean as i64;
    }
    let (gy, gcb, gcr) = (sy / n, scb / n, scr / n);

    let cx_n = cells_x as usize;
    let cy_n = cells_y as usize;
    // Pass 2 only writes the *derived* fields (`global_contrast_q8`,
    // `center_surround_q8`, `importance_q8`); the per-cell means it reads are set
    // in pass 1 and never mutated here, so neighbour reads can come straight from
    // `cells` (no snapshot copy needed).
    let mean_of = |c: &CellAnalysis| (c.luma_mean as i64, c.cb_mean as i64, c.cr_mean as i64);

    for gy_idx in 0..cy_n {
        for gx_idx in 0..cx_n {
            let i = gy_idx * cx_n + gx_idx;
            let (ly, lcb, lcr) = mean_of(&cells[i]);

            // Global contrast: L1 distance of (luma,Cb,Cr) mean from image mean.
            let global = (ly - gy).abs() + (lcb - gcb).abs() + (lcr - gcr).abs();

            // Center-surround: L1 distance from the average of up to 8 neighbours.
            let (mut ny, mut ncb, mut ncr, mut cnt) = (0i64, 0i64, 0i64, 0i64);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = gx_idx as i64 + dx;
                    let nyi = gy_idx as i64 + dy;
                    if nx < 0 || nyi < 0 || nx >= cx_n as i64 || nyi >= cy_n as i64 {
                        continue;
                    }
                    let (vy, vcb, vcr) = mean_of(&cells[nyi as usize * cx_n + nx as usize]);
                    ny += vy;
                    ncb += vcb;
                    ncr += vcr;
                    cnt += 1;
                }
            }
            let surround = if cnt > 0 {
                ((ly - ny / cnt).abs() + (lcb - ncb / cnt).abs() + (lcr - ncr / cnt).abs()) as i64
            } else {
                0
            };

            // Scale L1-over-3-channels (each 0..255) into q8 (0..=256):
            // ×2 so a moderate ~128-level total contrast (avg ~43/channel)
            // saturates — small photographic contrasts still register.
            let to_q8 = |v: i64| -> u16 { (v * 2).clamp(0, 256) as u16 };
            cells[i].global_contrast_q8 = to_q8(global);
            cells[i].center_surround_q8 = to_q8(surround);
            cells[i].importance_q8 = compute_importance(&cells[i]);
        }
    }
}

/// Fold local structure + contextual contrast into a 0..=256 visual-importance
/// weight: high for edges/text/high-contrast subjects, low for flat/noisy
/// background. Heuristic; the exact curve is tuned offline against the oracle
/// corpus. Consumed by rough-mode pruning and adaptive QP.
/// Map a quantization group's `importance` to a luma QP offset for adaptive
/// quantization. The offset is always `>= 0` (raise QP, shrink) — lossy still
/// encoding trades quality *for* size, so it never spends extra bits — but it is
/// importance-weighted: important regions (faces/edges/text) are raised the least
/// (`+1`) and flat/noisy background the most (`+8`), concentrating the loss where
/// the eye is least likely to fix. This is the single importance→QP map, applied
/// identically by the analysis (quantization) and writer (delta) passes so they
/// agree on every QG's QP — see [`crate::encoder`]'s adaptive-QP methods.
///
/// AQ is also why there is no per-region `lambda_scale`: raising a QG's QP
/// already raises its `lambda` (mode decision and RDOQ both derive lambda from
/// the per-CU QP), so steering lambda by importance *as well* would
/// double-count the same signal.
pub fn aq_qp_offset(importance_q8: u16) -> i32 {
    // `t` in 0..=1: 1 for fully important cells, 0 for flat/noisy.
    let t = (importance_q8.min(256) as f64) / 256.0;
    (1.0 + (1.0 - t) * 7.0).round() as i32 // +1..+8
}

/// Default magnitude (in QP steps) the bidirectional variance AQ offset is
/// clamped to when the caller does not pass an explicit clamp. The configurable
/// `--aq` presets use `aq_clamp = 2`; this is the legacy `BPG_PLACEBO_AQ`
/// default.
pub const VAR_AQ_CLAMP: f32 = 8.0;

/// Weight applied to the chroma log-activity term when combining it with the
/// luma log-variance into a single chroma-aware QG activity (Prangnell 2017's
/// luma+chroma activity for QP selection). Chroma carries less perceptual rate
/// than luma, so it is down-weighted rather than summed 1:1.
pub const CHROMA_AQ_WEIGHT: f32 = 0.35;

/// Chroma-aware QG activity: luma log-variance plus a down-weighted chroma
/// log-activity. `chroma_activity` is the per-cell combined Cb+Cr variance in
/// 8-bit units (already normalized across chroma formats by [`analyze_cell`]),
/// so it can be folded directly against the luma term.
#[inline]
fn combined_activity(luma_var: u32, chroma_activity: u16) -> f32 {
    let luma = ((1 + luma_var) as f32).log2();
    let chroma = ((1 + chroma_activity as u32) as f32).log2();
    luma + CHROMA_AQ_WEIGHT * chroma
}

/// Bidirectional, rate-neutral variance AQ QP offset (x265 `aq-mode`-style),
/// the alternative to the one-directional perceptual [`aq_qp_offset`]. The
/// offset is proportional to how a cell's `log2(1 + variance)` deviates from the
/// picture mean (`frame_mean_log2var`), so it both *raises* and *lowers* QP
/// around the slice QP (net roughly rate-neutral) instead of only shrinking.
///
/// - `perceptual = true` (x265 aq-mode 2 direction): high-variance/texture cells
///   get **higher** QP (masking hides the loss), flat cells get **lower** QP
///   (anti-banding). Optimizes perceived quality / SSIM.
/// - `perceptual = false` (PSNR/SSE-leaning): the opposite sign — high-variance
///   cells get **lower** QP (that is where reducing SSE costs the fewest bits).
///   Optimizes PSNR-at-equal-size.
///
/// `strength` scales the deviation (x265's `--aq-strength`, ~1.0 nominal); the
/// result is clamped to `±clamp` QP steps. Caller adds this to the slice QP and
/// clamps to the valid `[0, 51]` range.
pub fn aq_qp_offset_variance(
    variance: u32,
    frame_mean_log2var: f32,
    strength: f32,
    perceptual: bool,
    clamp: f32,
) -> i32 {
    let log2var = ((1 + variance) as f32).log2();
    let mut adj = strength * (log2var - frame_mean_log2var);
    if !perceptual {
        adj = -adj;
    }
    adj.round().clamp(-clamp, clamp) as i32
}

/// Chroma-aware perceptual variance AQ QP offset (Prangnell 2017 luma+chroma
/// activity). Like [`aq_qp_offset_variance`]'s `perceptual` direction, but the
/// per-cell activity folds in chroma log-activity ([`combined_activity`]) and is
/// compared to the picture-mean *activity* (`frame_mean_activity`). High
/// luma-or-chroma activity ⇒ higher QP (masking), flat-in-both ⇒ lower QP
/// (anti-banding). `strength` scales the deviation, clamped to `±clamp`.
pub fn aq_qp_offset_chroma(
    luma_var: u32,
    chroma_activity: u16,
    frame_mean_activity: f32,
    strength: f32,
    clamp: f32,
) -> i32 {
    let activity = combined_activity(luma_var, chroma_activity);
    let adj = strength * (activity - frame_mean_activity);
    adj.round().clamp(-clamp, clamp) as i32
}

/// Rank-based positive-only shrink offset — the in-tree mirror of the
/// experiment harness `positive` control. Every cell raises QP by `+1..+8`
/// according to where its `log2(1 + variance)` ranks between the picture
/// min/max, then clamped to `+clamp`. This is a deliberate size-bias control
/// (effective higher QP), **not** a quality feature; it is exposed only as
/// `AqMode::PositiveProbe` for parity with the experiment sweep.
pub fn aq_qp_offset_positive(
    luma_var: u32,
    frame_min_log2var: f32,
    frame_max_log2var: f32,
    clamp: f32,
) -> i32 {
    let log2var = ((1 + luma_var) as f32).log2();
    let span = frame_max_log2var - frame_min_log2var;
    let rank = if span <= f32::EPSILON {
        0.0
    } else {
        ((log2var - frame_min_log2var) / span).clamp(0.0, 1.0)
    };
    (1.0 + 7.0 * rank).round().clamp(1.0, clamp.max(1.0)) as i32
}

/// x265's `modeTwoConst` for the 16x16 quantization-group size — the constant
/// x265 uses in both the `AUTO_VARIANCE` picture-mean bias correction and the
/// `AUTO_VARIANCE_BIASED` dark/flat bias term (slicetype.cpp).
pub const AUTOVAR_MODE_TWO_CONST: f32 = 11.0;

/// x265 `acEnergyCu`-style chroma-plane energy weight per chroma format `cat`
/// (0 = mono, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4): the number of chroma samples
/// (Cb + Cr summed variance × samples-per-plane) a 16x16 luma block spans —
/// 2×8x8 = 64 per plane for 4:2:0, 128 for 4:2:2, 256 for 4:4:4.
fn autovar_chroma_weight(cat: u8) -> f32 {
    match cat {
        0 => 0.0,
        1 => 64.0,
        2 => 128.0,
        _ => 256.0,
    }
}

/// Per-cell `qp_adj0 = (energy + 1)^0.1` for the auto-variance AQ modes, where
/// `energy` reconstructs x265's `acEnergyCu` (sum over planes of
/// `N_samples × variance`, i.e. the block's AC energy) for a 16x16 block:
/// `256·var_luma + weight·var_chroma` with the format-dependent chroma weight
/// from [`autovar_chroma_weight`].
///
/// Approximation vs. x265: x265 measures each 16x16 block's own variance,
/// while our cell features are one 32x32 luma variance (`cell.variance`) and
/// one summed Cb+Cr chroma variance (`cell.chroma_activity`) per cell — we use
/// them as the 16x16-block variances of every block in the cell, i.e. one
/// energy sample per 32x32 QG instead of four 16x16 samples. Bit-depth
/// correction (x265's `1/(1 << 2·(depth−8))`) is identically 1 here because
/// [`analyze`] already right-shifts samples to 8-bit before computing the cell
/// features, so the variances are already in 8-bit units.
#[inline]
fn autovar_qp_adj0(luma_var: u32, chroma_activity: u16, chroma_weight: f32) -> f32 {
    let energy = 256.0 * luma_var as f32 + chroma_weight * chroma_activity as f32;
    (energy + 1.0).powf(0.1)
}

/// x265 aq-mode 2/3 (`AUTO_VARIANCE` / `AUTO_VARIANCE_BIASED`) QP offset for
/// one quantization group, ported from slicetype.cpp with qgSize-16 constants:
///
/// ```text
/// strength'              = strength × avg_adj
/// offset                 = strength' × (qp_adj0 − avg_adj_final)
/// offset (biased, +=)      strength × (1 − 11 / qp_adj0²)
/// ```
///
/// `qp_adj0` is the cell's `(energy + 1)^0.1` ([`AnalysisMaps::autovar_qp_adj0_at`]),
/// `avg_adj` its picture mean, and `avg_adj_final` the bias-corrected picture
/// reference ([`AnalysisMaps::autovar_avg_adj_final`]). Direction matches
/// `Perceptual`: high-energy cells get **+QP** (masking), low-energy cells
/// **−QP** (anti-banding); the `biased` term additionally pushes low-energy
/// (dark/flat, `qp_adj0² < 11`) cells further negative. `strength` is x265's
/// `--aq-strength` (~1.0 nominal); the result is rounded and clamped to
/// `±clamp` QP steps. Caller adds this to the slice QP and clamps to `[0, 51]`.
pub fn aq_qp_offset_autovar(
    qp_adj0: f32,
    avg_adj: f32,
    avg_adj_final: f32,
    strength: f32,
    biased: bool,
    clamp: f32,
) -> i32 {
    if avg_adj <= 0.0 || qp_adj0 <= 0.0 {
        return 0; // empty/degenerate map ⇒ uniform QP
    }
    let mut adj = strength * avg_adj * (qp_adj0 - avg_adj_final);
    if biased {
        adj += strength * (1.0 - AUTOVAR_MODE_TWO_CONST / (qp_adj0 * qp_adj0));
    }
    adj.round().clamp(-clamp, clamp) as i32
}

fn compute_importance(c: &CellAnalysis) -> u16 {
    // Structure: edge density, with a bonus for clean directional edges / text.
    let edge = c.edge_density_q8 as i32;
    let dir_bonus = match c.class {
        RegionClass::DirectionalEdge | RegionClass::TextLike => 96,
        RegionClass::Texture => 32,
        _ => 0,
    };
    // Contextual pop-out (the new axis): the stronger of global/surround.
    let contrast = c.global_contrast_q8.max(c.center_surround_q8) as i32;
    // Noise drags importance down: grain quantizes away and isn't where the
    // eye fixes.
    let noise_pen = if c.noise as i32 >= NOISE_HI as i32 {
        96
    } else {
        0
    };

    (edge / 2 + contrast + dir_bonus - noise_pen).clamp(0, 256) as u16
}

#[allow(clippy::too_many_arguments)]
fn analyze_cell(
    x0: i64,
    y0: i64,
    width: u32,
    height: u32,
    lum: &impl Fn(i64, i64) -> i32,
    src: Source<'_>,
    cat: u8,
    cw: u32,
    ch: u32,
    cx: u32,
    cy: u32,
    cell_cw: u32,
    cell_ch: u32,
    shift: u32,
) -> CellAnalysis {
    let x1 = (x0 + CELL as i64).min(width as i64);
    let y1 = (y0 + CELL as i64).min(height as i64);
    let n = ((x1 - x0) * (y1 - y0)).max(1) as i64;

    let mut sum: i64 = 0;
    let mut sum_sq: i64 = 0;
    let mut edge_count: i64 = 0;
    let mut flat_count: i64 = 0;
    let mut noise_sum: i64 = 0;
    let mut weak_count: i64 = 0;
    // 4 gradient-direction bins: 0=horizontal, 1=vertical, 2=diag '\', 3=diag '/'.
    let mut dir = [0i64; 4];

    for y in y0..y1 {
        for x in x0..x1 {
            let c = lum(x, y);
            sum += c as i64;
            sum_sq += (c * c) as i64;

            // 3x3 Sobel (clamped at picture edges).
            let tl = lum(x - 1, y - 1);
            let tc = lum(x, y - 1);
            let tr = lum(x + 1, y - 1);
            let ml = lum(x - 1, y);
            let mr = lum(x + 1, y);
            let bl = lum(x - 1, y + 1);
            let bc = lum(x, y + 1);
            let br = lum(x + 1, y + 1);
            let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
            let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
            let grad = gx.abs() + gy.abs();

            if grad > EDGE_GRAD {
                edge_count += 1;
                let ax = gx.abs();
                let ay = gy.abs();
                let bin = if ax >= 2 * ay {
                    0
                } else if ay >= 2 * ax {
                    1
                } else if (gx > 0) == (gy > 0) {
                    2
                } else {
                    3
                };
                dir[bin] += 1;
            }
            if grad < FLAT_GRAD {
                flat_count += 1;
            }
            if grad < WEAK_GRAD {
                // High-pass residual vs 3x3 box mean (noise proxy in weak-edge
                // areas), mirroring psnrtune's median-blur residual.
                let box_mean = (tl + tc + tr + ml + c + mr + bl + bc + br) / 9;
                noise_sum += (c - box_mean).abs() as i64;
                weak_count += 1;
            }
        }
    }

    let mean = sum / n;
    let variance = ((sum_sq / n) - mean * mean).max(0) as u32;
    let edge_density_q8 = ((edge_count * 256) / n) as u16;
    let flat_ratio_q8 = ((flat_count * 256) / n) as u16;
    let noise = if weak_count > 0 {
        (noise_sum / weak_count) as u16
    } else {
        0
    };

    let edge_total = dir.iter().sum::<i64>().max(1);
    let dir_max = *dir.iter().max().unwrap();
    let dir_dominance_q8 = ((dir_max * 256) / edge_total) as u16;
    let axis_aligned_q8 = (((dir[0] + dir[1]) * 256) / edge_total) as u16;
    let orient_entropy_q8 = entropy4_q8(&dir, edge_total);

    let (chroma_activity, cb_mean, cr_mean) =
        chroma_cell_activity(src, cat, cw, ch, cx, cy, cell_cw, cell_ch, shift);

    let class = classify(
        variance,
        edge_density_q8,
        orient_entropy_q8,
        dir_dominance_q8,
        axis_aligned_q8,
        noise,
        chroma_activity,
    );

    CellAnalysis {
        class,
        variance,
        edge_density_q8,
        flat_ratio_q8,
        orient_entropy_q8,
        dir_dominance_q8,
        noise,
        chroma_activity,
        luma_mean: mean as u16,
        cb_mean,
        cr_mean,
        // Contextual features are filled by the second pass in `analyze` (only
        // when requested); leave them zero for the per-cell-only path.
        global_contrast_q8: 0,
        center_surround_q8: 0,
        importance_q8: 0,
    }
}

// Gradient magnitude thresholds (|gx|+|gy|, 8-bit Sobel). A flat ramp gives a
// small constant gradient; a hard edge gives a large one.
const EDGE_GRAD: i32 = 48;
const FLAT_GRAD: i32 = 12;
const WEAK_GRAD: i32 = 24;

/// Shannon entropy of a 4-bin histogram, scaled to q8 (0 = single bin,
/// 256 = uniform over 4 bins = 2 bits).
fn entropy4_q8(dir: &[i64; 4], total: i64) -> u16 {
    if total <= 0 {
        return 0;
    }
    let t = total as f64;
    let mut h = 0.0f64;
    for &c in dir {
        if c > 0 {
            let p = c as f64 / t;
            h -= p * p.log2();
        }
    }
    // Max entropy for 4 bins is 2 bits.
    ((h / 2.0) * 256.0).round().clamp(0.0, 256.0) as u16
}

#[allow(clippy::too_many_arguments)]
fn classify(
    variance: u32,
    edge_density_q8: u16,
    orient_entropy_q8: u16,
    dir_dominance_q8: u16,
    axis_aligned_q8: u16,
    noise: u16,
    chroma_activity: u16,
) -> RegionClass {
    // Structured-luma classes take priority (their guards matter most).
    if edge_density_q8 >= TEXT_EDGE_Q8
        && orient_entropy_q8 < LOW_ENTROPY_Q8
        && axis_aligned_q8 >= AXIS_ALIGNED_Q8
        && noise < NOISE_HI
    {
        return RegionClass::TextLike;
    }
    if edge_density_q8 >= EDGE_DENSE_Q8 && dir_dominance_q8 >= DIR_DOMINANT_Q8 {
        return RegionClass::DirectionalEdge;
    }
    if noise >= NOISE_HI && variance < TEXTURE_VAR {
        return RegionClass::Noisy;
    }
    if variance >= TEXTURE_VAR && orient_entropy_q8 >= HIGH_ENTROPY_Q8 {
        return RegionClass::Texture;
    }
    // Luma is unstructured below here. Flat/Gradient take precedence (a smooth,
    // uniformly-coloured region has *low* chroma variance, so it must not be
    // mislabelled ChromaCritical); ChromaCritical then claims the remaining
    // low-detail cells that carry genuinely busy chroma.
    if variance < FLAT_VAR && edge_density_q8 < FLAT_EDGE_Q8 {
        return RegionClass::Flat;
    }
    if edge_density_q8 < GRAD_EDGE_Q8 {
        return RegionClass::Gradient;
    }
    if chroma_activity >= CHROMA_HI {
        return RegionClass::ChromaCritical;
    }
    RegionClass::Texture
}

impl AnalysisMaps {
    /// An empty map, for tiers that need neither AQ nor search steering.
    pub fn empty() -> AnalysisMaps {
        AnalysisMaps {
            cells_x: 0,
            cells_y: 0,
            cells: Vec::new(),
            frame_mean_log2var: 0.0,
            frame_min_log2var: 0.0,
            frame_max_log2var: 0.0,
            frame_mean_activity: 0.0,
            autovar_chroma_weight: 0.0,
            autovar_avg_adj: 0.0,
            autovar_avg_adj_final: 0.0,
        }
    }

    /// Resolve the steering policy for a coding block at `(x0, y0)` of size
    /// `1 << log2_size`, given the encode effort.
    pub fn policy_at(&self, x0: u32, y0: u32, log2_size: u8, effort: Effort) -> SearchPolicy {
        crate::effort::template(effort).resolve_policy(
            self.region_class_at(x0, y0, log2_size),
            self.importance_at(x0, y0, log2_size),
        )
    }

    /// Fold `f` over every analysis cell that a `1 << log2_size` block at
    /// `(x0, y0)` covers (one cell for blocks <= 32x32, up to four for a 64x64
    /// CU). Returns `init` unchanged for an empty map. Shared by the
    /// block-aggregating queries below.
    fn fold_cells<T: Copy>(
        &self,
        x0: u32,
        y0: u32,
        log2_size: u8,
        init: T,
        f: impl Fn(T, &CellAnalysis) -> T,
    ) -> T {
        if self.cells.is_empty() {
            return init;
        }
        let size = 1u32 << log2_size;
        let cx0 = x0 >> CELL_LOG2;
        let cy0 = y0 >> CELL_LOG2;
        let cx1 = ((x0 + size - 1) >> CELL_LOG2).min(self.cells_x - 1);
        let cy1 = ((y0 + size - 1) >> CELL_LOG2).min(self.cells_y - 1);
        let mut acc = init;
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                acc = f(acc, &self.cells[(cy * self.cells_x + cx) as usize]);
            }
        }
        acc
    }

    /// The protect-priority-maximal class over the cells a block covers. For
    /// blocks <= one cell this is just the containing cell; for a 64x64 CU it
    /// takes the most-structured of the (up to four) covered cells, so a large
    /// CU that contains any edge/text is not pruned as if it were flat.
    pub fn region_class_at(&self, x0: u32, y0: u32, log2_size: u8) -> RegionClass {
        self.fold_cells(x0, y0, log2_size, RegionClass::Flat, |best, c| {
            if c.class.index() > best.index() {
                c.class
            } else {
                best
            }
        })
    }

    /// The maximal `importance_q8` over the cells a block covers (0..=256).
    /// Max — not average — so a large CU/quantization-group that contains any
    /// important structure is treated as important and not over-pruned /
    /// over-quantized, mirroring [`Self::region_class_at`]'s protect-the-structure
    /// aggregation. Returns 0 when the contextual pass didn't run (every cell's
    /// `importance_q8` is then 0) or the map is empty, so a caller that didn't
    /// request context gets a no-op.
    pub fn importance_at(&self, x0: u32, y0: u32, log2_size: u8) -> u16 {
        self.fold_cells(x0, y0, log2_size, 0u16, |best, c| best.max(c.importance_q8))
    }

    /// Spatial luma variance of the cell covering `(x0, y0)` — the raw local
    /// complexity signal the bidirectional variance AQ steers on (independent of
    /// the importance/classification machinery). 0 when the map is empty.
    pub fn variance_at(&self, x0: u32, y0: u32) -> u32 {
        if self.cells.is_empty() {
            return 0;
        }
        let cx = (x0 / CELL).min(self.cells_x - 1);
        let cy = (y0 / CELL).min(self.cells_y - 1);
        self.cells[(cy * self.cells_x + cx) as usize].variance
    }

    /// Combined Cb+Cr activity (8-bit-units variance) of the cell covering
    /// `(x0, y0)` — the chroma signal the chroma-aware AQ
    /// ([`aq_qp_offset_chroma`]) folds in. 0 when the map is empty.
    pub fn chroma_activity_at(&self, x0: u32, y0: u32) -> u16 {
        if self.cells.is_empty() {
            return 0;
        }
        let cx = (x0 / CELL).min(self.cells_x - 1);
        let cy = (y0 / CELL).min(self.cells_y - 1);
        self.cells[(cy * self.cells_x + cx) as usize].chroma_activity
    }

    /// Picture mean of `log2(1 + variance)` — the AQ reference point.
    pub fn frame_mean_log2var(&self) -> f32 {
        self.frame_mean_log2var
    }

    /// Picture mean chroma-aware activity — the [`aq_qp_offset_chroma`]
    /// reference point.
    pub fn frame_mean_log2activity(&self) -> f32 {
        self.frame_mean_activity
    }

    /// The auto-variance `qp_adj0 = (energy + 1)^0.1` of the cell covering
    /// `(x0, y0)` — the per-QG input to [`aq_qp_offset_autovar`]. Recomputed on
    /// demand from the cell's stored variance/chroma-activity (cheap: one `powf`
    /// per QG, memoized per QG by the encoder's AQ target cache). 1.0 when the
    /// map is empty (zero energy).
    pub fn autovar_qp_adj0_at(&self, x0: u32, y0: u32) -> f32 {
        if self.cells.is_empty() {
            return 1.0;
        }
        let cx = (x0 / CELL).min(self.cells_x - 1);
        let cy = (y0 / CELL).min(self.cells_y - 1);
        let c = &self.cells[(cy * self.cells_x + cx) as usize];
        autovar_qp_adj0(c.variance, c.chroma_activity, self.autovar_chroma_weight)
    }

    /// Picture mean of the auto-variance `qp_adj0` (x265's `avg_adj`) — scales
    /// the effective strength in [`aq_qp_offset_autovar`]. 0 when the map is
    /// empty.
    pub fn autovar_avg_adj(&self) -> f32 {
        self.autovar_avg_adj
    }

    /// Bias-corrected auto-variance picture reference point (x265's post-loop
    /// `avg_adj`) — what each cell's `qp_adj0` is compared against. 0 when the
    /// map is empty.
    pub fn autovar_avg_adj_final(&self) -> f32 {
        self.autovar_avg_adj_final
    }

    /// Picture min/max `log2(1 + variance)` — the [`aq_qp_offset_positive`]
    /// rank span.
    pub fn frame_min_log2var(&self) -> f32 {
        self.frame_min_log2var
    }

    pub fn frame_max_log2var(&self) -> f32 {
        self.frame_max_log2var
    }

    /// Per-class cell counts, for `--debug-stats` (image-global; not per-worker).
    pub fn class_counts(&self) -> [u64; NUM_CLASSES] {
        let mut h = [0u64; NUM_CLASSES];
        for c in &self.cells {
            h[c.class.index()] += 1;
        }
        h
    }
}

impl AqOffsetMap {
    /// QP offset for the quantization group containing `(x0, y0)` (the map's
    /// cell grid is the active QG size, recorded at construction).
    pub fn offset_at(&self, x0: u32, y0: u32) -> i32 {
        let cx = (x0 >> self.cell_log2).min(self.cells_x.saturating_sub(1));
        let cy = (y0 >> self.cell_log2).min(self.cells_y.saturating_sub(1));
        self.offsets[(cy * self.cells_x + cx) as usize] as i32
    }

    /// Build an in-memory map from a per-QG **measured activity** signal — the
    /// two-pass AQ path. `activity[cy*cells_x + cx]` is whatever pass-1 measured
    /// for that quantization group (here: coded residual coefficient energy, an
    /// actual CTU-search outcome rather than a source-variance prediction).
    ///
    /// The offset is the perceptual (x265 aq-mode-2) direction on log-activity:
    /// high-activity QGs (genuinely complex/textured, where masking hides loss)
    /// get **+QP**, low-activity QGs (flat, where banding shows) get **−QP**,
    /// around the picture-mean log-activity so the redistribution is ~rate
    /// neutral. `strength` scales the deviation; result clamped to `±clamp`.
    pub fn from_qg_activity(
        cells_x: u32,
        cells_y: u32,
        activity: &[f64],
        strength: f32,
        clamp: u8,
    ) -> AqOffsetMap {
        let n = activity.len();
        let logs: Vec<f32> = activity
            .iter()
            .map(|&a| (1.0 + a.max(0.0)).ln() as f32)
            .collect();
        let mean = if n == 0 {
            0.0
        } else {
            logs.iter().sum::<f32>() / n as f32
        };
        let cl = clamp.max(1) as f32;
        let offsets = logs
            .iter()
            .map(|&lv| (strength * (lv - mean)).round().clamp(-cl, cl) as i8)
            .collect();
        AqOffsetMap {
            cells_x,
            cells_y,
            // Two-pass measured AQ is defined on the 32x32 QG grid (its pass-1
            // activity accumulator is 32x32-only; `resolve_aq_qg_log2` forces
            // qg_log2 = 5 for `TwoPassMeasured`).
            cell_log2: CELL_LOG2,
            offsets,
        }
    }
}

/// Per-16x16-cell AQ signal grid for the `aq_qg = 16` opt-in.
///
/// The coarse [`AnalysisMaps`] cell grid (32x32, [`CELL_LOG2`]) feeds **both**
/// search steering and AQ; its size cannot change without perturbing steered
/// tiers byte-identically. When 16x16 quantization groups are active, the AQ
/// offset functions instead read this separate fine grid — the same per-cell
/// luma-variance / chroma-activity math as [`analyze`] at 16x16 granularity,
/// plus the frame-level statistics the offset functions normalize against
/// (computed over the fine grid, so the AQ redistribution stays ~rate-neutral
/// on its own grid). Search steering keeps reading the untouched coarse grid.
///
/// The offset functions themselves ([`aq_qp_offset_variance`],
/// [`aq_qp_offset_chroma`], [`aq_qp_offset_positive`],
/// [`aq_qp_offset_autovar`]) are grid-agnostic and shared between both grids.
pub struct FineAqGrid {
    cells_x: u32,
    cells_y: u32,
    luma_var: Vec<u32>,
    chroma_act: Vec<u16>,
    frame_mean_log2var: f32,
    frame_min_log2var: f32,
    frame_max_log2var: f32,
    frame_mean_activity: f32,
    autovar_chroma_weight: f32,
    autovar_avg_adj: f32,
    autovar_avg_adj_final: f32,
}

/// log2 of the fine AQ cell size (16x16 luma samples — one cell per 16x16 QG).
pub const FINE_CELL_LOG2: u8 = 4;
const FINE_CELL: u32 = 1 << FINE_CELL_LOG2;

/// Build the per-16x16 AQ signal grid ([`FineAqGrid`]) from the source
/// picture. Scalar mirror of [`analyze`]'s per-cell luma-variance and
/// chroma-activity math at 16x16 granularity (no Sobel / classification /
/// contextual passes — the fine grid only feeds the variance-family AQ offset
/// functions). Run only when 16x16 quantization groups are requested.
pub fn analyze_fine(
    width: u32,
    height: u32,
    bit_depth: u8,
    cat: u8,
    src: Source<'_>,
) -> FineAqGrid {
    let cells_x = width.div_ceil(FINE_CELL).max(1);
    let cells_y = height.div_ceil(FINE_CELL).max(1);
    let n_cells = (cells_x * cells_y) as usize;
    let mut luma_var = Vec::with_capacity(n_cells);
    let mut chroma_act = Vec::with_capacity(n_cells);

    let shift = (bit_depth - 8) as u32;
    let luma_stride = width as usize;
    let (cw, ch) = chroma_dims(width, height, cat);
    // Chroma samples spanned by one 16x16 luma cell, per chroma format.
    let (cell_cw, cell_ch) = match cat {
        0 => (0, 0),
        1 => (FINE_CELL / 2, FINE_CELL / 2),
        2 => (FINE_CELL / 2, FINE_CELL),
        _ => (FINE_CELL, FINE_CELL),
    };

    // Variance of one clamped plane rectangle in 8-bit units (mirrors
    // `analyze_cell` / `chroma_cell_activity`).
    let var_of = |plane: &[u16], stride: usize, x0: u32, y0: u32, x1: u32, y1: u32| -> i64 {
        let n = ((x1 - x0) * (y1 - y0)) as i64;
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = (plane[y as usize * stride + x as usize] >> shift) as i64;
                sum += v;
                sum_sq += v * v;
            }
        }
        let mean = sum / n;
        ((sum_sq / n) - mean * mean).max(0)
    };

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let x0 = cx * FINE_CELL;
            let y0 = cy * FINE_CELL;
            let x1 = (x0 + FINE_CELL).min(width);
            let y1 = (y0 + FINE_CELL).min(height);
            luma_var.push(var_of(src.y, luma_stride, x0, y0, x1, y1) as u32);

            let act = if cat == 0 {
                0
            } else {
                let cx0 = cx * cell_cw;
                let cy0 = cy * cell_ch;
                let cx1 = (cx0 + cell_cw).min(cw);
                let cy1 = (cy0 + cell_ch).min(ch);
                if cx1 <= cx0 || cy1 <= cy0 {
                    0
                } else {
                    let v_cb = var_of(src.cb, cw as usize, cx0, cy0, cx1, cy1);
                    let v_cr = var_of(src.cr, cw as usize, cx0, cy0, cx1, cy1);
                    (v_cb + v_cr).min(u16::MAX as i64) as u16
                }
            };
            chroma_act.push(act);
        }
    }

    // Frame statistics over the fine grid — the same folds `analyze` performs
    // over the coarse cells, so each offset function sees its reference point
    // computed on the grid it reads per-QG signals from.
    let autovar_chroma_weight = autovar_chroma_weight(cat);
    let n = luma_var.len().max(1) as f32;
    let mut sum_lv = 0.0f32;
    let mut sum_act = 0.0f32;
    let mut min_lv = f32::INFINITY;
    let mut max_lv = f32::NEG_INFINITY;
    let mut sum_adj = 0.0f32;
    let mut sum_adj_pow2 = 0.0f32;
    for (&v, &c) in luma_var.iter().zip(&chroma_act) {
        let lv = ((1 + v) as f32).log2();
        sum_lv += lv;
        sum_act += combined_activity(v, c);
        min_lv = min_lv.min(lv);
        max_lv = max_lv.max(lv);
        let adj = autovar_qp_adj0(v, c, autovar_chroma_weight);
        sum_adj += adj;
        sum_adj_pow2 += adj * adj;
    }
    let avg_adj = sum_adj / n;
    let avg_adj_pow2 = sum_adj_pow2 / n;
    let avg_adj_final = avg_adj - 0.5 * (avg_adj_pow2 - AUTOVAR_MODE_TWO_CONST) / avg_adj;

    FineAqGrid {
        cells_x,
        cells_y,
        luma_var,
        chroma_act,
        frame_mean_log2var: sum_lv / n,
        frame_min_log2var: min_lv,
        frame_max_log2var: max_lv,
        frame_mean_activity: sum_act / n,
        autovar_chroma_weight,
        autovar_avg_adj: avg_adj,
        autovar_avg_adj_final: avg_adj_final,
    }
}

impl FineAqGrid {
    #[inline]
    fn cell_idx(&self, x0: u32, y0: u32) -> usize {
        let cx = (x0 >> FINE_CELL_LOG2).min(self.cells_x - 1);
        let cy = (y0 >> FINE_CELL_LOG2).min(self.cells_y - 1);
        (cy * self.cells_x + cx) as usize
    }

    /// Luma variance of the 16x16 cell covering `(x0, y0)` (8-bit units).
    pub fn variance_at(&self, x0: u32, y0: u32) -> u32 {
        self.luma_var[self.cell_idx(x0, y0)]
    }

    /// Combined Cb+Cr activity of the 16x16 cell covering `(x0, y0)`.
    pub fn chroma_activity_at(&self, x0: u32, y0: u32) -> u16 {
        self.chroma_act[self.cell_idx(x0, y0)]
    }

    /// Fine-grid mean of `log2(1 + variance)` — the AQ reference point.
    pub fn frame_mean_log2var(&self) -> f32 {
        self.frame_mean_log2var
    }

    /// Fine-grid min/max `log2(1 + variance)` — the [`aq_qp_offset_positive`]
    /// rank span.
    pub fn frame_min_log2var(&self) -> f32 {
        self.frame_min_log2var
    }

    pub fn frame_max_log2var(&self) -> f32 {
        self.frame_max_log2var
    }

    /// Fine-grid mean chroma-aware activity — the [`aq_qp_offset_chroma`]
    /// reference point.
    pub fn frame_mean_log2activity(&self) -> f32 {
        self.frame_mean_activity
    }

    /// The auto-variance `qp_adj0 = (energy + 1)^0.1` of the 16x16 cell
    /// covering `(x0, y0)` — here the cell size matches x265's native 16x16
    /// `acEnergyCu` block exactly (the coarse grid approximates it at 32x32).
    pub fn autovar_qp_adj0_at(&self, x0: u32, y0: u32) -> f32 {
        let i = self.cell_idx(x0, y0);
        autovar_qp_adj0(
            self.luma_var[i],
            self.chroma_act[i],
            self.autovar_chroma_weight,
        )
    }

    /// Fine-grid mean of the auto-variance `qp_adj0` (x265's `avg_adj`).
    pub fn autovar_avg_adj(&self) -> f32 {
        self.autovar_avg_adj
    }

    /// Bias-corrected auto-variance reference point over the fine grid.
    pub fn autovar_avg_adj_final(&self) -> f32 {
        self.autovar_avg_adj_final
    }
}

fn chroma_dims(width: u32, height: u32, cat: u8) -> (u32, u32) {
    match cat {
        0 => (0, 0),
        1 => (width.div_ceil(2), height.div_ceil(2)),
        2 => (width.div_ceil(2), height),
        _ => (width, height),
    }
}

/// Chroma samples spanned by one 32x32 luma cell, per chroma format.
fn chroma_cell_size(cat: u8) -> (u32, u32) {
    match cat {
        0 => (0, 0),
        1 => (CELL / 2, CELL / 2),
        2 => (CELL / 2, CELL),
        _ => (CELL, CELL),
    }
}

/// Combined Cb/Cr activity (variance, 8-bit units) plus the per-cell Cb and Cr
/// means over a luma cell's chroma region. Returns `(activity, cb_mean,
/// cr_mean)`; the means feed the contextual global/center-surround chroma
/// contrast (Xu & Wu 2014's LAB colour-distance feature, here in YCbCr).
#[allow(clippy::too_many_arguments)]
fn chroma_cell_activity(
    src: Source<'_>,
    cat: u8,
    cw: u32,
    ch: u32,
    cx: u32,
    cy: u32,
    cell_cw: u32,
    cell_ch: u32,
    shift: u32,
) -> (u16, u16, u16) {
    if cat == 0 {
        return (0, 128, 128);
    }
    let stride = cw as usize;
    let x0 = cx * cell_cw;
    let y0 = cy * cell_ch;
    let x1 = (x0 + cell_cw).min(cw);
    let y1 = (y0 + cell_ch).min(ch);
    if x1 <= x0 || y1 <= y0 {
        return (0, 0, 0);
    }
    let n = ((x1 - x0) * (y1 - y0)) as i64;
    let _ = cat;
    // Returns (variance, mean).
    let stat_of = |plane: &[u16]| -> (i64, i64) {
        let mut sum = 0i64;
        let mut sum_sq = 0i64;
        for y in y0..y1 {
            for x in x0..x1 {
                let v = (plane[y as usize * stride + x as usize] >> shift) as i64;
                sum += v;
                sum_sq += v * v;
            }
        }
        let mean = sum / n;
        (((sum_sq / n) - mean * mean).max(0), mean)
    };
    let (v_cb, m_cb) = stat_of(src.cb);
    let (v_cr, m_cr) = stat_of(src.cr);
    let act = (v_cb + v_cr).min(u16::MAX as i64) as u16;
    (act, m_cb as u16, m_cr as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 128x64 (4:4:4): left half a flat constant, right half 1-px vertical
    /// stripes (hard high-frequency edges). The classifier must produce both
    /// Flat cells on the left and structured (non-Flat) cells on the right.
    #[test]
    fn classifies_flat_vs_edges() {
        let (w, h) = (128usize, 64usize);
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = if c < w / 2 {
                    128
                } else if c % 2 == 0 {
                    16
                } else {
                    235
                };
            }
        }
        let gray = vec![128u16; w * h];
        let src = Source {
            y: &y,
            cb: &gray,
            cr: &gray,
        };
        let maps = analyze(w as u32, h as u32, 8, 3, src);
        let counts = maps.class_counts();
        assert_eq!(
            counts.iter().sum::<u64>(),
            4 * 2,
            "8 cells over a 128x64 pic"
        );
        assert!(counts[RegionClass::Flat.index()] >= 1, "flat left half");
        let structured: u64 = counts.iter().sum::<u64>() - counts[RegionClass::Flat.index()];
        assert!(structured >= 1, "edges on the right half: {counts:?}");
    }

    /// x265 AUTO_VARIANCE / AUTO_VARIANCE_BIASED offset math on the same
    /// flat-left / striped-right synthetic picture: high-energy cells must get
    /// +QP and flat cells −QP, the biased mode must push flat cells further
    /// negative, offsets must be monotone in cell energy, and strength = 0 must
    /// be a uniform no-op.
    #[test]
    fn autovar_offsets_signs_and_bias() {
        let (w, h) = (128usize, 64usize);
        let mut y = vec![0u16; w * h];
        for r in 0..h {
            for c in 0..w {
                y[r * w + c] = if c < w / 2 {
                    128
                } else if c % 2 == 0 {
                    16
                } else {
                    235
                };
            }
        }
        let gray = vec![128u16; w * h];
        let src = Source {
            y: &y,
            cb: &gray,
            cr: &gray,
        };
        let maps = analyze(w as u32, h as u32, 8, 3, src);

        let (avg, avg_final) = (maps.autovar_avg_adj(), maps.autovar_avg_adj_final());
        assert!(avg >= 1.0, "qp_adj0 >= 1 per cell, so avg_adj >= 1: {avg}");

        // Flat cell at (0,0): zero luma variance, zero chroma activity.
        let flat = maps.autovar_qp_adj0_at(0, 0);
        // Striped cell on the right half: huge variance.
        let busy = maps.autovar_qp_adj0_at(96, 0);
        assert!(flat < busy, "flat {flat} vs striped {busy}");

        let clamp = 20.0;
        let off = |adj0: f32, biased: bool, s: f32| {
            aq_qp_offset_autovar(adj0, avg, avg_final, s, biased, clamp)
        };

        // AutoVariance: busy up, flat down.
        assert!(off(busy, false, 1.0) > 0, "high-energy cell must get +QP");
        assert!(off(flat, false, 1.0) < 0, "flat cell must get -QP");
        // Monotone in qp_adj0 (both modes).
        for biased in [false, true] {
            assert!(off(flat, biased, 1.0) <= off((flat + busy) / 2.0, biased, 1.0));
            assert!(off((flat + busy) / 2.0, biased, 1.0) <= off(busy, biased, 1.0));
        }
        // Biased mode pushes low-energy (qp_adj0^2 < 11) cells further negative.
        assert!(
            off(flat, true, 1.0) < off(flat, false, 1.0),
            "dark/flat bias must lower flat-cell QP further: biased {} vs {}",
            off(flat, true, 1.0),
            off(flat, false, 1.0)
        );
        // strength = 0 is a uniform no-op in both modes.
        for adj0 in [flat, busy] {
            assert_eq!(off(adj0, false, 0.0), 0);
            assert_eq!(off(adj0, true, 0.0), 0);
        }
        // Degenerate stats (empty map) resolve to 0 offset.
        assert_eq!(aq_qp_offset_autovar(1.0, 0.0, 0.0, 1.0, true, clamp), 0);
    }

    #[test]
    fn placebo_gets_inert_policy() {
        let y = vec![128u16; 64 * 64];
        let src = Source {
            y: &y,
            cb: &y,
            cr: &y,
        };
        let maps = analyze(64, 64, 8, 3, src);
        let p = maps.policy_at(0, 0, 5, Effort::Placebo);
        assert!(p.angular_prune.is_none() && p.allow_early_term);
        assert_eq!((p.luma_rd_bias, p.chroma_rd_bias), (0, 0));
        assert!(!p.force_leaf && p.rmd_prune_factor.is_none());
        // A flat cell under a steered tier forces angular pruning.
        // Importance-based pruning factors and force-leaf depend on the
        // canonical template's PreanalysisTemplate settings.
        let p = maps.policy_at(0, 0, 5, Effort::Fast);
        assert_eq!(p.angular_prune, Some(true));
    }
}
