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
//! adaptive QP. Steering applies to the source-steered non-reference tiers
//! (`Fast`/`Balanced`/`Good`); `Fastest` uses the map only for AQ, and
//! `Best`/`Placebo`/`Reference` keep inert search steering and uniform QP.

use crate::encoder::Source;
use crate::Effort;

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
    /// comparison. Always `false` on the byte-exact reference tiers.
    pub force_leaf: bool,
    /// Prefer building split structure before the leaf path on structured
    /// regions. This is a search-order hint unless upgraded to
    /// [`Self::force_split`] by the effort resolver.
    pub prefer_split: bool,
    /// Force this CU/TU to split on high-importance structured regions. `false`
    /// defers to the normal split RD comparison. Always `false` on the
    /// byte-exact reference tiers.
    pub force_split: bool,
    /// RMD cost-curve pruning factor for luma mode candidates. `Some(f)` drops
    /// rough-mode candidates whose cost exceeds `f ×` the best (shi2013 knee /
    /// xiong2016 average-cost); `None` keeps the full effort-derived candidate
    /// set (the reference tiers).
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
/// Always run for the non-reference tiers (the reference tiers use
/// [`AnalysisMaps::empty`] instead). A cheap second cell-grid-level pass fills
/// the contextual features (`global_contrast_q8`, `center_surround_q8`,
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

    let frame_mean_log2var = if cells.is_empty() {
        0.0
    } else {
        let sum: f32 = cells
            .iter()
            .map(|c| ((1 + c.variance) as f32).log2())
            .sum();
        sum / cells.len() as f32
    };

    AnalysisMaps {
        cells_x,
        cells_y,
        cells,
        frame_mean_log2var,
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

/// Maximum magnitude (in QP steps) of the bidirectional variance AQ offset.
const VAR_AQ_CLAMP: f32 = 8.0;

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
/// result is clamped to ±[`VAR_AQ_CLAMP`] QP steps. Caller adds this to the
/// slice QP and clamps to the valid `[0, 51]` range.
pub fn aq_qp_offset_variance(
    variance: u32,
    frame_mean_log2var: f32,
    strength: f32,
    perceptual: bool,
) -> i32 {
    let log2var = ((1 + variance) as f32).log2();
    let mut adj = strength * (log2var - frame_mean_log2var);
    if !perceptual {
        adj = -adj;
    }
    adj.round().clamp(-VAR_AQ_CLAMP, VAR_AQ_CLAMP) as i32
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
    /// An empty map, for tiers that need neither AQ nor search steering
    /// (`Best`/`Placebo`/`Reference`) — they early-return [`INERT`] from
    /// [`Self::policy_at`] without indexing any cell, so the picture-wide
    /// feature pass is skipped entirely.
    pub fn empty() -> AnalysisMaps {
        AnalysisMaps {
            cells_x: 0,
            cells_y: 0,
            cells: Vec::new(),
            frame_mean_log2var: 0.0,
        }
    }

    /// Resolve the steering policy for a coding block at `(x0, y0)` of size
    /// `1 << log2_size`, given the encode effort. The byte-exact reference tiers
    /// ([`crate::is_reference_tier`]) always get [`INERT`] so their output is
    /// unchanged; every other tier is steered.
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

    /// Picture mean of `log2(1 + variance)` — the AQ reference point.
    pub fn frame_mean_log2var(&self) -> f32 {
        self.frame_mean_log2var
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

    #[test]
    fn reference_tiers_get_inert_policy() {
        let y = vec![128u16; 64 * 64];
        let src = Source {
            y: &y,
            cb: &y,
            cr: &y,
        };
        let maps = analyze(64, 64, 8, 3, src);
        // The byte-exact reference tiers are never steered.
        for e in [Effort::Placebo, Effort::Reference] {
            let p = maps.policy_at(0, 0, 5, e);
            assert!(p.angular_prune.is_none() && p.allow_early_term);
            assert_eq!((p.luma_rd_bias, p.chroma_rd_bias), (0, 0));
            assert!(!p.force_leaf && p.rmd_prune_factor.is_none());
        }
        // A flat cell under a steered tier forces angular pruning, and the
        // importance overlay arms rough-mode pruning + forced-leaf coding.
        let p = maps.policy_at(0, 0, 5, Effort::Balanced);
        assert_eq!(p.angular_prune, Some(true));
        assert_eq!(p.rmd_prune_factor, Some(1.10));
        assert!(
            p.force_leaf,
            "a flat low-importance cell should force a leaf"
        );
    }
}
