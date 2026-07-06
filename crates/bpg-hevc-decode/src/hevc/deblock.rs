//! Deblocking filter (H.265 Section 8.7.2)
//!
//! Applies strong/weak filtering at CU and TU boundaries to reduce blocking artifacts.
//! For I-slices (HEIC still images), all boundaries have bS=2 since both sides are intra-coded.

use alloc::vec::Vec;

use super::picture::{DEBLOCK_FLAG_HORIZ, DEBLOCK_FLAG_VERT, DecodedFrame};

/// Beta prime values for deblocking filter (Table 8-12)
/// Index 0-51 maps QP to beta prime threshold
#[rustfmt::skip]
static BETA_PRIME: [u16; 52] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24,
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56,
    58, 60, 62, 64,
];

/// tC prime values for deblocking filter (Table 8-23)
/// Index 0-53 maps to tC prime threshold
#[rustfmt::skip]
static TC_PRIME: [u16; 54] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,
     3,  3,  3,  4,  4,  4,  5,  5,  6,  6,  7,  8,  9, 10, 11, 13,
    14, 16, 18, 20, 22, 24,
];

/// Chroma QP mapping table (Table 8-10) for indices 30-42
#[rustfmt::skip]
static CHROMA_QP_TABLE: [i32; 13] = [
    29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37,
];

/// Map intermediate chroma QP to actual chroma QP (H.265 §8.6.1).
///
/// Table 8-10 is used only for `ChromaArrayType == 1` (4:2:0). 4:2:2 and
/// 4:4:4 use `QpC = Min(qPi, 51)` directly; using the 4:2:0 table there makes
/// the chroma deblocking threshold non-conformant at qPi > 29.
fn chroma_qp_mapping(qp_i: i32, chroma_array_type: u8) -> i32 {
    if chroma_array_type != 1 {
        return qp_i.min(51);
    }
    if qp_i < 30 {
        qp_i
    } else if qp_i >= 43 {
        qp_i - 6
    } else {
        CHROMA_QP_TABLE[(qp_i - 30) as usize]
    }
}

/// Apply the deblocking filter to a decoded frame.
///
/// `beta_offset` and `tc_offset` come from slice header (slice_beta_offset_div2 * 2
/// and slice_tc_offset_div2 * 2).
/// `cb_qp_offset` and `cr_qp_offset` come from PPS (pps_cb_qp_offset / pps_cr_qp_offset).
pub fn apply_deblocking_filter(
    frame: &mut DecodedFrame,
    beta_offset: i32,
    tc_offset: i32,
    cb_qp_offset: i32,
    cr_qp_offset: i32,
) {
    apply_deblocking_filter_threads(frame, beta_offset, tc_offset, cb_qp_offset, cr_qp_offset, 1);
}

/// Luma rows per parallel deblocking band. A multiple of 8 so band
/// boundaries land on both the vertical (4-row segment) and the horizontal
/// (8-row edge spacing) grids, for luma and for every chroma subsampling.
const LUMA_BAND_ROWS: usize = 64;

/// [`apply_deblocking_filter`] with the edge work distributed over up to
/// `threads` scoped threads, byte-identical to the serial filter (and free of
/// `unsafe`: bands are disjoint `chunks_mut` slices).
///
/// Parallelism exploits the H.265 edge geometry:
/// - a vertical edge segment at `(x, y)` touches only rows `y..y+4`, so the
///   vertical pass splits each plane into [`LUMA_BAND_ROWS`]-row bands and
///   every segment stays inside one band;
/// - a horizontal edge at `y` touches rows `y-4..y+4` (3 written + 4 read on
///   each side), so the horizontal pass splits planes 4 luma rows below the
///   band grid (`split_at_mut` + `chunks_mut`); those boundary rows are
///   touched by no horizontal edge of any other band;
/// - within one pass, distinct edges are ≥8 samples apart and never overlap,
///   so per-band processing order does not matter;
/// - luma and chroma live on different planes, so their jobs share a phase;
///   the horizontal phase starts only after the vertical phase fully joins.
pub fn apply_deblocking_filter_threads(
    frame: &mut DecodedFrame,
    beta_offset: i32,
    tc_offset: i32,
    cb_qp_offset: i32,
    cr_qp_offset: i32,
    threads: usize,
) {
    let (sub_x, sub_y) = match frame.chroma_format {
        1 => (2u32, 2u32),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    };
    let y_stride = frame.y_stride();
    let c_stride = frame.c_stride();
    let chroma = frame.chroma_format > 0 && !frame.cb_plane.is_empty() && c_stride > 0;
    let ctx = DeblockCtx {
        width: frame.width,
        height: frame.height,
        bit_depth: frame.bit_depth as i32,
        chroma_format: frame.chroma_format,
        y_stride,
        c_stride,
        c_rows: if chroma {
            frame.cb_plane.len() / c_stride
        } else {
            0
        },
        deblock_flags: &frame.deblock_flags,
        qp_map: &frame.qp_map,
        deblock_stride: frame.deblock_stride,
        beta_offset,
        tc_offset,
        sub_x,
        sub_y,
    };

    let c_band_rows = LUMA_BAND_ROWS / sub_y as usize;

    // Phase 1: all vertical edges (luma + chroma), banded on the row grid.
    {
        let mut jobs: Vec<BandJob<'_>> = Vec::new();
        for (k, band) in frame
            .y_plane
            .chunks_mut(LUMA_BAND_ROWS * y_stride)
            .enumerate()
        {
            jobs.push(BandJob::LumaVert {
                band_y0: (k * LUMA_BAND_ROWS) as u32,
                band,
            });
        }
        if chroma {
            for (plane, qp_offset) in [
                (&mut frame.cb_plane, cb_qp_offset),
                (&mut frame.cr_plane, cr_qp_offset),
            ] {
                for (k, band) in plane.chunks_mut(c_band_rows * c_stride).enumerate() {
                    jobs.push(BandJob::ChromaVert {
                        qp_offset,
                        band_cy0: (k * c_band_rows) as u32,
                        band,
                    });
                }
            }
        }
        run_band_jobs(&ctx, threads, jobs);
    }

    // Phase 2: all horizontal edges, banded 4 luma rows below the grid so
    // each edge row's full touch range lives in exactly one band.
    {
        let head = (4usize).min(frame.y_plane.len() / y_stride.max(1)) * y_stride;
        let c_head = (4 / sub_y as usize).min(ctx.c_rows) * c_stride;
        let mut jobs: Vec<BandJob<'_>> = Vec::new();
        let (_, rest) = frame.y_plane.split_at_mut(head);
        for (k, band) in rest.chunks_mut(LUMA_BAND_ROWS * y_stride).enumerate() {
            jobs.push(BandJob::LumaHoriz {
                band_y0: (4 + k * LUMA_BAND_ROWS) as u32,
                band,
            });
        }
        if chroma {
            for (plane, qp_offset) in [
                (&mut frame.cb_plane, cb_qp_offset),
                (&mut frame.cr_plane, cr_qp_offset),
            ] {
                let (_, rest) = plane.split_at_mut(c_head);
                for (k, band) in rest.chunks_mut(c_band_rows * c_stride).enumerate() {
                    jobs.push(BandJob::ChromaHoriz {
                        qp_offset,
                        band_cy0: (4 / sub_y as usize + k * c_band_rows) as u32,
                        band,
                    });
                }
            }
        }
        run_band_jobs(&ctx, threads, jobs);
    }
}

/// Shared read-only state for one deblocking pass.
struct DeblockCtx<'a> {
    width: u32,
    height: u32,
    bit_depth: i32,
    chroma_format: u8,
    y_stride: usize,
    c_stride: usize,
    /// Full chroma plane height in rows (`plane.len() / c_stride`).
    c_rows: usize,
    deblock_flags: &'a [u8],
    qp_map: &'a [i8],
    deblock_stride: u32,
    beta_offset: i32,
    tc_offset: i32,
    sub_x: u32,
    sub_y: u32,
}

/// One plane band plus the edge set it owns for the current pass.
enum BandJob<'a> {
    LumaVert {
        band_y0: u32,
        band: &'a mut [u16],
    },
    LumaHoriz {
        band_y0: u32,
        band: &'a mut [u16],
    },
    ChromaVert {
        qp_offset: i32,
        band_cy0: u32,
        band: &'a mut [u16],
    },
    ChromaHoriz {
        qp_offset: i32,
        band_cy0: u32,
        band: &'a mut [u16],
    },
}

fn run_band_job(ctx: &DeblockCtx<'_>, job: BandJob<'_>) {
    match job {
        BandJob::LumaVert { band_y0, band } => deblock_luma_vert_band(ctx, band_y0, band),
        BandJob::LumaHoriz { band_y0, band } => deblock_luma_horiz_band(ctx, band_y0, band),
        BandJob::ChromaVert {
            qp_offset,
            band_cy0,
            band,
        } => deblock_chroma_vert_band(ctx, qp_offset, band_cy0, band),
        BandJob::ChromaHoriz {
            qp_offset,
            band_cy0,
            band,
        } => deblock_chroma_horiz_band(ctx, qp_offset, band_cy0, band),
    }
}

/// Run the band jobs on up to `threads` scoped workers (serial when 1).
/// Bands are disjoint plane slices, so job order is irrelevant.
fn run_band_jobs(ctx: &DeblockCtx<'_>, threads: usize, jobs: Vec<BandJob<'_>>) {
    let workers = threads.clamp(1, jobs.len().max(1));
    if workers <= 1 {
        for job in jobs {
            run_band_job(ctx, job);
        }
        return;
    }
    let mut worker_jobs: Vec<Vec<BandJob<'_>>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, job) in jobs.into_iter().enumerate() {
        worker_jobs[i % workers].push(job);
    }
    std::thread::scope(|scope| {
        for jobs in worker_jobs {
            scope.spawn(move || {
                for job in jobs {
                    run_band_job(ctx, job);
                }
            });
        }
    });
}

/// Vertical luma edges whose 4-row segments start inside this band.
fn deblock_luma_vert_band(ctx: &DeblockCtx<'_>, band_y0: u32, band: &mut [u16]) {
    let band_rows = (band.len() / ctx.y_stride) as u32;
    let y_end = (band_y0 + band_rows).min(ctx.height);
    let mut x = 8u32;
    while x < ctx.width {
        let mut y = band_y0;
        while y < y_end {
            let bx = x / 4;
            let by = y / 4;
            let idx = (by * ctx.deblock_stride + bx) as usize;
            if idx < ctx.deblock_flags.len() && (ctx.deblock_flags[idx] & DEBLOCK_FLAG_VERT) != 0 {
                // Get QP on both sides
                let qp_q = ctx.qp_map[idx] as i32;
                let qp_p = if bx > 0 {
                    ctx.qp_map[(by * ctx.deblock_stride + bx - 1) as usize] as i32
                } else {
                    qp_q
                };

                filter_edge_luma(ctx, band, band_y0, x, y, true, qp_p, qp_q);
            }
            y += 4;
        }
        x += 8;
    }
}

/// Horizontal luma edge rows whose touch range `y-4..y+4` lies inside this
/// band (`band_y0` is 4 rows below the band grid, so that is
/// `y ∈ {band_y0+4, band_y0+12, ...}` up to 4 rows before the band end).
fn deblock_luma_horiz_band(ctx: &DeblockCtx<'_>, band_y0: u32, band: &mut [u16]) {
    let band_rows = (band.len() / ctx.y_stride) as u32;
    let band_y1 = band_y0 + band_rows;
    let mut y = band_y0 + 4;
    while y + 4 <= band_y1 && y < ctx.height {
        let mut x = 0u32;
        while x < ctx.width {
            let bx = x / 4;
            let by = y / 4;
            let idx = (by * ctx.deblock_stride + bx) as usize;
            if idx < ctx.deblock_flags.len() && (ctx.deblock_flags[idx] & DEBLOCK_FLAG_HORIZ) != 0 {
                let qp_q = ctx.qp_map[idx] as i32;
                let qp_p = if by > 0 {
                    ctx.qp_map[((by - 1) * ctx.deblock_stride + bx) as usize] as i32
                } else {
                    qp_q
                };

                filter_edge_luma(ctx, band, band_y0, x, y, false, qp_p, qp_q);
            }
            x += 4;
        }
        y += 8;
    }
}

/// Filter a single luma edge (4 samples along the edge)
///
/// For vertical edges: x is the boundary position, filtering samples at x-1..x-4 and x..x+3
/// For horizontal edges: y is the boundary position, filtering samples at y-1..y-4 and y..y+3
///
/// Uses direct plane access with stride-based indexing to avoid per-sample
/// bounds checks. `plane` is the band of the luma plane starting at row
/// `band_y0`; coordinates stay plane-global, and the pass banding guarantees
/// every sample the edge touches lies inside the band.
#[allow(clippy::too_many_arguments)]
fn filter_edge_luma(
    ctx: &DeblockCtx<'_>,
    plane: &mut [u16],
    band_y0: u32,
    x: u32,
    y: u32,
    vertical: bool,
    qp_p: i32,
    qp_q: i32,
) {
    let bit_depth = ctx.bit_depth;
    let max_val = (1i32 << bit_depth) - 1;

    // For I-slices, bS is always 2 at boundaries
    let bs = 2i32;

    // Compute thresholds
    let qp_l = (qp_q + qp_p + 1) >> 1;
    let q_beta = (qp_l + ctx.beta_offset).clamp(0, 51);
    let beta = (BETA_PRIME[q_beta as usize] as i32) << (bit_depth - 8);
    let q_tc = (qp_l + 2 * (bs - 1) + ctx.tc_offset).clamp(0, 53);
    let tc = (TC_PRIME[q_tc as usize] as i32) << (bit_depth - 8);

    if tc == 0 {
        return;
    }

    let stride = ctx.y_stride;
    let y_local = (y - band_y0) as usize;

    // Compute stride-based addressing:
    // - step_along: stride between adjacent samples along the edge (k direction)
    // - step_across: stride between adjacent samples perpendicular to edge (p/q direction)
    // - base_q: offset of q[0][0] in plane
    let (step_along, step_across, base_q) = if vertical {
        // Vertical edge: k steps in y (stride), p/q steps in x (1)
        (stride, 1usize, y_local * stride + x as usize)
    } else {
        // Horizontal edge: k steps in x (1), p/q steps in y (stride)
        (1usize, stride, y_local * stride + x as usize)
    };
    // base_p = one step before the boundary on the p side
    let base_p = base_q - step_across;

    // Bounds check: ensure all 4 samples on both sides are in-bounds
    // q side extends 3 steps across from base_q, plus 3 steps along
    // p side extends 3 steps across back from base_p (= base_q - step_across)
    if base_p < 3 * step_across {
        return;
    }
    let last_q = base_q + 3 * step_along + 3 * step_across;
    if last_q >= plane.len() {
        return;
    }

    // Read samples for edge decision at k=0 and k=3
    let k3 = 3 * step_along;
    let p0_0 = plane[base_p] as i32;
    let p1_0 = plane[base_p - step_across] as i32;
    let p2_0 = plane[base_p - 2 * step_across] as i32;
    let p3_0 = plane[base_p - 3 * step_across] as i32;
    let q0_0 = plane[base_q] as i32;
    let q1_0 = plane[base_q + step_across] as i32;
    let q2_0 = plane[base_q + 2 * step_across] as i32;
    let q3_0 = plane[base_q + 3 * step_across] as i32;

    let p0_3 = plane[base_p + k3] as i32;
    let p1_3 = plane[base_p + k3 - step_across] as i32;
    let p2_3 = plane[base_p + k3 - 2 * step_across] as i32;
    let p3_3 = plane[base_p + k3 - 3 * step_across] as i32;
    let q0_3 = plane[base_q + k3] as i32;
    let q1_3 = plane[base_q + k3 + step_across] as i32;
    let q2_3 = plane[base_q + k3 + 2 * step_across] as i32;
    let q3_3 = plane[base_q + k3 + 3 * step_across] as i32;

    // Edge decision (H.265 8.7.2.5.3)
    let dp0 = (p2_0 - 2 * p1_0 + p0_0).abs();
    let dp3 = (p2_3 - 2 * p1_3 + p0_3).abs();
    let dq0 = (q2_0 - 2 * q1_0 + q0_0).abs();
    let dq3 = (q2_3 - 2 * q1_3 + q0_3).abs();

    let dpq0 = dp0 + dq0;
    let dpq3 = dp3 + dq3;
    let dp = dp0 + dp3;
    let dq = dq0 + dq3;
    let d = dpq0 + dpq3;

    if d >= beta {
        return;
    }

    // Determine filter strength
    let d_sam0 = 2 * dpq0 < (beta >> 2)
        && (p3_0 - p0_0).abs() + (q0_0 - q3_0).abs() < (beta >> 3)
        && (p0_0 - q0_0).abs() < ((5 * tc + 1) >> 1);

    let d_sam3 = 2 * dpq3 < (beta >> 2)
        && (p3_3 - p0_3).abs() + (q0_3 - q3_3).abs() < (beta >> 3)
        && (p0_3 - q0_3).abs() < ((5 * tc + 1) >> 1);

    let strong = d_sam0 && d_sam3;
    let d_ep = dp < ((beta + (beta >> 1)) >> 3);
    let d_eq = dq < ((beta + (beta >> 1)) >> 3);

    // Apply filter for all 4 samples along the edge
    for k in 0..4usize {
        let k_off = k * step_along;

        // Read p[0..3] and q[0..3] for this sample
        let p0 = plane[base_p + k_off] as i32;
        let p1 = plane[base_p + k_off - step_across] as i32;
        let p2 = plane[base_p + k_off - 2 * step_across] as i32;
        let q0 = plane[base_q + k_off] as i32;
        let q1 = plane[base_q + k_off + step_across] as i32;
        let q2 = plane[base_q + k_off + 2 * step_across] as i32;

        if strong {
            let p3 = plane[base_p + k_off - 3 * step_across] as i32;
            let q3 = plane[base_q + k_off + 3 * step_across] as i32;
            let tc2 = 2 * tc;

            // Strong filter (H.265 8.7.2.5.7)
            let p0_f = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3)
                .clamp(p0 - tc2, p0 + tc2)
                .clamp(0, max_val);
            let p1_f = ((p2 + p1 + p0 + q0 + 2) >> 2)
                .clamp(p1 - tc2, p1 + tc2)
                .clamp(0, max_val);
            let p2_f = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3)
                .clamp(p2 - tc2, p2 + tc2)
                .clamp(0, max_val);
            let q0_f = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3)
                .clamp(q0 - tc2, q0 + tc2)
                .clamp(0, max_val);
            let q1_f = ((p0 + q0 + q1 + q2 + 2) >> 2)
                .clamp(q1 - tc2, q1 + tc2)
                .clamp(0, max_val);
            let q2_f = ((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3)
                .clamp(q2 - tc2, q2 + tc2)
                .clamp(0, max_val);

            plane[base_p + k_off] = p0_f as u16;
            plane[base_p + k_off - step_across] = p1_f as u16;
            plane[base_p + k_off - 2 * step_across] = p2_f as u16;
            plane[base_q + k_off] = q0_f as u16;
            plane[base_q + k_off + step_across] = q1_f as u16;
            plane[base_q + k_off + 2 * step_across] = q2_f as u16;
        } else {
            // Weak filter
            let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;

            if delta.abs() < 10 * tc {
                let delta = delta.clamp(-tc, tc);

                plane[base_p + k_off] = (p0 + delta).clamp(0, max_val) as u16;
                plane[base_q + k_off] = (q0 - delta).clamp(0, max_val) as u16;

                if d_ep {
                    let delta_p =
                        ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    plane[base_p + k_off - step_across] = (p1 + delta_p).clamp(0, max_val) as u16;
                }
                if d_eq {
                    let delta_q =
                        ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-(tc >> 1), tc >> 1);
                    plane[base_q + k_off + step_across] = (q1 + delta_q).clamp(0, max_val) as u16;
                }
            }
        }
    }
}

/// Chroma tc threshold for one plane at a flagged edge (I-slice bS=2).
fn chroma_tc(ctx: &DeblockCtx<'_>, qp_p: i32, qp_q: i32, qp_offset: i32) -> i32 {
    let qp_i = ((qp_q + qp_p + 1) >> 1) + qp_offset;
    let qp_c = chroma_qp_mapping(qp_i, ctx.chroma_format);
    let q_tc = (qp_c + 2 + ctx.tc_offset).clamp(0, 53);
    (TC_PRIME[q_tc as usize] as i32) << (ctx.bit_depth - 8)
}

/// Vertical chroma edges of one plane whose 4-chroma-row segments start
/// inside this band. Chroma deblocking only modifies p0 and q0 (one sample
/// each side); for I-slices all flagged edges have bS=2.
///
/// Matching libde265: vertical chroma edges sit every `8*sub_x` luma
/// columns, with `4*sub_y`-luma-row segments.
fn deblock_chroma_vert_band(ctx: &DeblockCtx<'_>, qp_offset: i32, band_cy0: u32, band: &mut [u16]) {
    let max_val = (1i32 << ctx.bit_depth) - 1;
    let c_stride = ctx.c_stride;
    let c_height = ctx.height / ctx.sub_y;
    let band_rows = (band.len() / c_stride) as u32;
    let x_step = 8 * ctx.sub_x;
    let y_step = 4 * ctx.sub_y;

    let y0 = band_cy0 * ctx.sub_y;
    let y1 = ((band_cy0 + band_rows) * ctx.sub_y).min(ctx.height);
    let mut x = x_step;
    while x < ctx.width {
        let mut y = y0;
        while y < y1 {
            let bx = x / 4;
            let by = y / 4;
            let idx = (by * ctx.deblock_stride + bx) as usize;
            if idx < ctx.deblock_flags.len() && (ctx.deblock_flags[idx] & DEBLOCK_FLAG_VERT) != 0 {
                let qp_q = ctx.qp_map[idx] as i32;
                let qp_p = if bx > 0 {
                    ctx.qp_map[(by * ctx.deblock_stride + bx - 1) as usize] as i32
                } else {
                    qp_q
                };

                let cx = x / ctx.sub_x;
                let cy = y / ctx.sub_y;
                let tc = chroma_tc(ctx, qp_p, qp_q, qp_offset);
                if tc != 0 {
                    // Process 4 chroma samples along the edge
                    let num_samples = 4u32.min(c_height.saturating_sub(cy));
                    for k in 0..num_samples {
                        let row = (cy + k) as usize;
                        if cx < 2 || cx as usize >= c_stride || row >= ctx.c_rows {
                            continue;
                        }
                        let base = (row - band_cy0 as usize) * c_stride;
                        let ci = cx as usize;
                        if ci + 1 >= c_stride {
                            continue;
                        }
                        let p1 = band[base + ci - 2] as i32;
                        let p0 = band[base + ci - 1] as i32;
                        let q0 = band[base + ci] as i32;
                        let q1 = band[base + ci + 1] as i32;

                        let delta = (((q0 - p0) * 4 + p1 - q1 + 4) >> 3).clamp(-tc, tc);
                        band[base + ci - 1] = (p0 + delta).clamp(0, max_val) as u16;
                        band[base + ci] = (q0 - delta).clamp(0, max_val) as u16;
                    }
                }
            }
            y += y_step;
        }
        x += x_step;
    }
}

/// Horizontal chroma edge rows of one plane whose touch range `cy-2..cy+2`
/// lies inside this band (`band_cy0` is `4/sub_y` chroma rows below the band
/// grid). Horizontal chroma edges sit every `8*sub_y` luma rows.
fn deblock_chroma_horiz_band(
    ctx: &DeblockCtx<'_>,
    qp_offset: i32,
    band_cy0: u32,
    band: &mut [u16],
) {
    let max_val = (1i32 << ctx.bit_depth) - 1;
    let c_stride = ctx.c_stride;
    let c_width = ctx.width / ctx.sub_x;
    let band_rows = (band.len() / c_stride) as u32;
    let band_cy1 = band_cy0 + band_rows;
    let x_step = 4 * ctx.sub_x;
    let y_step = 8 * ctx.sub_y;

    let mut y = y_step;
    while y < ctx.height {
        let cy = y / ctx.sub_y;
        if cy >= band_cy0 + 2 && cy + 2 <= band_cy1 {
            let mut x = 0u32;
            while x < ctx.width {
                let bx = x / 4;
                let by = y / 4;
                let idx = (by * ctx.deblock_stride + bx) as usize;
                if idx < ctx.deblock_flags.len()
                    && (ctx.deblock_flags[idx] & DEBLOCK_FLAG_HORIZ) != 0
                {
                    let qp_q = ctx.qp_map[idx] as i32;
                    let qp_p = if by > 0 {
                        ctx.qp_map[((by - 1) * ctx.deblock_stride + bx) as usize] as i32
                    } else {
                        qp_q
                    };

                    let cx = x / ctx.sub_x;
                    let tc = chroma_tc(ctx, qp_p, qp_q, qp_offset);
                    if tc != 0 {
                        // Process 4 chroma samples along the edge
                        let num_samples = 4u32.min(c_width.saturating_sub(cx));
                        for k in 0..num_samples {
                            let col = (cx + k) as usize;
                            if cy < 2 || col >= c_stride {
                                continue;
                            }
                            let row_q = cy as usize;
                            let row_p = row_q - 1;
                            if row_q + 1 >= ctx.c_rows || row_p < 1 {
                                continue;
                            }

                            let b0 = band_cy0 as usize;
                            let p1 = band[(row_p - 1 - b0) * c_stride + col] as i32;
                            let p0 = band[(row_p - b0) * c_stride + col] as i32;
                            let q0 = band[(row_q - b0) * c_stride + col] as i32;
                            let q1 = band[(row_q + 1 - b0) * c_stride + col] as i32;

                            let delta = (((q0 - p0) * 4 + p1 - q1 + 4) >> 3).clamp(-tc, tc);
                            band[(row_p - b0) * c_stride + col] =
                                (p0 + delta).clamp(0, max_val) as u16;
                            band[(row_q - b0) * c_stride + col] =
                                (q0 - delta).clamp(0, max_val) as u16;
                        }
                    }
                }
                x += x_step;
            }
        }
        y += y_step;
    }
}
