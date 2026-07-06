//! Sample Adaptive Offset filter (H.265 Section 8.7.3)
//!
//! Applied after deblocking to reduce banding and ringing artifacts.
//! Two modes per CTB: Band Offset (BO) and Edge Offset (EO).

use alloc::vec;
use alloc::vec::Vec;

use super::picture::DecodedFrame;

/// SAO parameters for one CTB
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaoInfo {
    /// SAO type per component: 0=off, 1=band offset, 2=edge offset
    /// [0]=Y, [1]=Cb, [2]=Cr
    pub sao_type_idx: [u8; 3],
    /// Edge offset class per component (0-3, only used when type==2)
    /// 0=horizontal, 1=vertical, 2=diagonal 135°, 3=diagonal 45°
    pub sao_eo_class: [u8; 3],
    /// Band position per component (0-31, only used when type==1)
    pub sao_band_position: [u8; 3],
    /// Signed offset values per component, 4 values each
    /// For band offset: offsets for 4 consecutive bands starting at band_position
    /// For edge offset: offsets[0]=cat1(+), [1]=cat2(+), [2]=cat3(-), [3]=cat4(-)
    pub sao_offset_val: [[i8; 4]; 3],
}

/// SAO map for the entire frame, stored at CTB granularity
pub struct SaoMap {
    pub data: Vec<SaoInfo>,
    pub width_ctbs: u32,
    pub height_ctbs: u32,
}

impl SaoMap {
    pub fn new(width_ctbs: u32, height_ctbs: u32) -> Self {
        Self {
            data: vec![SaoInfo::default(); (width_ctbs * height_ctbs) as usize],
            width_ctbs,
            height_ctbs,
        }
    }

    #[inline]
    pub fn get(&self, ctb_x: u32, ctb_y: u32) -> &SaoInfo {
        &self.data[(ctb_y * self.width_ctbs + ctb_x) as usize]
    }

    #[inline]
    pub fn get_mut(&mut self, ctb_x: u32, ctb_y: u32) -> &mut SaoInfo {
        &mut self.data[(ctb_y * self.width_ctbs + ctb_x) as usize]
    }
}

/// Edge offset direction lookup: (dx0, dy0, dx1, dy1) for each eo_class
/// eo_class 0: horizontal (left, right)
/// eo_class 1: vertical (above, below)
/// eo_class 2: diagonal 135° (upper-left, lower-right)
/// eo_class 3: diagonal 45° (upper-right, lower-left)
const EO_OFFSETS: [(i32, i32, i32, i32); 4] = [
    (-1, 0, 1, 0),  // class 0: horizontal
    (0, -1, 0, 1),  // class 1: vertical
    (-1, -1, 1, 1), // class 2: 135° diagonal
    (1, -1, -1, 1), // class 3: 45° diagonal
];

/// Apply SAO filter to the entire frame
pub fn apply_sao(frame: &mut DecodedFrame, sao_map: &SaoMap, ctb_size: u32) {
    apply_sao_threads(frame, sao_map, ctb_size, 1);
}

/// [`apply_sao`] with CTB rows distributed over up to `threads` scoped
/// threads. Byte-identical to the serial pass: every CTB writes only its own
/// rectangle (rows of CTBs own disjoint plane bands), band offsets are
/// per-pixel in-place, and edge offsets read from the immutable pre-SAO
/// plane clones.
pub fn apply_sao_threads(frame: &mut DecodedFrame, sao_map: &SaoMap, ctb_size: u32, threads: usize) {
    let width = frame.width;
    let height = frame.height;
    let bit_depth = frame.bit_depth;

    // Only clone planes that have edge offset (type 2), since edge offset
    // reads neighbors that may be modified. Band offset (type 1) is in-place.
    let mut need_y_clone = false;
    let mut need_cb_clone = false;
    let mut need_cr_clone = false;
    for sao in &sao_map.data {
        if sao.sao_type_idx[0] == 2 {
            need_y_clone = true;
        }
        if sao.sao_type_idx[1] == 2 {
            need_cb_clone = true;
        }
        if sao.sao_type_idx[2] == 2 {
            need_cr_clone = true;
        }
        if need_y_clone && need_cb_clone && need_cr_clone {
            break;
        }
    }

    let orig_y = if need_y_clone {
        frame.y_plane.clone()
    } else {
        Vec::new()
    };
    let orig_cb = if need_cb_clone {
        frame.cb_plane.clone()
    } else {
        Vec::new()
    };
    let orig_cr = if need_cr_clone {
        frame.cr_plane.clone()
    } else {
        Vec::new()
    };

    let y_stride = frame.y_stride();
    let c_stride = frame.c_stride();

    let (sub_x, sub_y) = match frame.chroma_format {
        1 => (2u32, 2u32),
        2 => (2, 1),
        3 => (1, 1),
        _ => (1, 1),
    };

    let ctx = SaoApplyCtx {
        sao_map,
        ctb_size,
        width,
        height,
        bit_depth,
        y_stride: y_stride as u32,
        c_stride: c_stride as u32,
        sub_x,
        sub_y,
        chroma: frame.chroma_format > 0,
        orig_y: &orig_y,
        orig_cb: &orig_cb,
        orig_cr: &orig_cr,
    };

    // Split the planes into per-CTB-row bands (disjoint `&mut`); each row job
    // owns its band of all three planes.
    let luma_chunk = (ctb_size as usize * y_stride).max(1);
    let chroma_chunk = ((ctb_size / sub_y) as usize * c_stride).max(1);
    let mut y_bands = frame.y_plane.chunks_mut(luma_chunk);
    let mut cb_bands = frame.cb_plane.chunks_mut(chroma_chunk);
    let mut cr_bands = frame.cr_plane.chunks_mut(chroma_chunk);
    let mut jobs: Vec<(u32, &mut [u16], Option<(&mut [u16], &mut [u16])>)> =
        Vec::with_capacity(sao_map.height_ctbs as usize);
    for ctb_y in 0..sao_map.height_ctbs {
        let Some(y_band) = y_bands.next() else { break };
        let c = match (cb_bands.next(), cr_bands.next()) {
            (Some(cb), Some(cr)) => Some((cb, cr)),
            _ => None,
        };
        jobs.push((ctb_y, y_band, c));
    }

    let workers = threads.clamp(1, jobs.len().max(1));
    if workers <= 1 {
        for (ctb_y, y_band, c) in jobs {
            apply_sao_ctb_row(&ctx, ctb_y, y_band, c);
        }
        return;
    }
    let mut worker_jobs: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
    for (i, job) in jobs.into_iter().enumerate() {
        worker_jobs[i % workers].push(job);
    }
    std::thread::scope(|scope| {
        for jobs in worker_jobs {
            let ctx = &ctx;
            scope.spawn(move || {
                for (ctb_y, y_band, c) in jobs {
                    apply_sao_ctb_row(ctx, ctb_y, y_band, c);
                }
            });
        }
    });
}

/// Shared read-only state for one SAO application pass.
struct SaoApplyCtx<'a> {
    sao_map: &'a SaoMap,
    ctb_size: u32,
    width: u32,
    height: u32,
    bit_depth: u8,
    y_stride: u32,
    c_stride: u32,
    sub_x: u32,
    sub_y: u32,
    chroma: bool,
    orig_y: &'a [u16],
    orig_cb: &'a [u16],
    orig_cr: &'a [u16],
}

/// Apply one CTB row's SAO parameters into that row's plane bands.
/// Coordinates stay global; only the destination indexing is band-relative
/// (`band_y0` rows above the band).
fn apply_sao_ctb_row(
    ctx: &SaoApplyCtx<'_>,
    ctb_y: u32,
    y_band: &mut [u16],
    mut c_bands: Option<(&mut [u16], &mut [u16])>,
) {
    let ctb_size = ctx.ctb_size;
    let bit_depth = ctx.bit_depth;
    let band_y0 = ctb_y * ctb_size;
    let c_band_y0 = band_y0 / ctx.sub_y;
    for ctb_x in 0..ctx.sao_map.width_ctbs {
        let sao = ctx.sao_map.get(ctb_x, ctb_y);
        let ctb_x_px = ctb_x * ctb_size;
        let ctb_y_px = ctb_y * ctb_size;

        // Luma
        match sao.sao_type_idx[0] {
            1 => {
                let x_end = (ctb_x_px + ctb_size).min(ctx.width);
                let y_end = (ctb_y_px + ctb_size).min(ctx.height);
                apply_sao_band_inplace(
                    y_band,
                    band_y0,
                    ctx.y_stride,
                    ctb_x_px,
                    ctb_y_px,
                    x_end,
                    y_end,
                    sao.sao_band_position[0],
                    &sao.sao_offset_val[0],
                    bit_depth,
                );
            }
            2 => {
                let x_end = (ctb_x_px + ctb_size).min(ctx.width);
                let y_end = (ctb_y_px + ctb_size).min(ctx.height);
                apply_sao_edge(
                    ctx.orig_y,
                    y_band,
                    band_y0,
                    ctx.y_stride,
                    ctx.width,
                    ctx.height,
                    ctb_x_px,
                    ctb_y_px,
                    x_end,
                    y_end,
                    sao.sao_eo_class[0],
                    &sao.sao_offset_val[0],
                    bit_depth,
                );
            }
            _ => {}
        }

        // Chroma (4:2:0: halved coordinates)
        if ctx.chroma {
            let Some((cb_band, cr_band)) = c_bands.as_mut() else {
                continue;
            };
            let cx_start = ctb_x_px / ctx.sub_x;
            let cy_start = ctb_y_px / ctx.sub_y;
            let cx_end = ((ctb_x_px + ctb_size) / ctx.sub_x).min(ctx.width / ctx.sub_x);
            let cy_end = ((ctb_y_px + ctb_size) / ctx.sub_y).min(ctx.height / ctx.sub_y);
            let c_w = ctx.width / ctx.sub_x;
            let c_h = ctx.height / ctx.sub_y;

            // Cb
            match sao.sao_type_idx[1] {
                1 => {
                    apply_sao_band_inplace(
                        cb_band,
                        c_band_y0,
                        ctx.c_stride,
                        cx_start,
                        cy_start,
                        cx_end,
                        cy_end,
                        sao.sao_band_position[1],
                        &sao.sao_offset_val[1],
                        bit_depth,
                    );
                }
                2 => {
                    apply_sao_edge(
                        ctx.orig_cb,
                        cb_band,
                        c_band_y0,
                        ctx.c_stride,
                        c_w,
                        c_h,
                        cx_start,
                        cy_start,
                        cx_end,
                        cy_end,
                        sao.sao_eo_class[1],
                        &sao.sao_offset_val[1],
                        bit_depth,
                    );
                }
                _ => {}
            }

            // Cr
            match sao.sao_type_idx[2] {
                1 => {
                    apply_sao_band_inplace(
                        cr_band,
                        c_band_y0,
                        ctx.c_stride,
                        cx_start,
                        cy_start,
                        cx_end,
                        cy_end,
                        sao.sao_band_position[2],
                        &sao.sao_offset_val[2],
                        bit_depth,
                    );
                }
                2 => {
                    apply_sao_edge(
                        ctx.orig_cr,
                        cr_band,
                        c_band_y0,
                        ctx.c_stride,
                        c_w,
                        c_h,
                        cx_start,
                        cy_start,
                        cx_end,
                        cy_end,
                        sao.sao_eo_class[2],
                        &sao.sao_offset_val[2],
                        bit_depth,
                    );
                }
                _ => {}
            }
        }
    }
}

/// Apply SAO edge offset to a single pixel with bounds checking. `dst` may be
/// a band of the full plane, `band_off` samples below the plane origin.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_sao_edge_pixel(
    src: &[u16],
    dst: &mut [u16],
    band_off: usize,
    row: usize,
    x: u32,
    dx0: i32,
    dy0: i32,
    dx1: i32,
    dy1: i32,
    stride: u32,
    plane_w: u32,
    plane_h: u32,
    max_val: i32,
    offset_table: &[i32; 5],
) {
    let nx0 = x as i32 + dx0;
    let ny0 = (row / stride as usize) as i32 + dy0;
    let nx1 = x as i32 + dx1;
    let ny1 = (row / stride as usize) as i32 + dy1;

    if nx0 < 0
        || nx0 >= plane_w as i32
        || ny0 < 0
        || ny0 >= plane_h as i32
        || nx1 < 0
        || nx1 >= plane_w as i32
        || ny1 < 0
        || ny1 >= plane_h as i32
    {
        return;
    }

    let idx = row + x as usize;
    let sample = src[idx] as i32;
    let n0 = src[(ny0 as u32 * stride + nx0 as u32) as usize] as i32;
    let n1 = src[(ny1 as u32 * stride + nx1 as u32) as usize] as i32;

    let sign0 = (sample - n0).signum();
    let sign1 = (sample - n1).signum();
    let edge_idx = (2 + sign0 + sign1) as usize;

    let offset = offset_table[edge_idx];
    if offset != 0 {
        dst[idx - band_off] = (sample + offset).clamp(0, max_val) as u16;
    }
}

/// Apply SAO band offset in-place (type 1). Reads and writes same buffer.
/// `plane` may be a band of the full plane starting at row `band_y0`
/// (`y_start >= band_y0`); coordinates stay plane-global.
#[allow(clippy::too_many_arguments)]
fn apply_sao_band_inplace(
    plane: &mut [u16],
    band_y0: u32,
    stride: u32,
    x_start: u32,
    y_start: u32,
    x_end: u32,
    y_end: u32,
    band_position: u8,
    offsets: &[i8; 4],
    bit_depth: u8,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let band_shift = bit_depth - 5;

    // Build lookup table for the 32 bands
    let mut band_table = [0i8; 32];
    for k in 0..4u8 {
        let band_idx = (band_position + k) & 31;
        band_table[band_idx as usize] = offsets[k as usize];
    }

    for y in y_start..y_end {
        let row = ((y - band_y0) * stride) as usize;
        for x in x_start..x_end {
            let idx = row + x as usize;
            let sample = (plane[idx] as i32).min(max_val);
            let band = (sample >> band_shift) as usize;
            let offset = band_table[band] as i32;
            if offset != 0 {
                plane[idx] = (sample + offset).clamp(0, max_val) as u16;
            }
        }
    }
}

/// Apply SAO edge offset (type 2). Reads from pre-cloned src (full plane,
/// global coordinates), writes to `dst`, which may be a band of the full
/// plane starting at row `band_y0`.
#[allow(clippy::too_many_arguments)]
fn apply_sao_edge(
    src: &[u16],
    dst: &mut [u16],
    band_y0: u32,
    stride: u32,
    plane_w: u32,
    plane_h: u32,
    x_start: u32,
    y_start: u32,
    x_end: u32,
    y_end: u32,
    eo_class: u8,
    offsets: &[i8; 4],
    bit_depth: u8,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let (dx0, dy0, dx1, dy1) = EO_OFFSETS[eo_class as usize & 3];

    let offset_table: [i32; 5] = [
        offsets[0] as i32,
        offsets[1] as i32,
        0,
        -(offsets[2] as i32),
        -(offsets[3] as i32),
    ];

    // Compute safe interior bounds where neighbor access never goes out of frame.
    let safe_x_start = x_start.max((-dx0).max(-dx1).max(0) as u32);
    let safe_x_end = x_end.min(plane_w - dx0.max(dx1).max(0) as u32);
    let safe_y_start = y_start.max((-dy0).max(-dy1).max(0) as u32);
    let safe_y_end = y_end.min(plane_h - dy0.max(dy1).max(0) as u32);

    let stride_u = stride as usize;
    let band_off = band_y0 as usize * stride_u;
    let dx0_u = dx0 as isize;
    let dy0_s = dy0 as isize * stride_u as isize;
    let dx1_u = dx1 as isize;
    let dy1_s = dy1 as isize * stride_u as isize;

    // Interior: no bounds checks needed
    for y in safe_y_start..safe_y_end {
        let row = y as usize * stride_u;
        for x in safe_x_start..safe_x_end {
            let idx = row + x as usize;
            let sample = src[idx] as i32;
            let n0_idx = (idx as isize + dy0_s + dx0_u) as usize;
            let n1_idx = (idx as isize + dy1_s + dx1_u) as usize;
            let n0 = src[n0_idx] as i32;
            let n1 = src[n1_idx] as i32;

            let sign0 = (sample - n0).signum();
            let sign1 = (sample - n1).signum();
            let edge_idx = (2 + sign0 + sign1) as usize;

            let offset = offset_table[edge_idx];
            if offset != 0 {
                dst[idx - band_off] = (sample + offset).clamp(0, max_val) as u16;
            }
        }
    }

    // Border rows/columns: with bounds checks
    for y in y_start..y_end {
        if y >= safe_y_start && y < safe_y_end {
            let row = y as usize * stride_u;
            for x in x_start..safe_x_start.min(x_end) {
                apply_sao_edge_pixel(
                    src,
                    dst,
                    band_off,
                    row,
                    x,
                    dx0,
                    dy0,
                    dx1,
                    dy1,
                    stride,
                    plane_w,
                    plane_h,
                    max_val,
                    &offset_table,
                );
            }
            for x in safe_x_end.max(x_start)..x_end {
                apply_sao_edge_pixel(
                    src,
                    dst,
                    band_off,
                    row,
                    x,
                    dx0,
                    dy0,
                    dx1,
                    dy1,
                    stride,
                    plane_w,
                    plane_h,
                    max_val,
                    &offset_table,
                );
            }
        } else {
            let row = y as usize * stride_u;
            for x in x_start..x_end {
                apply_sao_edge_pixel(
                    src,
                    dst,
                    band_off,
                    row,
                    x,
                    dx0,
                    dy0,
                    dx1,
                    dy1,
                    stride,
                    plane_w,
                    plane_h,
                    max_val,
                    &offset_table,
                );
            }
        }
    }
}
