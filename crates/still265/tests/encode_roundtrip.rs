//! Phase 5 milestone 1/2 integration test: the Rust still encoder must emit an
//! Annex-B IDR access unit that `bpg-hevc-decode` decodes to **exactly** the
//! samples the encoder reconstructed, with a sane PSNR against the source.

use bpg_hevc_decode::hevc::decode;
use still265::encoder::{encode, Source};
use still265::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};

/// Build a smooth 4:2:0 YCbCr source (gradients predict well under Planar),
/// scaled to the given bit depth.
fn make_source(w: usize, h: usize, bd: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1i32 << bd) - 1;
    let s = if bd == 8 { 1 } else { 4 }; // scale 8-bit-ish ranges up to 10-bit
    let mut y = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((16 + (i + j) % 200) as i32 * s).clamp(0, max) as u16;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cb = vec![0u16; cw * ch];
    let mut cr = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            cb[j * cw + i] = ((128 + (i as i32 - cw as i32 / 2) / 2) * s).clamp(0, max) as u16;
            cr[j * cw + i] = ((128 + (j as i32 - ch as i32 / 2) / 2) * s).clamp(0, max) as u16;
        }
    }
    (y, cb, cr)
}

/// Build a smooth 4:4:4 YCbCr source (full-resolution Cb/Cr), scaled to the
/// given bit depth.
fn make_source_444(w: usize, h: usize, bd: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1i32 << bd) - 1;
    let s = if bd == 8 { 1 } else { 4 };
    let mut y = vec![0u16; w * h];
    let mut cb = vec![0u16; w * h];
    let mut cr = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((16 + (i + j) % 200) as i32 * s).clamp(0, max) as u16;
            cb[j * w + i] = ((128 + (i as i32 - w as i32 / 2) / 2) * s).clamp(0, max) as u16;
            cr[j * w + i] = ((128 + (j as i32 - h as i32 / 2) / 2) * s).clamp(0, max) as u16;
        }
    }
    (y, cb, cr)
}

/// Build a higher-frequency 4:4:4 source to exercise transform split RD on
/// component geometry where luma/chroma TUs mirror each other.
fn make_textured_source_444(w: usize, h: usize, bd: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1i32 << bd) - 1;
    let s = if bd == 8 { 1 } else { 4 };
    let mut y = vec![0u16; w * h];
    let mut cb = vec![0u16; w * h];
    let mut cr = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            let checker = if ((i / 8) ^ (j / 8)) & 1 == 0 {
                42
            } else {
                186
            };
            y[j * w + i] = ((checker + ((i * 3 + j * 5) % 17) as i32) * s).clamp(0, max) as u16;
            cb[j * w + i] =
                ((96 + ((i / 4) as i32 % 48) - ((j / 8) as i32 % 16)) * s).clamp(0, max) as u16;
            cr[j * w + i] =
                ((160 - ((i / 8) as i32 % 32) + ((j / 4) as i32 % 24)) * s).clamp(0, max) as u16;
        }
    }
    (y, cb, cr)
}

/// Build a higher-frequency 4:2:0 source (half-resolution Cb/Cr) to exercise
/// transform split RD on luma TUs whose chroma siblings stay unsplit.
fn make_textured_source_420(w: usize, h: usize, bd: u8) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let max = (1i32 << bd) - 1;
    let s = if bd == 8 { 1 } else { 4 };
    let mut y = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            let checker = if ((i / 8) ^ (j / 8)) & 1 == 0 {
                42
            } else {
                186
            };
            y[j * w + i] = ((checker + ((i * 3 + j * 5) % 17) as i32) * s).clamp(0, max) as u16;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cb = vec![0u16; cw * ch];
    let mut cr = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            cb[j * cw + i] =
                ((96 + ((i / 4) as i32 % 48) - ((j / 8) as i32 % 16)) * s).clamp(0, max) as u16;
            cr[j * cw + i] =
                ((160 - ((i / 8) as i32 % 32) + ((j / 4) as i32 % 24)) * s).clamp(0, max) as u16;
        }
    }
    (y, cb, cr)
}

fn psnr(a: &[u16], b: &[u16], peak: f64) -> f64 {
    let mut sse = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        sse += d * d;
    }
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / a.len() as f64;
    10.0 * (peak * peak / mse).log10()
}

fn crop_plane(src: &[u16], stride: usize, width: usize, height: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let row = y * stride;
        out.extend_from_slice(&src[row..row + width]);
    }
    out
}

fn assert_plane_eq(actual: &[u16], expected: &[u16], stride: usize, label: &str) {
    if actual == expected {
        return;
    }
    let idx = actual
        .iter()
        .zip(expected.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    let x = idx % stride;
    let y = idx / stride;
    panic!(
        "{label} mismatch at ({x},{y}) idx={idx}: decoded={} recon={}",
        actual.get(idx).copied().unwrap_or_default(),
        expected.get(idx).copied().unwrap_or_default()
    );
}

fn cfg(w: u32, h: u32, qp: u8, bd: u8) -> StillHevcConfig {
    cfg_chroma(w, h, qp, bd, ChromaFormat::Yuv420)
}

fn cfg_chroma(w: u32, h: u32, qp: u8, bd: u8, chroma: ChromaFormat) -> StillHevcConfig {
    StillHevcConfig {
        width: w,
        height: h,
        bit_depth: bd,
        chroma,
        qp,
        effort: Effort::Balanced,
        sao: SaoMode::Off,
        deblock: DeblockMode::Off,
    }
}

fn run_bd(w: u32, h: u32, qp: u8, bd: u8) {
    let (y, cb, cr) = make_source(w as usize, h as usize, bd);
    let config = cfg(w, h, qp, bd);
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );

    let decoded = decode(&bytes).expect("decoder must accept the encoded stream");

    // Exactness: decoded samples == encoder reconstruction.
    assert_plane_eq(
        &decoded.y_plane,
        &recon.y_plane,
        decoded.width as usize,
        &format!("luma recon (qp={qp} bd={bd})"),
    );
    assert_plane_eq(
        &decoded.cb_plane,
        &recon.cb_plane,
        decoded.width.div_ceil(2) as usize,
        &format!("cb recon (qp={qp} bd={bd})"),
    );
    assert_plane_eq(
        &decoded.cr_plane,
        &recon.cr_plane,
        decoded.width.div_ceil(2) as usize,
        &format!("cr recon (qp={qp} bd={bd})"),
    );

    // Quality: PSNR vs source should be reasonable for a smooth image.
    let peak = ((1u32 << bd) - 1) as f64;
    let display_y = crop_plane(
        &decoded.y_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
    );
    let py = psnr(&y, &display_y, peak);
    assert!(py > 30.0, "luma PSNR too low: {py:.2} dB (qp={qp} bd={bd})");
    eprintln!(
        "bd={bd} qp={qp} {}x{}  luma PSNR = {py:.2} dB  ({} bytes)",
        w,
        h,
        bytes.len()
    );
}

fn run_444(w: u32, h: u32, qp: u8, bd: u8) {
    let (y, cb, cr) = make_source_444(w as usize, h as usize, bd);
    let config = cfg_chroma(w, h, qp, bd, ChromaFormat::Yuv444);
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );

    let decoded = decode(&bytes).expect("decoder must accept the encoded stream");

    assert_eq!(
        decoded.y_plane, recon.y_plane,
        "luma recon mismatch (444, qp={qp} bd={bd})"
    );
    assert_eq!(
        decoded.cb_plane, recon.cb_plane,
        "cb recon mismatch (444, qp={qp} bd={bd})"
    );
    assert_eq!(
        decoded.cr_plane, recon.cr_plane,
        "cr recon mismatch (444, qp={qp} bd={bd})"
    );

    let peak = ((1u32 << bd) - 1) as f64;
    let display_y = crop_plane(
        &decoded.y_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
    );
    let display_cb = crop_plane(
        &decoded.cb_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
    );
    let py = psnr(&y, &display_y, peak);
    let pcb = psnr(&cb, &display_cb, peak);
    assert!(
        py > 30.0,
        "luma PSNR too low: {py:.2} dB (444, qp={qp} bd={bd})"
    );
    assert!(
        pcb > 30.0,
        "chroma PSNR too low: {pcb:.2} dB (444, qp={qp} bd={bd})"
    );
    eprintln!(
        "444 bd={bd} qp={qp} {}x{}  luma PSNR = {py:.2} dB, cb PSNR = {pcb:.2} dB ({} bytes)",
        w,
        h,
        bytes.len()
    );
}

fn run_textured_444(w: u32, h: u32, qp: u8, bd: u8) {
    let (y, cb, cr) = make_textured_source_444(w as usize, h as usize, bd);
    let config = cfg_chroma(w, h, qp, bd, ChromaFormat::Yuv444);
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );

    let decoded = decode(&bytes).expect("decoder must accept textured 444 stream");
    assert_eq!(decoded.y_plane, recon.y_plane, "textured 444 luma mismatch");
    assert_eq!(decoded.cb_plane, recon.cb_plane, "textured 444 cb mismatch");
    assert_eq!(decoded.cr_plane, recon.cr_plane, "textured 444 cr mismatch");

    let peak = ((1u32 << bd) - 1) as f64;
    let display_y = crop_plane(
        &decoded.y_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
    );
    let py = psnr(&y, &display_y, peak);
    assert!(
        py > 18.0,
        "textured 444 luma PSNR too low: {py:.2} dB (qp={qp} bd={bd})"
    );
    eprintln!(
        "textured 444 bd={bd} qp={qp} {}x{}  luma PSNR = {py:.2} dB ({} bytes)",
        w,
        h,
        bytes.len()
    );
}

/// Exercises the same higher-frequency texture as
/// [`run_textured_444`]/[`make_textured_source_420`], but in 4:2:0 so the
/// luma TU-split RD path runs while chroma TUs (half-resolution) stay
/// unsplit siblings.
fn run_textured_420(w: u32, h: u32, qp: u8, bd: u8) {
    let (y, cb, cr) = make_textured_source_420(w as usize, h as usize, bd);
    let config = cfg(w, h, qp, bd);
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );

    let decoded = decode(&bytes).expect("decoder must accept textured 420 stream");
    assert_plane_eq(
        &decoded.y_plane,
        &recon.y_plane,
        decoded.width as usize,
        &format!("textured 420 luma recon (qp={qp} bd={bd})"),
    );
    assert_plane_eq(
        &decoded.cb_plane,
        &recon.cb_plane,
        decoded.width.div_ceil(2) as usize,
        &format!("textured 420 cb recon (qp={qp} bd={bd})"),
    );
    assert_plane_eq(
        &decoded.cr_plane,
        &recon.cr_plane,
        decoded.width.div_ceil(2) as usize,
        &format!("textured 420 cr recon (qp={qp} bd={bd})"),
    );

    let peak = ((1u32 << bd) - 1) as f64;
    let display_y = crop_plane(
        &decoded.y_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
    );
    let py = psnr(&y, &display_y, peak);
    assert!(
        py > 18.0,
        "textured 420 luma PSNR too low: {py:.2} dB (qp={qp} bd={bd})"
    );
    eprintln!(
        "textured 420 bd={bd} qp={qp} {}x{}  luma PSNR = {py:.2} dB ({} bytes)",
        w,
        h,
        bytes.len()
    );
}

fn run(w: u32, h: u32, qp: u8) {
    run_bd(w, h, qp, 8);
}

#[test]
fn single_ctu_64x64() {
    run(64, 64, 27);
}

#[test]
fn single_ctu_qps() {
    for qp in [18, 27, 32, 40] {
        run(64, 64, qp);
    }
}

#[test]
fn full_image_multi_ctu() {
    // 192x128 = 3x2 CTUs, exercises CTU raster order + neighbour prediction.
    run(192, 128, 30);
}

#[test]
fn non_ctu_aligned_boundary_splits() {
    // Exercises H.265's forced coding-quadtree splits at the picture boundary.
    run(131, 97, 30);
}

#[test]
fn yuv444_round_trip() {
    run_444(64, 64, 27, 8);
    run_444(128, 64, 30, 10);
}

#[test]
fn textured_yuv444_round_trip() {
    run_textured_444(128, 128, 30, 8);
    run_textured_444(96, 80, 32, 10);
}

#[test]
fn textured_yuv420_round_trip() {
    run_textured_420(128, 128, 30, 8);
    run_textured_420(96, 80, 32, 10);
}

#[test]
fn ten_bit_round_trip() {
    for qp in [22, 30, 38] {
        run_bd(64, 64, qp, 10);
    }
    run_bd(128, 128, 30, 10);
}
