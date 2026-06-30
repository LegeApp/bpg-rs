//! Invariant: the encoder's *internal* reconstruction (what its RD cost model
//! and the harness `psnr_y` are measured against) must equal what the decoder
//! produces from the emitted bitstream — in the **production-shaped** config
//! (`Effort::Slow` + SAO on + deblock on), not just `Balanced`/no-filter.
//!
//! Motivation: if SAO/deblock or any emit-path step made the encoder's internal
//! recon optimistic vs the true decode, the encoder would optimize against a
//! too-good target — which would surface as "codes more residual energy yet
//! worse *decoded* PSNR". This test forecloses that hypothesis at the config
//! where the still265-vs-x265 quality gap is actually measured.

use bpg_hevc_decode::hevc::decode;
use still265::encoder::{Source, encode};
use still265::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};

/// Textured 4:2:0 source: blocky base (exercises SAO edge/band offsets and
/// deblock edges) plus per-pixel high-frequency dither (exercises non-trivial
/// residual / TU split decisions).
fn textured_420(w: usize, h: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    for j in 0..h {
        for i in 0..w {
            let base = if ((i / 16) ^ (j / 16)) & 1 == 0 {
                60
            } else {
                170
            };
            let hf = ((i * 7 + j * 13) % 23) as i32 - 11;
            y[j * w + i] = (base + hf).clamp(0, 255) as u16;
        }
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cb = vec![0u16; cw * ch];
    let mut cr = vec![0u16; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            cb[j * cw + i] = (128 + ((i as i32 * 5) % 40 - 20)).clamp(0, 255) as u16;
            cr[j * cw + i] = (128 + ((j as i32 * 5) % 40 - 20)).clamp(0, 255) as u16;
        }
    }
    (y, cb, cr)
}

fn run_slow_sao(w: u32, h: u32, qp: u8) {
    let (y, cb, cr) = textured_420(w as usize, h as usize);
    let config = StillHevcConfig {
        width: w,
        height: h,
        bit_depth: 8,
        chroma: ChromaFormat::Yuv420,
        qp,
        effort: Effort::Slow,
        sao: SaoMode::On,
        deblock: DeblockMode::On,
        adaptive_qp: false,
        aq_mode: still265::AqMode::Off,
        aq_strength: 0.35,
        aq_clamp: 2,
        two_pass_gate: true,
    };
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );
    let decoded = decode(&bytes).expect("decoder must accept the Slow+SAO stream");

    // Compare the display region (crop window) plane-by-plane. Any mismatch
    // means the encoder's internal recon != the decoded output.
    let dw = decoded.width as usize;
    let cw = decoded.width.div_ceil(2) as usize;
    assert_display_eq(
        &decoded.y_plane,
        &recon.y_plane,
        dw,
        w as usize,
        h as usize,
        qp,
        "luma",
    );
    assert_display_eq(
        &decoded.cb_plane,
        &recon.cb_plane,
        cw,
        w.div_ceil(2) as usize,
        h.div_ceil(2) as usize,
        qp,
        "cb",
    );
    assert_display_eq(
        &decoded.cr_plane,
        &recon.cr_plane,
        cw,
        w.div_ceil(2) as usize,
        h.div_ceil(2) as usize,
        qp,
        "cr",
    );
}

fn assert_display_eq(
    decoded: &[u16],
    recon: &[u16],
    stride: usize,
    disp_w: usize,
    disp_h: usize,
    qp: u8,
    label: &str,
) {
    let mut mismatches = 0u64;
    let mut first: Option<(usize, usize, u16, u16)> = None;
    let mut max_abs = 0i32;
    for j in 0..disp_h {
        for i in 0..disp_w {
            let idx = j * stride + i;
            let (d, r) = (decoded[idx], recon[idx]);
            if d != r {
                mismatches += 1;
                max_abs = max_abs.max((d as i32 - r as i32).abs());
                first.get_or_insert((i, j, d, r));
            }
        }
    }
    assert!(
        mismatches == 0,
        "{label} recon!=decode (qp={qp}): {mismatches} px differ, max|Δ|={max_abs}, \
         first at {first:?} (decoded vs recon)"
    );
}

#[test]
fn slow_sao_recon_equals_decode_aligned() {
    // CTU-aligned (128x128 = 2x2 CTUs).
    run_slow_sao(128, 128, 28);
}

#[test]
fn slow_sao_recon_equals_decode_unaligned() {
    // Not CTU-aligned (exercises boundary CUs + crop window) at two QPs.
    run_slow_sao(200, 136, 28);
    run_slow_sao(200, 136, 36);
}
