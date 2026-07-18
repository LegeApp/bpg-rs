//! External AQ correctness regressions.
//!
//! These tests intentionally live in their own integration-test binary because
//! `BPG_AQ_OFFSET_MAP` is process-global. Keeping all env-var AQ cases in this
//! single test avoids contaminating unrelated encode tests that may run in
//! parallel inside another test binary.

use bpg_hevc_decode::hevc::decode;
use still265::encoder::{Source, encode};
use still265::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};

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

fn cfg(w: u32, h: u32, qp: u8, bd: u8) -> StillHevcConfig {
    StillHevcConfig {
        width: w,
        height: h,
        bit_depth: bd,
        chroma: ChromaFormat::Yuv420,
        qp,
        effort: Effort::Slow,
        sao: SaoMode::Off,
        deblock: DeblockMode::Off,
        adaptive_qp: false,
        aq_mode: still265::AqMode::Off,
        aq_strength: 0.35,
        aq_clamp: 2,
        two_pass_gate: true,
        psy_rd: 0.0,
        psy_rdoq: 0.0,
        aq_qg: 32,
    }
}

fn cfg_sao_deblock(w: u32, h: u32, qp: u8, bd: u8) -> StillHevcConfig {
    StillHevcConfig {
        sao: SaoMode::On,
        deblock: DeblockMode::On,
        ..cfg(w, h, qp, bd)
    }
}

fn map_rows(offset_at: impl Fn(u32, u32) -> i8) -> String {
    let mut out = String::from("# x,y,offset\n");
    for y in [0u32, 32, 64] {
        for x in [0u32, 32, 64] {
            out.push_str(&format!("{x},{y},{}\n", offset_at(x, y)));
        }
    }
    out
}

/// Map rows on the 16x16 QG grid (for `aq_qg = 16` encodes): 16-aligned
/// coordinates covering the whole 96x96 test picture.
fn map_rows16(offset_at: impl Fn(u32, u32) -> i8) -> String {
    let mut out = String::from("# x,y,offset (16x16 QGs)\n");
    for y in (0u32..96).step_by(16) {
        for x in (0u32..96).step_by(16) {
            out.push_str(&format!("{x},{y},{}\n", offset_at(x, y)));
        }
    }
    out
}

fn assert_display_eq(
    decoded: &[u16],
    recon: &[u16],
    stride: usize,
    disp_w: usize,
    disp_h: usize,
    label: &str,
) {
    for y in 0..disp_h {
        for x in 0..disp_w {
            let idx = y * stride + x;
            assert_eq!(
                decoded[idx], recon[idx],
                "{label} mismatch at ({x},{y}) idx={idx}: decoded={} recon={}",
                decoded[idx], recon[idx]
            );
        }
    }
}

fn run_external_map_case(name: &str, map_body: String) {
    run_external_map_case_with_config(name, map_body, cfg);
}

fn run_external_map_case_with_config(
    name: &str,
    map_body: String,
    config_for: impl Fn(u32, u32, u8, u8) -> StillHevcConfig,
) {
    // `BPG_AQ_OFFSET_MAP` is process-global and the harness runs `#[test]`s on
    // parallel threads — serialize every external-map case on one lock so the
    // tests can't clobber each other's map path mid-encode.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let w = 96u32;
    let h = 96u32;
    let bd = 8u8;
    let path = std::env::temp_dir().join(format!(
        "still265_external_aq_correctness_{}_{}.csv",
        std::process::id(),
        name
    ));
    std::fs::write(&path, map_body).unwrap();

    unsafe {
        std::env::set_var("BPG_AQ_OFFSET_MAP", &path);
    }

    let (y, cb, cr) = make_textured_source_420(w as usize, h as usize, bd);
    let config = config_for(w, h, 30, bd);
    assert!(
        still265::aq_active(&config),
        "{name}: external map should enable AQ syntax"
    );
    let (bytes, recon) = encode(
        &config,
        Source {
            y: &y,
            cb: &cb,
            cr: &cr,
        },
    );

    unsafe {
        std::env::remove_var("BPG_AQ_OFFSET_MAP");
    }
    let _ = std::fs::remove_file(&path);

    let decoded =
        decode(&bytes).unwrap_or_else(|err| panic!("{name}: decoder rejected stream: {err}"));
    assert_display_eq(
        &decoded.y_plane,
        &recon.y_plane,
        decoded.width as usize,
        w as usize,
        h as usize,
        &format!("{name}: luma"),
    );
    assert_display_eq(
        &decoded.cb_plane,
        &recon.cb_plane,
        decoded.width.div_ceil(2) as usize,
        w.div_ceil(2) as usize,
        h.div_ceil(2) as usize,
        &format!("{name}: cb"),
    );
    assert_display_eq(
        &decoded.cr_plane,
        &recon.cr_plane,
        decoded.width.div_ceil(2) as usize,
        w.div_ceil(2) as usize,
        h.div_ceil(2) as usize,
        &format!("{name}: cr"),
    );
}

#[test]
fn external_aq_420_offset_maps_recon_decode_round_trip() {
    run_external_map_case("zero", map_rows(|_, _| 0));
    run_external_map_case("constant_plus1", map_rows(|_, _| 1));
    run_external_map_case("constant_minus1", map_rows(|_, _| -1));
    run_external_map_case(
        "single_qg_plus1",
        map_rows(|x, y| if x == 32 && y == 32 { 1 } else { 0 }),
    );
    run_external_map_case("mixed_nonzero", mixed_nonzero_map());
    run_external_map_case_with_config(
        "mixed_nonzero_sao_deblock",
        mixed_nonzero_map(),
        cfg_sao_deblock,
    );
}

/// External offset map on the 16x16 QG grid (`aq_qg = 16`, PPS
/// `diff_cu_qp_delta_depth = 2`): the emitted stream must decode to exactly
/// the encoder's reconstruction, with and without SAO+deblock.
#[test]
fn external_aq_qg16_offset_maps_recon_decode_round_trip() {
    let cfg_qg16 = |w, h, qp, bd| StillHevcConfig {
        aq_qg: 16,
        ..cfg(w, h, qp, bd)
    };
    let cfg_qg16_sao_deblock = |w, h, qp, bd| StillHevcConfig {
        aq_qg: 16,
        ..cfg_sao_deblock(w, h, qp, bd)
    };
    // Per-16x16-cell offsets spanning the clamp range, varying between
    // horizontally/vertically adjacent QGs so the depth-2 QP prediction is
    // genuinely exercised.
    let fine = |x: u32, y: u32| (((x / 16 + 2 * (y / 16)) % 5) as i8) - 2;
    run_external_map_case_with_config("qg16_mixed", map_rows16(fine), cfg_qg16);
    run_external_map_case_with_config(
        "qg16_mixed_sao_deblock",
        map_rows16(fine),
        cfg_qg16_sao_deblock,
    );
}

fn mixed_nonzero_map() -> String {
    map_rows(|x, y| match (x, y) {
        (0, 0) => -2,
        (32, 0) => 1,
        (64, 0) => 2,
        (0, 32) => 0,
        (32, 32) => -1,
        (64, 32) => 2,
        (0, 64) => 1,
        (32, 64) => -2,
        (64, 64) => 0,
        _ => 0,
    })
}
