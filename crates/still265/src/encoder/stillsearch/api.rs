//! StillSearch public-in-crate entry points.
//!
//! StillSearch is split into focused modules: `depth` owns CTU dispatch and the
//! public-in-crate `StillSearch` handle; `cu`, `tu`, `rough`, `nxn`, and `eval`
//! own their respective search stages; `emit` is the only module that turns
//! internal plans into writer-facing syntax trees.

pub(in crate::encoder) use super::depth::StillSearch;

#[cfg(test)]
pub(in crate::encoder) static SPLIT_WINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(in crate::encoder) static LEAF_WINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Count of CU luma-mode decisions that selected a non-DC mode (test guard for
/// the rough+RD mode search actually doing work).
#[cfg(test)]
pub(in crate::encoder) static LUMA_NONDC_PICKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Count of CU decisions where the split candidate beat the leaf (Phase 6 guard).
#[cfg(test)]
pub(in crate::encoder) static CU_SPLIT_WINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
mod tests {
    fn scan_eval_function_bodies(src: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let bytes = src.as_bytes();
        let mut i = 0;
        while let Some(pos) = src[i..].find("fn eval_") {
            let fn_start = i + pos;
            let Some(open_rel) = src[fn_start..].find('{') else {
                break;
            };
            let open = fn_start + open_rel;
            let mut depth = 0i32;
            let mut end = open;
            for (off, b) in bytes[open..].iter().enumerate() {
                match *b {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + off + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            out.push(&src[fn_start..end]);
            i = end;
        }
        out
    }

    #[test]
    fn luma_mode_search_selects_directional_mode() {
        use crate::encoder::{Source, encode};
        use crate::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};
        use std::sync::atomic::Ordering;

        let (w, h) = (128usize, 128usize);
        let mut y = vec![0u16; w * h];
        for px in y.iter_mut().enumerate() {
            let i = px.0 % w;
            *px.1 = if (i / 4) & 1 == 0 { 60 } else { 200 };
        }
        let chroma = vec![128u16; (w / 2) * (h / 2)];
        let config = StillHevcConfig {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 27,
            effort: Effort::Balanced,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
            adaptive_qp: false,
        };
        super::LUMA_NONDC_PICKS.store(0, Ordering::Relaxed);
        let _ = encode(
            &config,
            Source {
                y: &y,
                cb: &chroma,
                cr: &chroma,
            },
        );
        let nondc = super::LUMA_NONDC_PICKS.load(Ordering::Relaxed);
        assert!(
            nondc > 0,
            "luma mode search never picked a non-DC mode on vertical-stripe content"
        );
    }

    #[test]
    fn cu_split_wins_on_mixed_direction_content() {
        use crate::encoder::{Source, encode};
        use crate::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};
        use std::sync::atomic::Ordering;

        let (w, h) = (128usize, 128usize);
        let mut y = vec![0u16; w * h];
        for j in 0..h {
            for i in 0..w {
                let v = if i < w / 2 {
                    if (i / 4) & 1 == 0 { 60 } else { 200 }
                } else if (j / 4) & 1 == 0 {
                    60
                } else {
                    200
                };
                y[j * w + i] = v;
            }
        }
        let chroma = vec![128u16; (w / 2) * (h / 2)];
        let config = StillHevcConfig {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 27,
            effort: Effort::Balanced,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
            adaptive_qp: false,
        };
        super::CU_SPLIT_WINS.store(0, Ordering::Relaxed);
        let _ = encode(
            &config,
            Source {
                y: &y,
                cb: &chroma,
                cr: &chroma,
            },
        );
        let splits = super::CU_SPLIT_WINS.load(Ordering::Relaxed);
        assert!(
            splits > 0,
            "CU split never won on mixed-direction content (Phase 6 inactive?)"
        );
    }

    #[test]
    fn tu_rd_split_wins_on_localized_detail() {
        use crate::encoder::{Source, encode};
        use crate::{ChromaFormat, DeblockMode, Effort, SaoMode, StillHevcConfig};
        use std::sync::atomic::Ordering;

        let (w, h) = (64usize, 64usize);
        let mut y = vec![128u16; w * h];
        for j in 0..16 {
            for i in 0..16 {
                y[j * w + i] = if (i ^ j) & 1 == 0 { 40 } else { 210 };
            }
        }
        let chroma = vec![128u16; (w / 2) * (h / 2)];

        let config = StillHevcConfig {
            width: w as u32,
            height: h as u32,
            bit_depth: 8,
            chroma: ChromaFormat::Yuv420,
            qp: 27,
            effort: Effort::Balanced,
            sao: SaoMode::Off,
            deblock: DeblockMode::Off,
            adaptive_qp: false,
        };

        super::SPLIT_WINS.store(0, Ordering::Relaxed);
        super::LEAF_WINS.store(0, Ordering::Relaxed);
        let _ = encode(
            &config,
            Source {
                y: &y,
                cb: &chroma,
                cr: &chroma,
            },
        );
        let splits = super::SPLIT_WINS.load(Ordering::Relaxed);
        let leaves = super::LEAF_WINS.load(Ordering::Relaxed);
        assert!(
            splits > 0,
            "TU RD split never won on localized-detail content (splits={splits}, leaves={leaves})"
        );
    }

    #[test]
    fn search_eval_helpers_do_not_mutate_shared_frame() {
        for body in scan_eval_function_bodies(include_str!("eval.rs")) {
            assert!(
                !body.contains("state.frame.plane_mut") && !body.contains("predict_intra_tiled"),
                "eval_* functions must write to overlays/scratch, not mutate state.frame directly:\n{body}"
            );
        }
    }
}
