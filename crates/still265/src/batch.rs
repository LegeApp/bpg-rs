//! Batch still-image encoding with memory-budget-aware concurrency.
//!
//! Encoding many images at once is bounded by RAM as much as by CPU: each
//! in-flight encode holds a reconstruction frame, and the parallel wavefront
//! path additionally holds worker-local state. Modern frameless WPP workers no
//! longer clone the reconstruction pixel planes, but still carry small maps and
//! row-strip buffers; `BPG_WPP_FRAMELESS=0` restores full worker frames. This
//! module surfaces a [`MemoryEstimate`] so a caller can size concurrency against
//! a RAM budget *before* spawning work, plus a simple image-parallel
//! [`encode_batch`] runner.
//!
//! ```no_run
//! use still265::{batch, StillHevcConfig, Effort, SaoMode, DeblockMode, ChromaFormat};
//! use still265::encoder::Source;
//!
//! let cfg = StillHevcConfig {
//!     width: 4096, height: 3072, bit_depth: 8, chroma: ChromaFormat::Yuv420,
//!     qp: 28, effort: Effort::Slow, sao: SaoMode::On, deblock: DeblockMode::On,
//!     adaptive_qp: false, aq_mode: still265::AqMode::Off, aq_strength: 0.35,
//!     aq_clamp: 2, two_pass_gate: true,
//! };
//! // How much RAM will one encode of this size take?
//! let est = batch::estimate_memory(&cfg);
//! println!("{:.0} MiB/instance", est.peak_mib());
//!
//! // Given an 8 GiB budget, how many should run at once?
//! let conc = batch::recommended_concurrency(&cfg, 8 << 30);
//! let jobs: Vec<batch::BatchJob> = /* ... */ Vec::new();
//! let results = batch::encode_batch(&jobs, conc);
//! # let _ = (results, conc);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use bpg_image::ChromaFormat;

use crate::encoder::{EncodeStats, Source, encode_with_stats};
use crate::{Effort, StillHevcConfig};

const CTB: u32 = 64;
const BYTES_PER_SAMPLE: u64 = 2; // planes are u16

/// Estimated peak working-set memory for a single encode instance.
///
/// All fields are bytes unless noted. `peak_bytes` is the figure to budget
/// against; it is a deliberate slight over-estimate (it includes a fixed
/// fraction for scratch/transform/CABAC buffers) so dividing a RAM budget by it
/// is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// One reconstruction frame's pixel planes (CTB-aligned, all components).
    pub frame_bytes: u64,
    /// The input source planes (display resolution, all components).
    pub source_bytes: u64,
    /// Number of per-worker frame clones the chosen effort will hold
    /// concurrently (0 for serial configs and for the default frameless WPP
    /// path; `BPG_WPP_FRAMELESS=0` restores one full frame per worker).
    pub worker_frames: u32,
    /// Worker-local memory not counted in `frame_bytes`: either full
    /// per-worker frames (`worker_frames * frame_bytes`) or, for frameless WPP,
    /// the worker-local metadata maps plus row-strip buffers.
    pub worker_bytes: u64,
    /// Estimated peak resident memory for the whole encode.
    pub peak_bytes: u64,
}

impl MemoryEstimate {
    /// Peak estimate in MiB.
    pub fn peak_mib(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

fn round_up(v: u32, to: u32) -> u32 {
    v.div_ceil(to) * to
}

/// Total samples across all coded components for `w x h` luma in `chroma`.
fn plane_samples(w: u32, h: u32, chroma: ChromaFormat) -> u64 {
    let luma = w as u64 * h as u64;
    match chroma {
        ChromaFormat::Gray => luma,
        // Two chroma planes; sub-sampled per format (round up each dim).
        ChromaFormat::Yuv420 => luma + 2 * (w.div_ceil(2) as u64 * h.div_ceil(2) as u64),
        ChromaFormat::Yuv422 => luma + 2 * (w.div_ceil(2) as u64 * h as u64),
        ChromaFormat::Yuv444 => luma * 3,
    }
}

/// Whether `config` runs the frame-cloning parallel wavefront path. Mirrors the
/// encoder's wavefront gate, including AQ's build-only WPP acceleration path.
fn is_parallel(config: &StillHevcConfig) -> bool {
    if crate::aq_active(config) {
        return crate::params::effective_wpp_build_enabled(config);
    }
    match config.effort {
        Effort::Placebo => true,
        _ => std::env::var("BPG_BEST2_PARALLEL")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true),
    }
}

/// Worker count the parallel path would use: the requested thread count
/// (`BPG_ENC_THREADS` or the runtime parallelism) capped by the CTB-row count.
fn worker_count(coded_h: u32) -> u32 {
    let ctbs_y = (coded_h / CTB).max(1);
    let requested = std::env::var("BPG_ENC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1) as u32;
    requested.min(ctbs_y)
}

/// Mirror the encoder's frameless WPP gate for memory estimation. Frameless is
/// the default when the import width is the 1-row strip default; explicit
/// wider imports or `BPG_WPP_FRAMELESS=0` require full worker frames.
fn frameless_wpp_estimate_enabled() -> bool {
    let frameless = std::env::var("BPG_WPP_FRAMELESS")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    let import_one = match std::env::var("BPG_WPP_IMPORT_ROWS") {
        Ok(v) if v.trim() == "full" => false,
        Ok(v) => v.trim().parse::<usize>().ok() == Some(1),
        Err(_) => true,
    };
    frameless && import_one
}

/// Estimate peak memory for one encode of `config`, accounting for the chosen
/// effort's parallelism (per-worker frame clones) and the current
/// `BPG_ENC_THREADS` / CPU count. Pure — does not encode anything.
pub fn estimate_memory(config: &StillHevcConfig) -> MemoryEstimate {
    let coded_w = round_up(config.width, CTB);
    let coded_h = round_up(config.height, CTB);

    // Pixel planes (u16) plus a generous allowance for the per-frame auxiliary
    // maps (qp map, mode/ct-depth/tu-depth maps at 4x4 granularity, deblock edge
    // flags): ~0.25x the plane bytes, rounded up.
    let frame_pixels = plane_samples(coded_w, coded_h, config.chroma) * BYTES_PER_SAMPLE;
    let frame_aux_bytes = frame_pixels / 4;
    let frame_bytes = frame_pixels + frame_aux_bytes;
    let source_bytes = plane_samples(config.width, config.height, config.chroma) * BYTES_PER_SAMPLE;

    let workers = if is_parallel(config) {
        worker_count(coded_h)
    } else {
        0
    };
    let frameless_workers = workers > 0 && frameless_wpp_estimate_enabled();
    let worker_frames = if frameless_workers { 0 } else { workers };
    let worker_bytes = if workers == 0 {
        0
    } else if frameless_workers {
        // Frameless WPP workers keep per-frame metadata maps (deblock/QP plus
        // encoder mode and CT-depth maps) but no reconstruction pixel planes.
        // They also keep one full-plane-width row strip per component. Add a
        // small fixed scratch allowance for CABAC/search/canvas workspaces.
        let row_strip_bytes =
            plane_samples(coded_w, 1, config.chroma).saturating_mul(BYTES_PER_SAMPLE);
        workers as u64 * (frame_aux_bytes + row_strip_bytes + 64 * 1024)
    } else {
        workers as u64 * frame_bytes
    };

    // The main reconstruction frame is always live; the parallel path adds
    // worker-local state. Plus the source, plus ~15% for transform/RDOQ/CABAC
    // scratch and the output bitstream.
    let base = frame_bytes + worker_bytes + source_bytes;
    let peak_bytes = base + base / 7;

    MemoryEstimate {
        frame_bytes,
        source_bytes,
        worker_frames,
        worker_bytes,
        peak_bytes,
    }
}

/// How many encodes of `config` can run concurrently within `ram_budget_bytes`,
/// clamped to `[1, available_parallelism]`. Note the parallel wavefront path is
/// already multi-threaded *within* one encode, so for batch throughput it is
/// usually best to run each encode serially (`BPG_ENC_THREADS=1`) and let this
/// concurrency provide the parallelism across images.
pub fn recommended_concurrency(config: &StillHevcConfig, ram_budget_bytes: u64) -> usize {
    let per = estimate_memory(config).peak_bytes.max(1);
    let by_ram = (ram_budget_bytes / per).max(1) as usize;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    by_ram.min(cores).max(1)
}

/// One unit of batch work: a per-image config and its source planes.
pub struct BatchJob<'a> {
    pub config: StillHevcConfig,
    pub source: Source<'a>,
}

/// The result of encoding one [`BatchJob`].
pub struct BatchOutput {
    pub annexb: Vec<u8>,
    pub stats: EncodeStats,
}

/// Encode `jobs` using up to `concurrency` worker threads (image-level
/// parallelism), preserving input order in the returned vector.
///
/// Each job is independent, so this scales cleanly across images. Remember that
/// the parallel wavefront path also threads *within* an encode: to avoid
/// oversubscription when `concurrency > 1`, set `BPG_ENC_THREADS=1` so each
/// encode runs serially and this provides the across-image parallelism. Size
/// `concurrency` with [`recommended_concurrency`] to stay within a RAM budget.
pub fn encode_batch(jobs: &[BatchJob<'_>], concurrency: usize) -> Vec<BatchOutput> {
    let n = jobs.len();
    if n == 0 {
        return Vec::new();
    }
    let conc = concurrency.max(1).min(n);
    if conc == 1 {
        return jobs
            .iter()
            .map(|j| {
                let (annexb, _recon, stats) = encode_with_stats(&j.config, j.source);
                BatchOutput { annexb, stats }
            })
            .collect();
    }

    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<BatchOutput>> = (0..n).map(|_| None).collect();
    std::thread::scope(|s| {
        // Each worker grabs the next job index until the queue drains, returning
        // (index, output) pairs that are merged back in input order after join.
        let handles: Vec<_> = (0..conc)
            .map(|_| {
                let next = &next;
                let jobs = &jobs;
                s.spawn(move || {
                    let mut local: Vec<(usize, BatchOutput)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let j = &jobs[i];
                        let (annexb, _recon, stats) = encode_with_stats(&j.config, j.source);
                        local.push((i, BatchOutput { annexb, stats }));
                    }
                    local
                })
            })
            .collect();
        for h in handles {
            for (i, out) in h.join().expect("batch worker panicked") {
                slots[i] = Some(out);
            }
        }
    });
    slots
        .into_iter()
        .map(|o| o.expect("every job encoded"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeblockMode, Effort, SaoMode};

    fn cfg(w: u32, h: u32, effort: Effort, chroma: ChromaFormat) -> StillHevcConfig {
        StillHevcConfig {
            width: w,
            height: h,
            bit_depth: 8,
            chroma,
            qp: 28,
            effort,
            sao: SaoMode::On,
            deblock: DeblockMode::On,
            adaptive_qp: false,
            aq_mode: crate::AqMode::Off,
            aq_strength: 0.35,
            aq_clamp: 2,
            two_pass_gate: true,
        }
    }

    #[test]
    fn estimate_scales_with_resolution_and_format() {
        let small = estimate_memory(&cfg(512, 512, Effort::Fast, ChromaFormat::Yuv420));
        let big = estimate_memory(&cfg(4096, 4096, Effort::Fast, ChromaFormat::Yuv420));
        assert!(big.peak_bytes > small.peak_bytes * 50);

        let yuv420 = estimate_memory(&cfg(1024, 1024, Effort::Fast, ChromaFormat::Yuv420));
        let yuv444 = estimate_memory(&cfg(1024, 1024, Effort::Fast, ChromaFormat::Yuv444));
        assert!(yuv444.frame_bytes > yuv420.frame_bytes);
    }

    #[test]
    fn aq_configs_follow_build_only_wpp_gate() {
        // AQ streams keep serial slice syntax, but CU-tree construction may
        // still use WPP row workers (build-only acceleration, `BPG_AQ_WPP`).
        // The memory estimate must mirror that gate.
        let mut fast_aq = cfg(1024, 768, Effort::Fast, ChromaFormat::Yuv420);
        fast_aq.adaptive_qp = true;
        let est = estimate_memory(&fast_aq);
        if crate::params::effective_wpp_build_enabled(&fast_aq) {
            assert!(est.worker_bytes > 0);
        } else {
            assert_eq!(est.worker_frames, 0);
            assert_eq!(est.worker_bytes, 0);
        }
    }

    #[test]
    fn uniform_speed_tiers_account_for_wavefront_worker_memory() {
        if std::env::var("BPG_BEST2_PARALLEL")
            .ok()
            .is_some_and(|v| v.trim() == "0")
        {
            return;
        }
        let slow = estimate_memory(&cfg(1024, 768, Effort::Slow, ChromaFormat::Yuv420));
        assert!(slow.worker_bytes > 0);
        if std::env::var("BPG_WPP_FRAMELESS")
            .ok()
            .is_some_and(|v| v.trim() == "0")
            || std::env::var("BPG_WPP_IMPORT_ROWS")
                .ok()
                .is_some_and(|v| v.trim() != "1")
        {
            assert!(slow.worker_frames > 0);
            assert!(slow.worker_frames <= 12);
        } else {
            assert_eq!(slow.worker_frames, 0);
            let full_clone_bytes = worker_count(round_up(768, CTB)) as u64 * slow.frame_bytes;
            assert!(slow.worker_bytes < full_clone_bytes / 2);
        }
    }

    #[test]
    fn recommended_concurrency_respects_budget() {
        let c = cfg(1024, 768, Effort::Fast, ChromaFormat::Yuv420);
        let per = estimate_memory(&c).peak_bytes;
        // A budget below one instance still yields at least 1.
        assert_eq!(recommended_concurrency(&c, per / 2), 1);
        // A large budget is capped by core count, not RAM.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(recommended_concurrency(&c, per * 10_000), cores.max(1));
    }
}
