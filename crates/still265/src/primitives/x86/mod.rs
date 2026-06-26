//! x86/x86_64 primitive backends.
//!
//! `sse2` kernels are unconditionally compiled (SSE2 is the x86_64 baseline).
//! `avx2` kernels are guarded with `#[target_feature(enable = "avx2")]` and
//! are only installed after `is_x86_feature_detected!("avx2")`.

pub mod avx2;
pub mod sse2;

/// Install the best available x86 kernels into `p`.
/// Called from `select_primitives` after the wide-SIMD layer has been applied.
pub(super) fn setup(p: &mut crate::primitives::Primitives) {
    // SSE2 is always available on x86_64.
    if std::arch::is_x86_feature_detected!("sse2") {
        sse2::setup(p);
    }

    if std::arch::is_x86_feature_detected!("avx2") {
        avx2::setup(p);
    }
}
