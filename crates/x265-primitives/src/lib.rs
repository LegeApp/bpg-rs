//! Safe wrappers over `x265-primitives-sys`'s `extern "C"` boundary onto a
//! subset of x265's scalar (`_c`) `EncoderPrimitives` (Phase 4 of
//! `FULL_RUST_CODEC_PLAN.md`).
//!
//! This is the intra vertical slice: square-block DCT/IDCT/DST/IDST,
//! quant/dequant, square-PU SATD/SAD, square-CU residual/reconstruction/copy/
//! SSE/variance/SA8D helpers (`common/pixel.cpp`), and DC/Planar/angular intra
//! prediction (`common/intrapred.cpp`).
//!
//! The crate-root modules ([`dct`], [`pixel`], [`intra`], [`recon`],
//! [`quant`]) are the 8-bit surface (`pixel == u8`). The [`bitdepth10`] module
//! mirrors **the same surface** for 10-bit (`pixel == u16`) through identically
//! shaped submodules (`bitdepth10::dct` etc.), built from the dual-compiled
//! x265 namespace (`HIGH_BIT_DEPTH=1`, see `x265-primitives-sys`). Both depths
//! are generated from the same crate-level macros, so they cannot drift.
//!
//! These wrappers call directly into x265's C++ scalar reference
//! implementations (`ENABLE_ASSEMBLY` off) -- the same code x265 falls back
//! to without a matching SIMD backend, and the "oracle" Phase 4's
//! acceptance criterion (primitive tests match x265 for fixed vectors across
//! supported bit depths) compares against. No behavior is reimplemented here.

use x265_primitives_sys as sys;

// --- shared, bit-depth-independent precondition helpers -------------------

/// Asserts a `size`x`size` block fits in `pix` at the given element `stride`.
fn check_block<T>(pix: &[T], stride: usize, size: usize) {
    assert!(
        stride >= size,
        "stride {stride} smaller than block size {size}"
    );
    assert!(
        pix.len() >= stride * (size - 1) + size,
        "pixel slice too short for a {size}x{size} block at stride {stride}"
    );
}

/// HEVC angular intra modes are 2..=34 (mode 0 = Planar, 1 = DC are separate).
fn check_angular_mode(mode: u8) {
    assert!(
        (2..=34).contains(&mode),
        "angular intra mode must be in 2..=34"
    );
}

// --- crate-level generating macros (instantiated per bit depth) -----------

/// Forward/inverse square DCT-II pair (coefficients are `i16`, bit-depth
/// independent in signature; the FFI symbol selects the depth-specialized
/// primitive).
macro_rules! transform_pair {
    ($name_fwd:ident, $name_inv:ident, $n:expr, $fwd_fn:path, $inv_fn:path) => {
        #[doc = concat!(
                                    "Forward DCT-II of a ", stringify!($n), "x", stringify!($n),
                                    " residual block (row-major, stride ", stringify!($n), ")."
                                )]
        pub fn $name_fwd(block: &[i16; $n * $n]) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe { $fwd_fn(block.as_ptr(), out.as_mut_ptr()) };
            out
        }

        #[doc = concat!(
                                    "Inverse DCT-II of a ", stringify!($n), "x", stringify!($n),
                                    " coefficient block (row-major, stride ", stringify!($n), ")."
                                )]
        pub fn $name_inv(coeffs: &[i16; $n * $n]) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe { $inv_fn(coeffs.as_ptr(), out.as_mut_ptr()) };
            out
        }
    };
}

/// 4x4 forward/inverse DST-VII pair used by HEVC intra luma residual coding.
macro_rules! dst_pair {
    ($fwd:ident, $inv:ident, $fwd_fn:path, $inv_fn:path) => {
        /// Forward 4x4 DST used by HEVC intra residual coding for selected
        /// luma transform blocks.
        pub fn $fwd(block: &[i16; 16]) -> [i16; 16] {
            let mut out = [0i16; 16];
            unsafe { $fwd_fn(block.as_ptr(), out.as_mut_ptr()) };
            out
        }

        /// Inverse 4x4 DST used by HEVC intra residual reconstruction.
        pub fn $inv(coeffs: &[i16; 16]) -> [i16; 16] {
            let mut out = [0i16; 16];
            unsafe { $inv_fn(coeffs.as_ptr(), out.as_mut_ptr()) };
            out
        }
    };
}

/// SATD/SAD pair over square luma PU blocks of pixel type `$px`.
macro_rules! cmp_pair {
    ($name_satd:ident, $name_sad:ident, $n:expr, $px:ty, $satd_fn:path, $sad_fn:path) => {
        #[doc = concat!("SATD of two ", stringify!($n), "x", stringify!($n), " luma blocks.")]
        pub fn $name_satd(pix1: &[$px], stride1: usize, pix2: &[$px], stride2: usize) -> u32 {
            crate::check_block(pix1, stride1, $n);
            crate::check_block(pix2, stride2, $n);
            unsafe {
                $satd_fn(
                    pix1.as_ptr(),
                    stride1 as isize,
                    pix2.as_ptr(),
                    stride2 as isize,
                ) as u32
            }
        }

        #[doc = concat!("SAD of two ", stringify!($n), "x", stringify!($n), " luma blocks.")]
        pub fn $name_sad(pix1: &[$px], stride1: usize, pix2: &[$px], stride2: usize) -> u32 {
            crate::check_block(pix1, stride1, $n);
            crate::check_block(pix2, stride2, $n);
            unsafe {
                $sad_fn(
                    pix1.as_ptr(),
                    stride1 as isize,
                    pix2.as_ptr(),
                    stride2 as isize,
                ) as u32
            }
        }
    };
}

/// DC/Planar/angular intra prediction plus reference-sample filtering for one
/// square block size and pixel type `$px`.
///
/// `refs` is x265's reference-sample layout: `refs[0]` is the above-left
/// corner, `refs[1..=size]` are the "above" row, and
/// `refs[2*size+1..=3*size]` are the "left" column (length `4*size + 1`;
/// the unused fourth quadrant keeps every size's buffer a uniform shape).
macro_rules! dc_planar {
    ($name_dc:ident, $name_planar:ident, $name_ang:ident, $name_filter:ident, $n:expr, $px:ty,
     $dc_fn:path, $planar_fn:path, $ang_fn:path, $filter_fn:path) => {
        #[doc = concat!("DC intra prediction for a ", stringify!($n), "x", stringify!($n), " block.")]
        pub fn $name_dc(refs: &[$px; 4 * $n + 1], strong_filter: bool) -> [$px; $n * $n] {
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe {
                $dc_fn(out.as_mut_ptr(), $n, refs.as_ptr(), strong_filter as i32);
            }
            out
        }

        #[doc = concat!("Planar intra prediction for a ", stringify!($n), "x", stringify!($n), " block.")]
        pub fn $name_planar(refs: &[$px; 4 * $n + 1]) -> [$px; $n * $n] {
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe {
                $planar_fn(out.as_mut_ptr(), $n, refs.as_ptr());
            }
            out
        }

        #[doc = concat!("Angular intra prediction for a ", stringify!($n), "x", stringify!($n), " block.")]
        pub fn $name_ang(refs: &[$px; 4 * $n + 1], mode: u8, filter: bool) -> [$px; $n * $n] {
            crate::check_angular_mode(mode);
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe {
                $ang_fn(out.as_mut_ptr(), $n, refs.as_ptr(), mode as i32, filter as i32);
            }
            out
        }

        #[doc = concat!("Filter intra reference samples for a ", stringify!($n), "x", stringify!($n), " block.")]
        pub fn $name_filter(refs: &[$px; 4 * $n + 1]) -> [$px; 4 * $n + 1] {
            let mut out: [$px; 4 * $n + 1] = [0; 4 * $n + 1];
            unsafe {
                $filter_fn(refs.as_ptr(), out.as_mut_ptr());
            }
            out
        }
    };
}

/// Residual/reconstruction/copy/distortion helpers for one square CU/TU size,
/// pixel type `$px`, and pixel value ceiling `$pmax` (255 at 8-bit, 1023 at
/// 10-bit) used by the `copy_sp` precondition check.
macro_rules! recon_fns {
    (
        $sub:ident, $add:ident, $copy_pp:ident, $copy_ps:ident, $copy_sp:ident,
        $copy_ss:ident, $sse_pp:ident, $sse_ss:ident, $var:ident, $ssd_s:ident,
        $sa8d:ident, $blockfill_s:ident, $transpose:ident, $cpy2d1d_shl:ident,
        $cpy2d1d_shr:ident, $cpy1d2d_shl:ident, $cpy1d2d_shr:ident, $n:expr,
        $px:ty, $pmax:expr,
        $sub_fn:path, $add_fn:path, $copy_pp_fn:path, $copy_ps_fn:path,
        $copy_sp_fn:path, $copy_ss_fn:path, $sse_pp_fn:path, $sse_ss_fn:path,
        $var_fn:path, $ssd_s_fn:path, $sa8d_fn:path, $blockfill_s_fn:path,
        $transpose_fn:path, $cpy2d1d_shl_fn:path, $cpy2d1d_shr_fn:path,
        $cpy1d2d_shl_fn:path, $cpy1d2d_shr_fn:path
    ) => {
        #[doc = concat!("Residual subtraction for a ", stringify!($n), "x", stringify!($n), " block: `src0 - src1`.")]
        pub fn $sub(src0: &[$px; $n * $n], src1: &[$px; $n * $n]) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe {
                $sub_fn(out.as_mut_ptr(), $n, src0.as_ptr(), src1.as_ptr(), $n, $n);
            }
            out
        }

        #[doc = concat!("Reconstruct a ", stringify!($n), "x", stringify!($n), " block: clip(pred + residual).")]
        pub fn $add(pred: &[$px; $n * $n], residual: &[i16; $n * $n]) -> [$px; $n * $n] {
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe {
                $add_fn(out.as_mut_ptr(), $n, pred.as_ptr(), residual.as_ptr(), $n, $n);
            }
            out
        }

        #[doc = concat!("Copy a ", stringify!($n), "x", stringify!($n), " pixel block.")]
        pub fn $copy_pp(src: &[$px; $n * $n]) -> [$px; $n * $n] {
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe { $copy_pp_fn(out.as_mut_ptr(), $n, src.as_ptr(), $n) };
            out
        }

        #[doc = concat!("Copy a ", stringify!($n), "x", stringify!($n), " pixel block to i16.")]
        pub fn $copy_ps(src: &[$px; $n * $n]) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe { $copy_ps_fn(out.as_mut_ptr(), $n, src.as_ptr(), $n) };
            out
        }

        #[doc = concat!("Copy a ", stringify!($n), "x", stringify!($n), " in-range i16 pixel block back to the pixel type.")]
        pub fn $copy_sp(src: &[i16; $n * $n]) -> [$px; $n * $n] {
            assert!(
                src.iter().all(|&v| (0..=$pmax).contains(&v)),
                "copy_sp source contains values outside the pixel range"
            );
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe { $copy_sp_fn(out.as_mut_ptr(), $n, src.as_ptr(), $n) };
            out
        }

        #[doc = concat!("Copy a ", stringify!($n), "x", stringify!($n), " i16 block.")]
        pub fn $copy_ss(src: &[i16; $n * $n]) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe { $copy_ss_fn(out.as_mut_ptr(), $n, src.as_ptr(), $n) };
            out
        }

        #[doc = concat!("SSE between two ", stringify!($n), "x", stringify!($n), " pixel blocks.")]
        pub fn $sse_pp(src0: &[$px; $n * $n], src1: &[$px; $n * $n]) -> u64 {
            unsafe { $sse_pp_fn(src0.as_ptr(), $n, src1.as_ptr(), $n) }
        }

        #[doc = concat!("SSE between two ", stringify!($n), "x", stringify!($n), " i16 blocks.")]
        pub fn $sse_ss(src0: &[i16; $n * $n], src1: &[i16; $n * $n]) -> u64 {
            unsafe { $sse_ss_fn(src0.as_ptr(), $n, src1.as_ptr(), $n) }
        }

        #[doc = concat!("x265 packed variance tuple for a ", stringify!($n), "x", stringify!($n), " pixel block: sum | (squared_sum << 32).")]
        pub fn $var(src: &[$px; $n * $n]) -> u64 {
            unsafe { $var_fn(src.as_ptr(), $n) }
        }

        #[doc = concat!("SSD of a ", stringify!($n), "x", stringify!($n), " i16 block against zero.")]
        pub fn $ssd_s(src: &[i16; $n * $n]) -> u64 {
            unsafe { $ssd_s_fn(src.as_ptr(), $n) }
        }

        #[doc = concat!("SA8D of two ", stringify!($n), "x", stringify!($n), " pixel blocks.")]
        pub fn $sa8d(src0: &[$px; $n * $n], src1: &[$px; $n * $n]) -> u32 {
            unsafe { $sa8d_fn(src0.as_ptr(), $n, src1.as_ptr(), $n) as u32 }
        }

        #[doc = concat!("Fill a ", stringify!($n), "x", stringify!($n), " i16 block.")]
        pub fn $blockfill_s(value: i16) -> [i16; $n * $n] {
            let mut out = [0i16; $n * $n];
            unsafe { $blockfill_s_fn(out.as_mut_ptr(), $n, value) };
            out
        }

        #[doc = concat!("Transpose a ", stringify!($n), "x", stringify!($n), " pixel block.")]
        pub fn $transpose(src: &[$px; $n * $n]) -> [$px; $n * $n] {
            let mut out: [$px; $n * $n] = [0; $n * $n];
            unsafe { $transpose_fn(out.as_mut_ptr(), src.as_ptr(), $n) };
            out
        }

        #[doc = concat!("Copy ", stringify!($n), "x", stringify!($n), " i16 2D-to-1D with left shift.")]
        pub fn $cpy2d1d_shl(src: &[i16; $n * $n], shift: i32) -> [i16; $n * $n] {
            assert!(shift >= 0, "shift must be non-negative");
            let mut out = [0i16; $n * $n];
            unsafe { $cpy2d1d_shl_fn(out.as_mut_ptr(), src.as_ptr(), $n, shift) };
            out
        }

        #[doc = concat!("Copy ", stringify!($n), "x", stringify!($n), " i16 2D-to-1D with rounded right shift.")]
        pub fn $cpy2d1d_shr(src: &[i16; $n * $n], shift: i32) -> [i16; $n * $n] {
            assert!(shift > 0, "shift must be positive");
            let mut out = [0i16; $n * $n];
            unsafe { $cpy2d1d_shr_fn(out.as_mut_ptr(), src.as_ptr(), $n, shift) };
            out
        }

        #[doc = concat!("Copy ", stringify!($n), "x", stringify!($n), " i16 1D-to-2D with left shift.")]
        pub fn $cpy1d2d_shl(src: &[i16; $n * $n], shift: i32) -> [i16; $n * $n] {
            assert!(shift >= 0, "shift must be non-negative");
            let mut out = [0i16; $n * $n];
            unsafe { $cpy1d2d_shl_fn(out.as_mut_ptr(), $n, src.as_ptr(), shift) };
            out
        }

        #[doc = concat!("Copy ", stringify!($n), "x", stringify!($n), " i16 1D-to-2D with rounded right shift.")]
        pub fn $cpy1d2d_shr(src: &[i16; $n * $n], shift: i32) -> [i16; $n * $n] {
            assert!(shift > 0, "shift must be positive");
            let mut out = [0i16; $n * $n];
            unsafe { $cpy1d2d_shr_fn(out.as_mut_ptr(), $n, src.as_ptr(), shift) };
            out
        }
    };
}

/// `count_nonzero`/`copy_cnt` coefficient helpers for one square block size
/// (coefficients are `i16`; the FFI symbol selects the depth).
macro_rules! count_nonzero_fn {
    ($name:ident, $copy_name:ident, $n:expr, $ffi:path, $copy_ffi:path) => {
        #[doc = concat!("Count non-zero coefficients in a ", stringify!($n), "x", stringify!($n), " block.")]
        pub fn $name(quant_coeff: &[i16; $n * $n]) -> u32 {
            unsafe { $ffi(quant_coeff.as_ptr()) as u32 }
        }

        #[doc = concat!("Copy a ", stringify!($n), "x", stringify!($n), " residual block while counting non-zero coefficients.")]
        pub fn $copy_name(residual: &[i16; $n * $n]) -> (u32, [i16; $n * $n]) {
            let mut coeff = [0i16; $n * $n];
            let count = unsafe { $copy_ffi(coeff.as_mut_ptr(), residual.as_ptr(), $n) };
            (count, coeff)
        }
    };
}

/// The four scalar quant/dequant entry points (coefficients are `i16`; the
/// FFI symbol selects the depth-specialized primitive).
macro_rules! quant_scalar {
    ($quant_fn:path, $nquant_fn:path, $deqn_fn:path, $deqs_fn:path) => {
        fn check_len(num: usize) {
            assert!(num <= 32 * 32, "too many coefficients: {num}");
            assert!(num % 16 == 0, "coefficient count must be a multiple of 16");
        }

        fn check_pair<T, U>(a: &[T], b: &[U]) {
            assert_eq!(
                a.len(),
                b.len(),
                "coefficient slices must have matching lengths"
            );
            check_len(a.len());
        }

        pub fn quant(
            coef: &[i16],
            quant_coeff: &[i32],
            q_bits: i32,
            add: i32,
        ) -> (u32, Vec<i16>, Vec<i32>) {
            check_pair(coef, quant_coeff);
            assert!(q_bits >= 8, "q_bits must be at least 8");

            let mut delta_u = vec![0i32; coef.len()];
            let mut q_coef = vec![0i16; coef.len()];
            let num_sig = unsafe {
                $quant_fn(
                    coef.as_ptr(),
                    quant_coeff.as_ptr(),
                    delta_u.as_mut_ptr(),
                    q_coef.as_mut_ptr(),
                    q_bits,
                    add,
                    coef.len() as i32,
                )
            };
            (num_sig, q_coef, delta_u)
        }

        /// Non-RDO quantization variant; x265 returns absolute quantized levels.
        pub fn nquant(coef: &[i16], quant_coeff: &[i32], q_bits: i32, add: i32) -> (u32, Vec<i16>) {
            check_pair(coef, quant_coeff);
            assert!((0..32).contains(&q_bits), "q_bits must be in 0..32");
            assert!(add >= 0, "add must be non-negative");
            assert!(
                (add as u32) < (1u32 << q_bits),
                "add must be less than 2^q_bits"
            );

            let mut q_coef = vec![0i16; coef.len()];
            let num_sig = unsafe {
                $nquant_fn(
                    coef.as_ptr(),
                    quant_coeff.as_ptr(),
                    q_coef.as_mut_ptr(),
                    q_bits,
                    add,
                    coef.len() as i32,
                )
            };
            (num_sig, q_coef)
        }

        pub fn dequant_normal(quant_coef: &[i16], scale: i32, shift: i32) -> Vec<i16> {
            check_len(quant_coef.len());
            assert!(scale < 32768, "dequant scale must be less than 32768");
            assert!((1..=10).contains(&shift), "shift must be in 1..=10");

            let mut coef = vec![0i16; quant_coef.len()];
            unsafe {
                $deqn_fn(
                    quant_coef.as_ptr(),
                    coef.as_mut_ptr(),
                    quant_coef.len() as i32,
                    scale,
                    shift,
                );
            }
            coef
        }

        pub fn dequant_scaling(
            quant_coef: &[i16],
            dequant_coef: &[i32],
            per: i32,
            shift: i32,
        ) -> Vec<i16> {
            check_pair(quant_coef, dequant_coef);

            let mut coef = vec![0i16; quant_coef.len()];
            unsafe {
                $deqs_fn(
                    quant_coef.as_ptr(),
                    dequant_coef.as_ptr(),
                    coef.as_mut_ptr(),
                    quant_coef.len() as i32,
                    per,
                    shift,
                );
            }
            coef
        }
    };
}

// --- 8-bit surface (`pixel == u8`) ----------------------------------------

/// Square-block forward/inverse DCT-II (H.265 8.6.4) and 4x4 DST, from
/// `common/dct.cpp`'s `dct4_c`/.../`idct32_c` via `setupDCTPrimitives_c`.
/// `block`/`coeffs` are row-major, stride == size.
pub mod dct {
    transform_pair!(
        forward_4x4,
        inverse_4x4,
        4,
        crate::sys::bpgprim_dct4x4,
        crate::sys::bpgprim_idct4x4
    );
    transform_pair!(
        forward_8x8,
        inverse_8x8,
        8,
        crate::sys::bpgprim_dct8x8,
        crate::sys::bpgprim_idct8x8
    );
    transform_pair!(
        forward_16x16,
        inverse_16x16,
        16,
        crate::sys::bpgprim_dct16x16,
        crate::sys::bpgprim_idct16x16
    );
    transform_pair!(
        forward_32x32,
        inverse_32x32,
        32,
        crate::sys::bpgprim_dct32x32,
        crate::sys::bpgprim_idct32x32
    );
    dst_pair!(
        forward_dst_4x4,
        inverse_dst_4x4,
        crate::sys::bpgprim_dst4x4,
        crate::sys::bpgprim_idst4x4
    );
}

/// Sum of Absolute (Transformed) Differences over square luma PU blocks
/// (`common/pixel.cpp`). 8-bit planes; `stride*` are element strides.
pub mod pixel {
    cmp_pair!(
        satd_4x4,
        sad_4x4,
        4,
        u8,
        crate::sys::bpgprim_satd_4x4,
        crate::sys::bpgprim_sad_4x4
    );
    cmp_pair!(
        satd_8x8,
        sad_8x8,
        8,
        u8,
        crate::sys::bpgprim_satd_8x8,
        crate::sys::bpgprim_sad_8x8
    );
    cmp_pair!(
        satd_16x16,
        sad_16x16,
        16,
        u8,
        crate::sys::bpgprim_satd_16x16,
        crate::sys::bpgprim_sad_16x16
    );
    cmp_pair!(
        satd_32x32,
        sad_32x32,
        32,
        u8,
        crate::sys::bpgprim_satd_32x32,
        crate::sys::bpgprim_sad_32x32
    );
}

/// DC/Planar/angular intra prediction (H.265 8.4.4.2) from
/// `common/intrapred.cpp`'s `setupIntraPrimitives_c`. See [`dc_planar`] for the
/// `refs` reference-sample layout.
pub mod intra {
    dc_planar!(
        dc_4x4,
        planar_4x4,
        angular_4x4,
        filter_refs_4x4,
        4,
        u8,
        crate::sys::bpgprim_intra_dc_4x4,
        crate::sys::bpgprim_intra_planar_4x4,
        crate::sys::bpgprim_intra_ang_4x4,
        crate::sys::bpgprim_intra_filter_4x4
    );
    dc_planar!(
        dc_8x8,
        planar_8x8,
        angular_8x8,
        filter_refs_8x8,
        8,
        u8,
        crate::sys::bpgprim_intra_dc_8x8,
        crate::sys::bpgprim_intra_planar_8x8,
        crate::sys::bpgprim_intra_ang_8x8,
        crate::sys::bpgprim_intra_filter_8x8
    );
    dc_planar!(
        dc_16x16,
        planar_16x16,
        angular_16x16,
        filter_refs_16x16,
        16,
        u8,
        crate::sys::bpgprim_intra_dc_16x16,
        crate::sys::bpgprim_intra_planar_16x16,
        crate::sys::bpgprim_intra_ang_16x16,
        crate::sys::bpgprim_intra_filter_16x16
    );
    dc_planar!(
        dc_32x32,
        planar_32x32,
        angular_32x32,
        filter_refs_32x32,
        32,
        u8,
        crate::sys::bpgprim_intra_dc_32x32,
        crate::sys::bpgprim_intra_planar_32x32,
        crate::sys::bpgprim_intra_ang_32x32,
        crate::sys::bpgprim_intra_filter_32x32
    );
}

/// Residual, reconstruction, copy, and distortion helpers from
/// `common/pixel.cpp`'s `setupPixelPrimitives_c`. Square luma CU/TU sizes,
/// row-major with stride == block size.
pub mod recon {
    recon_fns!(
        sub_ps_4x4,
        add_ps_4x4,
        copy_pp_4x4,
        copy_ps_4x4,
        copy_sp_4x4,
        copy_ss_4x4,
        sse_pp_4x4,
        sse_ss_4x4,
        var_4x4,
        ssd_s_4x4,
        sa8d_4x4,
        blockfill_s_4x4,
        transpose_4x4,
        cpy2d1d_shl_4x4,
        cpy2d1d_shr_4x4,
        cpy1d2d_shl_4x4,
        cpy1d2d_shr_4x4,
        4,
        u8,
        255i16,
        crate::sys::bpgprim_sub_ps_4x4,
        crate::sys::bpgprim_add_ps_4x4,
        crate::sys::bpgprim_copy_pp_4x4,
        crate::sys::bpgprim_copy_ps_4x4,
        crate::sys::bpgprim_copy_sp_4x4,
        crate::sys::bpgprim_copy_ss_4x4,
        crate::sys::bpgprim_sse_pp_4x4,
        crate::sys::bpgprim_sse_ss_4x4,
        crate::sys::bpgprim_var_4x4,
        crate::sys::bpgprim_ssd_s_4x4,
        crate::sys::bpgprim_sa8d_4x4,
        crate::sys::bpgprim_blockfill_s_4x4,
        crate::sys::bpgprim_transpose_4x4,
        crate::sys::bpgprim_cpy2Dto1D_shl_4x4,
        crate::sys::bpgprim_cpy2Dto1D_shr_4x4,
        crate::sys::bpgprim_cpy1Dto2D_shl_4x4,
        crate::sys::bpgprim_cpy1Dto2D_shr_4x4
    );
    recon_fns!(
        sub_ps_8x8,
        add_ps_8x8,
        copy_pp_8x8,
        copy_ps_8x8,
        copy_sp_8x8,
        copy_ss_8x8,
        sse_pp_8x8,
        sse_ss_8x8,
        var_8x8,
        ssd_s_8x8,
        sa8d_8x8,
        blockfill_s_8x8,
        transpose_8x8,
        cpy2d1d_shl_8x8,
        cpy2d1d_shr_8x8,
        cpy1d2d_shl_8x8,
        cpy1d2d_shr_8x8,
        8,
        u8,
        255i16,
        crate::sys::bpgprim_sub_ps_8x8,
        crate::sys::bpgprim_add_ps_8x8,
        crate::sys::bpgprim_copy_pp_8x8,
        crate::sys::bpgprim_copy_ps_8x8,
        crate::sys::bpgprim_copy_sp_8x8,
        crate::sys::bpgprim_copy_ss_8x8,
        crate::sys::bpgprim_sse_pp_8x8,
        crate::sys::bpgprim_sse_ss_8x8,
        crate::sys::bpgprim_var_8x8,
        crate::sys::bpgprim_ssd_s_8x8,
        crate::sys::bpgprim_sa8d_8x8,
        crate::sys::bpgprim_blockfill_s_8x8,
        crate::sys::bpgprim_transpose_8x8,
        crate::sys::bpgprim_cpy2Dto1D_shl_8x8,
        crate::sys::bpgprim_cpy2Dto1D_shr_8x8,
        crate::sys::bpgprim_cpy1Dto2D_shl_8x8,
        crate::sys::bpgprim_cpy1Dto2D_shr_8x8
    );
    recon_fns!(
        sub_ps_16x16,
        add_ps_16x16,
        copy_pp_16x16,
        copy_ps_16x16,
        copy_sp_16x16,
        copy_ss_16x16,
        sse_pp_16x16,
        sse_ss_16x16,
        var_16x16,
        ssd_s_16x16,
        sa8d_16x16,
        blockfill_s_16x16,
        transpose_16x16,
        cpy2d1d_shl_16x16,
        cpy2d1d_shr_16x16,
        cpy1d2d_shl_16x16,
        cpy1d2d_shr_16x16,
        16,
        u8,
        255i16,
        crate::sys::bpgprim_sub_ps_16x16,
        crate::sys::bpgprim_add_ps_16x16,
        crate::sys::bpgprim_copy_pp_16x16,
        crate::sys::bpgprim_copy_ps_16x16,
        crate::sys::bpgprim_copy_sp_16x16,
        crate::sys::bpgprim_copy_ss_16x16,
        crate::sys::bpgprim_sse_pp_16x16,
        crate::sys::bpgprim_sse_ss_16x16,
        crate::sys::bpgprim_var_16x16,
        crate::sys::bpgprim_ssd_s_16x16,
        crate::sys::bpgprim_sa8d_16x16,
        crate::sys::bpgprim_blockfill_s_16x16,
        crate::sys::bpgprim_transpose_16x16,
        crate::sys::bpgprim_cpy2Dto1D_shl_16x16,
        crate::sys::bpgprim_cpy2Dto1D_shr_16x16,
        crate::sys::bpgprim_cpy1Dto2D_shl_16x16,
        crate::sys::bpgprim_cpy1Dto2D_shr_16x16
    );
    recon_fns!(
        sub_ps_32x32,
        add_ps_32x32,
        copy_pp_32x32,
        copy_ps_32x32,
        copy_sp_32x32,
        copy_ss_32x32,
        sse_pp_32x32,
        sse_ss_32x32,
        var_32x32,
        ssd_s_32x32,
        sa8d_32x32,
        blockfill_s_32x32,
        transpose_32x32,
        cpy2d1d_shl_32x32,
        cpy2d1d_shr_32x32,
        cpy1d2d_shl_32x32,
        cpy1d2d_shr_32x32,
        32,
        u8,
        255i16,
        crate::sys::bpgprim_sub_ps_32x32,
        crate::sys::bpgprim_add_ps_32x32,
        crate::sys::bpgprim_copy_pp_32x32,
        crate::sys::bpgprim_copy_ps_32x32,
        crate::sys::bpgprim_copy_sp_32x32,
        crate::sys::bpgprim_copy_ss_32x32,
        crate::sys::bpgprim_sse_pp_32x32,
        crate::sys::bpgprim_sse_ss_32x32,
        crate::sys::bpgprim_var_32x32,
        crate::sys::bpgprim_ssd_s_32x32,
        crate::sys::bpgprim_sa8d_32x32,
        crate::sys::bpgprim_blockfill_s_32x32,
        crate::sys::bpgprim_transpose_32x32,
        crate::sys::bpgprim_cpy2Dto1D_shl_32x32,
        crate::sys::bpgprim_cpy2Dto1D_shr_32x32,
        crate::sys::bpgprim_cpy1Dto2D_shl_32x32,
        crate::sys::bpgprim_cpy1Dto2D_shr_32x32
    );
}

/// Quantization, dequantization, and coefficient-count helpers from
/// `common/dct.cpp`. Low-level x265 primitive calls (not the full `Quant`
/// class/RDOQ path). Coefficient slices must be multiples of 16 and at most
/// 1024 coefficients (4x4 through 32x32).
pub mod quant {
    quant_scalar!(
        crate::sys::bpgprim_quant,
        crate::sys::bpgprim_nquant,
        crate::sys::bpgprim_dequant_normal,
        crate::sys::bpgprim_dequant_scaling
    );

    count_nonzero_fn!(
        count_nonzero_4x4,
        copy_count_4x4,
        4,
        crate::sys::bpgprim_count_nonzero_4x4,
        crate::sys::bpgprim_copy_cnt_4x4
    );
    count_nonzero_fn!(
        count_nonzero_8x8,
        copy_count_8x8,
        8,
        crate::sys::bpgprim_count_nonzero_8x8,
        crate::sys::bpgprim_copy_cnt_8x8
    );
    count_nonzero_fn!(
        count_nonzero_16x16,
        copy_count_16x16,
        16,
        crate::sys::bpgprim_count_nonzero_16x16,
        crate::sys::bpgprim_copy_cnt_16x16
    );
    count_nonzero_fn!(
        count_nonzero_32x32,
        copy_count_32x32,
        32,
        crate::sys::bpgprim_count_nonzero_32x32,
        crate::sys::bpgprim_copy_cnt_32x32
    );
}

// --- 10-bit surface (`pixel == u16`) --------------------------------------

/// Full 10-bit (`pixel == u16`) mirror of the crate-root 8-bit surface,
/// built from the dual-compiled x265 namespace (`HIGH_BIT_DEPTH=1`,
/// `X265_DEPTH=10`; see `x265-primitives-sys`'s `shim/wrapper10.cpp`).
///
/// The submodules ([`bitdepth10::dct`], [`bitdepth10::pixel`],
/// [`bitdepth10::intra`], [`bitdepth10::recon`], [`bitdepth10::quant`]) expose
/// the same function names as their 8-bit counterparts, generated from the same
/// crate-level macros, with `u8` replaced by `u16` and the reconstruction
/// clip/`copy_sp` range raised to the 10-bit ceiling (1023).
pub mod bitdepth10 {
    /// 10-bit square-block DCT/IDCT/DST (coefficients remain `i16`).
    pub mod dct {
        transform_pair!(
            forward_4x4,
            inverse_4x4,
            4,
            crate::sys::bpgprim10_dct4x4,
            crate::sys::bpgprim10_idct4x4
        );
        transform_pair!(
            forward_8x8,
            inverse_8x8,
            8,
            crate::sys::bpgprim10_dct8x8,
            crate::sys::bpgprim10_idct8x8
        );
        transform_pair!(
            forward_16x16,
            inverse_16x16,
            16,
            crate::sys::bpgprim10_dct16x16,
            crate::sys::bpgprim10_idct16x16
        );
        transform_pair!(
            forward_32x32,
            inverse_32x32,
            32,
            crate::sys::bpgprim10_dct32x32,
            crate::sys::bpgprim10_idct32x32
        );
        dst_pair!(
            forward_dst_4x4,
            inverse_dst_4x4,
            crate::sys::bpgprim10_dst4x4,
            crate::sys::bpgprim10_idst4x4
        );
    }

    /// 10-bit SATD/SAD over square luma PU blocks (`pixel == u16`).
    pub mod pixel {
        cmp_pair!(
            satd_4x4,
            sad_4x4,
            4,
            u16,
            crate::sys::bpgprim10_satd_4x4,
            crate::sys::bpgprim10_sad_4x4
        );
        cmp_pair!(
            satd_8x8,
            sad_8x8,
            8,
            u16,
            crate::sys::bpgprim10_satd_8x8,
            crate::sys::bpgprim10_sad_8x8
        );
        cmp_pair!(
            satd_16x16,
            sad_16x16,
            16,
            u16,
            crate::sys::bpgprim10_satd_16x16,
            crate::sys::bpgprim10_sad_16x16
        );
        cmp_pair!(
            satd_32x32,
            sad_32x32,
            32,
            u16,
            crate::sys::bpgprim10_satd_32x32,
            crate::sys::bpgprim10_sad_32x32
        );
    }

    /// 10-bit DC/Planar/angular intra prediction (`pixel == u16`).
    pub mod intra {
        dc_planar!(
            dc_4x4,
            planar_4x4,
            angular_4x4,
            filter_refs_4x4,
            4,
            u16,
            crate::sys::bpgprim10_intra_dc_4x4,
            crate::sys::bpgprim10_intra_planar_4x4,
            crate::sys::bpgprim10_intra_ang_4x4,
            crate::sys::bpgprim10_intra_filter_4x4
        );
        dc_planar!(
            dc_8x8,
            planar_8x8,
            angular_8x8,
            filter_refs_8x8,
            8,
            u16,
            crate::sys::bpgprim10_intra_dc_8x8,
            crate::sys::bpgprim10_intra_planar_8x8,
            crate::sys::bpgprim10_intra_ang_8x8,
            crate::sys::bpgprim10_intra_filter_8x8
        );
        dc_planar!(
            dc_16x16,
            planar_16x16,
            angular_16x16,
            filter_refs_16x16,
            16,
            u16,
            crate::sys::bpgprim10_intra_dc_16x16,
            crate::sys::bpgprim10_intra_planar_16x16,
            crate::sys::bpgprim10_intra_ang_16x16,
            crate::sys::bpgprim10_intra_filter_16x16
        );
        dc_planar!(
            dc_32x32,
            planar_32x32,
            angular_32x32,
            filter_refs_32x32,
            32,
            u16,
            crate::sys::bpgprim10_intra_dc_32x32,
            crate::sys::bpgprim10_intra_planar_32x32,
            crate::sys::bpgprim10_intra_ang_32x32,
            crate::sys::bpgprim10_intra_filter_32x32
        );
    }

    /// 10-bit residual/reconstruction/copy/distortion helpers (`pixel == u16`,
    /// `copy_sp`/reconstruction clip ceiling 1023).
    pub mod recon {
        recon_fns!(
            sub_ps_4x4,
            add_ps_4x4,
            copy_pp_4x4,
            copy_ps_4x4,
            copy_sp_4x4,
            copy_ss_4x4,
            sse_pp_4x4,
            sse_ss_4x4,
            var_4x4,
            ssd_s_4x4,
            sa8d_4x4,
            blockfill_s_4x4,
            transpose_4x4,
            cpy2d1d_shl_4x4,
            cpy2d1d_shr_4x4,
            cpy1d2d_shl_4x4,
            cpy1d2d_shr_4x4,
            4,
            u16,
            1023i16,
            crate::sys::bpgprim10_sub_ps_4x4,
            crate::sys::bpgprim10_add_ps_4x4,
            crate::sys::bpgprim10_copy_pp_4x4,
            crate::sys::bpgprim10_copy_ps_4x4,
            crate::sys::bpgprim10_copy_sp_4x4,
            crate::sys::bpgprim10_copy_ss_4x4,
            crate::sys::bpgprim10_sse_pp_4x4,
            crate::sys::bpgprim10_sse_ss_4x4,
            crate::sys::bpgprim10_var_4x4,
            crate::sys::bpgprim10_ssd_s_4x4,
            crate::sys::bpgprim10_sa8d_4x4,
            crate::sys::bpgprim10_blockfill_s_4x4,
            crate::sys::bpgprim10_transpose_4x4,
            crate::sys::bpgprim10_cpy2Dto1D_shl_4x4,
            crate::sys::bpgprim10_cpy2Dto1D_shr_4x4,
            crate::sys::bpgprim10_cpy1Dto2D_shl_4x4,
            crate::sys::bpgprim10_cpy1Dto2D_shr_4x4
        );
        recon_fns!(
            sub_ps_8x8,
            add_ps_8x8,
            copy_pp_8x8,
            copy_ps_8x8,
            copy_sp_8x8,
            copy_ss_8x8,
            sse_pp_8x8,
            sse_ss_8x8,
            var_8x8,
            ssd_s_8x8,
            sa8d_8x8,
            blockfill_s_8x8,
            transpose_8x8,
            cpy2d1d_shl_8x8,
            cpy2d1d_shr_8x8,
            cpy1d2d_shl_8x8,
            cpy1d2d_shr_8x8,
            8,
            u16,
            1023i16,
            crate::sys::bpgprim10_sub_ps_8x8,
            crate::sys::bpgprim10_add_ps_8x8,
            crate::sys::bpgprim10_copy_pp_8x8,
            crate::sys::bpgprim10_copy_ps_8x8,
            crate::sys::bpgprim10_copy_sp_8x8,
            crate::sys::bpgprim10_copy_ss_8x8,
            crate::sys::bpgprim10_sse_pp_8x8,
            crate::sys::bpgprim10_sse_ss_8x8,
            crate::sys::bpgprim10_var_8x8,
            crate::sys::bpgprim10_ssd_s_8x8,
            crate::sys::bpgprim10_sa8d_8x8,
            crate::sys::bpgprim10_blockfill_s_8x8,
            crate::sys::bpgprim10_transpose_8x8,
            crate::sys::bpgprim10_cpy2Dto1D_shl_8x8,
            crate::sys::bpgprim10_cpy2Dto1D_shr_8x8,
            crate::sys::bpgprim10_cpy1Dto2D_shl_8x8,
            crate::sys::bpgprim10_cpy1Dto2D_shr_8x8
        );
        recon_fns!(
            sub_ps_16x16,
            add_ps_16x16,
            copy_pp_16x16,
            copy_ps_16x16,
            copy_sp_16x16,
            copy_ss_16x16,
            sse_pp_16x16,
            sse_ss_16x16,
            var_16x16,
            ssd_s_16x16,
            sa8d_16x16,
            blockfill_s_16x16,
            transpose_16x16,
            cpy2d1d_shl_16x16,
            cpy2d1d_shr_16x16,
            cpy1d2d_shl_16x16,
            cpy1d2d_shr_16x16,
            16,
            u16,
            1023i16,
            crate::sys::bpgprim10_sub_ps_16x16,
            crate::sys::bpgprim10_add_ps_16x16,
            crate::sys::bpgprim10_copy_pp_16x16,
            crate::sys::bpgprim10_copy_ps_16x16,
            crate::sys::bpgprim10_copy_sp_16x16,
            crate::sys::bpgprim10_copy_ss_16x16,
            crate::sys::bpgprim10_sse_pp_16x16,
            crate::sys::bpgprim10_sse_ss_16x16,
            crate::sys::bpgprim10_var_16x16,
            crate::sys::bpgprim10_ssd_s_16x16,
            crate::sys::bpgprim10_sa8d_16x16,
            crate::sys::bpgprim10_blockfill_s_16x16,
            crate::sys::bpgprim10_transpose_16x16,
            crate::sys::bpgprim10_cpy2Dto1D_shl_16x16,
            crate::sys::bpgprim10_cpy2Dto1D_shr_16x16,
            crate::sys::bpgprim10_cpy1Dto2D_shl_16x16,
            crate::sys::bpgprim10_cpy1Dto2D_shr_16x16
        );
        recon_fns!(
            sub_ps_32x32,
            add_ps_32x32,
            copy_pp_32x32,
            copy_ps_32x32,
            copy_sp_32x32,
            copy_ss_32x32,
            sse_pp_32x32,
            sse_ss_32x32,
            var_32x32,
            ssd_s_32x32,
            sa8d_32x32,
            blockfill_s_32x32,
            transpose_32x32,
            cpy2d1d_shl_32x32,
            cpy2d1d_shr_32x32,
            cpy1d2d_shl_32x32,
            cpy1d2d_shr_32x32,
            32,
            u16,
            1023i16,
            crate::sys::bpgprim10_sub_ps_32x32,
            crate::sys::bpgprim10_add_ps_32x32,
            crate::sys::bpgprim10_copy_pp_32x32,
            crate::sys::bpgprim10_copy_ps_32x32,
            crate::sys::bpgprim10_copy_sp_32x32,
            crate::sys::bpgprim10_copy_ss_32x32,
            crate::sys::bpgprim10_sse_pp_32x32,
            crate::sys::bpgprim10_sse_ss_32x32,
            crate::sys::bpgprim10_var_32x32,
            crate::sys::bpgprim10_ssd_s_32x32,
            crate::sys::bpgprim10_sa8d_32x32,
            crate::sys::bpgprim10_blockfill_s_32x32,
            crate::sys::bpgprim10_transpose_32x32,
            crate::sys::bpgprim10_cpy2Dto1D_shl_32x32,
            crate::sys::bpgprim10_cpy2Dto1D_shr_32x32,
            crate::sys::bpgprim10_cpy1Dto2D_shl_32x32,
            crate::sys::bpgprim10_cpy1Dto2D_shr_32x32
        );
    }

    /// 10-bit quant/dequant/coefficient-count helpers (coefficients `i16`).
    pub mod quant {
        quant_scalar!(
            crate::sys::bpgprim10_quant,
            crate::sys::bpgprim10_nquant,
            crate::sys::bpgprim10_dequant_normal,
            crate::sys::bpgprim10_dequant_scaling
        );

        count_nonzero_fn!(
            count_nonzero_4x4,
            copy_count_4x4,
            4,
            crate::sys::bpgprim10_count_nonzero_4x4,
            crate::sys::bpgprim10_copy_cnt_4x4
        );
        count_nonzero_fn!(
            count_nonzero_8x8,
            copy_count_8x8,
            8,
            crate::sys::bpgprim10_count_nonzero_8x8,
            crate::sys::bpgprim10_copy_cnt_8x8
        );
        count_nonzero_fn!(
            count_nonzero_16x16,
            copy_count_16x16,
            16,
            crate::sys::bpgprim10_count_nonzero_16x16,
            crate::sys::bpgprim10_copy_cnt_16x16
        );
        count_nonzero_fn!(
            count_nonzero_32x32,
            copy_count_32x32,
            32,
            crate::sys::bpgprim10_count_nonzero_32x32,
            crate::sys::bpgprim10_copy_cnt_32x32
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC-only absolute check (not just round-trip): for a constant
    /// residual block of value `v`, H.265's 4x4 DCT-II (`g_t4` = `{64,64,
    /// 64,64}` in row 0) concentrates all energy in the DC coefficient.
    /// First pass (shift=1): each row's E0=E1=2v, O0=O1=0, so
    /// coef[0..4) = (64*2v+64*2v+1)>>1 = 128v, rest 0. Second pass
    /// (shift=8): dst[0] = (64*256v+64*256v+128)>>8 = 128v, rest 0.
    #[test]
    fn dct_4x4_constant_block_is_pure_dc() {
        let block = [8i16; 16];
        let coeffs = dct::forward_4x4(&block);
        assert_eq!(coeffs[0], 128 * 8);
        assert!(coeffs[1..].iter().all(|&c| c == 0), "{coeffs:?}");
    }

    #[test]
    fn dct_4x4_round_trip() {
        let block: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let coeffs = dct::forward_4x4(&block);
        let back = dct::inverse_4x4(&coeffs);
        assert_eq!(back, block);
    }

    #[test]
    fn dst_4x4_zero_vector_is_zero() {
        let block = [0i16; 16];
        let coeffs = dct::forward_dst_4x4(&block);
        assert_eq!(coeffs, [0i16; 16]);
        assert_eq!(dct::inverse_dst_4x4(&coeffs), block);
    }

    #[test]
    fn dct_8x8_round_trip() {
        let mut block = [0i16; 64];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i as i16 * 3) % 17 - 8;
        }
        let coeffs = dct::forward_8x8(&block);
        let back = dct::inverse_8x8(&coeffs);
        assert_eq!(back, block);
    }

    #[test]
    fn dct_32x32_round_trip() {
        let mut block = [0i16; 32 * 32];
        for (i, b) in block.iter_mut().enumerate() {
            *b = ((i as i32 * 7) % 31 - 15) as i16;
        }
        let coeffs = dct::forward_32x32(&block);
        let back = dct::inverse_32x32(&coeffs);
        assert_eq!(back, block);
    }

    /// Absolute check: for two flat 4x4 blocks differing by a constant 10,
    /// SAD = 16 * 10 = 160. The 4x4 Hadamard transform of a constant
    /// difference concentrates all energy in DC (= SAD), and x265's
    /// `satd4` returns `(sum + 1) >> 1` of the Hadamard coefficients, so
    /// SATD = (160 + 1) >> 1 = 80.
    #[test]
    fn satd_sad_4x4_constant_difference() {
        let a = [100u8; 16];
        let b = [110u8; 16];
        assert_eq!(pixel::sad_4x4(&a, 4, &b, 4), 160);
        assert_eq!(pixel::satd_4x4(&a, 4, &b, 4), 80);
    }

    #[test]
    fn satd_sad_zero_for_identical_blocks() {
        let a = [42u8; 64];
        assert_eq!(pixel::sad_8x8(&a, 8, &a, 8), 0);
        assert_eq!(pixel::satd_8x8(&a, 8, &a, 8), 0);
    }

    /// Absolute check: DC/Planar prediction from uniform reference samples
    /// must reproduce that uniform value exactly (H.265 8.4.4.2.5 / 8.4.4.2.4
    /// both reduce to the constant when above == left == corner).
    #[test]
    fn intra_dc_and_planar_uniform_refs() {
        let refs = [50u8; 4 * 4 + 1];
        assert_eq!(intra::dc_4x4(&refs, false), [50u8; 16]);
        assert_eq!(intra::planar_4x4(&refs), [50u8; 16]);
        assert_eq!(intra::angular_4x4(&refs, 10, false), [50u8; 16]);

        let refs16 = [200u8; 4 * 16 + 1];
        assert_eq!(intra::dc_16x16(&refs16, false), [200u8; 256]);
        assert_eq!(intra::planar_16x16(&refs16), [200u8; 256]);
        assert_eq!(intra::angular_16x16(&refs16, 26, true), [200u8; 256]);
        assert_eq!(intra::filter_refs_16x16(&refs16), refs16);
    }

    #[test]
    fn residual_and_reconstruction_4x4_absolute() {
        let orig = [100u8; 16];
        let pred = [90u8; 16];
        let residual = recon::sub_ps_4x4(&orig, &pred);
        assert_eq!(residual, [10i16; 16]);
        assert_eq!(recon::add_ps_4x4(&pred, &residual), orig);
    }

    #[test]
    fn reconstruction_clips_to_8bit_range() {
        let pred = [250u8; 16];
        let residual = [20i16; 16];
        assert_eq!(recon::add_ps_4x4(&pred, &residual), [255u8; 16]);

        let pred = [5u8; 16];
        let residual = [-20i16; 16];
        assert_eq!(recon::add_ps_4x4(&pred, &residual), [0u8; 16]);
    }

    #[test]
    fn copy_helpers_preserve_values() {
        let pix = [17u8; 64];
        assert_eq!(recon::copy_pp_8x8(&pix), pix);
        assert_eq!(recon::copy_ps_8x8(&pix), [17i16; 64]);
        assert_eq!(recon::copy_sp_8x8(&[17i16; 64]), pix);
        assert_eq!(recon::copy_ss_8x8(&[17i16; 64]), [17i16; 64]);
    }

    #[test]
    fn sse_absolute_vectors() {
        let a = [10u8; 16 * 16];
        let b = [13u8; 16 * 16];
        assert_eq!(recon::sse_pp_16x16(&a, &b), 16 * 16 * 9);

        let a = [10i16; 16 * 16];
        let b = [13i16; 16 * 16];
        assert_eq!(recon::sse_ss_16x16(&a, &b), 16 * 16 * 9);
    }

    #[test]
    fn variance_ssd_sa8d_and_blockfill_absolute_vectors() {
        let flat = [3u8; 16];
        assert_eq!(recon::var_4x4(&flat), ((16 * 9u64) << 32) | (16 * 3u64));

        assert_eq!(recon::ssd_s_8x8(&[3i16; 64]), 64 * 9);
        assert_eq!(recon::sa8d_4x4(&[100u8; 16], &[110u8; 16]), 80);
        assert_eq!(recon::blockfill_s_8x8(-7), [-7i16; 64]);
    }

    #[test]
    fn transpose_and_shift_copy_absolute_vectors() {
        let src: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(
            recon::transpose_4x4(&src),
            [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
        );

        let coeff: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 9, 10, 11, 12];
        let shifted = [2, 4, 6, 8, 10, 12, 14, 16, -2, -4, -6, -8, 18, 20, 22, 24];
        assert_eq!(recon::cpy2d1d_shl_4x4(&coeff, 1), shifted);
        assert_eq!(recon::cpy1d2d_shl_4x4(&coeff, 1), shifted);
        assert_eq!(recon::cpy2d1d_shr_4x4(&[3i16; 16], 1), [2i16; 16]);
        assert_eq!(recon::cpy1d2d_shr_4x4(&[3i16; 16], 1), [2i16; 16]);
    }

    #[test]
    fn quant_absolute_vector() {
        let coef = [
            16i16, -16, 8, -8, 0, 1, -1, 32, 4, -4, 2, -2, 3, -3, 64, -64,
        ];
        let q = [256i32; 16];
        let (num_sig, q_coef, delta_u) = quant::quant(&coef, &q, 8, 0);
        assert_eq!(num_sig, 15);
        assert_eq!(q_coef, coef);
        assert!(delta_u.iter().all(|&v| v == 0));

        let (num_sig, q_abs) = quant::nquant(&coef, &q, 8, 0);
        assert_eq!(num_sig, 15);
        assert_eq!(q_abs, coef.iter().map(|v| v.abs()).collect::<Vec<_>>());
    }

    #[test]
    fn dequant_absolute_vectors() {
        let q = [1i16, -1, 2, -2, 0, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8];
        assert_eq!(quant::dequant_normal(&q, 64, 6), q);
        assert_eq!(quant::dequant_scaling(&q, &[16i32; 16], 0, 0), q);
    }

    #[test]
    fn count_nonzero_absolute_vector() {
        let coeff = [1i16, 0, -2, 0, 3, 0, 0, 4, 0, 0, 0, 0, -5, 0, 6, 0];
        assert_eq!(quant::count_nonzero_4x4(&coeff), 6);

        let (count, copied) = quant::copy_count_4x4(&coeff);
        assert_eq!(count, 6);
        assert_eq!(copied, coeff);
    }

    /// The full 10-bit surface mirrors the 8-bit one: same structural
    /// invariants, with the value ceiling raised to 1023. The 4x4 DCT DC
    /// coefficient differs from 8-bit (32*v vs 128*v) because the forward
    /// transform's first-pass shift scales with bit depth.
    #[test]
    fn bitdepth10_dct_and_compare_absolute_vectors() {
        let coeffs = bitdepth10::dct::forward_4x4(&[8i16; 16]);
        assert_eq!(coeffs[0], 32 * 8);
        assert!(coeffs[1..].iter().all(|&v| v == 0), "{coeffs:?}");
        assert_eq!(bitdepth10::dct::inverse_4x4(&coeffs), [8i16; 16]);

        // Larger sizes survive a forward+inverse pass. Unlike the 8-bit case
        // (whose per-pass shifts happen to reconstruct small integers
        // exactly), x265's 10-bit transform pair rounds intermediate results
        // and is only near-lossless, so this checks closeness, not equality.
        let mut block = [0i16; 32 * 32];
        for (i, b) in block.iter_mut().enumerate() {
            *b = ((i as i32 * 7) % 31 - 15) as i16;
        }
        let back = bitdepth10::dct::inverse_32x32(&bitdepth10::dct::forward_32x32(&block));
        assert!(
            back.iter()
                .zip(block.iter())
                .all(|(&r, &o)| (r - o).abs() <= 2),
            "10-bit 32x32 DCT round trip drifted by more than 2"
        );

        let a = [100u16; 16];
        let b = [110u16; 16];
        assert_eq!(bitdepth10::pixel::sad_4x4(&a, 4, &b, 4), 160);
        assert_eq!(bitdepth10::pixel::satd_4x4(&a, 4, &b, 4), 80);
        assert_eq!(
            bitdepth10::pixel::sad_16x16(&[7u16; 256], 16, &[7u16; 256], 16),
            0
        );
    }

    #[test]
    fn bitdepth10_intra_recon_quant_absolute_vectors() {
        // intra prediction from uniform 10-bit references
        assert_eq!(
            bitdepth10::intra::dc_4x4(&[512u16; 17], false),
            [512u16; 16]
        );
        assert_eq!(bitdepth10::intra::planar_8x8(&[300u16; 33]), [300u16; 64]);
        assert_eq!(
            bitdepth10::intra::angular_16x16(&[600u16; 65], 26, true),
            [600u16; 256]
        );
        assert_eq!(
            bitdepth10::intra::filter_refs_8x8(&[300u16; 33]),
            [300u16; 33]
        );

        // residual / reconstruction, clipping to the 10-bit ceiling
        let orig = [700u16; 16];
        let pred = [500u16; 16];
        let residual = bitdepth10::recon::sub_ps_4x4(&orig, &pred);
        assert_eq!(residual, [200i16; 16]);
        assert_eq!(bitdepth10::recon::add_ps_4x4(&pred, &residual), orig);
        assert_eq!(
            bitdepth10::recon::add_ps_4x4(&[1000u16; 16], &[100i16; 16]),
            [1023u16; 16]
        );

        // copy / distortion helpers
        assert_eq!(bitdepth10::recon::copy_sp_8x8(&[900i16; 64]), [900u16; 64]);
        assert_eq!(
            bitdepth10::recon::sse_pp_8x8(&[10u16; 64], &[13u16; 64]),
            64 * 9
        );
        assert_eq!(
            bitdepth10::recon::sa8d_4x4(&[100u16; 16], &[110u16; 16]),
            80
        );

        // quant / count helpers (coefficients are i16, identical API)
        let coeff = [1i16, 0, -2, 0, 3, 0, 0, 4, 0, 0, 0, 0, -5, 0, 6, 0];
        assert_eq!(bitdepth10::quant::count_nonzero_4x4(&coeff), 6);
        let q = [1i16, -1, 2, -2, 0, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8];
        assert_eq!(bitdepth10::quant::dequant_normal(&q, 64, 6), q);
    }
}
