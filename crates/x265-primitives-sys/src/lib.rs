//! Raw `extern "C"` declarations for `shim/wrapper.cpp`'s boundary over a
//! subset of x265's scalar `EncoderPrimitives` (see `build.rs` and
//! `shim/wrapper.cpp` for what is compiled and why). 8-bit only
//! (`pixel == u8`).
//!
//! Safe wrappers with documented preconditions live in the `x265-primitives`
//! crate; this crate is the unsafe FFI layer only.
#![allow(non_camel_case_types)]

use std::os::raw::c_int;

extern "C" {
    // --- DCT / IDCT ---------------------------------------------------
    pub fn bpgprim_dct4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim_idct4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim_dct8x8(src: *const i16, dst: *mut i16);
    pub fn bpgprim_idct8x8(src: *const i16, dst: *mut i16);
    pub fn bpgprim_dct16x16(src: *const i16, dst: *mut i16);
    pub fn bpgprim_idct16x16(src: *const i16, dst: *mut i16);
    pub fn bpgprim_dct32x32(src: *const i16, dst: *mut i16);
    pub fn bpgprim_idct32x32(src: *const i16, dst: *mut i16);
    pub fn bpgprim_dst4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim_idst4x4(src: *const i16, dst: *mut i16);

    // --- SATD / SAD (square luma PU sizes) ----------------------------
    pub fn bpgprim_satd_4x4(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_sad_4x4(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_satd_8x8(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_sad_8x8(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_satd_16x16(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_sad_16x16(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_satd_32x32(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_sad_32x32(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;

    // --- Intra prediction (DC / Planar) --------------------------------
    pub fn bpgprim_intra_dc_4x4(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_planar_4x4(dst: *mut u8, dst_stride: isize, src_pix: *const u8);
    pub fn bpgprim_intra_dc_8x8(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_planar_8x8(dst: *mut u8, dst_stride: isize, src_pix: *const u8);
    pub fn bpgprim_intra_dc_16x16(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_planar_16x16(dst: *mut u8, dst_stride: isize, src_pix: *const u8);
    pub fn bpgprim_intra_dc_32x32(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_planar_32x32(dst: *mut u8, dst_stride: isize, src_pix: *const u8);
    pub fn bpgprim_intra_ang_4x4(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_ang_8x8(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_ang_16x16(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_ang_32x32(
        dst: *mut u8,
        dst_stride: isize,
        src_pix: *const u8,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim_intra_filter_4x4(samples: *const u8, filtered: *mut u8);
    pub fn bpgprim_intra_filter_8x8(samples: *const u8, filtered: *mut u8);
    pub fn bpgprim_intra_filter_16x16(samples: *const u8, filtered: *mut u8);
    pub fn bpgprim_intra_filter_32x32(samples: *const u8, filtered: *mut u8);

    // --- Residual / reconstruction / copy / SSE helpers ----------------
    pub fn bpgprim_sub_ps_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u8,
        src1: *const u8,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim_add_ps_4x4(
        dst: *mut u8,
        dst_stride: isize,
        pred: *const u8,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim_copy_pp_4x4(dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize);
    pub fn bpgprim_copy_ps_4x4(dst: *mut i16, dst_stride: isize, src: *const u8, src_stride: isize);
    pub fn bpgprim_copy_sp_4x4(dst: *mut u8, dst_stride: isize, src: *const i16, src_stride: isize);
    pub fn bpgprim_copy_ss_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_sse_pp_4x4(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_sse_ss_4x4(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_var_4x4(pix: *const u8, stride: isize) -> u64;
    pub fn bpgprim_ssd_s_4x4(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim_sa8d_4x4(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_blockfill_s_4x4(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim_transpose_4x4(dst: *mut u8, src: *const u8, src_stride: isize);
    pub fn bpgprim_cpy2Dto1D_shl_4x4(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy2Dto1D_shr_4x4(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shl_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shr_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim_sub_ps_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u8,
        src1: *const u8,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim_add_ps_8x8(
        dst: *mut u8,
        dst_stride: isize,
        pred: *const u8,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim_copy_pp_8x8(dst: *mut u8, dst_stride: isize, src: *const u8, src_stride: isize);
    pub fn bpgprim_copy_ps_8x8(dst: *mut i16, dst_stride: isize, src: *const u8, src_stride: isize);
    pub fn bpgprim_copy_sp_8x8(dst: *mut u8, dst_stride: isize, src: *const i16, src_stride: isize);
    pub fn bpgprim_copy_ss_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_sse_pp_8x8(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_sse_ss_8x8(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_var_8x8(pix: *const u8, stride: isize) -> u64;
    pub fn bpgprim_ssd_s_8x8(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim_sa8d_8x8(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_blockfill_s_8x8(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim_transpose_8x8(dst: *mut u8, src: *const u8, src_stride: isize);
    pub fn bpgprim_cpy2Dto1D_shl_8x8(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy2Dto1D_shr_8x8(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shl_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shr_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim_sub_ps_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u8,
        src1: *const u8,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim_add_ps_16x16(
        dst: *mut u8,
        dst_stride: isize,
        pred: *const u8,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim_copy_pp_16x16(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
    );
    pub fn bpgprim_copy_ps_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
    );
    pub fn bpgprim_copy_sp_16x16(
        dst: *mut u8,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_copy_ss_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_sse_pp_16x16(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_sse_ss_16x16(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_var_16x16(pix: *const u8, stride: isize) -> u64;
    pub fn bpgprim_ssd_s_16x16(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim_sa8d_16x16(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_blockfill_s_16x16(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim_transpose_16x16(dst: *mut u8, src: *const u8, src_stride: isize);
    pub fn bpgprim_cpy2Dto1D_shl_16x16(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy2Dto1D_shr_16x16(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shl_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shr_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim_sub_ps_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u8,
        src1: *const u8,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim_add_ps_32x32(
        dst: *mut u8,
        dst_stride: isize,
        pred: *const u8,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim_copy_pp_32x32(
        dst: *mut u8,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
    );
    pub fn bpgprim_copy_ps_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u8,
        src_stride: isize,
    );
    pub fn bpgprim_copy_sp_32x32(
        dst: *mut u8,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_copy_ss_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim_sse_pp_32x32(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_sse_ss_32x32(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim_var_32x32(pix: *const u8, stride: isize) -> u64;
    pub fn bpgprim_ssd_s_32x32(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim_sa8d_32x32(
        pix1: *const u8,
        stride1: isize,
        pix2: *const u8,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim_blockfill_s_32x32(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim_transpose_32x32(dst: *mut u8, src: *const u8, src_stride: isize);
    pub fn bpgprim_cpy2Dto1D_shl_32x32(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy2Dto1D_shr_32x32(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shl_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim_cpy1Dto2D_shr_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    // --- Quant / dequant / coefficient helpers -------------------------
    pub fn bpgprim_quant(
        coef: *const i16,
        quant_coeff: *const i32,
        delta_u: *mut i32,
        q_coef: *mut i16,
        q_bits: c_int,
        add: c_int,
        num_coeff: c_int,
    ) -> u32;
    pub fn bpgprim_nquant(
        coef: *const i16,
        quant_coeff: *const i32,
        q_coef: *mut i16,
        q_bits: c_int,
        add: c_int,
        num_coeff: c_int,
    ) -> u32;
    pub fn bpgprim_dequant_normal(
        quant_coef: *const i16,
        coef: *mut i16,
        num: c_int,
        scale: c_int,
        shift: c_int,
    );
    pub fn bpgprim_dequant_scaling(
        quant_coef: *const i16,
        dequant_coef: *const i32,
        coef: *mut i16,
        num: c_int,
        per: c_int,
        shift: c_int,
    );
    pub fn bpgprim_count_nonzero_4x4(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim_count_nonzero_8x8(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim_count_nonzero_16x16(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim_count_nonzero_32x32(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim_copy_cnt_4x4(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim_copy_cnt_8x8(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim_copy_cnt_16x16(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim_copy_cnt_32x32(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;

    // --- 10-bit scalar primitive surface (`pixel == u16`) ---------------
    // Full parity with the 8-bit surface above; same signatures with
    // `pixel == u16`. Built from the same `wrapper_impl.h` bodies compiled
    // under HIGH_BIT_DEPTH=1 (see `shim/wrapper10.cpp`).
    pub fn bpgprim10_dct4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_idct4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_dct8x8(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_idct8x8(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_dct16x16(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_idct16x16(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_dct32x32(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_idct32x32(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_dst4x4(src: *const i16, dst: *mut i16);
    pub fn bpgprim10_idst4x4(src: *const i16, dst: *mut i16);

    // --- SATD / SAD (square luma PU sizes) ----------------------------
    pub fn bpgprim10_satd_4x4(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_sad_4x4(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_satd_8x8(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_sad_8x8(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_satd_16x16(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_sad_16x16(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_satd_32x32(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_sad_32x32(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;

    // --- Intra prediction (DC / Planar) --------------------------------
    pub fn bpgprim10_intra_dc_4x4(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_planar_4x4(dst: *mut u16, dst_stride: isize, src_pix: *const u16);
    pub fn bpgprim10_intra_dc_8x8(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_planar_8x8(dst: *mut u16, dst_stride: isize, src_pix: *const u16);
    pub fn bpgprim10_intra_dc_16x16(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_planar_16x16(dst: *mut u16, dst_stride: isize, src_pix: *const u16);
    pub fn bpgprim10_intra_dc_32x32(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_planar_32x32(dst: *mut u16, dst_stride: isize, src_pix: *const u16);
    pub fn bpgprim10_intra_ang_4x4(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_ang_8x8(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_ang_16x16(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_ang_32x32(
        dst: *mut u16,
        dst_stride: isize,
        src_pix: *const u16,
        dir_mode: c_int,
        b_filter: c_int,
    );
    pub fn bpgprim10_intra_filter_4x4(samples: *const u16, filtered: *mut u16);
    pub fn bpgprim10_intra_filter_8x8(samples: *const u16, filtered: *mut u16);
    pub fn bpgprim10_intra_filter_16x16(samples: *const u16, filtered: *mut u16);
    pub fn bpgprim10_intra_filter_32x32(samples: *const u16, filtered: *mut u16);

    // --- Residual / reconstruction / copy / SSE helpers ----------------
    pub fn bpgprim10_sub_ps_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u16,
        src1: *const u16,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim10_add_ps_4x4(
        dst: *mut u16,
        dst_stride: isize,
        pred: *const u16,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim10_copy_pp_4x4(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ps_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_sp_4x4(
        dst: *mut u16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ss_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_sse_pp_4x4(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_sse_ss_4x4(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_var_4x4(pix: *const u16, stride: isize) -> u64;
    pub fn bpgprim10_ssd_s_4x4(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim10_sa8d_4x4(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_blockfill_s_4x4(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim10_transpose_4x4(dst: *mut u16, src: *const u16, src_stride: isize);
    pub fn bpgprim10_cpy2Dto1D_shl_4x4(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy2Dto1D_shr_4x4(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shl_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shr_4x4(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim10_sub_ps_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u16,
        src1: *const u16,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim10_add_ps_8x8(
        dst: *mut u16,
        dst_stride: isize,
        pred: *const u16,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim10_copy_pp_8x8(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ps_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_sp_8x8(
        dst: *mut u16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ss_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_sse_pp_8x8(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_sse_ss_8x8(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_var_8x8(pix: *const u16, stride: isize) -> u64;
    pub fn bpgprim10_ssd_s_8x8(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim10_sa8d_8x8(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_blockfill_s_8x8(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim10_transpose_8x8(dst: *mut u16, src: *const u16, src_stride: isize);
    pub fn bpgprim10_cpy2Dto1D_shl_8x8(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy2Dto1D_shr_8x8(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shl_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shr_8x8(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim10_sub_ps_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u16,
        src1: *const u16,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim10_add_ps_16x16(
        dst: *mut u16,
        dst_stride: isize,
        pred: *const u16,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim10_copy_pp_16x16(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ps_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_sp_16x16(
        dst: *mut u16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ss_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_sse_pp_16x16(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_sse_ss_16x16(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_var_16x16(pix: *const u16, stride: isize) -> u64;
    pub fn bpgprim10_ssd_s_16x16(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim10_sa8d_16x16(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_blockfill_s_16x16(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim10_transpose_16x16(dst: *mut u16, src: *const u16, src_stride: isize);
    pub fn bpgprim10_cpy2Dto1D_shl_16x16(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy2Dto1D_shr_16x16(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shl_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shr_16x16(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    pub fn bpgprim10_sub_ps_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src0: *const u16,
        src1: *const u16,
        stride0: isize,
        stride1: isize,
    );
    pub fn bpgprim10_add_ps_32x32(
        dst: *mut u16,
        dst_stride: isize,
        pred: *const u16,
        residual: *const i16,
        pred_stride: isize,
        residual_stride: isize,
    );
    pub fn bpgprim10_copy_pp_32x32(
        dst: *mut u16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ps_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const u16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_sp_32x32(
        dst: *mut u16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_copy_ss_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        src_stride: isize,
    );
    pub fn bpgprim10_sse_pp_32x32(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_sse_ss_32x32(
        pix1: *const i16,
        stride1: isize,
        pix2: *const i16,
        stride2: isize,
    ) -> u64;
    pub fn bpgprim10_var_32x32(pix: *const u16, stride: isize) -> u64;
    pub fn bpgprim10_ssd_s_32x32(pix: *const i16, stride: isize) -> u64;
    pub fn bpgprim10_sa8d_32x32(
        pix1: *const u16,
        stride1: isize,
        pix2: *const u16,
        stride2: isize,
    ) -> c_int;
    pub fn bpgprim10_blockfill_s_32x32(dst: *mut i16, dst_stride: isize, val: i16);
    pub fn bpgprim10_transpose_32x32(dst: *mut u16, src: *const u16, src_stride: isize);
    pub fn bpgprim10_cpy2Dto1D_shl_32x32(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy2Dto1D_shr_32x32(
        dst: *mut i16,
        src: *const i16,
        src_stride: isize,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shl_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );
    pub fn bpgprim10_cpy1Dto2D_shr_32x32(
        dst: *mut i16,
        dst_stride: isize,
        src: *const i16,
        shift: c_int,
    );

    // --- Quant / dequant / coefficient helpers -------------------------
    pub fn bpgprim10_quant(
        coef: *const i16,
        quant_coeff: *const i32,
        delta_u: *mut i32,
        q_coef: *mut i16,
        q_bits: c_int,
        add: c_int,
        num_coeff: c_int,
    ) -> u32;
    pub fn bpgprim10_nquant(
        coef: *const i16,
        quant_coeff: *const i32,
        q_coef: *mut i16,
        q_bits: c_int,
        add: c_int,
        num_coeff: c_int,
    ) -> u32;
    pub fn bpgprim10_dequant_normal(
        quant_coef: *const i16,
        coef: *mut i16,
        num: c_int,
        scale: c_int,
        shift: c_int,
    );
    pub fn bpgprim10_dequant_scaling(
        quant_coef: *const i16,
        dequant_coef: *const i32,
        coef: *mut i16,
        num: c_int,
        per: c_int,
        shift: c_int,
    );
    pub fn bpgprim10_count_nonzero_4x4(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim10_count_nonzero_8x8(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim10_count_nonzero_16x16(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim10_count_nonzero_32x32(quant_coeff: *const i16) -> c_int;
    pub fn bpgprim10_copy_cnt_4x4(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim10_copy_cnt_8x8(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim10_copy_cnt_16x16(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
    pub fn bpgprim10_copy_cnt_32x32(
        coeff: *mut i16,
        residual: *const i16,
        residual_stride: isize,
    ) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct4x4_link_probe_calls_x265_scalar_primitive() {
        let src = [8i16; 16];
        let mut dst = [0i16; 16];

        unsafe { bpgprim_dct4x4(src.as_ptr(), dst.as_mut_ptr()) };

        assert_eq!(dst[0], 128 * 8);
        assert!(dst[1..].iter().all(|&v| v == 0), "{dst:?}");
    }

    #[test]
    fn ten_bit_link_probe_calls_x265_scalar_primitives() {
        let src = [8i16; 16];
        let mut dct = [0i16; 16];
        unsafe { bpgprim10_dct4x4(src.as_ptr(), dct.as_mut_ptr()) };
        assert_eq!(dct[0], 32 * 8);
        assert!(dct[1..].iter().all(|&v| v == 0), "{dct:?}");

        let a = [100u16; 16];
        let b = [110u16; 16];
        assert_eq!(
            unsafe { bpgprim10_sad_4x4(a.as_ptr(), 4, b.as_ptr(), 4) },
            160
        );
        assert_eq!(
            unsafe { bpgprim10_satd_4x4(a.as_ptr(), 4, b.as_ptr(), 4) },
            80
        );

        let refs = [512u16; 17];
        let mut pred = [0u16; 16];
        unsafe { bpgprim10_intra_dc_4x4(pred.as_mut_ptr(), 4, refs.as_ptr(), 0) };
        assert_eq!(pred, [512u16; 16]);

        let residual = [600i16; 16];
        let mut recon = [0u16; 16];
        unsafe {
            bpgprim10_add_ps_4x4(
                recon.as_mut_ptr(),
                4,
                pred.as_ptr(),
                residual.as_ptr(),
                4,
                4,
            )
        };
        assert_eq!(recon, [1023u16; 16]);
    }
}
