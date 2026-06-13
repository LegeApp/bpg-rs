/* Shared `extern "C"` body over a subset of x265's scalar (`_c`)
 * `EncoderPrimitives` dispatch table -- see `setupCPrimitives` in
 * `x265_4.1/source/common/primitives.cpp`.
 *
 * This header is included by `wrapper.cpp` (8-bit, `pixel == uint8_t`) and
 * `wrapper10.cpp` (10-bit, `pixel == uint16_t`). The two translation units
 * compile the *same* function bodies against x265's bit-depth-specialized
 * scalar primitives; only the exported symbol prefix differs. The includer
 * must, before including this header:
 *
 *   - `#define BPG_PREFIX bpgprim_`   (8-bit) or `bpgprim10_` (10-bit), and
 *   - provide a `const EncoderPrimitives &cprims();` accessor in scope.
 *
 * `ENABLE_ASSEMBLY` is off (matching `bpg-x265-sys`'s default), so these are
 * the same C reference implementations x265 falls back to on platforms
 * without a matching SIMD backend -- the "oracle" Phase 4's acceptance
 * criterion (primitive tests match x265 for fixed vectors across supported
 * bit depths) compares against.
 *
 * The `pixel` type is whatever `common.h` defines for the active
 * `HIGH_BIT_DEPTH`/`X265_DEPTH`, so the int16_t coefficient/transform
 * helpers are bit-depth independent in signature while the pixel-typed
 * helpers (SATD/SAD, intra prediction, residual/reconstruction) follow the
 * compile's `pixel` width.
 */
#ifndef BPG_PREFIX
#error "wrapper_impl.h requires BPG_PREFIX to be defined before inclusion"
#endif

/* Indirection so BPG_PREFIX is macro-expanded before the `##` paste. */
#define BPG_CAT_(a, b) a ## b
#define BPG_CAT(a, b) BPG_CAT_(a, b)
#define FN(name) BPG_CAT(BPG_PREFIX, name)

extern "C" {

// --- DCT / IDCT (common/dct.cpp) ---------------------------------------

void FN(dct4x4)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_4x4].dct(src, dst, 4);
}

void FN(idct4x4)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_4x4].idct(src, dst, 4);
}

void FN(dct8x8)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_8x8].dct(src, dst, 8);
}

void FN(idct8x8)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_8x8].idct(src, dst, 8);
}

void FN(dct16x16)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_16x16].dct(src, dst, 16);
}

void FN(idct16x16)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_16x16].idct(src, dst, 16);
}

void FN(dct32x32)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_32x32].dct(src, dst, 32);
}

void FN(idct32x32)(const int16_t *src, int16_t *dst)
{
    cprims().cu[BLOCK_32x32].idct(src, dst, 32);
}

void FN(dst4x4)(const int16_t *src, int16_t *dst)
{
    cprims().dst4x4(src, dst, 4);
}

void FN(idst4x4)(const int16_t *src, int16_t *dst)
{
    cprims().idst4x4(src, dst, 4);
}

// --- SATD / SAD (common/pixel.cpp), square luma PU sizes ----------------

#define BPGPRIM_CMP(W, H, PU) \
int FN(satd_ ## W ## x ## H)(const pixel *pix1, intptr_t stride1, const pixel *pix2, intptr_t stride2) \
{ \
    return (int)cprims().pu[PU].satd(pix1, stride1, pix2, stride2); \
} \
int FN(sad_ ## W ## x ## H)(const pixel *pix1, intptr_t stride1, const pixel *pix2, intptr_t stride2) \
{ \
    return (int)cprims().pu[PU].sad(pix1, stride1, pix2, stride2); \
}

BPGPRIM_CMP(4, 4, LUMA_4x4)
BPGPRIM_CMP(8, 8, LUMA_8x8)
BPGPRIM_CMP(16, 16, LUMA_16x16)
BPGPRIM_CMP(32, 32, LUMA_32x32)

#undef BPGPRIM_CMP

// --- Intra prediction (common/intrapred.cpp) -----------------------------
//
// `srcPix` layout matches x265's reference-sample buffer: srcPix[0] is the
// above-left corner, srcPix[1 .. width] are the "above" samples, and
// srcPix[2*width+1 .. 3*width] are the "left" samples (length 4*width+1
// total; the last quadrant is unused by DC/Planar). `bFilter` is the
// strong-intra-smoothing flag (H.265 8.4.4.2.3); pass 0 to disable.

#define BPGPRIM_INTRA(W, H, BLOCK) \
void FN(intra_dc_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const pixel *srcPix, int bFilter) \
{ \
    cprims().cu[BLOCK].intra_pred[DC_IDX](dst, dstStride, srcPix, 1, bFilter); \
} \
void FN(intra_planar_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const pixel *srcPix) \
{ \
    cprims().cu[BLOCK].intra_pred[PLANAR_IDX](dst, dstStride, srcPix, 0, 0); \
} \
void FN(intra_ang_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const pixel *srcPix, int dirMode, int bFilter) \
{ \
    cprims().cu[BLOCK].intra_pred[dirMode](dst, dstStride, srcPix, dirMode, bFilter); \
} \
void FN(intra_filter_ ## W ## x ## H)(const pixel *samples, pixel *filtered) \
{ \
    cprims().cu[BLOCK].intra_filter(samples, filtered); \
}

BPGPRIM_INTRA(4, 4, BLOCK_4x4)
BPGPRIM_INTRA(8, 8, BLOCK_8x8)
BPGPRIM_INTRA(16, 16, BLOCK_16x16)
BPGPRIM_INTRA(32, 32, BLOCK_32x32)

#undef BPGPRIM_INTRA

// --- Residual / reconstruction / copy / SSE helpers (common/pixel.cpp) ---

#define BPGPRIM_CU_HELPERS(W, H, BLOCK, PU) \
void FN(sub_ps_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, const pixel *src0, const pixel *src1, intptr_t stride0, intptr_t stride1) \
{ \
    cprims().cu[BLOCK].sub_ps(dst, dstStride, src0, src1, stride0, stride1); \
} \
void FN(add_ps_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const pixel *pred, const int16_t *residual, intptr_t predStride, intptr_t residualStride) \
{ \
    cprims().cu[BLOCK].add_ps[NONALIGNED](dst, dstStride, pred, residual, predStride, residualStride); \
} \
void FN(copy_pp_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const pixel *src, intptr_t srcStride) \
{ \
    cprims().pu[PU].copy_pp(dst, dstStride, src, srcStride); \
} \
void FN(copy_ps_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, const pixel *src, intptr_t srcStride) \
{ \
    cprims().cu[BLOCK].copy_ps(dst, dstStride, src, srcStride); \
} \
void FN(copy_sp_ ## W ## x ## H)(pixel *dst, intptr_t dstStride, const int16_t *src, intptr_t srcStride) \
{ \
    cprims().cu[BLOCK].copy_sp(dst, dstStride, src, srcStride); \
} \
void FN(copy_ss_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, const int16_t *src, intptr_t srcStride) \
{ \
    cprims().cu[BLOCK].copy_ss(dst, dstStride, src, srcStride); \
} \
uint64_t FN(sse_pp_ ## W ## x ## H)(const pixel *pix1, intptr_t stride1, const pixel *pix2, intptr_t stride2) \
{ \
    return (uint64_t)cprims().cu[BLOCK].sse_pp(pix1, stride1, pix2, stride2); \
} \
uint64_t FN(sse_ss_ ## W ## x ## H)(const int16_t *pix1, intptr_t stride1, const int16_t *pix2, intptr_t stride2) \
{ \
    return (uint64_t)cprims().cu[BLOCK].sse_ss(pix1, stride1, pix2, stride2); \
} \
uint64_t FN(var_ ## W ## x ## H)(const pixel *pix, intptr_t stride) \
{ \
    return cprims().cu[BLOCK].var(pix, stride); \
} \
uint64_t FN(ssd_s_ ## W ## x ## H)(const int16_t *pix, intptr_t stride) \
{ \
    return (uint64_t)cprims().cu[BLOCK].ssd_s[NONALIGNED](pix, stride); \
} \
int FN(sa8d_ ## W ## x ## H)(const pixel *pix1, intptr_t stride1, const pixel *pix2, intptr_t stride2) \
{ \
    return cprims().cu[BLOCK].sa8d(pix1, stride1, pix2, stride2); \
} \
void FN(blockfill_s_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, int16_t val) \
{ \
    cprims().cu[BLOCK].blockfill_s[NONALIGNED](dst, dstStride, val); \
} \
void FN(transpose_ ## W ## x ## H)(pixel *dst, const pixel *src, intptr_t srcStride) \
{ \
    cprims().cu[BLOCK].transpose(dst, src, srcStride); \
} \
void FN(cpy2Dto1D_shl_ ## W ## x ## H)(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift) \
{ \
    ALIGN_VAR_32(int16_t, alignedDst[32 * 32]); \
    cprims().cu[BLOCK].cpy2Dto1D_shl(alignedDst, src, srcStride, shift); \
    memcpy(dst, alignedDst, (W * H) * sizeof(int16_t)); \
} \
void FN(cpy2Dto1D_shr_ ## W ## x ## H)(int16_t *dst, const int16_t *src, intptr_t srcStride, int shift) \
{ \
    ALIGN_VAR_32(int16_t, alignedDst[32 * 32]); \
    cprims().cu[BLOCK].cpy2Dto1D_shr(alignedDst, src, srcStride, shift); \
    memcpy(dst, alignedDst, (W * H) * sizeof(int16_t)); \
} \
void FN(cpy1Dto2D_shl_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, const int16_t *src, int shift) \
{ \
    ALIGN_VAR_32(int16_t, alignedSrc[32 * 32]); \
    memcpy(alignedSrc, src, (W * H) * sizeof(int16_t)); \
    cprims().cu[BLOCK].cpy1Dto2D_shl[NONALIGNED](dst, alignedSrc, dstStride, shift); \
} \
void FN(cpy1Dto2D_shr_ ## W ## x ## H)(int16_t *dst, intptr_t dstStride, const int16_t *src, int shift) \
{ \
    ALIGN_VAR_32(int16_t, alignedSrc[32 * 32]); \
    memcpy(alignedSrc, src, (W * H) * sizeof(int16_t)); \
    cprims().cu[BLOCK].cpy1Dto2D_shr(dst, alignedSrc, dstStride, shift); \
}

BPGPRIM_CU_HELPERS(4, 4, BLOCK_4x4, LUMA_4x4)
BPGPRIM_CU_HELPERS(8, 8, BLOCK_8x8, LUMA_8x8)
BPGPRIM_CU_HELPERS(16, 16, BLOCK_16x16, LUMA_16x16)
BPGPRIM_CU_HELPERS(32, 32, BLOCK_32x32, LUMA_32x32)

#undef BPGPRIM_CU_HELPERS

// --- Quant / dequant / coefficient helpers (common/dct.cpp) -------------

uint32_t FN(quant)(const int16_t *coef, const int32_t *quantCoeff, int32_t *deltaU, int16_t *qCoef, int qBits, int add, int numCoeff)
{
    return cprims().quant(coef, quantCoeff, deltaU, qCoef, qBits, add, numCoeff);
}

uint32_t FN(nquant)(const int16_t *coef, const int32_t *quantCoeff, int16_t *qCoef, int qBits, int add, int numCoeff)
{
    ALIGN_VAR_32(int32_t, alignedQuant[32 * 32]);
    memcpy(alignedQuant, quantCoeff, numCoeff * sizeof(int32_t));
    return cprims().nquant(coef, alignedQuant, qCoef, qBits, add, numCoeff);
}

void FN(dequant_normal)(const int16_t *quantCoef, int16_t *coef, int num, int scale, int shift)
{
    ALIGN_VAR_32(int16_t, alignedCoef[32 * 32]);
    cprims().dequant_normal(quantCoef, alignedCoef, num, scale, shift);
    memcpy(coef, alignedCoef, num * sizeof(int16_t));
}

void FN(dequant_scaling)(const int16_t *quantCoef, const int32_t *dequantCoef, int16_t *coef, int num, int per, int shift)
{
    ALIGN_VAR_32(int16_t, alignedCoef[32 * 32]);
    cprims().dequant_scaling(quantCoef, dequantCoef, alignedCoef, num, per, shift);
    memcpy(coef, alignedCoef, num * sizeof(int16_t));
}

#define BPGPRIM_COUNT_NONZERO(W, H, BLOCK) \
int FN(count_nonzero_ ## W ## x ## H)(const int16_t *quantCoeff) \
{ \
    ALIGN_VAR_32(int16_t, alignedCoeff[32 * 32]); \
    memcpy(alignedCoeff, quantCoeff, (W * H) * sizeof(int16_t)); \
    return cprims().cu[BLOCK].count_nonzero(alignedCoeff); \
}

BPGPRIM_COUNT_NONZERO(4, 4, BLOCK_4x4)
BPGPRIM_COUNT_NONZERO(8, 8, BLOCK_8x8)
BPGPRIM_COUNT_NONZERO(16, 16, BLOCK_16x16)
BPGPRIM_COUNT_NONZERO(32, 32, BLOCK_32x32)

#undef BPGPRIM_COUNT_NONZERO

#define BPGPRIM_COPY_COUNT(W, H, BLOCK) \
uint32_t FN(copy_cnt_ ## W ## x ## H)(int16_t *coeff, const int16_t *residual, intptr_t residualStride) \
{ \
    ALIGN_VAR_32(int16_t, alignedCoeff[32 * 32]); \
    uint32_t count = cprims().cu[BLOCK].copy_cnt(alignedCoeff, residual, residualStride); \
    memcpy(coeff, alignedCoeff, (W * H) * sizeof(int16_t)); \
    return count; \
}

BPGPRIM_COPY_COUNT(4, 4, BLOCK_4x4)
BPGPRIM_COPY_COUNT(8, 8, BLOCK_8x8)
BPGPRIM_COPY_COUNT(16, 16, BLOCK_16x16)
BPGPRIM_COPY_COUNT(32, 32, BLOCK_32x32)

#undef BPGPRIM_COPY_COUNT

} // extern "C"

#undef FN
#undef BPG_CAT
#undef BPG_CAT_
#undef BPG_PREFIX
