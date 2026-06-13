# x265 x86 Kernel Guide for a Rust BPG/HEVC Still Encoder

This document summarizes the x86 assembly files in the uploaded `x86.zip`, explains what each kernel family does in x265, and gives an integration strategy for Rust code that may not preserve x265's original C++ primitive-table structure.

Scope: the archive contains 27 NASM/YASM-style `.asm` files from x265's x86 primitive layer. These are not ordinary standalone assembly files: most depend on `x86inc.asm`, `x86util.asm`, global constants from `const-a.asm`, and x265 naming/CPU-feature macros.

## 1. Executive summary

For a Rust still-image BPG encoder, do not integrate all kernels at once. The x265 assembly layer contains both still-intra-relevant kernels and many video/inter-prediction kernels. The highest-value order is:

1. **Pixel cost kernels**: SATD, SA8D, SAD, SSD. These accelerate mode decision.
2. **Intra prediction kernels**: planar, DC, angular prediction, all-angles prediction.
3. **Transform and quantization kernels**: DCT/DST, inverse DCT/DST, quant/RDOQ helpers.
4. **Residual/reconstruction utility kernels**: subtract, add, block copy/fill, nonzero count.
5. **Loop filters**: SAO and deblock, once those features are enabled.
6. **Inter/motion kernels**: mostly irrelevant for still-image-only BPG; defer unless implementing BPG animation or inter-frame HEVC.

The current Rust encoder's 296s full-image encode is unlikely to be fixed mainly by assembly. It first needs analysis-architecture cleanup: fewer full-RD trials, limited RDOQ, cached bit estimates, no repeated frame snapshotting, scratch buffers instead of `Vec` churn. After that, the assembly kernels below become effective.

## 2. Licensing note

Most files carry x265/x264 GPL headers, except `x86inc.asm`, which is ISC-style. If these kernels are linked into the Rust encoder, the resulting binary is likely governed by x265-compatible GPL/commercial licensing constraints. Do not treat the assembly layer as license-neutral just because it is called from Rust.

## 3. x265 assembly conventions

### 3.1 File format

The files use NASM/YASM syntax plus x264/x265 macro layers:

- `x86inc.asm`: macro abstraction for C ABI, register naming, CPU suffixes, symbol mangling, Windows/SysV differences.
- `x86util.asm`: SIMD helper macros and codec constants such as pixel size, DCT coefficient size, strides, and vector helpers.
- `const-a.asm`: global constant vectors referenced by many kernels through `cextern`.

### 3.2 Function names

Most exported functions are declared with `cglobal`. In `x86inc.asm`, `cglobal foo` is converted into a mangled symbol of roughly:

```text
<private_prefix>_<foo><cpu_suffix>
```

The default `private_prefix` is `X265_NS`, so the final names depend on build-time defines used by x265. For Rust integration, choose one stable namespace such as:

```text
BPGX265_8_pixel_satd_16x16_sse4
BPGX265_10_dct16_avx2
```

or preserve x265's exact names and call them through generated bindings. The first option is usually easier if you want a clean Rust primitive table.

### 3.3 CPU suffixes

The files use `INIT_XMM`, `INIT_YMM`, `INIT_ZMM`, and similar macros to emit variants for CPU features:

- MMX/MMX2: legacy, usually avoid in a new Rust build unless needed for parity.
- SSE2 / SSSE3 / SSE4: broad compatibility baseline.
- AVX / AVX2: high value on modern x86_64.
- AVX512: useful only with careful dispatch and benchmarking; can hurt clocks on some CPUs.
- XOP: legacy AMD extension; usually not worth carrying forward in a new design.

### 3.4 Pixel type suffixes

x265 kernel names often encode source/destination sample type:

- `pp`: pixel → pixel.
- `ps`: pixel → short/intermediate.
- `sp`: short/intermediate → pixel.
- `ss`: short/intermediate → short/intermediate.

In 8-bit builds, `pixel` is `u8`. In high-bit-depth builds, `pixel` is usually `u16` representing 10/12-bit samples. `short` is usually an intermediate signed 16-bit buffer for interpolation/residual stages.

For Rust, model these explicitly:

```rust
pub type Pel8 = u8;
pub type Pel16 = u16;
pub type I16 = i16;
```

Do not hide all variants behind a generic `T` at the FFI boundary. Keep the FFI signatures concrete, then wrap them in safe Rust traits.

## 4. Recommended Rust primitive architecture

Use a primitive table rather than direct calls scattered across the encoder.

```rust
pub struct PixelPrimitives8 {
    pub satd_4x4: Satd8Fn,
    pub satd_8x8: Satd8Fn,
    pub sad_16x16: Sad8Fn,
    pub ssd_16x16: Ssd8Fn,
}

pub struct TransformPrimitives8 {
    pub dct4: Dct8Fn,
    pub dst4: Dct8Fn,
    pub dct8: Dct8Fn,
    pub dct16: Dct8Fn,
    pub dct32: Dct8Fn,
    pub idct4: Idct8Fn,
    pub idst4: Idct8Fn,
    pub idct8: Idct8Fn,
    pub idct16: Idct8Fn,
    pub idct32: Idct8Fn,
}

pub struct IntraPrimitives8 {
    pub pred_planar: IntraPred8Fn,
    pub pred_dc: IntraPred8Fn,
    pub pred_ang: [Option<IntraPred8Fn>; 35],
    pub pred_all_angs: Option<AllAngsPred8Fn>,
}

pub struct Primitives8 {
    pub pixel: PixelPrimitives8,
    pub transform: TransformPrimitives8,
    pub intra: IntraPrimitives8,
    pub block: BlockPrimitives8,
    pub loop_filter: LoopFilterPrimitives8,
}
```

Example FFI wrapper style:

```rust
pub type Satd8Fn = unsafe extern "C" fn(
    pix1: *const u8,
    stride1: isize,
    pix2: *const u8,
    stride2: isize,
) -> i32;

pub fn satd_8x8(p: &Primitives8, a: Plane8<'_>, b: Plane8<'_>) -> u32 {
    assert!(a.width >= 8 && a.height >= 8);
    assert!(b.width >= 8 && b.height >= 8);
    unsafe {
        (p.pixel.satd_8x8)(
            a.ptr,
            a.stride as isize,
            b.ptr,
            b.stride as isize,
        ) as u32
    }
}
```

The `unsafe` should be isolated in one crate, e.g. `bpg-x265-asm`, with safe wrappers in `bpg-primitives`.

## 5. Build integration strategy

### 5.1 Best initial path: compile NASM/YASM objects from `build.rs`

Use NASM/YASM through a build script. The `nasm-rs` crate can work, but a manual `Command::new("nasm")` path is often simpler because x265's macro defines are non-trivial.

Pseudo-build script:

```rust
use std::{env, path::PathBuf, process::Command};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = PathBuf::from("asm/x86");

    let objects = [
        "const-a.asm",
        "pixel-a.asm",
        "sad-a.asm",
        "ssd-a.asm",
        "intrapred8.asm",
        "intrapred8_allangs.asm",
        "dct8.asm",
        "pixel-util8.asm",
        "pixeladd8.asm",
        "blockcopy8.asm",
    ];

    for file in objects {
        let obj = out_dir.join(file.replace(".asm", ".o"));
        let status = Command::new("nasm")
            .arg("-f").arg("elf64")
            .arg("-I").arg(&src)
            .arg("-DARCH_X86_64=1")
            .arg("-DHIGH_BIT_DEPTH=0")
            .arg("-DBIT_DEPTH=8")
            .arg("-DX265_NS=bpgx265_8")
            .arg(src.join(file))
            .arg("-o").arg(&obj)
            .status()
            .unwrap();
        assert!(status.success(), "nasm failed for {file}");
        println!("cargo:rustc-link-arg={}", obj.display());
    }
}
```

You will need platform-specific output format handling:

- Linux: `elf64`
- macOS: `macho64`
- Windows MSVC: `win64`
- Windows GNU: `win64`, linked through the GNU toolchain or a static archive.

### 5.2 Prefer Rust CPU detection over `cpu-a.asm`

For a Rust project, `std::is_x86_feature_detected!` is usually a cleaner dispatch layer than porting x265's CPUID machinery first:

```rust
pub fn select_primitives_8() -> &'static Primitives8 {
    if is_x86_feature_detected!("avx2") {
        &PRIMS_8_AVX2
    } else if is_x86_feature_detected!("sse4.1") {
        &PRIMS_8_SSE4
    } else if is_x86_feature_detected!("ssse3") {
        &PRIMS_8_SSSE3
    } else {
        &PRIMS_8_SCALAR
    }
}
```

Keep `cpu-a.asm` available only if exact x265 feature detection is needed later.

### 5.3 Do not use `global_asm!` for this archive initially

Rust `global_asm!` is not a good fit for these files because they rely heavily on NASM macros, x86inc, and generated symbols. Compile to object files instead.

## 6. File-by-file kernel guide

### 6.1 `const-a.asm` — global SIMD constants

**What it is:** constant vectors such as zero, one, rounding offsets, shuffle masks, masks, powers of two, sign vectors, etc.

**Used by:** almost every other assembly file via `cextern` references.

**Still-image relevance:** required if using x265 assembly directly.

**Rust integration:** compile and link this object before any dependent objects. No direct Rust API is normally needed. Treat it as data-only support.

---

### 6.2 `x86inc.asm` — assembly ABI/macro framework

**What it is:** the x264/x265 assembly macro layer. It abstracts C ABI differences, register naming, function prologues, stack alignment, symbol mangling, and CPU suffixes.

**Used by:** every real kernel file.

**Still-image relevance:** required for building the assembly as-is.

**Rust integration:** keep as include-only. Do not bind to it directly.

---

### 6.3 `x86util.asm` — codec SIMD macro helpers

**What it is:** helper macros and definitions for pixel size, DCT coefficient size, vector operations, transpose helpers, packed arithmetic, and x265-specific assumptions.

**Used by:** most real kernel files.

**Still-image relevance:** required for building the assembly as-is.

**Rust integration:** include-only. Be careful with `HIGH_BIT_DEPTH`, `BIT_DEPTH`, `SIZEOF_PIXEL`, and `SIZEOF_DCTCOEF` defines.

---

### 6.4 `cpu-a.asm` — CPU feature and barrier helpers

**Kernel families:** `cpu_cpuid`, `cpu_xgetbv`, `cpu_cpuid_test`, `cpu_emms`, `cpu_sfence`, `safe_intel_cpu_indicator_init`, `stack_align`.

**What it is:** low-level CPUID/XGETBV and old MMX cleanup/barrier helpers.

**Used for:** x265's own CPU dispatch and SIMD feature probing.

**Still-image relevance:** low. Rust can use `is_x86_feature_detected!` instead.

**Rust integration:** defer. If used, expose only a tiny unsafe FFI module. Prefer Rust feature detection unless matching x265's exact CPU-mask behavior is required.

---

### 6.5 `blockcopy8.asm` — block copy/fill and type conversion

**Kernel families:** `blockcopy_pp`, `blockcopy_sp`, `blockcopy_ps`, `blockcopy_ss`, `blockfill_s`, 1D/2D copy with shifts, count/copy helpers.

**What it is:** optimized rectangular block movement and conversion among pixel and intermediate buffers.

**Used for:** copying prediction/reconstruction blocks, moving residual/intermediate data, filling short buffers, converting between pixel and short domains.

**Still-image relevance:** medium to high. Useful for removing Rust-side loop overhead in block copy/fill and residual buffer setup, but not the first bottleneck unless profiling says copy dominates.

**Rust integration:** expose only a subset initially:

```rust
pub type BlockCopyPp8 = unsafe extern "C" fn(
    dst: *mut u8,
    dst_stride: isize,
    src: *const u8,
    src_stride: isize,
);

pub type BlockCopyPs8 = unsafe extern "C" fn(
    dst: *mut i16,
    dst_stride: isize,
    src: *const u8,
    src_stride: isize,
);
```

Do not wire every size. Add only sizes used by your encoder: 4, 8, 16, 32, and maybe 64.

---

### 6.6 `pixeladd8.asm` — add residual to prediction

**Kernel families:** `pixel_add_ps`, `pixel_add_ps_aligned`.

**What it is:** adds signed residual samples to a pixel prediction block and clips/stores to pixel output.

**Used for:** reconstruction after inverse transform.

**Still-image relevance:** high once the encoder is allocation-free. Your current Rust reconstruction loop is a direct candidate for this.

**Rust integration:** wrap as `reconstruct_block_8()` and `reconstruct_block_10()` style functions. Validate exact clipping against decoder output.

---

### 6.7 `pixel-util8.asm` — residual, quant, transpose, variance, utility kernels

**Kernel families:** `pixel_sub_ps`, `getResidual4/8/16/32`, `quant`, `nquant`, `dequant_normal`, `dequant_scaling`, `count_nonzero`, `transpose4/8/16/32/64`, `pixel_var`, `weight_pp`, `weight_sp`, `costCoeffNxN`, `scanPosLast`, `pixel_ssim_*`.

**What it is:** utility kernels around the transform/quant path and metric path.

**Used for:** residual generation, coefficient quant/dequant, nonzero counting, block transposition, variance/SSIM, and coefficient-cost helpers.

**Still-image relevance:** very high. This is one of the most important files for your current Rust encoder because it can replace repeated scalar residual/quant/count loops.

**Rust integration priority:**

1. `pixel_sub_ps` / `getResidual*` for residual generation.
2. `count_nonzero` for CBF decisions.
3. `quant` / `nquant` after your scalar quant path is verified.
4. transpose helpers only if your transform path needs them.
5. SSIM helpers only for metrics, not encoding.

---

### 6.8 `dct8.asm` — transform, inverse transform, RDOQ helpers

**Kernel families:** `dct4`, `dst4`, `dct8`, `dct16`, `dct32`, `idct4`, `idst4`, `idct8`, `idct16`, `idct32`, `denoise_dct`, `nonPsyRdoQuant*`, `psyRdoQuant*`.

**What it is:** forward and inverse integer transform kernels plus quantization/RDOQ helpers.

**Used for:** HEVC transform coding and reconstruction. The 4x4 intra luma path uses DST; other blocks use DCT.

**Still-image relevance:** extremely high. This maps directly to your `transform.rs` path.

**Rust integration:** replace the current transform wrapper with a primitive table:

```rust
pub type Dct8Fn = unsafe extern "C" fn(
    src: *const i16,
    dst: *mut i16,
    src_stride: isize,
);
```

Exact signatures must be verified from x265's C++ primitive declarations. The assembly argument order is usually visible in the `cglobal` declaration names, but x265's C++ headers are the final authority.

**Recommended order:** DCT/DST first, IDCT/IDST second, RDOQ helpers later. Your current RDOQ is algorithmically too expensive; do not simply call assembly RDOQ before redesigning the RDOQ pass.

---

### 6.9 `intrapred8.asm` — 8-bit intra prediction

**Kernel families:** `intra_pred_planar4/8/16/32`, `intra_pred_dc4/8/16/32`, `intra_pred_ang<size>_<mode>`, `intra_filter`.

**What it is:** 8-bit planar, DC, angular, and reference-filtering prediction kernels.

**Used for:** luma/chroma intra prediction during rough mode search, full RD trials, and final reconstruction.

**Still-image relevance:** extremely high. This is likely one of the most important kernel groups for your current slow encode, especially because your rough mode search tests many intra modes.

**Rust integration:** first create an intra-prediction API that writes to a scratch buffer instead of mutating `DecodedFrame`. Then wire these kernels behind it.

```rust
pub type IntraPred8Fn = unsafe extern "C" fn(
    dst: *mut u8,
    dst_stride: isize,
    refs: *const u8,
    dir_mode: i32,
    filter: i32,
);
```

The exact x265 signature may differ. Keep a scalar Rust implementation as the reference and run bit-exact tests for every size/mode.

---

### 6.10 `intrapred16.asm` — high-bit-depth intra prediction

**Kernel families:** high-bit-depth versions of planar, DC, angular, and reference filtering.

**What it is:** intra prediction for 10/12-bit builds using 16-bit pixel samples.

**Used for:** same as `intrapred8.asm`, but for high-bit-depth pixels.

**Still-image relevance:** high if your BPG encoder supports 10-bit and eventually 12/14-bit.

**Rust integration:** use separate `Primitives10` / `Primitives12` tables. Do not force 8-bit and 10-bit through the same FFI type.

---

### 6.11 `intrapred8_allangs.asm` — all-angular prediction batch kernel

**Kernel families:** `all_angs_pred_WxH`.

**What it is:** computes many/all angular intra predictions for a block size in one optimized batch.

**Used for:** rough intra mode search.

**Still-image relevance:** very high for your current encoder, because rough mode decision over 35 modes is expensive.

**Rust integration:** this is a strong candidate to integrate early. Instead of calling `predict_intra` 35 times, call an all-angles kernel to fill a prediction buffer bank, then compute SATD/SAD costs.

Design:

```rust
pub struct AllAngPredictions8<'a> {
    pub modes: &'a mut [[u8; 32 * 32]; 35],
}
```

For memory efficiency, consider generating only candidate groups at first, not all blocks for all modes globally.

---

### 6.12 `pixel-a.asm` — SATD, SA8D, psycho-visual costs, SSIM-ish helpers

**Kernel families:** `pixel_satd`, `pixel_sa8d`, `psyCost_pp`, `psyCost_ss`, `calc_satd`, `ssimDist`, `normFact`, shift helpers.

**What it is:** mode-decision distortion/cost kernels, especially Hadamard SATD and SA8D.

**Used for:** rough intra/inter mode decisions, psy-RD, and quality metrics.

**Still-image relevance:** extremely high. SATD/SA8D is central to fast intra mode pruning.

**Rust integration:** wire SATD before DCT if rough mode search is a bottleneck. Use safe wrappers per block size.

```rust
pub type Satd8Fn = unsafe extern "C" fn(
    pix1: *const u8,
    stride1: isize,
    pix2: *const u8,
    stride2: isize,
) -> i32;
```

Use `psyCost_*` later, only after your baseline RD is stable.

---

### 6.13 `sad-a.asm` — 8-bit SAD and multi-candidate SAD

**Kernel families:** `pixel_sad`, `pixel_sad_x3`, `pixel_sad_x4`, cacheline variants.

**What it is:** Sum of Absolute Differences for one or multiple candidate blocks.

**Used for:** motion estimation and cheap mode screening. `x3`/`x4` compare against multiple candidate blocks in one call.

**Still-image relevance:** medium. For still intra-only, SATD is usually more valuable than SAD, but SAD is useful for coarse first-pass mode pruning or content classification.

**Rust integration:** optional after SATD. The `x3/x4` variants are useful if you batch several candidate predictions.

---

### 6.14 `sad16-a.asm` — high-bit-depth SAD

**Kernel families:** high-bit-depth `pixel_sad`, `pixel_sad_x3`, `pixel_sad_x4`, `pixel_vsad`.

**What it is:** SAD for 10/12-bit pixel samples.

**Still-image relevance:** medium. Same role as `sad-a.asm`, but for high-bit-depth images.

**Rust integration:** defer until 10-bit high-speed mode search matters.

---

### 6.15 `ssd-a.asm` — SSD kernels

**Kernel families:** `pixel_ssd`, `pixel_ssd_ss`, `pixel_ssd_sp`, `pixel_ssd_s`.

**What it is:** Sum of Squared Differences between pixel/short buffers.

**Used for:** final distortion calculation, RD cost, quality metrics, sometimes analysis.

**Still-image relevance:** high. Your Rust encoder computes distortion many times. SSD kernels are a direct replacement for scalar loops.

**Rust integration:** expose a small set of sizes: 4x4, 8x8, 16x16, 32x32, 64x64. Use `ss`/`sp` only if your internal scratch format needs it.

---

### 6.16 `pixel-32.asm` — 32-bit/legacy pixel metric helpers

**Kernel families:** `pixel_sa8d_*_internal`, `intra_sa8d_x3`, `pixel_ssim_*_core`.

**What it is:** legacy or special-case metric helpers, including SA8D and SSIM core pieces.

**Still-image relevance:** low to medium. Useful only if matching x265's exact SA8D/SSIM behavior or specialized candidate evaluation.

**Rust integration:** defer.

---

### 6.17 `loopfilter.asm` — SAO and deblocking kernels

**Kernel families:** `saoCuOrgE0/E1/E2/E3`, `saoCuOrgB0`, `saoCuStats*`, `calSign`, `pelFilterLumaStrong_H/V`, `pelFilterChroma_H/V`.

**What it is:** Sample Adaptive Offset application/statistics and deblocking pixel filters.

**Used for:** in-loop filtering and SAO analysis/application.

**Still-image relevance:** high once SAO/deblock are implemented. Not useful before those features are enabled.

**Rust integration:** split into two groups:

1. SAO stats/search helpers.
2. SAO/deblock application helpers.

Keep the scalar Rust filter as reference. Loop filters are easy to get subtly wrong at boundaries; add exact decoder-equivalence tests before enabling asm.

---

### 6.18 `ipfilter8.asm` — 8-bit interpolation filters

**Kernel families:** `interp_8tap_vert`, `interp_8tap_horiz`, `interp_8tap_hv`, `interp_4tap_*`, `filterPixelToShort`.

**What it is:** luma/chroma interpolation for fractional-pel motion compensation, converting pixels to intermediate short buffers.

**Used for:** inter prediction / motion compensation in video.

**Still-image relevance:** low for single-image BPG. Not needed for all-intra still encoding.

**Rust integration:** defer unless implementing BPG animation/inter-frame HEVC. Do not spend early time here.

---

### 6.19 `ipfilter16.asm` — high-bit-depth interpolation filters

**What it is:** high-bit-depth version of `ipfilter8.asm`.

**Still-image relevance:** low for still-only encoding.

**Rust integration:** defer.

---

### 6.20 `h-ipfilter8.asm` — 8-bit horizontal interpolation filters

**Kernel families:** horizontal 4-tap/8-tap pixel/intermediate filters.

**Used for:** motion compensation.

**Still-image relevance:** low.

**Rust integration:** defer.

---

### 6.21 `h-ipfilter16.asm` — high-bit-depth horizontal interpolation filters

**What it is:** high-bit-depth horizontal interpolation.

**Still-image relevance:** low.

**Rust integration:** defer.

---

### 6.22 `h4-ipfilter16.asm` — high-bit-depth 4-tap horizontal chroma filters

**Kernel families:** `chroma_filter_pp`, `chroma_filter_ps`, 4-tap horizontal chroma interpolation.

**Used for:** chroma fractional interpolation in inter prediction.

**Still-image relevance:** low for all-intra still encoding. Not the same as BPG chroma downsampling/upsampling.

**Rust integration:** defer.

---

### 6.23 `v4-ipfilter8.asm` — 8-bit 4-tap vertical chroma filters

**Kernel families:** `interp_4tap_vert_*` in `pp`, `ps`, `sp`, `ss` variants.

**Used for:** chroma vertical fractional interpolation.

**Still-image relevance:** low for still-only encoding.

**Rust integration:** defer.

---

### 6.24 `v4-ipfilter16.asm` — high-bit-depth 4-tap vertical chroma filters

**What it is:** high-bit-depth vertical chroma interpolation.

**Still-image relevance:** low for still-only encoding.

**Rust integration:** defer.

---

### 6.25 `mc-a.asm` — motion compensation average/weight kernels

**Kernel families:** `pixel_avg`, `addAvg`, `pixel_avg_weight`, `mc_weight`, `mc_offset`, prefetch helpers.

**What it is:** weighted prediction and averaging of inter-prediction blocks.

**Used for:** P/B-frame motion compensation, weighted prediction, bidirectional prediction.

**Still-image relevance:** very low for single-image all-intra BPG.

**Rust integration:** defer unless implementing BPG animation with inter frames.

---

### 6.26 `mc-a2.asm` — frame/motion/cutree helpers

**Kernel families:** plane copy/deinterleave/interleave, `frame_init_lowres_core`, `frame_subsample_luma`, `mbtree_propagate_cost`, `cutree_fix8_pack/unpack`, integral init helpers, aligned memcpy/memzero.

**What it is:** video-analysis support: lowres frame generation, CU-tree/MB-tree propagation, plane formatting, and memory utilities.

**Used for:** lookahead, rate control, motion analysis, frame preprocessing.

**Still-image relevance:** mostly low. Some memory helpers may be useful, but the analysis concepts are video-oriented.

**Rust integration:** do not integrate early. If you build still-image spatial preanalysis, write it natively in Rust rather than dragging in CU-tree machinery.

---

### 6.27 `seaintegral.asm` — SEA integral-image helpers

**Kernel families:** `integral4v/8v/12v/16v/24v/32v`, `integral4h/8h/12h/16h/24h/32h`.

**What it is:** integral sums for successive elimination algorithm / motion-estimation acceleration.

**Used for:** fast motion search rejection.

**Still-image relevance:** low for all-intra BPG. Could be conceptually useful for image preanalysis, but these exact kernels are video-motion oriented.

**Rust integration:** defer.

## 7. Integration priority for your BPG-Rust encoder

### Phase A: primitive crate skeleton

Create `bpg-x265-asm`:

```text
bpg-x265-asm/
  build.rs
  asm/x86/*.asm
  src/lib.rs
  src/ffi.rs
  src/dispatch.rs
  src/wrappers.rs
```

Expose only stable safe wrappers. Keep raw FFI private.

### Phase B: mode-decision acceleration

Integrate:

- `pixel-a.asm`: SATD / SA8D.
- `ssd-a.asm`: SSD.
- `sad-a.asm` and `sad16-a.asm`: optional coarse SAD.
- `intrapred8_allangs.asm`: all-angles prediction, if signatures can be validated.
- `intrapred8.asm` / `intrapred16.asm`: direct prediction functions.

This attacks your current rough-mode bottleneck.

### Phase C: transform path acceleration

Integrate:

- `dct8.asm`: DCT/DST/IDCT/IDST.
- `pixel-util8.asm`: residual, count_nonzero, quant/dequant helpers.
- `pixeladd8.asm`: reconstruction add/clip.
- `blockcopy8.asm`: copy/fill support.

This attacks transform/quant/reconstruction overhead.

### Phase D: loop filters

Integrate:

- `loopfilter.asm`: SAO/deblock.

Only do this after scalar SAO/deblock are correct.

### Phase E: video/inter kernels, optional

Defer:

- `ipfilter8/16.asm`
- `h-ipfilter8/16.asm`
- `h4-ipfilter16.asm`
- `v4-ipfilter8/16.asm`
- `mc-a.asm`
- `mc-a2.asm`
- `seaintegral.asm`

These are mostly inter/video kernels.

## 8. Testing strategy

Every assembly wrapper should have three test levels:

1. **Scalar equivalence test**: compare asm output with existing scalar Rust for random blocks.
2. **Edge-value test**: all zeros, all max, checkerboards, gradients, negative residuals, extreme QPs.
3. **Encoder integration test**: encode a known image and verify the decoded pixels match encoder reconstruction.

Example:

```rust
#[test]
fn dct4_matches_scalar() {
    for seed in 0..1000u64 {
        let block = random_i16_block::<16>(seed);
        let scalar = scalar_dct4(&block);
        let asm = unsafe { asm_dct4(&block) };
        assert_eq!(asm, scalar, "seed={seed}");
    }
}
```

For kernels whose output is not bit-identical by design, such as SATD if x265 uses a particular rounding convention, compare against x265's C primitive rather than an independently written scalar implementation.

## 9. Practical caveats

### 9.1 Exact signatures must be recovered from x265 primitive headers

The `.asm` files show argument counts and sometimes argument names through `cglobal`, but the authoritative C signatures live in x265's primitive declarations. Before writing Rust FFI, inspect x265's C++ headers and wrapper assignments.

### 9.2 AVX/SSE transition cost

If mixing SSE and AVX functions, make sure the assembly emits `vzeroupper` as x86inc expects. Avoid jumping between AVX and SSE hot paths unnecessarily.

### 9.3 Memory alignment

Some kernels have aligned variants. Rust wrappers should either guarantee alignment or call unaligned variants. Do not lie to the assembly about alignment.

### 9.4 Stride units

Check whether stride is in pixels, bytes, shorts, or elements for each primitive. x265 commonly passes strides in elements for pixel buffers, but intermediate short buffers can differ. The wrapper API should make this explicit.

### 9.5 8-bit vs high-bit-depth builds

Do not use one object set for all bit depths unless you know the x265 macro build supports it. Build separate namespaces:

```text
bpgx265_8_*
bpgx265_10_*
bpgx265_12_*
```

Then dispatch by bit depth before CPU feature.

## 10. Minimal first integration target

A practical first milestone is:

```text
8-bit only
SSE4 + AVX2 only
SATD 4/8/16/32
SSD 4/8/16/32
DCT/DST/IDCT 4/8/16/32
intra planar/DC/angular for 4/8/16/32
pixel_add_ps
pixel_sub_ps
count_nonzero
```

Do not integrate AVX512 first. Do not integrate inter-prediction filters first. Do not integrate CU-tree/motion helpers for still images.

## 11. Mapping table

| File | Main kernel families | Still-image priority | Use in Rust encoder |
|---|---:|---:|---|
| `const-a.asm` | global constants | required | link support object |
| `x86inc.asm` | ABI/macros | required | include only |
| `x86util.asm` | SIMD utility macros | required | include only |
| `cpu-a.asm` | CPUID/XGETBV/barriers | low | prefer Rust CPU detection |
| `blockcopy8.asm` | block copy/fill/conversion | medium/high | copy/fill scratch and recon regions |
| `pixeladd8.asm` | add residual to prediction | high | reconstruction after inverse transform |
| `pixel-util8.asm` | residual, quant, dequant, nonzero, transpose | very high | transform/quant hot path |
| `dct8.asm` | DCT/DST/IDCT/RDOQ | very high | transform and reconstruction |
| `intrapred8.asm` | 8-bit intra prediction | very high | prediction for RD and final recon |
| `intrapred16.asm` | high-bit-depth intra prediction | high | 10/12-bit prediction |
| `intrapred8_allangs.asm` | all angular predictions | very high | rough mode decision acceleration |
| `pixel-a.asm` | SATD/SA8D/psy/SSIM helpers | very high | mode decision costs |
| `sad-a.asm` | 8-bit SAD/x3/x4 | medium | coarse candidate pruning |
| `sad16-a.asm` | high-bit-depth SAD/x3/x4 | medium | 10/12-bit coarse candidate pruning |
| `ssd-a.asm` | SSD/SSE costs | high | RD distortion |
| `pixel-32.asm` | SA8D/SSIM internals | low/medium | defer |
| `loopfilter.asm` | SAO/deblock | high later | enable after scalar filters pass |
| `ipfilter8.asm` | 8-bit interpolation | low | inter only; defer |
| `ipfilter16.asm` | high-bit-depth interpolation | low | inter only; defer |
| `h-ipfilter8.asm` | 8-bit horizontal interpolation | low | inter only; defer |
| `h-ipfilter16.asm` | high-bit-depth horizontal interpolation | low | inter only; defer |
| `h4-ipfilter16.asm` | high-bit-depth 4-tap chroma horiz | low | inter only; defer |
| `v4-ipfilter8.asm` | 8-bit 4-tap vertical chroma | low | inter only; defer |
| `v4-ipfilter16.asm` | high-bit-depth 4-tap vertical chroma | low | inter only; defer |
| `mc-a.asm` | motion-comp average/weight | very low | animation/inter only |
| `mc-a2.asm` | lowres/cutree/frame helpers | very low | video lookahead only |
| `seaintegral.asm` | SEA integral sums | very low | motion estimation only |

## 12. Final recommendation

Treat these kernels as a backend acceleration layer, not the architecture of your Rust encoder. Your Rust structure should remain:

```text
analysis policy -> scalar-correct primitive trait -> CPU-dispatched primitive table -> asm implementation
```

Do not mirror every x265 C++ primitive table one-to-one unless it helps. Build the primitive table your still-image encoder actually needs, then map x265 kernels into it.
