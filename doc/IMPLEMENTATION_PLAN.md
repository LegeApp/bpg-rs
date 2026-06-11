# BPG Optimization Implementation Plan

**Created**: January 19, 2026  
**Objective**: Complete BPG encoder optimization with assembly, JCTVC, and native library integration

---

## Overview

This plan implements three major optimizations for the BPG encoder:

1. **Assembly Optimizations (YASM)** - 2-3x faster encoding
2. **JCTVC Encoder Option** - 10-20% better compression
3. **Native Library Integration** - Eliminate subprocess overhead, direct FFI

---

## Phase 1: Verify Assembly-Enabled x265 Build

### Step 1.1: Verify Build Artifacts
- [x] Confirm x265 libraries built with assembly
- [ ] Check file sizes and timestamps
- [ ] Verify assembly object files present

### Step 1.2: Runtime Performance Test
- [ ] Encode test image with assembly-enabled encoder
- [ ] Compare encoding time vs non-assembly version
- [ ] Verify output quality unchanged

**Expected Results**:
- `bpgenc_native.exe` exists and runs
- Encoding time: ~1-2 seconds for 800x600 image
- Output file size similar to non-assembly version

---

## Phase 2: Build JCTVC Encoder

### Step 2.1: Prepare JCTVC Source
- [ ] Verify JCTVC source files in `jctvc/` directory
- [ ] Identify required source files from Makefile
- [ ] Create object file list for linking

**JCTVC Source Structure**:
```
jctvc/
├── TLibCommon/       # Common data structures
├── TLibEncoder/      # HEVC encoder implementation
├── TLibVideoIO/      # Video I/O
├── libmd5/           # MD5 checksums
├── TAppEncCfg.cpp    # Encoder configuration
├── TAppEncTop.cpp    # Encoder top-level
└── program_options_lite.cpp
```

### Step 2.2: Create JCTVC Build Script
- [ ] Write `build_jctvc.bat` to compile all JCTVC sources
- [ ] Generate object files with proper C++ flags
- [ ] Create `libjctvc.a` static library

**Compiler Flags**:
```batch
-O3 -fno-strict-aliasing -std=c++11
-DMSYS_UNIX -DMSYS_WIN32
```

### Step 2.3: Build BPG Encoder with JCTVC
- [ ] Create `build_bpg_with_jctvc.bat`
- [ ] Compile BPG encoder with `-DUSE_JCTVC`
- [ ] Link against `libjctvc.a` instead of x265
- [ ] Produce `bpgenc_jctvc.exe`

### Step 2.4: Test JCTVC Encoder
- [ ] Encode test image with JCTVC
- [ ] Compare file size vs x265 version
- [ ] Measure encoding time (expected: slower but better compression)

**Expected Results**:
- JCTVC output: 10-20% smaller file size
- Encoding time: 2-3x slower than x265
- Quality: Potentially better at same bitrate

---

## Phase 3: Native Library Integration

### Step 3.1: Design Native API
- [ ] Create `bpg_api.h` with C interface
- [ ] Define encoder context structure
- [ ] Design memory-to-memory encoding functions

**API Design**:
```c
// Encoder creation/destruction
BPGEncoder* bpg_encoder_create(int quality, int bit_depth, int lossless);
void bpg_encoder_destroy(BPGEncoder* enc);

// Encoding functions
int bpg_encode_from_file(BPGEncoder* enc, const char* input_path, 
                         uint8_t** output, size_t* output_size);
int bpg_encode_from_memory(BPGEncoder* enc, const uint8_t* rgba_data,
                           int width, int height, int stride,
                           uint8_t** output, size_t* output_size);

// Error handling
const char* bpg_get_error(BPGEncoder* enc);
```

### Step 3.2: Implement Native Encoder
- [ ] Create `bpg_api.c` implementation
- [ ] Integrate x265 encoder directly (no subprocess)
- [ ] Handle image loading (PNG, JPEG) via libpng/libjpeg
- [ ] Implement YUV conversion and BPG container writing

**Implementation Files**:
- `bpg_api.h` - Public C API
- `bpg_api.c` - Implementation
- `bpg_native.c` - x265 integration
- `bpg_image_io.c` - Image loading/saving

### Step 3.3: Build Native Library
- [ ] Create `build_bpg_native_lib.bat`
- [ ] Compile all native API sources
- [ ] Link with x265, libpng, libjpeg
- [ ] Produce `libbpg_native.a` static library

### Step 3.4: Create Test Executable
- [ ] Build `bpgenc_native_lib.exe` using native API
- [ ] Test encoding from command line
- [ ] Verify memory management (no leaks)

---

## Phase 4: Rust FFI Integration

### Step 4.1: Create Rust FFI Bindings
- [ ] Update `src/codecs/bpg_native.rs`
- [ ] Define FFI extern declarations
- [ ] Create safe Rust wrapper types

**Rust FFI Structure**:
```rust
#[repr(C)]
pub struct BPGEncoder {
    _private: [u8; 0],
}

extern "C" {
    fn bpg_encoder_create(...) -> *mut BPGEncoder;
    fn bpg_encode_from_memory(...) -> c_int;
    fn bpg_encoder_destroy(...);
}

pub struct NativeBPGEncoder {
    encoder: *mut BPGEncoder,
}
```

### Step 4.2: Update build.rs
- [ ] Remove subprocess exe copying logic
- [ ] Add native library linking
- [ ] Link x265, libpng, libjpeg, zlib
- [ ] Set proper library search paths

**build.rs Changes**:
```rust
println!("cargo:rustc-link-search=native=../BPG/libbpg-0.9.8");
println!("cargo:rustc-link-lib=static=bpg_native");
println!("cargo:rustc-link-lib=static=x265");
println!("cargo:rustc-link-lib=static=png");
println!("cargo:rustc-link-lib=static=jpeg");
```

### Step 4.3: Update CLI to Use Native Library
- [ ] Modify `src/main.rs` convert-bpg command
- [ ] Use `NativeBPGEncoder` instead of subprocess
- [ ] Handle errors properly
- [ ] Add progress reporting

### Step 4.4: Test Rust Integration
- [ ] Run `cargo build`
- [ ] Test `cargo run -- convert-bpg test.jpg -o out.bpg`
- [ ] Verify no DLL dependencies needed
- [ ] Check memory usage and performance

---

## Phase 5: Benchmarking and Comparison

### Step 5.1: Prepare Test Suite
- [ ] Create test images (various sizes/types)
- [ ] Define benchmark metrics (time, size, quality)
- [ ] Set up automated test script

**Test Images**:
- Small: 800x600 photo
- Medium: 1920x1080 photo
- Large: 4K photo
- Complex: High-detail image

### Step 5.2: Run Benchmarks
- [ ] x265 without assembly (baseline)
- [ ] x265 with assembly (optimized)
- [ ] JCTVC encoder (quality-focused)
- [ ] Native library (integration test)

**Metrics to Collect**:
- Encoding time (seconds)
- Output file size (bytes)
- Compression ratio
- Memory usage
- PSNR/SSIM (if available)

### Step 5.3: Document Results
- [ ] Create benchmark results table
- [ ] Generate comparison charts
- [ ] Document recommendations

---

## File Structure

```
BPG/
├── libbpg-0.9.8/
│   ├── x265.out/              # x265 with assembly
│   │   ├── 12bit/libx265.a
│   │   ├── 10bit/libx265.a
│   │   └── 8bit/libx265.a
│   ├── jctvc/                 # JCTVC source
│   │   └── libjctvc.a         # (to be built)
│   ├── bpg_api.h              # Native API header
│   ├── bpg_api.c              # Native API implementation
│   ├── libbpg_native.a        # Native library
│   ├── bpgenc_native.exe      # x265 encoder (subprocess)
│   ├── bpgenc_jctvc.exe       # JCTVC encoder (subprocess)
│   ├── bpgenc_native_lib.exe  # Native library test
│   ├── build_jctvc.bat
│   ├── build_bpg_with_jctvc.bat
│   └── build_bpg_native_lib.bat
└── IMPLEMENTATION_PLAN.md     # This file
```

---

## Success Criteria

### Phase 1: Assembly Build ✓
- [x] x265 builds with YASM assembly
- [ ] Encoding speed 2-3x faster than non-assembly
- [ ] Output quality identical

### Phase 2: JCTVC
- [ ] JCTVC encoder builds successfully
- [ ] Compression 10-20% better than x265
- [ ] Encoding time acceptable for archival use

### Phase 3: Native Library
- [ ] Native API compiles and links
- [ ] No subprocess overhead
- [ ] Memory-safe implementation
- [ ] Test executable works

### Phase 4: Rust Integration
- [ ] Rust builds without errors
- [ ] CLI uses native library
- [ ] No runtime DLL dependencies
- [ ] Performance improvement measurable

### Phase 5: Benchmarks
- [ ] All variants tested
- [ ] Results documented
- [ ] Recommendations provided

---

## Risk Mitigation

### JCTVC Build Issues
- **Risk**: JCTVC may have complex dependencies
- **Mitigation**: Start with minimal build, add features incrementally

### Native Library Memory Management
- **Risk**: Memory leaks or corruption
- **Mitigation**: Use valgrind/sanitizers, careful testing

### Rust FFI Safety
- **Risk**: Unsafe FFI boundary issues
- **Mitigation**: Extensive testing, proper error handling

### Performance Regression
- **Risk**: Native library slower than subprocess
- **Mitigation**: Profile and optimize, benchmark early

---

## Timeline Estimate

- **Phase 1**: 30 minutes (verification only)
- **Phase 2**: 2-3 hours (JCTVC build and test)
- **Phase 3**: 3-4 hours (native library implementation)
- **Phase 4**: 2-3 hours (Rust integration)
- **Phase 5**: 1-2 hours (benchmarking)

**Total**: 8-12 hours of development time

---

## Next Steps

Starting with **Phase 1, Step 1.1**: Verify x265 assembly build artifacts and test runtime performance.
