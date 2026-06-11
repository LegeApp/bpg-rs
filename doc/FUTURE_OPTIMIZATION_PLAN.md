# BPG Future Optimization Plan

## Overview

Plan to enhance the BPG encoder with:
1. NASM assembly optimizations for performance
2. JCTVC encoder option for better compression
3. Native library integration (no subprocess)

---

## 1. NASM Assembly Optimizations

### Current Status
- x265 built with `-DENABLE_ASSEMBLY=OFF` due to CMake 4.x ASM_YASM module issues
- Encoding speed is slower without assembly optimizations

### Implementation Plan

#### Step 1: Fix CMake ASM_YASM Integration
```bash
# Option A: Use older CMake version that works with YASM
pacman -S mingw-w64-x86_64-cmake-3.27

# Option B: Patch CMake to provide missing modules
# Create custom FindASM_YASM.cmake module
# Provide CMakeASM_YASMLinkerInformation.cmake
```

#### Step 2: Rebuild x265 with Assembly
```bash
cd d:\misc\arc\openarc\BPG\libbpg-0.9.8

# Clean previous build
Remove-Item -Recurse -Force x265.out

# Update build_x265.bat to remove -DENABLE_ASSEMBLY=OFF
# Build with assembly optimizations
.\build_x265_with_asm.bat
```

#### Step 3: Performance Testing
```bash
# Test encoding speed with assembly
.\bpgenc_native.exe -m 9 -q 25 -o test_asm.bpg test_input.jpg

# Compare speed vs non-assembly version
# Expected: 2-3x faster encoding
```

#### Files to Create/Modify
- `build_x265_with_asm.bat` - Build script with assembly
- `cmake/FindASM_YASM.cmake` - Custom CMake module
- `cmake/CMakeASM_YASMLinkerInformation.cmake` - Missing module

---

## 2. JCTVC Encoder Integration

### Current Status
- JCTVC source available in `jctvc/` directory
- Previous build attempt failed due to ar wildcard issues
- JCTVC provides better compression but slower encoding

### Implementation Plan

#### Step 1: Fix JCTVC Build Script
```batch
REM build_encoder_jctvc.bat - Fix ar command issues

REM Create file list for ar
dir /b /s jctvc\*.o > objfiles.txt
ar rcs jctvc/libjctvc.a @objfiles.txt
del objfiles.txt
```

#### Step 2: Create JCTVC-Enabled BPG Encoder
```batch
REM build_bpg_with_jctvc.bat

# Compile JCTVC components
gcc -c jctvc/TLibCommon/*.cpp
gcc -c jctvc/TLibEncoder/*.cpp
gcc -c jctvc/TLibVideoIO/*.cpp
gcc -c jctvc/libmd5/*.c
gcc -c jctvc/TAppEncCfg.cpp
gcc -c jctvc/TAppEncTop.cpp
gcc -c jctvc/program_options_lite.cpp

# Create JCTVC library
ar rcs libjctvc.a @objfiles.txt

# Build BPG with JCTVC
gcc -DUSE_JCTVC -c bpgenc.c
gcc -o bpgenc_jctvc.exe bpgenc.o libjctvc.a -lstdc++
```

#### Step 3: Add JCTVC Option to Rust
```rust
// In src/codecs/bpg.rs
pub enum BPGEncoderType {
    X265,    // Fast encoding
    JCTVC,   // Better compression
}

pub struct BPGEncoder {
    encoder_type: BPGEncoderType,
    quality: u8,
    lossless: bool,
}
```

#### Step 4: Performance Comparison
```bash
# Test both encoders
.\bpgenc_native.exe -q 25 -o x265.bpg test.jpg    # Fast
.\bpgenc_jctvc.exe -q 25 -o jctvc.bpg test.jpg    # Better compression

# Compare file sizes and encoding times
```

---

## 3. Native Library Integration (No Subprocess)

### Current Status
- BPG encoder called via subprocess from Rust
- Need to integrate directly as native library

### Implementation Plan

#### Step 1: Create Native BPG Library
```c
// bpg_api.h - Native C API
#ifndef BPG_API_H
#define BPG_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Encoder context (opaque)
typedef struct BPGEncoder BPGEncoder;

// Create encoder
BPGEncoder* bpg_encoder_create(int quality, int bit_depth, int lossless);

// Encode image data (in-memory)
int bpg_encode_image(
    BPGEncoder* enc,
    const uint8_t* y_data, const uint8_t* u_data, const uint8_t* v_data,
    int width, int height,
    uint8_t** output, size_t* output_size
);

// Free encoder
void bpg_encoder_destroy(BPGEncoder* enc);

// Get last error
const char* bpg_get_error(void);

#ifdef __cplusplus
}
#endif

#endif
```

#### Step 2: Implement Native API
```c
// bpg_api.c - Implementation
#include "bpg_api.h"
#include "bpgenc.h"
#include <stdlib.h>
#include <string.h>

struct BPGEncoder {
    // x265 encoder context
    void* x265_enc;
    int quality;
    int bit_depth;
    int lossless;
    char error_msg[256];
};

BPGEncoder* bpg_encoder_create(int quality, int bit_depth, int lossless) {
    BPGEncoder* enc = calloc(1, sizeof(BPGEncoder));
    if (!enc) return NULL;
    
    // Initialize x265 encoder
    // ... x265 initialization code ...
    
    return enc;
}

int bpg_encode_image(
    BPGEncoder* enc,
    const uint8_t* y_data, const uint8_t* u_data, const uint8_t* v_data,
    int width, int height,
    uint8_t** output, size_t* output_size
) {
    // Encode using x265 directly
    // ... encoding implementation ...
    return 0;
}
```

#### Step 3: Update Rust FFI
```rust
// src/codecs/bpg.rs - Native FFI
#[repr(C)]
pub struct BPGEncoder {
    _private: [u8; 0],
}

extern "C" {
    fn bpg_encoder_create(quality: c_int, bit_depth: c_int, lossless: c_int) -> *mut BPGEncoder;
    fn bpg_encode_image(
        enc: *mut BPGEncoder,
        y_data: *const u8,
        u_data: *const u8,
        v_data: *const u8,
        width: c_int,
        height: c_int,
        output: *mut *mut u8,
        output_size: *mut size_t,
    ) -> c_int;
    fn bpg_encoder_destroy(enc: *mut BPGEncoder);
    fn bpg_get_error() -> *const c_char;
}

// High-level Rust API
pub struct NativeBPGEncoder {
    encoder: *mut BPGEncoder,
}

impl NativeBPGEncoder {
    pub fn new(quality: u8, bit_depth: u8, lossless: bool) -> Result<Self> {
        let encoder = unsafe {
            bpg_encoder_create(quality as c_int, bit_depth as c_int, if lossless { 1 } else { 0 })
        };
        
        if encoder.is_null() {
            return Err(anyhow!("Failed to create BPG encoder"));
        }
        
        Ok(Self { encoder })
    }
    
    pub fn encode_image(
        &self,
        y_data: &[u8],
        u_data: &[u8],
        v_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let mut output: *mut u8 = ptr::null_mut();
        let mut output_size: size_t = 0;
        
        let result = unsafe {
            bpg_encode_image(
                self.encoder,
                y_data.as_ptr(),
                u_data.as_ptr(),
                v_data.as_ptr(),
                width as c_int,
                height as c_int,
                &mut output,
                &mut output_size,
            )
        };
        
        if result != 0 {
            return Err(anyhow!("BPG encoding failed"));
        }
        
        let vec = unsafe {
            Vec::from_raw_parts(output, output_size, output_size)
        };
        
        Ok(vec)
    }
}
```

#### Step 4: Build Integration
```rust
// build.rs - Link native library
fn main() {
    // Link native BPG library
    println!("cargo:rustc-link-search=native=../BPG/libbpg-0.9.8");
    println!("cargo:rustc-link-lib=static=bpg_native");
    
    // Link x265 libraries
    println!("cargo:rustc-link-search=native=../BPG/libbpg-0.9.8/x265.out/8bit");
    println!("cargo:rustc-link-search=native=../BPG/libbpg-0.9.8/x265.out/10bit");
    println!("cargo:rustc-link-search=native=../BPG/libbpg-0.9.8/x265.out/12bit");
    println!("cargo:rustc-link-lib=static=x265");
    println!("cargo:rustc-link-lib=static=png");
    println!("cargo:rustc-link-lib=static=jpeg");
    println!("cargo:rustc-link-lib=static=z");
}
```

---

## Implementation Priority

### Phase 1: Native Library Integration (Highest Priority)
- Eliminates subprocess overhead
- Direct memory-to-memory encoding
- Better error handling
- **Timeline**: 1-2 days

### Phase 2: NASM Assembly Optimizations
- 2-3x performance improvement
- Required for production use
- **Timeline**: 2-3 days (depends on CMake issues)

### Phase 3: JCTVC Encoder Option
- Better compression ratio
- Slower encoding (acceptable for archival)
- **Timeline**: 1-2 days

---

## Testing Strategy

### Performance Benchmarks
```bash
# Test encoding speed
time cargo run -- encode test.jpg

# Test compression ratio
ls -la test.jpg test.bpg

# Test memory usage
cargo run -- encode-large-image.jpg
```

### Quality Tests
- Encode test images at various quality levels
- Compare PSNR/SSIM metrics
- Visual quality assessment

### Integration Tests
- Test with different image formats (JPG, PNG, TIFF)
- Test with different bit depths (8, 10, 12)
- Test lossless vs lossy encoding

---

## Expected Improvements

| Optimization | Speed Improvement | Compression | Implementation Effort |
|---------------|-------------------|-------------|----------------------|
| Native Library | 10-20% (no subprocess) | Same | Medium |
| NASM Assembly | 200-300% | Same | High |
| JCTVC | -50% (slower) | +10-20% better | Medium |

---

## Files to Create

1. `build_x265_with_asm.bat` - Assembly-enabled x265 build
2. `build_bpg_with_jctvc.bat` - JCTVC encoder build
3. `build_bpg_native.bat` - Native library build
4. `bpg_api.h` - Native C API header
5. `bpg_api.c` - Native API implementation
6. `cmake/FindASM_YASM.cmake` - CMake module fix
7. `src/codecs/bpg_native.rs` - Native Rust FFI

---

## Dependencies

### For NASM Support
- CMake 3.27 (older version) OR custom CMake modules
- NASM assembler (already installed)

### For JCTVC
- JCTVC source (already available)
- No additional dependencies

### For Native Library
- Existing x265 libraries
- libpng, libjpeg (already installed)

---

## Rollout Plan

1. **Week 1**: Implement native library integration
2. **Week 2**: Fix NASM/CMake issues and rebuild with assembly
3. **Week 3**: Add JCTVC encoder option
4. **Week 4**: Performance testing and optimization

This plan provides a clear path to a fully optimized, native BPG encoder integrated directly into the OpenArc application.
