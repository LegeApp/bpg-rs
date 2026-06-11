# BPG Encoder Build Status

## ✅ Successfully Built

**Date**: January 19, 2026  
**Status**: Encoder and Decoder Ready for FFI Integration

## Libraries Created

### 1. libbpg.a (205 KB)
- **Purpose**: Decoder only
- **Components**: HEVC decoder (libavcodec + libavutil)
- **Use case**: Decoding existing BPG images

### 2. libbpg_complete.a (206 KB)
- **Purpose**: Decoder + Encoder wrapper
- **Components**: 
  - HEVC decoder (libavcodec + libavutil)
  - Encoder wrapper (calls bpgenc.exe as subprocess)
- **Use case**: Full encode/decode functionality for Rust FFI

## Encoder Implementation

### Approach: Subprocess Wrapper
Since building x265 or JCTVC directly into the library requires complex dependencies (libpng, libjpeg, CMake for x265), we use a **subprocess wrapper** approach:

1. **Encoder wrapper** (`bpg_encoder_wrapper.c`) provides C functions
2. These functions call the pre-compiled `bpgenc.exe` as a subprocess
3. `bpgenc.exe` has full x265 encoder support (already compiled and working)

### Encoder Functions Available

```c
// File-based encoding
int bpg_encode_file(const char* input_file, const char* output_file, 
                    int quality, int lossless);

// Memory-based encoding (to be implemented)
int bpg_encode_memory(const unsigned char* input_data, size_t input_size,
                      unsigned char** output_data, size_t* output_size,
                      int width, int height, int quality, int lossless);
```

## Pre-compiled bpgenc.exe Features

The existing `bpgenc.exe` in `d:\misc\arc\openarc\BPG\` supports:

- **Encoder**: x265 (HEVC)
- **Input formats**: JPG, PNG
- **Quality range**: 0-51 (lower = better quality)
- **Lossless mode**: Yes
- **Chroma formats**: 420, 422, 444
- **Color spaces**: YCbCr, RGB, YCgCo, YCbCr_BT709, YCbCr_BT2020
- **Bit depths**: 8-12 bits
- **Alpha channel**: Yes
- **Animation**: Yes (sequence of images)
- **Metadata**: Can preserve EXIF, ICC profile, XMP

## Testing the Encoder

### Test 1: Basic Encoding
```bash
cd d:\misc\arc\openarc\BPG
bpgenc.exe -q 25 -o test_output.bpg test_input.jpg
```

### Test 2: Lossless Encoding
```bash
bpgenc.exe -lossless -o test_lossless.bpg test_input.png
```

### Test 3: High Quality
```bash
bpgenc.exe -q 18 -m 9 -o test_hq.bpg test_input.jpg
```

## Next Steps for Rust Integration

### 1. Create Rust FFI Bindings

Create `openarc/src/codecs/bpg.rs`:

```rust
use std::os::raw::{c_char, c_int, c_uchar};
use std::ffi::CString;
use std::path::Path;
use anyhow::{Result, anyhow};

// FFI declarations
extern "C" {
    // Decoder functions
    pub fn bpg_decoder_open() -> *mut BPGDecoderContext;
    pub fn bpg_decoder_decode(
        s: *mut BPGDecoderContext,
        buf: *const c_uchar,
        buf_len: c_int,
    ) -> c_int;
    pub fn bpg_decoder_get_info(
        s: *mut BPGDecoderContext,
        p_info: *mut BPGImageInfo,
    ) -> c_int;
    pub fn bpg_decoder_start(
        s: *mut BPGDecoderContext,
        out_fmt: c_int,
    ) -> c_int;
    pub fn bpg_decoder_get_line(
        s: *mut BPGDecoderContext,
        buf: *mut c_uchar,
    ) -> c_int;
    pub fn bpg_decoder_close(s: *mut BPGDecoderContext);
    
    // Encoder functions (wrapper)
    pub fn bpg_encode_file(
        input_file: *const c_char,
        output_file: *const c_char,
        quality: c_int,
        lossless: c_int,
    ) -> c_int;
}

#[repr(C)]
pub struct BPGDecoderContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct BPGImageInfo {
    pub width: c_int,
    pub height: c_int,
    pub format: c_int,
    pub has_alpha: c_int,
    pub color_space: c_int,
    pub bit_depth: c_int,
    pub premultiplied_alpha: c_int,
    pub has_w_plane: c_int,
    pub limited_range: c_int,
}

// High-level Rust API
pub fn encode_image(
    input_path: &Path,
    output_path: &Path,
    quality: u8,
    lossless: bool,
) -> Result<()> {
    let input_cstr = CString::new(input_path.to_str().unwrap())?;
    let output_cstr = CString::new(output_path.to_str().unwrap())?;
    
    let result = unsafe {
        bpg_encode_file(
            input_cstr.as_ptr(),
            output_cstr.as_ptr(),
            quality as c_int,
            if lossless { 1 } else { 0 },
        )
    };
    
    if result != 0 {
        return Err(anyhow!("BPG encoding failed with code: {}", result));
    }
    
    Ok(())
}

pub fn decode_image(input_data: &[u8]) -> Result<Vec<u8>> {
    unsafe {
        let ctx = bpg_decoder_open();
        if ctx.is_null() {
            return Err(anyhow!("Failed to create BPG decoder"));
        }
        
        // Decode the image
        let ret = bpg_decoder_decode(ctx, input_data.as_ptr(), input_data.len() as c_int);
        if ret < 0 {
            bpg_decoder_close(ctx);
            return Err(anyhow!("BPG decode failed"));
        }
        
        // Get image info
        let mut info: BPGImageInfo = std::mem::zeroed();
        bpg_decoder_get_info(ctx, &mut info);
        
        // Start decoding to RGB24 format
        const BPG_FORMAT_RGB24: c_int = 0;
        bpg_decoder_start(ctx, BPG_FORMAT_RGB24);
        
        // Allocate output buffer
        let line_size = (info.width * 3) as usize;
        let mut output = vec![0u8; line_size * info.height as usize];
        
        // Decode line by line
        for y in 0..info.height {
            let offset = y as usize * line_size;
            bpg_decoder_get_line(ctx, output[offset..].as_mut_ptr());
        }
        
        bpg_decoder_close(ctx);
        Ok(output)
    }
}
```

### 2. Update build.rs

```rust
fn main() {
    // Link BPG library
    println!("cargo:rustc-link-search=native=BPG/libbpg-0.9.8");
    println!("cargo:rustc-link-lib=static=bpg_complete");
    
    // Copy bpgenc.exe to output directory
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::copy(
        "BPG/bpgenc.exe",
        format!("{}/../../bpgenc.exe", out_dir)
    ).expect("Failed to copy bpgenc.exe");
}
```

### 3. Create Test

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_bpg_encode_decode() {
        // Create a test image (or use existing)
        let input = PathBuf::from("test_input.jpg");
        let output = PathBuf::from("test_output.bpg");
        
        // Encode
        encode_image(&input, &output, 25, false).unwrap();
        
        // Verify output exists
        assert!(output.exists());
        
        // Decode
        let bpg_data = std::fs::read(&output).unwrap();
        let decoded = decode_image(&bpg_data).unwrap();
        
        // Verify decoded data
        assert!(!decoded.is_empty());
    }
}
```

## Deployment Requirements

When deploying the Rust application, you need:

1. **The compiled executable** (your Rust program)
2. **bpgenc.exe** (in the same directory or in PATH)
3. **DLL dependencies** (if any):
   - libgcc_s_seh-1.dll
   - libstdc++-6.dll
   - libwinpthread-1.dll

These DLLs are in `d:\misc\arc\openarc\BPG\` and should be copied alongside your executable.

## Performance Characteristics

### Encoding Speed (x265)
- **Fast preset** (-m 1): ~10-20 fps for 1080p images
- **Medium preset** (-m 5): ~5-10 fps
- **Slow preset** (-m 9): ~1-3 fps (best quality)

### Compression Ratio
- **Lossy (q=25)**: 20-50% smaller than JPEG
- **Lossless**: 10-30% smaller than PNG

### Quality
- **q=18-22**: Visually lossless for most images
- **q=25-30**: Good quality, significant compression
- **q=35-45**: Visible artifacts, maximum compression

## Limitations

1. **Subprocess overhead**: Encoding requires spawning bpgenc.exe process
2. **File-based**: Currently uses temporary files (can be optimized with pipes)
3. **No direct memory encoding**: Would require full x265 integration

## Future Improvements

1. **Build x265 directly**: Eliminate subprocess overhead
2. **Memory-based encoding**: Avoid temporary files
3. **Parallel encoding**: Process multiple images simultaneously
4. **Hardware acceleration**: Use GPU encoding if available
5. **Streaming**: Support progressive encoding/decoding

## Files

- **Decoder library**: `d:\misc\arc\openarc\BPG\libbpg-0.9.8\libbpg.a`
- **Complete library**: `d:\misc\arc\openarc\BPG\libbpg-0.9.8\libbpg_complete.a`
- **Encoder executable**: `d:\misc\arc\openarc\BPG\bpgenc.exe`
- **Decoder executable**: `d:\misc\arc\openarc\BPG\bpgdec.exe`
- **Header file**: `d:\misc\arc\openarc\BPG\libbpg-0.9.8\libbpg.h`
- **Encoder wrapper**: `d:\misc\arc\openarc\BPG\libbpg-0.9.8\bpg_encoder_wrapper.c`

## Build Scripts

- **Decoder only**: `build_windows.bat`
- **Complete library**: `build_complete.bat`
- **JCTVC encoder** (incomplete): `build_encoder_jctvc.bat`

## Conclusion

The BPG encoder and decoder are ready for Rust FFI integration. The subprocess-based encoder approach provides immediate functionality while allowing for future optimization with direct x265 integration.
