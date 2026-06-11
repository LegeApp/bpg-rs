# Building BPG with Both x265 AND JCTVC (Combined DLL)

This repo now builds a single in-process DLL (`openarc_bpg.dll`) that contains **both** x265 and JCTVC encoder support.

**Short Answer:** **2-4 hours** of build script work (mostly straightforward)

---

## Current State

### What We Have Now
- ✅ Combined **DLL** build: `BPG/libbpg-0.9.8/openarc_bpg.dll`
- ✅ Combined encoder selection in `bpgenc.c` (already supports `-e x265|jctvc` when compiled with both defines)
- ✅ OpenArc-side mapping: **Standard = x265**, **Slow = JCTVC** (no longer maps JCTVC to both)

### Key Discovery
The BPG source code **already supports both encoders** in the same binary! Lines 2165-2197 of `bpgenc.c`:

```c
typedef enum {
#if defined(USE_X265)
    HEVC_ENCODER_X265,
#endif
#if defined(USE_JCTVC)
    HEVC_ENCODER_JCTVC,
#endif
    HEVC_ENCODER_COUNT,
} HEVCEncoderEnum;

static HEVCEncoder *hevc_encoder_tab[HEVC_ENCODER_COUNT] = {
#if defined(USE_X265)
    &x265_hevc_encoder,
#endif
#if defined(USE_JCTVC)
    &jctvc_encoder,
#endif
};
```

The encoder is selected via `-e` command line flag (already implemented!).

---

## How To Build

### Recommended (fast): parallel build (all CPU cores)
From `BPG/libbpg-0.9.8`:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\build_openarc_combined_dll.ps1 -Clean
```

Optional: cap/override cores:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\build_openarc_combined_dll.ps1 -Clean -Jobs 12
```

### Legacy entrypoint: .bat (auto-uses parallel when available)

```powershell
cmd /c build_openarc_combined_dll.bat
```

To force the sequential legacy path:

```powershell
set OPENARC_NO_PARALLEL=1
cmd /c build_openarc_combined_dll.bat
```

## Output

- `openarc_bpg.dll` and `openarc_bpg.lib` are written to `BPG/libbpg-0.9.8/`.
- Runtime dependencies are the MinGW DLLs and codec deps printed by the build script.

**Steps:**

1. **Enable both flags in Makefile** (5 minutes)
   ```makefile
   # Change line 8-12 in Makefile from:
   USE_X265=y
   # USE_JCTVC=y
   
   # To:
   USE_X265=y
   USE_JCTVC=y
   ```

2. **Build x265 library** (30 minutes)
   - Download x265 source from VideoLAN
   - Build with MinGW: `cmake -G "MinGW Makefiles" && make`
   - Or use pre-built x265.lib from MSYS2

3. **Build with both encoders** (30 minutes)
   ```bash
   cd D:\misc\arc\openarc\BPG\libbpg-0.9.8
   make USE_X265=y USE_JCTVC=y bpgenc.exe
   ```

4. **Test encoder selection** (15 minutes)
   ```bash
   bpgenc.exe -e x265 -q 28 -o test_x265.bpg input.jpg
   bpgenc.exe -e jctvc -q 28 -o test_jctvc.bpg input.jpg
   bpgenc.exe -h  # Should show both encoders available
   ```

5. **Update Rust wrapper** (45 minutes)
   - Modify `bpg_subprocess.rs` to add `BpgEncoderType::X265`
   - Update tests to verify both encoders work
   - Document encoder selection

**Challenges:**
- ⚠️ Need x265 library compiled for MinGW (may already exist in MSYS2)
- ⚠️ Linking both C and C++ code (Makefile already handles this)
- ⚠️ Potential symbol conflicts (unlikely, clean separation in code)

**Result:**
- Single `bpgenc.exe` with both encoders (~2-3 MB)
- User selects via `-e x265` or `-e jctvc`
- About 6-7 additional DLLs from x265 dependencies

---

## Notes

- The bundled FFmpeg subset relies on `HAVE_AV_CONFIG_H=1` so `config.h/intmath.h` are consistently visible across compilation units.
- `libavcodec/*_template.c` sources are *included* by other units and should not be compiled standalone.

---

## Comparison Table

| Aspect | Combined Executable | Static Library |
|--------|-------------------|----------------|
| **Effort** | 2 hours | 4 hours |
| **Complexity** | Low | High |
| **Build Issues** | Minor (x265 linking) | Major (libavutil errors) |
| **Integration** | Subprocess (10-20ms overhead) | Direct FFI (no overhead) |
| **Maintenance** | Easy | Difficult |
| **Size** | 2-3 MB exe + DLLs | 15-20 MB library |
| **Recommended** | ✅ Yes | ⚠️ Only if FFI required |

---

## Practical Recommendation

### Best Approach: Combined Executable (Option 1)

**Why:**
1. BPG source already supports it (90% of code exists)
2. Quick to implement (just enable flags and rebuild)
3. Subprocess overhead negligible for typical encoding (1-5 sec)
4. Easy to maintain and update
5. Clean separation of concerns

**Workflow:**
```rust
// Your code stays the same, just update encoder_path
let encoder_path = "bpgenc-combined.exe";  // Has both x265 and JCTVC

// Fast encoding
encoder.encode_with_type(input, output, BpgEncoderType::X265)?;

// Best compression
encoder.encode_with_type(input, output, BpgEncoderType::Jctvc)?;
```

**When to use each:**
- **x265**: Real-time encoding, batch processing, preview generation
- **JCTVC**: Archival storage, final output, maximum compression

---

## Step-by-Step Build Guide (Estimated 2 hours)

### Prerequisites
```powershell
# Check if x265 is available in MSYS2
pacman -Ss x265

# If not installed:
pacman -S mingw-w64-x86_64-x265
```

### Build Script
```bash
#!/bin/bash
# build_bpg_combined.sh

cd /d/misc/arc/openarc/BPG/libbpg-0.9.8

# Clean previous builds
make clean
rm -f bpgenc.exe

# Build with both encoders
make USE_X265=y USE_JCTVC=y CONFIG_WIN32=y bpgenc.exe

# Rename to avoid confusion
mv bpgenc.exe bpgenc-combined.exe

# Test
./bpgenc-combined.exe -h | grep encoder
```

### Expected Output
```
-e encoder           select the HEVC encoder (x265, jctvc, default = x265)
```

### Integration Update (15 minutes)
Update `bpg_subprocess.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpgEncoderType {
    X265,   // Fast, good compression
    Jctvc,  // Slow, best compression
}
```

Test both:
```rust
// Test fast encoder
BpgEncoder::new(path, BpgConfig::fast())?.encode_file(input, output)?;

// Test best encoder
BpgEncoder::new(path, BpgConfig::best_compression())?.encode_file(input, output)?;
```

---

## Potential Issues & Solutions

### Issue 1: x265 not found during linking
**Solution:**
```bash
# Install system x265
pacman -S mingw-w64-x86_64-x265

# Or specify library path
export LIBRARY_PATH=/c/msys64/mingw64/lib:$LIBRARY_PATH
```

### Issue 2: Undefined references to x265 functions
**Solution:**
```makefile
# Ensure x265 library is linked (line ~118 in Makefile)
BPGENC_LIBS += -lx265 -lstdc++ -lpthread
```

### Issue 3: Both encoders compile but crash at runtime
**Solution:**
- Check encoder initialization in `x265_glue.c` and `jctvc_glue.cpp`
- Ensure proper encoder cleanup between calls
- Test each encoder separately first

---

## Timeline

| Task | Estimated Time |
|------|----------------|
| Install x265 dependencies | 15 minutes |
| Modify Makefile (enable both flags) | 5 minutes |
| Build combined executable | 30 minutes |
| Debug any build errors | 30 minutes |
| Test both encoders | 15 minutes |
| Update Rust wrapper | 30 minutes |
| Integration testing | 15 minutes |
| **Total** | **2 hours 20 minutes** |

Add buffer time: **3 hours total**

---

## Conclusion

**Effort Required: 2-4 hours (low to medium complexity)**

The BPG encoder already has the architecture to support both x265 and JCTVC simultaneously. Most of the work is:
1. Enabling the build flags
2. Linking the x265 library
3. Testing the result

The hard part (encoder selection logic, glue code, runtime switching) is **already implemented** in the BPG source code.

### Recommendation
✅ **Build the combined executable** - it's quick, clean, and maintainable. The subprocess approach you already have works great, and adding x265 support is straightforward.

❌ **Skip the static library approach** unless you absolutely need FFI - the libavutil compilation issues will consume hours of debugging for minimal benefit.
