# BPG Full Encoder Build Plan

## Dependencies Required

### 1. CMake
- **Status**: Installed (needs PATH update or direct path)
- **Purpose**: Build x265 library
- **Check**: `cmake --version`

### 2. libpng
- **Status**: Need to install
- **Purpose**: PNG image input/output
- **Options**:
  - MSYS2: `pacman -S mingw-w64-x86_64-libpng`
  - Manual: Download from http://www.libpng.org/pub/png/libpng.html

### 3. libjpeg
- **Status**: Need to install
- **Purpose**: JPEG image input
- **Options**:
  - MSYS2: `pacman -S mingw-w64-x86_64-libjpeg-turbo`
  - Manual: Download libjpeg-turbo from https://github.com/libjpeg-turbo/libjpeg-turbo

### 4. x265 source
- **Status**: Already present in `x265/` directory
- **Purpose**: HEVC encoder
- **Build**: Use CMake to build 8-bit, 10-bit, and 12-bit versions

## Build Steps

### Step 1: Install Dependencies via MSYS2 (Recommended)
```bash
# Open MSYS2 terminal
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-libpng
pacman -S mingw-w64-x86_64-libjpeg-turbo
```

### Step 2: Build x265
```bash
cd d:\misc\arc\openarc\BPG\libbpg-0.9.8

# Create build directories
mkdir -p x265.out/8bit x265.out/10bit x265.out/12bit

# Build 12-bit
cd x265.out/12bit
cmake ../../x265/source -G "MinGW Makefiles" -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN12=ON
mingw32-make

# Build 10-bit
cd ../10bit
cmake ../../x265/source -G "MinGW Makefiles" -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DENABLE_CLI=OFF -DMAIN10=ON
mingw32-make

# Build 8-bit (links 10-bit and 12-bit)
cd ../8bit
cmake ../../x265/source -G "MinGW Makefiles" -DLINKED_10BIT=ON -DLINKED_12BIT=ON -DENABLE_SHARED=OFF -DENABLE_CLI=OFF
mingw32-make
```

### Step 3: Build BPG with x265
```bash
cd d:\misc\arc\openarc\BPG\libbpg-0.9.8

# Use the build script with x265 support
# (Will create a new script for this)
```

## Alternative: Manual Dependency Download

If MSYS2 is not available, download pre-built libraries:

1. **libpng**: https://sourceforge.net/projects/libpng/files/
2. **libjpeg-turbo**: https://github.com/libjpeg-turbo/libjpeg-turbo/releases
3. Extract to a `deps/` folder and adjust include/lib paths

## Current Status

- [x] BPG decoder compiled
- [x] BPG encoder wrapper (subprocess) working
- [ ] CMake in PATH
- [ ] libpng installed
- [ ] libjpeg installed
- [ ] x265 built (8-bit, 10-bit, 12-bit)
- [ ] BPG encoder with native x265 compiled
- [ ] bpgenc.exe with full features built
