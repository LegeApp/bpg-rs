# BPG JCTVC Encoder Integration Guide

## Overview
This document outlines the necessary changes made to the BPG codebase to enable JCTVC encoder support. The JCTVC encoder offers better compression efficiency than the default x265 encoder but is significantly slower.

## Key Changes Made

### 1. Enable JCTVC in Makefile
```makefile
# Enable the JCTVC code (best quality but slow) for the encoder
USE_JCTVC=y
```

### 2. Add JCTVC Source Files and Compilation Rules
```makefile
# JCTVC source files
JCTVC_SRCS := \
    jctvc/TAppEncCfg.cpp \
    jctvc/TAppEncTop.cpp \
    jctvc/program_options_lite.cpp \
    $(wildcard jctvc/TLibCommon/*.cpp) \
    $(wildcard jctvc/TLibEncoder/*.cpp) \
    $(wildcard jctvc/TLibVideoIO/*.cpp) \
    jctvc/libmd5/libmd5.c

# Convert source files to object files
JCTVC_OBJS := $(patsubst %.cpp,%.o,$(filter %.cpp,$(JCTVC_SRCS))) \
              $(patsubst %.c,%.o,$(filter %.c,$(JCTVC_SRCS)))
```

### 3. Add JCTVC Compiler Flags
```makefile
# JCTVC compiler flags
JCTVC_CFLAGS := -I$(PWD)/jctvc \
               -I$(PWD)/jctvc/TLibCommon \
               -I$(PWD)/jctvc/TLibEncoder \
               -I$(PWD)/jctvc/TLibVideoIO \
               -I$(PWD)/jctvc/libmd5
JCTVC_CFLAGS += -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers \
               -Wno-misleading-indentation -Wno-class-memaccess
JCTVC_CFLAGS += -DMSYS_PROJECT -D_MSYS2 -D_CRT_SECURE_NO_DEPRECATE -D_CRT_SECURE_NO_WARNINGS \
               -D_CRT_NONSTDC_NO_WARNINGS -D_WIN32_WINNT=0x0600 -DUSE_JCTVC -D_ISOC99_SOURCE \
               -D_GNU_SOURCE -DHAVE_STRING_H -DHAVE_STDINT_H -DHAVE_INTTYPES_H -DHAVE_MALLOC_H \
               -D__STDC_LIMIT_MACROS

# Add to global CFLAGS
CFLAGS += $(JCTVC_CFLAGS)
CXXFLAGS += $(JCTVC_CFLAGS) -std=c++11
```

### 4. Add Build Rules for JCTVC
```makefile
# Compile JCTVC C++ source files
%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile JCTVC C source files
%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Build JCTVC static library
jctvc/libjctvc.a: $(JCTVC_OBJS)
	@mkdir -p jctvc
	$(AR) rcs $@ $^

# Add JCTVC to build
BPGENC_OBJS += jctvc_glue.o jctvc/libjctvc.a
```

### 5. Update Python Script for JCTVC Support
```python
# Add command line argument
parser.add_argument('--jctvc', action='store_true', help='Use JCTVC encoder (slower but better quality)')

# In the conversion function
if use_jctvc:
    cmd.extend(["-e", "jctvc"])
```

### 6. Create a Dedicated JCTVC Build Target
```makefile
# Target to build only JCTVC encoder
jctvc: clean
	$(MAKE) USE_JCTVC=y USE_X265= bpgenc$(EXE)
```

## Building and Usage

### Building JCTVC Encoder
```bash
make jctvc
```

### Using the Python Script with JCTVC
```bash
# With JCTVC encoder
python3 bpg-convert2.py --jctvc

# With default x265 encoder
python3 bpg-convert2.py
```

## Notes
- The JCTVC encoder is significantly slower than x265 but provides better compression efficiency
- The `--jctvc` flag must be explicitly provided to use the JCTVC encoder
- The build process automatically includes all necessary JCTVC source files and dependencies

## Troubleshooting
1. If you encounter compilation errors, ensure all JCTVC source files are present
2. Check that all include paths in the Makefile are correct
3. Verify that the compiler supports C++11 features
4. If you see linker errors, ensure all object files are being included in the final binary