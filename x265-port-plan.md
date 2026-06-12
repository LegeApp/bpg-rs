This is feasible, but the right target is **not “x265 in Rust”** in the general sense. The target should be:

**a Rust still-picture HEVC intra encoder derived from x265, preserving x265’s proven intra/RDO/primitives/CABAC logic, but deleting the video pipeline.**

That fits BPG much better than a complete port. BPG is explicitly a still-image format based on a subset of HEVC, with support for 4:2:0, 4:2:2, 4:4:4, grayscale, alpha, 8–14 bits, metadata, animation, and lossless mode. For your current bpg-rs direction, the relevant part is the single-image HEVC payload, not the video machinery. ([Fabrice Bellard's Home Page][1])

## Main feasibility judgment

Your idea is plausible because x265’s encoder architecture has a clean split between:

1. **HEVC coding machinery**: CTUs, CUs, intra prediction, transforms, quantization, CABAC, NAL/SPS/PPS/VPS, deblock, SAO.
2. **Video machinery**: GOP decisions, lookahead, B/P frames, motion estimation, references, rate control over time, frame threading, VBV, scenecut, weighted prediction, temporal filters.

For BPG still images, the second category is mostly dead weight.

The uploaded x265 4.1 tree confirms this split. The core you care about is mainly:

```text
source/common      ~27.5k lines top-level C++/headers, excluding arch asm dirs
source/encoder     ~43.6k lines C++/headers
source/common/x86  large asm/primitives layer
source/common/aarch64 / arm / ppc  platform primitives
```

The biggest files are `encoder.cpp`, `search.cpp`, `slicetype.cpp`, `analysis.cpp`, `ratecontrol.cpp`, `param.cpp`, `entropy.cpp`, and `cudata.cpp`. That points to the key strategy: **do not port `slicetype.cpp`, `ratecontrol.cpp`, motion/reference machinery, and CLI/I/O first. Port the intra encode path first.**

## What to keep

Keep these conceptually intact, even if rewritten idiomatically in Rust:

| Area                         | Why keep it                                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Assembly primitive files** | These are the speed-critical kernels. Keeping them avoids losing the main reason x265 is fast.                                                                                                                                                                      |
| **Primitive dispatch table** | x265’s C/C++ code routes DCT, SATD, pixel, intra, SAO, loopfilter, and quant functions through function-pointer tables. Recreate this in Rust around the existing asm.                                                                                              |
| **HEVC syntax writer**       | VPS/SPS/PPS/slice headers/NAL formatting must remain conformant. BPG-rs already expects x265-like HEVC output.                                                                                                                                                      |
| **CABAC / entropy coding**   | This is not optional. It is central to HEVC compression efficiency.                                                                                                                                                                                                 |
| **CTU/CU/TU data model**     | HEVC’s 64×64 CTU structure and recursive CU/TU partitioning are the heart of the encoder.                                                                                                                                                                           |
| **Intra prediction**         | This is the most important coding tool for still images.                                                                                                                                                                                                            |
| **DCT/DST, quant, RDOQ**     | Essential for quality/size tradeoff.                                                                                                                                                                                                                                |
| **Intra analysis/search**    | This is where x265 spends effort choosing prediction modes, splits, transform sizes, and R-D costs.                                                                                                                                                                 |
| **Deblock and SAO**          | Keep them as tunable features. They may help photos but hurt some sharp graphics/text.                                                                                                                                                                              |
| **10-bit path**              | BPG defaults historically favor 10-bit because it can improve compression/rounding behavior. x265 accepts input between 8 and 16 bits and shifts/masks to its internal depth, but practical x265 builds are usually 8/10/12-bit variants. ([x265 Documentation][2]) |

The biggest thing to preserve is **behavioral equivalence at first**. Do not start by “improving” x265. Start by making a narrowed Rust encoder produce valid BPG-compatible HEVC for simple images, then compare output and metrics against stock x265.

## What to drop

Drop these from the final still-image encoder:

| x265 area                                            |                Drop or reduce? | Reason                                                                                                                                                     |
| ---------------------------------------------------- | -----------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source/input`                                       |                           Drop | BPG-rs already owns image loading/conversion.                                                                                                              |
| `source/output`                                      |                           Drop | You do not need `.hevc`, `.y4m`, recon, raw video output wrappers.                                                                                         |
| CLI parser                                           |                           Drop | Replace with typed Rust preset/config structs.                                                                                                             |
| Dynamic HDR10 / Dolby Vision / RPU / film-grain SEIs |                 Drop initially | BPG metadata belongs in the BPG container, not HEVC video SEIs.                                                                                            |
| VMAF integration / CSV logging                       |                           Drop | Not part of encoder core.                                                                                                                                  |
| Lookahead                                            |                           Drop | Single still image has no future frames.                                                                                                                   |
| Scenecut                                             |                           Drop | No scene changes in one image.                                                                                                                             |
| B-frames / P-frames                                  |                           Drop | Still picture should be one intra frame. x265’s docs specify `--keyint 1` for all-intra and `--bframes 0` for no B frames. ([x265 Documentation][3])       |
| Motion estimation                                    | Drop after initial oracle work | No inter prediction in still-picture intra coding.                                                                                                         |
| Reference frames / DPB                               |              Minimize to stubs | One IDR/intra picture does not need real decoded-picture-buffer management.                                                                                |
| Weighted prediction                                  |                           Drop | Inter-video feature.                                                                                                                                       |
| Temporal MVP                                         |                           Drop | Inter-video feature.                                                                                                                                       |
| Cutree                                               |                           Drop | Temporal AQ tool; not useful for one image.                                                                                                                |
| ABR/VBV/two-pass rate control                        |                           Drop | Still images should use CQP, CRF-like single-image QP mapping, or direct target-size search.                                                               |
| Frame threading                                      |                           Drop | Multiple-frame parallelism is irrelevant.                                                                                                                  |
| WPP row threading                                    |                  Keep optional | Still useful for large single images, though x265 documents WPP as a parallelism gain with some compression-efficiency overhead. ([x265 Documentation][3]) |

Do **not** delete all this immediately. First use stock x265 as an oracle, then progressively amputate.

## What to port first

The first Rust port should be a narrow vertical slice:

```text
RGB/YCbCr planes from bpg-rs
        ↓
PicYuv / Frame / FrameData / CUData
        ↓
intra analysis + intra prediction
        ↓
transform + quant + RDO/RDOQ
        ↓
CABAC entropy
        ↓
NAL/VPS/SPS/PPS/slice bitstream
        ↓
BPG-rs HEVC rewrite/container layer
```

Port order:

### 1. Build a still-image x265 oracle

Before porting, make a tiny harness around stock x265 4.1:

```text
input image planes
→ x265 API
→ one IDR/all-intra HEVC bitstream
→ existing bpg-rs container path
→ decode and compare
```

Use settings roughly equivalent to:

```text
totalFrames = 1
keyint = 1
min-keyint = 1
bframes = 0
scenecut = 0
rc-lookahead = 0
open-gop = false
repeat-headers = true
profile = main444-stillpicture / main444-10 / main444-12 depending path
```

x265’s API is centered on an `x265_encoder` object, `x265_param`, `x265_picture`, `x265_encoder_open`, `x265_encoder_headers`, and `x265_encoder_encode`; parameters are copied into the encoder at open time, so your Rust-side API does not need to mimic the mutable C API long term. ([x265 Documentation][2])

This harness becomes the test oracle. Byte-identical output is useful early, but not required forever. Decode-equivalent, metric-equivalent, and size-equivalent are the real goals.

### 2. Port syntax and headers

Start with these:

```text
common/bitstream.*
encoder/nal.*
encoder/sei.*        only minimal needed SEIs, maybe none initially
common/slice.*       VPS/SPS/PPS/profile structures
encoder/level.*      profile/level logic
encoder/entropy.*    CABAC syntax writer
common/contexts.h
```

Goal: generate valid VPS/SPS/PPS and an empty/minimal intra slice structure.

This is a good first milestone because BPG-rs already has modified-SPS/HEVC rewrite logic. You want to prove that the Rust encoder emits the kind of HEVC header stream your BPG container code expects.

### 3. Port primitive dispatch before porting high-level analysis

Do **not** hard-code pure Rust DCT/SATD paths first and optimize later. That risks building the wrong abstraction.

Create a crate boundary like:

```text
x265-primitives-sys
  - builds original asm files
  - exposes C ABI symbols/wrappers

x265-primitives
  - safe Rust function pointer table
  - CPU detection
  - fallback C/Rust primitive implementations
```

Keep the assembly files unchanged. Use a small C/C++ or assembly-wrapper layer only where needed to expose stable unmangled names. The Rust side should see function pointers like:

```rust
pub struct PrimitiveSet {
    pub satd_8x8: SatdFn,
    pub dct_8x8: DctFn,
    pub intra_pred: IntraPredFns,
    pub quant: QuantFn,
    pub sao: SaoFns,
    // ...
}
```

The important part is not to rewrite all SIMD. Preserve x265’s asm as the acceleration substrate.

### 4. Port image/frame memory

Port or redesign:

```text
common/picyuv.*
common/yuv.*
common/shortyuv.*
common/frame.*
common/framedata.*
common/cudata.*
common/ringmem.*
common/piclist.*     maybe minimal/stubbed
```

In Rust, avoid a literal C++ class port where possible. Use:

```rust
struct Picture<T> {
    planes: [Plane<T>; 3],
    chroma: ChromaFormat,
    bit_depth: BitDepth,
}

struct Frame {
    original: Picture,
    reconstructed: Picture,
    data: FrameData,
}
```

For performance, use arena-style allocation and aligned plane buffers. The only acceptable `unsafe` islands should be:

```text
aligned allocation
asm primitive calls
raw pointer interop for primitive kernels
maybe unchecked indexing in deeply profiled kernels
```

Everything else can be safe Rust with explicit slices.

### 5. Port intra prediction, transform, quant, RDO

This is the real encoder.

Port:

```text
common/intrapred.*
common/predict.*
common/dct.*
common/quant.*
common/scalinglist.*
common/pixel.*
encoder/rdcost.h
encoder/search.*
encoder/analysis.*
encoder/bitcost.*
```

But port only the intra-relevant parts first.

In x265, `search.cpp` and `analysis.cpp` are mixed intra/inter logic. Your Rust version should split this into:

```text
intra_search.rs
intra_rdo.rs
cu_split.rs
transform_search.rs
mode_decision.rs
```

This is where the Rust port can become cleaner than the original. For BPG, you do not need inter mode evaluation, merge candidates, motion vectors, reference lists, or bidirectional prediction. Removing those branches should make the code easier to reason about and easier to tune.

### 6. Port loop filters as tunable modules

Port:

```text
common/deblock.*
encoder/sao.*
encoder/framefilter.*
```

But expose them as still-image tuning controls:

```rust
pub enum SaoMode {
    Off,
    PhotoDefault,
    ConservativeEdges,
    Full,
}

pub enum DeblockMode {
    Off,
    Weak,
    X265Default,
    Strong,
}
```

This matters because BPG still images are not all photos. SAO/deblock may help noisy photos but can soften UI, manga, scans, maps, and text. This fits your “tuning, not destructive optimizing” preference.

### 7. Replace rate control with still-image QP control

x265’s full rate control is designed around video. Keep it only as long as needed for comparison, then replace it.

For still images, use:

```text
CQP mode
CRF-like still-image QP mapping
optional target-size binary search
optional per-block QP offset map
```

x265 already has a useful hook: `x265_picture` supports per-image quantizer offsets, applied per 16×16 block or 8×8 block depending on `qg-size`, when adaptive quantization is enabled. ([GitHub][4])

That is directly relevant to your plan. Instead of trying to force all tuning through CLI options, bpg-rs can compute image features and feed a QP-offset map directly into the encoder.

Use this for conservative tuning:

```text
protect faces/skin-like smooth gradients
protect text/edges/line art
slightly relax noisy/grainy high-entropy regions
avoid overspending on visually chaotic foliage/noise
preserve flat gradients with 10/12-bit paths
```

This is not PNGQuant-style destructive preprocessing. It is encoder steering.

## Suggested crate layout

A clean Rust architecture would look like this:

```text
bpg-rs
  container, metadata, BPG header, HEVC rewrite, decoder integration

x265-rs
  public still-image encoder API

x265-core
  CTU/CU/TU, intra analysis, RDO, entropy, loop filters

x265-syntax
  NAL, VPS/SPS/PPS, slice headers, bitstream writer, CABAC

x265-primitives
  Rust primitive table, CPU dispatch, safe wrappers

x265-primitives-sys
  original asm files, C ABI shims, build.rs/nasm/cc integration

x265-oracle-tests
  optional test harness comparing against stock x265 4.1
```

The public API should not look like x265’s video API. It should look like an image encoder:

```rust
pub struct BpgHevcEncoder {
    config: StillHevcConfig,
    primitives: PrimitiveSet,
}

pub struct StillHevcConfig {
    pub bit_depth: BitDepth,
    pub chroma: ChromaFormat,
    pub quality: QualityMode,
    pub preset: StillPreset,
    pub sao: SaoMode,
    pub deblock: DeblockMode,
    pub effort: Effort,
}

pub enum StillPreset {
    Photo,
    ArchivalPhoto,
    GrainyPhoto,
    Screenshot,
    LineArt,
    TextScan,
    FlatIllustration,
    Lossless,
}
```

The final merged path should be:

```text
image decode / color convert in bpg-rs
→ feature analysis
→ x265-rs still encoder
→ raw HEVC access unit
→ bpg-rs SPS rewrite / BPG container
→ .bpg
```

## The most important deletion: inter prediction

The biggest simplification is deleting the inter encoder.

Final still-image x265-rs should not need:

```text
motion.cpp
motion.h
reference.cpp
reference.h
weightPrediction.cpp
large parts of search.cpp
large parts of analysis.cpp
most of dpb.cpp
most of slicetype.cpp
most of ratecontrol.cpp
```

This is where your performance and maintainability win comes from. You are not porting a video encoder; you are extracting x265’s excellent **intra still-picture encoder**.

## The dangerous parts

The hard parts are not the CLI, image I/O, or BPG container. The hard parts are:

1. **CABAC correctness**
   One wrong context update or bin write produces invalid or subtly worse streams.

2. **CU/TU recursion**
   HEVC’s recursive block decision logic is complex. Port it with tests around each recursion level.

3. **Primitive ABI**
   The asm functions assume exact layout, alignment, bit depth, and stride behavior.

4. **Bit depth variants**
   x265 is compile-time specialized for 8/10/12-bit. Your Rust design should avoid infecting the whole codebase with runtime branching. Prefer const generics or separate monomorphized modules.

5. **BPG profile mismatch**
   BPG supports up to 14 bits, but x265’s practical encoder paths are 8/10/12-bit. x265’s own headers list `main444-16-intra` and `main444-16-stillpicture` as not supported in the source you uploaded. So the realistic first targets are 8-bit, 10-bit, and maybe 12-bit.

6. **Licensing/patents**
   x265 is GPLv2 or commercial-license software, and the x265 docs explicitly note that the software license does not cover HEVC patent rights. ([x265 Documentation][5]) A Rust port derived from x265 inherits the same licensing reality.

## Presets worth building

For your actual BPG archival goal, I would design presets around source type rather than x265-style video names:

### `archival-photo`

Default photo mode.

```text
10-bit internal
4:4:4 or 4:2:0 depending user choice
SAO conservative/on
deblock weak/default
psy-rd enabled
RDOQ enabled
no preprocessing
spatial AQ enabled
```

### `grainy-photo`

For high-ISO/noisy photos.

```text
avoid aggressive local QP swings
weaker deblock
SAO tested both ways
less tendency to erase grain
maybe larger files accepted
```

x265 already has grain-related tuning for video ratecontrol, but for still images I would not copy it directly. I would make a still-image grain preset.

### `screenshot-ui`

For UI, signs, maps, text overlays, and hard edges.

```text
prefer 4:4:4
SAO off or conservative
deblock off/weak
transform-skip evaluation more important
protect edges with QP offsets
```

### `line-art-scan`

For manga, drawings, scanned line art.

```text
4:4:4 or grayscale
SAO off
deblock off/weak
strong edge preservation
possibly lossless or near-lossless
```

### `flat-illustration`

For anime/vector-like images.

```text
protect gradients
avoid ringing around flat-color boundaries
test SAO carefully
10/12-bit useful
```

### `lossless`

x265 has true lossless mode that bypasses scaling, transform, quantization, and in-loop filter processes; slower presets generally produce smaller lossless streams. ([x265 Documentation][3]) For BPG-rs, expose this as a real archival preset, not just `qp=0`.

## Recommended implementation plan

### Phase 1: stock x265 oracle inside bpg-rs

Use stock x265 4.1 as an internal comparison backend. Feed it the same planes bpg-rs will feed x265-rs. Save:

```text
input image
x265 params
raw HEVC
final BPG
decoded PNG
metrics
file size
```

This becomes your regression corpus.

### Phase 2: Rust HEVC syntax layer

Port:

```text
bitstream
NAL
VPS/SPS/PPS
profile/level
minimal slice header
CABAC shell
```

Goal: valid headers and a skeleton stream.

### Phase 3: primitive layer

Build the assembly bridge before porting the heavy encoder logic. Make tests for:

```text
SATD
SAD
DCT/IDCT
quant/dequant
intra prediction
SAO kernels
loop filter kernels
pixel copy/average/variance
```

Do not proceed until primitive results match stock x265 for fixed inputs.

### Phase 4: single CTU encode

Encode one 64×64 block, intra-only, one color format, one bit depth, no SAO/deblock.

Test images:

```text
flat gray
horizontal gradient
checkerboard
single black line
noise patch
small natural crop
```

### Phase 5: full image, one bit depth

Support a full 10-bit 4:4:4 or 4:2:0 image. Still no inter, no lookahead, no full ratecontrol.

At this point you should be able to produce a BPG that decodes.

### Phase 6: quality parity pass

Compare against stock x265 with all-intra settings:

```text
size ratio
PSNR/SSIM/MS-SSIM or preferred metric
visual diffs
decode validity
speed
```

Do not tune yet. First get boring parity.

### Phase 7: remove video leftovers

Only after parity:

```text
delete/stub DPB
delete slicetype/lookahead
delete motion/reference/weighted prediction
replace ratecontrol
simplify threading
remove CLI-shaped param parser
```

This avoids deleting something that was accidentally still needed.

### Phase 8: still-image tuning

Now use the Rust-native structure to add the parts that are hard to do through x265 CLI options:

```text
source classifier
per-block QP offsets
preset-specific SAO/deblock choices
preset-specific RDO decisions
direct BPG chroma/bit-depth policy
target-size search
feature-driven 4:2:0 vs 4:4:4 choice
```

This is where the port becomes more than a translation.

## The most important design rule

Keep the first working encoder **boringly faithful** to x265.

Then make it image-native.

A premature “clean Rust redesign” of `analysis.cpp` and `search.cpp` will probably lose compression efficiency. But a staged port lets you keep the algorithmic value of x265 while removing the parts that only exist because x265 is a video encoder.

The end state should be:

```text
bpg-rs + x265-rs
= private, Rust-native, still-image HEVC/BPG encoder
= x265 intra quality and assembly speed
= no video baggage
= direct image-aware tuning surface
```

That is a coherent project. It is ambitious, but the still-image restriction makes it much more realistic than a full x265 clone.

[1]: https://bellard.org/bpg/ "BPG Image format"
[2]: https://x265.readthedocs.io/en/master/api.html "Application Programming Interface — x265  documentation"
[3]: https://x265.readthedocs.io/en/master/cli.html "Command Line Options — x265  documentation"
[4]: https://github.com/videolan/x265/blob/master/source/x265.h "x265/source/x265.h at master · videolan/x265 · GitHub"
[5]: https://x265.readthedocs.io/en/stable/introduction.html "Introduction — x265  documentation"
