Yes, your revised target makes more sense: **encoder first**, but split into two encoder meanings:

1. **BPG encoder wrapper / muxer** — definitely worth porting.
2. **HEVC encoder backend** — keep `x265` as FFI, but consider porting the JCTVC/HM reference encoder if your goal is high-compression, slower-but-better BPG output.

The uploaded source confirms that `jctvc` is not assembly-heavy. The assembly-heavy part is `x265`, and that is exactly the part you should not port.

## Revised feasibility

| Piece                                | Port to Rust? | Reason                                                                 |
| ------------------------------------ | ------------: | ---------------------------------------------------------------------- |
| BPG header writer                    |           Yes | Small, format-specific, straightforward                                |
| BPG metadata/extensions              |           Yes | Simple enough                                                          |
| Color conversion / chroma conversion |           Yes | Manageable and useful                                                  |
| Alpha extraction / W-plane handling  |           Yes | BPG-specific and not too large                                         |
| Modified HEVC/SPS header packing     |           Yes | Bitstream fiddly, but contained                                        |
| x265 backend                         |   No, use FFI | Already optimized C++/asm                                              |
| JCTVC/HM reference encoder           |     Maybe yes | Plain C++ reference code, slow/stateful, possible modernization target |
| Native x265-like encoder             |            No | Too large, too optimized, wrong target                                 |

So the architecture should probably be:

```text
bpg-enc
  ├── bpg-format        // BPG container/header/extension writer
  ├── bpg-image         // planar image model, chroma conversion, alpha split
  ├── bpg-hevc-mux      // modified SPS/header extraction/packing
  ├── bpg-x265          // FFI backend
  └── bpg-jctvc-rs      // optional native reference encoder backend
```

The key point: **porting the BPG encoder does not require porting x265**. The BPG-specific encoder layer is mostly image preparation plus HEVC bitstream packaging.

## What `jctvc_glue.cpp` currently does

The existing JCTVC path is very crude. It does not integrate JCTVC as a clean library.

It does this:

```text
BPG Image
  → save temporary .yuv file
  → construct fake argv[] command-line options
  → call TAppEncTop::parseCfg(argc, argv)
  → call TAppEncTop::encode()
  → read temporary .bin HEVC bitstream
  → delete temp files
  → return bytes
```

That is a good sign for your porting goal, because the integration layer is bad in obvious ways.

Low-hanging improvements before even touching HEVC algorithms:

* remove temp YUV file I/O
* remove fake CLI parsing
* expose a real `encode_frame(ImagePlanes) -> Vec<u8>` API
* separate config from mutable encoder state
* remove global analysis/debug objects
* remove video-GOP assumptions for still-image mode
* make alpha/color encodes independent
* make per-image encoding reentrant
* make the bitstream output an in-memory writer from the start

This alone would make a Rust port cleaner than the original C++ integration.

## Is JCTVC worth porting?

Potentially, yes, but with one caveat:

> JCTVC is slow partly because the codebase is old and stateful, but also because reference HEVC encoding is algorithmically expensive.

Rust can fix structure, safety, threading, and data flow. It will not magically make full RD-search HEVC encoding cheap. But your idea is still reasonable because BPG has a narrower target than general HEVC video.

For BPG, you can initially ignore or defer a lot:

```text
Keep:
  intra image coding
  4:2:0 / 4:2:2 / 4:4:4
  8–14 bit
  lossless mode
  transform skip
  cross-component prediction
  SAO / deblocking options
  CABAC encoder

Defer:
  inter prediction
  motion estimation
  weighted prediction
  rate control
  complex GOP structures
  video IO
  MD5 SEI
  animation P-frame mode
```

That changes the task from “port a full HEVC encoder” to “port the still-image intra path of the HM encoder.”

That is still big, but much more plausible.

## Is the encoder harder than the decoder?

Usually, yes.

A decoder mostly follows the bitstream. An encoder must decide:

* coding unit partitioning
* prediction modes
* transform unit sizes
* quantization choices
* RD-cost tradeoffs
* transform skip decisions
* SAO decisions
* entropy context estimation
* lossless bypass decisions

But for a BPG-first port, you can make it less brutal by targeting the JCTVC path exactly rather than designing a new encoder. You are not inventing the search; you are porting and then simplifying it.

The hard part is not assembly. The hard part is turning stateful C++ encoder machinery into clean Rust without changing behavior too much.

## Best practical path

### Phase 1: Port the BPG encoder wrapper first

This should be first. It gives you a working Rust crate even before JCTVC is native.

Implement:

```rust
pub trait HevcEncoder {
    fn open(params: HevcEncodeParams) -> Result<Self, Error>
    where
        Self: Sized;

    fn encode_frame(&mut self, image: &ImagePlanes) -> Result<(), Error>;

    fn finish(self) -> Result<Vec<u8>, Error>;
}
```

Then implement:

```rust
pub struct BpgEncoder<E: HevcEncoder> {
    params: BpgEncodeParams,
    hevc: E,
    alpha_hevc: Option<E>,
}
```

The BPG layer should own:

* input image normalization
* alpha split
* chroma conversion
* padding to coding-block size
* bit-depth conversion
* BPG header writing
* extension writing
* frame-duration table
* final modified HEVC payload construction

This creates a stable Rust API:

```rust
let bpg = BpgEncoder::<X265Backend>::new(params)?
    .encode_image(&rgba)?
    .finish()?;
```

At this point, `X265Backend` can be FFI. You now have a Rust BPG encoder without touching JCTVC yet.

### Phase 2: Port `build_modified_hevc`, `build_modified_sps`, NAL handling

This is important because it is BPG-specific. In `bpgenc.c`, the encoder strips/repackages HEVC headers so the BPG file stores a compact modified HEVC payload.

Port these pieces early:

```text
extract_nal
escape/unescape NAL payload
Exp-Golomb reader/writer
build_modified_sps
build_modified_hevc
frame-duration payload packing
```

This part is bit-level but contained. It is exactly the kind of code where Rust helps.

### Phase 3: Keep x265 as the correctness backend

Use x265 FFI as your stable backend while you port the rest.

That gives you:

```text
Rust BPG encoder
  → Rust image prep
  → x265 FFI
  → Rust modified HEVC wrapper
  → valid .bpg
```

This becomes your oracle for the BPG-side logic. If this works, any later JCTVC issue is isolated to the HEVC encoder backend.

### Phase 4: Port JCTVC as an intra-only backend

Do not begin with the whole JCTVC app.

Start from the call graph needed for:

```text
single frame
intra-only
no file IO
fixed QP
fixed chroma format
fixed bit depth
in-memory output
```

The minimal target is:

```rust
pub struct JctvcIntraEncoder {
    cfg: Arc<EncConfig>,
    sps: Sps,
    pps: Pps,
    cabac_tables: CabacTables,
}

impl HevcEncoder for JctvcIntraEncoder {
    fn encode_frame(&mut self, image: &ImagePlanes) -> Result<(), Error> {
        // encode one intra picture
    }

    fn finish(self) -> Result<Vec<u8>, Error> {
        // return Annex-B or length-prefixed HEVC payload as needed
    }
}
```

Avoid porting `TAppEncTop` as the main abstraction. Treat it as a reference for wiring, not the Rust design.

## JCTVC port order

I would port in this order:

```text
1. Basic types
   CommonDef, TypeDef equivalents, chroma format, component IDs

2. Image buffers
   TComPicYuv → Plane<T>, Picture, PictureBuffer

3. Bitstream writer
   TComBitStream, NALwrite, AnnexBwrite

4. Parameter sets
   SPS, PPS, slice structures

5. CABAC encoder
   TEncBinCoderCABAC
   TEncSbac
   ContextModel
   ContextTables

6. Transform / quant
   TComTrQuant

7. Prediction
   TComPrediction
   intra prediction only first

8. CU/TU structure
   TComDataCU
   TComTU
   TComPicSym

9. RD cost
   TComRdCost

10. Search
   TEncSearch, but only intra search initially

11. CU encoder
   TEncCu

12. Slice / picture encoder
   TEncSlice
   TEncGOP, reduced to still-image flow

13. SAO / loop filter
   TEncSampleAdaptiveOffset
   TComLoopFilter
```

Do not start from `TAppEncTop.cpp`. Start from a narrow top-level still-image encoder and pull dependencies upward.

## Where threading can realistically help

You are right that JCTVC is single-threaded and archaic, but HEVC encoder parallelism has constraints.

### Easy parallelism

These are straightforward:

```text
alpha plane encode and color encode
multiple images
multiple animation frames, if intra-only
image conversion
chroma conversion
padding
candidate precomputation
cost table setup
```

### Medium parallelism

These can help but need care:

```text
SAO analysis per CTU/region
transform candidate evaluation
intra mode candidate scoring
RDO trials with cloned local contexts
```

A Rust port can make this cleaner by using local temporary state instead of mutating giant shared encoder objects.

### Hard parallelism

These are constrained:

```text
CABAC final bitstream writing
CTU traversal with left/top dependencies
intra prediction using reconstructed neighboring samples
deblocking dependencies
wavefront row synchronization
```

For still images, the realistic threading model is:

```text
parallel analysis / mode search
  → choose modes
  → serial or wavefront-constrained final entropy encode
```

or:

```text
tiles/slices in parallel
  → independent entropy contexts
  → compression ratio penalty
```

If you want exact JCTVC-like compression, avoid too many independent tiles/slices. If you accept a small compression penalty, tiles/slices give much simpler parallelism.

## What “rescuing JCTVC” should mean

I would not try to preserve the class structure. Preserve behavior, not architecture.

Replace:

```cpp
TEncTop
TEncGOP
TEncSlice
TEncCu
TEncSearch
TComDataCU
TComPic
TComPicYuv
global config/debug/analyze state
```

with more Rust-like structures:

```rust
struct EncConfig { ... }        // immutable after construction
struct SequenceParams { ... }   // SPS/PPS/VPS material
struct Picture { ... }          // planes + reconstructed planes
struct Ctu { ... }              // per-CTU state
struct CtuScratch { ... }       // reusable temp buffers
struct CabacWriter<W> { ... }
struct IntraSearch { ... }
struct TransformQuant { ... }
struct RdCost { ... }
struct PictureEncoder { ... }
```

The main design rule:

> Make analysis state local, make reference/config state immutable, and make final bitstream state explicit.

That is what enables threading later.

## Expected payoff

A cleaned-up Rust JCTVC intra encoder could plausibly improve over stock JCTVC because:

* no temp file round trips
* no fake CLI config path
* less virtual/class indirection
* fewer global mutable objects
* better cache-aware buffer layout
* Rayon-style parallel analysis
* still-image-only simplification
* easier feature gating

But it probably will not approach x265 speed unless you also implement heavy SIMD and years of encoder heuristics.

The likely useful niche is:

```text
x265 backend:
  fast, production default

jctvc-rs backend:
  slower, high-compression / experimental / archival mode

simple-rs backend later:
  maybe faster than jctvc-rs but lower compression
```

## The rough path I would use

1. **Port BPG muxer/writer.**
2. **Add x265 FFI backend.**
3. **Get valid `.bpg` output from Rust.**
4. **Port modified HEVC header construction.**
5. **Replace C image conversion with Rust.**
6. **Create `HevcEncoder` trait.**
7. **Stub `JctvcRsEncoder` behind the same trait.**
8. **Port JCTVC intra-only path.**
9. **Match JCTVC output approximately, not byte-for-byte at first.**
10. **Add alpha/lossless/high-bit-depth.**
11. **Add parallel intra search / SAO analysis.**
12. **Only later consider animation/inter support.**

## Bottom line

Your idea is more reasonable than “port all of HEVC.”

The good division is:

* **Do not port x265.** Its optimized assembly is an asset, not a blocker.
* **Do port the BPG encoder wrapper.** That is the obvious first win.
* **Consider porting JCTVC as a Rust intra-image reference encoder.** Not because it will be easy, but because its current integration and architecture are poor enough that a Rust rewrite could make it meaningfully more usable.
* **Do not start with full video HEVC.** Start with still-image intra BPG and grow outward.

So yes: encoder-first is the better plan, and JCTVC is the only HEVC backend in this tree that is plausibly worth “rescuing” into Rust.


Yes. There is probably a **real third-space opportunity** here.

Not “invent a new codec,” because BPG compatibility locks you into legal HEVC syntax. But a **BPG-specialized HEVC still-image encoder** could plausibly beat one or both existing backends on the dimensions you care about:

```text
x265:
  very fast, heavily optimized, video-oriented, heuristic-heavy

JCTVC/HM:
  very slow, reference-oriented, exhaustive-ish, compression-strong

BPG-specialized encoder:
  still-image-only, intra-only, BPG-aware, possibly parallel, less general than both
```

The opportunity exists because neither x265 nor JCTVC is optimized specifically for the problem:

> Encode one still image into an HEVC intra bitstream, then wrap it as BPG, with no concern for real-time video, GOP structure, streaming latency, VBV, motion search, scene changes, or long-running rate control.

x265 documentation itself describes the speed/compression tradeoff: faster presets take shortcuts, slower presets test more options to improve compression efficiency. ([x265.readthedocs.io][1]) BPG’s own README notes that x265 and JCTVC optimize different objective measures: x265 is tuned by default for SSIM, while JCTVC is tuned for PSNR. ([GitHub][2]) That gap alone suggests there is room for a still-image-specific encoder that chooses its own metric and search strategy.

## The shape of the opportunity

The possible third encoder is not:

```text
generic HEVC video encoder
```

It is:

```text
HEVC intra still-image encoder specialized for BPG
```

That narrower target lets you throw away huge parts of video encoder complexity:

```text
discard:
  motion estimation
  inter prediction
  B-frame/P-frame logic
  lookahead
  VBV / streaming rate control
  scene cut logic
  temporal AQ
  reference frame management
  GOP machinery
  most threading constraints caused by temporal prediction

keep:
  CTU/CU partition decisions
  intra prediction mode search
  transform decisions
  quantization
  CABAC encoding
  SAO / deblocking decisions
  chroma format decisions
  bit-depth handling
  lossless / near-lossless paths
```

That is still a serious encoder, but it is not x265.

## Where it might beat x265

A BPG-specific encoder could beat x265 in compression because x265 is carrying assumptions from video.

Potential advantages:

### 1. Still-image-first RD model

x265’s heuristics are designed around video perception and throughput. A BPG encoder could optimize for still-image metrics:

```text
PSNR
SSIM
MS-SSIM
Butteraugli-like perceptual distance
edge preservation
text/line art preservation
e-ink readability, if relevant to your use case
```

For still images, annoying artifacts are different from video artifacts. There is no motion masking. Fine edges, ringing, flat areas, and chroma bleed matter more.

### 2. More aggressive per-region decisions

A still image can be classified spatially:

```text
photo texture
flat graphic
text
line art
screen capture
dark low-noise area
high-detail foliage
skin / smooth gradient
```

Then choose different encoding search behavior per region.

For example:

```text
text / line art:
  favor 4:4:4
  preserve edges
  allow smaller transform blocks
  use stronger mode search

photo texture:
  allow larger transform blocks
  stronger chroma subsampling
  more quantization tolerance

flat graphics:
  exploit large blocks
  avoid unnecessary residual detail
  protect gradients from banding
```

x265 has adaptive quantization and psy tools, but it is not designed as a still-image semantic classifier.

### 3. BPG-aware color and alpha handling

BPG has specific modes: grayscale, YCbCr, RGB, YCgCo, alpha, high bit depth, CMYK-like W-plane handling. A BPG-specific encoder could make better front-end decisions before HEVC sees the data:

```text
choose YCbCr vs RGB vs YCgCo per image
choose 4:2:0 vs 4:2:2 vs 4:4:4 intelligently
choose alpha strategy
choose premultiplied alpha or straight alpha behavior
choose lossless for alpha but lossy for color
choose bit-depth preservation or downconversion
```

This is not “inside HEVC” exactly. It is encoder-front-end intelligence. But it affects final BPG efficiency a lot.

### 4. Better still-image rate allocation

Video rate control is the wrong mental model for one image.

A still-image encoder could do:

```text
first pass:
  analyze image complexity
  classify CTUs
  estimate perceptual importance

second pass:
  allocate bits spatially
  choose QP offsets
  decide which blocks deserve expensive search
```

There is academic work showing that better HEVC rate-control and pre-analysis can improve RD performance; one x265-focused paper reported a 10.3% BD-rate gain from improved use of pre-analysis and cost-guided rate control. ([arXiv][3]) That is video-oriented, but it supports the general point: x265’s default choices are not an optimal endpoint.

### 5. Better exhaustive search budget placement

JCTVC spends too much effort everywhere. x265 saves time everywhere.

A third encoder could spend effort selectively:

```text
expensive search only where it matters
cheap mode decision in simple flat areas
parallel candidate evaluation
early exit only when confidence is high
use JCTVC-like exhaustive search for ambiguous blocks
```

That could plausibly beat JCTVC in speed while approaching its compression, or beat x265 in compression while staying much faster than JCTVC.

There is also research on learned fast HEVC intra coding: one paper describes using a shallow CNN to reduce intra-mode complexity by up to 75.2% with negligible RD loss. ([arXiv][4]) You would not need to start with ML, but that points at the useful search space: predict which CU splits and intra modes are worth testing.

## Where it might beat JCTVC

JCTVC’s weakness is not just “it searches more.” It is also old architecture.

A BPG-specific Rust encoder could beat JCTVC on speed by removing:

```text
fake command-line configuration path
temporary YUV files
video sequence assumptions
single-threaded picture flow
global mutable state
class-heavy scratch structures
unused inter prediction machinery
debug/reference-model baggage
genericity for many configurations BPG does not need
```

More importantly, it can specialize:

```text
one picture
known chroma format
known bit depth
known profile target
known BPG wrapper constraints
fixed GOP size 1
no inter frames
no decoded picture buffer complexity
```

A JCTVC-derived Rust encoder could keep the strong intra/RD logic but reorganize it as:

```text
immutable config
per-CTU scratch
parallel analysis
explicit final entropy encode
arena-backed temporary buffers
cache-friendly planes
feature-gated tools
```

This could be much faster than JCTVC without becoming x265.

## The non-see-saw part

You are right that it is not necessarily a simple see-saw.

The x265/JCTVC endpoints are not mathematically optimal endpoints. They are historical artifacts:

```text
x265:
  optimized production video encoder

JCTVC/HM:
  standard reference model

BPG:
  still-image format that reuses HEVC
```

That leaves unoccupied territory:

```text
fast-ish still-image HEVC encoder
high-compression still-image HEVC encoder
perceptual still-image HEVC encoder
line-art/text-aware HEVC encoder
lossless/near-lossless BPG-specialized encoder
hybrid x265/JCTVC decision encoder
```

The best opportunity is probably **not** to write a full encoder from scratch. It is to build a new encoder around selective reuse of JCTVC’s intra machinery, but with a narrower BPG-specific control layer.

## Possible design directions

### Direction A: JCTVC-rs, cleaned and parallelized

This is the most direct route.

```text
start from JCTVC/HM
keep conformance behavior
delete video-oriented machinery
rewrite image-only intra path in Rust
parallelize analysis
serial or wavefront final CABAC
```

Goal:

```text
same or near-same size as JCTVC
much faster than JCTVC
still slower than x265
```

This is probably the safest “third encoder.”

### Direction B: x265 plus BPG-specific pre-analysis

This is lower effort.

```text
analyze image in Rust
choose x265 params per image
possibly per-zone QP maps if accessible
choose color/chroma mode better
choose psy/SSIM/PSNR tuning better
choose lossless/near-lossless modes better
```

Goal:

```text
better than naive x265 BPG
same x265 backend speed profile
no native HEVC encoder yet
```

This would help you learn where compression is being lost before committing to a full encoder port.

### Direction C: Hybrid encoder: JCTVC decisions, custom fast path

This is more ambitious.

Use JCTVC-like exhaustive decision making as a “teacher,” then build faster heuristics:

```text
run JCTVC on corpus
collect chosen CU splits, intra modes, transform decisions
train or derive heuristics
implement fast Rust mode predictor
fall back to exhaustive search for uncertain blocks
```

Goal:

```text
x265-ish speed class on simple areas
JCTVC-ish decisions on hard areas
```

This is where ML or handcrafted classifiers could enter.

### Direction D: Still-image psychovisual BPG encoder

This would not necessarily beat PSNR, but could beat visual quality per byte.

```text
optimize for still-image perceptual quality
protect edges and text
spend bits on gradients
avoid color bleeding
region-aware chroma decisions
```

Goal:

```text
better subjective BPG at same file size
especially for scanned documents, screenshots, illustrations, e-ink images
```

This is probably relevant to your Lege background.

### Direction E: Lossless / near-lossless specialist

BPG supports lossless modes. JCTVC is reported by libbpg as producing smaller images than x265 in lossless mode. A dedicated still-image lossless HEVC encoder might have a clearer target than lossy perceptual encoding.

Focus on:

```text
transform skip
residual DPCM
block partitioning for edges
YCgCo/RGB mode choice
alpha-specific compression
screen/text images
```

Goal:

```text
beat x265 lossless
match or beat JCTVC lossless faster
```

This might be the most technically concrete subproject.

## What a BPG-native encoder would exploit

A normal video encoder optimizes for a huge parameter space. A BPG-native encoder can hard-code assumptions:

```text
only intra
usually one frame
no latency constraints
can run two or three analysis passes
can use whole-image statistics
can classify regions globally
can tune chroma by image type
can spend more time on headers/mode decisions if bytes saved matter
can choose BPG color mode before HEVC encode
can use separate policies for alpha plane
```

That is where the opportunity lives.

## A plausible architecture

```text
bpg-enc-rs
  Image analyzer
    detects photo/text/line/flat/gradient regions
    estimates local complexity
    chooses color mode and chroma format

  Policy engine
    chooses QP map
    chooses search depth per CTU
    chooses lossless/lossy alpha behavior
    chooses deblock/SAO policy

  HEVC intra core
    CU partition search
    intra prediction mode search
    transform/quant
    CABAC writer

  BPG muxer
    BPG header
    modified HEVC payload
    metadata
```

The important design decision is that the encoder should not have a single global preset like:

```text
fast / slow / veryslow
```

It should have a per-region effort map:

```text
flat area:
  low effort, large blocks

edge/text:
  high effort, smaller blocks

texture:
  medium effort, perceptual quantization

gradient:
  protect against banding
```

## What to test first

Before writing a third encoder, I would run experiments to identify where gains exist.

### Experiment 1: Backend comparison grid

For a corpus:

```text
photos
scans
screenshots
line art
mixed text/photo pages
alpha PNGs
synthetic gradients
```

Encode with:

```text
x265 presets / QPs
JCTVC QPs
lossless modes
4:2:0 / 4:4:4
YCbCr / RGB / YCgCo
SAO on/off
deblock on/off
transform skip on/off
```

Measure:

```text
file size
PSNR
SSIM
MS-SSIM if available
decode output visual artifacts
encode time
```

This will show whether JCTVC wins generally or only in certain image classes.

### Experiment 2: x265 parameter mining

Before porting JCTVC, try to close the gap with better x265 parameters.

If x265 can get within 5–10% of JCTVC by changing settings, the native encoder target changes. If it remains 20–25% worse, the third encoder case gets stronger.

### Experiment 3: JCTVC profiling

Profile JCTVC on still images.

You need to know whether time goes mostly into:

```text
intra mode search
CU partitioning
transform/quant
CABAC bit estimation
SAO
I/O/config overhead
```

If 80% is in intra search, Rust cleanup alone will not solve it, but smarter pruning/parallelization might. If significant time is infrastructure overhead, the port has immediate payoff.

### Experiment 4: JCTVC decision extraction

Patch JCTVC to dump:

```text
chosen CU sizes
chosen intra modes
QP per block
SAO decisions
transform sizes
bits per CTU
distortion per CTU
```

Then inspect patterns. This becomes the map for a Rust replacement.

## A realistic target

The realistic win is probably one of these:

```text
Target 1:
  80–95% of JCTVC compression gain over x265
  3–10x faster than JCTVC

Target 2:
  same compression as x265
  better still-image visual quality

Target 3:
  better than x265 lossless
  much faster than JCTVC lossless

Target 4:
  image-class-specific wins:
    screenshots, scans, line art, alpha PNGs
```

The unrealistic target is:

```text
always smaller than JCTVC
always faster than x265
```

That would mean beating a highly optimized production encoder on speed while also beating a slow exhaustive reference encoder on compression. Possible in a narrow class, not realistic as a universal claim.

## Bottom line

Yes: there is likely a genuine optimization space between x265 and JCTVC because BPG is using HEVC for a narrow still-image use case neither encoder was primarily designed for.

The most promising third encoder shape is:

```text
JCTVC-derived intra HEVC encoder
  narrowed to BPG still images
  redesigned in Rust
  region-aware
  parallel analysis
  BPG-aware color/chroma/alpha front end
  x265 kept as the fast backend
```

That could plausibly beat x265 in compression, beat JCTVC in speed, and maybe beat one or both on certain image classes. The opportunity is not in changing BPG or HEVC syntax; it is in making better encoder decisions for still images under BPG’s constraints.

[1]: https://x265.readthedocs.io/en/stable/presets.html?utm_source=chatgpt.com "Preset Options - x265 Documentation"
[2]: https://github.com/mirrorer/libbpg?utm_source=chatgpt.com "libbpg (copy of the source code for easy access)"
[3]: https://arxiv.org/abs/2109.12294?utm_source=chatgpt.com "Revisiting Pre-analysis Information Based Rate Control in x265"
[4]: https://arxiv.org/abs/1907.02287?utm_source=chatgpt.com "Learned Fast HEVC Intra Coding"


Yes. Treat this as two projects that share the same core:

1. **Direction A:** a cleaned, idiomatic Rust port of the JCTVC/HM still-image intra encoder.
2. **Direction C:** a later hybrid encoder that uses JCTVC-rs as a “teacher” to build faster decision heuristics.

The important architectural rule is: **do not port x265, but do port the BPG layer, the HEVC/BPG wrapper, the image pipeline, and the JCTVC intra path.** x265 stays as an FFI backend and benchmark/oracle. The HM/JCTVC reference software exists to aid study, conformance, interoperability, and standard implementation work, while libbpg’s README notes that its JCTVC backend is slower but can be more efficient than x265 and that the two are tuned differently, with x265 defaulting to SSIM and JCTVC to PSNR. ([GitHub][1])

## Target architecture

Use this shape from the beginning:

```text
bpg-rs/
  crates/
    bpg-format/
      BPG header, extension data, metadata, ue7, container writer/parser

    bpg-image/
      image planes, color conversion, chroma subsampling, alpha split, padding

    bpg-hevc/
      NAL parsing, RBSP/EBSP handling, bit writer/reader, modified SPS builder,
      BPG modified HEVC payload writer

    bpg-encode/
      public BPG encoder API, backend-agnostic orchestration

    bpg-x265/
      x265 FFI backend; not pure Rust, not default dependency unless enabled

    bpg-jctvc-rs/
      Rust port of the still-image JCTVC/HM encoder path

    bpg-tools/
      CLI, corpus testing, comparison harness
```

Public API:

```rust
pub trait HevcStillEncoder {
    fn encode_still(&mut self, image: &ImagePlanes, params: &HevcEncodeParams)
        -> Result<HevcBitstream, EncodeError>;
}

pub enum Backend {
    X265,
    JctvcRs,
}

pub struct BpgEncoder {
    params: BpgEncodeParams,
    backend: Backend,
}
```

The BPG layer should not care whether the HEVC payload came from x265 or JCTVC-rs.

## What to port first

The current libbpg encoder structure gives you a clean split. In the uploaded source, `bpgenc.h` exposes the backend abstraction:

```c
HEVCEncoderContext *(*open)(const HEVCEncodeParams *params);
int (*encode)(HEVCEncoderContext *s, Image *img);
int (*close)(HEVCEncoderContext *s, uint8_t **pbuf);
```

Keep that abstraction, but make it Rust-native:

```rust
trait HevcEncoder {
    fn open(params: HevcEncodeParams) -> Result<Self, Error>
    where
        Self: Sized;

    fn encode_frame(&mut self, img: &ImagePlanes) -> Result<(), Error>;

    fn finish(self) -> Result<Vec<u8>, Error>;
}
```

The first deliverable should be:

```text
Rust BPG encoder
  → Rust image/color/alpha/padding logic
  → x265 FFI backend
  → Rust BPG modified HEVC writer
  → valid .bpg output
```

That gives you a working encoder before touching JCTVC.

## Phase 1: Port the BPG-specific encoder layer

Port these before JCTVC:

```text
BPG header writer
extension data writer
animation control extension
metadata preservation
alpha plane extraction
W-plane / alpha flags
image padding
chroma subsampling
bit-depth conversion
NAL start-code scanner
emulation-prevention removal
Exp-Golomb reader/writer
BPG modified SPS builder
BPG modified HEVC payload writer
```

This is the `bpgenc.c` material around:

```text
save_yuv1
extract_nal
find_nal_end
put_ue
get_ue_golomb
put_ue_golomb
build_modified_sps
build_modified_hevc
bpg_encoder_encode
bpg_encoder_encode_trailer
```

This is high-value and low-risk. It also isolates the hard encoder backend work from BPG packaging bugs.

### Tests for Phase 1

Use original `bpgenc` as the oracle:

```text
same input image
same x265 backend
old bpgenc output
new Rust bpg-rs output
decode both with stock bpgdec
compare pixels
compare BPG header fields
compare modified SPS parse result
```

Do not require byte-identical files. Require decode-identical pixels and valid metadata.

## Phase 2: Keep x265 as the fast backend

Do a minimal, clean x265 backend:

```rust
pub struct X265Backend {
    api: *const x265_api,
    encoder: *mut x265_encoder,
    picture: *mut x265_picture,
}
```

Feature gate it:

```toml
[features]
x265 = []
jctvc-rs = []
```

The current `x265_glue.c` config is small and should be mirrored closely:

```text
CQP mode
qp = params.qp
bLossless = params.lossless
internalCsp = I400/I420/I422/I444
keyframeMax = 1 for intra-only
totalFrames = 1 for still image
bRepeatHeaders = 1
bEmitInfoSEI = 0
preset = compress_level
tune = "ssim" currently
```

This gives you a stable backend for daily testing while JCTVC-rs is incomplete.

## Direction A: JCTVC-rs roadmap

Do **not** port `TAppEncTop` as the main design. In the uploaded source, `jctvc_glue.cpp` currently writes a temporary YUV file, constructs fake command-line args, calls `TAppEncTop::parseCfg`, calls `encode()`, then reads a temporary `.bin`. That is exactly the stateful infrastructure you want to eliminate.

Instead, design the Rust encoder around a direct in-memory still-image path:

```rust
pub struct JctvcStillEncoder {
    cfg: Arc<EncConfig>,
    seq: SequenceParams,
    scratch_pool: ScratchPool,
}

impl HevcEncoder for JctvcStillEncoder {
    fn encode_frame(&mut self, img: &ImagePlanes) -> Result<(), Error> {
        self.encode_intra_picture(img)
    }

    fn finish(self) -> Result<Vec<u8>, Error> {
        Ok(self.bitstream.into_bytes())
    }
}
```

### A1. Define the narrow initial target

Start with the BPG/JCTVC still-image settings already used by libbpg:

```text
single image
intra-only
GOPSize = 1
IntraPeriod = 1
Profile = main_444_16_intra
fixed QP / CQP
no temp YUV files
no CLI parser
no rate-control pass
no inter prediction
no animation P-frames
no decoded-picture-hash initially
8-bit first
4:2:0 first or 4:4:4 first, but not all formats at once
```

Given BPG’s still-image use, I would start with **8-bit 4:4:4 intra** because it avoids chroma subsampling complexity inside early tests, then add 4:2:0/4:2:2 after the core path works.

### A2. Port bottom-up, not top-down

Porting order:

```text
1. Common enums and constants
   ChromaFormat, ComponentId, ChannelType, SliceType, PredMode,
   text_type, part_size, scan order constants.

2. Plane and picture storage
   TComPicYuv → Plane<T>, PicturePlanes, ReconstructedPicture.
   Use contiguous Vec<T> with explicit stride.

3. Bitstream writer
   TComBitStream, NALwrite, Annex-B writer.
   This is needed early because every test depends on valid NAL output.

4. Parameter sets
   VPS/SPS/PPS/slice structures.
   Emit the same constrained BPG-compatible parameter sets.

5. CABAC encoder core
   ContextModel, ContextModel3DBuffer,
   TEncBinCoderCABAC, TEncSbac, TEncEntropy.

6. Transform and quantization
   TComTrQuant.
   Start with one bit depth and one transform size path.
   Then expand.

7. Intra prediction
   TComPrediction and intra mode primitives.
   Implement luma first, then chroma.

8. RD cost
   TComRdCost.
   Build a clear `DistortionMetric` abstraction:
     SSE/PSNR first, later SSIM/perceptual variants.

9. CU/TU structures
   TComDataCU, TComTU, TComPicSym.
   This is where you should simplify aggressively.

10. Intra search
   TEncSearch, intra-only subset.
   This is the central compression-quality module.

11. CU encoder
   TEncCu.
   Keep only intra CU recursion initially.

12. Picture/slice encoder
   TEncSlice / TEncGOP reduced to still-image encode.

13. In-loop tools
   SAO and deblocking.
   Add after basic valid output.

14. Range extensions and lossless
   transform skip, cross-component prediction, residual DPCM,
   high bit depth, lossless-specific flags.
```

The first real milestone is not “all of JCTVC ported.” It is:

```text
Rust JCTVC-rs emits a decodable HEVC intra bitstream
for one 8-bit 4:4:4 image,
wrapped by bpg-rs,
decoded correctly by stock bpgdec.
```

### A3. Rust design replacements

Replace the C++ class graph:

```text
TEncTop
TEncGOP
TEncSlice
TEncCu
TEncSearch
TComDataCU
TComPic
TComPicYuv
TComYuv
TComTrQuant
TEncSbac
```

with:

```rust
struct EncConfig { ... }          // immutable
struct SequenceParams { ... }     // VPS/SPS/PPS material
struct Picture<T> { ... }         // source + reconstructed planes
struct CtuState { ... }           // final chosen state
struct CtuScratch { ... }         // temporary candidate buffers
struct IntraSearch { ... }
struct TransformQuant { ... }
struct CabacEstimator { ... }     // bit estimation
struct CabacWriter<W> { ... }     // final entropy writer
struct PictureEncoder { ... }
```

Critical design rule:

```text
Config immutable.
Scratch local.
Final bitstream explicit.
No global mutable encoder state.
No fake CLI config path.
No temp files.
```

### A4. Separate analysis from final entropy writing

This is the key to performance.

Bad design:

```text
while encoding CTU:
  mutate global encoder state
  search modes
  write bits
  update contexts
  reconstruct
```

Better design:

```text
analysis pass:
  for each CTU/region:
    evaluate candidate splits/modes
    estimate bits using local cloned CABAC estimator
    choose mode tree
    store chosen syntax decisions

final pass:
  walk chosen CTU decisions in legal order
  write actual CABAC bitstream
  reconstruct final picture
```

Not every HEVC dependency lets you fully split these, but this should be the direction. It gives you space for parallel analysis without corrupting entropy state.

### A5. Parallelization plan

Start conservative.

#### Easy parallel wins

```text
image conversion
chroma conversion
alpha encode vs color encode
multiple images
multiple QP trials
candidate pre-analysis
per-CTU rough complexity classification
```

#### Medium wins

```text
parallel intra candidate scoring
parallel transform candidate testing
parallel SAO analysis
parallel mode preselection
```

#### Hard wins

```text
full CTU final encode
CABAC final write
neighbor-dependent reconstruction
deblocking
```

The practical architecture:

```text
1. Serial correctness path.
2. Parallel rough analysis.
3. Serial final encode.
4. Optional wavefront/tile experiments later.
```

Avoid tile-based parallelism at first unless you accept compression loss. Independent tiles/slices are easy to parallelize but tend to cost bits.

### A6. Compression-preserving speed work

After correctness:

```text
remove generic video path
remove inter prediction code from hot path
specialize intra-only
specialize known bit depth with const generics where useful
specialize chroma format where useful
reuse scratch buffers
replace pointer-heavy graph with indices and slices
cache scan orders and lookup tables
make transform/quant cache-friendly
SIMD only after scalar Rust is correct
```

Do not start with SIMD. Start with correct scalar Rust, then profile.

## Direction C: hybrid teacher/student roadmap

Direction C should come **after** Direction A has a working JCTVC-rs encoder or at least after you can instrument stock JCTVC heavily.

The goal:

```text
Use JCTVC-quality decisions as training/teacher data,
then build a faster BPG-specific mode decision system.
```

This does not need neural ML first. Begin with decision logging and hand-built predictors.

### C1. Add decision tracing to JCTVC

Patch either original JCTVC or your partial Rust port to dump per-block decisions:

```text
image id
QP
bit depth
chroma format
CTU position
local variance
edge strength
gradient direction histogram
texture score
flatness score
neighbor complexity
chosen CU split tree
chosen intra luma mode
chosen chroma mode
chosen transform size
transform skip yes/no
SAO decision
deblock effect
estimated bits
actual bits
distortion
RD cost
```

Write this to Parquet or JSONL.

For your background, Parquet is probably the better long-term choice because you can run repeated analysis over large corpora.

### C2. Build a decision-diff harness

For each image block, compare:

```text
x265 decision approximation
JCTVC decision
JCTVC-rs decision
fast heuristic decision
```

You want to know where JCTVC wins:

```text
Does it split smaller?
Does it choose different intra modes?
Does it spend more bits on edges?
Does it use transform skip more effectively?
Does SAO help?
Does cross-component prediction matter?
Does 4:4:4 account for most wins?
```

This tells you whether the third encoder needs a better HEVC core or just better front-end decisions.

### C3. Build heuristic tiers

Instead of one global preset, use per-region effort tiers:

```rust
enum EffortTier {
    TrivialFlat,
    SimpleGradient,
    TextOrLineArt,
    ComplexTexture,
    AmbiguousHighEffort,
}
```

Example policy:

```text
TrivialFlat:
  test large CUs first
  skip many angular modes
  early accept if residual cheap

SimpleGradient:
  protect banding
  test planar/DC/angular near gradient direction

TextOrLineArt:
  test smaller CUs
  test transform skip
  test more angular modes
  favor 4:4:4 if source warrants it

ComplexTexture:
  medium CU split search
  avoid exhaustive angular search unless RD estimate uncertain

AmbiguousHighEffort:
  use JCTVC-like exhaustive search
```

This is where you can beat the false see-saw. JCTVC spends too much effort everywhere; x265 prunes aggressively everywhere. Your encoder can spend effort selectively.

### C4. Student model options

Start with deterministic models:

```text
threshold rules
decision trees
small hand-written classifiers
logistic regression generated offline
lookup tables by feature bucket
```

Only later consider a tiny ML model.

A practical progression:

```text
C0:
  hand-tuned thresholds

C1:
  decision tree trained from JCTVC traces, exported as Rust match/ranges

C2:
  small gradient-boosted tree or random forest distilled to tables

C3:
  tiny neural predictor only if it beats simpler systems
```

For a Rust codec, a decision tree or table-driven predictor is more appealing than bundling ONNX just for mode prediction.

### C5. Fast uncertainty fallback

The hybrid encoder should not blindly trust the predictor.

Use:

```text
predict candidate set
estimate RD confidence
if confidence high:
  test reduced set
else:
  fall back to JCTVC-like exhaustive search
```

This gives you a safety valve. The predictor is allowed to be wrong on easy cases, but not on costly ambiguous ones.

### C6. Direction C milestones

```text
C1:
  dump JCTVC decisions for corpus

C2:
  reproduce JCTVC choices statistically
  e.g. "top-3 predicted intra modes contain JCTVC mode 95% of time"

C3:
  reduced-search encoder
  tests fewer candidates but keeps quality within target

C4:
  effort-tier encoder
  per-region search budget

C5:
  compare:
    x265
    stock JCTVC
    JCTVC-rs full search
    JCTVC-rs hybrid search

C6:
  tune for classes:
    photos
    scans
    screenshots
    line art
    alpha PNGs
```

## What “everything that can be ported” should mean

I would classify the source like this:

### Port to Rust early

```text
bpgenc.c BPG writer logic
bpgenc.h data model
image conversion
alpha splitting
BPG metadata
NAL extraction
modified SPS/HEVC payload
JCTVC glue behavior, but not its architecture
```

### Port to Rust for Direction A

```text
jctvc/TLibCommon
jctvc/TLibEncoder
reduced TAppEncTop responsibilities
NALwrite / SEIwrite
CABAC encoder
intra prediction
transform/quant
SAO/deblock
CU/TU decision logic
```

### Keep as FFI / external

```text
x265 core
x265 x86 asm
x265 C++ encoder machinery
```

### Replace rather than port literally

```text
program_options_lite
TAppEncCfg
TVideoIOYuv
temp-file YUV flow
global debug/analyze structures
CLI-oriented config plumbing
```

Those should disappear in the Rust design.

## Suggested milestone sequence

### Milestone 0: Corpus and oracle harness

Build the harness first.

```text
input corpus
encode with old bpgenc x265
encode with old bpgenc jctvc
encode with new Rust encoder
decode with stock bpgdec
measure:
  size
  encode time
  PSNR
  SSIM
  optional visual diffs
```

This prevents you from getting lost during the port.

### Milestone 1: Rust BPG muxer + x265 backend

Goal:

```text
bpg-rs can encode valid BPG through x265.
```

This proves:

```text
BPG header
metadata
alpha handling
modified HEVC payload
x265 FFI
decode compatibility
```

### Milestone 2: JCTVC direct C++ shim

Before a full Rust port, create a cleaner C++ shim around JCTVC:

```text
no temp input file
no temp output file if possible
no fake argv if possible
direct config struct
in-memory image planes
in-memory bitstream
```

This is an optional but useful bridge. It shows exactly which JCTVC internals are needed.

### Milestone 3: Rust bitstream + parameter sets

Port:

```text
bit writer
NAL writer
VPS/SPS/PPS writer
slice header writer
SEI writer
```

Goal:

```text
emit structurally valid HEVC headers for BPG profile target.
```

### Milestone 4: Rust CABAC encoder

Port CABAC separately with unit tests.

Test against known syntax fragments from JCTVC:

```text
same context init
same bins
same byte output
```

CABAC bugs are painful. Isolate it.

### Milestone 5: Minimal intra encoder

Support:

```text
8-bit
4:4:4
one CTU size
limited transform sizes
basic intra modes
fixed QP
no SAO
no deblock initially if decoder accepts output, or include simple path
```

Goal:

```text
stock BPG decoder can decode the output.
```

Compression can be bad at this point. Validity matters first.

### Milestone 6: Match JCTVC feature set needed by BPG

Add:

```text
4:2:0
4:2:2
gray
high bit depth
transform skip
cross-component prediction
SAO
deblocking
lossless mode
residual DPCM / range extension tools
alpha plane
```

### Milestone 7: Compression parity mode

Goal:

```text
JCTVC-rs full-search mode gets near stock JCTVC size/quality.
```

Do not optimize too early. First get the full-search reference mode.

### Milestone 8: Performance mode

Add:

```text
parallel pre-analysis
mode candidate pruning
effort tiers
scratch reuse
cache-aware layout
const-generic bit depth/chroma specialization
optional SIMD
```

### Milestone 9: Direction C hybrid mode

Use JCTVC-rs full-search traces to train or derive reduced-search policies.

Goal:

```text
approach JCTVC compression
with materially lower encode time
```

## Performance design notes

Use explicit planes:

```rust
pub struct Plane<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
    stride: usize,
}
```

Use scratch arenas:

```rust
pub struct CtuScratch<T> {
    pred: Vec<T>,
    residual: Vec<i16>,
    coeffs: Vec<i32>,
    recon: Vec<T>,
    candidate_costs: Vec<CandidateCost>,
}
```

Use immutable shared tables:

```rust
pub struct Tables {
    scan_orders: ScanOrders,
    cabac_init: CabacInitTables,
    transform: TransformTables,
    chroma_qp: ChromaQpTables,
}
```

Use backend specialization:

```rust
trait Sample: Copy + Default + Send + Sync {
    const BIT_DEPTH: usize;
}

struct Encoder8;
struct Encoder10;
struct Encoder12;
struct Encoder14;
```

Avoid making every hot function generic from day one. Start clear, profile, then specialize.

## What success should look like

Reasonable targets:

```text
Stage 1:
  Rust BPG+x265 equals old bpgenc+x265 functionally.

Stage 2:
  JCTVC-rs valid output, compression worse than JCTVC but decodable.

Stage 3:
  JCTVC-rs full mode within ~1–3% size of stock JCTVC at equal metric.

Stage 4:
  JCTVC-rs full mode faster than stock JCTVC due to removed infrastructure,
  memory layout, and some parallel analysis.

Stage 5:
  Hybrid mode gets most of JCTVC's size advantage over x265
  while being far faster than stock JCTVC.
```

The central idea is:

```text
x265 backend:
  production-fast baseline

JCTVC-rs full:
  high-compression reference mode

JCTVC-rs hybrid:
  BPG-specific practical mode
```

That gives the project a clean progression. You do not need to gamble immediately on a new encoder design. First port and clean the JCTVC intra path, then use it to generate the data needed to build a better BPG-specific encoder.

[1]: https://github.com/listenlink/HM?utm_source=chatgpt.com "listenlink/HM: The H.265 reference software HM"
