# bpg-rs: Rust BPG Encoder Scaffolding (Milestone 1)

> Snapshot of the approved implementation plan
> (`/home/dk/.claude/plans/that-changes-the-plan-vast-quiche.md`), plus a
> progress update appended at the bottom.

## Context

We benchmarked the existing C `bpgenc` with both the JCTVC and x265 backends on
`dusk.png` (3200x2528 photo) across QP, chroma format (4:2:0/4:4:4), and
compression-level sweeps (`test_runs/results.csv`). At equal quality
(rate-distortion, not equal-QP), JCTVC offers at most ~0-4% smaller files than
x265 in narrow conditions, and is *worse* than x265 on 4:4:4/SSIM by up to
~11%, while being 30-50x slower. Pushing both encoders to max effort (`-m 9`)
changed nothing measurably.

**Decision: drop JCTVC from the Rust port entirely.** The Rust port targets a
clean BPG container/encoder built around **x265 only**, via FFI. The long-term
payoff is not "port a better HEVC encoder" but a Rust-native BPG muxer/image
pipeline with an x265 backend, which can later host a parameter-search
"candidate optimizer" (multiple x265 encodes per image, pick the best) since
x265 is fast enough (~1-2s) to make that practical.

> **Superseded** — see "Roadmap update (tuning system)" near the end of this
> file. The "candidate optimizer" (pick-the-best-of-N-encodes) framing above
> has been replaced by a conservative, source-aware **tuning system** for
> archival photo encoding. This section is retained for historical context on
> why x265-only was chosen.

**Additional decision:** the existing system-installed `libx265` (apt
package, v4.1/X265_BUILD=215) happens to match the version vendored at
`/home/dk/Desktop/testers-linux/BPG/x265_4.1/` (source tree, release tag 4.1,
`X265_BUILD=215` per its CMakeLists.txt). The bundled 2015-era source in
`libbpg-0.9.8/x265/` (X265_BUILD=75) is **not** used. `bpg-x265-sys` will
**vendor and build x265 from `/home/dk/Desktop/testers-linux/BPG/x265_4.1/source`**
via `build.rs` + `cmake`, producing a static `libx265.a` + matching
`x265_config.h`, rather than depending on the system package. This makes the
build reproducible and keeps the port pinned to a known-good, recent x265.

## Goal — Milestone 1

```
cargo run -p bpg-tools -- encode ../dusk.png -o /tmp/out.bpg --backend x265 --qp 28 --format 420
```

produces a `.bpg` that:
1. decodes successfully with the existing stock `bpgdec`
   (`/home/dk/Desktop/testers-linux/BPG/libbpg-0.9.8/bpgdec`)
2. has PSNR/SSIM within ~±0.2 dB / ±0.002 of
   `bpgenc -e x265 -q 28 -f 420 -o ref.bpg dusk.png`'s output vs `dusk.png`

Scope: 8-bit input (PNG via `image` crate), still image only (no animation),
lossy CQP only, YCbCr color space, 4:2:0 and 4:4:4 chroma. Alpha, >8-bit,
lossless, animation are explicit `TODO(extension)` stubs — type design must
not preclude them.

## Workspace layout

```
bpg-rs/
  Cargo.toml                  # workspace
  crates/
    bpg-bitstream/            # ue7 varint + Exp-Golomb bit I/O (no deps)
    bpg-image/                # Plane<T>, Image, color convert, chroma subsample, pad
    bpg-format/                # BPG container header (BpgHeader), depends on bpg-bitstream
    bpg-hevc/                  # NAL extraction, build_modified_sps/build_modified_hevc, depends on bpg-bitstream
    bpg-encode/                # HevcEncoder trait + encode_still_image orchestration
    bpg-x265-sys/              # vendors+builds x265_4.1, raw FFI bindings (bindgen)
    bpg-x265/                  # safe wrapper implementing HevcEncoder for x265
    bpg-tools/                 # CLI (encode subcommand)
```

Dependency graph: `bpg-bitstream` -> `bpg-format`, `bpg-hevc`;
`bpg-image` standalone (uses `image` crate); `bpg-encode` depends on
`bpg-image`, `bpg-format`, `bpg-hevc`; `bpg-x265-sys` -> `bpg-x265` (implements
`bpg-encode::HevcEncoder`); `bpg-tools` ties `bpg-image`+`bpg-encode`+`bpg-x265`
together via `clap`.

## bpg-x265-sys: vendored x265 build

`build.rs`:
1. Locate vendored source at `../../../x265_4.1/source` (path relative to the
   crate).
2. Run `cmake -S <src> -B <OUT_DIR>/x265-build -DENABLE_SHARED=OFF
   -DENABLE_CLI=OFF -DCMAKE_BUILD_TYPE=Release` (8-bit profile only for M1).
3. Run `cmake --build <OUT_DIR>/x265-build -j`.
4. Emit `cargo:rustc-link-search=<OUT_DIR>/x265-build` and
   `cargo:rustc-link-lib=static=x265`, plus `-lstdc++ -lpthread -lm -ldl`.
5. bindgen against `x265_4.1/source/x265.h`, with include paths for both
   `x265_4.1/source` and `<OUT_DIR>/x265-build` (where cmake generates
   `x265_config.h`). Allowlist `x265_.*` types/functions and `X265_.*` vars.
6. Handle the versioned `x265_api_get_215` symbol: regex-extract `X265_BUILD`
   (215) from the generated `x265_config.h` in `build.rs`, and codegen a tiny
   `extern "C" { fn x265_api_get_215(bit_depth: c_int) -> *const x265_api; }`
   + `pub use x265_api_get_215 as x265_api_get_versioned;` into
   `OUT_DIR/api_get.rs`, included via `include!()`.
7. Test: call `x265_api_get_versioned(8)`, assert non-null and
   `(*api).bit_depth == 8`.

## Implementation order (with risk notes)

0. **Workspace scaffold** — DONE.
1. **bpg-bitstream** (low risk) — DONE. `write_ue7`/`read_ue7`,
   `BitReader`/`BitWriter` with Exp-Golomb `ue(v)`. 8 unit tests passing.
2. **bpg-image types + color conversion** (medium risk) — DONE.
   `ColorConvertState::new` ports `convert_init`/`rgb24_to_ycc` for the
   `YCbCr*` family (BT.601/709/2020); `Rgb`/`YCgCo` are `unimplemented!()`.
   Verified against hand-computed (Python) reference values for full-range
   and limited-range YCbCr.
3. **Chroma subsampling + padding** (HIGH RISK) — DONE.
   `bpg-image::chroma::decimate_to_420` ports `decimate2_hv` for
   `h_phase == 1` only (the phase `bpgenc` always uses for the 4:2:0 default
   path). `bpg-image::pad::pad_plane` ports `image_pad`. Both have unit
   tests; full byte-for-byte validation against the C decimator on real image
   data is still outstanding (see Open items).
4. **bpg-format container header** (low-medium risk) — DONE.
   `BpgHeader::write` ports the `img_header` construction
   (`bpgenc.c:2515-2573`). Unit test matches the exact header bytes of a real
   `bpgenc -e x265 -q 28 -f 420` run on a no-alpha version of `dusk.png`
   (3200x2528, YCbCr, full range, 4:2:0, 8-bit):
   `42 50 47 fb 20 00 99 00 93 60 00`.
5. **bpg-hevc NAL/SPS rewriting** (HIGHEST RISK) — IN PROGRESS.
   Source for `extract_nal`/`find_nal_end` (bpgenc.c:1496-1565),
   `build_modified_sps` (bpgenc.c:1733-2031), and `build_modified_hevc`
   (bpgenc.c:2061-2169) has been read and the bit-level transformation is
   understood:
   - `find_nal_end`/`extract_nal`: locate Annex-B start codes
     (`00 00 01` / `00 00 00 01`), strip emulation-prevention bytes (`00 00 03`
     -> `00 00`).
   - `build_modified_sps`: parses the VPS (NAL type 32) then the SPS (NAL type
     33) bit-by-bit up through `log2_min_cb_size` and friends, asserting a
     long list of HEVC-feature preconditions that x265's SPS is known to
     satisfy (e.g. `vps_id==0`, `max_sub_layers==0`, `sps_id==0`,
     `log2_max_poc_lsb==8`, `scaling_list_enable_flag==0`,
     `amp_enabled_flag==1`, `nb_st_rps==0`,
     `sps_temporal_mvp_enabled_flag==1`, no VUI HRD/display-window/range-ext
     features beyond a small allowed set). It then re-emits just the
     "modified SPS" tail (`log2_min_cb_size-3`, diff sizes, transform depth,
     sao/pcm/strong-intra-smoothing/extension flags) as a `ue(v)`-coded blob
     prefixed with its own `ue7` length.
   - `build_modified_hevc`: for the no-alpha, non-animated M1 case, this
     reduces to: take x265's raw Annex-B output, run `build_modified_sps` on
     it (which also returns the byte offset where the VPS+SPS NALs ended),
     emit the modified-SPS blob, then copy every remaining NAL (stripping the
     leading start code from the very first one only) verbatim into the
     output buffer. The `frame_duration_tab`/SEI-insertion logic is dead code
     for `frame_ticks == 1` (the still-image default), so it can be omitted
     entirely for M1.
   - Confirmed via `xxd` that real BPG payload bytes for `dusk_rgb.png` begin
     `03 92 47 40 44 ...` immediately after the container header — i.e. the
     modified-SPS blob starts with `ue7(len)=0x03` followed by 3 bytes of
     `ue(v)`-coded SPS tail. This is the byte-for-byte target for the Rust
     port's unit test.

   Not yet started: writing the actual Rust `bpg-hevc` crate
   (`extract_nals`, `ModifiedSps::from_hevc_stream`,
   `build_modified_hevc`) and its de-risking tests.

6. **bpg-x265-sys FFI** (medium risk, mechanical) — not started.
7. **bpg-x265 safe wrapper** (medium risk) — not started.
8. **bpg-encode orchestration** (integration risk) — not started.
9. **bpg-tools CLI** (low risk) — not started (placeholder `main.rs` only).
10. **End-to-end verification** — not started.

## Verification (Milestone 1 acceptance)

```bash
cd /home/dk/Desktop/testers-linux/BPG
./libbpg-0.9.8/bpgenc -e x265 -q 28 -f 420 -o /tmp/ref.bpg dusk.png

cd bpg-rs
cargo run -p bpg-tools -- encode ../dusk.png -o /tmp/out.bpg --backend x265 --qp 28 --format 420

cd ..
./libbpg-0.9.8/bpgdec -o /tmp/ref.png /tmp/ref.bpg
./libbpg-0.9.8/bpgdec -o /tmp/out.png /tmp/out.bpg

ffmpeg -hide_banner -i /tmp/ref.png -i dusk.png -lavfi "ssim;[0:v][1:v]psnr" -f null -
ffmpeg -hide_banner -i /tmp/out.png -i dusk.png -lavfi "ssim;[0:v][1:v]psnr" -f null -
ls -la /tmp/ref.bpg /tmp/out.bpg
```

Acceptance: (1) `out.bpg` decodes via stock `bpgdec`; (2) PSNR/SSIM of
`out.png` vs `dusk.png` within ~±0.2 dB / ±0.002 of `ref.png`'s. Byte-for-byte
match of the modified-HEVC payload against `ref.bpg` is a nice-to-have, not a
hard requirement (x265 bitstream determinism across runs/builds isn't
guaranteed).

## Open items to resolve during implementation

- Full byte-for-byte validation of `decimate_to_420` against the C
  `decimate2_hv` on real (non-flat) image data — current unit tests only cover
  flat/solid-color planes.
- Confirm relative path from `bpg-rs/crates/bpg-x265-sys/` to
  `/home/dk/Desktop/testers-linux/BPG/x265_4.1/source` once that crate's
  `build.rs` is written (expected: `../../../x265_4.1/source`).
- `dusk.png` itself has an alpha channel (RGBA); M1 ignores alpha
  (`has_alpha = false` always), so the end-to-end verification in step 10
  should compare against an RGB-only copy of `dusk.png` (e.g.
  `ffmpeg -i dusk.png -pix_fmt rgb24 dusk_rgb.png`), not the original RGBA
  file, when constructing the reference `.bpg`.

---

## Progress update (this session)

**Done so far:**
- Cargo workspace created at `bpg-rs/` with all 8 member crates scaffolded.
- `bpg-bitstream`: fully implemented and tested (ue7 varint, MSB-first
  bit I/O, Exp-Golomb).
- `bpg-image`: fully implemented and tested (`Plane<T>`, `Image`,
  `ChromaFormat`, `ColorSpace`, `ColorConvertState` for YCbCr/BT.709/BT.2020,
  `chroma::decimate_to_420` for `h_phase==1`, `pad::pad_plane`/
  `pad_to_cb_size`).
- `bpg-format`: fully implemented and tested (`BpgHeader::write`), verified
  byte-for-byte against a real `bpgenc` header for `dusk.png`'s dimensions.
- All of `libbpg-0.9.8`, `x265_4.1`, `test_runs`, `dusk.png`, and the new
  `bpg-rs` workspace have been committed to a new git repo at
  `/home/dk/Desktop/testers-linux/BPG` and pushed to
  `https://github.com/LegeApp/bpg-rs.git`.

**Next step:** implement the `bpg-hevc` crate (step 5 above) — this is the
highest-risk remaining piece. Concretely:
1. `extract_nals(annexb: &[u8]) -> Vec<Nal>` — Annex-B start-code scanning +
   emulation-prevention-byte removal, per `find_nal_end`/`extract_nal`.
2. `ModifiedSps::from_hevc_stream(hevc_bytes: &[u8]) -> Result<(Self, usize), HevcError>`
   — bit-level VPS+SPS parser/rewriter per `build_modified_sps`, with the same
   precondition checks (returning `HevcError::UnsupportedFeature` instead of
   `fprintf`+`return -1`).
3. `build_modified_hevc(color_stream: &[u8]) -> Result<Vec<u8>, HevcError>` —
   for M1 (no alpha, `frame_ticks==1`), call `ModifiedSps::from_hevc_stream`,
   emit its bytes, then copy the remaining NALs verbatim (stripping only the
   first start code).
4. De-risk by encoding a small test image with x265 directly (a throwaway
   harness or by temporarily hooking into `bpg-x265` once that exists) and
   comparing the Rust-rewritten output's modified-SPS bytes against the
   `03 92 47 40 44` prefix observed in the real `ref_noalpha.bpg`.

After `bpg-hevc`, proceed to step 6 (`bpg-x265-sys` vendored FFI build),
step 7 (`bpg-x265` safe wrapper), step 8 (`bpg-encode` orchestration), step 9
(`bpg-tools` CLI), and step 10 (end-to-end verification).

---

## Progress update (Milestone 1 COMPLETE)

**All remaining crates implemented; the M1 acceptance test passes with a
byte-for-byte identical decode against the C reference.**

- `bpg-hevc` (step 5): implemented and tested (7 unit tests). `find_nal_end`,
  `extract_nal`, `ModifiedSps::from_hevc_stream` (full VPS+SPS precondition
  parser/rewriter), and `build_modified_hevc` for the no-alpha/still-image
  case. Errors are returned as `HevcError` rather than `fprintf`+`-1`.
- `bpg-x265-sys` (step 6): `build.rs` builds the vendored x265 at
  `../../../x265_4.1/source` via the `cmake` crate (static lib, 8-bit,
  `ENABLE_CLI=OFF`, `ENABLE_ASSEMBLY=OFF` by default for a deterministic
  scalar build; `nasm` 2.16.03 IS available in this dev env, so set
  `BPG_X265_ENABLE_ASM=1` to build the SIMD kernels),
  then bindgens `x265.h`. **Simplification vs. the original plan:** instead of
  codegen-ing the build-number-versioned `x265_api_get_215` symbol, the safe
  wrapper calls the non-versioned `x265_api_query(bitDepth, X265_BUILD, &err)`
  function, which is exported directly and avoids the `include!()` glue.
- `bpg-x265` (step 7): safe `X265Encoder` implementing
  `bpg_encode::HevcEncoder`, ported from `x265_glue.c` (single-frame intra).
  The x265 RC-mode enum (`X265_RC_CQP=1`) and `x265_preset_names` array are
  `static`/anonymous in `x265.h` (not emitted by bindgen), so they are
  redefined as constants in the wrapper.
- `bpg-encode` (step 8): `HevcEncoder` trait, `HevcEncodeParams`,
  `EncodeError`, and `encode_still_image` orchestration (pad → encode →
  `build_modified_hevc` → header + payload).
- `bpg-tools` (step 9): `clap`-based CLI with the `encode` subcommand.

**End-to-end verification (step 10) — PASSED:**

```
cargo run --release -p bpg-tools -- encode ../dusk.png -o /tmp/out.bpg \
    --backend x265 --qp 28 --format 420
./libbpg-0.9.8/bpgdec -o /tmp/out.png /tmp/out.bpg          # decodes cleanly
```

- `out.bpg` is 1,097,884 bytes (cf. the C reference `x265_q28_420.bpg`,
  1,099,698 bytes — the C reference also carries an alpha plane).
- The payload begins `03 92 47 40 44` immediately after the container header
  — **exactly** the modified-SPS target byte sequence recorded above.
- Decoding `out.bpg` and the C reference and comparing both against `dusk.png`
  (RGB) gives **PSNR(ours vs C-ref) = ∞** — i.e. the decoded RGB pixels are
  *identical* to the C `bpgenc` output (both 39.4190 dB vs source). This far
  exceeds the ±0.2 dB / ±0.002 acceptance bar.

The only header-byte difference from the in-tree `test_runs/*.bpg` references
(`0x20` vs `0x30` at offset 4) is the `alpha1_flag`: those references were
encoded from the RGBA `dusk.png`, whereas M1 drops alpha by design.

**Build/test:** `cargo test` → 25 unit tests pass across the workspace. The
x265 static build runs from `bpg-x265-sys/build.rs` on first `cargo build`
(requires `cmake` + a C/C++ toolchain; no assembler needed for the default
scalar build. `nasm` 2.16.03 is available, so `BPG_X265_ENABLE_ASM=1` enables
the SIMD kernels).

**Open items / next milestones (superseded, see later sections):** alpha-plane
support (interleaved MSPS + `build_modified_hevc` alpha branch), >8-bit,
lossless, animation, 4:2:2, `c_h_phase==0` (MPEG2 siting), RGB/YCgCo color
spaces, and the x265 parameter-search "candidate optimizer" described in the
Context section. The type stubs (`TODO(extension)`) already leave room for
these. Animation was dropped (see "Progress update (10-bit support)") and the
candidate optimizer was replaced by the tuning system (see "Roadmap update
(tuning system)").

## Progress update (10-bit support)

Per a roadmap review: **alpha and animation are dropped** (alpha is rarely
present in PNGs from raw-image pipelines; their `TODO(extension)`/`has_alpha`
stubs are left as-is but deprioritized). **>8-bit (10/12-bit) is now
implemented**, ahead of the candidate optimizer (still last).

- `bpg-image`: `Image.planes` is now `Vec<Plane<u16>>` always, matching
  `bpgenc.c`'s internal `typedef uint16_t PIXEL` representation — values are
  bounded by `(1 << bit_depth) - 1` for whatever `bit_depth` (8/10/12) was
  selected at color-conversion time.
  - `convert.rs`: `ColorConvertState::rgb_to_ycc` is now generic over the
    input sample type (`P: Into<i64>`, i.e. `u8` or `u16`) and always returns
    `(u16, u16, u16)`; the existing `c_shift = 31 - out_bit_depth` etc.
    formulas were already bit-depth-generic.
  - `chroma.rs`: `decimate_to_420`/`decimate_row_h` now operate on
    `Plane<u16>`/`&[u16]` (arithmetic unchanged, only storage type).
  - `pad.rs`: `pad_plane` is now `pad_plane<T: Copy + Default>`.
  - `Image::from_rgb8(rgb, color_space, limited_range, bit_depth)` and new
    `Image::from_rgb16(rgb16, color_space, limited_range, bit_depth)` (for
    16-bit PNG input) both go through a shared `from_rgb_pixels` helper.
- `bpg-encode`: `encode_still_image` now accepts `bit_depth` 8, 10, or 12
  (was hard-`8`-only).
- `bpg-x265`: `encode_impl` now branches on `bit_depth`: for 8-bit it
  truncates the `u16` planes to a temporary `u8` buffer (mirroring
  `image_convert16to8`) with `stride` in samples; for 10/12-bit it passes the
  `u16` plane data directly with `stride * 2` (bytes), matching x265's
  `x265_picture.stride` "bytes between row starts" convention.
- `bpg-x265-sys`: `build.rs` now does an x265 **multilib build** for 10-bit:
  it first builds a 10-bit static lib (`HIGH_BIT_DEPTH=ON`, `MAIN12=OFF`,
  `EXPORT_C_API=OFF`, namespace `x265_10bit`) and copies it out as
  `libx265_main10.a`, then builds the main 8-bit lib (`EXPORT_C_API=ON`,
  `LINKED_10BIT=ON`, `EXTRA_LIB=<path to libx265_main10.a>`). Both archives
  are linked (`-lx265 -lx265_main10`); `x265_api_query(10, X265_BUILD, &err)`
  then dispatches into the linked-in `x265_10bit::` namespace. Set
  `BPG_X265_SKIP_10BIT=1` to build only the 8-bit lib for faster iteration
  (12-bit was not built — not needed for the 10-bit milestone).
- `bpg-tools`: new `-b`/`--bit-depth` option (8/10/12, default 8). 16-bit PNG
  input (`DynamicImage::ImageRgb16`/`ImageRgba16`) is read via
  `Image::from_rgb16`; other inputs go through `Image::from_rgb8` and may be
  upscaled to a higher output `bit_depth`.

**Verification:** a synthetic 64x64 16-bit PNG (generated via the `image`
crate) was encoded at `--bit-depth 10 --format 420 --qp 28`, producing a
115-byte `.bpg` that `bpgdec` decodes cleanly (`-b 16` round-trips to a
16-bit PNG with values close to the source, as expected for lossy QP28). The
existing 8-bit M1 acceptance test (`dusk.png`, byte-identical vs C reference,
1,097,884 bytes) still passes unchanged. `cargo test` → 27 unit tests pass.

**Remaining:** 12-bit (not built, would need a second multilib lib +
`MAIN12=ON`), the candidate optimizer (last), and the previously-listed
alpha/4:2:2/MPEG2-siting/RGB/YCgCo items.

## Roadmap update (reprioritization + tuning system)

Per a roadmap review, the remaining items above are reprioritized as follows.

**Deferred to much later:**
- **12-bit support** (second multilib lib with `MAIN12=ON`).
- **Lossless mode** (`bLossless=1`).

Both remain `TODO(extension)`/`unimplemented!()` stubs; nothing about the
current design blocks adding them later.

**Promoted to a near-term milestone:**
- **4:2:2 chroma** (`ChromaFormat::Yuv422`), alongside the already-implemented
  4:2:0. `bpg-image` needs a `decimate_to_422`/horizontal-only decimation
  analogous to `image_ycc444_to_ycc422`/`decimate2_h` in `bpgenc.c`
  (4:2:0's `decimate_to_420` does both horizontal and vertical decimation via
  `decimate2_hv`; 4:2:2 only decimates horizontally). `bpg-tools`'s
  `--format` enum gains a `422` variant alongside `420`/`444`.

**Replaces the "candidate optimizer" (described in the Context section and
referenced as "last" throughout this file): a "tuning" system.**

The original plan was a parameter-search "candidate optimizer": run multiple
x265 encodes per image and pick the result that scores best on some metric
(PSNR/SSIM/file size). On reflection this is the wrong default for an
*archival* photo encoder — metric-chasing tends to push toward destructive
choices (denoise/sharpen preprocessing, palette reduction, chroma
subsampling, or bit-depth reduction the user didn't ask for) purely because
they make a number go up, and it can silently strip metadata/ICC profiles
that matter for archival use. The replacement is **conservative,
source-aware presets ("tunes") plus an explicit archival policy**, with
optimizer-style candidate search demoted to an optional, tightly-bounded,
opt-in mode for later.

### Three-layer model

1. **Preset** — x265 speed/quality preset (`ultrafast`..`placebo`), already
   exposed via `-m/--compress-level`. Orthogonal to the rest.
2. **Tune** — a *source-character* hint describing what kind of image this
   is, used to pick x265 params (and, where allowed, chroma/bit-depth
   choices) that suit that source. This is a **bpg-rs concept**, distinct
   from x265's own `--tune` (`psnr`/`ssim`/`grain`/`zero-latency`/
   `fast-decode`/`animation`), which is about *encoder* behavior (e.g.
   disabling psychovisual optimizations for PSNR/SSIM). bpg-rs's `Tune` may
   select an x265 tune as part of its plan, but the two are not the same
   axis.
3. **ArchivalPolicy** — hard constraints/permissions that bound what any tune
   is allowed to do (e.g. "never strip metadata", "never subsample chroma
   without being asked").

### `Tune` enum

```rust
pub enum Tune {
    Auto,       // run bpg-analyze and pick one of the below
    Neutral,    // no source-specific adjustments; safe default
    Photo,      // typical digital-camera/phone photos
    FilmGrain,  // scanned film / grainy sources — preserve grain, don't denoise
    Slide,      // scanned slides/transparencies — flat fields, dust/scratches
    LowLight,   // high-ISO/noisy photos — avoid over-smoothing noise into banding
    Artwork,    // illustrations/digital art — sharp edges, flat color regions
    Screenshot, // UI screenshots/rendered text/graphics — sharp edges, exact colors
    Scan,       // document/photo scans
}
```

`Neutral` is the safe default when a tune can't be confidently chosen.

### `ArchivalPolicy`

```rust
pub struct ArchivalPolicy {
    /// Copy through EXIF/XMP/other metadata where the container supports it.
    pub preserve_metadata: bool,
    /// Copy through any embedded ICC color profile.
    pub preserve_icc_profile: bool,
    /// Permit a tune to choose 4:2:0/4:2:2 over 4:4:4 for this image.
    pub allow_chroma_subsampling: bool,
    /// Permit a tune to choose an output bit depth lower than the source.
    pub allow_down_bitdepth: bool,
    /// Permit any pixel-level preprocessing (denoise, sharpen, etc.).
    pub allow_preprocessing: bool,
    /// If true, tunes may lean toward metric fidelity (PSNR/SSIM) over
    /// perceptual choices when the two trade off.
    pub prefer_metric_fidelity: bool,
}
```

`--archival` on the CLI selects a policy with everything destructive turned
off (`allow_preprocessing = false`, `preserve_metadata = true`,
`preserve_icc_profile = true`); chroma subsampling and bit-depth choices are
still permitted unless the user also passes flags disabling them, since those
are normal lossy-codec parameters rather than "destructive edits".

### What tunes must NOT do

Regardless of `Tune`, no tune may (unless `ArchivalPolicy` explicitly allows
it):
- Apply denoise, sharpen, deband, or palette-reduction preprocessing to
  pixel data.
- Choose parameters purely to maximize a metric (PSNR/SSIM/file size) at the
  expense of the source's character (e.g. denoising grain to improve PSNR).
- Change chroma subsampling or bit depth without the policy permitting it.
- Drop metadata or ICC profiles.

### Per-tune behavior (x265 param choices, conceptually)

- **Neutral**: no adjustments beyond preset/qp; equivalent to today's
  behavior. Safe fallback.
- **Photo**: psy-rd / psy-rdoq tuned for natural photo content (close to
  x265 defaults); 4:2:0 acceptable if policy allows.
- **FilmGrain**: higher psy-rd, `--tune grain`-like emphasis, AQ settings
  that avoid smoothing grain into flat regions; prefer 4:4:4 or 4:2:2 over
  4:2:0 if policy allows subsampling, since grain carries chroma detail.
- **Slide**: similar to Photo but with AQ tuned for large flat fields with
  occasional fine detail (dust/scratches) — avoid banding in flats without
  smoothing out scratches.
- **LowLight**: AQ/psy-rd chosen to avoid amplifying sensor noise into
  blocking/banding; explicitly does *not* denoise (that would be
  preprocessing).
- **Artwork**: parameters favoring sharp edges and flat color regions
  (lower AQ strength, rdoq tuned for synthetic content); 4:4:4 preferred
  when policy allows, since artwork often has chroma edges.
- **Screenshot**: near-lossless-leaning QP defaults, 4:4:4 preferred, no
  chroma subsampling unless explicitly permitted — text/UI elements are
  chroma-sensitive.
- **Scan**: similar to Slide/Photo hybrid; AQ tuned for scanned-paper
  texture without over-smoothing.
- **Auto**: `bpg-analyze` extracts `ImageFeatures` from the source and maps
  them to one of the above via `TuneAnalysis`, falling back to `Neutral` when
  signals are inconclusive.

### `EncodePlan`

The output of "tune resolution" — a fully-resolved set of encoder params,
independent of how it was derived:

```rust
pub struct EncodePlan {
    pub chroma_format: ChromaFormat,
    pub x265_preset: &'static str,
    pub x265_tune: Option<&'static str>,   // x265's own --tune, if any
    pub x265_params: BTreeMap<String, String>, // extra x265_param_parse() overrides
    pub bpg_color_mode: ColorSpace,
    pub warnings: Vec<String>,             // e.g. "policy disallowed 4:2:0; using 4:4:4"
}
```

### New crates

- **`bpg-analyze`**: feature extraction (`ImageFeatures`: noise estimate,
  edge density, flat-region fraction, color histogram spread, grain
  signature, etc.) and `TuneAnalysis` (maps `ImageFeatures` -> a suggested
  `Tune` + human-readable reasoning string). Pure analysis, no encoding.
- **`bpg-tune`**: maps `(Tune, ArchivalPolicy, ImageFeatures)` -> `EncodePlan`.
  Depends on `bpg-analyze` for `Auto`. Contains the per-tune param tables
  described above.

Updated dependency graph: `bpg-analyze` is standalone (operates on
`bpg-image::Image`); `bpg-tune` depends on `bpg-analyze` + `bpg-image`;
`bpg-encode`/`bpg-tools` depend on `bpg-tune` to resolve an `EncodePlan`
before calling the `HevcEncoder`.

### CLI shape (future)

```
bpg-rs encode input.jpg output.bpg --quality 32 --tune auto --archival
bpg-rs analyze input.jpg     # prints ImageFeatures + chosen Tune + reasoning
```

`analyze` is a diagnostic/dry-run command: it explains what `--tune auto`
would choose and why, without encoding.

### Optional future: bounded candidate search

A `--candidate-check conservative` mode may be added later: given the
resolved `EncodePlan`, try a small, bounded set of nearby variations (e.g.
+/-1 QP, with/without an x265 tune) and pick among them using a quality
floor rather than a quality-maximizing search — i.e. "don't make it worse",
not "make it as small/high-scoring as possible". This is explicitly
secondary to the tuning system above and not required for the 4:2:2
milestone.

### Dataset/labeling note

If a labeled dataset is ever built to validate `Auto` tune selection, it
should target "did bpg-rs pick the *conservative, appropriate* tune for this
source" (agreement with a human's source-character judgment), not "which
params produced the smallest file at a given PSNR" — the latter reproduces
the metric-chasing problem the tuning system is meant to avoid.

## Progress update (4:2:2 chroma support)

**4:2:2 chroma is implemented**, per the reprioritization above.

- `bpg-image::chroma`: new `decimate_to_422(src: &Plane<u16>, bit_depth: u32)
  -> Plane<u16>`, ported from `image_ycc444_to_ycc422`/`decimate2_h`/
  `decimate2p1_simple` (`h_phase == 1` only, matching `decimate_to_420`'s
  restriction). Unlike `decimate_to_420`'s horizontal pass
  (`decimate2p1_simple16`, `shift = bit_depth - 7`, no saturation), 4:2:2 is
  a single horizontal-only pass with a **fixed `shift = 7`** (the DP1 taps
  sum to `2^7` regardless of bit depth) and saturates to
  `[0, (1 << bit_depth) - 1]` via `clamp_pix`, exactly mirroring
  `decimate2p1_simple`. Output plane keeps the source height and halves only
  the width (`(w+1)/2`).
- `Image::subsample_to_422(h_phase)` mirrors `subsample_to_420`: asserts
  4:4:4 input and `h_phase == 1`, decimates planes 1/2, sets
  `chroma_format = Yuv422`. `plane_shifts` already returned `(1, 0)` for
  `Yuv422` (h-shift only), so `pad_to_cb_size` needed no changes.
- `bpg-tools`: `--format` gained a `422` value; `run_encode` now matches on
  `Format::{Yuv420, Yuv422, Yuv444}` calling `subsample_to_420(1)` /
  `subsample_to_422(1)` / no-op respectively.
- `bpg-encode`/`bpg-x265`/`bpg-format`/`bpg-hevc` needed **no changes**:
  `ChromaFormat::Yuv422` -> `X265_CSP_I422` / `PixelFormat::Yuv422` mappings
  already existed, and the generic HEVC SPS chroma_format_idc rewrite in
  `bpg-hevc` only special-cases `chroma_format_idc == 3` (4:4:4 separate
  planes), so `idc == 2` (4:2:2) flows through unchanged.

**Verification:** `cargo run -p bpg-tools -- encode dusk.png -o out.bpg
--backend x265 --qp 28 --format 422` produces a 1,132,583-byte `.bpg` with
header byte 4 = `0x40` (pixel_format = 2 = YUV422, bit_depth_minus_8 = 0).
Built the C `bpgenc`/`bpgdec` reference (required installing
`libjpeg-dev`/`libx265-dev`/`ffmpeg` in this environment, since they were
missing) and ran `bpgenc -e x265 -q 28 -f 422 -o ref422.bpg dusk_rgb.png`
(system x265 3.5, vs the vendored 4.1 used by bpg-rs) — **the two `.bpg`
files are byte-for-byte identical** (`cmp` reports no difference), despite
the x265 version difference. `bpgdec` decodes the bpg-rs output cleanly;
`ffmpeg` PSNR/SSIM vs `dusk_rgb.png` is ~39.99 dB / 0.9865 SSIM, consistent
with QP28 4:2:2. The existing 8-bit 4:2:0 acceptance test (`dusk.png`,
1,097,884 bytes) is unchanged. `cargo test` -> 35 unit tests pass (8 new
4:2:2 tests in `bpg-image`).

**Remaining (per the reprioritization above):** alpha-plane support,
`c_h_phase==0` (MPEG2 chroma siting), RGB/YCgCo color spaces, and the
tuning system (`bpg-analyze`/`bpg-tune`). 12-bit and lossless remain deferred
to much later.

## Roadmap update (Rust backend correction)

This file began as the x265-backed BPG encoder plan. That milestone remains
valid, but it is no longer the long-term production architecture. The corrected
direction is:

1. The Rust still-HEVC encoder becomes the production backend.
2. Upstream x265 remains only as an oracle/dev tool for regression comparison,
   syntax checks, file-size/quality benchmarking, and phase-gate validation.
3. `bpg-x265`/`bpg-x265-sys` should become optional `oracle-x265` features once
   the Rust backend can encode BPG files accepted by `bpg-decode` and stock
   `bpgdec`.
4. Do not delete x265 yet; it is still the only reliable external comparator
   while the Rust encoder is being completed.

Immediate order:

- Implement `RustStillHevcEncoder` for `bpg_encode::HevcEncoder`, add
  `--backend rust`, then make Rust the default once decode-valid.
- Fix the backend source-geometry boundary before adding more format support:
  visible width/height, coded width/height, plane width/height/stride, bit
  depth, and chroma format must be explicit.
- Finish Rust-still 4:2:2 correctness.
- Add deblocking.
- Add SAO in two stages: syntax enabled with every CTB/component off, then real
  RD-selected SAO candidates.
- After those easy correctness items, focus on the hard compression work:
  SATD rough intra analysis, CABAC bit estimation, full RD mode decision,
  recursive CU/TU split search, chroma mode decision, RDOQ/sign hiding,
  primitive replacement, and hot-path allocation removal.

Material completion is not "SAO/deblock/4:2:2 exist." Completion means the Rust
backend is production default, x265 is oracle-only, 8-bit/10-bit 4:2:0/4:2:2/
4:4:4 round-trip exactly against encoder reconstruction including odd sizes,
deblock/SAO can be enabled and decoded exactly, unsupported BPG features fail
explicitly, and the Rust encoder is within the chosen x265 size/quality band on
the oracle corpus.

Reproducibility issue to fix: `bpg-rs/x265_4.1` is currently a symlink to an
absolute local path. If x265 is still required for any normal build, replace it
with a real vendored source tree, a git submodule, a build/setup script, or move
it fully behind documented oracle-only setup.

• Made concrete progress toward the active goal:

  Implemented a first-pass RDOQ path in crates/bpg-still-hevc/src/encoder.rs:444. After the plain x265-style quantizer, each nonzero coefficient is refined with candidate levels using dequantized transform-domain distortion plus the existing CABAC
  residual bit estimator. Zero-block candidates are handled explicitly so RDOQ can legally clear a block.

  Also improved roundtrip assertion diagnostics in crates/bpg-still-hevc/tests/encode_roundtrip.rs:97, and updated quantization comments in crates/bpg-still-hevc/src/transform.rs:1.

  I also localized the 4:2:0 TU split issue: optional 32x32-to-16x16 splitting can select plausible RD splits, but chroma recon diverges around the split block. I kept that gated to preserve correctness.

  Verified:

  cargo fmt
  cargo test -p bpg-still-hevc

  Remaining for the full goal: fix and enable 4:2:0 and 10-bit TU split RD, implement full CU split RD, and harden RDOQ beyond this first greedy coefficient-level pass.

## Progress update (4:2:0/10-bit TU split RD + full CU split RD)

• Made concrete progress toward the active goal:

  Root-caused the 4:2:0/10-bit TU split RD gating: `choose_luma_mode` and
  `choose_chroma_mode` call `predict_intra` for every SATD-search candidate
  mode, writing real sample values into `self.frame` without restoring it
  afterward. This corrupted `UNINIT_SAMPLE` sentinels (see
  `bpg-hevc-decode/src/hevc/picture.rs`) that later split/leaf RD trials in
  `build_tt` depend on for H.265 intra reference-sample availability rules.
  Fixed by snapshotting `self.frame` before each mode-search loop and
  restoring it after, in both functions
  (crates/bpg-still-hevc/src/encoder.rs).

  With that fix, `can_split_tt` no longer needs the bit-depth-8 gate or the
  log2_size restriction: it is now simply
  `log2_size <= MAX_TB_LOG2 && log2_size > MIN_TB_LOG2 && trafo_depth <
  MAX_INTRA_TT_DEPTH && !(cat == 1 && log2_size == 3)`. The remaining
  exclusion is 4:2:0's 8x8->4x4 luma split, where chroma is coded once at the
  8x8 parent node (`build_parent_chroma_tu`) and that path is not yet
  validated against decoder recon.

  Implemented full in-picture CU split RD (previously hardcoded
  `split = false` except at forced picture-boundary splits). New types
  `CuLeaf`/`CuNode` (encoder.rs) mirror the existing `Tt` split/leaf pattern
  at the coding-quadtree level. New `Encoder` methods `build_cu_leaf`,
  `build_cu_kids`, `build_cu` perform the split-vs-leaf RD comparison
  (snapshotting/restoring `self.frame`, `self.mode_map`, and
  `self.ct_depth_map` per trial). New free functions
  `estimate_split_cu_flag_bits`, `estimate_cu_leaf_bits`,
  `estimate_cu_node_bits`, `estimate_cu_kids_bits` mirror the `estimate_tt_*`
  bit-estimation helpers. New `write_cu` emits `split_cu_flag` plus the
  recursive CU syntax. `write_coding_quadtree` is now a thin two-call wrapper
  (`build_cu` then `write_cu`).

  Verified:

  cargo fmt -p bpg-still-hevc
  cargo test -p bpg-still-hevc --test encode_roundtrip -- --test-threads=1  (8/8 pass)
  cargo test  (full workspace, 33 test groups, 0 failures)

  Remaining for the full goal: harden RDOQ beyond the first greedy
  coefficient-level pass (`refine_levels_rdoq`).

## Progress update (RDOQ multi-pass hardening)

• Made concrete progress toward the active goal:

  Hardened `refine_levels_rdoq` (crates/bpg-still-hevc/src/encoder.rs) beyond
  its original single greedy coefficient-level pass: the per-coefficient
  {level, 0, level-1, level+1} candidate search now repeats (capped at 4
  passes) until a pass makes no level changes. This lets a level change for
  one coefficient (which shifts the significant_coeff_flag/last-position
  CABAC context for its neighbours) unlock a better choice for another
  coefficient on a subsequent pass, including the whole block converging to
  all-zero (cbf == 0) when warranted.

  Verified:

  cargo fmt -p bpg-still-hevc
  cargo test -p bpg-still-hevc --test encode_roundtrip -- --test-threads=1  (8/8 pass)
  cargo test  (full workspace, 0 failures)

  This completes all four items of the active goal: (a) 4:2:0 TU split RD,
  (b) 10-bit TU split RD, (c) full CU split RD, (d) multi-pass RDOQ
  hardening.

## Progress update (SATD-shortlist + full-RD luma mode decision, preset-controlled)

• Made concrete progress toward further x265 intra parity:

  Previously `choose_luma_mode` picked a single luma intra mode by SATD +
  estimated mode-signaling bits alone, and only that one mode ever reached
  full TU-split RD. x265 instead SATD-shortlists a handful of candidate modes
  and runs full RD (transform/RDOQ/CABAC cost) on each before committing.

  `choose_luma_mode` is now `choose_luma_mode_candidates`
  (crates/bpg-still-hevc/src/encoder.rs): it scores all 35 modes by SATD +
  mode bits as before, then returns the best `N` by that rough cost
  (ascending), where `N = luma_rd_candidates(effort)`:
  `Effort::Fast` -> 1, `Effort::Balanced` -> 2, `Effort::Best` -> 3 — mirroring
  x265 presets' candidate-count tradeoff (faster presets commit to the
  best-SATD mode immediately; slower presets spend extra full-RD trials on
  next-best SATD candidates).

  `build_cu_leaf` now loops over the shortlist: for each candidate luma mode
  it runs `choose_chroma_mode` + full `build_tt` (TU-split RD), computes the
  CU-level RD cost via `estimate_cu_leaf_bits` + `distortion_tt_region`, and
  keeps the lowest-cost candidate — snapshotting/restoring `self.frame` and
  `self.mode_map` between trials (same pattern as `build_cu`'s split-vs-leaf
  comparison). `Effort::Fast` shortlists one candidate, so this degenerates to
  the previous single-trial behavior.

  `Encoder` gained an `effort: Effort` field, set from
  `StillHevcConfig::effort` in `encode()`.

  Verified:

  cargo build -p bpg-still-hevc
  cargo fmt -p bpg-still-hevc
  cargo test -p bpg-still-hevc --test encode_roundtrip -- --test-threads=1  (8/8 pass; default
  `Effort::Balanced` shows small bytes/PSNR shifts vs. the single-candidate
  baseline, e.g. full_image_multi_ctu 624->613 bytes, single_ctu_qps qp=18
  56.03->59.77 dB — confirming the extra candidate is doing useful RD work)
  cargo test  (full workspace, 0 failures)

  Remaining for further parity: chroma mode decision is still SATD-only (no
  full RD per candidate); strong intra smoothing / reference-sample filtering
  parity with x265's 32x32 bilinear path hasn't been audited; `bpg-tools`
  doesn't yet expose `--effort` to pick `Fast`/`Balanced`/`Best` from the CLI.

## Progress update (chroma mode full RD + strong-intra-smoothing audit)

• Made concrete progress toward further x265 intra parity:

  Audited strong intra smoothing: `bpg-still-hevc/src/params.rs` already
  writes `strong_intra_smoothing_enabled_flag = 1` in the SPS (matching
  x265's default), and every encoder call into
  `bpg_hevc_decode::hevc::intra::predict_intra` already passes
  `strong_intra_smoothing_enabled = true`, so the decoder's 32x32 bilinear
  reference-sample path (`intra_prediction_sample_filtering` in
  `bpg-hevc-decode/src/hevc/intra.rs`) is already exercised consistently by
  both the encoder's trial predictions and its final reconstruction. No
  change needed — this gap was already closed.

  Extended chroma mode decision (`choose_chroma_mode`,
  crates/bpg-still-hevc/src/encoder.rs) the same way as luma: all 5 H.265
  Table 8-2 candidates (Planar/Vertical/Horizontal/DC/DM) are still scored by
  SATD + mode bits first, but now the best `chroma_rd_candidates(effort)` of
  those get a full RD trial via `code_block` (actual
  transform/quantize/RDOQ on the CU-level chroma TB(s)), with RD cost =
  `distortion + lambda * (residual_bits + mode_idx_bits)`. New
  `chroma_rd_candidates`: `Effort::Fast` -> 1 (original SATD-only decision,
  unchanged), `Effort::Balanced` -> 2, `Effort::Best` -> 5 (full search over
  all candidates).

  Verified:

  cargo build -p bpg-still-hevc
  cargo fmt -p bpg-still-hevc
  cargo test -p bpg-still-hevc --test encode_roundtrip -- --test-threads=1  (8/8 pass; default
  `Effort::Balanced` improves chroma quality at equal-or-smaller size, e.g.
  yuv444_round_trip cb PSNR 56.31->58.41 dB while luma bytes drop 141->139)
  cargo test  (full workspace, 0 failures; one x265-primitives test failure
  on the first parallel run was a transient/flaky rerun issue, not
  reproducible in isolation or on a clean re-run)

  Remaining for further parity: `bpg-tools` doesn't expose `--effort`
  (`Fast`/`Balanced`/`Best`) from the CLI — this is blocked on the
  not-yet-done "Rust backend correction" roadmap item (`RustStillHevcEncoder`
  for `bpg_encode::HevcEncoder`); `bpg-still-hevc::encoder::encode` currently
  has no caller outside its own test suite.

## Progress update (Rust backend wired into bpg-tools + stock-bpgdec decode-conformance fix)

  Done. The Rust-native still encoder is now a first-class `bpg-tools`
  backend and its output decodes with the **stock** `bpgdec` — the M1
  acceptance bar (`CLAUDE.md`).

  - **`RustStillHevcEncoder`** (`crates/bpg-still-hevc/src/backend.rs`)
    implements `bpg_encode::HevcEncoder`: converts the CTU-padded `Image`
    planes into a `Source`, builds a `StillHevcConfig` (SAO/deblock off),
    calls `encoder::encode`, and returns the Annex-B for
    `bpg_encode::encode_still_image` -> `bpg_hevc::build_modified_hevc`.
    Rejects non-4:2:0/4:4:4 chroma and non-8/10-bit depth as
    `EncodeError::Unsupported` (no panic).
  - **CLI**: `bpg-tools encode ... --backend rust --effort {fast,balanced,best}`.
  - **The blocking bug — `write_pps` was missing `pps_extension_present_flag`.**
    Stock `bpgdec`'s bundled (patched-ffmpeg) `ff_hevc_decode_nal_pps` reads
    `pps_extension_present_flag` immediately after
    `slice_segment_header_extension_present_flag`; the vendored Rust decoder's
    `parse_pps` stops one field earlier. Omitting the flag made `bpgdec`
    misread `rbsp_stop_one_bit` as `pps_extension_present_flag = 1`, then
    overread the PPS by a byte (`"Overread PPS by 8 bits"` -> `"PPS id out of
    range"` -> `"Could not decode image"`). Because the encoder and the
    vendored decoder *shared* the missing field, the internal roundtrip
    passed and hid it — only the stock decoder caught it. Fixed by writing
    `pps_extension_present_flag = 0` (`crates/bpg-still-hevc/src/params.rs`).
    Also fixed earlier in this effort: `write_vps` was truncated after
    `profile_tier_level` (made the *raw* Annex-B stream non-conformant for
    strict demuxers); now writes the full H.265 7.3.2.1 VPS body.
  - **How it was isolated**: built a debug `bpgdec` from `libbpg-0.9.8` with
    `-DUSE_AV_LOG -DDEBUG` so the bundled decoder's `av_log` errors print —
    that named the exact failing check. (Internal vendored-decoder roundtrips
    and `ffmpeg -f hevc` on the VPS-stripped rebuilt stream are *not* valid
    conformance checks for this; only the stock decoder is.)
  - **Regression guard**: `params::tests::pps_rbsp_terminates_for_conformant_decoder`
    parses `write_pps()` to completion *the way a conformant decoder does*
    (through `pps_extension_present_flag`) and asserts clean
    `rbsp_trailing_bits` termination — catching a missing/extra PPS field
    that the shared-blind-spot roundtrip cannot.

  Verification (all decode with stock `/usr/local/bin/bpgdec`, exit 0):
  `{420,444} x {fast,balanced,best} x {8,10-bit}` on a 200x200 multi-CTU
  crop of `dusk.png`, plus the 64x64 single-CTU crop. PSNR vs source is
  sane (e.g. 64x64 qp32 420: rust 37.6 dB vs x265 39.0 dB; 200x200 qp30:
  ~35 dB) — a valid, slightly-less-efficient encode, not garbage.
  Constraint honored: only `bpg-still-hevc` (`params.rs`/`backend.rs`) was
  changed; `bpg-hevc`'s `rewrite_sps`/`build_modified_hevc` were left
  untouched — the encoder was fixed to match x265/the decoder, not the
  reverse.

  Commands:
  cargo fmt
  cargo test  (full workspace, 0 failures)
  cargo run -p bpg-tools -- encode <png> -o out.bpg --backend rust \
      --effort best --qp 30 --format 444 -b 10   # decodes with stock bpgdec
