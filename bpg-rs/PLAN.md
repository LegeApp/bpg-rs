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
