# CLAUDE.md

Guidance for AI assistants (Claude Code and similar) working in this
repository.

## What this repository is

This repo holds the **BPG image format** reference materials plus an
in-progress **Rust port of the BPG encoder** (`bpg-rs/`). BPG ("Better
Portable Graphics") is Fabrice Bellard's HEVC-intra-based image format.

The active development work is the Rust port at `bpg-rs/`. Everything else
in the repo root is reference material the port is built from / validated
against:

```
.
├── bpg-rs/            # <-- Rust workspace under active development
├── libbpg-0.9.8/      # original C reference implementation (bpgenc/bpgdec/libbpg)
├── x265_4.1/          # vendored x265 source (release tag 4.1, X265_BUILD=215)
├── bpg_spec.txt        # the BPG file format specification
├── test_runs/          # benchmark .bpg outputs + scripts comparing JCTVC vs x265
├── doc/, llm-docs/     # historical notes from earlier (non-Rust) build attempts
├── outside-advice.md   # design discussion that informed the Rust port's architecture
├── dusk.png            # 3200x2528 RGBA test photo used for end-to-end verification
└── html/               # BPG JS decoder demo (from upstream libbpg)
```

`doc/`, `llm-docs/`, and `outside-advice.md` document earlier attempts to
build/integrate the **C** JCTVC/x265 encoders (Windows builds, OpenArc DLL
integration, etc.). They are historical context, not part of the active
Rust port — don't treat instructions in them as current unless cross-checked
against `bpg-rs/PLAN.md`.

## The Rust port: `bpg-rs/`

`bpg-rs/PLAN.md` is the source of truth for design decisions, scope, and
current progress. **Read it before making architectural changes** — it
records *why* things are the way they are (e.g. why JCTVC was dropped, why
x265 is vendored rather than using the system package, the exact byte
layouts being targeted). When you complete a milestone or change the plan,
update `bpg-rs/PLAN.md`'s progress section.

### Key decisions (see PLAN.md for full rationale)

- **x265 only** as the HEVC backend. JCTVC was benchmarked against x265 on
  `dusk.png` (`test_runs/results.csv`) and found not worth porting (≤4%
  smaller in narrow cases, worse on 4:4:4/SSIM, 30-50x slower).
- `bpg-x265-sys` will **vendor and build x265 from `x265_4.1/source`**
  (release tag 4.1, `X265_BUILD=215`) via `build.rs` + `cmake` + `bindgen`,
  rather than linking the system `libx265`. Path from
  `bpg-rs/crates/bpg-x265-sys/` to the vendored source is
  `../../../x265_4.1/source`.
- **Milestone 1 (M1) scope**: 8-bit PNG input, still images only (no
  animation), lossy CQP only, YCbCr color space (BT.601/709/2020), 4:2:0 and
  4:4:4 chroma, no alpha. Alpha, lossless, animation, RGB/YCgCo color
  spaces are explicit `TODO(extension)` / `unimplemented!()` stubs — types
  are designed so adding them later doesn't require redesign.
- **Roadmap decision (post-M1)**: alpha and animation are dropped from the
  near-term roadmap (alpha is rarely meaningful in PNGs from raw-image
  pipelines; the existing `TODO(extension)`/`has_alpha` stubs are left as-is
  but deprioritized). **10-bit support has been added** (see "Progress update
  (10-bit support)" in PLAN.md): `Image` stores samples as `u16` internally
  (matching `bpgenc.c`'s `PIXEL = uint16_t`), `bpg-x265-sys` does an x265
  multilib build (8-bit + linked 10-bit), and `bpg-tools` has a
  `-b/--bit-depth 8|10|12` option (12-bit lib not yet built).
- **Roadmap update (reprioritization + tuning system)**: 12-bit and lossless
  are deferred to much later. **4:2:2 chroma is a near-term priority**
  alongside the already-implemented 4:2:0. The "candidate optimizer"
  (pick-the-best-of-N-encodes) is **replaced** by a conservative,
  source-aware **tuning system** for archival photo encoding: a `Tune` enum
  (`Auto`/`Neutral`/`Photo`/`FilmGrain`/`Slide`/`LowLight`/`Artwork`/
  `Screenshot`/`Scan`), an `ArchivalPolicy` (controls metadata/ICC
  preservation, chroma subsampling, bit-depth reduction, and whether any
  preprocessing is allowed), and an `EncodePlan` (resolved x265 params) —
  produced by new `bpg-analyze` (feature extraction + tune classification)
  and `bpg-tune` (Tune + ArchivalPolicy + ImageFeatures -> EncodePlan) crates.
  No tune may apply destructive preprocessing, chase metrics at the expense
  of source character, or drop metadata/chroma/bit-depth without policy
  permission. A bounded `--candidate-check conservative` mode (small
  parameter neighborhood, quality floor not quality-maximizing) may follow
  later. See PLAN.md's "Roadmap update (reprioritization + tuning system)"
  for the full design.
- M1 acceptance target:
  ```
  cargo run -p bpg-tools -- encode ../dusk.png -o /tmp/out.bpg --backend x265 --qp 28 --format 420
  ```
  must produce a `.bpg` that decodes with the stock `bpgdec` and is within
  ~±0.2 dB PSNR / ±0.002 SSIM of the equivalent C `bpgenc -e x265` output.

### Workspace layout & status

`bpg-rs/Cargo.toml` is a Cargo workspace (resolver "2", edition 2021). Member
crates, in dependency order:

**Milestone 1 is complete and verified** — the `encode` CLI produces a `.bpg`
that decodes with the stock `bpgdec` and is byte-for-byte pixel-identical to
the C `bpgenc -e x265` reference (see PLAN.md's "Milestone 1 COMPLETE"
section). All eight crates are implemented and `cargo test` passes (35 tests).
**10-bit support has since been added** (see PLAN.md's "Progress update
(10-bit support)"), and **4:2:2 chroma is implemented** (see PLAN.md's
"Progress update (4:2:2 chroma support)") — its output is also byte-for-byte
identical to the C reference.

| Crate | Status | Purpose |
|---|---|---|
| `bpg-bitstream` | **Done**, tested | `ue7` base-128 varint (container header fields) + MSB-first `BitReader`/`BitWriter` with Exp-Golomb `ue(v)` (HEVC RBSP). Ported from `put_ue`/`get_ue`/`get_bits`/`put_bits`/`*_ue_golomb` in `bpgenc.c`. No deps. |
| `bpg-image` | **Done**, tested | `Plane<T>`, `Image` (`planes: Vec<Plane<u16>>`, matching `bpgenc.c`'s internal `PIXEL = uint16_t`), `ChromaFormat`, `ColorSpace`, RGB→YCbCr `ColorConvertState` (`convert.rs`, BT.601/709/2020, full+limited range, generic over input sample type and 8/10/12-bit output), 4:4:4→4:2:0 and 4:4:4→4:2:2 chroma decimation (`chroma.rs`, `decimate_to_420`/`decimate_to_422`, `h_phase==1` only, `u16`; `Image::subsample_to_420`/`subsample_to_422`), CTU padding (`pad.rs`, generic `pad_plane<T: Copy + Default>`). `Image::from_rgb8`/`from_rgb16` both take a `bit_depth`. Depends on `image` crate (PNG decode, incl. 16-bit). `Rgb`/`YCgCo` color spaces are `unimplemented!()`. |
| `bpg-format` | **Done**, tested | `BpgHeader` — the fixed-size BPG container header, byte-for-byte verified against a real `bpgenc` header for `dusk.png`'s dimensions. Depends on `bpg-bitstream`. |
| `bpg-hevc` | **Done**, tested | `find_nal_end`/`extract_nal` (Annex-B start codes + emulation-prevention removal), `ModifiedSps::from_hevc_stream` (VPS+SPS parse/rewrite with precondition checks → `HevcError`), `build_modified_hevc` (no-alpha still-image). Ported from `bpgenc.c:1496-2169`. Depends on `bpg-bitstream`. |
| `bpg-encode` | **Done** | `HevcEncoder` trait, `HevcEncodeParams`, `EncodeError`, `encode_still_image` orchestration (pad → encode → `build_modified_hevc` → header+payload). Accepts `bit_depth` 8/10/12. Depends on `bpg-image` + `bpg-format` + `bpg-hevc`. |
| `bpg-x265-sys` | **Done** | `build.rs` builds the vendored x265 (`../../../x265_4.1/source`) via the `cmake` crate + `bindgen` FFI from `wrapper.h`. Does an x265 **multilib build**: an 8-bit lib (`EXPORT_C_API=ON`, `LINKED_10BIT=ON`) linked against a separately-built 10-bit lib (`HIGH_BIT_DEPTH=ON`, `EXPORT_C_API=OFF`, namespace `x265_10bit`, copied out as `libx265_main10.a`). `ENABLE_ASSEMBLY=OFF` by default (no `nasm`/`yasm` in dev env; `BPG_X265_ENABLE_ASM=1` to enable; `BPG_X265_SKIP_10BIT=1` to build only the 8-bit lib). `links = "x265"`. Uses `x265_api_query` (non-versioned) rather than the `x265_api_get_215` symbol. |
| `bpg-x265` | **Done** | Safe `X265Encoder` implementing `bpg-encode::HevcEncoder`, ported from `x265_glue.c` (single-frame intra). For `bit_depth == 8` truncates the internal `u16` planes to `u8` (mirroring `image_convert16to8`); for 10/12-bit passes `u16` plane data with byte stride. Defines `X265_RC_CQP`/preset-name constants that bindgen omits. |
| `bpg-hevc-decode` | **Done**, tested | Vendored pure-Rust HEVC still-picture intra **decoder** (`src/hevc/*` copied verbatim from `heic-decoder-rs`), with a local `error::HevcError` and a minimal `heif::HevcDecoderConfig` shim replacing the upstream's container/cancellation deps. External deps trimmed to `archmage` + `safe_unaligned_simd`; edition 2024 (let-chains). Entry point `hevc::decode(annexb) -> DecodedFrame`. See `bpg-rs/FULL_RUST_CODEC_PLAN.md` (Phase 1). |
| `bpg-decode` | **Done**, tested | BPG still-image **decode** orchestration: BPG container -> `bpg-hevc::rebuild_annexb_from_bpg_payload` -> `bpg-hevc-decode` -> RGBA/RGB/BGR(A) output. Public API: `DecoderConfig`, `PixelLayout`, `DecodeOutput`, `ImageInfo`, `Limits`, `DecodeError`. Rejects animation/alpha/W-plane/MPEG2-siting/RGB/YCgCo **and 4:2:2/4:4:4** as `Unsupported` (the vendored decoder is **4:2:0-only** — 4:2:2/4:4:4 decode to garbage, surfaced by the oracle; see `FULL_RUST_CODEC_PLAN.md` Phase 1 follow-up). Fixture tests in `tests/fixtures.rs`. |
| `bpg-oracle` | **Done**, tested | x265 reference oracle (`gen`/`check`): a deterministic regression corpus (flat/gradient/checkerboard/lineart/noise/HDR-gradient/photo crop) encoded across {4:2:0,4:2:2,4:4:4}×{qp}×{8,10-bit} with the locked still-image settings. Records bpg size, rebuilt Annex-B, the ground-truth **stock-bpgdec** decode, and two metric axes (encode-quality PSNR vs source; Rust-decoder status/fidelity). `manifest.csv`+`SETTINGS.md` committed, `corpus/`+`out/` regenerable. The Phase-3 Rust-encoder comparison baseline. |
| `bpg-tools` | **Done** | `clap` CLI with the `encode` subcommand, a `-b/--bit-depth 8\|10\|12` option (16-bit PNG input read via `from_rgb16`), and `--format 420\|422\|444`, ties `bpg-image` + `bpg-encode` + `bpg-x265` together. |

Dependency graph: `bpg-bitstream` → `bpg-format`, `bpg-hevc`; `bpg-image` is
standalone (uses the `image` crate); `bpg-encode` depends on `bpg-image` +
`bpg-format` + `bpg-hevc`; `bpg-x265-sys` → `bpg-x265` (implements
`HevcEncoder`); `bpg-hevc-decode` is standalone (vendored decoder); `bpg-decode`
depends on `bpg-bitstream` + `bpg-format` + `bpg-hevc` + `bpg-hevc-decode`;
`bpg-tools` depends on all of the above via `clap`.

The full Rust BPG codec roadmap (vendoring the decoder, then a Rust
x265-derived still encoder) lives in `bpg-rs/FULL_RUST_CODEC_PLAN.md`. **Phase 1
(vendor the decoder, remove the external `heic-decoder-rs` path dependency) is
complete** — the repo is now self-contained.

**Next milestones**: alpha-plane support, `c_h_phase==0` (MPEG2 chroma
siting), RGB/YCgCo color spaces, and the **tuning system** (`bpg-analyze` +
`bpg-tune` crates, `Tune`/`ArchivalPolicy`/`EncodePlan` — see PLAN.md's
"Roadmap update (reprioritization + tuning system)"), which replaces the old
"candidate optimizer". **4:2:2 chroma is done** (see PLAN.md's "Progress
update (4:2:2 chroma support)"). **Deferred to much later**: 12-bit (a second
multilib lib with `MAIN12=ON`) and lossless. The `TODO(extension)` type stubs
already leave room for all of these. Animation is dropped from the roadmap.

### Building the x265 FFI crate

`bpg-x265-sys/build.rs` compiles the vendored x265 on first `cargo build`,
which needs `cmake` + a C/C++ toolchain. It builds x265 **twice** (a 10-bit
lib with `EXPORT_C_API=OFF`, then the main 8-bit lib with `LINKED_10BIT=ON`
linking the 10-bit archive) — see PLAN.md's "Progress update (10-bit
support)" for the full multilib recipe. Set `BPG_X265_SKIP_10BIT=1` to build
only the 8-bit lib for faster iteration (then `bit_depth == 10/12` encodes
will fail). The development environment has no assembler, so assembly is
disabled by default (correct but slower); set `BPG_X265_ENABLE_ASM=1` to
re-enable it where `nasm`/`yasm` is installed. Building `bpgdec` (to verify
output) is `make bpgdec` in `libbpg-0.9.8/`.

## Porting conventions

This is a **line-by-line port of `libbpg-0.9.8/bpgenc.c`** (and friends), not
a redesign. When implementing a new crate/module:

1. **Cite the C source** in a doc comment at the top of the file/module,
   e.g. `//! Ported from \`decimate2_hv\`/\`decimate2_h16\`/\`decimate2_v\` in
   \`libbpg-0.9.8/bpgenc.c\`.` — this is how every existing module
   (`bpg-bitstream`, `convert.rs`, `chroma.rs`, `pad.rs`, `bpg-format`) is
   documented, and lets future readers cross-check behavior against the
   original.
2. **Match the C arithmetic exactly** — same integer widths, same shift/round
   behavior, same clamping (`clamp_pix`), same rounding mode (`lrint` ==
   `f64::round_ties_even` here). Subtle differences (e.g. rounding direction)
   will produce a *different but plausible-looking* bitstream that fails
   byte-for-byte / PSNR comparisons against the C reference.
3. **Write unit tests with known reference values**, derived either from
   hand-computed/Python cross-checks or from real `bpgenc` output (e.g.
   `bpg-format`'s test asserts the exact header bytes for `dusk.png`'s
   dimensions: `42 50 47 fb 20 00 99 00 93 60 00`). When a feature is only
   partially ported (e.g. `decimate_to_420` only handles `h_phase == 1`),
   document the restriction with `assert_eq!`/`assert!` preconditions and a
   doc comment, not silent fallback.
4. **Use `TODO(extension)` for explicitly out-of-scope features** (alpha,
   12-bit, lossless, RGB/YCgCo color spaces) so the type design doesn't
   preclude adding them later, but don't implement them until their
   milestone. Animation is dropped from the roadmap entirely.
5. **Use `unimplemented!()`/precondition asserts**, not silent incorrect
   behavior, for unsupported inputs in this milestone.

## Build & test

The Cargo workspace lives in `bpg-rs/` — run all `cargo` commands from
there:

```bash
cd bpg-rs
cargo build              # builds all crates, incl. the vendored x265 multilib build (cmake)
cargo test                # runs unit tests across the workspace
cargo test -p bpg-image   # test a single crate
```

`cargo build`/`cargo test` succeed from a clean checkout, but the first build
compiles vendored x265 twice (8-bit + linked 10-bit, see "Building the x265
FFI crate" above), which needs `cmake` + a C/C++ toolchain and takes
significantly longer than a pure-Rust build. `cargo test` runs 35 unit tests
across `bpg-bitstream`, `bpg-image`, `bpg-format`, `bpg-hevc` (the other
crates have no unit tests of their own).

Toolchain: edition 2021, `cargo`/`rustc` 1.94+. Workspace deps: `image` 0.25
(PNG, incl. 16-bit), `clap` 4.6 (derive), `bindgen` 0.72 + `cmake` 0.1
(build-deps for `bpg-x265-sys`).

End-to-end verification compares against the C reference using
`libbpg-0.9.8/bpgenc`/`bpgdec` and `ffmpeg` SSIM/PSNR — see the
"Verification"/"Progress update" sections of `bpg-rs/PLAN.md` for the exact
commands.

## Reference materials cheat sheet

- `bpg_spec.txt` — authoritative BPG container format spec (`heic_file()`
  syntax etc.). `bpg-format` implements this.
- `libbpg-0.9.8/bpgenc.c` — the C encoder being ported. Line numbers cited in
  `PLAN.md` and crate doc comments refer to this file.
- `x265_4.1/source/` — vendored x265 4.1 source that `bpg-x265-sys` will
  build. Do not confuse with `libbpg-0.9.8/x265/` (an older, unused
  `X265_BUILD=75` copy).
- `test_runs/results.csv` + `run_grid.py`/`run_grid2.py` — the JCTVC-vs-x265
  benchmark data that motivated the x265-only decision.
- `dusk.png` — the canonical test image (3200x2528, RGBA) used for header
  byte-matching and end-to-end PSNR/SSIM checks. Note it has an alpha
  channel; M1 ignores alpha, so verification should compare against an
  RGB-only copy (`ffmpeg -i dusk.png -pix_fmt rgb24 dusk_rgb.png`).
