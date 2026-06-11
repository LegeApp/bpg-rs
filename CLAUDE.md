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
  4:4:4 chroma, no alpha. Alpha, >8-bit, lossless, animation, RGB/YCgCo color
  spaces are explicit `TODO(extension)` / `unimplemented!()` stubs — types
  are designed so adding them later doesn't require redesign.
- M1 acceptance target:
  ```
  cargo run -p bpg-tools -- encode ../dusk.png -o /tmp/out.bpg --backend x265 --qp 28 --format 420
  ```
  must produce a `.bpg` that decodes with the stock `bpgdec` and is within
  ~±0.2 dB PSNR / ±0.002 SSIM of the equivalent C `bpgenc -e x265` output.

### Workspace layout & status

`bpg-rs/Cargo.toml` is a Cargo workspace (resolver "2", edition 2021). Member
crates, in dependency order:

| Crate | Status | Purpose |
|---|---|---|
| `bpg-bitstream` | **Done**, tested | `ue7` base-128 varint (container header fields) + MSB-first `BitReader`/`BitWriter` with Exp-Golomb `ue(v)` (HEVC RBSP). Ported from `put_ue`/`get_ue`/`get_bits`/`put_bits`/`*_ue_golomb` in `bpgenc.c`. No deps. |
| `bpg-image` | **Done**, tested | `Plane<T>`, `Image`, `ChromaFormat`, `ColorSpace`, RGB→YCbCr `ColorConvertState` (`convert.rs`, BT.601/709/2020, full+limited range), 4:4:4→4:2:0 chroma decimation (`chroma.rs`, `h_phase==1` only), CTU padding (`pad.rs`). Depends on `image` crate (PNG decode). `Rgb`/`YCgCo` color spaces are `unimplemented!()`. |
| `bpg-format` | **Done**, tested | `BpgHeader` — the fixed-size BPG container header, byte-for-byte verified against a real `bpgenc` header for `dusk.png`'s dimensions. Depends on `bpg-bitstream`. |
| `bpg-hevc` | **Not started** (`// TODO` stub) | NAL extraction (Annex-B start codes, emulation-prevention removal) + `build_modified_sps`/`build_modified_hevc` — the highest-risk remaining piece. Ported from `bpgenc.c:1496-2169`. Depends on `bpg-bitstream`. |
| `bpg-encode` | **Not started** (`// TODO` stub) | `HevcEncoder` trait + `encode_still_image` orchestration. Depends on `bpg-image`, `bpg-format`, `bpg-hevc`. |
| `bpg-x265-sys` | **Not started** (`build.rs` is a no-op stub) | Vendored x265 build (cmake) + raw `bindgen` FFI bindings, including handling the versioned `x265_api_get_215` symbol. `links = "x265"`. |
| `bpg-x265` | **Not started** (`// TODO` stub) | Safe wrapper implementing `bpg-encode::HevcEncoder` over `bpg-x265-sys`. |
| `bpg-tools` | **Not started** (placeholder `main`) | CLI (`encode` subcommand) via `clap`, ties `bpg-image` + `bpg-encode` + `bpg-x265` together. |

Dependency graph: `bpg-bitstream` → `bpg-format`, `bpg-hevc`; `bpg-image` is
standalone (uses the `image` crate); `bpg-encode` depends on `bpg-image` +
`bpg-format` + `bpg-hevc`; `bpg-x265-sys` → `bpg-x265` (implements
`HevcEncoder`); `bpg-tools` depends on all of the above via `clap`.

**Current next step** (per PLAN.md): implement `bpg-hevc`
(`extract_nals`, `ModifiedSps::from_hevc_stream`, `build_modified_hevc`),
then `bpg-x265-sys`, `bpg-x265`, `bpg-encode`, `bpg-tools`, then end-to-end
verification.

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
   >8-bit, animation, lossless, RGB/YCgCo color spaces) so the type design
   doesn't preclude adding them later, but don't implement them in M1.
5. **Use `unimplemented!()`/precondition asserts**, not silent incorrect
   behavior, for unsupported inputs in this milestone.

## Build & test

The Cargo workspace lives in `bpg-rs/` — run all `cargo` commands from
there:

```bash
cd bpg-rs
cargo build              # builds all crates (bpg-x265-sys's build.rs is currently a no-op)
cargo test                # runs unit tests across the workspace
cargo test -p bpg-image   # test a single crate
```

As of this writing, `bpg-bitstream`, `bpg-image`, and `bpg-format` have
passing unit tests (8 + 2 + 8); the remaining crates are empty stubs and
build trivially. `cargo build`/`cargo test` should succeed from a clean
checkout without needing x265 — the vendored x265 build only kicks in once
`bpg-x265-sys/build.rs` is implemented.

Toolchain: edition 2021, `cargo`/`rustc` 1.94+. Workspace deps: `image`
0.25 (PNG only), `clap` 4.6 (derive), `bindgen` 0.72 + `cc` 1 (build-deps for
the future `bpg-x265-sys`).

End-to-end verification (once `bpg-tools` exists) compares against the C
reference using `libbpg-0.9.8/bpgenc`/`bpgdec` and `ffmpeg` SSIM/PSNR — see
the "Verification" section of `bpg-rs/PLAN.md` for the exact commands.

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
