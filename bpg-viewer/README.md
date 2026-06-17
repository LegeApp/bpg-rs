# bpg-view

A minimal, fast BPG image viewer. Thin window — almost the entire surface is the
image. Starts instantly, loads the file given on the command line, and reports a
clear error in-window (and on stderr) instead of crashing when a file can't be
decoded.

## Build

```bash
cargo build --release
# binary: target/release/bpg-view
```

Pure Rust — no system libraries required. Decoding is done by the `bpg-decode`
crate from the sibling `bpg-rs` workspace (path dependency
`../bpg-rs/crates/bpg-decode`), so there is **no `libbpgdec.so` runtime
dependency** anymore. This also brings bpg-rs's large-image support: it decodes
very large stills (e.g. 18432×9216 / ~170 MP heightmaps) that crash the C
`bpgdec`/some third-party viewers.

## Usage

```bash
bpg-view path/to/image.bpg
```

You can also launch it with no argument and drop a `.bpg` file onto the window.

### Controls

| Key / action      | Effect                |
|-------------------|-----------------------|
| Scroll / `+` `-`  | Zoom                  |
| Drag              | Pan                   |
| `F`               | Fit to window         |
| `1`               | Actual size (100%)    |
| Drop a file       | Open it               |
| `Q` / `Esc`       | Quit                  |

## Layout

- `src/main.rs`    — the viewer (egui/eframe, glow renderer)
- `src/decoder.rs` — thin wrapper over the pure-Rust `bpg-decode` crate
  (RGBA8 output; YCbCr→RGB / BT.601/709/2020 handled inside `bpg-decode`)
