//! Build the vendored x265 (release 4.1, `X265_BUILD=215`) as a static
//! library and generate raw FFI bindings against `x265.h`.
//!
//! Per `bpg-rs/PLAN.md`, x265 is vendored at `../../../x265_4.1/source`
//! (relative to this crate) and built via cmake rather than linking the
//! system package, so the BPG port stays pinned to a known-good x265.
//!
//! The environment used to develop this port has no `nasm`/`yasm`, so the
//! build disables x265 assembly (`ENABLE_ASSEMBLY=OFF`). This produces a
//! correct but slower encoder, which is fine for M1 (correctness, not speed).
//! Set `BPG_X265_ENABLE_ASM=1` to re-enable assembly where an assembler is
//! available.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/bpg-x265-sys -> crates -> bpg-rs -> repo root -> x265_4.1/source
    let x265_src = manifest_dir
        .join("../../../x265_4.1/source")
        .canonicalize()
        .expect("vendored x265 source not found at ../../../x265_4.1/source");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=BPG_X265_ENABLE_ASM");

    let enable_asm = env::var("BPG_X265_ENABLE_ASM").as_deref() == Ok("1");

    // --- Build x265 (static, 8-bit, no CLI) via cmake. ---
    let dst = cmake::Config::new(&x265_src)
        .define("ENABLE_SHARED", "OFF")
        .define("ENABLE_CLI", "OFF")
        .define("ENABLE_ASSEMBLY", if enable_asm { "ON" } else { "OFF" })
        .define("ENABLE_TESTS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("x265-static")
        .build();

    // cmake-rs builds into <dst>/build; the static lib and generated
    // x265_config.h both land there.
    let build_dir = dst.join("build");
    let config_h_dir = find_x265_config_h(&build_dir)
        .expect("generated x265_config.h not found under the cmake build dir");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=x265");
    // x265 is C++ and uses pthreads / libm / libdl.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");

    // --- Generate FFI bindings against x265.h. ---
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", x265_src.display()))
        .clang_arg(format!("-I{}", config_h_dir.display()))
        // Public API surface only.
        .allowlist_type("x265_.*")
        .allowlist_function("x265_.*")
        .allowlist_var("X265_.*")
        .allowlist_var("x265_api_query_errnames")
        .layout_tests(false)
        .generate()
        .expect("bindgen failed to generate x265 bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
}

/// cmake may place `x265_config.h` directly in `build_dir` or in a
/// subdirectory depending on generator; search for it.
fn find_x265_config_h(build_dir: &Path) -> Option<PathBuf> {
    let direct = build_dir.join("x265_config.h");
    if direct.exists() {
        return Some(build_dir.to_path_buf());
    }
    let mut stack = vec![build_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "x265_config.h") {
                return path.parent().map(|p| p.to_path_buf());
            }
        }
    }
    None
}
