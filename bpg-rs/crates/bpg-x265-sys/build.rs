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
//!
//! ## 10-bit support (multilib build)
//!
//! A single x265 build only supports one internal bit depth. To support
//! 10-bit BPGs alongside the 8-bit path, this build script follows x265's
//! standard "multilib" recipe (`build/linux/multilib.sh` upstream):
//!
//! 1. Build a 10-bit static lib with `HIGH_BIT_DEPTH=ON`, `MAIN12=OFF`,
//!    `EXPORT_C_API=OFF` (its public symbols land in the `x265_10bit::` C++
//!    namespace, set via the `X265_NS` macro), into a separate cmake build
//!    directory, and copy its `libx265.a` out as `libx265_main10.a`.
//! 2. Build the normal 8-bit static lib with `EXPORT_C_API=ON` (default,
//!    namespace `x265`), `LINKED_10BIT=ON`, and `EXTRA_LIB` pointing at
//!    `libx265_main10.a`.
//!
//! The main lib's `x265_api_query`/`x265_api_get` (in `encoder/api.cpp`)
//! dispatch `bitDepth == 10` requests into `x265_10bit::x265_api_query`.
//! At link time both static libs are passed to the linker in
//! `-lx265 -lx265_main10` order, which resolves the cross-namespace
//! references in a single left-to-right pass (the 10-bit lib has no
//! references back into the main lib).
//!
//! Set `BPG_X265_SKIP_10BIT=1` to build only the 8-bit lib (faster
//! iteration); `bpg-x265` will then only support `bit_depth == 8`.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/bpg-x265-sys -> crates -> bpg-rs -> repo root -> x265_4.1/source
    let x265_src = manifest_dir
        .join("../../../x265_4.1/source")
        .canonicalize()
        .expect("vendored x265 source not found at ../../../x265_4.1/source");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=BPG_X265_ENABLE_ASM");
    println!("cargo:rerun-if-env-changed=BPG_X265_SKIP_10BIT");

    let enable_asm = env::var("BPG_X265_ENABLE_ASM").as_deref() == Ok("1");
    let skip_10bit = env::var("BPG_X265_SKIP_10BIT").as_deref() == Ok("1");

    // --- Optionally build the 10-bit lib first (EXPORT_C_API=0, namespace
    // x265_10bit), so its archive can be linked into the main 8-bit build. ---
    let main10_lib_dir = if !skip_10bit {
        let dst10 = cmake::Config::new(&x265_src)
            .out_dir(out_dir.join("x265-10bit"))
            .define("ENABLE_SHARED", "OFF")
            .define("ENABLE_CLI", "OFF")
            .define("ENABLE_ASSEMBLY", if enable_asm { "ON" } else { "OFF" })
            .define("ENABLE_TESTS", "OFF")
            // Disable NUMA pool support: it makes the build depend on whether
            // libnuma is installed (CMake auto-detects it), which is
            // non-deterministic and would otherwise require linking -lnuma.
            // BPG encodes one intra frame, so NUMA thread pools are moot.
            .define("ENABLE_LIBNUMA", "OFF")
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("HIGH_BIT_DEPTH", "ON")
            .define("MAIN12", "OFF")
            .define("EXPORT_C_API", "OFF")
            .build_target("x265-static")
            .build();
        let build_dir10 = dst10.join("build");

        let main10_lib_dir = out_dir.join("x265-main10-lib");
        std::fs::create_dir_all(&main10_lib_dir).expect("failed to create main10 lib dir");
        std::fs::copy(
            build_dir10.join("libx265.a"),
            main10_lib_dir.join("libx265_main10.a"),
        )
        .expect("failed to copy 10-bit libx265.a to libx265_main10.a");
        Some(main10_lib_dir)
    } else {
        None
    };

    // --- Build the main x265 lib (8-bit, EXPORT_C_API=1, no CLI) via cmake,
    // linking in the 10-bit lib if present. ---
    let mut main_cfg = cmake::Config::new(&x265_src);
    main_cfg
        .define("ENABLE_SHARED", "OFF")
        .define("ENABLE_CLI", "OFF")
        .define("ENABLE_ASSEMBLY", if enable_asm { "ON" } else { "OFF" })
        .define("ENABLE_TESTS", "OFF")
        // See the 10-bit config above: keep NUMA off for a deterministic,
        // libnuma-independent build.
        .define("ENABLE_LIBNUMA", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("x265-static");
    if let Some(main10_lib_dir) = &main10_lib_dir {
        main_cfg
            .define(
                "EXTRA_LIB",
                main10_lib_dir.join("libx265_main10.a").to_str().unwrap(),
            )
            .define("LINKED_10BIT", "ON");
    }
    let dst = main_cfg.build();

    // cmake-rs builds into <dst>/build; the static lib and generated
    // x265_config.h both land there.
    let build_dir = dst.join("build");
    let config_h_dir = find_x265_config_h(&build_dir)
        .expect("generated x265_config.h not found under the cmake build dir");

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=x265");
    if let Some(main10_lib_dir) = &main10_lib_dir {
        println!("cargo:rustc-link-search=native={}", main10_lib_dir.display());
        println!("cargo:rustc-link-lib=static=x265_main10");
    }
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

    let out_path = out_dir.join("bindings.rs");
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
