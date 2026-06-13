//! Builds small, independent subsets of the vendored x265 4.1 source
//! (`../../../x265_4.1/source/common`) -- just the scalar (`_c`) primitive
//! translation units Phase 4 needs -- plus local `extern "C"` shims.
//!
//! This is deliberately *not* the cmake/bindgen multilib build used by
//! `bpg-x265-sys`: that builds the full encoder library (and its 10-bit
//! linked variant) for the public `x265_*` API. This crate only needs four
//! `common/*.cpp` translation units and stays a plain `cc` build so
//! `cargo build -p x265-primitives-sys` is fast to iterate on.
//!
//! `ENABLE_ASSEMBLY` is not defined (matching `bpg-x265-sys`'s default of
//! `OFF`): these are the scalar C reference implementations x265 uses on
//! platforms without a matching SIMD backend, which is the "oracle" Phase
//! 4 primitives are compared against (see FULL_RUST_CODEC_PLAN.md).
//! The full 8-bit wrapper is built with `HIGH_BIT_DEPTH=0`, `X265_DEPTH=8`,
//! `X265_NS=x265_8bit`. A narrower 10-bit link probe is built from the same
//! source files with `HIGH_BIT_DEPTH=1`, `X265_DEPTH=10`,
//! `X265_NS=x265_10bit`, following the same namespace split as x265's
//! multilib build.

use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_dir = Path::new(&manifest_dir);
    let x265_common = crate_dir.join("../../../x265_4.1/source/common");
    let x265_source = crate_dir.join("../../../x265_4.1/source");
    let shim = crate_dir.join("shim");

    // Namespaced as `x265_Nbit_prim` (not x265-sys's `x265_Nbit`/`x265`): a
    // binary linking both this crate and `bpg-x265-sys` would otherwise see
    // duplicate `x265_10bit::primitives` symbol definitions (this crate's
    // `wrapper10.cpp` and x265-sys's vendored `primitives.cpp` both define
    // `EncoderPrimitives X265_NS::primitives`).
    compile_primitives(
        &x265_common,
        &x265_source,
        &shim,
        "x265primitives",
        "0",
        "8",
        "x265_8bit_prim",
        "wrapper.cpp",
    );
    compile_primitives(
        &x265_common,
        &x265_source,
        &shim,
        "x265primitives10",
        "1",
        "10",
        "x265_10bit_prim",
        "wrapper10.cpp",
    );

    for f in [
        "constants.cpp",
        "dct.cpp",
        "pixel.cpp",
        "intrapred.cpp",
        "primitives.h",
        "common.h",
    ] {
        println!("cargo:rerun-if-changed={}", x265_common.join(f).display());
    }
    for f in [
        "entropy_tables.cpp",
        "wrapper.cpp",
        "wrapper10.cpp",
        "wrapper_impl.h",
        "slicetype.h",
        "x265_config.h",
    ] {
        println!("cargo:rerun-if-changed={}", shim.join(f).display());
    }
}

fn compile_primitives(
    x265_common: &Path,
    x265_source: &Path,
    shim: &Path,
    lib_name: &str,
    high_bit_depth: &str,
    depth: &str,
    namespace: &str,
    wrapper: &str,
) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .opt_level(2)
        .include(shim)
        .include(x265_common)
        .include(x265_source)
        .define("HIGH_BIT_DEPTH", high_bit_depth)
        .define("X265_DEPTH", depth)
        .define("X265_NS", namespace);

    if cfg!(target_arch = "x86_64") {
        build.define("X265_ARCH_X86", "1").define("X86_64", "1");
    } else if cfg!(target_arch = "x86") {
        build.define("X265_ARCH_X86", "1");
    }

    build
        .file(x265_common.join("constants.cpp"))
        .file(x265_common.join("dct.cpp"))
        .file(x265_common.join("pixel.cpp"))
        .file(x265_common.join("intrapred.cpp"))
        .file(shim.join("entropy_tables.cpp"))
        .file(shim.join(wrapper));

    build.compile(lib_name);
}
