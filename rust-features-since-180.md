A few are genuinely relevant, but first: on the official Rust release announcements page, the newest stable release I can verify is **1.95.0** from April 16, 2026. I do not see 1.96.0 listed there yet, so that may be beta/nightly or just not stable at the moment. ([Rust Blog][1])

For a Rust JPEG 2000 encoder, the most useful additions since 1.80 are these:

**1.86: `slice::get_disjoint_mut` and `HashMap::get_disjoint_mut`**
This is one of the most directly useful ones for codec code. JPEG 2000 has lots of places where you want simultaneous mutable access to disjoint regions: subbands, temporary row/column buffers, code-block state arrays, packet bookkeeping, or multiple tiles/components. Rust 1.86 added helpers for getting multiple mutable references safely in one shot, which can simplify buffer choreography and remove awkward split logic or unsafe indexing. ([Rust Blog][2])

**1.84: strict provenance APIs**
If your encoder uses low-level memory tricks, pointer tagging, custom allocators, SIMD-adjacent code, or manually managed scratch arenas, this matters. Rust 1.84 added strict provenance APIs specifically to avoid ambiguous pointer–integer casts and make low-level code easier to reason about and analyze. For codec internals, that can help when replacing old C-style pointer arithmetic with sounder Rust patterns. ([Rust Blog][3])

**1.86: safe functions may use `#[target_feature]`**
That is relevant if you want CPU-specialized hot paths for DWT, color transforms, quantization, or scan/bit-model logic without forcing those entry points themselves to be unsafe. It makes it cleaner to expose optimized implementations behind dispatch while keeping the public surface more idiomatic. ([Rust Blog][2])

**1.94: `array_windows`**
This is small but useful for signal-processing code. JPEG 2000 code often walks fixed-size neighborhoods or short filter windows. `array_windows` gives fixed-size slice windows as `&[T; N]`, which is nicer for wavelet lifting steps, neighborhood/context inspection, and compact inner-loop code than dynamically sized windows. ([Rust Blog][4])

**1.95: `cfg_select!`**
This is handy if you want one crate to choose between portable, SIMD, platform-specific, or feature-specific implementations at compile time. For example, different DWT kernels, endian-sensitive paths, or architecture-tuned entropy-coding helpers. It overlaps with `cfg-if`, but now it is in the standard toolchain. ([Rust Blog][5])

**1.95: `Vec::push_mut` / `insert_mut` and similar APIs**
These can be nice in codec builders and packet/code-block assembly code, where you push a new structure and immediately mutate it in place. It is not revolutionary, but it reduces a lot of “push then index last element” friction in state-heavy encoders. ([Rust Blog][5])

**1.83: expanded const capabilities**
Rust 1.83 expanded what can run in const contexts, including mutable references and interior mutability during constant evaluation. That can help if you want more tables, context-transition data, lifting constants, or compile-time-generated lookup structures without build scripts. For JPEG 2000, that is most relevant to MQ-coder tables, context tables, and fixed transform metadata. ([Rust Blog][6])

**1.81: improved standard sort implementations**
Not a headline feature, but potentially useful if parts of your encoder do sorting for rate allocation, layer truncation candidates, packet ordering preparation, or heuristic planning. Rust 1.81 updated both stable and unstable sort implementations for better runtime performance and compile time. ([Rust Blog][7])

**1.84 and 1.85: Cargo MSRV-aware resolver and Rust 2024 edition**
These matter more for project maintenance than raw encoder speed. The MSRV-aware resolver in 1.84 makes it easier to support an older minimum toolchain without dependency pain, and Rust 2024 in 1.85 gives you stricter unsafe boundaries and cleaner modernization pressure. For a codec crate with performance-sensitive unsafe sections, the 2024 edition’s stronger unsafe hygiene is a real quality-of-implementation win. ([Rust Blog][3])

If I were ranking them for a JPEG 2000 encoder specifically, I would put them in this rough order:

1. `get_disjoint_mut`
2. strict provenance APIs
3. safe `#[target_feature]`
4. `array_windows`
5. `cfg_select!`
6. expanded const evaluation
7. `push_mut`-style collection APIs

The biggest practical gains are likely to be in **safer low-level memory work**, **cleaner SIMD/path specialization**, and **less borrow-checker friction in buffer-heavy code**, not in any one giant language feature.

[1]: https://blog.rust-lang.org/releases/?utm_source=chatgpt.com "The Rust Release Announcements"
[2]: https://blog.rust-lang.org/2025/04/03/Rust-1.86.0/?utm_source=chatgpt.com "Announcing Rust 1.86.0 | Rust Blog"
[3]: https://blog.rust-lang.org/2025/01/09/Rust-1.84.0/?utm_source=chatgpt.com "Announcing Rust 1.84.0 | Rust Blog"
[4]: https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/?utm_source=chatgpt.com "Announcing Rust 1.94.0 | Rust Blog"
[5]: https://blog.rust-lang.org/2026/04/16/Rust-1.95.0/?utm_source=chatgpt.com "Announcing Rust 1.95.0 | Rust Blog"
[6]: https://blog.rust-lang.org/2024/11/28/Rust-1.83.0/?utm_source=chatgpt.com "Announcing Rust 1.83.0 | Rust Blog"
[7]: https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/?utm_source=chatgpt.com "Announcing Rust 1.81.0 | Rust Blog"
