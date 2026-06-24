# StillSearch v2 Overhaul Plan

## Summary

- **Plan-mode boundary:** no files change in this turn. First execution step writes this plan to root `plan.md`, then creates the StillSearch skeleton.
- Build **StillSearch** as the second-generation still265 intra search/split core under `crates/still265/src/encoder/stillsearch/`.
- Use a **hard compile cutover**: `rdo2`, `rdo_legacy`, and `snapshot` stay deleted; remove stale imports/references instead of restoring files from git history.
- StillSearch is **plan-first and arena-backed**: trials pass small IDs/descriptors; coeffs, recon patches, and child plans live in CTU-local arenas; writer-facing syntax trees are built only for the final winner.
- Current crate state: `cargo check -p still265` fails because `encoder/mod.rs` and `encoder/write.rs` still reference deleted `rdo2`/`rdo_legacy`/`snapshot` modules and removed `build_cu`/snapshot methods.

## Architecture Guardrails

### StillSearch core invariant

StillSearch is plan-first and arena-backed.

During search:
- candidate decisions return `Decision { cost, plan_id, recon_id, confidence }`
- coeffs, recon patches, and child plans live in `CtuWorkspace` arenas
- loser candidates are discarded by dropping IDs / resetting arena marks
- writer-facing `CuNode`/`Tt`/`CodedBlock` structures are created only for the final winner
- the shared reconstructed frame is mutated only by final commit

No hot trial path may:
- snapshot and restore the frame
- build recursive `Vec`-backed `CuNode`/`Tt` trees for losers
- re-run exact RDOQ in final commit when an exact retained winner is still valid

Trial decisions pass around small IDs and `Copy`-sized descriptors. Large data lives in CTU-local arenas. Hot search code must not clone `Vec`-backed trees or coefficient arrays.

### Boundary rules

- Do not restore `encoder/rdo2/*`, `encoder/rdo_legacy.rs`, or `encoder/snapshot.rs`.
- Do not use frame snapshot/restore as the normal trial mechanism.
- Reuse only lower layers: `cabac`, `contexts`, `transform`, `rdoq`, `residual`, `primitives`, `aq`, and writer-facing syntax structs.
- **No-runtime-`todo!` rule:** the skeleton phase may use `todo!()` only in functions not reachable from public encode APIs. Any public encode path must either (1) return a clear `Unsupported`/`NotImplemented` error before writing output, or (2) produce a deliberately minimal valid all-DC/all-leaf encode. No runtime `todo!()`, `unreachable!()`, or `panic!()` may remain in the public encode path after the skeleton commit.

## Key Changes

### 1. Root plan and reclassified syntax types

- Create root `plan.md` with this plan. Treat old `docs/plan.md` as obsolete rdo2 history; do not extend it.
- Reclassify writer-facing types: if `CuNode`/`Tt`/`CodedBlock` are still required by `write.rs`, treat them as **final syntax-tree types**, not search types. Move/rename them to `encoder/syntax.rs` if practical. They may be constructed only after StillSearch chooses a final plan and must never represent loser candidates or trial state.

### 2. StillSearch skeleton (lean file set first)

Create initially:

```text
crates/still265/src/encoder/stillsearch/
  mod.rs        // docs + guardrails + reexports
  api.rs        // StillSearch, CTU entrypoint, config/view types
  geom.rs       // CuGeom/TuGeom/PuGeom + chroma geometry
  cost.rs       // RdCost, Decision, Confidence, BlockEval
  workspace.rs  // CtuWorkspace, fixed scratch, candidate buffers
  arena.rs      // typed IDs + arenas (plans, coeffs, recon overlays)
  overlay.rs    // read-through recon overlay model
  emit.rs       // plan-to-writer bridge into final syntax structs
  ledger.rs     // work buckets/counters
```

Defer until real code exists (avoid empty-placeholder architecture):
- `rough.rs`, `eval/` (`mod.rs` + `block.rs`), `tu.rs`, `cu.rs`, `nxn.rs`, `chroma.rs`, `validate.rs` are added when their phase begins, not upfront.

### 3. Overlay model (specific)

- Prediction reads in order: **local candidate overlay → committed frame → unavailable-edge fallback**.
- For split/NxN trials, prior sibling/PU recon must be visible to later siblings/PUs in legal z-order **within the same candidate branch**, sourced from the candidate overlay, never from mutating the global frame.
- Only the winning branch's overlay commits to the shared frame, once.

### 4. Hard compile cutover

- In `encoder/mod.rs`, replace `mod rdo2; mod rdo_legacy; mod snapshot;` with `mod stillsearch;` (and `mod syntax;` if types are moved).
- **Delete, don't preserve, dead knobs:** any field, env gate, profiler label, or `EncodeStats` member that existed only to support rdo2/snapshot/legacy search is removed or moved behind a historical doc. Keep public `EncodeStats` stable only where externally useful (e.g. consumed by `bpg-tools`).
- Replace old `build_cu` call sites with the StillSearch CTU entrypoint.
- Force the initial path to **serial CTU build only**; stub/disable parallel snapshot-merge until overlay commit/worker merge exists.
- Keep public API stable: `crate::encoder::{encode, encode_with_stats, Source, EncodeStats}` and `RustStillHevcEncoder` behavior/config surface.

### 5. Implementation phases

1. **Skeleton compile** — add lean files; wire `encoder` to compile without rdo2/snapshot. `cargo check -p still265` passes. Public encode path returns a clear `NotImplemented` error if no encode is wired yet (no runtime `todo!`).
2. **Syntax bridge milestone** — implement the narrowest bridge first: `StillSearchPlan::LeafAllDc → CuNode/Tt/CodedBlock → write.rs`, proving the final syntax boundary before search complexity.
3. **First valid encode target** — 8-bit, 4:2:0, one CTU or tiny image, all CUs leaf, fixed luma mode (DC or planar), fixed chroma DM/DC, fixed TU size, no PartNxN, no split, deblock/SAO unchanged or disabled per existing config. Commit final recon once.
4. **Local TU/RQT** — full-vs-split TU with local scratch, null-CBF comparison, exact TU diagnostics, retain only winning coeff/recon.
5. **Luma decision** — x265-shaped rough search (planar/DC/angular, MPM-protected shortlist); staged cheap → exact close-call → final; reuse exact winner when context still valid.
6. **CU recursion** — `Leaf2Nx2N`, legal `PartNxN`, `Split` as arena-backed decisions; split branch-and-bound vs best leaf/NxN; overlay-commit winner once.
7. **PartNxN + chroma** — `NxnRoughSet`, `eval_nxn4_batch`, exact PU winner carry-forward; 4:2:0 chroma first, then 4:2:2/4:4:4.
8. **Parity + speed** — fixed-block diff tooling and C-vs-Rust decoded sweep first; only then horizontal angular SIMD, DST4 SIMD, residual-pricing cleanup, and later u8-native 8-bit path.

## Test Plan

- Phase 1: `cargo check -p still265`; `git diff --check`; confirm no `mod rdo2/rdo_legacy/snapshot` and no runtime `todo!`/`panic!` in the public encode path.
- Phase 2–3: `cargo test -p still265 --test transform_recon`, `--test residual_roundtrip`, `--test encode_roundtrip -- --nocapture`; tiny-image encode/decode smoke once CTU output exists.
- Parity milestones: fixed-block diff for 4×4/8×8/16×16/32×32 luma TUs; decoded Rust vs C/BPG-C bytes, PSNR-Y, RGB PSNR, SSIM/MS-SSIM; log TU size, mode, QP, CBF, nnz, abs-level sum, last position, residual energy, bits.
- Performance milestones: StillSearch ledger must show bucket movement before any speed claim. Buckets: rough luma, luma cheap, luma exact, TU leaf, TU split, NxN rough, NxN batch, chroma rough, chroma trial, RDOQ, residual price, final commit, writer, deblock, SAO.

## Rejection Criteria

- Restoring deleted files, importing from old paths, or copying old rdo2/snapshot logic from git history. (Docs and x265 source may be consulted; removed Rust search code may not be resurrected.)
- Leaving runtime `todo!()`/`unreachable!()`/`panic!()` in the public encode path after the skeleton commit.
- Frame snapshot/restore in a hot trial path.
- Materializing loser `CuNode`/`Tt` trees, or cloning `Vec`-backed coeff/recon arrays in hot search.
- Re-running exact RDOQ in final commit when a valid exact winner was retained.
- Preserving dead rdo2/snapshot/legacy env gates or stats solely to ease compilation.

## Assumptions and Defaults

- Module location: `crates/still265/src/encoder/stillsearch/`; cutover is hard, no dual rdo2 path.
- `plan.md` means root `/mnt/Samsung980_1TB/Rust-projects/bpg-rs/plan.md`.
- Initial writer stays `encoder/write.rs`; StillSearch converts final plans to current syntax structs.
- Parallel analysis is restored only after overlays can merge CTU results without snapshot/restore.
- Skeleton is compile-first, but no runtime `todo!` in public encode paths; valid encoding returns from the syntax-bridge milestone onward.
