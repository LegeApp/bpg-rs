# StillSearch Implementation Note

## Purpose

`StillSearch` is the new still-image intra search engine for `still265`.

This is no longer an attempt to port x265’s search/split architecture file-for-file. The old `rdo2` folder and `snapshot.rs` have been intentionally removed from the project. They must not be restored from git history, copied back, or used as a hidden reference implementation.

The task is to build a fresh still-image-only HEVC intra search architecture that keeps the valid algorithmic lessons from x265 and prior still265 work, but does not recreate the old Rust encoder architecture.

## Hard boundary

Do not resurrect:

```text
encoder/rdo2/*
encoder/rdo_legacy.rs
encoder/snapshot.rs
old build_cu / build_tt / build_cu_leaf_nxn logic
frame snapshot / restore as the normal trial mechanism
```

If something from old code seems useful, re-express the idea in the new architecture. Do not copy the structure.

Allowed reuse:

```text
cabac.rs
contexts.rs
transform.rs
rdoq.rs
residual.rs
primitives/*
writer-facing syntax structs if still present
write.rs as the initial bitstream boundary
decoder-derived validation tools
```

The old encoder was removed because the search core had become architecture-limited. The new work should not be a transplant.

## What StillSearch is

`StillSearch` should be a CTU-local decision graph engine.

Its job is to decide:

```text
CU split structure
2Nx2N vs NxN where legal
luma intra mode
chroma intra mode
TU split structure
coefficient levels / CBFs
local reconstructed samples for prediction
final writer-facing syntax plan
```

Its core design should be:

```text
CTU workspace
  source views
  recon overlays
  fixed candidate arrays
  coefficient arena
  plan arena
  per-size scratch blocks
  local entropy/context snapshots
  final commit path
```

Trials must write to local scratch or recon overlays. Loser trials must not mutate the shared reconstructed frame. The final winner commits once.

## What StillSearch is not

StillSearch is not:

```text
a ModeDepth/RQT clone
a port of x265 analysis.cpp/search.cpp
a continuation of rdo2
a collection of env-gated heuristics
a recursive tree builder that materializes CuNode/Tt for every candidate
a snapshot/restore engine
```

x265 remains a behavioral and algorithmic reference, especially for rough intra mode search, RQT legality, CABAC syntax costs, RDOQ behavior, and primitive expectations. But x265’s C object layout is not the design target.

## Main invariant

Approximate work may reject obvious losers. Approximate work may not decide close cases.

The intended decision shape is:

```text
rough screen
cheap trial
exact close-call trial
single final commit
```

But this must be implemented with new StillSearch structures, not the old rdo2 `EvalPolicy` pipeline.

Prefer explicit stage functions:

```rust
eval_luma_cheap(...)
eval_luma_exact(...)
eval_luma_final(...)

eval_chroma_cheap(...)
eval_chroma_exact(...)
eval_chroma_final(...)

eval_nxn4_batch(...)
```

Avoid a single branch-heavy generic evaluator in the hot path.

## Core data model

Use bounded, CTU-local storage. Avoid heap allocation during candidate trials.

Suggested skeleton:

```rust
pub struct StillSearch<'a> {
    params: &'a EncoderParams,
    source: SourceView<'a>,
    recon: ReconFrame,
    contexts: ContextState,
    workspace: CtuWorkspace,
    stats: SearchStats,
}

pub struct CtuWorkspace {
    source_cache: CtuSourceCache,
    recon_pool: ReconOverlayPool,
    scratch: BlockScratchSet,
    candidates: CandidateArena,
    coeffs: CoeffArena,
    plans: PlanArena,
}
```

Trial results should be small handles:

```rust
pub struct Decision {
    pub cost: RdCost,
    pub plan: PlanId,
    pub recon: ReconId,
    pub confidence: Confidence,
}
```

Search plans should be arena-backed:

```rust
pub enum CuPlan {
    Leaf2Nx2N {
        luma_mode: u8,
        chroma_mode: u8,
        tt: TtPlanId,
    },
    LeafNxN {
        pus: [PuPlan; 4],
        parent_chroma: Option<ChromaPlanId>,
    },
    Split {
        children: [Option<CuPlanId>; 4],
    },
}

pub enum TtPlan {
    Leaf {
        geom: TuGeom,
        luma: Option<BlockCoeffId>,
        cb: Option<BlockCoeffId>,
        cr: Option<BlockCoeffId>,
        cost: RdCost,
    },
    Split {
        children: [TtPlanId; 4],
        parent_chroma: Option<ChromaPlanId>,
        cost: RdCost,
    },
}
```

Only after a final winner is selected should plans be converted to writer-facing syntax objects, unless the writer is later refactored to consume plans directly.

## Recon model

Do not use frame snapshot/restore for normal trials.

Use a read-through recon model:

```text
prediction reads:
  1. local overlay samples
  2. committed reconstructed frame
  3. unavailable/edge fallback
```

A trial writes reconstructed samples into an overlay. A split child can read prior sibling overlays where legal. The selected branch’s overlay is committed once.

This is the central replacement for the deleted `snapshot.rs`.

## CU search approach

Implement CU search as candidate comparison, not recursive object construction.

High-level flow:

```rust
fn decide_cu(search: &mut StillSearch, geom: CuGeom, ctx: CtxId) -> Decision {
    let leaf = maybe_eval_leaf_2nx2n(search, geom, ctx);
    let nxn = maybe_eval_nxn(search, geom, ctx);
    let split = maybe_eval_split(search, geom, ctx, leaf.as_ref().map(|d| d.cost));

    choose_best(leaf, nxn, split)
}
```

Split evaluation should support branch-and-bound:

```text
If accumulated child cost already cannot beat the best available leaf/NxN cost,
abort the remaining split children.
```

But do not start by adding many new force-leaf or force-split heuristics. The previous project history already mined most of those.

## TU search approach

TU search is one of the main reasons StillSearch exists. It should be local, arena-backed, and measurable.

Implement:

```text
full TU leaf trial
split TU trial
null-CBF comparison
exact syntax cost
local recon output
plan result
```

The decision-diff notes showed that prior still265 behavior over-split smooth/mid regions and underperformed x265 in texture coefficient allocation. Therefore, TU search must expose detailed diagnostics:

```text
x, y, log2_tu
region class
leaf distortion / bits / cost
split distortion / bits / cost
winner
CBF
coefficient strength
recon residual energy
```

A future large-TU bias may be useful, but it should be implemented as a named policy with diagnostics, not as an anonymous heuristic.

## Intra mode search

Keep the x265-shaped rough-search idea, but not x265’s object layout.

For Best-equivalent search:

```text
score planar, DC, and angular modes
protect MPMs
keep x265-like shortlist
cheap-trial candidates
exact-recheck close candidates
final-code winner once
```

The rough search should be written around batchable prediction and SATD, not per-mode heap or tree structures.

Do not make angular exclusion part of the default architecture. Intra mode selection was not the main quality gap. Angular exclusion is a speed/quality tradeoff and should remain a separate policy until proven.

## PartNxN

Do not begin by pruning PartNxN. Prior attempts showed that it is often a close competitor.

StillSearch should make PartNxN cheaper:

```text
compute NxnRoughSet once per 8x8 CU
reuse rough 4x4 mode results
batch four 4x4 PU evaluations
carry exact winning PU coefficients forward
commit PUs sequentially only on the winning path
```

Suggested structure:

```rust
pub struct NxnRoughSet {
    pub parent_8x8: RoughModeSet,
    pub sub_4x4: [RoughModeSet; 4],
    pub family_diversity: u8,
    pub rough_gain_q8: i32,
}
```

## RDOQ and coefficient allocation

Do not assume the quality gap is solved by a cleaner search architecture.

StillSearch must make coefficient behavior easy to inspect. For each final block, log:

```text
TU size
mode
QP
region class
nnz
abs level sum
average nonzero level
last position
CBF
residual energy after recon
frac bits
```

The goal is to compare against decoded x265/BPG-C decisions and answer whether still265 is still under-coding textured coefficients or wasting bits in smooth/mid regions.

## Chroma

Chroma should be integrated cleanly but should not dominate the initial design.

Initial priority:

```text
correct chroma geometry
correct chroma mode syntax
correct Cb/Cr coefficient handling
4:2:0 first
then 4:2:2 and 4:4:4
```

Do not build chroma through copied legacy luma paths. It should use the same plan/eval/final model.

## Writer boundary

Initially, keep the existing writer boundary if possible.

StillSearch should produce a final plan, then convert that plan into whatever final syntax structs the writer currently expects.

Do not rewrite `write.rs` and StillSearch at the same time. Once StillSearch is stable, the writer can later be taught to consume plans directly.

## Instrumentation requirement

StillSearch must include a first-class work ledger from the beginning.

Each bucket should track:

```text
calls
wall_ns
component
log2 size
prediction calls
forward transforms
quant calls
RDOQ calls
inverse transforms
exact residual price calls
approx residual price calls
coeff bytes retained
recon overlay writes
final commits
allocations if practical
```

Minimum buckets:

```text
RoughLuma
LumaCheap
LumaExact
TuLeaf
TuSplit
NxnRough
NxnBatch
ChromaRough
ChromaTrial
Rdoq
ResidualPrice
FinalCommit
Writer
Deblock
Sao
```

No speed claim should be accepted without bucket movement.

## Environment gates

StillSearch-specific environment gates must be declared and parsed in:

```text
crates/still265/src/encoder/stillsearch/env.rs
```

Do not add ad-hoc `std::env::var*` checks in search modules. The current
StillSearch gates are:

```text
BPG_STILLSEARCH_PROFILE=1
  Populate per-bucket `stillsearch_ledger_ns` wall-clock timings.

BPG_STILLSEARCH_LUMA_CHEAP=0
  Disable the x265-style luma-only cheap pass and restore full recursive TU
  search for every rough-shortlisted luma mode.

BPG_STILLSEARCH_LUMA_CHEAP_EXACT_TOP=<usize>  [default: 1]
  Number of cheap-ranked luma candidates to remeasure with full TU search.
  1 is fastest (default). 2 remeasures the top-2 for marginal quality insurance.
  3 matches old exhaustive behavior. Sweep showed no quality loss at 1 vs 2 or 3.

BPG_STILLSEARCH_ROUGH_RD_CANDS=<usize>  [default: 2]
  Max rough-pass candidates carried into the cheap luma pass (before MPM union).
  2 is default (capped at 4 = ROUGH_RD_CANDS). Sweep showed no quality loss at
  2 vs 3 or 4 on test images; saving ~1.5s at 12MP.

BPG_STILLSEARCH_NXN_SKIP_SATD=<float>  [default: 1000]
  Skip PartNxN for 8×8 CUs whose rough SATD is below this threshold.
  1000 = default: smooth blocks skip NxN (PSNR unchanged or slightly better;
  saves ~3s at 12MP). NxN over-selects smooth blocks due to frozen-context bias.
  Set to 0 to disable. 2000 saves an additional ~1s (+2.5% bytes, PSNR improves).

BPG_STILLSEARCH_LUMA_CHEAP_RESIDUAL_PRICE=skip
  Diagnostic only. Rank the luma-cheap first pass without exact residual syntax
  pricing. The full winner pass remains exact. Defaults to exact because smoke
  testing showed good speed but worse rate.

BPG_STILLSEARCH_ANGULAR_EXCLUSION=game|iame|tsame
  Env-gated diagnostic angular-mode prefilter for rough luma search.
  Defaults to off. `BPG_BEST_ANGULAR_EXCLUSION` is accepted as a legacy alias.

BPG_ANGULAR_GAME_VAR=<float>
BPG_ANGULAR_IAME_FACTOR=<float>
BPG_ANGULAR_MIN_KEEP=<usize>
BPG_ANGULAR_MIN_LOG2=<u8>
  Parameters for the angular-exclusion diagnostic.
```

### Default speed point (Best effort, QP=28, 12MP)

As of 2026-06-24, the default Best effort averages ~15.5s on three 12MP test images
(4000×3000 crop). Quality vs C bpgenc -m8: −0.73 dB PSNR_y, −0.4% bytes.

Key call counts: LumaCheap ~3.9× cu_trials, LumaExact = 1× cu_trials, NxnRough
~0.94× cu_trials (skip=1000 fires on ~62% of 8×8 CUs).

For maximum speed (~14-15s average): set NXN_SKIP_SATD=2000 (saves 1s more,
slight byte increase but PSNR improves due to NxN bias correction).

## Implementation order

Recommended sequence:

```text
1. Create StillSearch module skeleton and compile without rdo2/snapshot.
2. Add CTU workspace, arenas, geometry types, and work ledger.
3. Implement fixed-mode fixed-TU luma final coding.
4. Add plan-to-writer bridge for one simple leaf CU.
5. Implement local TU full-vs-split with recon overlays.
6. Implement rough luma mode search.
7. Implement luma cheap/exact/final candidate flow.
8. Implement CU leaf-vs-split using local decisions, not frame rollback.
9. Add PartNxN through NxnRoughSet and batched 4x4 evaluation.
10. Add chroma mode and chroma residual path.
11. Add coefficient allocation diagnostics.
12. Run C-vs-Rust decoded quality sweep.
13. Only then optimize with SIMD, u8-native storage, and stronger policies.
```

## Rejection criteria

Reject any patch that:

```text
restores rdo2 or snapshot.rs
uses git history to copy old search code back
adds a new env gate instead of a named policy
allocates Vec children during hot trial search
materializes final CuNode/Tt trees for loser candidates
uses shared-frame mutation plus rollback for normal trials
claims parity without decoded C-side PSNR-Y comparison
improves one image but regresses the multi-image QP sweep
```

## Success criteria

The first successful StillSearch milestone is not “faster than x265.” The first milestone is:

```text
valid BPG output
no old rdo2/snapshot dependency
local recon overlays working
simple CU/TU plans emitted correctly
work ledger active
decoded output validated
```

The second milestone is:

```text
Best-equivalent search quality close to or better than old Best
fewer duplicated RDOQ/final replay operations
clearer TU/coefficient diagnostics
```

The third milestone is:

```text
speed improvements from architecture, not from unsafe pruning
C-vs-Rust quality gap reduced or precisely localized
path open for u8-native storage and SIMD kernels
```

## Summary

StillSearch should be a new architecture, not a rescue of the old one.

Use x265 as a reference for what decisions must be available and how HEVC intra search behaves. Do not use x265 as a mandate to copy `ModeDepth`/`RQT` object structure. The clean Rust design should be CTU-local, arena-backed, overlay-based, plan-first, and explicitly staged from rough search through exact final commit.

The old files were removed to force this boundary. Preserve that boundary.
