I read `the-state-of-things.md` as canonical and checked the current `crates/still265/src/encoder` layout plus the x265 4.1 `encoder/analysis.cpp`, `encoder/search.cpp`, and `encoder/search.h` reference. My read is:

The project does **not** need another incremental Best/Slow scheduler pass. It needs a **second-generation still265 execution core** that keeps the algorithmic lessons from rdo2, but discards the current encoder folder’s object/snapshot/search architecture.

## Executive plan

Do **not** rewrite the whole workspace. Rewrite the encoder execution core.

Keep these, but audit them:

| Keep                          | Reason                                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `cabac.rs`, `contexts.rs`     | Needed for entropy estimation/writing. The API may need wrappers, but not a rewrite first.                                                 |
| `rdoq.rs`                     | Keep as baseline, but build a parity harness against x265 RDOQ behavior.                                                                   |
| `transform.rs`, `primitives/` | Keep only after fixed-block parity testing. Forward DCT was already a real source of loss, so this layer must remain suspect until proven. |
| `residual.rs`                 | Keep the syntax writer/pricer logic, but split final syntax writing from trial pricing.                                                    |
| `encoder/write.rs`            | Keep initially as the final bitstream boundary. Do not rewrite writer and search at the same time.                                         |
| decoder crates                | Keep. The decoder-derived decision diff is one of the best tools you have.                                                                 |

Delete or quarantine these from the new implementation path:

| Replace                                | Reason                                                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `encoder/rdo2/*`                       | Correct direction, but too layered around snapshots, `CuNode`/`Tt` materialization, and generic block eval. |
| `encoder/rdo_legacy.rs`                | Should not remain reachable. It will contaminate the rewrite.                                               |
| `encoder/snapshot.rs`                  | Should become diagnostic fallback only. Normal trials should not roll back the shared frame.                |
| `encoder/types.rs` hot-path tree types | Fine as writer-facing final syntax objects, bad as trial/search objects.                                    |
| most env-gated experimental branches   | They encode stale investigations and invite agents to resurrect rejected paths.                             |

The new core should be named something like `encoder2` or `search_core`, but I would avoid a vague `rdo3`. This is bigger than RDO. It is an x265-style **ModeDepth/RQT execution engine** for still images.

---

# 1. First rule: separate quality parity from speed parity

The markdown and related docs contain two different issues:

1. **Speed:** Rust Best is still about 3× slower than x265-like C at the parity target.
2. **Quality:** Rust is still lower luma quality at comparable bitrate. Earlier notes point to coefficient allocation, TU oversplitting in smooth/mid regions, texture coefficient strength, loop filters, and prior transform mismatch.

Do not start by optimizing the new search core. Start by proving that a fixed block coded with fixed decisions produces x265-equivalent reconstruction behavior.

## Phase 0 — x265 parity lab

Create a dedicated tool, not an env gate:

```text
crates/bpg-tools/src/bin/x265_stage_diff.rs
```

It should compare still265 and x265/HM-like behavior for one fixed block:

```text
input block
fixed intra mode
fixed reference samples
fixed QP
fixed TU size
fixed chroma format
```

Dump and compare:

```text
prediction samples
residual
forward coefficients
initial quant levels
RDOQ levels
last position
significant-coeff groups
dequant coefficients
inverse residual
recon samples
CBF / coeff bits estimate
```

The first target is luma 8-bit 4:2:0, fixed 4×4/8×8/16×16/32×32 TUs.

**Stop condition:** for fixed mode+TU+QP, still265’s forward/quant/RDOQ/recon path is either bit-identical to the chosen x265 reference or the remaining difference is explicitly understood and documented.

Do this before the architecture rewrite, otherwise the new engine may faithfully reproduce the current quality deficit.

---

# 2. Rewrite strategy that prevents agents from copying old code

Your concern is valid. If the old encoder remains in-tree and compiling, agents will reuse it.

Use a hard quarantine.

## Commit 1: move old encoder out of the build

Recommended layout:

```text
crates/still265/src/
  encoder2/
    mod.rs
    core.rs
    ctu.rs
    cu.rs
    tu.rs
    mode.rs
    chroma.rs
    part_nxn.rs
    rdoq_bridge.rs
    recon.rs
    workspace.rs
    final_emit.rs
    stats.rs
  encoder_old_do_not_use/
    ...
```

Then add a build-facing rule:

```rust
// lib.rs
mod encoder2;
// no `mod encoder_old_do_not_use`
```

Better: move the old folder outside `src` entirely:

```text
docs/archived_encoder_2026_06_23/
```

Also add a short `ARCHITECTURE_RULES.md`:

```text
The new encoder must not call or copy:
- encoder/rdo_legacy.rs
- encoder/rdo2/*
- encoder/snapshot.rs
- old build_cu / build_tt / build_cu_leaf_nxn code

Allowed reuse:
- cabac/context/rdoq/transform/primitives/residual syntax helpers
- writer-facing final structs only after final decision is made
```

For agent workflows, this matters more than prose. Make old paths **not compile**.

---

# 3. New architecture: x265-shaped still-image core

The current encoder’s hot path is centered on:

```text
predict → residual → forward transform → quant/RDOQ → inverse/recon → residual price
```

That part is unavoidable. The architectural problem is that this work is invoked through layered candidate structures, frame snapshots, recursive materialized trees, and winner replay. x265 does roughly similar decisions, but with per-depth scratch objects, mode slots, local recon, and primitive tables.

The new core should be built around these objects.

## Core object model

```rust
struct Encoder2 {
    params: EncParams,
    source: SourcePlanes,
    recon: ReconFrame,
    maps: SyntaxMaps,
    cabac: CabacState,
    primitives: PrimitiveSet,
    ctu_ws: CtuWorkspace,
    stats: EncStats,
}
```

```rust
struct CtuWorkspace {
    depths: [DepthWorkspace; 4],       // 64, 32, 16, 8
    rqt: [RqtWorkspace; 4],            // TU depths / sizes
    candidates: CandidateWorkspace,
    coeff_arena: CoeffArena,
    plan_arena: PlanArena,
    recon_scratch: ReconScratchPool,
}
```

```rust
struct DepthWorkspace {
    modes: ModeSlots,
    fenc: LocalSourceBlock,
    pred: LocalPredBlock,
    best: Option<ModeId>,
    split: SplitSlot,
}
```

```rust
struct RqtWorkspace {
    pred: PlaneBlock,
    residual: I16Block,
    coeffs: I16Block,
    levels: I16Block,
    recon_residual: I16Block,
    recon: PlaneBlock,
}
```

The important difference: **trial branches write into local workspaces, not the shared frame.** Only the final winning path commits to `Encoder2.recon`.

---

# 4. Replace recursive materialized trees with plans

The current code uses `CuNode`, `Tt`, `LeafTu`, `CodedBlock { levels: Vec<i16> }` too early. In the new core, those should be final syntax objects only.

During search, use plans:

```rust
struct CuDecision {
    cost: RdCost,
    kind: CuKind,
    recon_id: ReconId,
    syntax_id: SyntaxId,
}

enum CuKind {
    Leaf2Nx2N { luma_mode: u8, chroma_mode: u8, tt: TtPlanId },
    LeafNxN { pus: [PuPlan; 4], parent_chroma: ChromaPlanId },
    Split { children: [Option<CuPlanId>; 4] },
}
```

```rust
enum TtPlan {
    Leaf {
        log2_size: u8,
        luma: BlockCoeffId,
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

Only after the CU winner is known:

```text
CuPlan/TtPlan → writer-facing CuNode/Tt, or directly emit syntax later
```

Initial version can still convert plans into the existing writer structs. Do **not** rewrite the bitstream writer at the same time.

---

# 5. Evaluation kernel v2

The current `rdo2_eval_leaf_block` is the right conceptual primitive, but it is too generic and too allocation-/policy-heavy for the long-term hot path.

Replace it with specialized entry points:

```rust
eval_luma_trial_cheap()
eval_luma_trial_exact()
eval_luma_final()
eval_chroma_trial_cheap()
eval_chroma_trial_exact()
eval_chroma_final()
eval_nxn4_batch()
```

The generic `EvalPolicy` abstraction was useful during rdo2 migration. For v2, it should resolve before the hot call. Avoid a branchy “one function does everything” design.

Each eval function should return:

```rust
struct BlockEval {
    distortion: u64,
    frac_bits: u64,
    cost: f64,
    cbf: bool,
    coeff_id: Option<BlockCoeffId>,
    recon_id: Option<BlockReconId>,
}
```

Rules:

```text
Cheap trial:
  plain quant
  approximate or cached residual bits
  no retained coeffs unless explicitly requested

Exact trial:
  full RDOQ
  exact residual bits
  can retain coeffs if candidate may be final

Final:
  should not recompute if exact retained coeffs are valid
```

The goal is to stop paying:

```text
trial exact work → final exact replay
```

unless the prediction context changed.

---

# 6. CU search v2

Implement the search in this order:

```text
compress_ctu()
  compress_cu(depth=0, 64x64)
    maybe evaluate leaf
    maybe evaluate NxN at 8x8
    maybe evaluate split
    choose best
    commit winner to local/current recon
```

But the internal representation should mirror x265’s flow:

```text
ModeDepth[depth].pred[2Nx2N]
ModeDepth[depth].pred[NxN]
ModeDepth[depth].pred[SPLIT]
ModeDepth[depth].best
```

For still image intra only, this can be much smaller than x265’s full inter-capable `ModeDepth`.

## Leaf vs split rule

Keep the rdo2 principle:

```text
cheap screen may reject obvious losers
close cases get exact recheck
winner is coded once
```

But do not implement it through frame rollback. Implement it as:

```text
leaf trial writes recon A
split trial writes recon B
winner copies/commits A or B
```

Use split branch-and-bound, but make it cost-based and local:

```rust
if accumulated_child_cost > best_leaf_cost * bound_margin {
    abort remaining children
}
```

Do not add more CU force-leaf/force-split heuristics as the first pass. The docs already show that these are mostly mined out or dangerous.

---

# 7. TU search v2

This is one of the main overhaul targets.

The decision-diff docs indicate still265 over-splits smooth/mid TUs compared with x265, while texture quality is tied to coefficient strength and bit allocation. So TU v2 should not just reproduce current rdo2 behavior.

## Required TU features

```text
full-vs-split local RQT scratch
null-CBF cost comparison
larger-TU bias experiment in smooth/mid regions
exact logging of leaf/split distortion/bits/cost
ability to replay x265-like limitTU later
```

The new TU function should look more like:

```rust
fn estimate_residual_qt(
    mode: &mut ModeSlot,
    cu: CuGeom,
    tu: TuGeom,
    depth_range: TuDepthRange,
    ctx: CabacContextId,
) -> TtPlanId
```

It should evaluate:

```text
full leaf if legal
split if legal
compare with exact syntax costs
store only winning coeff/recon
```

Add a gated but first-class policy:

```text
large_tu_bias_non_textured
```

Meaning:

```text
For 16x16/32x32 TUs in Flat/Gradient/Noisy/mid regions,
split must beat leaf by a margin before winning.
Never apply to TextLike / DirectionalEdge / ChromaCritical.
```

This addresses the measured smooth/mid oversplitting directly. It is not a CU-level heuristic.

---

# 8. RDOQ and coefficient allocation

The quality gap cannot be solved by broader search alone. Reference/Placebo-style exhaustive search not closing the gap means the new core must treat RDOQ/coeff allocation as a first-class parity target.

## RDOQ tasks

1. Build fixed-block RDOQ diff against x265.
2. Log, per block:

   ```text
   nnz
   abs level sum
   last position
   coeff-group significance
   greater1/greater2 bins
   residual energy after recon
   bits
   ```
3. Compare by:

   ```text
   region class
   TU size
   QP
   mode family
   ```
4. Specifically test the texture finding:

   ```text
   at matched bytes, x265 keeps stronger 4x4/8x8 texture coefficients
   ```

Possible fixes after measurement:

```text
RDOQ rate-model parity
coefficient-group zeroing parity
last-position decision parity
sign-data-hiding on all relevant high-quality tiers
texture-aware coefficient lambda adjustment only if x265 parity still differs
```

Avoid guessing. RDOQ changes can look good on one QP and break BD-rate elsewhere.

---

# 9. PartNxN overhaul

Do not prune PartNxN first. The docs are clear: PartNxN is expensive because it is often competitive.

The v2 plan:

## 9.1 Shared rough cache

Compute once per 8×8 CU:

```rust
struct NxnRoughSet {
    parent_8x8: RoughModeSet,
    sub_4x4: [RoughModeSet; 4],
    family_diversity: u8,
    rough_gain_q8: i32,
}
```

Use it for both:

```text
should attempt NxN?
actual 4x4 PU candidate lists
```

No duplicate rough search.

## 9.2 Batch four 4×4 PUs

Create:

```rust
eval_nxn4_batch()
```

It should reuse:

```text
DST4 scratch
RDOQ scratch
mode-bit state
residual pricing scratch
prediction border extraction
```

The math still happens four times, but setup overhead should fall.

## 9.3 Carry exact winners forward

If a 4×4 PU exact trial wins and its neighbor context remains valid, final commit must reuse:

```text
mode
levels
cbf
frac_bits
recon residual
```

No second forward transform/RDOQ.

---

# 10. Rough intra mode search

x265’s rough search shape is already basically copied:

```text
all angular modes
25% padded best threshold
MPM0 protected
8–9 RD candidates at slow levels
```

So the issue is not candidate count. It is implementation throughput.

## Required work

1. Keep all-angle search for Best parity.
2. Implement horizontal all-angles SIMD or transpose reuse.
3. Use 8-bit SATD directly for 8-bit sources.
4. Keep mode-cost calculation branch-light and table-driven.
5. Add angular exclusion only as a non-default speed/quality tradeoff.

Do not promote angular exclusion as a parity feature. The decision diff says intra mode selection is already close; angular exclusion is speed-only and can hurt edge/text content.

---

# 11. 8-bit-native path

This should be part of the overhaul, but not the first patch.

The current code stores source/recon/pred as `u16` widely even for 8-bit input. That is bad for cache and SIMD packing. Partial 8-bit attempts regressed because they added conversion overhead without removing the real 16-bit path.

Implement this as a real split:

```rust
enum EncoderImpl {
    EightBit(Encoder8),
    HighBit(Encoder16),
}
```

Hot paths should be monomorphized or duplicated, not dynamically branched per pixel.

For `Encoder8`:

```text
source: u8
recon: u8
prediction: u8
SATD/SSD: u8 kernels
residual: i16
coeffs/levels: i16
transform: i16
```

For `Encoder16`:

```text
current 10/12-bit path remains
```

Do not force the 8-bit rewrite to support every chroma/depth edge case immediately. Start with:

```text
8-bit 4:2:0 luma path
then chroma
then 4:4:4/4:2:2
```

---

# 12. Loop filter and reconstruction parity

The quality docs name deblocking/SAO as possible remaining below-RDO differences. Treat them as measurable stages, not tuning knobs.

## Deblock

Add a decoder-derived comparison:

```text
for each boundary:
  rust boundary strength
  beta/tc
  filter applied?
  sample deltas
```

Compare against x265-decoded output where possible. If x265’s deblock decisions differ systematically, fix the signaling/decision path before touching search.

## SAO

SAO is currently off by default. x265/BPG-C behavior must be matched in the benchmark configuration. If C uses SAO and Rust does not, speed/quality comparisons are contaminated.

Decide one of these:

```text
A. benchmark both with SAO off
B. enable Rust SAO for parity
C. document that Rust is intentionally non-SAO and compare separately
```

Do not mix these in the same “x265 parity” claim.

---

# 13. Primitive backend plan

x265’s advantage is partly boring: hand-tuned primitives.

Priority order:

```text
1. DST4 forward
2. inverse DST4 if encoder recon path does not already hit optimized decoder path
3. horizontal all-angles intra prediction
4. residual + SSD u8 path
5. RDOQ inner-loop tables and branch reduction
6. residual pricing hot loop
7. optional x86 asm backend
```

Do not start with a broad “SIMD everything” project. Use the work ledger and implement by measured bucket share.

---

# 14. Instrumentation is mandatory

Before v2 is considered real, add a stable work ledger independent of env-gated experiments.

Each bucket should report:

```text
calls
wall_ns
log2 size histogram
component histogram
prediction calls
forward transforms
quant calls
RDOQ calls
inverse transforms
exact residual pricings
approx residual pricings
retained coefficient bytes
recon copies
allocations, if practical
```

Buckets:

```text
RoughLumaAllAngles
LumaCheapTrial
LumaExactTrial
TtLeaf
TtSplit
NxnRough
NxnExact
ChromaRough
ChromaTrial
FinalCommit
Rdoq
ResidualPrice
Deblock
Sao
Writer
```

Promotion rule:

```text
No “faster” claim without bucket movement.
No quality-changing claim without multi-image QP sweep.
No parity claim without C decoded PSNR-Y and RGB metrics.
```

---

# 15. Concrete implementation sequence

This is the order I would give an agent.

## Stage A — quarantine and parity tools

```text
1. Move old encoder out of the compile path.
2. Add encoder2 skeleton that can encode one CTU using a trivial fixed mode.
3. Add fixed-block stage-diff tool.
4. Add decoded C-vs-Rust quality harness as mandatory regression output.
```

Success:

```text
builds
old rdo2 is not imported
one tiny image can encode/decode
fixed-block diff exists
```

## Stage B — final writer bridge

```text
5. Define CuPlan/TtPlan/BlockCoeff arenas.
6. Implement conversion from v2 plans to existing writer-facing CuNode/Tt.
7. Keep write.rs mostly unchanged.
```

Success:

```text
v2 can emit valid BPG through old writer boundary
```

## Stage C — local RQT/TU core

```text
8. Implement eval_luma_final for fixed mode/TU.
9. Implement TU full-vs-split with local RQT scratch.
10. Add null-CBF comparison and exact cost logging.
11. Add large-TU-bias experiment, off by default.
```

Success:

```text
fixed luma CUs encode correctly
TU maps can be dumped and compared against C
```

## Stage D — luma mode decision

```text
12. Implement x265-shaped rough luma mode search.
13. Implement candidate shortlist.
14. Implement cheap trial / exact close-call / final commit.
15. Reuse exact winner when valid.
```

Success:

```text
mode usage matches current Best/x265 on test images
no broad quality regression
```

## Stage E — CU recursion

```text
16. Implement ModeDepth-like 2Nx2N leaf slot.
17. Implement split slot with branch-and-bound.
18. Implement winner commit from local recon scratch.
19. Remove normal frame snapshot/restore from CU trials.
```

Success:

```text
valid full-frame encode
snapshot count near zero in normal search
```

## Stage F — PartNxN and chroma

```text
20. Add NxnRoughSet.
21. Add eval_nxn4_batch.
22. Add PartNxN exact winner carry-forward.
23. Add chroma rough/trial/final path.
24. Keep chroma lower priority unless parity says otherwise.
```

Success:

```text
PartNxN wins comparable to current Best
less PartNxN wall time per win
```

## Stage G — speed path

```text
25. Add horizontal all-angles SIMD.
26. Add DST4 SIMD.
27. Add u8-native source/recon path for 8-bit 4:2:0.
28. Add allocation/arena cleanup.
```

Success:

```text
single-thread Best-equivalent is within ~1.5× x265 before asm
threaded parity target moves toward ≤1.2×
```

## Stage H — parity sweep

```text
29. Rebuild/patch C wrapper for true x265 single-thread controls.
30. Run 7+ images × QP 20/26/32/38.
31. Report decoded PSNR-Y, RGB PSNR, SSIM/MS-SSIM, bytes, time, TU maps, coefficient strength.
32. Promote v2 only if it beats old Best on quality/time or clearly matches C better.
```

---

# 16. What not to do

Do **not** ask an agent to “improve rdo2.” That invites small edits to the old design.

Do **not** keep the old encoder folder compiling as a reference. That invites copy-back.

Do **not** focus on Slow/Best merging as the main route. It may make a useful preset later, but it will not fix the x265 parity problem.

Do **not** keep adding env gates. Convert proven decisions into named policies and delete dead branches.

Do **not** prune PartNxN first. Make it cheaper first.

Do **not** optimize snapshots first. The canonical markdown says snapshot/restore is not the main ceiling anymore.

Do **not** rewrite the bitstream writer at the same time as the search core. Keep one stable boundary.

---

# 17. Minimal agent handoff

Here is the tight version I would give to an implementation agent:

```text
Goal: build still265 encoder2, a new x265-shaped still-image intra execution core. Do not modify or reuse old encoder/rdo2 or rdo_legacy code except as a behavioral reference in docs. The old encoder must be moved out of the compile path before encoder2 work begins.

Keep cabac, contexts, transform, rdoq, residual syntax helpers, primitives, and write.rs as reusable lower layers, but do not call old build_cu/build_tt/build_cu_leaf_nxn/rdo2_eval_leaf_block.

Architecture target:
- per-CTU workspace
- per-depth ModeDepth-like mode slots
- RQT scratch per TU depth
- CuPlan/TtPlan arenas for trial decisions
- local recon scratch for leaf/split/NxN trials
- final winner committed once
- writer-facing CuNode/Tt materialized only after final decision

First deliverables:
1. encoder2 skeleton compiling with old encoder removed from lib.rs
2. fixed-block stage diff tool for prediction/residual/transform/quant/RDOQ/recon
3. luma fixed-mode fixed-TU encode path
4. local TU full-vs-split path
5. plan-to-existing-writer bridge
6. luma mode search with x265-style rough shortlist
7. CU leaf/split decision without frame snapshot rollback
8. PartNxN batch evaluator
9. chroma path
10. work ledger and C-vs-Rust decoded quality sweep

Reject changes that:
- import encoder/rdo2 or rdo_legacy
- add a new env-gated heuristic without deleting an old one
- improve speed only on one image/resolution
- reduce search work but increase final replay work
- claim x265 parity without decoded C-side PSNR-Y comparison
```

Bottom line: throw out the current `encoder` execution architecture, not the whole codec. Keep the proven lower-level pieces behind a parity harness, then rebuild the search core around x265’s real strengths: fixed workspaces, local recon, plan-first decisions, retained exact winners, specialized small-block paths, and 8-bit-native storage.

Yes. Since this is **still-image-only HEVC intra**, you do not have to copy x265’s full `ModeDepth`/`RQT` architecture literally. The HEVC syntax still forces CU/TU/RQT-like decisions, but the **implementation structure** can be cleaner and potentially faster than x265’s general video encoder architecture.

The right target is not “ModeDepth/RQT but in Rust.” It is:

> **CTU-local decision graph + fixed arenas + recon overlays + batched block evaluation**

x265’s architecture is excellent C engineering for a full video encoder. It carries inter modes, rate control, temporal reuse, frame threading, VBV concerns, many presets, and years of accumulated compatibility. A still encoder can be narrower.

Rust 1.90+ does not magically make this fast, but modern stable Rust is well past that baseline now; the Rust project’s latest stable release is 1.96.0 as of May 2026, while 1.90.0 was released in September 2025. The useful point is that you can assume mature const generics, strong enum/type APIs, good LTO/linking behavior, and stable systems-programming ergonomics, while still using explicit SIMD/crate/x86 intrinsics where needed. ([Rust Blog][1])

## The core improvement over x265’s structure

x265’s `ModeDepth`/`RQT` is basically:

```text
per-depth mode slots
per-depth prediction/recon buffers
recursive CU/TU search
copy best mode into picture
```

That is good, but for still265 v2 I would make the **decision graph** explicit:

```text
CTU workspace
  ├── source views
  ├── recon overlays
  ├── candidate arrays
  ├── coefficient arena
  ├── plan arena
  ├── per-size scratch blocks
  └── final commit path
```

The key difference is that trials should not “own” trees or copied YUV buffers. They should own **plan IDs** and **scratch IDs**.

## Proposed architecture: `CtuDecisionGraph`

Instead of `ModeDepth` as the central concept, use:

```rust
struct CtuWorkspace {
    source: CtuSourceCache,
    recon: ReconOverlayPool,
    blocks: BlockScratchSet,
    candidates: CandidateArena,
    coeffs: CoeffArena,
    plans: PlanArena,
    stats: CtuStats,
}
```

Then each CU/TU trial produces:

```rust
struct Decision {
    cost: RdCost,
    plan: PlanId,
    recon: ReconId,
    confidence: Confidence,
}
```

The hot search becomes:

```rust
fn decide_cu(ws: &mut CtuWorkspace, geom: CuGeom, ctx: CtxId) -> Decision {
    let leaf = maybe_eval_leaf(ws, geom, ctx);
    let nxn = maybe_eval_nxn(ws, geom, ctx);
    let split = maybe_eval_split(ws, geom, ctx, leaf.cost);

    choose_best([leaf, nxn, split])
}
```

Only the winner is committed:

```rust
fn commit_decision(enc: &mut Encoder2, ws: &CtuWorkspace, decision: Decision) {
    enc.recon.commit(ws.recon[decision.recon]);
    enc.writer_plan.push(ws.plans[decision.plan]);
}
```

This is more idiomatic Rust than x265-style object mutation, and it is also more suitable for avoiding accidental recomputation.

## Why this can beat a literal ModeDepth/RQT port

### 1. Still-image constraints are narrower

You do not need:

```text
inter prediction
motion search
reference frames
VBV
lookahead
B-frame machinery
temporal AQ
frame-level pipeline latency
most rate-control paths
```

So your core can be shaped entirely around:

```text
intra mode
CU split
TU split
RDOQ
chroma mode
PartNxN
loop filters
```

That means fewer mode slots, fewer invalid states, smaller workspaces, and less branching.

### 2. You can make all candidate storage fixed-capacity

x265 often uses arrays internally, but still has a general-purpose C object structure.

In Rust, define bounded candidate containers:

```rust
type ModeList = ArrayVec<ModeCandidate, 16>;
type ChildList = ArrayVec<CuPlanId, 4>;
type TuChildList = ArrayVec<TtPlanId, 4>;
```

No `Vec<Tt>` during search. No `Vec<CuNode>` during search. No recursive allocation. The writer can still receive a final materialized tree after the decision.

### 3. You can use type-state to prevent replay bugs

The current code has policy flags like:

```text
CheapTrial
ExactTrial
Final
retain_levels
commit
```

In v2, make those states different types instead of runtime flags:

```rust
struct CheapEval;
struct ExactEval;
struct FinalEval;

fn eval_block<S: EvalStage>(...) -> BlockEval<S>;
```

Or even simpler, separate functions:

```rust
eval_luma_cheap()
eval_luma_exact()
eval_luma_final()
```

That is less abstract, but probably faster and safer.

The important invariant becomes compile-time visible:

```text
Cheap result cannot be emitted.
Exact retained result can be committed if prediction context is unchanged.
Final result must leave recon committed.
```

### 4. Recon overlays are better than frame snapshots

ModeDepth/RQT copies working YUV buffers. Your current Rust path snapshots/restores frame regions. A better Rust-specific model is a read-through overlay:

```rust
struct ReconView<'a> {
    base: &'a ReconFrame,
    overlay: &'a LocalReconPatch,
}
```

Prediction reads:

```text
local overlay first
committed frame second
edge/unavailable third
```

Trials write only to the overlay. The winning overlay is committed once.

This avoids the two worst patterns:

```text
shared-frame mutation during loser trials
snapshot/restore rollback
```

### 5. Batched small-block work is easier in a clean Rust design

PartNxN and 4×4/8×8 TUs are tiny and overhead-sensitive. A literal x265-ish port may still call the full block pipeline repeatedly.

A better still-specific design has batch entry points:

```rust
eval_4x4_luma_batch::<4>()
eval_nxn4_batch()
eval_chroma_pair_batch()
```

For PartNxN:

```text
extract four 4×4 source blocks once
derive four MPM sets
rough-score four PUs
batch DST4 / RDOQ setup
commit PU recon sequentially only for the winner path
```

You still respect HEVC’s PU ordering, but you reduce setup churn.

### 6. Whole-image preanalysis can be stronger than x265’s video assumptions

x265 is optimized for video, where per-frame work must be bounded and temporal consistency matters. A still encoder can afford a richer prepass:

```text
texture map
smooth/mid/edge map
saliency/importance map
chroma-critical map
estimated TU depth preference
estimated coefficient budget pressure
```

This does not replace RDO. It guides where to spend exact work.

Useful still-specific policies:

```text
large-TU bias in smooth/mid regions
texture-preserving RDOQ checks
selective PartNxN islands
saliency-weighted AQ
chroma simplification outside important regions
```

This is one area where still265 can eventually exceed x265, because x265’s defaults are not purely “best still image at arbitrary encode time.”

## The architecture I would build

### Module layout

```text
encoder2/
  mod.rs
  params.rs
  geom.rs
  ctu.rs
  workspace.rs
  source.rs
  recon.rs
  candidates.rs
  rough.rs
  eval/
    mod.rs
    luma.rs
    chroma.rs
    transform.rs
    rdoq.rs
    residual_bits.rs
  search/
    cu.rs
    tu.rs
    nxn.rs
    chroma.rs
  plan.rs
  commit.rs
  final_emit.rs
  stats.rs
```

### Main data flow

```text
image preanalysis
  ↓
CTU workspace reset
  ↓
rough candidate generation
  ↓
cheap block evaluation
  ↓
exact close-call evaluation
  ↓
plan selection
  ↓
single final commit
  ↓
writer-facing syntax emission
```

### Key internal representations

```rust
#[derive(Clone, Copy)]
struct CuGeom {
    x: u16,
    y: u16,
    log2_size: u8,
    depth: u8,
}

#[derive(Clone, Copy)]
struct TuGeom {
    x: u16,
    y: u16,
    log2_size: u8,
    depth: u8,
    component: Component,
}

#[derive(Clone, Copy)]
struct RdCost {
    distortion: u64,
    frac_bits: u64,
    cost_q: i64, // fixed-point, not f64 if possible
}
```

I would strongly consider replacing `f64` RD costs with fixed-point integer costs in the hot path. x265’s RD cost model is integer-heavy. Rust `f64` is not automatically bad, but a fixed-point `i64` cost can reduce ordering ambiguity, improve reproducibility, and simplify trace diffs.

Example:

```rust
fn rd_cost(dist: u64, bits_scaled: u64, lambda_q16: u64) -> u64 {
    dist + ((bits_scaled * lambda_q16) >> 16)
}
```

## The better-than-x265 candidate engine

The major conceptual upgrade would be a **candidate DAG** rather than a recursive tree builder.

For a CU:

```text
candidate 0: 2Nx2N leaf, mode M
candidate 1: 2Nx2N leaf, mode N
candidate 2: NxN
candidate 3: split
```

Each candidate points to reusable pieces:

```text
rough prediction result
mode bits
TU plan
coeff block IDs
recon patch ID
```

So if two decisions share an evaluated block, they can reuse it safely. This is especially useful for:

```text
rough 4×4 modes reused by PartNxN
luma winner reused by chroma search
exact trial reused by final commit
TU leaf result reused by full-vs-split decision
```

This is not “memoize everything.” Intra prediction depends on reconstructed neighbors, so many things are context-sensitive. But a CTU-local DAG lets you reuse the pieces that are genuinely invariant.

## Where x265 is still hard to beat

Do not assume the clean Rust architecture will automatically beat x265.

x265 still has advantages:

```text
hand-written assembly
mature primitive tables
years of RDOQ tuning
cache-aware C data structures
carefully evolved intra heuristics
```

A clean Rust design can beat a literal port architecturally, but only if the hot loops are not overly “high-level idiomatic.” For codecs, idiomatic Rust should mean:

```text
safe public structure
flat hot loops
fixed-capacity buffers
const-generic size specialization
no trait objects in hot paths
no heap allocation in trials
carefully contained unsafe for SIMD/alignment where justified
```

Not:

```text
iterator-heavy inner transforms
boxed trait pipelines
generic recursive tree objects
allocating Vecs for every candidate
runtime policy enums inside every pixel/block loop
```

## What I would not copy from x265

I would not copy x265’s full `ModeDepth` object structure one-to-one.

I would copy these ideas:

```text
per-depth workspaces
candidate slots
RQT scratch
local recon buffers
primitive tables
winner copied once
branch-and-bound split search
x265 rough intra shortlist shape
```

I would not copy:

```text
full Mode object hierarchy
general inter/intra mode machinery
deep mutable object graph
video-rate-control assumptions
C-style broad structs with many inactive fields
```

## Best target architecture in one sentence

Use **x265’s algorithms and primitive discipline**, but implement them as a **still-image-only CTU decision graph with fixed arenas, typed eval stages, local recon overlays, batched small-block evaluation, and whole-image preanalysis**.

That is plausibly better than ModeDepth/RQT for this project. Not because RQT is wrong, but because x265’s RQT implementation is solving a larger problem than still265 needs to solve.

[1]: https://blog.rust-lang.org/releases/latest/?utm_source=chatgpt.com "Announcing Rust 1.96.0"
