I have both subagent reports and the x265 code comparison. I’ll synthesize a candid verdict, with concrete “if we continue” targets ranked by expected ceiling versus rewrite value.## Executive verdict

**I do not think the current still265 architecture is “fundamentally unrescuable,” but I also do not think a Slow/Best merge will get you to bpg-c/x265 speed parity at comparable quality.** The evidence says the current architecture can probably yield another **~20–40%** in realistic speed from targeted work, maybe more on specific content/resolutions, but not the **2–4×** still needed across the board.

The honest assessment is:

1. **There is no obvious missed one-line/search-knob speed win left.**
2. **A full CU/TU search rewrite is not yet justified as the next move**, because rdo2 already embodies the right broad idea—cheap screen, exact close-call, winner replay—and most naive pruning attempts have been measured and rejected.
3. **The biggest remaining gap vs x265 is likely implementation throughput + data layout + specialized primitives**, not simply “we search too much.”
4. If the goal is **bpg-c speed parity**, you probably need a **second-generation search/eval core**, but not necessarily a totally different RDO algorithm: more like an x265-style tightly integrated `ModeDepth`/`RQT` search engine with fixed-size aligned scratch, 8-bit-native pixel paths, late materialization, and hand-specialized kernels.

So: **do not throw away still265 yet**, but also **do not expect Slow+Best tuning to close the gap**.

---

## Current speed situation from README

The README’s latest high-res timings show still265 still substantially behind bpg-c/x265:

- **12 MP:** C `1.56s`; still265 `best 4.83s` = **3.1× slower**, `slow 2.77s` = **1.8× slower**, `fastadaptive 2.06s` = **1.3× slower**.
- **50 MP:** C `2.90s`; still265 `best 10.70s` = **3.7× slower**, `slow 8.20s` = **2.8× slower**, `fastadaptive 5.70s` = **1.9× slower**.

The README also notes that Best adds NxN search and exhaustive rough-mode search—about **15× Slow’s CU trials**—for only **1.3–1.7× wall-clock**, and it buys real size/quality on textured content. That already tells us the cost is not just a dumb multiplicative explosion; a lot of the extra Best search is doing useful work.

---

## What the current architecture is really doing

The unit of expensive work is:

```text
rdo2_eval_leaf_block =
predict → residual → forward transform → quant/RDOQ → inverse/recon → residual price
```

This is explicitly the hot unit in `crates/still265/src/encoder/rdo2/tu.rs`.

### Best’s multiplicative structure

For a 64×64 CTU, Best effectively does a full CU quadtree down to 8×8:

| CU size | Count |
|---|---:|
| 64×64 | 1 |
| 32×32 | 4 |
| 16×16 | 16 |
| 8×8 | 64 |
| **Total possible CU leaf decisions** | **85** |

Best has important prunes:

- `best_cu_no_64_leaf` skips the 64×64 leaf below QP35.
- `best_cu_early_16` is on.
- CU split branch-and-bound aborts split evaluation once accumulated child cost cannot beat leaf.
- But for most remaining nodes, it still builds the leaf and then tries split children.

At each CU leaf, it then does:

1. **Rough luma mode search**
   - Usually scores all 35 modes.
   - Uses the x265-like “within 25% of best or MPM0” shortlist.
   - Keeps up to about 8–9 RD candidates.
2. **Cheap luma candidate screen**
   - One single-block `rdo2_eval_leaf_block` per candidate for ≤32×32.
3. **Close-call exact recheck**
   - Top 2 or 3 candidates if close.
4. **Winner full TU search**
   - EvaluateBoth transform-tree leaf vs split recursively.
5. **Chroma mode search**
   - 5 rough chroma modes.
   - narrowed RD candidates, each requiring Cb+Cr block evals.
6. **PartNxN at 8×8**
   - Evaluates 2Nx2N and NxN.
   - NxN adds four 4×4 PUs, each with its own rough luma search and exact/RDOQ candidate evals.

The top multiplicative sources are:

1. **CU quadtree × TU quadtree.**
   - 32×32 TU screen: 85 luma TU evals per CU leaf.
   - 16×16 TU screen: 21.
   - 8×8 TU screen: 5.
2. **PartNxN.**
   - Per 8×8 CU, NxN adds roughly `4 × (8–9 RDOQ candidates + final)` luma evals.
3. **All-mode rough search and 8–9 luma candidates per CU leaf.**

That said, this structure is not insane. x265 placebo also does heavy intra RDO. The problem is that x265’s implementation is far more efficient per unit and has mature local gating/caching.

---

## What x265 is doing differently

I reviewed the vendored x265 4.1 source.

### 1. x265 also evaluates leaf + split, but its top-level path is tighter

In `analysis.cpp::compressIntraCU`, x265:

- checks the 2Nx2N intra mode if `mightNotSplit`;
- checks NxN for 8×8 CUs;
- then recurses into children if `mightSplit`;
- can use `bEnableSplitRdSkip` to stop split recursion when accumulated child cost exceeds current best.

That is structurally similar to still265’s CU branch-and-bound.

But x265’s data path is much tighter:

- `ModeDepth` arrays per depth.
- `m_rqt[]` contexts and scratch.
- Copying `Yuv` blocks / context state rather than snapshotting broad frame regions.
- Highly specialized per-size primitives.

### 2. x265’s luma mode search is similar in *shape*, but faster in implementation

In `search.cpp::estIntraPredQT`, x265:

- scores DC, planar, and all angular modes with SATD/SA8D;
- computes `maxCandCount = 2 + rdLevel + ((depth + initTuDepth) >> 1)`;
  - placebo `rdLevel = 6`, so candidate cap is 8–9, just like still265;
- selects modes within 25% of best or MPM0;
- runs simple RDO for shortlist candidates without TU splits;
- then **remeasures only the best mode allowing TU splits**.

still265 has essentially copied this selection logic. That means **rough-mode shortlist count is probably not the big overlooked mismatch**. The issue is that still265’s all-angle prediction/SATD path and later eval path cost more.

### 3. x265’s TU search has mature gating and caching

In `search.cpp::estimateResidualQT`, x265 has:

- `bCheckSplit`
- `bCheckFull`
- `limitTU` mechanisms
- `m_cacheTU` save/load paths
- null-CBF optimization
- optional transform-skip
- full-vs-split shortcuts where split is not considered if full is all-zero/minimal energy
- cached BFS/DFS/neighbor-limited TU decisions.

Important: bpgenc placebo sets `limitTU=0`, so not all of those are active for the exact reference. But the x265 codebase is built around RQT scratch/caches and context stores; still265 is still more tree-materialization oriented.

---

## Prior speed work: what is exhausted

The project docs and subagent review are very clear.

### Exhausted or mostly exhausted

#### 1. Trial residual bit approximation

Tried. Bad.

- `BPG_BEST_TRIAL_APPROX_BITS=1` gave only ~5% speed but **+9% PSNR-Y BD-rate** regression.
- Verdict: ranking corruption.

#### 2. Wholesale Balanced scheduler merge

Tried. Bad.

- `BPG_BEST_SCHEDULER=balanced` gave ~1.39× speed but **+7.96% PSNR-Y BD-rate**, worse MS-SSIM too.
- Verdict: Pareto-dominated.

#### 3. PartNxN pruning

Tried multiple ways. Bad.

- Conservative prune: no real speed, +2.5% BD.
- Exact branch-and-bound: bites too late, even slower.
- Verdict: “You cannot cheaply prune a close competitor.”

#### 4. More CU force-leaf above 16×16

Tried/diagnosed. Unsafe.

- 32×32/64×64 mistake rates too high.
- safe16 is reasonable, but not a huge wall-clock win.

#### 5. Force-split predictors

Mostly bad.

- False positives cause expensive child search.
- Even true positives do not guarantee wall-time win.

#### 6. Snapshot/restore optimization

The snapshot hypothesis was mostly disproven.

- Later profiles put snapshot/restore around **~1.3%**.
- Some snapshot/allocation cleanups helped, but not enough for parity.

#### 7. TU reuse

The user-provided handoff says `BPG_BEST2_TT_REUSE`:

```text
2.18s → 2.30s, slower
518,229 → 513,703 B, ~0.9% smaller
```

So it is a compression lever, not speed.

---

## What still looks legitimately unexhausted

These are not “guaranteed parity” levers. They are the best remaining bets.

### 1. Full 8-bit-native pipeline

This is probably the largest unbuilt implementation-throughput project.

Current still265 uses `u16` planes widely even for 8-bit input. That means:

- double memory bandwidth,
- larger snapshots,
- worse cache behavior,
- more conversion pressure,
- less natural SIMD packing.

Docs warn partial 8-bit attempts regressed. But a **partial** path is exactly how this goes wrong. The viable version must be end-to-end:

```text
source 8-bit
recon/frame 8-bit
prediction output 8-bit
SATD/SSE 8-bit
residual i16
transform i16
snapshots 8-bit
deblock/SAO compatible path
```

Expected ceiling: maybe **10–30%**, content/machine dependent. Could be bigger if memory bandwidth is a hidden bottleneck at high resolution, but I would not promise 2×.

### 2. Missing specialized kernels / SIMD / asm

Current profile:

```text
quant + rdoq        32%
fwd_transform       23%
rough SATD search   17%
residual_price      10%
exact_residual_bits 7%
predict             7%
inv_transform       3%
```

Even if “portable SIMD” exists, x265 has years of per-size hand-tuned assembly. still265 still lacks or only partially covers:

- horizontal all-angles intra prediction,
- forward DST4,
- encoder inverse-transform routing,
- residual pricing/CABAC-cost optimization,
- hand-written asm backend.

This is not glamorous, but if you want bpg-c speed parity, this is unavoidable.

Expected ceiling: **20–50% cumulative** if done aggressively, maybe more with asm and 8-bit-native layout. Not a simple tuning pass.

### 3. Winner coding once, especially PartNxN carry-forward

There is still duplicated work in places.

`best2_cu_reuse` is on, so CU-level final replay is mostly handled. But PartNxN still has a relevant opportunity:

- default PartNxN screens each 4×4 PU exactly,
- then final-codes winner again,
- `BPG_NXN_CARRY` exists and should avoid redundant per-PU re-RDOQ,
- it is default-off.

The code comments say it should be byte-identical in the eligible path. This is worth testing broadly and probably promoting if it holds.

Expected ceiling: **a few percent**, maybe more on content with heavy PartNxN.

### 4. Chroma trial-RDOQ / exact-recheck cleanup

Docs say chroma is not the largest headroom, but the current path still does:

- rough chroma,
- scratch candidate evals,
- exact rechecks when close,
- then final winner coding downstream.

The previous conclusion was “small blocks, low headroom,” but with quality now being accepted via chroma-QP reallocation, there may be a new speed opportunity:

If chroma is intentionally lower priority, the encoder can probably use a much cheaper chroma decision path without affecting accepted luma quality much.

This is not pure quality-preserving speed, but under the newly accepted quality target it may be valid.

Expected ceiling: **5–10%**, possibly more if chroma-QP offset is used by default for luma-focused comparisons.

### 5. Normal-QP Best early termination still has some room

`best_cu_no_64_leaf` and `best_cu_early_16` are already on. But docs say broader early termination at normal QP is still underused.

The promising shape is not blunt force-leaf/split; it is classifier-gated refinement:

```text
Slow-like shallow result first
classify blocks into:
  - safe accept Slow
  - must escalate to Best
  - cheap exact recheck only
```

This is effectively a new scheduler above Best, not a rewrite of RDO. But prior attempts show this is hard:

- Slow is fast but loses size/quality on textured content.
- SlowPlus deep repair helped but did not close gap.
- More exact close-call leaf-vs-split did not solve structural under-splitting.

Expected ceiling: **content dependent**. On smooth/high-res, could approach Slow speed with Best-like quality. On textured mid-res, likely not.

---

## “Slow + Best merge” feasibility

### Why it is tempting

Slow is much faster:

- README 12 MP: Best 4.83s vs Slow 2.77s.
- README 50 MP: Best 10.70s vs Slow 8.20s.

And on smooth content, Slow can be close to Best or even edge luma PSNR.

### Why it cannot be the universal solution

Slow’s speed comes from **not doing the deep search that Best uses to win textured content**.

Docs show:

- Best codes ~2.5× more CUs than Slow.
- Best wins much more 4×4 NxN.
- SlowPlus deep repair recovers part of the gap, but only part.
- At 12 MP, Best is ~11% smaller than Slow at equal/slightly better PSNR.

So a Slow/Best merge will become a classifier problem:

```text
Can we identify the blocks where Slow is enough and only run Best where necessary?
```

That is plausible, but prior simple classifiers were either unsafe or low-impact. The remaining version would need to be more like a real learned/hand-tuned decision system, not another env knob.

### My honest take

A Slow/Best merge is **worth exploring as a new tier**, but I would not bet the project on it reaching bpg-c parity.

Expected result if done well:

- **10–30% faster than Best**
- near-Best quality on many photos
- still slower than bpg-c at high resolution
- tricky BD-rate regressions on textured images

---

## Is the architecture fundamentally flawed?

### Not fundamentally flawed

The architecture is not obviously stupid. It has:

- x265-like rough mode shortlist;
- rdo2 cheap screen / exact close-call;
- CU split branch-and-bound;
- PartNxN support;
- CTU parallelism / tiling;
- scratch-backed RDOQ and residual pricing;
- SIMD coverage in key spots.

It is a real encoder.

### But it is not x265’s architecture

x265’s speed is not just “better heuristics.” It is the result of:

- per-depth `ModeDepth` / `RQT` scratch;
- delayed materialization;
- highly specialized primitive tables;
- fixed-size aligned buffers;
- assembly kernels;
- efficient context save/restore;
- years of maturity in memory layout.

still265 has moved in that direction, but it is still much more “Rust tree + snapshots + generic eval unit” than x265’s hand-tuned search machine.

So if the bar is **beat x265 placebo speed at similar quality**, the current architecture probably needs a second-generation eval/search core.

Not because rdo2 was wrong, but because rdo2 is still layered on data structures and primitives that are not competitive enough.

---

## What I would do next: ranked plan

### Phase 1 — Do not rewrite yet; quantify remaining exact headroom

Run a clean 4MP/12MP/50MP benchmark matrix for these *already-existing* or low-risk gates:

1. `BPG_NXN_CARRY=1`
2. `BPG_NXN_PU_RDOQ_TOP=K` for K = 3, 5, 7
3. `BPG_SMOOTH_TU_LEAF=4` and `=5`
4. `BPG_BEST_CU_ZERO_LEAF_32=1`
5. `BPG_CU_EARLY_K` values at QP 24–36
6. chroma cheaper path under accepted chroma-QP offset
7. `BPG_BEST2_TT_REUSE=1` as optional size mode, not speed

Use equal file-size / luma-PSNR as the accepted quality metric, not equal QP.

If none gives >10% cleanly, stop tuning knobs.

### Phase 2 — Build throughput infrastructure, not more heuristics

Highest priority:

1. End-to-end 8-bit-native path.
2. Forward DST4 SIMD.
3. Horizontal angular prediction SIMD.
4. residual/CABAC price hot-loop optimization.
5. hand-written asm backend for the top 2–3 primitives.
6. avoid `Vec`/tree materialization for hot TU split kids where possible.

This is likely the only path to large speed gains without quality loss.

### Phase 3 — Prototype “Best search core v2” only if Phase 2 is insufficient

Not a full “throw everything away,” but a new search core for Best:

```text
- fixed per-depth CU/TU workspaces
- no heap tree allocation during search
- explicit ModeDepth-like candidate slots
- local recon buffers per depth
- context save/restore instead of broad snapshots
- winner materialized once at the boundary
- optional 8-bit frame path
```

This is basically: **keep the algorithmic knowledge from rdo2, replace the execution engine.**

That is the rewrite I would endorse if speed parity is mandatory.

---

## Bottom line

**No, I do not see a large overlooked Slow/Best tuning lever that will magically put still265 at bpg-c speed while preserving Best quality.**

The remaining search knobs have mostly been mined. The best untried search-side idea is a serious Slow/Best classifier, but that is a project, not a knob, and it will probably be content-dependent.

**If “must match x265 speed” is a hard requirement, the next serious investment should be a new x265-style search/eval execution core and/or aggressive primitive/data-layout work—not another RDO rewrite and not another shallow scheduler merge.**

My recommendation:

- **Keep current still265/rdo2 as the correctness and quality baseline.**
- **Bank the chroma-QP quality result as a luma-parity mode.**
- **Do one short final speed-knob sweep on carry-forward / chroma simplification / existing early gates.**
- If that does not produce a clean >20–30% improvement, **start Best-core-v2 around fixed scratch, 8-bit-native storage, and x265-like ModeDepth/RQT execution.**
