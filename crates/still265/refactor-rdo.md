Assuming “performance” means compression performance/quality: I would **not** rewrite `rdo.rs` as a cleaner version of the current file. I would rewrite it around a different search architecture.

The current file mixes rough mode decision, candidate evaluation, frame mutation, snapshot/restore, transform coding, RDOQ, residual-bit pricing, TU split, CU split, chroma search, and final reconstruction in one recursive engine. That makes it correct, but it also means weak candidates often pay almost the same cost as serious candidates. The current `code_block_internal()` path does prediction, residual, transform, quant/RDOQ, reconstruction, and residual pricing for every block trial, while luma/chroma/TU/CU search recursively call into it. 

If I were rewriting it, I would build an **x265-style staged decision engine**:

```text
rough screen
  -> cheap RD trial
    -> exact close-call recheck
      -> final replay/commit
```

Not:

```text
every plausible candidate
  -> full RD/RDOQ/reconstruct/context-cost path
```

## 1. Separate “evaluate” from “commit”

This is the biggest architectural change.

Current style:

```text
trial_code_block()
  predict into frame
  residual
  transform
  quant/RDOQ
  reconstruct into frame
  later compute distortion by reading frame again
  maybe restore snapshot
```

New style:

```text
eval_block(commit = false)
  predict into scratch
  residual
  transform
  quant/RDOQ policy
  inverse/recon residual if needed
  compute distortion from residual - recon_residual
  do not write to frame

commit_block()
  replay chosen block into frame once
```

For a leaf block, distortion does not require writing to the reconstructed frame. You can compute it as:

```text
distortion = SSE(original_residual - reconstructed_residual)
```

That avoids:

```text
write candidate recon into frame
read frame back for distortion
snapshot/restore frame for losing candidates
```

You only need to commit reconstructed samples when later predictions depend on them, such as inside a split branch or the final chosen path.

This single design change would simplify much of the file.

## 2. Make block evaluation mode-explicit

I would replace `code_block_internal()` with a `BlockCoder` that supports several evaluation modes.

Something like:

```rust
enum EvalKind {
    RoughSatd,
    CheapTrial,
    ExactTrial,
    Final,
}

struct EvalPolicy {
    rdoq: RdoqPolicy,
    residual_bits: ResidualBitPolicy,
    reconstruct: ReconstructPolicy,
    commit: bool,
    keep_levels: bool,
}

enum RdoqPolicy {
    Off,
    CheapSingleScan,
    FullSingleScan,
}

enum ResidualBitPolicy {
    None,
    Approx,
    Exact,
}
```

Then the hot path becomes:

```rust
fn eval_block(
    &mut self,
    block: BlockDesc,
    mode: IntraPredMode,
    ctxs: &Contexts,
    policy: EvalPolicy,
    scratch: &mut SearchScratch,
) -> BlockEval
```

`BlockEval` should contain:

```rust
struct BlockEval {
    distortion: u64,
    frac_bits: u64,
    nnz: u32,
    cbf: bool,
    cost: f64,
    levels_ref: Option<LevelsHandle>,
    confidence: DecisionConfidence,
}
```

The important point: **not all trials deserve the same policy**.

For example:

```text
rough screen:
  RDOQ off
  residual bits approximate or none
  no frame commit

cheap trial:
  plain quant or cheap RDOQ
  approximate bits
  no frame commit unless split children need it

exact close-call:
  full RDOQ
  exact residual bits
  commit only inside branch scope

final:
  full final coding
  commit
```

The current code has some of these concepts, but they are attached to effort templates and work stages in a way that still funnels many trials through the same heavy block path. I would make the policy explicit at the call site.

## 3. Use close-call escalation as the core rule

The rewrite should preserve compression quality by escalating close calls, not by doing exact work everywhere.

Every decision should follow this shape:

```rust
let cheap_a = eval_candidate(a, CheapTrial);
let cheap_b = eval_candidate(b, CheapTrial);

if clearly_better(cheap_a, cheap_b) {
    choose cheap winner;
} else {
    let exact_a = eval_candidate(a, ExactTrial);
    let exact_b = eval_candidate(b, ExactTrial);
    choose exact winner;
}
```

This applies to:

```text
luma mode candidates
chroma mode candidates
TU leaf vs split
CU leaf vs split
PartNxN vs 2Nx2N
```

This is how I would keep quality. Not by forcing old Best’s exact trial behavior, but by saying:

```text
If the decision is obvious, do not spend exact work.
If the decision is close, spend exact work.
```

That should recover most of the quality lost by broad trial-RDOQ gating while keeping most of the speed.

## 4. Rewrite TU search first

The transform-tree logic should be the first target because it multiplies block calls heavily.

I would make TU decision return a compact `TtDecision` plus cost, not a fully materialized tree full of cloned coded blocks.

Pseudo-structure:

```rust
fn analyze_tt(
    &mut self,
    pos: BlockPos,
    luma_mode: u8,
    chroma_mode: u8,
    ctxs: &Contexts,
    policy: &SearchPolicy,
    scratch: &mut SearchScratch,
) -> TtDecision {
    if cannot_split {
        return analyze_tt_leaf_exact_or_cheap(...);
    }

    let leaf = analyze_leaf(..., CheapTrial);

    if leaf_is_obvious_enough(&leaf, policy) {
        return leaf;
    }

    let split = analyze_split(..., CheapTrial);

    if close(leaf.cost, split.cost) {
        let leaf_exact = analyze_leaf(..., ExactTrial);
        let split_exact = analyze_split(..., ExactTrial);
        return min_cost(leaf_exact, split_exact);
    }

    min_cost(leaf, split)
}
```

For split evaluation, children need intra dependencies, so they must be committed in branch order. But this should happen inside a **branch checkpoint**, not through ad hoc snapshot objects everywhere:

```rust
let checkpoint = frame.checkpoint(region);
let child0 = analyze_child(..., commit = true);
let child1 = analyze_child(..., commit = true);
...
frame.restore(checkpoint);
```

For leaf evaluation, most candidates can be non-committing.

## 5. Rewrite CU search as branch-and-bound with cheap lower bounds

CU search should use a similar staged design.

Pseudo-flow:

```rust
fn analyze_cu(...) -> CuDecision {
    let leaf = analyze_cu_leaf(..., CheapTrial);

    if cu_leaf_early_accept(&leaf, features, neighbors, policy) {
        return exact_if_needed(leaf);
    }

    let split_bound = estimate_split_lower_bound(...);

    if split_bound > leaf.cost * policy.split_margin {
        return exact_if_needed(leaf);
    }

    let split = analyze_cu_split(..., CheapTrial, branch_and_bound = leaf.cost);

    if close(leaf.cost, split.cost) {
        return exact_recheck_leaf_vs_split(...);
    }

    min_cost(leaf, split)
}
```

The current branch-and-bound only helps after building some children. I would add a cheaper pre-split bound from source variance / SATD / preanalysis before building any child. It does not need to be perfect. It only needs to skip obvious non-splits.

## 6. Luma mode decision should produce ranked candidates with confidence

`decide_luma_modes()` should not only return `Vec<u8>`. It should return scores and gaps.

For example:

```rust
struct ModeCandidate {
    mode: u8,
    rough_cost: u64,
    rank: u8,
    ratio_q8: u16, // cost / best
    is_mpm: bool,
}

struct LumaModePlan {
    candidates: SmallVec<[ModeCandidate; 10]>,
    best_gap_q8: u16,
    confidence: DecisionConfidence,
}
```

Then the later RD search can say:

```text
rank 1:
  full cheap trial

rank 2:
  cheap trial, exact only if close

rank 3+:
  only in TextLike/detail regions or if rough score is very close
```

The current code sorts scores and throws away useful information. It also allocates vectors in places where bounded arrays would work. 

## 7. Chroma should be late and narrow

I would not let chroma participate broadly during luma search except in explicitly chroma-critical regions.

Flow:

```text
1. Pick luma mode/tree.
2. Rough-score chroma modes.
3. If non-ChromaCritical:
     exact-test only rank 1, maybe rank 2 if close.
4. If ChromaCritical:
     allow wider exact testing.
```

Current chroma search can rough-score all modes and then full-RD code multiple Cb/Cr candidates. That is too expensive for most 4:2:0 photos. The chroma gate already points in the right direction; I would make it a central policy, not a patch on the current function.

## 8. PartNxN must be a prechecked special path

PartNxN is valuable for quality, but it should never be “try it because eligible.”

I would make the path:

```text
8x8 2Nx2N rough cost
four 4x4 PU rough costs
syntax penalty estimate
residual-pressure estimate

if four_4x4_gain is large enough:
    run PartNxN cheap trial
    exact recheck only if close
else:
    skip
```

PartNxN should mostly appear in:

```text
text-like detail
fine directional edges
blocks with high 8x8 residual pressure
low/mid QP where small-block precision matters
```

Not flat photo regions.

## 9. No hot heap allocation

This would be a strict rule.

The rewrite should use a `SearchScratch` object with maximum-size buffers:

```rust
struct SearchScratch {
    src16: [u16; 64 * 64],
    src8: [u8; 64 * 64],
    pred16: [u16; 64 * 64],
    pred8: [u8; 64 * 64],
    residual: [i16; 64 * 64],
    coeffs: [i16; 32 * 32],
    levels: [i16; 32 * 32],
    recon_residual: [i16; 32 * 32],

    mode_scores: [(u64, u8); 35],
    mode_candidates: [ModeCandidate; 35],

    rdoq: RdoqScratch,
    residual_bits: ResidualBitScratch,
    frame_undo: FrameUndoScratch,
}
```

No `Vec` in the inner loops for:

```text
mode lists
candidate scores
Tt kids
Cu kids
RDOQ records
rank arrays
prefix/suffix arrays
source blocks
predicted u8 conversion
```

If dynamic structure is needed, use:

```text
SmallVec / ArrayVec
arena indices
fixed [Option<NodeId>; 4]
```

The current `source_block()` and rough SATD path are examples of what I would remove: source blocks are copied into fresh `Vec<u16>`, and 8-bit rough SATD converts predicted `u16` into `u8` on every score. 

## 10. Add true 8-bit path

For 8-bit images, the rough search should not produce `u16` predictions and then downconvert.

I would add:

```rust
predict_intra_into_u8(...)
intra_pred_allangs_u8(...)
satd_u8_strided(...)
ssd_u8_strided(...)
sub_residual_u8_to_i16(...)
```

The high-res comparison is mostly 8-bit. A generic `u16` internal path is convenient but punishes cache and memory bandwidth. Keep `u16` for 10/12-bit; specialize 8-bit.

## 11. RDOQ should be scratch-based and policy-controlled

The rewrite would change RDOQ in two ways.

First, make it allocation-free:

```rust
fn rdoq_single_scan_into(
    ctxs: &Contexts,
    coeffs: &[i16],
    desc: BlockDesc,
    scratch: &mut RdoqScratch,
) -> RdoqResult
```

`RdoqScratch` owns:

```text
CoeffRec[1024]
rank_of[1024]
suff_zero[1025]
pref_normal[1025]
levels[1024]
coded_sb_flags
```

Second, avoid full RDOQ unless policy requires it:

```text
CheapTrial:
  no RDOQ or cheap quant

ExactTrial:
  full single-scan RDOQ

Final:
  full final RDOQ
```

The recent diagnostic showed trial RDOQ gating gives a large speedup but changes output. That says the right answer is not “never gate RDOQ.” It is “gate it, then exact-recheck close decisions.”

## 12. Residual bits should be a service, not scattered calls

I would make residual pricing explicit:

```rust
trait ResidualPricer {
    fn approx_bits(&mut self, levels: &[i16], desc: BlockDesc) -> u64;
    fn exact_bits(&mut self, levels: &[i16], desc: BlockDesc, ctxs: &Contexts) -> u64;
}
```

And the decision engine chooses the pricing mode.

For cheap trials:

```text
approx bits
```

For close rechecks:

```text
exact bits
```

For final output:

```text
only exact if needed for stats/decision; otherwise the writer will code it
```

The current residual estimator correctly mirrors CABAC residual syntax, but that makes it expensive when used broadly in search. 

## 13. Plan first, replay once

The rewrite should favor this model:

```text
analyze:
  produce CuDecision / TtDecision / BlockDecision tree
  with costs and, only for winners, optional retained levels

finalize:
  replay winning decision tree once
  commit reconstruction
  write syntax later from the same decision tree
```

The current code frequently mutates reconstruction state during trial search and then restores it. That is correct but expensive. A plan/replay model makes losing candidates cheaper.

For exact preservation of reconstruction, final replay remains authoritative.

## 14. New module layout

I would split `rdo.rs` into modules like this:

```text
encoder/rdo/
  mod.rs              public entry points
  scratch.rs          SearchScratch, arenas, fixed buffers
  block.rs            BlockCoder / eval_block / commit_block
  rough.rs            RMD, SATD scoring, luma/chroma rough candidates
  luma.rs             luma mode decision and escalation
  chroma.rs           chroma mode decision and cache
  tu.rs               transform-tree decision
  cu.rs               coding-tree decision
  part_nxn.rs         PartNxN precheck and analysis
  cost.rs             RD cost, syntax-bit estimates, close-call logic
  policy.rs           resolved search policy from EffortTemplate
  replay.rs           final replay/commit from decisions
```

`rdo.rs` should not be a 3000-line recursive state machine. It should be a coordinator.

## 15. Quality-preservation strategy

I would not try to preserve old Best output. I would preserve compression performance with these safeguards:

```text
1. Every approximate decision records confidence.
2. Close decisions are exact-rechecked.
3. Exact recheck can change the winner.
4. Trace records:
     approximate winner
     exact winner
     changed_winner
     cost gap
     metric impact
5. Full sweep validates BD-rate, PSNR, SSIM, MS-SSIM, Butteraugli.
```

The core invariant:

```text
Approximate search may reject obvious losers.
Approximate search may not decide close cases without exact recheck.
```

That is the production-quality compromise.

## 16. What I would not do

I would not:

```text
rewrite the current recursive code line-for-line
try to make old Best bit-identical
parallelize CABAC write
add more diagnostic env gates without making them policy
start with SAO/deblock
keep allocating Vecs inside candidate loops
continue mutating/restoring frame for every weak candidate
```

The current bottleneck is not the syntax writer. It is search and trial coding.

## 17. The first concrete rewrite slice

I would not rewrite everything in one pass. I would create `rdo2` behind a feature/env gate and start with **TU decision only**.

Target:

```text
rdo2::tu
```

Scope:

```text
- New BlockCoder with CheapTrial and ExactTrial.
- New SearchScratch.
- New analyze_luma_tt() using:
    cheap leaf
    cheap split
    exact close-call recheck
- Existing CU/luma/chroma code calls into rdo2 TU path.
- Final output still goes through existing writer structures.
```

Acceptance counters:

```text
cheap_tu_decisions
exact_tu_escalations
exact_recheck_changed_winner
trial_rdoq_blocks_saved
exact_residual_estimates_saved
BD-rate / metric delta
```

If that works, move upward:

```text
1. TU rewrite
2. luma candidate rewrite
3. chroma rewrite
4. CU rewrite
5. PartNxN rewrite
6. final replay cleanup
```

## My summary answer

If I owned the rewrite, I would build a **staged, scratch-based, close-call-escalating RDO engine**.

The key changes would be:

```text
- Separate evaluation from frame commit.
- Make block trials policy-explicit.
- Use cheap trials for broad search.
- Exact-recheck only close decisions.
- Store compact plans, not fully materialized losing trees.
- Replay/commit winners once.
- Remove hot-loop Vec allocations.
- Add true 8-bit prediction/scoring.
- Make RDOQ and residual pricing scratch-based services.
```

That is the path most likely to close the x265 gap without giving back the quality gains.
