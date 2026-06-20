# rdo2 RDO-engine refactor — progress log

Single source of truth for the staged rewrite of the still265 intra RDO search
engine (`crates/still265/src/encoder/rdo_legacy.rs`, formerly `rdo.rs`) into a staged,
scratch-based, close-call-escalating engine under `encoder/rdo2/`, per
root `plan.md`. Side advice for further work:
`crates/rdo2-instructions-for-refactor.md`.

**Why** (measured): `docs/best-scheduler-quality-sweep-2026-06-19.md` showed
quality-preserving search-work reduction in the current architecture tops out
~26%; the monolithic engine funnels every candidate through the same heavy path.
The rewrite screens cheaply and escalates only close decisions.

**Posture:** do NOT preserve old Best bytes; validate by BD-rate sweep. Each
slice gated by env (`BPG_RDO2_TU` etc.), default off, until it passes round-trip
+ BD-rate. Writer/CU/luma/chroma stay on the old path until their slice lands.

**Architecture guardrail (2026-06-19):** `rdo.rs` has been renamed to
`rdo_legacy.rs` to make lingering ownership explicit. It may be used as
migration bridge and compatibility glue only. It is acceptable for early slices
to reuse stable old services (prediction kernels, transform/quant/RDOQ kernels,
residual pricing, snapshots while still needed), but new staged decisions and
Best winner replay must move into `encoder/rdo2/*` instead of accreting inside
the old recursive engine. Reuse kernels and data contracts; do not reuse old
control flow as the design. The rewrite goal is still to retire the old
monolithic recursion, not to grow a second architecture inside the same file.

_Resolved 2026-06-19:_ the slice-3 ranked-luma decision (previously inlined in
`build_cu_leaf`) now lives in `encoder/rdo2/luma.rs::rdo2_cu_luma_ranked`.
`build_cu_leaf` is a call-site interceptor: it snapshots the CU region, builds
the candidate/MPM lists and budget, then delegates the cheap-screen + close-call
escalation + final-code to `rdo2::luma`. Pure code-move (behaviour-neutral both
gate states); `estimate_luma_cu_trial_bits` bumped to `pub(in crate::encoder)`.
Round-trip 26 + transform 3 green with gates on and off.

**Integration contract (must not break):** the writer (`write.rs`) is a pure
bitstream pass over materialized `Tt`/`LeafTu`/`CodedBlock{levels,cbf}` trees;
reconstruction is committed into `state.frame` during analysis. So rdo2 must
materialize the same `Tt` and leave the winner's recon committed. Round-trip
tests assert encoder-recon == in-tree-decoder-recon (internal consistency), not
byte-stability — they survive decision changes.

---

## Core-principles scorecard (refactor-rdo.md north star)

The rewrite is judged against nine principles. Current standing (2026-06-19):

| # | principle | status | evidence / gap |
|---|---|---|---|
| 3 | cheap trials for broad search | ✅ done | `best_tt_cheap_trial` screens; slice 2 = 22% faster, BD-neutral |
| 4 | exact-recheck only close decisions | ✅ done | `policy::close`/`is_close_call` escalation in slices 1–4, 6 |
| 5 | compact plans, not materialized losing trees | ✅ done | `tt_to_plan`/`TtPlan`, `cu_to_plan`/`CuPlan`; slice 2 plan→replay |
| 2 | block trials policy-explicit | 🟢 adopting | `EvalPolicy`/`EvalKind`/`RdoqPolicy`/`ResidualBitPolicy` (policy.rs) drive `rdo2_eval_leaf_block`; the native TU screen (slice 8b, `BPG_RDO2_TT_NATIVE`) now costs the luma+chroma structure decision through explicit `CheapTrial` policy calls instead of the implicit `best_tt_cheap_trial` flag (byte-identical to legacy). Only the narrow scoped parent-chroma call still rides the flag |
| 6 | replay/commit winners once | 🟡 partial | slice 2 replays once; PartNxN screen materializes a losing tree then re-codes the winner on escalation |
| 1 | separate evaluation from frame commit | 🟢 adopting, validated | `rdo2_eval_leaf_block` ranks PartNxN PUs (`BPG_RDO2_NXN`, quality-neutral ~4% faster); native TU screen (slice 8c) drops the losing-leaf frame snapshot, **−6.6% single-thread**, byte-identical. Remaining: per-depth scratch recon so split trials never touch the shared frame. |
| 7 | remove hot-loop Vec allocations | 🟡 started | `SearchScratch` owns luma/chroma trace buffers, recycles cheap/plain quant levels, keeps RDOQ/residual-pricing scratch; slice 8c drops the native TU screen's `tu_depth` + losing-leaf frame snapshots; slice 8d pools the remaining `base_frame` snapshot in `frame_snapshot_pool`. Remaining: TU split kids, final/materialized levels, and pooling the snapshots in the hotter `build_cu_leaf`/luma-candidate loops |
| 9 | RDOQ & residual pricing as scratch services | 🟡 started | `rdoq_single_scan_into` + `RdoqScratch` landed; `rdo2/residual.rs::ResidualPricer` now owns the rdo2 pricing boundary, counters, trace split, and scratch-backed exact estimator; legacy wrappers remain for old paths |
| 8 | true 8-bit prediction/scoring | 🔴 not started (in rdo2) | legacy `src8`/SATD only; no rdo2 8-bit scoring path |

**Read:** the staged-search *shape* (3/4/5) is proven and delivered the slice-2
win. Further per-slice speed (4, 6) is **blocked on the evaluation layer**:
cheap screens still funnel through predict→transform→quant→**commit→read-back**,
so a −60% search-work reduction (PartNxN) buys ~0% wall-clock. The next
high-leverage move is therefore **not more slices over the legacy path** but
landing principles **1, 2, 7, 9** — wiring the hot screens onto
`rdo2_eval_leaf_block` + `SearchScratch` so reductions in *work* become
reductions in *time*. Then re-run slices 4/6 over the non-committing evaluator.

---

## Status board

| slice | scope | gate | state |
|---|---|---|---|
| 0 | scaffold: module, policy, scratch, gate flag, eval primitive | `BPG_RDO2_TU` | ✅ landed |
| 1 | staged `analyze_luma_tt` (luma TU leaf-vs-split, close-call) + wire | `BPG_RDO2_TU` | ✅ landed (round-trip green; BD-neutral; 7% faster) |
| 2 | `analyze_tt` (luma+chroma) via cheap-screen + exact replay; wire `build_tt` (Best's hot path) | `BPG_RDO2_TU` | ✅ landed (BD-neutral; **22% faster on Best**) |
| 3 | luma candidate decision (ranked + escalation) | `BPG_RDO2_LUMA` | ✅ gated candidate (round-trip green; +17% over slice 2; MS-SSIM borderline) |
| 4 | chroma late-and-narrow | `BPG_RDO2_CHROMA` | ⚠ measured negative — gated OFF (timing-neutral, +0.1–0.24% BD-rate); extraction kept |
| 5 | CU branch-and-bound + cheap pre-split bound | `BPG_RDO2_CU` | ✅ already present — `decide_cu` has early-term + split lower-bound abort; low headroom |
| 6 | PartNxN via non-committing evaluator (principle #1) | `BPG_RDO2_NXN` | ✅ quality-neutral (−0.01% PSNR-Y), **~4% faster** from one loop — validates the eval-layer lever |
| 6a | luma candidate boundary extraction | — | ✅ behaviour-neutral port: luma candidate trial + exact close-call escalation now live in `rdo2/luma.rs`; prepares fuller scratch-frame/plan rewrite |
| 6b | evaluator distortion contract | — | ✅ `rdo2_eval_leaf_block` now computes clipped pixel-domain SSE from scratch buffers, matching legacy distortion without frame commit/read-back |
| 6c | luma leaf-screen scratch candidate costs | `BPG_RDO2_LUMA_SCRATCH` | ✅ gated implementation: Best leaf-screen candidate ranking uses non-committing `LeafEval` cost records, no losing `Tt`; tests + smoke green |
| 6d | luma scratch exact escalation | `BPG_RDO2_LUMA_LEGACY_ESCALATE` fallback | ✅ normal scratch-screen close calls exact-recheck via rdo2 `EvalKind::ExactTrial`; legacy materializing escalation avoided |
| 6e | SearchScratch luma trace buffers | — | ✅ Phase 4 started: luma ranked/exact trace candidate Vecs are pooled in `SearchScratch`; allocation audit recorded |
| 6f | SearchScratch cheap levels buffer | — | ✅ non-committing cheap/plain quant trials recycle `SearchScratch.levels` instead of allocating returned losing levels |
| 6g | RDOQ scratch service skeleton | — | ✅ Phase 5 started: rdo2 full single-scan RDOQ uses `RdoqScratch` |
| 6h | RDOQ borrowed result model | — | ✅ `RdoqResult` borrows scratch-owned levels; rdo2 copies only committed/materialized winners |
| 6i | residual pricing service boundary | — | ✅ rdo2 pricing boundary + counters landed |
| 6j | root final residual-pricing elision | — | ✅ conditional Phase 6 slice: root CTU final replay can skip duplicate exact residual pricing when rdo2 TU is active |
| 6k | skipped-final trace accounting | — | ✅ trace CSVs now report final residual pricing elision separately from exact and approximate priced blocks |
| 6l | scratch-owned exact residual pricing | — | ✅ rdo2 exact residual pricing uses `ResidualPricingScratch`; legacy wrapper remains for old paths |
| 6m | rdo2 residual pricing trace split | — | ✅ `search_summary.csv` now reports rdo2 approximate vs exact residual pricing counts from `ResidualPricer` |
| 6n | chroma candidate allocation cleanup | — | ✅ chroma RD trace candidates are pooled and per-candidate coded blocks use a fixed two-entry holder |
| 7 | full luma stack sweep / promotion gate | Best default; escapes: `BPG_RDO2_TU=0`, `BPG_RDO2_NXN=0`, `BPG_RDO2_LUMA=0`, `BPG_RDO2_LUMA_SCRATCH=0` | ✅ promoted for Best with `BPG_RDO2_LUMA_CLOSE_MULT=2`: ~1.24x faster, +0.487% PSNR-Y BD, +0.582% MS-SSIM BD |
| 8a | scoped TU trial state | — | ✅ Phase 8 started: all cheap/exact TU trial flag writes now go through `Encoder::with_tt_trial_flags`; legacy recursive screen remains |
| 8b | native rdo2 TU cheap screen | Best default; escape `BPG_RDO2_TT_NATIVE=0` | ✅ **promoted default-on for Best**: `rdo2_analyze_tt` screens the luma+chroma tree with a native rdo2 recursion costing each leaf through `rdo2_eval_leaf_block` (explicit `EvalKind::CheapTrial`) instead of re-entering legacy `build_tt`; **byte-identical** to the legacy screen across 4:2:0/4:2:2/4:4:4, QP 24/28/32, and 2000x1500 (SHA-256 match) |
| 8c | native TU screen: drop redundant snapshots | Best default; escape `BPG_RDO2_TT_NOSNAP=0` | ✅ **promoted default-on for Best**: native screen no longer snapshots `tu_depth` (redundant) and defers the per-node `leaf_frame` snapshot, recoding the deterministic leaf on the rare leaf-beats-split case. Byte-identical (SHA-256 across 4:2:0/4:2:2, 1 MP + 2000x1500); **−6.6% single-thread encode**, parallel-neutral |
| 8d | pool native TU screen snapshot buffers | Best default; escape `BPG_RDO2_TT_POOL=0` | ✅ **promoted default-on for Best**: the per-node `base_frame` snapshot reuses a `SearchScratch.frame_snapshot_pool` free-list instead of allocating 3 plane `Vec`s/node. Byte-identical (4:2:0/4:2:2); timing **neutral** (−1.4% single-thread on a heavily-loaded machine, within noise) — an allocation-churn reduction (principle #7), not a demonstrated wall-clock win here |
| 8e | chroma RD cost-only scratch screen | `BPG_RDO2_CHROMA=1`, `BPG_RDO2_CHROMA_SCRATCH=1` | ✅ gated implementation: one-block chroma RD candidates now use non-committing `rdo2_eval_leaf_block` cost records for cheap screen + exact close-call recheck, never materializing candidate caches; 4:2:2 stacked chroma falls back. Smoke shows legacy `chroma_trial` RDOQ rows eliminated for 4:2:0; exact close-call RDOQ is now routed through rdo2. Not promoted pending BD sweep. |
| 9a | legacy leakage ledger + parent-chroma native cut | `BPG_RDO2_REPORT_LEGACY=1` | ✅ landed: Best-only legacy counters identify remaining rdo1 call-graph ownership; native rdo2 TT no longer calls legacy `build_parent_chroma_tu` during the cheap screen. |
| 9b | rdo2 final winner replay | Best default via `BPG_RDO2_TU` | ✅ landed: rdo2 owns final `Tt`/`CuNode` replay in `rdo2/finalize.rs` and materializes final blocks through `rdo2_eval_leaf_block(EvalKind::Final)`. Smoke removes `legacy_final_code_tt` and cuts legacy `code_block_internal` from 247k to 158k on 1000x750 Best. |
| 9c | rdo2 rough luma ownership | Best default | ✅ landed: rough luma candidate-list selection moved to `rdo2/rough.rs` while preserving the same SATD scorer; smoke removes `legacy_decide_luma_modes` (61k calls) from the Best report. |
| 9d | rdo2 chroma mode ownership | Best default | ✅ landed: rough chroma SATD screen + narrowing moved into `rdo2/chroma.rs`; smoke removes `legacy_decide_chroma_mode` (26.8k calls) from the Best report. |
| 9e | rdo2 luma leaf-screen fallback | Best default | ✅ landed: large-CU luma leaf-screen fallback now materializes luma-only TTs through `rdo2_eval_leaf_block` under the current trial policy; smoke removes `legacy_build_luma_tt_leaf` (10.5k calls). |
| 9f | rdo2 CU recursion ownership | Best default | ✅ landed: CU build/decision recursion moved into `rdo2/cu.rs`; smoke removes `legacy_build_cu` + `legacy_decide_cu` (30.9k calls). |
| 9g | rdo2 PartNxN final blocks | Best default via `BPG_RDO2_NXN` | ✅ landed: the Best/rdo2 PartNxN decision-time materialization now final-codes the four 4x4 luma PUs and parent chroma through `rdo2_final_code_tt_leaf` / `rdo2_final_parent_chroma_tu` instead of legacy `final_code_tt_leaf` / `build_parent_chroma_tu`. This keeps lower-effort and gate-off paths unchanged while cutting another important Best dependency on rdo1 block finalization. |
| 9h | rdo2 CU leaf + PartNxN ownership | Best default | ✅ landed: `build_cu_leaf`, `build_cu_leaf_nxn`, rough PartNxN precheck, and `decide_cu_8x8_part` moved out of `rdo_legacy.rs` into `rdo2/luma.rs` and `rdo2/cu.rs`. Legacy fallback calls remain only inside explicit non-Best/gate-off branches; the default Best CU leaf and PartNxN decision surfaces are now owned by rdo2. |
| 9i | rdo2 chroma RD default | Best default; escape `BPG_RDO2_CHROMA=0` | ✅ landed: Best now defaults `BPG_RDO2_CHROMA=1`, and the scratch chroma evaluator handles both one-block and stacked two-block chroma candidates. Default Best chroma candidate ranking/exact recheck no longer needs the legacy `trial_code_block` path. |
| 9j | remove Best rdo2 escape hatches | Best forced rdo2 | ✅ landed: `Effort::Best` now forces rdo2 TU, NxN, luma, luma scratch, chroma, native TT, no-snapshot, and TT-pool gates on regardless of `BPG_RDO2_* = 0`, and ignores `BPG_RDO2_LUMA_LEGACY_ESCALATE`. The env gates remain only for non-Best migration/debug paths. |
| 9k | guard remaining legacy fallbacks | Best forced rdo2 | ✅ landed: remaining rdo2 fallback branches that call legacy luma TT, TT screen, chroma candidate, PartNxN, or final CU helpers now hard-fail if reached under `Effort::Best`. This makes accidental Best regression to rdo1 visible at compile-checked runtime boundaries while lower-effort migration paths remain available. |
| 9l | rdo2 syntax-cost ownership | Best/default rdo2 | ✅ landed: CABAC syntax estimators used by rdo2 moved into `rdo2/cost.rs`; rdo2 modules no longer import CU/luma/chroma syntax-cost helpers from `rdo_legacy.rs`. Dead legacy copies were removed where no longer used by the remaining legacy path. |
| 9m | remove legacy CU final replay | rdo2 CU replay | ✅ landed: `rdo2/cu.rs` now always replays CU winners through `rdo2_final_code_cu`; legacy `final_code_cu`, `final_code_cu_inner`, and `final_code_cu_nxn` were deleted from `rdo_legacy.rs`. |
| 9n | remove legacy chroma candidate helper | rdo2 chroma RD | ✅ landed: `rdo2/chroma.rs` now scores chroma RD candidates and exact rechecks only through `rdo2_eval_leaf_block`; the local fallback that called legacy `trial_code_block` was deleted along with its now-unused chroma cache snapshot support. |
| 9o | PartNxN adaptive RDOQ screen audit | default full-RDOQ; diagnostic `BPG_NXN_ADAPTIVE=1` | ✅ parked: the adaptive PartNxN candidate screen stayed byte-identical on the 3-image 1000x750 QP28 smoke, but after fixing `BPG_NXN_EXACT` to avoid double exact-pricing, it did not reduce `trial_rdoq_blocks` and added plain-screen forward transforms. Default Best therefore keeps the original full-RDOQ ranking; `BPG_NXN_ADAPTIVE=1` remains only for diagnostics. |
| 10a | post-rdo2 baseline + WorkBucket ledger | `BPG_TRACE_SEARCH=<dir>` | ✅ landed: `docs/current-rdo2-baseline-2026-06-20.md` records the post-`074eed5` 1000/2000/4000 baselines and profile traces. `SearchTrace` now writes `work_ledger.csv` with per-bucket calls, wall time, size/component histograms, transform/RDOQ/residual-pricing counts, and source/snapshot/allocation placeholders. Hot rdo2 eval call sites are attributed to rough luma/chroma, luma candidate cheap/exact, TT leaf cheap/exact, NxN PU exact/cheap, chroma cheap/exact, and final replay. The old Best scheduler gate was rechecked on 1000x750 QP28 and default/off/small-luma had identical bytes and counters, so it is diagnostic-only again. |

---

## Log

### Slice 0 — scaffold + eval primitive (✅ 2026-06-19)

- New `encoder/rdo2/{mod,policy,scratch,tu}.rs`. `mod rdo2;` in `encoder/mod.rs`.
- `EvalKind`/`EvalPolicy`/`RdoqPolicy`/`ResidualBitPolicy` + `close()` in `policy.rs`.
- `SearchScratch` (pooled `pred`/`residual`/`coeffs`/`transform_tmp`) on `Encoder`,
  `mem::take`-borrowed; `BPG_RDO2_TU` gate flag (`rdo2_tu`), default off.
- `Encoder::rdo2_eval_leaf_block(ctxs,x,y,log2,c_idx,mode,qp,policy) -> LeafEval`:
  non-committing predict→residual→transform→quant/RDOQ→price; clipped
  pixel-domain distortion computed from scratch buffers; commits
  `clamp(pred+recon)` to the frame only when `policy.commit`. Mirrors
  `code_block_internal` kernel-for-kernel.
- Verify: `cargo build -p still265` clean; `encode_roundtrip` (26) + `transform_recon`
  (3) all pass with the scaffold present (gate off ⇒ behavior-neutral).

### Slice 1 — staged analyze_luma_tt (✅ round-trip; 2026-06-19)

- `rdo2/tu.rs`: `rdo2_analyze_luma_tt` + `rdo2_luma_subtree`/`rdo2_luma_split`/
  `rdo2_luma_leaf`. Mirrors `build_luma_tt`'s contract (returns `Tt`, leaves the
  winner's luma recon committed). Leaf screened with a **cheap, non-committing**
  eval; leaf-vs-split decision escalates to an **exact leaf recheck only when
  costs are close** (`policy::close`, `budget.close_call_margin`). Terminal/forced
  leaves coded exact. `tt_bits_luma` accessor added to `rdo.rs` to expose the
  module-private `estimate_tt_bits` (cat=0).
- Wired: `build_luma_tt` early-returns to `rdo2_analyze_luma_tt` when `rdo2_tu`.
- **Note on scope:** for `Best` the luma-mode search uses the leaf-screen path, so
  `build_luma_tt` is the **non-Best** TU path; Best's real luma+chroma TU decision
  is `build_tt` (slice 2). Slice 1 therefore validates the staged engine + commit
  contract end-to-end on the lower tiers before touching Best's hot path.
- Verify: `cargo build` clean; **`BPG_RDO2_TU=1` round-trip (26) + transform (3)
  all pass** (encoder recon == in-tree decoder across all tiers/formats).
- **BD-rate sanity (effort=good, 2 img × QP 24/28/32/36, gate off vs on):
  +0.12% PSNR-RGB, +0.15% SSIM-RGB (neutral, within ±1%); encode 0.93x (7%
  faster).** Confirms cheap-screen + exact-close-call preserves quality — the
  architecture works where the wholesale balanced scheduler (+8%) failed.
  Artifacts: `target/rdo2s1-good-{off,on}/`.

### Slice 2 — analyze_tt for build_tt (cheap-screen + exact replay) (✅ 2026-06-19)

- Approach (refactor-rdo.md §13): instead of reimplementing chroma, `rdo2_analyze_tt`
  screens the whole subtree cheaply via the **legacy `build_tt` forced cheap**
  (`best_tt_cheap_trial`: no trial RDOQ + approx bits) under an `in_rdo2`
  re-entry guard, takes the chosen structure as a `TtPlan` (`tt_to_plan`),
  restores the frame, and **replays it exactly via `final_code_tt`**. The cheap
  screen already recorded the winner's tu-depth map (kept); only the recon is
  recomputed exact. Reuses validated chroma/parent-chroma/final-code paths.
- `in_rdo2` guard field on `Encoder`; `build_tt` early-returns to `rdo2_analyze_tt`
  when `rdo2_tu && !in_rdo2`. `best_tt_cheap_trial` only bites for `Best` (the
  target tier); non-Best screens at normal quality (correct, no speedup).
- Verify: `cargo build` clean; **`BPG_RDO2_TU=1` round-trip (26) + transform (3)
  all pass** (all tiers/formats).
- **Best-tier sweep (7 img × QP 24/28/32/36, single-thread), gate off vs on:
  BD-rate +0.07% PSNR-Y, +0.15% MS-SSIM, +0.06% SSIM-RGB, +0.08% PSNR-RGB
  (all within ±0.15% — neutral); size +0.04%; encode 3.699s → 2.875s = 0.78x
  (22% faster).** The staged cheap-decision + exact-winner-replay closes part of
  the single-thread gap at ~zero quality cost — what the wholesale balanced
  scheduler (+8% / less speed) could not. Artifacts: `target/q1-baseline-best/`
  (off), `target/rdo2s2-best-on/` (on).
- Next levers (still ~3x vs x265): the cheap screen still runs full candidate
  breadth; slices 3/4/5 cut luma-candidate, chroma, and CU search work.

### Slice 3 — luma candidate screening (ranked + escalation) (✅ gated; 2026-06-19)

- New independent `BPG_RDO2_LUMA` gate. `BPG_RDO2_TU` no longer implicitly enables
  the luma-mode cheap screen, so TU and luma slices can be A/B tested separately.
- `build_cu_leaf` screens the ranked luma candidates with cheap Best trials
  (`best_tt_cheap_trial`: no trial RDOQ + approximate residual bits), tracks best,
  runner-up, and third-place costs, then exact-rechecks the top two when the
  decision is close. If the third cheap candidate is also within the close-call
  margin, it is included in the exact recheck as a guardrail. The chosen CU is
  still final-coded exactly afterwards.
- Added counters:
  `rdo2_luma_cheap_cu_decisions`, `rdo2_luma_exact_escalations`,
  `rdo2_luma_exact_changed_winner`.
- Verify: `BPG_RDO2_LUMA=1` and `BPG_RDO2_TU=1 BPG_RDO2_LUMA=1` both pass
  `encode_roundtrip` (26) + `transform_recon` (3).
- Counter sanity on `20240501_110934.png`, QP 28, Best, single-thread:
  15,019 cheap luma CU decisions; 4,926 exact escalations; 2,101 changed winners.
- **Best-tier sweep (7 img × QP 24/28/32/36, single-thread), gate off vs
  `BPG_RDO2_TU=1 BPG_RDO2_LUMA=1`: BD-rate +0.78% PSNR-Y, +1.13% MS-SSIM,
  +0.72% SSIM-RGB, +0.79% PSNR-RGB; size +0.71%; encode 0.65x baseline.**
  Relative to slice 2 (`BPG_RDO2_TU=1`), slice 3 adds +0.72% PSNR-Y,
  +0.98% MS-SSIM, +0.66% SSIM-RGB, +0.71% PSNR-RGB; size +0.60%; encode 0.84x
  slice 2 (about 17% faster). Artifacts: `target/rdo2s3-best-on-top3/`.
- Interpretation: speed win is real and PSNR/SSIM stay within the ±1% posture,
  but MS-SSIM is slightly above the original-baseline target. Keep gated; next
  tuning choices are a wider exact-recheck policy or moving to slice 4 and
  recovering quality through narrower chroma decisions.

### Slice 4 — chroma late-and-narrow (⚠ measured negative; gated OFF 2026-06-19)

- New `BPG_RDO2_CHROMA` gate (`Best` only). **Architecture-first:** the chroma RD
  decision moved out of `rdo.rs::decide_chroma_mode` into
  `encoder/rdo2/chroma.rs::rdo2_chroma_rd_decision`; `decide_chroma_mode` keeps
  only the shared rough SATD screen + top-`n` narrowing and delegates the RD
  decision. Gate-off path is behaviour-neutral (builds the same full-quality
  leaf cache); round-trip 26 + transform 3 green gate on/off and full-stack.
- Gated path cheap-screens the narrowed candidates (`best_tt_cheap_trial`: no
  trial RDOQ + approx bits — already chroma-applicable) and exact-rechecks the
  top two on close calls. **Key structural finding:** in the Best hot path
  `rdo2_analyze_tt` re-codes chroma exactly from the plan and *ignores* the
  chroma leaf cache, so any winner cache built here is discarded — an initial
  design that re-coded the winner exactly to keep the cache valid was pure waste
  and measured slower. Redesigned to a cost-only screen returning
  `leaf_cache: None` (downstream `final_code_tt` produces exact chroma regardless).
- **Best-tier sweep (7 img × QP 24/28/32/36, single-thread), vs slice 3
  (`BPG_RDO2_TU=1 BPG_RDO2_LUMA=1`): BD-rate +0.15% PSNR-Y, +0.10% MS-SSIM,
  +0.22% SSIM-RGB, +0.24% PSNR-RGB; size +0.10%; encode timing-neutral**
  (single-image best-of-3 at Best: 2.24s both; the batch sweep's ~5% was load
  noise). Artifacts: `target/rdo2s4-best-on/`.
- Counters (`20240502_184356.png`, QP 28, Best): 13,486 cheap chroma CU
  decisions, 1,789 exact escalations (13%), 226 changed winners — but chroma
  trial blocks are ~65k of 524k total trials, all small 4×4/8×8 where RDOQ is
  already cheap, so the screen removes little work.
- **Verdict:** structurally low headroom — chroma mode selection is a small,
  already-cheap fraction of the encode and the dominant chroma cost is the exact
  final coding (unchanged by this slice). Pareto loss (no speed, small quality
  cost). **Gate kept default OFF.** The `rdo2/chroma.rs` extraction is kept as a
  clean architectural boundary; the cheap-screen lever is not promoted. Move on
  to slice 5 (CU branch-and-bound), which targets the real cost center (the CU
  quadtree / luma TU search where the trial-block volume concentrates).

### Slice 5 — CU branch-and-bound (✅ already present; 2026-06-19)

- Investigation, not new code: `decide_cu` (rdo.rs) **already** implements the
  branch-and-bound the slice envisioned — a leaf-cost upper bound, an incremental
  split lower-bound abort (`partial_split_cost >= leaf_cost` breaks the 4-kid
  loop, `cu_split_bound_aborts`), plus `cu_early_terminate` and `ForceLeaf`/
  `ForceSplit`/`PreferSplit` budget shortcuts. Counters on Best
  (`20240502_184356.png`, QP 28): `cu_split_bound_aborts: 2269`,
  `cu_early_terminations: 0`, `cu_force_leaf: 0`. Low remaining headroom; no
  `BPG_RDO2_CU` gate added.
- This reframed the search for the real cost center → PartNxN (slice 6).

### Slice 6 — PartNxN precheck (🚧 measured cost center; 2026-06-19)

- **The measured cost center.** Best counters (`20240502_184356.png`, QP 28):
  `partnxn_cu_trials: 196039`, `partnxn_code_block_calls: 292055` (**~46% of all
  629k `code_block_calls`**), `partnxn_attempts: 9965` → `wins: 3397` (34%),
  `losses: 6568` (66%). Every interior 8×8 CU RD-compares the normal `Part2Nx2N`
  against a full four-4×4-PU `PartNxN` search (~20 trials/attempt); two thirds of
  that search loses. Screening obvious losers is the largest remaining lever
  (potential ~20% on top of slice 2's 22%).
- **Negative result first — the existing `partnxn_prune` heuristics are
  Pareto-worse on Best** (7 img × QP, vs slice 3):
  - conservative: **+2.54% PSNR-Y, +2.26% MS-SSIM** BD-rate; size +1.39%;
    encode 1.013x (no speedup).
  - aggressive: **+3.07% PSNR-Y, +2.79% MS-SSIM** BD-rate; size +1.59%;
    encode 1.046x (slower).
  Why: `should_try_partnxn_8x8` itself runs five rough mode-searches per CU
  (8×8 + four 4×4), so skipping saves little time; and its rough-gain/diverse
  signal poorly predicts actual RD wins, so it wrongly skips many of the 34%
  PartNxN winners → big quality loss. **Do not enable `BPG_PARTNXN_PRUNE` on
  Best.** Artifacts: `target/nxn-{conservative,aggressive}/`.
- **Decision-pruning is a dead end for PartNxN** — three approaches, all
  measured, none ship: (a) legacy `partnxn_prune` heuristic Pareto-worse (+2.5%
  BD, no speed); (b) cheap single-mode RD screen quality-neutral but wall-clock-
  neutral (the screen's per-PU `decide_luma_modes` dominates), and a *cheap-trial*
  variant lost +0.55% PSNR-Y; (c) exact branch-and-bound (accumulate per-PU luma
  cost, abort when it can't beat `c2`) is BD-neutral but **bites too late** —
  `partnxn_cu_trials` only −2% (PartNxN's four 4×4 PUs each cost ~¼ of the 8×8
  `c2`, so the bound only crosses near the end), netting ~8% *slower*. You cannot
  cheaply prune a close competitor.

### x265 architecture study → principle #1 is the lever (✅ validated 2026-06-19)

- Re-read `x265_4.1/source/encoder/search.cpp::estIntraPredQT` (1581–1709) and the
  `m_rqt[]` setup. x265's intra speed comes from: a sa8d rough screen over all 35
  modes (`intra_pred_allangs`, no transform) keeping only modes within 1.25% of
  best; full RD on the few survivors where **each trial writes recon/resi/coeff
  into per-depth scratch (`m_rqt[depth]`) and restores a tiny CABAC *context*
  array (`m_entropyCoder.load`)** — it **never snapshots/restores pixel frame
  regions**; only the *winner* is copied into the recon picture (1687–1692).
- Our anti-pattern: the hot trial loops `snapshot_frame_region` /
  `restore_frame_region` **per candidate** (each allocates a `Vec` and copies up
  to three pixel planes — 65 call sites in `rdo.rs`), commit recon to the shared
  frame, then read it back via `distortion_block`. Our bit estimation is already
  frozen-context, so we need *no* CABAC store/load — we have it easier than x265.
- `rdo2_eval_leaf_block` (slice 0) already is the x265 pattern: predict→residual→
  transform→quant→price into `SearchScratch`, clipped pixel-domain scratch
  distortion, commit only when asked. It was just unused. **Wired it into the PartNxN PU candidate
  ranking** (`build_cu_leaf_nxn`, `use_eval` arg, gated `BPG_RDO2_NXN`, Best):
  trials are non-committing (no snapshot/restore, no read-back); only the winning
  PU is final-coded/committed (needed for the next PU's prediction).
- **Result (vs fresh same-session gate-OFF Best, 7 img × QP):** BD-rate
  −0.009% PSNR-Y, +0.035% MS-SSIM, −0.036% SSIM-RGB, −0.021% PSNR-RGB —
  **quality-neutral**; encode **78.4s → 75.3s = 0.961x (~4% faster)** from
  converting one loop; round-trip 26 + transform 3 green. Artifacts:
  `target/{off-fresh,nxn-eval}/`.
- **This is the path.** The ~4% is one loop; the majority of trial work is in the
  `build_cu_leaf` luma-candidate loop, `build_tt`/`build_luma_tt`, and chroma —
  all still on snapshot/restore. Moving each onto `rdo2_eval_leaf_block` (with
  clipped scratch distortion + `SearchScratch`, committing only winners) is
  principles 1+7 and should compound toward the x265 gap. Next: the
  `build_cu_leaf` luma-candidate ranking (the biggest single trial loop).
- **Negative follow-up — direct luma leaf-screen swap is not enough.** Tried
  replacing only the safe/common `build_cu_leaf` luma candidate leaf-screen case
  (`Best`, `BPG_RDO2_LUMA`, `best_luma_leaf_screen`, `log2_cb_size <= MAX_TB_LOG2`)
  with a non-committing `rdo2_eval_leaf_block` wrapper that materialized a luma
  `Tt::Leaf` for `estimate_luma_cu_trial_bits`. Round-trip stayed green, but the
  full 7-image Best sweep was Pareto-negative vs `target/nxn-eval`: BD-rate
  +0.77% PSNR-Y, +0.95% MS-SSIM, +0.67% SSIM-RGB, +0.83% PSNR-RGB; size +0.64%;
  encode 1.006x. Reverted. Interpretation: the evaluator primitive is valid, but
  piecemeal replacement inside the old luma ranking changes decisions without
  enough wall-clock gain. Revisit this loop only with a fuller `rdo2/luma`
  scratch-frame/plan design that can preserve the candidate decision contract and
  avoid per-candidate restore across split/dependency cases, not as a one-leaf
  shortcut.

### Slice 6a — rdo2-owned luma candidate boundary (✅ 2026-06-19)

- Behaviour-neutral architecture slice after the failed one-leaf shortcut:
  `rdo2/luma.rs` now owns both the legacy-compatible luma candidate trial helper
  and the exact close-call escalation helper. The old `rdo.rs`
  `escalate_luma_close_call` method was removed; `rdo.rs::build_cu_leaf` remains
  a call-site bridge that snapshots the CU baseline and delegates the ranked
  decision to `rdo2_cu_luma_ranked`.
- Why this matters: the next speed slice needs the full luma candidate contract
  in one place before changing the evaluator. The direct leaf-only
  `rdo2_eval_leaf_block` swap perturbed decisions without enough wall-clock gain;
  this extraction sets up a fuller `rdo2/luma` scratch-frame/plan design that can
  preserve split/dependency behaviour and still commit only winners.
- Verify: `cargo build -p still265`; `BPG_RDO2_TU=1 BPG_RDO2_LUMA=1
  BPG_RDO2_NXN=1 cargo test -p still265 --test encode_roundtrip --test
  transform_recon` → 26 + 3 green.

### Slice 6b — scratch evaluator distortion contract (✅ 2026-06-19)

- Corrected `rdo2_eval_leaf_block` to compute distortion as clipped pixel-domain
  SSE from scratch buffers (`src` vs `clamp(pred + recon_residual)`), instead of
  residual-domain SSE. This keeps the x265-style non-committing property (no
  frame write/read-back for losers) while matching the legacy committed path's
  distortion contract more closely.
- Why this matters: the reverted direct luma leaf-screen swap likely exposed this
  mismatch. PartNxN tolerated it, but broad luma candidate ranking is more
  sensitive. Future luma-eval experiments should be retested after this fix,
  preferably behind a separate small gate, rather than changing the established
  `BPG_RDO2_LUMA` behavior silently.
- Verify: `cargo build -p still265`; `BPG_RDO2_TU=1 BPG_RDO2_LUMA=1
  BPG_RDO2_NXN=1 cargo test -p still265 --test encode_roundtrip --test
  transform_recon` → 26 + 3 green.

### Slice 6c — luma leaf-screen scratch candidate costs (✅ gated 2026-06-19)

- Added `BPG_RDO2_LUMA_SCRATCH` as a separate gate; it does not silently change
  the established `BPG_RDO2_LUMA` behavior. Scope is deliberately the safe Best
  leaf-screen case first: `Best` + `BPG_RDO2_LUMA` + `best_luma_leaf_screen` +
  `log2_cb_size <= MAX_TB_LOG2`.
- New path in `rdo2/luma.rs`: losing luma candidates use
  `rdo2_eval_leaf_block` with `EvalKind::CheapTrial`, then an rdo2-local luma
  syntax bit estimate (`part_mode`, luma intra mode, CBF luma, residual bits).
  The result is a compact cost record, not a materialized `Tt`. Close-call exact
  rechecks and final winner coding still use the current writer-compatible
  materialization path, preserving validity while moving the broad screen off
  per-candidate frame restore/commit/read-back.
- Added counters:
  `rdo2_luma_scratch_candidates`,
  `rdo2_luma_scratch_exact_escalations`,
  `rdo2_luma_scratch_changed_winner`,
  `rdo2_luma_scratch_legacy_evals_skipped`,
  `rdo2_luma_scratch_snapshot_restores_saved`.
- Smoke (`20240501_110934.png`, QP 28, Best, 4:2:0, single-thread, all current
  rdo2 gates): 92,425 scratch candidates; 92,425 legacy materializing evals
  skipped; 277,275 per-candidate frame/map/tu-depth restores avoided by counter;
  4,936 exact escalations; 2,372 changed winners; valid output
  `/tmp/rdo2-luma-scratch.bpg`.
- Verify: `cargo build -p still265`; `BPG_RDO2_TU=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 BPG_RDO2_NXN=1 cargo test -p still265 --test
  encode_roundtrip --test transform_recon` → 26 + 3 green; release
  `bpg-tools` build and smoke encode green. Full BD sweep intentionally deferred
  until the whole luma candidate loop replacement reaches milestone quality.

### Slice 6d — rdo2 exact luma escalation (✅ gated 2026-06-19)

- Implemented Phase 3 from root `plan.md` for the scratch-screen path:
  close-call luma candidates now exact-recheck with
  `rdo2_eval_leaf_block(..., EvalKind::ExactTrial)` rather than materializing
  legacy luma `Tt` trials. The chosen winner is still final-coded through the
  existing writer-compatible path.
- Added `BPG_RDO2_LUMA_LEGACY_ESCALATE=1` as a temporary fallback. Normal
  `BPG_RDO2_LUMA_SCRATCH=1` operation avoids legacy exact escalation; the legacy
  helper remains only for fallback / non-scratch bridge cases while full luma
  cutover is incomplete.
- Added counters:
  `rdo2_luma_scratch_exact_rechecks` and
  `rdo2_luma_scratch_legacy_exact_escalations_avoided`.
- Smoke (`20240501_110934.png`, QP 28, Best, 4:2:0, single-thread, current
  release binary, all current rdo2 luma/TU/NxN gates): 92,247 scratch candidates;
  11,560 exact scratch rechecks; 4,838 legacy exact escalations avoided; 92,247
  legacy materializing evals skipped; 276,741 per-candidate restores avoided by
  counter; valid output `/tmp/rdo2-luma-exact.bpg` at 121,497 bytes, 2.72s.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `cargo build --release -p bpg-tools`; smoke encode green.

### Slice 6e — SearchScratch luma trace buffers (✅ 2026-06-19)

- Started Phase 4 from root `plan.md`: `SearchScratch` now owns reusable
  `luma_trace_cands` and `luma_exact_trace_cands` buffers. `rdo2/luma.rs` takes,
  clears, and returns these buffers for the ranked luma screen and nested exact
  escalation instead of constructing fresh local `Vec<CandRec>` buffers.
- Old code bypassed/deleted: local `Vec<crate::trace::CandRec>::new()` allocation
  sites in the rdo2 luma ranked decision, scratch exact recheck, and legacy
  fallback escalation.
- Principle advanced: #7 (remove hot-loop Vec allocations) for the active luma
  scratch path, while preserving the existing trace contract.
- Allocation audit:
  removed: rdo2/luma trace candidate Vec construction for ranked screen and both
  escalation paths.
  remaining: `rdo2/chroma.rs` trace/coded candidate allocation was removed in
  slice 6n; `rdo2/tu.rs` still uses `Vec::with_capacity(4)` for split kids,
  needed until the TU plan/arena slice; `rdo2_eval_leaf_block` still receives
  owned `levels` from quant/RDOQ, scheduled for Phase 5 RDOQ scratch service.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green.

### Slice 6f — SearchScratch cheap levels buffer (✅ 2026-06-19)

- Continued Phase 4 from root `plan.md`: `SearchScratch` now owns a reusable
  `levels` buffer. `rdo2_eval_leaf_block` routes `RdoqPolicy::Off` through
  `transform::quantize_into` using that buffer.
- For non-committing `CheapTrial` + approximate-bit evaluations (the broad
  `BPG_RDO2_LUMA_SCRATCH` screen), the evaluator computes CBF, distortion, and
  residual bits, then returns an empty `CodedBlock.levels` and recycles the
  quantized levels buffer back into `SearchScratch`. Exact/committed evaluations
  still move levels into `CodedBlock` because writer-compatible materialization
  needs them.
- Old allocation bypassed/deleted: `transform::quantize` allocation in the hot
  non-committing cheap evaluator path.
- Principle advanced: #7 (remove hot-loop Vec allocations) and #9 preparation
  (levels ownership moves toward evaluator-owned scratch). Full exact RDOQ level
  ownership is still scheduled for Phase 5.
- Allocation audit update:
  removed: broad luma scratch-screen cheap trials no longer allocate/move owned
  losing `levels` Vecs.
  remaining: exact `rdoq_single_scan` still returns owned levels; committed cheap
  TU paths still retain levels for `Tt`; TU split kids remain a temporary bridge.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green.

### Slice 6g — RDOQ scratch service skeleton (✅ 2026-06-19)

- Started Phase 5 from root `plan.md`: `rdoq.rs` now exposes
  `RdoqScratch` and `rdoq_single_scan_into(...)`. The scratch object owns the
  single-scan RDOQ working arrays (`CoeffRec` records, rank map, prefix/suffix
  cost arrays, and temporary levels) instead of allocating them inside each rdo2
  exact trial block.
- `rdo2_eval_leaf_block` now routes `RdoqPolicy::FullSingleScan` through the
  scratch-backed RDOQ path. The old `rdoq_single_scan(...)` function remains as a
  compatibility wrapper for legacy `rdo.rs` control flow.
- Important remaining Phase 5 debt: the scratch API still returns owned
  `Vec<i16>` levels. Non-committing rdo2 exact trials recycle that buffer back
  into `SearchScratch`, so losing exact trials no longer retain owned level
  vectors, but the fuller `RdoqResult` borrow/copy model from `plan.md` is still
  pending for final/materialized paths.
- Allocation audit update:
  removed from rdo2 full-RDOQ trials: fresh RDOQ `recs`, `rank_of`,
  `suff_zero`, `pref_normal`, and `levels` allocations.
  remaining: legacy `rdoq_single_scan` creates a temporary scratch wrapper; final
  and committed materialization still need owned levels for writer-compatible
  `CodedBlock`; TU split kids remain a temporary bridge. Chroma candidate buffers
  were removed later in slice 6n.
- Smoke (`20240501_110934.png`, QP 28, Best, 4:2:0, single-thread, current
  release binary, all current rdo2 luma/TU/NxN gates): 92,247 scratch candidates;
  11,560 exact scratch rechecks; 4,838 legacy exact escalations avoided; 92,247
  legacy materializing evals skipped; 276,741 per-candidate restores avoided by
  counter; valid output `/tmp/rdo2-rdoq-scratch.bpg` at 121,497 bytes, 2.51s.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; smoke encode green.

### Slice 6h — RDOQ borrowed result model (✅ 2026-06-19)

- Completed the remaining Phase 5 result-ownership cut for rdo2: `RdoqResult`
  now borrows the scratch-owned levels slice from `RdoqScratch` instead of
  moving a `Vec<i16>` out of scratch.
- `rdo2_eval_leaf_block` now treats levels as either owned plain-quant levels or
  borrowed RDOQ levels. Non-committing full-RDOQ trials use the borrowed levels
  for sign-data hiding, inverse transform, clipped scratch distortion, and
  residual bit pricing, then return an empty `CodedBlock.levels`. Committed /
  writer-materialized RDOQ paths copy levels into `CodedBlock` only when needed.
- Legacy compatibility remains isolated: `rdoq_single_scan(...)` still returns
  `(Vec<i16>, u32)` by creating a temporary `RdoqScratch` and copying the
  borrowed result for old `rdo.rs` callers.
- Allocation audit update:
  removed from rdo2 full-RDOQ loser trials: moving/taking owned level Vecs out of
  scratch and recycling them afterward.
  remaining: final/materialized winner paths still allocate/copy levels for the
  writer contract; legacy `rdoq_single_scan` still allocates its temporary
  scratch/copy until old RDO flow is retired; TU split kids remain a temporary
  bridge.
- Smoke (`20240501_110934.png`, QP 28, Best, 4:2:0, single-thread, current
  release binary, all current rdo2 luma/TU/NxN gates): 92,247 scratch candidates;
  11,560 exact scratch rechecks; 4,838 legacy exact escalations avoided; 92,247
  legacy materializing evals skipped; 276,741 per-candidate restores avoided by
  counter; valid output `/tmp/rdo2-rdoq-result.bpg` at 121,497 bytes, 2.69s.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; smoke encode green.

### Slice 6i — rdo2 residual pricing service boundary (🟡 2026-06-19)

- Started Phase 6 from root `plan.md`: added `rdo2/residual.rs` with
  `ResidualPricer`, and routed `rdo2_eval_leaf_block` through it. Residual-bit
  policy selection now lives behind an rdo2 service boundary instead of an
  inline match inside the block evaluator.
- Added rdo2-specific debug counters:
  `rdo2_residual_approx_pricings` and `rdo2_residual_exact_pricings`. These are
  separate from the existing global `residual_bit_estimates` counter so the
  active rdo2 stack can account broad cheap pricing vs exact close-call/final
  pricing volume.
- Current limitation resolved by slice 6l: this boundary/control-flow slice
  initially delegated exact pricing to the existing `residual_frac_bits` helper;
  rdo2 exact pricing now uses `ResidualPricingScratch`.
- Smoke (`20240501_110934.png`, QP 28, Best, 4:2:0, single-thread, current
  release binary, all current rdo2 luma/TU/NxN gates): 86,148 rdo2 approximate
  residual pricings; 194,893 rdo2 exact residual pricings; valid output
  `/tmp/rdo2-residual-pricer.bpg` at 121,497 bytes, 2.49s.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; smoke encode green.

### Slice 6j — root final residual-pricing elision (✅ conditional; 2026-06-19)

- Continued Phase 6 from root `plan.md`: added a scoped
  `elide_final_residual_pricing` analysis flag. During rdo2 root CTU winner
  replay (`BPG_RDO2_TU=1`, `ct_depth == 0`), final `CodedBlock` materialization
  can now skip the duplicate exact residual-bit estimate because the writer will
  emit those coefficients immediately afterward.
- Guardrails: the flag is **not** active for non-root final-coded children,
  because parent CU decisions still estimate their RD cost from
  `CodedBlock.frac_bits`.
- Added counter:
  `rdo2_residual_final_pricings_elided`.
- Smoke finding: default Best currently uses the `best2_cu_reuse` winner-direct
  path, so this replay elision is bypassed there (`0` elisions on the standard
  all-gates Best smoke). With `BPG_BEST2_LUMA_FASTRD=1`, which disables the
  winner-reuse path and exercises root final replay, the same 1000x750 QP 28
  smoke recorded `32,863` final residual pricings elided and decoded
  successfully.
- Old work bypassed: exact final residual pricing in root replay only; trial
  exact pricing, close-call exact pricing, and non-root final-coded children keep
  exact `frac_bits` for RD decisions.
- Principle advanced: #9 (residual pricing is explicit and avoidable when not
  needed by a decision). Remaining Phase 6 work is scratch-owned exact CABAC
  pricing and broader rdo2 approximate/exact trace splitting.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; Best QP 28
  single-thread smoke encode/decode green with `BPG_BEST2_LUMA_FASTRD=1`.

### Slice 6k — skipped-final residual-pricing trace accounting (✅ 2026-06-19)

- Continued Phase 6 trace integration: `SearchTrace` now has an explicit
  final-pricing-elided bucket in stage totals, per-component/block-size volume,
  CTU accounting, and `search_summary.csv`. This lets the root final replay
  elision stay enabled when `BPG_TRACE_SEARCH` is active without misclassifying
  skipped exact final estimates as approximate priced blocks.
- Old ambiguity removed: trace `code_block_volume.csv` now separates
  `exact_residual_estimates`, `approx_residual_priced_blocks`, and
  `final_pricings_elided`.
- Smoke (`20240501_110934_1000x750.png`, QP 28, Best, single-thread,
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 BPG_BEST2_LUMA_FASTRD=1 BPG_TRACE_SEARCH=...`):
  debug stats reported `rdo2_residual_final_pricings_elided: 32681`, and
  `target/rdo2-trace-elide-smoke/search_summary.csv` reported
  `final_residual_pricings_elided,32681`. `stage_table.csv` and
  `code_block_volume.csv` expose matching final-code skipped-pricing columns.
  Output decoded successfully.
- Principle advanced: #9 traceability. Remaining Phase 6 work is the
  scratch-owned exact CABAC residual estimator and, separately, a broader trace
  split for rdo2 approximate vs exact pricing calls.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; trace-enabled Best
  QP 28 smoke encode/decode green.

### Slice 6l — scratch-owned exact residual pricing (✅ 2026-06-19)

- Completed the Phase 6 exact-pricing ownership cut for rdo2 evaluator paths:
  `residual.rs` now exposes `ResidualPricingScratch` and
  `estimate_residual_bits_into(...)`. `SearchScratch` owns that pricing scratch,
  and `rdo2/residual.rs::ResidualPricer` routes `ResidualBitPolicy::Exact`
  through it instead of calling legacy `Encoder::residual_frac_bits`.
- Accounting preserved: rdo2 exact pricing still increments
  `rdo2_residual_exact_pricings`, the global `residual_bit_estimates`, and
  residual-bit profiler timing. Legacy `residual_frac_bits` remains for old
  `rdo.rs` control flow until that path is retired.
- Added direct equivalence coverage in `tests/residual_roundtrip.rs`: every
  residual writer/decoder round-trip case now also checks
  `estimate_residual_bits_into` against the original `estimate_residual_bits`.
- Old code bypassed: rdo2 exact residual pricing no longer uses the legacy
  `Encoder::residual_frac_bits` wrapper or its local context clone. The exact
  CABAC estimator still replays syntax from a preserved context state, but that
  state is now owned and reused by `SearchScratch`.
- Principle advanced: #9 (RDOQ & residual pricing as scratch services). Remaining
  Phase 6 work is trace/detail polish outside the scratch-owned exact-pricing
  boundary; legacy exact pricing stays isolated until old RDO flow is retired.
- Smoke (`20240501_110934_1000x750.png`, QP 28, Best, single-thread, all current
  rdo2 luma/TU/NxN gates): `rdo2_residual_approx_pricings: 105073`,
  `rdo2_residual_exact_pricings: 276552`; valid output
  `target/rdo2-residual-scratch-pricer-smoke.bpg` at 143,046 bytes, decoded
  successfully.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `cargo test -p still265 --test residual_roundtrip -- --nocapture` → 8 green;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; Best QP 28
  single-thread smoke encode/decode green.

### Slice 6m — rdo2 residual pricing trace split (✅ 2026-06-19)

- Completed the remaining Phase 6 trace split: `SearchTrace` now records
  `rdo2_residual_approx_pricings` and `rdo2_residual_exact_pricings` directly
  from the `ResidualPricer` boundary, and `search_summary.csv` emits both
  counters beside the existing luma/TU aggregate rows.
- Old ambiguity removed: trace consumers no longer need to infer rdo2 trial
  pricing mode from global exact-estimate counts or per-block volume tables.
  Final-pricing elision remains a separate counter from slice 6k.
- Principle advanced: #9 traceability. rdo2 residual pricing is now explicit,
  scratch-backed for exact trials, and visible in both debug stats and search
  summary traces. Legacy wrapper usage remains only for old paths outside rdo2.
- Smoke (`20240501_110934_1000x750.png`, QP 28, Best, single-thread, all current
  rdo2 luma/TU/NxN gates): debug stats and
  `target/rdo2-trace-pricing-smoke/search_summary.csv` both reported
  `rdo2_residual_approx_pricings: 105150`,
  `rdo2_residual_exact_pricings: 276036`, and
  `final_residual_pricings_elided: 0`; valid output
  `target/rdo2-trace-pricing-smoke.bpg` at 143,070 bytes decoded successfully.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `cargo test -p still265 --test residual_roundtrip -- --nocapture` → 8 green;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 cargo test -p still265 --test encode_roundtrip --
  --nocapture` → 26 green; same gates for `transform_recon` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`.

### Slice 6n — chroma candidate allocation cleanup (✅ 2026-06-19)

- Continued Phase 4 allocation cleanup in the rdo2 chroma boundary:
  `SearchScratch` now owns `chroma_trace_cands`, and
  `rdo2/chroma.rs::rdo2_chroma_rd_decision` takes/clears/returns that buffer
  instead of constructing a fresh trace-candidate `Vec`.
- Replaced `code_chroma_candidate`'s per-candidate
  `Vec<(CodedBlock, CodedBlock)>` with a fixed two-entry holder. This matches
  the chroma leaf geometry contract: one chroma transform block normally, or two
  stacked blocks for 4:2:2. The final `ChromaLeafCache` still receives owned
  `CodedBlock`s for the writer-compatible cache.
- Old allocation bypassed/deleted: chroma RD candidate trace Vec construction
  and per-candidate coded-block Vec allocation.
- Allocation audit update:
  removed: rdo2 chroma trace candidate allocation and coded-block holder
  allocation.
  remaining: TU split kids still allocate through `Vec<Tt>` because
  `Tt::Split` and the writer contract own child vectors; final/materialized
  winner levels still allocate/copy into `CodedBlock` until the writer/tree
  contract changes.
- Smoke (`20240501_110934_1000x750.png`, QP 28, Best, single-thread, all current
  rdo2 luma/TU/NxN gates plus `BPG_RDO2_CHROMA=1`): trace/debug stats exercised
  the chroma path with `rdo2_chroma_cheap_cu_decisions: 14067`,
  `rdo2_chroma_exact_escalations: 2049`, and
  `rdo2_chroma_exact_changed_winner: 452`; valid output
  `target/rdo2-chroma-alloc-smoke.bpg` at 143,230 bytes decoded successfully.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1 BPG_RDO2_CHROMA=1 cargo test -p still265 --test
  encode_roundtrip -- --nocapture` → 26 green; same gates for
  `transform_recon` → 3 green; `git diff --check`;
  `cargo build --release -p bpg-tools`.

### Phase 7 sweep — full rdo2 luma stack (✅ promoted for Best; 2026-06-19)

- Ran the planned 7-image Best sweep at QP 24/28/32/36, 4:2:0, single-thread,
  using local `bpgenc_native.exe` / `bpgdec.exe` and the existing
  `scripts/headsup_quality.py` harness. Artifacts:
  `target/phase7-rdo2-luma-sweep/results.csv`,
  `summary_by_variant.csv`, `bdrate_by_image.csv`, plus custom
  `rdo2_vs_default_bdrate.csv` and `rdo2_vs_default_bdrate_by_image.csv`.
- Variants compared: default Best, `BPG_RDO2_TU=1`,
  `BPG_RDO2_TU=1 BPG_RDO2_NXN=1`, full rdo2 luma stack
  (`BPG_RDO2_TU=1 BPG_RDO2_NXN=1 BPG_RDO2_LUMA=1
  BPG_RDO2_LUMA_SCRATCH=1`), and close-margin multiplier variants m2/m10
  through `BPG_RDO2_LUMA_CLOSE_MULT`.
- Result vs default Best (positive BD-rate = worse):
  - TU only: +0.066% PSNR-Y, +0.146% MS-SSIM, +0.062% SSIM-RGB, +0.081%
    PSNR-RGB; timing effectively neutral (1.00x in this Windows run).
  - TU+NxN: same quality as TU-only in this sweep; 1.02x speedup.
  - Full rdo2 luma stack: +0.628% PSNR-Y, +0.860% MS-SSIM, +0.529% SSIM-RGB,
    +0.612% PSNR-RGB; 1.21x speedup.
  - Full rdo2 luma m2: +0.487% PSNR-Y, +0.582% MS-SSIM, +0.306% SSIM-RGB,
    +0.484% PSNR-RGB; 1.24x speedup; worst-image MS-SSIM +0.94%.
  - Full rdo2 luma m10: +0.416% PSNR-Y, +0.607% MS-SSIM, +0.394% SSIM-RGB,
    +0.413% PSNR-RGB; 1.12x speedup; worst-image MS-SSIM +1.48%.
- Promotion decision: promote the rdo2 TU + NxN + luma scratch stack default-on
  for Best using `BPG_RDO2_LUMA_CLOSE_MULT=2` as the default multiplier.
  Explicit escapes remain `BPG_RDO2_TU=0`, `BPG_RDO2_NXN=0`,
  `BPG_RDO2_LUMA=0`, `BPG_RDO2_LUMA_SCRATCH=0`, and
  `BPG_RDO2_LUMA_CLOSE_MULT=<positive-float>`. `BPG_RDO2_CHROMA` remains
  default-off.

### Slice 8a — scoped TU trial state (✅ 2026-06-19)

- Started Phase 8 from root `plan.md`: added `Encoder::with_tt_trial_flags`,
  a scoped helper for the remaining cheap/exact TU trial bridge. The helper
  saves/restores the legacy `best_tt_cheap_trial` and `best_tt_exact_trial`
  fields and asserts that a scope cannot be both cheap and exact.
- Replaced all ad hoc writes to those fields outside the helper:
  `rdo2/tu.rs::rdo2_analyze_tt`, `rdo2/luma.rs::rdo2_cu_luma_ranked`,
  `rdo2/chroma.rs::rdo2_chroma_rd_decision`, and the legacy
  `build_tt` close-escalation bridge now enter cheap/exact trial mode through
  the helper.
- Boundary of this slice: it does not remove the legacy recursive cheap screen.
  `rdo2_analyze_tt` still re-enters `build_tt` under the `in_rdo2` guard to get
  a `TtPlan`, then replays the winner exactly. The next Phase 8 slice should
  replace that screen with explicit rdo2 policy calls instead of relying on the
  legacy flag-backed path.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `cargo test -p still265 --test encode_roundtrip -- --nocapture` → 26 green;
  `cargo test -p still265 --test transform_recon -- --nocapture` → 3 green;
  `git diff --check`; `cargo build --release -p bpg-tools`; default-env Best
  QP 28 smoke encode/decode green with rdo2 luma/TU counters active and
  `BPG_RDO2_CHROMA` still off.

### Slice 8b — native rdo2 TU cheap screen (✅ promoted for Best 2026-06-19)

- The Phase 8 cutover the plan's "immediate next command" asked for: replace
  `rdo2_analyze_tt`'s legacy `build_tt` cheap re-entry with explicit rdo2 policy
  calls. New gate `BPG_RDO2_TT_NATIVE` (default off). When set, `rdo2_analyze_tt`
  screens the luma+chroma transform tree with a **native** rdo2 recursion
  (`rdo2_tt_subtree_native` / `rdo2_tt_split_native` / `rdo2_tt_leaf_cheap`) that
  costs every cheap trial block through `rdo2_eval_leaf_block` at an explicit
  `EvalKind::CheapTrial` policy, instead of setting the implicit
  `best_tt_cheap_trial` flag and re-entering the legacy recursive `build_tt`
  under the `in_rdo2` guard.
- The leaf-vs-split control flow (forced leaf/split, prefer-split,
  early-terminate, neighbour-leaf limit, close-call recording, tu-depth winner
  map) is copied verbatim from `build_tt`; only the block coder changed. Parent
  chroma (the narrow 4:2:0 `log2 == 3` case) still flows through the shared
  `build_parent_chroma_tu`, kept cheap via the **scoped** `with_tt_trial_flags`
  helper rather than a global flag — so the only remaining `best_tt_cheap_trial`
  write on this path is that one narrow, scoped call.
- New accessor `Encoder::tt_bits_full` exposes the module-private
  `estimate_tt_bits` (cat-aware) so the native screen prices subtrees without
  re-entering legacy code; mirrors the existing `tt_bits_luma`.
- **Validation — byte-identical to the legacy screen.** Best QP 24/28/32 across
  `20240501_110934`, `20240502_151356`, `20240502_184356` and chroma formats
  4:2:0 / 4:2:2 / 4:4:4 produced bit-for-bit identical `.bpg` output legacy vs
  `BPG_RDO2_TT_NATIVE=1`, all decoding cleanly with `bpgdec.exe`. The native
  cheap screen reproduces the legacy cheap-screen structure decisions exactly,
  so the cutover is faithful (no BD sweep needed to prove neutrality — identical
  bytes ⇒ identical quality). Single-image encode timing is neutral: this slice
  removes the legacy `build_tt` re-entry but still commits cheap recon and uses
  snapshot/restore, so it does not yet realise principle #1's non-committing
  speed win — that is the follow-up slice.
- Old control flow bypassed (gate on): the `in_rdo2`-guarded `build_tt`
  re-entry and its `best_tt_cheap_trial` global-flag screen for the TU structure
  decision. Legacy `build_tt`/`build_tt_split`/`build_tt_leaf` remain as the
  gate-off path and for `final_code_tt`'s exact replay.
- Principle advanced: #2 (block trials policy-explicit) for the TU structure
  screen, and Phase 8's objective "rdo2 owns transform-tree decisions".
- **Promotion (2026-06-19):** `BPG_RDO2_TT_NATIVE` is now **default-on for Best**
  (env escape `BPG_RDO2_TT_NATIVE=0` reverts to the legacy screen). Justified by
  byte-identity: default-Best output equals the `=0` escape bit-for-bit, and the
  26 round-trip tests (covering efforts/formats/bit-depths) pass with the native
  screen, so there is no quality or output risk. High-res confirmation:
  2000x1500 QP 28 SHA-256 match legacy vs native.
- Next steps: (1) delete the `in_rdo2`/`best_tt_cheap_trial` `build_tt` structure
  screen once the native default has baked (keep `final_code_tt` replay and the
  `build_tt` gate-off fallback); (2) make the native leaf-vs-split decision
  non-committing (cost both branches from `SearchScratch`, commit only the
  winner) to convert the structural cutover into a wall-clock win (principle #1)
  — the lever against the high-res gap.
- Verify: `cargo fmt`; `cargo build -p still265`;
  `BPG_RDO2_TT_NATIVE=1 cargo test -p still265 --test encode_roundtrip --test
  transform_recon -- --nocapture` → 26 + 3 green; gate-off
  `cargo test -p still265 --test encode_roundtrip --test transform_recon` →
  26 + 3 green; `git diff --check` clean; `cargo build --release -p bpg-tools`;
  smoke encode/decode green.

### Slice 8c — native TU screen drops redundant snapshots (✅ promoted for Best 2026-06-19)

- First wall-clock win from the native TU screen (principle #1/#7). Two changes
  to `rdo2_tt_subtree_native`, both byte-identical:
  1. **No `tu_depth` snapshots.** The legacy `build_tt` snapshotted and restored
     the `tu_depth` map around the leaf/split trials. Leaf coding never writes
     that map (only `record_tu_winner` does, and it sets the final value), and
     the split recursion's child writes are correct without a base restore, so
     the snapshots/restores were pure overhead. Removed from the main,
     prefer-split, and (implicitly) terminal paths.
  2. **Deferred `leaf_frame` snapshot.** The legacy path snapshotted the leaf
     reconstruction before trialling the split, so it could restore the leaf on
     a leaf win. Instead the leaf is committed; early-terminate and
     neighbour-leaf wins return it directly (no snapshot, no recode); only the
     rare "leaf still beats split after comparison" case restores `base` and
     recodes the **deterministic** leaf. This removes one frame snapshot per
     compared node at the cost of a recode only on that minority branch.
- Gate `BPG_RDO2_TT_NOSNAP` (default on for Best, `=0` restores the snapshot path
  for A/B timing). Justified by byte-identity: nosnap == snapshot == legacy
  SHA-256 across 4:2:0 and 4:2:2 at 1 MP, and nosnap == snapshot at 2000x1500;
  all decode cleanly. The 26 round-trip + 3 transform tests pass on both the
  nosnap and snapshot paths.
- **Timing (2000x1500 QP 28 Best, best-of-N):** single-thread 16.22s → 15.15s =
  **−6.6%** (the plan's diagnostic baseline); multi-thread parallel-neutral
  (3.10s → 3.06s, within noise). The single-thread win is the meaningful signal;
  the parallel path already amortises snapshot cost across workers.
- Principle advanced: #1 (separate evaluation from commit — losing leaf no
  longer materialised as a saved frame region) and #7 (remove hot-loop
  allocations — `snapshot_frame_region`/`snapshot_tu_depth_region` each alloc a
  backing `Vec`). Next: the deeper x265-style win is per-depth scratch recon
  buffers so trials never touch the shared frame at all (larger slice).
- Verify: `cargo fmt`; `cargo build -p still265`;
  `cargo test -p still265 --test encode_roundtrip --test transform_recon` →
  26 + 3 green (nosnap default); `BPG_RDO2_TT_NOSNAP=0` round-trip → 26 green;
  `git diff --check` clean; `cargo build --release -p bpg-tools`; SHA-256
  identity + decode checks green.

### Slice 8d — pool the native TU screen's snapshot buffers (✅ promoted for Best 2026-06-19)

- Continues principle #7. The native TU screen's one remaining per-node frame
  snapshot (`base_frame`) now reuses a free-list, `SearchScratch
  .frame_snapshot_pool`, instead of allocating a fresh `FrameSnapshot` (a
  `Vec<PlaneSnapshot>` plus up to three plane `data` `Vec`s) per compared node.
  `snapshot_frame_region_pooled` pops a recycled snapshot and refills its plane
  `data` buffers (clearing retains capacity); `recycle_frame_snapshot` returns
  it (pool capped at 8). LIFO matches the nested TU recursion.
- To recycle at a single site despite the screen's many early returns, the
  leaf-vs-split body was extracted into `rdo2_tt_compare_native(&base_frame,
  …)`; the entry `rdo2_tt_subtree_native` acquires `base_frame` (pooled when
  `BPG_RDO2_TT_POOL`, else fresh) and recycles it once after the compare
  returns. The rare PreferSplit `split_frame` and the `BPG_RDO2_TT_NOSNAP=0`
  `leaf_frame` are left un-pooled (rare paths).
- Gate `BPG_RDO2_TT_POOL` (default on for Best; `=0` allocates per node for
  A/B). Byte-identical: pool == no-pool SHA-256 for 4:2:0 and 4:2:2, decode
  clean; 26 round-trip + 3 transform tests pass with the pool on, and 26
  round-trip with `BPG_RDO2_TT_POOL=0`.
- **Timing: neutral.** On a heavily-loaded machine (single-thread 2000x1500
  runs ranged 14–19s for *both* settings), the min-of-N delta was −1.4% (pool
  slightly faster). An early reading of −18.9% was a load artifact — the
  un-pooled side caught high-load runs; repeated interleaving did not reproduce
  it. This is expected: pooling removes the *allocation* but not the *copy*, so
  its ceiling is well below slice 8c's full snapshot removal (−6.6%). Kept
  default-on as a low-risk, byte-identical allocation-churn reduction that
  should help most where `malloc` is the bottleneck (other allocators, the
  many-worker parallel path); it is **not** claimed as a wall-clock win on this
  machine.
- Honest caveat recorded for future tuners: this machine's load makes
  single-image timing deltas under ~5% unreliable; trust interleaved min-of-N
  and prefer the 7-image harness for anything marginal.
- Verify: `cargo fmt`; `cargo build -p still265` (no new warnings);
  `cargo test -p still265 --test encode_roundtrip --test transform_recon` →
  26 + 3 green; `BPG_RDO2_TT_POOL=0` round-trip → 26 green; `git diff --check`
  clean; `cargo build --release -p bpg-tools`; SHA-256 identity + decode checks.

### Profiling the rdo2 hot path — snapshot hypothesis disproven (2026-06-19)

The slices 8c/8d premise was that snapshot/restore traffic dominates the
high-res gap. **Direct measurement refutes this.** The `BPG_PROFILE` profiler
existed but (a) only wrapped the *legacy* `code_block`/`rdoq`/`residual` paths
that the rdo2 default bypasses, and (b) was never merged from the fork-workers
into the reported `state`, so it always printed ~0. Fixed both: added rdo2-path
inner timers (`eval_predict`/`eval_transform`/`eval_quant_rdoq`/`eval_recon`/
`eval_residual_price`, plus `rough_search` for `decide_luma_modes`) and a
worker→state prof merge in `write.rs`. All gated by `BPG_PROFILE`, inert
otherwise.

Measured breakdown (2000x1500, Best, QP 28, single-thread, 14.7s encode):

| bucket | ms | % |
|---|---:|---:|
| quant + RDOQ (rdo2 3160 + legacy 1637) | 4797 | **33%** |
| legacy `code_block` total (chroma RD / final / PartNxN) | 3855 | 26% |
| forward+inverse transform (rdo2) | 2162 | 15% |
| residual pricing (rdo2 1037 + legacy 1110) | 2147 | 15% |
| rough SATD mode search (35 modes/PU) | 1769 | 12% |
| **snapshot + restore** | **191** | **1.3%** |

**Consequence:** the per-depth-scratch-recon / `m_rqt` rewrite (and the snapshot
pooling of 8d) target a 1.3% slice — they will not move the high-res gap. The
real cost is the **volume of block-eval kernels**: RDOQ (33%), transforms (15%),
residual pricing (15%), and the rough SATD search (12%). The `code_block_volume`
trace shows RDOQ runs on ~98% of blocks (`rdoq_blocks` 254K ≈ `code_blocks`
259K), split `final_code` 140K / `chroma_trial` 84K / `tu_decision` 30K — i.e.
the staged "cheap screen, RDOQ only on close calls" goal is **not** realised:
chroma mode selection still RDOQ-codes every trial (legacy path, `rdo2_chroma`
off), and final replay RDOQ-codes the winner. Real levers, in rough order:
1. Cut trial RDOQ + exact residual pricing to *truly* final + close-call only —
   the chroma decision (84K trial RDOQ) is the biggest unconverted offender.
2. Faster kernels: the rough SATD search horizontal-all-angles is still scalar
   (`docs/remaining-simd-coverage-2026-06-18.md`), forward DST4 is scalar; these
   are 12%+ of time with known SIMD wins.
3. Stop coding the winner twice (cheap screen + exact final replay): ~85K blocks
   are priced both approximately and exactly (344K pricings over 259K blocks).

This redirects Phase 8's "remaining" list: the m_rqt item is **de-prioritised**
(1.3% ceiling); the kernel-volume and chroma-RDOQ items are promoted.

### Current timing vs C bpgenc (`-m9`, single-thread, 2026-06-19)

- Re-measured the default promoted Best stack (rdo2 TU + NxN + luma scratch,
  `BPG_RDO2_LUMA_CLOSE_MULT=2`, all default-on) against `bpgenc_native.exe -m9`
  with one-CPU affinity, via `bpg-highres-compare`. Synthetic
  `20240501_110934` at three sizes
  (`target/highres-compare-best-m9-rdo2-promoted/summary.md`):

  | size | C total | Rust total | Rust/C |
  |---:|---:|---:|---:|
  | 1000x750  | 5.039s | 4.357s  | **0.86x** (Rust faster) |
  | 2000x1500 | 8.566s | 15.862s | 1.85x |
  | 4000x3000 | 21.673s | 64.266s | 2.97x |

- Interpretation: at ~1 MP single-thread Rust Best is at parity / slightly
  faster than C `-m9`; the **widening high-resolution gap persists** (~1.85x at
  3 MP, ~2.97x at 12 MP), consistent with `docs/handoff-memory-usage-findings.md`
  (build/RD dominates; serial write and SAO are not the bottleneck). Caveat:
  the C `-m9` numbers in this run are ~2x higher than the older handoff table
  (1.969/5.393/15.920s) — likely concurrent machine load and/or a different
  `bpgenc` binary, so treat the **ratios** (load-affecting both encoders) as the
  reliable signal, not the absolute C seconds. The rdo2 promotion did not move
  the absolute 1 MP Rust time materially (~4.35s here vs ~4.38s pre-rdo2
  default); its measured 1.24x win was on the 7-image `headsup_quality.py`
  harness, not this single synthetic image. The high-res scaling gap remains the
  dominant performance story and is the next speed target after the Phase 8
  structural cutover.

### Slice 8e — chroma RD cost-only scratch screen (✅ gated 2026-06-19)

- Implemented the revised measured lever: `BPG_RDO2_CHROMA_SCRATCH` routes
  one-block chroma RD candidates through `rdo2_eval_leaf_block` instead of the
  legacy `trial_code_block`/`code_chroma_candidate` path when
  `BPG_RDO2_CHROMA=1` and `Effort::Best`.
- Scope is intentionally narrow: `count == 1` chroma geometry (4:2:0 and 4:4:4)
  uses the scratch cost-only path; stacked 4:2:2 (`count == 2`) increments a
  fallback counter and stays on the existing materializing path because the
  second stacked block can depend on the first block's reconstruction.
- The scratch screen evaluates Cb+Cr with `EvalKind::CheapTrial` (plain quant,
  approximate bits, non-committing), exact-rechecks close top-two decisions with
  `EvalKind::ExactTrial`, and returns `leaf_cache: None`. Final chroma output is
  still materialized exactly by `final_code_tt`, preserving the writer contract.
- Added counters:
  `rdo2_chroma_scratch_candidates`,
  `rdo2_chroma_scratch_cheap_evals`,
  `rdo2_chroma_scratch_exact_evals`,
  `rdo2_chroma_scratch_exact_escalations`,
  `rdo2_chroma_scratch_changed_winner`,
  `rdo2_chroma_scratch_legacy_evals_skipped`, and
  `rdo2_chroma_scratch_stacked_fallbacks`. The high-res harness now exports
  these fields in `results.csv`.
- Smoke (`20240501_110934`, 2000x1500, QP 28, Best, single-thread,
  `BPG_PROFILE=1`, `BPG_TRACE_SEARCH=...`):
  - default promoted Best: `search_summary.csv` reported `code_blocks=953534`,
    `rdoq_blocks=934774`; legacy profile bucket `code_block=3788 ms`, `rdoq=1583 ms`.
  - old chroma gate (`BPG_RDO2_CHROMA=1 BPG_RDO2_CHROMA_SCRATCH=0`):
    `code_blocks=1040974`, legacy-traced `rdoq_blocks=701488`;
    `chroma_trial` RDOQ remained 30,256 blocks (Cb+Cr 4x4/8x8/16x16).
  - scratch gate (`BPG_RDO2_CHROMA=1 BPG_RDO2_CHROMA_SCRATCH=1`):
    `code_blocks=689948`, legacy-traced `rdoq_blocks=671232`; no
    `chroma_trial` RDOQ rows; counters: 160,385 scratch candidates, 320,770
    cheap eval blocks, 30,256 exact eval blocks, 7,564 exact escalations, 1,161
    changed winners, 351,026 legacy eval blocks skipped, 0 stacked fallbacks.
    The 30,256 exact eval blocks match the old gate's remaining chroma-trial
    RDOQ count: this slice moves close-call RDOQ to rdo2 instead of eliminating
    it.
- Timing caveat: the rerun used to export the new CSV counters hit heavy machine
  load (20.7s encode), while earlier interleaved profile smokes were 14.5s
  default, 13.49s old gate, 13.57s scratch. Treat the structural trace deltas as
  reliable; require the 7-image harness before default-on promotion.
- Verify: `cargo fmt`; `cargo build -p still265`; default
  `encode_roundtrip` (26) + `transform_recon` (3) green;
  `BPG_RDO2_CHROMA=1 BPG_RDO2_CHROMA_SCRATCH=1` same tests green;
  `git diff --check`; `cargo build --release -p bpg-tools`; 2000x1500 smoke
  encodes decoded through the high-res harness.

### Slice 9a — legacy leakage ledger + native parent chroma cut (✅ 2026-06-19)

- Renamed the old monolithic module to `encoder/rdo_legacy.rs` and wired
  `encoder/mod.rs`, `rdo2/luma.rs`, and `rdo2/chroma.rs` to reference it
  explicitly. This keeps the old code available as a reference while making
  remaining Best call-graph leaks visible.
- Added `EncodeStats::legacy` and `BPG_RDO2_REPORT_LEGACY=1`. The report is
  Best-only and counts high-level old-path entry points plus the block-coding
  funnel (`trial_code_block`, `estimate_block`, `code_block_internal`) so each
  cut can prove work moved out of rdo1.
- Removed the scoped legacy `build_parent_chroma_tu` call from native rdo2 TT
  cheap screening. `rdo2_parent_chroma_tu_cheap` now uses
  `rdo2_eval_leaf_block(EvalKind::CheapTrial)` for the 4:2:0 parent Cb/Cr
  blocks, matching the existing scope (`cat == 1`, `log2_size == 3`) without
  reviving snapshot/restore work.
- Baseline leakage smoke (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`) before the finalizer
  move: total 662,590; `legacy_code_block_internal=247,404`;
  `legacy_estimate_block=116,418`; `legacy_trial_code_block=116,418`;
  `legacy_final_code_tt=22,468`; `legacy_build_tt_leaf=15,116`;
  `legacy_build_parent_chroma_tu=15,278`.
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; chroma scratch-gated same tests green.

### Slice 9b — rdo2 final winner replay (✅ 2026-06-19)

- Added `encoder/rdo2/finalize.rs` as the rdo2-owned writer-compatible replay
  boundary. It materializes `Tt`/`CuNode` plans for the existing writer, but
  final transform blocks are coded through
  `rdo2_eval_leaf_block(EvalKind::Final)` instead of legacy
  `final_code_block`/`code_block_internal`.
- Retargeted rdo2 TU final replay from `final_code_tt` to
  `rdo2_final_code_tt`; retargeted the rdo2 luma path from
  `decide_and_final_code_tt` to `rdo2_analyze_tt` when `BPG_RDO2_TU` is active;
  and retargeted Best CU final replay to `rdo2_final_code_cu` when rdo2 TU is
  active. Legacy wrappers remain for non-migrated paths.
- The rdo2 finalizer also owns final parent-chroma replay for the same narrow
  4:2:0 parent-TU case, avoiding the legacy `build_parent_chroma_tu` final
  helper during rdo2 replay.
- Leakage smoke after the cut (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`): total 475,535;
  `legacy_final_code_tt` is absent; `legacy_build_tt_leaf` drops to 297;
  `legacy_code_block_internal` drops to 157,515; `legacy_estimate_block` and
  `legacy_trial_code_block` drop to 88,215; `legacy_build_parent_chroma_tu`
  drops to 11,598. The remaining big leaks are now CU ownership, luma rough
  search, chroma mode decision, luma leaf-screen materialization, and PartNxN /
  chroma fallback materialization.
- Smoke output decoded successfully:
  `target/rdo2-legacy-smoke/rdo2-finalize-report.bpg` →
  `target/rdo2-legacy-smoke/rdo2-finalize-report.png` (1000x750).
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; `BPG_RDO2_CHROMA=1
  BPG_RDO2_CHROMA_SCRATCH=1` same tests green.

### Slice 9c — rdo2 rough luma ownership (✅ 2026-06-19)

- Added `encoder/rdo2/rough.rs` and moved the rough luma candidate-list owner
  (`decide_luma_modes`) out of `rdo_legacy.rs`. The scoring helpers remain
  shared for this slice, so SATD costs, pruning, batched angular prediction, and
  candidate ordering are unchanged.
- This is an ownership cut, not a speed claim: `rough_satd_search` remains the
  same work bucket, but the default Best call graph no longer enters rdo1 for
  rough luma mode selection.
- Leakage smoke after the move (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`): total 414,040;
  `legacy_decide_luma_modes` is absent (was 61,495 after 9b);
  `legacy_code_block_internal=157,515`; `legacy_trial_code_block=88,215`;
  `legacy_estimate_block=88,215`; `legacy_decide_chroma_mode=26,845`;
  `legacy_build_parent_chroma_tu=11,598`;
  `legacy_build_luma_tt_leaf=10,467`. Remaining high-value ownership cuts are
  CU decision/build ownership, chroma mode ownership, luma leaf-screen
  materialization, and the PartNxN/chroma fallback materializers still using
  legacy block coding.
- Smoke output decoded successfully:
  `target/rdo2-legacy-smoke/rdo2-rough-report.bpg` →
  `target/rdo2-legacy-smoke/rdo2-rough-report.png` (1000x750).
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; `git diff --check` clean.

### Slice 9d — rdo2 chroma mode ownership (✅ 2026-06-19)

- Moved the chroma mode front half (`chroma_mode_from_idx` and
  `decide_chroma_mode`) from `rdo_legacy.rs` into `encoder/rdo2/chroma.rs`.
  The module now owns the full chroma decision boundary: rough SATD screen,
  candidate narrowing, gated cheap/exact RD decision, and cache/scratch return.
- Behaviour is intended unchanged: the moved code keeps the same SATD scoring,
  `best2_chroma_gate`/`best2_chroma_protect` policy, winner-rank accounting,
  and delegation to `rdo2_chroma_rd_decision`.
- Leakage smoke after the move (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`): total 387,195;
  `legacy_decide_chroma_mode` is absent (was 26,845 after 9c);
  `legacy_code_block_internal=157,515`; `legacy_trial_code_block=88,215`;
  `legacy_estimate_block=88,215`; `legacy_build_parent_chroma_tu=11,598`;
  `legacy_build_luma_tt_leaf=10,467`.
- Smoke output decoded successfully:
  `target/rdo2-legacy-smoke/rdo2-chroma-owner-report.bpg` →
  `target/rdo2-legacy-smoke/rdo2-chroma-owner-report.png` (1000x750).
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; `BPG_RDO2_CHROMA=1
  BPG_RDO2_CHROMA_SCRATCH=1` same tests green.

### Slice 9e — rdo2 luma leaf-screen fallback (✅ 2026-06-19)

- Added rdo2 luma-only materializing leaf-screen helpers in `rdo2/luma.rs`.
  They preserve the current trial policy (`CheapTrial`, `ExactTrial`, or
  `Final`, committed) but code leaves through `rdo2_eval_leaf_block` instead of
  the legacy `build_luma_tt_leaf_screen` / `trial_code_block` funnel.
- Removed the now-unused legacy `build_luma_tt_leaf_screen` wrapper. The
  ordinary non-Best `build_luma_tt_leaf` remains for non-migrated luma-TU paths.
- Leakage smoke after the move (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`): total 362,772;
  `legacy_build_luma_tt_leaf` is absent (was 10,467 after 9d);
  `legacy_code_block_internal=152,863`; `legacy_trial_code_block=83,563`;
  `legacy_estimate_block=83,563`; `legacy_build_parent_chroma_tu=11,598`.
  Remaining visible default-Best rdo1 ownership is mainly CU build/decision
  scaffolding and the materializing chroma/PartNxN fallback block-coding funnel.
- Smoke output decoded successfully:
  `target/rdo2-legacy-smoke/rdo2-luma-screen-report.bpg` →
  `target/rdo2-legacy-smoke/rdo2-luma-screen-report.png` (1000x750).
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; `git diff --check` clean.

### Slice 9f — rdo2 CU recursion ownership (✅ 2026-06-19)

- Added `encoder/rdo2/cu.rs` and moved the CU recursion owner out of
  `rdo_legacy.rs`: `build_cu`, `decide_cu`, `build_cu_kids`, and
  `cu_trial_result`. This keeps the same leaf-vs-split decisions, bound abort,
  AQ QP setup/restore, and final replay behaviour, but default Best no longer
  enters the legacy module for CU scaffolding.
- Kept transitional dependencies explicit: CU leaves, PartNxN construction, and
  some block bit-estimation helpers still call legacy-compatible services. The
  remaining report now isolates those block/materialization paths instead of
  counting every CU recursion step as rdo1 ownership.
- Leakage smoke after the move (`20240501_110934_1000x750.png`, QP 28, Best,
  single-thread, `BPG_PROFILE=1 BPG_RDO2_REPORT_LEGACY=1`): total 331,884;
  `legacy_build_cu` and `legacy_decide_cu` are absent (15,444 each after 9e);
  remaining buckets: `legacy_code_block_internal=152,863`,
  `legacy_trial_code_block=83,563`, `legacy_estimate_block=83,563`,
  `legacy_build_parent_chroma_tu=11,598`, `legacy_build_tt_leaf=297`.
- Smoke output decoded successfully:
  `target/rdo2-legacy-smoke/rdo2-cu-owner-report.bpg` →
  `target/rdo2-legacy-smoke/rdo2-cu-owner-report.png` (1000x750).
- Verify: `cargo fmt`; `cargo build -p still265`; default `encode_roundtrip`
  (26) + `transform_recon` (3) green; `BPG_RDO2_CHROMA=1
  BPG_RDO2_CHROMA_SCRATCH=1` same tests green.

### Slice 9g — Best-only rdo2 RDO path hardening (✅ 2026-06-20)

- Hardened `Effort::Best` so the important RDO gates run through rdo2 even
  when legacy environment toggles are set off: TU, luma screen/scratch, chroma
  RD/scratch, native TT, no-snapshot replay, TT pooling, and PartNxN are all
  forced on for Best.
- Moved the remaining CU leaf and PartNxN decision/build ownership into rdo2,
  including the 8x8 PartNxN path, rough PartNxN precheck, and native final
  replay through `rdo2_final_code_tt`.
- Added rdo2-owned syntax cost helpers in `encoder/rdo2/cost.rs` and switched
  rdo2 CU/luma/chroma code to those estimators instead of reaching back into
  `rdo_legacy.rs`.
- Removed the old legacy CU final replay helpers and the now-dead luma-only TT
  builder/final replay block (`build_luma_tt*`, `final_code_tt*`, `decide_tt`,
  `decide_and_final_code_tt`). The rdo2 luma candidate screen now uses scratch
  block evaluation and exact rdo2 rechecks rather than falling back to legacy
  materializing luma trials.
- Chroma candidate scoring now stays in rdo2 scratch evaluation for both the
  cheap screen and exact recheck; the stale `ChromaDecision` leaf-cache return
  path was removed because rdo2 final TT no longer consumes it.
- Verify: `cargo fmt -p still265`; `cargo check -p still265`;
  `cargo build -p still265` clean. No comparison or round-trip tests were run
  for this slice.

### Slice 9h — rdo2 metrics ownership + legacy Best guards (✅ 2026-06-20)

- Added `encoder/rdo2/metrics.rs` and moved generic hot-path RDO primitives out
  of `rdo_legacy.rs`: rough SATD scoring, RD/lambda costs, distortion
  measurement, TU split predicates, TT bit wrappers, CU early termination, and
  CU/TT plan conversion now live under rdo2.
- Reused rdo2's syntax estimator for `tt_bits_luma` / `tt_bits_full` by making
  the existing `rdo2::cost::estimate_tt_bits` visible within the encoder.
- Removed the stale legacy `estimate_intra_luma_mode_bits` duplicate; rdo2's
  cost module is now the sole owner of that luma-mode syntax estimate.
- Replaced the old legacy Best counters in the remaining block/TT legacy
  implementation with hard guards. If `Effort::Best` enters
  `code_block_internal`, `estimate_block`, `trial_code_block`,
  `build_parent_chroma_tu`, `build_tt_leaf`, or `build_tt`, it now trips the
  explicit "Best must not enter legacy RDO" invariant instead of silently doing
  important work in rdo1.
- Verify: `cargo fmt -p still265`; `cargo check -p still265`;
  `cargo build -p still265` clean. No comparison or round-trip tests were run
  for this slice.

### Slice 9i — fix round-trip regressions; all tiers share rdo2 CU path (⚠ 2026-06-20)

- Slice 9g/9h were compile-only; the round-trip suite was actually **red**. Root
  cause: `build_cu`/`decide_cu`/`build_cu_leaf`/`rdo2_cu_luma_ranked` are now the
  shared CU path for **every** effort tier (not just Best), so their rdo2-only
  assumptions panicked the whole ladder.
- Fixed three migration panics:
  - `rdo2_cu_luma_ranked` panicked (`unsupported luma candidate geometry`) for CUs
    larger than the max TB (64x64): added `rdo2_eval_luma_candidate_subtree`, which
    screens a large CU's luma candidate by materialising its forced luma transform
    subtree (`rdo2_luma_subtree`, now `pub(in crate::encoder)`) and restoring; the
    scratch-only exact recheck is skipped for those CUs.
  - `rdo2_chroma_rd_decision` panicked for non-Best tiers because `scratch` was
    gated on the Best-only `rdo2_chroma_scratch` flag. Decoupled: the scratch
    evaluator is the shared chroma coding primitive for all tiers (gated only on
    geometry, `count <= 2`); the `cheap` Best screen stays a separate Best-only
    speed trick.
- Fixed a correctness bug in `rdo2_final_code_cu_inner`'s leaf path: it emitted the
  **screen-time** `mpm` carried in the plan, but the decoder derives MPM from final
  neighbour modes. Now recomputes MPM from final neighbours (the PartNxN final path
  already did this).
- Result: `encode_roundtrip` 25/26 green (was 15/26). Other test binaries green.
- **Open blocker:** `synthetic_pattern_matrix` (Balanced, smooth gradient, partial
  CTB) still fails with a CABAC desync in the rdo2 final-replay path. All syntax
  through `cbf_luma` decodes in sync (mode/MPM/chroma/pred verified equal), but the
  luma residual desyncs (encoder writes levels `[9,-5]`, decoder reads `[1,1,1,1,1]`).
  Effort-independent; the distinguishing factor vs Best is that Best uses the
  committed screen `winner` directly (`best2_cu_reuse`) while non-Best runs the
  final replay. Needs an encoder-side syntax trace to diff against the decoder's
  `se_trace`. See memory `rdo2-boundary-cabac-desync`.
- Timing (the migration's Best path): 2000x1500, Best, QP 28, single-thread Rust =
  **12.02s encode** (20240501_110934), vs the last-known ~14.5-14.7s baseline — a
  ~17% speedup from the rdo2 work.
- Verify: `cargo build -p still265 -p bpg-hevc-decode` clean; `cargo test -p still265`
  = only `synthetic_pattern_matrix` fails; `bpg-highres-compare --effort best
  --sizes 2000x1500 --qp 28 --rust-single-thread --skip-c`.

### Slice 10 — collapse non-Best tiers onto Best; SIMD forward quantize (✅ 2026-06-20)

- **Effort ladder collapse.** Per the new direction (rebuild the ladder on a tuned
  rdo2 `Best`), `encode_with_stats` now coerces `Fastest`/`Fast`/`Balanced`/`Good`
  to `Effort::Best` at entry (rebinds `config` to a clone with `effort = Best`).
  Only `Best` (rdo2) and the byte-exact `Placebo`/`Reference` keep distinct
  behaviour; the non-Best budgets/methods are now dead code. This makes the whole
  still265 suite green — including `synthetic_pattern_matrix`, because Best uses the
  committed screen `winner` directly (`best2_cu_reuse`) and never enters the
  non-Best final-replay path where the slice-9i CABAC desync lives. That confirms
  the desync was isolated to the now-unused non-Best replay; it is no longer on any
  live path.
- **Profiling baseline** (1000x750, Best, QP 28, serial single-thread): quant+rdoq
  1131ms, fwd_transform 577ms, residual_price 368ms, rough_satd 279ms,
  exact_residual_bits 273ms, predict 132ms; 816k forward transforms, 316k RDOQ
  blocks, PartNxN = 73% of CU trials.
- **SIMD forward quantize.** `transform::quantize_into` now dispatches through a new
  `primitives::quantize` kernel with a `wide` portable-SIMD backend
  (`wide_simd::quantize`), mirroring the existing `dequantize`. All quant constants
  fit i32 (max `|c|*scale + add` ~9e8 < 2.1e9), so the SIMD path is bit-identical to
  the scalar reference — enforced by `wide_simd::tests::quantize_matches_scalar`
  over every (bd, log2, qp, len). Result: quant+rdoq 1131→1029ms, overall encode
  2.948→2.851s (~3.3%) on the controlled 1000x750 serial profile; remaining
  quant+rdoq is RDOQ (sequential, harder to SIMD). encode_roundtrip green through
  the SIMD path (`--features wide-simd`).
- Next speed levers (bigger, need quality validation): cut redundant forward
  transforms (winner is transformed at cheap+exact+final with identical residual),
  reduce RDOQ/exact-bit-pricing call counts, or SIMD the scalar 4x4 DST (PartNxN).
