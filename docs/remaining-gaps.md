# Remaining gaps / known issues

This file tracks known, intentionally-deferred gaps and discrepancies found
during development that are out of scope for the change that found them.

## `still265` paper-driven search pruning (effort tiers) + parallelism finding

Two encoder-side, bitstream-neutral fast-mode-decision techniques from the
literature are applied to the fast tiers (the chosen modes/splits are always
coded validly, so decode is unaffected; only RD-optimality / output size
changes):

- **CU-split early termination** (liang2013 §3.2, `Encoder::cu_early_terminate`):
  skip the sub-tree split RD trial when the coded CU leaf has zero residual
  (`Balanced`) or is additionally cheap/smooth (`Fast`/`Fastest`).
- **Global angular mode exclusion** (heindel2016 §3.1,
  `Encoder::prune_angular_modes`): drop the angular rough-mode sweep on
  homogeneous luma blocks (source-block variance threshold), keeping Planar/DC
  + MPM.

Tier scale: `Fastest` aggressive, `Fast` moderate, `Balanced` light
(near-lossless), `Good`/`Best` off (quality-preserving), and
`Placebo`/`Reference` off (byte-exact reference, byte-identical to the
pre-pruning encoder). Averaged over the 16-image 1024x768 photo `test-set`
(4:2:0): `fastest` -30..-50% time / +0.3..+1.8% size, `fast` -20..-40% /
+0.1..+0.6%, `balanced` -10% / ~0%.
The current `Fastest` has since moved to a more aggressive Kvazaar-like profile
(sparse progressive RMD, DM-only chroma, approximate bits, fixed-leaf small
TUs); see `docs/effort-ladder-reshape.md` for the latest sample numbers.

Two papers from the same review are **not actionable** for stock-`bpgdec`-
compatible BPG: chen2016 (iterative-filtering intra prediction adds a new mode +
flag the decoder must understand) and kim2008 (fast intra-skip in *inter*
frames — BPG is all-intra still). strutz2017 (adaptive colour-space selection)
is deferred (needs a YCgCo forward+inverse transform on both sides; gains are
mostly screen-content, not archival photos).

**Lossless speed work — findings:**

- *Compact nonzero-coefficient list (P1):* not pursued. The common RDOQ path
  (`rdoq::rdoq_single_scan`, used by every tier except `Placebo`/`Reference`) is inherently
  O(block area) — each scan position needs a level decision, so there are no
  zeros to skip. The only skippable-zero loop is `refine_levels_rdoq_limited`
  (exact-greedy path only), whose own cost is dominated by the CABAC estimate,
  not the dense distortion sum. Payoff is negligible.

- *CTU / wavefront parallelism (P3): implemented for the `Placebo` tier, and now
  reused as the default for `Best`* (the `BEST_PARALLEL` template; revert with
  `BPG_BEST2_PARALLEL=0`). The frozen-slice-init wavefront below was first built
  for `Placebo`; default `Best` now runs the same path over the Best search
  budget (~3× faster, ~0.1 dB / ~0.4% efficiency cost from the frozen context —
  see `docs/quality-speed-size-targets.md` Phase 5). The
  encoder is single-pass interleaved (`write_coding_quadtree` runs `build_cu`
  then `write_cu` per CTU, and `build_cu` prices RD against the running CABAC
  context `write_cu` evolves), so a bit-identical parallel encode is impossible.
  The current `Placebo` tier therefore splits from the exact running-context path:
  it runs a
  CTU-wavefront-parallel analysis (`encode_slice_data_parallel`) — diagonal
  steps `t = cx + 2*cy`, each step's CTUs built in parallel by worker
  `Encoder`s that clone the global reconstruction, with a frozen slice-init RD
  context — followed by a serial CABAC write. RD priced against the frozen
  context is what decouples analysis from entropy state; the result is
  bit-identical to the same analysis run single-threaded (`BPG_ENC_THREADS=1`,
  the determinism oracle) and differs from `Reference` (the serial exact path)
  only by the frozen-vs-running context (~0.1% size). Measured ~3x faster than
  `Reference` on a 12-CTU-row 1024x768 image. The implementation is **safe Rust**
  (no `unsafe`): `std::thread::scope` with a shared read-only `&global` during
  the parallel section and per-worker owned frames merged serially after the
  barrier. Each worker keeps a persistent frame that mirrors the global
  reconstruction: every built CTU's region is replayed into all worker frames
  after each step's barrier, so workers always see correct earlier-step
  neighbours with `O(n_ctus * ctb^2 * workers)` sync (not a per-step full-frame
  clone), which is what lets it scale to 4K-7K. Measured: 1024x768 ~3x vs
  `Reference`; 4096x3072 ~5x (42.0s -> 8.4s, and ~15% faster than the earlier
  per-step-clone version, a gap that widens with resolution). Limitations:
  speedup is bounded by the wavefront ramp and the CTU row count (`n_threads`
  capped at `ctbs_y`); each worker holds a full frame (`~n_threads * frame`
  RAM); `Placebo` + `--sao` falls back to the serial path.

- *Accumulated-cost (branch-and-bound) CU-split termination (lemmetti2018 §III):
  implemented for **all** tiers, exact.* In `Encoder::build_cu` the four split
  children are now built one at a time, accumulating each child's distortion and
  full node-bit estimate plus the always-coded non-negative parent
  `split_cu_flag` bits. Both child halves are exactly additive over the quadrants
  (`distortion_tt_region` is per-pixel SSE over disjoint regions;
  `estimate_cu_kids_bits` is by construction the ordered sum of the per-child
  `estimate_cu_node_bits` priced against the same frozen context), so the running
  accumulated cost is a monotone lower bound on `split_cost`. Once it reaches
  the already-known leaf cost, the split provably cannot win and is
  abandoned without building/pricing the remaining children. The decision is
  identical to full evaluation (a non-aborted split reuses the accumulated
  distortion/bits to form the bit-identical `split_cost`; an aborted split would
  have lost the `split_cost < leaf_cost` test anyway), so output is
  **byte-identical across the ladder** (verified before the later
  Best/Placebo/Reference rename vs the pre-change binary over 4 photos ×
  {420,422,444} × {fastest,fast,balanced,good,best,placebo}).
  **Finding: exact, but marginal on photographic input.** Before the larger
  trial-reuse work, on a 1024x768 photo (qp30, 420, placebo) it fired 1414 times
  and removed ~3% of CU-split RD trials (`cu_trials` 48960 -> 47493,
  `code_block_calls` -2.1%), but **wall-clock was unchanged** and the dominant
  `residual_bit_estimates` counter barely moved (-0.05%). With TT+CU winner
  reuse and the parent split flag included in the bound, the qp28 photo check
  remains byte-identical and lands at old-`placebo`/current-`reference` 805,959
  and old-`best`/current-`placebo` 804,000
  `code_block_calls`. The reason the remaining bound yield is small is
  structural: the bound only fires when the leaf cost is low, i.e. on smooth CUs
  whose skipped children are themselves cheap (little residual to estimate),
  while the encoder's cost is dominated by residual bit-estimation / RDOQ on the
  *textured* CUs the prune never touches. Kept because it is exact, free (it
  replaces the single whole-region distortion/bits pass with an equivalent
  per-child accumulation), and genuinely wires in the paper's method — but it is
  not a meaningful speed lever here. A *lossy* variant
  (looser bound, or bounding against the running best across siblings before the
  leaf is fully priced) could prune more but would change output and is out of
  scope for the exact tiers.

## `still265` source-derived preanalysis steering (adaptive search budget)

`crates/still265/src/preanalysis.rs` builds a cheap, read-only, per-32x32-cell
structure map from the source picture (luma variance, Sobel edge density,
4-bin gradient-orientation entropy + dominance, a weak-edge high-pass noise
proxy, and per-cell chroma variance), classifies each cell
(`Flat`/`Gradient`/`DirectionalEdge`/`TextLike`/`Texture`/`Noisy`/
`ChromaCritical`), and resolves a `SearchPolicy` per coding block. It is built
once (`encode_with_stats`), shared with the parallel `Placebo` workers by `Arc`,
and consulted read-only at four hooks: `prune_angular_modes` (force/guard the
local verdict), `cu_early_terminate` (guard structured CUs), and the
luma/chroma RD-candidate counts. The features mirror the *semantics* of the
offline Python profiler `psnrtune-dev/psnrtune_characteristics.py` (no
OpenCV/CC/Canny here), which serves as the offline oracle for tuning the
thresholds. The map never changes syntax — only which encoder-side shortcuts
are taken — so decode validity is preserved (verified: stock `bpgdec` accepts
all steered outputs).

**Tier scope (after measurement).** Search steering is active for
`Fast`/`Balanced`/`Good`; `Fastest` uses the map only for AQ, while
`Best`/`Placebo`/`Reference` early-return an inert policy and skip the feature
pass entirely. `Fastest` was dropped from steering because its local pruning is
already maximal: steering produced byte-identical output with only added
overhead. Candidate *expansion* (extra RD trials on structured cells) is a
quality lever reserved for `Good`; the speed-leaning tiers only
prune/guard/reduce.

**Finding: a modest, content-dependent win on photos; the larger gains need
mixed content.** On an 8-image 1024x768 camera-photo subset (qp28, 4:2:0, vs
the pre-change binary): `balanced` -6.4% time / +0.03% size, `good` -5.9% /
+0.08%, `fast` ~neutral (+0.0% / +0.06%), `fastest` byte-identical. The wins
come from forcing angular pruning on `Gradient`/`Flat` cells (which `Balanced`'s
light local heuristic leaves unpruned). The ceiling is limited *on this corpus*
because camera photos are almost entirely "structured" at 32x32 (≈93% of cells
`Texture`/`DirectionalEdge`/`TextLike`), so there is little flat area to prune
and the extra-search levers gave no size benefit (hence expansion was narrowed
to the rare `TextLike`/`ChromaCritical` cells). The protective guards
(don't-prune-angular / don't-early-terminate on edge/text cells) are expected to
pay off mainly on **screenshot/scan/line-art** content, which the photo-only
test-set does not exercise — that is the next thing to measure. Possible future
work: finer-than-32x32 features, a smarter `DirectionalEdge`/`Texture` split,
and (deferred, needs syntax work) transform-skip on `TextLike` cells and local
QP.

## `still265` adaptive quantization (default-on) + monochrome gap

Per-CU adaptive quantization (`cu_qp_delta`, 32x32 quantization groups) is now
**default-on for the speed/size-oriented effort tiers** (`Fastest`/`Fast`/
`Balanced`/`Good`); the high-quality uniform-QP tiers
(`Best`/`Placebo`/`Reference`) stay AQ-free. The QP offset
is importance-weighted from the preanalysis map (low-importance/flat/noisy
regions raised the most, `+1..+8`), concentrating quantization loss where the eye
is least likely to fix. This replaced the earlier experimental `Lossier` dial:
there is no lossless mode, so AQ is simply part of the lossy path. The single
gate is `still265::aq_active` (`!is_reference_tier && chroma != Gray`), consulted
by both the PPS writer and the encoder so `cu_qp_delta_enabled_flag` and the
per-CU QP plan can never disagree. Verified by reconstruction-equality round-trip
(encoder recon == in-tree decoder) across 4:2:0/4:2:2/4:4:4 × smooth/textured ×
CTU-aligned/not × deblock/SAO × 8/10/12-bit.

**Known gap — monochrome (4:0:0) is excluded from AQ.** In monochrome there is no
chroma CBF to trigger an early `cu_qp_delta`, so on flat luma the delta is
deferred past the first TU and the QG QP *prediction* (`Encoder::aq_cu_begin`,
mirroring the decoder's `decode_quantization_parameters`) diverges from the
decoder on **non-CTU-aligned** pictures — the `gray_non_ctu_aligned` round-trip
(67x53) fails with AQ on, while monochrome 64x64 and all of 4:2:0/4:2:2/4:4:4 at
the same odd dimensions pass. Monochrome therefore falls back to uniform QP via
the `aq_active` gate (still a correct encode, just no adaptive QP). Re-enabling it
needs the monochrome QG-prediction / per-TU `store_block_qp` path reconciled with
the decoder at partial-CTB boundaries (then drop the `chroma != Gray` clause and
the `pps_aq_off_for_monochrome` test). Monochrome BPG is rare, so this is
deferred.

**Known gap — cropped-padding recon divergence at the display boundary
(monochrome, decision-dependent).** Separately from the AQ gap above (this
happens with AQ *off*), the encoder prunes the CU coding structure at the
**display** width/height (`build_cu_kids`/`build_cu_inner`/`write_cu` all gate on
`display_width`/`display_height`), so a non-CTU-aligned picture's coded-but-
cropped padding strip (`x ∈ [display_width, coded_width)`, likewise for height)
is reconstructed inconsistently with the decoder when a boundary CU *splits* in a
way that prunes the child covering that strip. This is **display-invisible** —
the conformance window crops it — and only surfaced because a search-effort
decision shift (the `FastRd` context-aware residual-bit model) steered a
boundary monochrome CU into a splitting structure; the `gray_non_ctu_aligned`
round-trip's *display region* stays bit-exact, while 4 padding columns of the
bottom-right CTB differ by 2–3. The recon-equality test therefore asserts the
display rectangle for monochrome (`assert_display_eq`); color (4:2:0/4:2:2/4:4:4)
still asserts the full CTB-aligned plane. A real fix would prune the coding
structure at the **coded** picture size (`frame.width`/`frame.height`) so the
padding strip is coded and reconstructed identically to the decoder; deferred
because it never affects the decoded image.

## `still265` effort ladder + CABAC bit-cost estimator (findings)

The encoder exposes a seven-step `Effort` ladder (`Fastest`/`Fast`/`Balanced`/
`Good`/`Best`/`Placebo`/`Reference`, HandBrake/x265-style). The current `Best`
is the practical max-quality tier between `Good` and `Placebo`; the current
`Placebo` is the former parallel exhaustive `Best`; `Reference` is the former
serial running-context `Placebo`. The knobs are `rough_luma_modes` (shortlist breadth),
`luma_rd_candidates` (size-aware for `Fast`: a 2nd candidate only on 16x16+
CUs), `chroma_rd_candidates`, and single-scan-vs-exact RDOQ (`Placebo`/
`Reference` keep exact greedy).
Those knobs are now QP-aware on the non-reference tiers: at QP 38+ and again at
QP 44+, the encoder trims rough-mode breadth, luma/chroma RD candidates,
limited-RDOQ coverage, and selected fast-tier split early-outs. `Best`,
`Placebo`, and `Reference` remain exhaustive at rough-mode search.

The multi-candidate non-reference path also decouples chroma from luma mode
selection: luma candidates are ranked using a legal DM-chroma proxy, then
`choose_chroma_mode` runs once for the winning luma mode. The exhaustive
joint luma/chroma search is retained for `Placebo`/`Reference`.

Profiling context: `estimate_residual_bits` is the hottest path, but it is
**already** as cheap per call as the standard HEVC tricks allow — the cost
model is table-driven (`cabac::ENTROPY_BITS` = x265 `g_entropyBits`, no
arithmetic-coder simulation), and each coded block's residual cost is memoized
on the block (`CodedBlock::frac_bits`) so the transform-tree estimator
(`estimate_tt_inner`) only re-prices the cheap split/cbf flags, never the
residual. So the encoder is **call-bound**: estimator invocations scale with RD
search breadth (measured before the rename at ~69k at `Fastest`, ~138k at
`Balanced`, ~1.08M at old-`Best`/current-`Placebo` on a 768x768 crop), which is
exactly what the effort ladder controls.

Output-preserving cleanup done alongside: the single-context estimate helpers
(`estimate_intra_luma_mode_bits`, `estimate_intra_chroma_mode_bits`,
`estimate_split_cu_flag_bits`, and the `part_mode` price in
`estimate_cu_leaf_bits`) previously cloned all 170 CABAC contexts to mutate
one; they now copy just the single touched `ContextModel` (bit-identical).

Not done (deferred, would change output / larger refactors): widening the
incremental RDOQ cache validity to cut full re-estimates in the exact-greedy
tiers; a compact
nonzero-coefficient list so the residual walk tracks `nnz` rather than block
area; coarse-grained CTU/wavefront threading (the inner estimate cannot be
vectorized or threaded, but independent CUs/CTUs can run in parallel).

## BPG-C `-q` vs x265 Avg QP calibration

`bpgenc -e x265 -q N` does not report the same effective x265 slice QP as
still265 `--qp N` for the 8-bit all-intra BPG tests. Verbose BPG-C probes show
`-q 28` reports x265 `Avg QP:25.00`, while `-q 31` reports `Avg QP:28.00`.
Heads-up tests that want equal effective QP should therefore pass BPG-C
`q = still265_qp + 3` and record the actual x265 encode QP separately from the
comparison QP. The unshifted user-facing `q=N` comparison is still useful as a
BPG compatibility check, but it gives x265 a 3-QP quality advantage and should
not be read as an equal-QP architecture comparison.

## Full x265 RDOQ / signhide / transform-skip parity plan

**Efficiency-gap measurement (the reason for this plan).** A QP-sweep BD-style
comparison (still265 `Best` vs `bpgenc -m9` = x265 *placebo*, both deblock-on,
SAO differs — see below) shows that at **equal output size** x265 is ahead by
**~1.4 dB PSNR and ~0.015 SSIM** consistently across QPs. The headline
equal-QP "+44% size / +1.17 dB" is an operating-point artifact (still265 `Best`
at QP N sits at a higher-rate point than x265 at QP N); the equal-*size* gap is
the real coding-efficiency deficit. Of that ~1.4 dB, ~0.3 dB is the missing SAO
(x265 `-m9` runs deblock+SAO; the rust runs were SAO-off), leaving ~0.8-1.1 dB
of pure coefficient-coding deficit — which is what this plan targets. Note the
search *structure* already mirrors x265 placebo (split-free candidate eval +
single full RQT on the winner via `best_luma_leaf_screen`; x265-style
`2+rdLevel+depth/2` candidate cap with the 25% threshold + MPM0 forcing; rough
SAD lambda via `best2_rough_lambda`), so the deficit is in coefficient coding,
not search breadth. The speed gap (~2.4x) is primitive throughput (scalar
quant/dequant/inverse-transform/DST4/intra-prediction vs x265 assembly), not
over-search.

**RD lambda parity (`best2_rd_lambda`, done).** The SSE-domain RD/RDOQ lambda
now matches `x265_lambda2_tab` (`0.038*exp(0.234*qp)`) instead of the legacy HM
intra factor (0.57), which was ~16-24% lower (under-penalizing bits). Default-on
for `Best` (`BPG_BEST2_RD_LAMBDA=0` reverts); the rough SAD lambda stays on the
legacy base so `best2_rough_lambda` is unchanged. **Finding: low-impact alone
(~0.2-0.3% size)** — still265's current RDOQ doesn't make aggressive
lambda-driven zeroing decisions, so lambda has little to bite on. Correct for
parity, but confirms the size lever is RDOQ-2, not lambda.

1. **Sign-data hiding — DONE (`sdh_active`/`BPG_BEST2_SDH`, default-on for
   `Best`).** `sign_data_hiding_enabled_flag = 1`; the quant path makes each
   coding group's hidden sign parity-consistent via a faithful port of x265
   `Quant::signBitHidingHDQ` (`crate::residual::apply_sign_data_hiding`), run on
   the final levels before reconstruction; the writer omits exactly those signs.
   Reference tiers stay SDH-free (bit-stable). Verified: `all_effort_tiers_round_trip`
   (in-tree decoder bit-exact for `Best`+SDH), stock `bpgdec` accepts the stream
   with no new divergence, and **−2.7..−3.3% size** on a photo at QP 28-36.
2. **RDOQ level-2 rate model — DONE (`best2_rdoq2`/`BPG_BEST2_RDOQ2`, default-on
   for `Best`), but FINDING: neutral on photos.** `rdoq_single_scan` now tracks
   the x265 `rdoqQuant` greater1/greater2 context state (`ctxSet`/`c1`/`c2`/
   `c1Idx`/`c2Idx`) and adapted Rice parameter, replacing the frozen-context
   magnitude model (faithful port of `getICRateCost` via `ic_level_bits`).
   Verified bit-exact round-trip (`all_effort_tiers_round_trip`) and stock
   `bpgdec`. **A/B: −0.12%..+0.03% size, ±0.015 dB — essentially zero**, because
   the per-coefficient level decision is distortion-dominated, so a more accurate
   *rate* barely moves it (same outcome as the lambda switch). still265 already
   had the structural RDOQ pieces (last-position + CSBF zero-forcing); making the
   rate exact does not close the gap.
3. Add transform skip for eligible 4x4 TUs: syntax flag, forward/recon path,
   quant/RDOQ pricing, and RD selection against the normal transform. **Expected
   ~0 on photos** (transform-skip helps screen/text/line-art); pursue only if a
   screen-content corpus is in scope.

**Re-diagnosis (post SDH + lambda + RDOQ-2).** With all three coefficient-coding
levers on, the equal-size gap vs x265 `-m9` is **still ~1.0-1.2 dB (~+15% BD-rate)**
across photos/QPs. SDH recovered ~3% (~0.15 dB); lambda and RDOQ-2 were neutral.
So the coefficient-coding layer is **not** where the deficit lives. The remaining
gap is in **(a) in-loop filtering** — x265 `-m9` runs SAO (~0.3 dB), still265
`Best` runs SAO-off — and **(b) bit allocation / perceptual RDO**: x265 `-m9`
uses adaptive quantization (and psy-rd), while `Best` is deliberately uniform-QP
with AQ off. The highest-value remaining work for the photo corpus is therefore
AQ/psy-rd for `Best` and (a faster) SAO — not further coefficient-coding tools.

**AQ for `Best` — implemented (`BPG_BEST_AQ=psnr|perceptual`, `BPG_BEST_AQ_STRENGTH`,
default-OFF), FINDING: does not help photos.** A bidirectional, rate-neutral
variance AQ (`preanalysis::aq_qp_offset_variance`, x265 aq-mode style, steering on
the raw per-cell `variance` not the importance map) was added for `Best`, with the
search fast paths it is incompatible with gated off under AQ:
`best_luma_leaf_screen` / `best_tu_neighbor_limit` / `best2_tt_reuse` /
`best2_cu_reuse`, and — critically — the **parallel WPP path** (`BEST_PARALLEL`),
whose frozen-slice-init context does not carry per-QG QP prediction across worker
boundaries (it produced valid-but-garbage ~20 dB encodes until serial-gated). With
those gates, recon is bit-exact (`all_effort_tiers_round_trip` under `BPG_BEST_AQ`)
and stock `bpgdec` agrees. **A/B (QP sweep, photos): RD-neutral-to-negative.**
`psnr`-mode is roughly RD-neutral on PSNR (e.g. +3.3% size for +0.19 dB, worse than
just lowering QP) and slightly negative on SSIM; on busier images it is clearly
negative (+11% size / −3.5 dB). `perceptual`-mode degrades PSNR badly at higher QP
(flat QGs pushed past QP 40 → banding). This matches the project's prior conclusion
that the preanalysis signal separates digital-vs-photo content but not intra-photo
complexity well enough to drive a useful PSNR-leaning allocation. Kept default-off
and available for screen/mixed-content experimentation; **not** the photo-gap lever.

**Net finding across the parity batch (lambda, SDH, RDOQ-2, AQ): only SDH (~3% /
~0.15 dB) moved the photo corpus.** The residual ~1.0-1.2 dB equal-size gap vs x265
`-m9` is not recoverable from the RD-lambda, the RDOQ rate model, or QP allocation —
it is SAO (~0.3 dB, deferred) plus, most likely, x265's core intra mode/partition RDO
and psy-rd, which would need a separate investigation (recommended next: build the
x265 CLI for `--no-aq`/`--no-sao`/`--no-psy` ablations to attribute the remainder
precisely).
4. Run corpus A/B gates after each phase against x265 m9 at equal effective QP:
   size, PSNR/SSIM, encode time, and stock `bpgdec` compatibility. Promote the
   phase into `Best` only when it improves compression or closes a confirmed
   x265 parity gap without a disproportionate speed regression.

## `bpg-tools decode --format rgb` vs. stock `bpgdec` PNG output differs by a
   few levels on natural/textured images (pre-existing, format-independent)

Decoding the *same* `.bpg` bitstream with `bpg-tools decode` (Rust
`bpg-decode` + `image` PNG encode) and with the stock C `bpgdec` produces PNGs
that differ by a few pixel levels in a large fraction of pixels (e.g. for
`crates/bpg-decode/tests/fixtures/lena512color.bpg`, an x265-encoded fixture:
mean abs diff ~0.5, max 26, ~42% of pixels differ by >=1). For a still265-
encoded 4:2:0/4:2:2 stream the mean diff is larger (~3.7-6.5, max up to ~75),
but the discrepancy exists for **both** 4:2:0 and 4:2:2 at comparable
magnitude, so it is not specific to either chroma format or to the still265
4:2:2 encode path (see the 4:2:2 chroma cbf-gating fix in `still265::encoder`,
which was verified independently via the internal
`still265::encoder::encode` -> `bpg_hevc_decode::hevc::decode` round trip,
which is bit-exact).

Likely cause: a rounding/range difference in the YCbCr -> RGB conversion
between `bpg-image`/`image`'s PNG encode path and `bpgdec`'s libpng path (not
yet root-caused). Worth investigating separately with a raw-plane comparison
(bypassing PNG RGB conversion) to confirm whether the HEVC decode itself
(`bpg-hevc-decode`) matches `bpgdec`'s internal YCbCr planes exactly.

## `still265` primitive SIMD dispatch: only 8-bit SATD is vectorized

`still265::primitives` (Chunk 6) introduces a function-pointer dispatch table
(`Primitives`/`PRIMITIVES`) selected once from the runtime CPU features and the
`BPG_PRIMITIVES` env var (`scalar`/`simd`/`asm`/`auto`).

Optimized kernels implemented so far:

- **8-bit SATD** (`satd_u8`): SSE2, `x86_64`, always built.
- **10/12-bit SATD** (`satd_u16`), **SSD / RD distortion** (`ssd_u16`),
  **residual subtraction** (`sub_residual`), **forward 1-D DCT**
  (`forward_dct_1d`): portable `wide`-SIMD (`primitives::wide_simd`), behind the
  optional **`wide-simd`** feature (one pure-Rust dependency, `wide`). These map
  to x265's high-bit-depth `pixel_satd` (`pixel-a.asm`), `pixel_ssd`
  (`ssd-a.asm`), `pixel_sub_ps` (`pixel-util8.asm`), and `dct4/8/16/32`
  (`dct8.asm`). The `satd_u16` kernel uses i32 lanes (so 10/12-bit diffs do not
  overflow) and `wide::i32x4::transpose`, applying the order-independent
  column-first/transpose/row Hadamard so its 16 coefficients are the transpose
  of the scalar's — identical absolute-value sum. The distortion/residual
  kernels take the SIMD fast path only for interior blocks (no source-edge
  clamping); edge blocks fall back to the scalar clamped loop. The DCT kernel
  only vectorizes `n >= 8` (DCT4/DST4 use the scalar tail).

Still scalar-only (not yet routed / vectorized):

- **Quant/dequant loops, inverse transform, forward DST4**: scalar only.
- **`asm` backend**: `BPG_PRIMITIVES=asm` is accepted but resolves to the best
  available SIMD path; there are no hand-written assembly kernels (the default
  build stays pure Rust).
- **AVX2 / aarch64 NEON**: no arch-specific tier beyond the 8-bit SATD SSE2
  kernel; `wide` auto-selects whatever the target supports at compile time.

All optimized kernels are byte-identical to the scalar reference: enforced by
`primitives::tests::sse2_satd_u8_*` and `primitives::wide_simd::tests::*`
(per-kernel scalar-equivalence over random patterns/sizes/bit-depths), and by
end-to-end byte-identical encodes (scalar build vs `wide-simd` build, and
`BPG_PRIMITIVES=scalar` on the `wide-simd` build) across 8/10/12-bit ×
4:2:0/4:2:2/4:4:4 × fast/balanced/best. `BPG_PRIMITIVES=scalar` forces the
reference path for A/B testing.

Measured speedup (768x768 dusk crop, qp28, best-of-N wall-clock, `wide-simd`
vs scalar build): 8-bit balanced 4:4:4 ~20%, best 4:4:4 ~10%, balanced 4:2:0
~8%, best 4:2:0 ~1% (that tier is dominated by CABAC `estimate_residual_bits`,
which is inherently sequential and not a SIMD target). 12-bit (with the
`satd_u16` kernel also active) balanced/fast 4:4:4 and balanced 4:2:0 all
~20%.

## `still265` encoder allocation reduction: partial (Chunk 7)

Done (all verified bitstream-identical): the per-mode predicted-block and
8-bit-truncation buffers in the luma/chroma rough-mode loops are pooled
(`predict_intra_into` + `pred`/`pred8` scratch buffers), `code_block`'s residual
buffer is pooled across calls via an `Encoder::scratch_residual` field
(`mem::take`), and forward-transform coefficient/transpose scratch buffers are
pooled through `transform::forward_transform_into`.

Not yet done: the RDOQ `levels` (which escape in `CodedBlock`) and the
inverse-transform `res` buffer are still freshly allocated per call — pooling
the latter requires threading scratch buffers through
`transform::reconstruct_residual` and the decoder's `inverse_transform`
signatures. The frame/map snapshot buffers
(`snapshot_plane`/`snapshot_frame_region`/`snapshot_*_region`) used for RD
rollback also allocate per call; pooling them touches RD-rollback correctness
and is deferred to avoid risking bitstream changes. None of these change
output; they are pure speed work.

## `still265` RDOQ stage (c): net win, occasional sub-1% local loss

The single-scan RDOQ (`rdoq::rdoq_single_scan`, used by `Balanced`/`Fast`) now
implements stage (c): middle sub-block `coded_sub_block_flag` all-zero forcing
(Chunk 8). Measured against the pre-stage-(c) encoder over a 2-image ×
{420,422,444} × {qp 18..44} grid: **−433 bytes total, 12 size wins, 2 small
regressions** (max +10 bytes / 0.2%). The regressions are inherent to the
frozen-context cost estimate (it does not re-price the `prev_csbf` context
shift that zeroing imposes on lower-frequency neighbours) and match how
HM/x265's greedy sub-block RDOQ behaves; a slack margin was tried and made it
worse, so the decision is left bare (`zero < keep`). Stage (c) only affects the
single-scan path, never `Effort::Placebo`/`Effort::Reference` (exact greedy),
and output stays decoder-compatible (verified with stock `bpgdec`).

Not done in Chunk 8: the goal also mentioned testing `Balanced` chroma RD
candidate expansion (top-2/3 trials); `Balanced` still uses one chroma
candidate, while `Good`'s chroma RD candidate count now contracts at high QP.
