Yes, x265 has SAO SIMD kernels. But your 7× slowdown is mostly **not** because `sao.rs` lacks SIMD. The current SAO path is algorithmically expensive before SIMD even enters the picture.

## The main problem: SAO triggers two full serial encodes

In `crates/still265/src/encoder/mod.rs`, SAO-on does this:

```rust
let _ = encode_slice_data(&mut state, None, slice_qp_y);
...
let sao_map = state.decide_sao_map(ctb);

state.frame = DecodedFrame::with_params(...);
...
state.analysis_cache.clear();

let bytes = encode_slice_data(&mut state, Some(&sao_map), slice_qp_y);
```

So SAO currently means:

```text
1. Full encode without SAO.
2. Deblock.
3. Analyze SAO map.
4. Throw away frame/state.
5. Clear analysis cache.
6. Full encode again with SAO syntax.
7. Deblock again.
8. Apply SAO.
```

And worse: this uses `encode_slice_data(...)`, not the normal `encode_slice_data_parallel(...)` path used when SAO is off.

So for `Best`, SAO appears to convert:

```text
one parallel encode
```

into:

```text
two serial full encodes + SAO analysis + SAO application
```

That alone can plausibly explain a 7× slowdown.

The first optimization is therefore **not SIMD**. It is to stop doing full analysis twice.

## Correct architecture

You want a two-stage pipeline, but not two full encoders.

### Current bad model

```text
pass 1: analyze + encode + reconstruct
SAO decision
pass 2: analyze + encode + reconstruct again
```

### Better model

```text
pass 1: analyze + reconstruct + cache final decisions
SAO decision
pass 2: replay/write cached final decisions with SAO syntax
```

The second pass should be a **bitstream replay**, not another RDO search.

For still image intra-only encoding, this is especially reasonable. SAO depends on the reconstructed frame after deblock. It does not need CU mode analysis to be redone. The final CTU decisions, intra modes, TU splits, quantized coefficients, CBFs, and QP decisions should be reused.

So add something like:

```rust
encode_slice_analysis(...)
    -> ReconstructedFrame + FinalCtuDecisions

decide_sao_map(reconstructed_after_deblock)

write_slice_from_decisions(decisions, Some(sao_map))
    -> bytes
```

This avoids the catastrophic second RDO pass.

The only caveat is CABAC context drift: SAO syntax is coded before CTU syntax, so the CABAC state entering the CTU is slightly different. But that should affect final bit counts, not require redoing mode/RDO decisions. x265 does not perform full CU re-analysis just because SAO syntax exists.

## Immediate low-risk fix

At minimum, change the first pass to use the normal parallel path:

```rust
let _ = if state.effort_template.parallel_analysis {
    encode_slice_data_parallel(&mut state, slice_qp_y)
} else {
    encode_slice_data(&mut state, None, slice_qp_y)
};
```

That does not solve the double-encode design, but it should reduce the worst regression quickly.

The larger fix is: **do not clear `analysis_cache` and do not rerun full RDO in pass 2**. If the current cache is not sufficient to replay a CTU exactly, that is the next structure to add.

## Second problem: SAO stats scan the same pixels too many times

Current SAO analysis does:

```rust
best_sao_component():
    BO scan once
    EO scan four times
```

For luma, that is **5 full scans per CTB**.

For 4:2:0 chroma, it does:

```rust
Cb BO scan
Cr BO scan
Cb EO x4
Cr EO x4
```

So roughly:

```text
luma: 5 full luma scans
chroma: 10 quarter-size scans
total: about 7.5 luma-equivalent frame scans
```

That is not insane by itself, but your implementation makes it more expensive than necessary:

```rust
let src = self.src_sample(c_idx, x, y) as i64;
```

is called inside every BO/EO pass. EO does bounds checks per pixel. EO uses generic `(dx, dy)` arithmetic and two sign comparisons per pixel. x265 avoids much of this.

## What x265 does differently

x265 computes a CTU-local `diff` buffer once:

```cpp
diff[y * MAX_CU_SIZE + x] = fenc - rec;
```

Then all SAO stats consume:

```text
diff[]
rec[]
```

instead of repeatedly loading source and computing `src - recon`.

Relevant x265 primitive types exist in `source/common/primitives.h`:

```cpp
saoCuStatsBO
saoCuStatsE0
saoCuStatsE1
saoCuStatsE2
saoCuStatsE3

saoCuOrgB0
saoCuOrgE0
saoCuOrgE1
saoCuOrgE2
saoCuOrgE3
```

On x86, x265 has SAO assembly in:

```text
source/common/x86/loopfilter.asm
```

and registers SIMD versions in:

```text
source/common/x86/asm-primitives.cpp
```

Notably:

```text
saoCuOrgB0_sse4 / avx2
saoCuOrgE0..E3_sse4 / avx2
saoCuStatsE0..E3_sse4 / avx2
```

`SAO_BO` stats assembly exists in `loopfilter.asm`, but x265’s x86 registration has `saoCuStatsBO_sse4` commented out in at least one registration path, so BO stats may fall back to C on x86. AArch64 has NEON/SVE/SVE2 SAO stats implementations, including BO.

So yes: **there are x265 SAO kernels**, but copying them directly has licensing implications if your crate is not GPL-compatible.

## Concrete optimization order

### 1. Remove the second full RDO pass

This is the priority.

Add a replay path:

```rust
write_slice_from_final_decisions(
    state: &mut EncoderState,
    decisions: &[FinalCtuDecision],
    sao_map: Option<&SaoMap>,
    slice_qp_y: i32,
) -> Vec<u8>
```

This pass should only:

```text
- write SAO syntax
- write split flags / modes / transform tree from cached decisions
- write quantized coefficient levels from cached decisions
- update CABAC contexts
- reconstruct if needed
```

It should not:

```text
- run RMD
- run full RD mode search
- run chroma mode decision
- run RDOQ
- estimate residual bits repeatedly
```

If this is done, SAO should stop being a 7× option. It may still cost 10–40% depending on stats/application, but not 7×.

### 2. Keep the first pass parallel

For SAO map generation, the first reconstructed frame does not need serial analysis. Use the same encode path as SAO-off.

```text
SAO off:
    parallel Best encode

SAO on:
    parallel Best encode for reconstruction
    SAO map decision
    cheap serial/parallel bitstream replay
```

The replay itself probably has to be deterministic raster order for CABAC, but it should be cheap.

### 3. Precompute CTU diff once

Replace repeated `src_sample()` calls inside BO/EO with a CTU-local diff buffer.

Current:

```rust
for every SAO mode:
    for every pixel:
        src_sample()
        recon
        src - recon
```

Better:

```rust
let mut diff = [i16; 64 * 64];

for pixel in CTU:
    diff[i] = src[i] - recon[i];

sao_bo_stats(diff, recon)
sao_e0_stats(diff, recon)
sao_e1_stats(diff, recon)
sao_e2_stats(diff, recon)
sao_e3_stats(diff, recon)
```

For 8/10/12-bit, `i16` is enough:

```text
8-bit:   -255..255
10-bit:  -1023..1023
12-bit:  -4095..4095
```

Use `i32` accumulators for sums/counts, then `i64` only for final reduction/cost math.

### 4. Specialize EO classes

Current `sao_eo_stats()` is generic:

```rust
let (dx0, dy0, dx1, dy1) = EO_OFFSETS[eo_class];
...
nx0 = x + dx0
ny0 = y + dy0
...
bounds check
...
signum()
```

Replace it with four functions:

```rust
sao_eo0_stats_horizontal()
sao_eo1_stats_vertical()
sao_eo2_stats_diag_135()
sao_eo3_stats_diag_45()
```

Each function should use direct indexing and precomputed safe ranges.

For example, horizontal EO does not need arbitrary neighbor math:

```rust
left  = rec[idx - 1]
right = rec[idx + 1]
```

Vertical:

```rust
up   = rec[idx - stride]
down = rec[idx + stride]
```

Diagonal:

```rust
up_left    = rec[idx - stride - 1]
down_right = rec[idx + stride + 1]
```

This matters because SAO stats are pure tight loops.

### 5. Remove per-pixel boundary checks

x265 computes `startX`, `endX`, `startY`, `endY` per CTU and then runs fast interior loops. Your EO stats currently check bounds inside the pixel loop.

Do this instead:

```text
interior CTUs:
    no bounds checks

edge CTUs:
    use clipped start/end ranges
```

For most CTUs in a 1024×1024 image, boundaries are irrelevant. Do not pay for them per pixel.

### 6. Use rolling sign buffers like x265

For EO, x265 avoids computing both neighbor signs independently for every pixel.

Horizontal example:

```cpp
signLeft = sign(rec[0] - rec[-1])
for x:
    signRight = sign(rec[x] - rec[x + 1])
    edgeType = signRight + signLeft + 2
    signLeft = -signRight
```

Your current code computes:

```rust
sign0 = sign(recon - n0)
sign1 = sign(recon - n1)
```

for every pixel. That is twice the comparison work in some EO modes.

For vertical/diagonal, x265 uses small `upBuff` arrays to carry signs between rows. Port that pattern.

### 7. Make SAO decision parallel

After reconstruction/deblock, SAO map decisions are mostly independent per CTU. The merge flags are written later when neighboring parameters match; they do not require serial decision.

So:

```rust
let infos: Vec<SaoInfo> = (0..num_ctus)
    .into_par_iter()
    .map(|addr| decide_sao_ctu(addr))
    .collect();
```

Then build `SaoMap` from the vector.

This is safe if each CTU decision only reads:

```text
source frame
reconstructed frame
bit depth
component dimensions
```

and writes one `SaoInfo`.

### 8. Optimize `apply_sao`

Your decoder-side `apply_sao()` clones entire planes if any CTU uses EO:

```rust
let orig_y = frame.y_plane.clone()
```

That is simple and correct, but it is not x265-like. x265 uses CTU line buffers, not whole-frame clones.

For encoder speed, ask whether you need to apply SAO at all after writing the bitstream. For actual file output, the decoder will apply SAO. The encoder only needs post-SAO reconstruction if you are computing PSNR/SSIM or using the frame as reference. For a still image with no future refs, you can make final `apply_sao()` optional:

```text
normal encode output:
    write SAO syntax, do not apply SAO internally

metrics/debug mode:
    apply SAO to reconstructed frame
```

That alone may shave time in normal operation.

If you do need it, use line-buffer EO application rather than full-plane clone.

## SAO RD decision should charge bits

Your current SAO choice appears to select any mode with positive distortion reduction:

```rust
if best.4 > 0 {
    enable SAO
}
```

x265 does not do that. x265 charges syntax bits using lambda:

```cpp
cost = distortion + lambda * bits
```

This is not mainly a speed fix, but it matters for size efficiency. Some SAO choices may improve SSE slightly while not being worth their syntax cost.

So change the decision from:

```text
choose if reduction > 0
```

to:

```text
choose if reduction - lambda * estimated_sao_bits > 0
```

Approximate bits are fine at first:

```text
type flag bits
offset unary bits
sign bits
band position bits
EO class bits
merge flag bits
```

This may reduce the `+0.85% size` while keeping most of the `+0.38 dB`.

## Should you port x265’s SAO SIMD?

Eventually, yes. But only after the architecture fix.

The likely payoff order is:

```text
1. Avoid double full encode.
2. Replay cached CTU decisions.
3. Parallelize SAO map decision.
4. Precompute diff once per CTU.
5. Specialized scalar EO/BO loops.
6. SIMD SAO stats/application.
```

If you port SIMD first, you may make a bad architecture faster, but SAO will still be structurally too expensive.

## Practical target

A reasonable high-quality still-image SAO overhead should be closer to:

```text
+10% to +40%
```

not:

```text
+700%
```

For a 1024×1024 image, SAO stats and application are only a few full-frame scans. They should not dominate more than the encoder’s intra RDO unless you are re-running the encoder.

So the short answer is:

```text
Yes, x265 has SAO SIMD kernels.

But the 7× slowdown is mainly from the current SAO pipeline:
two full serial encodes, cache clearing, and repeated pixel scans.

Fix the double-encode/replay design first.
Then port x265-style SAO stats/application kernels.
```
