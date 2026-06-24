Yes, it makes sense, and it is decoder-friendly **if expressed as legal HEVC syntax**.

But one correction first: x265 does **not** generally have a maximum block size of 16×16 for 4K. HEVC/x265 normally uses **64×64 CTUs** by default. What x265 may do is *choose* many 16×16 or smaller CUs after recursive analysis, depending on content, preset, and RD settings. There is also an x265 `--ctu` option, but 64×64 CTUs are normal HEVC.

Your idea is better stated as:

> Use 64×64 CTUs, allow only 64×64 leaf or terminal 64→32 split, and never recurse into 16×16 / 8×8 CU search.

That is coherent.

## Decoder friendliness

This is decoder-friendly.

HEVC supports 64×64 CTUs. A CTU may remain a 64×64 CU or split into four 32×32 CUs. The decoder just follows the split flags in the bitstream.

Two options:

```text id="wzlmhg"
Option A:
    Keep normal SPS min CU size, e.g. 8x8.
    Encoder simply never signals splits below 32x32.

Option B:
    Signal min CU size = 32x32 / max CU depth = 1.
    Decoder knows 64→32 is the deepest possible split.
```

Option A is probably safer for compatibility and easier integration: keep the bitstream parameter set conventional, but make the encoder policy shallow.

Also remember: even with a 64×64 CU, HEVC transform blocks max out at 32×32. So a 64×64 leaf still needs valid transform-tree handling. That is fine. You can have:

```text id="1r7l1m"
64x64 CU
    prediction unit: large
    transform units: 32x32 / 16x16 / etc. as allowed
```

So “no recursive CU split” does not mean “no residual adaptation.”

## Why it makes sense for still images

For still-image encoding, this is a very plausible direction.

Video encoders care about:

```text id="0mocwx"
motion interaction
temporal consistency
real-time throughput
per-frame complexity budget
many frame types
scene changes
reference structure
```

BPG/still265 cares about one intra image.

Your experiments already show that:

```text id="1sumkp"
Floor:
    64x64 leaf-heavy encode
    usable quality
    very fast
    only +6.4% bytes vs Best on the test image

FloorPlus:
    64 leaf + shallow 64→32 repair
    recovers 50.6% of the Floor→Best byte gap
    still far faster than Best
```

So yes: the data supports a shallow-CU architecture.

The deeper 16×16 / 8×8 / PartNxN machinery appears to be a **marginal compression polish layer**, not the core still-image coding mechanism, at least on large natural photos.

## What this policy would be

Call it something like:

```text id="nc0f04"
Shallow64
FloorShallow
Still64
FloorPlus2Strict
```

The rule:

```text id="7fu7xk"
At each 64x64 CTU, choose between:

A. 64x64 CU leaf
B. four terminal 32x32 CU leaves

No 32→16 CU split.
No 16→8 CU split.
No PartNxN.
```

Inside each 64 or 32 leaf, you may still allow:

```text id="bx0k9v"
odds-gated angular mode search
TU split
RDOQ on final winner
chroma repair
SAO/deblock as normal
```

So the split tree stays shallow, but prediction and transform coding are not crippled.

## Strict version

The strict version is:

```text id="8k0jzn"
64 leaf
or
64→32 terminal split
```

No child repair. No recursive auction.

That is essentially a cleaner formalization of the current `FloorPlus`.

Search candidates:

```text id="jh1jdw"
Candidate 0:
    64x64 Floor leaf

Candidate 1:
    enhanced 64x64 leaf
    odds-gated modes
    optional TU split

Candidate 2:
    terminal 64→32 split
    each 32 child uses odds-gated leaf/TU search
```

Choose by real RD cost.

This should be very stable and decoder-safe.

## Soft version

The soft version allows one exception:

```text id="e310h6"
64→32 split wins
then repair only the worst 32x32 child
```

But this violates the strict “never past first level” idea if that repair includes 32→16 CU split.

So I would split the experiments:

```text id="x6r8vg"
Shallow64Strict:
    no CU split below 32

Shallow64Soft:
    one worst-child 32→16 terminal split allowed
```

Run them separately. Do not blur them.

Given your theory, `Shallow64Strict` is the pure test.

## Why it could improve size efficiency

Shallow CU search may improve size efficiency per unit time because it avoids the bad bargain:

```text id="36dghi"
deep recursive search:
    many syntax decisions
    many small blocks
    high mode/TU/RDOQ search cost
    small byte savings

shallow search:
    fewer split flags
    fewer modes
    fewer coeff trees
    better context locality
    more stable large-block prediction
```

For natural photos, large CUs often work well because the image has smooth gradients and correlated regions. Detail can often be handled by transform subdivision rather than CU subdivision.

The important design idea is:

> Use CU split for coarse region separation, not for every local texture detail.

Local texture detail should first be handled by:

```text id="c3fjpf"
angular mode choice
TU split
coefficient coding
possibly chroma repair
```

Only then consider deeper CU split.

## Possible downside

This will not be universally optimal.

It may lose more bytes or visible detail on:

```text id="fp7e22"
text over image
screenshots
line art
dense foliage
fabric
hair
high-noise images
small images
sharp synthetic edges
```

Those are the cases where 16×16 / 8×8 CUs can matter.

But that does not mean the policy is bad. It may mean you need a fallback classifier:

```text id="v5v05k"
photo-like CTU:
    Shallow64

text/screen/detail CTU:
    allow deeper repair
```

For BPG archival photos, shallow should probably be the default experiment.

## Implementation plan

### 1. Add explicit CU-depth policy

In effort config:

```rust id="ymti23"
pub enum CuDepthPolicy {
    FullRecursive,
    ForceLeaf,
    Shallow64Strict,
    Shallow64Soft,
}
```

For `Shallow64Strict`:

```rust id="pqp3uj"
max_cu_split_depth = 1;
allow_64_to_32 = true;
allow_32_to_16 = false;
allow_16_to_8 = false;
allow_part_nxn = false;
```

### 2. Keep SPS conventional first

Do not initially change SPS/min CU signalling.

Keep decoder parameters as they are, but enforce encoder policy:

```text id="nnpr66"
encoder never emits split below 32
```

Later, test signalling min CU size 32 if you want. That may save tiny syntax overhead, but compatibility risk is not worth mixing into the first experiment.

### 3. Candidate set per CTU

For each CTU:

```text id="c59rd2"
A. Floor 64 leaf
B. Enhanced 64 leaf
C. Terminal 64→32 split
```

Optional:

```text id="3jqumq"
D. Terminal 64→32 split with odds-gated child modes/TU
```

No other CU shapes.

### 4. Odds-gated mode search

Inside 64 and 32 leaves:

```text id="vpwb3a"
always include:
    Planar
    DC
    MPM
    best rough mode

optionally include:
    nearby angular modes chosen by odds gate
```

This lets 32×32 children improve without exploding into full mode search.

### 5. TU repair remains allowed

This is important.

For a 64×64 leaf, allow TU subdivision where legal/useful. This gives residual adaptation without CU recursion.

You want to test:

```text id="tclrr9"
CU tree shallow
TU tree adaptive
```

not:

```text id="adw81d"
everything shallow
```

The latter may artificially handicap the idea.

## What to measure

For the known image, compare:

```text id="0dnu7x"
Floor
FloorPlus
Shallow64Strict
Shallow64Strict + odds modes
Shallow64Strict + odds modes + TU repair
Best
```

Metrics:

```text id="7lb6bw"
bytes
time
gap recovered
64 leaf wins
64 enhanced leaf wins
64→32 split wins
32 child count
TU split count
mode candidates evaluated
fwd transforms
trial RDOQ blocks
```

Target:

```text id="k4wwq5"
Shallow64Strict:
    recover >=60% of Floor→Best gap
    time <=22s

Shallow64Strict + odds/TU:
    recover >=70% of gap
    time <=28s
```

## Bottom line

Yes, the idea makes sense.

A shallow 64×64 CTU policy is legal, decoder-friendly, and strongly supported by your Floor/FloorPlus results. The clean experiment is:

```text id="bshkue"
64x64 CTU
choose 64 leaf or terminal 64→32 split
no recursive CU split below 32
no PartNxN
keep TU/mode repair adaptive
```

That should become a named experimental effort, separate from FloorPlus and Best. It directly tests the hypothesis that for still-image BPG, **large-block intra coding plus shallow repair beats video-style recursive CU search on time/size efficiency**.
