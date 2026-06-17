# still265 vs x265 quality baseline — 2026-06-17

Session baseline for the "close the quality gap vs x265" work, regenerated on
the current `decoder-chroma-support` checkout (`D:\Rust-projects\bpg-rs`) with a
freshly built `target/release/bpg-tools.exe`. The prior baseline in
`target/headsup-quality/` was produced from a different checkout path
(`D:\isolated-dev\BPG\bpg-rs`) and is superseded by this one.

## Harness

`scripts/headsup_quality.py`, 7-image main `test-set/*.png` (1024x768 photos),
4:2:0, QP sweep {24,28,32,36}, decoded with stock `bpgdec.exe`, PSNR/SSIM over
RGB24 via ffmpeg.

```
python scripts/headsup_quality.py --images "test-set/*.png" --qps 24,28,32,36 \
  --rust-efforts best --x265-levels 9 \
  --bpgenc ./bpgenc_native.exe --bpgdec ./bpgdec.exe \
  --outdir target/quality-session/baseline
```

`x265` here = `bpgenc -e x265 -m9` = x265 4.1 placebo (SAO + psy-rd on);
still265 `best` is SAO-off.

## Equal-QP (averaged over the sweep)

| encoder        | avg size (B) | avg bpp | avg PSNR (dB) | avg SSIM |
|----------------|-------------:|--------:|--------------:|---------:|
| still265 best  | 94924        | 0.9656  | 34.439        | 0.93261  |
| x265 m9        | 95548        | 0.9720  | 35.115        | 0.93665  |

Equal-QP: x265 is **+0.68 dB** at ~equal size.

## Equal-size (nearest-size matching, rust − x265)

- mean PSNR delta: **−0.676 dB**
- mean size delta: **+2.49 %** (still265 is larger)
- mean SSIM delta: **−0.0040**

still265 `best` is worse on *both* axes (larger and lower PSNR), consistent with
the documented "spends residual bits to repair worse prediction" hypothesis.
This is the number to beat.
</content>
</invoke>
