# Oracle x265 still-image settings (locked)

These are asserted by `bpg-x265`'s `X265Encoder` AFTER any tune overrides, so
every oracle encode uses exactly this configuration:

- preset: index 8 (`bpg-tools encode` default)
- tune: `ssim` (EncoderTuning::neutral — the byte-identical-to-C-reference path)
- rate control: CQP (`rc.rateControlMode = X265_RC_CQP`, `rc.qp = <qp>`)
- intra only / still: `keyframeMax = 1`, `totalFrames = 1`
- `bRepeatHeaders = 1` (VPS/SPS/PPS precede the slice)
- `bEnableAMP = 1`, `bEnableRectInter = 1` (BPG modified-SPS requires AMP)
- `bEmitInfoSEI = 0`, `decodedPictureHashSEI = 0`
- color: ColorSpace::YCbCr (BT.601, matrix_coefficients = 6), full range
- chroma siting: JPEG (c_h_phase = 1)

## Metrics

- `psnr`: **encode quality** — stock `bpgdec` (the ground-truth reference
decoder) decoded vs the source. High across all chroma formats confirms the
encoder produces correct 4:2:0/4:2:2/4:4:4 BPG.
- `rust_decode` / `rust_psnr`: the in-repo `bpg-decode` status on the same BPG
(`ok`/`unsupported`/`fail`) and, when it decodes, its fidelity vs the bpgdec
reference. The Rust decoder currently supports 4:2:0 only, so 4:2:2/4:4:4
rows read `unsupported` (see FULL_RUST_CODEC_PLAN.md Phase 1 follow-up).

The corpus is generated deterministically by `bpg-oracle gen`; `bpg-oracle
check` regenerates and compares bpg_bytes (exact) and encode-quality PSNR
(±0.01 dB) against the committed manifest.csv. Byte-exactness assumes the same
vendored x265 build (x265 4.1, ENABLE_ASSEMBLY=OFF in the dev env) and the same
bpgdec.
