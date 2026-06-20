# Current rdo2 Baseline (2026-06-20)

Baseline after:

- `44e92d6 Checkpoint rdo2 encoder refactor`
- `074eed5 Promote Best trial RDOQ gate default`

The harness uses `--sizes`; the `--only-size` examples in `plan.md` are stale.
On this Linux build, `--c-single-thread` is Windows-affinity-only, so the C
columns below use the local `./libbpg/bpgenc` default threading. Rust
single-thread runs use `--rust-single-thread` plus `BPG_ENC_THREADS=1` through
the harness.

## Commands

```sh
./target/release/bpg-highres-compare \
  --effort best --compress-level 9 --bpgenc ./libbpg/bpgenc \
  --sizes 1000x750,2000x1500,4000x3000 --max-sources 1 --runs 1 \
  --rust-single-thread \
  --work-dir target/highres-current-best-m9-rust-singlethread

./target/release/bpg-highres-compare \
  --effort best --compress-level 9 --bpgenc ./libbpg/bpgenc \
  --sizes 1000x750,2000x1500,4000x3000 --max-sources 1 --runs 1 \
  --work-dir target/highres-current-best-m9-threaded

env BPG_TRACE_SEARCH=target/trace-best-current-1000 BPG_PROFILE=1 \
  BPG_BEST2_PARALLEL=0 BPG_ENC_THREADS=1 \
  ./target/release/bpg-highres-compare \
  --effort best --compress-level 9 --bpgenc ./libbpg/bpgenc \
  --sizes 1000x750 --max-sources 1 --runs 1 --rust-single-thread --skip-c \
  --work-dir target/highres-current-best-trace-1000

env BPG_TRACE_SEARCH=target/trace-best-current-4000 BPG_PROFILE=1 \
  BPG_BEST2_PARALLEL=0 BPG_ENC_THREADS=1 \
  ./target/release/bpg-highres-compare \
  --effort best --compress-level 9 --bpgenc ./libbpg/bpgenc \
  --sizes 4000x3000 --max-sources 1 --runs 1 --rust-single-thread --skip-c \
  --work-dir target/highres-current-best-trace-4000
```

## High-Resolution Timing

Rust single-thread:

| size | C total s | Rust total s | Rust encode s | Rust/C total | build s | write s | restore GB | restore fanout s | encode s/MP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000x750 | 0.350 | 2.938 | 2.922 | 8.39 | 2.866 | 0.017 | 0.09 | 0.001 | 3.896 |
| 2000x1500 | 0.764 | 10.510 | 10.449 | 13.75 | 10.237 | 0.055 | 0.38 | 0.004 | 3.483 |
| 4000x3000 | 1.899 | 38.753 | 38.528 | 20.41 | 37.781 | 0.174 | 1.57 | 0.015 | 3.211 |

Default threaded:

| size | C total s | Rust total s | Rust encode s | Rust/C total | build s | write s | restore GB | restore fanout s | encode s/MP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000x750 | 0.346 | 0.907 | 0.891 | 2.62 | 0.834 | 0.017 | 0.11 | 0.006 | 1.189 |
| 2000x1500 | 0.811 | 2.304 | 2.242 | 2.84 | 2.019 | 0.058 | 0.54 | 0.041 | 0.747 |
| 4000x3000 | 1.902 | 7.514 | 7.274 | 3.95 | 6.436 | 0.194 | 2.23 | 0.158 | 0.606 |

## Profile Traces

1000x750, Best QP28, Rust single-thread:

| metric | value |
|---|---:|
| rust encode s | 2.937 |
| bytes | 143077 |
| cu trials | 391568 |
| trial RDOQ blocks | 316289 |
| forward transforms | 816003 |
| inverse transforms | 622199 |
| residual bit estimates | 402658 |
| frame snapshots/restores | 281787 / 247353 |
| bytes snapshotted/restored | 83921932 / 90127628 |
| PartNxN attempts/wins/losses | 11562 / 6477 / 5085 |
| rdo2 chroma cheap/exact evals | 81880 / 8180 |

Inner profile:

| bucket | ms |
|---|---:|
| snapshot+restore | 20.5 |
| rdo2 predict | 132.2 |
| rdo2 forward transform | 581.1 |
| rdo2 quant+rdoq | 1035.5 |
| rdo2 inverse transform | 62.6 |
| rdo2 residual price | 368.3 |
| rough SATD search | 278.1 |

4000x3000, Best QP28, Rust single-thread:

| metric | value |
|---|---:|
| rust encode s | 42.337 |
| bytes | 1234647 |
| cu trials | 6044321 |
| trial RDOQ blocks | 4846973 |
| forward transforms | 12406571 |
| inverse transforms | 8377840 |
| residual bit estimates | 5398534 |
| frame snapshots/restores | 4328877 / 4117617 |
| bytes snapshotted/restored | 1367216206 / 1614067840 |
| PartNxN attempts/wins/losses | 173174 / 57836 / 115338 |
| rdo2 chroma cheap/exact evals | 1213452 / 136648 |

Inner profile:

| bucket | ms |
|---|---:|
| snapshot+restore | 338.3 |
| rdo2 predict | 2151.4 |
| rdo2 forward transform | 10322.3 |
| rdo2 quant+rdoq | 12934.1 |
| rdo2 inverse transform | 1006.8 |
| rdo2 residual price | 4246.0 |
| rough SATD search | 4424.7 |

## Interpretation

- Build/RD dominates both serial and threaded paths; write and restore fanout
  remain secondary.
- The most important measured buckets are still quant/RDOQ, forward transform,
  residual pricing, and rough SATD.
- Snapshot/restore is visible but smaller than the kernel buckets, so the next
  structural work should be measured with a per-stage work ledger before large
  scratch-recon rewrites.
- `BPG_BEST_TRIAL_RDOQ_GATE` no longer appears to be a useful primary speed
  lever on the current rdo2 Best path; it should be treated as compatibility or
  diagnostic surface unless a broader trace proves otherwise.

## WorkBucket Smoke

After the baseline, `SearchTrace` gained `work_ledger.csv`. A 1000x750 QP28
trace smoke (`target/trace-work-ledger-smoke`) produced these largest buckets:

| bucket | calls | wall ms | RDOQ | exact bits | approx bits |
|---|---:|---:|---:|---:|---:|
| LumaCandidateCheap | 104231 | 423.685 | 0 | 0 | 104231 |
| TtLeafCheap | 147530 | 430.697 | 0 | 0 | 147530 |
| TtLeafExact | 18776 | 383.263 | 18776 | 18776 | 0 |
| FinalReplay | 147297 | 361.774 | 147297 | 147297 | 0 |
| NxnPuExact | 285891 | 348.840 | 285891 | 285891 | 0 |
| RoughLumaAllAngles | 61558 | 277.173 | 0 | 0 | 0 |

This table is now the required attribution surface for the next slices. It
shows why future work should target exact NxN/TT/final RDOQ, final replay reuse,
and rough-luma throughput rather than scheduler flag tuning.
