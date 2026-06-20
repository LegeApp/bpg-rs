# x265 Wrapper Single-Thread Controls

The local `libbpg/x265_glue.c` wrapper was patched so benchmark runs can use
x265's real threading controls instead of relying only on process affinity.

Supported environment:

```sh
BPG_X265_SINGLE_THREAD=1
BPG_X265_PARAMS='frame-threads=1,pools=none,wpp=0,pmode=0'
```

`BPG_X265_SINGLE_THREAD=1` applies:

```text
frame-threads=1
pools=none
```

`BPG_X265_PARAMS` accepts comma- or semicolon-separated `name=value` entries and
passes each entry through `x265_param_parse`. Invalid names or values fail the
encode instead of being silently ignored.

Verification on 2026-06-20:

```sh
make -C libbpg bpgenc
BPG_X265_SINGLE_THREAD=1 \
BPG_X265_PARAMS='frame-threads=1,pools=none,wpp=0,pmode=0' \
  ./libbpg/bpgenc -o /tmp/bpg-x265-override-smoke.bpg \
  -q 28 -f 420 -b 8 -m 9 \
  target/highres-compare/generated/20240501_110934_1000x750.png
```

The wrapper produced `/tmp/bpg-x265-override-smoke.bpg` successfully. A deliberate
bad override, `BPG_X265_PARAMS='not-a-real-x265-param=1'`, exited with code 1
and printed `x265: invalid parameter override not-a-real-x265-param=1`.

`bpg-highres-compare --c-single-thread` now sets those x265 params for rebuilt
wrappers and still pins the process to one CPU as a compatibility fallback for
stock wrappers. Additional rebuilt-wrapper x265 params can be passed with
repeatable `--c-x265-param name=value`, for example:

```sh
./target/release/bpg-highres-compare \
  --c-single-thread \
  --c-x265-param wpp=1 \
  --c-x265-param pmode=0
```
