#!/usr/bin/env python3
"""Interleaved bpg-highres-compare timing for thin-vs-fat LTO binaries.

Example:

    python3 scripts/lto_compare.py \
      --thin /tmp/bpg-highres-compare-thin \
      --fat /tmp/bpg-highres-compare-fat \
      --input-dir test-set/one-12mp \
      --rounds 5 --work-root /tmp/lto_cmp_12mp

The harness alternates run order by round (thin/fat, then fat/thin) to reduce
thermal/background-load bias, parses `results.csv`, and hashes the emitted Rust
`.bpg` files to verify that the two binaries produce identical streams.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


def parse_stats(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in [
        "phase_total_us",
        "phase_build_us",
        "phase_wpp_wait_us",
        "wpp_row_parks",
        "wpp_row_takeovers",
    ]:
        m = re.search(rf"^\s*{re.escape(name)}:\s*(\d+)", stdout, re.M)
        if m:
            out[name] = int(m.group(1))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_rust_bpg(work_dir: Path) -> Path:
    hits = sorted((work_dir / "rust").glob("*/*.bpg"))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one rust .bpg under {work_dir}, found {len(hits)}")
    return hits[0]


def run_one(
    label: str,
    exe: Path,
    round_idx: int,
    args: argparse.Namespace,
    extra_env: dict[str, str],
) -> dict[str, object]:
    work_dir = args.work_root / f"r{round_idx:02d}_{label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--input-dir",
        str(args.input_dir),
        "--max-sources",
        str(args.max_sources),
        "--native",
        "--skip-c",
        "--qp",
        str(args.qp),
        "--efforts",
        args.effort,
        "--runs",
        "1",
        "--debug-stats",
        "--work-dir",
        str(work_dir),
    ]
    env = os.environ.copy()
    env.update(extra_env)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (work_dir / "stdout.log").write_text(proc.stdout)
    (work_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    with (work_dir / "results.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    bpg = find_rust_bpg(work_dir)
    result: dict[str, object] = {
        "round": round_idx,
        "label": label,
        "encode_s": float(row["rust_encode_s"]),
        "bytes": int(row["rust_bpg_bytes"]),
        "sha256": sha256(bpg),
        "bpg": str(bpg),
    }
    result.update(parse_stats(proc.stdout + "\n" + proc.stderr))
    return result


def summarize(rows: list[dict[str, object]]) -> None:
    print("\nper-run")
    for r in rows:
        print(r)

    print("\nsummary")
    by_label: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_label.setdefault(str(r["label"]), []).append(r)

    for label in ["thin", "fat"]:
        rs = by_label[label]
        times = [float(r["encode_s"]) for r in rs]
        print(
            f"{label:5s} min={min(times):.6f}s "
            f"mean={statistics.mean(times):.6f}s "
            f"median={statistics.median(times):.6f}s "
            f"runs={[round(t, 6) for t in times]}"
        )
        print(
            f"      parks={[r.get('wpp_row_parks') for r in rs]} "
            f"takeovers={[r.get('wpp_row_takeovers') for r in rs]} "
            f"wait_us={[r.get('phase_wpp_wait_us') for r in rs]}"
        )

    thin_hashes = {r["sha256"] for r in by_label["thin"]}
    fat_hashes = {r["sha256"] for r in by_label["fat"]}
    print("\nbyte identity")
    print(f"thin hashes: {sorted(thin_hashes)}")
    print(f"fat hashes:  {sorted(fat_hashes)}")
    if thin_hashes == fat_hashes and len(thin_hashes) == 1:
        print("OK: all thin/fat outputs are byte-identical")
    else:
        print("MISMATCH: thin/fat output hashes differ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--thin", type=Path, required=True)
    p.add_argument("--fat", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--max-sources", type=int, default=1)
    p.add_argument("--qp", type=int, default=28)
    p.add_argument("--effort", default="slow")
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help="extra env assignment, e.g. BPG_WPP_HANDOFF_RELEASE=stall-only",
    )
    args = p.parse_args()

    extra_env: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        extra_env[k] = v

    args.work_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for round_idx in range(1, args.rounds + 1):
        order = [("thin", args.thin), ("fat", args.fat)]
        if round_idx % 2 == 0:
            order.reverse()
        for label, exe in order:
            print(f"RUN round={round_idx} label={label}", flush=True)
            rows.append(run_one(label, exe, round_idx, args, extra_env))

    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
