#!/usr/bin/env python3
"""Rust still265 vs BPG-C/x265 heads-up quality benchmark.

This intentionally excludes JCTVC. It compares:

  - Rust: bpg-tools encode --effort <effort>
  - BPG-C: bpgenc -e x265 -m <level>

For every input image / chroma format / QP, the script records encode time,
output size, and a full-reference metric panel: ffmpeg PSNR/SSIM-RGB plus (via
scripts/metrics.py) PSNR-Y, MS-SSIM (+dB), VIFp, PSNR-HVS-M, VMAF, XPSNR, and
optionally Butteraugli / piq learned metrics. Outputs:

  results.csv               per-encode rows (all metrics)
  summary_by_variant.csv    per-variant means
  metric_aggregates.csv     per-variant mean / median / worst-image per metric
  nearest_size_rust_vs_x265 equal-size deltas
  bdrate_by_image.csv       Bjontegaard BD-rate (rust vs x265) per metric

The metric panel adds ~1s/image (mostly VMAF); use --no-extra-metrics for the
ffmpeg-only fast path, or --no-vmaf to skip just the ffmpeg perceptual metrics.
It is serial by default so timings are less noisy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

try:
    import metrics as metric_panel
except Exception:  # pragma: no cover - metrics deps optional
    metric_panel = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = "test-set/**/*.png"
DEFAULT_QPS = "28"
DEFAULT_FORMATS = "420"
DEFAULT_RUST_EFFORTS = "best"
DEFAULT_X265_LEVELS = "9"
DEFAULT_BPG_TOOLS = ROOT / "target/release" / ("bpg-tools.exe" if os.name == "nt" else "bpg-tools")

# Perceptual metric-panel columns added beyond the ffmpeg psnr_rgb/ssim_rgb.
# `True` => higher is better (used for worst-image and BD-metric aggregation).
EXTRA_METRIC_DIRECTION = {
    "psnr_y": True,
    "ms_ssim": True,
    "ms_ssim_db": True,
    "vifp": True,
    "psnr_hvs_m": True,
    "vmaf": True,
    "xpsnr": True,
    "butteraugli": False,
    "haarpsi": True,
    "dists": False,
    "lpips": False,
}
EXTRA_METRIC_KEYS = list(EXTRA_METRIC_DIRECTION)


@dataclass(frozen=True)
class Variant:
    encoder: str
    setting: str

    @property
    def label(self) -> str:
        return f"{self.encoder}-{self.setting}"


def parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in parse_csv_list(raw)]


def parse_env_assignments(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        for part in parse_csv_list(item):
            if "=" not in part:
                raise SystemExit(f"--rust-env must be KEY=VALUE, got: {part}")
            key, value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise SystemExit(f"--rust-env has empty key: {part}")
            out[key] = value
    return out


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def require_tool(path_or_name: str, label: str) -> str:
    path = Path(path_or_name)
    if path.parent != Path(".") or path.is_absolute():
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        raise SystemExit(f"{label} is not executable: {path_or_name}")
    found = shutil.which(path_or_name)
    if found:
        return found
    raise SystemExit(f"{label} not found in PATH: {path_or_name}")


def image_list(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        base = ROOT if not Path(pattern).is_absolute() else Path("/")
        matches = sorted(base.glob(pattern) if base == ROOT else Path("/").glob(pattern.lstrip("/")))
        out.extend(p for p in matches if p.is_file())
    # Preserve order while removing duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if not unique:
        raise SystemExit(f"no images matched: {patterns}")
    return unique


def ffmpeg_metrics(ffmpeg: str, decoded_png: Path, original_png: Path) -> tuple[float, float]:
    # Force RGB24 on both sides so the metric is over the same displayed image
    # representation, independent of each decoder's PNG color-type choice.
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(decoded_png),
        "-i",
        str(original_png),
        "-filter_complex",
        "[0:v]format=rgb24,split=2[dec1][dec2];"
        "[1:v]format=rgb24,split=2[ref1][ref2];"
        "[dec1][ref1]ssim=stats_file=-;[dec2][ref2]psnr=stats_file=-",
        "-f",
        "null",
        "-",
    ]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg metrics failed for {decoded_png}\n{r.stderr}")

    psnr = math.nan
    ssim = math.nan
    for line in (r.stderr + "\n" + r.stdout).splitlines():
        if "Parsed_psnr" in line and "average:" in line:
            m = re.search(r"average:([0-9.]+|inf)", line)
            if m:
                psnr = float("inf") if m.group(1) == "inf" else float(m.group(1))
        elif "Parsed_ssim" in line and "All:" in line:
            m = re.search(r"All:([0-9.]+)", line)
            if m:
                ssim = float(m.group(1))
    if math.isnan(psnr) or math.isnan(ssim):
        raise RuntimeError(f"could not parse ffmpeg PSNR/SSIM for {decoded_png}")
    return psnr, ssim


def encode_rust(
    bpg_tools: str,
    image: Path,
    out_bpg: Path,
    effort: str,
    qp: int,
    fmt: str,
    extra_args: list[str],
    debug_stats_csv: Path | None,
    extra_env: dict[str, str],
) -> tuple[float, str]:
    cmd = [
        bpg_tools,
        "encode",
        str(image),
        "-o",
        str(out_bpg),
        "--effort",
        effort,
        "--qp",
        str(qp),
        "--format",
        fmt,
    ]
    if debug_stats_csv is not None:
        cmd.extend(["--debug-stats-csv", str(debug_stats_csv)])
    cmd.extend(extra_args)
    env = os.environ.copy()
    env.update(extra_env)
    t0 = time.perf_counter()
    r = run(cmd, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"Rust encode failed: {' '.join(cmd)}\n{r.stderr}\n{r.stdout}")
    return elapsed, r.stderr + r.stdout


def encode_x265(
    bpgenc: str,
    image: Path,
    out_bpg: Path,
    level: int,
    qp: int,
    fmt: str,
    extra_args: list[str],
) -> tuple[float, str]:
    cmd = [
        bpgenc,
        "-e",
        "x265",
        "-m",
        str(level),
        "-q",
        str(qp),
        "-f",
        fmt,
        "-o",
        str(out_bpg),
        str(image),
        *extra_args,
    ]
    t0 = time.perf_counter()
    r = run(cmd)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"BPG-C/x265 encode failed: {' '.join(cmd)}\n{r.stderr}\n{r.stdout}")
    return elapsed, r.stderr + r.stdout


def decode_bpg(bpgdec: str, in_bpg: Path, out_png: Path) -> None:
    r = run([bpgdec, "-o", str(out_png), str(in_bpg)])
    if r.returncode != 0:
        raise RuntimeError(f"bpgdec failed for {in_bpg}\n{r.stderr}\n{r.stdout}")


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.with_suffix("").as_posix())


def row_key(row: dict[str, object]) -> tuple[str, str, int, str]:
    return (
        str(row["image"]),
        str(row["format"]),
        int(row["qp"]),
        str(row["encoder"]),
    )


def average(rows: Iterable[dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return mean(vals) if vals else math.nan


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _values(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v in ("", None):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def summarize(rows: list[dict[str, object]], outdir: Path) -> None:
    # Which extra metrics actually have data in this run.
    present_metrics = [k for k in EXTRA_METRIC_KEYS if _values(rows, k)]

    summary_rows: list[dict[str, object]] = []
    by_variant: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["encoder"]), str(row["setting"]), str(row["format"]))
        by_variant.setdefault(key, []).append(row)
    for (encoder, setting, fmt), group in sorted(by_variant.items()):
        srow = {
            "encoder": encoder,
            "setting": setting,
            "format": fmt,
            "runs": len(group),
            "avg_size_bytes": round(average(group, "size_bytes"), 3),
            "avg_bpp": round(average(group, "bpp"), 6),
            "avg_encode_s": round(average(group, "encode_s"), 3),
            "avg_psnr_rgb": round(average(group, "psnr_rgb"), 5),
            "avg_ssim_rgb": round(average(group, "ssim_rgb"), 7),
        }
        for m in present_metrics:
            vals = _values(group, m)
            srow[f"avg_{m}"] = round(mean(vals), 6) if vals else ""
        summary_rows.append(srow)

    write_csv(
        outdir / "summary_by_variant.csv",
        summary_rows,
        [
            "encoder",
            "setting",
            "format",
            "runs",
            "avg_size_bytes",
            "avg_bpp",
            "avg_encode_s",
            "avg_psnr_rgb",
            "avg_ssim_rgb",
            *[f"avg_{m}" for m in present_metrics],
        ],
    )

    # Per-variant mean / median / worst for each metric (incl. PSNR/SSIM-RGB).
    # "worst" = the least-favourable image (min for higher-is-better metrics).
    agg_metrics: list[tuple[str, bool]] = [("psnr_rgb", True), ("ssim_rgb", True)] + [
        (m, EXTRA_METRIC_DIRECTION[m]) for m in present_metrics
    ]
    agg_rows: list[dict[str, object]] = []
    for (encoder, setting, fmt), group in sorted(by_variant.items()):
        for metric, higher_better in agg_metrics:
            vals = _values(group, metric)
            if not vals:
                continue
            worst = min(vals) if higher_better else max(vals)
            agg_rows.append(
                {
                    "encoder": encoder,
                    "setting": setting,
                    "format": fmt,
                    "metric": metric,
                    "higher_is_better": higher_better,
                    "mean": round(mean(vals), 6),
                    "median": round(median(vals), 6),
                    "worst": round(worst, 6),
                    "runs": len(vals),
                }
            )
    write_csv(
        outdir / "metric_aggregates.csv",
        agg_rows,
        ["encoder", "setting", "format", "metric", "higher_is_better",
         "mean", "median", "worst", "runs"],
    )

    _write_bdrate(rows, outdir, present_metrics)


def _write_bdrate(rows: list[dict[str, object]], outdir: Path, present_metrics: list[str]) -> None:
    """Per-image BD-rate (Bjontegaard) of each non-x265 variant vs x265 over the
    QP sweep, for the higher-is-better quality metrics. Negative = the test
    encoder needs fewer bytes at equal quality (test is better)."""
    if metric_panel is None:
        return
    bd_metrics = ["psnr_rgb"] + [m for m in present_metrics if EXTRA_METRIC_DIRECTION[m]]

    by_if: dict[tuple[str, str], dict[str, dict[str, list[dict[str, object]]]]] = {}
    for r in rows:
        img, fmt, enc, setting = (
            str(r["image"]), str(r["format"]), str(r["encoder"]), str(r["setting"]),
        )
        by_if.setdefault((img, fmt), {}).setdefault(enc, {}).setdefault(setting, []).append(r)

    out: list[dict[str, object]] = []
    for (img, fmt), encs in sorted(by_if.items()):
        x265s = encs.get("x265", {})
        if not x265s:
            continue
        for tenc, tsettings in encs.items():
            if tenc == "x265":
                continue
            for tset, tgroup in tsettings.items():
                for xset, xgroup in x265s.items():
                    rec: dict[str, object] = {
                        "image": img,
                        "format": fmt,
                        "test": f"{tenc}-{tset}",
                        "ref": f"x265-{xset}",
                        "points": min(len(tgroup), len(xgroup)),
                    }
                    for metric in bd_metrics:
                        tp = [
                            (float(r["size_bytes"]), float(r[metric]))
                            for r in tgroup
                            if r.get(metric) not in ("", None)
                        ]
                        xp = [
                            (float(r["size_bytes"]), float(r[metric]))
                            for r in xgroup
                            if r.get(metric) not in ("", None)
                        ]
                        bd = None
                        if len(tp) >= 4 and len(xp) >= 4:
                            bd = metric_panel.bd_rate(
                                [s for s, _ in xp], [m for _, m in xp],
                                [s for s, _ in tp], [m for _, m in tp],
                            )
                        rec[f"bdrate_{metric}"] = round(bd, 4) if bd is not None else ""
                    out.append(rec)

    if out:
        write_csv(
            outdir / "bdrate_by_image.csv",
            out,
            ["image", "format", "test", "ref", "points"]
            + [f"bdrate_{m}" for m in bd_metrics],
        )

    nearest_rows: list[dict[str, object]] = []
    rust_rows = [r for r in rows if r["encoder"] == "rust"]
    x265_rows = [r for r in rows if r["encoder"] == "x265"]
    for rr in rust_rows:
        candidates = [
            xr
            for xr in x265_rows
            if xr["image"] == rr["image"] and xr["format"] == rr["format"]
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda xr: abs(float(xr["size_bytes"]) - float(rr["size_bytes"])))
        nearest_rows.append(
            {
                "image": rr["image"],
                "format": rr["format"],
                "rust_setting": rr["setting"],
                "rust_qp": rr["qp"],
                "rust_encode_qp": rr.get("encode_qp", rr["qp"]),
                "rust_size": rr["size_bytes"],
                "rust_psnr_rgb": rr["psnr_rgb"],
                "rust_ssim_rgb": rr["ssim_rgb"],
                "rust_encode_s": rr["encode_s"],
                "x265_setting": nearest["setting"],
                "x265_qp": nearest["qp"],
                "x265_encode_qp": nearest.get("encode_qp", nearest["qp"]),
                "x265_size": nearest["size_bytes"],
                "x265_psnr_rgb": nearest["psnr_rgb"],
                "x265_ssim_rgb": nearest["ssim_rgb"],
                "x265_encode_s": nearest["encode_s"],
                "size_delta_pct": round(
                    100.0 * (float(rr["size_bytes"]) - float(nearest["size_bytes"]))
                    / float(nearest["size_bytes"]),
                    5,
                ),
                "psnr_delta_rgb": round(float(rr["psnr_rgb"]) - float(nearest["psnr_rgb"]), 5),
                "ssim_delta_rgb": round(float(rr["ssim_rgb"]) - float(nearest["ssim_rgb"]), 7),
                "speed_ratio_rust_over_x265": round(
                    float(rr["encode_s"]) / max(float(nearest["encode_s"]), 1e-9), 5
                ),
            }
        )
    write_csv(
        outdir / "nearest_size_rust_vs_x265.csv",
        nearest_rows,
        [
            "image",
            "format",
            "rust_setting",
            "rust_qp",
            "rust_encode_qp",
            "rust_size",
            "rust_psnr_rgb",
            "rust_ssim_rgb",
            "rust_encode_s",
            "x265_setting",
            "x265_qp",
            "x265_encode_qp",
            "x265_size",
            "x265_psnr_rgb",
            "x265_ssim_rgb",
            "x265_encode_s",
            "size_delta_pct",
            "psnr_delta_rgb",
            "ssim_delta_rgb",
            "speed_ratio_rust_over_x265",
        ],
    )

    variance_rows: list[dict[str, object]] = []
    by_image: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_image.setdefault((str(row["image"]), str(row["format"])), []).append(row)
    for (image, fmt), group in sorted(by_image.items()):
        sizes = [float(r["size_bytes"]) for r in group]
        variance_rows.append(
            {
                "image": image,
                "format": fmt,
                "min_size": int(min(sizes)),
                "max_size": int(max(sizes)),
                "max_over_min": round(max(sizes) / max(min(sizes), 1.0), 5),
                "runs": len(group),
            }
        )
    write_csv(outdir / "size_variance_by_image.csv", variance_rows, ["image", "format", "min_size", "max_size", "max_over_min", "runs"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="comma-separated glob(s), default: %(default)s")
    ap.add_argument("--qps", default=DEFAULT_QPS, help="comma-separated QPs, default: %(default)s")
    ap.add_argument("--formats", default=DEFAULT_FORMATS, help="comma-separated chroma formats, default: %(default)s")
    ap.add_argument("--rust-efforts", default=DEFAULT_RUST_EFFORTS, help="comma-separated Rust efforts; 'reference' is rejected")
    ap.add_argument("--x265-levels", default=DEFAULT_X265_LEVELS, help="comma-separated BPG-C -m levels, default: %(default)s")
    ap.add_argument("--bpg-tools", default=str(DEFAULT_BPG_TOOLS))
    ap.add_argument("--bpgenc", default="bpgenc")
    ap.add_argument("--bpgdec", default="bpgdec")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--outdir", default=str(ROOT / "target/headsup-quality"))
    ap.add_argument("--rust-extra", default="", help="extra args appended to bpg-tools encode, shell-split on whitespace")
    ap.add_argument(
        "--rust-env",
        action="append",
        default=[],
        help="extra Rust encode environment assignment(s), KEY=VALUE; repeat or comma-separate",
    )
    ap.add_argument(
        "--rust-label-suffix",
        default="",
        help="suffix appended to Rust setting/variant labels for env-gated experiments, e.g. -ttcu",
    )
    ap.add_argument(
        "--rust-debug-stats-csv",
        default="",
        help="append still265 per-encode search counters to this CSV; relative paths are under --outdir",
    )
    ap.add_argument("--x265-extra", default="", help="extra args appended to bpgenc, shell-split on whitespace")
    ap.add_argument(
        "--x265-qp-offset",
        type=int,
        default=0,
        help="offset added to the QP passed to bpgenc; use 3 to match x265 Avg QP to Rust QP",
    )
    ap.add_argument("--keep-artifacts", action="store_true", help="keep decoded PNGs and BPG files")
    ap.add_argument("--resume", action="store_true", help="skip rows already present in results.csv")
    ap.add_argument(
        "--no-extra-metrics",
        action="store_true",
        help="skip the perceptual metric panel (MS-SSIM/VIFp/PSNR-HVS-M/VMAF/...); ffmpeg PSNR/SSIM only",
    )
    ap.add_argument("--no-vmaf", action="store_true", help="skip the ffmpeg libvmaf/xpsnr metrics (faster)")
    ap.add_argument("--no-piq", action="store_true", help="skip piq learned metrics (lpips/dists/haarpsi)")
    ap.add_argument(
        "--butteraugli-bin",
        default=None,
        help="path to a butteraugli binary for the perceptual diff metric (else $BUTTERAUGLI_BIN / PATH)",
    )
    args = ap.parse_args()

    efforts = parse_csv_list(args.rust_efforts)
    if any(e.lower() == "reference" for e in efforts):
        raise SystemExit("Rust 'reference' effort is intentionally excluded from this benchmark")

    qps = parse_int_list(args.qps)
    formats = parse_csv_list(args.formats)
    x265_levels = parse_int_list(args.x265_levels)
    images = image_list(parse_csv_list(args.images))

    bpg_tools = require_tool(args.bpg_tools, "bpg-tools")
    bpgenc = require_tool(args.bpgenc, "bpgenc")
    bpgdec = require_tool(args.bpgdec, "bpgdec")
    ffmpeg = require_tool(args.ffmpeg, "ffmpeg")

    rust_extra = shlex.split(args.rust_extra) if args.rust_extra else []
    rust_env = parse_env_assignments(args.rust_env)
    x265_extra = shlex.split(args.x265_extra) if args.x265_extra else []

    outdir = Path(args.outdir)
    artifacts = outdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    results_csv = outdir / "results.csv"
    rust_debug_stats_csv = Path(args.rust_debug_stats_csv) if args.rust_debug_stats_csv else None
    if rust_debug_stats_csv is not None and not rust_debug_stats_csv.is_absolute():
        rust_debug_stats_csv = outdir / rust_debug_stats_csv
    if rust_debug_stats_csv is not None:
        rust_debug_stats_csv.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            rust_debug_stats_csv.unlink(missing_ok=True)
    fieldnames = [
        "image",
        "width",
        "height",
        "pixels",
        "format",
        "qp",
        "encode_qp",
        "encoder",
        "setting",
        "variant",
        "size_bytes",
        "bpp",
        "encode_s",
        "psnr_rgb",
        "ssim_rgb",
        *EXTRA_METRIC_KEYS,
        "bpg_path",
    ]

    panel_enabled = metric_panel is not None and not args.no_extra_metrics
    if not args.no_extra_metrics and metric_panel is None:
        print("note: scripts/metrics.py unavailable (missing numpy/scipy?); "
              "extra metric panel disabled", flush=True)

    rows: list[dict[str, object]] = []
    existing_keys: set[tuple[str, str, str, int, int, str, str]] = set()
    if args.resume and results_csv.exists():
        with results_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing_keys.add(
                    (
                        row["image"],
                        row["format"],
                        row["encoder"],
                        int(row["qp"]),
                        int(row.get("encode_qp") or row["qp"]),
                        row["setting"],
                        row["variant"],
                    )
                )

    def emit(row: dict[str, object]) -> None:
        rows.append(row)
        write_csv(results_csv, rows, fieldnames)
        summarize(rows, outdir)
        print(json.dumps(row, sort_keys=True), flush=True)

    variants = [Variant("rust", e) for e in efforts] + [Variant("x265", f"m{m}") for m in x265_levels]

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment guard
        raise SystemExit(f"Pillow is required for image dimensions: {exc}") from exc

    for image in images:
        try:
            rel_image = image.resolve().relative_to(ROOT.resolve())
        except ValueError:
            rel_image = image
        with Image.open(image) as im:
            width, height = im.size
        pixels = width * height
        stem = safe_stem(Path(str(rel_image)))
        for fmt in formats:
            for qp in qps:
                for variant in variants:
                    encode_qp = qp + args.x265_qp_offset if variant.encoder == "x265" else qp
                    setting_label = variant.setting
                    variant_label = variant.label
                    if variant.encoder == "rust" and args.rust_label_suffix:
                        setting_label = f"{variant.setting}{args.rust_label_suffix}"
                        variant_label = f"rust-{setting_label}"
                    key = (
                        str(rel_image),
                        fmt,
                        variant.encoder,
                        qp,
                        encode_qp,
                        setting_label,
                        variant_label,
                    )
                    if key in existing_keys:
                        continue
                    base = f"{stem}_{fmt}_q{qp}_eq{encode_qp}_{variant_label}"
                    bpg_path = artifacts / f"{base}.bpg"
                    png_path = artifacts / f"{base}.png"
                    if variant.encoder == "rust":
                        elapsed, _ = encode_rust(
                            bpg_tools,
                            image,
                            bpg_path,
                            variant.setting,
                            qp,
                            fmt,
                            rust_extra,
                            rust_debug_stats_csv,
                            rust_env,
                        )
                    else:
                        level = int(variant.setting.removeprefix("m"))
                        elapsed, _ = encode_x265(
                            bpgenc,
                            image,
                            bpg_path,
                            level,
                            encode_qp,
                            fmt,
                            x265_extra,
                        )
                    decode_bpg(bpgdec, bpg_path, png_path)
                    psnr_rgb, ssim_rgb = ffmpeg_metrics(ffmpeg, png_path, image)
                    size = bpg_path.stat().st_size
                    row = {
                        "image": str(rel_image),
                        "width": width,
                        "height": height,
                        "pixels": pixels,
                        "format": fmt,
                        "qp": qp,
                        "encode_qp": encode_qp,
                        "encoder": variant.encoder,
                        "setting": setting_label,
                        "variant": variant_label,
                        "size_bytes": size,
                        "bpp": round((size * 8.0) / pixels, 8),
                        "encode_s": round(elapsed, 5),
                        "psnr_rgb": round(psnr_rgb, 5),
                        "ssim_rgb": round(ssim_rgb, 7),
                        "bpg_path": str(bpg_path if args.keep_artifacts else ""),
                    }
                    if panel_enabled:
                        try:
                            panel = metric_panel.compute_panel(
                                png_path,
                                image,
                                ffmpeg=ffmpeg,
                                butteraugli_bin=args.butteraugli_bin,
                                want_vmaf=not args.no_vmaf,
                                want_piq=not args.no_piq,
                            )
                            for key in EXTRA_METRIC_KEYS:
                                v = panel.get(key)
                                row[key] = "" if v is None else round(float(v), 6)
                        except Exception as exc:  # keep the sweep going
                            print(f"warning: metric panel failed for {png_path}: {exc}",
                                  flush=True)
                    if not args.keep_artifacts:
                        png_path.unlink(missing_ok=True)
                        bpg_path.unlink(missing_ok=True)
                    emit(row)

    metadata = {
        "images": [str(p) for p in images],
        "qps": qps,
        "formats": formats,
        "rust_efforts": efforts,
        "rust_env": rust_env,
        "rust_label_suffix": args.rust_label_suffix,
        "x265_levels": x265_levels,
        "x265_qp_offset": args.x265_qp_offset,
        "bpg_tools": bpg_tools,
        "bpgenc": bpgenc,
        "bpgdec": bpgdec,
        "ffmpeg": ffmpeg,
        "note": "JCTVC intentionally excluded; Rust reference effort intentionally excluded.",
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
