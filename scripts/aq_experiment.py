#!/usr/bin/env python3
"""Run still265 AQ experiments against a uniform-QP baseline.

This is intentionally a measurement harness, not a preset. It exercises the
immutable external AQ map path (`BPG_AQ_OFFSET_MAP`) and answers the important
question: at equal bytes, does an AQ curve beat the uniform-QP curve?

Correctness gate: nonzero external 4:2:0 maps are covered by
`external_aq_correctness.rs` (zero, constant +/-1, single-QG, mixed maps, plus
the SAO/deblock replay path). Re-run that test before trusting AQ sweeps after
touching the QP/final-RDOQ/slice-replay code.

AQ rows are also decoded with both stock `bpgdec` and `bpg-tools decode`; the
stock/Rust PSNR is recorded as a diagnostic, and can be made fatal with
`--fail-stock-rust-check`. The RD curves themselves are measured from stock
`bpgdec` output.

Outputs under --outdir:
  results.csv              per encode + metric row
  aq_map_stats.csv         min/mean/max dQP and below/above-base shares
  equal_byte_deltas.csv    AQ metric minus uniform interpolation at same bytes
  bdrate_vs_uniform.csv    BD-rate of each AQ variant vs uniform per metric
  maps/*.csv               generated x,y,offset AQ maps

Example:
  python3 scripts/aq_experiment.py \
    --images 'test-set/test-set-4mp/*.png' \
    --qps 24,26,28,30,32,34 \
    --variants perceptual,psnr
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"Pillow is required: {exc}") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import metrics as metric_panel
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"scripts/metrics.py dependencies are required: {exc}") from exc

QG = 32
METRICS = [
    "psnr_rgb",
    "psnr_y",
    "ms_ssim",
    "ms_ssim_db",
    "vifp",
    "psnr_hvs_m",
    "vmaf",
    "xpsnr",
]


@dataclass(frozen=True)
class Variant:
    name: str
    map_mode: str | None
    strength: float = 0.35
    # When set, this is an in-tree `--aq <cli_aq>` variant (e.g. "two-pass")
    # encoded directly by bpg-tools, not an external offset map. `strength`/
    # `clamp` are forwarded as --aq-strength/--aq-clamp overrides.
    cli_aq: str | None = None
    clamp: int = 2
    # psy-rd / psy-rdoq strengths forwarded as --psy-rd/--psy-rdoq when > 0.
    # Composable with cli_aq (e.g. two-pass AQ + psy-rd in one variant).
    psy_rd: float = 0.0
    psy_rdoq: float = 0.0
    # AQ quantization-group size forwarded as --aq-qg when 16 (32 = default).
    aq_qg: int = 32


def parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(p) for p in parse_csv(raw)]


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


def image_list(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        base = ROOT if not Path(pattern).is_absolute() else Path("/")
        matches = sorted(base.glob(pattern) if base == ROOT else Path("/").glob(pattern.lstrip("/")))
        out.extend(p for p in matches if p.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if not unique:
        raise SystemExit(f"no images matched: {list(patterns)}")
    return unique


def safe_stem(path: Path) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(path.with_suffix("")))


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def ffmpeg_psnr_rgb(ffmpeg: str, ref: Path, test: Path) -> float:
    proc = run([ffmpeg, "-hide_banner", "-i", str(ref), "-i", str(test), "-lavfi", "psnr", "-f", "null", "-"])
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg psnr failed\n{proc.stderr}")
    m = re.search(r"average:([0-9.]+|inf)", proc.stderr)
    if not m:
        raise RuntimeError(f"ffmpeg psnr output did not contain an average PSNR\n{proc.stderr}")
    return math.inf if m.group(1) == "inf" else float(m.group(1))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def luma_grid(path: Path) -> tuple[int, int, list[int]]:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        pix = list(rgb.getdata())
    y = [
        int(round(0.2126 * r + 0.7152 * g + 0.0722 * b))
        for (r, g, b) in pix
    ]
    return w, h, y


def chroma_grids(path: Path) -> tuple[int, int, list[int], list[int]]:
    """Full-resolution Cb/Cr planes (8-bit) from the source, via YCbCr.

    The chroma activity map is computed at full resolution and in 8-bit units
    regardless of the eventual 4:2:0/4:2:2/4:4:4 subsampling, mirroring the
    in-tree `chroma_cell_activity` (which works in 8-bit units too)."""
    with Image.open(path) as im:
        ycc = im.convert("YCbCr")
        w, h = ycc.size
        pix = list(ycc.getdata())
    cb = [p[1] for p in pix]
    cr = [p[2] for p in pix]
    return w, h, cb, cr


def block_variance(y: list[int], w: int, h: int, x0: int, y0: int, n: int = QG) -> float:
    xs = range(x0, min(x0 + n, w))
    ys = range(y0, min(y0 + n, h))
    vals = [y[yy * w + xx] for yy in ys for xx in xs]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) * (v - mean) for v in vals) / len(vals)


# Chroma weight applied to the chroma log-activity term — mirrors the in-tree
# `preanalysis::CHROMA_AQ_WEIGHT`.
CHROMA_AQ_WEIGHT = 0.35


def make_aq_map(
    image: Path,
    out: Path,
    *,
    mode: str,
    strength: float,
    clamp: int,
    shift: int,
) -> dict[str, object]:
    w, h, y = luma_grid(image)
    cells_x = math.ceil(w / QG)
    cells_y = math.ceil(h / QG)
    vars_: list[float] = []
    for qgy in range(cells_y):
        for qgx in range(cells_x):
            vars_.append(block_variance(y, w, h, qgx * QG, qgy * QG))
    logs = [math.log2(1.0 + v) for v in vars_]
    mean_log = sum(logs) / max(1, len(logs))

    # Chroma-aware activity = luma log-variance + weighted chroma log-activity,
    # only computed when needed (the YCbCr conversion is not free).
    activity: list[float] = logs
    mean_activity = mean_log
    if mode == "perceptual_chroma":
        cw, ch, cb, cr = chroma_grids(image)
        chroma_logs: list[float] = []
        for qgy in range(cells_y):
            for qgx in range(cells_x):
                vcb = block_variance(cb, cw, ch, qgx * QG, qgy * QG)
                vcr = block_variance(cr, cw, ch, qgx * QG, qgy * QG)
                chroma_logs.append(math.log2(1.0 + vcb + vcr))
        activity = [lv + CHROMA_AQ_WEIGHT * cl for lv, cl in zip(logs, chroma_logs)]
        mean_activity = sum(activity) / max(1, len(activity))

    offsets: list[int] = []
    rows: list[str] = [
        "# x,y,offset,mode,variance,log2var,mean_log2var,activity,mean_activity",
        f"# mode={mode} strength={strength} clamp={clamp} shift={shift}",
    ]
    for idx, (var, logv, act) in enumerate(zip(vars_, logs, activity)):
        qgx = idx % cells_x
        qgy = idx // cells_x
        if mode == "perceptual":
            raw = strength * (logv - mean_log)
        elif mode == "perceptual_chroma":
            raw = strength * (act - mean_activity)
        elif mode == "psnr":
            raw = -strength * (logv - mean_log)
        elif mode == "positive":
            # Explicit positive-only shrinker for comparison. This should not
            # be mistaken for quality-preserving AQ unless it beats uniform QP
            # at equal bytes.
            rank = 0.0 if max(logs) == min(logs) else (logv - min(logs)) / (max(logs) - min(logs))
            raw = 1.0 + 7.0 * rank
        else:
            raise SystemExit(
                f"unknown AQ variant {mode!r}; use perceptual, perceptual_chroma, psnr, positive"
            )
        off = int(round(raw)) + shift
        off = max(-clamp, min(clamp, off))
        offsets.append(off)
        rows.append(
            f"{qgx * QG},{qgy * QG},{off},{mode},{var:.6f},{logv:.6f},{mean_log:.6f},"
            f"{act:.6f},{mean_activity:.6f}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    total = len(offsets) or 1
    below = sum(1 for v in offsets if v < 0)
    above = sum(1 for v in offsets if v > 0)
    zero = sum(1 for v in offsets if v == 0)
    return {
        "map_path": str(out),
        "mode": mode,
        "qgs": len(offsets),
        "min_dqp": min(offsets) if offsets else 0,
        "mean_dqp": round(sum(offsets) / total, 5),
        "max_dqp": max(offsets) if offsets else 0,
        "below_base_qgs": below,
        "at_base_qgs": zero,
        "above_base_qgs": above,
        "below_base_pct": round(100.0 * below / total, 3),
        "above_base_pct": round(100.0 * above / total, 3),
    }


def encode_decode_measure(
    *,
    bpg_tools: str,
    bpgdec: str,
    ffmpeg: str,
    image: Path,
    outdir: Path,
    variant: Variant,
    qp: int,
    fmt: str,
    map_path: Path | None,
    effort: str,
    keep_artifacts: bool,
    no_vmaf: bool,
    no_piq: bool,
    check_stock_rust: bool,
    fail_stock_rust_check: bool,
    stock_rust_min_psnr: float,
) -> dict[str, object]:
    stem = safe_stem(image.relative_to(ROOT) if image.is_relative_to(ROOT) else image)
    base = f"{stem}_{fmt}_q{qp}_{variant.name}"
    bpg = outdir / "artifacts" / f"{base}.bpg"
    png = outdir / "artifacts" / f"{base}.png"
    rust_png = outdir / "artifacts" / f"{base}.rust.png"
    bpg.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if map_path is not None:
        env["BPG_AQ_OFFSET_MAP"] = str(map_path)
    cmd = [
        bpg_tools,
        "encode",
        str(image),
        "-o",
        str(bpg),
        "--effort",
        effort,
        "-q",
        str(qp),
        "-f",
        fmt,
    ]
    if variant.cli_aq is not None:
        cmd += [
            "--aq",
            variant.cli_aq,
            "--aq-strength",
            f"{variant.strength}",
            "--aq-clamp",
            f"{variant.clamp}",
        ]
    if variant.psy_rd > 0.0:
        cmd += ["--psy-rd", f"{variant.psy_rd}"]
    if variant.psy_rdoq > 0.0:
        cmd += ["--psy-rdoq", f"{variant.psy_rdoq}"]
    if variant.aq_qg == 16:
        cmd += ["--aq-qg", "16"]
    is_aq = map_path is not None or variant.cli_aq is not None or variant.psy_rd > 0.0 or variant.psy_rdoq > 0.0
    t0 = time.perf_counter()
    enc = run(cmd, env=env)
    if enc.returncode != 0:
        raise RuntimeError(f"encode failed for {variant.name} q{qp}\n{enc.stderr}")
    dec = run([bpgdec, "-o", str(png), str(bpg)])
    if dec.returncode != 0:
        raise RuntimeError(f"decode failed for {variant.name} q{qp}\n{dec.stderr}")
    stock_rust_psnr: str | float = ""
    if check_stock_rust and is_aq:
        rust_dec = run([bpg_tools, "decode", str(bpg), "-o", str(rust_png), "--format", "rgb"])
        if rust_dec.returncode != 0:
            raise RuntimeError(f"rust decode failed for {variant.name} q{qp}\n{rust_dec.stderr}")
        psnr = ffmpeg_psnr_rgb(ffmpeg, png, rust_png)
        stock_rust_psnr = "inf" if math.isinf(psnr) else round(psnr, 6)
        if psnr < stock_rust_min_psnr:
            msg = (
                f"warning: stock bpgdec and Rust decode differ for {variant.name} q{qp}: "
                f"RGB PSNR {psnr:.3f} dB < {stock_rust_min_psnr:.3f} dB"
            )
            if fail_stock_rust_check:
                raise RuntimeError(msg.removeprefix("warning: "))
            print(msg, file=sys.stderr)
    elapsed = time.perf_counter() - t0
    panel = metric_panel.compute_panel(
        png,
        image,
        ffmpeg=ffmpeg,
        want_vmaf=not no_vmaf,
        want_piq=not no_piq,
    )
    size = bpg.stat().st_size
    with Image.open(image) as im:
        w, h = im.size
    row: dict[str, object] = {
        "image": str(image.relative_to(ROOT) if image.is_relative_to(ROOT) else image),
        "width": w,
        "height": h,
        "pixels": w * h,
        "format": fmt,
        "qp": qp,
        "variant": variant.name,
        "size_bytes": size,
        "bpp": round(size * 8.0 / (w * h), 8),
        "encode_decode_s": round(elapsed, 5),
        "stock_rust_psnr_rgb": stock_rust_psnr,
    }
    for key in METRICS:
        val = panel.get(key)
        row[key] = "" if val is None else round(float(val), 6)
    if keep_artifacts:
        row["bpg_path"] = str(bpg)
        row["png_path"] = str(png)
    else:
        bpg.unlink(missing_ok=True)
        png.unlink(missing_ok=True)
        rust_png.unlink(missing_ok=True)
        row["bpg_path"] = ""
        row["png_path"] = ""
    return row


def interp_uniform(rows: list[dict[str, object]], image: str, fmt: str, metric: str, bpp: float) -> float | None:
    pts = [
        (float(r["bpp"]), float(r[metric]))
        for r in rows
        if r["image"] == image and r["format"] == fmt and r["variant"] == "uniform" and r.get(metric) not in ("", None)
    ]
    if len(pts) < 2:
        return None
    pts.sort()
    if bpp < pts[0][0] or bpp > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= bpp <= x1:
            if x1 == x0:
                return y0
            # Interpolate quality over log-rate, which tracks RD curves better
            # than linear byte interpolation.
            t = (math.log(bpp) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return None


def write_equal_byte_deltas(rows: list[dict[str, object]], outdir: Path) -> None:
    out: list[dict[str, object]] = []
    for r in rows:
        if r["variant"] == "uniform":
            continue
        rec: dict[str, object] = {
            "image": r["image"],
            "format": r["format"],
            "qp": r["qp"],
            "variant": r["variant"],
            "size_bytes": r["size_bytes"],
            "bpp": r["bpp"],
        }
        for metric in METRICS:
            if r.get(metric) in ("", None):
                rec[f"{metric}_delta_vs_uniform_equal_bytes"] = ""
                continue
            base = interp_uniform(rows, str(r["image"]), str(r["format"]), metric, float(r["bpp"]))
            rec[f"{metric}_delta_vs_uniform_equal_bytes"] = (
                "" if base is None else round(float(r[metric]) - base, 6)
            )
        out.append(rec)
    fields = ["image", "format", "qp", "variant", "size_bytes", "bpp"] + [
        f"{m}_delta_vs_uniform_equal_bytes" for m in METRICS
    ]
    write_csv(outdir / "equal_byte_deltas.csv", out, fields)


def write_bdrate(rows: list[dict[str, object]], outdir: Path) -> None:
    out: list[dict[str, object]] = []
    keys = sorted({(r["image"], r["format"]) for r in rows})
    variants = sorted({r["variant"] for r in rows if r["variant"] != "uniform"})
    for image, fmt in keys:
        base = [r for r in rows if r["image"] == image and r["format"] == fmt and r["variant"] == "uniform"]
        for variant in variants:
            test = [r for r in rows if r["image"] == image and r["format"] == fmt and r["variant"] == variant]
            if len(base) < 4 or len(test) < 4:
                continue
            rec: dict[str, object] = {"image": image, "format": fmt, "variant": variant}
            for metric in METRICS:
                try:
                    br = [float(r["bpp"]) for r in base if r.get(metric) not in ("", None)]
                    bm = [float(r[metric]) for r in base if r.get(metric) not in ("", None)]
                    tr = [float(r["bpp"]) for r in test if r.get(metric) not in ("", None)]
                    tm = [float(r[metric]) for r in test if r.get(metric) not in ("", None)]
                    bd = metric_panel.bd_rate(br, bm, tr, tm)
                except Exception:
                    bd = None
                rec[f"bdrate_{metric}"] = "" if bd is None else round(bd, 4)
            out.append(rec)
    fields = ["image", "format", "variant"] + [f"bdrate_{m}" for m in METRICS]
    write_csv(outdir / "bdrate_vs_uniform.csv", out, fields)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="test-set/test-set-4mp/*.png")
    ap.add_argument("--qps", default="24,26,28,30,32,34")
    ap.add_argument("--formats", default="420")
    ap.add_argument("--variants", default="perceptual,psnr,positive")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument(
        "--strengths",
        default="",
        help="comma list of AQ strengths to sweep, e.g. 0.15,0.20,0.25,0.35. "
        "When set, each variant is emitted once per strength as <mode>_s<NNN> "
        "(0.20 -> _s020); overrides --strength.",
    )
    ap.add_argument("--clamp", type=int, default=2)
    ap.add_argument("--effort", default="slow", help="bpg-tools --effort (fast|slow|placebo)")
    ap.add_argument("--shift", type=int, default=0, help="integer offset added to every generated dQP")
    ap.add_argument("--outdir", default=str(ROOT / "target/aq-experiment"))
    ap.add_argument("--bpg-tools", default=str(ROOT / "target/release/bpg-tools"))
    ap.add_argument("--bpgdec", default="bpgdec")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--keep-artifacts", action="store_true")
    ap.add_argument("--no-vmaf", action="store_true")
    ap.add_argument("--no-piq", action="store_true")
    ap.add_argument(
        "--skip-stock-rust-check",
        action="store_true",
        help="do not compare stock bpgdec output against bpg-tools decode for AQ rows",
    )
    ap.add_argument(
        "--stock-rust-min-psnr",
        type=float,
        default=40.0,
        help="diagnostic RGB PSNR threshold between stock and Rust decodes for AQ rows",
    )
    ap.add_argument(
        "--fail-stock-rust-check",
        action="store_true",
        help="abort when stock/Rust decode PSNR is below --stock-rust-min-psnr",
    )
    ap.add_argument(
        "--allow-unsafe-nonzero-aq",
        action="store_true",
        help=argparse.SUPPRESS,  # compatibility no-op; nonzero external AQ is now correctness-tested
    )
    args = ap.parse_args()

    requested_variants = parse_csv(args.variants)

    bpg_tools = require_tool(args.bpg_tools, "bpg-tools")
    bpgdec = require_tool(args.bpgdec, "bpgdec")
    ffmpeg = require_tool(args.ffmpeg, "ffmpeg")
    images = image_list(parse_csv(args.images))
    qps = parse_ints(args.qps)
    fmts = parse_csv(args.formats)

    sweep = [float(s) for s in parse_csv(args.strengths)] if args.strengths.strip() else None

    def make_variant(v: str, name: str, strength: float) -> Variant:
        # A trailing "+qg16" on any token requests 16x16 quantization groups.
        # "cli:<arg>" → in-tree `--aq <arg>` variant (e.g. cli:two-pass);
        # "psy:<rd>[:<rdoq>][:aq=<preset>]" → psy-rd/psy-rdoq variant, its
        # optional aq= suffix composing with in-tree AQ (--strengths does not
        # apply to psy:* variants — the psy values are in the token itself);
        # otherwise an external offset-map variant whose map_mode is the name.
        aq_qg = 32
        if "+qg16" in v:
            aq_qg = 16
            v = v.replace("+qg16", "")
        if v.startswith("psy:"):
            parts = v[len("psy:"):].split(":")
            psy_rd = float(parts[0])
            psy_rdoq = float(parts[1]) if len(parts) > 1 and not parts[1].startswith("aq=") else 0.0
            aq = next((p[len("aq="):] for p in parts[1:] if p.startswith("aq=")), None)
            return Variant(
                name.replace(":", "_").replace("=", ""),
                None,
                strength,
                cli_aq=aq,
                clamp=args.clamp,
                psy_rd=psy_rd,
                psy_rdoq=psy_rdoq,
                aq_qg=aq_qg,
            )
        if v.startswith("cli:"):
            arg = v[len("cli:"):]
            return Variant(
                name.replace("cli:", ""), None, strength, cli_aq=arg, clamp=args.clamp, aq_qg=aq_qg
            )
        return Variant(name, v, strength, clamp=args.clamp, aq_qg=aq_qg)

    aq_variants: list[Variant] = []
    for v in requested_variants:
        if sweep is None:
            aq_variants.append(make_variant(v, v, args.strength))
        else:
            for s in sweep:
                suffix = f"s{int(round(s * 100)):03d}"
                aq_variants.append(make_variant(v, f"{v}_{suffix}", s))
    variants = [Variant("uniform", None)] + aq_variants
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    map_stats: list[dict[str, object]] = []
    result_fields = [
        "image",
        "width",
        "height",
        "pixels",
        "format",
        "qp",
        "variant",
        "size_bytes",
        "bpp",
        "encode_decode_s",
        "stock_rust_psnr_rgb",
        *METRICS,
        "bpg_path",
        "png_path",
    ]
    map_fields = [
        "image",
        "format",
        "variant",
        "map_path",
        "mode",
        "qgs",
        "min_dqp",
        "mean_dqp",
        "max_dqp",
        "below_base_qgs",
        "at_base_qgs",
        "above_base_qgs",
        "below_base_pct",
        "above_base_pct",
    ]

    for image in images:
        rel = image.relative_to(ROOT) if image.is_relative_to(ROOT) else image
        for fmt in fmts:
            maps: dict[str, Path] = {}
            for variant in variants:
                if variant.map_mode is None:
                    continue
                map_path = outdir / "maps" / f"{safe_stem(Path(str(rel)))}_{fmt}_{variant.name}.csv"
                stats = make_aq_map(
                    image,
                    map_path,
                    mode=variant.map_mode,
                    strength=variant.strength,
                    clamp=args.clamp,
                    shift=args.shift,
                )
                stats.update({"image": str(rel), "format": fmt, "variant": variant.name})
                map_stats.append(stats)
                maps[variant.name] = map_path
            write_csv(outdir / "aq_map_stats.csv", map_stats, map_fields)
            for qp in qps:
                for variant in variants:
                    row = encode_decode_measure(
                        bpg_tools=bpg_tools,
                        bpgdec=bpgdec,
                        ffmpeg=ffmpeg,
                        image=image,
                        outdir=outdir,
                        variant=variant,
                        qp=qp,
                        fmt=fmt,
                        map_path=maps.get(variant.name),
                        effort=args.effort,
                        keep_artifacts=args.keep_artifacts,
                        no_vmaf=args.no_vmaf,
                        no_piq=args.no_piq,
                        check_stock_rust=not args.skip_stock_rust_check,
                        fail_stock_rust_check=args.fail_stock_rust_check,
                        stock_rust_min_psnr=args.stock_rust_min_psnr,
                    )
                    rows.append(row)
                    write_csv(outdir / "results.csv", rows, result_fields)
                    write_equal_byte_deltas(rows, outdir)
                    write_bdrate(rows, outdir)
                    print(json.dumps(row, sort_keys=True), flush=True)

    write_csv(outdir / "aq_map_stats.csv", map_stats, map_fields)
    write_equal_byte_deltas(rows, outdir)
    write_bdrate(rows, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
