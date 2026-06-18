#!/usr/bin/env python3
"""Full-reference image-quality metric panel for the heads-up codec tester.

Separates pixel fidelity, structure, and perceptual quality so a PSNR loss can
be judged against structural/perceptual wins (see the metric-panel advice in
docs/). Every metric is best-effort: if a backend is missing the metric is
reported as ``None`` and the rest still run.

Always available (numpy/scipy/PIL):
  psnr_rgb, psnr_y, ms_ssim, ms_ssim_db, vifp, psnr_hvs_m
Via ffmpeg (libvmaf / xpsnr filters): vmaf, xpsnr
Optional (only if installed): butteraugli (external binary via $BUTTERAUGLI_BIN
  or --butteraugli-bin), and piq-based lpips/dists/haarpsi if `piq` imports.

Higher-is-better: psnr_*, ms_ssim*, vifp, vmaf, xpsnr, haarpsi.
Lower-is-better:  butteraugli, lpips, dists.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
    from scipy.fftpack import dct as _dct

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - environment guard
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Image loading / colour
# ---------------------------------------------------------------------------

def load_rgb(path) -> np.ndarray:
    """Load an image as float64 RGB in [0, 255], shape (H, W, 3)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def to_y(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma in [0, 255] (matches the common PSNR-HVS / VIF references)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------------------------------------------------------------------------
# Pixel fidelity
# ---------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(peak * peak / mse)


# ---------------------------------------------------------------------------
# Structure: SSIM / MS-SSIM (Wang 2003)
# ---------------------------------------------------------------------------

def _ssim_cs(x: np.ndarray, y: np.ndarray, c1: float, c2: float):
    """Return (mean SSIM, mean contrast-structure) over an 11-tap Gaussian."""
    mu_x = gaussian_filter(x, 1.5, truncate=3.5)
    mu_y = gaussian_filter(y, 1.5, truncate=3.5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = gaussian_filter(x * x, 1.5, truncate=3.5) - mu_x2
    sy = gaussian_filter(y * y, 1.5, truncate=3.5) - mu_y2
    sxy = gaussian_filter(x * y, 1.5, truncate=3.5) - mu_xy
    cs = (2 * sxy + c2) / (sx + sy + c2)
    ssim = ((2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs
    return float(ssim.mean()), float(cs.mean())


def _downsample2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0] - img.shape[0] % 2, img.shape[1] - img.shape[1] % 2
    a = img[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def ms_ssim(y1: np.ndarray, y2: np.ndarray, peak: float = 255.0):
    """5-scale MS-SSIM on luma; returns (ms_ssim, ms_ssim_db) or (None, None)."""
    if not _HAVE_SCIPY:
        return None, None
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    x, y = y1.astype(np.float64), y2.astype(np.float64)
    cs_vals, ssim_last = [], None
    n = len(weights)
    for i in range(n):
        if min(x.shape) < 11:
            n = i
            break
        ssim_full, cs = _ssim_cs(x, y, c1, c2)
        ssim_last = ssim_full
        cs_vals.append(max(cs, 1e-8))
        if i < len(weights) - 1:
            x, y = _downsample2(x), _downsample2(y)
    if not cs_vals or ssim_last is None:
        return None, None
    w = weights[:n]
    w = w / w.sum()
    val = 1.0
    for i in range(n - 1):
        val *= cs_vals[i] ** w[i]
    val *= max(ssim_last, 1e-8) ** w[n - 1]
    val = float(min(max(val, 0.0), 1.0))
    db = -10.0 * math.log10(1.0 - val) if val < 1.0 else float("inf")
    return val, db


# ---------------------------------------------------------------------------
# Perceptual: VIFp (Sheikh & Bovik, pixel-domain multi-scale)
# ---------------------------------------------------------------------------

def vifp(ref: np.ndarray, dist: np.ndarray) -> float | None:
    if not _HAVE_SCIPY:
        return None
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)
    eps = 1e-10
    sigma_nsq = 2.0
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sd = n / 5.0
        if scale > 1:
            ref = gaussian_filter(ref, sd)[::2, ::2]
            dist = gaussian_filter(dist, sd)[::2, ::2]
        if min(ref.shape) < 2:
            break
        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = gaussian_filter(ref * ref, sd) - mu1_sq
        s2 = gaussian_filter(dist * dist, sd) - mu2_sq
        s12 = gaussian_filter(ref * dist, sd) - mu1_mu2
        s1 = np.clip(s1, 0, None)
        s2 = np.clip(s2, 0, None)
        g = s12 / (s1 + eps)
        sv = s2 - g * s12
        g = np.where(s1 < eps, 0.0, g)
        sv = np.where(s1 < eps, s2, sv)
        s1 = np.where(s1 < eps, 0.0, s1)
        g = np.where(s2 < eps, 0.0, g)
        sv = np.where(s2 < eps, 0.0, sv)
        sv = np.clip(sv, eps, None)
        num += float(np.sum(np.log10(1.0 + g * g * s1 / (sv + sigma_nsq))))
        den += float(np.sum(np.log10(1.0 + s1 / sigma_nsq)))
    if den <= 0:
        return None
    return num / den


# ---------------------------------------------------------------------------
# Perceptual: PSNR-HVS-M (Ponomarenko et al., DCT-domain CSF + masking)
# ---------------------------------------------------------------------------

# Normalised 8x8 DCT contrast-sensitivity table (csf_cof), Ponomarenko 2007.
_CSF = np.array(
    [
        [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
        [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
        [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
        [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
        [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
        [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
        [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
        [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950],
    ]
)


def _blocks_8x8(img: np.ndarray) -> np.ndarray:
    """Stack of non-overlapping 8x8 blocks, shape (nblocks, 8, 8)."""
    h, w = img.shape[0] - img.shape[0] % 8, img.shape[1] - img.shape[1] % 8
    a = img[:h, :w]
    return (
        a.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def _dct2_blocks(blocks: np.ndarray) -> np.ndarray:
    return _dct(_dct(blocks, axis=1, norm="ortho"), axis=2, norm="ortho")


def _mask_energy(dct_blocks: np.ndarray) -> np.ndarray:
    """Per-block reduced contrast-masking factor (csf-weighted AC energy)."""
    m = (dct_blocks * _CSF) ** 2
    e = m.sum(axis=(1, 2)) - m[:, 0, 0]
    return np.sqrt(np.clip(e, 0, None) / 64.0)


def psnr_hvs_m(y_ref: np.ndarray, y_dist: np.ndarray, peak: float = 255.0) -> float | None:
    """DCT-domain PSNR with CSF weighting + contrast masking (Ponomarenko).

    Vectorised over all 8x8 blocks. DC keeps its full weighted error; AC errors
    below the local masking threshold (`mask/16`) are suppressed.
    """
    if not _HAVE_SCIPY:
        return None
    if y_ref.shape[0] < 8 or y_ref.shape[1] < 8:
        return None
    da = _dct2_blocks(_blocks_8x8(y_ref))
    db = _dct2_blocks(_blocks_8x8(y_dist))
    mask = np.maximum(_mask_energy(da), _mask_energy(db))  # (nb,)
    thr = (mask / 16.0)[:, None, None]
    diff = np.abs(da - db)
    reduced = np.where(diff < thr, 0.0, diff - thr)
    ac = np.ones((8, 8), dtype=bool)
    ac[0, 0] = False
    u = np.where(ac[None], reduced, diff)
    acc = float(np.sum((u * _CSF) ** 2))
    count = da.shape[0] * 64
    if count == 0:
        return None
    mse = acc / count
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(peak * peak / mse)


# ---------------------------------------------------------------------------
# ffmpeg-backed metrics: VMAF (libvmaf) and XPSNR
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ffmpeg_filters(ffmpeg: str) -> str:
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                           capture_output=True, text=True)
        return r.stdout + r.stderr
    except Exception:
        return ""


def vmaf(ffmpeg: str, dist_png, ref_png) -> float | None:
    if "libvmaf" not in _ffmpeg_filters(ffmpeg):
        return None
    # Write the JSON log to a *relative* path with cwd set to a temp dir: an
    # absolute Windows path contains a drive-letter colon, which ffmpeg's filter
    # parser treats as an option separator and silently mis-parses.
    workdir = tempfile.mkdtemp()
    logname = "vmaf.json"
    dist_abs = os.path.abspath(str(dist_png))
    ref_abs = os.path.abspath(str(ref_png))
    try:
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin",
            "-i", dist_abs, "-i", ref_abs,
            "-lavfi",
            "[0:v]format=yuv420p[d];[1:v]format=yuv420p[r];"
            f"[d][r]libvmaf=log_fmt=json:log_path={logname}",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        with open(os.path.join(workdir, logname)) as f:
            data = json.load(f)
        return float(data["pooled_metrics"]["vmaf"]["mean"])
    except Exception:
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def xpsnr(ffmpeg: str, dist_png, ref_png) -> float | None:
    if "xpsnr" not in _ffmpeg_filters(ffmpeg):
        return None
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin",
        "-i", str(dist_png), "-i", str(ref_png),
        "-lavfi", "[0:v]format=yuv420p[d];[1:v]format=yuv420p[r];[d][r]xpsnr",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stderr + r.stdout
    # "XPSNR  y: .. u: .. v: .. (minimum: .. )"  or "XPSNR average ..."
    m = re.search(r"XPSNR\s+average[^0-9]*([0-9.]+|inf)", out)
    if not m:
        m = re.search(r"XPSNR\s+y:\s*([0-9.]+|inf)", out)
    if not m:
        return None
    v = m.group(1)
    return float("inf") if v == "inf" else float(v)


# ---------------------------------------------------------------------------
# Optional: Butteraugli (external binary) and piq learned metrics
# ---------------------------------------------------------------------------

def butteraugli(ref_png, dist_png, binary: str | None = None) -> float | None:
    binary = binary or os.environ.get("BUTTERAUGLI_BIN") or shutil.which("butteraugli")
    if not binary:
        return None
    try:
        r = subprocess.run([binary, str(ref_png), str(dist_png)],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        for line in out:
            m = re.search(r"([0-9]+\.[0-9]+)", line)
            if m:
                return float(m.group(1))
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def _piq():
    try:
        import torch  # noqa: F401
        import piq  # noqa: F401

        return piq
    except Exception:
        return None


def piq_metrics(ref_rgb: np.ndarray, dist_rgb: np.ndarray) -> dict:
    """LPIPS / DISTS / HaarPSI via `piq` if importable, else empty."""
    piq = _piq()
    if piq is None:
        return {}
    import torch

    def t(a):
        return torch.from_numpy(a.transpose(2, 0, 1)[None] / 255.0).float()

    x, y = t(dist_rgb), t(ref_rgb)
    out = {}
    try:
        out["haarpsi"] = float(piq.haarpsi(x, y, data_range=1.0))
    except Exception:
        pass
    try:
        out["dists"] = float(piq.DISTS()(x, y))
    except Exception:
        pass
    try:
        out["lpips"] = float(piq.LPIPS()(x, y))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

# Column order + direction for the heads-up table. True = higher is better.
PANEL_KEYS = [
    ("psnr_rgb", True),
    ("psnr_y", True),
    ("ms_ssim", True),
    ("ms_ssim_db", True),
    ("vifp", True),
    ("psnr_hvs_m", True),
    ("vmaf", True),
    ("xpsnr", True),
    ("butteraugli", False),
    ("haarpsi", True),
    ("dists", False),
    ("lpips", False),
]


def compute_panel(
    dist_png,
    ref_png,
    *,
    ffmpeg: str = "ffmpeg",
    butteraugli_bin: str | None = None,
    want_vmaf: bool = True,
    want_piq: bool = True,
) -> dict:
    """Compute every available metric for one (decoded, reference) PNG pair."""
    ref_rgb = load_rgb(ref_png)
    dist_rgb = load_rgb(dist_png)
    if ref_rgb.shape != dist_rgb.shape:
        raise ValueError(f"shape mismatch {ref_rgb.shape} vs {dist_rgb.shape}")
    ry, dy = to_y(ref_rgb), to_y(dist_rgb)

    panel: dict[str, float | None] = {k: None for k, _ in PANEL_KEYS}
    panel["psnr_rgb"] = psnr(dist_rgb, ref_rgb)
    panel["psnr_y"] = psnr(dy, ry)
    panel["ms_ssim"], panel["ms_ssim_db"] = ms_ssim(ry, dy)
    panel["vifp"] = vifp(ry, dy)
    panel["psnr_hvs_m"] = psnr_hvs_m(ry, dy)
    if want_vmaf:
        panel["vmaf"] = vmaf(ffmpeg, dist_png, ref_png)
        panel["xpsnr"] = xpsnr(ffmpeg, dist_png, ref_png)
    panel["butteraugli"] = butteraugli(ref_png, dist_png, butteraugli_bin)
    if want_piq:
        panel.update(piq_metrics(ref_rgb, dist_rgb))
    return panel


# ---------------------------------------------------------------------------
# BD-rate (Bjontegaard) over a rate/quality sweep
# ---------------------------------------------------------------------------

def bd_rate(rate_a, metric_a, rate_b, metric_b) -> float | None:
    """Average % bitrate change of B vs A at equal quality (negative = B better).

    Standard Bjontegaard piecewise-cubic integral over log-rate. Needs >= 4
    overlapping points. Returns None if the metric range does not overlap.
    """
    try:
        ra = np.log10(np.asarray(rate_a, dtype=float))
        rb = np.log10(np.asarray(rate_b, dtype=float))
        ma = np.asarray(metric_a, dtype=float)
        mb = np.asarray(metric_b, dtype=float)
        if len(ra) < 4 or len(rb) < 4:
            return None
        # Sort by quality metric (x = metric, y = log-rate).
        oa, ob = np.argsort(ma), np.argsort(mb)
        ma, ra = ma[oa], ra[oa]
        mb, rb = mb[ob], rb[ob]
        pa = np.polyfit(ma, ra, 3)
        pb = np.polyfit(mb, rb, 3)
        lo = max(ma.min(), mb.min())
        hi = min(ma.max(), mb.max())
        if hi <= lo:
            return None
        ia = np.polyint(pa)
        ib = np.polyint(pb)
        int_a = np.polyval(ia, hi) - np.polyval(ia, lo)
        int_b = np.polyval(ib, hi) - np.polyval(ib, lo)
        avg = (int_b - int_a) / (hi - lo)
        return float((10.0 ** avg - 1.0) * 100.0)
    except Exception:
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute the metric panel for two images.")
    ap.add_argument("reference")
    ap.add_argument("distorted")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--butteraugli-bin", default=None)
    args = ap.parse_args()
    p = compute_panel(args.distorted, args.reference, ffmpeg=args.ffmpeg,
                      butteraugli_bin=args.butteraugli_bin)
    width = max(len(k) for k in p)
    for k, _ in PANEL_KEYS:
        v = p.get(k)
        print(f"{k:<{width}} : {'n/a' if v is None else f'{v:.5f}'}")
