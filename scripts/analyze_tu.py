#!/usr/bin/env python3
"""Analyze the BPG_TU_DIAG CSV: TU leaf-vs-split margins + flip-rate by region/size.

Joins each encoder TU decision with the source-luma local variance (so we can
bucket by smooth/mid/textured the same way bpg-decision-diff does), and reports:
  - per (region x TU size): n, split-win %, and for split-wins the distribution
    of the relative flip margin (split_won_by / leaf_cost)
  - flip% at candidate large-TU-bias thresholds (fraction of split-wins that a
    given relative leaf-bias would flip back to the larger leaf)

Usage: analyze_tu.py <tu_diag.csv> <source.png>
"""
import sys, csv
import numpy as np
from PIL import Image

SMOOTH, TEXTURED = 20.0, 200.0
REGION = {0: "smooth", 1: "mid", 2: "textured"}  # fallback if variance unused
BIAS = [0.005, 0.01, 0.02, 0.05]  # relative leaf-bias thresholds to test


def luma(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]


def main():
    csv_path, png_path = sys.argv[1], sys.argv[2]
    Y = luma(png_path)
    H, W = Y.shape
    # integral images for O(1) block variance
    P = np.zeros((H + 1, W + 1)); P2 = np.zeros((H + 1, W + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(Y, 0), 1)
    P2[1:, 1:] = np.cumsum(np.cumsum(Y * Y, 0), 1)

    def var(x, y, s):
        x2, y2 = min(x + s, W), min(y + s, H)
        n = (x2 - x) * (y2 - y)
        if n <= 0:
            return 0.0
        ssum = P[y2, x2] - P[y, x2] - P[y2, x] + P[y, x]
        ssq = P2[y2, x2] - P2[y, x2] - P2[y2, x] + P2[y, x]
        return ssq / n - (ssum / n) ** 2

    def cls(v):
        return "smooth" if v < SMOOTH else ("textured" if v > TEXTURED else "mid")

    # bucket[(region, log2)] = list of (winner_split, split_won_by, leaf_cost)
    buckets = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            x, y = int(row["x"]), int(row["y"])
            l2 = int(row["log2"])
            s = 1 << l2
            r = cls(var(x, y, s))
            key = (r, l2)
            buckets.setdefault(key, []).append(
                (int(row["winner_split"]), float(row["split_won_by"]), float(row["leaf_cost"]))
            )

    print(f"{'region':<9} {'TU':<6} {'n':>7} {'split%':>7} {'medRelMgn':>10} "
          + " ".join(f"flip@{int(b*1000)/10}%".rjust(9) for b in BIAS))
    order = {"smooth": 0, "mid": 1, "textured": 2}
    for key in sorted(buckets, key=lambda k: (order[k[0]], k[1])):
        rows = buckets[key]
        n = len(rows)
        splits = [(won, lc) for w, won, lc in rows if w == 1]
        nsplit = len(splits)
        # relative margins of split wins
        rel = [won / lc for won, lc in splits if lc > 0]
        med = float(np.median(rel)) if rel else 0.0
        # flip%: of ALL decisions, how many split-wins flip to leaf under bias b
        flips = []
        for b in BIAS:
            f = sum(1 for won, lc in splits if lc > 0 and won < b * lc)
            flips.append(100.0 * f / n if n else 0.0)
        print(f"{key[0]:<9} {1<<key[1]:>2}x{1<<key[1]:<3} {n:>7} {100*nsplit/n:>6.1f}% "
              f"{med:>10.4f} " + " ".join(f"{x:>8.1f}%" for x in flips))


if __name__ == "__main__":
    main()
