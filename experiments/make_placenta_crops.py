"""Cuts placenta slices (EMPIAR-10967) into 1024 x 1024 pictures at MitoFinder's 5 nm zoom, brightness matched to
EPFL, ready for click_centres.py (test first, then teach).

    py -3 experiments/make_placenta_crops.py 12     # stack 12 -> data/placenta_crops     (mostly blood vessel)
    py -3 experiments/make_placenta_crops.py 39     # stack 39 -> data/placenta39_crops   (sharp, dense mitochondria)

Crops sit on a fixed grid. Crops that are mostly blood or empty fluid are skipped by an image-only rule (edge
density); when a slice has more tissue crops than needed, a fixed-seed random pick decides. Never MitoFinder.
Test and teach slices alternate, far enough apart (1 um or more) to show different cells.
(v0, 13 Oct: 3 slices 150 nm apart gave near-identical pictures, half blood, about 6 mitochondria in all.)
"""
import os
import random
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SIDE = 1024                # crop size at 5 nm
STACKS = {
    "12": dict(src="placenta", out="placenta_crops", nm=4.2, grid=2, per_slice=4, tissue=0.04,
               slices={"test": (30, 90, 150, 210, 270), "teach": (60, 120, 180, 240, 300)}),   # 30 slices = 1.5 um apart
    "39": dict(src="placenta39", out="placenta39_crops", nm=5.0, grid=4, per_slice=4, max_background=0.5,
               slices={"test": (5, 45, 85), "teach": (25, 65, 99)}),                           # 20 slices = 1 um apart
}
# tissue: share of pixels on an edge. Stack 12: blood-only crops 0.015-0.028, tissue 0.046-0.055.
# max_background (stack 39, which is sharp and mostly tissue, so edge density mislabels smooth tissue as empty):
# the share of a crop that is smooth like empty space (local spread below the slice's 20th percentile). Empty crops
# measured 0.53-0.79, tissue crops 0.00-0.45 (slices 5, 45, 99).

cfg = STACKS[sys.argv[1] if len(sys.argv) > 1 else "12"]
src, out = os.path.join(ROOT, "data", cfg["src"]), os.path.join(ROOT, "data", cfg["out"])
pick = random.Random(0)
for group, zs in cfg["slices"].items():
    os.makedirs(os.path.join(out, group), exist_ok=True)
    for z in zs:
        a = tifffile.imread(os.path.join(src, f"s{z:04d}.tif")).astype(np.float32)
        if cfg["nm"] != 5.0:
            a = ndimage.zoom(a, cfg["nm"] / 5, order=1)
        a = np.clip((a - a.mean()) / a.std() * 30.4 + 140.5, 0, 255).astype(np.uint8)
        margin = (a.shape[0] - cfg["grid"] * SIDE) // 2
        if "max_background" in cfg:
            f = a.astype(np.float32)
            m = ndimage.uniform_filter(f, 15)
            spread = ndimage.uniform_filter(np.sqrt(np.maximum(ndimage.uniform_filter(f * f, 15) - m * m, 0)), 65)
            smooth = spread < np.percentile(spread, 20)
        tissue = []
        for r in range(cfg["grid"]):
            for c in range(cfg["grid"]):
                y, x = margin + r * SIDE, margin + c * SIDE
                crop = a[y:y + SIDE, x:x + SIDE]
                if "max_background" in cfg:
                    score = float(smooth[y:y + SIDE, x:x + SIDE].mean())
                    keep, why = score <= cfg["max_background"], f"background {score:.2f}"
                else:
                    score = float((ndimage.gaussian_gradient_magnitude(crop.astype(np.float32), 3) > 4).mean())
                    keep, why = score >= cfg["tissue"], f"edge density {score:.3f}"
                if keep:
                    tissue.append((r, c, crop))
                else:
                    print(f"  skip {group} z{z} r{r}c{c}: {why}")
        chosen = pick.sample(tissue, min(cfg["per_slice"], len(tissue)))
        for r, c, crop in chosen:
            Image.fromarray(crop).save(os.path.join(out, group, f"z{z}_r{r}c{c}.png"))
        print(f"{group} z{z}: {len(tissue)} tissue crops, kept {len(chosen)}")
print("crops in", out)
