"""Cuts placenta slices (EMPIAR-10967, stack 12, 4.2 nm pixels) into 1024 x 1024 pictures at MitoFinder's 5 nm zoom,
brightness matched to EPFL, ready for click_centres.py (data/placenta_crops/test first, then teach).

Crops sit on a fixed 2 x 2 grid, never chosen by looking at what MitoFinder finds (that would flatter its score).
Crops that are mostly blood or empty fluid are skipped by an image-only rule (edge density), never by MitoFinder.
Test and teach slices alternate 30 slices (1.5 um) apart, so they show different cells. (v0, 13 Oct: 3 slices 150 nm
apart on a fixed grid gave near-identical pictures, half of them blood, about 6 mitochondria in all.)
Run: py -3 experiments/make_placenta_crops.py
"""
import os

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC, OUT = os.path.join(ROOT, "data", "placenta"), os.path.join(ROOT, "data", "placenta_crops")
NM = 4.2                   # stack 12 pixel size, from the EMPIAR-10967 entry
SIDE = 1024                # crop size at 5 nm
SLICES = {"test": (30, 90, 150, 210, 270), "teach": (60, 120, 180, 240, 300)}   # 30 slices (1.5 um) apart: different cells
TISSUE = 0.04            # share of pixels on an edge; blood-only crops measured 0.015-0.028, tissue 0.046-0.055 (v0 set)

os.makedirs(OUT, exist_ok=True)
for group, zs in SLICES.items():
    for z in zs:
        a = tifffile.imread(os.path.join(SRC, f"s{z:04d}.tif")).astype(np.float32)
        a = ndimage.zoom(a, NM / 5, order=1)                                  # 3000 px at 4.2 nm -> 2520 px at 5 nm
        a = np.clip((a - a.mean()) / a.std() * 30.4 + 140.5, 0, 255).astype(np.uint8)
        margin = (a.shape[0] - 2 * SIDE) // 2
        for r in range(2):
            for c in range(2):
                y, x = margin + r * SIDE, margin + c * SIDE
                crop = a[y:y + SIDE, x:x + SIDE]
                edges = float((ndimage.gaussian_gradient_magnitude(crop.astype(np.float32), 3) > 4).mean())
                if edges < TISSUE:
                    print(f"  skip {group} z{z} r{r}c{c}: edge density {edges:.3f}")
                    continue
                os.makedirs(os.path.join(OUT, group), exist_ok=True)
                Image.fromarray(crop).save(os.path.join(OUT, group, f"z{z}_r{r}c{c}.png"))
        print(group, z, "done")
print("crops in", OUT)
