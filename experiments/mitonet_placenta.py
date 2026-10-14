"""MitoNet (Conrad & Narayan, Cell Systems 2023; weights CC-BY-4.0, Zenodo 6861565) on the placenta test pictures,
scored against Gokul's clicks exactly like MitoFinder. MitoNet's own default settings, nothing tuned on these answers:
input normalised with its mean 0.57571 / std 0.12765, mito where sigmoid(sem_logits) > 0.5, centres = 7x7 local maxima
of the centre heatmap above 0.1 (its nms_kernel and nms_threshold), kept only inside predicted mitochondria.
Two ways to turn its output into dots are reported: its own centres, and one deepest-inside point per mitochondrion blob.
Run: py -3 experiments/mitonet_placenta.py [quantized|full]
"""
import json
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from dots_fix import match

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEST = os.path.join(ROOT, "data", "placenta39_crops", "test")
WEIGHTS = os.path.join(ROOT, "data", "mitonet", "MitoNet_v1_quantized.pth" if "full" not in sys.argv else "MitoNet_v1.pth")
OUT = os.path.join(ROOT, "data", "mitonet", "test_masks")


def run(model, img):
    x = (img.astype(np.float32) / 255.0 - 0.57571) / 0.12765
    with torch.no_grad():
        out = model(torch.from_numpy(x)[None, None])
    sem = torch.sigmoid(out["sem_logits"])[0, 0].numpy()
    ctr = out["ctr_hmp"][0, 0].numpy()
    return sem, ctr


def centre_dots(sem, ctr):
    peaks = (ctr == ndimage.maximum_filter(ctr, size=7)) & (ctr > 0.1) & (sem > 0.5)
    labels, n = ndimage.label(peaks)
    return [tuple(c) for c in ndimage.center_of_mass(peaks, labels, range(1, n + 1))] if n else []


def blob_dots(sem, min_area=200):
    mask = sem > 0.5
    labels, n = ndimage.label(mask)
    if not n:
        return []
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    depth = ndimage.distance_transform_edt(mask)
    pts = ndimage.maximum_position(depth, labels, range(1, n + 1))
    return [tuple(p) for p, a in zip(pts, areas) if a >= min_area]      # ignore specks smaller than ~ 70 x 70 nm


def main():
    model = torch.jit.load(WEIGHTS, map_location="cpu").eval()
    answers = json.load(open(os.path.join(TEST, "answers.json")))
    os.makedirs(OUT, exist_ok=True)
    totals = {"MitoNet centres": [0, 0, 0], "MitoNet blobs": [0, 0, 0]}
    for f in sorted(p for p in os.listdir(TEST) if p.endswith(".png")):
        img = np.asarray(Image.open(os.path.join(TEST, f)))
        sem, ctr = run(model, img)
        np.save(os.path.join(OUT, f.replace(".png", ".npy")), (sem * 255).astype(np.uint8))
        truth = [tuple(d) for d in answers.get(f, [])]
        line = f"  {f:16s} your dots {len(truth):2d}"
        for name, dots in (("MitoNet centres", centre_dots(sem, ctr)), ("MitoNet blobs", blob_dots(sem))):
            m, mi, fa = match(dots, truth)
            t = totals[name]; t[0] += m; t[1] += mi; t[2] += fa
            line += f" | {name}: {len(dots):3d} dots, {m} matched"
        print(line, flush=True)
    print(f"\n{os.path.basename(WEIGHTS)} on {len(os.listdir(TEST)) - 1} placenta test pictures (your {sum(map(len, answers.values()))} dots):")
    for name, (found, missed, fake) in totals.items():
        r = found / max(found + missed, 1); p = found / max(found + fake, 1)
        print(f"  {name:16s} found {found} of {found + missed} ({r:.0%}), {fake} fake dots ({p:.0%} real), F1 {2 * p * r / max(p + r, 1e-9):.2f}")


if __name__ == "__main__":
    main()
