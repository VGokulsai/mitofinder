"""Notebook 03 scored the taught model (v1h) with v1's old cutoff of 0.4. Here the cutoff and input mode are chosen
on a THIRD MitoEM-H block (slices 192-255, never taught on, never tested), then v1 and v1h are each scored once
on the notebook's test block (slices 448-499).
Run: py -3 experiments/human_cutoff.py
"""
import json
import os
import urllib.request

import numcodecs
import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import find_boundaries

from dots_fix import UNet, blobs, match, peaks

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = "https://huggingface.co/datasets/pytc/MitoVerse/resolve/main/data/wei20/mitoEM-H.zarr/"
CACHE = os.path.join(ROOT, "data", "mitoem")
META = {a: json.loads(urllib.request.urlopen(BASE + a + "/.zarray").read()) for a in ("img", "mito")}
RADIUS = 15                     # 75 nm at 5 nm per pixel, as in notebooks 02 and 03


def chunk(arr, z, y, x):
    path = os.path.join(CACHE, f"{arr}.{z}.{y}.{x}")
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + f"{arr}/{z}.{y}.{x}", path)
    m = META[arr]
    return np.frombuffer(numcodecs.get_codec(m["compressor"]).decode(open(path, "rb").read()), m["dtype"]).reshape(m["chunks"])


def block(z):
    img = np.concatenate([np.concatenate([chunk("img", z, y, x) for x in (3, 4)], 2) for y in (3, 4)], 1)
    lab = np.concatenate([np.concatenate([chunk("mito", z, y, x) for x in (3, 4)], 2) for y in (3, 4)], 1)
    return img, lab


def dots_in_slice(lab):
    ids = np.unique(lab)[1:]
    if not len(ids):
        return []
    inside = (lab > 0) & ~find_boundaries(lab, mode="inner")
    depth = ndimage.distance_transform_edt(inside)
    return [tuple(map(int, p)) for p in ndimage.maximum_position(depth, lab, ids)]


def prepare(img, lab):
    """Same steps as notebook 03: match EPFL brightness, 8 nm -> 5 nm, crop to a multiple of 8."""
    a = img.astype(np.float32)
    a = np.clip((a - a.mean()) / a.std() * 30.4 + 140.5, 0, 255).astype(np.uint8)
    a = np.stack([ndimage.zoom(s, 8 / 5, order=1) for s in a])
    a = a[:, :a.shape[1] // 8 * 8, :a.shape[2] // 8 * 8]
    truth = [[(y * 1.6, x * 1.6) for y, x in dots_in_slice(lab[z]) if y * 1.6 < a.shape[1] and x * 1.6 < a.shape[2]]
             for z in range(len(lab))]
    return a, truth


def load(name):
    m = UNet()
    m.load_state_dict(torch.load(os.path.join(ROOT, name), map_location="cpu"))
    return m.eval()


@torch.no_grad()
def glows(model, stack, zs, copies):
    out = {}
    for z in zs:
        s = stack[[z, z, z]] if copies else stack[z - 1:z + 2]
        out[z] = model(torch.from_numpy(s.astype(np.float32)[None] / 255.0))[0, 0].numpy()
    return out


def score(glow, truth, rule):
    found = missed = fake = 0
    for z, g in glow.items():
        m, mi, fa = match(rule(g), truth[z])
        found += m; missed += mi; fake += fa
    r = found / max(found + missed, 1); p = found / max(found + fake, 1)
    return 2 * p * r / max(p + r, 1e-9), r, p


def main():
    torch.set_num_threads(os.cpu_count())
    rules = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        rules[f"peaks {t}"] = lambda g, t=t: peaks(g, t)
        rules[f"blobs {t}"] = lambda g, t=t: blobs(g, t)

    tune, tune_truth = prepare(*block(3))                   # slices 192-255
    test_img, test_lab = block(7)
    test, test_truth = prepare(test_img[:52], test_lab[:52])  # slices 448-499, as in notebook 03
    print("tuning block", tune.shape, "| test block", test.shape, flush=True)

    for name in ("mitofinder_v1.pt", "mitofinder_v1h.pt"):
        model = load(name)
        best = None
        for copies in (True, False):
            g = glows(model, tune, range(2, len(tune) - 1, 4), copies)
            for rule, fn in rules.items():
                f1 = score(g, tune_truth, fn)[0]
                if best is None or f1 > best[0]:
                    best = (f1, copies, rule)
        f1, copies, rule = best
        print(f"{name}: chosen on tuning block -> {rule}, {'same slice x3' if copies else 'neighbours'} (tuning F1 {f1:.2f})", flush=True)
        g = glows(model, test, range(1, len(test) - 1), copies)
        f1, r, p = score(g, test_truth, rules[rule])
        old = score(g, test_truth, rules["peaks 0.4"])
        print(f"  TEST, chosen rule: found {r:.0%}, real dots {p:.0%}, F1 {f1:.2f}   (old cutoff 0.4: F1 {old[0]:.2f})", flush=True)


if __name__ == "__main__":
    main()
