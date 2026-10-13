"""Can we cut MitoFinder v1's fake dots without retraining?

Predicts glow maps for every EPFL slice once (cached in data/), then compares ways of turning glows
into dots. Choices are made on the TRAINING volume; the TEST volume is only scored at the end.
Run: py -3 experiments/dots_fix.py
"""
import os
import sys

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

ROOT = os.path.join(os.path.dirname(__file__), "..")
MATCH = 15


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=3, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList([Block(cin, c[0]), Block(c[0], c[1]), Block(c[1], c[2])])
        self.middle = Block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2), nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.merge = nn.ModuleList([Block(c[3], c[2]), Block(c[2], c[1]), Block(c[1], c[0])])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        bridges = []
        for block in self.down:
            x = block(x); bridges.append(x); x = F.max_pool2d(x, 2)
        x = self.middle(x)
        for up, merge, bridge in zip(self.up, self.merge, reversed(bridges)):
            x = merge(torch.cat([up(x), bridge], 1))
        return torch.sigmoid(self.out(x))


def dots_in_slice(m):
    labels, n = ndimage.label(m)
    depth = ndimage.distance_transform_edt(m)
    return [tuple(map(int, p)) for p in ndimage.maximum_position(depth, labels, range(1, n + 1))] if n else []


def match(pred, truth):
    pairs = sorted((np.hypot(p[0] - t[0], p[1] - t[1]), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t = set(), set()
    for d, i, j in pairs:
        if d > MATCH:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return len(used_t), len(truth) - len(used_t), len(pred) - len(used_p)


def peaks(glow, threshold, size=MATCH):
    """v1: one dot per bright peak (plateaus counted once)."""
    top = (glow == ndimage.maximum_filter(glow, size=size)) & (glow > threshold)
    labels, n = ndimage.label(top)
    return [tuple(int(round(v)) for v in c) for c in ndimage.center_of_mass(top, labels, range(1, n + 1))] if n else []


def blobs(glow, threshold):
    """Fix: one dot per glowing blob, at its brightest pixel."""
    labels, n = ndimage.label(glow > threshold)
    return [tuple(map(int, p)) for p in ndimage.maximum_position(glow, labels, range(1, n + 1))] if n else []


def glows_for(name, model):
    path = os.path.join(ROOT, "data", f"glow_v1_{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    img = tifffile.imread(os.path.join(ROOT, "data", f"{name}.tif"))
    out = np.zeros(img.shape, np.float16)
    with torch.no_grad():
        for z in range(1, img.shape[0] - 1):
            x = torch.from_numpy(img[z - 1:z + 2].astype(np.float32)[None] / 255.0)
            out[z] = model(x)[0, 0].numpy()
            print(f"\r{name}: slice {z}/{img.shape[0] - 2}", end="", flush=True)
    print()
    np.save(path, out)
    return out


def score(method, glows, truth, zs):
    found = missed = fake = 0
    for z in zs:
        m, mi, fa = match(method(glows[z].astype(np.float32)), truth[z])
        found += m; missed += mi; fake += fa
    r = found / max(found + missed, 1); p = found / max(found + fake, 1)
    return 2 * p * r / max(p + r, 1e-9), r, p, found, missed, fake


def main():
    torch.set_num_threads(os.cpu_count())
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(ROOT, "mitofinder_v1.pt"), map_location="cpu"))
    model.eval()
    sets = {}
    for name in ("training", "testing"):
        mask = tifffile.imread(os.path.join(ROOT, "data", f"{name}_groundtruth.tif")) > 0
        sets[name] = (glows_for(name, model), [dots_in_slice(mask[z]) for z in range(mask.shape[0])], range(1, mask.shape[0] - 1))

    tg, tt, tz = sets["testing"]
    f1, r, p, *_ = score(lambda g: peaks(g, 0.4), tg, tt, tz)
    print(f"check, v1 as in Colab (peaks, 0.4) on test: F1 {f1:.3f} recall {r:.0%} precision {p:.0%}")
    if abs(f1 - 0.77) > 0.02:
        sys.exit("does not reproduce the Colab score (0.77): stop before trusting anything below")

    candidates = {}
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        candidates[f"peaks t={t}"] = lambda g, t=t: peaks(g, t)
        candidates[f"peaks t={t} spread 31"] = lambda g, t=t: peaks(g, t, 2 * MATCH + 1)
        candidates[f"blobs t={t}"] = lambda g, t=t: blobs(g, t)

    g, tr, zs = sets["training"]
    print("\nchosen on the TRAINING volume:")
    results = sorted(((score(m, g, tr, zs), name) for name, m in candidates.items()), reverse=True)
    for (f1, r, p, *_), name in results[:8]:
        print(f"  {name:24s} F1 {f1:.3f}  recall {r:.0%}  precision {p:.0%}")
    best = results[0][1]
    f1, r, p, found, missed, fake = score(candidates[best], tg, tt, tz)
    print(f"\nTEST, best choice '{best}': found {found} of {found + missed} (recall {r:.0%}), {fake} fake (precision {p:.0%}), F1 {f1:.2f}")


if __name__ == "__main__":
    main()
