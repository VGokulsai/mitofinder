"""MitoFinder's placenta score against Gokul's clicks (data/placenta_crops/test/answers.json).

Each model uses the dot rule it already had (v1: one dot per blob at 0.7, chosen on EPFL training slices); nothing is
chosen on these answers. The cutoff sweep at the end is shown for understanding only, not used to pick a score.
Run: py -3 experiments/placenta_score.py [model.pt] [cutoff]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from dots_fix import UNet, blobs, match

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEST = os.path.join(ROOT, "data", "placenta_crops", "test")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mitofinder_v1.pt"
    cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    answers = json.load(open(os.path.join(TEST, "answers.json")))
    pictures = sorted(f for f in os.listdir(TEST) if f.endswith(".png"))
    model = UNet()
    model.load_state_dict(torch.load(os.path.join(ROOT, name), map_location="cpu"))
    model.eval()
    glows = {}
    with torch.no_grad():
        for f in pictures:
            a = np.asarray(Image.open(os.path.join(TEST, f))).astype(np.float32) / 255.0
            glows[f] = model(torch.from_numpy(np.stack([a, a, a]))[None])[0, 0].numpy()   # slices 50 nm apart: same slice x3

    def score(t, show=False):
        found = missed = fake = 0
        for f in pictures:
            pred = blobs(glows[f], t)
            m, mi, fa = match(pred, [tuple(d) for d in answers.get(f, [])])      # dots_fix.MATCH = 15 px = 75 nm
            found += m; missed += mi; fake += fa
            if show:
                print(f"  {f:16s} your dots {len(answers.get(f, [])):2d} | MitoFinder dots {len(pred):3d} | matched {m}")
        r = found / max(found + missed, 1); p = found / max(found + fake, 1)
        return found, missed, fake, r, p, 2 * p * r / max(p + r, 1e-9)

    print(f"{name}, one dot per blob at {cutoff}, on {len(pictures)} placenta test pictures:")
    found, missed, fake, r, p, f1 = score(cutoff, show=True)
    print(f"PLACENTA: found {found} of {found + missed} (recall {r:.0%}), {fake} fake dots (precision {p:.0%}), F1 {f1:.2f}")
    print("\ncutoff sweep (for understanding only):")
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        found, missed, fake, r, p, f1 = score(t)
        print(f"  {t:.2f}: found {r:.0%}, real dots {p:.0%}, F1 {f1:.2f}, {found + fake} dots in all")


if __name__ == "__main__":
    main()
