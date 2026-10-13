"""Use MitoFinder on any electron-microscope picture.

    py -3 find_mitos.py picture.png           # a single image (png, jpg or tif)
    py -3 find_mitos.py stack.tif 80          # slice 80 of a 3D stack (uses the slices above and below too)
    py -3 find_mitos.py try                   # every new picture in the folder "try"

Saves picture_mitofinder.png next to the input, with a red + on every mitochondrion it finds, and opens it.
Trained on mouse brain at 5 nm per pixel: pictures at a very different zoom will score worse.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
from dots_fix import UNet, blobs  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mitofinder_v1.pt")


def to_uint8(a):
    """Match the brightness and contrast MitoFinder trained on (EPFL: mean 140.5, spread 30.4).
    A plain 0-255 stretch is not enough: it made v1 find nothing at all."""
    a = a.astype(np.float32)
    a = (a - a.mean()) / max(a.std(), 1e-6) * 30.4 + 140.5
    return np.clip(a, 0, 255).astype(np.uint8)


def load(path, z=None):
    """Returns the 3 slices the model reads: (above, this one, below)."""
    if path.lower().endswith((".tif", ".tiff")):
        a = tifffile.imread(path)
    else:
        a = np.asarray(Image.open(path).convert("L"))
    if a.ndim == 3 and a.shape[-1] in (3, 4):           # colour tif
        a = a[..., :3].mean(-1)
    if a.ndim == 2:                                     # one picture: show it to the model three times
        a = to_uint8(a)
        return np.stack([a, a, a]), a
    z = a.shape[0] // 2 if z is None else z
    z = int(np.clip(z, 1, a.shape[0] - 2))
    three = to_uint8(a[z - 1:z + 2])
    return three, three[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if os.path.isdir(sys.argv[1]):                      # a folder: every picture in it not done yet
        folder = sys.argv[1]
        todo = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")) and "_mitofinder" not in f
                and not os.path.exists(os.path.join(folder, os.path.splitext(f)[0] + "_mitofinder.png"))]
        print(f"{len(todo)} new picture(s) in {folder}" + ("" if todo else " (delete a _mitofinder result to redo it)"))
        for p in todo:
            run(p)
        return
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)


def run(path, z=None):
    three, picture = load(path, z)
    h, w = picture.shape
    H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8          # the U-Net halves the size 3 times
    x = np.pad(three, ((0, 0), (0, H - h), (0, W - w)), mode="reflect").astype(np.float32) / 255.0

    model = UNet()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        glow = model(torch.from_numpy(x)[None])[0, 0].numpy()[:h, :w]
    dots = blobs(glow, 0.7)      # experiments/dots_fix.py: test F1 0.76 -> 0.84 vs v1's peaks at 0.4

    out = os.path.splitext(path)[0] + "_mitofinder.png"
    fig, ax = plt.subplots(1, 2, figsize=(16, 8 * h / w + 1))
    ax[0].imshow(picture, cmap="gray")
    ax[0].scatter([d[1] for d in dots], [d[0] for d in dots], s=70, c="red", marker="+")
    ax[0].set_title(f"MitoFinder found {len(dots)} mitochondria")
    ax[1].imshow(glow, cmap="magma", vmin=0, vmax=1)
    ax[1].set_title("its glow map")
    for a in ax:
        a.axis("off")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"{len(dots)} mitochondria -> {out}")
    if os.name == "nt":
        os.startfile(out)


if __name__ == "__main__":
    main()
