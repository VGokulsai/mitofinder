"""Writes notebooks/01_look_at_the_data.ipynb. Run: py -3 notebooks/make_01.py"""
import json
import pathlib

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# MitoFinder 01: look at the data

Before building any AI, researchers look at their data. In this notebook we:

1. download a real electron-microscope dataset (brain tissue, EPFL, Switzerland), where experts outlined every mitochondrion;
2. look at the images and the outlines side by side;
3. measure three things that decide how we build the model: **how much of the image is mitochondria**, **how many mitochondria** there are and **how big** they are, and **how they continue across slices** (it is 3D).

Run each cell with **Shift + Enter**. Nothing here needs a GPU.
""")

md("""
## 1. Download the data

Four files, about 130 MB each. `training` and `testing` are the microscope images; the `_groundtruth` files are the experts' outlines (white = mitochondrion, black = everything else).
""")

code("""
!pip -q install tifffile
import os, urllib.request
BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/"
FILES = ["training.tif", "training_groundtruth.tif", "testing.tif", "testing_groundtruth.tif"]
os.makedirs("data", exist_ok=True)
for f in FILES:
    path = "data/" + f
    if not os.path.exists(path):
        print("downloading", f, "...")
        urllib.request.urlretrieve(BASE + f, path)
print("done:", sorted(os.listdir("data")))
""")

md("""
## 2. Load it and check the basics

A `.tif` here is a **stack**: many 2D slices on top of each other, like pages of a book. `shape` tells us (slices, height, width).
""")

code("""
import numpy as np, tifffile
img  = tifffile.imread("data/training.tif")
mask = tifffile.imread("data/training_groundtruth.tif")
print("image:", img.shape, img.dtype, "values", img.min(), "to", img.max())
print("mask: ", mask.shape, mask.dtype, "values", np.unique(mask))
""")

md("""
**Question for you:** each voxel (3D pixel) is about 5 x 5 x 5 nanometres. How big is this whole block in micrometres? (1 micrometre = 1000 nanometres.) Write your answer here: ...
""")

md("""
## 3. Look at one slice

Left: what the microscope saw. Middle: the experts' outline. Right: both together (mitochondria in green).
""")

code("""
import matplotlib.pyplot as plt
def show(z):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img[z], cmap="gray");  ax[0].set_title(f"slice {z}: microscope image")
    ax[1].imshow(mask[z], cmap="gray"); ax[1].set_title("expert outline (white = mitochondrion)")
    ax[2].imshow(img[z], cmap="gray")
    ax[2].imshow(np.ma.masked_where(mask[z] == 0, mask[z]), cmap="Greens", alpha=0.45, vmin=0, vmax=255)
    ax[2].set_title("overlay")
    for a in ax: a.axis("off")
    plt.show()
show(80)
""")

md("""
**Try:** change `80` to other slice numbers (0 to the number of slices minus 1). Find one mitochondrion and look at its inner folds (*cristae*). Then find something dark that is **not** outlined: that is what fools volunteers and AI.
""")

md("""
## 4. How much of the image is mitochondria?

This number matters a lot: if mitochondria are only a small part of the image, a lazy AI could answer "no mitochondrion anywhere" and still be right most of the time. We will have to stop it doing that.
""")

code("""
fraction = (mask > 0).mean()
print(f"{fraction:.1%} of all voxels are mitochondria")
print(f"an AI that always says 'not mitochondria' would be right {1 - fraction:.1%} of the time, and completely useless")
""")

md("""
## 5. How many mitochondria, and how big?

`label` gives every separate white blob in a slice its own number, so we can count them and measure their areas.
""")

code("""
from scipy import ndimage
counts, areas = [], []
for z in range(mask.shape[0]):
    labels, n = ndimage.label(mask[z] > 0)
    counts.append(n)
    if n:
        areas += list(np.bincount(labels.ravel())[1:])
areas = np.array(areas)
print("mitochondria per slice: average", round(np.mean(counts), 1), "| min", min(counts), "| max", max(counts))
print("area of one (pixels): median", int(np.median(areas)), "| smallest", areas.min(), "| largest", areas.max())
plt.figure(figsize=(8, 4)); plt.hist(areas, bins=60); plt.xlabel("area of one mitochondrion in a slice (pixels)"); plt.ylabel("how many"); plt.show()
""")

md("""
**Question for you:** the smallest blobs are tiny. Are they tiny mitochondria, or the edge of a big one that is just starting or ending in this slice? (Hint: look at section 6.)
""")

md("""
## 6. It is 3D: follow one mitochondrion through the slices

We cut out a small window and show 8 slices in a row. Watch a mitochondrion grow, shrink and disappear: the same object, seen page by page.
""")

code("""
# pick the biggest mitochondrion in the middle slice and follow it
mid = mask.shape[0] // 2
labels, n = ndimage.label(mask[mid] > 0)
biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
ys, xs = np.where(labels == biggest)
cy, cx, r = int(ys.mean()), int(xs.mean()), 90
fig, ax = plt.subplots(2, 8, figsize=(20, 5.5))
for i, z in enumerate(range(mid - 14, mid + 18, 4)):
    crop = lambda a: a[z, max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    ax[0, i].imshow(crop(img), cmap="gray"); ax[0, i].set_title(f"slice {z}")
    ax[1, i].imshow(crop(mask), cmap="gray")
    ax[0, i].axis("off"); ax[1, i].axis("off")
plt.suptitle("top: image, bottom: expert outline"); plt.show()
""")

md("""
## 7. Turn outlines into dots, like the volunteers

On Placenta Profiles, volunteers do not draw outlines: in **Mito Mapper** they click the **centre** of each mitochondrion. So our AI will learn to find centres too, and its answers will look exactly like the volunteers' dots. Here we compute a centre for every outlined mitochondrion in a slice: these dots are what we will teach the AI to find.
""")

code("""
def centres(z):
    labels, n = ndimage.label(mask[z] > 0)
    return ndimage.center_of_mass(mask[z] > 0, labels, range(1, n + 1))

z = 80
dots = centres(z)
plt.figure(figsize=(9, 9))
plt.imshow(img[z], cmap="gray")
plt.scatter([c[1] for c in dots], [c[0] for c in dots], s=60, c="red", marker="+")
plt.title(f"slice {z}: {len(dots)} mitochondrion centres (what a perfect volunteer would click)")
plt.axis("off"); plt.show()
""")

md("""
**Question for you:** look at a long, bent mitochondrion. Is its centre inside it? (The centre of a banana can be outside the banana.) We will need to handle that.
""")

md("""
## What we learned (fill in)

1. The block is ... micrometres across.
2. ...% of voxels are mitochondria, so our AI must be trained so that ignoring them is punished.
3. A typical mitochondrion covers about ... pixels in one slice.
4. Mitochondria continue across slices, so ...

**Next notebook:** we cut the slices into small training patches and build our first AI (a U-Net).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).with_name("01_look_at_the_data.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
