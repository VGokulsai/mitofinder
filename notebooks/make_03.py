"""Writes notebooks/03_human_tissue.ipynb. Run: py -3 notebooks/make_03.py"""
import json
import pathlib

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# MitoFinder 03: does it work on human tissue?

MitoFinder v1 learned on **mouse** brain (EPFL). Placenta is **human** tissue, from a different microscope, at a different zoom. Before placenta, we test a smaller jump: **human brain** (MitoEM-H, from Harvard; 10,552 mitochondria outlined by experts).

Every difference between two datasets can confuse an AI. Here are the three we know about:

| | EPFL (what v1 learned) | MitoEM-H (today) |
|---|---|---|
| tissue | mouse brain | human brain |
| one pixel is | 5 nm | **8 nm** (mitochondria look 1.6x smaller) |
| slices are | 5 nm apart | **30 nm apart** (neighbours look less alike) |
| brightness | average 140 | different in different places (108 to 145) |

We test v1 as it is, then fix one difference at a time and watch the score. Last, we teach it with a few human slices.

**Before you start:** Runtime -> Change runtime type -> **T4 GPU** -> Save, then **Runtime -> Disconnect and delete runtime** and connect again (the GPU only arrives with a fresh runtime). Have `mitofinder_v1.pt` ready: you downloaded it after notebook 2.
""")

md("## 1. Setup: GPU, libraries, your model")

code("""
!pip -q install numcodecs
import os, json, time, urllib.request, numpy as np, numcodecs, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import find_boundaries

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(no GPU: see the top of the notebook)")
QUICK = device == "cpu"   # on a CPU: a tiny version, just to check everything works

if not os.path.exists("mitofinder_v1.pt"):
    from google.colab import files
    print("choose mitofinder_v1.pt from your computer:")
    files.upload()

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

def load_v1():
    m = UNet().to(device)
    m.load_state_dict(torch.load("mitofinder_v1.pt", map_location=device))
    return m.eval()

v1 = load_v1()
print("MitoFinder v1 loaded")
""")

md("""
## 2. Download human brain pieces

MitoEM-H is huge (500 slices of 4096 x 4096 pixels), so we download only two blocks, each 1024 x 1024 pixels:

- a **teaching block**: slices 0 to 63 (used only in section 6)
- a **test block**: slices 448 to 499, about 12 micrometres deeper: different cells, never used for teaching

The data comes from the MitoVerse collection on Hugging Face, stored in small pieces ("chunks") of 64 slices x 512 x 512 pixels.
""")

code("""
BASE = "https://huggingface.co/datasets/pytc/MitoVerse/resolve/main/data/wei20/mitoEM-H.zarr/"
os.makedirs("mitoem", exist_ok=True)
META = {a: json.loads(urllib.request.urlopen(BASE + a + "/.zarray").read()) for a in ("img", "mito")}

def chunk(arr, z, y, x):
    path = f"mitoem/{arr}.{z}.{y}.{x}"
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + f"{arr}/{z}.{y}.{x}", path)
    m = META[arr]
    return np.frombuffer(numcodecs.get_codec(m["compressor"]).decode(open(path, "rb").read()), m["dtype"]).reshape(m["chunks"])

def block(z, ys=(3, 4), xs=(3, 4)):
    img = np.concatenate([np.concatenate([chunk("img", z, y, x) for x in xs], 2) for y in ys], 1)
    lab = np.concatenate([np.concatenate([chunk("mito", z, y, x) for x in xs], 2) for y in ys], 1)
    return img, lab

t = time.time()
teach_img, teach_lab = block(0)
test_img, test_lab = block(7)
test_img, test_lab = test_img[:52], test_lab[:52]     # slices 448-499; the rest of that chunk is empty padding
print(f"downloaded in {time.time() - t:.0f}s | teaching block {teach_img.shape} | test block {test_img.shape}")
print(f"average brightness: teaching {teach_img.mean():.0f}, test {test_img.mean():.0f} (EPFL was 140)")

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(test_img[26], cmap="gray"); ax[0].set_title("human brain, test slice")
ax[1].imshow(test_img[26], cmap="gray"); ax[1].imshow(np.ma.masked_where(test_lab[26] == 0, test_lab[26] % 20), cmap="tab20", alpha=0.5)
ax[1].set_title("experts' outlines (each mitochondrion its own colour)")
for a in ax: a.axis("off")
plt.show()
""")

md("""
**Look:** compare this with the mouse brain in notebook 1. Same kind of mitochondria (bean shape, stripes inside), but smaller on screen, because each pixel here covers more tissue (8 nm instead of 5).

## 3. The answer sheet

Same rule as notebook 2: one dot per mitochondrion, at the point **deepest inside** it. One new detail: here the experts gave every mitochondrion its own number, so two that touch are still two dots (in notebook 2 touching ones could merge).
""")

code("""
def dots_in_slice(lab):
    \"\"\"(y, x) of the deepest-inside pixel of every mitochondrion (each has its own number in lab).\"\"\"
    ids = np.unique(lab)[1:]
    if not len(ids):
        return []
    inside = (lab > 0) & ~find_boundaries(lab, mode="inner")     # cut touching mitochondria apart
    depth = ndimage.distance_transform_edt(inside)
    return [tuple(map(int, p)) for p in ndimage.maximum_position(depth, lab, ids)]

test_dots = [dots_in_slice(test_lab[z]) for z in range(len(test_lab))]
print(sum(map(len, test_dots)), "mitochondrion dots in the test block,", round(np.mean([len(d) for d in test_dots]), 1), "per slice")
""")

md("""
## 4. The tools from notebook 2: glows to dots, and the score

A dot counts as found if it is within **75 nm** of a real centre, the same distance as in notebook 2. In pixels that number changes with the zoom: 15 pixels at 5 nm, but only about 9 pixels at 8 nm.
""")

code("""
MATCH_NM = 75

def peaks(glow, threshold=0.4, size=15):
    top = (glow == ndimage.maximum_filter(glow, size=size)) & (glow > threshold)
    labels, n = ndimage.label(top)
    return [tuple(int(round(v)) for v in c) for c in ndimage.center_of_mass(top, labels, range(1, n + 1))] if n else []

def match(pred, truth, radius):
    pairs = sorted((np.hypot(p[0] - t[0], p[1] - t[1]), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t = set(), set()
    for d, i, j in pairs:
        if d > radius: break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return len(used_t), len(truth) - len(used_t), len(pred) - len(used_p)

@torch.no_grad()
def glow_of(model, stack, z, copies=False):
    s = stack[[z, z, z]] if copies else stack[z - 1:z + 2]
    return model(torch.from_numpy(s.astype(np.float32)[None] / 255.0).to(device))[0, 0].float().cpu().numpy()

def score(model, stack, nm_per_pixel, copies=False):
    \"\"\"stack: the test block after any changes. Dots are scaled to its zoom.\"\"\"
    model.eval()
    zoom = 8 / nm_per_pixel
    radius = MATCH_NM / nm_per_pixel
    found = missed = fake = 0
    for z in range(1, len(stack) - 1, 8 if QUICK else 1):
        truth = [(y * zoom, x * zoom) for y, x in test_dots[z] if y * zoom < stack.shape[1] and x * zoom < stack.shape[2]]   # skip dots in the cropped-off edge
        m, mi, fa = match(peaks(glow_of(model, stack, z, copies), size=int(radius)), truth, radius)
        found += m; missed += mi; fake += fa
    r = found / max(found + missed, 1); p = found / max(found + fake, 1)
    return {"found": f"{r:.0%}", "real dots": f"{p:.0%}", "F1": round(2 * p * r / max(p + r, 1e-9), 2)}

results = {}
""")

md("""
## 5. Fix one difference at a time

**Test A: v1 exactly as it is.** Human tissue, wrong zoom, wrong brightness. Prediction before you run it: will it find more or less than half?
""")

code("""
results["A. v1 as it is"] = score(v1, test_img, nm_per_pixel=8)
print(results)
""")

md("""
**Test B: match the brightness.** In building the MitoFinder app we found v1 finds *nothing* when the contrast is stretched. So we shift every test slice to the brightness and contrast of the EPFL images (average 140.5, spread 30.4).
""")

code("""
def match_brightness(stack):
    a = stack.astype(np.float32)
    a = (a - a.mean()) / a.std() * 30.4 + 140.5
    return np.clip(a, 0, 255).astype(np.uint8)

test_b = match_brightness(test_img)
results["B. + brightness"] = score(v1, test_b, nm_per_pixel=8)
print(results["B. + brightness"])
""")

md("""
**Test C: match the zoom.** Make each slice 1.6x bigger (8 nm / 5 nm), so mitochondria are the size v1 knows.
""")

code("""
def match_zoom(stack):
    return np.stack([ndimage.zoom(s, 8 / 5, order=1) for s in stack])

test_c = match_zoom(test_b)
test_c = test_c[:, :test_c.shape[1] // 8 * 8, :test_c.shape[2] // 8 * 8]      # the U-Net halves the size 3 times
results["C. + zoom"] = score(v1, test_c, nm_per_pixel=5)
print(results["C. + zoom"])
""")

md("""
**Test D: the slices problem.** v1 reads 3 slices that were 5 nm apart, so they looked almost the same. Here neighbours are 30 nm apart and can look quite different. What if we show it the **same** slice three times instead?
""")

code("""
results["D. + same slice x3"] = score(v1, test_c, nm_per_pixel=5, copies=True)
for k, v in results.items(): print(f"{k:22s} {v}")
""")

md("""
**Write down:** which fix helped most? Did any make it worse? The best of C and D is what v1 can do on human tissue without new lessons.

## 6. Teach it with a few human slices (fine-tuning)

Now we let MitoFinder practise on the **teaching block** (64 human slices, never the test block). We start from v1's brain, not from zero, so it should need only a few minutes. This is called **fine-tuning**.

Notice what it learns from: only the **dots** (as glows). That matters for placenta: there are no expert outlines there, but **your clicks** are dots.
""")

code("""
SIGMA = 6
PATCH = 128 if QUICK else 256
STEPS = 10 if QUICK else 800
COPIES = results["D. + same slice x3"]["F1"] > results["C. + zoom"]["F1"]     # use whichever input worked better
print("teaching with", "the same slice x3" if COPIES else "3 neighbouring slices")

teach = match_zoom(match_brightness(teach_img))
teach_dots = [[(y * 1.6, x * 1.6) for y, x in dots_in_slice(teach_lab[z])] for z in range(len(teach_lab))]

def glow_map(shape, dots):
    g = np.zeros(shape, np.float32)
    for y, x in dots:
        g[min(int(round(y)), shape[0] - 1), min(int(round(x)), shape[1] - 1)] = 1.0
    return np.minimum(ndimage.gaussian_filter(g, SIGMA) * (2 * np.pi * SIGMA ** 2), 1.0)

teach_glow = np.stack([glow_map(teach.shape[1:], d) for d in teach_dots]).astype(np.float16)

def random_patch():
    z = np.random.randint(1, len(teach) - 1)
    if np.random.rand() < 0.7 and teach_dots[z]:
        cy, cx = teach_dots[z][np.random.randint(len(teach_dots[z]))]
        y = int(np.clip(cy - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, teach.shape[1] - PATCH))
        x = int(np.clip(cx - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, teach.shape[2] - PATCH))
    else:
        y, x = np.random.randint(0, teach.shape[1] - PATCH + 1), np.random.randint(0, teach.shape[2] - PATCH + 1)
    zs = [z, z, z] if COPIES else [z - 1, z, z + 1]
    a = teach[zs, y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
    b = teach_glow[z, y:y + PATCH, x:x + PATCH].astype(np.float32)[None]
    k = np.random.randint(4)
    a, b = np.rot90(a, k, (1, 2)), np.rot90(b, k, (1, 2))
    if np.random.rand() < 0.5:
        a, b = a[:, :, ::-1], b[:, :, ::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)

model = load_v1()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
losses, t = [], time.time()
model.train()
for step in range(1, STEPS + 1):
    pairs = [random_patch() for _ in range(8)]
    x = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device)
    y = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device)
    pred = model(x)
    loss = ((1 + 20 * y) * (pred - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if step % max(1, STEPS // 8) == 0:
        print(f"step {step}/{STEPS}  loss {np.mean(losses[-50:]):.5f}  ({time.time() - t:.0f}s)")
torch.save(model.state_dict(), "mitofinder_v1h.pt")

results["E. taught on human"] = score(model, test_c, nm_per_pixel=5, copies=COPIES)
for k, v in results.items(): print(f"{k:22s} {v}")
""")

md("## 7. See it: before and after teaching")

code("""
z = len(test_c) // 2
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
truth = [(y * 1.6, x * 1.6) for y, x in test_dots[z] if y * 1.6 < test_c.shape[1] and x * 1.6 < test_c.shape[2]]
for a, (name, m) in zip(ax, [("v1", v1), ("taught on human", model)]):
    p = peaks(glow_of(m, test_c, z, COPIES))
    a.imshow(test_c[z], cmap="gray")
    a.scatter([d[1] for d in truth], [d[0] for d in truth], s=140, facecolors="none", edgecolors="lime", linewidths=2)
    a.scatter([d[1] for d in p], [d[0] for d in p], s=60, c="red", marker="+")
    a.set_title(f"{name}: green circles = experts, red + = MitoFinder"); a.axis("off")
plt.show()
""")

md("""
## What we learned (fill in)

1. On human brain, v1 as it is scored F1 = ..., and after matching brightness and zoom F1 = ...
2. The difference that mattered most was ...
3. After teaching with 64 human slices: F1 = ... It learned only from dots, so for placenta we can teach it with ...

**Download** `mitofinder_v1h.pt` (Files -> right-click -> Download) before closing.

**Next notebook:** real placenta images, where nobody has drawn outlines. You click the centres on a few slices, and those clicks become the answer sheet.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).with_name("03_human_tissue.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
