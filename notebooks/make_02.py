"""Writes notebooks/02_build_mitofinder.ipynb. Run: py -3 notebooks/make_02.py"""
import json
import pathlib

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# MitoFinder 02: build the AI

We build an AI that looks at an electron-microscope image and puts a dot on the centre of every mitochondrion, the same job Placenta Profiles volunteers do in Mito Mapper.

**Before you start:** Runtime -> Change runtime type -> **T4 GPU** -> Save. Then run the cells top to bottom with Shift + Enter.

The plan: answer sheet (glowing dots) -> patches -> U-Net -> training -> dots -> honest score on images it has never seen.
""")

md("## 1. Setup: GPU, libraries, data")

code("""
!pip -q install tifffile
import os, time, urllib.request, numpy as np, tifffile, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage

device = "cuda" if torch.cuda.is_available() else "cpu"
print("training on:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(no GPU: training will be slow; turn the GPU on, see the top)")

QUICK = device == "cpu"   # on a CPU we run a tiny version, just to check everything works

BASE = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/"
os.makedirs("data", exist_ok=True)
for f in ["training.tif", "training_groundtruth.tif", "testing.tif", "testing_groundtruth.tif"]:
    if not os.path.exists("data/" + f):
        print("downloading", f); urllib.request.urlretrieve(BASE + f, "data/" + f)

train_img, train_mask = tifffile.imread("data/training.tif"), tifffile.imread("data/training_groundtruth.tif") > 0
test_img,  test_mask  = tifffile.imread("data/testing.tif"),  tifffile.imread("data/testing_groundtruth.tif") > 0
print("train", train_img.shape, "| test", test_img.shape)
""")

md("""
## 2. The answer sheet: one dot per mitochondrion, as a soft glow

From notebook 1 (question 4): the plain centre can fall **outside** a bent mitochondrion. So we use the point **deepest inside** each one: the pixel farthest from its edge. A careful volunteer would click there.

A single pixel is too small a target to learn from, so each dot becomes a **soft glow**: brightest at the centre, fading over a few pixels. The AI learns to draw glows; later we turn glows back into dots.
""")

code("""
SIGMA = 6        # glow radius in pixels (about 30 nm)

def dots_in_slice(m):
    \"\"\"(y, x) of the deepest-inside pixel of every mitochondrion in one slice.\"\"\"
    labels, n = ndimage.label(m)
    depth = ndimage.distance_transform_edt(m)
    return [tuple(map(int, p)) for p in ndimage.maximum_position(depth, labels, range(1, n + 1))] if n else []

def glow_map(shape, dots):
    g = np.zeros(shape, np.float32)
    for y, x in dots:
        g[y, x] = 1.0
    g = ndimage.gaussian_filter(g, SIGMA) * (2 * np.pi * SIGMA ** 2)   # every mitochondrion's glow peaks at exactly 1
    return np.minimum(g, 1.0)

def make_answers(mask):
    dots = [dots_in_slice(mask[z]) for z in range(mask.shape[0])]
    glows = np.stack([glow_map(mask.shape[1:], d) for d in dots]).astype(np.float16)
    return dots, glows

t = time.time()
train_dots, train_glow = make_answers(train_mask)
test_dots,  test_glow  = make_answers(test_mask)
print(f"answer sheets ready in {time.time() - t:.0f}s: {sum(map(len, train_dots))} training dots, {sum(map(len, test_dots))} test dots")

z = 80
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(train_img[z], cmap="gray"); ax[0].scatter([d[1] for d in train_dots[z]], [d[0] for d in train_dots[z]], c="red", marker="+", s=80)
ax[0].set_title("deepest-inside dots"); ax[1].imshow(train_glow[z].astype(np.float32), cmap="magma"); ax[1].set_title("the glow map the AI must learn to draw")
for a in ax: a.axis("off")
plt.show()
""")

md("""
## 3. Patches: small 3D-aware pieces to learn from

The AI does not get whole images. It gets **256 x 256 patches**, each made of **3 slices** (the one we care about, plus the one above and below), so it can see a mitochondrion as a 3D sausage and not be fooled by a tiny tip (question 3).

Each patch is randomly **flipped and rotated**: a mitochondrion is still a mitochondrion upside down, and this gives the AI more variety to learn from.

**The 5.6% problem again:** most random patches contain no mitochondrion at all, so the AI would spend most of its time looking at nothing. So 70% of patches are picked **around a real mitochondrion** (placed at a random spot in the patch, not always the middle) and 30% fully at random, so it still learns what "nothing here" looks like.
""")

code("""
PATCH = 128 if QUICK else 256

FOCUS = 0.7     # share of patches picked around a real mitochondrion (random patches are mostly empty)

def random_patch(img, glow, dots=None):
    dots = train_dots if dots is None else dots
    z = np.random.randint(1, img.shape[0] - 1)
    if np.random.rand() < FOCUS and dots[z]:
        cy, cx = dots[z][np.random.randint(len(dots[z]))]           # a real mitochondrion...
        y = int(np.clip(cy - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, img.shape[1] - PATCH))
        x = int(np.clip(cx - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, img.shape[2] - PATCH))   # ...somewhere in the patch, not always centred
    else:
        y = np.random.randint(0, img.shape[1] - PATCH + 1)
        x = np.random.randint(0, img.shape[2] - PATCH + 1)
    a = img[z - 1:z + 2, y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0   # 3 slices
    b = glow[z, y:y + PATCH, x:x + PATCH].astype(np.float32)[None]
    k = np.random.randint(4)
    a, b = np.rot90(a, k, (1, 2)), np.rot90(b, k, (1, 2))
    if np.random.rand() < 0.5:
        a, b = a[:, :, ::-1], b[:, :, ::-1]
    return np.ascontiguousarray(a), np.ascontiguousarray(b)

def batch(n):
    pairs = [random_patch(train_img, train_glow) for _ in range(n)]
    x = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device)
    y = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device)
    return x, y

x, y = batch(4)
print("a batch of patches:", tuple(x.shape), "(patches, slices, height, width) -> answers", tuple(y.shape))
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    ax[0, i].imshow(x[i, 1].cpu(), cmap="gray"); ax[1, i].imshow(y[i, 0].cpu(), cmap="magma")
    ax[0, i].axis("off"); ax[1, i].axis("off")
plt.suptitle("top: middle slice of each patch, bottom: its answer"); plt.show()
""")

md("""
## 4. The U-Net: MitoFinder's brain

- **Going down (left side of the U):** the image is shrunk step by step. Early layers notice edges, middle layers notice stripes and curves, deep layers notice "bean shape with stripes inside".
- **Going up (right side):** it is grown back to full size to draw the glow map.
- **Bridges across the U:** fine details from the way down are handed straight across, so the glows land in exactly the right place.

Each `Block` is two small filters (3 x 3 pixels) that slide over the image. The network learns the numbers inside those filters during training.
""")

code("""
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
        for block in self.down:              # down the U
            x = block(x); bridges.append(x); x = F.max_pool2d(x, 2)
        x = self.middle(x)
        for up, merge, bridge in zip(self.up, self.merge, reversed(bridges)):   # up the U
            x = merge(torch.cat([up(x), bridge], 1))
        return torch.sigmoid(self.out(x))

model = UNet().to(device)
print(f"MitoFinder has {sum(p.numel() for p in model.parameters()):,} numbers to learn")
""")

md("""
## 5. Training

Each step: show a batch of patches, compare the AI's glow map with the answer, measure how wrong it is (the **loss**), and nudge every number a little to be less wrong.

The 5.6% trap from notebook 1: most of an image has no glow, so an AI that draws nothing would already look "mostly right". To stop that, mistakes **near a real dot cost 20 times more**.
""")

code("""
STEPS = 20 if QUICK else 3000
BATCH = 4 if QUICK else 8
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS)

def loss_fn(pred, target):
    weight = 1 + 20 * target                       # missing a mitochondrion costs much more
    return (weight * (pred - target) ** 2).mean()

losses, t = [], time.time()
model.train()
for step in range(1, STEPS + 1):
    x, y = batch(BATCH)
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    losses.append(loss.item())
    if step % max(1, STEPS // 10) == 0:
        print(f"step {step:5d}/{STEPS}  loss {np.mean(losses[-50:]):.5f}  ({time.time() - t:.0f}s)")
plt.figure(figsize=(8, 3)); plt.plot(losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss (how wrong)"); plt.title("MitoFinder learning"); plt.show()
torch.save(model.state_dict(), "mitofinder_v1.pt")
print("saved mitofinder_v1.pt")
""")

md("""
## 6. From glows to dots, and an honest score

On the **test** images (never seen in training) the AI draws its glow map; every bright peak becomes a dot. Then we match its dots with the experts' dots: a match is a pair less than 15 pixels (75 nm) apart.

- **found** (recall): the share of real mitochondria it put a dot on
- **trustworthy** (precision): the share of its dots that really are mitochondria
- **F1**: one number combining both (1.0 = perfect)
""")

code("""
MATCH = 15      # pixels: a dot counts if it is this close to a real centre

@torch.no_grad()
def predict(img, z):
    model.eval()
    x = torch.from_numpy(img[z - 1:z + 2].astype(np.float32)[None] / 255.0).to(device)
    return model(x)[0, 0].float().cpu().numpy()

def peaks(glow, threshold=0.4):
    \"\"\"One dot per bright spot. A flat top can have many equal pixels, so each connected
    group of top pixels counts once (its centre).\"\"\"
    top = (glow == ndimage.maximum_filter(glow, size=MATCH)) & (glow > threshold)
    labels, n = ndimage.label(top)
    return [tuple(int(round(v)) for v in c) for c in ndimage.center_of_mass(top, labels, range(1, n + 1))] if n else []

def match(pred, truth):
    \"\"\"Greedy matching: closest pairs first. Returns (matched, missed, fake).\"\"\"
    pairs = sorted((np.hypot(p[0] - t[0], p[1] - t[1]), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t = set(), set()
    for d, i, j in pairs:
        if d > MATCH: break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return len(used_t), len(truth) - len(used_t), len(pred) - len(used_p)

zs = range(1, test_img.shape[0] - 1, 16 if QUICK else 1)
found = missed = fake = 0
for z in zs:
    m, mi, fa = match(peaks(predict(test_img, z)), test_dots[z])
    found += m; missed += mi; fake += fa
recall = found / max(found + missed, 1); precision = found / max(found + fake, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)
print(f"on {len(zs)} test slices it has never seen:")
print(f"  found {found} of {found + missed} mitochondria (recall {recall:.0%}), missed {missed}")
print(f"  {fake} fake dots (precision {precision:.0%})")
print(f"  F1 = {f1:.2f}")
""")

md("## 7. See it: the AI's dots next to the experts' dots")

code("""
z = test_img.shape[0] // 2
glow = predict(test_img, z)
p = peaks(glow)
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
ax[0].imshow(test_img[z], cmap="gray")
ax[0].scatter([d[1] for d in test_dots[z]], [d[0] for d in test_dots[z]], s=140, facecolors="none", edgecolors="lime", linewidths=2, label="experts")
ax[0].scatter([d[1] for d in p], [d[0] for d in p], s=60, c="red", marker="+", label="MitoFinder")
ax[0].legend(loc="lower right"); ax[0].set_title(f"test slice {z}: green circles = experts, red + = MitoFinder")
ax[1].imshow(glow, cmap="magma"); ax[1].set_title("MitoFinder's glow map")
for a in ax: a.axis("off")
plt.show()
""")

md("""
## What we learned (fill in)

1. MitoFinder v1 found ...% of the mitochondria and ...% of its dots were real (F1 = ...).
2. Where it goes wrong (look at section 7): ...
3. Idea to make it better: ...

**Download your model:** in the left sidebar, Files -> right-click `mitofinder_v1.pt` -> Download. (Colab deletes files when the session ends.)

**Next notebook:** does MitoFinder still work on **human** tissue (MitoEM-H)? That is the real test before placenta.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).with_name("02_build_mitofinder.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
