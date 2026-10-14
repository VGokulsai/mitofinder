"""Writes notebooks/04_placenta.ipynb. Run: py -3 notebooks/make_04.py"""
import json
import pathlib

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# MitoFinder 04: placenta, taught from your clicks

Nobody has drawn outlines around placenta mitochondria, so this time **your clicks are the answer sheet**:
- **teaching pictures** (12): MitoFinder practises on most of them; the last slice's pictures are kept aside to choose its cutoff
- **test pictures** (12): scored once at the very end, never seen while learning or choosing

As it is, MitoFinder v1 scores F1 0.04 on your test pictures (it found 4 of your 28 mitochondria). From notebook 3 we know teaching works on new tissue (human brain: 0.41 -> 0.58) **if the cutoff is chosen again afterwards**. Let's see if it works for placenta.

**Before you start:** Runtime -> Change runtime type -> **T4 GPU** -> Save, then **Disconnect and delete runtime** and connect again. Have `placenta_pack.zip` ready (it is in your `mitofinder` folder).
""")

md("## 1. Setup: GPU, your pictures, your clicks, the model")

code("""
import os, json, time, zipfile, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(no GPU: see the top of the notebook)")
QUICK = device == "cpu"

if not os.path.exists("placenta_pack.zip"):
    from google.colab import files
    print("choose placenta_pack.zip from your mitofinder folder:")
    files.upload()
zipfile.ZipFile("placenta_pack.zip").extractall("pack")

def load_set(name):
    folder = f"pack/{name}"
    answers = json.load(open(f"{folder}/answers.json"))
    names = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    pics = {n: np.asarray(Image.open(f"{folder}/{n}")) for n in names}
    dots = {n: [tuple(d) for d in answers.get(n, [])] for n in names}
    return pics, dots

teach_pics, teach_dots = load_set("teach")
test_pics, test_dots = load_set("test")
print(f"teaching: {len(teach_pics)} pictures, {sum(map(len, teach_dots.values()))} of your dots")
print(f"test:     {len(test_pics)} pictures, {sum(map(len, test_dots.values()))} of your dots")
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
        for block in self.down:
            x = block(x); bridges.append(x); x = F.max_pool2d(x, 2)
        x = self.middle(x)
        for up, merge, bridge in zip(self.up, self.merge, reversed(bridges)):
            x = merge(torch.cat([up(x), bridge], 1))
        return torch.sigmoid(self.out(x))

def load(path):
    m = UNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()

start = load("pack/mitofinder_v1h.pt")
print("starting from MitoFinder v1h (taught on human brain in notebook 3)")
""")

md("""
## 2. The tools: glows to dots, and the score

Same as before: a dot counts as found if it is within 15 pixels (75 nm) of one of your dots. These placenta slices are 50 nm apart, so MitoFinder sees the **same picture three times** instead of three neighbouring slices.
""")

code("""
MATCH = 15

@torch.no_grad()
def glow_of(model, pic):
    a = pic.astype(np.float32) / 255.0
    return model(torch.from_numpy(np.stack([a, a, a]))[None].to(device))[0, 0].float().cpu().numpy()

def peaks(glow, t):
    top = (glow == ndimage.maximum_filter(glow, size=MATCH)) & (glow > t)
    labels, n = ndimage.label(top)
    return [tuple(c) for c in ndimage.center_of_mass(top, labels, range(1, n + 1))] if n else []

def blobs(glow, t):
    labels, n = ndimage.label(glow > t)
    return [tuple(p) for p in ndimage.maximum_position(glow, labels, range(1, n + 1))] if n else []

def match(pred, truth):
    pairs = sorted((np.hypot(p[0] - t[0], p[1] - t[1]), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth))
    used_p, used_t = set(), set()
    for d, i, j in pairs:
        if d > MATCH: break
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return len(used_t), len(truth) - len(used_t), len(pred) - len(used_p)

def score(model, pics, dots, rule):
    found = missed = fake = 0
    for n, pic in pics.items():
        m, mi, fa = match(rule(glow_of(model, pic)), dots[n])
        found += m; missed += mi; fake += fa
    r = found / max(found + missed, 1); p = found / max(found + fake, 1)
    return {"found": f"{found} of {found + missed}", "real dots": f"{p:.0%}", "F1": round(2 * p * r / max(p + r, 1e-9), 2)}

RULES = {f"{kind} {t}": (lambda g, t=t, f=fn: f(g, t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for kind, fn in (("peaks", peaks), ("blobs", blobs))}

def choose_rule(model, pics, dots):
    \"\"\"Pick the dot rule with the best F1 on the kept-aside pictures.\"\"\"
    return max(RULES, key=lambda r: score(model, pics, dots, RULES[r])["F1"])
""")

md("""
## 3. Split the teaching pictures: 8 to practise on, 2 kept aside to choose the cutoff

The kept-aside pictures are the last teaching slice, the one farthest along the stack.
""")

code("""
last = max(int(n[1:].split("_")[0]) for n in teach_pics)            # the teaching slice farthest along the stack
aside = [n for n in teach_pics if n.startswith(f"z{last}_")]
practice = [n for n in teach_pics if n not in aside]
aside_pics, aside_dots = {n: teach_pics[n] for n in aside}, {n: teach_dots[n] for n in aside}
print("practise on:", practice)
print("kept aside: ", aside, "with", sum(len(aside_dots[n]) for n in aside), "of your dots")

results = {}
rule0 = "peaks 0.7"                                      # v1h's cutoff, chosen on human brain
results["A. v1h as it is (human cutoff)"] = score(start, test_pics, test_dots, RULES[rule0])
rule1 = choose_rule(start, aside_pics, aside_dots)
results[f"B. v1h, cutoff re-chosen ({rule1})"] = score(start, test_pics, test_dots, RULES[rule1])
for k, v in results.items(): print(f"{k:42s} {v}")
""")

md("""
## 4. Teach it from your clicks

Each of your dots becomes a soft glow, exactly like notebook 2. Patches are flipped and rotated, and their **brightness and graininess are jiggled** a little, so MitoFinder cannot rely on one exact look. Remember: any mitochondrion you did not click is taught as "nothing here".
""")

code("""
SIGMA, PATCH = 6, (128 if QUICK else 256)
STEPS = 10 if QUICK else 600

def glow_map(shape, dots):
    g = np.zeros(shape, np.float32)
    for y, x in dots:
        g[min(int(round(y)), shape[0] - 1), min(int(round(x)), shape[1] - 1)] = 1.0
    return np.minimum(ndimage.gaussian_filter(g, SIGMA) * (2 * np.pi * SIGMA ** 2), 1.0)

targets = {n: glow_map(teach_pics[n].shape, teach_dots[n]) for n in practice}

def random_patch():
    n = practice[np.random.randint(len(practice))]
    pic, glow, dots = teach_pics[n], targets[n], teach_dots[n]
    H, W = pic.shape
    if dots and np.random.rand() < 0.7:
        cy, cx = dots[np.random.randint(len(dots))]
        y = int(np.clip(cy - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, H - PATCH))
        x = int(np.clip(cx - np.random.randint(PATCH // 8, PATCH - PATCH // 8), 0, W - PATCH))
    else:
        y, x = np.random.randint(0, H - PATCH + 1), np.random.randint(0, W - PATCH + 1)
    a = pic[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0
    a = (a - 0.55) * np.random.uniform(0.8, 1.25) + 0.55 + np.random.uniform(-0.05, 0.05)     # contrast and brightness
    a = np.clip(a + np.random.normal(0, np.random.uniform(0, 0.03), a.shape), 0, 1)           # graininess
    b = glow[y:y + PATCH, x:x + PATCH]
    k = np.random.randint(4)
    a, b = np.rot90(a, k), np.rot90(b, k)
    if np.random.rand() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    return np.ascontiguousarray(np.stack([a, a, a])), np.ascontiguousarray(b[None])

model = load("pack/mitofinder_v1h.pt")
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
losses, t = [], time.time()
model.train()
for step in range(1, STEPS + 1):
    pairs = [random_patch() for _ in range(8)]
    x = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float32)).to(device)
    y = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.float32)).to(device)
    loss = ((1 + 20 * y) * (model(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if step % max(1, STEPS // 6) == 0:
        print(f"step {step}/{STEPS}  loss {np.mean(losses[-50:]):.5f}  ({time.time() - t:.0f}s)")
model.eval()
torch.save(model.state_dict(), "mitofinder_v1p.pt")

rule2 = choose_rule(model, aside_pics, aside_dots)
results[f"C. taught on placenta ({rule2})"] = score(model, test_pics, test_dots, RULES[rule2])
for k, v in results.items(): print(f"{k:42s} {v}")
""")

md("## 5. See it: your dots (green) and MitoFinder's (red), before and after teaching")

code("""
for n in list(test_pics)[:3]:
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a, (title, m, r) in zip(ax, [("before (v1h)", start, rule1), ("after teaching on your clicks", model, rule2)]):
        p = RULES[r](glow_of(m, test_pics[n]))
        a.imshow(test_pics[n], cmap="gray")
        a.scatter([d[1] for d in test_dots[n]], [d[0] for d in test_dots[n]], s=200, facecolors="none", edgecolors="lime", linewidths=2)
        a.scatter([d[1] for d in p], [d[0] for d in p], s=80, c="red", marker="+")
        a.set_title(f"{n}: {title}"); a.axis("off")
    plt.show()
""")

md("""
## What we learned (fill in)

1. On placenta, v1h as it is scored F1 = ...; after teaching on my clicks, F1 = ...
2. Where it still goes wrong (section 5): ...
3. With only ... of my dots to learn from, the next thing to try is ...

**Download** `mitofinder_v1p.pt` (Files -> right-click -> Download) before closing.

Test set: 9 pictures and few dots, so one more or one fewer match moves the score a lot. Treat it as a first reading, not a final number.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(__file__).with_name("04_placenta.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
