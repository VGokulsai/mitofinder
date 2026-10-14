"""MitoNet proposes mitochondrion centres on the placenta TEACHING pictures, as yellow circles for click_centres.py.
Nothing becomes an answer until Gokul accepts it. Dots within 15 px of one he already placed are skipped.
Never run on the test pictures: their answers must stay his alone, or MitoNet would be grading itself.
Run: py -3 experiments/make_suggestions.py
"""
import json
import os
import warnings

import numpy as np
import torch
from PIL import Image

from mitonet_placenta import centre_dots, run

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEACH = os.path.join(ROOT, "data", "placenta39_crops", "teach")

model = torch.jit.load(os.path.join(ROOT, "data", "mitonet", "MitoNet_v1_quantized.pth"), map_location="cpu").eval()
answers = json.load(open(os.path.join(TEACH, "answers.json")))
suggestions = {}
for f in sorted(p for p in os.listdir(TEACH) if p.endswith(".png")):
    sem, ctr = run(model, np.asarray(Image.open(os.path.join(TEACH, f))))
    mine = answers.get(f, [])
    new = [[round(float(y), 1), round(float(x), 1)] for y, x in centre_dots(sem, ctr)
           if all(np.hypot(y - a, x - b) > 15 for a, b in mine)]
    suggestions[f] = new
    print(f"{f:16s} your dots {len(mine):2d} | MitoNet suggests {len(new):2d} more")
json.dump(suggestions, open(os.path.join(TEACH, "suggestions.json"), "w"), indent=1)
print("total suggestions:", sum(map(len, suggestions.values())))
