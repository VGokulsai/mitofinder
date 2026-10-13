"""Runs a notebook's code cells top to bottom on this laptop (CPU, so notebooks use their QUICK mode).
Figures are saved as <folder>/fig1.png, fig2.png ... instead of shown.

    py -3 notebooks/run_local.py notebooks/03_human_tissue.ipynb <work folder>
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb_path, work = os.path.abspath(sys.argv[1]), sys.argv[2]
os.makedirs(work, exist_ok=True)
os.chdir(work)
shown = [0]


def save_instead_of_show(*a, **k):
    shown[0] += 1
    plt.savefig(f"fig{shown[0]}.png", dpi=60, bbox_inches="tight")
    plt.close("all")


plt.show = save_instead_of_show
cells = [c["source"] for c in json.load(open(nb_path, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for i, src in enumerate(cells, 1):
    src = "".join(src) if isinstance(src, list) else src
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!"))   # pip lines: installed here already
    t = time.time()
    print(f"--- cell {i}/{len(cells)}", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)
    print(f"    ({time.time() - t:.0f}s)", flush=True)
print("ALL CELLS RAN")
