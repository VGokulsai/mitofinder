"""Click the centre of every mitochondrion, like Mito Mapper. Your clicks become MitoFinder's answer sheet.

    py -3 click_centres.py data/placenta_crops

  left click   put a dot on a mitochondrion's centre
  right click  remove the nearest dot
  n / p        next / previous picture
  z            zoom tool on or off (the magnifier in the toolbar does the same); h = see the whole picture again
  q            quit

If the folder has suggestions.json (dots another AI proposed), they show as yellow circles:
  left click on a yellow circle   accept it (it becomes your red +)
  right click on a yellow circle  reject it
  a                               accept every yellow circle left on this picture (after you have looked at them)
A suggestion only counts once you accept it.

Every click is saved straight away to answers.json in the same folder, so you can stop and come back.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
for k in ("keymap.pan", "keymap.back", "keymap.forward"):    # p and the arrow keys are ours: next/previous picture
    plt.rcParams[k] = []
import numpy as np
from PIL import Image

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "placenta_crops")
pictures = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
if not pictures:
    sys.exit(f"no .png pictures in {folder}")
save_path = os.path.join(folder, "answers.json")
answers = json.load(open(save_path)) if os.path.exists(save_path) else {}
suggest_path = os.path.join(folder, "suggestions.json")
suggestions = json.load(open(suggest_path)) if os.path.exists(suggest_path) else {}
NEAR = 20               # pixels: a click this close to a yellow circle accepts it
state = {"i": 0}

fig, ax = plt.subplots(figsize=(11, 11))
fig.canvas.manager.set_window_title("MitoFinder: click the centres")


def save():
    with open(save_path, "w") as f:
        json.dump(answers, f, indent=1)
    if suggestions:
        with open(suggest_path, "w") as f:
            json.dump(suggestions, f, indent=1)


def nearest(dots, y, x):
    if not dots:
        return None, float("inf")
    d = [np.hypot(a - y, b - x) for a, b in dots]
    i = int(np.argmin(d))
    return i, d[i]


def draw(keep_view=False):
    name = pictures[state["i"]]
    view = (ax.get_xlim(), ax.get_ylim()) if keep_view else None
    ax.clear()
    ax.imshow(np.asarray(Image.open(os.path.join(folder, name))), cmap="gray")
    dots = answers.get(name, [])
    if dots:
        ax.scatter([d[1] for d in dots], [d[0] for d in dots], s=90, c="red", marker="+", linewidths=2)
    todo = suggestions.get(name, [])
    if todo:
        ax.scatter([d[1] for d in todo], [d[0] for d in todo], s=900, facecolors="none", edgecolors="yellow", linewidths=4)
    done = sum(1 for p in pictures if p in answers)
    extra = f"  |  {len(todo)} YELLOW to check" if todo else ""
    ax.set_title(f"{name}  ({state['i'] + 1} of {len(pictures)})  |  {len(dots)} dots here{extra}  |  {done} pictures started\n"
                 "left click = dot / accept yellow, right click = remove / reject, a = accept all yellow here, n/p, z = zoom, h")
    ax.axis("off")
    if view:
        ax.set_xlim(view[0]); ax.set_ylim(view[1])
    fig.canvas.draw_idle()


def on_click(event):
    toolbar = fig.canvas.toolbar
    if event.inaxes is not ax or (toolbar is not None and toolbar.mode):      # the zoom/pan tool is on: let it work
        return
    name = pictures[state["i"]]
    dots = answers.setdefault(name, [])
    todo = suggestions.get(name, [])
    si, sd = nearest(todo, event.ydata, event.xdata)
    if event.button == 1:
        if si is not None and sd <= NEAR:
            dots.append(todo.pop(si))                     # accept the suggestion
        else:
            dots.append([round(event.ydata, 1), round(event.xdata, 1)])
    elif event.button == 3:
        di, dd = nearest(dots, event.ydata, event.xdata)
        if si is not None and sd <= dd:
            todo.pop(si)                                  # reject the suggestion
        elif di is not None:
            dots.pop(di)
    save()
    draw(keep_view=True)


def on_key(event):
    if event.key in ("n", "right"):
        state["i"] = min(state["i"] + 1, len(pictures) - 1); draw()
    elif event.key in ("p", "left"):
        state["i"] = max(state["i"] - 1, 0); draw()
    elif event.key == "z":
        fig.canvas.toolbar.zoom()
    elif event.key == "a":
        name = pictures[state["i"]]
        answers.setdefault(name, []).extend(suggestions.get(name, []))
        suggestions[name] = []
        save(); draw(keep_view=True)
    elif event.key == "q":
        plt.close(fig)


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
draw()
plt.show()
print(f"saved {sum(len(v) for v in answers.values())} dots on {len(answers)} pictures -> {save_path}")
