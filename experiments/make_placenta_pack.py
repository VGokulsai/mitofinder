"""Zips the placenta pictures, your clicks and mitofinder_v1h.pt into placenta_pack.zip for notebook 04.
Run: py -3 experiments/make_placenta_pack.py [output.zip]
"""
import json
import os
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "placenta_pack.zip")
crops = os.path.join(ROOT, "data", "placenta_crops")

for group in ("test", "teach"):
    answers = os.path.join(crops, group, "answers.json")
    if not os.path.exists(answers):
        sys.exit(f"no clicks yet for {group}: run 'Click {'placenta' if group == 'test' else 'teaching'}.bat' first")
    print(f"{group}: {sum(len(v) for v in json.load(open(answers)).values())} dots")

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for group in ("test", "teach"):
        for f in sorted(os.listdir(os.path.join(crops, group))):
            if f.endswith((".png", ".json")):
                z.write(os.path.join(crops, group, f), f"{group}/{f}")
    z.write(os.path.join(ROOT, "mitofinder_v1h.pt"), "mitofinder_v1h.pt")
print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
