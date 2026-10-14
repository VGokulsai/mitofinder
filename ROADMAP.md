# MitoFinder roadmap

Updated 14 Oct 2026. Every number below was measured in this repo or read from the source named next to it.

## Where it stands

| Step | Result |
|---|---|
| Notebook 01: look at the data (EPFL mouse brain, 5 nm) | 5.6% of voxels are mitochondria, about 15 per slice |
| Notebook 02: MitoFinder v1, trained on a Colab T4 (3,000 steps) | test F1 0.77 |
| `experiments/dots_fix.py`: cutoff 0.4 -> 0.7, chosen on training slices | **test F1 0.84** |
| Notebook 03 + `experiments/human_cutoff.py`: human brain (MitoEM-H) | v1 0.41 -> **v1h 0.58** after teaching on 64 human slices (cutoff re-chosen on a separate block) |
| Placenta (EMPIAR-10967 stack 39, CC0), 28 centres Gokul clicked on 12 test pictures | v1 0.04, v1h 0.12 |
| Notebook 04: v1h taught on Gokul's 13 teaching dots (v1p) | 0.09-0.12: false alarms down about 4x, but too shy (unmarked mitochondria were taught as background) |
| **MitoNet** (published generalist, CC-BY-4.0), its default settings | **0.37** (8 of 28 found, 7 fake); excellent on clear mitochondria, misses the faint grey ovals |

Talk post on Placenta Profiles: 14 Oct (discussion 4155123), DM to smith_p. No reply yet; we don't wait.

## Next steps, in order

1. **Suggest-and-correct click tool** (me): MitoNet proposes dots on the teaching pictures, Gokul accepts, removes or adds. Complete answers in minutes instead of hunting for every mitochondrion.
2. **Re-teach MitoFinder** on the complete teaching answers (notebook 04 again, Colab). Score once on the 28 test dots. Question: does it close the gap with MitoNet?
3. **README + make it public** (Gokul says "push"): every score including the failures, the MitoNet comparison, the one-click app.
4. **bioRxiv preprint** (free, no age rule): methods, the three tissues, the placenta lesson about incomplete labels. Everything else links to it.
5. **Show it** (Gokul writes): Talk thread update, image.sc forum, BioImage Model Zoo upload.
6. **Conferences** (see memory: where-to-show-research): IEEE ISBI 2027 1-page abstract due **1 Feb 2027** (Lausanne, 25-28 May); Focus on Microscopy abstract due **7 Jan 2027** (Osaka; financial support applications by **21 Oct 2026**).
7. **The disease step**: pre-eclampsia / growth-restriction images are with the Placenta Profiles team. Ask once there is a public, scored tool. IRIS 2027 project.
8. **Then project two: TB from cough sounds** (Synapse account + parent-signed minor addendum first).

## Rules we keep

- Choose settings on training data, score the test data once.
- Classify on Zooniverse with your own eyes, never MitoFinder's or MitoNet's.
- In the suggest-and-correct tool, a dot counts only after Gokul has looked at it.
- It measures; it never diagnoses.
