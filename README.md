# MitoFinder

An AI that puts a dot on the centre of every mitochondrion in an electron-microscope image, the same job volunteers do in [Science Scribbler: Placenta Profiles](https://www.zooniverse.org/projects/msbrhonclif/science-scribbler-placenta-profiles) (Mito Mapper).

It learns from dots, not outlines. Placenta images have no expert outlines, but volunteer clicks are dots, so they can be its answer sheet.

## Scores

A dot counts as found if it lands within 75 nm of an expert's centre. F1 combines "how many it found" and "how many of its dots are real" (1.0 is perfect). Settings are chosen on training data; each test score is measured once.

| Model | Test data | Found | Dots that are real | F1 |
|---|---|---|---|---|
| v1, cutoff 0.4 | EPFL mouse brain, 163 unseen slices | 93% | 65% | 0.77 |
| v1, cutoff 0.7 (chosen on training slices) | same | 82% | 85% | **0.84** |
| v1 on human brain, no changes | MitoEM-H, every 8th of 50 slices | 1% | 7% | 0.01 |
| + brightness matched | same | 12% | 82% | 0.21 |
| + zoom matched (8 nm to 5 nm) | same | 30% | 52% | 0.38 |
| + same slice x3 instead of neighbours | same | 44% | 41% | 0.42 |

The human-brain rows come from a laptop trial; the full run with fine-tuning is notebook 03 on a GPU.

What went wrong, and what it taught:
- v1 put several dots along long, tube-shaped mitochondria. Raising the cutoff from 0.4 to 0.7 fixed most of it; counting one dot per glowing blob did no better.
- Stretching image contrast made v1 find nothing at all. Every picture is now shifted to the brightness it learned on.
- On human tissue the zoom and the slice spacing (30 nm, not 5 nm) mattered as much as the tissue itself.

## Use it

Double-click `MitoFinder.bat`: it marks every new picture in `try/` and opens the result. Or:

```
py -3 find_mitos.py picture.png
py -3 find_mitos.py stack.tif 80
```

## Notebooks (run on Google Colab with a T4 GPU)

1. `01_look_at_the_data`: the EPFL data, how rare mitochondria are, how big, how they continue across slices
2. `02_build_mitofinder`: answer sheet of glowing dots, 3-slice patches, a U-Net, training, scoring
3. `03_human_tissue`: human brain (MitoEM-H), one difference fixed at a time, then teaching it on human slices
4. placenta: next (see [ROADMAP.md](ROADMAP.md))

## Data

- EPFL CVLab, Electron Microscopy Dataset (CA1 hippocampus, 5 nm)
- MitoEM-H (Wei et al. 2020), read from [MitoVerse](https://huggingface.co/datasets/pytc/MitoVerse)
- Placenta: [EMPIAR-10967](https://www.ebi.ac.uk/empiar/EMPIAR-10967/), SBF SEM of human term placental villi (R. M. Lewis, Southampton), CC0

MitoFinder measures. It does not diagnose anything.
