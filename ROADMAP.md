# MitoFinder roadmap

Written 13 Oct 2026. Every number below was measured in this repo or read from the source named next to it.

## Where it stands

| Step | Result |
|---|---|
| Notebook 01: look at the data (EPFL mouse brain, 5 nm) | 5.6% of voxels are mitochondria, about 15 per slice |
| Notebook 02: MitoFinder v1, trained on a Colab T4 (3,000 steps) | test F1 0.77: found 93%, 65% of dots real |
| `experiments/dots_fix.py`: cutoff 0.4 -> 0.7, chosen on training slices | **test F1 0.84**: found 82%, 85% of dots real |
| `MitoFinder.bat` / `find_mitos.py` | works on any picture; matches EPFL brightness first (without that, v1 finds nothing) |
| Notebook 03: human brain (MitoEM-H), laptop trial | v1 F1 0.01 as-is -> 0.21 brightness -> 0.38 zoom -> 0.42 same slice x3. Fine-tuning needs the Colab GPU |

## Next steps, in order

1. **Run notebook 03 on Colab** (T4, disconnect and reconnect, Run all, upload `mitofinder_v1.pt`). Record the A-E table and download `mitofinder_v1h.pt`.
2. **Notebook 04: placenta.** Data: EMPIAR-10967, *SBF SEM of Human term placental villi* (Rohan Lewis, Southampton; 90.2 GB; one 16-bit TIFF per slice, mostly 4.2 nm pixels, slices 50 nm apart; healthy pregnancies). Confirm its licence on the EMPIAR page before publishing anything made from it.
   - Download about 20 slices of one stack. Match brightness and zoom (4.2 -> 5 nm). Slices 50 nm apart, so use the same slice x3.
   - **Answer sheet = your clicks.** A small click tool: you mark mitochondrion centres on about 10 slices, just like Mito Mapper.
   - Score v1 and v1h on half your slices, teach on the other half, score again.
3. **Make it public** (your push): GitHub repo with a README table of every score, including the failures.
4. **Show it to the people who use it** (you write the posts):
   - Placenta Profiles Talk board, with the placenta score, asking whether it would help the team.
   - image.sc forum (tag `bioimageio`), where bioimage-analysis researchers look.
   - Later: BioImage Model Zoo (bioimage.io), so any lab can run it.
5. **The disease step:** pre-eclampsia and fetal growth restriction images are with the Placenta Profiles team, not public. Ask them, once step 4 shows a real placenta score. If they agree: compare mitochondria counts and sizes between healthy and pre-eclampsia placentas. That is an IRIS 2027 project.

## Rules we keep

- Choose settings on training data, score the test data once.
- Classify on Zooniverse with your own eyes, never MitoFinder's.
- It measures; it never diagnoses.
