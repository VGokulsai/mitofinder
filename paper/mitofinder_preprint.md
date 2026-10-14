# MitoFinder: learning mitochondrion centres from point annotations, and why it fails on placenta without more labels

**Gokul Sai**
Independent student researcher, Hyderabad, India
Code and models: github.com/VGokulsai/mitofinder

## Abstract

Citizen-science projects such as *Science Scribbler: Placenta Profiles* ask volunteers to click the centre of each mitochondrion in electron-microscopy (EM) images, partly so that computers can later be trained to do the same. We built MitoFinder, a small U-Net that learns from points rather than outlines. Each annotated centre becomes a Gaussian "glow" target, and predicted glows are turned back into points. We report point-detection F1, where a prediction counts if it lies within 75 nm of an annotated centre; all settings were chosen on held-out data and each test set was scored once. On mouse hippocampus (EPFL, 5 nm), MitoFinder reaches F1 0.84 on 163 unseen slices. On human cortex (MitoEM-H) it scores 0.41 after matching brightness and zoom, and 0.58 after fine-tuning on 64 human slices. On human term placenta (EMPIAR-10967, CC0), for which no public mitochondria labels exist, the brain-trained models score 0.04 to 0.12 against 28 centres annotated by the author. Fine-tuning on 13 author-annotated placenta centres cut false detections about four-fold but left the model too conservative. The published generalist MitoNet scores 0.37 on the same points. Our results suggest that the bottleneck for placenta is labelled data rather than architecture, and volunteer point annotations are the input this model lacks. All code, notebooks, models and the author's annotations are released.

## 1. Introduction

Placental mitochondria differ between healthy and pre-eclamptic pregnancies; electron microscopy has shown degenerative changes in pre-eclamptic placental mitochondria [7]. Serial block-face scanning EM (SBF-SEM) can image them across whole volumes, but measuring them by hand does not scale. Supervised segmentation models need pixel outlines, which are slow to draw. For placenta, we found no public outline set: none of the 212 labelled mitochondria volumes in the MitoVerse collection is placental [8].

Point annotations are much cheaper. *Placenta Profiles* (Zooniverse, launched November 2020; teams at Manchester, Southampton, the Rosalind Franklin Institute and Diamond Light Source) collects them at scale in its Mito Mapper workflow [6]. This work asks how far a model trained only on centre points can get, and what it needs to work on placenta.

## 2. Data

| Set | Tissue | Pixel size | Labels used | Use |
|---|---|---|---|---|
| EPFL CA1 hippocampus [1] | mouse brain, FIB-SEM | 5 nm isotropic | expert masks, 165 + 165 slices | training / test |
| MitoEM-H [2] via MitoVerse [8] | human cortex, ssSEM | 8 x 8 x 30 nm | expert instance masks; three 1024 x 1024 blocks | fine-tune (z 0-63), choose settings (z 192-255), test (z 448-499) |
| EMPIAR-10967, stack 39 [5] | human term placenta, SBF-SEM | 5 nm, 50 nm slices | author's clicks | fine-tune (12 crops, 13 centres), test (12 crops, 28 centres) |

Point targets were derived from masks as the deepest interior point of each object (the maximum of the distance transform). Unlike the centroid, this point stays inside curved mitochondria. Placenta crops (1024 x 1024 at 5 nm) were taken on a fixed grid from slices 20 apart (1 µm), so test and training crops show different cells. Crops that were mostly empty background were removed by an image-only rule. The remaining crops were drawn with a fixed random seed; no model output was used to choose them.

## 3. Method

**Targets.** Each centre becomes a Gaussian of σ = 6 px, scaled by 2πσ² so that its peak is exactly 1 and all objects carry equal weight.

**Model.** A 2-D U-Net [4] (base 32 channels, three down-sampling levels, about 1.9 M parameters) takes three adjacent slices as channels and outputs one sigmoid glow map. For anisotropic or widely spaced data, the same slice is repeated three times.

**Training.** Patches are 256 x 256 px; 70% are centred near a labelled object and 30% are random. Augmentation is random 90° rotation and flipping. The loss is the mean of (1 + 20·target)·(prediction − target)², so errors near a real centre cost up to 21 times more. On EPFL we trained for 3,000 steps with batch size 8 and Adam with a one-cycle schedule (maximum learning rate 2e-3) on a Colab T4 GPU. Fine-tuning starts from the previous model with learning rate 3e-4, for 800 steps (human) or 600 steps (placenta, adding brightness, contrast and noise jitter).

**From glows to points.** Points are local maxima above a cutoff, or one point per connected region above the cutoff. The cutoff and rule are always chosen on data not used for testing.

**Input matching.** Every image is shifted to the training brightness and contrast (mean 140.5, s.d. 30.4 on a 0-255 scale) and resampled to 5 nm pixels.

**Scoring.** Predictions and annotations are matched greedily, closest pairs first, within 15 px (75 nm). Recall is matched / annotated, precision is matched / predicted, and F1 is their harmonic mean.

**Baseline.** MitoNet v1 [3] (quantised weights, CC-BY-4.0) was run with its default thresholds. Its points are the predicted instance centres that fall inside predicted mitochondria.

## 4. Results

| Test data | Model and settings | Found | Precision | F1 |
|---|---|---|---|---|
| EPFL, 163 unseen slices (2,596 centres) | MitoFinder v1, cutoff 0.4 | 93% | 65% | 0.77 |
| same | cutoff 0.7, chosen on training slices | 82% | 85% | **0.84** |
| MitoEM-H test block (50 slices) | v1, no input matching (every 8th slice) | 1% | 7% | 0.01 |
| same | + brightness and zoom matched, settings from a separate block | 42% | 40% | 0.41 |
| same | **v1h**: fine-tuned on 64 human slices | 62% | 55% | **0.58** |
| Placenta, 12 crops, 28 centres | v1 (cutoff from EPFL) | 14% | 2% | 0.04 |
| same | v1h (cutoff from human brain) | 21% | 8% | 0.12 |
| same | v1p: v1h fine-tuned on 13 placenta centres, cutoff chosen on held-out crops | 14% | 7% | 0.09 |
| same | MitoNet v1, default settings | 29% | 53% | **0.37** |

**Input statistics matter as much as tissue.** With raw MitoEM-H input, the EPFL model found almost nothing (F1 0.01). Matching brightness, then zoom, then slice spacing recovered F1 0.41 without new training. In the other direction, stretching the contrast of an EPFL test slice to the full 0-255 range made the model detect nothing on it.

**Fine-tuning helps across tissue, but only if the cutoff is chosen again.** The model fine-tuned on human cortex scored 0.38 with the old cutoff and 0.58 once the cutoff was re-chosen on a separate block. It also did better on placenta (0.04 to 0.12), which it never saw during training.

**Sparse labels teach the wrong lesson.** On placenta, fine-tuning on 13 centres reduced false detections from 1,600-5,000 to 50-160, depending on the cutoff, but the model then missed most mitochondria. The training crops contained many mitochondria that were not clicked, and the loss treated them as background.

## 5. Discussion and limitations

The placenta result is a data result. MitoFinder learned brain mitochondria well (F1 0.84) from about 2,500 annotated centres. Placenta has a different texture, contrast and set of look-alikes, including microvilli and vesicles, and 13 points were not enough to teach the model these differences. Two fixes follow from this. The first is complete annotation inside small regions, so that an unlabelled pixel really means "not a mitochondrion". The second is many more points, which is what Mito Mapper collects. MitoNet did better on our placenta test without any placenta training, and could propose candidates for people to accept or reject. Its outputs still need review: on our training crops, MitoNet and the author agreed on only 1 of 13 centres, and some of its proposals fell on microvilli.

The study has clear limitations. The placenta answer key comes from one non-expert annotator (the author), on 24 crops from a single stack from an uncomplicated pregnancy. The placenta numbers are therefore rough; one match more or less moves F1 by several points. Faint, low-contrast candidates were the least certain: in one crop, MitoNet detected none of the 11 the author marked. All results are 2-D detection of centres, not segmentation or 3-D instance tracking. No disease samples were used. The tool measures; it does not diagnose.

## 6. Code, data and model availability

Code, Colab notebooks, trained models (v1, v1h, v1p), the author's placenta annotations and every experiment script are at github.com/VGokulsai/mitofinder. Data sources: the EPFL CVLab EM dataset [1]; MitoEM-H [2] via MitoVerse [8]; EMPIAR-10967 [5] (CC0); MitoNet weights [3] (Zenodo 6861565, CC-BY-4.0).

## Acknowledgements

I thank the *Science Scribbler: Placenta Profiles* team and its volunteers, whose project motivated this work; R. M. Lewis and colleagues for releasing EMPIAR-10967; the EPFL CVLab, the MitoEM authors and the MitoVerse maintainers for open data; and R. Conrad and K. Narayan for releasing MitoNet. This work was done independently, without institutional affiliation or funding, using free Google Colab GPUs. Claude (Anthropic) was used as a coding and writing assistant; all experiments were run and checked by the author.

## References

1. Lucchi A. et al. Supervoxel-based segmentation of mitochondria in EM image stacks with learned shape features. IEEE Transactions on Medical Imaging, 2012.
2. Wei D. et al. MitoEM dataset: large-scale 3D mitochondria instance segmentation from EM images. MICCAI, 2020.
3. Conrad R. & Narayan K. Instance segmentation of mitochondria in electron microscopy images with a generalist deep learning model trained on a diverse dataset. Cell Systems, 2023.
4. Ronneberger O., Fischer P. & Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI, 2015.
5. Lewis R. M. et al. SBF SEM of human term placental villi. EMPIAR-10967, 2022. doi:10.6019/EMPIAR-10967.
6. Science Scribbler: Placenta Profiles. Zooniverse, 2020-present.
7. Comparative proteomics analysis suggests that placental mitochondria are involved in the development of pre-eclampsia. PLoS ONE, 2013. PMC3650054.
8. MitoVerse: a generalist 3D-EM mitochondria segmentation benchmark. PyTorch Connectomics, Hugging Face dataset pytc/MitoVerse, 2026.
