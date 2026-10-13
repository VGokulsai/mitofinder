# What already exists, what won, and what we copy

Collected 13 Oct 2026. Every number was read at the source linked next to it; anything not confirmed says so.

## 1. MitoFinder (mitochondria in electron-microscope images)

### Already built
| Who | What | Where it stands |
|---|---|---|
| Placenta Profiles team (Southampton, Manchester, Rosalind Franklin Institute, Diamond, Zooniverse) | Volunteers click mitochondrion centres (Mito Mapper) and check them (Mito Mop-Up), partly to train AI | ~40,000 Mop-Up subjects finished by Aug 2026, crowd said 43% yes / 57% no. In their words: "We have no ground truth for our data" (smith_p, Zooniverse staff, Talk board, 28 Aug 2026) |
| MitoNet / empanada (Conrad & Narayan, NCI; Cell Systems 2023) | General mitochondria AI, napari plugin | The best-known general model. A fair comparison for us later |
| MitoVerse (PyTorch Connectomics lab, Hugging Face, updated Jun 2026) | 212 labelled volumes, 29,833 mitochondria, 13 sources | **No placenta** |
| Southampton (R. M. Lewis) | Placenta images, EMPIAR-10967, CC0 | No mitochondria labels |

### Competitions
- **Regeneron ISEF 2023, 2025, 2026:** no AI-for-electron-microscopy project and no placenta project in the award lists (2026 read to about 85%). Mitochondria projects that won were wet-lab: 2023 CELL036 First Award (ER-mitochondria contacts); 2026 BCHM009 Second, BMED055 Third, CELL036 Fourth. The closest tool-building one: 2026 MATS077T, automating an electron microscope with AI agents (materials), Fourth.
- **IRIS (India):** nothing similar found. Closest medical-AI winner: MOUTHSCOPE (oral pre-cancer, imaging + AI), Grand Award 2022.
- ISEF 2026 prizes per category: 1st $6,000, 2nd $2,400, 3rd $1,200, 4th $600.

### What we copy
- From the crowd projects: learn from **dots**, not outlines. Clicks are cheap, outlines are not.
- From MitoVerse / MitoEM practice: fix one difference at a time (brightness, zoom, slice spacing) and score after each.
- From every good paper: choose settings on one set, score the test set once.

### What we do that they did not
- Placenta, which nobody's benchmark has.
- Taught from a volunteer's clicks, scored against a volunteer's clicks.
- Every failure published next to the wins (v1 placenta score as-is: F1 0.02).

## 2. TB from cough sounds (project two)

### Already built
| Who | What | Score | What went wrong |
|---|---|---|---|
| CODA TB DREAM challenge, 11 teams (OFID 2025, PMC12502651) | CNNs, gradient boosting, LightGBM on cough sounds; 2,143 people, 7 countries incl. India | Best cough-only AUROC **0.743** (Blue Team); #2 tie was a **high-school team** (0.731) | Patient info alone scored 0.817, so the cough added little. All trained from scratch. 0.5 s clips. 714,922 longer recordings unused. HIV not an input |
| Same models on new patients in Peru (PMC13462625) | No retraining | Blue Team 0.743 -> **0.615**; high-school team 0.731 -> 0.541 | Different phone, different local diseases, different ages |
| Zambia study (arXiv 2509.09746) | Speech foundation models (wav2vec2 etc.), 3-second clips, averaged per person | AUROC 0.852 (audio), 0.921 with patient info | Never says each person stayed on one side of the split; one region only |
| EPFL, Zhang et al. (arXiv 2608.25846, 26 Aug 2026) | Tested cough models across 3 datasets | Up to 0.755 inside a dataset, often **below 0.6** outside | Models group coughs **by phone and dataset, not by TB**. Device-diverse training helps |
| Google HeAR (arXiv 2403.02522) | Health-sound foundation model, 300M clips, 2 s / 16 kHz input | Built to hold up across microphones | Terms: Health AI Developer Foundations; Hugging Face login |
| IIT Kanpur with the Health Ministry | TB-from-cough model in India | Not checked | |

### Competitions
| Year | Who | Project | Result |
|---|---|---|---|
| ISEF 2023 | Chandra Suda (Arkansas) | Smartphone cough model; claims 94% sensitivity / 87% specificity | Fourth Award, $500 |
| ISEF 2024 | Michael Xu & Faisal Nabulsi | SCOMAE: pre-trained on lots of audio, then TB coughs; AUROC 0.843 | Fourth Award, $500 + USAID First Award (global health) |
| ISEF 2025 | Nathan Ma & Samer Rahman | Lung-sound classification | Fourth Award |
| IRIS | none found | | |

Both TB winners reported one score on one dataset and stopped at Fourth.

### What we copy
- Pre-trained sound models (HeAR, wav2vec2), not training from scratch (Zambia, SCOMAE).
- Several seconds of audio, averaged per person (Zambia: 3 s best).
- Count the coughs (Yuanfang Guan lab, 4th in CODA).
- Combine with patient info by a simple stacking model (Zambia).

### What we do that they did not
1. Beat the honest bar: patient info alone (0.817), not 0.743.
2. Leave one country out, India included; report the worst country.
3. Make it phone-proof: device-diverse training and augmentation, then show it transfers. Failure across sites is already published (EPFL, Aug 2026), so our claim must be a fix that measurably works.
4. Use the 714,922 unused recordings. Give the model HIV status.
5. Split by person, one locked test.

## Where to show the work
| Place | Who reads it |
|---|---|
| Placenta Profiles Talk, "Questions for the Research Team" | The project's scientists and staff (smith_p replies there almost daily) |
| image.sc forum (tag bioimageio) | Bioimage-analysis researchers and tool builders |
| BioImage Model Zoo (bioimage.io) | Labs that download ready-made microscopy models |
| bioRxiv / medRxiv | Biologists / health researchers |
| IRIS 2027 -> Regeneron ISEF | Last chance: about Aug-Oct 2027 |
| Breakthrough Junior Challenge | 2-minute science video, ages 13-18 |
