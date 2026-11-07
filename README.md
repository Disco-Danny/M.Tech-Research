# Adaptive frame-budget sign recognition

An M.Tech research proof of concept for mentor review, comparing CPU-based
frame-processing policies on isolated Indian Sign Language words and a small
Nepali alphabet-sign subset.

**Research question:** Can a four-to-eight-frame adaptive policy reduce feature
computation while retaining recognition quality relative to fixed sampling?

This is isolated-sign classification, **not** continuous sign-to-text
translation. The methods are three frame-processing policies, not three
neural-network architectures. Novelty and publication readiness have not yet
been established.

## Methods

All methods use 48-by-48 grayscale images, a simplified cell-normalized
HOG-style descriptor, and a standardized linear SGD classifier with hinge loss.
One image yields 324 values; one clip yields 2,916 values.

| Method | Frame and feature policy |
|---|---|
| All frames | Describe every frame, average image descriptors into eight temporal bins, and classify. |
| Uniform 8 | Describe up to eight evenly spaced frames and classify. |
| Adaptive 4 to 8 | Describe a nested four-frame subset first. If its top-two classifier score gap is below a validation-selected threshold, compute additional descriptors and use the eight-frame classifier. |

Short clips use fewer unique frames and interpolate descriptors to maintain
the input size. All methods append a mean adjacent-selected-frame difference
descriptor. The adaptive path caches and reuses image and transition work
within the clip. It stores two classifiers.

**Important:** Uniform and adaptive decoding both traverse the compressed
video and retrieve up to eight images. Adaptive processing initially describes
four of those images; it does not exit the video early or promise skipped
codec decoding.

## Repository contents

```text
speed_poc.py                         Benchmark implementation and docstrings
sign_speed_poc.ipynb                 Interactive review and visualization
requirements-speed.txt              Pinned benchmark dependencies
tests/test_speed_poc.py              Six data-independent contract tests
speed_poc_py_explained.txt           Detailed implementation explanation
sign_speed_poc_notebook_explained.txt Cell-by-cell notebook explanation
results/speed_poc/isl/               ISL metrics, timings, predictions, models
results/speed_poc/nepali/            Nepali metrics, timings, predictions, model
```

Report-building scripts, historical reports/presentations, the stale earlier
frame-selection notebook, GRU/CDE experiments, hand-landmarker assets, local
environments, and duplicate review-package copies are intentionally excluded.
**Datasets are intentionally excluded.** This repository contains code and
saved results for review, not video data. Dataset folders are ignored by Git
to avoid accidentally publishing them in a later commit.

## Quick start

The recorded benchmark and tests were run with **Python 3.9.6**. The pinned
dependencies are the reference environment; compatibility with other Python
versions has not been certified.

```sh
git clone https://github.com/Disco-Danny/M.Tech-Research.git
cd M.Tech-Research
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-speed.txt
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\Activate.ps1` rather than `source`.
Run commands from the repository root.

- The unit tests use synthetic inputs and temporary indexing files. They do
  not need the real videos or replace the saved results.
- `python speed_poc.py` performs dataset-backed smoke checks, including one
  video from each dataset. **It does not run the full benchmark and cannot run
  from this repository until the separately obtained datasets are added.**
- No pretrained neural model or MediaPipe installation is needed.

### Inspect saved results without retraining

Open the JSON and CSV files directly, or display a table:

```sh
python -c "import json; from pathlib import Path; import speed_poc; report = json.loads(Path('results/speed_poc/isl/metrics.json').read_text()); print(speed_poc.table(report)); print(speed_poc.table(report, kind='speed'))"
```

Replace `isl` with `nepali` to inspect that dataset.

### Run the full experiment

```sh
python -c "from pathlib import Path; import speed_poc; speed_poc.run(Path.cwd(), repeats=2)"
```

This trains the models and measures all three policies separately. It can take
several minutes and **overwrites** the tracked metrics, predictions, timings,
and fold models. Preserve or commit a snapshot before experimenting. The two
repeats are timing measurements, not extra accuracy samples.

Before running, obtain the datasets separately from the researcher and place
`Cropped_Data/` and `External_Data_NS23/` at the repository root. ISL clips must
retain their `*_Cropped/*.avi` folder/filename structure. The Nepali folder must
contain the videos and original `manifest.csv`, including `local_file`, `split`,
`gloss`, and `volunteer` fields. Do not guess or regenerate the original splits.

### Notebook

Open `sign_speed_poc.ipynb` in VS Code or Jupyter with the reference environment
selected. Start from the repository root and execute cells in order.
`RERUN=False` loads the current saved JSON reports; `RERUN=True` retrains and
overwrites benchmark artifacts.

The upload copy has **cleared execution counts and embedded outputs** to avoid
publishing stale tables, animations, and machine-specific paths. Running the
notebook regenerates its displays from the saved results and separately
provided videos. Its setup cell runs dataset-backed checks and its animation
reads a video, so it cannot execute end-to-end without the datasets. For a
data-free review, inspect notebook source and the JSON/CSV files, or use the
saved-result command above.
The detailed explanation guides retain historical October results and are
useful for concepts, not authoritative current timing values.

## Evaluation protocol

### ISL

- 412 clips, eight word classes, 26 filename-derived groups.
- Five-fold stratified group cross-validation, fixed seed 42.
- Each clip receives one out-of-fold prediction.
- Inside each outer training fold, one grouped inner holdout selects the gate.
- Filename groups are not verified signer identities.

### Nepali subset

- 15 clips, five alphabet glosses: GA, KA, LA, PHA, RA.
- Ten training clips from S2/S3; five held-out clips from S1.
- The manifest specifies local file, gloss, volunteer, source path, and split.
- One training volunteer is used for inner fitting and the other for gate
  validation; final models are refitted on both training volunteers.
- Five test clips are only a decoding/evaluation demonstration, not strong
  evidence of generalization.

The scaler and classifier are fitted only on permitted training rows.
Label-free features are precomputed for fitting, but those descriptor caches
are not shared across independently measured inference methods.
The gate seeks the smallest rescue fraction whose validation macro F1 is
within **0.02 absolute** of the eight-frame model. Test labels are not used
to select it. The validation constraint is not a held-out performance guarantee.

## Latest recorded results

**Run date: 5 November 2026.** One benchmark run, two timing repetitions per
method per held-out clip. The upload does not represent a new benchmark run.

| Dataset | Method | Accuracy | Macro F1 | Mean latency/clip |
|---|---|---:|---:|---:|
| ISL | All frames | 93.20% | 0.9316 | 147.77 ms |
| ISL | Uniform 8 | 91.75% | 0.9164 | 46.56 ms |
| ISL | Adaptive 4 to 8 | 91.26% | 0.9109 | 46.09 ms |
| Nepali | All frames | 60.00% | 0.4667 | 146.01 ms |
| Nepali | Uniform 8 | 60.00% | 0.5000 | 21.35 ms |
| Nepali | Adaptive 4 to 8 | 60.00% | 0.4667 | 21.01 ms |

The adaptive method reduced descriptor work but had only a small observed
latency difference from Uniform 8 on this run. This is not evidence of a
statistically significant speed or accuracy improvement.

Each result folder includes:

| File | Meaning |
|---|---|
| `metrics.json` | Aggregate quality, latency, resource proxies, fold groups, thresholds, and recorded platform/OpenCV information. |
| `predictions.csv` | Held-out labels, group IDs, and predictions for all three policies. Paths were converted to repository-relative form for portability; labels and predictions are unchanged. |
| `timings.csv` | Per-clip, per-repeat timing rows: total, loading, feature/policy, and model milliseconds. |
| `fold_*.pkl` | Trained pipelines, class lists, and thresholds. Five files for ISL, one for Nepali. |

ISL has 412 prediction rows and 2,472 timing rows; Nepali has five prediction
rows and 30 timing rows. Metrics, timings, and model files were
copied unchanged from the recorded run. Prediction paths are repository-relative
references to videos that are not included.

Only deserialize pickle files you trust: pickle loading can execute code.
Use the reference package versions when loading stored scikit-learn models.

## What the guide should review

Suggested reading order: this README -> `speed_poc.py` ->
`tests/test_speed_poc.py` -> saved JSON/CSV results -> notebook and explanations.
Please reference function names and line numbers in feedback or GitHub issues.

1. Correctness of descriptor construction, temporal pooling, interpolation,
   short-clip behavior, and adaptive cache reuse.
2. Separation of outer tests and inner gate selection; verified group metadata.
3. Transfer of a raw score threshold from inner models to larger refitted
   models, whose margin scale can change.
4. Fairness of inference timing and resource accounting.
5. Additional fixed four-frame and matched sampling baselines.
6. Experiments required before a journal/conference submission: larger
   signer-verified test sets, repeated trials, group-aware uncertainty estimates,
   threshold/quality trade-off ablations, and background robustness.

## Limitations and reproducibility notes

- ISL group provenance and dataset source/license details need documentation.
- Nepali word/alphabet tasks and ISL word tasks are separate; results must not
  be pooled into a multilingual accuracy claim.
- Whole-image features can learn background shortcuts and lose fine details.
- HOG is simplified, not an exact Dalal--Triggs implementation.
- Raw margin gaps are not calibrated probabilities.
- Operating-system file caches may be warm. Thread pools are limited to one
  during `run`, but latency still depends on hardware, load, and codecs.
- Descriptor counts, gray-buffer sizes, and serialized model sizes are
  resource proxies, not measured energy, FLOPs, or peak RAM.
- Two repetitions do not support robust tail-latency or significance claims.
- Package pins, fixed seeds, and recorded fold groups help
  reproduce this snapshot once the original datasets are supplied; a comprehensive experiment provenance record and
  uncertainty analysis remain future work.

## Dataset attribution and access

No datasets or dataset manifests are distributed in this repository.
Contact the researcher for the exact evaluation subsets. No general license
for the code has been assigned here.

- ISL: 412 cropped word clips; the original source citation and formal
  redistribution license still need documentation. Seek owner clarification
  before reuse.
- NSL23: the source dataset card states **CC BY 4.0**. Preserve the original
  attribution and local split manifest when obtaining or reusing the subset.
- NSL23 source: <https://huggingface.co/datasets/Titung/nepali-sign-language-video>.
- Jhuma Sunuwar, Samarjeet Borah, and Aditi Kharga, "NSL23 dataset for alphabets
  of Nepali sign language," *Data in Brief* (2024),
  <https://doi.org/10.1016/j.dib.2024.109986>.
