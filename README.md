# Multimodal Survival Prediction — Minicourse

A step-by-step tutorial for building a multimodal (WSI + mpMRI + clinical) prediction pipeline for
prostate cancer biochemical recurrence (BCR), on the
[CHIMERA challenge](https://chimera.grand-challenge.org/) Task 1 dataset.

The labels support **two tasks**, from the same `BCR` / `time_to_follow-up/BCR` fields:
**classification** (binary BCR event) and **survival analysis** (time-to-event, censoring-aware).

## Course roadmap

| # | Notebook | Content |
|---|----------|---------|
| 00 | `00_environment_setup.ipynb` | Install `uv`, sync the environment, register the Jupyter kernel |
| 01 | `01_data_preparation.ipynb` | Download data, WSI patch prep, clinical encoding, label prep + split |
| 02 | `02_feature_extraction.ipynb` | H-optimus-0 (WSI) and MRI-PTPCa (MRI) embeddings — single-pass, no layer aggregation |
| 03 | `03_fusion_models.ipynb` | Early / intermediate / late fusion architectures |
| 04 | `04_train.ipynb` | Train both tasks across all three fusion strategies and every CV fold |
| 05 | `05_inference_evaluation.ipynb` | Inference, slide-to-patient aggregation, post-processing, evaluation & visualization |

A companion course site (GitHub Pages, `docs/`) walks through the same material with worked
explanations and links each section to its notebook, open-able directly in Colab.

## Structure

```
minicourse-multimodal/
├── notebooks/               # the course itself — run these in order, starting at 00
├── src/                     # model-architecture code shared verbatim across notebooks
│                            # (mri_ptpca.py: adapted third-party model; fusion_models.py:
│                            #  finalized versions of what notebook 03 develops/explains)
├── data/                    # created by the notebooks; gitignored
│   ├── raw/task1/           # CHIMERA data (downloaded, or symlinked from an existing copy)
│   ├── prepared/            # clinical embeddings, WSI patch manifests
│   ├── labels/              # train/test label CSVs with CV folds
│   └── features/            # WSI/MRI model embeddings (notebook 02+)
├── pyproject.toml           # uv-managed project + dependencies
└── README.md
```

**Convention:** everything that's part of the tutorial — data wrangling, preprocessing,
training loops — is written directly in the notebooks, with explanations, so you see every step
run. `src/` holds two kinds of exceptions: large third-party model architectures that genuinely
can't be inlined without hurting readability (`mri_ptpca.py`), and finalized code a notebook
develops/explains inline but that a *later* notebook also needs verbatim to avoid ~150 lines of
duplication (`fusion_models.py` — notebook 03 defines and explains every class in it; notebook 04
imports the same file to actually train them).

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/). See
`notebooks/00_environment_setup.ipynb` for the full walkthrough (installing `uv`, syncing the
environment, registering it as a Jupyter kernel). Quick version:

```bash
uv sync
uv run python -m ipykernel install --user --name minicourse-multimodal --display-name "minicourse-multimodal"
```

WSI patching/feature extraction needs the OpenSlide C library. `openslide-bin` (in
`pyproject.toml`) bundles a prebuilt copy, so no separate system install is needed — handy on
clusters where you can't `apt`/`brew` install things yourself.

Downloading the dataset needs AWS CLI credentials configured, though the CHIMERA bucket itself is
public and `--no-sign-request` (or symlink in an existing copy — see notebook 01, Step 1).

## Attribution

- Dataset & task: [CHIMERA challenge](https://chimera.grand-challenge.org/), Task 1 (prostate
  cancer biochemical recurrence prediction).
