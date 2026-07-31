# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A step-by-step multimodal-survival-prediction minicourse (WSI + mpMRI + clinical → prostate
cancer biochemical recurrence), based on
[HIMF-Surv](https://github.com/bowang-lab/HIMF-Surv) (local reference copy:
`/scratch/yws0322/minicourse/HIMF-Surv`) and the same CHIMERA challenge Task 1 dataset. See
`README.md` for the full course roadmap and dataset/attribution details.

Deliberate differences vs. HIMF-Surv:
- No layer-wise feature aggregation — just the final embedding per patch/scan (last transformer
  block only, both for WSI and MRI).
- WSI patch embeddings use **H-optimus-0** (`hf-hub:bioptimus/H-optimus-0` via timm), not UNI.
  1536-dim output, 224x224 input patches at 0.5 microns/pixel, gated on HuggingFace (needs a token
  — see `.env` below). Normalization mean/std: `(0.707223, 0.578729, 0.703617)` /
  `(0.211883, 0.230117, 0.177517)` — not standard ImageNet stats.
- MRI-PTPCa's third input channel is real data here, not zero-filled: HIMF-Surv's
  `feature_extractors/mri.py` only ever loads T2W + ADC and zeros the model's expected third
  (DWI) input, but CHIMERA actually ships a matching `_hbv.mha` (high-b-value) series per scan —
  `src/mri_ptpca.extract_embedding` uses it as the real DWI input, falling back to zero-fill only
  if a scan is missing its HBV file.
- Three fusion strategies (early / intermediate / late) shown side by side instead of one fixed
  hierarchical-fusion architecture. All three pool WSI patches with the *same* `ABMIL` module and
  build their task head through the *same* `build_task_head`/`PredictionHead` (a 3-layer MLP,
  matching HIMF-Surv's `MLPPredictionHead` exactly) — so the comparison between fusion strategies
  isolates the fusion mechanism itself, not incidental differences in pooling or head capacity.
  Intermediate fusion uses one self-attention pass + mean pool (not a deep transformer stack) at
  `shared_dim=256` (HIMF-Surv uses 1536 with an 18-layer/12-head transformer — deliberately
  shrunk here so all three strategies stay in a comparable parameter regime on this course's
  ~95-patient cohort instead of intermediate fusion dwarfing the other two and overfitting).
- Multi-slide patients (CHIMERA patients have 1–12 WSI slides each — HIMF-Surv's own
  `feature_extractors/wsi.py` globs only `*_1.tif`, i.e. it silently keeps just the first slide
  per patient and drops the rest) are handled here instead by using every slide, following the
  actual convention of [nnMIL](https://arxiv.org/abs/2511.14907) (verified against a local copy
  of its code at `/home/yws0322/scratch/autoMIL/benchmarks/lib/nnMIL`): each of a patient's
  slides becomes its own **training row** — that patient's label (BCR event/time) is copied onto
  every one of their slides, and their MRI/clinical vectors are repeated per slide — so training
  itself is slide-level, with no patient-level merging inside the model at all. Patient-level
  results only get computed at **evaluation** time, by grouping a batch's predictions by
  patient_id and aggregating: mean probability + majority vote for the hard label
  (classification, matching nnMIL's `classification_predictor.py`), mean risk score (survival,
  matching nnMIL's `survival_loss.py::survival_c_index`) — never a mean of raw logits/hazards.
  This means `ABMIL`/`EarlyFusionModel`/`IntermediateFusionModel`/`LateFusionModel` in
  `src/fusion_models.py` need **no multi-slide-specific code at all**: each still just takes one
  WSI bag per batch row, exactly as before. All of the actual multi-slide logic (grouping a
  patient's slide files into training rows with a duplicated label, and the evaluation-time
  aggregation) lives in notebook 04's dataset/evaluation code, per this project's notebooks-first
  convention below — it was deliberately kept out of `src/fusion_models.py` and out of notebook 03
  (an earlier pass merged predictions patient-side *inside* `forward()` via
  `expand_batch_to_slides`/`merge_slide_predictions`, which was reverted once nnMIL's actual
  code showed it aggregates post-hoc at evaluation instead, on probabilities/risk scores rather
  than raw logits).

**Non-Python dependencies handled via pure-Python wheels, not system installs** (important on
shared clusters where users can't `apt`/`brew` install things): `openslide-bin` bundles the
OpenSlide C library so plain `openslide-python` doesn't need a separate system library.

`data/raw/task1` is a symlink to an already-downloaded copy of the CHIMERA dataset on shared
cluster storage (`/home/yws0322/projects/rrg-jma/shared/Multimodal/ProstateCancer/chimera`, 95
patients) rather than something notebook 01 actually re-downloads in this environment.

Two downstream tasks share the same labels (`BCR`, `time_to_follow-up/BCR`): binary
**classification** and censoring-aware **survival analysis**.

## The core convention: notebooks first, `src/` only when unavoidable

**Everything that is part of the tutorial goes directly into the notebook as explained, executable
cells — data download, preprocessing, patch extraction, label prep, training loops, evaluation.**
Do not extract this kind of logic into an importable `.py` module "for reuse" — reuse across
notebooks happens by reading/writing files on disk (patch manifests, embeddings, label CSVs), not
by importing shared pipeline functions.

`src/` holds two distinct kinds of exceptions:

1. Large third-party model architectures that genuinely cannot be pasted into a notebook without
   hurting it: `src/mri_ptpca.py` (the `CNNViTMM`/`VisionNet` model classes, adapted from
   `HIMF-Surv/feature_extractors/mri.py`, plus `load_mri_ptpca_model` and `extract_embedding`).
   Notebook 02 still writes the MRI *preprocessing* function (`preprocess_mri_scan`) and
   `histogram_balance` inline, since those are the part actually worth reading — only the model
   plumbing is in `src/`.
2. Code that *is* the lesson in one notebook, fully defined and explained there, but that a
   *later* notebook also needs verbatim to actually run: `src/fusion_models.py` (`ABMIL`,
   `PredictionHead`/`build_task_head`, `discretize_time`/`NLLLoss`/`concordance_index`,
   `EarlyFusionModel`/`IntermediateFusionModel`/`LateFusionModel`). Notebook 03 develops and
   explains every one of these inline with dummy-tensor sanity checks; notebook 04 imports the
   same file to train them for real. The alternative (04 redefining ~150 lines of model code
   identically) was rejected as pure duplication with no teaching value on the second pass —
   this was an explicit user decision, not a default the convention grants automatically.

Before writing a `.py` file, first ask: could a learner reasonably read this as part of the
notebook narrative? If yes, it belongs in the notebook. If a *later* notebook also needs the exact
same code to run (not just to read), that's when promoting it to `src/` after the teaching
notebook is done with it becomes reasonable — check with the user before assuming this applies
elsewhere, since it was decided case-by-case here.

This was corrected explicitly during initial setup — an earlier draft of notebook 01 wrapped the
download/patching/encoding/label logic into `mmcourse/data_prep/*.py` helper functions, and the
user asked for it to be un-done and moved inline into the notebook instead. The folder itself was
named `mmcourse/` first, briefly renamed to `minicourse_multimodal/` (matching the repo name), and
settled on `src/` — parallel to `notebooks/`, since the repo name was already taken.

## Environment: uv

The project is managed with [uv](https://docs.astral.sh/uv/) — `pyproject.toml` is the source of
truth for dependencies (`[tool.uv] package = false`: this is an environment, not an installable
package — `src/` is imported via `sys.path`, not an editable install), `uv.lock` pins them.
`notebooks/00_environment_setup.ipynb` is the onboarding walkthrough (install uv, `uv sync`,
register the Jupyter kernel, load `.env`). When adding a new dependency needed by a later
notebook, run `uv add <package> --project <repo root>` (updates `pyproject.toml` + `uv.lock` +
installs it) rather than hand-editing either file.

**Secrets:** `.env` (gitignored) holds `HF_TOKEN`, needed to download the gated H-optimus-0
weights in notebook 02. `.env.example` (committed) documents the expected keys. Notebooks load it
with `python-dotenv` + `huggingface_hub.login(token=...)` rather than requiring an interactive
`huggingface-cli login`.

## Building/editing notebooks

There's no `nbformat` installed by default — it comes from `uv sync` once notebook 00 has been
run, but don't assume it's present in an arbitrary shell. Notebooks are built by writing a small
Python script (in a scratch location, not committed) that assembles `{"cell_type", "source", ...}`
dicts into the nbformat-4 JSON structure and writes it with `json.dump`. Use triple-single-quoted
(`'''...'''`) Python strings for cell source text so that code cells containing `"""docstrings"""`
don't collide with the wrapper string's delimiters. Once a notebook already exists, prefer
`NotebookEdit` for further changes (it requires reading the notebook first).

After writing/editing a notebook: `json.load` the file (well-formed JSON) and `ast.parse()` every
code cell's joined source (syntactically valid Python) at minimum. But since the CHIMERA data
itself is present (via the `data/raw/task1` symlink) and `uv run` gives a real environment, prefer
going further where possible:

- If the notebook has no hard external blocker (no real data dependency, no gated-model access
  needed), concatenate every code cell's source in order and run the whole thing as one script
  with `uv run python <script>` — this is a real top-to-bottom execution check, not just syntax
  validation, and it's cheap when the notebook is self-contained (e.g. notebook 03's
  architecture-only, dummy-tensor sanity checks).
- If it does depend on real data or gated model access, extract the relevant function
  definitions with a small script and run them against real files under `data/raw/task1`
  (`uv run python -c "..."`) — this caught real bugs during development that syntax-checking
  alone would have missed (e.g. the `openslide-python` system-library gap, fixed by switching to
  `openslide-bin`). Full end-to-end execution of those notebooks still isn't practical from an
  arbitrary shell (H-optimus-0 needs a real HF-gated token + access grant; a substitute non-gated
  `timm` ViT of the same architecture shape is a reasonable stand-in to test the surrounding
  plumbing).

## Data

`data/` (raw downloads, prepared artifacts, labels, features) is gitignored — it's regenerated by
running the notebooks in order (or symlinked in, for the raw data — see notebook 01). Don't commit
anything under it.
