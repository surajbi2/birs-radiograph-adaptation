# BIRS: budget-indexed readout shrinkage

Code and result files for the paper *Budget-indexed readout shrinkage: harm-constrained local adaptation of
radiograph classifiers built on ImageNet and foundation encoders*.

Authors: Suraj Kumar Ramagiri and Parshuram M. Kamble, Central University of Karnataka.

A site that receives a radiograph classifier with only a few labelled studies must decide whether to refit the
model's final linear layer (the readout) and how far. BIRS interpolates between the transferred readout and a
readout refitted on the local studies, with a weight that depends only on the number of labelled studies:

    w(lambda) = (1 - lambda) * w_source + lambda * w_refit

For each label budget the weight maximises the mean AUC gain over the transferred model, subject to at most 5% of
simulated deployments being made worse. The schedule is calibrated once, on transfers between anatomical regions
of MURA, and applied unchanged to external datasets. Where no non-zero weight meets the cap, the site keeps the
transferred model.

## Contents

| Path | What it holds |
|---|---|
| `code/` | The pipeline for the ImageNet backbones (DenseNet-121, ConvNeXt-Tiny, Swin-Tiny): data layer, training, readout experiments, the BIRS schedule (`birs_schedule.py`) and `make_paper_outputs.py`, which computes every number in the paper |
| `code/paper/` | `paper_numbers.json`: the 2,929 values of the paper, each with a key and a note saying how it was computed (written by `make_paper_outputs.py`) |
| `results/` | The result files `make_paper_outputs.py` reads (per-cell CSVs), each with a provenance header |
| `foundation_arm/` | The foundation-encoder arm (SKELEX, RAD-DINO, MedSigLIP-448, DINOv3-B; fine-tuned and frozen): scripts, per-cell results, fine-tuning results and the summary computed from them |
| `manuscript/` | Scripts that draw the paper's figures and write its foundation-encoder tables from `paper_numbers.json` |
| `PROVENANCE.md` | The hash of every script as it ran and as published, and notes on scripts edited after they ran |

## Reproduce the paper's numbers (CPU, a few minutes)

Every number in the paper is computed from the files in `results/` and `foundation_arm/`. Create an environment
with Python 3.11 or later and install the CPU requirements:

```bash
git clone https://github.com/surajbi2/birs-radiograph-adaptation.git
cd birs-radiograph-adaptation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then, from `code/`:

```bash
cd code
python make_paper_outputs.py --results-root ../results --out ./paper_check
```

The script records the SHA-256 of every input it reads and checks the foundation-arm cells against the hashes in the
foundation-arm summary. To confirm that the run reproduces the paper's values:

```bash
python -c "import json; v = lambda p: {k: x['value'] for k, x in json.load(open(p))['numbers'].items()}; a, b = v('paper_check/paper_numbers.json'), v('paper/paper_numbers.json'); print(len(a), 'values,', 'identical' if a == b else 'DIFFERENT')"
```

which prints `2929 values, identical`. The script also writes working figures and tables into `paper_check/`; the
figures and tables of the paper are made by the scripts in `manuscript/`.

Tables and figures of the paper:

```bash
cd manuscript
python source_model_aucs.py          # source-model wrist AUCs
python foundation_tables.py          # Tables 3 and 7, Supplementary Tables S8-S11
python make_data_figures.py          # Figs. 2-4
python make_overview_thumbnails.py   # radiograph thumbnails for Fig. 1 (needs the datasets)
python make_overview_figure.py       # Fig. 1, as a draw.io file
```

## Run the full pipeline (GPU)

The full pipeline needs the four datasets and a GPU. Each script's docstring gives its exact command. The code calls
its two phases Stage A (source models and transfer between MURA anatomies, written to `results/source_models/`) and
Stage B (dataset shift, external datasets and readout experiments, written to `results/transfer/`).

1. **Index the data:** `mskdata.py --build`, then `mskdata.py --verify`. `mskdata.py` (data layer) and `mskmodel.py`
   (models, training, metrics) are the libraries every stage imports.
2. **Source models and transfer between anatomies (Stage A):** `train_source_models.py`, `source_model_tables.py`;
   `joint_encoder_vs_ceiling.py` and `joint_encoder_report.py` (one encoder trained on all seven anatomies);
   `layerwise_unfreezing.py` with `layerwise_block_maps.py` (which encoder blocks must change);
   `measure_appearance.py` and `shortcut_reliance.py` (how much low-level appearance explains).
3. **Dataset shift and external datasets (Stage B):** `dataset_shift.py` (MURA wrist to GRAZPEDWRI-DX, with the
   preprocessing arms of `preprocessing_variants.py`), `stratify_scores.py`, `fracatlas_transfer.py`,
   `fracatlas_match_stats.py`, `mfxr_external.py` (MFXR reference models and decomposition).
4. **Readout experiments on label budgets:** `readout_probe.py` (merged by `merge_wrist_probe.py`),
   `readout_penalty_sweep.py`, `oracle_readout_cv.py` (de-duplicated by `dedupe_oracle_results.py`),
   `readout_angle.py`, `readout_shrinkage.py` (six MURA targets, GRAZPEDWRI-DX, MFXR),
   `fracatlas_budget_cells.py` (FracAtlas hand), `l2sp_comparator.py` (L2-SP comparator).
5. **BIRS and its evaluation:** `birs_schedule.py`, `external_harm_analyses.py`, `loo_anatomy_policies.py`,
   `bootstrap_intervals.py`.
6. **Cohorts and data audits:** `cohort_counts.py`, `check_patient_overlap.py`, `audit_bit_depth.py`,
   `graz_label_crosstab.py`, `graz_label_audit.py`, `graz_confound_audit.py`, `graz_indomain_audit.py`.
7. **Foundation encoders** (`foundation_arm/scripts/`, with Hugging Face Transformers 4.57.6):
   `feature_reproducibility_gate.py` (gate G2), `arm_a_finetune.py` (fine-tuned arm), `extract_features.py`
   (features for both arms), `arm_b_readout.py` (frozen-arm source readouts), `budget_cells.py gate` then
   `budget_cells.py run` (budget cells); then `foundation_arm/analysis/foundation_summary.py` (summary) and
   `foundation_summary_verify.py` (an independent second pass over it).
8. **Numbers, tables and figures:** `make_paper_outputs.py`, then the scripts in `manuscript/`.

### Paths

The ImageNet-arm stages use the layout `~/diagnosis/{code,data,checkpoints,results}` by default (`mskmodel.py`);
most stages take `--index`, `--ckpt-root` and `--out-dir` to point elsewhere. The foundation-arm scripts import the
ImageNet-arm code from `~/diagnosis/code`, so copy or link `code/` there. Folders that depend on your machine are
placeholders; set them before running:

| Placeholder | Where | What it should point to |
|---|---|---|
| `/path/to/foundation_arm_workdir` | `FA`, `OUT_ROOT`, `G2_DIR`, `OUT` at the top of the five `foundation_arm/scripts/` | A folder for the gate files, fine-tuned weights, features and cells |
| `/path/to/imagenet_feature_cache` | `DEFAULT_CACHE` in `l2sp_comparator.py` (or `--cache-dir`), `PAPER_CACHE` in `extract_features.py` and `budget_cells.py` | The feature cache written by `l2sp_comparator.py extract`; all three must name the same folder |
| `/path/to/imagenet_arm_results` | `RES` in `foundation_arm/analysis/foundation_summary.py` | A folder holding `results/transfer/` and `paper_2026-10-04b/paper_numbers.json`, the ImageNet-arm numbers the summary starts from |
| `/path/to/birs/code` | `layerwise_block_maps.py` (run as a script) | This repository's `code/` folder |

`make_paper_outputs.py` reads the foundation-arm files from `FA_ROOT`, which defaults to `../foundation_arm`, the
repository layout when it is run from `code/`. Hugging Face models are read from `$HF_HOME`.

## Data

The datasets are public and are not redistributed here:

| Dataset | Source |
|---|---|
| MURA v1.1 | Rajpurkar et al., arXiv:1712.06957; Stanford AIMI |
| GRAZPEDWRI-DX | Nagy et al., Scientific Data 9, 222 (2022), doi:10.1038/s41597-022-01328-z |
| FracAtlas | Abedeen et al., Scientific Data 10, 521 (2023), doi:10.1038/s41597-023-02432-4 |
| MFXR | Li et al., Scientific Data (2026), doi:10.1038/s41597-026-07547-y |

## Model weights

ImageNet backbones come from timm (`densenet121.ra_in1k`, `convnext_tiny.in12k_ft_in1k`,
`swin_tiny_patch4_window7_224.ms_in1k`). Foundation encoders come from Hugging Face (`skhoha/SKELEX`,
`microsoft/rad-dino`, `google/medsiglip-448`, `facebook/dinov3-vitb16-pretrain-lvd1689m`); MedSigLIP and DINOv3 are
gated and require accepting their licences. Fine-tuned weights and extracted features are not included: SKELEX is
released under CC BY-NC-ND 4.0, which does not allow sharing modified copies, and features of gated models are not
redistributed. `foundation_arm/weights_manifest.json` records the exact commits used.

## Environment

Python 3.11, PyTorch 2.5.1 (CUDA 12.1), timm 1.0.26, scikit-learn 1.8.0, pandas 3.0.2 and NumPy 2.4.3;
`requirements.txt` lists what the CPU steps need and `requirements-full.txt` is the complete environment of the
GPU runs.

## Citation

If you use this code or the result files, please cite the paper (full reference to be added on publication):

    S. K. Ramagiri, P. M. Kamble. Budget-indexed readout shrinkage: harm-constrained local adaptation of radiograph
    classifiers built on ImageNet and foundation encoders.

## Licence

The code is released under the MIT licence (`LICENSE`). The result files, tables and figures are released under
CC BY 4.0 (`LICENSE-DATA.md`). The datasets and pretrained weights are not part of this repository and
keep their own licences.
