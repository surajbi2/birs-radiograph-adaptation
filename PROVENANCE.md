# Provenance

## Published scripts and the versions that ran

For publication the scripts were renamed by what they do, their docstrings rewritten, path defaults that pointed at the authors' machines replaced by placeholders such as `/path/to/foundation_arm_workdir`, and messages that cited section numbers of an earlier draft reworded. Nothing else in the code changed: for every pipeline script, the same renames, placeholders and messages applied to the version that ran give a syntax tree, docstrings removed, identical to the published one (`AST equal`). The result files record the SHA-256 of the script that wrote them, and some record its name; the table gives both.

| Published file | Name in result headers | As-run SHA-256 | Published SHA-256 | AST equal |
|---|---|---|---|---|
| `code/audit_bit_depth.py` | - | `e1033d9aab2dc841` | `0f1a3103d4e930e2` | yes |
| `code/bootstrap_intervals.py` | - | `8fabc7b3736a2ccc` | `83f5a20a2c591d20` | yes |
| `code/check_patient_overlap.py` | - | `1b8b77c49678a734` | `84e0ce73c9be0264` | yes |
| `code/cohort_counts.py` | `cohort_counts.py` | `0b219b530665008f` | `974bc8116d53e1eb` | yes |
| `code/dedupe_oracle_results.py` | - | `f3d05b6c2d16e37f` | `a147f0c17925f794` | yes |
| `code/layerwise_block_maps.py` | `layerwise_freeze.py` | `ae88e7886388fb93` | `191d58f9a3e22d61` | yes |
| `code/layerwise_unfreezing.py` | `layerwise_run.py` | `ae9cafaacd9bfb47` | `b4c6d3385aa1b0c7` | yes |
| `code/make_paper_outputs.py` | - | `8230aba39f56e984` | `9d574744419bc48c` | yes |
| `code/graz_label_crosstab.py` | `make_graz_label_crosstab.py` | `b3dc8ce86be2dfa1` | `220bdbf90c98a081` | yes |
| `code/measure_appearance.py` | `measure_appearance.py` | `acff28b87aad6400` | `455fa199d666f0c4` | yes |
| `code/merge_wrist_probe.py` | - | `5fbb883dd4781ae9` | `b6fd10a01987d69f` | yes |
| `code/mskdata.py` | `mskdata.py` | `2dc8663e416b8eaf` | `c25bf41e1605db30` | yes |
| `code/mskmodel.py` | `mskmodel.py` | `4c3c5c1614d4b51a` | `38382e3a3f1e8aa9` | yes |
| `code/external_harm_analyses.py` | - | `9f10296bfe85f7d3` | `787c0fbf149603a4` | yes |
| `code/loo_anatomy_policies.py` | - | `9ab4f8e5040cad4b` | `9048be78e2f03ca8` | yes |
| `code/shortcut_reliance.py` | `shortcut_reliance.py` | `4bd3d75e01e8e083` | `a07038aa3a1d2fb5` | yes |
| `code/source_model_tables.py` | `stage_a_analyze.py` | `2dd40ea8e42dbf03` | `a576eb4d198f5995` | yes |
| `code/train_source_models.py` | `stage_a_run.py` | `5a17e811eb9a36d4` | `44829009e0971d30` | yes |
| `code/readout_penalty_sweep.py` | `stage_b11_creg_sweep.py` | `a742ac68ddbe8cc7` | `4883baaedd763189` | yes |
| `code/oracle_readout_cv.py` | `stage_b12_oracle_creg.py` | `92c312d8a1c2c9c5` | `961c28b2e4cded11` | yes |
| `code/readout_shrinkage.py` | `stage_b13_shrinkage.py` | `a3d78622e726bd19` | `334e1589b3afdaa5` | yes |
| `code/fracatlas_budget_cells.py` | `stage_b14_frac_budget.py` | `4f4d2ca2d21d3fbf` | `41bd53ac5f771fa7` | yes |
| `code/mfxr_external.py` | `stage_b15_mfxr.py` | `bf1736a1b28db2b9` | `32b03ffe4bfb3e30` | yes |
| `code/birs_schedule.py` | `stage_b16_lambda_schedule.py` | `5c58a0b4ce14c7a0` | `903b274c3bbac08c` | yes |
| `code/readout_angle.py` | `stage_b_angle.py` | `1490bff9933e36e3` | `b42070ee2869cd38` | yes |
| `code/graz_confound_audit.py` | - | `dd627d5e9c9c2312` | `fa40ffa29787a245` | yes |
| `code/graz_indomain_audit.py` | - | `9945ab696f087c89` | `5c5c7f13678b71aa` | yes |
| `code/fracatlas_match_stats.py` | - | `7198a5140ff1c859` | `a97036dbad0323fd` | yes |
| `code/fracatlas_transfer.py` | `stage_b_fracatlas.py` | `f8e48824fd7d7954` | `8616ac81fa747d5f` | yes |
| `code/graz_label_audit.py` | - | `1a2649d9ffc920c3` | `4c7f9d6a26092083` | yes |
| `code/preprocessing_variants.py` | `stage_b_prep.py` | `014363ea3abbe851` | `fb967398a5eb4a82` | yes |
| `code/readout_probe.py` | `stage_b_probe.py` | `6ccbbfc5abd77b59` | `cd1256fb5a29dd69` | yes |
| `code/dataset_shift.py` | `stage_b_run.py` | `7ee1bc9766fdc603` | `ebdf1bc2831899d7` | yes |
| `code/stratify_scores.py` | `stage_b_stratify.py` | `533144c2650a192b` | `e296b4ef731d8b6e` | yes |
| `code/joint_encoder_vs_ceiling.py` | - | `3f8adada817a36d2` | `b62a6d204dba3651` | yes |
| `code/joint_encoder_report.py` | - | `bf0c87ab9a509962` | `fc82c7216b66d999` | yes |
| `code/l2sp_comparator.py` | `stage_b17_l2sp.py` | `d86c4f24d396e23e` | `eabd9ac4d083f247` | yes |
| `foundation_arm/scripts/feature_reproducibility_gate.py` | - | `2e2e2956401e6a1f` | `db7248c17551b289` | yes |
| `foundation_arm/scripts/arm_a_finetune.py` | `arm_a_finetune.py` | `d2e8ba7abe477726` | `d600ffde032cc195` | yes |
| `foundation_arm/scripts/extract_features.py` | - | `fed9ce7457d3afc3` | `eec1d311d43022bd` | yes |
| `foundation_arm/scripts/arm_b_readout.py` | - | `9c86255f9a2c15da` | `ee383244a6b83224` | yes |
| `foundation_arm/scripts/budget_cells.py` | `fa_cells.py` | `d4ba7cbd7074faa8` | `7a3b530bc89aed54` | yes |
| `foundation_arm/analysis/foundation_summary.py` | - | `baab5945290b0cb1` | `53360d4301b63ac1` | yes |
| `foundation_arm/analysis/foundation_summary_verify.py` | - | `e4893598f7bf4e49` | `ba735f449e4645b1` | yes |
| `manuscript/foundation_tables.py` | - | `a93ca5cfbf6b12fd` | `e2fbb124d818c537` | no (repository paths; outputs compared by verify_repo.py) |
| `manuscript/source_model_aucs.py` | - | `2a9446036eef2ca1` | `ab47fa5f16ac8a33` | no (repository paths; outputs compared by verify_repo.py) |
| `manuscript/make_data_figures.py` | - | `2cf301d305b05942` | `aa2178a237b144c9` | no (repository paths; outputs compared by verify_repo.py) |
| `manuscript/make_overview_figure.py` | - | `4a5ba75770cab627` | `f49ec050665902e9` | no (repository paths; outputs compared by verify_repo.py) |
| `manuscript/make_overview_thumbnails.py` | - | `f023fe7eefea47bd` | `9e99a13ff559fe5e` | no (repository paths; outputs compared by verify_repo.py) |

The foundation-arm scripts are the versions of the DINOv3 run. SKELEX, RAD-DINO and MedSigLIP-448 ran with earlier versions of `arm_a_finetune.py` and `extract_features.py` that lack only the DINOv3 entries and the `config_kw` hook, which has no effect for the other encoders.

## Result files

Every file under `results/` and `foundation_arm/cells/` is byte-identical to the file that was read; their hashes are recorded in `code/paper/paper_numbers.json` and `foundation_arm/analysis/foundation_summary.json`, and `make_paper_outputs.py` checks the foundation-arm cells against them. Their provenance headers therefore still give the script names and folders of the original compute environment. Result files whose earlier name is recorded in another shipped file:

| Published file | Name when written |
|---|---|
| `results/transfer/birs_schedule.csv` | `stage_b/stage_b16_lambda_schedule.csv` |

The folders `results/source_models/` and `results/transfer/` were `stage_a/` and `stage_b/` when written.

## Regenerated outputs

`code/paper/paper_numbers.json` and `foundation_arm/analysis/foundation_summary.json` record the names of their input files, so they were regenerated by the published code from the published inputs. Every value is identical to the values the paper was written from (`paper_numbers.json` with SHA-256 `d316e96dc1b21d08...`); only file names and notes differ.

## Scripts edited after they produced their results

The result files in `results/` are the files the paper was built from. Each one's SHA-256 is
recorded in the `inputs` block of `code/paper/paper_numbers.json`, and every file shipped here
matches its recorded hash. The scripts in `code/` are their current versions. For the scripts
below, the file was saved again **after** it wrote the result the paper uses, and the version that
actually ran was not kept. Each such result file is therefore the output of an earlier version of
the script shipped here. None of these stages has been re-run with its current version, because the
stages need the trained models and the datasets, which are not redistributed.

Times are the file-system times on the compute server (UTC+05:30).

| Script | Result file it wrote | Result written | Script last saved | What is known about the difference |
|---|---|---|---|---|
| `readout_shrinkage.py` | `readout_shrinkage_XR_WRIST.csv` | 2026-09-19, 14:24–15:00 | 2026-09-19 17:21 | The shipped version skips a budget sample that contains only one class at two points, line 489 and inside `raw_space_readout()` at line 180, so such a sample yields no row. The shipped result file lacks exactly such cells: 7 of the 2,700 anatomy cells (6 at 10 studies, 1 at 25) and 1 of the 900 GRAZPEDWRI-DX and MFXR cells (MFXR, 10 studies). Whether the version that ran had both checks is not known. |
| `mfxr_external.py` | `mfxr_external_XR_WRIST.csv` | 2026-09-19, 14:53–15:02 | 2026-09-19 16:06 | Not recorded. |
| `oracle_readout_cv.py` | `oracle_readout_cv_XR_WRIST.csv` | 2026-08-08 (run); 2026-08-12 14:51 (de-duplicated) | 2026-08-12 14:48 | The run's output contained duplicated rows. `dedupe_oracle_results.py` removed them three minutes after the script was last saved, and the de-duplicated file is the one recorded and shipped. |
| `readout_angle.py` | `readout_angle_XR_WRIST.csv` | 2026-08-05, 17:19–18:20 | 2026-08-05 17:52 | Saved while the run was in progress. Python reads a script once at start-up, so the run used the version from before 17:52. |
| `graz_indomain_audit.py` | `graz_indomain_audit.csv` | 2026-08-04 19:28 | 2026-08-04 19:38 | Not recorded. |
| `mskdata.py` (library) | every result file written before 2026-09-19 14:16 | before 2026-09-19 | 2026-09-19 14:16 | The edit added the MFXR dataset to the data index; it was made by a one-off patch script, which is not shipped. The results written earlier used the version before that edit. |

Every other script that wrote a result file used here was last saved before, or within about a
minute of, the start of its run.

`mskmodel.py`, the model and feature-extraction library, has not changed since 2026-07-28. Every
result file was therefore written with the version shipped here, apart from the provenance change
described below.
