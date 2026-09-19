# Execution dependencies

Run commands from the repository root. Review KNOWN_ISSUES.md first. Full analysis commands below are staged branches, not a validated end-to-end pipeline.

| Stage | Inputs needed | Command |
|---|---|---|
| Offline archive checks | Included files only | `python run_reproduction.py` |
| Frozen patient-phase checks | Included files; audit dependencies | `python scripts/check_frozen_associations.py` |
| Frozen cell manifest check | Included files; pandas | `python 01_single_cell_selection_and_cnmf/reconstruct_final_cell_universe.py` |
| cNMF fitting | Raw count h5ad, final IDs and feature manifest | `python 01_single_cell_selection_and_cnmf/cnmf_model_selection_and_programs.py` |
| K6 projection | Raw counts plus preceding cNMF consensus output | `python 01_single_cell_selection_and_cnmf/project_k6_to_all_cells.py` |
| NMF sensitivity | Raw/annotated/malignant h5ad, frozen usage, fitted cNMF files | `python 01_single_cell_selection_and_cnmf/cnmf_sensitivity_analysis.py --modes actual_input,projected` |
| Raw redox | Raw counts, malignant h5ad and frozen usage | `python 02_redox_branch_and_tf_activity/compute_redox_scores.py` |
| Scaled redox | Scaled malignant h5ad and frozen usage | `python 02_redox_branch_and_tf_activity/compute_scaled_branch_sensitivity.py` |
| TF activity | Malignant h5ad and included CollecTRI | `python 02_redox_branch_and_tf_activity/compute_tf_activity_ulm.py` |
| Frozen scFEA | Included module tables | `python 02_redox_branch_and_tf_activity/validate_scfea_results.py` |
| Post-treatment immune scores | TME h5ad and included sample score table | `python 03_immune_association/patient_level_p1_and_immune_scores.py` |
| Mixed models | Raw counts and frozen malignant score table | `python 03_immune_association/mixed_effects_chemokine_model.py` |
| Exact chemokine robustness | Previous mixed-model cell table plus raw counts | `python 03_immune_association/leave_one_patient_out_and_fdr.py` |
| Additional branch specificity | Raw counts and included P1 cell/patient-phase tables | `python 03_immune_association/pseudobulk_chemokine_correlation.py` |
| TCGA | Staged 520-primary-tumor matrix | `python 04_external_public_validation/tcga_score_and_correlation.py` |
| GSE159067 | Official expression file; optional clinical metadata | `python 04_external_public_validation/gse159067_score_and_correlation.py` |
| DepMap | Network access or cached pinned 24Q2 files; included spectra | `python 05_depmap/depmap_expression_and_dependency.py` |

Projection writes new usage under `results/cnmf/`; downstream scripts still read the included frozen `data/derived/` usage. No automatic promotion is performed because final label provenance is unresolved. Archive checks inspect bundled data irrespective of external data-root environment variables.

## Methods that need explicit reporting

- Patient-level immune code uses 20,000 permutations, 2,000 bootstrap replicates and a four-test BH family (two exposures by two outcomes).
- Exact CXCL9/CXCL10 mixed-model BH is a two-test family; the four module outcomes are corrected separately.
- DepMap standardizes genes over the full expression model set, then restricts to 57 HNSCC models. P1 top50 and weighted top200 exclude MT-/RPL/RPS genes; weighted scores normalize nonnegative spectral loadings. BH includes 20 tests.
- Original scFEA inference, final filtering and several figure-specific analyses must be supplied before claiming full reproducibility.
