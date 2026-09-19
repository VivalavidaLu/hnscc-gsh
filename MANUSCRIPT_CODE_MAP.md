# Manuscript panel and table map

Reviewed against the current main manuscript and supplementary figures on 2026-09-20. Paths are repository-relative. “Available” means code exists, not that a complete raw-data rerun passed. “Frozen” means only derived numerical evidence is supplied. Final publication layouts are generally not regenerated.

| Panel | Script / input | Supported output and remaining gap |
|---|---|---|
| 1A | `01_single_cell_selection_and_cnmf/malignant_cell_classification.py`; `reconstruct_final_cell_universe.py` | Threshold flags and manifest containment; full selection pipeline absent |
| 1B | No corresponding implementation | Whole-cohort QC/Harmony/Leiden/UMAP missing |
| 1C | No corresponding implementation | Annotation and marker dotplot missing |
| 1D | `01_single_cell_selection_and_cnmf/project_k6_to_all_cells.py`; frozen final labels | Counts match manuscript; usage/label discrepancy unresolved; final UMAP absent |
| 1E | No corresponding implementation | Patient-by-program pseudobulk top100 overlap missing |
| 1F | `data/derived/cNMF_K6_gene_spectra_score.txt` | Frozen spectra available; representative expression/detection dotplot missing |
| 2A | `02_redox_branch_and_tf_activity/compute_redox_scores.py` | `results/redox/raw_count/gene_mean_by_program_from_raw_lognorm__*.csv`; final heatmap absent |
| 2B | Same script | `branch_score_summary_from_raw_lognorm__*.csv`; must identify final label source |
| 2C | Same script | Program expression means and raw-count detection; final dotplot absent |
| 2D | Same script | `per_patient_P1_vs_other_from_raw_lognorm__*.csv`; >=10 cells per group per patient |
| 2E | `02_redox_branch_and_tf_activity/validate_scfea_results.py`; seven-module frozen table | Frozen contrast consistency only; original scFEA inference absent |
| 2F | `02_redox_branch_and_tf_activity/compute_tf_activity_ulm.py` | `results/redox/tf_activity/tables/pseudobulk_group_contrasts_tf_activity.csv`; custom implementation differs from Methods; no final plot |
| 3A | No corresponding implementation | Nine-patient paired fraction/Wilcoxon source missing |
| 3B | `03_immune_association/patient_level_p1_and_immune_scores.py` | `post_patient_P1_abundance_TME_correlations.csv`; requires TME h5ad; historical panel filenames differ |
| 3C | Same script | Patient score table has four variables; historical heatmap plots only three; final layout missing |
| 3D | `03_immune_association/mixed_effects_chemokine_model.py`; `leave_one_patient_out_and_fdr.py` | Module models and exact CXCL9/10 models available; exact gene models reproduced from frozen covariates; final combined panel missing |
| 3E | `03_immune_association/leave_one_patient_out_and_fdr.py`; frozen patient-phase table | CXCL10 detection correlation reproduced by `scripts/check_frozen_associations.py`; final scatter absent |
| 4A | Laboratory source outside archive | Representative mIHC image; original image/processing unavailable |
| 4B | Laboratory source outside archive | Same boundary |
| 4C | Clinical source outside archive | 25-patient measurements, exact tests and BH source absent |
| 4D | Clinical source outside archive | Composite-score source and correlations absent |
| 4E | `04_external_public_validation/tcga_score_and_correlation.py` | `results/external_validation/tcga_hnsc/tcga_hnsc_statistics.json`; score/rho/CI available, binned trend CI and staging implementation absent |
| 4F | `04_external_public_validation/gse159067_score_and_correlation.py` | `GSE159067_direct_correlations.csv`; statistics available, final scatter absent |
| 5A | No corresponding implementation | Exact proxy gene sets, usage-tertile selection and Cohen d missing |
| 5B | No corresponding implementation | Tertile expression and GCLC/GCLM/GSS detection missing |
| 5C | `05_depmap/depmap_expression_and_dependency.py` | `results/depmap/signature_dependency_correlations.csv`; weighted P1/GCLC statistics available; plot absent |
| 5D | Same script | Target-excluded rho/P/BH available; bootstrap CI and final forest plot absent |
| 5E | Laboratory source outside archive | GSH source measurements/statistics absent |
| 5F | Laboratory source outside archive | Original blots absent |
| 5G | Laboratory source outside archive | Densitometry and paired statistical source absent |
| 5H | Laboratory source outside archive | ELISA source and log10-scale paired statistics absent |
| S1A | `01_single_cell_selection_and_cnmf/cnmf_model_selection_and_programs.py` | K4-8 fitting/consensus available; final comparison summary and plot absent |
| S1B | `01_single_cell_selection_and_cnmf/cnmf_sensitivity_analysis.py` | Lightweight NMF reference-agreement analysis; run both actual_input/projected modes; not direct input-vs-projection agreement |
| S1C | Frozen K6 usage | 600-cell balanced sampling and final heatmap implementation absent |
| S1D | cNMF sensitivity script / frozen labels | Patient-by-program counts available; final percentage plot absent |
| S2A | Redox raw-count and `compute_scaled_branch_sensitivity.py` | Both scales supported but label sources must be aligned |
| S2B | Same scripts | Program branch means available; final standardized heatmap absent |
| S2C | `data/processed/suppFigS2C_scFEA_full_ranked_modules_P1_vs_nonP1_all_modules.csv` | Correct 168-module contrast available; image shows 25, legend says 168; historical 25-row table is a different contrast |
| S3A | Patient-level immune script | Patient LOO correlations available; historical panel filename differs |
| S3B | `03_immune_association/leave_one_patient_out_and_fdr.py` | Exact-gene LOO; frozen recomputation gives CXCL9 8/9 and CXCL10 9/9 negative |
| S3C | No matching implementation | Three-score exact-CXCL10 analysis missing; SLC3A2 residual/standardized ambiguity |
| S4A | Laboratory source outside archive | Representative IHC source absent |
| S4B | GSE159067 script | Four exposures x two outcomes, rho/CI/BH available; final plot absent |
| S5A | DepMap continuous score output | High/low cutoffs, group comparison and final plot missing |
| S5B | DepMap correlation output | Alternative-score rho/P available; bootstrap CI absent |
| S5C | DepMap correlation output | 20-test matrix can be assembled; current image visually agrees with current manuscript; no raw-data rerun here |
| S5D | DepMap script | `leave_one_model_out_correlations.csv` available; final plot absent |
| S5E | Laboratory source outside archive | Present in current supplementary document; four-cell-line blot source absent |
| Table 1 | Main manuscript | Categorical counts sum to 25; no clinical source table/statistical code |
| Supplementary reagent table | Referenced in Methods | Not supplied in the two documents or archive |

The broad `pseudobulk_chemokine_correlation.py` analysis targets a CXCL9/10/11 module and is not a substitute for the S3C exact-CXCL10 analysis. Do not interpret historical figure filenames as the current manuscript panel map.
