# Manuscript figure and code map

| Location | Folder and entry point | Analysis and scope |
| --- | --- | --- |
| Figure 1 and S1, discovery model components | 01 · classify_malignant_cells.py, fit_programs.py, project_programs.py | The check_cell_universe.py entry point verifies final membership and six program counts. CNV/entropy classification component, balanced cNMF fitting (3,564 cells), six-program discovery projection. Normalize over projection features, use consensus weights for fitted cells and normalize weights within cells. |
| S1E–F and donor-exclusion results | 01 · fit_programs.py, match_spectra.py, refit_associations.py | Exclude LSCC21. Fit fixed and reselected features with the same 200 K6 seeds. Perform one-to-one spectrum matching and evaluate native variance-scaled TPM projection for the patient-level sensitivity analyses. |
| Figure 2B and S2A | 02 · branch_scores.py | Branch expression and scaled-expression contrasts. |
| Figure 2E and S2C | Separate scFEA workflow | Flux inference is performed upstream. S2C displays the 15 highest and 10 lowest module effect sizes from the P1 versus non-P1 contrast. |
| Figure 2F | 02 · regulon_activity.py | Direct decoupler ULM, GSH genes removed from CollecTRI targets; P1 high/low pseudobulks by patient and phase. |
| Figure 3A, S3 and Supplementary Data | 03 · aggregate_response_scores.py, pathological_response.py | Patient-phase percentages, mean usages and seven-gene GSH scores. The --paired-summary option supplies the Figure 3A signed-rank test across all nine pairs. Baseline and post-minus-pre comparisons by pathological response. Exact rank tests, Hodges–Lehmann differences, pointwise confidence intervals and joint BH across 26 tests. |
| Figure 3B, S4E | 03 · pooled_activation.py | Post-treatment pooled T/NK activation and myeloid inflammation, with patient omission. |
| Figure 3C, S4A–D | 03 · aggregate_immune_scores.py, lineage_associations.py | CD4, CD8 and NK activation, CD8 states and CXCR3 detection. Matched treatment-phase comparisons and stratification by response. |
| Figure 3D–E, S4F–G | 03 · chemokine_associations.py, chemokine_score_sensitivity.py | CXCL9 and CXCL10 mixed models (the two chemokine terms in Figure 3D), patient-phase correlations, clustered bootstrap and standardized SLC3A2 sensitivity. |
| S1G and external projection | 04 · external_single_cell.py | Fixed six-spectrum projection. Report donor/phase eligibility and signature-half coherence adjusted for depth and mitochondrial fraction. |
| Figure 4D | 04 · tissue_correlations.py | Statistical code for the two exploratory tissue correlations and their joint BH correction. |
| Figures 4E–F, S5B and bulk clinical analyses | 04 · score_bulk_signatures.py, primary_bulk_associations.py, bulk_correlations.py, clinical_endpoints.py | Primary bulk correlations, signature coverage, composition adjustment and clinical models. |
| Figures 5A–B | 05 · expression_proxies.py | Nine expression proxies, three GSH pathway genes and the P1 usage tertiles. |
| Figures 5C–D, S6A–D | 05 · depmap_dependency.py | Primary and target-excluded correlations with Chronos dependency, two GCLC upper-versus-lower quartile contrasts, the nine bootstrap intervals for 5D and S6B, and leave-one-model-out sensitivity. |

Folder 05 also includes paired_assays.py for the paired independent-experiment comparisons in the cell-line panels of Figure 5 and Supplementary Figure 6, including the CCK-8 analysis in S6H. Experimental measurements are available by request to the corresponding authors.
