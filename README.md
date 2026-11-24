# HNSCC glutathione and CXCL10 analyses

Code for *SLC7A11/GCLC-associated glutathione metabolism in relation to CXCL10 expression and CD8⁺ T-cell infiltration in immunotherapy-treated HNSCC*.

| Folder | Computational analyses |
| --- | --- |
| 01_single_cell_selection_and_cnmf | Malignant-cell selection, cNMF projection, donor exclusion and spectrum recovery |
| 02_redox_branch_and_tf_activity | Metabolic branch contrasts, scFEA module summaries and CollecTRI ULM |
| 03_immune_association | Pathological response, immune lineages, CD8 states and P1 chemokine models |
| 04_external_public_validation | GSE195832 single-cell projection, TCGA-HNSC and GSE159067 validation, clinical endpoints and tissue correlations |
| 05_depmap | Expression proxies, DepMap 24Q2 dependencies and paired assay statistics |

The [panel map](MANUSCRIPT_CODE_MAP.md) maps manuscript figures to their analysis scripts and identifies steps that require additional inputs.

### Running an analysis

Install the analysis dependencies using `python -m pip install -r requirements.txt`. Supply the [analysis-ready inputs](DATA_SOURCES.md) through command-line arguments. Each analysis entry point supports `--help`. For example, after preparing one row per patient and phase with `aggregate_response_scores.py`:

```sh
python 03_immune_association/pathological_response.py --phase patient_phase.csv --out response.csv
```

### Data access

This repository contains analysis source code and the gene lists used for external signature scoring. Analysis-ready datasets, fitted model parameters and result tables are separate inputs or outputs. Clinical source records and experimental measurements are held by the authors. Public cohort accessions and input definitions are listed in [Data sources](DATA_SOURCES.md). Requests for supporting clinical and experimental data are reviewed by the corresponding authors, Huanzi Lu (luhz3@mail.sysu.edu.cn) and Yujie Liang (liangyj35@mail.sysu.edu.cn).
