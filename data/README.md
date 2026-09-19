# Input layout

Large source and working expression files are not committed. Place them under the following generic layout, or point `HNSCC_DATA_DIR` to a directory with the same structure.

```text
data/
├── raw/
│   ├── single_cell_counts.h5ad
│   ├── tcga_hnsc_primary_tumor_expression.csv
│   ├── GSE159067_IHN_log2cpm_data.txt.gz
│   └── depmap_24q2/                 # downloaded automatically when absent
├── processed/
│   ├── single_cell_annotated.h5ad
│   ├── malignant_6165.h5ad
│   ├── tme_annotated.h5ad
│   ├── malignant_classification.csv                   # included
│   ├── section3_p1_immune_pressure_sample_merged.csv  # included
│   ├── cell_level_malignant_ifn_apm_scores.csv        # included
│   ├── p1_cell_level_scores_with_key_genes.csv        # included
│   ├── p1_patient_phase_cxcl9_cxcl10.csv              # included
│   └── gse159067_sample_metadata.csv  # optional covariate sensitivity analysis
├── reference/
│   └── collectri_human.csv            # included frozen TF-target prior
└── derived/                         # versioned with this repository
    ├── malignant_PURE_FINAL_cellids.csv
    ├── malignant_DEFINITIVE_FINAL_cellids.csv
    ├── cnmf_feature_genes_1508.txt     # submitted ordered cNMF feature set
    ├── cNMF_K6_gene_spectra_score.txt
    └── cNMF_K6_usage_6165.csv          # continuous usage; final labels are in the included processed table
```

The frozen CollecTRI human TF–target prior is versioned under `reference/` so the TF-activity calculation does not depend on a changing online resource.

## Required conventions

- `single_cell_counts.h5ad`: count matrix derived from GSE301741; cell identifiers must match the included manifests. Required observation fields used across scripts include `Patient`, `Timepoint` and the cell-type/program fields named in the scripts.
- `single_cell_annotated.h5ad`, `malignant_6165.h5ad`, `tme_annotated.h5ad`: analysis-ready objects derived from GSE301741. These are working formats, not additional source datasets.
- The versioned CSV files under `processed/` are computational classification, score and pseudobulk tables derived from GSE301741. They contain no laboratory measurements and allow the key statistical models to be inspected without the large expression objects.
- `cnmf_feature_genes_1508.txt` is the exact ordered 1,508-gene feature manifest used for the submitted cNMF analysis. The cNMF script validates this manifest against the staged count matrix and does not add marker genes at runtime.
- Final P1-P6 assignments are stored in the `program` column of `cell_level_malignant_ifn_apm_scores.csv`; they reproduce the manuscript counts. The continuous K=6 usage table supports usage-based scores and the usage-argmax sensitivity analysis.
- `tcga_hnsc_primary_tumor_expression.csv`: transpose UCSC Xena dataset `TCGA.HNSC.sampleMap/HiSeqV2` to sample-by-gene orientation, retain TCGA sample-type code `01`, and keep one aliquot per patient. The expected table has 520 rows and 20,531 gene columns; its byte count and SHA-256 are recorded in `DATA_SOURCES.md`.
- `GSE159067_IHN_log2cpm_data.txt.gz`: use the GEO supplementary file without renaming. The primary 102-sample correlation runs directly from this file. `gse159067_sample_metadata.csv` is needed only for the optional covariate models.
- DepMap files are pinned to Public 24Q2 and downloaded by `05_depmap/depmap_expression_and_dependency.py`.

Input checks vary by script; some scores use the available genes only. Check gene-coverage outputs and KNOWN_ISSUES.md before interpretation. Working h5ad construction and final label provenance are not fully supplied.

- scFEA frozen module tables supporting Figure 2E and Supplementary Figure 2C are included under `data/processed/`.
