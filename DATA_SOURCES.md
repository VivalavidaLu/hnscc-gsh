# Sources and analytic inputs

| Resource | Input used |
| --- | --- |
| [GSE301741](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE301741) | Discovery cohort. Use consistent cell IDs and gene symbols across the malignant-cell count matrix, cell annotations and cNMF consensus. The malignant cohort has 6,165 cells from 13 patients. |
| [GSE195832](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195832) | GSM-prefixed 10x matrices from GEO; epithelial and reference-cell annotations with barcode, donor, treatment phase, CNV status, library size and mitochondrial fraction. |
| [TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC) | Primary tumors, one sample per patient. Analysis-scale expression, HPV status, ABSOLUTE purity and methylation-derived leukocyte fraction. Match DNA data by the first 15 barcode characters and take median values across technical aliquots. |
| [GSE159067](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159067) | Targeted expression panel and GEO clinical annotations. Align samples, remove control probes and keep HPV as a categorical field. RECIST response compares CR/PR with SD/PD. PFS and OS use months and binary event indicators. |
| [DepMap Public 24Q2](https://doi.org/10.25452/figshare.plus.25880521) | Protein-coding expression (log TPM), model metadata and Chronos CRISPR GeneEffect. Dependency strength is negative GeneEffect. |
| CollecTRI | Human signed regulatory network with source, target and weight. The seven GSH genes are removed from the target set before ULM. |

### Staging inputs

The code starts from the analysis stage named in each script. Discovery cNMF uses a raw-count malignant AnnData, ordered training cell IDs, a discovery feature list and a cNMF replicate parameter table. Discovery projection additionally uses training-cell consensus usages. The external single-cell model uses the fitted discovery TPM spectra and their scale table. GEO supplies expression matrices. Request study-derived inputs, including training membership, fitted spectra, malignant-cell calls and CNV annotations, from the corresponding authors.

Provide verified response labels separately as a two-column Patient,Response file. Use R and NR for verified labels and omit unresolved patients. For Figure 3, Patient, TreatmentPhase and SampleID identify sampling units. The response definition in GSE301741 is at least 10% tumor regression by area. Eight of the nine paired patients have verified labels. Immune aggregation requires full-library counts, subtype labels and sequencing/sorting metadata. GSH score standardization for pathological response uses all malignant cells; the score-removal analysis in S4G standardizes genes within P1.

For the external bulk cohorts, use sample-by-gene CSVs with gene symbols as columns. Score each measured gene within its cohort. GSE159067 clinical fields are Response, Age, HPV, immunotherapy line, OS_time, OS_event, PFS_time and PFS_event. Preserve all HPV categories. TCGA-HNSC composition-adjusted models use ABSOLUTE_purity and leukocyte_fraction. The external P1 top-50 score includes available genes in the discovery ranking; the DepMap P1 signature separately excludes mitochondrial and ribosomal genes.

For Figure 5A/B, input expression is log1p(raw gene count / full-library total × 10,000). Each gene contributes equal positive weight. The detection fractions returned by the script are on a 0–1 scale.

Tissue correlations use one complete set of staining scores per specimen. Paired assay tests use independent experiments as the unit. The CCK-8 entry point averages six wells per condition, subtracts the matched blank and normalizes to the control. The CXCL10 option expects dilution-corrected concentrations for log10 analysis. Leave values that could not be quantified blank. Plotting substitutions are excluded from inference. Complete pairs are selected separately for each contrast.

Clinical and experimental source records are available for consideration upon reasonable request to luhz3@mail.sysu.edu.cn or liangyj35@mail.sysu.edu.cn.
