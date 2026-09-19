# Public data sources

| Dataset | Public source | Release file or record | Repository use |
|---|---|---|---|
| GSE301741 | NCBI GEO, accession `GSE301741` | Raw sample matrices and `GSE301741_Seurat_Object_QCpass_137020cells_withMetaData.rds` | Single-cell reconstruction and analysis. Large locally staged `.h5ad` objects are not committed. |
| TCGA-HNSC | TCGA data distributed through the UCSC Xena TCGA hub | Dataset `TCGA.HNSC.sampleMap/HiSeqV2`; retain sample-type code `01` and one aliquot per patient. The staged 520 × 20,531 sample-by-gene CSV used for the reported result is 60,632,651 bytes with SHA-256 `fad3b56e2578a9c8bd9ae6a221ea9124d087f86c037dafa7fe4e864432b37828`. | Bulk GSH/cystine–CXCL10 validation. |
| GSE159067 | NCBI GEO, accession `GSE159067` | `GSE159067_IHN_log2cpm_data.txt.gz` (2,087,402 bytes; SHA-256 `efc32fc5bbd2a13689d6cf5fac7ffae09aae59d1c1dcc82a438c6ca9288c2224`); metadata derived from the public series record | Independent 102-sample validation using the original GEO supplementary filename. |
| DepMap | Broad DepMap Public 24Q2 | DOI `10.25452/figshare.plus.25880521`; `Model.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `CRISPRGeneEffect.csv` | HNSCC expression-score and CRISPR-dependency analyses. |
| CollecTRI | OmniPath/CollecTRI human TF–target prior | Frozen source–target–weight table in `data/reference/collectri_human.csv` | Pseudobulk ULM/MLM TF-activity analysis. |

The repository contains computational code and non-sensitive derived manifests only. Public datasets remain under their source licences and access terms.
