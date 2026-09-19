# HNSCC glutathione and CXCL10 analyses

Computational source accompanying **Glutathione metabolism associates with reduced CXCL10 and immune activity after neoadjuvant immunochemotherapy in HNSCC**.

**Release candidate: partial computational reproducibility.** This archive contains analysis code and frozen derived tables. It does not regenerate every manuscript panel. A discrepancy between continuous cNMF usage and final program labels remains unresolved. Read [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before interpreting or rerunning analyses.

## Start here

```bash
python run_reproduction.py
python run_reproduction.py --strict
```

These offline checks use only the Python standard library: syntax, checksums, cell manifests, label consistency and matched scFEA contrasts. The default reports known issues; `--strict` exits nonzero while manuscript consistency issues remain. Neither command refits cNMF or reconstructs all figures.

For the numerical checks on frozen patient-phase tables:

```bash
python -m pip install -r requirements-audit.txt
python scripts/check_frozen_associations.py
```

For the original analysis scripts, use a separate environment with the supplied `requirements.txt`. Its historical environment declaration is retained; that entire dependency set was not reinstalled during release preparation. The original manuscript environment and release environment differ. See [REPRODUCTION_CHECK.md](REPRODUCTION_CHECK.md).

## Analyses

| Directory | Scope |
|---|---|
| `01_single_cell_selection_and_cnmf/` | Classification thresholds, frozen cell manifests, cNMF fitting/projection and sensitivity |
| `02_redox_branch_and_tf_activity/` | Redox branch scores, custom ULM/MLM TF activity and frozen scFEA validation |
| `03_immune_association/` | Patient aggregation, mixed models, chemokine correlations and leave-one-patient-out checks |
| `04_external_public_validation/` | TCGA-HNSC and GSE159067 associations |
| `05_depmap/` | Pinned DepMap Public 24Q2 expression/dependency associations |
| `data/` | Small derived tables, manifests and frozen network prior |

[MANUSCRIPT_CODE_MAP.md](MANUSCRIPT_CODE_MAP.md) maps each figure panel and table to its available implementation and missing components. [WORKFLOW.md](WORKFLOW.md) gives execution dependencies. [data/README.md](data/README.md) and [DATA_SOURCES.md](DATA_SOURCES.md) describe required external inputs.

Large expression matrices and clinical/experimental measurements are not included. The original data-processing steps that create some working matrices are also absent; downloading a public dataset alone does not fill these gaps.

Scripts normally use `data/` and `results/`. Most full analyses accept `HNSCC_DATA_DIR` and `HNSCC_RESULTS_DIR`; offline checks intentionally inspect the included repository tables. Explicit provenance fields use logical paths. Third-party errors and tracebacks can still contain runtime paths; generated results and logs are excluded from the source archive.

## Citation and license

See [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), and [DATA_LICENSE.md](DATA_LICENSE.md). Existing author-supplied license declarations are retained; source datasets and the network prior remain subject to their original terms. This candidate has no assigned DOI and no confirmed public repository URL.
