# Verification status

Release preparation dated 2026-09-20.

## Checks executed during preparation

- Offline archive checks: Python AST syntax, all 16 included data-file SHA-256 hashes and byte counts, cell-set inclusion and final label counts.
- All seven Figure 2E module effects/means compared with the matching P1-versus-non-P1 168-module table. M80 d=0.995787084361. The original mixed-contrast assertion was corrected without changing a data file.
- Frozen usage argmax versus final labels: 602 mismatches, including 256 P1 membership changes. This remains an unresolved scientific provenance issue.
- Frozen patient-phase data: 13 eligible units from nine patients; CXCL10 mean rho=-0.5878007454, P=0.0346287959; detection rho=-0.5989438875, P=0.0305391222; CXCL9 mean rho=-0.2033623432, P=0.5051743132. Patient LOO negative in 9/9 CXCL10 and 8/9 CXCL9 analyses.
- Original exact-gene mixed-model function rerun using supplied frozen P1 covariates: CXCL9 beta=-0.073585, FDR=0.007461; CXCL10 beta=-0.055769, FDR=0.017553; both converged. Covariates and expression were not rebuilt from raw counts.
- All ten embedded manuscript/supplement images visually inspected. Current S5 numbers and S5E have been updated; S2C still displays 25 modules despite a 168-module legend.

The numerical audit used Python 3.13 with NumPy 2.5.1, pandas 2.3.3, SciPy 1.18.0 and statsmodels 0.14.6. DOCX conversion used local MarkItDown 0.1.6 in a separate Python 3.12 runtime, with plugins disabled. The release's original `requirements.txt` declares a different historical environment and is retained for provenance, not presented as freshly validated.

## Historical results supplied with the source

The original release recorded checks dated 2026-09-15: TCGA rho=-0.3278413; GSE159067 dual-gene/CXCL10 rho=-0.337646; 57-model DepMap weighted-P1/GCLC rho=0.299002, P=0.023861, BH-FDR=0.238609; GSH-without-GCLC/GCLC rho=0.135922. These historical records were not independently rerun from the large matrices during this preparation.

No full cNMF refit, raw scFEA inference, external-cohort rerun, wet-laboratory statistical rerun, or reconstruction of every final figure passed in this preparation. A successful archive check is not a claim of complete manuscript reproducibility.
