# DepMap release provenance

The manuscript DepMap analysis used **DepMap Public 24Q2**, released by Broad DepMap on 23 May 2024.

- Dataset: Broad DepMap (2024), *DepMap 24Q2 Public*
- DOI: https://doi.org/10.25452/figshare.plus.25880521
- Figshare article ID: `25880521`
- Model metadata: `Model.csv`, Figshare file ID `46489732`
- Expression: `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (unstranded log2(TPM+1)), Figshare file ID `46490878`
- CRISPR effect: `CRISPRGeneEffect.csv`, Figshare file ID `46489063`; the release script streams only the required target-gene columns
- Access date used for the release audit: 14 September 2026

The complete pinned files retain 57 HNSCC models after the model-metadata filter and intersection with expression and CRISPR data. More negative Chronos `GeneEffect` values indicate stronger dependency; the script reports dependency strength as `-GeneEffect`, so larger values indicate stronger dependency.

The script pins the three Figshare file IDs, byte counts and MD5 checksums. It uses a valid cached file without network access and downloads only missing or invalid files. Re-running the analysis with a newer release is a new sensitivity analysis and should not silently replace the 24Q2 result reported in the manuscript.
