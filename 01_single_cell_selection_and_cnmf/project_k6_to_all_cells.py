"""Project the fitted K=6 cNMF solution to all 6,165 frozen malignant cells."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.optimize import nnls


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
RAW_COUNTS = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
FINAL_IDS = DATA_ROOT / "derived" / "malignant_DEFINITIVE_FINAL_cellids.csv"
FEATURE_GENES = DATA_ROOT / "derived" / "cnmf_feature_genes_1508.txt"
RUN = RESULTS_ROOT / "cnmf" / "cnmf_runs" / "malignant_final_v2"
SPECTRA = RUN / "malignant_final_v2.gene_spectra_score.k_6.dt_2_0.txt"
CONSENSUS_USAGE = RUN / "malignant_final_v2.usages.k_6.dt_2_0.consensus.txt"
OUTPUT = RESULTS_ROOT / "cnmf"
EXPECTED_COUNTS = {"1": 1857, "2": 830, "3": 904, "4": 850, "5": 1081, "6": 643}


def main() -> None:
    final_ids = pd.read_csv(FINAL_IDS)["cell_id"].astype(str)
    adata = sc.read_h5ad(RAW_COUNTS, backed="r")
    missing = final_ids[~final_ids.isin(adata.obs_names)]
    if len(missing):
        raise RuntimeError(f"Raw count object is missing {len(missing)} frozen cells")
    cells = final_ids.to_numpy()
    selected = adata[cells].to_memory()
    adata.file.close()

    features = [line.strip() for line in FEATURE_GENES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(features) != 1508 or len(set(features)) != 1508:
        raise RuntimeError("Expected 1,508 unique frozen cNMF features")
    missing_genes = [gene for gene in features if gene not in selected.var_names]
    if missing_genes:
        raise RuntimeError(f"Raw count object is missing {len(missing_genes)} cNMF features")

    spectra = pd.read_csv(SPECTRA, sep="\t", index_col=0)
    consensus = pd.read_csv(CONSENSUS_USAGE, sep="\t", index_col=0)
    programs = spectra.index.astype(str)
    if programs.tolist() != ["1", "2", "3", "4", "5", "6"]:
        raise RuntimeError(f"Unexpected K=6 program labels: {programs.tolist()}")

    sc.pp.normalize_total(selected, target_sum=1e4)
    sc.pp.log1p(selected)
    matrix = selected[:, features].X
    matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    basis = spectra.loc[:, features].to_numpy(dtype=float).T

    projected = np.zeros((len(cells), 6), dtype=float)
    for index, row in enumerate(matrix):
        projected[index], _ = nnls(basis, row)
    usage = pd.DataFrame(projected, index=cells, columns=programs)

    shared = consensus.index.intersection(usage.index)
    consensus_values = consensus.loc[shared, programs].astype(float)
    consensus_values = consensus_values.div(consensus_values.sum(axis=1), axis=0)
    usage.loc[shared, programs] = consensus_values
    usage = usage.div(usage.sum(axis=1), axis=0)
    if usage.isna().any().any():
        raise RuntimeError("K=6 projection produced missing usage values")

    assignment = usage.idxmax(axis=1).rename("cNMF_program")
    counts = assignment.value_counts().sort_index().astype(int).to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Program counts differ from the submitted analysis: {counts}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage.rename_axis("cell_id").to_csv(OUTPUT / "cNMF_K6_usage_projected_6165.csv")
    assignment.rename_axis("cell_id").to_csv(OUTPUT / "cNMF_K6_program_assignments_6165.csv")
    print(f"Projected K=6 usage for {len(usage):,} cells; program counts: {counts}")


if __name__ == "__main__":
    main()
