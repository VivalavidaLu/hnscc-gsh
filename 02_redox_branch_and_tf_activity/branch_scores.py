"""Estimate the six metabolic branch contrasts from counts or a pre-scaled matrix."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

GENE_SETS = {
    "gsh_cysteine": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "phase2_detox": [
        "NQO1",
        "AKR1C1",
        "AKR1C2",
        "AKR1B10",
        "ALDH3A1",
        "GSTA4",
        "GSTM1",
        "GSTP1",
    ],
    "thioredoxin_prdx": ["TXN", "TXNRD1", "PRDX1", "SRXN1"],
    "heme_iron": ["HMOX1", "FTH1", "FTL", "SLC40A1", "TFRC"],
    "nrf2_core": ["NFE2L2", "KEAP1", "MAFK", "MAFG", "BACH1"],
    "aa_transport": ["SLC1A5", "SLC7A5", "SLC38A1", "SLC38A2", "SLC1A4", "SLC7A8"],
}


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the six metabolic branch contrasts from counts or a pre-scaled matrix."
    )
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Malignant cell AnnData with aligned discovery labels in obs.cNMF_program",
    )
    parser.add_argument("--scale", required=True, type=Path, help="counts or scaled")
    parser.add_argument("--out", required=True, type=Path, help="Summary CSV")
    args = parser.parse_args()
    a = ad.read_h5ad(args.matrix)
    labels = a.obs.cNMF_program.astype(str).str.extract(r"(\d+)")[0].astype(int)
    total = np.asarray(a.X.sum(axis=1)).ravel() if str(args.scale) == "counts" else None
    if str(args.scale) not in {"counts", "scaled"}:
        raise ValueError("Choose counts or scaled.")
    rows = []
    for branch, genes in GENE_SETS.items():
        present = [g for g in genes if g in a.var_names]
        x = dense_matrix(a[:, present].X).astype(float)
        if total is not None:
            x = np.log1p(x / total[:, None] * 10000)
        scores = x.mean(axis=1)
        rows.append(
            dict(
                branch=branch,
                genes=";".join(present),
                cohens_d=cohen_d(scores[labels == 1], scores[labels != 1]),
            )
        )
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
