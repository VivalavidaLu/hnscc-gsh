"""Project the six discovery programs using feature-subset normalization."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.optimize import nnls


def main():
    parser = argparse.ArgumentParser(
        description="Project the six discovery programs using feature-subset normalization."
    )
    parser.add_argument(
        "--counts",
        required=True,
        type=Path,
        help="Raw-count AnnData, malignant cells by genes; cell IDs and gene symbols as axes",
    )
    parser.add_argument(
        "--spectra",
        required=True,
        type=Path,
        help="cNMF gene score spectra, tab-delimited; six program rows and gene columns",
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Ordered cNMF features, one symbol per line",
    )
    parser.add_argument(
        "--consensus",
        required=True,
        type=Path,
        help="Training-cell consensus usages, tab-delimited; cell IDs by program",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV of normalized usages with cell IDs",
    )
    args = parser.parse_args()
    adata = ad.read_h5ad(args.counts, backed="r")
    spec = pd.read_csv(args.spectra, sep="\t", index_col=0)
    spec.index = spec.index.astype(int)
    spec = spec.loc[list(range(1, 7))]
    features = args.features.read_text().split()
    features = [g for g in features if g in spec.columns]
    counts = adata[:, features].X
    counts = (
        counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    ).astype(np.float32)
    total = counts.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Every cell needs a positive feature-library size.")
    expr = np.log1p(counts / total[:, None] * np.float32(10000))
    basis = spec[features].to_numpy(float).T
    cons = pd.read_csv(args.consensus, sep="\t", index_col=0)
    cons.columns = cons.columns.astype(int)
    if not cons.index.is_unique or not cons.index.isin(adata.obs_names).all():
        raise ValueError("Invalid training cell identifiers.")
    cons = cons.loc[:, spec.index]
    weights = np.array(
        [
            cons.loc[c].to_numpy() if c in cons.index else nnls(basis, x)[0]
            for c, x in zip(adata.obs_names, expr)
        ]
    )
    if np.any(weights.sum(1) <= 0):
        raise ValueError("Unassigned cell")
    usage = pd.DataFrame(
        weights / weights.sum(1)[:, None], index=adata.obs_names, columns=spec.index
    )
    usage.to_csv(args.out, index_label="cell_id")


if __name__ == "__main__":
    main()
