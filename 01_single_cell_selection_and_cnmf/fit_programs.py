"""Fit discovery cNMF or a paired feature-sensitivity analysis after donor exclusion."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from cnmf import cNMF
from cnmf.cnmf import save_df_to_npz


def select_features(rem):
    genes = rem.var_names.tolist()
    counts = sparse.csr_matrix(rem.X)
    depth = np.asarray(counts.sum(axis=1)).ravel()
    if (depth <= 0).any():
        raise ValueError("Feature selection requires positive library sizes.")
    logX = counts.multiply(1e4 / depth[:, None]).tocsr()
    logX.data = np.log1p(logX.data)
    ranks = np.zeros(rem.n_vars)
    for pt in sorted(rem.obs.Patient.unique()):
        y = logX[rem.obs.Patient.eq(pt).to_numpy()]
        m = np.asarray(y.mean(axis=0)).ravel()
        v = np.asarray(y.power(2).mean(axis=0)).ravel() - m * m
        ok = np.asarray((y > 0).sum(axis=0)).ravel() >= max(3, int(y.shape[0] * 0.05))
        disp = np.where(ok & (m > 1e-8), v / (m + 1e-8), 0)
        order = np.argsort(-disp)
        r = np.empty_like(order)
        r[order] = np.arange(len(order))
        ranks += r
    ranking = pd.Series(ranks / rem.obs.Patient.nunique(), index=genes)
    fresh = ranking.nsmallest(1500).index.tolist()
    canonical = [
        "KRT14",
        "KRT5",
        "LAMA3",
        "COL17A1",
        "MKI67",
        "TOP2A",
        "CENPF",
        "CCNB1",
        "VIM",
        "TGFBI",
        "CAV1",
        "FN1",
        "MX1",
        "IFITM1",
        "ISG15",
        "IFI27",
        "SPRR2D",
        "KLK13",
        "S100A12",
        "SLURP2",
    ]
    fresh += [g for g in canonical if g in genes and g not in fresh]

    return fresh


def main():
    parser = argparse.ArgumentParser(
        description="Fit discovery cNMF or a paired feature-sensitivity analysis after donor exclusion."
    )
    parser.add_argument(
        "--counts",
        required=True,
        type=Path,
        help="Malignant-cell raw-count AnnData with Patient in obs",
    )
    parser.add_argument(
        "--training",
        required=True,
        type=Path,
        help="Training-cell identifiers, one per line, in discovery order",
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Ordered discovery features, one per line",
    )
    parser.add_argument(
        "--parameters",
        required=True,
        type=Path,
        help="cNMF replicate parameter table (TSV from the original cNMF run, including nmf_seed and n_components)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=Path,
        help="discovery, fixed_features or reselected_features",
    )
    parser.add_argument("--out", required=True, type=Path, help="cNMF output directory")
    args = parser.parse_args()
    mode = str(args.mode)
    if mode not in {"discovery", "fixed_features", "reselected_features"}:
        raise ValueError(mode)
    adata = ad.read_h5ad(args.counts)
    training = args.training.read_text().split()
    features = args.features.read_text().split()
    if mode != "discovery":
        adata = adata[adata.obs.Patient.ne("LSCC21")].copy()
        training = [c for c in training if c in adata.obs_names]
        if len(training) != 3064 or adata.n_obs != 4436:
            raise ValueError("Unexpected exclusion cohort.")
    if mode == "reselected_features":
        features = select_features(adata)
    parameters = pd.read_csv(args.parameters, sep="\t", index_col=0)
    ks = [4, 5, 6, 7, 8] if mode == "discovery" else [6]
    parameters = parameters[parameters.n_components.isin(ks)].reset_index(drop=True)
    if not (parameters.groupby("n_components").size().reindex(ks) == 200).all():
        raise ValueError("Expected 200 replicate seeds per K.")
    args.out.mkdir(parents=True, exist_ok=True)
    selected = adata[training, features].copy()
    selected.obs = pd.DataFrame(index=selected.obs_names)
    selected.var = pd.DataFrame(index=selected.var_names)
    input_path = args.out / (mode + "_counts.h5ad")
    selected.write_h5ad(input_path)
    gene_file = args.out / (mode + "_features.txt")
    gene_file.write_text("\n".join(features) + "\n")
    model = cNMF(output_dir=str(args.out), name=mode)
    model.prepare(
        counts_fn=str(input_path),
        components=ks,
        n_iter=200,
        seed=42,
        genes_file=str(gene_file),
        num_highvar_genes=len(features),
        max_NMF_iter=1000,
    )
    save_df_to_npz(parameters, model.paths["nmf_replicate_parameters"])
    model.factorize(worker_i=0, total_workers=1)
    model.combine(components=ks, skip_missing_files=False)
    for k in ks:
        model.consensus(k=k, density_threshold=2.0, show_clustering=False)


if __name__ == "__main__":
    main()
