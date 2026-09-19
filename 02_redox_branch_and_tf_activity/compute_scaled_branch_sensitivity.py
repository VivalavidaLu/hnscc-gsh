import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
H5AD = DATA_ROOT / "processed" / "malignant_6165.h5ad"
USAGE = DATA_ROOT / "derived" / "cNMF_K6_usage_6165.csv"
OUT = RESULTS_ROOT / "redox" / "scaled_expression"
OUT.mkdir(parents=True, exist_ok=True)


GENE_SETS = {
    "gsh_cysteine": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "phase2_detox": ["NQO1", "AKR1C1", "AKR1C2", "AKR1B10", "ALDH3A1", "GSTA4", "GSTM1", "GSTP1"],
    "thioredoxin_prdx": ["TXN", "TXNRD1", "PRDX1", "SRXN1"],
    "heme_iron": ["HMOX1", "FTH1", "FTL", "SLC40A1", "TFRC"],
    "nrf2_core": ["NFE2L2", "KEAP1", "MAFK", "MAFG", "BACH1"],
    "aa_transport": ["SLC1A5", "SLC7A5", "SLC38A1", "SLC38A2", "SLC1A4", "SLC7A8"],
}


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def get_gene_vector(adata, gene):
    idx = adata.var_names.get_loc(gene)
    x = adata.X[:, idx]
    if sparse.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
    return x.astype(float)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    adata = ad.read_h5ad(H5AD)
    usage = pd.read_csv(USAGE, index_col=0)
    common = adata.obs_names.intersection(usage.index)
    adata = adata[common, :].copy()
    usage = usage.loc[common]

    usage_cols = list(usage.columns)
    usage_numeric = usage.apply(pd.to_numeric, errors="coerce")
    dominant = usage_numeric.idxmax(axis=1).astype(str)

    obs = adata.obs.copy()
    obs["dominant_program_from_usage"] = dominant.values
    if "cNMF_program" in obs.columns:
        obs["cNMF_program"] = obs["cNMF_program"].astype(str)

    all_genes = sorted(set(g for genes in GENE_SETS.values() for g in genes))
    present = [g for g in all_genes if g in adata.var_names]
    missing = [g for g in all_genes if g not in adata.var_names]

    meta_rows = [
        {"item": "h5ad", "value": H5AD.name},
        {"item": "usage", "value": USAGE.name},
        {"item": "n_cells_common", "value": str(adata.n_obs)},
        {"item": "n_genes_h5ad", "value": str(adata.n_vars)},
        {"item": "has_raw", "value": str(adata.raw is not None)},
        {"item": "layers", "value": ";".join(adata.layers.keys())},
        {"item": "uns_log1p", "value": str(adata.uns.get("log1p", ""))},
        {"item": "usage_columns", "value": ";".join(usage_cols)},
        {"item": "present_genes", "value": ";".join(present)},
        {"item": "missing_genes", "value": ";".join(missing)},
    ]
    pd.DataFrame(meta_rows).to_csv(OUT / "input_provenance.csv", index=False)

    expr = {g: get_gene_vector(adata, g) for g in present}

    program_col = "dominant_program_from_usage"
    p1_mask = obs[program_col] == "1"
    records = []
    for gene in present:
        x = expr[gene]
        p = stats.mannwhitneyu(x[p1_mask], x[~p1_mask], alternative="two-sided").pvalue
        records.append(
            {
                "gene": gene,
                "branch": next(k for k, v in GENE_SETS.items() if gene in v),
                "n_P1": int(p1_mask.sum()),
                "n_other": int((~p1_mask).sum()),
                "mean_P1": float(x[p1_mask].mean()),
                "mean_other": float(x[~p1_mask].mean()),
                "pct_P1": float((x[p1_mask] > 0).mean() * 100),
                "pct_other": float((x[~p1_mask] > 0).mean() * 100),
                "cohen_d_P1_vs_other": cohen_d(x[p1_mask], x[~p1_mask]),
                "mw_p": float(p),
            }
        )
    df = pd.DataFrame(records)
    df["mw_q"] = bh_fdr(df["mw_p"].values)
    df.sort_values("cohen_d_P1_vs_other", ascending=False).to_csv(
        OUT / "gene_level_P1_vs_other_from_h5ad.csv", index=False
    )

    prog_records = []
    for gene in present:
        x = expr[gene]
        for prog in sorted(obs[program_col].unique(), key=lambda s: int(s)):
            mask = obs[program_col] == prog
            prog_records.append(
                {
                    "gene": gene,
                    "branch": next(k for k, v in GENE_SETS.items() if gene in v),
                    "program": prog,
                    "n_cells": int(mask.sum()),
                    "mean": float(x[mask].mean()),
                    "pct_positive": float((x[mask] > 0).mean() * 100),
                }
            )
    prog_df = pd.DataFrame(prog_records)
    rank_rows = []
    for gene, sub in prog_df.groupby("gene"):
        sub = sub.sort_values("mean", ascending=False).copy()
        sub["mean_rank_desc"] = np.arange(1, len(sub) + 1)
        rank_rows.append(sub)
    prog_rank = pd.concat(rank_rows, ignore_index=True)
    prog_rank.to_csv(OUT / "gene_mean_by_program_from_h5ad.csv", index=False)

    per_patient_records = []
    for gene in present:
        x = expr[gene]
        for patient, idx in obs.groupby("Patient", observed=False).indices.items():
            idx = np.asarray(idx)
            in_p1 = p1_mask.values[idx]
            if in_p1.sum() < 10 or (~in_p1).sum() < 10:
                continue
            per_patient_records.append(
                {
                    "gene": gene,
                    "branch": next(k for k, v in GENE_SETS.items() if gene in v),
                    "Patient": patient,
                    "n_P1": int(in_p1.sum()),
                    "n_other": int((~in_p1).sum()),
                    "mean_P1": float(x[idx][in_p1].mean()),
                    "mean_other": float(x[idx][~in_p1].mean()),
                    "cohen_d_P1_vs_other": cohen_d(x[idx][in_p1], x[idx][~in_p1]),
                }
            )
    pp = pd.DataFrame(per_patient_records)
    pp.to_csv(OUT / "per_patient_P1_vs_other_from_h5ad.csv", index=False)

    consistency = []
    for gene, sub in pp.groupby("gene"):
        consistency.append(
            {
                "gene": gene,
                "branch": sub["branch"].iloc[0],
                "n_testable_patients": int(len(sub)),
                "n_positive": int((sub["cohen_d_P1_vs_other"] > 0).sum()),
                "n_negative": int((sub["cohen_d_P1_vs_other"] < 0).sum()),
                "median_d": float(sub["cohen_d_P1_vs_other"].median()),
                "mean_d": float(sub["cohen_d_P1_vs_other"].mean()),
                "min_n_P1": int(sub["n_P1"].min()),
                "max_n_P1": int(sub["n_P1"].max()),
            }
        )
    pd.DataFrame(consistency).sort_values("branch").to_csv(
        OUT / "per_patient_direction_summary_from_h5ad.csv", index=False
    )

    branch_summary = []
    for branch, genes in GENE_SETS.items():
        pg = [g for g in genes if g in present]
        if not pg:
            continue
        mat = np.vstack([expr[g] for g in pg]).T
        score = mat.mean(axis=1)
        p = stats.mannwhitneyu(score[p1_mask], score[~p1_mask], alternative="two-sided").pvalue
        means_by_program = {}
        for prog in sorted(obs[program_col].unique(), key=lambda s: int(s)):
            mask = obs[program_col] == prog
            means_by_program[f"P{prog}_mean"] = float(score[mask].mean())
        top_program = max(means_by_program, key=means_by_program.get).replace("_mean", "")
        branch_summary.append(
            {
                "branch": branch,
                "genes_present": ";".join(pg),
                "n_genes_present": len(pg),
                "P1_mean": float(score[p1_mask].mean()),
                "other_mean": float(score[~p1_mask].mean()),
                "cohen_d_P1_vs_other": cohen_d(score[p1_mask], score[~p1_mask]),
                "mw_p": float(p),
                "top_program_by_mean": top_program,
                **means_by_program,
            }
        )
    bdf = pd.DataFrame(branch_summary)
    bdf["mw_q"] = bh_fdr(bdf["mw_p"].values)
    bdf.to_csv(OUT / "branch_score_summary_from_h5ad.csv", index=False)

    dominance_check = pd.DataFrame(
        {
            "dominant_program": obs[program_col].value_counts().sort_index(),
            "h5ad_cNMF_program": obs["cNMF_program"].value_counts().sort_index()
            if "cNMF_program" in obs.columns
            else pd.Series(dtype=int),
        }
    )
    dominance_check.to_csv(OUT / "program_count_crosscheck.csv")

    print("Wrote scaled-expression audit files under the configured results directory.")
    print("Branch summary:")
    print(bdf[["branch", "n_genes_present", "cohen_d_P1_vs_other", "top_program_by_mean"]].to_string(index=False))
    print("Key genes:")
    key = df[df["gene"].isin(["SLC7A11", "GCLC", "GCLM", "NQO1", "HMOX1", "TXN", "TXNRD1", "PRDX1", "NFE2L2"])]
    print(key[["gene", "branch", "cohen_d_P1_vs_other", "mean_P1", "mean_other", "pct_P1", "pct_other"]].to_string(index=False))


if __name__ == "__main__":
    main()
