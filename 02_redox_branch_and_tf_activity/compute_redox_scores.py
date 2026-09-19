import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
RAW_H5AD = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
MALIGNANT_H5AD = DATA_ROOT / "processed" / "malignant_6165.h5ad"
USAGE = DATA_ROOT / "derived" / "cNMF_K6_usage_6165.csv"
OUT = RESULTS_ROOT / "redox" / "raw_count"
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
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


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


def dense_matrix(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def branch_for_gene(gene):
    for branch, genes in GENE_SETS.items():
        if gene in genes:
            return branch
    raise KeyError(gene)


def main():
    malignant = ad.read_h5ad(MALIGNANT_H5AD, backed="r")
    usage = pd.read_csv(USAGE, index_col=0)
    common_cells = malignant.obs_names.intersection(usage.index)
    malignant_obs = malignant.obs.loc[common_cells].copy()
    malignant.file.close()

    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    missing_cells = common_cells.difference(raw.obs_names)
    if len(missing_cells) > 0:
        raise RuntimeError(f"{len(missing_cells)} malignant cells missing from raw h5ad")

    all_genes = sorted(set(g for genes in GENE_SETS.values() for g in genes))
    present = [g for g in all_genes if g in raw.var_names]
    missing_genes = [g for g in all_genes if g not in raw.var_names]

    cell_pos = raw.obs_names.get_indexer(common_cells)
    gene_pos = raw.var_names.get_indexer(present)
    counts = dense_matrix(raw.X[cell_pos, :][:, gene_pos]).astype(float)
    raw_obs = raw.obs.iloc[cell_pos].copy()
    raw.file.close()

    if "nCount_RNA" not in raw_obs.columns:
        raise RuntimeError("RAW h5ad lacks nCount_RNA; cannot normalize selected genes without full-matrix sums")
    total = pd.to_numeric(raw_obs["nCount_RNA"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise RuntimeError("Invalid nCount_RNA values")

    lognorm = np.log1p(counts / total[:, None] * 1e4)
    usage = usage.loc[common_cells].apply(pd.to_numeric, errors="coerce")
    dominant = usage.idxmax(axis=1).astype(str)

    obs = raw_obs.copy()
    obs["dominant_program_from_usage"] = dominant.values
    obs["cNMF_program_malignant_h5ad"] = malignant_obs["cNMF_program"].astype(str).values
    label_sources = {
        "usage_idxmax": "dominant_program_from_usage",
        "h5ad_cNMF_program": "cNMF_program_malignant_h5ad",
    }

    pd.DataFrame(
        [
            {"item": "raw_h5ad", "value": RAW_H5AD.name},
            {"item": "malignant_h5ad_for_cell_ids", "value": MALIGNANT_H5AD.name},
            {"item": "usage", "value": USAGE.name},
            {"item": "n_cells_common", "value": str(len(common_cells))},
            {"item": "n_raw_genes", "value": str(raw.n_vars if hasattr(raw, "n_vars") else "")},
            {"item": "normalization", "value": "log1p(raw_count / nCount_RNA * 1e4)"},
            {"item": "present_genes", "value": ";".join(present)},
            {"item": "missing_genes", "value": ";".join(missing_genes)},
        ]
    ).to_csv(OUT / "input_provenance.csv", index=False)

    gene_to_col = {g: i for i, g in enumerate(present)}

    latest_bdf = None
    latest_gene_df = None
    for source_name, label_col in label_sources.items():
        labels = obs[label_col].astype(str)
        p1_mask = labels.to_numpy() == "1"

        gene_records = []
        for gene in present:
            j = gene_to_col[gene]
            x = lognorm[:, j]
            c = counts[:, j]
            p = stats.mannwhitneyu(x[p1_mask], x[~p1_mask], alternative="two-sided").pvalue
            gene_records.append(
                {
                    "label_source": source_name,
                    "gene": gene,
                    "branch": branch_for_gene(gene),
                    "n_P1": int(p1_mask.sum()),
                    "n_other": int((~p1_mask).sum()),
                    "mean_lognorm_P1": float(x[p1_mask].mean()),
                    "mean_lognorm_other": float(x[~p1_mask].mean()),
                    "pct_count_positive_P1": float((c[p1_mask] > 0).mean() * 100),
                    "pct_count_positive_other": float((c[~p1_mask] > 0).mean() * 100),
                    "mean_raw_count_P1": float(c[p1_mask].mean()),
                    "mean_raw_count_other": float(c[~p1_mask].mean()),
                    "cohen_d_lognorm_P1_vs_other": cohen_d(x[p1_mask], x[~p1_mask]),
                    "mw_p_lognorm": float(p),
                }
            )
        gene_df = pd.DataFrame(gene_records)
        gene_df["mw_q_lognorm"] = bh_fdr(gene_df["mw_p_lognorm"].values)
        gene_df.sort_values("cohen_d_lognorm_P1_vs_other", ascending=False).to_csv(
            OUT / f"gene_level_P1_vs_other_from_raw_lognorm__{source_name}.csv", index=False
        )

        program_records = []
        for gene in present:
            x = lognorm[:, gene_to_col[gene]]
            c = counts[:, gene_to_col[gene]]
            for prog in sorted(labels.unique(), key=lambda s: int(s)):
                mask = labels.to_numpy() == prog
                program_records.append(
                    {
                        "label_source": source_name,
                        "gene": gene,
                        "branch": branch_for_gene(gene),
                        "program": prog,
                        "n_cells": int(mask.sum()),
                        "mean_lognorm": float(x[mask].mean()),
                        "pct_count_positive": float((c[mask] > 0).mean() * 100),
                        "mean_raw_count": float(c[mask].mean()),
                    }
                )
        prog_df = pd.DataFrame(program_records)
        ranked = []
        for gene, sub in prog_df.groupby("gene"):
            sub = sub.sort_values("mean_lognorm", ascending=False).copy()
            sub["mean_lognorm_rank_desc"] = np.arange(1, len(sub) + 1)
            ranked.append(sub)
        pd.concat(ranked, ignore_index=True).to_csv(
            OUT / f"gene_mean_by_program_from_raw_lognorm__{source_name}.csv", index=False
        )

        patient_records = []
        patient_vals = obs["Patient"].astype(str).to_numpy()
        for gene in present:
            x = lognorm[:, gene_to_col[gene]]
            for patient in sorted(pd.unique(patient_vals)):
                idx = patient_vals == patient
                in_p1 = p1_mask[idx]
                if in_p1.sum() < 10 or (~in_p1).sum() < 10:
                    continue
                patient_records.append(
                    {
                        "label_source": source_name,
                        "gene": gene,
                        "branch": branch_for_gene(gene),
                        "Patient": patient,
                        "n_P1": int(in_p1.sum()),
                        "n_other": int((~in_p1).sum()),
                        "cohen_d_lognorm_P1_vs_other": cohen_d(x[idx][in_p1], x[idx][~in_p1]),
                        "mean_lognorm_P1": float(x[idx][in_p1].mean()),
                        "mean_lognorm_other": float(x[idx][~in_p1].mean()),
                    }
                )
        pp = pd.DataFrame(patient_records)
        pp.to_csv(OUT / f"per_patient_P1_vs_other_from_raw_lognorm__{source_name}.csv", index=False)

        consistency = []
        for gene, sub in pp.groupby("gene"):
            consistency.append(
                {
                    "label_source": source_name,
                    "gene": gene,
                    "branch": sub["branch"].iloc[0],
                    "n_testable_patients": int(len(sub)),
                    "n_positive": int((sub["cohen_d_lognorm_P1_vs_other"] > 0).sum()),
                    "n_negative": int((sub["cohen_d_lognorm_P1_vs_other"] < 0).sum()),
                    "median_d": float(sub["cohen_d_lognorm_P1_vs_other"].median()),
                    "mean_d": float(sub["cohen_d_lognorm_P1_vs_other"].mean()),
                    "min_n_P1": int(sub["n_P1"].min()),
                    "max_n_P1": int(sub["n_P1"].max()),
                }
            )
        pd.DataFrame(consistency).sort_values(["branch", "gene"]).to_csv(
            OUT / f"per_patient_direction_summary_from_raw_lognorm__{source_name}.csv", index=False
        )

        branch_records = []
        for branch, genes in GENE_SETS.items():
            pg = [g for g in genes if g in present]
            mat = lognorm[:, [gene_to_col[g] for g in pg]]
            score = mat.mean(axis=1)
            means = {}
            for prog in sorted(labels.unique(), key=lambda s: int(s)):
                mask = labels.to_numpy() == prog
                means[f"P{prog}_mean_lognorm"] = float(score[mask].mean())
            top = max(means, key=means.get).replace("_mean_lognorm", "")
            p = stats.mannwhitneyu(score[p1_mask], score[~p1_mask], alternative="two-sided").pvalue
            branch_records.append(
                {
                    "label_source": source_name,
                    "branch": branch,
                    "genes_present": ";".join(pg),
                    "n_genes_present": len(pg),
                    "P1_mean_lognorm": float(score[p1_mask].mean()),
                    "other_mean_lognorm": float(score[~p1_mask].mean()),
                    "cohen_d_lognorm_P1_vs_other": cohen_d(score[p1_mask], score[~p1_mask]),
                    "mw_p_lognorm": float(p),
                    "top_program_by_mean_lognorm": top,
                    **means,
                }
            )
        bdf = pd.DataFrame(branch_records)
        bdf["mw_q_lognorm"] = bh_fdr(bdf["mw_p_lognorm"].values)
        bdf.to_csv(OUT / f"branch_score_summary_from_raw_lognorm__{source_name}.csv", index=False)

        pd.DataFrame(
            {
                "program": sorted(labels.unique(), key=lambda s: int(s)),
                "n_cells": [int((labels == p).sum()) for p in sorted(labels.unique(), key=lambda s: int(s))],
            }
        ).to_csv(OUT / f"program_counts__{source_name}.csv", index=False)

        latest_bdf = bdf
        latest_gene_df = gene_df

    print("Wrote raw-count audit files under the configured results directory.")
    print("Branch summary:")
    print(
        latest_bdf[
            [
                "label_source",
                "branch",
                "n_genes_present",
                "cohen_d_lognorm_P1_vs_other",
                "top_program_by_mean_lognorm",
            ]
        ].to_string(index=False)
    )
    key = latest_gene_df[
        latest_gene_df["gene"].isin(["SLC7A11", "GCLC", "GCLM", "NQO1", "HMOX1", "TXN", "TXNRD1", "PRDX1", "NFE2L2"])
    ]
    print("Key genes:")
    print(
        key[
            [
                "label_source",
                "gene",
                "branch",
                "cohen_d_lognorm_P1_vs_other",
                "mean_lognorm_P1",
                "mean_lognorm_other",
                "pct_count_positive_P1",
                "pct_count_positive_other",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
