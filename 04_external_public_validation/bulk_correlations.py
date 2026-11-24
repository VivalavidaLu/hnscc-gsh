"""Estimate unadjusted and composition-adjusted CXCL10 associations."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def rank_residual_correlation(df, x, y, covariates):
    d = df[[x, y] + covariates].dropna()
    n = len(d)
    if n < 4 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return dict(n=n, rho=np.nan, p=np.nan, df=np.nan)
    a = d[x].rank().to_numpy()
    b = d[y].rank().to_numpy()
    parts = [np.ones((n, 1))]
    for c in covariates:
        if pd.api.types.is_numeric_dtype(d[c]):
            parts.append(d[c].rank().to_numpy()[:, None])
        else:
            parts.append(pd.get_dummies(d[c], drop_first=True, dtype=float).to_numpy())
    X = np.column_stack(parts)
    rank = np.linalg.matrix_rank(X)
    dfree = n - rank - 1
    a = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    b = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    r = float(np.clip(a @ b / den, -1, 1)) if den > 0 else np.nan
    p = (
        2 * stats.t.sf(abs(r) * np.sqrt(dfree / max(1 - r * r, 1e-15)), dfree)
        if dfree > 0
        else np.nan
    )
    return dict(n=n, rho=r, p=float(p), df=dfree, design_rank=int(rank))


def partial_ci(df, x, y, covariates, n_boot=2000, seed=301741):
    d = df[[x, y] + covariates].dropna().reset_index(drop=True)
    out = rank_residual_correlation(d, x, y, covariates)
    rng = np.random.default_rng(seed)
    rr = []
    for _ in range(n_boot):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        rr.append(rank_residual_correlation(b, x, y, covariates)["rho"])
    out["ci_low"], out["ci_high"] = np.nanpercentile(rr, [2.5, 97.5])
    return out


def fdr(df, groups, pcol="p"):
    df = df.copy()
    df["q"] = np.nan
    for _, d in df.groupby(groups, dropna=False):
        good = d[pcol].notna()
        idx = d.index[good]
        df.loc[idx, "q"] = multipletests(d.loc[idx, pcol], method="fdr_bh")[1]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate unadjusted and composition-adjusted CXCL10 associations."
    )
    parser.add_argument(
        "--tcga",
        required=True,
        type=Path,
        help="TCGA-HNSC patient scores and covariates CSV",
    )
    parser.add_argument(
        "--panel", required=True, type=Path, help="GSE159067 scores and covariates CSV"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output CSV")
    args = parser.parse_args()
    rows = []
    for cohort, path in [("TCGA", args.tcga), ("GSE159067", args.panel)]:
        df = pd.read_csv(path, dtype={"HPV": str})
        models = [("unadjusted_all", df, [])]
        if cohort == "TCGA":
            matched = df.dropna(subset=["ABSOLUTE_purity", "leukocyte_fraction"])
            models += [
                (
                    "composition_markers_all",
                    df,
                    ["HPV", "epithelial", "immune", "stromal"],
                ),
                ("unadjusted_matched", matched, []),
                ("DNA_purity_HPV", matched, ["HPV", "ABSOLUTE_purity"]),
                (
                    "DNA_purity_leukocyte_stroma_HPV",
                    matched,
                    ["HPV", "ABSOLUTE_purity", "leukocyte_fraction", "stromal"],
                ),
            ]
        else:
            models.append(
                (
                    "composition_markers_HPV",
                    df,
                    ["HPV", "epithelial", "immune", "stromal"],
                )
            )
        for model, d, covs in models:
            for exposure in ["P1_top50", "GSH", "SLC7A11_GCLC"]:
                result = partial_ci(d, exposure, "CXCL10", covs)
                rows.append(
                    dict(
                        cohort=cohort,
                        model=model,
                        exposure=exposure,
                        outcome="CXCL10",
                        covariates=";".join(covs),
                        **result,
                    )
                )
    fdr(pd.DataFrame(rows), ["cohort", "model"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
