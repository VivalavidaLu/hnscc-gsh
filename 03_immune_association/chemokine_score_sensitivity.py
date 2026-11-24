"""Test the full GSH score, a two-gene deletion and standardized SLC3A2 within P1."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


from statsmodels.stats.multitest import multipletests

SEED = 20260718
N_BOOT = 5000


def boot(d: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    d = d[["Patient", x, y]].dropna().reset_index(drop=True)
    patients = d.Patient.unique()
    group_index = [
        np.flatnonzero(d.Patient.to_numpy() == patient) for patient in patients
    ]
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    rng, values = np.random.default_rng(SEED), np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate(
            [
                group_index[j]
                for j in rng.integers(0, len(group_index), len(group_index))
            ]
        )
        values[i] = stats.spearmanr(xv[idx], yv[idx]).statistic
    return tuple(map(float, np.quantile(values, [0.025, 0.5, 0.975])))


def main():
    parser = argparse.ArgumentParser(
        description="Test the full GSH score, a two-gene deletion and standardized SLC3A2 within P1."
    )
    parser.add_argument(
        "--cells",
        required=True,
        type=Path,
        help="P1-only CSV with Patient, Phase, and log1p(CP10K) columns named by gene symbol",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output correlation and donor-omission summary CSV",
    )
    args = parser.parse_args()
    cells = pd.read_csv(args.cells)
    genes = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]
    z = (cells[genes] - cells[genes].mean()) / cells[genes].std(ddof=0)
    scores = {
        "Original 7-gene GSH score": genes,
        "Remove SLC7A11/GCLC": [g for g in genes if g not in ["SLC7A11", "GCLC"]],
        "Standardized SLC3A2": ["SLC3A2"],
    }
    rows = []
    for score, selected in scores.items():
        cells["score"] = z[selected].mean(axis=1)
        grp = cells.groupby(["Patient", "Phase"])
        phase = grp[["score", "CXCL9", "CXCL10"]].mean()[grp.size() >= 10].reset_index()
        for gene in ["CXCL9", "CXCL10"]:
            r, p = stats.spearmanr(phase.score, phase[gene])
            low, _, high = boot(phase, "score", gene)
            retained = []
            for donor in phase.Patient.unique():
                subset = phase[phase.Patient != donor]
                retained.append(stats.spearmanr(subset.score, subset[gene]).statistic)
            rows.append(
                dict(
                    feature=score,
                    feature_type="core_removal",
                    cluster_ci95_low=low,
                    cluster_ci95_high=high,
                    loo_fraction_negative=float(np.mean(np.array(retained) < 0)),
                    gene=gene,
                    rho=r,
                    nominal_p=p,
                    loo_negative=int((np.array(retained) < 0).sum()),
                    loo_total=len(retained),
                )
            )
    result = pd.DataFrame(rows)
    result["fdr_within_gene_feature_type"] = result.groupby("gene").nominal_p.transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
