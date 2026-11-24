"""Compute expression proxy and individual gene effects by P1 usage tertile."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PROXIES = {
    "Palbociclib": ["CDK4", "CDK6", "CCND1", "CCND2", "CCND3", "RB1"],
    "JPH203": ["SLC7A5", "SLC3A2"],
    "Erastin": ["SLC7A11", "SLC3A2", "GPX4", "ACSL4", "TFRC"],
    "Sulfasalazine": ["SLC7A11", "SLC3A2"],
    "ML385": ["NFE2L2", "KEAP1", "SLC7A11", "GCLC", "GCLM", "NQO1", "TXNRD1"],
    "BSO": ["GCLC", "GCLM", "GSS"],
    "Echinomycin": ["HIF1A", "SLC2A1", "LDHA", "PDK1", "VEGFA", "CA9"],
    "Everolimus": ["MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1", "RPS6", "TSC1", "TSC2"],
    "Rapamycin": ["MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1", "RPS6"],
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute expression proxy and individual gene effects by P1 usage tertile."
    )
    parser.add_argument(
        "--expression",
        required=True,
        type=Path,
        help="Malignant cell × gene CSV of full-library log1p(CP10K), indexed by cell ID",
    )
    parser.add_argument(
        "--usage",
        required=True,
        type=Path,
        help="Six-program usage CSV indexed by cell ID",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    expr = pd.read_csv(args.expression, index_col=0)
    usage = pd.read_csv(args.usage, index_col=0)
    if (
        not expr.index.is_unique
        or set(expr.index) != set(usage.index)
        or not usage.index.is_unique
    ):
        raise ValueError("Expected identical unique cell IDs in both inputs.")
    u = usage.loc[expr.index, [str(i) for i in range(1, 7)]]
    p1 = u["1"] / u.sum(axis=1)
    low, high = p1.quantile([1 / 3, 2 / 3])
    lo = p1.le(low)
    hi = p1.ge(high)
    if (lo & hi).any():
        raise ValueError("The tertiles overlap.")

    def effect(values):
        h = values[hi]
        l = values[lo]
        return dict(
            n_high=len(h),
            n_low=len(l),
            mean_high=h.mean(),
            mean_low=l.mean(),
            sd_high=h.std(),
            sd_low=l.std(),
            cohens_d=cohen_d(h, l),
        )

    rows = []
    for name, genes in PROXIES.items():
        rows.append(
            dict(
                drug=name,
                genes_used=";".join(genes),
                **effect(expr[genes].mean(axis=1)),
            )
        )
    individual = []
    for gene in ["GCLC", "GCLM", "GSS"]:
        values = expr[gene]
        individual.append(
            dict(
                gene=gene,
                pct_detected_high=values[hi].gt(0).mean(),
                pct_detected_low=values[lo].gt(0).mean(),
                **effect(values),
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("cohens_d", ascending=False).to_csv(
        args.out / "expression_proxies.csv", index=False
    )
    pd.DataFrame(individual).to_csv(args.out / "gene_effects.csv", index=False)


if __name__ == "__main__":
    main()
