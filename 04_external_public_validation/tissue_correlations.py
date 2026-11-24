"""Compute the two exploratory tissue correlations and their joint BH correction."""

import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        required=True,
        type=Path,
        help="One row per specimen with numeric SLC7A11, GCLC, CXCL10 and CD8 staining scores",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output correlation summary CSV"
    )
    args = parser.parse_args()
    data = pd.read_csv(args.scores, usecols=["SLC7A11", "GCLC", "CXCL10", "CD8"])
    if data.isna().any().any():
        raise ValueError("Expected complete scores for the paired stains.")
    markers = data[["SLC7A11", "GCLC"]]
    composite = ((markers - markers.mean()) / markers.std(ddof=1)).mean(axis=1)
    rows = []
    for outcome in ["CXCL10", "CD8"]:
        rho, p = spearmanr(composite, data[outcome])
        rows.append({"outcome": outcome, "n": len(data), "rho": rho, "p": p})
    result = pd.DataFrame(rows)
    result["q"] = multipletests(result.p, method="fdr_bh")[1]
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
