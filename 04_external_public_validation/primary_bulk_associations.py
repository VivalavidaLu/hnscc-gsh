"""Estimate the two prespecified bulk CXCL10 associations in Figure 4E/F."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 301741


def tcga_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = np.column_stack([x, y])
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        sampled = values[rng.integers(0, len(values), len(values))]
        estimates[index] = stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic
    return tuple(np.nanquantile(estimates, [0.025, 0.975]))


def panel_ci(
    x: pd.Series, y: pd.Series, seed: int = 301741, n_boot: int = 10000
) -> tuple[float, float]:
    pair = pd.concat([x, y], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sampled = pair[rng.integers(0, len(pair), len(pair))]
        estimates[i] = stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the two prespecified bulk CXCL10 associations in Figure 4E/F."
    )
    parser.add_argument(
        "--tcga", required=True, type=Path, help="TCGA-HNSC scored sample CSV"
    )
    parser.add_argument(
        "--panel", required=True, type=Path, help="GSE159067 scored sample CSV"
    )
    parser.add_argument("--out", required=True, type=Path, help="Results CSV")
    args = parser.parse_args()
    rows = []
    for cohort, path, exposure in [
        ("TCGA", args.tcga, "GSH"),
        ("GSE159067", args.panel, "SLC7A11_GCLC"),
    ]:
        d = pd.read_csv(path)[[exposure, "CXCL10"]].dropna()
        x, y = d[exposure], d.CXCL10
        rho, p = stats.spearmanr(x, y)
        interval = (
            tcga_ci(x.to_numpy(), y.to_numpy()) if cohort == "TCGA" else panel_ci(x, y)
        )
        rows.append(
            dict(
                cohort=cohort,
                exposure=exposure,
                n=len(d),
                rho=rho,
                p=p,
                ci_low=interval[0],
                ci_high=interval[1],
            )
        )
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
