"""Compare baseline and paired change between pathological response groups."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon
from patient_statistics import compare
from statsmodels.stats.multitest import multipletests


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and paired change between pathological response groups."
    )
    parser.add_argument(
        "--phase",
        required=True,
        type=Path,
        help="CSV with one row per Patient and Phase, Response (R/NR/Unresolved), program fractions in percent, usages, GSH_all_z",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output CSV for the 26 primary comparisons",
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        help="Optional CSV for the paired P1 fraction change across all patients in Figure 3A",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.phase)
    if data.duplicated(["Patient", "Phase"]).any():
        raise ValueError("Expected one row per patient and phase.")
    pre = data[data.Phase.eq("pre")].set_index("Patient")
    post = data[data.Phase.eq("post")].set_index("Patient")
    paired = pre.index.intersection(post.index)
    if args.paired_summary:
        change = (
            post.loc[paired, "P1_fraction"] - pre.loc[paired, "P1_fraction"]
        ).dropna()
        p_value = (
            wilcoxon(change, alternative="two-sided").pvalue
            if change.ne(0).any()
            else 1.0
        )
        pd.DataFrame(
            [
                {
                    "n_pairs": len(change),
                    "median_percentage_point_change": change.median(),
                    "paired_wilcoxon_p": p_value,
                }
            ]
        ).to_csv(args.paired_summary, index=False)
    paired = paired[pre.loc[paired, "Response"].isin(["R", "NR"])]
    if not pre.loc[paired, "Response"].equals(post.loc[paired, "Response"]):
        raise ValueError("Inconsistent response labels.")
    groups = pre.loc[paired, "Response"]
    metrics = [
        f"P{i}_{suffix}" for suffix in ["fraction", "usage"] for i in range(1, 7)
    ] + ["GSH_all_z"]
    rows = []
    for contrast, values in [
        ("pre", pre.loc[paired, metrics]),
        ("delta", post.loc[paired, metrics] - pre.loc[paired, metrics]),
    ]:
        for metric in metrics:
            result = compare(
                values.loc[groups.eq("R"), metric], values.loc[groups.eq("NR"), metric]
            )
            rows.append(dict(contrast=contrast, metric=metric, **result))
    table = pd.DataFrame(rows)
    table["BH_q_global_26"] = multipletests(table.exact_P, method="fdr_bh")[1]
    table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
