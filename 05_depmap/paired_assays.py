"""Analyze matched independent experiments using technical-well averages or assay endpoints."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def cck8(data):
    keys = ["cell_line", "experiment", "condition"]
    if (
        data[keys + ["well", "absorbance"]].isna().any().any()
        or data.duplicated(keys + ["well"]).any()
    ):
        raise ValueError(
            "Each technical well requires a unique identifier and a measured absorbance."
        )
    wells = data.groupby(keys).absorbance
    if not wells.size().eq(6).all():
        raise ValueError(
            "The CCK-8 analysis uses six wells per condition and experiment."
        )
    means = wells.mean().unstack("condition")
    denominator = means["Ctrl"] - means["Blank"]
    if denominator.isna().any() or denominator.le(0).any():
        raise ValueError("Expected a positive matched blank-corrected vehicle value.")
    return (
        means.drop(columns="Blank").sub(means.Blank, axis=0).div(denominator, axis=0)
        * 100
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="CSV with cell_line, experiment, condition, and well and absorbance (CCK-8) or value (preprocessed endpoint). For CXCL10, leave nonquantifiable measurements blank; plotting-only values must be excluded.",
    )
    parser.add_argument(
        "--assay", required=True, choices=["cck8", "cxcl10", "normalized"]
    )
    parser.add_argument("--reference", required=True, help="Reference condition label")
    parser.add_argument("--treatment", required=True, help="Comparison condition label")
    parser.add_argument("--out", required=True, type=Path, help="Output comparison CSV")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    if args.assay == "cck8":
        wide = cck8(df)
    else:
        wide = df.pivot(
            index=["cell_line", "experiment"], columns="condition", values="value"
        )
        if args.assay == "cxcl10":
            if (wide <= 0).any().any():
                raise ValueError(
                    "ELISA concentrations must be positive for the log10 analysis."
                )
            wide = np.log10(wide)
    records = []
    for line, group in wide.groupby(level="cell_line"):
        pair = group[[args.reference, args.treatment]]
        if args.assay == "cxcl10":
            pair = pair.dropna()
        if pair.isna().any().any() or len(pair) < 2:
            raise ValueError(
                "At least two matched independent experiments with both values are required."
            )
        a, b = pair[args.reference], pair[args.treatment]
        test = ttest_rel(a, b, alternative="two-sided")
        records.append(
            {
                "cell_line": line,
                "reference": args.reference,
                "treatment": args.treatment,
                "n_pairs": len(pair),
                "difference_treatment_minus_reference": (b - a).mean(),
                "paired_t_reference_minus_treatment": test.statistic,
                "unadjusted_p": test.pvalue,
            }
        )
    pd.DataFrame(records).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
