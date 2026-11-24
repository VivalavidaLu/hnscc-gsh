"""Score discovery signatures and tissue composition in bulk cohorts."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json


def zscore(x):
    a = np.asarray(x, dtype=float)
    sd = np.nanstd(a, axis=0, ddof=0)
    return (a - np.nanmean(a, axis=0)) / np.where(sd > 0, sd, np.nan)


def signature(expr, genes):
    present = [g for g in genes if g in expr.columns]
    assert present
    return np.nanmean(zscore(expr[present]), axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Score discovery signatures and tissue composition in bulk cohorts."
    )
    parser.add_argument(
        "--expression",
        required=True,
        type=Path,
        help="Sample-by-gene CSV indexed by sample ID, normalized expression and unique genes",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Sample-indexed CSV with HPV_status, endpoints or mapped tissue composition covariates",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="CSV with covariates and standardized scores",
    )
    args = parser.parse_args()
    expr = pd.read_csv(args.expression, index_col=0).drop(
        columns="sampleID", errors="ignore"
    )
    meta = pd.read_csv(args.metadata, index_col=0).loc[expr.index].copy()
    if not expr.index.is_unique or not meta.index.is_unique:
        raise ValueError("Expected one row per sample.")
    plan = json.loads(
        Path(__file__).with_name("signature_definitions.json").read_text()
    )
    definitions = {
        "P1_top50": plan["primary_signature"],
        "P1_top100": plan["top100"],
        "GSH": plan["GSH"],
        "SLC7A11_GCLC": ["SLC7A11", "GCLC"],
        **plan["composition_markers"],
    }
    for label, genes in definitions.items():
        meta[label] = zscore(signature(expr, genes))
    meta["CXCL10"] = expr.CXCL10
    meta["HPV"] = meta.HPV_status.astype(str)
    meta.to_csv(args.out, index_label="sample")


if __name__ == "__main__":
    main()
