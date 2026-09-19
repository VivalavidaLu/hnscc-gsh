"""Apply the prespecified CNV/entropy rules used to identify malignant cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ENTROPY_THRESHOLD = 0.375
CNV_BURDEN_THRESHOLD = 0.05
def classify(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mixing_entropy_norm", "cnv_burden", "in_cnv_groupings"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["low_entropy"] = result["mixing_entropy_norm"] < ENTROPY_THRESHOLD
    result["high_cnv"] = result["cnv_burden"] >= CNV_BURDEN_THRESHOLD
    result["tier1"] = result["low_entropy"] & result["high_cnv"]
    result["tier2"] = (~result["in_cnv_groupings"].fillna(False).astype(bool)) & result["low_entropy"]
    result["MALIGNANT_DEFINITIVE"] = result["tier1"] | result["tier2"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="Cell-level CNV/entropy classification table")
    parser.add_argument("output_csv", type=Path, help="Output table with malignant-cell flags")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, index_col=0)
    result = classify(frame)
    observed = int(result["MALIGNANT_DEFINITIVE"].sum())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv)
    print(f"Definitive malignant cells: {observed:,}/{len(result):,}")


if __name__ == "__main__":
    main()
