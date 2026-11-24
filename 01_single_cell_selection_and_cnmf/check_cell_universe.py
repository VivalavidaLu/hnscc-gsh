"""Check final malignant-cell membership and counts against the discovery projection."""

import argparse
from pathlib import Path
import pandas as pd


EXPECTED_GSE301741_EPITHELIAL_PURITY_CELLS = 10_856
EXPECTED_DISCOVERY_PROGRAM_COUNTS = [1857, 830, 904, 850, 1081, 643]


def read_ids(path):
    data = pd.read_csv(path, usecols=["cell_id"])
    if data.cell_id.isna().any() or data.cell_id.duplicated().any():
        raise ValueError("Each cell must have exactly one identifier.")
    return set(data.cell_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epithelial",
        required=True,
        type=Path,
        help="Epithelial-purity-retained cell IDs (CSV, cell_id)",
    )
    parser.add_argument(
        "--final", required=True, type=Path, help="Final discovery cells (CSV, cell_id)"
    )
    parser.add_argument(
        "--classification",
        required=True,
        type=Path,
        help="Classification CSV (cell_id and MALIGNANT_DEFINITIVE)",
    )
    parser.add_argument(
        "--usage",
        required=True,
        type=Path,
        help="Normalized six-program projection indexed by cell ID",
    )
    args = parser.parse_args()
    purity = read_ids(args.epithelial)
    final = read_ids(args.final)
    classified = pd.read_csv(
        args.classification, usecols=["cell_id", "MALIGNANT_DEFINITIVE"]
    )
    if classified.cell_id.duplicated().any():
        raise ValueError("Duplicate classification records.")
    positive = (
        classified.MALIGNANT_DEFINITIVE.astype("string").str.lower().isin(["true", "1"])
    )
    evidence = set(classified.loc[positive, "cell_id"])
    usage = pd.read_csv(args.usage, index_col=0)[list("123456")]
    if (
        not usage.index.is_unique
        or set(usage.index) != final
        or not final <= purity
        or not final <= evidence
    ):
        raise ValueError("Inconsistent final cohort membership.")
    counts = usage.idxmax(axis=1).value_counts().sort_index()
    if len(purity) != EXPECTED_GSE301741_EPITHELIAL_PURITY_CELLS:
        raise ValueError(
            f"GSE301741 epithelial-purity set: expected "
            f"{EXPECTED_GSE301741_EPITHELIAL_PURITY_CELLS} cells, found {len(purity)}."
        )
    if counts.tolist() != EXPECTED_DISCOVERY_PROGRAM_COUNTS:
        raise ValueError(
            f"GSE301741 discovery projection: expected P1–P6 counts "
            f"{EXPECTED_DISCOVERY_PROGRAM_COUNTS}, found {counts.tolist()}."
        )
    print(f"{len(purity)} epithelial-purity cells; {len(final)} final cells.")
    print(counts.to_string())


if __name__ == "__main__":
    main()
