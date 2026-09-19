"""Audit the frozen 10,856 → 6,165 cell chain and final program labels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data" / "derived"
CLASSIFICATION = REPOSITORY / "data" / "processed" / "malignant_classification.csv"
PROGRAM_ASSIGNMENTS = REPOSITORY / "data" / "processed" / "cell_level_malignant_ifn_apm_scores.csv"
EXPECTED_PROGRAM_COUNTS = {"P1": 1857, "P2": 830, "P3": 904, "P4": 850, "P5": 1081, "P6": 643}


def ids(filename: str) -> pd.Index:
    frame = pd.read_csv(DATA / filename)
    if "keep" in frame.columns:
        frame = frame.loc[frame["keep"].astype(bool)]
    values = pd.Index(frame["cell_id"].astype(str), name="cell_id")
    if values.has_duplicates:
        raise ValueError(f"Duplicate cell IDs in {filename}")
    return values


def main() -> None:
    classification = pd.read_csv(CLASSIFICATION)
    required = {"cell_id", "MALIGNANT_DEFINITIVE"}
    missing = required.difference(classification.columns)
    if missing:
        raise ValueError(f"Missing classification columns: {sorted(missing)}")
    pure = ids("malignant_PURE_FINAL_cellids.csv")
    reference = ids("malignant_DEFINITIVE_FINAL_cellids.csv")
    definitive = set(classification.loc[classification["MALIGNANT_DEFINITIVE"].astype(bool), "cell_id"].astype(str))
    assignments = pd.read_csv(PROGRAM_ASSIGNMENTS, usecols=["cell_id", "program"])
    assignments["cell_id"] = assignments["cell_id"].astype(str)

    assert len(pure) == 10856
    assert len(reference) == 6165 and set(reference) <= set(pure)
    assert set(reference) <= definitive
    assert not assignments["cell_id"].duplicated().any()
    assert set(assignments["cell_id"]) == set(reference)
    program_counts = assignments["program"].value_counts().sort_index().astype(int).to_dict()
    assert program_counts == EXPECTED_PROGRAM_COUNTS

    print(json.dumps({
        "epithelial_purity": len(pure),
        "final_analysis_universe": len(reference),
        "final_within_epithelial_purity": True,
        "all_final_cells_have_malignant_evidence": True,
        "program_counts": program_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
