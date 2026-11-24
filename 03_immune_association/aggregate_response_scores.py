"""Aggregate program fractions, usage and the standardized all-malignant GSH score by patient and phase."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate program fractions, usage and the standardized all-malignant GSH score by patient and phase."
    )
    parser.add_argument(
        "--counts",
        required=True,
        type=Path,
        help="Full-library raw-count AnnData for the 6,165 malignant cells; obs.Patient and obs.TreatmentPhase",
    )
    parser.add_argument(
        "--usage",
        required=True,
        type=Path,
        help="Normalized six-program usage CSV indexed by cell ID",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output patient–phase scores CSV"
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Patient and Response (R or NR) for patients with verified labels; omit unassigned patients.",
    )
    args = parser.parse_args()
    labels_table = pd.read_csv(args.responses, usecols=["Patient", "Response"])
    if (
        labels_table.Patient.duplicated().any()
        or not labels_table.Response.isin(["R", "NR"]).all()
    ):
        raise ValueError(
            "Supply one verified R or NR per patient in the response file."
        )
    response = labels_table.set_index("Patient").Response
    a = ad.read_h5ad(args.counts)
    u = pd.read_csv(args.usage, index_col=0).loc[
        a.obs_names, [str(i) for i in range(1, 7)]
    ]
    if not np.allclose(u.sum(axis=1), 1) or u.isna().any().any():
        raise ValueError("Expected normalized usages.")
    genes = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]
    counts = a[:, genes].X
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    total = a.obs.nCount_RNA.to_numpy(dtype=np.float64)
    expr = np.log1p(
        counts.astype(np.float64) / total.astype(np.float64)[:, None] * 10000
    )
    scores = pd.DataFrame(
        {
            "Patient": a.obs.Patient.to_numpy(),
            "Phase": a.obs.TreatmentPhase.astype(str).str.lower().to_numpy(),
            "GSH_all_z": ((expr - expr.mean(0)) / expr.std(0)).mean(1),
        }
    )
    labels = u.idxmax(axis=1)
    for i in range(1, 7):
        scores[f"P{i}_usage"] = u[str(i)].to_numpy()
        scores[f"P{i}_fraction"] = 100 * labels.eq(str(i)).to_numpy(dtype=float)
    phase = scores.groupby(["Patient", "Phase"]).mean().reset_index()
    phase["Response"] = phase.Patient.map(response).fillna("Unresolved")
    phase.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
