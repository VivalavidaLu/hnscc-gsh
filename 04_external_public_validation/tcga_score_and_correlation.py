"""Reproduce the primary TCGA-HNSC GSH/cystine–CXCL10 association.

Input is a sample-by-gene expression table restricted to primary tumours. The
first column is the sample identifier and HGNC gene symbols are column names.
No TCGA source data are distributed with this repository.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
EXPR_PATH = DATA_ROOT / "raw" / "tcga_hnsc_primary_tumor_expression.csv"
OUT_DIR = RESULTS_ROOT / "external_validation" / "tcga_hnsc"

GSH_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]
EXPECTED_SAMPLES = 520
SEED = 301741


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - values.mean()) / values.std(ddof=0)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = np.column_stack([x, y])
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        sampled = values[rng.integers(0, len(values), len(values))]
        estimates[index] = stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic
    return tuple(np.nanquantile(estimates, [0.025, 0.975]))


def main() -> None:
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    expr.index = expr.index.astype(str)
    if expr.index.has_duplicates:
        raise RuntimeError("TCGA expression table contains duplicate sample identifiers")
    if len(expr) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES} primary tumours, found {len(expr)}")
    missing = sorted(set(GSH_GENES + ["CXCL10"]) - set(expr.columns))
    if missing:
        raise RuntimeError(f"Missing required genes: {missing}")

    scores = pd.DataFrame(index=expr.index)
    scores["GSH_cystine_score"] = expr[GSH_GENES].apply(zscore, axis=0).mean(axis=1)
    scores["CXCL10_expr"] = zscore(expr["CXCL10"])
    scores = scores.dropna()
    rho, p_value = stats.spearmanr(scores["GSH_cystine_score"], scores["CXCL10_expr"])
    ci_low, ci_high = bootstrap_spearman(
        scores["GSH_cystine_score"].to_numpy(), scores["CXCL10_expr"].to_numpy()
    )
    leave_one_out = []
    values = scores.to_numpy()
    for index in range(len(values)):
        keep = np.ones(len(values), dtype=bool)
        keep[index] = False
        leave_one_out.append(stats.spearmanr(values[keep, 0], values[keep, 1]).statistic)

    relative_input = str(EXPR_PATH.relative_to(REPOSITORY)) if EXPR_PATH.is_relative_to(REPOSITORY) else EXPR_PATH.name
    report = {
        "input": relative_input,
        "input_sha256": sha256(EXPR_PATH),
        "sample_selection": "TCGA-HNSC primary tumour aliquots; one sample per patient",
        "n": int(len(scores)),
        "gsh_genes": GSH_GENES,
        "score_definition": "mean of per-gene cohort z scores; population SD (ddof=0)",
        "spearman_rho": float(rho),
        "two_sided_asymptotic_p": float(p_value),
        "bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "bootstrap_replicates": 5000,
        "bootstrap_seed": SEED,
        "leave_one_out_rho_range": [float(np.min(leave_one_out)), float(np.max(leave_one_out))],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "tcga_hnsc_scores.csv", index_label="sample")
    (OUT_DIR / "tcga_hnsc_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
