"""Test P1 and GSH signatures in DepMap Public 24Q2."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re
from scipy import stats
from statsmodels.stats.multitest import multipletests

EXPECTED_DEPMAP_24Q2_HNSCC_MODELS = 57

TARGETS = ["GCLC", "GCLM", "GSS", "SLC7A11", "GPX4"]

GSH_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]


def gene_symbol(column: str) -> str:
    return re.sub(r"\s+\(\d+\)$", "", str(column))


def p1_definitions(path) -> tuple[list[str], pd.Series]:
    spectra = pd.read_csv(path, sep="\t", index_col=0)
    row = spectra.loc[1] if 1 in spectra.index else spectra.loc["1"]
    ranked = row.sort_values(ascending=False)
    ranked = ranked[~ranked.index.str.match(r"^(MT-|RPL|RPS)")]
    return ranked.head(50).index.tolist(), ranked.head(200)


def read_expression(path: Path, genes: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    mapping: dict[str, str] = {}
    for column in header[1:]:
        mapping.setdefault(gene_symbol(column), column)
    present = [gene for gene in genes if gene in mapping]
    frame = pd.read_csv(path, usecols=[header[0], *[mapping[g] for g in present]])
    return frame.rename(
        columns={header[0]: "ModelID", **{mapping[g]: g for g in present}}
    ).set_index("ModelID")


def read_crispr_targets(path: Path) -> pd.DataFrame:
    return read_expression(path, TARGETS)


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)


def build_scores(
    expression: pd.DataFrame, top50: list[str], weighted: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_columns(expression)
    top50_used = [gene for gene in top50 if gene in z]
    weighted_used = [gene for gene in weighted.index if gene in z]
    weights = weighted.loc[weighted_used].clip(lower=0)
    weights = weights / weights.sum()
    scores = pd.DataFrame(index=z.index)
    scores["P1_top50"] = z[top50_used].mean(axis=1)
    scores["P1_weighted"] = z[weighted_used].mul(weights, axis=1).sum(axis=1)
    present_gsh = [gene for gene in GSH_GENES if gene in z]
    scores["GSH_cystine"] = z[present_gsh].mean(axis=1)
    for target in TARGETS:
        present = [gene for gene in GSH_GENES if gene != target and gene in z]
        scores[f"GSH_without_{target}"] = z[present].mean(axis=1)
    coverage = pd.DataFrame(
        [
            {
                "signature": "P1_top50",
                "requested": 50,
                "used": len(top50_used),
                "genes": ";".join(top50_used),
            },
            {
                "signature": "P1_weighted",
                "requested": 200,
                "used": len(weighted_used),
                "genes": ";".join(weighted_used),
            },
            {
                "signature": "GSH_cystine",
                "requested": len(GSH_GENES),
                "used": len(present_gsh),
                "genes": ";".join(present_gsh),
            },
        ]
    )
    return scores, coverage


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signature in ["P1_top50", "P1_weighted", "GSH_cystine"]:
        for target in TARGETS:
            pair = frame[[signature, f"{target}_dependency_strength"]].dropna()
            rho, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append(
                {
                    "signature": signature,
                    "dependency_gene": target,
                    "n": len(pair),
                    "rho": rho,
                    "p_value": p_value,
                }
            )
    for target in TARGETS:
        signature = f"GSH_without_{target}"
        pair = frame[[signature, f"{target}_dependency_strength"]].dropna()
        rho, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append(
            {
                "signature": signature,
                "dependency_gene": target,
                "n": len(pair),
                "rho": rho,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["BH_FDR_20_tests"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def leave_one_model_out(frame: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test in tests.itertuples(index=False):
        x = test.signature
        y = f"{test.dependency_gene}_dependency_strength"
        pair = frame[[x, y]].dropna()
        for model_id in pair.index:
            subset = pair.drop(index=model_id)
            rho, p_value = stats.spearmanr(subset[x], subset[y])
            rows.append(
                {
                    "signature": x,
                    "dependency_gene": test.dependency_gene,
                    "omitted_model": model_id,
                    "n": len(subset),
                    "rho": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def cliffs_delta(high: np.ndarray, low: np.ndarray) -> float:
    return float(
        (np.sum(high[:, None] > low[None, :]) - np.sum(high[:, None] < low[None, :]))
        / (len(high) * len(low))
    )


def quartile_contrasts(model):
    quartiles = []
    for signature in ("P1_weighted", "GSH_without_GCLC"):
        q1, q3 = model[signature].quantile([0.25, 0.75])
        low = model.loc[model[signature] <= q1, "GCLC_dependency_strength"].to_numpy(
            float
        )
        high = model.loc[model[signature] >= q3, "GCLC_dependency_strength"].to_numpy(
            float
        )
        test = stats.mannwhitneyu(high, low, alternative="two-sided", method="exact")
        quartiles.append(
            {
                "signature": signature,
                "n_low": len(low),
                "n_high": len(high),
                "median_low": float(np.median(low)),
                "median_high": float(np.median(high)),
                "cliffs_delta": cliffs_delta(high, low),
                "mann_whitney_p": float(test.pvalue),
            }
        )
    quartile_fdr = multipletests(
        [row["mann_whitney_p"] for row in quartiles], method="fdr_bh"
    )[1]
    for row, adjusted in zip(quartiles, quartile_fdr):
        row["BH_FDR_2_quartile_tests"] = float(adjusted)
    return pd.DataFrame(quartiles)


def bootstrap_interval(x, y, seed, replicates=20000):
    """Resample matched cell-line models and rerank each resample."""
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        draw = rng.integers(0, len(x), size=(size, len(x)))
        a = stats.rankdata(x[draw], axis=1)
        b = stats.rankdata(y[draw], axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        usable = denominator > 0
        estimates.extend((a[usable] * b[usable]).sum(axis=1) / denominator[usable])
    return np.quantile(estimates, [0.025, 0.975])


def panel_intervals(model):
    rows = []

    def add(panel, signature, target, seed):
        x = model[signature].to_numpy(float)
        y = model[f"{target}_dependency_strength"].to_numpy(float)
        low, high = bootstrap_interval(x, y, seed)
        rows.append(
            dict(
                panel=panel,
                signature=signature,
                target=target,
                n=len(x),
                rho=stats.spearmanr(x, y).statistic,
                ci_low=low,
                ci_high=high,
            )
        )

    for i, signature in enumerate(
        ["P1_top50", "P1_weighted", "GSH_cystine", "GSH_without_GCLC"]
    ):
        add("S6B", signature, "GCLC", 301760 + i)
    for i, target in enumerate(TARGETS):
        add("5D", f"GSH_without_{target}", target, 301742 + i)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Test P1 and GSH signatures in DepMap Public 24Q2."
    )
    parser.add_argument(
        "--expression",
        required=True,
        type=Path,
        help="DepMap protein-coding logTPM CSV",
    )
    parser.add_argument(
        "--crispr", required=True, type=Path, help="DepMap Chronos gene effect CSV"
    )
    parser.add_argument(
        "--models", required=True, type=Path, help="DepMap Model metadata CSV"
    )
    parser.add_argument(
        "--spectra", required=True, type=Path, help="cNMF gene score spectra TSV"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    expression_path, crispr_path, model_path = args.expression, args.crispr, args.models
    top50, weighted = p1_definitions(args.spectra)
    requested = list(dict.fromkeys([*weighted.index, *GSH_GENES]))
    expression = read_expression(expression_path, requested)
    scores, coverage = build_scores(expression, top50, weighted)

    models = pd.read_csv(model_path, low_memory=False)
    models["is_hnscc"] = (
        models["OncotreePrimaryDisease"]
        .fillna("")
        .eq("Head and Neck Squamous Cell Carcinoma")
    )
    metadata = models.set_index("ModelID")
    hnscc_ids = metadata.index[metadata["is_hnscc"]]

    gene_effect = read_crispr_targets(crispr_path)
    dependency = (-gene_effect).add_suffix("_dependency_strength")
    common = scores.index.intersection(dependency.index).intersection(hnscc_ids)
    if len(common) != EXPECTED_DEPMAP_24Q2_HNSCC_MODELS:
        raise RuntimeError(
            f"DepMap Public 24Q2: expected {EXPECTED_DEPMAP_24Q2_HNSCC_MODELS} "
            f"HNSCC models with expression and CRISPR data, found {len(common)}."
        )
    analysis = scores.loc[common].join(dependency.loc[common])
    meta_columns = [
        c for c in ["CellLineName", "CCLEName", "OncotreeSubtype"] if c in metadata
    ]
    analysis = analysis.join(metadata.loc[common, meta_columns])

    tests = correlations(analysis)
    loo = leave_one_model_out(analysis, tests)
    args.out.mkdir(parents=True, exist_ok=True)
    panel_intervals(analysis).to_csv(args.out / "dependency_intervals.csv", index=False)
    quartile_contrasts(analysis).to_csv(
        args.out / "dependency_quartiles.csv", index=False
    )
    coverage.to_csv(args.out / "signature_gene_coverage.csv", index=False)
    analysis.to_csv(
        args.out / "hnscc_model_scores_and_dependencies.csv", index_label="ModelID"
    )
    tests.to_csv(args.out / "signature_dependency_correlations.csv", index=False)
    loo.to_csv(args.out / "leave_one_model_out_correlations.csv", index=False)


if __name__ == "__main__":
    main()
