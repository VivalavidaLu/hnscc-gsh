"""Patient-level audit of GSE159067 redox-CXCL10 associations for Result 4."""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests


EXPR_PATH = DATA_ROOT / "raw" / "GSE159067_IHN_log2cpm_data.txt.gz"
CLIN_PATH = DATA_ROOT / "processed" / "gse159067_sample_metadata.csv"
OUT_DIR = RESULTS_ROOT / "external_validation" / "gse159067"
GSH_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - values.mean()) / values.std(ddof=0)


def score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    available = [gene for gene in genes if gene in expr.index]
    return pd.concat([zscore(expr.loc[gene]) for gene in available], axis=1).mean(axis=1)


def bootstrap_spearman(x: pd.Series, y: pd.Series, seed: int = 301741, n_boot: int = 10000) -> tuple[float, float]:
    pair = pd.concat([x, y], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sampled = pair[rng.integers(0, len(pair), len(pair))]
        estimates[i] = stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def partial_spearman(data: pd.DataFrame, x: str, y: str, covariates: list[str]) -> dict[str, float | int | str]:
    columns = [x, y, *covariates]
    clean = data[columns].dropna().copy()
    ranked = clean[[x, y]].rank(method="average")
    design_parts = []
    for covariate in covariates:
        if clean[covariate].dtype == object or str(clean[covariate].dtype).startswith("category"):
            design_parts.append(pd.get_dummies(clean[covariate].astype(str), prefix=covariate, drop_first=True, dtype=float))
        else:
            design_parts.append(clean[[covariate]].rank(method="average"))
    design = add_constant(pd.concat(design_parts, axis=1), has_constant="add")
    rx = OLS(ranked[x].astype(float), design.astype(float)).fit().resid
    ry = OLS(ranked[y].astype(float), design.astype(float)).fit().resid
    rho = stats.pearsonr(rx, ry).statistic
    design_rank = int(np.linalg.matrix_rank(design.astype(float).to_numpy()))
    degrees_of_freedom = len(clean) - design_rank - 1
    if degrees_of_freedom <= 0:
        raise RuntimeError("Insufficient residual degrees of freedom for partial correlation")
    t_statistic = rho * np.sqrt(degrees_of_freedom / (1 - rho**2))
    p_value = 2 * stats.t.sf(abs(t_statistic), df=degrees_of_freedom)
    return {
        "exposure": x,
        "outcome": y,
        "covariates": "+".join(covariates),
        "n": len(clean),
        "design_rank_including_intercept": design_rank,
        "residual_df": degrees_of_freedom,
        "partial_rho": rho,
        "p_value": p_value,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(EXPR_PATH, sep="\t", compression="infer", index_col="ID_REF", low_memory=False)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    if expr.columns.duplicated().any():
        raise RuntimeError("Duplicate expression identifiers detected")
    if expr.shape[1] != 102:
        raise RuntimeError(f"Expected 102 expression samples, found {expr.shape[1]}")
    common = expr.columns
    data = pd.DataFrame(index=common)
    clinical = None
    if CLIN_PATH.exists():
        clinical = pd.read_csv(CLIN_PATH, index_col=0)
        if clinical.index.duplicated().any():
            raise RuntimeError("Duplicate metadata identifiers detected")
        if set(common) != set(clinical.index):
            raise RuntimeError("Expression and metadata sample identifiers do not match")
        data = clinical.loc[common].copy()
    for gene in ["SLC7A11", "GCLC", "CXCL9", "CXCL10"]:
        data[gene] = expr.loc[gene, common]
    data["GSH_cystine"] = score(expr, GSH_GENES)
    data["SLC7A11_GCLC"] = score(expr, ["SLC7A11", "GCLC"])
    data["CXCL9_10"] = score(expr, ["CXCL9", "CXCL10"])

    exposures = ["SLC7A11", "GCLC", "SLC7A11_GCLC", "GSH_cystine"]
    outcomes = ["CXCL10", "CXCL9_10"]
    rows = []
    for exposure in exposures:
        for outcome in outcomes:
            pair = data[[exposure, outcome]].dropna()
            result = stats.spearmanr(pair[exposure], pair[outcome])
            ci_low, ci_high = bootstrap_spearman(pair[exposure], pair[outcome])
            rows.append({
                "exposure": exposure,
                "outcome": outcome,
                "n": len(pair),
                "rho": result.statistic,
                "p_value": result.pvalue,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
            })
    correlations = pd.DataFrame(rows)
    correlations["BH_FDR_8_tests"] = multipletests(correlations["p_value"], method="fdr_bh")[1]

    sensitivity_rows = []
    if clinical is not None:
        required = {"HPV_status", "site", "Sex", "immunotherapy line", "Age", "hot phenotype"}
        missing_metadata = required.difference(data.columns)
        if missing_metadata:
            raise RuntimeError(f"Missing metadata columns: {sorted(missing_metadata)}")
        data["HPV_status_cat"] = data["HPV_status"].astype(str)
        data["site_cat"] = data["site"].astype(str)
        data["sex_cat"] = data["Sex"].astype(str)
        data["line_rank"] = pd.to_numeric(data["immunotherapy line"], errors="coerce")
        data["age_rank"] = pd.to_numeric(data["Age"], errors="coerce")
        data["hot_score_numeric"] = pd.to_numeric(data["hot phenotype"], errors="coerce")
        for exposure in ["GSH_cystine", "SLC7A11_GCLC"]:
            sensitivity_rows.append(partial_spearman(data, exposure, "CXCL10", ["HPV_status_cat", "site_cat"]))
            sensitivity_rows.append(partial_spearman(
                data,
                exposure,
                "CXCL10",
                ["HPV_status_cat", "site_cat", "age_rank", "sex_cat", "line_rank"],
            ))
            sensitivity_rows.append(partial_spearman(data, exposure, "CXCL10", ["hot_score_numeric"]))
    sensitivity = pd.DataFrame(sensitivity_rows)
    if not sensitivity.empty:
        sensitivity["BH_FDR_6_sensitivity_tests"] = multipletests(sensitivity["p_value"], method="fdr_bh")[1]

    qc = pd.DataFrame([
        {
            "expression_rows": expr.shape[0],
            "expression_samples": expr.shape[1],
            "metadata_rows": clinical.shape[0] if clinical is not None else 0,
            "matched_unique_samples": len(common),
            "gsh_genes_present": ";".join(g for g in GSH_GENES if g in expr.index),
            "gsh_genes_missing": ";".join(g for g in GSH_GENES if g not in expr.index),
        }
    ])
    correlations.to_csv(OUT_DIR / "GSE159067_direct_correlations.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "GSE159067_partial_spearman_sensitivity.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT_DIR / "GSE159067_qc.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUT_DIR / "GSE159067_patient_level_scores.csv", index_label="sample", encoding="utf-8-sig")
    print(qc.to_string(index=False))
    print("\nDirect correlations\n", correlations.to_string(index=False))
    if not sensitivity.empty:
        print("\nSensitivity models\n", sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
