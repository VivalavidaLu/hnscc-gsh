"""Fit P1 cell mixed models and patient-phase chemokine sensitivity analyses."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
import warnings

GENES = ["CXCL9", "CXCL10"]

EXPOSURE = "GSH_cystine_score"

MIN_P1_CELLS = 10

N_BOOT = 5000

SEED = 20260717


def bh(p: pd.Series) -> pd.Series:
    values = pd.to_numeric(p, errors="coerce").to_numpy(float)
    out = np.full(len(values), np.nan)
    good = np.isfinite(values)
    index = np.flatnonzero(good)[np.argsort(values[good])]
    adjusted = values[index] * len(index) / np.arange(1, len(index) + 1)
    out[index] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return pd.Series(out, index=p.index)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (
        (s - s.mean()) / sd
        if np.isfinite(sd) and sd > 0
        else pd.Series(np.nan, index=s.index)
    )


def spearman(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    d = frame[[x, y]].dropna()
    rho, p = stats.spearmanr(d[x], d[y])
    return float(rho), float(p), int(len(d))


def cluster_bootstrap(
    frame: pd.DataFrame, x: str, y: str
) -> tuple[float, float, float, float]:
    d = frame[["Patient", x, y]].dropna().reset_index(drop=True)
    patients = d["Patient"].drop_duplicates().tolist()
    group_indices = [
        np.flatnonzero(d["Patient"].to_numpy() == patient) for patient in patients
    ]
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    rng = np.random.default_rng(SEED)
    values = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([group_indices[j] for j in draws])
        values[i] = np.corrcoef(stats.rankdata(xv[idx]), stats.rankdata(yv[idx]))[0, 1]
    values = values[np.isfinite(values)]
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.5)),
        float(np.quantile(values, 0.975)),
        float((values < 0).mean()),
    )


def mixed_gene_models(p1: pd.DataFrame) -> pd.DataFrame:
    covariates = [
        "phase_post",
        "P1_fraction_patient_phase",
        "S_phase_score",
        "G2M_phase_score",
        "stemness_proxy_score",
        "EMT_score",
        "hypoxia_score",
    ]
    rows = []
    for gene in GENES:
        outcome = f"{gene}_z"
        d = p1[[outcome, EXPOSURE, "Patient"] + covariates].dropna().copy()
        for column in [outcome, EXPOSURE] + covariates:
            d[f"z_{column}"] = zscore(d[column])
        formula = f"z_{outcome} ~ z_{EXPOSURE} + " + " + ".join(
            f"z_{column}" for column in covariates
        )
        record = {
            "gene": gene,
            "n_cells": len(d),
            "n_patients": d["Patient"].nunique(),
            "formula": formula,
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = mixedlm(formula, d, groups=d["Patient"]).fit(
                    reml=False, method="lbfgs", maxiter=300, disp=False
                )
            record.update(
                beta=float(fit.params[f"z_{EXPOSURE}"]),
                se=float(fit.bse[f"z_{EXPOSURE}"]),
                p_value=float(fit.pvalues[f"z_{EXPOSURE}"]),
                converged=bool(fit.converged),
                status="ok",
            )
        except Exception as exc:
            record.update(
                beta=np.nan,
                se=np.nan,
                p_value=np.nan,
                converged=False,
                status=f"failed: {type(exc).__name__}",
            )
        rows.append(record)
    out = pd.DataFrame(rows)
    out["fdr"] = bh(out["p_value"])
    out["ci_low"] = out["beta"] - 1.96 * out["se"]
    out["ci_high"] = out["beta"] + 1.96 * out["se"]
    return out


def pseudobulk(p1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregations = {"n_P1_cells": ("cell_id", "size"), EXPOSURE: (EXPOSURE, "mean")}
    for gene in GENES:
        aggregations[f"{gene}_mean_logCP10K"] = (f"{gene}_logCP10K", "mean")
        aggregations[f"{gene}_detection_fraction"] = (f"{gene}_detected", "mean")
    pp = (
        p1.groupby(["Patient", "Phase", "Patient_Phase"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    primary = pp.loc[pp["n_P1_cells"].ge(MIN_P1_CELLS)].copy()
    rows = []
    for gene in GENES:
        for measure in [f"{gene}_mean_logCP10K", f"{gene}_detection_fraction"]:
            rho, p, n = spearman(primary, EXPOSURE, measure)
            low, med, high, negative = cluster_bootstrap(primary, EXPOSURE, measure)
            rows.append(
                {
                    "gene": gene,
                    "measure": measure,
                    "rho": rho,
                    "nominal_p": p,
                    "n_patient_phase": n,
                    "n_patients": primary["Patient"].nunique(),
                    "cluster_ci95_low": low,
                    "cluster_median": med,
                    "cluster_ci95_high": high,
                    "bootstrap_fraction_negative": negative,
                }
            )
    loo_rows = []
    for gene in GENES:
        measure = f"{gene}_mean_logCP10K"
        full_rho, full_p, _ = spearman(primary, EXPOSURE, measure)
        for patient in sorted(primary["Patient"].unique()):
            rho, p, n = spearman(
                primary.loc[~primary["Patient"].eq(patient)], EXPOSURE, measure
            )
            loo_rows.append(
                {
                    "gene": gene,
                    "left_out_patient": patient,
                    "rho": rho,
                    "nominal_p": p,
                    "n_patient_phase": n,
                    "full_rho": full_rho,
                    "full_nominal_p": full_p,
                }
            )
    return pp, pd.DataFrame(rows), pd.DataFrame(loo_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fit P1 cell mixed models and patient-phase chemokine sensitivity analyses."
    )
    parser.add_argument(
        "--cells",
        required=True,
        type=Path,
        help="P1-only scored cells CSV with mixed-model covariates, chemokine expression and detection fields (see function definitions)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    cells = pd.read_csv(args.cells)
    args.out.mkdir(parents=True, exist_ok=True)
    mixed_gene_models(cells).to_csv(
        args.out / "chemokine_mixed_models.csv", index=False
    )
    _, summary, loo = pseudobulk(cells)
    summary.to_csv(args.out / "chemokine_correlations.csv", index=False)
    loo.to_csv(args.out / "leave_one_patient_out.csv", index=False)


if __name__ == "__main__":
    main()
