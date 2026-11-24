"""Estimate the four primary post-treatment P1/immune correlations and leave-one-patient-out sensitivity."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED = 301741

PRIMARY_EXPOSURES = ["P1_fraction", "P1_usage_mean"]

PRIMARY_OUTCOMES = ["tnk_activation_core_score", "macrophage_inflammatory_score"]


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    out = np.full(p.size, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(out, index=p_values.index)
    order = np.where(valid)[0][np.argsort(p[valid])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return pd.Series(out, index=p_values.index)


def spearman_bootstrap_permutation(
    x: pd.Series, y: pd.Series, seed: int, n_boot: int = 2_000, n_perm: int = 20_000
) -> dict:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    x_values, y_values = paired["x"].to_numpy(float), paired["y"].to_numpy(float)
    n = len(paired)
    rho, p_asymptotic = stats.spearmanr(x_values, y_values)
    rng = np.random.default_rng(seed)
    if n_boot:
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            indices = rng.integers(0, n, n)
            boot[i] = stats.spearmanr(x_values[indices], y_values[indices]).statistic
        ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    if n_perm:
        rank_x = stats.rankdata(x_values)
        rank_y = stats.rankdata(y_values)
        z_x = (rank_x - rank_x.mean()) / rank_x.std(ddof=1)
        z_y = (rank_y - rank_y.mean()) / rank_y.std(ddof=1)
        exceed = 0
        for start in range(0, n_perm, 5_000):
            batch = min(5_000, n_perm - start)
            indices = np.array([rng.permutation(n) for _ in range(batch)])
            permuted_rho = (z_x @ z_y[indices].T) / (n - 1)
            exceed += int((np.abs(permuted_rho) >= abs(rho) - 1e-12).sum())
        p_permutation = float((exceed + 1) / (n_perm + 1))
    else:
        p_permutation = np.nan
    return {
        "n": n,
        "rho": float(rho),
        "p_asymptotic": float(p_asymptotic),
        "p_permutation": p_permutation,
        "ci95_boot_low": float(ci_low),
        "ci95_boot_high": float(ci_high),
    }


def correlations(patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, loo_rows = [], []
    analyses = [
        ("primary_all", patient),
        ("sensitivity_malignant_ge50", patient.loc[patient["malignant_cells"] >= 50]),
    ]
    for analysis_name, data in analyses:
        for exposure in PRIMARY_EXPOSURES:
            for outcome in PRIMARY_OUTCOMES + ["immune_dampened_summary_index"]:
                result = spearman_bootstrap_permutation(
                    data[exposure], data[outcome], SEED + len(rows)
                )
                rows.append(
                    {
                        "analysis": analysis_name,
                        "exposure": exposure,
                        "outcome": outcome,
                        **result,
                    }
                )
                if analysis_name == "primary_all" and outcome in PRIMARY_OUTCOMES:
                    for omitted in data["Patient"]:
                        loo = data.loc[data["Patient"] != omitted]
                        loo_result = spearman_bootstrap_permutation(
                            loo[exposure],
                            loo[outcome],
                            SEED + len(loo_rows),
                            n_boot=0,
                            n_perm=0,
                        )
                        loo_rows.append(
                            {
                                "exposure": exposure,
                                "outcome": outcome,
                                "omitted_patient": omitted,
                                **loo_result,
                            }
                        )
    summary = pd.DataFrame(rows)
    primary_mask = (
        summary["analysis"].eq("primary_all")
        & summary["outcome"].isin(PRIMARY_OUTCOMES)
        & summary["exposure"].isin(PRIMARY_EXPOSURES)
    )
    summary["p_bh_primary_family"] = np.nan
    summary.loc[primary_mask, "p_bh_primary_family"] = bh_fdr(
        summary.loc[primary_mask, "p_permutation"]
    )
    summary["inference_role"] = np.where(
        primary_mask,
        "primary family: P1 fraction and P1 usage across two pre-specified TME transcriptional outcomes",
        "secondary/descriptive",
    )
    loo = pd.DataFrame(loo_rows)
    loo_summary = (
        loo.groupby(["exposure", "outcome"], observed=True)
        .agg(
            n_loo=("rho", "size"),
            n_negative=("rho", lambda values: int((values < 0).sum())),
            n_positive=("rho", lambda values: int((values > 0).sum())),
            rho_min=("rho", "min"),
            rho_max=("rho", "max"),
        )
        .reset_index()
    )
    return summary, loo, loo_summary


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the four primary post-treatment P1/immune correlations and leave-one-patient-out sensitivity."
    )
    parser.add_argument(
        "--patients",
        required=True,
        type=Path,
        help="Post-treatment patient CSV with P1_fraction, P1_usage_mean, malignant_cells, TNK and myeloid scores, immune_dampened_summary_index and Patient",
    )
    parser.add_argument("--out", required=True, type=Path, help="Directory for results")
    args = parser.parse_args()
    patients = pd.read_csv(args.patients)
    if not patients.Patient.is_unique:
        raise ValueError("One row per patient required.")
    summary, loo, loo_summary = correlations(patients)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "pooled_correlations.csv", index=False)
    loo.to_csv(args.out / "patient_omission.csv", index=False)
    loo_summary.to_csv(args.out / "patient_omission_summary.csv", index=False)


if __name__ == "__main__":
    main()
