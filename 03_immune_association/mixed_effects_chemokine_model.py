#!/usr/bin/env python
from __future__ import annotations

import os

import json
import warnings
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.multitest import multipletests


RAW_H5AD = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
CELL_SCORES = DATA_ROOT / "processed" / "cell_level_malignant_ifn_apm_scores.csv"
OUT = RESULTS_ROOT / "immune" / "mixed_effects"
TAB = OUT / "tables"
FIG = OUT / "figures"
LOG = OUT / "logs"
for d in [OUT, TAB, FIG, LOG]:
    d.mkdir(parents=True, exist_ok=True)


GENESETS = {
    "S_phase": [
        "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
        "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1",
        "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
        "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
        "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
    ],
    "G2M_phase": [
        "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
        "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2",
        "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
        "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
        "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1",
        "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5",
        "CENPA",
    ],
    "EMT": [
        "VIM", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "TGFBI", "ITGA5",
        "ITGB1", "COL1A1", "COL1A2", "COL3A1", "MMP2", "MMP9", "SPARC", "SERPINE1",
        "CDH2", "LUM", "FAP", "ACTA2", "TAGLN",
    ],
    "hypoxia": [
        "HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "ENO1", "PGK1", "ALDOA", "BNIP3",
        "NDRG1", "ADM", "ANGPTL4", "EGLN3", "P4HA1", "P4HA2", "PDK1", "HK2", "GPI",
        "MIF", "SERPINE1",
    ],
    "stemness_proxy": [
        "SOX2", "KLF4", "MYC", "NANOG", "POU5F1", "PROM1", "ALDH1A1", "ALDH1A3",
        "EPCAM", "KRT5", "KRT14", "TP63", "ITGA6", "CD44", "BMI1", "LGR5", "AXL",
    ],
}

READOUTS = [
    "IFN_gamma_response_score",
    "JAK_STAT_score",
    "CXCL9_10_module_score",
    "combined_IFN_APM_score",
]
EXPOSURE = "GSH_cystine_score"
COVARIATES = [
    "P1_fraction_patient_phase",
    "S_phase_score",
    "G2M_phase_score",
    "stemness_proxy_score",
    "EMT_score",
    "hypoxia_score",
]


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def zscore_series(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (v - v.mean()) / sd


def bh(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.sum():
        out.loc[ok] = multipletests(p.loc[ok], method="fdr_bh")[1]
    return out


def rank_residualize(y: pd.Series, cov: pd.DataFrame) -> pd.Series:
    frame = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), cov], axis=1)
    frame = frame.dropna()
    out = pd.Series(np.nan, index=y.index)
    if frame.shape[0] < cov.shape[1] + 3:
        return out
    ranked = frame.rank(axis=0)
    x = np.column_stack([np.ones(ranked.shape[0]), ranked.drop(columns="y").to_numpy(float)])
    beta, *_ = np.linalg.lstsq(x, ranked["y"].to_numpy(float), rcond=None)
    out.loc[ranked.index] = ranked["y"].to_numpy(float) - x @ beta
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> tuple[float, float, int]:
    sub = df[[x, y] + covariates].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] < len(covariates) + 5:
        return np.nan, np.nan, int(sub.shape[0])
    rx = rank_residualize(sub[x], sub[covariates])
    ry = rank_residualize(sub[y], sub[covariates])
    ok = rx.notna() & ry.notna()
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(rx.loc[ok], ry.loc[ok])
    return float(rho), float(p), int(ok.sum())


def score_extra_modules(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ad.read_h5ad(RAW_H5AD)
    ids = pd.Index(cell_df["cell_id"].astype(str))
    present_cells = ids.intersection(raw.obs_names.astype(str))
    if len(present_cells) != len(ids):
        missing = ids.difference(raw.obs_names.astype(str))
        pd.Series(missing, name="cell_id").to_csv(TAB / "missing_cells_from_raw_for_covariate_scores.csv", index=False)
    genes = sorted({g for gs in GENESETS.values() for g in gs if g in raw.var_names})
    x = dense(raw[present_cells.tolist(), genes].X).astype(np.float32)
    counts = pd.to_numeric(raw.obs.loc[present_cells, "nCount_RNA"], errors="coerce").to_numpy(np.float32)
    counts[counts <= 0] = np.nan
    expr = pd.DataFrame(np.log1p((x / counts[:, None]) * 10000.0), index=present_cells, columns=genes)
    zexpr = expr.apply(zscore_series, axis=0)

    scored = cell_df.set_index("cell_id").loc[present_cells].copy()
    coverage = []
    for name, geneset in GENESETS.items():
        present = [g for g in geneset if g in zexpr.columns]
        scored[f"{name}_score"] = zexpr[present].mean(axis=1) if present else np.nan
        coverage.append(
            {
                "module": name,
                "n_defined": len(geneset),
                "n_present": len(present),
                "coverage_fraction": len(present) / len(geneset),
                "genes_present": ";".join(present),
                "genes_missing": ";".join([g for g in geneset if g not in present]),
                "score_definition": "mean z-scored CP10k-log1p expression; stemness is a curated module proxy, not CytoTRACE",
            }
        )
    raw.file.close()
    scored = scored.reset_index()
    return scored, pd.DataFrame(coverage)


def prepare_analysis_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = pd.read_csv(CELL_SCORES)
    cell["cell_id"] = cell["cell_id"].astype(str)
    cell["Phase"] = cell["Phase"].astype(str).str.lower()
    cell["Patient_Phase"] = cell["Patient"].astype(str) + "_" + cell["Phase"].astype(str)
    cell, coverage = score_extra_modules(cell)
    coverage.to_csv(TAB / "covariate_module_gene_coverage.csv", index=False, encoding="utf-8-sig")

    p1_fraction = (
        cell.groupby(["Patient", "Phase"], observed=True)["is_P1"]
        .mean()
        .rename("P1_fraction_patient_phase")
        .reset_index()
    )
    cell = cell.merge(p1_fraction, on=["Patient", "Phase"], how="left")
    p1 = cell[cell["is_P1"].astype(bool)].copy()

    agg_cols = [EXPOSURE, "AA_transport_score"] + READOUTS + COVARIATES
    pp = (
        p1.groupby(["Patient", "Phase", "Patient_Phase"], observed=True)
        .agg(
            n_P1_cells=("cell_id", "size"),
            **{c: (c, "mean") for c in agg_cols if c in p1.columns},
        )
        .reset_index()
    )
    phase_dummies = pd.get_dummies(pp["Phase"], prefix="phase", drop_first=True, dtype=float)
    pp = pd.concat([pp, phase_dummies], axis=1)

    cell.to_csv(TAB / "cell_level_scores_with_covariates.csv", index=False, encoding="utf-8-sig")
    pp.to_csv(TAB / "P1_patient_phase_scores_with_covariates.csv", index=False, encoding="utf-8-sig")
    return p1, pp


def run_mixed_models(p1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_df = p1[[EXPOSURE, "Patient", "Phase"] + READOUTS + COVARIATES].copy()
    model_df["phase_post"] = (model_df["Phase"].astype(str) == "post").astype(float)
    fixed_cov = ["phase_post"] + COVARIATES
    for y in READOUTS:
        cols = [y, EXPOSURE, "Patient"] + fixed_cov
        d = model_df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name != "Patient" else s).dropna()
        for c in [y, EXPOSURE] + fixed_cov:
            d[f"z_{c}"] = zscore_series(d[c])
        formula = f"z_{y} ~ z_{EXPOSURE} + " + " + ".join([f"z_{c}" for c in fixed_cov])
        record = {"analysis": "cell_level_mixedlm_patient_random_intercept", "readout": y, "n": int(d.shape[0]), "n_patients": int(d["Patient"].nunique())}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = mixedlm(formula, d, groups=d["Patient"]).fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            term = f"z_{EXPOSURE}"
            record.update(
                beta=float(fit.params.get(term, np.nan)),
                se=float(fit.bse.get(term, np.nan)),
                p_value=float(fit.pvalues.get(term, np.nan)),
                converged=bool(getattr(fit, "converged", False)),
                model_status="ok",
            )
        except Exception as exc:
            record.update(beta=np.nan, se=np.nan, p_value=np.nan, converged=False, model_status=f"failed: {type(exc).__name__}: {exc}")
        rows.append(record)
    out = pd.DataFrame(rows)
    out["fdr"] = bh(out["p_value"])
    out.to_csv(TAB / "cell_level_mixed_effect_models.csv", index=False, encoding="utf-8-sig")
    return out


def run_patient_phase_models(pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_covs = [c for c in pp.columns if c.startswith("phase_")]
    covs = phase_covs + COVARIATES
    rows = []
    corr_rows = []
    for y in READOUTS:
        cols = [y, EXPOSURE] + covs
        d = pp[cols].apply(pd.to_numeric, errors="coerce").dropna()
        record = {"analysis": "patient_phase_ols_hc3", "readout": y, "n": int(d.shape[0]), "n_patients": int(pp.loc[d.index, "Patient"].nunique())}
        if d.shape[0] >= len(covs) + 4:
            zd = d.copy()
            for c in cols:
                zd[f"z_{c}"] = zscore_series(zd[c])
            formula = f"z_{y} ~ z_{EXPOSURE}"
            for c in covs:
                formula += f" + z_{c}"
            try:
                fit = ols(formula, data=zd).fit(cov_type="HC3")
                term = f"z_{EXPOSURE}"
                record.update(beta=float(fit.params.get(term, np.nan)), se=float(fit.bse.get(term, np.nan)), p_value=float(fit.pvalues.get(term, np.nan)), model_status="ok")
            except Exception as exc:
                record.update(beta=np.nan, se=np.nan, p_value=np.nan, model_status=f"failed: {type(exc).__name__}: {exc}")
        else:
            record.update(beta=np.nan, se=np.nan, p_value=np.nan, model_status="too_few_patient_phase_units_for_full_covariate_model")
        rows.append(record)

        rho, p, n = partial_spearman(pp, EXPOSURE, y, covs)
        corr_rows.append({"analysis": "patient_phase_partial_spearman", "readout": y, "rho": rho, "p_value": p, "n": n, "covariates": ";".join(covs)})

        simple = pp[[EXPOSURE, y]].dropna()
        if simple.shape[0] >= 4:
            srho, sp = stats.spearmanr(simple[EXPOSURE], simple[y])
        else:
            srho, sp = np.nan, np.nan
        corr_rows.append({"analysis": "patient_phase_simple_spearman", "readout": y, "rho": float(srho), "p_value": float(sp), "n": int(simple.shape[0]), "covariates": ""})

    model = pd.DataFrame(rows)
    model["fdr"] = bh(model["p_value"])
    corr = pd.DataFrame(corr_rows)
    corr["fdr"] = corr.groupby("analysis")["p_value"].transform(bh)
    model.to_csv(TAB / "patient_phase_ols_hc3_models.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(TAB / "patient_phase_spearman_and_partial_spearman.csv", index=False, encoding="utf-8-sig")
    return model, corr


def make_figures(p1: pd.DataFrame, pp: pd.DataFrame, mixed: pd.DataFrame, ppm: pd.DataFrame, corr: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="paper", font="Arial")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.linewidth": 0.7})

    fig = plt.figure(figsize=(12.5, 8.6), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.05, 1.05], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:])

    scatter_y = "JAK_STAT_score"
    sns.regplot(data=pp, x=EXPOSURE, y=scatter_y, ax=ax_a, scatter_kws={"s": 45, "alpha": 0.85, "edgecolor": "white", "linewidths": 0.4}, line_kws={"color": "#B8323B", "lw": 1.6}, color="#2C7A7B")
    for _, r in pp.iterrows():
        ax_a.text(r[EXPOSURE], r[scatter_y], str(r["Patient"]).replace("OCSCC", "O").replace("LSCC", "L").replace("OPSCC", "OP"), fontsize=5.8, alpha=0.75)
    ax_a.set_title("P1 patient-phase association")
    ax_a.set_xlabel("P1 GSH/cystine score")
    ax_a.set_ylabel("P1 JAK/STAT score")
    ax_a.text(-0.12, 1.08, "A", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

    simple = corr[corr["analysis"].eq("patient_phase_simple_spearman")].copy()
    part = corr[corr["analysis"].eq("patient_phase_partial_spearman")].copy()
    heat = pd.concat([
        simple.assign(model="simple"),
        part.assign(model="adjusted"),
    ])
    mat = heat.pivot(index="readout", columns="model", values="rho").reindex(READOUTS)
    sns.heatmap(mat, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Spearman rho"}, ax=ax_b)
    ax_b.set_title("Patient-phase correlations")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.text(-0.22, 1.08, "B", transform=ax_b.transAxes, fontsize=12, fontweight="bold")

    forest = mixed.set_index("readout").reindex(READOUTS).reset_index()
    y = np.arange(len(forest))
    ax_c.axvline(0, color="#59636E", lw=0.8)
    ax_c.errorbar(forest["beta"], y, xerr=1.96 * forest["se"], fmt="o", color="#B8323B", ecolor="#7A8793", elinewidth=1.2, capsize=2)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([x.replace("_score", "").replace("_", " ") for x in forest["readout"]])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Mixed model beta for GSH/cystine")
    ax_c.set_title("Cell-level mixed models")
    ax_c.text(-0.22, 1.08, "C", transform=ax_c.transAxes, fontsize=12, fontweight="bold")

    cov_mat = pp[[EXPOSURE] + COVARIATES + READOUTS].corr(method="spearman")
    keep = [EXPOSURE] + COVARIATES
    sns.heatmap(cov_mat.loc[keep, READOUTS], cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.35, linecolor="white", cbar=False, ax=ax_d)
    ax_d.set_title("Covariate audit")
    ax_d.set_xlabel("immune readout")
    ax_d.set_ylabel("exposure/covariate")
    ax_d.text(-0.22, 1.08, "D", transform=ax_d.transAxes, fontsize=12, fontweight="bold")

    pp_plot = ppm.set_index("readout").reindex(READOUTS).reset_index()
    ax_e.axvline(0, color="#59636E", lw=0.8)
    y2 = np.arange(len(pp_plot))
    ax_e.errorbar(pp_plot["beta"], y2, xerr=1.96 * pp_plot["se"], fmt="o", color="#2C7A7B", ecolor="#7A8793", elinewidth=1.2, capsize=2)
    ax_e.set_yticks(y2)
    ax_e.set_yticklabels([x.replace("_score", "").replace("_", " ") for x in pp_plot["readout"]])
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Patient-phase OLS beta for GSH/cystine (HC3 SE)")
    ax_e.set_title("Patient-phase adjusted model")
    ax_e.text(-0.08, 1.08, "E", transform=ax_e.transAxes, fontsize=12, fontweight="bold")
    sns.despine(fig=fig)
    fig.savefig(FIG / "Fig_P1_GSH_IFN_mixed_effects_audit.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig_P1_GSH_IFN_mixed_effects_audit.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_report(p1: pd.DataFrame, pp: pd.DataFrame, mixed: pd.DataFrame, ppm: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = [
        "# P1-GSH versus IFN/JAK/CXCL mixed-effect audit",
        "",
        "## Scope",
        "",
        "This analysis tests whether the P1-internal GSH/cystine score remains associated with IFN/JAK/CXCL readouts after accounting for patient structure and major covariates. Stemness is represented by a curated module proxy, not CytoTRACE.",
        "",
        "## Data used",
        "",
        f"- P1 cells: {p1.shape[0]}",
        f"- Patients with P1 cells: {p1['Patient'].nunique()}",
        f"- Patient-phase units: {pp.shape[0]}",
        "",
        "## Cell-level mixed model",
        "",
        mixed.to_csv(index=False).strip(),
        "",
        "## Patient-phase adjusted OLS",
        "",
        ppm.to_csv(index=False).strip(),
        "",
        "## Patient-phase Spearman audit",
        "",
        corr.to_csv(index=False).strip(),
        "",
        "## Interpretation boundary",
        "",
        "Negative adjusted coefficients support an association between higher P1 GSH/cystine state and lower IFN/JAK/CXCL response. These models reduce patient/covariate confounding but remain observational and do not prove causality.",
    ]
    (OUT / "P1_GSH_IFN_mixed_effects_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p1, pp = prepare_analysis_tables()
    mixed = run_mixed_models(p1)
    ppm, corr = run_patient_phase_models(pp)
    make_figures(p1, pp, mixed, ppm, corr)
    write_report(p1, pp, mixed, ppm, corr)
    meta = {
        "n_P1_cells": int(p1.shape[0]),
        "n_patients": int(p1["Patient"].nunique()),
        "n_patient_phase_units": int(pp.shape[0]),
        "readouts": READOUTS,
        "exposure": EXPOSURE,
        "covariates": COVARIATES,
        "stemness_boundary": "curated stemness module proxy, not CytoTRACE",
    }
    (LOG / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
