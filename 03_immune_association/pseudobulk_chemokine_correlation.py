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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats
import statsmodels.api as sm


RAW_H5AD = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
P1_CELL_TABLE = DATA_ROOT / "processed" / "p1_cell_level_scores_with_key_genes.csv"
PSEUDOBULK_TABLE = DATA_ROOT / "processed" / "p1_patient_phase_cxcl9_cxcl10.csv"
OUT = RESULTS_ROOT / "immune" / "branch_specificity"
TAB = OUT / "tables"
FIG = OUT / "figures"
LOG = OUT / "logs"

PRIMARY_Y = "CXCL9_10_module_score"
PRIMARY_MIN_CELLS = 10
MIN_CELL_THRESHOLDS = [1, 5, 10, 20, 50]


BRANCH_GENESETS = {
    "GSH_cystine": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "SLC7A11_GCLC": ["SLC7A11", "GCLC"],
    "AA_transport": ["SLC1A5", "SLC7A5", "SLC38A1", "SLC38A2", "SLC1A4", "SLC7A8"],
    "NRF2_detox": ["NQO1", "AKR1C1", "AKR1C2", "AKR1B10", "ALDH3A1", "GSTA4", "GSTM1", "GSTP1", "HMOX1", "TXNRD1"],
    "thioredoxin_PRDX": ["TXN", "TXNRD1", "PRDX1", "PRDX2", "PRDX4", "PRDX6", "SRXN1"],
    "heme_iron": ["HMOX1", "FTH1", "FTL", "SLC40A1", "TFRC", "FPN1", "SLC48A1"],
    "hypoxia": ["HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "ENO1", "PGK1", "ALDOA", "BNIP3", "NDRG1", "ADM", "ANGPTL4", "EGLN3", "P4HA1", "P4HA2", "PDK1", "HK2", "GPI", "MIF", "SERPINE1"],
    "EMT": ["VIM", "FN1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "TGFBI", "ITGA5", "ITGB1", "COL1A1", "COL1A2", "COL3A1", "MMP2", "MMP9", "SPARC", "SERPINE1", "CDH2", "LUM", "FAP", "ACTA2", "TAGLN"],
    "cell_cycle": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5", "STMN1", "CENPF", "CCNB1", "CCNB2", "CDK1", "UBE2C"],
    "stemness": ["SOX2", "KLF4", "MYC", "NANOG", "POU5F1", "PROM1", "ALDH1A1", "ALDH1A3", "EPCAM", "KRT5", "KRT14", "TP63", "ITGA6", "CD44", "BMI1", "LGR5", "AXL"],
}

BRANCH_LABELS = {
    "GSH_cystine": "GSH/cystine",
    "SLC7A11_GCLC": "SLC7A11/GCLC",
    "AA_transport": "AA transport",
    "NRF2_detox": "NRF2 detox",
    "thioredoxin_PRDX": "Thioredoxin/PRDX",
    "heme_iron": "Heme/iron",
    "hypoxia": "Hypoxia",
    "EMT": "EMT",
    "cell_cycle": "Cell cycle",
    "stemness": "Stemness",
}

BRANCH_CLASS = {
    "GSH_cystine": "main_GSH_arm",
    "SLC7A11_GCLC": "main_GSH_arm",
    "AA_transport": "related_transport",
    "NRF2_detox": "alternative_NRF2_stress",
    "thioredoxin_PRDX": "alternative_NRF2_stress",
    "heme_iron": "alternative_NRF2_stress",
    "hypoxia": "known_confounder",
    "EMT": "known_confounder",
    "cell_cycle": "known_confounder",
    "stemness": "known_confounder",
}


def ensure_dirs() -> None:
    for d in [OUT, TAB, FIG, LOG]:
        d.mkdir(parents=True, exist_ok=True)


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (v - v.mean()) / sd


def bh(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(out, index=pvals.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranks = np.arange(1, len(order) + 1)
    vals = p[order] * len(order) / ranks
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    out[order] = np.clip(vals, 0, 1)
    return pd.Series(out, index=pvals.index)


def spearman_pair(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.shape[0] < 4 or d[x].nunique() < 3 or d[y].nunique() < 3:
        return np.nan, np.nan, int(d.shape[0])
    rho, p = stats.spearmanr(d[x], d[y])
    return float(rho), float(p), int(d.shape[0])


def rank_residualize(y: pd.Series, covars: pd.DataFrame) -> pd.Series:
    d = pd.concat([y.rename("y"), covars], axis=1).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if d.shape[0] < covars.shape[1] + 4:
        return out
    yr = d["y"].rank(method="average")
    xr = d.drop(columns=["y"]).apply(lambda c: pd.to_numeric(c, errors="coerce").rank(method="average"))
    xr = sm.add_constant(xr, has_constant="add")
    try:
        fit = sm.OLS(yr, xr).fit()
        out.loc[d.index] = fit.resid
    except Exception:
        pass
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> tuple[float, float, int]:
    cols = [x, y] + covariates
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if d.shape[0] < len(covariates) + 5:
        return np.nan, np.nan, int(d.shape[0])
    rx = rank_residualize(d[x], d[covariates])
    ry = rank_residualize(d[y], d[covariates])
    ok = rx.notna() & ry.notna()
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    rho, p = stats.spearmanr(rx.loc[ok], ry.loc[ok])
    return float(rho), float(p), int(ok.sum())


def ols_hc3(df: pd.DataFrame, x: str, y: str, covariates: list[str]) -> dict:
    cols = [x, y] + covariates
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    rec = {"n": int(d.shape[0]), "n_covariates": len(covariates)}
    if d.shape[0] < len(covariates) + 5:
        rec.update(beta=np.nan, se=np.nan, p_value=np.nan, adj_r2=np.nan, aic=np.nan, model_status="too_few_units")
        return rec
    dz = d.copy()
    for c in cols:
        dz[c] = zscore(dz[c])
    X = sm.add_constant(dz[[x] + covariates], has_constant="add")
    try:
        fit = sm.OLS(dz[y], X).fit(cov_type="HC3")
        rec.update(
            beta=float(fit.params.get(x, np.nan)),
            se=float(fit.bse.get(x, np.nan)),
            p_value=float(fit.pvalues.get(x, np.nan)),
            adj_r2=float(fit.rsquared_adj),
            aic=float(fit.aic),
            model_status="ok",
        )
    except Exception as exc:
        rec.update(beta=np.nan, se=np.nan, p_value=np.nan, adj_r2=np.nan, aic=np.nan, model_status=f"failed:{type(exc).__name__}:{exc}")
    return rec


def score_branch_modules() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = pd.read_csv(P1_CELL_TABLE)
    p1["cell_id"] = p1["cell_id"].astype(str)
    p1["Patient"] = p1["Patient"].astype(str)
    p1["Phase"] = p1["Phase"].astype(str).str.lower()
    p1["Patient_Phase"] = p1["Patient"] + "_" + p1["Phase"]

    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    ids = pd.Index(p1["cell_id"].astype(str))
    raw_ids = pd.Index(raw.obs_names.astype(str))
    present_cells = ids.intersection(raw_ids)
    if len(present_cells) != len(ids):
        pd.Series(ids.difference(raw_ids), name="cell_id").to_csv(TAB / "missing_cells_from_raw_h5ad.csv", index=False)

    all_genes = sorted({g for genes in BRANCH_GENESETS.values() for g in genes})
    present_genes = [g for g in all_genes if g in raw.var_names]
    x = dense(raw[present_cells.tolist(), present_genes].X).astype(np.float32)
    counts = pd.to_numeric(raw.obs.loc[present_cells, "nCount_RNA"], errors="coerce").to_numpy(np.float32)
    counts[counts <= 0] = np.nan
    expr = pd.DataFrame(np.log1p((x / counts[:, None]) * 10000.0), index=present_cells, columns=present_genes)
    raw.file.close()

    p1 = p1.set_index("cell_id").loc[present_cells].copy()
    zexpr = expr.apply(zscore, axis=0)
    coverage = []
    for branch, genes in BRANCH_GENESETS.items():
        present = [g for g in genes if g in expr.columns]
        p1[f"branch_{branch}_score"] = zexpr[present].mean(axis=1) if present else np.nan
        p1[f"branch_{branch}_mean_logCP10K"] = expr[present].mean(axis=1) if present else np.nan
        coverage.append(
            {
                "branch": branch,
                "branch_label": BRANCH_LABELS[branch],
                "branch_class": BRANCH_CLASS[branch],
                "n_genes_defined": len(genes),
                "n_genes_present": len(present),
                "coverage_fraction": len(present) / len(genes),
                "genes_present": ";".join(present),
                "genes_missing": ";".join([g for g in genes if g not in present]),
                "score_definition": "mean z-scored CP10K-log1p expression in P1 cells",
            }
        )
    p1 = p1.reset_index()
    p1.to_csv(TAB / "P1_cell_level_branch_specificity_scores.csv", index=False, encoding="utf-8-sig")
    cov = pd.DataFrame(coverage)
    cov.to_csv(TAB / "branch_gene_coverage.csv", index=False, encoding="utf-8-sig")
    return p1, cov


def make_pseudobulk(p1: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(PSEUDOBULK_TABLE)
    base["Patient"] = base["Patient"].astype(str)
    base["Phase"] = base["Phase"].astype(str).str.lower()
    branch_cols = []
    for branch in BRANCH_GENESETS:
        branch_cols.append(f"branch_{branch}_score")
        branch_cols.append(f"branch_{branch}_mean_logCP10K")
    agg_map = {c: (c, "mean") for c in branch_cols if c in p1.columns}
    branch_pp = p1.groupby(["Patient", "Phase", "Patient_Phase"], observed=True).agg(**agg_map).reset_index()
    pp = base.merge(branch_pp, on=["Patient", "Phase", "Patient_Phase"], how="left", validate="one_to_one")
    pp["phase_pre"] = (pp["Phase"] == "pre").astype(float)
    pp.to_csv(TAB / "P1_patient_phase_branch_specificity_scores.csv", index=False, encoding="utf-8-sig")
    return pp


def run_models(pp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    covariates_basic = ["phase_pre", "P1_fraction_patient_phase"]
    covariates_core = ["phase_pre", "P1_fraction_patient_phase", "cell_cycle_score", "stemness_proxy_score", "EMT_score", "hypoxia_score"]
    rows = []
    ols_rows = []
    loo_rows = []
    for min_cells in MIN_CELL_THRESHOLDS:
        d0 = pp[pp["n_P1_cells"] >= min_cells].copy()
        for branch in BRANCH_GENESETS:
            x = f"branch_{branch}_score"
            for analysis, covs in [
                ("simple_spearman", []),
                ("partial_spearman_phase_p1fraction", covariates_basic),
                ("partial_spearman_core_covariates", covariates_core),
            ]:
                if analysis == "simple_spearman":
                    rho, p, n = spearman_pair(d0, x, PRIMARY_Y)
                else:
                    rho, p, n = partial_spearman(d0, x, PRIMARY_Y, covs)
                rows.append(
                    {
                        "analysis": analysis,
                        "min_P1_cells": min_cells,
                        "branch": branch,
                        "branch_label": BRANCH_LABELS[branch],
                        "branch_class": BRANCH_CLASS[branch],
                        "exposure": x,
                        "readout": PRIMARY_Y,
                        "rho": rho,
                        "p_value": p,
                        "n": n,
                        "covariates": ";".join(covs),
                    }
                )
            for analysis, covs in [
                ("ols_hc3_none", []),
                ("ols_hc3_phase_p1fraction", covariates_basic),
                ("ols_hc3_core_covariates", covariates_core),
            ]:
                rec = ols_hc3(d0, x, PRIMARY_Y, covs)
                rec.update(
                    {
                        "analysis": analysis,
                        "min_P1_cells": min_cells,
                        "branch": branch,
                        "branch_label": BRANCH_LABELS[branch],
                        "branch_class": BRANCH_CLASS[branch],
                        "exposure": x,
                        "readout": PRIMARY_Y,
                        "covariates": ";".join(covs),
                    }
                )
                ols_rows.append(rec)

    corr = pd.DataFrame(rows)
    corr["fdr_within_analysis_min_cells"] = corr.groupby(["analysis", "min_P1_cells"])["p_value"].transform(bh)
    corr.to_csv(TAB / "branch_specificity_CXCL910_correlations.csv", index=False, encoding="utf-8-sig")

    ols = pd.DataFrame(ols_rows)
    ols["fdr_within_analysis_min_cells"] = ols.groupby(["analysis", "min_P1_cells"])["p_value"].transform(bh)
    ols.to_csv(TAB / "branch_specificity_CXCL910_ols_hc3.csv", index=False, encoding="utf-8-sig")

    main = pp[pp["n_P1_cells"] >= PRIMARY_MIN_CELLS].copy()
    for branch in BRANCH_GENESETS:
        x = f"branch_{branch}_score"
        full_rho, full_p, full_n = spearman_pair(main, x, PRIMARY_Y)
        for patient in sorted(main["Patient"].unique()):
            d = main[main["Patient"] != patient].copy()
            rho, p, n = spearman_pair(d, x, PRIMARY_Y)
            loo_rows.append(
                {
                    "branch": branch,
                    "branch_label": BRANCH_LABELS[branch],
                    "left_out_patient": patient,
                    "rho": rho,
                    "p_value": p,
                    "n": n,
                    "full_rho": full_rho,
                    "full_p_value": full_p,
                    "full_n": full_n,
                    "expected_negative_retained": bool(pd.notna(rho) and rho < 0),
                }
            )
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(TAB / "branch_specificity_CXCL910_leave_one_patient_out.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for branch in BRANCH_GENESETS:
        sub = corr[(corr["branch"] == branch) & (corr["min_P1_cells"] == PRIMARY_MIN_CELLS)]
        sub_loo = loo[loo["branch"] == branch]
        simple = sub[sub["analysis"] == "simple_spearman"].iloc[0]
        basic = sub[sub["analysis"] == "partial_spearman_phase_p1fraction"].iloc[0]
        core = sub[sub["analysis"] == "partial_spearman_core_covariates"].iloc[0]
        ols_sub = ols[(ols["branch"] == branch) & (ols["min_P1_cells"] == PRIMARY_MIN_CELLS)]
        ols_basic = ols_sub[ols_sub["analysis"] == "ols_hc3_phase_p1fraction"].iloc[0]
        neg_models = int((sub["rho"] < 0).sum() + (ols_sub["beta"] < 0).sum())
        valid_models = int(sub["rho"].notna().sum() + ols_sub["beta"].notna().sum())
        summary_rows.append(
            {
                "branch": branch,
                "branch_label": BRANCH_LABELS[branch],
                "branch_class": BRANCH_CLASS[branch],
                "simple_rho": simple["rho"],
                "simple_p": simple["p_value"],
                "partial_phase_p1fraction_rho": basic["rho"],
                "partial_phase_p1fraction_p": basic["p_value"],
                "partial_core_covariates_rho": core["rho"],
                "partial_core_covariates_p": core["p_value"],
                "ols_phase_p1fraction_beta": ols_basic["beta"],
                "ols_phase_p1fraction_p": ols_basic["p_value"],
                "negative_direction_models": neg_models,
                "valid_direction_models": valid_models,
                "direction_retention_fraction": neg_models / valid_models if valid_models else np.nan,
                "loo_negative_retained": int(sub_loo["expected_negative_retained"].sum()),
                "loo_total": int(sub_loo.shape[0]),
                "loo_retention_fraction": float(sub_loo["expected_negative_retained"].mean()),
                "branch_specificity_rank_score": (
                    (abs(simple["rho"]) if pd.notna(simple["rho"]) and simple["rho"] < 0 else 0)
                    + (abs(basic["rho"]) if pd.notna(basic["rho"]) and basic["rho"] < 0 else 0)
                    + (abs(ols_basic["beta"]) if pd.notna(ols_basic["beta"]) and ols_basic["beta"] < 0 else 0)
                    + float(sub_loo["expected_negative_retained"].mean())
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("branch_specificity_rank_score", ascending=False)
    summary.to_csv(TAB / "branch_specificity_CXCL910_summary_ranked.csv", index=False, encoding="utf-8-sig")
    return corr, ols, loo


def style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_panels(pp: pd.DataFrame, corr: pd.DataFrame, ols: pd.DataFrame, loo: pd.DataFrame) -> None:
    style()
    summary = pd.read_csv(TAB / "branch_specificity_CXCL910_summary_ranked.csv")
    order = summary["branch"].tolist()
    label_map = dict(zip(summary["branch"], summary["branch_label"]))

    main_corr = corr[(corr["min_P1_cells"] == PRIMARY_MIN_CELLS) & (corr["analysis"] == "simple_spearman")].copy()
    main_corr["branch"] = pd.Categorical(main_corr["branch"], categories=order[::-1], ordered=True)
    plt.figure(figsize=(2.75, 2.75))
    ax = plt.gca()
    colors = main_corr["branch_class"].map(
        {
            "main_GSH_arm": "#d73027",
            "related_transport": "#fc8d59",
            "alternative_NRF2_stress": "#4575b4",
            "known_confounder": "#969696",
        }
    )
    ax.barh(main_corr["branch"].astype(str), main_corr["rho"], color=colors, alpha=0.92)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    ax.set_xlabel("Spearman rho with P1 CXCL9/10")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)
    savefig("Panel_A_branch_simple_spearman_forest")

    basic_ols = ols[(ols["min_P1_cells"] == PRIMARY_MIN_CELLS) & (ols["analysis"] == "ols_hc3_phase_p1fraction")].copy()
    basic_ols["branch"] = pd.Categorical(basic_ols["branch"], categories=order[::-1], ordered=True)
    plt.figure(figsize=(2.75, 2.75))
    ax = plt.gca()
    colors = basic_ols["branch_class"].map(
        {
            "main_GSH_arm": "#d73027",
            "related_transport": "#fc8d59",
            "alternative_NRF2_stress": "#4575b4",
            "known_confounder": "#969696",
        }
    )
    ax.barh(basic_ols["branch"].astype(str), basic_ols["beta"], color=colors, alpha=0.92)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    ax.set_xlabel("Standardized beta\nadjusted for phase + P1 fraction")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)
    savefig("Panel_B_branch_adjusted_beta_phase_P1fraction")

    heat = corr[corr["min_P1_cells"].isin(MIN_CELL_THRESHOLDS) & corr["analysis"].eq("simple_spearman")].pivot(index="branch", columns="min_P1_cells", values="rho")
    heat = heat.loc[order]
    plt.figure(figsize=(2.55, 2.55))
    ax = sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "rho", "shrink": 0.72},
    )
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    ax.set_xlabel("Minimum P1 cells")
    ax.set_ylabel("")
    savefig("Panel_C_branch_min_cell_threshold_heatmap")

    loo_sum = loo.groupby(["branch", "branch_label"], as_index=False)["expected_negative_retained"].mean()
    loo_sum["branch"] = pd.Categorical(loo_sum["branch"], categories=order[::-1], ordered=True)
    plt.figure(figsize=(2.45, 2.65))
    ax = plt.gca()
    ax.barh(loo_sum["branch"].astype(str), loo_sum["expected_negative_retained"], color="#756bb1", alpha=0.9)
    ax.set_xlim(0, 1.02)
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
    ax.set_xlabel("LOO fraction retaining negative direction")
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)
    savefig("Panel_D_branch_leave_one_patient_out_retention")

    top_branches = ["GSH_cystine", "SLC7A11_GCLC", "AA_transport", "NRF2_detox", "thioredoxin_PRDX", "heme_iron"]
    sub = pp[pp["n_P1_cells"] >= PRIMARY_MIN_CELLS].copy()
    mat = sub[[f"branch_{b}_score" for b in top_branches] + [PRIMARY_Y]].corr(method="spearman")
    mat.index = [c.replace("branch_", "").replace("_score", "") for c in mat.index]
    mat.columns = [c.replace("branch_", "").replace("_score", "") for c in mat.columns]
    rename = {b: BRANCH_LABELS.get(b, b) for b in top_branches}
    rename["CXCL9_10_module"] = "CXCL9/10"
    mat = mat.rename(index=rename, columns=rename)
    plt.figure(figsize=(2.65, 2.35))
    ax = sns.heatmap(mat, cmap="vlag", center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor="white", cbar_kws={"label": "Spearman rho", "shrink": 0.7})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    savefig("Panel_E_redox_branch_correlation_matrix")


def write_report(corr: pd.DataFrame, ols: pd.DataFrame, loo: pd.DataFrame) -> None:
    summary = pd.read_csv(TAB / "branch_specificity_CXCL910_summary_ranked.csv")
    top = summary.iloc[0]
    gsh = summary[summary["branch"] == "GSH_cystine"].iloc[0]
    anchor = summary[summary["branch"] == "SLC7A11_GCLC"].iloc[0]
    aa = summary[summary["branch"] == "AA_transport"].iloc[0]
    nrf2 = summary[summary["branch"] == "NRF2_detox"].iloc[0]
    prdx = summary[summary["branch"] == "thioredoxin_PRDX"].iloc[0]
    heme = summary[summary["branch"] == "heme_iron"].iloc[0]

    lines = [
        "# P1内部CXCL9/10分支特异性比较",
        "",
        "## 做了什么",
        "",
        "只取P1细胞，在RAW h5ad中重新计算多个分支的CP10K-log1p表达分数，然后按patient + phase聚合为pseudobulk单位。统一比较每个分支与P1内部CXCL9/10 score的关系。主分析使用至少10个P1细胞的patient-phase单位，并输出1、5、20、50个P1细胞阈值敏感性、phase/P1 fraction调整模型和leave-one-patient-out。",
        "",
        "比较的分支包括：GSH/cystine、SLC7A11/GCLC、AA transport、NRF2 detox、thioredoxin/PRDX、heme/iron、hypoxia、EMT、cell cycle和stemness。",
        "",
        "## 排名结果",
        "",
        f"- 综合方向稳定性排名第一的是：{top['branch_label']}。",
        f"- GSH/cystine: simple rho={gsh['simple_rho']:.3f}, p={gsh['simple_p']:.4g}; phase+P1 fraction partial rho={gsh['partial_phase_p1fraction_rho']:.3f}, p={gsh['partial_phase_p1fraction_p']:.4g}; LOO负方向保留={int(gsh['loo_negative_retained'])}/{int(gsh['loo_total'])}。",
        f"- SLC7A11/GCLC: simple rho={anchor['simple_rho']:.3f}, p={anchor['simple_p']:.4g}; phase+P1 fraction partial rho={anchor['partial_phase_p1fraction_rho']:.3f}, p={anchor['partial_phase_p1fraction_p']:.4g}; LOO负方向保留={int(anchor['loo_negative_retained'])}/{int(anchor['loo_total'])}。",
        f"- AA transport: simple rho={aa['simple_rho']:.3f}, p={aa['simple_p']:.4g}; phase+P1 fraction partial rho={aa['partial_phase_p1fraction_rho']:.3f}, p={aa['partial_phase_p1fraction_p']:.4g}; LOO负方向保留={int(aa['loo_negative_retained'])}/{int(aa['loo_total'])}。",
        f"- NRF2 detox: simple rho={nrf2['simple_rho']:.3f}, p={nrf2['simple_p']:.4g}; LOO负方向保留={int(nrf2['loo_negative_retained'])}/{int(nrf2['loo_total'])}。",
        f"- Thioredoxin/PRDX: simple rho={prdx['simple_rho']:.3f}, p={prdx['simple_p']:.4g}; LOO负方向保留={int(prdx['loo_negative_retained'])}/{int(prdx['loo_total'])}。",
        f"- Heme/iron: simple rho={heme['simple_rho']:.3f}, p={heme['simple_p']:.4g}; LOO负方向保留={int(heme['loo_negative_retained'])}/{int(heme['loo_total'])}。",
        "",
        "## 说人话",
        "",
        "这一步不是问“P1是不是免疫抑制”，而是问一个更窄的问题：CXCL9/10-low更贴近哪条肿瘤内在分支。若GSH/cystine或SLC7A11/GCLC比detox、thioredoxin、heme/iron、hypoxia、EMT、cell cycle和stemness更稳定地对应CXCL9/10-low，就支持我们的主线：P1不是泛NRF2或泛stress全部升高，而是GSH/cystine分支和CXCL9/10-low耦合更突出。",
        "",
        "这仍然是观察性分支比较，不等于证明GSH分支因果压低CXCL9/10。其作用是界定关联主要集中于SLC7A11/GCLC-GSH分支，而不是泛NRF2信号。",
        "",
        "## 输出文件",
        "",
        "- `tables/branch_gene_coverage.csv`",
        "- `tables/P1_patient_phase_branch_specificity_scores.csv`",
        "- `tables/branch_specificity_CXCL910_correlations.csv`",
        "- `tables/branch_specificity_CXCL910_ols_hc3.csv`",
        "- `tables/branch_specificity_CXCL910_summary_ranked.csv`",
        "- `figures/Panel_A_branch_simple_spearman_forest.png/pdf`",
        "- `figures/Panel_B_branch_adjusted_beta_phase_P1fraction.png/pdf`",
        "- `figures/Panel_C_branch_min_cell_threshold_heatmap.png/pdf`",
        "- `figures/Panel_D_branch_leave_one_patient_out_retention.png/pdf`",
        "- `figures/Panel_E_redox_branch_correlation_matrix.png/pdf`",
    ]
    (OUT / "P1_CXCL910_branch_specificity_report_CN.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "primary_min_P1_cells": PRIMARY_MIN_CELLS,
        "top_branch": str(top["branch"]),
        "top_branch_label": str(top["branch_label"]),
        "gsh_simple_rho": float(gsh["simple_rho"]),
        "slc7a11_gclc_simple_rho": float(anchor["simple_rho"]),
        "aa_transport_simple_rho": float(aa["simple_rho"]),
        "nrf2_detox_simple_rho": float(nrf2["simple_rho"]),
        "output": "results/immune/branch_specificity",
    }
    (LOG / "branch_specificity_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    p1, coverage = score_branch_modules()
    pp = make_pseudobulk(p1)
    corr, ols, loo = run_models(pp)
    plot_panels(pp, corr, ols, loo)
    write_report(corr, ols, loo)
    print("Completed branch-specificity analysis.")


if __name__ == "__main__":
    main()
