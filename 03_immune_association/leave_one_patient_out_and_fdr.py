from __future__ import annotations

import os

import json
import warnings
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.formula.api import mixedlm


INPUT = RESULTS_ROOT / "immune" / "mixed_effects" / "tables" / "cell_level_scores_with_covariates.csv"
RAW_H5AD = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
OUT = RESULTS_ROOT / "immune" / "cxcl9_cxcl10_robustness"
TAB = OUT / "tables"
REPORT = OUT / "report"
GENES = ["CXCL9", "CXCL10"]
EXPOSURE = "GSH_cystine_score"
MIN_P1_CELLS = 10
N_BOOT = 5000
SEED = 20260717


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=s.index)


def bh(p: pd.Series) -> pd.Series:
    values = pd.to_numeric(p, errors="coerce").to_numpy(float)
    out = np.full(len(values), np.nan)
    good = np.isfinite(values)
    index = np.flatnonzero(good)[np.argsort(values[good])]
    adjusted = values[index] * len(index) / np.arange(1, len(index) + 1)
    out[index] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return pd.Series(out, index=p.index)


def spearman(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    d = frame[[x, y]].dropna()
    rho, p = stats.spearmanr(d[x], d[y])
    return float(rho), float(p), int(len(d))


def cluster_bootstrap(frame: pd.DataFrame, x: str, y: str) -> tuple[float, float, float, float]:
    d = frame[["Patient", x, y]].dropna().reset_index(drop=True)
    patients = d["Patient"].drop_duplicates().tolist()
    group_indices = [np.flatnonzero(d["Patient"].to_numpy() == patient) for patient in patients]
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    rng = np.random.default_rng(SEED)
    values = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([group_indices[j] for j in draws])
        values[i] = np.corrcoef(stats.rankdata(xv[idx]), stats.rankdata(yv[idx]))[0, 1]
    values = values[np.isfinite(values)]
    return float(np.quantile(values, .025)), float(np.quantile(values, .5)), float(np.quantile(values, .975)), float((values < 0).mean())


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(INPUT)
    cells["cell_id"] = cells["cell_id"].astype(str)
    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    if not pd.Index(cells["cell_id"]).isin(pd.Index(raw.obs_names.astype(str))).all():
        raise ValueError("Cell ID mismatch between audited table and raw h5ad.")
    matrix = dense(raw[cells["cell_id"].tolist(), GENES].X).astype(float)
    counts = pd.to_numeric(raw.obs.loc[cells["cell_id"], "nCount_RNA"], errors="coerce").to_numpy(float)
    raw.file.close()
    expression = np.log1p(matrix / counts[:, None] * 10000.0)
    for index, gene in enumerate(GENES):
        cells[f"{gene}_logCP10K"] = expression[:, index]
        cells[f"{gene}_z"] = zscore(cells[f"{gene}_logCP10K"])
        cells[f"{gene}_detected"] = cells[f"{gene}_logCP10K"].gt(0)
    cells["Phase"] = cells["Phase"].astype(str).str.lower()
    cells["phase_post"] = cells["Phase"].eq("post").astype(float)
    return cells, pd.DataFrame({"gene": GENES, "expression_definition": "CP10K-log1p from single_cell_counts.h5ad"})


def mixed_gene_models(p1: pd.DataFrame) -> pd.DataFrame:
    covariates = ["phase_post", "P1_fraction_patient_phase", "S_phase_score", "G2M_phase_score", "stemness_proxy_score", "EMT_score", "hypoxia_score"]
    rows = []
    for gene in GENES:
        outcome = f"{gene}_z"
        d = p1[[outcome, EXPOSURE, "Patient"] + covariates].dropna().copy()
        for column in [outcome, EXPOSURE] + covariates:
            d[f"z_{column}"] = zscore(d[column])
        formula = f"z_{outcome} ~ z_{EXPOSURE} + " + " + ".join(f"z_{column}" for column in covariates)
        record = {"gene": gene, "n_cells": len(d), "n_patients": d["Patient"].nunique(), "formula": formula}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = mixedlm(formula, d, groups=d["Patient"]).fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            record.update(beta=float(fit.params[f"z_{EXPOSURE}"]), se=float(fit.bse[f"z_{EXPOSURE}"]), p_value=float(fit.pvalues[f"z_{EXPOSURE}"]), converged=bool(fit.converged), status="ok")
        except Exception as exc:
            record.update(beta=np.nan, se=np.nan, p_value=np.nan, converged=False, status=f"failed: {type(exc).__name__}")
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
    pp = p1.groupby(["Patient", "Phase", "Patient_Phase"], observed=True).agg(**aggregations).reset_index()
    primary = pp.loc[pp["n_P1_cells"].ge(MIN_P1_CELLS)].copy()
    rows = []
    for gene in GENES:
        for measure in [f"{gene}_mean_logCP10K", f"{gene}_detection_fraction"]:
            rho, p, n = spearman(primary, EXPOSURE, measure)
            low, med, high, negative = cluster_bootstrap(primary, EXPOSURE, measure)
            rows.append({"gene": gene, "measure": measure, "rho": rho, "nominal_p": p, "n_patient_phase": n, "n_patients": primary["Patient"].nunique(), "cluster_ci95_low": low, "cluster_median": med, "cluster_ci95_high": high, "bootstrap_fraction_negative": negative})
    loo_rows = []
    for gene in GENES:
        measure = f"{gene}_mean_logCP10K"
        full_rho, full_p, _ = spearman(primary, EXPOSURE, measure)
        for patient in sorted(primary["Patient"].unique()):
            rho, p, n = spearman(primary.loc[~primary["Patient"].eq(patient)], EXPOSURE, measure)
            loo_rows.append({"gene": gene, "left_out_patient": patient, "rho": rho, "nominal_p": p, "n_patient_phase": n, "full_rho": full_rho, "full_nominal_p": full_p})
    return pp, pd.DataFrame(rows), pd.DataFrame(loo_rows)


def write_report_utf8(mixed: pd.DataFrame, pphase: pd.DataFrame, loo: pd.DataFrame) -> None:
    lines = [
        "# Result 3.4：CXCL9 与 CXCL10 单基因审计",
        "",
        "本分析将 CXCL9 与 CXCL10 从旧的含 CXCL11 模块中拆开，使用相同的原始表达、患者随机截距模型、patient-phase pseudobulk、检出率和 LOO 规则。该分析用于比较单基因关联的一致性，而非通过追溯性选择放大患者层结论。",
    ]
    for gene in GENES:
        cell = mixed.loc[mixed["gene"].eq(gene)].iloc[0]
        expr = pphase.loc[(pphase["gene"].eq(gene)) & (pphase["measure"].eq(f"{gene}_mean_logCP10K"))].iloc[0]
        detect = pphase.loc[(pphase["gene"].eq(gene)) & (pphase["measure"].eq(f"{gene}_detection_fraction"))].iloc[0]
        gene_loo = loo.loc[loo["gene"].eq(gene)]
        lines.extend([
            "",
            f"## {gene}",
            f"- 细胞层 mixed model：beta={cell['beta']:.3f}，FDR={cell['fdr']:.4g}。",
            f"- patient-phase 平均表达：rho={expr['rho']:.3f}，名义 P={expr['nominal_p']:.3f}，患者簇 bootstrap 95% CI=[{expr['cluster_ci95_low']:.3f}, {expr['cluster_ci95_high']:.3f}]。",
            f"- patient-phase 检出率：rho={detect['rho']:.3f}，名义 P={detect['nominal_p']:.3f}。",
            f"- 平均表达 LOO：{int((gene_loo['rho'] < 0).sum())}/{len(gene_loo)} 次为负方向。",
        ])
    lines.extend([
        "",
        "## 判定",
        "以细胞层和患者层结果共同判断：若单个基因在两层均同向，则视为较强的关联证据；若患者层精度较宽，则仅报告方向与不确定性。",
        "",
    ])
    (REPORT / "Result3_34_CXCL9_CXCL10_individual_gene_audit_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for folder in [TAB, REPORT]:
        folder.mkdir(parents=True, exist_ok=True)
    cells, coverage = load_data()
    p1 = cells.loc[cells["is_P1"].astype(bool)].copy()
    mixed = mixed_gene_models(p1)
    pp, pphase, loo = pseudobulk(p1)
    coverage.to_csv(TAB / "CXCL9_CXCL10_gene_coverage.csv", index=False, encoding="utf-8-sig")
    p1.to_csv(TAB / "P1_cells_CXCL9_CXCL10_expression.csv", index=False, encoding="utf-8-sig")
    pp.to_csv(TAB / "P1_patient_phase_CXCL9_CXCL10_expression.csv", index=False, encoding="utf-8-sig")
    mixed.to_csv(TAB / "P1_CXCL9_CXCL10_cell_mixed_models.csv", index=False, encoding="utf-8-sig")
    pphase.to_csv(TAB / "P1_CXCL9_CXCL10_patient_phase_statistics.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(TAB / "P1_CXCL9_CXCL10_LOO.csv", index=False, encoding="utf-8-sig")
    write_report_utf8(mixed, pphase, loo)
    (OUT / "analysis_metadata.json").write_text(json.dumps({"genes": GENES, "primary_min_P1_cells": MIN_P1_CELLS, "bootstrap_resamples": N_BOOT, "seed": SEED}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Completed CXCL9/CXCL10 robustness analysis.")


if __name__ == "__main__":
    main()
