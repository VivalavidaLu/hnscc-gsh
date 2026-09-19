"""Reanalyse post-treatment P1 abundance and TME transcriptional context.

This script intentionally does not use the deprecated P1_fraction x GSH score.
The primary exposure is post-treatment P1 fraction among malignant cells; mean
P1 cNMF usage is a convergent, non-independent sensitivity exposure. The two
primary outcomes are a core T/NK activation score and an inflammatory-myeloid
transcriptional score. All inference is patient-level.
"""

from __future__ import annotations

import os

import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT = DATA_ROOT / "processed" / "section3_p1_immune_pressure_sample_merged.csv"
TME_H5AD = DATA_ROOT / "processed" / "tme_annotated.h5ad"
OUT = RESULTS_ROOT / "immune" / "patient_level"
TABLES = OUT / "tables"
INDIVIDUAL_FIGURES = OUT / "individual_panels"
REPORTS = OUT / "report"

TNK_SUBTYPES = {
    "CD4_Tn", "CD8_T_prolif", "CD8_Tem", "CD8_Tem_stress", "CD8_Tex",
    "IFN_T", "NK", "Tfh-like", "Treg", "Treg_resting",
}
CORE_TNK_GENES = ["IFNG", "TNF", "CD69"]
PRIMARY_OUTCOMES = ["tnk_activation_core_score", "macrophage_inflammatory_score"]
PRIMARY_EXPOSURES = ["P1_fraction", "P1_usage_mean"]
SEED = 301741


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_obs_column(obs: h5py.Group, key: str):
    obj = obs[key]
    encoding = obj.attrs.get("encoding-type", "")
    encoding = decode(encoding) if isinstance(encoding, bytes) else encoding
    if isinstance(obj, h5py.Group) and encoding == "categorical":
        categories = [decode(x) for x in obj["categories"][:]]
        return [categories[i] if i >= 0 else np.nan for i in obj["codes"][:]]
    if isinstance(obj, h5py.Group) and encoding == "nullable-string-array":
        values, mask = obj["values"][:], obj["mask"][:]
        return [np.nan if is_missing else decode(value) for value, is_missing in zip(values, mask)]
    values = obj[:]
    return [decode(x) for x in values] if values.dtype.kind in {"S", "O"} else values


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


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    values = pd.to_numeric(frame[value], errors="coerce")
    weights = pd.to_numeric(frame[weight], errors="coerce")
    keep = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep])) if keep.any() else np.nan


def compute_core_tnk_scores() -> pd.DataFrame:
    """Read only genes needed for the corrected T/NK activation sensitivity score."""
    with h5py.File(TME_H5AD, "r") as handle:
        obs = handle["obs"]
        meta = pd.DataFrame({
            "Patient": read_obs_column(obs, "Patient"),
            "SampleID": read_obs_column(obs, "SampleID"),
            "Phase": pd.Series(read_obs_column(obs, "TreatmentPhase")).astype(str).str.lower(),
            "cell_subtype": read_obs_column(obs, "cell_subtype"),
        })
        genes = [decode(x) for x in handle["var"]["gene_id"]["values"][:]]
        gene_index = {gene: index for index, gene in enumerate(genes)}
        missing = [gene for gene in CORE_TNK_GENES if gene not in gene_index]
        if missing:
            raise RuntimeError(f"Core T/NK genes missing from TME matrix: {missing}")
        target_columns = {gene: gene_index[gene] for gene in CORE_TNK_GENES}
        matrix = handle["X"]
        tnk_rows = np.flatnonzero(meta["cell_subtype"].isin(TNK_SUBTYPES).to_numpy())
        scores = np.zeros((len(tnk_rows), len(CORE_TNK_GENES)), dtype=float)
        gene_position = {gene: index for index, gene in enumerate(CORE_TNK_GENES)}
        for output_row, row in enumerate(tnk_rows):
            start, end = int(matrix["indptr"][row]), int(matrix["indptr"][row + 1])
            columns = matrix["indices"][start:end]
            values = matrix["data"][start:end]
            for gene, column in target_columns.items():
                locations = np.flatnonzero(columns == column)
                if locations.size:
                    scores[output_row, gene_position[gene]] = float(values[locations[0]])
    meta = meta.iloc[tnk_rows].copy()
    meta["tnk_activation_core_score"] = scores.mean(axis=1)
    rows = []
    for keys, sub in meta.groupby(["Patient", "SampleID", "Phase"], observed=True):
        rows.append({
            "Patient": keys[0], "SampleID": keys[1], "Phase": keys[2],
            "tnk_core_n_cells": int(len(sub)),
            "tnk_activation_core_score": float(sub["tnk_activation_core_score"].mean()),
        })
    return pd.DataFrame(rows)


def make_patient_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(INPUT)
    core = compute_core_tnk_scores()
    sample = sample.merge(core, on=["Patient", "SampleID", "Phase"], how="left", validate="one_to_one")
    post = sample.loc[sample["Phase"].astype(str).str.lower().eq("post")].copy()
    post = post.loc[post["malignant_cells"].notna()].copy()
    rows = []
    for patient, sub in post.groupby("Patient", observed=True):
        row = {
            "Patient": patient,
            "n_post_samples": int(sub["SampleID"].nunique()),
            "post_sample_ids": ";".join(sorted(sub["SampleID"].astype(str).unique())),
            "cell_fraction_values": ";".join(sorted(set(";".join(sub["CellFraction_values"].astype(str)).split(";")))),
            "malignant_cells": int(sub["malignant_cells"].sum()),
            "P1_cells": int(sub["P1_cells"].sum()),
            "total_tme_cells": int(sub["total_tme_cells"].sum()),
            "TNK_cells": int(sub["TNK_cells"].sum()),
            "Myeloid_cells": int(sub["Myeloid_cells"].sum()),
        }
        row["P1_fraction"] = row["P1_cells"] / row["malignant_cells"]
        row["P1_usage_mean"] = weighted_mean(sub, "P1_usage_mean", "malignant_cells")
        row["tnk_activation_core_score"] = weighted_mean(sub, "tnk_activation_core_score", "tnk_core_n_cells")
        row["tcell_activation_legacy_score"] = weighted_mean(sub, "tcell_activation_ifn_score", "tcell_activation_ifn_n_cells")
        row["macrophage_inflammatory_score"] = weighted_mean(sub, "macrophage_inflammatory_score", "macrophage_inflammatory_n_cells")
        row["treg_context_score"] = weighted_mean(sub, "treg_context_score", "treg_context_n_cells")
        row["tcell_exhaustion_checkpoint_score"] = weighted_mean(
            sub, "tcell_exhaustion_checkpoint_score", "tcell_exhaustion_checkpoint_n_cells"
        )
        rows.append(row)
    patient = pd.DataFrame(rows).sort_values("Patient").reset_index(drop=True)
    for column in [
        "tnk_activation_core_score", "macrophage_inflammatory_score",
        "treg_context_score", "tcell_exhaustion_checkpoint_score",
    ]:
        values = patient[column]
        patient[f"z_{column}"] = (values - values.mean()) / values.std(ddof=0)
    patient["immune_dampened_summary_index"] = (
        -patient["z_tnk_activation_core_score"]
        - patient["z_macrophage_inflammatory_score"]
        + patient["z_treg_context_score"]
        + patient["z_tcell_exhaustion_checkpoint_score"]
    ) / 4
    return sample, patient


def spearman_bootstrap_permutation(x: pd.Series, y: pd.Series, seed: int, n_boot: int = 2_000, n_perm: int = 20_000) -> dict:
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
    analyses = [("primary_all", patient), ("sensitivity_malignant_ge50", patient.loc[patient["malignant_cells"] >= 50])]
    for analysis_name, data in analyses:
        for exposure in PRIMARY_EXPOSURES:
            for outcome in PRIMARY_OUTCOMES + ["immune_dampened_summary_index"]:
                result = spearman_bootstrap_permutation(data[exposure], data[outcome], SEED + len(rows))
                rows.append({"analysis": analysis_name, "exposure": exposure, "outcome": outcome, **result})
                if analysis_name == "primary_all" and outcome in PRIMARY_OUTCOMES:
                    for omitted in data["Patient"]:
                        loo = data.loc[data["Patient"] != omitted]
                        loo_result = spearman_bootstrap_permutation(
                            loo[exposure], loo[outcome], SEED + len(loo_rows), n_boot=0, n_perm=0
                        )
                        loo_rows.append({"exposure": exposure, "outcome": outcome, "omitted_patient": omitted, **loo_result})
    summary = pd.DataFrame(rows)
    primary_mask = (
        summary["analysis"].eq("primary_all")
        & summary["outcome"].isin(PRIMARY_OUTCOMES)
        & summary["exposure"].isin(PRIMARY_EXPOSURES)
    )
    summary["p_bh_primary_family"] = np.nan
    summary.loc[primary_mask, "p_bh_primary_family"] = bh_fdr(summary.loc[primary_mask, "p_permutation"])
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


def save_individual_panel(figure: plt.Figure, stem: str) -> None:
    for extension, kwargs in {
        "pdf": {}, "svg": {}, "png": {"dpi": 600}, "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        figure.savefig(INDIVIDUAL_FIGURES / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def individual_scatter_panel(patient: pd.DataFrame, row: pd.Series, outcome: str, y_label: str, stem: str) -> None:
    figure, axis = plt.subplots(figsize=(3.25, 2.65))
    x = patient["P1_fraction"] * 100
    y = patient[outcome]
    axis.scatter(x, y, s=42, color="#0072B2", alpha=0.92, edgecolor="white", linewidth=0.55, zorder=3)
    slope, intercept, _, _ = stats.theilslopes(y, x)
    grid_x = np.linspace(x.min(), x.max(), 100)
    axis.plot(grid_x, intercept + slope * grid_x, color="#1F2933", linewidth=1.2, zorder=2)
    axis.set_xlabel("P1 fraction among malignant cells (%)", fontsize=8)
    axis.set_ylabel(y_label, fontsize=8)
    axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E6E9EC", linewidth=0.55)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=0.5)
    save_individual_panel(figure, stem)


def individual_effect_panel(summary: pd.DataFrame, exposure: str, stem: str, color: str, annotation: str) -> None:
    data = summary.loc[
        summary["analysis"].eq("primary_all")
        & summary["outcome"].isin(PRIMARY_OUTCOMES)
        & summary["exposure"].eq(exposure)
    ].copy()
    data["order"] = data["outcome"].map({"tnk_activation_core_score": 0, "macrophage_inflammatory_score": 1})
    data = data.sort_values("order")
    labels = [
        "Core T/NK activation",
        "Inflammatory myeloid",
    ]
    figure, axis = plt.subplots(figsize=(3.55, 1.8))
    positions = np.arange(len(data))[::-1]
    for position, row in zip(positions, data.itertuples()):
        axis.plot([row.ci95_boot_low, row.ci95_boot_high], [position, position], color=color, linewidth=2.2, solid_capstyle="round")
        axis.scatter(row.rho, position, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        q = "q=" + f"{row.p_bh_primary_family:.3f}" if pd.notna(row.p_bh_primary_family) else "secondary"
        axis.text(0.47, position, q, va="center", fontsize=6.7, color="#39434D")
    axis.axvline(0, color="#68737D", linewidth=0.8)
    axis.set_xlim(-1.04, 0.72)
    axis.set_yticks(positions, labels, fontsize=6.7)
    axis.set_xlabel("Spearman rho with 95% bootstrap CI", fontsize=7.2)
    axis.tick_params(labelsize=6.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#E6E9EC", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=0.55)
    save_individual_panel(figure, stem)


def individual_loo_panel(summary: pd.DataFrame, loo: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(2.95, 1.5))
    labels = ["T/NK activation", "Inflammatory myeloid"]
    outcomes = ["tnk_activation_core_score", "macrophage_inflammatory_score"]
    styles = {
        "tnk_activation_core_score": {"color": "#0072B2", "marker": "o", "y": 0.72},
        "macrophage_inflammatory_score": {"color": "#D55E00", "marker": "^", "y": 0.28},
    }
    for outcome in outcomes:
        style = styles[outcome]
        position = style["y"]
        subset = loo.loc[(loo["exposure"] == "P1_fraction") & (loo["outcome"] == outcome)].sort_values("rho")
        offsets = np.linspace(-0.10, 0.10, len(subset))
        axis.scatter(
            subset["rho"], position + offsets, s=22, color=style["color"], marker=style["marker"],
            edgecolor="white", linewidth=0.45, zorder=3,
        )
        axis.plot([subset["rho"].min(), subset["rho"].max()], [position, position], color=style["color"], linewidth=2.0, solid_capstyle="round")
        full = summary.loc[
            (summary["analysis"] == "primary_all") & (summary["exposure"] == "P1_fraction") & (summary["outcome"] == outcome), "rho"
        ].iloc[0]
        axis.scatter(full, position, marker="D", s=36, color=style["color"], edgecolor="white", linewidth=0.55, zorder=4)
    axis.axvline(0, color="#68737D", linewidth=0.8)
    axis.set_xlim(-0.95, 0.12)
    axis.set_ylim(0.05, 0.95)
    axis.set_yticks([0.72, 0.28], labels, fontsize=6.3)
    axis.set_xlabel("LOO Spearman rho", fontsize=7.0)
    axis.tick_params(labelsize=6.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#E6E9EC", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=0.55)
    save_individual_panel(figure, "Fig3D_post_P1_fraction_LOO_stability")


def individual_ranked_profile_panel(patient: pd.DataFrame) -> None:
    ordered = patient.sort_values("P1_fraction").reset_index(drop=True)
    columns = ["P1_fraction", "tnk_activation_core_score", "macrophage_inflammatory_score"]
    labels = ["P1 fraction", "Core T/NK activation", "Inflammatory myeloid"]
    matrix = np.vstack([
        (ordered[column] - ordered[column].mean()) / ordered[column].std(ddof=0) for column in columns
    ])
    figure, axis = plt.subplots(figsize=(4.15, 2.05))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    axis.set_yticks(np.arange(3), labels, fontsize=7.0)
    axis.set_xticks(np.arange(len(ordered)), ordered["Patient"], rotation=55, ha="right", fontsize=5.3)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("Patient-level z-score", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.8)
    figure.tight_layout(pad=0.45)
    save_individual_panel(figure, "Fig3E_post_P1_fraction_ranked_TME_profile")


def make_individual_panels(patient: pd.DataFrame, summary: pd.DataFrame, loo: pd.DataFrame) -> None:
    primary = summary.loc[summary["analysis"].eq("primary_all")]
    tnk = primary.loc[(primary["exposure"] == "P1_fraction") & (primary["outcome"] == "tnk_activation_core_score")].iloc[0]
    myeloid = primary.loc[(primary["exposure"] == "P1_fraction") & (primary["outcome"] == "macrophage_inflammatory_score")].iloc[0]
    individual_scatter_panel(
        patient, tnk, "tnk_activation_core_score", "Core T/NK activation score",
        "Fig3A_post_P1_fraction_vs_TNK_activation",
    )
    individual_scatter_panel(
        patient, myeloid, "macrophage_inflammatory_score", "Inflammatory-myeloid score",
        "Fig3B_post_P1_fraction_vs_inflammatory_myeloid",
    )
    individual_effect_panel(
        summary, "P1_fraction", "Fig3C_post_P1_fraction_TME_effect_sizes", "#0072B2",
        "Primary exposure: P1 fraction; bars show 95% bootstrap CI",
    )
    individual_effect_panel(
        summary, "P1_usage_mean", "FigS3_post_P1_usage_TME_sensitivity", "#009E73",
        "Sensitivity exposure: continuous P1 usage; bars show 95% bootstrap CI",
    )
    individual_loo_panel(summary, loo)
    individual_ranked_profile_panel(patient)


def write_report(patient: pd.DataFrame, summary: pd.DataFrame, loo_summary: pd.DataFrame):
    primary = summary.loc[(summary["analysis"] == "primary_all") & summary["outcome"].isin(PRIMARY_OUTCOMES)]
    lines = [
        "# Result 3: 治疗后 P1 丰度与 TME 转录背景",
        "",
        "## 设计与边界",
        "",
        "本重分析的主暴露是治疗后患者级 P1 fraction（P1 恶性细胞数/全部恶性细胞数）；P1 cNMF usage 仅作为同一量化框架内的收敛性敏感性暴露。分析不使用 P1 fraction × GSH/cystine composite，因此不将 P1 丰度与 P1 内部 GSH 状态混为一个暴露。",
        "",
        f"主分析包含 {len(patient)} 位具有可测恶性细胞暴露的治疗后患者。另有 TME 对象中未测得恶性细胞的患者被排除，而非归为 P1-low。每位患者的多个治疗后样本先按相关细胞群数量加权聚合，避免多样本患者构成独立重复。",
        "",
        "两个预设 TME 转录终点为：(i) T/NK 细胞内核心 activation score（IFNG、TNF、CD69 的平均 log-normalized expression）；(ii) 髓系细胞内 inflammatory-myeloid score。后者衡量髓系细胞的炎性转录状态，不等同于髓系细胞浸润数量。旧脚本所用 T/NK IFN score 含 CXCL9/10/11，存在细胞来源不匹配问题，故只作为历史/敏感性指标，不作为本重分析的主终点。",
        "",
        "## 患者级结果",
        "",
    ]
    for _, row in primary.iterrows():
        role = "主暴露" if row["exposure"] == "P1_fraction" else "usage 敏感性"
        lines.append(
            f"- {role}: {row['exposure']} 与 {row['outcome']} 的 Spearman rho={row['rho']:+.3f}，"
            f"95% bootstrap CI [{row['ci95_boot_low']:+.3f}, {row['ci95_boot_high']:+.3f}]，"
            f"置换 p={row['p_permutation']:.4f}，四项主检验 BH-FDR={row['p_bh_primary_family']:.4f}，n={int(row['n'])}。"
        )
    lines += [
        "",
        "## 稳定性与限制",
        "",
        "- 留一患者审计用于评估同队列内结果是否由单一患者主导；它不是独立外部验证。",
        "- 总恶性细胞数不少于 50 的患者敏感性分析用于降低低分母 P1 fraction 的影响。",
        "- 本研究为治疗后横断面相关，无法判断 P1 丰度是免疫低反应背景的原因、结果或共同后果。",
        "- TME 对象来自不同 CellFraction 分选；因此细胞比例不得解释为组织中的绝对浸润比例。主分析优先使用细胞群内部的转录分数以降低该问题，但不能消除取样异质性。",
        "- immune_dampened_summary_index 是对核心结果的描述性汇总，因含主终点成分而不构成独立证据。",
        "",
        "## 正文可写表述",
        "",
        "在治疗后患者级分析中，P1 fraction 较高的肿瘤与较低的 T/NK 细胞 activation 转录状态及较低的炎性髓系转录状态相关。连续 P1 usage 对炎性髓系终点给出同向结果，而对 T/NK activation 仅呈同向趋势；两项 P1 fraction 关联在留一患者审计中均保持负方向。由于样本量有限、分析为横断面且 TME 由分选对象构建，这些结果支持 P1 丰度所处的免疫低反应背景，而不支持 P1 驱动免疫抑制或疗效耐受的因果结论。",
        "",
        "## Figure 3A-B 图注（英文）",
        "",
        "Figure 3A-B. Post-treatment P1 abundance is associated with a less activated immune transcriptional context. Each dot represents one patient-level aggregate across all available post-treatment samples (n=13), rather than an individual cell or sample. P1 fraction was calculated as the total number of P1 malignant cells divided by the total number of malignant cells for each patient. In panel A, the y-axis shows the core T/NK activation score, calculated as the pooled mean expression of IFNG, TNF, and CD69 within T/NK cells. In panel B, the y-axis shows the inflammatory-myeloid transcriptional score calculated within myeloid cells. All points are plotted at a fixed size. Solid lines indicate Theil-Sen trend estimates for visualization. Associations were evaluated using two-sided patient-level Spearman correlations with permutation-based P values. P values were adjusted using the Benjamini-Hochberg procedure. Panel A: rho=-0.615, permutation P=0.028, FDR q=0.038. Panel B: rho=-0.698, permutation P=0.011, FDR q=0.038.",
        "",
        "## Figure 3A-B 图注（中文）",
        "",
        "图3A-B。治疗后 P1 丰度与较低的免疫转录活性背景相关。每个点代表一位患者将所有可用治疗后样本合并后的患者级聚合值（n=13），不代表单个细胞或单个样本。P1 fraction 定义为该患者 P1 恶性细胞数占恶性细胞总数的比例。A图纵轴为 T/NK 细胞内 IFNG、TNF 和 CD69 的平均表达分数；B图纵轴为髓系细胞内炎性髓系转录分数。所有点使用相同大小。实线为 Theil-Sen 趋势线，仅用于可视化。关联采用患者级双侧 Spearman 相关和置换检验，并使用 Benjamini-Hochberg 方法校正多重检验。A图 rho=-0.615，置换 P=0.028，FDR q=0.038；B图 rho=-0.698，置换 P=0.011，FDR q=0.038。",
    ]
    (REPORTS / "Result3_post_P1_abundance_TME_audit_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for directory in [TABLES, INDIVIDUAL_FIGURES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)
    sample, patient = make_patient_table()
    summary, loo, loo_summary = correlations(patient)
    sample.to_csv(TABLES / "post_sample_with_corrected_TNK_score.csv", index=False)
    patient.to_csv(TABLES / "post_patient_P1_abundance_TME_scores.csv", index=False)
    summary.to_csv(TABLES / "post_patient_P1_abundance_TME_correlations.csv", index=False)
    loo.to_csv(TABLES / "post_patient_P1_abundance_TME_leave_one_out.csv", index=False)
    loo_summary.to_csv(TABLES / "post_patient_P1_abundance_TME_leave_one_out_summary.csv", index=False)
    make_individual_panels(patient, summary, loo)
    write_report(patient, summary, loo_summary)
    metadata = {
        "primary_unit": "post-treatment patient aggregate",
        "primary_exposure": "P1_fraction",
        "sensitivity_exposure": "P1_usage_mean",
        "primary_outcomes": PRIMARY_OUTCOMES,
        "n_patients": int(len(patient)),
        "cell_downsampling": False,
        "permutation_iterations": 20000,
        "bootstrap_iterations": 2000,
        "seed": SEED,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
