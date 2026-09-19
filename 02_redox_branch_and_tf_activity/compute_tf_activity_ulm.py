from __future__ import annotations

import os

import json
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


H5AD = DATA_ROOT / "processed" / "malignant_6165.h5ad"
OUT = RESULTS_ROOT / "redox" / "tf_activity"
TAB = OUT / "tables"
FIG = OUT / "figures"
LOG = OUT / "logs"
for p in [TAB, FIG, LOG]:
    p.mkdir(parents=True, exist_ok=True)

COLLECTRI = DATA_ROOT / "reference" / "collectri_human.csv"

FOCUS_TFS = [
    "NFE2L2", "ATF4", "BACH1", "HIF1A", "JUN", "FOS", "FOSL1", "CEBPB",
    "STAT1", "IRF1", "IRF7", "RELA", "NFKB1", "TP63", "KLF5", "SOX2",
    "MYC", "E2F1",
]
REDOX_TFS = ["NFE2L2", "ATF4", "BACH1", "HIF1A", "JUN", "FOS", "FOSL1", "CEBPB"]
IFN_TFS = ["STAT1", "IRF1", "IRF7", "RELA", "NFKB1"]
CONTROL_TFS = ["TP63", "KLF5", "SOX2", "MYC", "E2F1"]
GSH_SIGNATURE_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]

SCORE_SETS = {
    "GSH_cystine_score": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "AA_transport_score": ["SLC1A5", "SLC7A5", "SLC38A1", "SLC38A2", "SLC1A4", "SLC7A8"],
    "NRF2_detox_nonGSH_score": ["NQO1", "HMOX1", "TXN", "TXNRD1", "PRDX1", "SRXN1", "AKR1C1", "AKR1C2", "AKR1C3"],
    "IFN_gamma_response_score": ["STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP2", "GBP5", "IDO1"],
    "JAK_STAT_score": ["STAT1", "STAT2", "IRF1", "JAK1", "JAK2", "SOCS1", "SOCS3", "IFIT1", "IFIT2", "IFIT3", "ISG15"],
    "CXCL9_10_module_score": ["CXCL9", "CXCL10", "CXCL11"],
    "MHC_I_APM_score": ["B2M", "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5"],
    "combined_IFN_APM_score": [
        "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP2", "B2M",
        "HLA-A", "HLA-B", "HLA-C", "TAP1", "TAP2", "PSMB8", "PSMB9", "NLRC5",
    ],
    "S_phase_score": ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"],
    "G2M_phase_score": ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"],
}


def dense(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.astype(float)
    sd = arr.std(axis=0, ddof=0).replace(0, np.nan)
    return ((arr - arr.mean(axis=0)) / sd).fillna(0.0)


def module_scores(expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=expr.index)
    rows = []
    for name, genes in SCORE_SETS.items():
        present = [g for g in genes if g in expr.columns]
        if present:
            scores[name] = zscore_df(expr[present]).mean(axis=1)
        else:
            scores[name] = np.nan
        rows.append({
            "score": name,
            "n_defined": len(genes),
            "n_present": len(present),
            "genes_present": ";".join(present),
            "genes_missing": ";".join([g for g in genes if g not in expr.columns]),
        })
    return scores, pd.DataFrame(rows)


def build_groups(meta: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["program_group"] = np.where(out["cNMF_program"].astype(str).eq("1"), "P1",
                            np.where(out["cNMF_program"].astype(str).eq("5"), "P5", "nonP1_nonP5"))
    out["P1_GSH_group"] = "not_P1"
    p1_idx = out.index[out["program_group"].eq("P1")]
    q25, q75 = scores.loc[p1_idx, "GSH_cystine_score"].quantile([0.25, 0.75])
    out.loc[p1_idx[scores.loc[p1_idx, "GSH_cystine_score"] <= q25], "P1_GSH_group"] = "P1_GSH_low_Q1"
    out.loc[p1_idx[scores.loc[p1_idx, "GSH_cystine_score"] >= q75], "P1_GSH_group"] = "P1_GSH_high_Q4"
    out.loc[p1_idx[(scores.loc[p1_idx, "GSH_cystine_score"] > q25) & (scores.loc[p1_idx, "GSH_cystine_score"] < q75)], "P1_GSH_group"] = "P1_GSH_mid"
    return out


def pseudobulk(expr: pd.DataFrame, meta: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = pd.concat([meta, scores], axis=1)
    rows = []
    matrices = []
    for (patient, phase, group), idx in cell.groupby(["Patient", "TreatmentPhase", "program_group"], observed=True).groups.items():
        idx = list(idx)
        if len(idx) < 5:
            continue
        rows.append({"pb_id": f"{patient}|{phase}|{group}", "Patient": patient, "TreatmentPhase": phase, "contrast_class": "program", "group": group, "n_cells": len(idx)})
        matrices.append(expr.loc[idx].mean(axis=0))
    p1 = cell[cell["program_group"].eq("P1")]
    for (patient, phase, ggroup), idx in p1.groupby(["Patient", "TreatmentPhase", "P1_GSH_group"], observed=True).groups.items():
        if ggroup == "P1_GSH_mid":
            continue
        idx = list(idx)
        if len(idx) < 5:
            continue
        rows.append({"pb_id": f"{patient}|{phase}|{ggroup}", "Patient": patient, "TreatmentPhase": phase, "contrast_class": "P1_GSH", "group": ggroup, "n_cells": len(idx)})
        matrices.append(expr.loc[idx].mean(axis=0))
    pb_meta = pd.DataFrame(rows)
    pb_expr = pd.DataFrame(matrices, index=pb_meta["pb_id"], columns=expr.columns)
    return pb_expr, pb_meta


def prepare_net(expr_genes: pd.Index, prior: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not COLLECTRI.exists():
        raise FileNotFoundError("Missing reference/collectri_human.csv under the configured data directory")
    net = pd.read_csv(COLLECTRI)
    net = net[["source", "target", "weight"]].dropna().drop_duplicates()
    net = net[net["source"].isin(FOCUS_TFS) & net["target"].isin(expr_genes)].copy()
    if prior == "gsh_targets_removed":
        net = net[~net["target"].isin(GSH_SIGNATURE_GENES)].copy()
    counts = net.groupby("source")["target"].nunique().reindex(FOCUS_TFS).fillna(0).astype(int).reset_index()
    counts.columns = ["TF", "n_targets_in_expression"]
    counts["prior"] = prior
    keep_tfs = counts.loc[counts["n_targets_in_expression"] >= 5, "TF"].tolist()
    net = net[net["source"].isin(keep_tfs)].copy()
    counts["used_tmin5"] = counts["TF"].isin(keep_tfs)
    return net, counts


def adjacency(net: pd.DataFrame) -> pd.DataFrame:
    adj = net.pivot_table(index="target", columns="source", values="weight", aggfunc="mean").fillna(0.0)
    return adj


def ulm_scores(pb_expr: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    genes = adj.index
    y = pb_expr[genes].to_numpy(dtype=float)
    w = adj.to_numpy(dtype=float)
    n = len(genes)
    df = n - 2
    y_center = y - y.mean(axis=1, keepdims=True)
    w_center = w - w.mean(axis=0, keepdims=True)
    cov = y_center @ w_center / (n - 1)
    y_sd = y.std(axis=1, ddof=1).reshape(-1, 1)
    w_sd = w.std(axis=0, ddof=1).reshape(1, -1)
    r = cov / (y_sd * w_sd)
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt(df / ((1.0 - r) * (1.0 + r)))
    return pd.DataFrame(t, index=pb_expr.index, columns=adj.columns)


def mlm_scores(pb_expr: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    genes = adj.index
    X = adj.to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    inv = np.linalg.pinv(X.T @ X)
    betas = []
    tvals = []
    df = X.shape[0] - X.shape[1]
    for _, row in pb_expr[genes].iterrows():
        y = row.to_numpy(dtype=float)
        beta = inv @ X.T @ y
        resid = y - X @ beta
        sse = float((resid ** 2).sum() / max(df, 1))
        se = np.sqrt(np.diag(sse * inv))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = beta / se
        tvals.append(t[1:])
    return pd.DataFrame(tvals, index=pb_expr.index, columns=adj.columns)


def effect_table(df: pd.DataFrame, group_col: str, a: str, b: str, cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        x = df.loc[df[group_col].eq(a), col].dropna().astype(float)
        y = df.loc[df[group_col].eq(b), col].dropna().astype(float)
        if len(x) < 3 or len(y) < 3:
            p = np.nan
            cliff = np.nan
        else:
            p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
            cliff = float(np.sign(x.to_numpy()[:, None] - y.to_numpy()[None, :]).mean())
        rows.append({"contrast": label, "variable": col, "group_a": a, "group_b": b, "n_a": len(x), "n_b": len(y), "median_a": x.median() if len(x) else np.nan, "median_b": y.median() if len(y) else np.nan, "delta_median": (x.median() - y.median()) if len(x) and len(y) else np.nan, "cliffs_delta": cliff, "p_value": p})
    out = pd.DataFrame(rows)
    mask = out["p_value"].notna()
    out["fdr"] = np.nan
    if mask.any():
        out.loc[mask, "fdr"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def spearman_grid(df: pd.DataFrame, x_cols: list[str], y_cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        for y in y_cols:
            sub = df[[x, y]].dropna()
            if sub.shape[0] < 5 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[x], sub[y])
            rows.append({"analysis": label, "x": x, "y": y, "n": int(sub.shape[0]), "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    mask = out["p_value"].notna()
    out["fdr"] = np.nan
    if mask.any():
        out.loc[mask, "fdr"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def residualize(y: pd.Series, covars: pd.DataFrame) -> pd.Series:
    tmp = pd.concat([y.rename("y"), covars], axis=1).dropna()
    out = pd.Series(np.nan, index=y.index)
    if tmp.shape[0] <= covars.shape[1] + 2:
        return out
    X = tmp[covars.columns].to_numpy(dtype=float)
    X = np.column_stack([np.ones(X.shape[0]), X])
    yy = tmp["y"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    out.loc[tmp.index] = yy - X @ beta
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, covars: list[str], label: str) -> dict:
    tmp = df[[x, y] + covars].dropna()
    if tmp.shape[0] < len(covars) + 5:
        return {"analysis": label, "x": x, "y": y, "covars": ";".join(covars), "n": int(tmp.shape[0]), "partial_rho": np.nan, "p_value": np.nan}
    ranks = tmp.rank()
    rx = residualize(ranks[x], ranks[covars])
    ry = residualize(ranks[y], ranks[covars])
    sub = pd.concat([rx, ry], axis=1).dropna()
    rho, p = stats.pearsonr(sub.iloc[:, 0], sub.iloc[:, 1]) if sub.shape[0] >= 5 else (np.nan, np.nan)
    return {"analysis": label, "x": x, "y": y, "covars": ";".join(covars), "n": int(sub.shape[0]), "partial_rho": rho, "p_value": p}


def main() -> None:
    start = time.time()
    a = ad.read_h5ad(H5AD)
    expr = pd.DataFrame(dense(a.X), index=a.obs_names, columns=a.var_names)
    meta = a.obs[["Patient", "TreatmentPhase", "SampleID", "cNMF_program", "nCount_RNA", "nFeature_RNA", "percent.mt"]].copy()
    scores, coverage = module_scores(expr)
    coverage.to_csv(TAB / "pseudobulk_module_score_gene_coverage.csv", index=False)
    meta = build_groups(meta, scores)
    pb_expr, pb_meta = pseudobulk(expr, meta, scores)
    pb_scores, _ = module_scores(pb_expr)
    pb_meta = pb_meta.set_index("pb_id")
    all_activity = []
    target_counts = []
    for prior in ["full_prior", "gsh_targets_removed"]:
        net, counts = prepare_net(pb_expr.columns, prior)
        target_counts.append(counts)
        net.to_csv(TAB / f"collectri_focus_{prior}_used.csv", index=False)
        adj = adjacency(net)
        for method, func in [("ulm", ulm_scores), ("mlm", mlm_scores)]:
            act = func(pb_expr, adj).reindex(columns=FOCUS_TFS)
            act.columns = [f"{prior}__{method}__{c}" for c in act.columns]
            all_activity.append(act)
    pd.concat(target_counts, ignore_index=True).to_csv(TAB / "collectri_focus_target_counts_pseudobulk.csv", index=False)
    activity = pd.concat(all_activity, axis=1)
    activity.to_csv(TAB / "pseudobulk_decoupler_compatible_tf_activity.csv")
    pb = pd.concat([pb_meta, pb_scores, activity], axis=1)
    pb.to_csv(TAB / "pseudobulk_decoupler_activity_with_scores.csv")
    tf_cols = list(activity.columns)
    contrasts = []
    prog = pb[pb["contrast_class"].eq("program")]
    contrasts.append(effect_table(prog, "group", "P1", "P5", tf_cols, "pseudobulk_P1_vs_P5"))
    contrasts.append(effect_table(prog[prog["group"].isin(["P1", "nonP1_nonP5"])], "group", "P1", "nonP1_nonP5", tf_cols, "pseudobulk_P1_vs_nonP1_nonP5"))
    p1g = pb[pb["contrast_class"].eq("P1_GSH")]
    contrasts.append(effect_table(p1g, "group", "P1_GSH_high_Q4", "P1_GSH_low_Q1", tf_cols, "pseudobulk_P1_GSH_high_vs_low"))
    contrasts = pd.concat(contrasts, ignore_index=True)
    contrasts.to_csv(TAB / "pseudobulk_group_contrasts_tf_activity.csv", index=False)
    p1_prog = prog[prog["group"].eq("P1")].copy()
    readouts = ["GSH_cystine_score", "AA_transport_score", "NRF2_detox_nonGSH_score", "IFN_gamma_response_score", "JAK_STAT_score", "CXCL9_10_module_score", "MHC_I_APM_score", "combined_IFN_APM_score", "S_phase_score", "G2M_phase_score"]
    corr = spearman_grid(p1_prog, tf_cols, readouts, "P1_pseudobulk_TF_activity_vs_readouts")
    corr.to_csv(TAB / "P1_pseudobulk_tf_activity_vs_readout_spearman.csv", index=False)
    partial_rows = []
    for prior in ["full_prior", "gsh_targets_removed"]:
        for method in ["ulm", "mlm"]:
            redox = [f"{prior}__{method}__{tf}" for tf in ["NFE2L2", "ATF4", "BACH1"] if f"{prior}__{method}__{tf}" in p1_prog.columns]
            ifn = [f"{prior}__{method}__{tf}" for tf in ["STAT1", "IRF1"] if f"{prior}__{method}__{tf}" in p1_prog.columns]
            p1_prog[f"{prior}__{method}__redox_tf_mean"] = p1_prog[redox].mean(axis=1)
            p1_prog[f"{prior}__{method}__ifn_tf_mean"] = p1_prog[ifn].mean(axis=1)
            partial_rows.append(partial_spearman(p1_prog, "GSH_cystine_score", "JAK_STAT_score", ["S_phase_score", "G2M_phase_score"], f"{prior}__{method}__GSH_vs_JAK_adjust_cell_cycle"))
            partial_rows.append(partial_spearman(p1_prog, "GSH_cystine_score", "JAK_STAT_score", ["S_phase_score", "G2M_phase_score", f"{prior}__{method}__redox_tf_mean"], f"{prior}__{method}__GSH_vs_JAK_adjust_cell_cycle_redoxTF"))
            partial_rows.append(partial_spearman(p1_prog, f"{prior}__{method}__redox_tf_mean", f"{prior}__{method}__ifn_tf_mean", ["S_phase_score", "G2M_phase_score"], f"{prior}__{method}__redoxTF_vs_ifnTF_adjust_cell_cycle"))
    pd.DataFrame(partial_rows).to_csv(TAB / "P1_pseudobulk_partial_spearman_anti_circularity.csv", index=False)
    p1_prog.to_csv(TAB / "P1_pseudobulk_with_decoupler_branch_means.csv")
    metadata = {
        "analysis": "focused_decoupler_compatible_prior_based_TF_activity_on_pseudobulk",
        "method_boundary": "Uses official CollecTRI prior and decoupler ULM/MLM equations implemented transparently on patient-phase pseudobulk. This is not pySCENIC and not de novo regulon inference.",
        "anti_circularity": {
            "full_prior": "All focused CollecTRI TF-target edges.",
            "gsh_targets_removed": f"Same prior after removing GSH signature genes: {GSH_SIGNATURE_GENES}.",
            "primary_unit": "Patient-phase/program pseudobulk and P1-GSH quartile pseudobulk, not individual cells.",
            "controls": CONTROL_TFS,
        },
        "n_cells": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "n_pseudobulk_profiles": int(pb.shape[0]),
        "runtime_seconds": round(time.time() - start, 2),
    }
    with open(LOG / "focused_decoupler_pseudobulk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
