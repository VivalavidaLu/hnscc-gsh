"""Compare direct decoupler ULM activities between upper and lower P1 GSH quartiles."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import decoupler as dc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

FOCUS_TFS = [
    "NFE2L2",
    "ATF4",
    "BACH1",
    "HIF1A",
    "JUN",
    "FOS",
    "FOSL1",
    "CEBPB",
    "STAT1",
    "IRF1",
    "IRF7",
    "RELA",
    "NFKB1",
    "TP63",
    "KLF5",
    "SOX2",
    "MYC",
    "E2F1",
]

GSH_SIGNATURE_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]

SCORE_SETS = {
    "GSH_cystine_score": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "AA_transport_score": [
        "SLC1A5",
        "SLC7A5",
        "SLC38A1",
        "SLC38A2",
        "SLC1A4",
        "SLC7A8",
    ],
    "NRF2_detox_nonGSH_score": [
        "NQO1",
        "HMOX1",
        "TXN",
        "TXNRD1",
        "PRDX1",
        "SRXN1",
        "AKR1C1",
        "AKR1C2",
        "AKR1C3",
    ],
    "IFN_gamma_response_score": [
        "STAT1",
        "IRF1",
        "CXCL9",
        "CXCL10",
        "CXCL11",
        "GBP1",
        "GBP2",
        "GBP5",
        "IDO1",
    ],
    "JAK_STAT_score": [
        "STAT1",
        "STAT2",
        "IRF1",
        "JAK1",
        "JAK2",
        "SOCS1",
        "SOCS3",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "ISG15",
    ],
    "CXCL9_10_module_score": ["CXCL9", "CXCL10", "CXCL11"],
    "MHC_I_APM_score": [
        "B2M",
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "TAP1",
        "TAP2",
        "PSMB8",
        "PSMB9",
        "NLRC5",
    ],
    "combined_IFN_APM_score": [
        "STAT1",
        "IRF1",
        "CXCL9",
        "CXCL10",
        "CXCL11",
        "GBP1",
        "GBP2",
        "B2M",
        "HLA-A",
        "HLA-B",
        "HLA-C",
        "TAP1",
        "TAP2",
        "PSMB8",
        "PSMB9",
        "NLRC5",
    ],
    "S_phase_score": [
        "MCM5",
        "PCNA",
        "TYMS",
        "FEN1",
        "MCM2",
        "MCM4",
        "RRM1",
        "UNG",
        "GINS2",
        "MCM6",
        "CDCA7",
        "DTL",
        "PRIM1",
        "UHRF1",
        "HELLS",
        "RFC2",
        "RPA2",
        "NASP",
        "RAD51AP1",
        "GMNN",
        "WDR76",
        "SLBP",
        "CCNE2",
        "UBR7",
        "POLD3",
        "MSH2",
        "ATAD2",
        "RAD51",
        "RRM2",
        "CDC45",
        "CDC6",
        "EXO1",
        "TIPIN",
        "DSCC1",
        "BLM",
        "CASP8AP2",
        "USP1",
        "CLSPN",
        "POLA1",
        "CHAF1B",
        "BRIP1",
        "E2F8",
    ],
    "G2M_phase_score": [
        "HMGB2",
        "CDK1",
        "NUSAP1",
        "UBE2C",
        "BIRC5",
        "TPX2",
        "TOP2A",
        "NDC80",
        "CKS2",
        "NUF2",
        "CKS1B",
        "MKI67",
        "TMPO",
        "CENPF",
        "TACC3",
        "FAM64A",
        "SMC4",
        "CCNB2",
        "CKAP2L",
        "CKAP2",
        "AURKB",
        "BUB1",
        "KIF11",
        "ANP32E",
        "TUBB4B",
        "GTSE1",
        "KIF20B",
        "HJURP",
        "CDCA3",
        "HN1",
        "CDC20",
        "TTK",
        "CDC25C",
        "KIF2C",
        "RANGAP1",
        "NCAPD2",
        "DLGAP5",
        "CDCA2",
        "CDCA8",
        "ECT2",
        "KIF23",
        "HMMR",
        "AURKA",
        "PSRC1",
        "ANLN",
        "LBR",
        "CKAP5",
        "CENPE",
        "CTCF",
        "NEK2",
        "G2E3",
        "GAS2L3",
        "CBX5",
        "CENPA",
    ],
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
        rows.append(
            {
                "score": name,
                "n_defined": len(genes),
                "n_present": len(present),
                "genes_present": ";".join(present),
                "genes_missing": ";".join([g for g in genes if g not in expr.columns]),
            }
        )
    return scores, pd.DataFrame(rows)


def build_groups(meta: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["program_group"] = np.where(
        out["cNMF_program"].astype(str).eq("1"),
        "P1",
        np.where(out["cNMF_program"].astype(str).eq("5"), "P5", "nonP1_nonP5"),
    )
    out["P1_GSH_group"] = "not_P1"
    p1_idx = out.index[out["program_group"].eq("P1")]
    q25, q75 = scores.loc[p1_idx, "GSH_cystine_score"].quantile([0.25, 0.75])
    out.loc[p1_idx[scores.loc[p1_idx, "GSH_cystine_score"] <= q25], "P1_GSH_group"] = (
        "P1_GSH_low_Q1"
    )
    out.loc[p1_idx[scores.loc[p1_idx, "GSH_cystine_score"] >= q75], "P1_GSH_group"] = (
        "P1_GSH_high_Q4"
    )
    out.loc[
        p1_idx[
            (scores.loc[p1_idx, "GSH_cystine_score"] > q25)
            & (scores.loc[p1_idx, "GSH_cystine_score"] < q75)
        ],
        "P1_GSH_group",
    ] = "P1_GSH_mid"
    return out


def pseudobulk(
    expr: pd.DataFrame, meta: pd.DataFrame, scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell = pd.concat([meta, scores], axis=1)
    rows = []
    matrices = []
    for (patient, phase, group), idx in cell.groupby(
        ["Patient", "TreatmentPhase", "program_group"], observed=True
    ).groups.items():
        idx = list(idx)
        if len(idx) < 5:
            continue
        rows.append(
            {
                "pb_id": f"{patient}|{phase}|{group}",
                "Patient": patient,
                "TreatmentPhase": phase,
                "contrast_class": "program",
                "group": group,
                "n_cells": len(idx),
            }
        )
        matrices.append(expr.loc[idx].mean(axis=0))
    p1 = cell[cell["program_group"].eq("P1")]
    for (patient, phase, ggroup), idx in p1.groupby(
        ["Patient", "TreatmentPhase", "P1_GSH_group"], observed=True
    ).groups.items():
        if ggroup == "P1_GSH_mid":
            continue
        idx = list(idx)
        if len(idx) < 5:
            continue
        rows.append(
            {
                "pb_id": f"{patient}|{phase}|{ggroup}",
                "Patient": patient,
                "TreatmentPhase": phase,
                "contrast_class": "P1_GSH",
                "group": ggroup,
                "n_cells": len(idx),
            }
        )
        matrices.append(expr.loc[idx].mean(axis=0))
    pb_meta = pd.DataFrame(rows)
    pb_expr = pd.DataFrame(matrices, index=pb_meta["pb_id"], columns=expr.columns)
    return pb_expr, pb_meta


def prepare_net(
    expr_genes: pd.Index, prior: str, network_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not network_file.exists():
        raise FileNotFoundError(f"Missing official CollecTRI file: {network_file}")
    net = pd.read_csv(network_file)
    net = net[["source", "target", "weight"]].dropna().drop_duplicates()
    net = net[net["source"].isin(FOCUS_TFS) & net["target"].isin(expr_genes)].copy()
    if prior == "gsh_targets_removed":
        net = net[~net["target"].isin(GSH_SIGNATURE_GENES)].copy()
    counts = (
        net.groupby("source")["target"]
        .nunique()
        .reindex(FOCUS_TFS)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    counts.columns = ["TF", "n_targets_in_expression"]
    counts["prior"] = prior
    keep_tfs = counts.loc[counts["n_targets_in_expression"] >= 5, "TF"].tolist()
    net = net[net["source"].isin(keep_tfs)].copy()
    counts["used_tmin5"] = counts["TF"].isin(keep_tfs)
    return net, counts


def cliff_delta(x: pd.Series, y: pd.Series) -> float:
    return float(np.sign(x.to_numpy()[:, None] - y.to_numpy()[None, :]).mean())


def contrast_table(activity: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    group_col = "group"
    selected = metadata[metadata["contrast_class"].eq("P1_GSH")].copy()
    rows = []
    for tf in activity.columns:
        x = activity.loc[selected.index[selected[group_col].eq("P1_GSH_high_Q4")], tf]
        y = activity.loc[selected.index[selected[group_col].eq("P1_GSH_low_Q1")], tf]
        p_value = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
        rows.append(
            {
                "TF": tf,
                "n_high": len(x),
                "n_low": len(y),
                "median_high": x.median(),
                "median_low": y.median(),
                "median_difference": x.median() - y.median(),
                "cliffs_delta": cliff_delta(x, y),
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["fdr"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare direct decoupler ULM activities between upper and lower P1 GSH quartiles."
    )
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Analysis AnnData with the discovery expression matrix X and Patient, TreatmentPhase, cNMF_program in obs",
    )
    parser.add_argument(
        "--network",
        required=True,
        type=Path,
        help="Human CollecTRI CSV with source, target and weight",
    )
    parser.add_argument("--out", required=True, type=Path, help="TF contrast CSV")
    args = parser.parse_args()
    a = ad.read_h5ad(args.matrix)
    expr = pd.DataFrame(dense(a.X), index=a.obs_names, columns=a.var_names)
    scores, _ = module_scores(expr)
    meta = build_groups(a.obs[["Patient", "TreatmentPhase", "cNMF_program"]], scores)
    pb, groups = pseudobulk(expr, meta, scores)
    net, _ = prepare_net(pb.columns, "gsh_targets_removed", args.network)
    fitted = dc.mt.ulm(pb, net, tmin=5, raw=False, empty=True, verbose=False)
    activity = fitted[0] if isinstance(fitted, tuple) else fitted
    contrast_table(
        pd.DataFrame(activity, index=pb.index)[FOCUS_TFS], groups.set_index("pb_id")
    ).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
