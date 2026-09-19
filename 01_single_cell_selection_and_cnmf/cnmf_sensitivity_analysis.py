"""
Supplemental cNMF sensitivity, P2 purity, and malignant-cell provenance audit.

This script is intentionally independent from the original discovery scripts.
It does not overwrite the discovery outputs. Results are written below
``results/cnmf_sensitivity``.

Main analyses:
  1. Lightweight NMF sensitivity on the fixed 6,165 projected malignant-cell
     universe with seed-specific patient-stratified downsampling.
  2. Optional replay on the original 3,564-cell cNMF norm-count input.
  3. P2 MHC-II/IFN epithelial purity and ambient/TME marker audit using raw
     counts normalized by per-cell nCount_RNA.
  4. Patient-mixing, CNV/entropy, and cNMF-input provenance checks.

Important interpretation boundary:
  The sensitivity analysis is a lightweight NMF replay, not a full cNMF
  consensus rerun. It tests whether the same major axes are recoverable under
  alternative seeds/downsampling from the same malignant-cell universe.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
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
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


CNMF_INPUT = RESULTS_ROOT / "cnmf"
RUN = CNMF_INPUT / "cnmf_runs" / "malignant_final_v2"
RAW_H5AD = DATA_ROOT / "raw" / "single_cell_counts.h5ad"
FINAL_H5AD = DATA_ROOT / "processed" / "single_cell_annotated.h5ad"
MALIGNANT_H5AD = DATA_ROOT / "processed" / "malignant_6165.h5ad"
PROJECTED_USAGE = DATA_ROOT / "derived" / "cNMF_K6_usage_6165.csv"
CLASSIFICATION = DATA_ROOT / "processed" / "malignant_classification.csv"
NORM_COUNTS = RUN / "cnmf_tmp" / "malignant_final_v2.norm_counts.h5ad"
OUT = RESULTS_ROOT / "cnmf_sensitivity"


PROGRAM_LABELS = {
    "1": "Stress/metabolic-cytoskeletal mixed",
    "2": "MHC-II/IFN-associated basal-like",
    "3": "ECM/pEMT",
    "4": "Cycling",
    "5": "Basal IFN/detox-stress",
    "6": "Inflammatory terminal squamous",
}

MARKER_SETS = {
    "epithelial": [
        "EPCAM",
        "KRT5",
        "KRT14",
        "KRT15",
        "KRT16",
        "KRT17",
        "KRT8",
        "KRT18",
        "KRT19",
        "TP63",
        "ITGA6",
    ],
    "mhc_ii_ifn": [
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DRB5",
        "HLA-DPA1",
        "HLA-DPB1",
        "HLA-DQA1",
        "HLA-DQB1",
        "HLA-DMA",
        "CD74",
        "CIITA",
        "IFI27",
        "IFI35",
        "IFI6",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "ISG15",
        "ISG20",
        "MX1",
        "OAS1",
        "OASL",
        "IRF1",
        "IRF7",
        "STAT1",
        "TAP1",
    ],
    "immune_pan": ["PTPRC", "B2M"],
    "t_nk": ["CD3D", "CD3E", "TRAC", "TRBC1", "TRBC2", "NKG7", "GNLY", "KLRD1"],
    "myeloid": ["LYZ", "LST1", "S100A8", "S100A9", "FCGR3A", "FCER1G"],
    "b_cell": ["MS4A1", "CD79A", "CD79B", "BANK1"],
    "plasma": ["JCHAIN", "MZB1", "XBP1", "IGHG1", "IGKC"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "COL3A1"],
    "endothelial": ["PECAM1", "VWF", "KDR", "CLDN5", "RAMP2"],
}

LINEAGE_CONTAMINATION_SETS = [
    "immune_pan",
    "t_nk",
    "myeloid",
    "b_cell",
    "plasma",
    "fibroblast",
    "endothelial",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default="11,23,37",
        help="Comma-separated random seeds. Default: 11,23,37.",
    )
    parser.add_argument(
        "--ks",
        default="5,6,7",
        help="Comma-separated K values for lightweight NMF replay. Default: 5,6,7.",
    )
    parser.add_argument(
        "--modes",
        default="projected",
        help=(
            "Comma-separated modes: projected and/or actual_input. "
            "projected uses the fixed 6,165-cell universe with downsampling; "
            "actual_input replays the original 3,564-cell cNMF norm-count matrix."
        ),
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=3564,
        help="Downsample size for projected mode. Default matches original cNMF input n=3564.",
    )
    parser.add_argument("--max-iter", type=int, default=450)
    parser.add_argument("--skip-nmf", action="store_true")
    return parser.parse_args()


def ensure_out() -> None:
    for sub in ["tables", "figures", "logs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame, max_rows: int | None = None, max_col_width: int = 90) -> str:
    if df.empty:
        return "_No rows._"
    tmp = df.copy()
    if max_rows is not None:
        tmp = tmp.head(max_rows)
    tmp = tmp.replace({np.nan: ""})

    def clean(x: object) -> str:
        s = str(x).replace("\n", " ").replace("|", "\\|")
        return s if len(s) <= max_col_width else s[: max_col_width - 3] + "..."

    cols = list(tmp.columns)
    lines = ["| " + " | ".join(clean(c) for c in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def read_spectra(k: int) -> pd.DataFrame:
    path = RUN / f"malignant_final_v2.gene_spectra_score.k_{k}.dt_2_0.txt"
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def read_usage(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "cell_id" in df.columns:
            df = df.set_index("cell_id")
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def dominant(usage: pd.DataFrame) -> pd.Series:
    return usage.idxmax(axis=1).astype(str)


def counts_to_lognorm(
    counts: sparse.spmatrix | np.ndarray,
    ncounts: np.ndarray,
    dense: bool = True,
) -> np.ndarray | sparse.spmatrix:
    scale = (1e4 / np.maximum(ncounts.astype(float), 1.0)).astype(np.float32)
    if sparse.issparse(counts):
        x = counts.tocsr().astype(np.float32, copy=True)
        x = x.multiply(scale[:, None])
        x.data = np.log1p(x.data)
        if dense:
            return x.toarray().astype(np.float32, copy=False)
        return x
    x = np.asarray(counts, dtype=np.float32)
    x = x * scale[:, None]
    np.log1p(x, out=x)
    return x


def load_raw_slice_lognorm(
    cell_ids: list[str] | None,
    genes: list[str],
    dense: bool = True,
) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray, list[str], list[str]]:
    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    try:
        present = [g for g in genes if g in raw.var_names]
        missing = sorted(set(genes) - set(present))
        if cell_ids is None:
            sub = raw[:, present]
            used_cells = raw.obs_names.astype(str).tolist()
            ncounts = raw.obs["nCount_RNA"].to_numpy(dtype=float)
        else:
            missing_cells = sorted(set(cell_ids) - set(raw.obs_names.astype(str)))
            if missing_cells:
                raise ValueError(f"{len(missing_cells)} requested cells are missing from RAW h5ad.")
            sub = raw[cell_ids, present]
            used_cells = list(cell_ids)
            ncounts = raw.obs.loc[cell_ids, "nCount_RNA"].to_numpy(dtype=float)
        counts = sub.X
        lognorm = counts_to_lognorm(counts, ncounts, dense=dense)
        return lognorm, ncounts, present, missing
    finally:
        raw.file.close()


def load_raw_slice_counts(
    cell_ids: list[str] | None,
    genes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    try:
        present = [g for g in genes if g in raw.var_names]
        missing = sorted(set(genes) - set(present))
        if cell_ids is None:
            sub = raw[:, present]
            used_cells = raw.obs_names.astype(str).tolist()
            ncounts = raw.obs["nCount_RNA"].to_numpy(dtype=float)
        else:
            sub = raw[cell_ids, present]
            used_cells = list(cell_ids)
            ncounts = raw.obs.loc[cell_ids, "nCount_RNA"].to_numpy(dtype=float)
        counts = sub.X
        if sparse.issparse(counts):
            counts = counts.toarray()
        return np.asarray(counts), ncounts, present, missing
    finally:
        raw.file.close()


def stratified_sample_indices(labels: pd.Series, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = labels.astype(str)
    n = min(n, len(labels))
    counts = labels.value_counts().sort_index()
    raw_alloc = counts / counts.sum() * n
    alloc = np.floor(raw_alloc).astype(int)
    alloc[alloc == 0] = 1
    while alloc.sum() > n:
        candidates = alloc[alloc > 1].index
        if len(candidates) == 0:
            break
        frac = (raw_alloc - alloc).loc[candidates].sort_values()
        alloc.loc[frac.index[0]] -= 1
    while alloc.sum() < n:
        remaining = counts - alloc
        candidates = remaining[remaining > 0].index
        frac = (raw_alloc - alloc).loc[candidates].sort_values(ascending=False)
        alloc.loc[frac.index[0]] += 1

    sampled: list[int] = []
    pos = pd.Series(np.arange(len(labels)), index=labels.index)
    for group, take in alloc.items():
        group_positions = pos[labels.eq(group)].to_numpy()
        sampled.extend(rng.choice(group_positions, size=min(take, len(group_positions)), replace=False).tolist())
    sampled = np.asarray(sampled, dtype=int)
    rng.shuffle(sampled)
    return sampled


def top_genes_from_vector(values: np.ndarray, genes: list[str], n: int = 50) -> set[str]:
    order = np.argsort(values)[::-1][:n]
    return {genes[i] for i in order}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else np.nan


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(stats.spearmanr(a, b, nan_policy="omit").correlation)


def run_lightweight_nmf_sensitivity(args: argparse.Namespace, mal_obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ref_spectra = read_spectra(6)
    ref_usage_6165 = read_usage(PROJECTED_USAGE)
    ref_usage_6165 = ref_usage_6165[[str(i) for i in range(1, 7)]]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    all_match_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_split_rows: list[dict] = []
    all_size_rows: list[dict] = []

    for mode in modes:
        if mode == "projected":
            genes = [g for g in ref_spectra.columns.astype(str).tolist() if g in ref_spectra.columns]
            cells = mal_obs.index.astype(str).tolist()
            x, _, present_genes, missing = load_raw_slice_lognorm(cells, genes, dense=True)
            x = np.asarray(x, dtype=np.float32)
            cell_meta = mal_obs.loc[cells].copy()
            downsample = True
            mode_note = "6165 raw-count log-normalized replay with patient-stratified downsampling"
        elif mode == "actual_input":
            obj = ad.read_h5ad(NORM_COUNTS)
            try:
                x = obj.X
                if sparse.issparse(x):
                    x = x.toarray()
                x = np.asarray(x, dtype=np.float32)
                present_genes = obj.var_names.astype(str).tolist()
                cells = obj.obs_names.astype(str).tolist()
            finally:
                obj.file.close()
            cell_meta = mal_obs.reindex(cells).copy()
            missing = []
            downsample = False
            mode_note = "original 3564 cNMF norm-count input replay without cell downsampling"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ref_common_genes = [g for g in present_genes if g in ref_spectra.columns]
        gene_pos = {g: i for i, g in enumerate(present_genes)}
        ref_pos = [gene_pos[g] for g in ref_common_genes]
        x_use = x[:, ref_pos]
        genes_use = ref_common_genes
        ref_spectra_use = ref_spectra[genes_use]

        ref_usage = ref_usage_6165.reindex(cells).dropna()
        cells_available = ref_usage.index.tolist()
        available_pos = pd.Series(np.arange(len(cells)), index=cells).loc[cells_available].to_numpy()
        x_available = x_use[available_pos, :]
        meta_available = cell_meta.loc[cells_available].copy()
        ref_dom_all = dominant(ref_usage)

        for seed in seeds:
            if downsample:
                sample_positions = stratified_sample_indices(
                    meta_available["Patient"].astype(str),
                    args.sample_n,
                    seed,
                )
            else:
                sample_positions = np.arange(x_available.shape[0])

            sub_cells = meta_available.index[sample_positions].astype(str).tolist()
            x_sub = x_available[sample_positions, :]
            ref_dom = ref_dom_all.loc[sub_cells]

            for k in ks:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = NMF(
                        n_components=k,
                        init="nndsvdar",
                        solver="cd",
                        beta_loss="frobenius",
                        max_iter=args.max_iter,
                        tol=1e-4,
                        random_state=seed,
                    )
                    w = model.fit_transform(x_sub)
                    h = model.components_

                run_dom = pd.Series(
                    [f"C{i + 1}" for i in np.argmax(w, axis=1)],
                    index=sub_cells,
                    name="run_component",
                )
                ari = adjusted_rand_score(ref_dom.astype(str), run_dom.astype(str))
                nmi = normalized_mutual_info_score(ref_dom.astype(str), run_dom.astype(str))

                warn_text = "; ".join(sorted({str(warn.message) for warn in caught}))
                all_metric_rows.append(
                    {
                        "mode": mode,
                        "mode_note": mode_note,
                        "seed": seed,
                        "k": k,
                        "n_cells": int(x_sub.shape[0]),
                        "n_genes": int(x_sub.shape[1]),
                        "ari_vs_reference_k6_assignment": float(ari),
                        "nmi_vs_reference_k6_assignment": float(nmi),
                        "reconstruction_error": float(model.reconstruction_err_),
                        "n_iter": int(model.n_iter_),
                        "warnings": warn_text,
                        "missing_reference_genes": ";".join(missing),
                    }
                )

                comp_labels = [f"C{i + 1}" for i in range(k)]
                comp_top50 = {
                    comp_labels[i]: top_genes_from_vector(h[i, :], genes_use, 50) for i in range(k)
                }
                comp_sizes = run_dom.value_counts().reindex(comp_labels, fill_value=0)
                for comp, n_comp in comp_sizes.items():
                    all_size_rows.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "k": k,
                            "component": comp,
                            "n_cells": int(n_comp),
                            "pct_cells": float(n_comp / len(run_dom) * 100),
                        }
                    )

                best_match: dict[str, str] = {}
                for ref_p in [str(i) for i in range(1, 7)]:
                    ref_vec = ref_spectra_use.loc[ref_p].to_numpy(dtype=float)
                    ref_top = set(ref_spectra_use.loc[ref_p].sort_values(ascending=False).head(50).index)
                    rows = []
                    for i, comp in enumerate(comp_labels):
                        jac = jaccard(ref_top, comp_top50[comp])
                        rho = safe_spearman(ref_vec, h[i, :])
                        rows.append((comp, jac, rho))
                    rows = sorted(rows, key=lambda z: (np.nan_to_num(z[1]), np.nan_to_num(z[2])), reverse=True)
                    best_comp, best_jac, best_rho = rows[0]
                    best_match[ref_p] = best_comp

                    ref_cells = ref_dom[ref_dom.eq(ref_p)].index
                    recovery = float((run_dom.loc[ref_cells] == best_comp).mean()) if len(ref_cells) else np.nan
                    cell_j = jaccard(set(ref_cells), set(run_dom[run_dom.eq(best_comp)].index))
                    all_match_rows.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "k": k,
                            "reference_program": ref_p,
                            "reference_label": PROGRAM_LABELS[ref_p],
                            "best_component": best_comp,
                            "top50_jaccard": float(best_jac),
                            "spearman_spectra": float(best_rho),
                            "reference_program_cells_in_subset": int(len(ref_cells)),
                            "cell_recovery_to_best_component": recovery,
                            "cell_jaccard_to_best_component": cell_j,
                        }
                    )

                p5_comp = best_match["5"]
                p6_comp = best_match["6"]
                p5_cells = ref_dom[ref_dom.eq("5")].index
                p6_cells = ref_dom[ref_dom.eq("6")].index
                all_split_rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "k": k,
                        "p5_best_component": p5_comp,
                        "p6_best_component": p6_comp,
                        "p5_p6_distinct_components": bool(p5_comp != p6_comp),
                        "p5_recovery": float((run_dom.loc[p5_cells] == p5_comp).mean())
                        if len(p5_cells)
                        else np.nan,
                        "p6_recovery": float((run_dom.loc[p6_cells] == p6_comp).mean())
                        if len(p6_cells)
                        else np.nan,
                        "p5_to_p6_component_fraction": float((run_dom.loc[p5_cells] == p6_comp).mean())
                        if len(p5_cells)
                        else np.nan,
                        "p6_to_p5_component_fraction": float((run_dom.loc[p6_cells] == p5_comp).mean())
                        if len(p6_cells)
                        else np.nan,
                    }
                )

    match = pd.DataFrame(all_match_rows)
    metrics = pd.DataFrame(all_metric_rows)
    split = pd.DataFrame(all_split_rows)
    sizes = pd.DataFrame(all_size_rows)

    match.to_csv(OUT / "tables" / "lightweight_nmf_program_matching_long.csv", index=False)
    metrics.to_csv(OUT / "tables" / "lightweight_nmf_assignment_metrics.csv", index=False)
    split.to_csv(OUT / "tables" / "lightweight_nmf_p5_p6_split.csv", index=False)
    sizes.to_csv(OUT / "tables" / "lightweight_nmf_component_sizes.csv", index=False)

    return {"match": match, "metrics": metrics, "split": split, "sizes": sizes}


def mannwhitney_summary(values: pd.Series, groups: pd.Series, target: str) -> dict:
    a = values[groups.eq(target)].dropna().to_numpy()
    b = values[~groups.eq(target)].dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return {"p": np.nan, "auc": np.nan, "rank_biserial": np.nan}
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = stat / (len(a) * len(b))
    return {
        "p": float(p),
        "auc": float(auc),
        "rank_biserial": float(2 * auc - 1),
    }


def module_scores(lognorm: pd.DataFrame, marker_sets: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=lognorm.index)
    for name, genes in marker_sets.items():
        present = [g for g in genes if g in lognorm.columns]
        out[name] = lognorm[present].mean(axis=1) if present else np.nan
    out["lineage_contamination"] = out[LINEAGE_CONTAMINATION_SETS].mean(axis=1)
    return out


def run_p2_purity_audit(mal_obs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_marker_genes = sorted({g for genes in MARKER_SETS.values() for g in genes})

    mal_cells = mal_obs.index.astype(str).tolist()
    mal_counts, mal_ncounts, present, missing = load_raw_slice_counts(mal_cells, all_marker_genes)
    mal_lognorm = counts_to_lognorm(mal_counts, mal_ncounts, dense=True)
    mal_expr = pd.DataFrame(mal_lognorm, index=mal_cells, columns=present)
    mal_raw_counts = pd.DataFrame(mal_counts, index=mal_cells, columns=present)
    mal_scores = module_scores(mal_expr, MARKER_SETS)
    mal_scores["program"] = mal_obs.loc[mal_scores.index, "cNMF_program"].astype(str)
    mal_scores["program_label"] = mal_scores["program"].map(PROGRAM_LABELS)
    mal_scores["Patient"] = mal_obs.loc[mal_scores.index, "Patient"].astype(str)

    group_cols = ["program", "program_label"]
    score_cols = ["epithelial", "mhc_ii_ifn", "lineage_contamination"] + LINEAGE_CONTAMINATION_SETS
    by_program = (
        mal_scores.groupby(group_cols, observed=True)[score_cols]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    by_program.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in by_program.columns
    ]

    stat_rows = []
    for col in score_cols:
        s = mannwhitney_summary(mal_scores[col], mal_scores["program"], "2")
        stat_rows.append(
            {
                "comparison": "P2_vs_nonP2",
                "score": col,
                "p": s["p"],
                "auc_p2_greater_than_nonp2": s["auc"],
                "rank_biserial": s["rank_biserial"],
                "p2_median": float(mal_scores.loc[mal_scores["program"].eq("2"), col].median()),
                "nonp2_median": float(mal_scores.loc[~mal_scores["program"].eq("2"), col].median()),
            }
        )
    stats_df = pd.DataFrame(stat_rows)

    flag_genes = [
        "HLA-DRA",
        "CD74",
        "KRT14",
        "KRT5",
        "EPCAM",
        "PTPRC",
        "CD3D",
        "LYZ",
        "MS4A1",
        "JCHAIN",
        "COL1A1",
        "PECAM1",
    ]
    flag_genes = [g for g in flag_genes if g in mal_raw_counts.columns]
    flags = pd.DataFrame(index=mal_raw_counts.index)
    for g in flag_genes:
        flags[f"{g}_positive"] = mal_raw_counts[g].to_numpy() > 0
    epithelial_flag_genes = [g for g in ["KRT14", "KRT5", "EPCAM"] if f"{g}_positive" in flags]
    immune_flag_genes = [g for g in ["PTPRC", "CD3D", "LYZ", "MS4A1", "JCHAIN"] if f"{g}_positive" in flags]
    flags["hla_dra_cd74_double_positive"] = (
        flags.get("HLA-DRA_positive", False) & flags.get("CD74_positive", False)
    )
    flags["epithelial_marker_positive"] = flags[[f"{g}_positive" for g in epithelial_flag_genes]].any(axis=1)
    flags["immune_marker_positive"] = flags[[f"{g}_positive" for g in immune_flag_genes]].any(axis=1)
    flags["hla_cd74_epithelial_without_immune_marker"] = (
        flags["hla_dra_cd74_double_positive"]
        & flags["epithelial_marker_positive"]
        & ~flags["immune_marker_positive"]
    )
    flags["program"] = mal_scores["program"]
    flag_summary = (
        flags.groupby("program", observed=True)
        .agg(["sum", "mean"])
        .reset_index()
    )
    flag_summary.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in flag_summary.columns
    ]
    flag_summary["program_label"] = flag_summary["program"].map(PROGRAM_LABELS)

    # Cell-type reference distributions using only marker genes from raw counts.
    final = ad.read_h5ad(FINAL_H5AD, backed="r")
    try:
        celltype_obs = final.obs[["cell_type", "Patient", "nCount_RNA"]].copy()
    finally:
        final.file.close()
    raw_all_counts, _, present_all, missing_all = load_raw_slice_counts(None, all_marker_genes)
    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    try:
        all_ncounts = raw.obs["nCount_RNA"].to_numpy(dtype=float)
        all_cells = raw.obs_names.astype(str).tolist()
    finally:
        raw.file.close()
    all_lognorm = counts_to_lognorm(raw_all_counts, all_ncounts, dense=True)
    all_expr = pd.DataFrame(all_lognorm, index=all_cells, columns=present_all)
    all_scores = module_scores(all_expr, MARKER_SETS)
    all_scores = all_scores.join(celltype_obs[["cell_type"]], how="left")
    celltype_ref = (
        all_scores.groupby("cell_type", observed=True)[["epithelial", "mhc_ii_ifn", "lineage_contamination"]]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    celltype_ref.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in celltype_ref.columns
    ]

    p2 = mal_scores[mal_scores["program"].eq("2")]
    cor_rows = []
    for xcol in ["epithelial", "lineage_contamination", "immune_pan", "myeloid", "t_nk"]:
        rho = safe_spearman(p2["mhc_ii_ifn"].to_numpy(), p2[xcol].to_numpy())
        cor_rows.append({"program": "2", "x_score": xcol, "y_score": "mhc_ii_ifn", "spearman": rho})
    corr_df = pd.DataFrame(cor_rows)

    mal_scores.to_csv(OUT / "tables" / "p2_purity_single_cell_scores.csv")
    by_program.to_csv(OUT / "tables" / "p2_purity_scores_by_program.csv", index=False)
    stats_df.to_csv(OUT / "tables" / "p2_purity_p2_vs_nonp2_tests.csv", index=False)
    flag_summary.to_csv(OUT / "tables" / "p2_purity_marker_positive_flags_by_program.csv", index=False)
    celltype_ref.to_csv(OUT / "tables" / "p2_purity_celltype_reference_scores.csv", index=False)
    corr_df.to_csv(OUT / "tables" / "p2_purity_mhc_vs_contamination_correlations.csv", index=False)

    with open(OUT / "logs" / "p2_purity_missing_marker_genes.json", "w", encoding="utf-8") as f:
        json.dump({"malignant_missing": missing, "all_cells_missing": missing_all}, f, indent=2)

    return {
        "single_cell": mal_scores,
        "by_program": by_program,
        "stats": stats_df,
        "flags": flag_summary,
        "celltype_ref": celltype_ref,
        "corr": corr_df,
    }


def shannon_entropy(counts: pd.Series) -> float:
    vals = counts[counts > 0].astype(float).to_numpy()
    if len(vals) == 0:
        return np.nan
    p = vals / vals.sum()
    return float(-(p * np.log(p)).sum())


def run_provenance_audit(mal_obs: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    cls = pd.read_csv(CLASSIFICATION)
    cls["cell_id"] = cls["cell_id"].astype(str)
    cls = cls.set_index("cell_id", drop=False)
    mal = mal_obs.copy()
    mal.index = mal.index.astype(str)
    mal["program"] = mal["cNMF_program"].astype(str)
    joined = mal.join(cls, how="left", rsuffix="_classification")

    usage_6165 = read_usage(PROJECTED_USAGE)
    usage6_path = RUN / "malignant_final_v2.usages.k_6.dt_2_0.consensus.txt"
    usage3564 = read_usage(usage6_path) if usage6_path.exists() else pd.DataFrame()
    cnmf_cells = usage3564.index.astype(str).tolist()

    final = ad.read_h5ad(FINAL_H5AD, backed="r")
    raw = ad.read_h5ad(RAW_H5AD, backed="r")
    norm = ad.read_h5ad(NORM_COUNTS, backed="r") if NORM_COUNTS.exists() else None
    try:
        basic = {
            "raw_postdoublet_cells": int(raw.n_obs),
            "final_annotated_cells": int(final.n_obs),
            "final_epithelial_tumor_cells": int((final.obs["cell_type"].astype(str) == "Epithelial/Tumor").sum()),
            "classification_table_rows": int(cls.shape[0]),
            "classification_malignant_definitive_true": int(cls["MALIGNANT_DEFINITIVE"].fillna(False).astype(bool).sum())
            if "MALIGNANT_DEFINITIVE" in cls
            else None,
            "projected_malignant_cells": int(mal.shape[0]),
            "projected_patients": int(mal["Patient"].nunique()),
            "projected_cells_with_classification_row": int(joined["cell_id"].notna().sum()),
            "projected_malignant_definitive_true": int(
                joined["MALIGNANT_DEFINITIVE"].fillna(False).astype(bool).sum()
            )
            if "MALIGNANT_DEFINITIVE" in joined
            else None,
            "projected_usage_rows": int(usage_6165.shape[0]),
            "cnmf_consensus_usage_rows": int(usage3564.shape[0]),
            "cnmf_norm_counts_shape": [int(norm.n_obs), int(norm.n_vars)] if norm is not None else None,
            "cnmf_input_cells_in_projected_6165": int(pd.Index(cnmf_cells).isin(mal.index).sum()),
            "cnmf_input_cells_in_projected_usage": int(pd.Index(cnmf_cells).isin(usage_6165.index).sum()),
        }
    finally:
        final.file.close()
        raw.file.close()
        if norm is not None:
            norm.file.close()

    program_rows = []
    patient_counts = (
        joined.groupby(["program", "Patient"], observed=True).size().rename("n_cells").reset_index()
    )
    patient_counts.to_csv(OUT / "tables" / "program_patient_counts_projected_6165.csv", index=False)

    for program, sub in joined.groupby("program", observed=True):
        pat_counts = sub["Patient"].astype(str).value_counts()
        h = shannon_entropy(pat_counts)
        n_pat = int(pat_counts.shape[0])
        max_pat = pat_counts.index[0]
        max_pct = float(pat_counts.iloc[0] / pat_counts.sum() * 100)
        row = {
            "program": str(program),
            "program_label": PROGRAM_LABELS.get(str(program), ""),
            "n_cells": int(sub.shape[0]),
            "n_patients": n_pat,
            "max_patient": max_pat,
            "max_patient_pct": max_pct,
            "shannon_patient_entropy": h,
            "normalized_patient_entropy": float(h / math.log(n_pat)) if n_pat > 1 else np.nan,
            "effective_n_patients": float(math.exp(h)) if not np.isnan(h) else np.nan,
        }
        for col in [
            "mixing_entropy",
            "mixing_entropy_norm",
            "mpf",
            "cnv_burden",
            "del_burden",
            "amp_burden",
        ]:
            if col in sub:
                row[f"{col}_median"] = float(pd.to_numeric(sub[col], errors="coerce").median())
                row[f"{col}_mean"] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        for col in [
            "high_cnv",
            "low_entropy",
            "tier1",
            "tier2",
            "MALIGNANT_DEFINITIVE",
            "MALIGNANT_FINAL",
        ]:
            if col in sub:
                row[f"{col}_pct_true"] = float(sub[col].fillna(False).astype(bool).mean() * 100)
        if "classification_definitive" in sub:
            row["classification_definitive_counts"] = json.dumps(
                sub["classification_definitive"].astype(str).value_counts().to_dict(), ensure_ascii=False
            )
        program_rows.append(row)

    program_summary = pd.DataFrame(program_rows).sort_values("program")
    program_summary.to_csv(OUT / "tables" / "program_patient_entropy_cnv_summary.csv", index=False)

    universe_rows = []
    universes = {"projected_6165": pd.Index(mal.index)}
    if cnmf_cells:
        universes["cnmf_consensus_3564"] = pd.Index(cnmf_cells)
    for name, cell_index in universes.items():
        sub = joined.reindex(cell_index)
        universe_rows.append(
            {
                "universe": name,
                "n_cells": int(len(cell_index)),
                "n_patients": int(sub["Patient"].nunique()),
                "with_classification_row": int(sub["cell_id"].notna().sum()),
                "MALIGNANT_DEFINITIVE_pct_true": float(
                    sub["MALIGNANT_DEFINITIVE"].fillna(False).astype(bool).mean() * 100
                )
                if "MALIGNANT_DEFINITIVE" in sub
                else np.nan,
                "high_cnv_pct_true": float(sub["high_cnv"].fillna(False).astype(bool).mean() * 100)
                if "high_cnv" in sub
                else np.nan,
                "low_entropy_pct_true": float(sub["low_entropy"].fillna(False).astype(bool).mean() * 100)
                if "low_entropy" in sub
                else np.nan,
                "median_mixing_entropy_norm": float(
                    pd.to_numeric(sub["mixing_entropy_norm"], errors="coerce").median()
                )
                if "mixing_entropy_norm" in sub
                else np.nan,
                "median_mpf": float(pd.to_numeric(sub["mpf"], errors="coerce").median())
                if "mpf" in sub
                else np.nan,
                "median_cnv_burden": float(pd.to_numeric(sub["cnv_burden"], errors="coerce").median())
                if "cnv_burden" in sub
                else np.nan,
            }
        )
    universe_summary = pd.DataFrame(universe_rows)
    universe_summary.to_csv(OUT / "tables" / "cell_universe_provenance_summary.csv", index=False)

    joined[
        [
            c
            for c in [
                "Patient",
                "TreatmentPhase",
                "program",
                "tumor_leiden",
                "mixing_entropy",
                "mixing_entropy_norm",
                "mpf",
                "cnv_burden",
                "high_cnv",
                "low_entropy",
                "tier1",
                "tier2",
                "MALIGNANT_DEFINITIVE",
                "classification_definitive",
            ]
            if c in joined.columns
        ]
    ].to_csv(OUT / "tables" / "projected_6165_cell_level_provenance.csv")

    with open(OUT / "logs" / "provenance_summary.json", "w", encoding="utf-8") as f:
        json.dump(basic, f, indent=2, ensure_ascii=False)

    return {
        "basic": basic,
        "program_summary": program_summary,
        "universe_summary": universe_summary,
        "patient_counts": patient_counts,
    }


def make_figures(
    nmf: dict[str, pd.DataFrame] | None,
    purity: dict[str, pd.DataFrame],
    prov: dict[str, pd.DataFrame | dict],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    if nmf is not None and not nmf["metrics"].empty:
        metrics = nmf["metrics"].copy()
        match = nmf["match"].copy()
        split = nmf["split"].copy()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        sns.boxplot(data=metrics, x="k", y="ari_vs_reference_k6_assignment", hue="mode", ax=axes[0])
        axes[0].set_title("Assignment ARI vs reference K=6")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("ARI")
        sns.boxplot(data=metrics, x="k", y="nmi_vs_reference_k6_assignment", hue="mode", ax=axes[1])
        axes[1].set_title("Assignment NMI vs reference K=6")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("NMI")

        split_summary = (
            split.groupby(["mode", "k"], observed=True)["p5_p6_distinct_components"]
            .mean()
            .reset_index()
        )
        sns.barplot(data=split_summary, x="k", y="p5_p6_distinct_components", hue="mode", ax=axes[2])
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("P5/P6 matched to distinct components")
        axes[2].set_xlabel("K")
        axes[2].set_ylabel("Fraction of seeds")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / "supplemental_lightweight_nmf_assignment_metrics.png", dpi=300)
        fig.savefig(OUT / "figures" / "supplemental_lightweight_nmf_assignment_metrics.pdf")
        plt.close(fig)

        for mode, sub in match.groupby("mode", observed=True):
            pivot = (
                sub.groupby(["reference_program", "k"], observed=True)["top50_jaccard"]
                .median()
                .reset_index()
                .pivot(index="reference_program", columns="k", values="top50_jaccard")
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis", ax=ax)
            ax.set_title(f"Median top50 Jaccard ({mode})")
            ax.set_xlabel("NMF K")
            ax.set_ylabel("Reference K=6 program")
            fig.tight_layout()
            fig.savefig(OUT / "figures" / f"supplemental_top50_jaccard_heatmap_{mode}.png", dpi=300)
            fig.savefig(OUT / "figures" / f"supplemental_top50_jaccard_heatmap_{mode}.pdf")
            plt.close(fig)

    scores = purity["single_cell"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, col, title in zip(
        axes,
        ["mhc_ii_ifn", "epithelial", "lineage_contamination"],
        ["MHC-II/IFN", "Epithelial", "Lineage contamination"],
    ):
        sns.violinplot(
            data=scores,
            x="program",
            y=col,
            order=[str(i) for i in range(1, 7)],
            inner="quartile",
            cut=0,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("cNMF program")
        ax.set_ylabel("Mean log-normalized marker score")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "supplemental_p2_purity_program_scores.png", dpi=300)
    fig.savefig(OUT / "figures" / "supplemental_p2_purity_program_scores.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_scores = scores.copy()
    sns.scatterplot(
        data=plot_scores,
        x="lineage_contamination",
        y="mhc_ii_ifn",
        hue="program",
        palette="tab10",
        s=12,
        alpha=0.55,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("P2 MHC-II/IFN signal vs lineage markers")
    ax.set_xlabel("Lineage contamination score")
    ax.set_ylabel("MHC-II/IFN score")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "supplemental_p2_mhc_vs_contamination_scatter.png", dpi=300)
    fig.savefig(OUT / "figures" / "supplemental_p2_mhc_vs_contamination_scatter.pdf")
    plt.close(fig)

    program_summary = prov["program_summary"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.barplot(
        data=program_summary,
        x="program",
        y="normalized_patient_entropy",
        order=[str(i) for i in range(1, 7)],
        ax=axes[0],
        color="#4C78A8",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Patient mixing entropy")
    axes[0].set_xlabel("Program")
    axes[0].set_ylabel("Normalized entropy")
    sns.barplot(
        data=program_summary,
        x="program",
        y="max_patient_pct",
        order=[str(i) for i in range(1, 7)],
        ax=axes[1],
        color="#F58518",
    )
    axes[1].set_title("Dominant-patient contribution")
    axes[1].set_xlabel("Program")
    axes[1].set_ylabel("Max patient %")
    y_col = "cnv_burden_median" if "cnv_burden_median" in program_summary else "mpf_median"
    sns.barplot(
        data=program_summary,
        x="program",
        y=y_col,
        order=[str(i) for i in range(1, 7)],
        ax=axes[2],
        color="#54A24B",
    )
    axes[2].set_title(y_col.replace("_", " "))
    axes[2].set_xlabel("Program")
    axes[2].set_ylabel("Median")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "supplemental_patient_entropy_cnv_summary.png", dpi=300)
    fig.savefig(OUT / "figures" / "supplemental_patient_entropy_cnv_summary.pdf")
    plt.close(fig)


def summarize_nmf(nmf: dict[str, pd.DataFrame] | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nmf is None or nmf["metrics"].empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    metrics = nmf["metrics"]
    match = nmf["match"]
    split = nmf["split"]
    metrics_summary = (
        metrics.groupby(["mode", "k"], observed=True)
        .agg(
            seeds=("seed", "nunique"),
            n_cells=("n_cells", "median"),
            ari_median=("ari_vs_reference_k6_assignment", "median"),
            ari_min=("ari_vs_reference_k6_assignment", "min"),
            nmi_median=("nmi_vs_reference_k6_assignment", "median"),
            nmi_min=("nmi_vs_reference_k6_assignment", "min"),
            reconstruction_error_median=("reconstruction_error", "median"),
            n_iter_median=("n_iter", "median"),
        )
        .reset_index()
    )
    program_summary = (
        match.groupby(["mode", "k", "reference_program", "reference_label"], observed=True)
        .agg(
            top50_jaccard_median=("top50_jaccard", "median"),
            top50_jaccard_min=("top50_jaccard", "min"),
            spearman_median=("spearman_spectra", "median"),
            cell_recovery_median=("cell_recovery_to_best_component", "median"),
            cell_recovery_min=("cell_recovery_to_best_component", "min"),
        )
        .reset_index()
    )
    split_summary = (
        split.groupby(["mode", "k"], observed=True)
        .agg(
            seeds=("seed", "nunique"),
            p5_p6_distinct_seed_fraction=("p5_p6_distinct_components", "mean"),
            p5_recovery_median=("p5_recovery", "median"),
            p6_recovery_median=("p6_recovery", "median"),
            p5_to_p6_component_fraction_median=("p5_to_p6_component_fraction", "median"),
            p6_to_p5_component_fraction_median=("p6_to_p5_component_fraction", "median"),
        )
        .reset_index()
    )
    metrics_summary.to_csv(OUT / "tables" / "lightweight_nmf_assignment_metrics_summary.csv", index=False)
    program_summary.to_csv(OUT / "tables" / "lightweight_nmf_program_matching_summary.csv", index=False)
    split_summary.to_csv(OUT / "tables" / "lightweight_nmf_p5_p6_split_summary.csv", index=False)
    return metrics_summary, program_summary, split_summary


def write_report(
    args: argparse.Namespace,
    nmf: dict[str, pd.DataFrame] | None,
    purity: dict[str, pd.DataFrame],
    prov: dict[str, pd.DataFrame | dict],
) -> None:
    metrics_summary, program_summary, split_summary = summarize_nmf(nmf)

    p2_tests = purity["stats"].copy()
    p2_tests_small = p2_tests[
        p2_tests["score"].isin(["mhc_ii_ifn", "epithelial", "lineage_contamination", "immune_pan", "myeloid", "t_nk"])
    ][
        [
            "score",
            "p2_median",
            "nonp2_median",
            "auc_p2_greater_than_nonp2",
            "rank_biserial",
            "p",
        ]
    ]

    flags = purity["flags"].copy()
    flag_cols = [
        c
        for c in flags.columns
        if c.startswith("hla_cd74_epithelial_without_immune_marker_mean")
        or c.startswith("hla_dra_cd74_double_positive_mean")
        or c.startswith("immune_marker_positive_mean")
    ]
    flag_small = flags[["program", "program_label"] + flag_cols].copy()
    for c in flag_cols:
        flag_small[c] = flag_small[c] * 100

    prov_basic = prov["basic"]
    program_prov = prov["program_summary"].copy()
    prov_cols = [
        c
        for c in [
            "program",
            "program_label",
            "n_cells",
            "n_patients",
            "max_patient",
            "max_patient_pct",
            "normalized_patient_entropy",
            "effective_n_patients",
            "mixing_entropy_norm_median",
            "mpf_median",
            "cnv_burden_median",
            "high_cnv_pct_true",
            "low_entropy_pct_true",
            "MALIGNANT_DEFINITIVE_pct_true",
        ]
        if c in program_prov.columns
    ]
    program_prov_small = program_prov[prov_cols]

    manifest_paths = [
        RAW_H5AD,
        FINAL_H5AD,
        MALIGNANT_H5AD,
        PROJECTED_USAGE,
        CLASSIFICATION,
        NORM_COUNTS,
        RUN / "malignant_final_v2.gene_spectra_score.k_6.dt_2_0.txt",
        RUN / "malignant_final_v2.usages.k_6.dt_2_0.consensus.txt",
    ]
    manifest = []
    for path in manifest_paths:
        manifest.append(
            {
                "path": ("data/" + path.relative_to(DATA_ROOT).as_posix()) if path.is_relative_to(DATA_ROOT) else (("results/" + path.relative_to(RESULTS_ROOT).as_posix()) if path.is_relative_to(RESULTS_ROOT) else path.name),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else None,
                "mtime": path.stat().st_mtime if path.exists() else None,
                "sha256": file_sha256(path) if path.exists() and path.stat().st_size < 2_500_000_000 else None,
            }
        )
    with open(OUT / "logs" / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("# 补充审计：cNMF sensitivity、P2 purity 与 malignant provenance\n")
    lines.append("## 结论摘要\n")
    lines.append(
        "- 这次补充分析是轻量级 NMF/cNMF 稳健性审计，不是从 FASTQ 或完整 cNMF consensus 层面的推倒重跑。它的价值是检验现有 K=6 主轴是否在固定 malignant universe、不同随机种子和下采样下仍可恢复。\n"
    )
    if not metrics_summary.empty:
        proj_k6 = metrics_summary[
            metrics_summary["mode"].eq("projected") & metrics_summary["k"].eq(6)
        ]
        if not proj_k6.empty:
            row = proj_k6.iloc[0]
            lines.append(
                f"- 在固定 6,165 malignant cell universe 的 projected-mode 轻量 NMF replay 中，K=6 与参考 K=6 assignment 的中位 ARI={row['ari_median']:.3f}，NMI={row['nmi_median']:.3f}。这些数值应解释为对主轴可重复性的辅助证据，而不是替代原始 cNMF consensus。\n"
            )
        actual_k6 = metrics_summary[
            metrics_summary["mode"].eq("actual_input") & metrics_summary["k"].eq(6)
        ]
        if not actual_k6.empty:
            row = actual_k6.iloc[0]
            lines.append(
                f"- 在原始 3,564-cell norm-count 输入矩阵的 actual-input replay 中，K=6 与参考 K=6 assignment 的中位 ARI={row['ari_median']:.3f}，NMI={row['nmi_median']:.3f}。这说明主程序轴可以部分恢复，但普通单次 NMF 不能等同于 cNMF consensus。\n"
            )
        split_k6 = split_summary[
            split_summary["mode"].eq("projected") & split_summary["k"].eq(6)
        ]
        if not split_k6.empty:
            row = split_k6.iloc[0]
            lines.append(
                f"- P5/P6 在 projected-mode K=6 中被匹配到不同 component 的 seed 比例为 {row['p5_p6_distinct_seed_fraction']:.2f}；P5 median recovery={row['p5_recovery_median']:.3f}，P6 median recovery={row['p6_recovery_median']:.3f}。这直接决定 P5/P6 拆分能写多强。\n"
            )
        actual_split_k6 = split_summary[
            split_summary["mode"].eq("actual_input") & split_summary["k"].eq(6)
        ]
        if not actual_split_k6.empty:
            row = actual_split_k6.iloc[0]
            lines.append(
                f"- 但在 actual-input replay 中，P5/P6 被匹配到不同 component 的 seed 比例为 {row['p5_p6_distinct_seed_fraction']:.2f}。因此，P5/P6 拆分应写为“cNMF consensus 支持且有生物学解释，但对 replay 设置敏感”，不能写成不可动摇的稳定分型。\n"
            )
    lines.append(
        "- P2 purity 审计使用 raw counts，经 nCount_RNA 归一化后计算 marker scores。重点不是证明 P2 绝无 ambient RNA，而是确认 MHC-II/IFN 信号是否与上皮背景共存，且不伴随强 PTPRC/T/NK/myeloid/B/plasma/fibroblast/endothelial lineage signal。\n"
    )
    lines.append(
        "- provenance 审计输出了 6,165 投影细胞、3,564 cNMF consensus input、entropy/CNV 分类表之间的交集关系，以及每个 program 的患者混合熵和 CNV/entropy 指标。患者偏倚必须透明报告。\n"
    )

    lines.append("\n## 轻量 NMF sensitivity\n\n")
    if metrics_summary.empty:
        lines.append("_NMF sensitivity was skipped._\n")
    else:
        lines.append("### Assignment-level 稳定性\n\n")
        lines.append(md_table(metrics_summary.round(4)))
        lines.append("\n\n### Program-level top50/spectra/cell recovery\n\n")
        compact_program = program_summary[
            (program_summary["k"].isin([5, 6, 7]))
            & (program_summary["reference_program"].isin(["2", "3", "4", "5", "6"]))
        ].round(4)
        lines.append(md_table(compact_program, max_rows=60))
        lines.append("\n\n### P5/P6 split\n\n")
        lines.append(md_table(split_summary.round(4)))

    lines.append("\n## P2 purity / ambient 检查\n\n")
    lines.append("### P2 vs non-P2 marker-score tests\n\n")
    lines.append(md_table(p2_tests_small.round(5)))
    lines.append("\n\n### Raw-count positive flag summary by program (%)\n\n")
    lines.append(md_table(flag_small.round(2)))
    lines.append("\n\n### P2 MHC-II/IFN 与 lineage contamination 相关性\n\n")
    lines.append(md_table(purity["corr"].round(4)))

    lines.append("\n## Patient mixing / CNV / entropy provenance\n\n")
    lines.append("### Universe summary\n\n")
    lines.append(md_table(prov["universe_summary"].round(4)))
    lines.append("\n\n### Program-level provenance\n\n")
    lines.append(md_table(program_prov_small.round(4)))
    lines.append("\n\n### Basic provenance counts\n\n")
    lines.append("```json\n")
    lines.append(json.dumps(prov_basic, ensure_ascii=False, indent=2))
    lines.append("\n```\n")

    lines.append("\n## 输出文件\n\n")
    lines.append("- `tables/lightweight_nmf_assignment_metrics.csv` 和 summary 表：NMF replay 稳定性。\n")
    lines.append("- `tables/p2_purity_*`：P2 epithelial/ambient 审计。\n")
    lines.append("- `tables/program_patient_entropy_cnv_summary.csv`：每个 program 的患者混合与 CNV/entropy 汇总。\n")
    lines.append("- `figures/`：对应补充图。\n")
    lines.append("- `logs/input_manifest.json`：输入文件 provenance manifest。\n")

    (OUT / "supplemental_cnmf_sensitivity_purity_provenance_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    ensure_out()

    mal = ad.read_h5ad(MALIGNANT_H5AD, backed="r")
    try:
        mal_obs = mal.obs.copy()
        mal_obs.index = mal_obs.index.astype(str)
    finally:
        mal.file.close()

    nmf = None
    if not args.skip_nmf:
        nmf = run_lightweight_nmf_sensitivity(args, mal_obs)
    purity = run_p2_purity_audit(mal_obs)
    prov = run_provenance_audit(mal_obs)
    make_figures(nmf, purity, prov)
    write_report(args, nmf, purity, prov)

    print("Wrote supplemental audit to results/cnmf_sensitivity (configured output root).")


if __name__ == "__main__":
    main()
