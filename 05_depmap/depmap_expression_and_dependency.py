"""Reproduce the DepMap Public 24Q2 P1/GSH dependency analysis.

The script downloads the pinned public release, derives P1 expression scores
from the included cNMF K=6 spectra, retains HNSCC models identified by
DepMap metadata, and evaluates CRISPR dependency associations.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.stats.multitest import multipletests


REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HNSCC_DATA_DIR", REPOSITORY / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("HNSCC_RESULTS_DIR", REPOSITORY / "results")).resolve()
CACHE = DATA_ROOT / "raw" / "depmap_24q2"
OUT = RESULTS_ROOT / "depmap"
SPECTRA = DATA_ROOT / "derived" / "cNMF_K6_gene_spectra_score.txt"

FILES = {
    "Model.csv": {
        "url": "https://ndownloader.figshare.com/files/46489732",
        "size": 559182,
        "md5": "e5b60d1ce7636542d93f45494019eded",
    },
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": {
        "url": "https://ndownloader.figshare.com/files/46490878",
        "size": 460868099,
        "md5": "f00c434db50d811544ac4d047003979f",
    },
    "CRISPRGeneEffect.csv": {
        "url": "https://ndownloader.figshare.com/files/46489063",
        "size": 419330454,
        "md5": "d10d73692672380b3792231549e97d5a",
    },
}
GSH_GENES = ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"]
TARGETS = ["GCLC", "GCLM", "GSS", "SLC7A11", "GPX4"]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str) -> Path:
    if name not in FILES:
        raise ValueError(f"Unrecognized DepMap file: {name}")
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / name
    metadata = FILES[name]
    expected_size = int(metadata["size"])
    expected_md5 = str(metadata["md5"])
    if destination.exists() and destination.stat().st_size == expected_size:
        if md5sum(destination) == expected_md5:
            return destination
    if destination.exists():
        destination.unlink()
    with requests.get(str(metadata["url"]), stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_content(4 * 1024 * 1024):
                if block:
                    handle.write(block)
    observed_md5 = md5sum(destination)
    if destination.stat().st_size != expected_size or observed_md5 != expected_md5:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Integrity check failed for {name}")
    return destination


def gene_symbol(column: str) -> str:
    return re.sub(r"\s+\(\d+\)$", "", str(column))


def p1_definitions() -> tuple[list[str], pd.Series]:
    spectra = pd.read_csv(SPECTRA, sep="\t", index_col=0)
    row = spectra.loc[1] if 1 in spectra.index else spectra.loc["1"]
    ranked = row.sort_values(ascending=False)
    ranked = ranked[~ranked.index.str.match(r"^(MT-|RPL|RPS)")]
    return ranked.head(50).index.tolist(), ranked.head(200)


def read_expression(path: Path, genes: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    mapping: dict[str, str] = {}
    for column in header[1:]:
        mapping.setdefault(gene_symbol(column), column)
    present = [gene for gene in genes if gene in mapping]
    frame = pd.read_csv(path, usecols=[header[0], *[mapping[g] for g in present]])
    return frame.rename(columns={header[0]: "ModelID", **{mapping[g]: g for g in present}}).set_index("ModelID")


def read_crispr_targets(path: Path) -> pd.DataFrame:
    return read_expression(path, TARGETS)


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)


def build_scores(expression: pd.DataFrame, top50: list[str], weighted: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_columns(expression)
    top50_used = [gene for gene in top50 if gene in z]
    weighted_used = [gene for gene in weighted.index if gene in z]
    weights = weighted.loc[weighted_used].clip(lower=0)
    weights = weights / weights.sum()
    scores = pd.DataFrame(index=z.index)
    scores["P1_top50"] = z[top50_used].mean(axis=1)
    scores["P1_weighted"] = z[weighted_used].mul(weights, axis=1).sum(axis=1)
    present_gsh = [gene for gene in GSH_GENES if gene in z]
    scores["GSH_cystine"] = z[present_gsh].mean(axis=1)
    for target in TARGETS:
        present = [gene for gene in GSH_GENES if gene != target and gene in z]
        scores[f"GSH_without_{target}"] = z[present].mean(axis=1)
    coverage = pd.DataFrame([
        {"signature": "P1_top50", "requested": 50, "used": len(top50_used), "genes": ";".join(top50_used)},
        {"signature": "P1_weighted", "requested": 200, "used": len(weighted_used), "genes": ";".join(weighted_used)},
        {"signature": "GSH_cystine", "requested": len(GSH_GENES), "used": len(present_gsh), "genes": ";".join(present_gsh)},
    ])
    return scores, coverage


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signature in ["P1_top50", "P1_weighted", "GSH_cystine"]:
        for target in TARGETS:
            pair = frame[[signature, f"{target}_dependency_strength"]].dropna()
            rho, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append({"signature": signature, "dependency_gene": target, "n": len(pair), "rho": rho, "p_value": p_value})
    for target in TARGETS:
        signature = f"GSH_without_{target}"
        pair = frame[[signature, f"{target}_dependency_strength"]].dropna()
        rho, p_value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append({"signature": signature, "dependency_gene": target, "n": len(pair), "rho": rho, "p_value": p_value})
    result = pd.DataFrame(rows)
    result["BH_FDR_20_tests"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def leave_one_model_out(frame: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test in tests.itertuples(index=False):
        x = test.signature
        y = f"{test.dependency_gene}_dependency_strength"
        pair = frame[[x, y]].dropna()
        for model_id in pair.index:
            subset = pair.drop(index=model_id)
            rho, p_value = stats.spearmanr(subset[x], subset[y])
            rows.append({
                "signature": x,
                "dependency_gene": test.dependency_gene,
                "omitted_model": model_id,
                "n": len(subset),
                "rho": rho,
                "p_value": p_value,
            })
    return pd.DataFrame(rows)


def main() -> None:
    model_path = download("Model.csv")
    expression_path = download("OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    crispr_path = download("CRISPRGeneEffect.csv")
    top50, weighted = p1_definitions()
    requested = list(dict.fromkeys([*weighted.index, *GSH_GENES]))
    expression = read_expression(expression_path, requested)
    scores, coverage = build_scores(expression, top50, weighted)

    models = pd.read_csv(model_path, low_memory=False)
    models["is_hnscc"] = models["OncotreePrimaryDisease"].fillna("").eq("Head and Neck Squamous Cell Carcinoma")
    metadata = models.set_index("ModelID")
    hnscc_ids = metadata.index[metadata["is_hnscc"]]

    gene_effect = read_crispr_targets(crispr_path)
    dependency = (-gene_effect).add_suffix("_dependency_strength")
    common = scores.index.intersection(dependency.index).intersection(hnscc_ids)
    if len(common) != 57:
        raise RuntimeError(f"Expected 57 HNSCC models in the pinned release, found {len(common)}")
    analysis = scores.loc[common].join(dependency.loc[common])
    meta_columns = [c for c in ["CellLineName", "CCLEName", "OncotreeSubtype"] if c in metadata]
    analysis = analysis.join(metadata.loc[common, meta_columns])

    tests = correlations(analysis)
    loo = leave_one_model_out(analysis, tests)
    OUT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT / "signature_gene_coverage.csv", index=False)
    analysis.to_csv(OUT / "hnscc_model_scores_and_dependencies.csv", index_label="ModelID")
    tests.to_csv(OUT / "signature_dependency_correlations.csv", index=False)
    loo.to_csv(OUT / "leave_one_model_out_correlations.csv", index=False)
    summary = {
        "release": "DepMap Public 24Q2",
        "hnscc_models": len(common),
        "tests": len(tests),
        "p1_weighted_vs_gclc": tests.query("signature == 'P1_weighted' and dependency_gene == 'GCLC'").iloc[0].to_dict(),
        "gsh_without_gclc_vs_gclc": tests.query("signature == 'GSH_without_GCLC' and dependency_gene == 'GCLC'").iloc[0].to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
