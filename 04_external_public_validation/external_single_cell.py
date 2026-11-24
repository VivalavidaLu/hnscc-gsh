"""Project GSE195832 cells onto six fixed discovery spectra and summarize coherence."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gzip, json
from scipy.io import mmread
from scipy import sparse, stats
from scipy.optimize import nnls


def rank_residual_correlation(df, x, y, covariates):
    d = df[[x, y] + covariates].dropna()
    n = len(d)
    if n < 4 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return dict(n=n, rho=np.nan, p=np.nan, df=np.nan)
    a = d[x].rank().to_numpy()
    b = d[y].rank().to_numpy()
    parts = [np.ones((n, 1))]
    for c in covariates:
        if pd.api.types.is_numeric_dtype(d[c]):
            parts.append(d[c].rank().to_numpy()[:, None])
        else:
            parts.append(pd.get_dummies(d[c], drop_first=True, dtype=float).to_numpy())
    X = np.column_stack(parts)
    rank = np.linalg.matrix_rank(X)
    dfree = n - rank - 1
    a = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    b = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    r = float(np.clip(a @ b / den, -1, 1)) if den > 0 else np.nan
    p = (
        2 * stats.t.sf(abs(r) * np.sqrt(dfree / max(1 - r * r, 1e-15)), dfree)
        if dfree > 0
        else np.nan
    )
    return dict(n=n, rho=r, p=float(p), df=dfree, design_rank=int(rank))


def main():
    parser = argparse.ArgumentParser(
        description="Project GSE195832 cells onto six fixed discovery spectra and summarize coherence by donor and phase."
    )
    parser.add_argument(
        "--matrices",
        required=True,
        type=Path,
        help="Directory of public GSE195832 GSM-prefixed 10x files",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Cell metadata CSV with GSM, barcode, sample tag, cell ID, patient, phase, n_counts, pct_mt, score_group and cnv_status as used in code",
    )
    parser.add_argument(
        "--tpm", required=True, type=Path, help="Discovery cNMF gene_spectra_tpm TSV"
    )
    parser.add_argument(
        "--scale",
        required=True,
        type=Path,
        help="Discovery cNMF scale TSV containing __std",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    meta = pd.read_csv(args.annotations)
    spec = pd.read_csv(args.tpm, sep="\t", index_col=0)
    scaling = pd.read_csv(args.scale, sep="\t", index_col=0)["__std"]
    args.out.mkdir(parents=True, exist_ok=True)

    def save(table, name):
        table.to_csv(args.out / name, index=False)

    PLAN = json.loads(
        Path(__file__).with_name("signature_definitions.json").read_text()
    )
    wanted = list(
        dict.fromkeys(
            list(spec.columns) + PLAN["GSH"] + PLAN["AA"] + ["CXCL9", "CXCL10"]
        )
    )
    frames = []
    exprs = []
    coverage = []
    geneorder = None
    for gsm, d in meta.groupby("gsm", sort=True):
        tag = d.sample_tag.iloc[0]
        prefix = f"{gsm}_{tag}"
        print("Reading", gsm, tag, "selected cells", len(d), flush=True)
        with gzip.open(args.matrices / f"{prefix}features.tsv.gz", "rt") as f:
            genes = [line.rstrip().split("\t")[1] for line in f]
        with gzip.open(args.matrices / f"{prefix}barcodes.tsv.gz", "rt") as f:
            barcodes = [line.strip() for line in f]
        raw = mmread(args.matrices / f"{prefix}matrix.mtx.gz").tocsr().T.tocsr()
        assert raw.shape == (len(barcodes), len(genes))
        ids = pd.Index(barcodes).get_indexer(d.barcode)
        assert (ids >= 0).all()
        counts = raw[ids]
        del raw
        depth = np.asarray(counts.sum(axis=1)).ravel()
        assert np.allclose(depth, d.n_counts)
        unique = pd.Index(genes).drop_duplicates()
        codes = unique.get_indexer(genes)
        mapper = sparse.csr_matrix(
            (np.ones(len(codes)), (np.arange(len(codes)), codes)),
            shape=(len(codes), len(unique)),
        )
        counts = (counts @ mapper).tocsr()
        genes = list(unique)
        keep = [g for g in wanted if g in unique]
        if geneorder is None:
            geneorder = keep
        assert keep == geneorder
        subset = counts[:, unique.get_indexer(keep)].toarray().astype(float)
        log = np.log1p(subset / depth[:, None] * 1e4)
        exprs.append(pd.DataFrame(log, index=d.cell_id, columns=keep))
        shared = [g for g in spec.columns if g in unique]
        idx = pd.Index(keep).get_indexer(shared)
        E = subset[:, idx]
        s = scaling.loc[shared].to_numpy()
        H = spec[shared].to_numpy().T / s[:, None]
        E = E / E.sum(axis=1)[:, None] * 1e6 / s
        U = np.array([nnls(H, row)[0] for row in E])
        assert np.all(U.sum(axis=1) > 0)
        fit_error = np.linalg.norm(E - U @ H.T, axis=1) / np.linalg.norm(E, axis=1)
        U /= U.sum(axis=1)[:, None]
        for j in range(6):
            d[f"P{j + 1}_usage"] = U[:, j]
        d["dominant_program"] = U.argmax(axis=1) + 1
        d["relative_reconstruction_error"] = fit_error
        d["CNV_supported"] = d.cnv_status.isin(
            ["CNV_malignant_like", "CNV_high_conf_malignant"]
        )
        d["CNV_high"] = d.cnv_status.eq("CNV_high_conf_malignant")
        d["log_depth"] = np.log1p(depth)
        frames.append(d)
        coverage.append(
            dict(
                gsm=gsm,
                patient=d.patient.iloc[0],
                phase=d.phase.iloc[0],
                shared_spectrum_genes=len(shared),
                total_spectrum_genes=len(spec.columns),
                top50_present=len(set(PLAN["primary_signature"]) & set(genes)),
            )
        )
        del counts, subset, E, H, U
    cells = pd.concat(frames).set_index("cell_id", drop=False)
    expr = pd.concat(exprs).loc[cells.index]
    sig = PLAN["primary_signature"]
    gsh = PLAN["GSH"]
    aa = PLAN["AA"]
    # Score parameters computed once on the epithelial compartment, no outcome labels used.
    epi = cells.score_group.eq("epithelial_like")
    mu = expr.loc[epi].mean()
    sd = expr.loc[epi].std(ddof=0).replace(0, np.nan)
    zz = (expr - mu) / sd
    sets = {
        "P1_top50": sig,
        "P1_half_odd": sig[::2],
        "P1_half_even": sig[1::2],
        "GSH": gsh,
        "AA": aa,
        "GSH_nonoverlap": [g for g in gsh if g not in sig],
        "AA_nonoverlap": [g for g in aa if g not in sig],
    }
    for label, gs in sets.items():
        cells[label] = zz[[g for g in gs if g in zz]].mean(axis=1)
    save(pd.DataFrame(coverage), "single_cell_signature_coverage.csv")
    selected = cells[cells.CNV_supported].copy()
    summaries = []
    corrs = []
    profiles = []
    for (patient, phase), dall in cells.groupby(["patient", "phase"], sort=True):
        d = dall[dall.CNV_supported]
        row = dict(
            patient=patient,
            phase=phase,
            gsm=dall.gsm.iloc[0],
            n_epithelial=int(dall.score_group.eq("epithelial_like").sum()),
            n_CNV_supported=len(d),
            n_CNV_high=int(d.CNV_high.sum()),
            evaluable=len(d) >= 30,
        )
        for score in ["P1_usage", "P1_top50"]:
            row[score + "_mean"] = d[score].mean()
            row[score + "_median"] = d[score].median()
            row[score + "_q25"] = d[score].quantile(0.25)
            row[score + "_q75"] = d[score].quantile(0.75)
        row["P1_dominant_cells"] = int(d.dominant_program.eq(1).sum())
        row["P1_dominant_fraction"] = d.dominant_program.eq(1).mean()
        summaries.append(row)
        if len(d) < 30:
            continue
        for subset, dd in [("CNV_supported", d), ("CNV_high", d[d.CNV_high])]:
            if len(dd) < 30:
                continue
            for x, y in [
                ("P1_half_odd", "P1_half_even"),
                ("P1_usage", "P1_top50"),
                ("P1_top50", "GSH_nonoverlap"),
                ("P1_top50", "AA_nonoverlap"),
            ]:
                for label, covs in [
                    ("unadjusted", []),
                    ("depth_mt_adjusted", ["log_depth", "pct_mt"]),
                ]:
                    est = rank_residual_correlation(dd, x, y, covs)
                    # Within-sample correlations describe program coherence, not donor-level inferential tests.
                    est.pop("p", None)
                    corrs.append(
                        dict(
                            patient=patient,
                            phase=phase,
                            subset=subset,
                            x=x,
                            y=y,
                            model=label,
                            **est,
                        )
                    )
        for gene in sig:
            if gene in zz:
                profiles.append(
                    dict(
                        patient=patient,
                        phase=phase,
                        gene=gene,
                        mean_z=float(zz.loc[d.index, gene].mean()),
                        detected_fraction=float((expr.loc[d.index, gene] > 0).mean()),
                    )
                )
    save(pd.DataFrame(summaries), "donor_projection.csv")
    save(pd.DataFrame(corrs), "donor_coherence.csv")


if __name__ == "__main__":
    main()
