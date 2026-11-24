"""Evaluate equal-patient metabolic contrasts and patient-level immune associations after LSCC21 exclusion."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import anndata as ad
from scipy import stats, sparse
from scipy.optimize import nnls
from cnmf.cnmf import load_df_from_npz
from statsmodels.stats.multitest import multipletests

EXPECTED_POST_TREATMENT_DONORS_AFTER_LSCC21_EXCLUSION = 12

GENESETS = {
    "GSH_cystine": ["SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR", "GPX4"],
    "AA_transport": ["SLC1A5", "SLC7A5", "SLC38A1", "SLC38A2", "SLC1A4", "SLC7A8"],
    "NRF2_core": ["NFE2L2", "KEAP1", "MAFK", "MAFG", "BACH1"],
    "phase2_detox": [
        "NQO1",
        "AKR1C1",
        "AKR1C2",
        "AKR1B10",
        "ALDH3A1",
        "GSTA4",
        "GSTM1",
        "GSTP1",
    ],
    "thioredoxin_PRDX": ["TXN", "TXNRD1", "PRDX1", "SRXN1"],
    "heme_iron": ["HMOX1", "FTH1", "FTL", "SLC40A1", "TFRC"],
}


def spectrum(run, name, kind="score", k=6):
    df = pd.read_csv(
        run / f"{name}.gene_spectra_{kind}.k_{k}.dt_2_0.txt", sep="\t", index_col=0
    )
    df.index = df.index.astype(int)
    return df


def dcohen(a, b):
    v = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (
        len(a) + len(b) - 2
    )
    return (np.mean(a) - np.mean(b)) / np.sqrt(v) if v > 0 else np.nan


def corr_ci(x, y):
    a = np.c_[x, y]
    a = a[np.isfinite(a).all(axis=1)]
    n = len(a)
    result = dict(n_patients=n, rho=np.nan, CI_low=np.nan, CI_high=np.nan, P=np.nan)
    if n < 4 or min(len(np.unique(a[:, 0])), len(np.unique(a[:, 1]))) < 2:
        return result
    rng = np.random.default_rng(301741)
    rx = stats.rankdata(a[:, 0])
    ry = stats.rankdata(a[:, 1])
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    rho = float(rx @ ry / den)
    perm = np.array([rng.permutation(ry) for _ in range(19999)]) @ rx / den
    p = (1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / 20000
    ix = rng.integers(0, n, size=(5000, n))
    bx = stats.rankdata(a[ix, 0], axis=1)
    by = stats.rankdata(a[ix, 1], axis=1)
    bx -= bx.mean(axis=1)[:, None]
    by -= by.mean(axis=1)[:, None]
    denom = np.sqrt(np.sum(bx * bx, axis=1) * np.sum(by * by, axis=1))
    ok = denom > 0
    boot = np.sum(bx[ok] * by[ok], axis=1) / denom[ok]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    result.update(rho=rho, CI_low=float(lo), CI_high=float(hi), P=float(p))
    return result


def fdr(frame, groups):
    frame["BH_q"] = np.nan
    for _, sub in frame.groupby(groups, dropna=False):
        ids = sub.index[sub.P.notna()]
        if len(ids):
            frame.loc[ids, "BH_q"] = multipletests(
                frame.loc[ids, "P"], method="fdr_bh"
            )[1]
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate equal-patient metabolic contrasts and patient-level immune associations after LSCC21 exclusion."
    )
    parser.add_argument(
        "--counts",
        required=True,
        type=Path,
        help="Malignant full-library raw-count AnnData; Patient, TreatmentPhase, nCount_RNA and cNMF_program in obs",
    )
    parser.add_argument(
        "--run", required=True, type=Path, help="cNMF output subdirectory for one refit"
    )
    parser.add_argument(
        "--program",
        required=True,
        type=Path,
        help="Matched program number from match_spectra.py",
    )
    parser.add_argument(
        "--immune",
        required=True,
        type=Path,
        help="Post-treatment CSV with Patient, tnk_activation_core_score and macrophage_inflammatory_score",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(args.counts)
    a = a[a.obs.Patient.ne("LSCC21")].copy()
    meta = a.obs.copy()
    meta["Phase"] = meta.TreatmentPhase.astype(str).str.lower()
    allgenes = sorted(set(sum(GENESETS.values(), []) + ["CXCL9", "CXCL10"]))
    exp = np.log1p(
        a[:, allgenes].X.toarray() / meta.nCount_RNA.to_numpy()[:, None] * 1e4
    )
    expr = pd.DataFrame(exp, index=meta.index, columns=allgenes)
    for name, genes in GENESETS.items():
        meta[name] = expr[genes].mean(axis=1)
    for gene in ["SLC7A11", "GCLC", "CXCL9", "CXCL10"]:
        meta[gene] = expr[gene]
    meta["CXCL10_positive"] = expr.CXCL10.gt(0).astype(float)

    run = args.run
    spec = spectrum(run, run.name, "tpm")
    features = spec.columns
    x = a[:, features].X
    x = (x.toarray() if sparse.issparse(x) else np.asarray(x)).astype(float)
    scale = (
        load_df_from_npz(run / "cnmf_tmp" / f"{run.name}.tpm_stats.df.npz")
        .loc[features, "__std"]
        .to_numpy()
    )
    x = x / x.sum(axis=1)[:, None] * 1e6 / scale
    basis = spec.to_numpy().T / scale[:, None]
    weights = np.array([nnls(basis, row)[0] for row in x])
    u = pd.DataFrame(
        weights / weights.sum(1)[:, None], index=a.obs_names, columns=spec.index
    )
    analyses = {run.name + "_native": (u, int(args.program.name))}
    full_cohort_labels = (
        meta.cNMF_program.astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    immune = (
        pd.read_csv(args.immune)
        .set_index("Patient")
        .drop(index="LSCC21", errors="ignore")
    )

    def save(frame, name):
        frame.to_csv(args.out / name, index=False)

    outcomes = ["tnk_activation_core_score", "macrophage_inflammatory_score"]
    metab = []
    mdonor = []
    imrows = []
    postrows = []
    phaseall = []
    chem = []
    concord = []
    for model, (u, p1) in analyses.items():
        x = meta.copy()
        x["P1"] = u.idxmax(axis=1).eq(p1)
        x["P1_usage"] = u[p1]
        x["P1_fraction"] = x.P1.astype(float)
        z = expr.loc[x.P1, GENESETS["GSH_cystine"]]
        sd = z.std(ddof=0).replace(0, np.nan)
        x.loc[x.P1, "GSH_P1_z"] = ((z - z.mean()) / sd).mean(axis=1)

        concord.append(
            dict(
                model=model,
                matched_program=p1,
                P1_cells=int(x.P1.sum()),
                all_cells=len(x),
                patients_with_P1=int(x[x.P1].Patient.nunique()),
                p1_membership_agreement_with_full_cohort=float(
                    np.mean(x.P1.to_numpy() == full_cohort_labels.eq(1).to_numpy())
                ),
            )
        )
        for name in GENESETS:
            ds = []
            for pt, g in x.groupby("Patient", observed=True):
                v = g.loc[g.P1, name].to_numpy()
                w = g.loc[~g.P1, name].to_numpy()
                d = dcohen(v, w) if min(len(v), len(w)) >= 10 else np.nan
                mdonor.append(
                    dict(
                        model=model,
                        Patient=pt,
                        branch=name,
                        n_P1=len(v),
                        n_other=len(w),
                        cohens_d=d,
                    )
                )
                if np.isfinite(d):
                    ds.append(d)
            ds = np.array(ds)
            rng = np.random.default_rng(301741)
            if len(ds) >= 3:
                boot = np.mean(
                    ds[rng.integers(0, len(ds), size=(5000, len(ds)))], axis=1
                )
                lo, hi = np.quantile(boot, [0.025, 0.975])
                signs = np.array(list(itertools.product([-1, 1], repeat=len(ds))))
                null = np.mean(signs * ds, axis=1)
                p = float(np.mean(np.abs(null) >= abs(ds.mean()) - 1e-12))
            else:
                lo = hi = p = np.nan
            metab.append(
                dict(
                    model=model,
                    branch=name,
                    n_patients=len(ds),
                    n_positive=int(np.sum(ds > 0)),
                    mean_d=float(ds.mean()) if len(ds) else np.nan,
                    CI_low=lo,
                    CI_high=hi,
                    P=p,
                )
            )
        post = (
            x[x.Phase.eq("post")]
            .groupby("Patient", observed=True)
            .agg(
                n_malignant=("P1", "size"),
                n_P1=("P1", "sum"),
                P1_fraction=("P1_fraction", "mean"),
                P1_usage=("P1_usage", "mean"),
            )
            .join(immune[outcomes], validate="one_to_one")
        )
        if len(post) != EXPECTED_POST_TREATMENT_DONORS_AFTER_LSCC21_EXCLUSION:
            raise ValueError(
                f"GSE301741 after LSCC21 exclusion: expected "
                f"{EXPECTED_POST_TREATMENT_DONORS_AFTER_LSCC21_EXCLUSION} "
                f"post-treatment donors, found {len(post)}."
            )
        if post[outcomes].isna().any().any():
            raise ValueError("Immune outcomes are missing for post-treatment donors.")
        post.insert(0, "model", model)
        postrows.append(post.reset_index())
        for exposure in ["P1_fraction", "P1_usage"]:
            for outcome in outcomes:
                imrows.append(
                    dict(
                        model=model,
                        exposure=exposure,
                        outcome=outcome,
                        **corr_ci(post[exposure], post[outcome]),
                    )
                )
        phase = []
        for (pt, ph), g in x.groupby(["Patient", "Phase"], observed=True):
            sub = g[g.P1]
            r = dict(
                model=model,
                Patient=pt,
                Phase=ph,
                n_malignant=len(g),
                n_P1=len(sub),
                P1_fraction=g.P1.mean(),
                P1_usage=g.P1_usage.mean(),
            )
            for name in ["GSH_P1_z", "CXCL9", "CXCL10", "CXCL10_positive"]:
                r[name] = sub[name].mean() if len(sub) >= 10 else np.nan
            phase.append(r)
        phase = pd.DataFrame(phase)
        phaseall.append(phase)
        for ph in ["pre", "post"]:
            s = phase[phase.Phase.eq(ph)]
            for outcome in ["CXCL10", "CXCL10_positive", "CXCL9"]:
                chem.append(
                    dict(
                        model=model,
                        phase=ph,
                        outcome=outcome,
                        **corr_ci(s.GSH_P1_z, s[outcome]),
                    )
                )
        print("ASSOCIATIONS", model, "P1 cells", int(x.P1.sum()), flush=True)
    save(pd.DataFrame(concord), "assignment_summary.csv")
    save(pd.DataFrame(mdonor), "metabolic_effects_per_donor.csv")
    save(fdr(pd.DataFrame(metab), ["model"]), "metabolic_associations.csv")
    save(pd.concat(postrows), "post_patient_source_data.csv")
    save(fdr(pd.DataFrame(imrows), ["model", "exposure"]), "immune_associations.csv")
    save(pd.concat(phaseall), "patient_phase_P1_GSH_CXCL10.csv")
    save(fdr(pd.DataFrame(chem), ["model", "phase"]), "chemokine_associations.csv")


if __name__ == "__main__":
    main()
