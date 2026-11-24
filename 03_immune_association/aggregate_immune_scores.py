"""Construct patient-phase and protocol-matched immune summaries from annotated public cells."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


CD8 = ["CD8_Tem", "CD8_Tem_stress", "CD8_Tex", "CD8_T_prolif"]

CD4 = ["CD4_Tn", "Treg", "Treg_resting", "Tfh-like"]


def build(tnk, mal, ni=30, nm=10):
    rows = []
    for (pt, phase), sub in tnk.groupby(["Patient", "Phase"]):
        r = {
            "Patient": pt,
            "Phase": phase,
            "Response": sub.Response.iloc[0],
            "n_TNK_retained": len(sub),
            "protocols": ";".join(sorted(sub.protocol.unique())),
            "n_libraries": sub.SampleID.nunique(),
            "median_UMI": sub.nCount_RNA.median(),
        }
        for lin in ["CD8", "CD4", "NK"]:
            ss = sub[sub.lineage.eq(lin)]
            n = len(ss)
            r[lin + "_n"] = n
            r[lin + "_TNK_fraction"] = n / len(sub)
            r[lin + "_median_UMI"] = ss.nCount_RNA.median() if n else np.nan
            r[lin + "_activation"] = ss.activation.mean() if n >= ni else np.nan
            r[lin + "_CXCR3_n"] = int(ss.CXCR3_detected.sum())
            r[lin + "_CXCR3_fraction"] = ss.CXCR3_detected.mean() if n >= ni else np.nan
            if lin == "CD8":
                for score in ["cytotoxicity", "proliferation", "exhaustion_related"]:
                    r["CD8_" + score] = ss[score].mean() if n >= ni else np.nan
                pos = ss[ss.CXCR3_detected]
                r["CD8_CXCR3pos_cytotoxicity"] = (
                    pos.cytotoxicity.mean() if n >= ni and len(pos) >= 10 else np.nan
                )
                for typ in CD8:
                    tt = ss[ss.cell_subtype.eq(typ)]
                    r[typ + "_n"] = len(tt)
                    r[typ + "_fraction"] = len(tt) / n if n >= ni else np.nan
                    r[typ + "_cytotoxicity"] = (
                        tt.cytotoxicity.mean() if len(tt) >= 10 else np.nan
                    )
        for lin in ["uncertain_T", "uncertain_TNK"]:
            r[lin + "_n"] = int(sub.lineage.eq(lin).sum())
        mm = mal[(mal.Patient == pt) & (mal.Phase == phase)]
        r["n_malignant"] = len(mm)
        r["n_P1"] = int(mm.cNMF_program.eq(1).sum())
        r["P1_fraction_unfiltered"] = r["n_P1"] / len(mm) if len(mm) else np.nan
        r["P1_fraction"] = r["P1_fraction_unfiltered"] if len(mm) >= nm else np.nan
        r["P1_usage"] = mm.P1_usage.mean() if len(mm) >= nm else np.nan
        r["malignant_CXCL10"] = mm.CXCL10_logCP10K.mean() if len(mm) >= nm else np.nan
        r["malignant_CXCL10_fraction"] = (
            mm.CXCL10_detected.mean() if len(mm) >= nm else np.nan
        )
        pm = mm[mm.cNMF_program.eq(1)]
        r["P1_CXCL10"] = pm.CXCL10_logCP10K.mean() if len(pm) >= 10 else np.nan
        r["eligibility_CD8"] = (
            "eligible" if r["CD8_n"] >= ni else f"fewer_than_{ni}_CD8"
        )
        r["eligibility_malignant"] = (
            "eligible" if len(mm) >= nm else f"fewer_than_{nm}_malignant"
        )
        rows.append(r)
    return pd.DataFrame(rows)


def technical_match(tnk, mal):
    maskt = np.zeros(len(tnk), bool)
    maskm = np.zeros(len(mal), bool)
    rows = []
    for pt, sub in tnk.groupby("Patient"):
        pre = set(sub.loc[sub.Phase.eq("pre"), "protocol"])
        post = set(sub.loc[sub.Phase.eq("post"), "protocol"])
        common = pre & post
        rows.append(
            {
                "Patient": pt,
                "pre_protocols": ";".join(sorted(pre)),
                "post_protocols": ";".join(sorted(post)),
                "shared_protocols": ";".join(sorted(common)),
                "included_in_technical_pair": bool(common),
            }
        )
        maskt |= tnk.Patient.eq(pt).to_numpy() & tnk.protocol.isin(common).to_numpy()
        maskm |= mal.Patient.eq(pt).to_numpy() & mal.protocol.isin(common).to_numpy()

    return tnk[maskt].copy(), mal[maskm].copy()


GENE_SETS = {
    "activation": ["IFNG", "TNF", "CD69"],
    "cytotoxicity": ["PRF1", "GZMB", "GZMH", "GNLY", "NKG7"],
    "proliferation": ["MKI67", "TOP2A", "UBE2C", "STMN1"],
    "exhaustion_related": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Construct patient-phase and protocol-matched immune summaries from annotated public cells."
    )
    parser.add_argument(
        "--immune",
        required=True,
        type=Path,
        help="Raw-count TNK AnnData; cell_subtype, Patient, TreatmentPhase, SampleID and nCount_RNA in obs",
    )
    parser.add_argument(
        "--tumor",
        required=True,
        type=Path,
        help="Full-library malignant raw-count AnnData with the same sampling metadata",
    )
    parser.add_argument(
        "--usage",
        required=True,
        type=Path,
        help="Six-program usage CSV indexed by cell ID",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--responses",
        required=True,
        type=Path,
        help="Patient and Response (R or NR) for patients with verified labels; omit unassigned patients.",
    )
    args = parser.parse_args()
    labels_table = pd.read_csv(args.responses, usecols=["Patient", "Response"])
    if (
        labels_table.Patient.duplicated().any()
        or not labels_table.Response.isin(["R", "NR"]).all()
    ):
        raise ValueError(
            "Supply one verified R or NR per patient in the response file."
        )
    response = labels_table.set_index("Patient").Response

    def features(a, genes):
        subset = a[:, genes]
        x = subset.layers["counts"] if "counts" in subset.layers else subset.X
        values = x.data if sparse.issparse(x) else np.asarray(x)
        if np.any(values < 0) or not np.equal(values, np.floor(values)).all():
            raise ValueError("Supply raw, integer counts in layers[counts] or X.")
        return pd.DataFrame(
            x.toarray() if sparse.issparse(x) else x, index=a.obs_names, columns=genes
        )

    immune = ad.read_h5ad(args.immune)
    tumor = ad.read_h5ad(args.tumor)
    tnk = immune.obs.copy()
    tnk["Response"] = tnk.Patient.astype(str).map(response).fillna("unassigned")
    mal = tumor.obs.copy()
    sets = GENE_SETS
    markers = ["CD3D", "CD3E", "TRAC", "KLRD1", "GNLY", "TYROBP", "FCER1G"]
    raw = features(immune, sorted(set(markers + sum(sets.values(), []) + ["CXCR3"])))
    expr = np.log1p(raw.div(tnk.nCount_RNA, axis=0) * 10000)
    for name, genes in sets.items():
        tnk[name] = expr[genes].mean(axis=1)
    tnk["CXCR3_detected"] = raw.CXCR3.gt(0)
    t = raw[markers[:3]].gt(0).sum(axis=1)
    nk = raw[markers[3:]].gt(0).sum(axis=1)
    tnk["lineage"] = np.select(
        [
            tnk.cell_subtype.isin(CD8),
            tnk.cell_subtype.isin(CD4),
            tnk.cell_subtype.eq("NK") & (t < 2) & (nk >= 2),
            tnk.cell_subtype.eq("NK"),
        ],
        ["CD8", "CD4", "NK", "uncertain_TNK"],
        default="uncertain_T",
    )
    usage = pd.read_csv(args.usage, index_col=0).loc[
        mal.index, [str(i) for i in range(1, 7)]
    ]
    mal["P1_usage"] = usage["1"]
    mal["cNMF_program"] = usage.idxmax(axis=1).astype(int)
    cx = features(tumor, ["CXCL10"]).iloc[:, 0]
    mal["CXCL10_logCP10K"] = np.log1p(cx / mal.nCount_RNA * 10000)
    mal["CXCL10_detected"] = cx.gt(0)
    for df in [tnk, mal]:
        df["Phase"] = df.TreatmentPhase.astype(str).str.lower()
        df["protocol"] = (
            df.SampleID.str.extract(
                r"-(CD45neg|CD45pos|CD45ratio|CD3)-", expand=False
            ).fillna("unspecified")
            + "_"
            + df.SampleID.str.extract(r"-(3|5)$", expand=False).fillna("unknown")
        )
    args.out.mkdir(parents=True, exist_ok=True)
    build(tnk, mal).to_csv(args.out / "patient_phase.csv", index=False)
    build(*technical_match(tnk, mal)).to_csv(
        args.out / "matched_patient_phase.csv", index=False
    )


if __name__ == "__main__":
    main()
