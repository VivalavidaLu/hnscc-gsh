"""Test patient-level immune associations, matched phase changes and pathological response."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
from itertools import permutations, product
from scipy import stats
from statsmodels.stats.multitest import multipletests
from patient_statistics import compare as response_compare

PRIMARY = ["CD8_cytotoxicity", "CD8_CXCR3_fraction", "CD8_CXCR3pos_cytotoxicity"]

SECONDARY = [
    "CD8_activation",
    "CD4_activation",
    "NK_activation",
    "CD8_proliferation",
    "CD8_exhaustion_related",
    "CD4_CXCR3_fraction",
    "NK_CXCR3_fraction",
]

METRICS = PRIMARY + SECONDARY

EXPOSURES = ["P1_fraction", "malignant_CXCL10"]


def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def bh(d, groups):
    d["BH_q"] = np.nan
    for _, idx in d.groupby(groups, dropna=False).groups.items():
        vals = d.loc[idx, "P"].fillna(1).to_numpy()
        q = multipletests(vals, method="fdr_bh")[1]
        d.loc[idx, "BH_q"] = np.where(d.loc[idx, "P"].notna(), q, np.nan)
    return d


def spearman(a, b, key, infer=True):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    n = len(a)
    row = {
        "n": n,
        "rho": np.nan,
        "P": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "bootstrap_valid": 0,
        "status": "fewer_than_6_patients",
    }
    if n < 3 or min(len(np.unique(a)), len(np.unique(b))) < 2:
        row["status"] = "insufficient_or_constant"
        return row
    ar = stats.rankdata(a)
    br = stats.rankdata(b)
    ar -= ar.mean()
    br -= br.mean()
    den = np.linalg.norm(ar) * np.linalg.norm(br)
    rho = float(ar @ br / den)
    row["rho"] = rho
    assert np.isclose(rho, stats.spearmanr(a, b).statistic)
    if not infer:
        row["status"] = "descriptive_only_no_P_value"
        return row
    if n < 6:
        return row
    # Identical patient rank vectors must yield identical Monte Carlo P values
    # across sensitivity labels; this prevents simulation noise masquerading as
    # a threshold effect when the eligible patients do not actually change.
    rng = np.random.default_rng(seed(ar.tobytes().hex() + br.tobytes().hex()))
    if n <= 8:
        perm = np.array(list(permutations(br)), dtype=float)
        null = perm @ ar / den
        row["P"] = float(np.mean(abs(null) >= abs(rho) - 1e-12))
        row["permutations"] = len(null)
        row["status"] = "exact_patient_permutation"
    else:
        exceed = 0
        B = 99999
        for start in range(0, B, 10000):
            q = min(10000, B - start)
            v = rng.permuted(np.tile(br, (q, 1)), axis=1) @ ar / den
            exceed += int(np.sum(abs(v) >= abs(rho) - 1e-12))
        row["P"] = (exceed + 1) / (B + 1)
        row["permutations"] = B
        row["status"] = "Monte_Carlo_patient_permutation"
    ix = rng.integers(0, n, size=(10000, n))
    ax = stats.rankdata(a[ix], axis=1)
    bx = stats.rankdata(b[ix], axis=1)
    ax -= ax.mean(axis=1, keepdims=True)
    bx -= bx.mean(axis=1, keepdims=True)
    bot = np.linalg.norm(ax, axis=1) * np.linalg.norm(bx, axis=1)
    v = np.sum(ax * bx, axis=1) / np.where(bot == 0, np.nan, bot)
    v = v[np.isfinite(v)]
    row["bootstrap_valid"] = len(v)
    if len(v):
        row["CI95_low"], row["CI95_high"] = np.quantile(v, [0.025, 0.975])
    return row


def corr_grid(d, label, metrics=METRICS, exposures=EXPOSURES):
    rows = []
    for metric in metrics:
        for exposure in exposures:
            dd = d[["Patient", metric, exposure]].dropna()
            assert dd.Patient.is_unique
            result = spearman(dd[exposure], dd[metric], label + metric + exposure)
            rows.append(
                {
                    "analysis": label,
                    "family": "primary" if metric in PRIMARY else "secondary",
                    "metric": metric,
                    "exposure": exposure,
                    "patients": ";".join(dd.Patient),
                    **result,
                }
            )
    return bh(pd.DataFrame(rows), ["analysis", "family"])


def deltas(d):
    pre = d[d.Phase.eq("pre")].set_index("Patient")
    post = d[d.Phase.eq("post")].set_index("Patient")
    ix = pre.index.intersection(post.index)
    if not pre.loc[ix, "Response"].equals(post.loc[ix, "Response"]):
        raise ValueError("Patient response labels must agree across treatment phases.")
    result = pd.DataFrame(
        {"Patient": ix, "Response": pre.loc[ix, "Response"].to_numpy()}
    )
    for metric in METRICS + EXPOSURES + ["P1_usage", "malignant_CXCL10_fraction"]:
        result[metric] = (
            post.loc[ix, metric].to_numpy() - pre.loc[ix, metric].to_numpy()
        )
    result["pre_protocols"] = pre.loc[ix, "protocols"].to_numpy()
    result["post_protocols"] = post.loc[ix, "protocols"].to_numpy()
    return result


def responses(base, delta):
    rows = []
    for label, df in [
        ("baseline", base[base.Phase.eq("pre")]),
        ("post", base[base.Phase.eq("post")]),
        ("delta_technical_matched", delta),
    ]:
        for metric in METRICS:
            a = df.loc[df.Response.eq("R"), metric].dropna().to_numpy()
            b = df.loc[df.Response.eq("NR"), metric].dropna().to_numpy()
            rr = response_compare(a, b)
            rows.append(
                {
                    "analysis": label,
                    "metric": metric,
                    "R_patients": ";".join(
                        df.loc[df.Response.eq("R") & df[metric].notna(), "Patient"]
                    ),
                    "NR_patients": ";".join(
                        df.loc[df.Response.eq("NR") & df[metric].notna(), "Patient"]
                    ),
                    **rr,
                    "P": rr["exact_P"],
                }
            )
    d = pd.DataFrame(rows)
    d["family"] = "response_all_30"
    return bh(d, ["family"])


def main():
    parser = argparse.ArgumentParser(
        description="Test patient-level immune associations, matched phase changes and pathological response."
    )
    parser.add_argument(
        "--phase",
        required=True,
        type=Path,
        help="Patient-phase CSV from aggregate_immune_scores.py",
    )
    parser.add_argument(
        "--matched",
        required=True,
        type=Path,
        help="Matched-protocol patient-phase CSV from the same script",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    base = pd.read_csv(args.phase)
    delta = deltas(pd.read_csv(args.matched))
    args.out.mkdir(parents=True, exist_ok=True)
    corr_grid(base[base.Phase.eq("post")], "post_primary_definition").drop(
        columns="patients"
    ).to_csv(args.out / "post_associations.csv", index=False)
    responses(base, delta).drop(columns=["R_patients", "NR_patients"]).to_csv(
        args.out / "response_comparisons.csv", index=False
    )
    stage = []
    for metric in METRICS:
        dd = delta.loc[delta[metric].notna(), ["Patient", metric]]
        a = dd[metric].to_numpy()
        n = len(a)
        row = {
            "metric": metric,
            "n_pairs": n,
            "median_change": float(np.median(a)) if n else np.nan,
            "mean_change": float(np.mean(a)) if n else np.nan,
            "P": np.nan,
            "patients": ";".join(dd.Patient),
            "status": "descriptive_fewer_than_6_pairs",
        }
        if n >= 6:
            null = np.array(list(product([-1, 1], repeat=n))) @ a / n
            obs = abs(a.mean())
            row.update(
                P=float(np.mean(abs(null) >= obs - 1e-12)),
                status="exact_patient_sign_flip",
            )
        stage.append(row)
    stage = pd.DataFrame(stage)
    stage["family"] = "paired_stage"
    bh(stage, ["family"]).drop(columns="patients").to_csv(
        args.out / "paired_stage.csv", index=False
    )


if __name__ == "__main__":
    main()
