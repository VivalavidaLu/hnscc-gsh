"""Test response, progression-free and overall survival per SD of P1 score."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests


def main():
    parser = argparse.ArgumentParser(
        description="Test response, progression-free and overall survival per SD of P1 score."
    )
    parser.add_argument(
        "--scores",
        required=True,
        type=Path,
        help="GSE159067 score table with clinical fields in DATA_SOURCES.md",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Output CSV of ORs and HRs"
    )
    args = parser.parse_args()
    rows = []
    ph = []
    modelterms = []
    diagnostics = []
    df = pd.read_csv(args.scores, dtype={"HPV": str})
    df["Age"] = pd.to_numeric(df.Age)
    df["line"] = pd.to_numeric(df["immunotherapy line"])
    assert df[["OS_time", "PFS_time"]].gt(0).all().all()
    assert set(df.OS_event.unique()) <= {0, 1} and set(df.PFS_event.unique()) <= {0, 1}
    for score in ["P1_top50", "P1_top100"]:
        use = df[df.Response.isin(["CR", "PR", "SD", "PD"])].copy()
        y = use.Response.isin(["CR", "PR"]).astype(int)
        model = sm.GLM(
            y, sm.add_constant(use[[score]]), family=sm.families.Binomial()
        ).fit()
        b = model.params[score]
        lo, hi = model.conf_int().loc[score]
        rows.append(
            dict(
                signature=score,
                endpoint="RECIST_response",
                model="unadjusted",
                effect="OR",
                n=len(use),
                events=int(y.sum()),
                estimate=float(np.exp(b)),
                ci_low=float(np.exp(lo)),
                ci_high=float(np.exp(hi)),
                p=float(model.pvalues[score]),
            )
        )
        diagnostics.append(
            dict(
                signature=score,
                endpoint="RECIST_response",
                model="unadjusted",
                converged=bool(model.converged),
                deviance=float(model.deviance),
            )
        )
        for endpoint in ["PFS", "OS"]:
            for label, covs in [
                ("unadjusted", []),
                ("clinical_adjusted", ["Age", "HPV", "line"]),
                (
                    "clinical_composition_adjusted",
                    ["Age", "HPV", "line", "epithelial", "immune", "stromal"],
                ),
            ]:
                vars = [score] + covs + [endpoint + "_time", endpoint + "_event"]
                d = df[vars].dropna()
                d = (
                    pd.get_dummies(d, columns=["HPV"], drop_first=True, dtype=float)
                    if "HPV" in covs
                    else d
                )
                fit = CoxPHFitter()
                fit.fit(
                    d, duration_col=endpoint + "_time", event_col=endpoint + "_event"
                )
                r = fit.summary.loc[score]
                rows.append(
                    dict(
                        signature=score,
                        endpoint=endpoint,
                        model=label,
                        effect="HR",
                        n=len(d),
                        events=int(d[endpoint + "_event"].sum()),
                        estimate=r["exp(coef)"],
                        ci_low=r["exp(coef) lower 95%"],
                        ci_high=r["exp(coef) upper 95%"],
                        p=r["p"],
                    )
                )
                test = proportional_hazard_test(fit, d, time_transform="rank").summary
                for term, rr in test.iterrows():
                    ph.append(
                        dict(
                            signature=score,
                            endpoint=endpoint,
                            model=label,
                            term=term,
                            test="rank_time_schoenfeld",
                            statistic=rr["test_statistic"],
                            p=rr["p"],
                        )
                    )
                for term, rr in fit.summary.iterrows():
                    modelterms.append(
                        dict(
                            signature=score,
                            endpoint=endpoint,
                            model=label,
                            term=term,
                            coef=rr["coef"],
                            se=rr["se(coef)"],
                            HR=rr["exp(coef)"],
                            p=rr["p"],
                        )
                    )
                # Independent implementation check, same unadjusted model and Efron tie handling.
                if label == "unadjusted":
                    other = sm.PHReg(
                        d[endpoint + "_time"],
                        d[[score]],
                        status=d[endpoint + "_event"],
                        ties="efron",
                    ).fit()
                    err = float(abs(other.params[0] - r["coef"]))
                    diagnostics.append(
                        dict(
                            signature=score,
                            endpoint=endpoint,
                            model=label,
                            independent_PHReg_coefficient_difference=err,
                        )
                    )
                    assert err < 1e-4
    res = pd.DataFrame(rows)
    res["q"] = np.nan
    for score in ["P1_top50", "P1_top100"]:
        ix = res.index[(res.signature == score) & (res.model == "unadjusted")]
        res.loc[ix, "q"] = multipletests(res.loc[ix, "p"], method="fdr_bh")[1]
        for modelname in ["clinical_adjusted", "clinical_composition_adjusted"]:
            ix = res.index[(res.signature == score) & (res.model == modelname)]
            res.loc[ix, "q"] = multipletests(res.loc[ix, "p"], method="fdr_bh")[1]

    res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
