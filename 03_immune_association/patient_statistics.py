from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy import stats


@lru_cache(None)
def allocations(n, m):
    return np.array(list(combinations(range(n), m)), dtype=int)


def exact_rank_p(a, b):
    a = np.round(np.asarray(a, float), 12)
    b = np.round(np.asarray(b, float), 12)
    ranks = stats.rankdata(np.r_[a, b], method="average")
    idx = allocations(len(ranks), len(a))
    dist = ranks[idx].sum(axis=1)
    obs = ranks[: len(a)].sum()
    return float(
        min(1, 2 * min(np.mean(dist <= obs + 1e-10), np.mean(dist >= obs - 1e-10)))
    )


def hl_ci(a, b, alpha=0.05):
    a = np.round(np.asarray(a, float), 12)
    b = np.round(np.asarray(b, float), 12)
    diffs = np.round((a[:, None] - b).ravel(), 12)
    cuts = np.unique(diffs)
    step = max(1, float(np.ptp(cuts)) + 1)
    edges = np.r_[-np.inf, cuts, np.inf]
    accepted = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        theta = (
            hi - step
            if np.isneginf(lo)
            else lo + step
            if np.isposinf(hi)
            else (lo + hi) / 2
        )
        if exact_rank_p(a - theta, b) > alpha:
            accepted.append((lo, hi))
    for theta in cuts:
        if exact_rank_p(a - theta, b) > alpha:
            accepted.append((theta, theta))
    assert accepted, "Empty inverted confidence set"
    return (
        float(np.median(diffs)),
        float(min(x[0] for x in accepted)),
        float(max(x[1] for x in accepted)),
    )


def compare(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    row = {
        "n_R": len(a),
        "n_NR": len(b),
        "median_R": float(np.median(a)) if len(a) else np.nan,
        "median_NR": float(np.median(b)) if len(b) else np.nan,
        "mean_R": float(np.mean(a)) if len(a) else np.nan,
        "mean_NR": float(np.mean(b)) if len(b) else np.nan,
    }
    row["mean_difference_R_minus_NR"] = row["mean_R"] - row["mean_NR"]
    if len(a) and len(b):
        row["HL_effect"] = float(np.median((a[:, None] - b).ravel()))
    else:
        row["HL_effect"] = np.nan
    row.update(
        CI95_low=np.nan,
        CI95_high=np.nan,
        exact_P=np.nan,
        inference_status="descriptive_only_fewer_than_2_patients_in_a_group",
    )
    if min(len(a), len(b)) >= 2:
        h, lo, hi = hl_ci(a, b)
        row.update(
            HL_effect=h,
            CI95_low=lo,
            CI95_high=hi,
            exact_P=exact_rank_p(a, b),
            inference_status="exact_patient_level",
        )
        # Independent implementation of the rank permutation P-value.
        check = stats.permutation_test(
            (a, b),
            lambda x, y: stats.mannwhitneyu(x, y, method="asymptotic").statistic,
            n_resamples=np.inf,
            alternative="two-sided",
        ).pvalue
        assert abs(check - row["exact_P"]) < 1e-10, (a, b, check, row)
    return row
