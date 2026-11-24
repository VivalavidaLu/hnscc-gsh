"""Match programs across a donor-exclusion refit using Spearman correlations and optimal assignment."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine


def match(original, new, orig_tpm, new_tpm):
    shared = original.columns.intersection(new.columns)
    rows = []
    for i in original.index:
        for j in new.index:
            x = original.loc[i, shared].to_numpy()
            y = new.loc[j, shared].to_numpy()
            r = dict(
                original_program=int(i),
                new_program=int(j),
                n_shared_genes=len(shared),
                spearman=float(stats.spearmanr(x, y).statistic),
                pearson=float(stats.pearsonr(x, y).statistic),
                cosine_tpm=1
                - float(cosine(orig_tpm.loc[i, shared], new_tpm.loc[j, shared])),
            )
            for n in [50, 100]:
                a = set(original.loc[i].nlargest(n).index)
                b = set(new.loc[j].nlargest(n).index)
                r[f"top{n}_overlap"] = len(a & b)
                r[f"top{n}_jaccard"] = len(a & b) / len(a | b)
            rows.append(r)
    allpairs = pd.DataFrame(rows)
    m = allpairs.pivot(
        index="original_program", columns="new_program", values="spearman"
    )
    a, b = linear_sum_assignment(-m.to_numpy())
    mapping = dict(zip(m.index[a].astype(int), m.columns[b].astype(int)))
    allpairs["one_to_one_match"] = [
        mapping.get(i) == j
        for i, j in zip(allpairs.original_program, allpairs.new_program)
    ]
    return allpairs, mapping


def main():
    parser = argparse.ArgumentParser(
        description="Match programs across a donor-exclusion refit using Spearman correlations and optimal assignment."
    )
    parser.add_argument(
        "--original-score",
        required=True,
        type=Path,
        help="original score cNMF spectrum (program × gene TSV)",
    )
    parser.add_argument(
        "--refit-score",
        required=True,
        type=Path,
        help="refit score cNMF spectrum (program × gene TSV)",
    )
    parser.add_argument(
        "--original-tpm",
        required=True,
        type=Path,
        help="original tpm cNMF spectrum (program × gene TSV)",
    )
    parser.add_argument(
        "--refit-tpm",
        required=True,
        type=Path,
        help="refit tpm cNMF spectrum (program × gene TSV)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Pairwise similarities and matched pairs, CSV",
    )
    args = parser.parse_args()
    inputs = [
        pd.read_csv(p, sep="\t", index_col=0)
        for p in (
            args.original_score,
            args.refit_score,
            args.original_tpm,
            args.refit_tpm,
        )
    ]
    pairs, _ = match(*inputs)
    pairs.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
