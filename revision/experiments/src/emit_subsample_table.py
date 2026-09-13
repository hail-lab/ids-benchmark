"""
Emit the LaTeX body of the subsample / iteration-budget table from the
scikit-learn re-run, plus the prose figures that depend on it.

The superseded version of this table was produced on cuML while the benchmark
it is compared against ran on scikit-learn, so its absolute values were not
comparable with Table 4 and the conclusion drawn from it -- that the submitted
SVM figure was depressed by its iteration budget -- did not hold for the
implementation the paper reports.

Aggregation is by configuration, never by n_train: a full-data arm spans
1,333,333 and 1,333,334 rows depending on the fold-size remainder, and grouping
on that split produced a mean over two of three folds in the superseded table.

Usage:  python emit_subsample_table.py
"""

import pandas as pd

from utils import TABLES

SRC = TABLES / "e5_subsample_curve_sklearn.csv"
BENCH = TABLES.parent.parent.parent / "outputs" / "tables" / "benchmark_results.csv"

DATASETS = [("cicids2017", "CICIDS2017"), ("unsw_nb15", "UNSW-NB15")]
ROWS = [
    ("kNN", "k5", 50_000, r"k-NN ($k$=5)"),
    ("kNN", "k5", 100_000, r"k-NN ($k$=5)"),
    ("kNN", "k5", 200_000, r"k-NN ($k$=5)"),
    ("SVM", "rbf_maxiter5000", 50_000, r"SVM, \texttt{max\_iter}=5{,}000"),
    ("SVM", "rbf_maxiter5000_100k", 100_000, r"SVM, \texttt{max\_iter}=5{,}000"),
    ("SVM", "rbf_maxiter50000", 50_000, r"SVM, \texttt{max\_iter}=50{,}000"),
    ("LinearSVC", "linear", "full", r"LinearSVC"),
]


def thousands(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


CAPS = (50_000, 100_000, 200_000)


def cap_of(n: int):
    """The cap a fold was *intended* to run at.

    Grouping on raw n_train is wrong in both directions. k-NN's three sizes are
    three separate experiments and must stay separate; LinearSVC's 1,333,333
    and 1,333,334 are one experiment whose fold sizes differ by the remainder
    of an integer split, and separating them is what produced a mean over two
    of three folds in the superseded table.
    """
    return next((c for c in CAPS if n == c), "full")


def main():
    d = pd.read_csv(SRC)
    d["cap"] = d.n_train.map(cap_of)
    g = d.groupby(["dataset", "model", "config", "cap"]).agg(
        f1=("f1_macro", "mean"), sd=("f1_macro", "std"),
        conv=("converged", "all"), folds=("fold", "count"),
        n_min=("n_train", "min"), n_max=("n_train", "max"))

    print("% ---- tab:subsample (scikit-learn) ----")
    for ds, label in DATASETS:
        rows = [r for r in ROWS if (ds, r[0], r[1], r[2]) in g.index]
        if not rows:
            continue
        print(f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}")
        for model, config, cap, pretty in rows:
            r = g.loc[(ds, model, config, cap)]
            n = thousands(r.n_min) if r.n_min == r.n_max else \
                f"{thousands(r.n_min)}--{thousands(r.n_max)}"
            conv = "---" if model == "kNN" else ("yes" if r.conv else "no")
            star = "" if r.folds == 3 else r"$^{\dagger}$"
            print(f" & {pretty:36s} & {n:>14s} & {r.f1:.4f}{star} & {conv} \\\\")
        print(r"\hline")

    print("\n% prose figures:")
    for ds, label in DATASETS:
        try:
            k50 = d[(d.dataset == ds) & (d.model == "kNN") &
                    (d.n_train == 50_000)].f1_macro.mean()
            k200 = d[(d.dataset == ds) & (d.model == "kNN") &
                     (d.n_train == 200_000)].f1_macro.mean()
            print(f"%   {label}: k-NN 50k->200k costs {k200 - k50:+.4f}")
        except KeyError:
            pass
        try:
            lo = g.loc[(ds, "SVM", "rbf_maxiter5000", 50_000)]
            hi = g.loc[(ds, "SVM", "rbf_maxiter50000", 50_000)]
            print(f"%   {label}: SVM budget 5k->50k moves "
                  f"{lo.f1:.4f} -> {hi.f1:.4f} ({hi.f1 - lo.f1:+.4f}), "
                  f"converged {lo.conv} -> {hi.conv}")
        except KeyError:
            pass
        try:
            b = pd.read_csv(BENCH)
            sub = b[(b.dataset == ds) & (b.model == "SVM") &
                    (b.task == "binary")]
            if len(sub):
                print(f"%   {label}: submitted benchmark SVM = "
                      f"{sub.f1_macro.iloc[0]:.4f} (5 folds)")
        except Exception:                                 # noqa: BLE001
            pass

    print("\n% $\\dagger$ marks cells run on fewer than three folds.")


if __name__ == "__main__":
    main()
