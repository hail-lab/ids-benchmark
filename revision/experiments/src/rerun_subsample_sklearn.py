"""
Re-run the subsample / iteration-budget study on scikit-learn.

Why
---
e5_subsample_curve.csv was produced on Colab with cuML. The benchmark it is
compared against in Table 4 is scikit-learn, so the study does not currently
answer the question the reviewers asked -- whether the 50,000-row cap
understates *the reported* SVM and k-NN figures. Absolute values differ
markedly between the two implementations (k-NN 0.9698 against 0.9971 for the
same nominal configuration), so the comparison has to be redone on the library
the paper reports.

An abandoned scikit-learn attempt survives as e5_subsample_curve_v1_backup.csv
with four cells; those are consistent with the values produced here and with an
independent single-fold check, so this is a continuation rather than a new
direction.

Cost
----
Measured, per fold: k-NN 42 s at 50k, 77 s at 100k; SVM 792 s at 50k with
max_iter=5000; SVM 7,972 s at 100k. The 100k SVM arm therefore runs one fold
only, which the output records -- the published table already carries a
single-fold cell for the same reason.

Cells are ordered cheapest first and every completed cell is written
immediately, so the run can be stopped at any point and resumed, and the
cheap arms are banked before the expensive ones start.

Usage:  python rerun_subsample_sklearn.py [--check] [--datasets cicids2017]
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from utils import (TABLES, RANDOM_STATE, N_JOBS, dataset_path,
                   safe_feature_cols, log)

OUT = TABLES / "e5_subsample_curve_sklearn.csv"
N_FOLDS = 3
KEY = ["dataset", "model", "config", "n_train", "fold"]

# (model, config, cap, folds to run) -- cheapest first
CELLS = [
    ("kNN",       "k5",                    50_000, N_FOLDS),
    ("kNN",       "k5",                   100_000, N_FOLDS),
    ("kNN",       "k5",                   200_000, N_FOLDS),
    ("LinearSVC", "linear",                  None, N_FOLDS),
    ("SVM",       "rbf_maxiter5000",       50_000, N_FOLDS),
    ("SVM",       "rbf_maxiter50000",      50_000, N_FOLDS),
    ("SVM",       "rbf_maxiter5000_100k", 100_000, 1),
]


def build(model, config):
    if model == "kNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)
    if model == "LinearSVC":
        return LinearSVC(dual="auto", max_iter=5000, random_state=RANDOM_STATE)
    budget = 50_000 if "50000" in config else 5_000
    # probability estimates are omitted: they cost ~2x and do not change
    # predictions, verified at 0.8880 either way on an identical fold
    return SVC(kernel="rbf", max_iter=budget, random_state=RANDOM_STATE)


def converged(model, config):
    if not hasattr(model, "n_iter_"):
        return None
    budget = 50_000 if "50000" in config else 5_000
    return bool(np.max(np.atleast_1d(model.n_iter_)) < budget)


def main(datasets, check_only):
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = {tuple(str(v) for v in r) for r in prev[KEY].values}
        log.info("resuming: %d cell(s) already recorded", len(done))

    plan = [(ds, *c) for ds in datasets for c in CELLS]
    print(f"{'dataset':12s} {'model':10s} {'config':22s} {'cap':>9s} {'folds':>6s}")
    for ds, m, cfg, cap, nf in plan:
        print(f"{ds:12s} {m:10s} {cfg:22s} {str(cap or 'full'):>9s} {nf:>6d}")
    if check_only:
        print("\n--check: nothing was trained")
        return

    for ds in datasets:
        df = pd.read_parquet(dataset_path(ds))
        feats = safe_feature_cols(df.columns)
        X = df[feats].to_numpy(dtype=np.float32)
        y = df["label_binary"].to_numpy()
        del df
        log.info("[%s] %d rows x %d features", ds, len(X), len(feats))

        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            needed = [c for c in CELLS if fold <= c[3]]
            if not needed:
                continue
            X_tr, X_te = X[tr].copy(), X[te]
            lo = X_tr.min(0)
            sc = X_tr.max(0) - lo
            sc[sc == 0] = 1.0
            X_tr -= lo
            X_tr /= sc
            X_te = (X_te - lo) / sc
            y_tr, y_te = y[tr], y[te]

            for model_name, config, cap, _ in needed:
                if cap is None or len(X_tr) <= cap:
                    Xs, ys = X_tr, y_tr
                else:
                    idx = np.random.RandomState(RANDOM_STATE).choice(
                        len(X_tr), cap, replace=False)
                    Xs, ys = X_tr[idx], y_tr[idx]
                n_train = len(Xs)

                key = (ds, model_name, config, str(n_train), str(fold))
                if key in done:
                    log.info("[%s] skip %s/%s n=%d fold %d (done)",
                             ds, model_name, config, n_train, fold)
                    continue

                t0 = time.time()
                est = build(model_name, config)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    est.fit(Xs, ys)
                pred = est.predict(X_te)
                row = dict(
                    dataset=ds, model=model_name, config=config,
                    n_train=n_train, fold=fold, backend="sklearn",
                    accuracy=round(accuracy_score(y_te, pred), 4),
                    balanced_accuracy=round(balanced_accuracy_score(y_te, pred), 4),
                    precision_macro=round(precision_score(y_te, pred,
                                                          average="macro",
                                                          zero_division=0), 4),
                    recall_macro=round(recall_score(y_te, pred, average="macro",
                                                    zero_division=0), 4),
                    f1_macro=round(f1_score(y_te, pred, average="macro"), 4),
                    converged=converged(est, config),
                    fit_seconds=round(time.time() - t0, 1),
                )
                out = pd.DataFrame([row])
                if OUT.exists():
                    out = pd.concat([pd.read_csv(OUT), out], ignore_index=True)
                out.to_csv(OUT, index=False)
                done.add(key)
                log.info("[%s] %s/%s n=%d fold %d: F1=%.4f conv=%s (%.0fs)",
                         ds, model_name, config, n_train, fold,
                         row["f1_macro"], row["converged"], row["fit_seconds"])

        del X, y

    d = pd.read_csv(OUT)
    # Group by config, never by n_train. Splitting a full-data arm across
    # 1,333,333 and 1,333,334 rows -- the remainder of an integer fold
    # split -- is what produced a LinearSVC mean over two of three folds in
    # the superseded table, and that 0.7942 reached the manuscript.
    summary = (d.groupby(["dataset", "model", "config"])
                 .agg(f1=("f1_macro", "mean"), sd=("f1_macro", "std"),
                      conv=("converged", "all"), folds=("fold", "count"),
                      n_min=("n_train", "min"), n_max=("n_train", "max"))
                 .round(4))
    print("\n" + summary.to_string())
    split = summary[summary.n_min != summary.n_max]
    if len(split):
        print("\nnote: these arms span a fold-size remainder; the means"
              " above cover all folds:")
        print(split[["n_min", "n_max", "folds"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cicids2017,unsw_nb15")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    main([d.strip() for d in a.datasets.split(",") if d.strip()], a.check)
