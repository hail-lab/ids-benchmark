"""
Revision E8c — standardised effect sizes and confidence intervals.

Addresses R1 (report Kendall's W / pairwise effect sizes), R6.8 and R7.3
(Friedman on three datasets has little power; pairwise fold-level tests with
CIs are needed) and R6.7 (per-class per-fold counts for rare classes).

Computes, from results that already exist plus the new revision CSVs:
  1. Kendall's W for the Friedman test on the main benchmark (per task),
     with the interpretation band, plus the Friedman statistic recomputed.
  2. Pairwise model comparisons across datasets: Wilcoxon signed-rank where
     n permits, matched-pairs rank-biserial correlation, and Cliff's delta.
  3. Bootstrap 95% CIs for the ablation feature-removal deltas (E3 folds).
  4. Per-class, per-fold sample counts for each dataset's multi-class task,
     flagging classes with fewer than MIN_PER_FOLD samples per fold.

Usage:  python e8c_effect_sizes.py
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.model_selection import StratifiedKFold

from utils import (
    dataset_path,
    DATA_CLEAN, ORIG_TABLES, TABLES, RANDOM_STATE, PAPER_DATASETS,
    MODEL_NAMES, log,
)

N_FOLDS = 5
MIN_PER_FOLD = 5
N_BOOT = 10_000


def kendalls_w(scores):
    """Kendall's W from a (blocks × treatments) score matrix; higher is better."""
    n, k = scores.shape                      # n blocks (datasets), k models
    ranks = np.vstack([rankdata(-scores[i]) for i in range(n)])
    R = ranks.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    return 12 * S / (n ** 2 * (k ** 3 - k))


def _w_band(w):
    if w < 0.1:
        return "negligible"
    if w < 0.3:
        return "small"
    if w < 0.5:
        return "moderate"
    return "strong"


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def part1_kendall():
    res = pd.read_csv(ORIG_TABLES / "benchmark_results.csv")
    rows = []
    for task in ("binary", "multi"):
        sub = res[res["task"] == task]
        ds_present = [d for d in PAPER_DATASETS if d in sub["dataset"].values]
        models = [m for m in MODEL_NAMES
                  if sub[sub["model"] == m]["dataset"].nunique() == len(ds_present)]
        S = np.array([[sub[(sub["dataset"] == d) & (sub["model"] == m)]
                       ["f1_macro"].iloc[0] for m in models] for d in ds_present])
        stat, p = friedmanchisquare(*[S[:, j] for j in range(S.shape[1])])
        W = kendalls_w(S)
        rows.append(dict(task=task, n_datasets=len(ds_present),
                         n_models=len(models),
                         friedman_chi2=round(float(stat), 4),
                         friedman_p=round(float(p), 4),
                         kendalls_w=round(float(W), 3),
                         effect_band=_w_band(W)))
        log.info("[%s] Friedman chi2=%.3f p=%.4f, Kendall's W=%.3f (%s)",
                 task, stat, p, W, _w_band(W))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "e8c_kendalls_w.csv", index=False)
    return df


def part2_pairwise():
    res = pd.read_csv(ORIG_TABLES / "benchmark_results.csv")
    rows = []
    for task in ("binary", "multi"):
        sub = res[res["task"] == task]
        ds_present = [d for d in PAPER_DATASETS if d in sub["dataset"].values]
        models = [m for m in MODEL_NAMES
                  if sub[sub["model"] == m]["dataset"].nunique() == len(ds_present)]
        for a, b in itertools.combinations(models, 2):
            xa = np.array([sub[(sub["dataset"] == d) & (sub["model"] == a)]
                           ["f1_macro"].iloc[0] for d in ds_present])
            xb = np.array([sub[(sub["dataset"] == d) & (sub["model"] == b)]
                           ["f1_macro"].iloc[0] for d in ds_present])
            d = xa - xb
            try:
                stat, p = wilcoxon(xa, xb)
            except ValueError:
                stat, p = np.nan, np.nan
            # matched-pairs rank-biserial correlation
            nz = d[d != 0]
            if len(nz):
                r = rankdata(np.abs(nz))
                rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
            else:
                rbc = 0.0
            rows.append(dict(
                task=task, model_a=a, model_b=b, n_datasets=len(ds_present),
                mean_diff_f1=round(float(d.mean()), 4),
                wilcoxon_p=None if np.isnan(p) else round(float(p), 4),
                rank_biserial=round(float(rbc), 3),
                cliffs_delta=round(float(cliffs_delta(xa, xb)), 3),
                note="n=3 datasets: p-values are underpowered; "
                     "effect sizes are descriptive",
            ))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "e8c_pairwise_effects.csv", index=False)
    log.info("Pairwise effects → %d comparisons", len(df))
    return df


def part3_ablation_ci():
    path = TABLES / "e3_nested_fs.csv"
    if not path.exists():
        log.warning("Skipping ablation CIs — run e3_nested_fs.py first")
        return pd.DataFrame()
    e3 = pd.read_csv(path)
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for ds, g in e3.groupby("dataset"):
        base = g[g["config"] == "No FS"].set_index("fold")["f1_macro"]
        for config, gc in g.groupby("config"):
            if config == "No FS":
                continue
            sel = gc.set_index("fold")["f1_macro"]
            common = base.index.intersection(sel.index)
            d = (sel[common] - base[common]).to_numpy()
            boot = [rng.choice(d, len(d), replace=True).mean()
                    for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            try:
                _, p = wilcoxon(sel[common], base[common])
            except ValueError:
                p = np.nan
            rows.append(dict(
                dataset=ds, config=config, n_folds=len(d),
                mean_delta_f1=round(float(d.mean()), 4),
                ci95_low=round(float(lo), 4), ci95_high=round(float(hi), 4),
                wilcoxon_p=None if np.isnan(p) else round(float(p), 4),
            ))
            log.info("[%s] %s vs No FS: Δ=%.4f [%.4f, %.4f]",
                     ds, config, d.mean(), lo, hi)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "e8c_ablation_ci.csv", index=False)
    return df


def part4_class_counts():
    rows = []
    for ds in PAPER_DATASETS:
        df = pd.read_parquet(dataset_path(ds),
                             columns=["label_multi", "label_original"])
        y = df["label_multi"].to_numpy()
        names = df.groupby("label_multi")["label_original"].first().to_dict()
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        per_fold = {c: [] for c in np.unique(y)}
        for _, te in skf.split(np.zeros(len(y)), y):
            cnt = pd.Series(y[te]).value_counts().to_dict()
            for c in per_fold:
                per_fold[c].append(int(cnt.get(c, 0)))
        for c, counts in per_fold.items():
            rows.append(dict(
                dataset=ds, class_id=int(c), class_name=names.get(c, str(c)),
                n_total=int((y == c).sum()),
                min_per_fold=min(counts), max_per_fold=max(counts),
                mean_per_fold=round(float(np.mean(counts)), 1),
                unstable=min(counts) < MIN_PER_FOLD,
            ))
        log.info("[%s] %d classes, %d flagged unstable (<%d per validation fold)",
                 ds, len(per_fold),
                 sum(1 for r in rows if r["dataset"] == ds and r["unstable"]),
                 MIN_PER_FOLD)
    df = pd.DataFrame(rows).sort_values(["dataset", "n_total"])
    df.to_csv(TABLES / "e8c_per_class_fold_counts.csv", index=False)
    return df


if __name__ == "__main__":
    part1_kendall()
    part2_pairwise()
    part3_ablation_ci()
    part4_class_counts()
    log.info("E8c done → %s", TABLES)
