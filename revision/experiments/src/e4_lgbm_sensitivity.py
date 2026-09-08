"""
Revision E4 — sensitivity analysis of the LightGBM multi-class collapse.

Addresses R6.6, R7 (SMOTE untested) and R1 (explanation must be scoped as
empirical).  On CICIDS2017 multi-class (the F1=0.197 collapse case) LightGBM
is re-run under seven configurations, 5-fold stratified CV:

    default          — the paper configuration (softmax objective, no weights)
    class_weight     — class_weight="balanced"
    ova              — one-vs-all objective (multiclassova)
    ova+weight       — one-vs-all + balanced class weights
    smote            — SMOTE oversampling fitted inside each training fold
    undersample      — random majority undersampling inside each fold
    tuned            — deeper trees + lower min_child_samples + higher lr

SMOTE/undersampling arms cap the training fold at SAMPLE_CAP rows (stratified)
to keep synthetic-sample generation tractable; the cap is reported.

Usage:  python e4_lgbm_sensitivity.py [--dataset cicids2017]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from utils import (
    dataset_path,
    DATA_CLEAN, TABLES, RANDOM_STATE, N_JOBS, safe_feature_cols, log,
)
from evaluation import compute_metrics

OUT = TABLES / "e4_lgbm_sensitivity.csv"
N_FOLDS = 5
SAMPLE_CAP = 300_000  # training-fold cap for resampling arms

BASE = dict(
    n_estimators=400, max_depth=-1, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
)

CONFIGS = {
    "default":      dict(BASE),
    "class_weight": dict(BASE, class_weight="balanced"),
    "ova":          dict(BASE, objective="multiclassova"),
    "ova+weight":   dict(BASE, objective="multiclassova", class_weight="balanced"),
    "smote":        dict(BASE),
    "undersample":  dict(BASE),
    "tuned":        dict(BASE, num_leaves=127, min_child_samples=5,
                         learning_rate=0.1, n_estimators=600),
}


def _resample(kind, X_tr, y_tr):
    """Cap the fold, then apply SMOTE or random undersampling.

    SMOTE interpolates between a sample and its k nearest same-class
    neighbours, so a class with fewer than k+1 members cannot be synthesised
    from.  On CICIDS2017 the rarest classes have single-digit totals
    (Heartbleed: 7), and after the stratified cap some fall to one training
    row, which makes an unconditional SMOTE call fail outright.  We therefore
    oversample only the classes that can support it and leave the rest at
    their observed frequency, rather than dropping them (they are part of the
    task) or abandoning the arm.  `_resample` returns the resampled data plus
    a note recording which classes were excluded, so the manuscript can state
    exactly what SMOTE was and was not applied to.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    if len(X_tr) > SAMPLE_CAP:
        # stratified cap preserving all classes
        keep = []
        for c in np.unique(y_tr):
            idx = np.where(y_tr == c)[0]
            n = max(1, int(round(len(idx) * SAMPLE_CAP / len(y_tr))))
            keep.append(rng.choice(idx, min(n, len(idx)), replace=False))
        keep = np.concatenate(keep)
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    if kind != "smote":
        from imblearn.under_sampling import RandomUnderSampler
        X_r, y_r = RandomUnderSampler(
            random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
        return X_r, y_r, ""

    from imblearn.over_sampling import SMOTE
    labels, counts = np.unique(y_tr, return_counts=True)
    size = dict(zip(labels.tolist(), counts.tolist()))
    target = int(counts.max())

    # k must be < the smallest class we actually oversample; cap at 5.
    eligible = {c: n for c, n in size.items() if n >= 2}
    if not eligible:
        return X_tr, y_tr, "smote skipped: no class with >=2 samples"
    k = max(1, min(5, min(eligible.values()) - 1))
    # after fixing k, a class needs > k members to be a SMOTE source
    eligible = {c: n for c, n in size.items() if n > k}
    excluded = sorted(c for c in size if c not in eligible)
    if not eligible:
        return X_tr, y_tr, "smote skipped: no class exceeds k=%d" % k

    strategy = {c: target for c in eligible}
    sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k,
                    sampling_strategy=strategy)
    X_r, y_r = sampler.fit_resample(X_tr, y_tr)
    note = ("smote k=%d; oversampled %d/%d classes to %d; left at natural "
            "frequency (too few samples): %s"
            % (k, len(eligible), len(size), target,
               ",".join(map(str, excluded)) or "none"))
    return X_r, y_r, note


def main(ds="cicids2017", only=None):
    # Resume: keep rows already computed for other configurations.
    existing = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    done = set()
    if not existing.empty:
        done = set(zip(existing["config"].astype(str),
                       existing["fold"].astype(str)))

    df = pd.read_parquet(dataset_path(ds))
    feat_cols = safe_feature_cols(df.columns)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy(dtype=np.int32)
    del df
    n_classes = len(np.unique(y))
    log.info("[%s] %d rows × %d features, %d classes", ds, len(X),
             len(feat_cols), n_classes)

    rows = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr0, X_te = X[tr].copy(), X[te].copy()
        y_tr0, y_te = y[tr], y[te]
        lo, hi = X_tr0.min(0), X_tr0.max(0)
        sc = np.where(hi - lo == 0, 1, hi - lo)
        X_tr0 = (X_tr0 - lo) / sc
        X_te = (X_te - lo) / sc

        for name, params in CONFIGS.items():
            if only and name != only:
                continue
            if (name, str(fold)) in done:
                log.info("fold %d %-12s skipped (already computed)", fold, name)
                continue
            X_tr, y_tr = X_tr0, y_tr0
            note = ""
            if name in ("smote", "undersample"):
                try:
                    X_tr, y_tr, note = _resample(name, X_tr0, y_tr0)
                except Exception as exc:
                    log.error("[fold %d] %s resampling failed: %s", fold, name, exc)
                    continue
                if note:
                    log.info("fold %d %s: %s", fold, name, note)
            t0 = time.time()
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            m = compute_metrics(y_te, y_pred, None, n_classes)
            m.update(dataset=ds, config=name, fold=fold,
                     n_train=len(X_tr), note=note,
                     fit_seconds=round(time.time() - t0, 1))
            rows.append(m)
            log.info("fold %d %-12s F1=%.4f balAcc=%.4f (%.0fs, n=%d)",
                     fold, name, m["f1_macro"], m["balanced_accuracy"],
                     m["fit_seconds"], len(X_tr))
            out_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            out_df = out_df.drop_duplicates(subset=["config", "fold"], keep="last")
            out_df.to_csv(OUT, index=False)

    log.info("E4 done → %s", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--only", choices=sorted(CONFIGS),
                        help="run a single configuration (resumes the rest)")
    args = parser.parse_args()
    main(args.dataset, only=args.only)
