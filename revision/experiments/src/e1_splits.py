"""
Revision E1 — random-stratified vs temporal vs host/session-group CV.

Addresses R6.1/R6.2 (row-level stratified CV allows correlated flows in train
and test) and partially R3.1 (temporal split as a proxy for drift).

Schemes per dataset (built from {ds}_grouped.parquet, see prep_groups.py):
  cicids2017 : stratified | temporal (leave-later-days-out) | group (leave-one-day-out)
  unsw_nb15  : stratified | temporal (forward chaining on stime) | group (source host)
  ton_iot    : stratified | group (source host)   [no timestamp available]

Models: XGBoost, LightGBM, RandomForest.  Tasks: binary and multi.
For non-stratified schemes, test samples whose class is absent from the
training fold are excluded from multi-class scoring and their share reported.

Usage:  python e1_splits.py [--dataset DS] [--scheme stratified|temporal|group]
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
import lightgbm as lgb
import xgboost as xgb

from utils import (
    DATA_R1, TABLES, RANDOM_STATE, N_JOBS, PAPER_DATASETS,
    safe_feature_cols, log,
)
from evaluation import compute_metrics

E1_MODELS = ["XGBoost", "LightGBM", "RandomForest"]
N_FOLDS = 5
OUT = TABLES / "e1_split_schemes.csv"


def _make_model(name, n_classes):
    if name == "XGBoost":
        return xgb.XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
        )
    return RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )


# ── Split generators: yield (fold_id, train_idx, test_idx) ────────────

def splits_stratified(df, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        yield fold, tr, te


def splits_group(df, y):
    groups = df["meta_group"].values
    n_groups = df["meta_group"].nunique()
    n_splits = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups), 1):
        yield fold, tr, te


def splits_temporal_cicids(df, y):
    """Two chronological splits over capture days (meta_time = day index)."""
    day = df["meta_time"].values
    # fold 1: train Mon-Wed (0-2), test Thu-Fri (3-4)
    yield 1, np.where(day <= 2)[0], np.where(day >= 3)[0]
    # fold 2: train Mon-Thu (0-3), test Fri (4)
    yield 2, np.where(day <= 3)[0], np.where(day == 4)[0]


def splits_temporal_time(df, y, n_chunks=5):
    """Forward-chaining on meta_time: train on first k chunks, test on chunk k+1."""
    order = np.argsort(df["meta_time"].values, kind="stable")
    bounds = np.linspace(0, len(order), n_chunks + 1).astype(int)
    chunks = [order[bounds[i]:bounds[i + 1]] for i in range(n_chunks)]
    for k in range(1, n_chunks):
        tr = np.concatenate(chunks[:k])
        te = chunks[k]
        yield k, tr, te


SCHEMES = {
    "cicids2017": {
        "stratified": splits_stratified,
        "temporal": splits_temporal_cicids,
        "group": splits_group,          # leave-one-day-out (5 day groups)
    },
    "unsw_nb15": {
        "stratified": splits_stratified,
        "temporal": splits_temporal_time,
        "group": splits_group,          # source-host groups
    },
    "ton_iot": {
        "stratified": splits_stratified,
        "group": splits_group,          # source-host groups
    },
}


def run_dataset(ds, scheme_filter=None):
    path = DATA_R1 / f"{ds}_grouped.parquet"
    if not path.exists():
        log.error("Missing %s — run prep_groups.py first", path)
        return []
    df = pd.read_parquet(path)
    feat_cols = safe_feature_cols(df.columns)
    X_full = df[feat_cols].to_numpy(dtype=np.float32)
    log.info("[%s] %d rows × %d features, %d groups",
             ds, len(df), len(feat_cols), df["meta_group"].nunique())

    rows = []
    for task in ("binary", "multi"):
        y = df[f"label_{task}"].to_numpy()
        for scheme, gen in SCHEMES[ds].items():
            if scheme_filter and scheme != scheme_filter:
                continue
            for fold, tr, te in gen(df, y):
                y_tr, y_te = y[tr], y[te]
                # keep only test samples whose class occurs in training
                seen = np.unique(y_tr)
                mask = np.isin(y_te, seen)
                excluded = 1.0 - mask.mean()
                if mask.mean() == 0 or len(seen) < 2:
                    log.warning("[%s/%s/%s] fold %d skipped (degenerate split)",
                                ds, task, scheme, fold)
                    continue
                te_use = te[mask]
                y_te_use = y_te[mask]

                # remap labels to a contiguous range for XGBoost/LightGBM
                remap = {c: i for i, c in enumerate(seen)}
                y_tr_m = np.vectorize(remap.get)(y_tr)
                y_te_m = np.vectorize(remap.get)(y_te_use)
                n_classes = len(seen)

                X_tr = X_full[tr].copy()
                X_te = X_full[te_use].copy()
                fmin, fmax = X_tr.min(axis=0), X_tr.max(axis=0)
                scale = fmax - fmin
                scale[scale == 0] = 1.0
                X_tr = (X_tr - fmin) / scale
                X_te = (X_te - fmin) / scale

                for mname in E1_MODELS:
                    t0 = time.time()
                    model = _make_model(mname, n_classes)
                    model.fit(X_tr, y_tr_m)
                    y_pred = model.predict(X_te)
                    probs = model.predict_proba(X_te)
                    m = compute_metrics(y_te_m, y_pred, probs, n_classes)
                    m.update(dataset=ds, task=task, scheme=scheme, fold=fold,
                             model=mname, n_train=len(tr), n_test=len(te_use),
                             excluded_test_frac=round(excluded, 4),
                             n_classes_seen=n_classes,
                             fit_seconds=round(time.time() - t0, 1))
                    rows.append(m)
                    log.info("[%s/%s/%s] fold %d %s — F1=%.4f (excl %.1f%%, %.0fs)",
                             ds, task, scheme, fold, mname, m["f1_macro"],
                             excluded * 100, m["fit_seconds"])
                    _save(rows)
                del X_tr, X_te
                gc.collect()
    return rows


_all_rows = []

def _save(rows):
    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset=["dataset", "task", "scheme", "fold", "model"], keep="last")
    new.to_csv(OUT, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PAPER_DATASETS)
    parser.add_argument("--scheme", choices=["stratified", "temporal", "group"])
    args = parser.parse_args()
    for ds in ([args.dataset] if args.dataset else PAPER_DATASETS):
        run_dataset(ds, scheme_filter=args.scheme)
    log.info("E1 done → %s", OUT)
