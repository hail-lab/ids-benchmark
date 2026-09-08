"""
Revision E3 — feature-selection ablation with selection nested inside folds.

Addresses R6.4: MI ranking, RF importance, and the hybrid MI→RF selection are
re-fitted on the TRAINING portion of every CV fold, then applied to the
validation fold — eliminating any selection leakage.  Also records per-fold
selected feature sets so ranking stability across folds can be reported
(supports R6.12).

Configs: No FS | MI top-30 | RF top-15 | Hybrid MI→RF top-15.
Model: XGBoost (binary), 5-fold stratified CV, all three datasets.

Usage:  python e3_nested_fs.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from utils import (
    dataset_path,
    DATA_CLEAN, TABLES, RANDOM_STATE, N_JOBS, PAPER_DATASETS,
    safe_feature_cols, log,
)
from evaluation import compute_metrics

OUT = TABLES / "e3_nested_fs.csv"
OUT_FEATS = TABLES / "e3_nested_fs_selected_features.json"

TOP_K_FILTER = 30
TOP_K_MODEL = 15
MI_SAMPLE = 200_000
MAX_ROWS = 200_000  # same cap as the original ablation for comparability
N_FOLDS = 5


def _select(config, X_tr, y_tr, feat_names, rng):
    """Return selected column indices, computed on training data only."""
    if config == "No FS":
        return list(range(X_tr.shape[1]))
    n = min(MI_SAMPLE, len(X_tr))
    sub = rng.choice(len(X_tr), n, replace=False)
    if config in ("MI only (top 30)", "Hybrid MI-RF (15)"):
        mi = mutual_info_classif(X_tr[sub], y_tr[sub], random_state=RANDOM_STATE)
        mi_rank = list(np.argsort(mi)[::-1][:TOP_K_FILTER])
        if config == "MI only (top 30)":
            return mi_rank
        rf = RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS,
                                    random_state=RANDOM_STATE)
        rf.fit(X_tr[sub][:, mi_rank], y_tr[sub])
        top = np.argsort(rf.feature_importances_)[::-1][:TOP_K_MODEL]
        return [mi_rank[i] for i in top]
    # RF only
    rf = RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS,
                                random_state=RANDOM_STATE)
    rf.fit(X_tr[sub], y_tr[sub])
    return list(np.argsort(rf.feature_importances_)[::-1][:TOP_K_MODEL])


CONFIGS = ["No FS", "MI only (top 30)", "RF only (top 15)", "Hybrid MI-RF (15)"]


def main():
    rows, selected_log = [], {}
    rng = np.random.RandomState(RANDOM_STATE)

    for ds in PAPER_DATASETS:
        df = pd.read_parquet(dataset_path(ds))
        feat_cols = safe_feature_cols(df.columns)
        if len(df) > MAX_ROWS:
            df = df.sample(MAX_ROWS, random_state=RANDOM_STATE)
        X = df[feat_cols].to_numpy(dtype=np.float32)
        y = df["label_binary"].to_numpy(dtype=np.int32)
        del df
        log.info("[%s] %d rows × %d features", ds, len(X), len(feat_cols))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X[tr].copy(), X[te].copy()
            y_tr, y_te = y[tr], y[te]
            lo, hi = X_tr.min(0), X_tr.max(0)
            sc = np.where(hi - lo == 0, 1, hi - lo)
            X_tr = (X_tr - lo) / sc
            X_te = (X_te - lo) / sc

            for config in CONFIGS:
                sel = _select(config, X_tr, y_tr, feat_cols, rng)
                model = xgb.XGBClassifier(
                    n_estimators=400, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    eval_metric="logloss", random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                )
                model.fit(X_tr[:, sel], y_tr)
                y_pred = model.predict(X_te[:, sel])
                probs = model.predict_proba(X_te[:, sel])
                m = compute_metrics(y_te, y_pred, probs, 2)
                m.update(dataset=ds, config=config, fold=fold,
                         n_features=len(sel))
                rows.append(m)
                selected_log.setdefault(ds, {}).setdefault(config, []).append(
                    [feat_cols[i] for i in sel])
                log.info("[%s] fold %d %-18s F1=%.4f (k=%d)",
                         ds, fold, config, m["f1_macro"], len(sel))

        pd.DataFrame(rows).to_csv(OUT, index=False)
        with open(OUT_FEATS, "w") as fh:
            json.dump(selected_log, fh, indent=1)

    # Fold-stability of selections (mean pairwise Jaccard across folds)
    stab_rows = []
    for ds, cfgs in selected_log.items():
        for config, folds in cfgs.items():
            if config == "No FS":
                continue
            sets = [set(f) for f in folds]
            jac = [len(a & b) / len(a | b)
                   for i, a in enumerate(sets) for b in sets[i + 1:]]
            stab_rows.append({"dataset": ds, "config": config,
                              "mean_jaccard": round(float(np.mean(jac)), 3)})
    pd.DataFrame(stab_rows).to_csv(TABLES / "e3_selection_stability.csv",
                                   index=False)
    log.info("E3 done → %s", OUT)


if __name__ == "__main__":
    main()
