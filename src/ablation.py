"""
01_IDS_Benchmark — Ablation & Cross-Dataset Transfer
1. Feature-selection ablation (no FS vs MI-only vs RF-only vs hybrid)
2. Cross-dataset generalisation (train on A, test on B)

Usage
-----
    python src/ablation.py                          # all ablation experiments
    python src/ablation.py --cross-dataset          # cross-dataset transfer only
"""

import argparse
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from utils import DATA_CLEAN, DATASETS, FIGURES, TABLES, RANDOM_STATE, N_JOBS, log
from evaluation import compute_metrics

# ── Feature-selection ablation ────────────────────────────────────────

TOP_K_FILTER = 30   # MI filter
TOP_K_MODEL = 15    # RF importance
MI_SAMPLE = 200_000
ABLATION_MAX_ROWS = 200_000  # cap large datasets for ablation tractability


def _load_parquet_lean(path, max_rows=None):
    """Load parquet efficiently: subsample in pyarrow, then convert to float32."""
    import gc
    import pyarrow.parquet as pq
    table = pq.read_table(str(path))
    n_total = table.num_rows
    if max_rows and n_total > max_rows:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = np.sort(rng.choice(n_total, max_rows, replace=False))
        table = table.take(idx)
    df = table.to_pandas()
    del table
    gc.collect()
    for c in df.select_dtypes(include=["int64", "float64"]).columns:
        df[c] = df[c].astype("float32")
    gc.collect()
    return df


def hybrid_fs(X_train, y_train, feature_names):
    """2-stage hybrid feature selection: MI filter → RF importance."""
    n = min(MI_SAMPLE, len(X_train))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n, replace=False)
    mi = mutual_info_classif(X_train[idx], y_train[idx], random_state=RANDOM_STATE)
    mi_rank = np.argsort(mi)[::-1][:TOP_K_FILTER]
    mi_features = [feature_names[i] for i in mi_rank]

    rf = RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    rf.fit(X_train[:, mi_rank], y_train)
    imp = rf.feature_importances_
    top_k = np.argsort(imp)[::-1][:TOP_K_MODEL]
    selected = [mi_features[i] for i in top_k]
    selected_idx = [mi_rank[i] for i in top_k]
    return selected, selected_idx


def run_fs_ablation() -> pd.DataFrame:
    """Compare no-FS vs MI-only vs RF-only vs hybrid on each dataset (binary)."""
    rows = []
    for ds in DATASETS:
        path = DATA_CLEAN / f"{ds}.parquet"
        if not path.exists():
            continue
        df = _load_parquet_lean(path, max_rows=ABLATION_MAX_ROWS)
        exclude = {"label_original", "label_binary", "label_multi", "dataset"}
        feat_cols = [c for c in df.columns if c not in exclude]
        log.info("Loaded %s: %d rows, %d features", ds, len(df), len(feat_cols))
        X = df[feat_cols].values.astype("float32")
        y = df["label_binary"].values.astype("int32")
        del df

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
        )
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        configs = {
            "No FS": list(range(len(feat_cols))),
        }

        # MI only
        n = min(MI_SAMPLE, len(X_train))
        idx_sub = np.random.RandomState(RANDOM_STATE).choice(len(X_train), n, replace=False)
        mi = mutual_info_classif(X_train[idx_sub], y_train[idx_sub], random_state=RANDOM_STATE)
        configs["MI only (top 30)"] = list(np.argsort(mi)[::-1][:TOP_K_FILTER])

        # RF only
        rf = RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        configs["RF only (top 15)"] = list(np.argsort(rf.feature_importances_)[::-1][:TOP_K_MODEL])

        # Hybrid
        _, hyb_idx = hybrid_fs(X_train, y_train, feat_cols)
        configs["Hybrid MI→RF (15)"] = hyb_idx

        for config_name, selected_idx in configs.items():
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS,
                eval_metric="logloss",
            )
            model.fit(X_train[:, selected_idx], y_train)
            y_pred = model.predict(X_test[:, selected_idx])
            y_proba = model.predict_proba(X_test[:, selected_idx])
            metrics = compute_metrics(y_test, y_pred, y_proba, 2)
            metrics["dataset"] = ds
            metrics["config"] = config_name
            metrics["n_features"] = len(selected_idx)
            rows.append(metrics)
            log.info("[%s] %s — F1=%.4f (n_feat=%d)", ds, config_name,
                     metrics["f1_macro"], len(selected_idx))

    out_df = pd.DataFrame(rows)
    out = TABLES / "ablation_feature_selection.csv"
    out_df.to_csv(out, index=False)
    log.info("FS ablation → %s", out)
    return out_df


# ── Cross-dataset transfer ────────────────────────────────────────────

def run_cross_dataset() -> pd.DataFrame:
    """Train XGBoost on dataset A, test on dataset B (all pairs)."""
    loaded = {}
    for ds in DATASETS:
        path = DATA_CLEAN / f"{ds}.parquet"
        if not path.exists():
            continue
        df = _load_parquet_lean(path, max_rows=ABLATION_MAX_ROWS)
        exclude = {"label_original", "label_binary", "label_multi", "dataset"}
        feat_cols = sorted(c for c in df.columns if c not in exclude)
        loaded[ds] = (df, feat_cols)

    # Find common features across all loaded datasets
    if len(loaded) < 2:
        log.warning("Need ≥2 datasets for cross-dataset transfer")
        return pd.DataFrame()

    common_features = None
    for ds, (df, fc) in loaded.items():
        s = set(fc)
        common_features = s if common_features is None else common_features & s
    common_features = sorted(common_features)
    log.info("Common features across datasets: %d", len(common_features))

    if len(common_features) < 5:
        log.warning("Too few common features (%d) — datasets may have incompatible schemas. "
                     "Consider feature mapping.", len(common_features))
        return pd.DataFrame()

    rows = []
    for source, target in itertools.permutations(loaded.keys(), 2):
        df_src, _ = loaded[source]
        df_tgt, _ = loaded[target]

        X_src = df_src[common_features].values
        y_src = df_src["label_binary"].values
        X_tgt = df_tgt[common_features].values
        y_tgt = df_tgt["label_binary"].values

        scaler = MinMaxScaler()
        X_src = scaler.fit_transform(X_src)
        X_tgt = scaler.transform(X_tgt)

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS,
            eval_metric="logloss",
        )
        model.fit(X_src, y_src)
        y_pred = model.predict(X_tgt)
        y_proba = model.predict_proba(X_tgt)
        metrics = compute_metrics(y_tgt, y_pred, y_proba, 2)
        metrics["train_dataset"] = source
        metrics["test_dataset"] = target
        metrics["n_common_features"] = len(common_features)
        rows.append(metrics)
        log.info("Transfer %s → %s — F1=%.4f", source, target, metrics["f1_macro"])

    out_df = pd.DataFrame(rows)
    out = TABLES / "cross_dataset_transfer.csv"
    out_df.to_csv(out, index=False)
    log.info("Cross-dataset results → %s", out)
    return out_df


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--fs-ablation", action="store_true")
    args = parser.parse_args()

    if args.cross_dataset:
        run_cross_dataset()
    elif args.fs_ablation:
        run_fs_ablation()
    else:
        run_fs_ablation()
        run_cross_dataset()
