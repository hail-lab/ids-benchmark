"""
LightGBM multi-class validation experiments on CICIDS2017.

Tests three alternative configurations to verify whether the F1=0.197
collapse is a configuration artefact:
  (i)   is_unbalance=True (default softmax objective)
  (ii)  class_weight='balanced' via compute_sample_weight
  (iii) objective='multiclassova' with is_unbalance=True

All three use 5-fold stratified CV, identical to the main benchmark.
"""

import gc
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils import DATA_CLEAN, TABLES, RANDOM_STATE, N_JOBS, log

N_FOLDS = 5


def load_lean(dataset_name: str) -> pd.DataFrame:
    path = DATA_CLEAN / f"{dataset_name}.parquet"
    pf = pq.ParquetFile(str(path))
    chunks = []
    for batch in pf.iter_batches(batch_size=50_000):
        chunk = batch.to_pandas()
        for c in chunk.columns:
            if c in ("label_binary", "label_multi"):
                chunk[c] = chunk[c].astype("int32")
            elif chunk[c].dtype in ("int64", "float64"):
                chunk[c] = chunk[c].astype("float32")
        chunks.append(chunk)
        del batch
        gc.collect()
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def run_lgbm_config(name, X, y, extra_params, use_sample_weight=False):
    """Run a single LightGBM configuration with 5-fold CV, return per-fold F1."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fit_kwargs = {}
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y_train)
            fit_kwargs["sample_weight"] = sw

        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
            **extra_params,
        )
        t0 = time.time()
        model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_val)
        elapsed = time.time() - t0

        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)
        log.info("  [%s] fold %d — F1=%.4f (%.1fs)", name, fold, f1, elapsed)
        del model
        gc.collect()

    return fold_f1s


def main():
    log.info("Loading cicids2017...")
    df = load_lean("cicids2017")

    exclude = {"label_original", "label_binary", "label_multi", "dataset"}
    feat_cols = [c for c in df.columns if c not in exclude]

    y_multi = df["label_multi"].values.astype("int32")
    n_classes = len(np.unique(y_multi))
    log.info("Multi-class: %d classes, %d samples", n_classes, len(y_multi))

    # Build X
    for c in list(df.columns):
        if c in ("label_original", "dataset", "label_binary", "label_multi"):
            df.drop(columns=[c], inplace=True)
    gc.collect()

    n_rows = len(df)
    n_feat = len(feat_cols)
    X = np.empty((n_rows, n_feat), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    del df
    gc.collect()
    log.info("X loaded: %.1f MB (%d x %d)", X.nbytes / 1e6, n_rows, n_feat)

    results = {}

    # --- Baseline: default (matches original benchmark) ---
    log.info("\n=== Config 0: DEFAULT (original benchmark config) ===")
    f1s = run_lgbm_config("default", X, y_multi, {})
    results["default"] = {"mean": np.mean(f1s), "std": np.std(f1s), "folds": f1s}

    # --- Config (i): is_unbalance=True ---
    log.info("\n=== Config (i): is_unbalance=True ===")
    f1s = run_lgbm_config("is_unbalance", X, y_multi, {"is_unbalance": True})
    results["is_unbalance"] = {"mean": np.mean(f1s), "std": np.std(f1s), "folds": f1s}

    # --- Config (ii): class_weight='balanced' via sample_weight ---
    log.info("\n=== Config (ii): class_weight='balanced' (sample_weight) ===")
    f1s = run_lgbm_config("class_weight_balanced", X, y_multi, {}, use_sample_weight=True)
    results["class_weight_balanced"] = {"mean": np.mean(f1s), "std": np.std(f1s), "folds": f1s}

    # --- Config (iii): multiclassova + is_unbalance ---
    log.info("\n=== Config (iii): multiclassova + is_unbalance=True ===")
    f1s = run_lgbm_config("ova_unbalance", X, y_multi,
                          {"objective": "multiclassova", "is_unbalance": True})
    results["ova_unbalance"] = {"mean": np.mean(f1s), "std": np.std(f1s), "folds": f1s}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("LIGHTGBM VALIDATION RESULTS — CICIDS2017 MULTI-CLASS")
    print("=" * 70)
    for name, r in results.items():
        folds_str = ", ".join(f"{f:.4f}" for f in r["folds"])
        print(f"  {name:30s}: F1 = {r['mean']:.4f} ± {r['std']:.4f}  [{folds_str}]")

    # Save to CSV
    rows = []
    for name, r in results.items():
        rows.append({
            "config": name,
            "f1_mean": round(r["mean"], 4),
            "f1_std": round(r["std"], 4),
            **{f"fold_{i+1}": round(f, 4) for i, f in enumerate(r["folds"])},
        })
    out = TABLES / "lgbm_validation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
