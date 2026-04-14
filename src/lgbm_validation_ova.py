"""Re-run only the missing config (iii): multiclassova + is_unbalance."""
import gc, sys, time, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from utils import DATA_CLEAN, TABLES, RANDOM_STATE, N_JOBS, log

N_FOLDS = 5

def load_lean(dataset_name):
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
        del batch; gc.collect()
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    return df

def main():
    log.info("Loading cicids2017...")
    df = load_lean("cicids2017")
    exclude = {"label_original", "label_binary", "label_multi", "dataset"}
    feat_cols = [c for c in df.columns if c not in exclude]
    y = df["label_multi"].values.astype("int32")
    log.info("Classes: %d, samples: %d", len(np.unique(y)), len(y))

    for c in list(df.columns):
        if c in ("label_original", "dataset", "label_binary", "label_multi"):
            df.drop(columns=[c], inplace=True)
    gc.collect()

    X = np.empty((len(df), len(feat_cols)), dtype=np.float32)
    for i, c in enumerate(feat_cols):
        if c in df.columns:
            X[:, i] = df[c].values.astype(np.float32)
    del df; gc.collect()
    log.info("X: %.1f MB", X.nbytes / 1e6)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multiclassova", is_unbalance=True,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
        )
        t0 = time.time()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        f1 = f1_score(y[va], pred, average="macro", zero_division=0)
        fold_f1s.append(f1)
        log.info("  fold %d — F1=%.4f (%.1fs)", fold, f1, time.time() - t0)
        del model; gc.collect()

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\nova_unbalance: F1 = {mean_f1:.4f} +/- {std_f1:.4f}  [{', '.join(f'{f:.4f}' for f in fold_f1s)}]")

    # Append to existing CSV
    csv_path = TABLES / "lgbm_validation.csv"
    existing = pd.read_csv(csv_path)
    row = {"config": "ova_unbalance", "f1_mean": round(mean_f1, 4), "f1_std": round(std_f1, 4)}
    row.update({f"fold_{i+1}": round(f, 4) for i, f in enumerate(fold_f1s)})
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    print(f"Appended to {csv_path}")

if __name__ == "__main__":
    main()
