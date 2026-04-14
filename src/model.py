"""
01_IDS_Benchmark — Model Training
Trains 8 ML/DL models on each dataset (binary + multi-class) with
unified hyper-parameter search and 5-fold stratified CV.

Usage
-----
    python src/model.py                            # all datasets, all models
    python src/model.py --dataset cicids2017       # one dataset
    python src/model.py --model XGBoost            # one model across all datasets
"""

import argparse
import gc
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb

# PyTorch is imported lazily to avoid its ~2GB memory overhead
# when running sklearn-only models on large datasets.
torch = None
nn = None

def _ensure_torch():
    """Lazy-import PyTorch. Called only when DL models are needed."""
    global torch, nn, DataLoader, TensorDataset
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader as _DL, TensorDataset as _TDS
    torch = _torch
    nn = _nn
    DataLoader = _DL
    TensorDataset = _TDS

from utils import (
    DATA_CLEAN, DATASETS, MODEL_NAMES, MODELS as MODEL_DIR,
    TABLES, RANDOM_STATE, N_JOBS, log,
)

# ── Hyper-parameters ──────────────────────────────────────────────────
HP = {
    "XGBoost": dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device="cuda",
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    ),
    "RandomForest": dict(
        n_estimators=200, max_depth=None,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    ),
    "LightGBM": dict(
        n_estimators=400, max_depth=-1, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        device="gpu",
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1,
    ),
    "SVM": dict(
        kernel="rbf", probability=True, max_iter=5000,
        random_state=RANDOM_STATE,
    ),
    "kNN": dict(n_neighbors=5, n_jobs=N_JOBS),
    "MLP": dict(
        hidden_layer_sizes=(128, 64), max_iter=100,
        early_stopping=True, random_state=RANDOM_STATE,
    ),
}

N_FOLDS = 5
DL_EPOCHS = 50
DL_BATCH = 1024
DL_LR = 1e-3
SVM_MAX_SAMPLES = 50_000   # SVM does not scale to millions of rows
KNN_MAX_SAMPLES = 50_000   # kNN distance computation is O(n²)


# ── Sklearn model factory ─────────────────────────────────────────────

def _make_sklearn(name: str, n_classes: int):
    """Return an sklearn-compatible estimator."""
    if name == "XGBoost":
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        return xgb.XGBClassifier(objective=obj, eval_metric="logloss", **HP[name])
    if name == "RandomForest":
        return RandomForestClassifier(**HP[name])
    if name == "LightGBM":
        return lgb.LGBMClassifier(**HP[name])
    if name == "SVM":
        return SVC(**HP[name])
    if name == "kNN":
        return KNeighborsClassifier(**HP[name])
    if name == "MLP":
        return MLPClassifier(**HP[name])
    raise ValueError(f"Unknown sklearn model: {name}")


# ── PyTorch DL models (lazy-loaded to avoid memory overhead) ──────────

def _make_cnn1d(n_features: int, n_classes: int):
    """Create CNN1D model (requires torch to be loaded)."""
    class CNN1D(nn.Module):
        def __init__(self, n_feat, n_cls):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_cls),
            )
        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x).squeeze(-1)
            return self.fc(x)
    return CNN1D(n_features, n_classes)


def _make_bilstm(n_features: int, n_classes: int, hidden: int = 64):
    """Create BiLSTM model (requires torch to be loaded)."""
    class BiLSTMClassifier(nn.Module):
        def __init__(self, n_feat, n_cls, hid):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=hid, num_layers=2,
                batch_first=True, bidirectional=True, dropout=0.2,
            )
            self.fc = nn.Sequential(
                nn.Linear(hid * 2, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_cls),
            )
        def forward(self, x):
            x = x.unsqueeze(-1)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
    return BiLSTMClassifier(n_features, n_classes, hidden)


DL_PATIENCE = 10  # early-stopping patience


def _train_dl(model_name, X_train, y_train, X_val, y_val, n_classes, device):
    """Train a PyTorch model with early stopping, return (model, val_probs)."""
    _ensure_torch()
    n_features = X_train.shape[1]
    if model_name == "CNN1D":
        model = _make_cnn1d(n_features, n_classes).to(device)
    else:
        model = _make_bilstm(n_features, n_classes).to(device)
    # Class-weighted cross-entropy to handle imbalance
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = (1.0 / class_counts) * class_counts.sum() / n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimiser = torch.optim.Adam(model.parameters(), lr=DL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=DL_BATCH, shuffle=True)

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    val_loader = DataLoader(val_ds, batch_size=DL_BATCH, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(DL_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

        # Validation loss for early stopping (batched to avoid OOM)
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += criterion(model(xb), yb).item() * len(yb)
                val_n += len(yb)
        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= DL_PATIENCE:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Batched prediction to avoid OOM
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    probs = torch.cat(all_probs, dim=0).numpy()
    return model, probs


# ── Unified training loop ─────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature column names (everything except labels/meta)."""
    exclude = {"label_original", "label_binary", "label_multi", "dataset"}
    return [c for c in df.columns if c not in exclude]


def train_single(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_classes: int,
    dataset_name: str,
    task: str,
) -> dict:
    """
    5-fold stratified CV for one model on one dataset+task.
    Returns dict with per-fold metrics and averaged results.
    """
    device = None  # set lazily for DL models
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    best_f1 = -1
    model_path = MODEL_DIR / f"{dataset_name}_{task}_{model_name}.joblib"
    is_dl = model_name in ("CNN1D", "BiLSTM")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # In-place MinMax scaling (avoids sklearn's internal copy that OOMs)
        fmin = X_train.min(axis=0)
        fmax = X_train.max(axis=0)
        scale = fmax - fmin
        scale[scale == 0] = 1.0  # constant features
        X_train -= fmin
        X_train /= scale
        X_val = (X_val - fmin) / scale  # val is small, copy is fine

        # Sub-sample for SVM and kNN
        if model_name == "SVM" and len(X_train) > SVM_MAX_SAMPLES:
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(X_train), SVM_MAX_SAMPLES, replace=False
            )
            X_train, y_train = X_train[idx], y_train[idx]
        elif model_name == "kNN" and len(X_train) > KNN_MAX_SAMPLES:
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(X_train), KNN_MAX_SAMPLES, replace=False
            )
            X_train, y_train = X_train[idx], y_train[idx]

        t0 = time.time()

        if is_dl:
            _ensure_torch()
            if device is None:
                # Force CPU for BiLSTM to avoid GPU memory issues with large datasets
                if model_name == "BiLSTM":
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, val_probs = _train_dl(
                model_name, X_train, y_train, X_val, y_val, n_classes, device
            )
            y_pred = val_probs.argmax(axis=1)
        else:
            model = _make_sklearn(model_name, n_classes)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            if hasattr(model, "predict_proba"):
                val_probs = model.predict_proba(X_val)
            else:
                val_probs = None

        fit_time = time.time() - t0

        # Compute metrics (imported from evaluation.py at runtime to avoid circular)
        from evaluation import compute_metrics
        metrics = compute_metrics(y_val, y_pred, val_probs, n_classes)
        metrics["fold"] = fold
        metrics["fit_seconds"] = round(fit_time, 2)
        fold_results.append(metrics)

        # Save best model to disk immediately (don't keep in memory)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            if is_dl:
                torch.save(model.state_dict(), model_path.with_suffix(".pt"))
            else:
                joblib.dump({"model": model, "features": feature_names}, model_path)

        log.info(
            "[%s][%s][%s] fold %d — F1=%.4f  Acc=%.4f  (%.1fs)",
            dataset_name, task, model_name, fold,
            metrics["f1_macro"], metrics["accuracy"], fit_time,
        )

        # Aggressive memory cleanup between folds
        del X_train, X_val, y_train, y_val, model, y_pred, val_probs
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Saved best %s → %s", model_name, model_path.name)

    # Average across folds
    avg = {}
    numeric_keys = [k for k in fold_results[0] if k not in ("fold",)]
    for k in numeric_keys:
        vals = [r[k] for r in fold_results]
        avg[k] = round(np.mean(vals), 4)
        avg[f"{k}_std"] = round(np.std(vals), 4)
    avg["model"] = model_name
    avg["dataset"] = dataset_name
    avg["task"] = task
    avg["n_folds"] = N_FOLDS
    return avg


def _existing_combos() -> set:
    """Return set of (model, dataset, task) already in benchmark_results.csv."""
    out = TABLES / "benchmark_results.csv"
    if not out.exists():
        return set()
    df = pd.read_csv(out)
    return set(zip(df["model"], df["dataset"], df["task"]))


def run_dataset(dataset_name: str, model_filter: str = None,
                skip_existing: bool = True) -> list:
    """Train all models on one dataset, both binary & multi-class."""
    import pyarrow.parquet as pq

    path = DATA_CLEAN / f"{dataset_name}.parquet"
    if not path.exists():
        log.warning("Skipping %s — not preprocessed yet", dataset_name)
        return []

    # Memory-efficient loading: read schema first, then only needed columns
    schema = pq.read_schema(path)
    exclude = {"label_original", "label_binary", "label_multi", "dataset"}
    feat_cols = [f.name for f in schema if f.name not in exclude]

    # Read labels separately (tiny memory footprint)
    labels_df = pd.read_parquet(path, columns=["label_binary", "label_multi"])
    y_binary = labels_df["label_binary"].values.copy()
    y_multi = labels_df["label_multi"].values.copy()
    n_rows = len(labels_df)
    del labels_df
    gc.collect()

    # Build feature matrix column-by-column into pre-allocated float32 array
    # (avoids pandas intermediate copies that caused OOM)
    X = np.empty((n_rows, len(feat_cols)), dtype=np.float32)
    chunk_cols = 10
    for start in range(0, len(feat_cols), chunk_cols):
        batch = feat_cols[start : start + chunk_cols]
        chunk_df = pd.read_parquet(path, columns=batch)
        for j, col in enumerate(batch):
            X[:, start + j] = chunk_df[col].to_numpy(dtype=np.float32)
        del chunk_df
    gc.collect()
    log.info("Loaded %s: %d rows × %d features (%.1f MB float32)",
             dataset_name, n_rows, len(feat_cols),
             X.nbytes / (1024 ** 2))

    models_to_run = [model_filter] if model_filter else MODEL_NAMES
    done = _existing_combos() if skip_existing else set()
    all_results = []

    for task, label_col in [("binary", "label_binary"), ("multi", "label_multi")]:
        y = y_binary if task == "binary" else y_multi
        n_classes = len(np.unique(y))
        log.info("── %s / %s — %d classes, %d samples, %d features ──",
                 dataset_name, task, n_classes, len(y), len(feat_cols))

        for mname in models_to_run:
            if mname not in MODEL_NAMES:
                log.warning("Unknown model: %s", mname)
                continue
            if (mname, dataset_name, task) in done:
                log.info("Skipping %s/%s/%s — already completed", mname, dataset_name, task)
                continue

            # Free memory from previous model
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                result = train_single(mname, X, y, feat_cols, n_classes, dataset_name, task)
                all_results.append(result)
                # Incremental save after every model run
                _save_results(all_results)
            except Exception as exc:
                log.error("FAILED %s/%s/%s: %s", mname, dataset_name, task, exc, exc_info=True)
                continue

    # Cleanup
    del X
    gc.collect()

    return all_results


def _save_results(results: list) -> None:
    """Append-safe incremental save: merge with any existing results."""
    out = TABLES / "benchmark_results.csv"
    new_df = pd.DataFrame(results)
    if out.exists():
        old_df = pd.read_csv(out)
        # Drop duplicates (same model+dataset+task) keeping new
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["model", "dataset", "task"], keep="last",
        )
    else:
        combined = new_df
    combined.to_csv(out, index=False)
    log.info("Results saved → %s (%d rows)", out, len(combined))


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IDS benchmark models")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-train even if results exist")
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else DATASETS
    all_results = []

    for ds in targets:
        results = run_dataset(ds, model_filter=args.model,
                              skip_existing=not args.no_skip)
        all_results.extend(results)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n" + results_df.to_string(index=False))
