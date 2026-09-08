"""
Revision E8a — inference-latency measurement for all saved models.

Addresses R1 (operational IDS are latency-sensitive; training time alone is
insufficient).  Loads every trained model saved by the original benchmark
(outputs/models) and measures on CPU:

    batch_us_per_sample   — 10k-row batch prediction, microseconds/sample
    single_ms_median      — median latency of 200 single-row predict() calls

DL checkpoints (.pt) are rebuilt from the architecture definitions in
model.py and measured the same way.

Usage:  python e8a_latency.py
"""

import time

import joblib
import numpy as np
import pandas as pd

from utils import (
    dataset_path,
    DATA_CLEAN, ORIG_MODELS, TABLES, RANDOM_STATE, PAPER_DATASETS,
    safe_feature_cols, log,
)

OUT = TABLES / "e8a_inference_latency.csv"
BATCH_N = 10_000
SINGLE_N = 200


def _sample_X(ds, feat_names, n):
    df = pd.read_parquet(dataset_path(ds), columns=feat_names)
    idx = np.random.RandomState(RANDOM_STATE).choice(len(df), min(n, len(df)),
                                                     replace=False)
    X = df.iloc[idx].to_numpy(dtype=np.float32)
    lo, hi = X.min(0), X.max(0)
    sc = np.where(hi - lo == 0, 1, hi - lo)
    return (X - lo) / sc


def _measure(predict, X):
    predict(X[:100])  # warm-up
    t0 = time.perf_counter()
    predict(X)
    batch_us = (time.perf_counter() - t0) / len(X) * 1e6

    times = []
    for i in range(SINGLE_N):
        row = X[i % len(X):i % len(X) + 1]
        t0 = time.perf_counter()
        predict(row)
        times.append((time.perf_counter() - t0) * 1e3)
    return batch_us, float(np.median(times))


def _parse_stem(stem):
    """Split '{dataset}_{task}_{Model}' — dataset names contain underscores."""
    for d in PAPER_DATASETS:
        for t in ("binary", "multi"):
            prefix = f"{d}_{t}_"
            if stem.startswith(prefix):
                return d, t, stem[len(prefix):]
    return None, None, None


def main():
    rows = []
    for path in sorted(ORIG_MODELS.glob("*.joblib")):
        stem = path.stem
        ds, task, mname = _parse_stem(stem)
        if ds is None:
            continue
        try:
            bundle = joblib.load(path)
            model, feats = bundle["model"], bundle["features"]
            X = _sample_X(ds, feats, BATCH_N)
            batch_us, single_ms = _measure(model.predict, X)
            rows.append(dict(dataset=ds, task=task, model=mname,
                             batch_us_per_sample=round(batch_us, 2),
                             single_ms_median=round(single_ms, 3),
                             n_features=len(feats), device="cpu"))
            log.info("%-40s batch=%.1f µs/sample  single=%.2f ms",
                     stem, batch_us, single_ms)
        except Exception as exc:
            log.error("FAILED %s: %s", stem, exc)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    # DL checkpoints
    import torch
    import model as model_mod
    model_mod._ensure_torch()
    for path in sorted(ORIG_MODELS.glob("*.pt")):
        stem = path.stem
        ds, task, mname = _parse_stem(stem)
        if ds is None:
            continue
        try:
            df_cols = pd.read_parquet(dataset_path(ds)).columns
            feats = safe_feature_cols(df_cols)
            state = torch.load(path, map_location="cpu", weights_only=True)
            n_classes = state[[k for k in state if k.endswith("fc.3.bias")][0]].shape[0] \
                if any(k.endswith("fc.3.bias") for k in state) else \
                list(state.values())[-1].shape[0]
            net = (model_mod._make_cnn1d if mname == "CNN1D"
                   else model_mod._make_bilstm)(len(feats), n_classes)
            net.load_state_dict(state)
            net.eval()
            X = _sample_X(ds, feats, BATCH_N)

            def predict(arr, _net=net):
                with torch.no_grad():
                    return _net(torch.tensor(arr)).argmax(1).numpy()

            batch_us, single_ms = _measure(predict, X)
            rows.append(dict(dataset=ds, task=task, model=mname,
                             batch_us_per_sample=round(batch_us, 2),
                             single_ms_median=round(single_ms, 3),
                             n_features=len(feats), device="cpu"))
            log.info("%-40s batch=%.1f µs/sample  single=%.2f ms",
                     stem, batch_us, single_ms)
        except Exception as exc:
            log.error("FAILED %s: %s", stem, exc)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    log.info("E8a done → %s (%d models)", OUT, len(rows))


if __name__ == "__main__":
    main()
