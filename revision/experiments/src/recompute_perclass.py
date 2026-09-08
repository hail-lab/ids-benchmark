"""
Recompute per-class F1 on the label-corrected datasets.

Needed because the revision merged UNSW-NB15's duplicated ``Backdoor'' and
``Backdoors'' classes (11 classes -> 10).  The per-class table in the submitted
paper still reports them separately, with the anomalous pair F1 = 0.152 and
0.019 that the merge exists to remove, so that table has to be recomputed
rather than edited by hand.

CICIDS2017 is also recomputed: its web-attack class names were mojibake in the
submitted version.  Only the names change there - no classes are merged - so
the F1 values should reproduce, which doubles as a check that this script
agrees with the original pipeline.

Uses the same estimator and protocol as the benchmark: XGBoost with the
submitted hyperparameters, five-fold stratified CV, GPU where available.

Usage:  python recompute_perclass.py [--dataset unsw_nb15]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from utils import (
    TABLES, RANDOM_STATE, N_JOBS, PAPER_DATASETS, dataset_path,
    safe_feature_cols, log,
)

OUT = TABLES / "per_class_f1_corrected.csv"
N_FOLDS = 5

# the submitted benchmark's XGBoost configuration
HP = dict(objective="multi:softprob", n_estimators=400, max_depth=6,
          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=N_JOBS)


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        pass
    return {"tree_method": "hist"}


def run(ds):
    df = pd.read_parquet(dataset_path(ds))
    feats = safe_feature_cols(df.columns)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy()
    names = df.groupby("label_multi")["label_original"].first().to_dict()
    del df

    log.info("[%s] %d rows x %d features, %d classes", ds, len(X), len(feats),
             len(np.unique(y)))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    reports = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = XGBClassifier(**HP, **_device())
        model.fit(X[tr], y[tr])
        model.set_params(device="cpu")
        pred = model.predict(X[te])
        reports.append(classification_report(y[te], pred, output_dict=True,
                                             zero_division=0))
        log.info("[%s] fold %d/%d done (%.0fs elapsed)", ds, fold, N_FOLDS,
                 time.time() - t0)

    rows = []
    for c in sorted(np.unique(y)):
        f1s = [r.get(str(c), {}).get("f1-score", 0.0) for r in reports]
        rows.append(dict(dataset=ds, class_id=int(c),
                         class_name=names.get(c, str(c)),
                         samples=int((y == c).sum()),
                         f1_mean=round(float(np.mean(f1s)), 4),
                         f1_std=round(float(np.std(f1s)), 4)))
    out = pd.DataFrame(rows).sort_values("samples", ascending=False)
    print(f"\n=== {ds} ===")
    print(out[["class_name", "samples", "f1_mean", "f1_std"]]
          .to_string(index=False))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=PAPER_DATASETS)
    a = p.parse_args()

    targets = [a.dataset] if a.dataset else PAPER_DATASETS
    frames = []
    if OUT.exists():
        prev = pd.read_csv(OUT)
        frames.append(prev[~prev.dataset.isin(targets)])
    for ds in targets:
        frames.append(run(ds))
    pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
    log.info("saved -> %s", OUT)
