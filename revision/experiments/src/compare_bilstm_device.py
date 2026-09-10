"""
Is the BiLSTM refit's device choice material? Measure it on one fold.

The submitted pipeline pins the BiLSTM to CPU, where refitting UNSW-NB15
multi-class on the corrected labels costs 6.3 hours per fold -- about 31 hours
for the five folds needed to fill a single table cell. Before paying that, this
reproduces fold 1 exactly and changes only the device.

Everything else is held identical to ``model.train_single``: the same
StratifiedKFold(5, shuffle=True, random_state=42) split, the same fold-local
in-place min--max scaling, the same architecture from ``_make_bilstm``, the same
Adam at lr 1e-3, batch 1024, 50 epochs with patience 10 and
ReduceLROnPlateau, and the same class-weighted cross-entropy. No mixed
precision and no resident batching, so the only difference from the CPU run is
where the kernels execute.

The CPU result for this fold is 0.4761 (22,723 s). If the GPU agrees closely,
the device is immaterial at the precision the paper reports and the remaining
folds can be run on GPU; if it does not, that is worth knowing before any
cross-device number reaches a table.

Usage:  python compare_bilstm_device.py [--cpu-reference 0.4761]
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import REVISION, RANDOM_STATE, dataset_path, safe_feature_cols, log

SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as bench          # noqa: E402
from evaluation import compute_metrics   # noqa: E402

DATASET, TASK, FOLD = "unsw_nb15", "multi", 1


def main(reference: float):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- nothing to compare against")
    bench._ensure_torch()

    df = pd.read_parquet(dataset_path(DATASET))
    feats = safe_feature_cols(df.columns)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy()
    n_cls = int(len(np.unique(y)))
    del df

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, va = next(iter(skf.split(X, y)))          # fold 1, as train_single numbers it
    X_tr, X_va = X[tr].copy(), X[va]
    y_tr, y_va = y[tr], y[va]

    # identical fold-local scaling
    fmin = X_tr.min(axis=0)
    scale = X_tr.max(axis=0) - fmin
    scale[scale == 0] = 1.0
    X_tr -= fmin
    X_tr /= scale
    X_va = (X_va - fmin) / scale

    log.info("fold %d: %d train / %d val rows, %d features, %d classes",
             FOLD, len(X_tr), len(X_va), len(feats), n_cls)
    log.info("epochs=%d batch=%d lr=%g patience=%d",
             bench.DL_EPOCHS, bench.DL_BATCH, bench.DL_LR, bench.DL_PATIENCE)

    t0 = time.time()
    _, probs = bench._train_dl("BiLSTM", X_tr, y_tr, X_va, y_va, n_cls,
                               torch.device("cuda"))
    elapsed = time.time() - t0
    m = compute_metrics(y_va, probs.argmax(axis=1), probs, n_cls)

    gpu = m["f1_macro"]
    print("\n" + "=" * 62)
    print(f"BiLSTM / {DATASET} / {TASK} / fold {FOLD}")
    print(f"  CPU (reference) : {reference:.4f}   22,723 s")
    print(f"  GPU (this run)  : {gpu:.4f}   {elapsed:,.0f} s"
          f"   ({22723 / max(elapsed, 1):.0f}x faster)")
    print(f"  difference      : {gpu - reference:+.4f}")
    print(f"  balanced acc    : {m['balanced_accuracy']:.4f}")
    print("=" * 62)
    verdict = ("immaterial at the reported precision -- safe to finish on GPU"
               if abs(gpu - reference) <= 0.005 else
               "MATERIAL -- do not mix devices within a table")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-reference", type=float, default=0.4761)
    a = ap.parse_args()
    main(a.cpu_reference)
