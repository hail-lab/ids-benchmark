"""
Recompute the UNSW-NB15 multi-class benchmark on the label-corrected data.

Why this exists
---------------
The revision merged UNSW-NB15's duplicated ``Backdoor``/``Backdoors`` labels,
reducing that task from 11 classes to 10.  Every revision experiment reads the
corrected copy through ``utils.dataset_path`` and so reports 10-class figures,
but the main benchmark table was carried over from the original submission and
still holds the 11-class numbers -- while being labelled "10 classes".  Macro F1
is an unweighted mean over classes, so the merge moves it materially: XGBoost
goes from 0.5214 (11 classes, including the anomalous 0.152/0.019 pair) to
roughly 0.571.

Only UNSW-NB15 multi-class is affected.  CICIDS2017's correction changed label
*names* only (mojibake), and ToN-IoT was not corrected, so their numbers stand.

This re-runs all eight models under the submitted protocol -- same
hyperparameters, same five-fold stratified split, same seed -- changing nothing
but the labels.  The three tree models double as a check: e1_splits.py already
computed them on the corrected data under its own stratified scheme, so the
values here should agree with it closely.

Model artifacts are written to a revision-local directory; the submitted
artifacts in outputs/models/ are left untouched.

Usage:  python recompute_unsw_multi.py [--models XGBoost,SVM] [--dry-run]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import TABLES, REVISION, DATA_CLEAN, dataset_path, safe_feature_cols, log

# the submitted pipeline lives in the project's src/, one level above the
# revision folder; import it rather than duplicating its configuration
SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as bench          # noqa: E402  (needs sys.path set first)

OUT = TABLES / "unsw_multi_corrected.csv"
MODELS = ["XGBoost", "RandomForest", "LightGBM", "SVM", "kNN", "MLP",
          "CNN1D", "BiLSTM"]
DATASET, TASK = "unsw_nb15", "multi"


def _load(path: Path):
    """Feature matrix and multi-class labels, identifiers excluded.

    Note that ``model.get_feature_cols`` cannot be used here: it drops only the
    label columns, and unlike the other two cleaned parquets the UNSW-NB15 one
    retains srcip/dstip/sport/dsport/stime/ltime so that the leakage harness can
    add them back.  Feeding its output to the model would train on identifiers.
    ``safe_feature_cols`` applies the identifier filter and yields the 39
    features the paper reports.
    """
    df = pd.read_parquet(path)
    feats = safe_feature_cols(df.columns)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy()
    del df
    return X, y, feats


def main(models, dry, validate):
    # keep the submitted model artifacts intact
    local_models = REVISION / "results_r1" / "models_corrected"
    local_models.mkdir(parents=True, exist_ok=True)
    bench.MODEL_DIR = local_models

    if validate:
        # reproduce the submitted 11-class XGBoost figure (0.5214) to confirm
        # this harness matches the original protocol before trusting its
        # corrected output
        Xo, yo, fo = _load(DATA_CLEAN / f"{DATASET}.parquet")
        log.info("validation run: %d features, %d classes",
                 len(fo), len(np.unique(yo)))
        r = bench.train_single("XGBoost", Xo, yo, fo, int(len(np.unique(yo))),
                               DATASET, TASK)
        print(f"\nvalidation: XGBoost on original labels "
              f"({len(np.unique(yo))} classes) = {r['f1_macro']:.4f}"
              f"   (submitted value 0.5214)")
        return

    path = dataset_path(DATASET)
    log.info("reading %s", path)
    X, y, feats = _load(path)
    n_classes = int(len(np.unique(y)))

    log.info("%s/%s: %d rows x %d features, %d classes",
             DATASET, TASK, len(X), len(feats), n_classes)
    if n_classes != 10:
        raise SystemExit(f"expected 10 classes after the merge, got {n_classes}"
                         " -- is data_r1/unsw_nb15.parquet the corrected copy?")
    if len(feats) != 39:
        raise SystemExit(f"expected 39 features, got {len(feats)}")

    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        done = set(prev.model)
        log.info("resuming; already done: %s", sorted(done))

    if dry:
        print(f"would run: {[m for m in models if m not in done]}")
        return

    for name in models:
        if name in done:
            continue
        t0 = time.time()
        log.info("=== %s ===", name)

        note = ""
        if name == "LightGBM" and bench.HP["LightGBM"].get("device") == "gpu":
            # LightGBM's GPU histogram builder aborts on this data with
            # "Check failed: (best_split_info.left_count) > (0)". The CPU
            # builder is unaffected and the model specification is unchanged,
            # so the fit falls back to CPU and the fallback is recorded here
            # and in the manuscript rather than being silently absorbed.
            bench.HP["LightGBM"]["device"] = "cpu"
            note = "LightGBM refitted on CPU (GPU histogram builder aborts)"
            log.warning("%s", note)

        res = bench.train_single(name, X, y, feats, n_classes, DATASET, TASK)
        res["n_classes"] = n_classes
        res["labels"] = "corrected (Backdoor merged)"
        res["note"] = note

        row = pd.DataFrame([res])
        if OUT.exists():
            row = pd.concat([pd.read_csv(OUT), row], ignore_index=True)
        row.to_csv(OUT, index=False)
        log.info("%s: F1 = %.4f +- %.4f  (%.0fs)  -> %s",
                 name, res["f1_macro"], res["f1_macro_std"],
                 time.time() - t0, OUT.name)

    # --- report against the submitted 11-class numbers -------------------
    new = pd.read_csv(OUT)
    old = pd.read_csv(REVISION.parent / "outputs" / "tables" /
                      "benchmark_results.csv")
    old = old[(old.dataset == DATASET) & (old.task == TASK)]
    print(f"\n{'model':14s} {'11-class':>9s} {'10-class':>9s} {'delta':>8s}")
    for _, r in new.iterrows():
        o = old[old.model == r.model]
        ov = float(o.f1_macro.iloc[0]) if len(o) else float("nan")
        print(f"{r.model:14s} {ov:9.4f} {r.f1_macro:9.4f} "
              f"{r.f1_macro - ov:+8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(MODELS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="reproduce the submitted 11-class XGBoost figure")
    a = ap.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()], a.dry_run,
         a.validate)
