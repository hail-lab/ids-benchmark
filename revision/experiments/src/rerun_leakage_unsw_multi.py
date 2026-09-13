"""
Re-run the UNSW-NB15 multi-class leakage comparison on the corrected labels.

Why
---
The leakage experiment predates the Backdoor/Backdoors merge, so its UNSW-NB15
multi-class arm describes an 11-class task. Its clean-arm values are the old
benchmark numbers (XGBoost 0.5214 where Table 5 now reports 0.5745), which
makes fig_leakage_unsw_nb15_multi disagree visibly with the table, and the
aggregate inflation statistics are computed partly from those cells.

Only the leaky arm needs running: recompute_unsw_multi.py already produced the
clean arm on the corrected labels.

How
---
The cleaned UNSW-NB15 parquet retains srcip, sport, dstip, dsport, stime and
ltime -- cleanliness is enforced when the feature matrix is built, not when the
file is written, precisely so that this experiment can add them back. So both
arms run on *identical rows* with identical labels, and the only difference is
which columns enter the model. That is a tighter pairing than rebuilding a
separate leaky parquet from the raw files, which is what the original script
did, and it removes sampling noise from the delta.

Protocol is otherwise unchanged: same five folds, same seed, same fold-local
scaling, same hyperparameters.

Models are run cheapest first so that a failure late in the run still leaves
the earlier results on disk; the MLP accounts for most of the wall time.

Usage:  python rerun_leakage_unsw_multi.py [--check] [--models kNN,XGBoost]
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from utils import (TABLES, REVISION, META_COLS, dataset_path,
                   safe_feature_cols, log)

SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model as bench          # noqa: E402

OUT = TABLES / "unsw_multi_leaky_corrected.csv"
CLEAN = TABLES / "unsw_multi_corrected.csv"
DATASET, TASK = "unsw_nb15", "multi"

# cheapest first; the five models the submitted leakage experiment covers for
# this dataset and task (BiLSTM diverged, LightGBM and 1D-CNN did not converge)
MODELS = ["kNN", "XGBoost", "RandomForest", "SVM", "MLP"]


def load_leaky():
    """Feature matrix retaining the identifier columns, plus labels."""
    df = pd.read_parquet(dataset_path(DATASET))
    clean_feats = safe_feature_cols(df.columns)
    leaky_feats = [c for c in df.columns if c not in META_COLS]
    added = [c for c in leaky_feats if c not in clean_feats]

    X = df[leaky_feats].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy()
    del df
    return X, y, leaky_feats, clean_feats, added


def main(models, check_only):
    X, y, leaky, clean, added = load_leaky()
    n_cls = int(len(np.unique(y)))

    print(f"rows          : {len(X):,}")
    print(f"clean features: {len(clean)}")
    print(f"leaky features: {len(leaky)}  (+{len(added)})")
    print(f"added back    : {added}")
    print(f"classes       : {n_cls}")

    if n_cls != 10:
        raise SystemExit(f"expected 10 classes after the merge, got {n_cls}")
    if len(added) != 8:
        raise SystemExit(f"expected 8 identifier columns, got {len(added)}")

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).model)
        print(f"already done  : {sorted(done)}")
    todo = [m for m in models if m not in done]
    print(f"to run        : {todo}")

    if check_only:
        print("\n--check: nothing was trained")
        return

    # keep the submitted model artifacts intact
    local = REVISION / "results_r1" / "models_leaky_corrected"
    local.mkdir(parents=True, exist_ok=True)
    bench.MODEL_DIR = local

    for name in todo:
        t0 = time.time()
        log.info("=== %s (leaky, %d features) ===", name, len(leaky))
        res = bench.train_single(name, X.copy(), y, leaky, n_cls, DATASET, TASK)
        res["arm"] = "leaky"
        res["n_features"] = len(leaky)
        res["n_classes"] = n_cls
        row = pd.DataFrame([res])
        if OUT.exists():
            row = pd.concat([pd.read_csv(OUT), row], ignore_index=True)
        row.to_csv(OUT, index=False)
        log.info("%s: F1 = %.4f +- %.4f  (%.0fs)", name, res["f1_macro"],
                 res["f1_macro_std"], time.time() - t0)

    # --- inflation against the corrected clean arm -----------------------
    if not OUT.exists() or not CLEAN.exists():
        return
    lk = pd.read_csv(OUT).set_index("model")
    cl = pd.read_csv(CLEAN).set_index("model")
    shared = [m for m in MODELS if m in lk.index and m in cl.index]
    print(f"\n{'model':14s} {'clean':>8s} {'leaky':>8s} {'inflation':>10s}")
    for m in shared:
        c, l = cl.loc[m, "f1_macro"], lk.loc[m, "f1_macro"]
        print(f"{m:14s} {c:8.4f} {l:8.4f} {l - c:+10.4f}")
    if len(shared) == len(MODELS):
        d = [lk.loc[m, "f1_macro"] - cl.loc[m, "f1_macro"] for m in shared]
        print(f"\nUNSW-NB15 multi-class inflation: mean {np.mean(d):+.4f}, "
              f"median {np.median(d):+.4f}")
        print("Next: python replot_leakage_unsw.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--check", action="store_true",
                    help="validate inputs and print the plan without training")
    a = ap.parse_args()
    main([m.strip() for m in a.models.split(",") if m.strip()], a.check)
