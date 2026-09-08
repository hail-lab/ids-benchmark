"""
Revision E5 (GPU) - model-comparison fairness checks, built to run on a free
Colab T4 inside one session.

Addresses R1.15 (no class_weight for RF/SVM/MLP), R1.16 / R3.4 / R6.5 (SVM and
k-NN capped at 50k while other models saw up to 1.6M rows).

Why this is a GPU script
------------------------
A first, CPU-only version of Part B grew the RBF-SVM training set at fixed
max_iter=5000 and produced a spurious result: macro F1 on CICIDS2017 fell from
0.890 at 50k to 0.252 at 100k while taking 2.2 h to fit.  That is iteration
starvation - the same optimiser budget spread over a larger problem - not a
property of SVM given more data.  Separating the two effects needs several
SVM fits at several sizes, which is impractical on CPU and routine on GPU.

Part A - imbalance-handling parity (binary + multi, 3 datasets, N_FOLDS folds)
    RandomForest : default            vs  class_weight="balanced"
    SVM (RBF)    : default            vs  class_weight="balanced"
    MLP          : plain cross-entropy vs class-weighted cross-entropy
    The MLP is a PyTorch reimplementation of the benchmark's 128-64 network
    precisely so that a *weighted loss* arm exists; scikit-learn's
    MLPClassifier supports neither class_weight nor sample_weight, which is
    why the submitted paper had no such arm at all.

Part B - subsampling sensitivity (binary task)
    kNN       : 50k / 100k / 200k
    SVM RBF   : 50k / 100k at max_iter=5000     (the paper's configuration)
    SVM RBF   : 50k / 100k at max_iter=50000    (does the cap bind?)
    LinearSVC : 50k / 200k / full training fold (a scalable SVM variant)
    Every fit records whether the optimiser hit its iteration cap, so a poor
    score can be attributed to non-convergence rather than to the data.

Backend
-------
Uses cuML (RAPIDS) for RF / SVM / kNN when a GPU is present, and falls back to
scikit-learn otherwise.  The backend actually used is recorded in every row.
Comparisons are always made *within* one backend (default vs balanced, or
across n at fixed configuration), so the library choice never confounds a
comparison; it only needs to be disclosed, and the `backend` column does that.

Resumable: results are keyed and already-completed cells are skipped, so a
Colab disconnect costs at most one fit.

Usage
-----
    python e5_fairness_gpu.py --part B
    python e5_fairness_gpu.py --part A
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

from utils import (
    dataset_path,
    DATA_CLEAN, TABLES, RANDOM_STATE, N_JOBS, PAPER_DATASETS,
    safe_feature_cols, log,
)

OUT_A = TABLES / "e5_class_weight.csv"
OUT_B = TABLES / "e5_subsample_curve.csv"

SVM_CAP = 50_000
N_FOLDS = 3
KEY_A = ["dataset", "task", "model", "config", "fold"]
KEY_B = ["dataset", "model", "config", "n_train", "fold"]

# Cells too costly to repeat on every fold (still run on fold 1).
COSTLY_FOLD1_ONLY = {("SVM", "rbf_maxiter50000_100k")}


# ------------------------------------------------------------- backend

def _detect_backend():
    """Return ('cuml'|'sklearn', module namespace dict)."""
    try:
        import cuml  # noqa: F401
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.neighbors import KNeighborsClassifier as cuKNN
        from cuml.svm import SVC as cuSVC, LinearSVC as cuLinearSVC
        log.info("Backend: cuML %s (GPU)", cuml.__version__)
        return "cuml", dict(RF=cuRF, KNN=cuKNN, SVC=cuSVC, LinearSVC=cuLinearSVC)
    except Exception as exc:
        from sklearn.ensemble import RandomForestClassifier as skRF
        from sklearn.neighbors import KNeighborsClassifier as skKNN
        from sklearn.svm import SVC as skSVC, LinearSVC as skLinearSVC
        log.warning("cuML unavailable (%s) - falling back to scikit-learn (CPU). "
                    "Expect Part B to be slow.", type(exc).__name__)
        return "sklearn", dict(RF=skRF, KNN=skKNN, SVC=skSVC, LinearSVC=skLinearSVC)


BACKEND, M = _detect_backend()


def _rf():
    if BACKEND == "cuml":
        return M["RF"](n_estimators=200, random_state=RANDOM_STATE)
    return M["RF"](n_estimators=200, n_jobs=N_JOBS, random_state=RANDOM_STATE)


def _svc(max_iter):
    kw = dict(kernel="rbf", max_iter=max_iter)
    if BACKEND != "cuml":
        kw.update(random_state=RANDOM_STATE)
    return M["SVC"](**kw)


def balanced_resample(X, y, seed=RANDOM_STATE):
    """Class-balanced bootstrap of the training fold, same size as the input.

    Why this rather than `class_weight="balanced"`.  An earlier version passed
    class weights to the estimators, and it silently did nothing on the GPU
    backend: cuML's RandomForest accepts no sample weights, so the "balanced"
    arm refit the identical model and came out bit-identical to the default on
    all 18 folds, while cuML's SVC prints "Sample weights are currently ignored
    for multi class classification".  A comparison that quietly reduces to
    fitting the same model twice is worse than no comparison at all.

    Resampling achieves the same intent - every class contributes equally to
    the fit - without depending on backend support, and it is verifiable: the
    returned labels are balanced by construction.  Rare classes are duplicated,
    which is exactly what an inverse-frequency weight does.  Total size is held
    at len(X) so the balanced and default arms train on equally many rows and
    differ only in class composition.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = max(1, len(y) // len(classes))
    picks = []
    for c in classes:
        idx = np.where(y == c)[0]
        # with replacement so classes smaller than per_class are duplicated
        picks.append(rng.choice(idx, per_class, replace=len(idx) < per_class))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return X[sel], y[sel]


def _knn():
    if BACKEND == "cuml":
        return M["KNN"](n_neighbors=5)
    return M["KNN"](n_neighbors=5, n_jobs=N_JOBS)


def _linear_svc(max_iter=5000):
    if BACKEND == "cuml":
        return M["LinearSVC"](max_iter=max_iter)
    return M["LinearSVC"](dual="auto", max_iter=max_iter,
                          random_state=RANDOM_STATE)


# ------------------------------------------------------------- PyTorch MLP

def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


# A large batch makes each epoch cheap but takes 8x fewer optimiser steps per
# epoch than a 1024 batch, so the epoch budget is raised to compensate.  The
# budget is deliberately generous: this experiment exists to answer a fairness
# objection, and reporting a model that stopped because it ran out of epochs
# would repeat exactly the iteration-starvation problem documented for SVM in
# Part B.  Early stopping ends the fit as soon as validation loss plateaus, so
# in practice most fits finish well inside the cap, and `converged` is recorded
# for every fit so any that do not are visible.
MLP_BATCH = 8192
MLP_MAX_EPOCHS = 600
MLP_PATIENCE = 15


def train_mlp(X_tr, y_tr, X_te, n_classes, weighted, max_epochs=MLP_MAX_EPOCHS,
              patience=MLP_PATIENCE, batch=MLP_BATCH, lr=1e-3, val_frac=0.2):
    """128-64 ReLU MLP mirroring the benchmark's architecture.

    `weighted=True` uses class-weighted cross-entropy - the arm scikit-learn's
    MLPClassifier cannot express, and the one R1.15 asked for.
    Returns (y_pred, converged); converged is False if the epoch budget ran out
    before early stopping triggered.

    Performance note.  An earlier version streamed 1024-row batches through a
    DataLoader of CPU tensors, which made a single fit on CICIDS2017 take
    ~18 minutes: for a network this small the cost is per-batch Python and
    host-to-device copies, not arithmetic, so the GPU idled.  Here the fold is
    moved to the device once and batches are taken by slicing a shuffled index,
    with a correspondingly larger batch.  Both imbalance arms use identical
    settings, so the comparison the experiment exists to make is unaffected.
    """
    torch, nn = _torch()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)

    n_val = max(1, int(len(X_tr) * val_frac))
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(len(X_tr))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Resident on the device for the whole fit; a 1.6M x 77 float32 fold is
    # ~0.5 GB, comfortably inside a T4.  Falls back to CPU tensors if the
    # allocation fails, which only costs speed.
    try:
        Xt = torch.as_tensor(X_tr[tr_idx]).to(dev)
        yt = torch.as_tensor(y_tr[tr_idx], dtype=torch.long).to(dev)
        Xv = torch.as_tensor(X_tr[va_idx]).to(dev)
        yv = torch.as_tensor(y_tr[va_idx], dtype=torch.long).to(dev)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        dev = torch.device("cpu")
        Xt = torch.as_tensor(X_tr[tr_idx])
        yt = torch.as_tensor(y_tr[tr_idx], dtype=torch.long)
        Xv = torch.as_tensor(X_tr[va_idx])
        yv = torch.as_tensor(y_tr[va_idx], dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    ).to(dev)

    if weighted:
        counts = np.maximum(np.bincount(y_tr, minlength=n_classes), 1).astype(np.float32)
        w = (1.0 / counts) * counts.sum() / n_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = len(Xt)

    best, bad, best_state, converged = float("inf"), 0, None, False
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=dev)
        for i in range(0, n_train, batch):
            idx = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            criterion(model(Xt[idx]), yt[idx]).backward()
            opt.step()

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), 65_536):
                xb, yb = Xv[i:i + 65_536], yv[i:i + 65_536]
                tot += criterion(model(xb), yb).item() * len(yb)
        vloss = tot / len(Xv)

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                converged = True
                break

    del Xt, yt, Xv, yv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
        model.to(dev)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), 8192):
            xb = torch.tensor(X_te[i:i + 8192]).to(dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds), converged


# ------------------------------------------------------------- helpers

def _metrics(y_true, y_pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
    }


def _load(ds):
    df = pd.read_parquet(dataset_path(ds))
    cols = safe_feature_cols(df.columns)
    X = df[cols].to_numpy(dtype=np.float32)
    ys = {t: df[f"label_{t}"].to_numpy(dtype=np.int32) for t in ("binary", "multi")}
    del df
    return X, ys


def _scale(X_tr, X_te):
    lo, hi = X_tr.min(0), X_tr.max(0)
    sc = np.where(hi - lo == 0, 1, hi - lo).astype(np.float32)
    return ((X_tr - lo) / sc).astype(np.float32), ((X_te - lo) / sc).astype(np.float32)


def _subsample(X, y, cap, seed=RANDOM_STATE):
    if cap is None or len(X) <= cap:
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), cap, replace=False)
    return X[idx], y[idx]


def _fit_predict(model, X_tr, y_tr, X_te, sample_weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if sample_weight is not None:
            try:
                model.fit(X_tr, y_tr, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te)).astype(int).ravel()


def _converged(model, max_iter):
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None or max_iter is None or max_iter < 0:
        return None
    try:
        return bool(np.max(np.atleast_1d(np.asarray(n_iter))) < max_iter)
    except Exception:
        return None


def _append(path, row, key):
    """Append one result row, keyed so a re-run overwrites rather than duplicates.

    `converged` is None for models with no iteration cap (k-NN), which makes it
    an all-NA column on the first write.  Concatenating an all-NA column raises
    a pandas FutureWarning about dtype inference, so the column is given an
    explicit object dtype up front; the values written are unchanged.
    """
    df = pd.DataFrame([row])
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype(object)
    if path.exists():
        old = pd.read_csv(path)
        if "converged" in old.columns:
            old["converged"] = old["converged"].astype(object)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=key, keep="last")
    df.to_csv(path, index=False)


def _done(path, key):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if not set(key).issubset(df.columns):
        return set()
    return set(map(tuple, df[key].astype(str).values))


def _balanced_sw(y):
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
    counts[counts == 0] = 1
    w = len(y) / (len(np.unique(y)) * counts)
    return w[y].astype(np.float32)


# ------------------------------------------------------------- Part A

def part_a(datasets=None):
    """`datasets` restricts the run to a subset, so a long session can be split
    into bounded chunks; completed cells are skipped either way."""
    done = _done(OUT_A, KEY_A)
    for ds in (datasets or PAPER_DATASETS):
        X, ys = _load(ds)
        for task in ("binary", "multi"):
            y = ys[task]
            n_classes = int(y.max()) + 1
            skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            for fold, (tr, te) in enumerate(skf.split(X, y), 1):
                X_tr, X_te = _scale(X[tr].copy(), X[te].copy())
                y_tr, y_te = y[tr], y[te]
                X_svm, y_svm = _subsample(X_tr, y_tr, SVM_CAP)

                # `balanced` arms train on a class-balanced bootstrap of the
                # same fold (see balanced_resample); the MLP instead uses a
                # weighted loss, which is the arm scikit-learn cannot express
                # and which R1.15 asked about directly.
                jobs = [
                    ("RandomForest", "default",           False, X_tr,  y_tr),
                    ("RandomForest", "balanced_resample", True,  X_tr,  y_tr),
                    ("SVM",          "default",           False, X_svm, y_svm),
                    ("SVM",          "balanced_resample", True,  X_svm, y_svm),
                    ("MLP",          "default",           False, X_tr,  y_tr),
                    ("MLP",          "weighted_loss",     True,  X_tr,  y_tr),
                ]
                for mname, cfg, balanced, Xf, yf in jobs:
                    if (str(ds), task, mname, cfg, str(fold)) in done:
                        log.info("skip %s/%s %s/%s fold %d", ds, task, mname, cfg, fold)
                        continue
                    t0 = time.time()
                    converged = None
                    if mname == "MLP":
                        y_pred, converged = train_mlp(Xf, yf, X_te, n_classes,
                                                      weighted=balanced)
                    else:
                        Xb, yb = (balanced_resample(Xf, yf) if balanced
                                  else (Xf, yf))
                        if balanced:
                            # cheap proof the arm did something, logged per fit
                            _, cnt = np.unique(yb, return_counts=True)
                            log.info("   balanced fold: %d rows, class counts "
                                     "min=%d max=%d", len(yb), cnt.min(), cnt.max())
                        if mname == "RandomForest":
                            model = _rf()
                            y_pred = _fit_predict(model, Xb, yb, X_te)
                        else:
                            model = _svc(5000)
                            y_pred = _fit_predict(model, Xb, yb, X_te)
                            converged = _converged(model, 5000)

                    m = _metrics(y_te, y_pred)
                    m.update(dataset=ds, task=task, model=mname, config=cfg,
                             fold=fold, n_train=len(Xf), backend=BACKEND,
                             converged=converged,
                             fit_seconds=round(time.time() - t0, 1))
                    _append(OUT_A, m, KEY_A)
                    log.info("[%s/%s] fold %d %s/%-13s F1=%.4f (%.0fs)",
                             ds, task, fold, mname, cfg, m["f1_macro"],
                             m["fit_seconds"])
    log.info("E5-A done -> %s", OUT_A)


# ------------------------------------------------------------- Part B

def _cells():
    return [
        ("kNN",       "k5",                   50_000,  lambda: _knn(),            None),
        ("kNN",       "k5",                  100_000,  lambda: _knn(),            None),
        ("kNN",       "k5",                  200_000,  lambda: _knn(),            None),
        ("SVM",       "rbf_maxiter5000",      50_000,  lambda: _svc(5000),        5000),
        ("SVM",       "rbf_maxiter5000_100k",100_000,  lambda: _svc(5000),        5000),
        ("SVM",       "rbf_maxiter50000",     50_000,  lambda: _svc(50000),      50000),
        ("SVM",       "rbf_maxiter50000_100k",100_000, lambda: _svc(50000),      50000),
        ("LinearSVC", "linear",               50_000,  lambda: _linear_svc(),     5000),
        ("LinearSVC", "linear",              200_000,  lambda: _linear_svc(),     5000),
        ("LinearSVC", "linear",                 None,  lambda: _linear_svc(),     5000),
    ]


def part_b(datasets=None):
    """`datasets` restricts the run to a subset; completed cells are skipped."""
    done = _done(OUT_B, KEY_B)
    for ds in (datasets or ("cicids2017", "unsw_nb15")):
        X, ys = _load(ds)
        y = ys["binary"]
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = _scale(X[tr].copy(), X[te].copy())
            y_tr, y_te = y[tr], y[te]
            for mname, cfg, cap, factory, max_iter in _cells():
                if (mname, cfg) in COSTLY_FOLD1_ONLY and fold != 1:
                    continue
                Xs, ysub = _subsample(X_tr, y_tr, cap)
                n_train = len(Xs)
                if (str(ds), mname, cfg, str(n_train), str(fold)) in done:
                    log.info("skip %s %s/%s n=%d fold %d", ds, mname, cfg,
                             n_train, fold)
                    continue
                t0 = time.time()
                model = factory()
                y_pred = _fit_predict(model, Xs, ysub, X_te)
                m = _metrics(y_te, y_pred)
                m.update(dataset=ds, model=mname, config=cfg, n_train=n_train,
                         fold=fold, backend=BACKEND,
                         converged=_converged(model, max_iter),
                         fit_seconds=round(time.time() - t0, 1))
                _append(OUT_B, m, KEY_B)
                log.info("[%s] fold %d %-9s/%-22s n=%7d F1=%.4f conv=%s (%.0fs)",
                         ds, fold, mname, cfg, n_train, m["f1_macro"],
                         m["converged"], m["fit_seconds"])
    log.info("E5-B done -> %s", OUT_B)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["A", "B"])
    a = p.parse_args()
    log.info("Backend: %s", BACKEND)
    if a.part in (None, "B"):
        part_b()
    if a.part in (None, "A"):
        part_a()
