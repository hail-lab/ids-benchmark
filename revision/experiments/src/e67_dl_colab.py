"""
Revision E6 + E7 — deep-learning tuning and FT-Transformer benchmark.

Designed to run on a free Colab T4 GPU (also runs locally on CUDA).  Every
fold/trial result is appended to CSV immediately, so a session disconnect
loses at most one fold.

E6 (R7.1, R1) — hyperparameter search for CNN1D and BiLSTM.
    Random search over N_TRIALS configurations (lr, width, depth, dropout,
    batch size) using a single stratified train/val split; the best config is
    then evaluated with the full 5-fold protocol so it is directly comparable
    with the fixed-hyperparameter numbers in the submitted paper.

E7 (R3.2) — FT-Transformer, a state-of-the-art tabular transformer,
    benchmarked with the same 5-fold protocol, class-weighted loss and
    preprocessing as every other model.  Implemented inline (no rtdl
    dependency): feature tokeniser + transformer encoder + CLS head.

Usage
-----
    python e67_dl_colab.py --exp e6 --dataset ton_iot --model CNN1D
    python e67_dl_colab.py --exp e7 --dataset ton_iot --task binary
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
)

from utils import (
    dataset_path,
    DATA_CLEAN, TABLES, RANDOM_STATE, PAPER_DATASETS, safe_feature_cols, log,
)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Identical to evaluation.compute_metrics, inlined so this module runs on
    Colab without pulling in matplotlib/seaborn/shap via evaluation.py."""
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    roc = pr = np.nan
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
                pr = average_precision_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                    average="macro")
        except ValueError:
            pass
    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bacc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "roc_auc": round(roc, 4) if not np.isnan(roc) else np.nan,
        "pr_auc": round(pr, 4) if not np.isnan(pr) else np.nan,
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5

# Epoch budget.  Measured on UNSW-NB15 binary, fold 1, against a 50-epoch
# reference (macro F1 = 0.9754, 39.2 min/fold):
#     50 epochs  F1 = 0.9754   39.2 min
#     20 epochs  F1 = 0.9740   15.5 min   (-0.0014)
#     10 epochs  F1 = 0.9737    7.6 min   (-0.0017)
# Validation loss is essentially flat after the first few epochs, so the extra
# epochs buy 0.0014 F1 for 2.5x the compute.  Twenty is used because it keeps
# the full 5-fold protocol on full data affordable on a free Colab GPU, which
# matters more here than the fourth decimal place: the alternatives for saving
# the same time - subsampling the training data or dropping folds - would each
# weaken the comparison this experiment exists to make, whereas trimming an
# epoch budget that has already converged does not.  `epochs` is recorded for
# every fold so any fit that stops early is visible.
MAX_EPOCHS = 20
PATIENCE = 5

N_TRIALS = 20
# Cap rows for the tuning search only (full 5-fold uses everything)
SEARCH_ROWS = 400_000

OUT_E6 = TABLES / "e6_dl_tuning.csv"
OUT_E6_FOLDS = TABLES / "e6_dl_tuned_folds.csv"
OUT_E7 = TABLES / "e7_ft_transformer.csv"


# ── Architectures ─────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, n_feat, n_cls, width=64, depth=2, dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for i in range(depth):
            c_out = width * (2 ** i)
            layers += [nn.Conv1d(c_in, c_out, 3, padding=1),
                       nn.BatchNorm1d(c_out), nn.ReLU()]
            c_in = c_out
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(c_in, 64), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))


class BiLSTM(nn.Module):
    def __init__(self, n_feat, n_cls, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(64, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class FTTransformer(nn.Module):
    """Feature tokeniser + transformer encoder + CLS head (Gorishniy et al., 2021)."""

    def __init__(self, n_feat, n_cls, d_token=64, n_blocks=3, n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_feat, d_token))
        self.bias = nn.Parameter(torch.empty(n_feat, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(),
                                  nn.Linear(d_token, n_cls))

    def forward(self, x):
        # x: (B, F) numeric → tokens (B, F, d)
        tok = x.unsqueeze(-1) * self.weight + self.bias
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        return self.head(self.encoder(tok)[:, 0])


def build(name, n_feat, n_cls, hp):
    if name == "CNN1D":
        return CNN1D(n_feat, n_cls, hp["width"], hp["depth"], hp["dropout"])
    if name == "BiLSTM":
        return BiLSTM(n_feat, n_cls, hp["hidden"], hp["layers"], hp["dropout"])
    return FTTransformer(n_feat, n_cls, hp["d_token"], hp["n_blocks"],
                         hp["n_heads"], hp["dropout"])


# ── Training loop (shared) ────────────────────────────────────────────

def train_eval(name, hp, X_tr, y_tr, X_va, y_va, n_cls, max_epochs=MAX_EPOCHS):
    torch.manual_seed(RANDOM_STATE)
    model = build(name, X_tr.shape[1], n_cls, hp).to(DEVICE)

    counts = np.maximum(np.bincount(y_tr, minlength=n_cls), 1).astype(np.float32)
    w = (1.0 / counts) * counts.sum() / n_cls
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp.get("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=3, min_lr=1e-6)
    # Throughput notes.  The first version of this loop streamed CPU tensors
    # through a DataLoader in fp32, which made one FT-Transformer fold on
    # UNSW-NB15 take 86 minutes and put the full experiment far outside a free
    # Colab GPU budget.  Two changes address that without altering the model:
    #   * the fold is moved to the device once and batched by slicing a
    #     shuffled index, so no host-to-device copy happens per step;
    #   * training runs under automatic mixed precision, which is where most of
    #     the speedup comes from for an attention model on a T4.
    # Evaluation stays in fp32 so reported metrics are unaffected.
    bs = int(hp["batch"])
    try:
        Xt = torch.as_tensor(X_tr).to(DEVICE)
        yt = torch.as_tensor(y_tr, dtype=torch.long).to(DEVICE)
        Xv = torch.as_tensor(X_va).to(DEVICE)
        yv = torch.as_tensor(y_va, dtype=torch.long).to(DEVICE)
        on_device = True
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        Xt = torch.as_tensor(X_tr)
        yt = torch.as_tensor(y_tr, dtype=torch.long)
        Xv = torch.as_tensor(X_va)
        yv = torch.as_tensor(y_va, dtype=torch.long)
        on_device = False
        log.warning("fold did not fit on the GPU; batching from host memory")

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_train = len(Xt)

    def _batch(t, idx):
        out = t[idx]
        return out if on_device else out.to(DEVICE, non_blocking=True)

    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train, device=Xt.device)
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb = _batch(Xt, idx), _batch(yt, idx)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), 16_384):
                sl = slice(i, i + 16_384)
                xb = Xv[sl] if on_device else Xv[sl].to(DEVICE)
                yb = yv[sl] if on_device else yv[sl].to(DEVICE)
                tot += criterion(model(xb), yb).item() * len(yb)
        vl = tot / len(Xv)
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xv), 16_384):
            xb = Xv[i:i + 16_384]
            if not on_device:
                xb = xb.to(DEVICE)
            probs.append(torch.softmax(model(xb), 1).float().cpu())
    probs = torch.cat(probs).numpy()

    del Xt, yt, Xv, yv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return compute_metrics(y_va, probs.argmax(1), probs, n_cls), epoch + 1


# ── Data ──────────────────────────────────────────────────────────────

def load(ds, task, max_rows=None):
    df = pd.read_parquet(dataset_path(ds))
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE)
    feats = safe_feature_cols(df.columns)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df[f"label_{task}"].to_numpy(dtype=np.int64)
    return X, y, feats


def scale(X_tr, X_va):
    lo, hi = X_tr.min(0), X_tr.max(0)
    sc = np.where(hi - lo == 0, 1, hi - lo).astype(np.float32)
    return ((X_tr - lo) / sc).astype(np.float32), ((X_va - lo) / sc).astype(np.float32)


def _append(path, row, key=None):
    """Append a result row; with `key`, a repeat overwrites rather than duplicates."""
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        if key:
            df = df.drop_duplicates(subset=key, keep="last")
    df.to_csv(path, index=False)


def _done(path, key):
    """Set of already-computed keys, so an interrupted run can resume.

    Sessions here are long (E7 on CICIDS2017 runs for hours) and free Colab
    disconnects, so every fold and every search trial is checked against what
    is already on disk before it is recomputed.
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if not set(key).issubset(df.columns):
        return set()
    return set(map(tuple, df[key].astype(str).values))


# ── E6: random search + tuned 5-fold ──────────────────────────────────

def sample_hp(name, rng):
    hp = dict(lr=float(10 ** rng.uniform(-4, -2.3)),
              batch=int(rng.choice([256, 512, 1024, 2048])),
              dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.5])),
              weight_decay=float(10 ** rng.uniform(-6, -3)))
    if name == "CNN1D":
        hp.update(width=int(rng.choice([32, 64, 128])),
                  depth=int(rng.choice([2, 3, 4])))
    else:
        hp.update(hidden=int(rng.choice([32, 64, 128])),
                  layers=int(rng.choice([1, 2, 3])))
    return hp


KEY_E6 = ["dataset", "task", "model", "trial"]
KEY_E6_FOLDS = ["dataset", "task", "model", "config", "fold"]


def run_e6(ds, task, name):
    # --- search phase (resumable trial by trial) ---
    done_trials = _done(OUT_E6, KEY_E6)
    rng = np.random.RandomState(RANDOM_STATE)
    # Draw every configuration up front so the sampling sequence is identical
    # whether or not the run was interrupted; only unfinished trials are fitted.
    plan = [sample_hp(name, rng) for _ in range(N_TRIALS)]
    todo = [t for t in range(N_TRIALS)
            if (str(ds), task, name, str(t)) not in done_trials]

    if todo:
        log.info("[E6 %s/%s/%s] %d of %d trials to run", ds, task, name,
                 len(todo), N_TRIALS)
        X, y, _ = load(ds, task, max_rows=SEARCH_ROWS)
        n_cls = len(np.unique(y))
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
        X_tr, X_va = scale(X_tr, X_va)
        for trial in todo:
            hp = plan[trial]
            t0 = time.time()
            try:
                m, ep = train_eval(name, hp, X_tr, y_tr, X_va, y_va, n_cls,
                                   max_epochs=25)
            except RuntimeError as exc:      # OOM on a large config
                log.error("trial %d failed: %s", trial, exc)
                torch.cuda.empty_cache()
                continue
            _append(OUT_E6, dict(dataset=ds, task=task, model=name, trial=trial,
                                 **hp, f1_macro=m["f1_macro"],
                                 balanced_accuracy=m["balanced_accuracy"],
                                 epochs=ep, seconds=round(time.time() - t0, 1)),
                    key=KEY_E6)
            log.info("[E6 %s/%s/%s] trial %2d/%d F1=%.4f %s", ds, task, name,
                     trial, N_TRIALS, m["f1_macro"], hp)
        del X, y, X_tr, X_va
    else:
        log.info("[E6 %s/%s/%s] search already complete", ds, task, name)

    # Best configuration is read back from disk, so it is the best across all
    # trials ever run rather than only those fitted in this session.
    trials = pd.read_csv(OUT_E6)
    mine = trials[(trials.dataset == ds) & (trials.task == task)
                  & (trials.model == name)]
    if mine.empty:
        log.error("[E6] no completed trials for %s/%s/%s", ds, task, name)
        return
    best_row = mine.loc[mine.f1_macro.idxmax()]
    hp_cols = [c for c in mine.columns if c in
               ("lr", "batch", "dropout", "weight_decay", "width", "depth",
                "hidden", "layers")]
    best_hp = {c: best_row[c] for c in hp_cols if pd.notna(best_row[c])}
    for k in ("batch", "width", "depth", "hidden", "layers"):
        if k in best_hp:
            best_hp[k] = int(best_hp[k])
    best_f1 = float(best_row.f1_macro)
    log.info("[E6] best %s on %s/%s: F1=%.4f %s", name, ds, task, best_f1,
             best_hp)

    # --- full 5-fold with the tuned configuration (also resumable) ---
    done_folds = _done(OUT_E6_FOLDS, KEY_E6_FOLDS)
    todo_folds = [f for f in range(1, N_FOLDS + 1)
                  if (str(ds), task, name, "tuned", str(f)) not in done_folds]
    if not todo_folds:
        log.info("[E6 tuned] %s/%s/%s already complete", ds, task, name)
        return
    log.info("[E6 tuned] %s/%s/%s folds to run: %s", ds, task, name, todo_folds)

    X, y, _ = load(ds, task)
    n_cls = len(np.unique(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        if fold not in todo_folds:
            continue
        Xa, Xb = scale(X[tr].copy(), X[va].copy())
        t0 = time.time()
        m, ep = train_eval(name, best_hp, Xa, y[tr], Xb, y[va], n_cls)
        _append(OUT_E6_FOLDS, dict(dataset=ds, task=task, model=name,
                                   fold=fold, config="tuned",
                                   hp=json.dumps(best_hp), epochs=ep,
                                   fit_seconds=round(time.time() - t0, 1), **m),
                key=KEY_E6_FOLDS)
        log.info("[E6 tuned] %s/%s/%s fold %d/%d F1=%.4f", ds, task, name,
                 fold, N_FOLDS, m["f1_macro"])


# ── E7: FT-Transformer 5-fold ─────────────────────────────────────────

FT_HP = dict(lr=1e-3, batch=1024, dropout=0.1, weight_decay=1e-5,
             d_token=64, n_blocks=3, n_heads=8)


KEY_E7 = ["dataset", "task", "model", "fold"]


def run_e7(ds, task):
    done = _done(OUT_E7, KEY_E7)
    todo = [f for f in range(1, N_FOLDS + 1)
            if (str(ds), task, "FTTransformer", str(f)) not in done]
    if not todo:
        log.info("[E7] %s/%s already complete - nothing to do", ds, task)
        return
    log.info("[E7] %s/%s folds to run: %s", ds, task, todo)

    X, y, _ = load(ds, task)
    n_cls = len(np.unique(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        if fold not in todo:
            continue
        Xa, Xb = scale(X[tr].copy(), X[va].copy())
        t0 = time.time()
        m, ep = train_eval("FTTransformer", FT_HP, Xa, y[tr], Xb, y[va], n_cls)
        _append(OUT_E7, dict(dataset=ds, task=task, model="FTTransformer",
                             fold=fold, epochs=ep,
                             fit_seconds=round(time.time() - t0, 1), **m),
                key=KEY_E7)
        log.info("[E7] %s/%s fold %d/%d F1=%.4f (%.0fs)", ds, task, fold,
                 N_FOLDS, m["f1_macro"], time.time() - t0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["e6", "e7"], required=True)
    p.add_argument("--dataset", choices=PAPER_DATASETS, required=True)
    p.add_argument("--task", choices=["binary", "multi"], default="binary")
    p.add_argument("--model", choices=["CNN1D", "BiLSTM"], default="CNN1D")
    a = p.parse_args()
    log.info("Device: %s", DEVICE)
    if a.exp == "e6":
        run_e6(a.dataset, a.task, a.model)
    else:
        run_e7(a.dataset, a.task)
