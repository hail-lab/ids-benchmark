"""
Revision E8b — multi-class SHAP attribution and fold-stability of rankings.

Addresses R3.3 (which features drive misclassification of rare attack
classes), R6.12 (SHAP only ran on XGBoost binary; no per-attack-type view, no
stability check) and R1 (SHAP protocol underspecified).

Protocol, run per dataset on the multi-class task:
  1. 5-fold stratified CV with XGBoost (the paper's best tree model).
  2. In each fold, TreeSHAP is computed on a class-stratified sample of the
     VALIDATION fold (SAMPLE_PER_CLASS per class, capped at MAX_SAMPLE), so
     explanations never come from training rows.
  3. Per-class mean |SHAP| is averaged across folds  → heatmap
     (features × attack classes).
  4. Ranking stability: Spearman correlation and top-10 Jaccard between the
     per-fold feature rankings.
  5. Misclassification attribution: for each rare class, mean |SHAP| over
     the samples of that class that the model got WRONG, contrasted with the
     ones it got right.

Outputs
  e8b_shap_perclass_{ds}.csv       features × classes mean |SHAP|
  e8b_shap_stability.csv           per-dataset Spearman / Jaccard
  e8b_shap_misclassified.csv       rare-class correct vs wrong attribution
  fig_shap_multiclass_{ds}.pdf     heatmap

Usage:  python e8b_shap_multi.py [--dataset DS]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from utils import (
    dataset_path,
    DATA_CLEAN, TABLES, FIGURES, RANDOM_STATE, N_JOBS, PAPER_DATASETS,
    safe_feature_cols, log,
)

N_FOLDS = 5
SAMPLE_PER_CLASS = 200   # per class, per fold
MAX_SAMPLE = 2_000       # total cap per fold (matches the paper's SHAP budget)
TOP_K_PLOT = 20


def _xgb_device():
    """GPU training kwargs when a CUDA device is present, else CPU.

    Fitting fifteen-class XGBoost on two million rows is the expensive part of
    this experiment; on Colab's two vCPUs it is impractical, on a T4 it is
    routine.  TreeExplainer itself runs on CPU either way.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {"tree_method": "hist", "device": "cuda"}
    except Exception:
        pass
    return {"tree_method": "hist"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "font.family": "serif",
})


def _stratified_sample(y, rng):
    """Indices: up to SAMPLE_PER_CLASS per class, then capped at MAX_SAMPLE."""
    picks = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = min(SAMPLE_PER_CLASS, len(idx))
        picks.append(rng.choice(idx, n, replace=False))
    picks = np.concatenate(picks)
    if len(picks) > MAX_SAMPLE:
        picks = rng.choice(picks, MAX_SAMPLE, replace=False)
    return np.sort(picks)


def run_dataset(ds):
    df = pd.read_parquet(dataset_path(ds))
    feat_cols = safe_feature_cols(df.columns)
    class_names = (df.groupby("label_multi")["label_original"]
                     .first().sort_index().tolist())
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label_multi"].to_numpy(dtype=np.int32)
    del df
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y, minlength=n_classes)
    log.info("[%s] %d rows × %d features, %d classes", ds, len(X),
             len(feat_cols), n_classes)

    rng = np.random.RandomState(RANDOM_STATE)
    fold_perclass = []   # (n_classes, n_features) per fold
    fold_rankings = []   # global feature ranking per fold
    mis_rows = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr].copy(), X[te].copy()
        y_tr, y_te = y[tr], y[te]
        lo, hi = X_tr.min(0), X_tr.max(0)
        sc = np.where(hi - lo == 0, 1, hi - lo)
        X_tr = (X_tr - lo) / sc
        X_te = (X_te - lo) / sc

        model = xgb.XGBClassifier(
            objective="multi:softprob", n_estimators=400, max_depth=6,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=N_JOBS, **_xgb_device(),
        )
        model.fit(X_tr, y_tr)
        # Explanation runs on CPU; move the booster back so XGBoost does not
        # warn about predicting from GPU-resident trees on host arrays.
        model.set_params(device="cpu")

        sample_idx = _stratified_sample(y_te, rng)
        X_s, y_s = X_te[sample_idx], y_te[sample_idx]
        y_pred_s = model.predict(X_s)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_s)
        # normalise to array (n_samples, n_features, n_classes)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        log.info("[%s] fold %d SHAP on %d samples, shape %s",
                 ds, fold, len(X_s), sv.shape)

        # mean |SHAP| per (class, feature): samples OF that class,
        # attribution FOR that class's output
        pc = np.zeros((n_classes, len(feat_cols)))
        for c in range(n_classes):
            mask = y_s == c
            if mask.sum() == 0:
                continue
            cls_axis = min(c, sv.shape[-1] - 1)
            pc[c] = np.abs(sv[mask, :, cls_axis]).mean(axis=0)
        fold_perclass.append(pc)
        fold_rankings.append(np.abs(sv).mean(axis=(0, 2)))

        # rare-class misclassification attribution
        rare = np.argsort(class_counts)[:5]
        for c in rare:
            mask = y_s == c
            if mask.sum() < 5:
                continue
            cls_axis = min(c, sv.shape[-1] - 1)
            correct = mask & (y_pred_s == c)
            wrong = mask & (y_pred_s != c)
            for label, m in (("correct", correct), ("misclassified", wrong)):
                if m.sum() == 0:
                    continue
                imp = np.abs(sv[m, :, cls_axis]).mean(axis=0)
                top = np.argsort(imp)[::-1][:5]
                mis_rows.append(dict(
                    dataset=ds, fold=fold,
                    attack_class=class_names[c] if c < len(class_names) else str(c),
                    n_class_total=int(class_counts[c]), outcome=label,
                    n_samples=int(m.sum()),
                    top_features="; ".join(
                        f"{feat_cols[i]}={imp[i]:.4f}" for i in top),
                ))

    # ── aggregate across folds ────────────────────────────────────────
    mean_pc = np.mean(fold_perclass, axis=0)
    pc_df = pd.DataFrame(mean_pc.T, index=feat_cols,
                         columns=[class_names[c] if c < len(class_names) else str(c)
                                  for c in range(n_classes)])
    pc_df.to_csv(TABLES / f"e8b_shap_perclass_{ds}.csv")

    # stability across folds
    R = np.array(fold_rankings)
    spear, jac = [], []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            spear.append(spearmanr(R[i], R[j]).statistic)
            a = set(np.argsort(R[i])[::-1][:10])
            b = set(np.argsort(R[j])[::-1][:10])
            jac.append(len(a & b) / len(a | b))
    stab = dict(dataset=ds, mean_spearman=round(float(np.mean(spear)), 3),
                min_spearman=round(float(np.min(spear)), 3),
                mean_top10_jaccard=round(float(np.mean(jac)), 3),
                n_folds=N_FOLDS)
    log.info("[%s] ranking stability: Spearman=%.3f, top-10 Jaccard=%.3f",
             ds, stab["mean_spearman"], stab["mean_top10_jaccard"])

    # ── heatmap ───────────────────────────────────────────────────────
    top_feats = pc_df.mean(axis=1).sort_values(ascending=False).index[:TOP_K_PLOT]
    H = pc_df.loc[top_feats]
    # column-normalise so rare classes remain visible
    Hn = H / H.max(axis=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * H.shape[1] + 3), 7))
    im = ax.imshow(Hn.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels(H.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(H.shape[0]))
    ax.set_yticklabels(H.index, fontsize=7)
    ax.set_title(f"Per-class SHAP attribution — {ds} (XGBoost, multi-class)\n"
                 f"mean |SHAP| over {N_FOLDS} folds, column-normalised",
                 fontsize=10, pad=10)
    fig.colorbar(im, ax=ax, label="normalised mean |SHAP|", shrink=0.8)
    fig.tight_layout()
    out = FIGURES / f"fig_shap_multiclass_{ds}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out.name)

    return stab, mis_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PAPER_DATASETS)
    args = parser.parse_args()

    stabs, mis = [], []
    for ds in ([args.dataset] if args.dataset else PAPER_DATASETS):
        s, m = run_dataset(ds)
        stabs.append(s)
        mis.extend(m)
        pd.DataFrame(stabs).to_csv(TABLES / "e8b_shap_stability.csv", index=False)
        pd.DataFrame(mis).to_csv(TABLES / "e8b_shap_misclassified.csv", index=False)
    log.info("E8b done → %s", TABLES)
