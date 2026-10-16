"""
Re-render the three multi-class confusion matrices so displayed rows sum to 1.00.

Editor comment 2(i), round 2: the captions say the matrices are row-normalized,
yet several displayed rows do not sum to 1.00. The normalization was always
correct (cm / row sum, plot_confusion_corrected.py). What the reader saw was
the display: each cell rounded independently to two decimals, and cells below
0.005 left blank, so a row such as UNSW-NB15 Shellcode showed values summing to
1.02 and ToN-IoT backdoor to 0.99.

This script changes the display only. Each row is rounded by the largest-
remainder method: every cell is floored to a whole hundredth, and the
hundredths still missing from the row are given to the cells with the largest
remainders. The displayed values then sum to exactly 1.00 in every row, and no
displayed value differs from the true proportion by 0.01 or more. A cell is left
blank only when its displayed value is 0.00. The row's sample count is added to
its label so that small-support rows (e.g. Heartbleed, n = 2) can be read
for what they are.

Models, split and scaling are unchanged from the round-1 figures (same
fitted XGBoost models, same stratified 80/20 split with seed 42, min-max
scaling from the training part), so no number reported in the text changes.
ToN-IoT is included here for the first time in the revision tooling. Its
labels needed no correction, so it uses the submitted model and the clean
dataset, exactly as the original figure did.

Outputs (revision/results_r2):
  figures/fig_cm_{dataset}_multi.pdf
  tables/cm_{dataset}_multi.csv      raw counts, true proportions, displayed values

Usage:  python plot_confusion_r2.py
"""

import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import PROJECT, RANDOM_STATE, REVISION, dataset_path, log

# Round-2 outputs live beside round 1, which is read but never written.
RESULTS_R1 = REVISION / "results_r1"
FIGURES = REVISION / "results_r2" / "figures"
TABLES = REVISION / "results_r2" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import confusion_matrices as cmmod          # noqa: E402  (rcParams, labels)

MODELS = {
    # CICIDS2017: the label fix changed names only; the submitted model stands
    "cicids2017": PROJECT / "outputs" / "models" / "cicids2017_multi_XGBoost.joblib",
    # ToN-IoT: no label correction was needed; the submitted model stands
    "ton_iot": PROJECT / "outputs" / "models" / "ton_iot_multi_XGBoost.joblib",
    # UNSW-NB15: refitted on the 10 merged classes in round 1
    "unsw_nb15": RESULTS_R1 / "models_corrected" / "unsw_nb15_multi_XGBoost.joblib",
}


def largest_remainder(row, total=100):
    """Integer hundredths for one row of proportions, summing exactly to total.

    Ties in the remainder go to the cell with the larger true value, then to
    the lower index, so the result is deterministic.
    """
    scaled = np.asarray(row, dtype=float) * total
    base = np.floor(scaled + 1e-12).astype(int)
    short = total - int(base.sum())
    if short < 0 or short > len(row):
        raise ValueError(f"row does not sum to 1: {row}")
    order = sorted(range(len(row)), key=lambda j: (-(scaled[j] - base[j]), -scaled[j], j))
    for j in order[:short]:
        base[j] += 1
    return base


def _self_test():
    # Shellcode-like row: independent rounding gives 1.02, this must give 1.00.
    r = largest_remainder([0.114, 0.054, 0.024, 0.144, 0.024, 0.64])
    assert r.sum() == 100, r
    assert all(abs(r[j] - v * 100) < 1 for j, v in enumerate(
        [0.114, 0.054, 0.024, 0.144, 0.024, 0.64])), r
    assert largest_remainder([1.0, 0.0]).tolist() == [100, 0]
    assert largest_remainder([1 / 3, 1 / 3, 1 / 3]).sum() == 100


def plot(cm_norm, shown, labels, dataset):
    n = len(labels)
    font = 8 if n <= 10 else 6.5
    fig, ax = plt.subplots(figsize=(9, 8) if n <= 10 else (12, 10))
    annot = np.array([[f"{shown[i, j] / 100:.2f}" if shown[i, j] else ""
                       for j in range(n)] for i in range(n)])
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues", ax=ax,
        xticklabels=[lab.split(" (n=")[0] for lab in labels], yticklabels=labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Recall (row-normalized)", "shrink": 0.75},
        vmin=0, vmax=1, annot_kws={"fontsize": font},
    )
    ax.set_title(f"Confusion Matrix — {cmmod.DATASET_LABELS.get(dataset, dataset)} "
                 f"(Multi-class, XGBoost)", fontsize=12, pad=14)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=font)
    ax.tick_params(axis="y", rotation=0, labelsize=font)
    fig.tight_layout(pad=1.5)
    out = FIGURES / f"fig_cm_{dataset}_multi.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def run(ds):
    saved = joblib.load(MODELS[ds])
    model, feats = saved["model"], saved["features"]

    df = pd.read_parquet(dataset_path(ds))
    y = df["label_multi"].to_numpy()
    X = df[feats].to_numpy(dtype=np.float32)
    names = df.groupby("label_multi")["label_original"].first().to_dict()
    del df

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    fmin, fmax = X_tr.min(axis=0), X_tr.max(axis=0)
    scale = fmax - fmin
    scale[scale == 0] = 1.0
    X_te = (X_te - fmin) / scale

    y_pred = model.predict(X_te)
    classes = sorted(np.unique(y_te))
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    support = cm.sum(axis=1)
    if (support == 0).any():
        raise SystemExit(f"[{ds}] a class has no test samples")
    cm_norm = cm / support[:, None]

    shown = np.vstack([largest_remainder(r) for r in cm_norm])
    bad = [i for i in range(len(classes)) if shown[i].sum() != 100]
    if bad:
        raise SystemExit(f"[{ds}] displayed rows do not sum to 1.00: {bad}")
    worst = np.abs(shown / 100 - cm_norm).max()
    if worst >= 0.01:
        raise SystemExit(f"[{ds}] display error {worst:.4f} >= 0.01")

    labels = [f"{names.get(c, c)} (n={support[i]:,})" for i, c in enumerate(classes)]
    out = plot(cm_norm, shown, labels, ds)

    rows = []
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            rows.append(dict(true=names.get(c, c), pred=names.get(d, d),
                             count=int(cm[i, j]), proportion=cm_norm[i, j],
                             displayed=shown[i, j] / 100))
    pd.DataFrame(rows).to_csv(TABLES / f"cm_{ds}_multi.csv", index=False)

    acc = np.trace(cm) / cm.sum()
    log.info("[%s] %d classes, %d test rows, accuracy %.4f, max display error %.4f -> %s",
             ds, len(classes), cm.sum(), acc, worst, out.name)
    return cm_norm, shown, labels


if __name__ == "__main__":
    _self_test()
    for ds in MODELS:
        run(ds)
