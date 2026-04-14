"""
Confusion Matrix Generation
Generates normalised confusion matrix heatmaps for the best model
(XGBoost) on each dataset for both binary and multi-class tasks.

Outputs:
  - outputs/figures/fig_cm_{dataset}_{task}.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix

from utils import (
    DATA_CLEAN, DATASETS, FIGURES, MODELS as MODEL_DIR,
    RANDOM_STATE, log,
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.size": 10,
    "font.family": "serif",
})

DATASET_LABELS = {
    "cicids2017": "CICIDS2017",
    "ton_iot": "ToN-IoT",
    "unsw_nb15": "UNSW-NB15",
}

# Models to generate confusion matrices for
TARGET_MODELS = ["XGBoost"]


def generate_confusion_matrices():
    """Generate confusion matrix figures for each dataset/task combo.
    Uses saved best-fold model with MinMax scaling on a held-out
    stratified sample (20%) for fast, unbiased CM generation.
    """
    for ds in DATASETS:
        data_path = DATA_CLEAN / f"{ds}.parquet"
        if not data_path.exists():
            log.info("Skipping %s — no clean parquet", ds)
            continue

        df = pd.read_parquet(data_path)
        feat_cols = [
            c for c in df.columns
            if c not in ("label_binary", "label_multi", "label_original", "dataset")
        ]

        for task in ["binary", "multi"]:
            label_col = f"label_{task}"
            y_all = df[label_col].values
            X_all = df[feat_cols].values

            for model_name in TARGET_MODELS:
                model_path = MODEL_DIR / f"{ds}_{task}_{model_name}.joblib"
                if not model_path.exists():
                    log.warning("Model not found: %s", model_path)
                    continue

                log.info("Generating CM for %s / %s / %s", model_name, ds, task)
                saved = joblib.load(model_path)
                model = saved["model"]

                # Use a stratified 80/20 split — predict on the 20% held-out
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=0.2,
                    stratify=y_all, random_state=RANDOM_STATE,
                )

                # MinMax scale using training stats
                fmin = X_train.min(axis=0)
                fmax = X_train.max(axis=0)
                scale = fmax - fmin
                scale[scale == 0] = 1.0
                X_test_scaled = (X_test - fmin) / scale

                y_pred = model.predict(X_test_scaled)
                all_y_true = y_test
                all_y_pred = y_pred

                # Get class labels
                if task == "binary":
                    class_names = ["Benign", "Attack"]
                else:
                    # Map numeric labels back to original names
                    if "label_original" in df.columns:
                        label_map = {}
                        for orig, multi in zip(
                            df["label_original"], df["label_multi"]
                        ):
                            label_map[multi] = orig
                        classes = sorted(np.unique(all_y_true))
                        class_names = [
                            label_map.get(c, str(c)) for c in classes
                        ]
                    else:
                        classes = sorted(np.unique(all_y_true))
                        class_names = [str(c) for c in classes]

                # Compute normalised confusion matrix (row-normalised)
                cm = confusion_matrix(all_y_true, all_y_pred)
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                cm_norm = np.nan_to_num(cm_norm)  # handle 0/0

                _plot_cm(cm_norm, cm, class_names, ds, task, model_name)


def _plot_cm(
    cm_norm, cm_raw, class_names, dataset, task, model_name
):
    """Plot a confusion matrix heatmap."""
    n_classes = len(class_names)
    ds_label = DATASET_LABELS.get(dataset, dataset)
    task_label = "Binary" if task == "binary" else "Multi-class"

    if n_classes <= 3:
        fig_size = (5.5, 4.5)
        font_size = 11
    elif n_classes <= 10:
        fig_size = (9, 8)
        font_size = 8
    else:
        fig_size = (12, 10)
        font_size = 6.5

    fig, ax = plt.subplots(figsize=fig_size)

    # Annotation: show percentage and raw count for small CMs,
    # just percentage for large ones
    if n_classes <= 5:
        annot = np.array([
            [f"{cm_norm[i, j]:.2f}\n({cm_raw[i, j]:,})"
             for j in range(n_classes)]
            for i in range(n_classes)
        ])
    else:
        annot = np.array([
            [f"{cm_norm[i, j]:.2f}" if cm_norm[i, j] >= 0.005 else ""
             for j in range(n_classes)]
            for i in range(n_classes)
        ])

    sns.heatmap(
        cm_norm, annot=annot, fmt="",
        cmap="Blues", ax=ax,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Recall (row-normalised)", "shrink": 0.75},
        vmin=0, vmax=1,
        annot_kws={"fontsize": font_size},
    )

    ax.set_title(
        f"Confusion Matrix — {ds_label} ({task_label}, {model_name})",
        fontsize=12, pad=14,
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=font_size)
    ax.tick_params(axis="y", rotation=0, labelsize=font_size)
    fig.tight_layout(pad=1.5)

    out = FIGURES / f"fig_cm_{dataset}_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out.name)


if __name__ == "__main__":
    generate_confusion_matrices()
    log.info("Done — confusion matrix figures in %s", FIGURES)
