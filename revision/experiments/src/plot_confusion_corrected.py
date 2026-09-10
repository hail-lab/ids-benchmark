"""
Regenerate the multi-class confusion matrices on the label-corrected data.

The submitted figures predate both label fixes, and it shows:

  * UNSW-NB15 draws an 11x11 matrix with separate ``Backdoor`` and ``Backdoors``
    rows -- the very defect the revision corrects -- under a caption that says
    10 classes.
  * CICIDS2017's web-attack class names come from the mis-decoded labels.

CICIDS2017's correction changed label *names* only, so its trained model and
its numbers are unaffected and the existing artifact is reused; only the tick
labels change. UNSW-NB15 needs the model refitted on the merged labels, which
recompute_unsw_multi.py has already produced in results_r1/models_corrected/.

Plotting is delegated to the submitted pipeline's ``_plot_cm`` so the figures
remain visually identical to the ones they replace.

Usage:  python plot_confusion_corrected.py
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import REVISION, FIGURES, RANDOM_STATE, dataset_path, log

SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import confusion_matrices as cmmod          # noqa: E402

# where to find a model trained on the corrected labels for each dataset
MODELS = {
    # CICIDS2017: names changed, mapping did not -- the submitted model stands
    "cicids2017": REVISION.parent / "outputs" / "models"
                  / "cicids2017_multi_XGBoost.joblib",
    # UNSW-NB15: refitted on 10 merged classes
    "unsw_nb15": REVISION / "results_r1" / "models_corrected"
                 / "unsw_nb15_multi_XGBoost.joblib",
}


def run(ds: str):
    mpath = MODELS[ds]
    if not mpath.exists():
        log.error("[%s] no model at %s -- skipped", ds, mpath)
        return False

    saved = joblib.load(mpath)
    model, feats = saved["model"], saved["features"]

    df = pd.read_parquet(dataset_path(ds))
    y = df["label_multi"].to_numpy()
    X = df[feats].to_numpy(dtype=np.float32)
    names_map = df.groupby("label_multi")["label_original"].first().to_dict()
    del df

    log.info("[%s] %d features, %d classes", ds, len(feats), len(np.unique(y)))

    # same held-out 20% and same scaling as the submitted figures
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    fmin, fmax = X_tr.min(axis=0), X_tr.max(axis=0)
    scale = fmax - fmin
    scale[scale == 0] = 1.0
    X_te = (X_te - fmin) / scale

    y_pred = model.predict(X_te)
    classes = sorted(np.unique(y_te))
    class_names = [str(names_map.get(c, c)) for c in classes]

    cm = confusion_matrix(y_te, y_pred, labels=classes)
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    cmmod.FIGURES = FIGURES          # write beside the other revision figures
    cmmod._plot_cm(cm_norm, cm, class_names, ds, "multi", "XGBoost")
    print(f"{ds}: {len(classes)} classes -> {class_names}")
    return True


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    ok = [run(ds) for ds in MODELS]
    if not all(ok):
        raise SystemExit("one or more figures were not regenerated")
