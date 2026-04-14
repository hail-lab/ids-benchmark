"""
Post-training pipeline: generates all numbers needed for the paper.
Run AFTER all model training runs are complete.

Outputs:
  1. per_class_f1.csv          — per-class F1 for XGBoost multi-class
  2. degradation_table.csv     — binary-to-multi F1 degradation
  3. efficiency_table.csv      — training times per model/dataset/task
  4. wilcoxon_results.csv      — Wilcoxon signed-rank tests
  5. LaTeX table snippets       — printed to stdout
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from utils import (
    DATA_CLEAN, DATASETS, MODEL_NAMES, MODELS as MODEL_DIR,
    TABLES, RANDOM_STATE, log,
)

# ── 1. Per-class F1 for XGBoost multi-class ──────────────────────────
def per_class_f1():
    """Compute per-class F1 for XGBoost multi-class on both datasets."""
    results = []
    for ds in DATASETS:
        path = DATA_CLEAN / f"{ds}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        feat_cols = [c for c in df.columns if c not in
                     ("label_binary", "label_multi", "label_original", "dataset")]
        X = df[feat_cols].values
        y = df["label_multi"].values

        # Load saved XGBoost model
        model_path = MODEL_DIR / f"{ds}_multi_XGBoost.joblib"
        if not model_path.exists():
            log.warning("Model not found: %s", model_path)
            continue

        saved = joblib.load(model_path)
        model = saved["model"]

        # Get class names from label_original
        if "label_original" in df.columns:
            label_map = dict(zip(df["label_multi"], df["label_original"]))
        else:
            label_map = {i: str(i) for i in np.unique(y)}

        # 5-fold CV to get averaged per-class F1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_reports = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            fold_reports.append(report)

        # Average per-class F1 across folds
        classes = sorted(np.unique(y))
        for c in classes:
            f1s = [r.get(str(c), {}).get("f1-score", 0) for r in fold_reports]
            count = (y == c).sum()
            name = label_map.get(c, str(c))
            results.append({
                "dataset": ds,
                "class_id": c,
                "class_name": name,
                "samples": count,
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            })

    out = pd.DataFrame(results)
    out.to_csv(TABLES / "per_class_f1.csv", index=False)
    print("\n=== Per-Class F1 (XGBoost Multi-Class) ===")
    for ds in DATASETS:
        sub = out[out["dataset"] == ds]
        if sub.empty:
            continue
        print(f"\n{ds}:")
        for _, row in sub.iterrows():
            print(f"  {row['class_name']:25s}  {row['samples']:>8d}  F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}")
    return out


# ── 2. Binary-to-Multi Degradation ──────────────────────────────────
def degradation_table():
    """Compute relative F1 degradation from binary to multi-class."""
    csv = TABLES / "benchmark_results.csv"
    df = pd.read_csv(csv)

    results = []
    for m in MODEL_NAMES:
        for ds in DATASETS:
            bin_row = df[(df["model"]==m) & (df["dataset"]==ds) & (df["task"]=="binary")]
            mul_row = df[(df["model"]==m) & (df["dataset"]==ds) & (df["task"]=="multi")]
            if bin_row.empty or mul_row.empty:
                continue
            f1_bin = bin_row.iloc[0]["f1_macro"]
            f1_mul = mul_row.iloc[0]["f1_macro"]
            delta = (f1_bin - f1_mul) / f1_bin * 100 if f1_bin > 0 else 0
            results.append({
                "model": m, "dataset": ds,
                "f1_binary": f1_bin, "f1_multi": f1_mul,
                "degradation_pct": delta,
            })

    out = pd.DataFrame(results)
    out.to_csv(TABLES / "degradation_table.csv", index=False)
    print("\n=== Binary-to-Multi Degradation (%) ===")
    pivot = out.pivot_table(index="model", columns="dataset", values="degradation_pct")
    print(pivot.to_string(float_format="%.1f"))
    return out


# ── 3. Efficiency Table ──────────────────────────────────────────────
def efficiency_table():
    """Print training times table."""
    csv = TABLES / "benchmark_results.csv"
    df = pd.read_csv(csv)

    print("\n=== Training Time per Fold (seconds) ===")
    for ds in DATASETS:
        print(f"\n{ds}:")
        for task in ["binary", "multi"]:
            sub = df[(df["dataset"]==ds) & (df["task"]==task)]
            for _, row in sub.iterrows():
                print(f"  {row['model']:12s} {task:6s}  {row['fit_seconds']:>10.1f}s")


# ── 4. Wilcoxon Signed-Rank Tests ───────────────────────────────────
def wilcoxon_tests():
    """Compute Wilcoxon signed-rank test for best vs each alternative."""
    csv = TABLES / "benchmark_results.csv"
    df = pd.read_csv(csv)

    results = []
    for task in ["binary", "multi"]:
        sub = df[df["task"] == task]

        # Find best model (highest mean F1 across datasets)
        mean_f1 = sub.groupby("model")["f1_macro"].mean()
        best = mean_f1.idxmax()

        for other in MODEL_NAMES:
            if other == best:
                continue
            # Collect paired observations (one per dataset)
            pairs_best = []
            pairs_other = []
            for ds in DATASETS:
                b = sub[(sub["model"]==best) & (sub["dataset"]==ds)]
                o = sub[(sub["model"]==other) & (sub["dataset"]==ds)]
                if b.empty or o.empty:
                    continue
                pairs_best.append(b.iloc[0]["f1_macro"])
                pairs_other.append(o.iloc[0]["f1_macro"])

            if len(pairs_best) < 2:
                W, p = 0.0, 1.0
            else:
                try:
                    W, p = wilcoxon(pairs_best, pairs_other)
                except Exception:
                    W, p = 0.0, 1.0

            sig = "Yes" if p < 0.05 else "No"
            results.append({
                "task": task, "best": best, "vs": other,
                "W": W, "p_value": p, "significant": sig,
            })

    out = pd.DataFrame(results)
    out.to_csv(TABLES / "wilcoxon_results.csv", index=False)
    print("\n=== Wilcoxon Signed-Rank Tests ===")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    print("=" * 80)
    print("POST-TRAINING PIPELINE")
    print("=" * 80)

    csv = TABLES / "benchmark_results.csv"
    df = pd.read_csv(csv)
    print(f"\nLoaded {len(df)} results from {csv}")
    expected = len(MODEL_NAMES) * len(DATASETS) * 2
    if len(df) < expected:
        print(f"WARNING: Only {len(df)}/{expected} results found! Some may be missing.")
        print("Missing combinations:")
        existing = set(zip(df["model"], df["dataset"], df["task"]))
        for m in MODEL_NAMES:
            for ds in DATASETS:
                for t in ["binary", "multi"]:
                    if (m, ds, t) not in existing:
                        print(f"  {m}/{ds}/{t}")

    degradation_table()
    efficiency_table()
    wilcoxon_tests()

    if "--perclass" in sys.argv:
        per_class_f1()

    print("\n\nDone! Check outputs/tables/ for CSV files.")
