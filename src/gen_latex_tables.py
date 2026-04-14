"""Generate LaTeX table code from benchmark_results.csv for the paper."""
import pandas as pd
from utils import TABLES, DATASETS

MODEL_ORDER = ["XGBoost", "RandomForest", "LightGBM", "SVM", "kNN", "MLP", "CNN1D", "BiLSTM"]

DATASET_LABELS = {
    "cicids2017": "CICIDS2017",
    "cicids2018": "CICIDS2018",
    "ton_iot": "ToN-IoT",
    "unsw_nb15": "UNSW-NB15",
}

def fmt(val, std, decimals=4):
    """Format value±std for LaTeX."""
    return f"{val:.{decimals}f}$\\pm${std:.{decimals}f}"

def fmt_time(t):
    """Format time with comma thousands."""
    if t >= 1000:
        return f"{t:,.1f}"
    return f"{t:.1f}"

def gen_table(df, task):
    """Generate LaTeX table rows for a task (binary or multi)."""
    sub = df[df["task"] == task].copy()
    active_ds = [d for d in DATASETS if d in sub["dataset"].values]

    # Find best F1 per dataset
    best_f1 = {d: sub[sub["dataset"] == d]["f1_macro"].max() for d in active_ds}

    rows = []
    for m in MODEL_ORDER:
        parts = []
        skip = False
        for d in active_ds:
            r = sub[(sub["model"] == m) & (sub["dataset"] == d)]
            if r.empty:
                skip = True
                break
            r = r.iloc[0]
            f1_str = fmt(r["f1_macro"], r["f1_macro_std"])
            if abs(r["f1_macro"] - best_f1[d]) < 1e-6:
                f1_str = "\\textbf{" + f1_str + "}"
            parts.append(f"{f1_str} & {r['accuracy']:.4f} & {fmt_time(r['fit_seconds'])}")
        if skip:
            continue
        name = m.replace("_", "\\_")
        row = f"{name:12s} & " + "    & ".join(parts) + "  \\\\"
        rows.append(row)

    return "\n".join(rows)

if __name__ == "__main__":
    csv_path = TABLES / "benchmark_results.csv"
    df = pd.read_csv(csv_path)
    active_ds = [d for d in DATASETS if d in df["dataset"].values]
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Active datasets: {active_ds}\n")

    print("=" * 80)
    print("BINARY TABLE")
    print("=" * 80)
    print(gen_table(df, "binary"))

    print("\n" + "=" * 80)
    print("MULTI-CLASS TABLE")
    print("=" * 80)
    print(gen_table(df, "multi"))

    # SOTA "This work" rows
    print("\n" + "=" * 80)
    print("SOTA TABLE — 'This work' rows")
    print("=" * 80)
    for ds in active_ds:
        best_bin = df[(df["dataset"]==ds) & (df["task"]=="binary")]
        best_mul = df[(df["dataset"]==ds) & (df["task"]=="multi")]
        if best_bin.empty or best_mul.empty:
            print(f"% {ds} — missing binary or multi results")
            continue
        best_bin = best_bin.loc[best_bin["f1_macro"].idxmax()]
        best_mul = best_mul.loc[best_mul["f1_macro"].idxmax()]
        ds_label = DATASET_LABELS.get(ds, ds)
        print(f"\\textbf{{This work}} & {ds_label} & {best_bin['model']} & {best_bin['f1_macro']:.3f} & {best_mul['f1_macro']:.3f} & \\textbf{{Yes}} & 5-fold CV \\\\")

    # Degradation table
    print("\n" + "=" * 80)
    print("DEGRADATION TABLE")
    print("=" * 80)
    for m in MODEL_ORDER:
        vals = []
        for ds in active_ds:
            b = df[(df["model"]==m) & (df["dataset"]==ds) & (df["task"]=="binary")]
            mu = df[(df["model"]==m) & (df["dataset"]==ds) & (df["task"]=="multi")]
            if b.empty or mu.empty:
                vals.append("---")
            else:
                f1b = b.iloc[0]["f1_macro"]
                f1m = mu.iloc[0]["f1_macro"]
                delta = (f1b - f1m) / f1b * 100 if f1b > 0 else 0
                vals.append(f"{delta:.1f}")
        print(f"{m:12s} & " + " & ".join(vals) + " \\\\")

    # Efficiency table
    print("\n" + "=" * 80)
    print("EFFICIENCY TABLE")
    print("=" * 80)
    for m in MODEL_ORDER:
        vals = []
        for ds in active_ds:
            for task in ["binary", "multi"]:
                row = df[(df["model"]==m) & (df["dataset"]==ds) & (df["task"]==task)]
                if row.empty:
                    vals.append("---")
                else:
                    vals.append(fmt_time(row.iloc[0]["fit_seconds"]))
        print(f"{m:12s} & " + " & ".join(vals) + " \\\\")
