"""
Leakage Before/After Comparison
Compares benchmark results with and without data leakage to demonstrate
the impact of proper preprocessing.

Outputs:
  - outputs/tables/leakage_comparison.csv
  - outputs/figures/fig_leakage_comparison.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import TABLES, FIGURES, MODEL_NAMES, DATASETS, log

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

DATASET_LABELS = {
    "cicids2017": "CICIDS2017",
    "ton_iot": "ToN-IoT",
    "unsw_nb15": "UNSW-NB15",
}


def build_comparison():
    """Build a merged comparison table of leaky vs clean results."""
    clean_path = TABLES / "benchmark_results.csv"
    leaky_path = TABLES / "benchmark_results_backup_leaky.csv"

    if not leaky_path.exists():
        log.error("Leaky backup not found: %s", leaky_path)
        return None

    clean = pd.read_csv(clean_path)
    leaky = pd.read_csv(leaky_path)

    # Merge on model/dataset/task
    merged = clean.merge(
        leaky[["model", "dataset", "task", "f1_macro", "accuracy"]],
        on=["model", "dataset", "task"],
        suffixes=("_clean", "_leaky"),
        how="inner",
    )

    merged["f1_inflation"] = merged["f1_macro_leaky"] - merged["f1_macro_clean"]
    merged["f1_inflation_pct"] = (
        merged["f1_inflation"] / merged["f1_macro_clean"] * 100
    )
    merged["accuracy_inflation"] = merged["accuracy_leaky"] - merged["accuracy_clean"]

    # Sort for readability
    merged = merged.sort_values(["task", "dataset", "model"]).reset_index(drop=True)
    out_path = TABLES / "leakage_comparison.csv"
    merged.to_csv(out_path, index=False)
    log.info("Saved leakage comparison → %s (%d rows)", out_path, len(merged))
    return merged


def fig_leakage_comparison(merged: pd.DataFrame):
    """Grouped bar chart: leaky vs clean F1 side by side."""
    for task in ["binary", "multi"]:
        sub = merged[merged["task"] == task].copy()
        if sub.empty:
            continue

        datasets_present = [d for d in DATASETS if d in sub["dataset"].values]
        models_present = [m for m in MODEL_NAMES if m in sub["model"].values]

        n_models = len(models_present)
        n_datasets = len(datasets_present)

        # Create one figure per dataset for clarity
        for ds in datasets_present:
            ds_sub = sub[sub["dataset"] == ds]
            if ds_sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(max(9, n_models * 1.2), 5.5))
            x = np.arange(n_models)
            width = 0.32

            vals_leaky = []
            vals_clean = []
            for m in models_present:
                row = ds_sub[ds_sub["model"] == m]
                vals_leaky.append(row["f1_macro_leaky"].values[0] if len(row) else 0)
                vals_clean.append(row["f1_macro_clean"].values[0] if len(row) else 0)

            bars_leaky = ax.bar(
                x - width / 2, vals_leaky, width,
                label="With leakage",
                color="#D55E00", edgecolor="white", linewidth=0.5, alpha=0.85,
            )
            bars_clean = ax.bar(
                x + width / 2, vals_clean, width,
                label="Without leakage (proper)",
                color="#0072B2", edgecolor="white", linewidth=0.5, alpha=0.85,
            )

            # Annotate inflation above bars
            for i, (vl, vc) in enumerate(zip(vals_leaky, vals_clean)):
                diff = vl - vc
                if diff > 0.001:
                    ax.annotate(
                        f"+{diff:.3f}",
                        xy=(x[i], max(vl, vc)),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color="#D55E00",
                        fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(models_present, fontsize=9, rotation=25, ha="right")
            ax.set_ylabel("F1-macro", fontsize=11)
            ds_label = DATASET_LABELS.get(ds, ds)
            task_label = "Binary" if task == "binary" else "Multi-class"
            ax.set_title(
                f"Data Leakage Impact — {ds_label} ({task_label})",
                fontsize=12, pad=12,
            )
            ax.set_ylim(0, min(1.12, max(vals_leaky + vals_clean) * 1.10))
            ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", visible=False)
            fig.tight_layout(pad=1.5)
            out = FIGURES / f"fig_leakage_{ds}_{task}.pdf"
            fig.savefig(out)
            plt.close(fig)
            log.info("Saved %s", out.name)


def print_latex_table(merged: pd.DataFrame):
    """Print a LaTeX table suitable for inclusion in the paper."""
    print("\n%% === Leakage Comparison LaTeX Table ===")
    for task in ["binary", "multi"]:
        sub = merged[merged["task"] == task]
        if sub.empty:
            continue
        task_label = "Binary" if task == "binary" else "Multi-class"
        print(f"\n% Task: {task_label}")
        for _, row in sub.iterrows():
            ds_label = DATASET_LABELS.get(row["dataset"], row["dataset"])
            model = row["model"]
            f1_leaky = row["f1_macro_leaky"]
            f1_clean = row["f1_macro_clean"]
            diff = row["f1_inflation"]
            pct = row["f1_inflation_pct"]
            print(
                f"{model:12s} & {ds_label:10s} & {f1_leaky:.4f} & "
                f"{f1_clean:.4f} & {diff:+.4f} & {pct:+.1f}\\% \\\\"
            )


if __name__ == "__main__":
    merged = build_comparison()
    if merged is not None:
        fig_leakage_comparison(merged)
        print_latex_table(merged)
        # Summary statistics
        print("\n=== Leakage Impact Summary ===")
        for task in ["binary", "multi"]:
            sub = merged[merged["task"] == task]
            if sub.empty:
                continue
            print(f"\n{task.upper()}:")
            print(f"  Mean F1 inflation:   {sub['f1_inflation'].mean():.4f}")
            print(f"  Max  F1 inflation:   {sub['f1_inflation'].max():.4f} "
                  f"({sub.loc[sub['f1_inflation'].idxmax(), 'model']}/"
                  f"{sub.loc[sub['f1_inflation'].idxmax(), 'dataset']})")
            print(f"  Mean %% inflation:    {sub['f1_inflation_pct'].mean():.1f}%")
        log.info("Done.")
