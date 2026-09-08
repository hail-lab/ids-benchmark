"""
01_IDS_Benchmark — Evaluation, SHAP Explainability & Figure Generation
Computes metrics, SHAP beeswarm/force plots, and paper figures.

All figures are designed for IEEE Access column width with:
- No overlapping text or labels
- Numbered markers with separate legend (scatter plots)
- Adequate margins — nothing overflows borders
- Colour-blind-friendly palette

Usage
-----
    python evaluation.py                  # generate all figures from saved results
    python evaluation.py --shap           # also run SHAP analysis (slow)
"""

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score,
)

from utils import (
    DATA_CLEAN, DATASETS, MODEL_NAMES, FIGURES, TABLES,
    MODELS as MODEL_DIR, RANDOM_STATE, log,
)

# ── Global plot style ─────────────────────────────────────────────────
DATASET_LABELS = {
    "cicids2017": "CICIDS2017",
    "cicids2018": "CICIDS2018",
    "ton_iot": "ToN-IoT",
    "unsw_nb15": "UNSW-NB15",
}

# Colour-blind-friendly palette (Wong 2011 + extras)
MODEL_COLORS = {
    "XGBoost": "#0072B2", "RandomForest": "#009E73", "LightGBM": "#E69F00",
    "SVM": "#D55E00", "kNN": "#CC79A7", "MLP": "#56B4E9",
    "CNN1D": "#F0E442", "BiLSTM": "#000000",
}
MODEL_MARKERS = {
    "XGBoost": "o", "RandomForest": "s", "LightGBM": "D", "SVM": "^",
    "kNN": "v", "MLP": "P", "CNN1D": "X", "BiLSTM": "*",
}
DATASET_COLORS = {"cicids2017": "#0072B2", "ton_iot": "#D55E00",
                  "cicids2018": "#009E73", "unsw_nb15": "#CC79A7"}
DATASET_MARKERS = {"cicids2017": "o", "ton_iot": "s", "cicids2018": "D", "unsw_nb15": "^"}

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
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def _ds_label(ds: str) -> str:
    return DATASET_LABELS.get(ds, ds)


# ── Metrics computation (called from model.py during training) ────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    roc = np.nan
    pr = np.nan
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc = roc_auc_score(y_true, y_proba[:, 1])
                pr = average_precision_score(y_true, y_proba[:, 1])
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
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


# ── Figure 1: Performance heatmap ────────────────────────────────────

def fig1_heatmap(results_df: pd.DataFrame) -> None:
    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="model", columns="dataset", values="f1_macro")
        pivot = pivot.reindex(index=[m for m in MODEL_NAMES if m in pivot.index])
        pivot = pivot.reindex(columns=[d for d in DATASETS if d in pivot.columns])
        pivot.columns = [_ds_label(c) for c in pivot.columns]

        n_ds = len(pivot.columns)
        fig_w = max(5.5, 3.5 + n_ds * 2.4)
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlGnBu",
            linewidths=1.0, linecolor="white", ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
            cbar_kws={"label": "F1-macro", "shrink": 0.75},
        )
        task_label = "Binary" if task == "binary" else "Multi-class"
        ax.set_title(f"F1-macro — {task_label} Classification", fontsize=13, pad=14)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)
        fig.tight_layout(pad=1.5)
        out = FIGURES / f"fig1_heatmap_{task}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out.name)


# ── Figure 2: Grouped bar chart ──────────────────────────────────────

def fig2_grouped_bars(results_df: pd.DataFrame) -> None:
    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task].copy()
        if sub.empty:
            continue
        datasets_present = [d for d in DATASETS if d in sub["dataset"].values]
        models_present = [m for m in MODEL_NAMES if m in sub["model"].values]
        n_datasets = len(datasets_present)
        n_models = len(models_present)

        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6))
        x = np.arange(n_models)
        bar_width = 0.7 / max(n_datasets, 1)
        offsets = np.linspace(
            -(n_datasets - 1) * bar_width / 2,
            (n_datasets - 1) * bar_width / 2,
            n_datasets,
        )

        for i, ds in enumerate(datasets_present):
            vals, stds = [], []
            for m in models_present:
                row = sub[(sub["model"] == m) & (sub["dataset"] == ds)]
                vals.append(row["f1_macro"].values[0] if len(row) else 0)
                stds.append(row["f1_macro_std"].values[0] if (len(row) and "f1_macro_std" in row.columns) else 0)
            ax.bar(
                x + offsets[i], vals, bar_width,
                label=_ds_label(ds), yerr=stds, capsize=2,
                color=DATASET_COLORS.get(ds, f"C{i}"),
                edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models_present, fontsize=9.5, rotation=25, ha="right")
        ax.set_ylabel("F1-macro", fontsize=11)
        task_label = "Binary" if task == "binary" else "Multi-class"
        ax.set_title(f"Model Comparison — {task_label} Classification", fontsize=13, pad=12)
        y_max = max(sub["f1_macro"].max() * 1.08, 0.5)
        ax.set_ylim(0, min(1.10, y_max))
        ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        fig.tight_layout(pad=1.5)
        out = FIGURES / f"fig2_grouped_bar_{task}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out.name)


# ── Figure 3: SHAP beeswarm ──────────────────────────────────────────

def fig3_shap_beeswarm(datasets_to_run: list = None) -> None:
    targets = datasets_to_run or DATASETS
    for ds in targets:
        model_path = MODEL_DIR / f"{ds}_binary_XGBoost.joblib"
        if not model_path.exists():
            log.warning("No saved model for %s — skipping SHAP", ds)
            continue
        data_path = DATA_CLEAN / f"{ds}.parquet"
        if not data_path.exists():
            continue
        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]
        # Read only the needed columns to reduce memory
        df = pd.read_parquet(data_path, columns=features)
        n_sample = min(2000, len(df))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(df), n_sample, replace=False)
        X_sample = df.iloc[idx][features].values
        del df  # free memory

        log.info("Computing SHAP for %s (%d samples)…", ds, n_sample)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample, feature_names=features,
            show=False, max_display=20, plot_size=None,
        )
        plt.title(f"SHAP Feature Importance — {_ds_label(ds)} (XGBoost, Binary)",
                  fontsize=12, pad=14)
        plt.tight_layout(pad=1.5)
        out = FIGURES / f"fig3_shap_beeswarm_{ds}.pdf"
        plt.savefig(out)
        plt.close("all")
        log.info("Saved %s", out.name)


# ── Figure 4: Multi-metric grouped bar ───────────────────────────────

def fig4_multi_metric(results_df: pd.DataFrame) -> None:
    metrics_to_plot = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1-macro", "Precision", "Recall"]
    metric_colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]

    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task].copy()
        if sub.empty:
            continue
        datasets_present = [d for d in DATASETS if d in sub["dataset"].values]
        models_present = [m for m in MODEL_NAMES if m in sub["model"].values]

        for ds in datasets_present:
            ds_sub = sub[sub["dataset"] == ds]
            n_metrics = len(metrics_to_plot)
            n_models = len(models_present)

            fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6))
            x = np.arange(n_models)
            bar_width = 0.7 / n_metrics
            offsets = np.linspace(
                -(n_metrics - 1) * bar_width / 2,
                (n_metrics - 1) * bar_width / 2,
                n_metrics,
            )

            for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
                vals = []
                for m in models_present:
                    row = ds_sub[ds_sub["model"] == m]
                    vals.append(row[metric].values[0] if len(row) else 0)
                ax.bar(x + offsets[i], vals, bar_width, label=label,
                       color=metric_colors[i], edgecolor="white", linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(models_present, fontsize=9.5, rotation=25, ha="right")
            ax.set_ylabel("Score", fontsize=11)
            task_label = "Binary" if task == "binary" else "Multi-class"
            ax.set_title(
                f"Multi-metric Comparison — {_ds_label(ds)} ({task_label})",
                fontsize=13, pad=12,
            )
            ax.set_ylim(0, 1.08)
            ax.legend(loc="lower left", ncol=2, framealpha=0.95, fontsize=8.5)
            ax.grid(axis="y", alpha=0.25)
            ax.grid(axis="x", visible=False)
            fig.tight_layout(pad=1.5)
            out = FIGURES / f"fig4_multi_metric_{ds}_{task}.pdf"
            fig.savefig(out)
            plt.close(fig)
            log.info("Saved %s", out.name)


# ── Figure 5: Time vs F1 scatter (shape+fill markers, no numbers) ────

def fig5_time_vs_f1(results_df: pd.DataFrame) -> None:
    """Scatter plot using unique shape per model and filled/open markers per dataset.
    Two-part legend: model shapes and dataset fill styles.
    """
    from matplotlib.lines import Line2D

    # Fill styles: first dataset = filled, second = open (facecolor='none')
    ds_list = [d for d in DATASETS if d in results_df["dataset"].values]

    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task].copy()
        if sub.empty or "fit_seconds" not in sub.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        for _, row in sub.iterrows():
            m = row["model"]
            ds = row["dataset"]
            color = MODEL_COLORS.get(m, "#333333")
            marker = MODEL_MARKERS.get(m, "o")
            ds_idx = ds_list.index(ds) if ds in ds_list else 0
            if ds_idx == 0:
                # Filled marker
                ax.scatter(
                    row["fit_seconds"], row["f1_macro"],
                    color=color, marker=marker,
                    s=120, zorder=5, edgecolors="black", linewidths=0.6,
                )
            else:
                # Open marker (outline only)
                ax.scatter(
                    row["fit_seconds"], row["f1_macro"],
                    facecolors="none", edgecolors=color, marker=marker,
                    s=120, zorder=5, linewidths=1.8,
                )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"
        ))
        ax.set_xlabel("Training Time per Fold (seconds, log scale)", fontsize=11)
        ax.set_ylabel("F1-macro", fontsize=11)
        task_label = "Binary" if task == "binary" else "Multi-class"
        ax.set_title(f"Efficiency vs Performance — {task_label}", fontsize=13, pad=14)

        # Pad axes
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin * 0.5, xmax * 2.0)
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.06
        ax.set_ylim(ymin - pad, ymax + pad)

        # Legend part 1: model shapes (filled, with colour)
        model_handles = [
            Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
                   markerfacecolor=MODEL_COLORS[m], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=9, label=m, linestyle="None")
            for m in MODEL_NAMES if m in sub["model"].values
        ]
        # Legend part 2: dataset fill style
        ds_handles = []
        for i, ds in enumerate(ds_list):
            if i == 0:
                ds_handles.append(
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="gray", markeredgecolor="black",
                           markeredgewidth=0.5, markersize=9,
                           label=f"{_ds_label(ds)} (filled)", linestyle="None"))
            else:
                ds_handles.append(
                    Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="none", markeredgecolor="gray",
                           markeredgewidth=1.8, markersize=9,
                           label=f"{_ds_label(ds)} (open)", linestyle="None"))

        leg1 = ax.legend(handles=model_handles, loc="lower left", fontsize=7.5,
                         ncol=2, framealpha=0.95, title="Models", title_fontsize=8)
        ax.add_artist(leg1)
        ax.legend(handles=ds_handles, loc="lower right", fontsize=7.5,
                  framealpha=0.95, title="Datasets", title_fontsize=8)

        fig.tight_layout(pad=1.5)
        out = FIGURES / f"fig5_time_vs_f1_{task}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out.name)


# ── Figure 6: Ablation bar chart ─────────────────────────────────────

def fig6_ablation_bars() -> None:
    path = TABLES / "ablation_feature_selection.csv"
    if not path.exists():
        log.info("Fig6 ablation — run ablation.py first")
        return
    df = pd.read_csv(path)
    datasets_present = df["dataset"].unique()
    bar_colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]

    for ds in datasets_present:
        sub = df[df["dataset"] == ds]
        configs = sub["config"].values
        f1_vals = sub["f1_macro"].values
        n_feats = sub["n_features"].values

        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars = ax.bar(range(len(configs)), f1_vals,
                      color=bar_colors[:len(configs)],
                      edgecolor="white", linewidth=0.8, width=0.6)

        # Place value + feature count INSIDE bar if bar is tall enough, else above
        for bar, v, nf in zip(bars, f1_vals, n_feats):
            label = f"{v:.4f}\n({nf} feat)"
            y_pos = bar.get_height() - 0.02
            va = "top"
            color = "white"
            if bar.get_height() < 0.15:
                y_pos = bar.get_height() + 0.01
                va = "bottom"
                color = "black"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va=va, fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=9.5, rotation=15, ha="right")
        ax.set_ylabel("F1-macro", fontsize=11)
        ax.set_title(f"Feature Selection Ablation — {_ds_label(ds)} (Binary, XGBoost)",
                     fontsize=12, pad=14)
        ax.set_ylim(0, min(1.08, max(f1_vals) * 1.06))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        fig.tight_layout(pad=1.5)
        out = FIGURES / f"fig6_ablation_{ds}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out.name)


# ── Figure 7: Cross-dataset transfer heatmap ─────────────────────────

def fig7_cross_dataset_heatmap() -> None:
    path = TABLES / "cross_dataset_transfer.csv"
    if not path.exists():
        log.info("Fig7 cross-dataset — run ablation.py first")
        return
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="train_dataset", columns="test_dataset", values="f1_macro")
    pivot.index = [_ds_label(d) for d in pivot.index]
    pivot.columns = [_ds_label(d) for d in pivot.columns]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        linewidths=1.0, linecolor="white", ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        cbar_kws={"label": "F1-macro", "shrink": 0.75},
        vmin=0, vmax=1,
    )
    ax.set_title("Cross-Dataset Generalisation (XGBoost, Binary)", fontsize=12, pad=14)
    ax.set_ylabel("Train Dataset", fontsize=11)
    ax.set_xlabel("Test Dataset", fontsize=11)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout(pad=1.5)
    out = FIGURES / f"fig7_cross_dataset_transfer.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out.name)


# ── Figure 8: Training time horizontal bar ────────────────────────────

def fig8_training_time(results_df: pd.DataFrame) -> None:
    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task].copy()
        if sub.empty or "fit_seconds" not in sub.columns:
            continue
        datasets_present = [d for d in DATASETS if d in sub["dataset"].values]
        models_present = [m for m in MODEL_NAMES if m in sub["model"].values]
        n_ds = len(datasets_present)
        n_models = len(models_present)

        fig, ax = plt.subplots(figsize=(10, max(5, n_models * 0.8)))
        y = np.arange(n_models)
        bar_h = 0.7 / max(n_ds, 1)
        offsets = np.linspace(-(n_ds - 1) * bar_h / 2, (n_ds - 1) * bar_h / 2, n_ds)

        global_max = 0
        for i, ds in enumerate(datasets_present):
            vals = []
            for m in models_present:
                row = sub[(sub["model"] == m) & (sub["dataset"] == ds)]
                vals.append(row["fit_seconds"].values[0] if len(row) else 0)
            global_max = max(global_max, max(vals))
            bars = ax.barh(y + offsets[i], vals, bar_h,
                           label=_ds_label(ds),
                           color=DATASET_COLORS.get(ds, f"C{i}"),
                           edgecolor="white", linewidth=0.5)
            # Time labels inside bar end (to avoid overflow)
            for bar, v in zip(bars, vals):
                if v > 0:
                    txt = f"{v:,.0f}s" if v >= 10 else f"{v:.1f}s"
                    # Place inside bar if bar is wide enough, else outside
                    if v > global_max * 0.15:
                        ax.text(bar.get_width() * 0.97,
                                bar.get_y() + bar.get_height() / 2,
                                txt, va="center", ha="right",
                                fontsize=7.5, color="white", fontweight="bold")
                    else:
                        ax.text(bar.get_width() + global_max * 0.02,
                                bar.get_y() + bar.get_height() / 2,
                                txt, va="center", ha="left", fontsize=7.5)

        ax.set_yticks(y)
        ax.set_yticklabels(models_present, fontsize=10)
        ax.set_xlabel("Training Time per Fold (seconds)", fontsize=11)
        task_label = "Binary" if task == "binary" else "Multi-class"
        ax.set_title(f"Training Time Comparison — {task_label}", fontsize=13, pad=14)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"
        ))
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
        fig.tight_layout(pad=1.5)
        out = FIGURES / f"fig8_training_time_{task}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("Saved %s", out.name)


# ── Wilcoxon signed-rank test ─────────────────────────────────────────

def wilcoxon_test(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ["binary", "multi"]:
        sub = results_df[results_df["task"] == task]
        if sub.empty:
            continue
        mean_f1 = sub.groupby("model")["f1_macro"].mean()
        best = mean_f1.idxmax()
        log.info("Best model (%s, mean F1): %s (%.4f)", task, best, mean_f1[best])

        for other in MODEL_NAMES:
            if other == best:
                continue
            a = sub[sub["model"] == best]["f1_macro"].values
            b = sub[sub["model"] == other]["f1_macro"].values
            n = min(len(a), len(b))
            if n < 2:
                continue
            stat, pval = wilcoxon(a[:n], b[:n])
            rows.append({
                "task": task, "model_a": best, "model_b": other,
                "f1_a": round(float(np.mean(a)), 4),
                "f1_b": round(float(np.mean(b)), 4),
                "statistic": round(stat, 4), "p_value": round(pval, 4),
                "significant_005": pval < 0.05,
            })
    wdf = pd.DataFrame(rows)
    if not wdf.empty:
        out = TABLES / "wilcoxon_test.csv"
        wdf.to_csv(out, index=False)
        log.info("Wilcoxon results → %s", out)
    return wdf


# ── Main pipeline ─────────────────────────────────────────────────────

def generate_all_figures(run_shap: bool = False) -> None:
    results_path = TABLES / "benchmark_results.csv"
    if not results_path.exists():
        log.error("No results found at %s — run model.py first", results_path)
        return
    results_df = pd.read_csv(results_path)
    log.info("Loaded %d result rows", len(results_df))

    fig1_heatmap(results_df)
    fig2_grouped_bars(results_df)
    fig4_multi_metric(results_df)
    fig5_time_vs_f1(results_df)
    fig6_ablation_bars()
    fig7_cross_dataset_heatmap()
    fig8_training_time(results_df)
    wilcoxon_test(results_df)

    if run_shap:
        fig3_shap_beeswarm()

    log.info("All figures saved to %s", FIGURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    args = parser.parse_args()
    generate_all_figures(run_shap=args.shap)
