"""
Friedman Test and Critical Difference Diagram

Performs the Friedman test and Nemenyi post-hoc analysis across all
models and datasets, then generates a Critical Difference (CD) diagram.

Requires: scikit-posthocs (pip install scikit-posthocs)

Outputs:
  - outputs/tables/friedman_results.csv
  - outputs/figures/fig_cd_diagram_{task}.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

from utils import TABLES, FIGURES, MODEL_NAMES, DATASETS, log

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.size": 10,
    "font.family": "serif",
})


def _build_rank_matrix(results_df, task):
    """
    Build a matrix: rows = datasets, columns = models, values = F1-macro.
    Only includes datasets/models present in the data.
    """
    sub = results_df[results_df["task"] == task].copy()
    if sub.empty:
        return None, None

    datasets_present = [d for d in DATASETS if d in sub["dataset"].values]
    models_present = [m for m in MODEL_NAMES if m in sub["model"].values]

    # Build score matrix (datasets × models)
    scores = np.full((len(datasets_present), len(models_present)), np.nan)
    for i, ds in enumerate(datasets_present):
        for j, m in enumerate(models_present):
            row = sub[(sub["dataset"] == ds) & (sub["model"] == m)]
            if not row.empty:
                scores[i, j] = row.iloc[0]["f1_macro"]

    # Drop models with any NaN
    valid_cols = ~np.isnan(scores).any(axis=0)
    scores = scores[:, valid_cols]
    models_present = [m for m, v in zip(models_present, valid_cols) if v]

    return scores, models_present


def friedman_test(results_df):
    """Run Friedman test + compute average ranks for each task."""
    all_results = []

    for task in ["binary", "multi"]:
        scores, models = _build_rank_matrix(results_df, task)
        if scores is None or scores.shape[0] < 2:
            log.warning(
                "Friedman test for %s: need ≥2 datasets, have %d",
                task, 0 if scores is None else scores.shape[0],
            )
            continue

        n_datasets, n_models = scores.shape
        log.info(
            "Friedman test (%s): %d datasets × %d models",
            task, n_datasets, n_models,
        )

        # Compute ranks per dataset (higher F1 → rank 1)
        ranks = np.zeros_like(scores)
        for i in range(n_datasets):
            ranks[i] = rankdata(-scores[i])  # negative for descending

        avg_ranks = ranks.mean(axis=0)

        # Friedman test (requires ≥3 groups and ≥2 blocks ideally)
        if n_models >= 3 and n_datasets >= 2:
            try:
                stat, p_value = friedmanchisquare(*[scores[:, j] for j in range(n_models)])
            except Exception as e:
                log.warning("Friedman test failed: %s", e)
                stat, p_value = np.nan, np.nan
        else:
            stat, p_value = np.nan, np.nan
            log.warning("Not enough groups/blocks for Friedman test")

        for j, m in enumerate(models):
            all_results.append({
                "task": task,
                "model": m,
                "avg_rank": round(avg_ranks[j], 2),
                "friedman_stat": round(stat, 4) if not np.isnan(stat) else np.nan,
                "friedman_p": round(p_value, 4) if not np.isnan(p_value) else np.nan,
                "n_datasets": n_datasets,
            })

        # Print summary
        task_label = "Binary" if task == "binary" else "Multi-class"
        print(f"\n=== Friedman Test — {task_label} ===")
        print(f"  Datasets: {n_datasets}, Models: {n_models}")
        if not np.isnan(stat):
            print(f"  χ² = {stat:.4f}, p = {p_value:.4f}")
            print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        print(f"\n  Average Ranks (lower = better):")
        sorted_idx = np.argsort(avg_ranks)
        for idx in sorted_idx:
            print(f"    {models[idx]:15s}  rank = {avg_ranks[idx]:.2f}")

        # CD diagram
        _plot_cd_diagram(avg_ranks, models, n_datasets, task, stat, p_value)

    out_df = pd.DataFrame(all_results)
    if not out_df.empty:
        out_path = TABLES / "friedman_results.csv"
        out_df.to_csv(out_path, index=False)
        log.info("Saved → %s", out_path)
    return out_df


def _nemenyi_cd(n_models, n_datasets, alpha=0.05):
    """
    Compute the Nemenyi critical difference.
    CD = q_α * sqrt(k*(k+1) / (6*N))
    where k = number of models, N = number of datasets
    q_α values from Demšar (2006) Table 5.
    """
    # q_α values for α=0.05, indexed by k (number of groups)
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q = q_alpha.get(n_models)
    if q is None:
        log.warning("No q_α tabulated for k=%d models", n_models)
        return None
    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    return cd


def _plot_cd_diagram(avg_ranks, models, n_datasets, task, stat, p_value):
    """
    Draw a Critical Difference diagram.
    Models whose average rank difference is within CD are connected.
    """
    n_models = len(models)
    cd = _nemenyi_cd(n_models, n_datasets)

    # Sort models by average rank
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx]
    sorted_names = [models[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Draw axis line
    rank_min = 1
    rank_max = n_models
    margin = 0.5
    ax.plot(
        [rank_min - margin, rank_max + margin], [0, 0],
        "k-", linewidth=1.5,
    )

    # Place tick marks for each rank position
    for r in range(1, n_models + 1):
        ax.plot([r, r], [-0.05, 0.05], "k-", linewidth=1)
        ax.text(r, 0.12, str(r), ha="center", va="bottom", fontsize=9)

    # Place models alternating above and below
    y_positions = []
    for i in range(n_models):
        if i % 2 == 0:
            y = -0.3 - (i // 2) * 0.25
        else:
            y = 0.3 + (i // 2) * 0.25
        y_positions.append(y)

    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        y = y_positions[i]
        # Line from axis to model name
        ax.plot([rank, rank], [0, y * 0.6], "k-", linewidth=0.8)
        ax.plot(rank, y * 0.6, "ko", markersize=3)
        # Model name
        va = "top" if y < 0 else "bottom"
        ax.text(
            rank, y, f"{name}\n({rank:.1f})",
            ha="center", va=va, fontsize=8.5,
            fontweight="bold" if i == 0 else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat" if i == 0 else "white",
                      edgecolor="gray", alpha=0.8),
        )

    # Draw CD bar if available
    if cd is not None:
        cd_y = max(y_positions) + 0.4 if max(y_positions) > 0 else 0.8
        ax.plot([rank_min, rank_min + cd], [cd_y, cd_y], "r-", linewidth=2.5)
        ax.text(
            rank_min + cd / 2, cd_y + 0.1,
            f"CD = {cd:.2f}", ha="center", fontsize=9,
            color="red", fontweight="bold",
        )

        # Draw connecting bars for groups that are not significantly different
        groups = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                    groups.append((i, j))

        # Merge overlapping groups
        bar_y = min(y_positions) - 0.4 if min(y_positions) < 0 else -0.8
        drawn_bars = []
        for start, end in groups:
            r1, r2 = sorted_ranks[start], sorted_ranks[end]
            # Check if this bar is subsumed by an existing one
            subsumed = False
            for dr1, dr2 in drawn_bars:
                if r1 >= dr1 and r2 <= dr2:
                    subsumed = True
                    break
            if not subsumed:
                ax.plot(
                    [r1 - 0.05, r2 + 0.05], [bar_y, bar_y],
                    linewidth=3, color="#0072B2", alpha=0.6,
                    solid_capstyle="round",
                )
                bar_y -= 0.15
                drawn_bars.append((r1, r2))

    # Title and formatting
    task_label = "Binary" if task == "binary" else "Multi-class"
    title = f"Critical Difference Diagram — {task_label}"
    if not np.isnan(p_value):
        title += f"\n(Friedman χ² = {stat:.2f}, p = {p_value:.4f})"
    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xlim(rank_min - margin - 0.3, rank_max + margin + 0.3)
    ax.axis("off")
    fig.tight_layout(pad=2.0)

    out = FIGURES / f"fig_cd_diagram_{task}.pdf"
    fig.savefig(out)
    plt.close(fig)
    log.info("Saved %s", out.name)


if __name__ == "__main__":
    results_path = TABLES / "benchmark_results.csv"
    if not results_path.exists():
        log.error("No results at %s", results_path)
    else:
        results_df = pd.read_csv(results_path)
        friedman_test(results_df)
        log.info("Done.")
