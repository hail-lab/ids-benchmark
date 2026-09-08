"""
Figure for E1: macro-F1 under stratified vs grouped vs temporal partitioning.

The partitioning result is the largest effect in the revision, and reporting it
only as a table understates it.  This draws one panel per task, datasets on the
x-axis, one bar per scheme, averaged over the three tree models with the
across-model range as the error bar.

ToN-IoT has no usable capture timestamp, so its temporal bar is absent by
design rather than missing; the panel labels that explicitly.

Usage:  python plot_e1_splits.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import TABLES, FIGURES, log

# same conventions as src/evaluation.py
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2, "font.size": 10, "font.family": "serif",
    "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8, "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8", "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

DATASET_NAMES = {"cicids2017": "CICIDS2017", "ton_iot": "ToN-IoT",
                 "unsw_nb15": "UNSW-NB15"}
SCHEMES = ["stratified", "group", "temporal"]
SCHEME_LABELS = {
    "stratified": "Stratified 5-fold (row-level)",
    "group": "Grouped (capture day / source host)",
    "temporal": "Temporal (train early, test late)",
}
# Wong 2011, as elsewhere in the project
SCHEME_COLORS = {"stratified": "#0072B2", "group": "#D55E00",
                 "temporal": "#009E73"}

OUT = FIGURES / "fig_e1_split_schemes.pdf"


def main():
    df = pd.read_csv(TABLES / "e1_split_schemes.csv")

    # mean over folds first, so each model contributes one number per cell
    per_model = (df.groupby(["dataset", "task", "scheme", "model"])["f1_macro"]
                   .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    datasets = ["cicids2017", "ton_iot", "unsw_nb15"]
    x = np.arange(len(datasets))
    width = 0.26

    for ax, task in zip(axes, ["binary", "multi"]):
        sub = per_model[per_model.task == task]
        for i, scheme in enumerate(SCHEMES):
            means, lo, hi = [], [], []
            for ds in datasets:
                v = sub[(sub.dataset == ds) & (sub.scheme == scheme)]["f1_macro"]
                if len(v) == 0:            # ToN-IoT has no temporal scheme
                    means.append(np.nan); lo.append(0.0); hi.append(0.0)
                else:
                    m = v.mean()
                    means.append(m)
                    lo.append(m - v.min())
                    hi.append(v.max() - m)
            pos = x + (i - 1) * width
            ax.bar(pos, means, width, yerr=[lo, hi], capsize=3,
                   color=SCHEME_COLORS[scheme], edgecolor="black",
                   linewidth=0.5, error_kw=dict(lw=0.8),
                   label=SCHEME_LABELS[scheme] if task == "binary" else None)
            # label above the error bar, not the bar, so the two do not collide
            for p, m, h in zip(pos, means, hi):
                if np.isnan(m):
                    ax.text(p, 0.02, "n/a", ha="center", va="bottom",
                            fontsize=7, rotation=90, color="0.4")
                else:
                    ax.text(p, m + h + 0.02, f"{m:.3f}", ha="center",
                            va="bottom", fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_NAMES[d] for d in datasets])
        ax.set_title("Binary classification" if task == "binary"
                     else "Multi-class classification")
        ax.set_ylim(0, 1.15)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Macro F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.09), frameon=True)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    log.info("saved -> %s", OUT)

    # print the deltas quoted in the text, so the figure and prose agree
    b = per_model[per_model.task == "binary"]
    for ds in datasets:
        s = b[(b.dataset == ds) & (b.scheme == "stratified")]["f1_macro"].mean()
        g = b[(b.dataset == ds) & (b.scheme == "group")]["f1_macro"].mean()
        print(f"{DATASET_NAMES[ds]:12s} binary  stratified {s:.4f} -> "
              f"grouped {g:.4f}   (delta {s - g:+.4f})")


if __name__ == "__main__":
    main()
