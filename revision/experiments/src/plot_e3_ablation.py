"""
Regenerate the three feature-selection ablation figures from the nested results.

The submitted figures were drawn from an ablation whose selectors were fitted
once on a train/test split.  The revision refits each selector inside every
training fold (e3_nested_fs.py), which changes the numbers slightly and, on
UNSW-NB15, reverses the sign of the hybrid selector's effect.  Leaving the old
figures beside the corrected table would put two different sets of numbers for
the same experiment in one paper.

Style follows the submitted figures: one panel per dataset, bars in the same
order and palette, value and feature count printed inside each bar.  Error bars
(range across the five folds) are new -- the fold spread is what makes the
near-equality of the four configurations readable.

Usage:  python plot_e3_ablation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import TABLES, FIGURES, log

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2, "font.size": 10, "font.family": "serif",
    "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

DATASET_NAMES = {"cicids2017": "CICIDS2017", "ton_iot": "ToN-IoT",
                 "unsw_nb15": "UNSW-NB15"}
N_FEATURES = {"cicids2017": 77, "ton_iot": 36, "unsw_nb15": 39}
ORDER = ["No FS", "MI only (top 30)", "RF only (top 15)", "Hybrid MI-RF (15)"]
LABELS = {"No FS": "No FS", "MI only (top 30)": "MI only (top 30)",
          "RF only (top 15)": "RF only (top 15)",
          "Hybrid MI-RF (15)": "Hybrid MI$\\to$RF (15)"}
# the submitted figures' palette (Wong 2011)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]


def main():
    df = pd.read_csv(TABLES / "e3_nested_fs.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    for ds, g in df.groupby("dataset"):
        means, lo, hi, nfeat = [], [], [], []
        for cfg in ORDER:
            v = g[g.config == cfg]["f1_macro"]
            m = v.mean()
            means.append(m)
            lo.append(m - v.min())
            hi.append(v.max() - m)
            n = g[g.config == cfg]["n_features"]
            nfeat.append(int(n.iloc[0]) if len(n) else N_FEATURES[ds])

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = np.arange(len(ORDER))
        ax.bar(x, means, 0.62, yerr=[lo, hi], capsize=3, color=COLORS,
               edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
        for xi, (m, nf) in enumerate(zip(means, nfeat)):
            ax.text(xi, m - 0.055, f"{m:.4f}\n({nf} feat)", ha="center",
                    va="top", fontsize=8.5, fontweight="bold", color="white")

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in ORDER], rotation=18, ha="right")
        ax.set_ylabel("F1-macro")
        ax.set_title(f"Feature Selection Ablation — {DATASET_NAMES[ds]}\n"
                     "(Binary, XGBoost, selector refitted in-fold)")
        ax.set_ylim(0, 1.06)
        ax.set_axisbelow(True)
        fig.tight_layout()

        out = FIGURES / f"fig6_ablation_{ds}.pdf"
        fig.savefig(out)
        plt.close(fig)
        log.info("saved -> %s", out.name)
        print(f"{DATASET_NAMES[ds]:12s} " +
              "  ".join(f"{c.split(' (')[0]}={m:.4f}"
                        for c, m in zip(ORDER, means)))


if __name__ == "__main__":
    main()
