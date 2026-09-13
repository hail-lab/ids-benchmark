"""
Rebuild the UNSW-NB15 multi-class leakage figure and statistics on the
corrected labels, and report whether the paper's headline figures move.

Run after rerun_leakage_unsw_multi.py. It splices the corrected UNSW-NB15
multi-class cells into a copy of the leakage comparison table, redraws
fig_leakage_unsw_nb15_multi.pdf, and prints the aggregate inflation before and
after so any change to the 0.018 / 0.092 figures quoted in the abstract is
visible rather than silent.

The original table is left untouched; the corrected one is written alongside it.

Usage:  python replot_leakage_unsw.py
"""

import shutil
import sys

import numpy as np
import pandas as pd

from utils import TABLES, FIGURES, REVISION, log

SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import leakage_comparison as lc          # noqa: E402

ORIG = REVISION.parent / "outputs" / "tables" / "leakage_comparison.csv"
OUT = TABLES / "leakage_comparison_corrected.csv"
LEAKY = TABLES / "unsw_multi_leaky_corrected.csv"
CLEAN = TABLES / "unsw_multi_corrected.csv"


def stats(df, label):
    out = {}
    for task in ("binary", "multi"):
        s = df[df.task == task]
        out[task] = (s.f1_inflation.mean(), s.f1_inflation.median(), len(s))
        print(f"  {label:9s} {task:6s}  n={len(s):2d}  "
              f"mean {s.f1_inflation.mean():.4f}  "
              f"median {s.f1_inflation.median():.4f}")
    return out


def main():
    for p in (LEAKY, CLEAN):
        if not p.exists():
            raise SystemExit(f"missing {p.name} -- run the refits first")

    orig = pd.read_csv(ORIG)
    leaky = pd.read_csv(LEAKY).set_index("model")
    clean = pd.read_csv(CLEAN).set_index("model")

    mask = (orig.dataset == "unsw_nb15") & (orig.task == "multi")
    models = sorted(orig[mask].model)
    missing = [m for m in models if m not in leaky.index or m not in clean.index]
    if missing:
        raise SystemExit(f"no corrected result for {missing}")

    corrected = orig.copy()
    for m in models:
        row = (corrected.dataset == "unsw_nb15") & (corrected.task == "multi") \
              & (corrected.model == m)
        c = float(clean.loc[m, "f1_macro"])
        l = float(leaky.loc[m, "f1_macro"])
        corrected.loc[row, "f1_macro_clean"] = c
        corrected.loc[row, "f1_macro_leaky"] = l
        corrected.loc[row, "f1_inflation"] = l - c
        corrected.loc[row, "f1_inflation_pct"] = (l - c) / c * 100
        if "accuracy_clean" in corrected.columns:
            corrected.loc[row, "accuracy_clean"] = float(clean.loc[m, "accuracy"])
            corrected.loc[row, "accuracy_leaky"] = float(leaky.loc[m, "accuracy"])
            corrected.loc[row, "accuracy_inflation"] = (
                float(leaky.loc[m, "accuracy"]) - float(clean.loc[m, "accuracy"]))
    corrected.to_csv(OUT, index=False)
    log.info("wrote %s", OUT.name)

    print("\naggregate inflation:")
    before = stats(orig, "original")
    after = stats(corrected, "corrected")

    print("\nfigures quoted in the abstract and highlights:")
    for task, name in (("binary", "binary"), ("multi", "multi-class")):
        b, a = before[task][0], after[task][0]
        flag = "" if round(b, 3) == round(a, 3) else "   <-- CHANGES"
        print(f"  mean {name:11s} {b:.4f} -> {a:.4f}  "
              f"(paper prints {round(a, 3):.3f}){flag}")
        b, a = before[task][1], after[task][1]
        flag = "" if round(b, 3) == round(a, 3) else "   <-- CHANGES"
        print(f"  med. {name:11s} {b:.4f} -> {a:.4f}  "
              f"(paper prints {round(a, 3):.3f}){flag}")

    # redraw only the panel that changed
    lc.FIGURES = FIGURES
    FIGURES.mkdir(parents=True, exist_ok=True)
    lc.fig_leakage_comparison(corrected)
    target = "fig_leakage_unsw_nb15_multi.pdf"
    src = FIGURES / target
    if not src.exists():
        raise SystemExit(f"{target} was not regenerated")
    shutil.copy2(src, REVISION / "manuscript" / target)
    print(f"\nupdated {target}")

    u = corrected[(corrected.dataset == "unsw_nb15") & (corrected.task == "multi")]
    print("\nUNSW-NB15 multi-class cells now plotted:")
    print(u[["model", "f1_macro_clean", "f1_macro_leaky", "f1_inflation"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
