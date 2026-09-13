"""
Redraw the benchmark figures that plot UNSW-NB15 multi-class F1.

Three figures were generated before the duplicated Backdoor labels were merged
and still show the 11-class values, which now disagree with Tables 5 and 7:

  fig2_grouped_bar_multi   per-model macro F1 by dataset
  fig5_time_vs_f1_multi    training time against macro F1
  fig_cd_diagram_multi     critical-difference diagram -- the worst of the
                           three, since the old Friedman statistics are drawn
                           into its title (chi2 = 13.22, p = 0.0669) and the
                           manuscript text now reports 12.22 and 0.094

Training times are deliberately taken from the original benchmark run rather
than from the refit. The refit was performed on a machine under varying load --
one MLP fold took 4,557 s and the next 1,050 s on identical work -- so its
timings measure contention, not the models. Tables 5 and 13 report the original
figures for the same reason, and the figures must agree with them.

The binary figures are unaffected: the label correction touched only
UNSW-NB15's multi-class task.

Usage:  python replot_benchmark_figures.py
"""

import shutil
import sys

import pandas as pd

from utils import TABLES, FIGURES, REVISION, log

SRC = REVISION.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import evaluation as ev          # noqa: E402
import friedman_cd as fcd        # noqa: E402

ORIG = REVISION.parent / "outputs" / "tables" / "benchmark_results.csv"
CORR = TABLES / "benchmark_results_corrected.csv"


def figure_source() -> pd.DataFrame:
    """Corrected F1 and accuracy, with the original run's training times."""
    corr = pd.read_csv(CORR)
    orig = pd.read_csv(ORIG)[["model", "dataset", "task",
                              "fit_seconds", "fit_seconds_std"]]
    merged = corr.drop(columns=[c for c in ("fit_seconds", "fit_seconds_std")
                                if c in corr.columns])
    merged = merged.merge(orig, on=["model", "dataset", "task"], how="left")
    missing = merged.fit_seconds.isna().sum()
    if missing:
        raise SystemExit(f"{missing} row(s) have no original timing")
    return merged


def main():
    df = figure_source()
    log.info("figure source: %d rows, corrected F1 with original timings",
             len(df))

    # both modules write to their own FIGURES; point them at the revision's
    ev.FIGURES = FIGURES
    fcd.FIGURES = FIGURES
    FIGURES.mkdir(parents=True, exist_ok=True)

    ev.fig2_grouped_bars(df)
    ev.fig5_time_vs_f1(df)
    fcd.friedman_test(df)

    # only the multi-class variants changed; copy those into the manuscript
    changed = ["fig2_grouped_bar_multi.pdf", "fig5_time_vs_f1_multi.pdf",
               "fig_cd_diagram_multi.pdf"]
    manu = REVISION / "manuscript"
    for name in changed:
        src = FIGURES / name
        if not src.exists():
            raise SystemExit(f"expected {name} to be regenerated")
        shutil.copy2(src, manu / name)
        print(f"  updated {name}")

    u = df[(df.dataset == "unsw_nb15") & (df.task == "multi")]
    print("\nUNSW-NB15 multi-class values now plotted:")
    print(u[["model", "f1_macro", "fit_seconds"]]
          .sort_values("f1_macro", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
