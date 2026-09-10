"""
Build the label-corrected benchmark table and refresh everything derived from it.

The UNSW-NB15 multi-class results in outputs/tables/benchmark_results.csv were
computed before the duplicated ``Backdoor``/``Backdoors`` labels were merged, so
they describe an 11-class task.  recompute_unsw_multi.py re-ran all eight models
on the 10-class data; this script splices those numbers in and rewrites the
statistics that were computed from the stale table.

CICIDS2017 and ToN-IoT are copied through unchanged: CICIDS2017's correction
touched label *names* only and ToN-IoT was not corrected, so their numbers stand.

Outputs:
  results_r1/tables/benchmark_results_corrected.csv   -- the merged table
  results_r1/tables/e8c_kendalls_w.csv                -- recomputed
  results_r1/tables/e8c_pairwise_effects.csv          -- recomputed

Usage:  python merge_corrected_benchmark.py
"""

import pandas as pd

from utils import TABLES, REVISION, log

import e8c_effect_sizes as e8c

ORIG = REVISION.parent / "outputs" / "tables" / "benchmark_results.csv"
CORR = TABLES / "unsw_multi_corrected.csv"
OUT = TABLES / "benchmark_results_corrected.csv"


def main():
    orig = pd.read_csv(ORIG)
    corr = pd.read_csv(CORR)

    expected = {"XGBoost", "RandomForest", "LightGBM", "SVM", "kNN", "MLP",
                "CNN1D", "BiLSTM"}
    missing = expected - set(corr.model)
    if missing:
        raise SystemExit(f"recompute incomplete, missing: {sorted(missing)}")

    # drop the stale UNSW multi-class block and splice the corrected one in
    keep = orig[~((orig.dataset == "unsw_nb15") & (orig.task == "multi"))]
    corr = corr[[c for c in corr.columns if c in orig.columns]]
    merged = pd.concat([keep, corr], ignore_index=True)
    merged.to_csv(OUT, index=False)
    log.info("wrote %s (%d rows)", OUT.name, len(merged))

    # --- what changed -----------------------------------------------------
    old = orig[(orig.dataset == "unsw_nb15") & (orig.task == "multi")]
    print(f"\n{'model':14s} {'11-class':>9s} {'10-class':>9s} {'delta':>8s}")
    for _, r in corr.iterrows():
        o = old[old.model == r.model]
        ov = float(o.f1_macro.iloc[0]) if len(o) else float("nan")
        print(f"{r.model:14s} {ov:9.4f} {r.f1_macro:9.4f} {r.f1_macro - ov:+8.4f}")

    print("\nUNSW-NB15 multi-class ranking:")
    print("  before:", " > ".join(old.sort_values("f1_macro", ascending=False)
                                  .model.tolist()))
    print("  after :", " > ".join(corr.sort_values("f1_macro", ascending=False)
                                  .model.tolist()))

    # --- recompute the statistics that read the benchmark table -----------
    # e8c reads ORIG_TABLES/benchmark_results.csv; point it at the merged file
    # by writing the merged table where it looks, via a temporary override
    e8c.ORIG_TABLES = TABLES
    tmp = TABLES / "benchmark_results.csv"
    merged.to_csv(tmp, index=False)
    try:
        w = e8c.part1_kendall()
        p = e8c.part2_pairwise()
    finally:
        tmp.unlink(missing_ok=True)

    print("\nKendall's W (recomputed on corrected labels):")
    print(w.to_string(index=False))

    print("\nLargest pairwise gaps (multi-class):")
    m = p[p.task == "multi"].reindex(
        p[p.task == "multi"].mean_diff_f1.abs().sort_values(ascending=False).index)
    print(m.head(5)[["model_a", "model_b", "mean_diff_f1",
                     "rank_biserial", "cliffs_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
