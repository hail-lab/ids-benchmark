"""
Coverage-adjusted reading of the grouped/temporal multi-class results (E1).

Reviewer 6, round 2, comment 2: classes present in a held-out partition but
absent from its training partition are excluded from scoring, so the grouped
multi-class results are conditional on known-class coverage. The reviewer asks
for an open-set score or, at minimum, a coverage-adjusted interpretation.

No model is retrained. The E1 splits are deterministic (GroupKFold, seeded
StratifiedKFold, fixed day boundaries), so each fold's class composition is
reconstructed from the data. As a check on that reconstruction, the excluded
fraction it implies must equal the excluded_test_frac that E1 saved for every
fold, or the script stops.

Two quantities can then be adjusted exactly, because a flow whose class was
never seen in training is necessarily misclassified:

  accuracy_adj          = accuracy * (1 - excluded)
  balanced_accuracy_adj = balanced_accuracy * K_seen / (K_seen + K_unseen)

where balanced accuracy is mean per-class recall over the classes present in
the scored test set (K_seen of them), and each unseen class adds a recall of
zero. Macro F1 cannot be adjusted the same way: its class set includes any
class the model predicted, and per-sample predictions were not retained.

Reads:   revision/results_r1/tables/e1_split_schemes.csv
         revision/data_r1/{ds}_grouped.parquet  (built by prep_groups.py)
Writes:  revision/results_r2/tables/e1_coverage_folds.csv
         Revision_R2/results_r2/tables/e1_coverage_summary.csv

Usage:  python e1_coverage.py
"""

import numpy as np
import pandas as pd

from utils import DATA_R1, REVISION, log

RESULTS_R1 = REVISION / "results_r1"
TABLES = REVISION / "results_r2" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
from e1_splits import SCHEMES

E1_CSV = RESULTS_R1 / "tables" / "e1_split_schemes.csv"


def fold_coverage(ds):
    df = pd.read_parquet(DATA_R1 / f"{ds}_grouped.parquet",
                         columns=["label_multi", "meta_group", "meta_time"])
    y = df["label_multi"].to_numpy()
    rows = []
    for scheme, gen in SCHEMES[ds].items():
        for fold, tr, te in gen(df, y):
            seen = set(np.unique(y[tr]))
            test_classes = set(np.unique(y[te]))
            unseen = test_classes - seen
            excluded = 1.0 - np.isin(y[te], list(seen)).mean()
            rows.append(dict(dataset=ds, scheme=scheme, fold=fold,
                             k_test=len(test_classes),
                             k_seen=len(test_classes & seen),
                             k_unseen=len(unseen),
                             excluded_recomputed=round(excluded, 4)))
    return pd.DataFrame(rows)


def main():
    e1 = pd.read_csv(E1_CSV)
    e1 = e1[e1["task"] == "multi"]

    cov = pd.concat([fold_coverage(ds) for ds in sorted(e1["dataset"].unique())])
    m = e1.merge(cov, on=["dataset", "scheme", "fold"], how="left", validate="m:1")

    if m["k_test"].isna().any():
        raise SystemExit("some E1 folds have no reconstructed split")
    mismatch = m[(m["excluded_recomputed"] - m["excluded_test_frac"]).abs() > 1e-4]
    if len(mismatch):
        raise SystemExit("split reconstruction does not reproduce E1's excluded "
                         f"fractions:\n{mismatch[['dataset','scheme','fold','excluded_test_frac','excluded_recomputed']]}")
    log.info("reconstruction reproduces E1's excluded fraction on all %d rows", len(m))

    m["class_coverage"] = m["k_seen"] / m["k_test"]
    m["accuracy_adj"] = m["accuracy"] * (1 - m["excluded_test_frac"])
    m["balanced_accuracy_adj"] = m["balanced_accuracy"] * m["k_seen"] / m["k_test"]
    m.to_csv(TABLES / "e1_coverage_folds.csv", index=False)

    s = (m.groupby(["dataset", "scheme"])
          .agg(f1_macro=("f1_macro", "mean"),
               excluded=("excluded_test_frac", "mean"),
               class_coverage=("class_coverage", "mean"),
               min_class_coverage=("class_coverage", "min"),
               k_unseen_max=("k_unseen", "max"),
               accuracy=("accuracy", "mean"),
               accuracy_adj=("accuracy_adj", "mean"),
               balanced_accuracy=("balanced_accuracy", "mean"),
               balanced_accuracy_adj=("balanced_accuracy_adj", "mean"),
               n=("f1_macro", "size"))
          .round(4).reset_index())
    s.to_csv(TABLES / "e1_coverage_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s)


if __name__ == "__main__":
    main()
