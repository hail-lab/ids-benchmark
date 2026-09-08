"""
Sequential driver for the local (CPU) revision experiments.

Runs one experiment at a time so that each gets the full machine, and logs a
one-line status per stage.  Safe to re-run: every stage is skipped if its
output CSV already exists, unless --force is given.

Usage:  python run_remaining.py [--force] [--only STAGE]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils import TABLES, log

SRC = Path(__file__).resolve().parent

# (stage name, script, args, expected output CSV)
STAGES = [
    ("e1_unsw",   "e1_splits.py",          ["--dataset", "unsw_nb15"],  None),
    ("e1_cicids", "e1_splits.py",          ["--dataset", "cicids2017"], None),
    ("e4_lgbm",   "e4_lgbm_sensitivity.py", [],                         "e4_lgbm_sensitivity.csv"),
    ("e3_fs",     "e3_nested_fs.py",       [],                          "e3_nested_fs.csv"),
    ("e8b_shap",  "e8b_shap_multi.py",     [],                          "e8b_shap_stability.csv"),
    ("e5_curve",  "e5_fairness.py",        ["--part", "B"],             "e5_subsample_curve.csv"),
    ("e5_weight", "e5_fairness.py",        ["--part", "A"],             "e5_class_weight.csv"),
    ("e8c_stats", "e8c_effect_sizes.py",   [],                          None),  # always re-run last
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()

    for name, script, extra, out_csv in STAGES:
        if args.only and args.only != name:
            continue
        if out_csv and not args.force and (TABLES / out_csv).exists():
            log.info("SKIP %s — %s already exists", name, out_csv)
            continue
        log.info("=" * 60)
        log.info("START %s: %s %s", name, script, " ".join(extra))
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-u", str(SRC / script), *extra],
                              cwd=SRC)
        mins = (time.time() - t0) / 60
        if proc.returncode == 0:
            log.info("DONE  %s in %.1f min", name, mins)
        else:
            log.error("FAIL  %s (exit %d) after %.1f min — continuing",
                      name, proc.returncode, mins)
    log.info("All stages finished.")


if __name__ == "__main__":
    main()
