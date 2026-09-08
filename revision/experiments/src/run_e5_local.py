"""
Run E5 Part A locally (no Colab quota needed).

Locally there is no cuML - RAPIDS is Linux-only - so Random Forest and SVM run
on scikit-learn/CPU instead of the GPU.  That makes this much slower than the
~35 min it takes on a Colab T4: expect roughly 6-8 h, dominated by the RBF-SVM
fits (about 13 min each on this CPU versus 2 s on a GPU).

The upside is that the work is almost entirely CPU-bound, while E7 is
GPU-bound.  The two can therefore run at the same time in separate terminals
with little contention, which is the fastest route to finishing both.

Part B is skipped automatically - it is already complete in results_r1.

Resumable: every fit is written as it completes and finished fits are skipped.

Usage
-----
    python run_e5_local.py                  # all three datasets
    python run_e5_local.py --dataset ton_iot
    python run_e5_local.py --check          # readiness + what is left
"""

import argparse
import time

import pandas as pd
import psutil
import torch

from utils import PAPER_DATASETS, log
import e5_fairness_gpu as e5

MIN_FREE_GB = 4.0
ORDER = ["ton_iot", "unsw_nb15", "cicids2017"]      # cheapest first

# 6 jobs per fold x 3 folds x 2 tasks
JOBS_PER_DATASET = 6 * e5.N_FOLDS * 2


def check(strict=True):
    free = psutil.virtual_memory().available / 1024 ** 3
    print(f"RAM available : {free:5.1f} GB   (need {MIN_FREE_GB:.1f})")
    print(f"backend       : {e5.BACKEND}"
          + ("  (GPU)" if e5.BACKEND == "cuml"
             else "  (CPU - expect 6-8 h; this is normal on Windows)"))
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}  "
              f"(used by the MLP arm only)")
    if free < MIN_FREE_GB:
        print(f"\n!! Only {free:.1f} GB RAM free - close browsers and re-run.")
        if strict:
            raise SystemExit("Aborting: not enough free memory.")
        return False
    if strict:
        print("\nReady.")
    return True


def remaining():
    done = e5._done(e5.OUT_A, e5.KEY_A)
    out = []
    for ds in ORDER:
        n = sum(1 for k in done if k[0] == ds)
        if n < JOBS_PER_DATASET:
            out.append((ds, JOBS_PER_DATASET - n))
    return out


def main(datasets=None):
    check()
    todo = [d for d in remaining() if datasets is None or d[0] in datasets]
    if not todo:
        log.info("Nothing left - E5 Part A is complete.")
        return
    log.info("results -> %s", e5.OUT_A)
    log.info("to run: %s", ", ".join(f"{d}({n} fits)" for d, n in todo))

    t_start = time.time()
    for ds, n in todo:
        log.info("=" * 60)
        log.info("E5 Part A - %s  (%d fit(s) remaining)", ds, n)
        t0 = time.time()
        e5.part_a(datasets=[ds])
        log.info("%s finished in %.1f min", ds, (time.time() - t0) / 60)

    log.info("=" * 60)
    log.info("E5 Part A done in %.1f h", (time.time() - t_start) / 3600)
    if e5.OUT_A.exists():
        a = pd.read_csv(e5.OUT_A)
        piv = a.pivot_table(index=["dataset", "task", "model"],
                            columns="config", values="f1_macro").round(4)
        print("\n" + piv.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=PAPER_DATASETS)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check(strict=False)
        todo = remaining()
        print()
        if not todo:
            print("E5 Part A is complete - nothing left to run.")
        else:
            for ds, n in todo:
                print(f"  {ds}: {n} of {JOBS_PER_DATASET} fits remaining")
    else:
        main([a.dataset] if a.dataset else None)
