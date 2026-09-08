"""
Run E6 (CNN1D / BiLSTM hyperparameter search) locally.

Pure PyTorch, so it runs on the GPU here exactly as it would on Colab - no
cuML involved and no CPU fallback penalty. Expect roughly 1-3 h in total.

Because it competes with E7 for the GPU, run this *after* E7 finishes rather
than alongside it.

Resumable at trial and fold level: an interrupted search continues from the
next unfinished trial, and the best configuration is chosen from all trials
ever recorded, not just this session's.

Usage
-----
    python run_e6_local.py                 # all four combinations
    python run_e6_local.py --check
"""

import argparse
import time

import psutil
import torch

from utils import log
import e67_dl_colab as dl

MIN_FREE_GB = 4.0

# (dataset, task, model) - the multi-class task is where R7.1's objection bites
COMBOS = [
    ("ton_iot", "multi", "CNN1D"),
    ("ton_iot", "multi", "BiLSTM"),
    ("unsw_nb15", "multi", "CNN1D"),
    ("unsw_nb15", "multi", "BiLSTM"),
]


def check(strict=True):
    free = psutil.virtual_memory().available / 1024 ** 3
    print(f"RAM available : {free:5.1f} GB   (need {MIN_FREE_GB:.1f})")
    if torch.cuda.is_available():
        gpu_free, gpu_total = [x / 1024 ** 3 for x in torch.cuda.mem_get_info()]
        print(f"GPU free      : {gpu_free:5.1f} GB of {gpu_total:.1f} GB")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        if gpu_free < 2.0:
            print("\n!! Little GPU memory free - is E7 still running? "
                  "Let it finish first.")
            if strict:
                raise SystemExit("Aborting: GPU busy.")
            return False
    else:
        print("GPU           : none - E6 on CPU is impractical")
        if strict:
            raise SystemExit("Aborting: no CUDA device.")
        return False
    if free < MIN_FREE_GB:
        print(f"\n!! Only {free:.1f} GB RAM free - close browsers and re-run.")
        if strict:
            raise SystemExit("Aborting: not enough free memory.")
        return False
    if strict:
        print("\nReady.")
    return True


def remaining():
    trials = dl._done(dl.OUT_E6, dl.KEY_E6)
    folds = dl._done(dl.OUT_E6_FOLDS, dl.KEY_E6_FOLDS)
    out = []
    for ds, task, name in COMBOS:
        n_tr = sum(1 for k in trials if k[:3] == (ds, task, name))
        n_fo = sum(1 for k in folds if (k[0], k[1], k[2]) == (ds, task, name))
        if n_tr < dl.N_TRIALS or n_fo < dl.N_FOLDS:
            out.append((ds, task, name, dl.N_TRIALS - n_tr, dl.N_FOLDS - n_fo))
    return out


def main():
    check()
    todo = remaining()
    if not todo:
        log.info("Nothing left - E6 is complete.")
        return
    log.info("results -> %s , %s", dl.OUT_E6, dl.OUT_E6_FOLDS)
    for ds, task, name, nt, nf in todo:
        log.info("  %s/%s/%s: %d trial(s), %d tuned fold(s) remaining",
                 ds, task, name, max(nt, 0), max(nf, 0))

    t_start = time.time()
    for ds, task, name, _, _ in todo:
        log.info("=" * 60)
        log.info("E6 %s / %s / %s", ds, task, name)
        t0 = time.time()
        dl.run_e6(ds, task, name)
        log.info("finished in %.1f min", (time.time() - t0) / 60)

    log.info("=" * 60)
    log.info("E6 done in %.1f h", (time.time() - t_start) / 3600)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check(strict=False)
        todo = remaining()
        print()
        if not todo:
            print("E6 is complete - nothing left to run.")
        else:
            for ds, task, name, nt, nf in todo:
                print(f"  {ds}/{task}/{name}: {max(nt,0)} trials, "
                      f"{max(nf,0)} folds remaining")
    else:
        main()
