"""
Run E7 (FT-Transformer) locally, overnight, on the RTX 3060.

Why local rather than Colab: E7 needs roughly 7-8 h of GPU on this machine,
which is more than a free Colab day allows and would span two sessions with a
disconnect risk each time.  Locally there is no quota and no session limit, so
it can simply run to completion while you sleep.  It is also gentler on the
machine than the earlier CPU experiments were: the fold is held on the GPU and
batched by index slicing, so it uses about one core of twenty, not half the
CPU.

The binding constraint is system RAM at load time, not the GPU.  Reading the
CICIDS2017 parquet peaks around 3 GB, and this script refuses to start if that
is not available - an out-of-memory kill six hours in would be worse than a
message now.

Resumable: every fold is written as it completes and finished folds are
skipped, so an interruption costs at most the fold in progress.

Usage
-----
    python run_e7_local.py                 # UNSW-NB15 then CICIDS2017
    python run_e7_local.py --dataset unsw_nb15
    python run_e7_local.py --check         # report readiness and exit
"""

import argparse
import time

import psutil
import torch

from utils import TABLES, log
import e67_dl_colab as dl

MIN_FREE_GB = 4.0          # peak parquet load is ~3 GB; leave headroom
MIN_GPU_FREE_GB = 3.0

ORDER = ["unsw_nb15", "cicids2017"]   # cheaper first


def _resources():
    vm = psutil.virtual_memory()
    free_ram = vm.available / 1024 ** 3
    gpu_free = gpu_total = None
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_free, gpu_total = free_b / 1024 ** 3, total_b / 1024 ** 3
    return free_ram, gpu_free, gpu_total


def check(strict=True):
    free_ram, gpu_free, gpu_total = _resources()
    print(f"RAM available : {free_ram:5.1f} GB   (need {MIN_FREE_GB:.1f})")
    if gpu_free is None:
        print("GPU           : none detected - E7 on CPU is not practical")
        if strict:
            raise SystemExit("Aborting: no CUDA device.")
        return False
    print(f"GPU free      : {gpu_free:5.1f} GB of {gpu_total:.1f} GB   "
          f"(need {MIN_GPU_FREE_GB:.1f})")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")

    ok = True
    if free_ram < MIN_FREE_GB:
        print(f"\n!! Only {free_ram:.1f} GB RAM free. Close browsers and other")
        print("   heavy apps, then re-run. Loading CICIDS2017 needs ~3 GB and")
        print("   an OOM kill part-way through would waste the whole run.")
        ok = False
    if gpu_free < MIN_GPU_FREE_GB:
        print(f"\n!! Only {gpu_free:.1f} GB GPU memory free. Browsers use the GPU")
        print("   for compositing; closing them frees it.")
        ok = False
    if ok:
        print("\nReady.")
    if strict and not ok:
        raise SystemExit("Aborting: not enough free memory (see above).")
    return ok


def remaining():
    """(dataset, task) pairs still to do, from what is already on disk."""
    done = dl._done(dl.OUT_E7, dl.KEY_E7)
    todo = []
    for ds in ORDER:
        for task in ("binary", "multi"):
            missing = [f for f in range(1, dl.N_FOLDS + 1)
                       if (ds, task, "FTTransformer", str(f)) not in done]
            if missing:
                todo.append((ds, task, missing))
    return todo


def main(datasets=None):
    check()
    log.info("device=%s max_epochs=%d patience=%d",
             dl.DEVICE, dl.MAX_EPOCHS, dl.PATIENCE)
    log.info("results -> %s", dl.OUT_E7)

    todo = [t for t in remaining()
            if datasets is None or t[0] in datasets]
    if not todo:
        log.info("Nothing left to run - E7 is complete for %s",
                 datasets or ORDER)
        return
    total_folds = sum(len(m) for _, _, m in todo)
    log.info("%d fold(s) to run: %s", total_folds,
             ", ".join(f"{d}/{t}:{len(m)}" for d, t, m in todo))

    t_start = time.time()
    for ds, task, missing in todo:
        log.info("=" * 60)
        log.info("E7 %s / %s  (%d fold(s) remaining)", ds, task, len(missing))
        t0 = time.time()
        dl.run_e7(ds, task)
        log.info("%s/%s finished in %.1f min", ds, task, (time.time() - t0) / 60)
        free_ram, gpu_free, _ = _resources()
        log.info("after: RAM free %.1f GB, GPU free %.1f GB", free_ram, gpu_free)

    log.info("=" * 60)
    log.info("E7 done in %.1f h -> %s", (time.time() - t_start) / 3600,
             dl.OUT_E7)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ORDER, help="run just one dataset")
    p.add_argument("--check", action="store_true",
                   help="report readiness and remaining work, then exit")
    a = p.parse_args()

    if a.check:
        check(strict=False)
        todo = remaining()
        print()
        if not todo:
            print("E7 is complete - nothing left to run.")
        else:
            for ds, task, missing in todo:
                print(f"  {ds}/{task}: folds {missing}")
            print(f"\n{sum(len(m) for _, _, m in todo)} fold(s) remaining; "
                  f"roughly 20-30 min each on this GPU.")
    else:
        main([a.dataset] if a.dataset else None)
