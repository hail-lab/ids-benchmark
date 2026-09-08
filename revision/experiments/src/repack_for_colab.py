"""
Repack the Colab upload datasets so they are smaller and upload more reliably.

Colab's browser upload is unreliable for very large files - a truncated
transfer surfaces later as an opaque "Parquet magic bytes not found" error.
Two changes cut the payload by roughly a third:

  * feature columns stored as float32.  This is lossless with respect to what
    the experiments actually see, because every loader already does
    `df[cols].to_numpy(dtype=np.float32)` at read time.  Label columns are
    left untouched.
  * zstd compression instead of the pyarrow default.

Optionally also splits each output into fixed-size parts (--split-mb), which
the notebook reassembles; uploading several ~64 MB parts is far more reliable
than one large file.

Verification: after writing, the repacked file is re-read and the float32 view
of every feature column is compared against the original, so a silent
precision or ordering change cannot slip through.

Usage
-----
    python repack_for_colab.py
    python repack_for_colab.py --split-mb 64
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import PAPER_DATASETS, dataset_path, log

OUT_DIR = (Path(__file__).resolve().parents[1] / "colab_upload" / "data")
LABEL_COLS = {"label_original", "label_binary", "label_multi", "dataset"}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def repack(name: str, split_mb: int | None) -> dict:
    src = dataset_path(name)
    df = pd.read_parquet(src)
    before = src.stat().st_size

    feat = [c for c in df.columns if c not in LABEL_COLS]
    original = {c: df[c].to_numpy(dtype=np.float32) for c in feat}
    for c in feat:
        df[c] = df[c].astype("float32")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out,
                   compression="zstd")
    after = out.stat().st_size

    # verify the float32 view is bit-identical to what the loaders would build
    check = pd.read_parquet(out)
    assert list(check.columns) == list(df.columns), "column order changed"
    assert len(check) == len(df), "row count changed"
    for c in feat:
        if not np.array_equal(check[c].to_numpy(dtype=np.float32),
                              original[c], equal_nan=True):
            raise AssertionError(f"{name}: column {c} changed on repack")
    for c in LABEL_COLS & set(df.columns):
        if not check[c].equals(df[c]):
            raise AssertionError(f"{name}: label column {c} changed on repack")

    rows = len(check)
    log.info("[%s] %.1f -> %.1f MiB (%.0f%%), %s rows, verified",
             name, before / 1024**2, after / 1024**2, 100 * after / before,
             f"{rows:,}")

    info = {"name": out.name, "bytes": after, "rows": rows, "md5": md5(out)}

    if split_mb:
        part_bytes = split_mb * 1024 * 1024
        parts, idx = [], 0
        with open(out, "rb") as fh:
            while True:
                block = fh.read(part_bytes)
                if not block:
                    break
                p = OUT_DIR / f"{out.name}.part{idx:02d}"
                p.write_bytes(block)
                parts.append(p.name)
                idx += 1
        log.info("[%s] split into %d part(s) of <=%d MiB", name, len(parts),
                 split_mb)
        info["parts"] = parts

    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mb", type=int, default=None,
                    help="also write fixed-size parts of this many MiB")
    args = ap.parse_args()

    infos = [repack(ds, args.split_mb) for ds in PAPER_DATASETS]

    print("\nPaste this into the notebook's EXPECTED table:\n")
    print("EXPECTED = {")
    for i in infos:
        print(f"    {i['name']!r:24s}: ({i['bytes']:>11_}, {i['rows']:>9_}),")
    print("}")
    total = sum(i["bytes"] for i in infos)
    print(f"\ntotal upload: {total/1024**2:.0f} MiB")
    if args.split_mb:
        print("\nparts written:")
        for i in infos:
            for p in i.get("parts", []):
                print("   ", p)
