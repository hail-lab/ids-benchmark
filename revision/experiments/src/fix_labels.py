"""
Revision R1 - write label-corrected copies of the clean parquets.

Two defects were found in the submitted pipeline while answering R6.10:

 1. UNSW-NB15 carries both "Backdoor" and "Backdoors" for the same attack
    category, and the submitted version scored them as two distinct classes.
    Table 7 of the submission therefore reported Backdoor (1,394 samples,
    F1 = 0.152) and Backdoors (421, F1 = 0.019) separately, penalising the
    classifier for confusing two labels that denote the same thing.

 2. CICIDS2017's web-attack labels contain a Latin-1 en-dash. Reading those
    files as UTF-8 corrupted three class names, which then propagated into
    every per-class table.

Corrected copies are written to Revision_R1/data_r1/ and are picked up
automatically by utils.dataset_path(); the originals in data/clean/ are left
untouched so the submitted results remain reproducible.

Usage:  python fix_labels.py
"""

import pandas as pd

from utils import DATA_CLEAN, DATA_R1, log

# Same normalisation as prep_groups.py, applied here to the clean parquets.
CIC_LABEL_FIX = {
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \x96 XSS": "Web Attack - XSS",
    "Web Attack \x96 Sql Injection": "Web Attack - SQL Injection",
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - SQL Injection",
    "Web Attack – Brute Force": "Web Attack - Brute Force",
    "Web Attack – XSS": "Web Attack - XSS",
    "Web Attack – Sql Injection": "Web Attack - SQL Injection",
}

# "None" is a string-conversion artifact: UNSW-NB15 leaves attack_cat empty for
# benign flows, and astype(str) on a genuine Python None yields "None", which
# the submitted pipeline's ["", "nan", "NaN"] replacement list did not catch.
# The class was still separated correctly; only its printed name was wrong.
UNSW_LABEL_FIX = {"Backdoors": "Backdoor", "None": "Normal"}

FIXES = {"cicids2017": CIC_LABEL_FIX, "unsw_nb15": UNSW_LABEL_FIX}


def fix(name: str) -> None:
    src = DATA_CLEAN / f"{name}.parquet"
    if not src.exists():
        log.warning("missing %s", src)
        return
    df = pd.read_parquet(src)
    before = df["label_original"].nunique()

    mapping = FIXES[name]
    df["label_original"] = df["label_original"].astype(str).str.strip()
    hits = df["label_original"].isin(mapping).sum()
    df["label_original"] = df["label_original"].replace(mapping)

    # Re-derive the integer multi-class code from the corrected names so the
    # codes stay contiguous and consistent with the label list.
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    after = df["label_original"].nunique()

    out = DATA_R1 / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    log.info("[%s] %d rows relabelled; classes %d -> %d; saved %s (%.1f MB)",
             name, hits, before, after, out.name, out.stat().st_size / 1e6)
    counts = df["label_original"].value_counts()
    log.info("[%s] classes: %s", name, dict(counts))


if __name__ == "__main__":
    for ds in FIXES:
        fix(ds)
    log.info("Done. utils.dataset_path() will now resolve to these copies.")
