"""
Revision E1 — build "grouped" parquets that retain split metadata.

The submission-time parquets carry no temporal or host information (identifier
columns are dropped and rows are shuffled by the stratified cap), so
alternative CV schemes cannot be built from them.  This script re-loads the
raw CSVs and produces {dataset}_grouped.parquet in Revision_R1/data_r1 with
the SAME cleaning as preprocessing.py plus extra metadata columns:

    meta_group : capture day (cicids2017) or source host (ton_iot, unsw_nb15)
    meta_time  : start time where available (unsw_nb15 stime); else file order

Usage:  python prep_groups.py [--dataset cicids2017|ton_iot|unsw_nb15]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import DATA_RAW, DATA_R1, RANDOM_STATE, PAPER_DATASETS, log
from preprocessing import LEAKAGE_PATTERNS

MAX_ROWS = 2_000_000  # same cap as preprocessing.py


def _clean_keep_meta(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """preprocessing.clean() logic, but meta_* columns pass through untouched."""
    meta_cols = {"label_original", "label_binary", "label_multi", "dataset",
                 "meta_group", "meta_time"}

    drop_cols = [
        c for c in df.columns
        if any(pat in c for pat in LEAKAGE_PATTERNS) and c not in meta_cols
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    log.info("[%s] Dropped %d identifier/time columns", dataset_name, len(drop_cols))

    for col in ["label", "type", "attack_type", "attack_cat"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encode_cols = [c for c in obj_cols if c not in meta_cols]
    for col in encode_cols:
        df[col] = df[col].astype("category").cat.codes.astype("int16")

    feature_cols = [c for c in df.columns if c not in meta_cols]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="all", subset=feature_cols, inplace=True)
    df[feature_cols] = df[feature_cols].fillna(0)

    if len(df) > MAX_ROWS:
        df, _ = train_test_split(
            df, train_size=MAX_ROWS,
            stratify=df["label_binary"], random_state=RANDOM_STATE,
        )
        log.info("[%s] Capped to %d rows (stratified)", dataset_name, MAX_ROWS)

    log.info("[%s] Clean rows: %d, features: %d", dataset_name, len(df),
             len([c for c in df.columns if c not in meta_cols]))
    return df


# ── Loaders with metadata ─────────────────────────────────────────────

DAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}

# CICIDS2017 web-attack labels contain a Latin-1 en-dash that becomes mojibake
# when the files are read as UTF-8.  Reading as latin-1 and normalising here
# gives stable, printable class names (R6.10 reproducibility).
CIC_LABEL_FIX = {
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \x96 XSS": "Web Attack - XSS",
    "Web Attack \x96 Sql Injection": "Web Attack - SQL Injection",
    "Web Attack – Brute Force": "Web Attack - Brute Force",
    "Web Attack – XSS": "Web Attack - XSS",
    "Web Attack – Sql Injection": "Web Attack - SQL Injection",
}


def load_cicids2017_grouped() -> pd.DataFrame:
    raw_dir = DATA_RAW / "cicids2017"
    dfs = []
    for f in sorted(raw_dir.glob("*.csv")):
        day = f.name.split("-")[0].capitalize()
        d = pd.read_csv(f, encoding="latin-1", low_memory=False)
        d.columns = d.columns.str.strip().str.lower().str.replace(" ", "_")
        d["meta_group"] = day
        d["meta_time"] = DAY_ORDER.get(day, 9)
        dfs.append(d)
        log.info("  %s: %d rows (%s)", f.name, len(d), day)
    df = pd.concat(dfs, ignore_index=True)

    label_col = [c for c in df.columns if "label" in c][0]
    df["label_original"] = (df[label_col].astype(str).str.strip()
                            .replace(CIC_LABEL_FIX))
    df["label_binary"] = (df["label_original"] != "BENIGN").astype(int)
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "cicids2017"
    df = df.drop(columns=[label_col], errors="ignore")
    return df


def load_ton_iot_grouped() -> pd.DataFrame:
    raw_dir = DATA_RAW / "ton_iot"
    df = pd.read_csv(raw_dir / "train_test_network.csv", low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # host group BEFORE identifier columns are dropped
    df["meta_group"] = df["src_ip"].astype(str)
    df["meta_time"] = np.arange(len(df))  # no timestamp available; keep file order

    df["label_original"] = df["type"].astype(str).str.strip()
    df["label_binary"] = df["label"].astype(int)
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "ton_iot"
    return df


def load_unsw_nb15_grouped() -> pd.DataFrame:
    raw_dir = DATA_RAW / "unsw_nb15"
    # Official 49-column schema for UNSW-NB15_{1..4}.csv (features file is empty
    # in this copy, so names are hardcoded from the dataset documentation).
    col_names = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
        "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
        "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
        "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
        "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "attack_cat", "label",
    ]
    dfs = []
    for i in (1, 2, 3, 4):
        f = raw_dir / f"UNSW-NB15_{i}.csv"
        d = pd.read_csv(f, header=None, names=col_names, low_memory=False)
        dfs.append(d)
        log.info("  %s: %d rows", f.name, len(d))
    df = pd.concat(dfs, ignore_index=True)

    df["meta_group"] = df["srcip"].astype(str)
    df["meta_time"] = pd.to_numeric(df["stime"], errors="coerce")

    df["label_original"] = df["attack_cat"].astype(str).str.strip()
    df.loc[df["label_original"].isin(["", "nan", "NaN"]), "label_original"] = "Normal"
    # Harmonise the known label typo (R6.10)
    df["label_original"] = df["label_original"].replace({"Backdoors": "Backdoor"})
    df["label_binary"] = df["label"].fillna(0).astype(int)
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "unsw_nb15"
    return df


LOADERS = {
    "cicids2017": load_cicids2017_grouped,
    "ton_iot": load_ton_iot_grouped,
    "unsw_nb15": load_unsw_nb15_grouped,
}


def build(name: str) -> None:
    log.info("── Building grouped parquet for %s ──", name)
    df = LOADERS[name]()
    df = _clean_keep_meta(df, name)
    out = DATA_R1 / f"{name}_grouped.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    log.info("Saved → %s (%.1f MB)", out, out.stat().st_size / 1e6)
    log.info("Groups: %d unique", df["meta_group"].nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PAPER_DATASETS)
    args = parser.parse_args()
    for ds in ([args.dataset] if args.dataset else PAPER_DATASETS):
        build(ds)
