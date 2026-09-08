"""
01_IDS_Benchmark — Unified Preprocessing
Loads raw CSVs for each of the 4 datasets, applies a common cleaning
pipeline, and outputs parquet files with a unified schema.

Usage
-----
    python src/preprocessing.py                    # all datasets
    python src/preprocessing.py --dataset cicids2017
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import DATA_RAW, DATA_CLEAN, DATASETS, RANDOM_STATE, log

warnings.filterwarnings("ignore", category=FutureWarning)

# Maximum rows per dataset to cap memory (stratified sample if exceeded)
MAX_ROWS = 2_000_000

# Columns to drop (leakage / identifiers) — common across datasets
# Patterns are matched as substrings against normalised (lowercase, underscore-
# separated) column names.  Both space- and underscore-separated forms are
# included so that native Zeek column names (src_ip) and CICFlowMeter names
# (source_ip) are both caught.
LEAKAGE_PATTERNS = [
    "flow_id", "flow id",
    "source_ip", "source ip", "src_ip", "src ip", "srcip",
    "destination_ip", "destination ip", "dst_ip", "dst ip", "dstip",
    "source_port", "source port", "src_port", "src port", "srcport",
    "destination_port", "destination port", "dst_port", "dst port", "dstport",
    "timestamp", "time",
    "sport", "dport", "saddr", "daddr",
]


# ── Dataset-specific loaders ──────────────────────────────────────────

def _load_cicids2017() -> pd.DataFrame:
    """Load CICIDS2017 CSVs and normalise column names."""
    raw_dir = DATA_RAW / "cicids2017"
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {raw_dir} — run data_collection.py first")
    log.info("Loading %d CICIDS2017 CSV(s)…", len(csvs))
    dfs = [pd.read_csv(f, encoding="utf-8", low_memory=False) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Label column
    label_col = [c for c in df.columns if "label" in c][0]
    df["label_original"] = df[label_col].astype(str).str.strip()
    df["label_binary"] = (df["label_original"] != "BENIGN").astype(int)
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "cicids2017"
    return df


def _load_cicids2018() -> pd.DataFrame:
    """Load CSE-CIC-IDS2018 CSVs."""
    raw_dir = DATA_RAW / "cicids2018"
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {raw_dir} — run data_collection.py first")
    log.info("Loading %d CICIDS2018 CSV(s)…", len(csvs))
    dfs = [pd.read_csv(f, encoding="utf-8", low_memory=False) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    label_col = [c for c in df.columns if "label" in c][0]
    df["label_original"] = df[label_col].astype(str).str.strip()
    df["label_binary"] = (df["label_original"] != "Benign").astype(int)
    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "cicids2018"
    return df


def _load_ton_iot() -> pd.DataFrame:
    """Load ToN-IoT network subset."""
    raw_dir = DATA_RAW / "ton_iot"
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {raw_dir} — run data_collection.py first")
    log.info("Loading %d ToN-IoT CSV(s)…", len(csvs))
    dfs = [pd.read_csv(f, low_memory=False) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ToN-IoT uses 'label' (0/1) and 'type' (attack name)
    if "type" in df.columns:
        df["label_original"] = df["type"].astype(str).str.strip()
    elif "attack_type" in df.columns:
        df["label_original"] = df["attack_type"].astype(str).str.strip()
    else:
        df["label_original"] = "unknown"

    if "label" in df.columns:
        df["label_binary"] = df["label"].astype(int)
    else:
        df["label_binary"] = (df["label_original"] != "normal").astype(int)

    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "ton_iot"
    return df


def _load_unsw_nb15() -> pd.DataFrame:
    """Load UNSW-NB15 (4-part CSVs)."""
    raw_dir = DATA_RAW / "unsw_nb15"
    csvs = sorted(raw_dir.glob("*.csv"))
    # Try to find feature names file
    feat_file = raw_dir / "UNSW-NB15_features.csv"
    if feat_file.exists():
        feat_df = pd.read_csv(feat_file, encoding="latin-1")
        col_names = feat_df["Name"].str.strip().str.lower().tolist()
    else:
        col_names = None

    data_csvs = [f for f in csvs if "feature" not in f.name.lower()
                 and "GT" not in f.name and "LIST" not in f.name
                 and "training" not in f.name.lower()
                 and "testing" not in f.name.lower()]
    if not data_csvs:
        raise FileNotFoundError(f"No data CSVs in {raw_dir} — run data_collection.py first")
    log.info("Loading %d UNSW-NB15 CSV(s)…", len(data_csvs))

    dfs = []
    for f in data_csvs:
        d = pd.read_csv(f, header=None if col_names else 0, low_memory=False)
        if col_names and len(d.columns) == len(col_names):
            d.columns = col_names
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # UNSW-NB15 has 'attack_cat' and 'label' columns
    if "attack_cat" in df.columns:
        df["label_original"] = df["attack_cat"].astype(str).str.strip()
        # Treat empty / NaN attack_cat as Normal
        df.loc[df["label_original"].isin(["", "nan", "NaN"]), "label_original"] = "Normal"
    else:
        df["label_original"] = "unknown"

    if "label" in df.columns:
        df["label_binary"] = df["label"].fillna(0).astype(int)
    else:
        df["label_binary"] = (df["label_original"] != "Normal").astype(int)

    df["label_multi"] = df["label_original"].astype("category").cat.codes
    df["dataset"] = "unsw_nb15"
    return df


LOADERS = {
    "cicids2017": _load_cicids2017,
    "cicids2018": _load_cicids2018,
    "ton_iot": _load_ton_iot,
    "unsw_nb15": _load_unsw_nb15,
}


# ── Common cleaning pipeline ─────────────────────────────────────────

def clean(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Unified cleaning: drop leakage, handle inf/NaN, encode categoricals."""
    n_raw = len(df)
    log.info("[%s] Raw rows: %d, columns: %d", dataset_name, n_raw, len(df.columns))

    # 1. Drop leakage / identifier columns
    drop_cols = [
        c for c in df.columns
        if any(pat in c for pat in LEAKAGE_PATTERNS)
        and c not in ("label_original", "label_binary", "label_multi", "dataset")
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    log.info("[%s] Dropped %d leakage columns", dataset_name, len(drop_cols))

    # 2. Also drop raw label source columns (keep our unified ones)
    for col in ["label", "type", "attack_type", "attack_cat"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3. Encode remaining object/category columns
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    meta_cols = {"label_original", "dataset"}
    encode_cols = [c for c in obj_cols if c not in meta_cols]
    for col in encode_cols:
        df[col] = df[col].astype("category").cat.codes.astype("int16")
    if encode_cols:
        log.info("[%s] Encoded %d categorical columns", dataset_name, len(encode_cols))

    # 4. Coerce all feature columns to numeric
    feature_cols = [c for c in df.columns if c not in meta_cols | {"label_binary", "label_multi"}]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Replace inf → NaN, then drop rows that are entirely NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how="all", subset=feature_cols, inplace=True)

    # 6. Fill remaining NaN with 0 (safe for tree-based models; neutral for scaled)
    df[feature_cols] = df[feature_cols].fillna(0)

    # 7. Cap to MAX_ROWS (stratified on label_binary)
    if len(df) > MAX_ROWS:
        df, _ = train_test_split(
            df, train_size=MAX_ROWS,
            stratify=df["label_binary"], random_state=RANDOM_STATE,
        )
        log.info("[%s] Capped to %d rows (stratified)", dataset_name, MAX_ROWS)

    log.info("[%s] Clean rows: %d, features: %d",
             dataset_name, len(df),
             len([c for c in df.columns if c not in meta_cols | {"label_binary", "label_multi"}]))
    return df


def preprocess_dataset(name: str) -> None:
    """Load, clean, and save one dataset as parquet."""
    log.info("── Preprocessing %s ──", name)
    df = LOADERS[name]()
    df = clean(df, name)

    out_path = DATA_CLEAN / f"{name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Saved → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # Print class distribution
    for task in ["label_binary", "label_multi"]:
        dist = df[task].value_counts().to_dict()
        log.info("[%s] %s distribution: %s", name, task, dist)


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IDS datasets")
    parser.add_argument(
        "--dataset", choices=DATASETS,
        help="Preprocess one dataset (default: all available)",
    )
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else DATASETS
    for ds in targets:
        raw_dir = DATA_RAW / ds
        if not raw_dir.exists() or not list(raw_dir.glob("*.csv")):
            log.warning("Skipping %s — no raw data found. Run data_collection.py first.", ds)
            continue
        preprocess_dataset(ds)

    log.info("Done. Clean parquets in %s", DATA_CLEAN)
