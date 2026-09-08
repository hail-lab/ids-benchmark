"""Shared paths, logging, and configuration for IDS Benchmark project.

Revision-R1 variant: reads data from the original project tree, but writes
every output (tables, figures, models) into Revision_R1/results_r1 so the
original submission outputs stay untouched.
"""

import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
# Normally .../01_IDS_Benchmark/Revision_R1/experiments/src/utils.py, so the
# project tree is three levels up.  On Colab this file is uploaded to a flat
# location such as /content/code, where those levels do not exist, so the
# derivation is guarded: missing ancestors fall back to the file's own folder
# and the notebook overrides DATA_CLEAN / DATA_R1 / TABLES explicitly.
_SRC = Path(__file__).resolve().parent


def _ancestor(path: Path, n: int) -> Path:
    parents = path.parents
    return parents[n] if n < len(parents) else path


REVISION   = _ancestor(_SRC, 1)       # .../Revision_R1
PROJECT    = _ancestor(_SRC, 2)       # .../01_IDS_Benchmark
ROOT       = PROJECT                  # kept for backwards compatibility

DATA_RAW   = PROJECT / "data" / "raw"
DATA_CLEAN = PROJECT / "data" / "clean"
DATA_R1    = REVISION / "data_r1"     # grouped/clean parquets built for the revision
FIGURES    = REVISION / "results_r1" / "figures"
TABLES     = REVISION / "results_r1" / "tables"
MODELS     = REVISION / "results_r1" / "models"

for p in [DATA_R1, FIGURES, TABLES, MODELS]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only or non-existent parent (e.g. a flat Colab upload).
        # The caller overrides these paths before use.
        pass

# Original submission outputs (read-only from revision code)
ORIG_TABLES = PROJECT / "outputs" / "tables"
ORIG_MODELS = PROJECT / "outputs" / "models"

# ── Logging ────────────────────────────────────────────────────────────
# force=True matters in notebooks: Colab and IPython install a root handler
# before user code runs, which makes a plain basicConfig() a silent no-op and
# leaves the root level at WARNING, so every log.info below would disappear.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── Constants ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS = -1  # use all cores

# Dataset names (canonical keys used throughout the pipeline)
DATASETS = ["cicids2017", "cicids2018", "ton_iot", "unsw_nb15"]

# The three datasets used in the paper
PAPER_DATASETS = ["cicids2017", "ton_iot", "unsw_nb15"]

# Model names
MODEL_NAMES = [
    "XGBoost", "RandomForest", "LightGBM", "SVM",
    "kNN", "MLP", "CNN1D", "BiLSTM",
]

# Evaluation metrics
METRICS = [
    "accuracy", "balanced_accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "roc_auc", "pr_auc",
]

# Columns that must never enter a feature matrix.  The stored
# unsw_nb15.parquet still carries identifier/time columns from the
# leakage experiment (srcip, sport, dstip, dsport, stime, ltime), so every
# revision script drops these defensively at load time.
META_COLS = {"label_original", "label_binary", "label_multi", "dataset"}
IDENTIFIER_PATTERNS = [
    "flow_id", "source_ip", "src_ip", "srcip",
    "destination_ip", "dst_ip", "dstip",
    "source_port", "src_port", "srcport",
    "destination_port", "dst_port", "dstport",
    "timestamp", "stime", "ltime",
    "sport", "dport", "saddr", "daddr",
]


def dataset_path(name: str):
    """Path to a dataset, preferring the revision-corrected copy.

    Revision R1 corrects two defects in the shared clean parquets (the
    duplicated UNSW-NB15 "Backdoor"/"Backdoors" labels and the mis-decoded
    CICIDS2017 web-attack labels).  Corrected files are written to
    Revision_R1/data_r1/ rather than overwriting the originals, so the
    submitted results remain reproducible; every revision script resolves
    datasets through this function and therefore picks up the corrected copy
    automatically when one exists.
    """
    corrected = DATA_R1 / f"{name}.parquet"
    return corrected if corrected.exists() else DATA_CLEAN / f"{name}.parquet"


def safe_feature_cols(columns) -> list:
    """Feature columns with meta, meta_* and identifier columns removed."""
    out = []
    for c in columns:
        if c in META_COLS or c.startswith("meta_"):
            continue
        if any(pat in c for pat in IDENTIFIER_PATTERNS):
            continue
        out.append(c)
    return out
