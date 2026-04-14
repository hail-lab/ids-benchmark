"""Shared paths, logging, and configuration for IDS Benchmark project."""

import logging
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
FIGURES    = ROOT / "outputs" / "figures"
TABLES     = ROOT / "outputs" / "tables"
MODELS     = ROOT / "outputs" / "models"

for p in [DATA_RAW, DATA_CLEAN, FIGURES, TABLES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS = -1  # use all cores

# Dataset names (canonical keys used throughout the pipeline)
DATASETS = ["cicids2017", "cicids2018", "ton_iot", "unsw_nb15"]

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
