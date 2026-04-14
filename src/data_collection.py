"""
01_IDS_Benchmark — Data Collection
Downloads the four IDS datasets into data/raw/.

Datasets
--------
1. CICIDS2017  — CIC IDS 2017 (flow-based CSVs, ~2.8 M rows)
2. CICIDS2018  — CSE-CIC-IDS2018 (AWS-hosted CSVs)
3. TON_IoT     — ToN-IoT network dataset
4. UNSW-NB15   — UNSW-NB15 (4 CSV parts)

Usage
-----
    python src/data_collection.py                 # download all
    python src/data_collection.py --dataset cicids2017   # one dataset
"""

import argparse
import zipfile
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from utils import DATA_RAW, log

# ── Download registry ──────────────────────────────────────────────────
# Each entry: dataset_key -> list of (url, local_filename)
# NOTE: Some URLs may require manual download or acceptance of terms.
#       The script prints instructions for any dataset that cannot be
#       fetched automatically.
DOWNLOAD_REGISTRY = {
    "cicids2017": {
        "instructions": (
            "CICIDS2017 must be downloaded manually from:\n"
            "  https://www.unb.ca/cic/datasets/ids-2017.html\n"
            "Click 'CSV files' and extract all CSVs into:\n"
            "  data/raw/cicids2017/\n"
            "Expected files: Friday-WorkingHours-*.csv, "
            "Monday-WorkingHours.pcap_ISCX.csv, etc."
        ),
        "auto_urls": [],
    },
    "cicids2018": {
        "instructions": (
            "CICIDS2018 must be downloaded from the AWS bucket:\n"
            "  https://www.unb.ca/cic/datasets/ids-2018.html\n"
            "Use the AWS CLI:\n"
            "  aws s3 sync --no-sign-request "
            '"s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" '
            "data/raw/cicids2018/\n"
            "Or download each CSV manually from the page above."
        ),
        "auto_urls": [],
    },
    "ton_iot": {
        "instructions": (
            "ToN-IoT network dataset:\n"
            "  https://research.unsw.edu.au/projects/toniot-datasets\n"
            "Download the 'Network' subset CSVs and place in:\n"
            "  data/raw/ton_iot/\n"
            "Expected: Network_dataset_cleaned.csv or similar."
        ),
        "auto_urls": [],
    },
    "unsw_nb15": {
        "instructions": (
            "UNSW-NB15 dataset:\n"
            "  https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "Download UNSW-NB15_1.csv through UNSW-NB15_4.csv,\n"
            "UNSW-NB15_features.csv, and UNSW-NB15_GT.csv into:\n"
            "  data/raw/unsw_nb15/"
        ),
        "auto_urls": [],
    },
}


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a single file with progress bar."""
    if dest.exists():
        log.info("Already exists: %s — skipping", dest.name)
        return
    log.info("Downloading %s → %s", url, dest.name)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _extract(archive: Path, dest_dir: Path) -> None:
    """Auto-extract zip or tar.gz."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
        log.info("Extracted %s", archive.name)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
        log.info("Extracted %s", archive.name)


def download_dataset(name: str) -> None:
    """Download (or print instructions for) a single dataset."""
    info = DOWNLOAD_REGISTRY[name]
    dest_dir = DATA_RAW / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Auto-download any available URLs
    for url, fname in info.get("auto_urls", []):
        _download_file(url, dest_dir / fname)
        if fname.endswith((".zip", ".tar.gz", ".tgz")):
            _extract(dest_dir / fname, dest_dir)

    # Print manual instructions if no auto URLs
    if not info["auto_urls"]:
        log.warning("Manual download required for %s:", name)
        print("\n" + "=" * 60)
        print(info["instructions"])
        print("=" * 60 + "\n")

    # Verify
    csvs = list(dest_dir.glob("*.csv")) + list(dest_dir.rglob("*.csv"))
    if csvs:
        log.info("✓ %s: found %d CSV file(s) in %s", name, len(csvs), dest_dir)
    else:
        log.warning("✗ %s: no CSV files found yet — follow instructions above", name)


def verify_all() -> dict:
    """Return dict of {dataset: n_csv_files}."""
    status = {}
    for ds in DOWNLOAD_REGISTRY:
        csvs = list((DATA_RAW / ds).rglob("*.csv")) if (DATA_RAW / ds).exists() else []
        status[ds] = len(csvs)
        state = "✓" if csvs else "✗"
        log.info("%s %s: %d CSV(s)", state, ds, len(csvs))
    return status


# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IDS benchmark datasets")
    parser.add_argument(
        "--dataset", choices=list(DOWNLOAD_REGISTRY.keys()),
        help="Download a specific dataset (default: all)",
    )
    args = parser.parse_args()

    if args.dataset:
        download_dataset(args.dataset)
    else:
        for ds_name in DOWNLOAD_REGISTRY:
            download_dataset(ds_name)

    print("\n── Download Status ──")
    verify_all()
