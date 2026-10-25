"""Download the Kaggle credit card fraud dataset into data/.

Credentials come from ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY),
never from the project .env — that file is mounted into the API and consumer
containers, which have no need for a personal Kaggle token.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"
DEFAULT_DEST = "data/creditcard.csv"


def fetch(dest: str = DEFAULT_DEST, force: bool = False) -> Path:
    """Download the dataset and copy the CSV to `dest`. Returns its path."""
    dest_path = Path(dest)
    if dest_path.exists() and not force:
        print(f"{dest_path} already exists; pass --force to re-download")
        return dest_path

    import kagglehub

    # kagglehub caches under ~/.cache/kagglehub and returns the cache dir,
    # so the CSV still has to be copied to where train.py looks for it.
    cache_dir = Path(kagglehub.dataset_download(DATASET))

    matches = list(cache_dir.rglob(CSV_NAME))
    if not matches:
        raise FileNotFoundError(f"{CSV_NAME} not found under {cache_dir}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dest_path)

    size_mb = dest_path.stat().st_size / 1_000_000
    print(f"Dataset ready at {dest_path} ({size_mb:.1f} MB)")
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch(dest=args.dest, force=args.force)
