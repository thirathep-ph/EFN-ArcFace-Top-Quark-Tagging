"""Download Top Tagging Reference Dataset from Zenodo (record 2603256)."""

import argparse
import os
import sys
import urllib.request

ZENODO_RECORD = "https://zenodo.org/api/records/2603256/files"
FILES = ["train.h5", "val.h5", "test.h5"]
MIN_TOTAL_BYTES = 1_700_000_000  # ~1.7 GB across the three files


def _progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size and total_size > 0:
        percent = min(100, int(100 * downloaded / total_size))
        sys.stdout.write(f"\r  {percent}% ({downloaded / 1e9:.2f} GB)")
    else:
        sys.stdout.write(f"\r  {downloaded / 1e9:.2f} GB")
    sys.stdout.flush()


def download_file(url, dest):
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download Top Tagging dataset")
    parser.add_argument("--dest", type=str, default="data/top_tagging",
                        help="Destination directory (default: data/top_tagging)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    existing = [f for f in os.listdir(args.dest) if f.endswith(".h5")]
    total = sum(os.path.getsize(os.path.join(args.dest, f)) for f in existing)
    if total >= MIN_TOTAL_BYTES:
        print(f"Dataset already exists in {args.dest} ({total / 1e9:.2f} GB)")
        return

    for fname in FILES:
        dest = os.path.join(args.dest, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
            print(f"{fname} already present, skipping")
            continue
        url = f"{ZENODO_RECORD}/{fname}/content"
        download_file(url, dest)

    total = sum(os.path.getsize(os.path.join(args.dest, f))
                for f in os.listdir(args.dest) if f.endswith(".h5"))
    print(f"Done. Total {total / 1e9:.2f} GB in {args.dest}")


if __name__ == "__main__":
    main()
"""