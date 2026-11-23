"""Cache FastJet physics features to parquet for fast re-reading.

Computes 10 substructure features via FastJet C/A R=2.0 from test.h5
and saves to parquet. Subsequent runs read from parquet (~seconds)
instead of recomputing (~minutes).

Usage:
    python scripts/analysis/cache_physics_features.py [--max-events N] [--force]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: arcefn imports (which load awkward) must come before pyarrow —
# importing pyarrow first crashes the awkward native import in this venv.
from arcefn.data.loader import load_awkward          # noqa: E402
from arcefn.utils.physics import compute_features     # noqa: E402
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS   # noqa: E402

import numpy as np                                    # noqa: E402
import pyarrow as pa                                  # noqa: E402
import pyarrow.parquet as pq                          # noqa: E402

FEATURE_NAMES = [
    "Mass", "mSD", "Mult", "nSD",
    "sqrt_d12", "sqrt_d23", "Tau21", "Tau32",
    "zg", "theta_g",
]

OUT_PATH = EXPERIMENTS / "robustness_s16_m05" / "test_features.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events to cache (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if parquet exists")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not args.force:
        print(f"Cache exists: {OUT_PATH}. Use --force to recompute.")
        return

    h5_path = str(DATA_DIR / "test.h5")
    print(f"Loading {h5_path} (max_events={args.max_events})...")
    events, labels, _ = load_awkward(h5_path, max_events=args.max_events, lazy=False)
    n = len(events)
    print(f"  Loaded {n:,} events, labels shape={labels.shape}")

    print("Computing FastJet features (C/A R=2.0)...")
    features = compute_features(events)
    print(f"  Features shape: {features.shape}")

    # Build table from dict
    table = pa.table({
        "event_idx": pa.array(np.arange(n), type=pa.int64()),
        "label": pa.array(labels, type=pa.int8()),
        **{FEATURE_NAMES[i]: pa.array(features[:, i], type=pa.float32())
           for i in range(len(FEATURE_NAMES))},
    })
    print(f"Writing parquet to {OUT_PATH} ...")
    pq.write_table(table, OUT_PATH, compression="zstd")

    # Verify round-trip
    table2 = pq.read_table(OUT_PATH)
    print(f"  Verified: {table2.num_rows:,} rows x {table2.num_columns} cols")
    print(f"  File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
