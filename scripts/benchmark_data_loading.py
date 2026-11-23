"""Benchmark: H5 vs NumPy (mmap) data loading speed.

Usage (on Kaggle or local):
    python -m scripts.benchmark_data_loading [--data-dir data/top_tagging] [--n-events 100000]
"""
import argparse
import os
import time

import numpy as np


def bench_h5(path, n_events, trials=3):
    import tables as tb
    times = []
    for _ in range(trials):
        t0 = time.time()
        f = tb.open_file(path, "r")
        tbl = f.root.table.table
        v0 = tbl.read(field="values_block_0", start=0, stop=n_events)
        v1 = tbl.read(field="values_block_1", start=0, stop=n_events)
        es = v0[:, 0:800:4]
        pxs = v0[:, 1:800:4]
        pys = v0[:, 2:800:4]
        pzs = v0[:, 3:800:4]
        labels = v1[:, 1]
        f.close()
        times.append(time.time() - t0)
    return min(times)


def bench_npy(base, n_events, trials=3):
    times = []
    for _ in range(trials):
        t0 = time.time()
        es = np.load(base + "_E.npy", mmap_mode="r")[:n_events]
        pxs = np.load(base + "_px.npy", mmap_mode="r")[:n_events]
        pys = np.load(base + "_py.npy", mmap_mode="r")[:n_events]
        pzs = np.load(base + "_pz.npy", mmap_mode="r")[:n_events]
        labels = np.load(base + "_labels.npy", mmap_mode="r")[:n_events]
        # force read
        _ = es.sum() + pxs.sum() + pys.sum() + pzs.sum() + labels.sum()
        times.append(time.time() - t0)
    return min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/top_tagging")
    parser.add_argument("--n-events", type=int, default=100000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    h5_path = os.path.join(args.data_dir, "train.h5")
    npy_base = os.path.join(args.data_dir, "train")

    print(f"Benchmarking {args.n_events:,} events, {args.trials} trials (best of)\n")

    if os.path.exists(h5_path):
        t = bench_h5(h5_path, args.n_events, args.trials)
        print(f"  H5 read:   {t:.3f}s  ({args.n_events/t:,.0f} events/s)")
    else:
        print("  H5 file not found, skipping")

    if os.path.exists(npy_base + "_E.npy"):
        t = bench_npy(npy_base, args.n_events, args.trials)
        print(f"  NPY read:  {t:.3f}s  ({args.n_events/t:,.0f} events/s)")
    else:
        print("  .npy files not found. Run: python -m scripts.convert_h5_to_npy")
        print("  Skipping NPY benchmark")

    print("\nDone.")


if __name__ == "__main__":
    main()