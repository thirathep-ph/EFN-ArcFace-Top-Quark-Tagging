"""One-time conversion: H5 (Top Tagging) -> NumPy arrays (mmap-friendly).

Reads train.h5 / val.h5 / test.h5 once and writes *.npy (float32) alongside
them. Subsequent training runs load from .npy via mmap (no h5 overhead).

Usage:
    python -m scripts.convert_h5_to_npy [--data-dir data/top_tagging]
"""
import argparse
import os

import numpy as np

N_CONS = 200


def convert_one(h5_path: str, out_dir: str = None) -> None:
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    out_dir = out_dir or os.path.dirname(h5_path)
    base = os.path.join(out_dir, stem)
    out_e = base + "_E.npy"
    if os.path.exists(out_e):
        print(f"  {base}: .npy already exist, skipping")
        return

    print(f"  Reading {h5_path} ...")
    import tables as tb
    f = tb.open_file(h5_path, "r")
    tbl = f.root.table.table
    n = tbl.shape[0]
    v0 = tbl.read(field="values_block_0")
    v1 = tbl.read(field="values_block_1")
    es = v0[:, 0:N_CONS * 4:4].astype(np.float32)
    pxs = v0[:, 1:N_CONS * 4:4].astype(np.float32)
    pys = v0[:, 2:N_CONS * 4:4].astype(np.float32)
    pzs = v0[:, 3:N_CONS * 4:4].astype(np.float32)
    labels = v1[:, 1].astype(np.int8)
    f.close()

    weights = np.ones(n, dtype=np.float32)

    np.save(base + "_E.npy", es)
    np.save(base + "_px.npy", pxs)
    np.save(base + "_py.npy", pys)
    np.save(base + "_pz.npy", pzs)
    np.save(base + "_labels.npy", labels)
    np.save(base + "_weights.npy", weights)
    print(f"  Saved {base}_*.npy  ({n:,} events, {es.nbytes/1e9:.2f} GB each)")


def main():
    parser = argparse.ArgumentParser(description="Convert H5 to NumPy arrays")
    parser.add_argument("--data-dir", type=str, default="data/top_tagging")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for .npy (default: same as data-dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        path = os.path.join(args.data_dir, split + ".h5")
        if os.path.exists(path):
            print(f"[{split}]")
            convert_one(path, out_dir)
        else:
            print(f"  {path} not found, skipping")

    print("Done.")


if __name__ == "__main__":
    main()