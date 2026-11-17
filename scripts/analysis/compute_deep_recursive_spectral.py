#!/usr/bin/env python
"""Deep Recursive Spectral Clustering

Recursively runs spectral clustering on each sub-subclass until no further
substructure is found (k=1) or minimum size threshold is reached.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from arcefn.utils.clustering import spectral_clustering_subclass, estimate_clusters_eigengap
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT = EXPERIMENTS / "robustness_s16_m05" / "deep_recursive_spectral.json"

SUBCLASS_NAMES = {0: "QCD_Dense", 1: "QCD_Dilute", 2: "Top_Dilute", 3: "Top_Dense"}

MIN_SIZE = 500  # minimum events to attempt clustering
MAX_DEPTH = 6   # safety limit


def cluster_recursive(x: np.ndarray, depth: int, path: str, min_size: int = MIN_SIZE) -> dict:
    """Recursively cluster until k=1 or min_size reached."""
    n = len(x)
    if n < min_size:
        return {"n": int(n), "depth": depth, "path": path, "stopped": "min_size", "has_substructure": False}

    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

    # Eigengap analysis
    optimal_k, eigenvalues = estimate_clusters_eigengap(x_norm, max_k=6)
    eigvals = eigenvalues.tolist() if eigenvalues is not None else None

    if optimal_k <= 1 or depth >= MAX_DEPTH:
        reason = "k=1" if optimal_k <= 1 else "max_depth"
        return {
            "n": int(n),
            "depth": depth,
            "path": path,
            "optimal_k": optimal_k,
            "eigenvalues": eigvals,
            "stopped": reason,
            "has_substructure": False
        }

    # Run spectral clustering
    sub_labels, k_found = spectral_clustering_subclass(x_norm, max_k=optimal_k)
    counts = Counter(sub_labels)
    order = sorted(counts.keys(), key=lambda k: -counts[k])
    label_map = {old: new for new, old in enumerate(order)}
    sub_labels_mapped = np.array([label_map[l] for l in sub_labels])
    sizes = [int(counts[old]) for old in order]

    print(f"  {'  '*depth}{path}: k={k_found} (sizes: {sizes})")

    # Recurse on each sub-cluster
    children = {}
    for new_idx, old_idx in enumerate(order):
        child_mask = sub_labels == old_idx
        child_x = x[child_mask]
        child_path = f"{path}.{new_idx}"
        children[f"sub{new_idx}"] = cluster_recursive(child_x, depth + 1, child_path, min_size)

    return {
        "n": int(n),
        "depth": depth,
        "path": path,
        "optimal_k": optimal_k,
        "k_found": int(k_found),
        "eigenvalues": eigvals,
        "has_substructure": True,
        "sub_sizes": sizes,
        "children": children
    }


def main() -> None:
    t0 = time.time()
    print("Loading embeddings...")
    emb_np = np.load(EMBEDDINGS)
    embeddings = emb_np["embeddings"]
    labels = emb_np["labels"]
    print(f"  {len(embeddings):,} events")

    # Get primary subclass labels
    print("Getting primary subclass partition...")
    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, "deep_recursive_spectral")

    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = preds == cls
        x = embeddings[mask]
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        sc_labels, n_clusters = spectral_clustering_subclass(x_norm, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda k: -counts[k])
        label_map = {order[0]: 0, order[1]: 1} if len(order) >= 2 else {order[0]: 0}
        sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
        subclass_labels[mask] = sc_labels + (2 * cls)

    print("Primary subclass counts:")
    for sc in range(4):
        n = (subclass_labels == sc).sum()
        print(f"  {SUBCLASS_NAMES[sc]}: {n:,} ({100 * n / len(labels):.1f}%)")

    # Deep recursive clustering on each primary subclass
    report = {
        "n_events": len(embeddings),
        "primary_partition": "predicted_labels_primary",
        "min_size": MIN_SIZE,
        "max_depth": MAX_DEPTH,
        "subclasses": {}
    }

    for sc_id, sc_name in SUBCLASS_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Deep recursive SC on {sc_name}...")
        mask = subclass_labels == sc_id
        x = embeddings[mask]
        print(f"  n = {len(x):,}")
        report["subclasses"][sc_name] = cluster_recursive(x, depth=1, path=sc_name)

    # Save
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"\n{'='*50}")
    print(f"Saved: {OUT}")
    print(f"Total time: {time.time() - t0:.0f}s")

    # Print summary
    def print_tree(node: dict, indent: int = 0):
        prefix = "  " * indent
        if node.get("has_substructure"):
            print(f"{prefix}{node['path']}: k={node['k_found']} (sizes: {node['sub_sizes']})")
            for child in node.get("children", {}).values():
                print_tree(child, indent + 1)
        else:
            print(f"{prefix}{node['path']}: STOP ({node['stopped']}, n={node['n']})")

    print("\nTree summary:")
    for sc_name, data in report["subclasses"].items():
        print_tree(data)


if __name__ == "__main__":
    main()