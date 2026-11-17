#!/usr/bin/env python
"""Recursive Spectral Clustering on Subclasses

Runs spectral clustering (fast global cosine + eigengap) on each of the 4
primary subclasses to check for sub-substructure.

Uses the same parameter-free pipeline as the primary clustering.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from arcefn.utils.clustering import spectral_clustering_subclass, estimate_clusters_eigengap
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT = EXPERIMENTS / "robustness_s16_m05" / "recursive_spectral_subclasses.json"

SUBCLASS_NAMES = {0: "QCD_Dense", 1: "QCD_Dilute", 2: "Top_Dilute", 3: "Top_Dense"}


def main() -> None:
    t0 = time.time()
    print("Loading embeddings...")
    emb_np = np.load(EMBEDDINGS)
    embeddings = emb_np["embeddings"]
    labels = emb_np["labels"]
    print(f"  {len(embeddings):,} events")

    # Get primary subclass labels (same as compute_subclass_clustering.py)
    print("Getting primary subclass partition...")
    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, "recursive_spectral")

    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = preds == cls
        x = embeddings[mask]
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        sc_labels, n_clusters = spectral_clustering_subclass(x_norm, max_k=5)
        # Map to 0/1 by size (0 = larger = Dense, 1 = smaller = Dilute)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda k: -counts[k])
        label_map = {order[0]: 0, order[1]: 1} if len(order) >= 2 else {order[0]: 0}
        sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
        subclass_labels[mask] = sc_labels + (2 * cls)

    print("Primary subclass counts:")
    for sc in range(4):
        n = (subclass_labels == sc).sum()
        print(f"  {SUBCLASS_NAMES[sc]}: {n:,} ({100 * n / len(labels):.1f}%)")

    # Recursive spectral clustering on each subclass
    report = {
        "n_events": len(embeddings),
        "primary_partition": "predicted_labels_primary",
        "subclasses": {}
    }

    for sc_id, sc_name in SUBCLASS_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Recursive SC on {sc_name}...")
        mask = subclass_labels == sc_id
        x = embeddings[mask]
        n = len(x)
        print(f"  n = {n:,}")

        if n < 50:
            print("  Too few events, skipping")
            report["subclasses"][sc_name] = {"n": int(n), "skipped": True, "reason": "too_few_events"}
            continue

        # L2-normalize
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

        # Eigengap analysis (up to k=6)
        print("  Computing eigengap...")
        optimal_k, eigenvalues = estimate_clusters_eigengap(x_norm, max_k=6)
        print(f"  Eigengap suggests k = {optimal_k}")

        if eigenvalues is not None:
            eigvals = eigenvalues.tolist()
        else:
            eigvals = None

        if optimal_k <= 1:
            print("  No sub-structure found (k=1)")
            report["subclasses"][sc_name] = {
                "n": int(n),
                "optimal_k": 1,
                "eigenvalues": eigvals,
                "has_substructure": False
            }
            continue

        # Run spectral clustering with optimal k
        print(f"  Running spectral clustering with k={optimal_k}...")
        sub_labels, k_found = spectral_clustering_subclass(x_norm, max_k=optimal_k)
        print(f"  Found k = {k_found}")

        # Map labels by size (0 = largest)
        counts = Counter(sub_labels)
        order = sorted(counts.keys(), key=lambda k: -counts[k])
        label_map = {old: new for new, old in enumerate(order)}
        sub_labels_mapped = np.array([label_map[l] for l in sub_labels])

        sizes = [int(counts[old]) for old in order]
        print(f"  Sub-subclass sizes: {sizes}")

        report["subclasses"][sc_name] = {
            "n": int(n),
            "optimal_k": optimal_k,
            "k_found": int(k_found),
            "eigenvalues": eigvals,
            "has_substructure": k_found > 1,
            "sub_sizes": sizes,
            "labels": sub_labels_mapped.tolist()  # for downstream physics validation
        }

    # Save
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"\n{'='*50}")
    print(f"Saved: {OUT}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("\nSummary:")
    for sc_name, data in report["subclasses"].items():
        if data.get("skipped"):
            print(f"  {sc_name}: SKIPPED ({data['reason']})")
        elif data.get("has_substructure"):
            print(f"  {sc_name}: k={data['k_found']} (sizes: {data['sub_sizes']})")
        else:
            print(f"  {sc_name}: NO sub-structure (k=1)")


if __name__ == "__main__":
    main()