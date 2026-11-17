#!/usr/bin/env python
"""Leiden Clustering on k-NN Graph with Resolution Sweep

Builds k-NN graph on embeddings and runs Leiden at multiple resolutions
to find natural clustering scales. Compatible with physics validation.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import igraph as ig
    import leidenalg
except ImportError:
    print("Installing igraph and leidenalg...")
    import subprocess
    subprocess.check_call(["pip", "install", "python-igraph", "leidenalg"])
    import igraph as ig
    import leidenalg

from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT = EXPERIMENTS / "robustness_s16_m05" / "leiden_clustering.json"

# Resolution sweep
RESOLUTIONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
K_NN = 30  # k-NN graph parameter


def build_knn_graph(x_norm: np.ndarray, k: int = K_NN) -> ig.Graph:
    """Build k-NN graph from L2-normalized embeddings using cosine similarity."""
    n = len(x_norm)
    # Cosine similarity = dot product on L2-normalized vectors
    # Use sklearn's NearestNeighbors for efficiency
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=-1).fit(x_norm)
    distances, indices = nbrs.kneighbors(x_norm)
    
    # Build edge list (skip self-loops)
    edges = []
    weights = []
    for i in range(n):
        for j_idx in range(1, k+1):  # skip 0 (self)
            j = indices[i, j_idx]
            if i < j:  # undirected, add once
                # Convert cosine distance to similarity
                w = 1.0 - distances[i, j_idx]
                if w > 0:
                    edges.append((i, j))
                    weights.append(float(w))
    
    g = ig.Graph()
    g.add_vertices(n)
    g.add_edges(edges)
    g.es["weight"] = weights
    return g


def leiden_at_resolution(g: ig.Graph, resolution: float, seed: int = 42) -> np.ndarray:
    """Run Leiden at given resolution, return cluster labels."""
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed
    )
    return np.array(partition.membership)


def main() -> None:
    t0 = time.time()
    print("Loading embeddings...")
    emb_np = np.load(EMBEDDINGS)
    embeddings = emb_np["embeddings"]
    labels = emb_np["labels"]
    print(f"  {len(embeddings):,} events")

    # Get primary subclass labels (same as before)
    print("Getting primary subclass partition...")
    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, "leiden_clustering")

    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = preds == cls
        x = embeddings[mask]
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        # Use spectral clustering for primary partition (consistent with paper)
        from arcefn.utils.clustering import spectral_clustering_subclass
        sc_labels, _ = spectral_clustering_subclass(x_norm, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda k: -counts[k])
        label_map = {order[0]: 0, order[1]: 1} if len(order) >= 2 else {order[0]: 0}
        sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
        subclass_labels[mask] = sc_labels + (2 * cls)

    SUBCLASS_NAMES = {0: "QCD_Dense", 1: "QCD_Dilute", 2: "Top_Dilute", 3: "Top_Dense"}

    # Run Leiden on EACH primary subclass separately (more meaningful)
    report = {
        "n_events": len(embeddings),
        "primary_partition": "predicted_labels_primary",
        "k_nn": K_NN,
        "resolutions": RESOLUTIONS,
        "subclasses": {}
    }

    for sc_id, sc_name in SUBCLASS_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Leiden on {sc_name}...")
        mask = subclass_labels == sc_id
        x = embeddings[mask]
        n = len(x)
        print(f"  n = {n:,}")

        if n < 100:
            print("  Too few events, skipping")
            continue

        # L2-normalize
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)

        # Build k-NN graph
        print(f"  Building k-NN graph (k={K_NN})...")
        g = build_knn_graph(x_norm, k=K_NN)
        print(f"  Graph: {g.vcount()} vertices, {g.ecount()} edges")

        # Run Leiden at each resolution
        sc_results = {}
        for res in RESOLUTIONS:
            print(f"  Resolution γ={res}...")
            leiden_labels = leiden_at_resolution(g, res)
            n_clusters = len(np.unique(leiden_labels))
            counts = Counter(leiden_labels)
            sizes = sorted(counts.values(), reverse=True)
            print(f"    k={n_clusters}, sizes: {sizes[:10]}{'...' if len(sizes)>10 else ''}")
            
            sc_results[f"gamma_{res}"] = {
                "n_clusters": int(n_clusters),
                "sizes": sizes,
                "labels": leiden_labels.tolist()
            }

        report["subclasses"][sc_name] = {
            "n": int(n),
            "leiden_results": sc_results
        }

    # Save
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"\n{'='*50}")
    print(f"Saved: {OUT}")
    print(f"Total time: {time.time() - t0:.0f}s")

    # Summary
    print("\nSummary:")
    for sc_name, data in report["subclasses"].items():
        print(f"  {sc_name} (n={data['n']:,}):")
        for gamma_key, res_data in data["leiden_results"].items():
            print(f"    {gamma_key}: k={res_data['n_clusters']}, sizes={res_data['sizes'][:5]}")


if __name__ == "__main__":
    main()