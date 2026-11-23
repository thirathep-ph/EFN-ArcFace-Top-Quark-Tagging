"""Compute Hopkins statistic + silhouette score on embeddings to prove no discrete clusters.

Hopkins statistic: H ~ 0.5 => data is uniformly distributed (no cluster tendency);
H ~ 0 => data is clustered. Standard cluster tendency test.

Uses sklearn NearestNeighbors for memory-efficient distance queries.

Usage:
    python scripts/analysis/hopkins_silhouette.py [--max-events 50000] [--sample 3000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arcefn.utils.paths import EXPERIMENTS

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
OUT_JSON = EXPERIMENTS / "robustness_s16_m05" / "hopkins_silhouette.json"


def hopkins_statistic(X: np.ndarray, n_sample: int = 3000, seed: int = 42) -> float:
    """Hopkins cluster tendency statistic.

    Returns H in [0, 1]. H ~ 0.5 => uniform/random (no clusters);
    H ~ 0 => strong cluster tendency.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = min(n_sample, n // 10)

    # Random points in the data's bounding box
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    rand_points = rng.uniform(xmin, xmax, size=(m, d))

    # Random data points
    data_idx = rng.choice(n, size=m, replace=False)
    data_points = X[data_idx]

    # Fit nearest-neighbor index on data (cosine distance on normalized vectors)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)

    # u: distance from random points to nearest data point
    u_dist, _ = nn.kneighbors(rand_points, n_neighbors=1, return_distance=True)
    u = u_dist.ravel()

    # w: distance from random data points to nearest *other* data point
    # Use 2 neighbors (self + nearest other), take the second
    w_dist, _ = nn.kneighbors(data_points, n_neighbors=2, return_distance=True)
    w = w_dist[:, 1].ravel()

    u_sum = float(np.sum(u))
    w_sum = float(np.sum(w))
    if u_sum + w_sum == 0:
        return 0.5
    H = u_sum / (u_sum + w_sum)
    return H


def silhouette_sample(X: np.ndarray, labels: np.ndarray, n_sample: int = 3000, seed: int = 42) -> float:
    """Silhouette score on a subsample (full O(n^2) is infeasible at 404k).

    Near 0 => no cluster structure; near 1 => well-separated clusters.
    Uses cluster labels = class labels (top/qcd) as the clustering.
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = len(X)
    m = min(n_sample, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]
    ys = labels[idx]
    return float(silhouette_score(Xs, ys, metric="cosine"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args()

    d = np.load(EMBEDDINGS)
    X = d["embeddings"]
    y = d["labels"]
    if args.max_events is not None:
        X = X[: args.max_events]
        y = y[: args.max_events]
    print(f"Loaded embeddings: {X.shape}, labels: {y.shape}")

    # L2-normalize (embeddings are already normalized in practice, but be safe)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    H = hopkins_statistic(Xn, n_sample=args.sample)
    print(f"Hopkins statistic: {H:.4f} (0.5 = uniform/no clusters, 0 = clustered)")

    try:
        sil = silhouette_sample(Xn, y, n_sample=args.sample)
        print(f"Silhouette score (class labels): {sil:.4f} (near 0 = no cluster structure)")
    except Exception as e:
        print(f"Silhouette failed: {e}")
        sil = None

    result = {
        "hopkins_statistic": H,
        "silhouette_score": sil,
        "n_events": int(len(X)),
        "n_sample": int(args.sample),
        "interpretation": "Hopkins ~0.5 => uniform, no discrete clusters; silhouette near 0 => no cluster structure",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
