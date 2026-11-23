"""Prototype-distance evolution analysis.

Groups events by distance from their predicted class prototype (ArcFace
class center) and tracks how physics observables evolve from prototype
core → decision boundary. Two passes: per-class (top/qcd), 5 quantile bins
of distance-to-own-prototype (equal-frequency).

Usage:
    python scripts/analysis/compute_prototype_distance_evolution.py --max-events 404000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# NOTE: arcefn imports (which load awkward) must come before matplotlib —
# importing pyarrow first crashes the awkward native import in this venv.
from arcefn.data.loader import load_awkward
from arcefn.utils.physics import load_cached_features
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
TEST_H5 = EXPERIMENTS.parent / "data" / "top_tagging" / "test.h5"
OUT_DIR = EXPERIMENTS / "robustness_s16_m05"
OUT_JSON = OUT_DIR / "prototype_distance_evolution.json"
OUT_NPZ = OUT_DIR / "prototype_distance_evolution.npz"

# Feature indices from compute_features() in arcefn/utils/physics.py
# [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
IDX_MASS = 0
IDX_TAU32 = 7
IDX_ZG = 8
IDX_THETA_G = 9
IDX_NCONS = 2  # Mult

N_BINS = 5
RANDOM_STATE = 42

# CERN/HEP-style palette
CERN_BLUE = "#0053a6"
CERN_RED = "#c8171f"
CLASS_COLORS = {0: CERN_BLUE, 1: CERN_RED}
CLASS_TAGS = {0: "qcd", 1: "top"}


def angular_distance(X_norm: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Cosine angular distance on unit sphere: arccos(cos(theta)) in radians."""
    cos_sim = np.clip(X_norm @ centers.T, -1.0, 1.0)
    return np.arccos(cos_sim)  # shape (n_events, n_centers)


def bin_by_normalized_distance(dist_to_own: np.ndarray, n_bins: int = N_BINS):
    """Bin events by quantile of distance to own prototype (equal-frequency per bin).
    Returns bin edges and bin indices for each event.
    """
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(dist_to_own, q)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([dist_to_own.min(), dist_to_own.max()])
    elif len(edges) < n_bins + 1:
        edges = np.linspace(dist_to_own.min(), dist_to_own.max(), n_bins + 1)
    bin_idx = np.digitize(dist_to_own, edges) - 1
    bin_idx = np.clip(bin_idx, 0, max(len(edges) - 2, 0))
    return edges, bin_idx


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=100000,
                        help="Max events to load from embeddings + test.h5 (default: 100000)")
    args = parser.parse_args()
    max_events = args.max_events

    # 1. Load embeddings + labels (subset)
    d = np.load(EMBEDDINGS)
    X = d["embeddings"][:max_events]
    y_true = d["labels"][:max_events]
    print(f"Loaded embeddings: {X.shape}, labels: {y_true.shape} (max_events={max_events})")

    # 2. Predictions from checkpoint (margin-free inference)
    preds = arcface_preds_from_centers(X, CHECKPOINT)
    correct = (preds == y_true)

    # 3. Normalize embeddings to unit sphere
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    # 4. Load class centers
    import torch
    sd = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    key = "arcface_head.class_centers" if "arcface_head.class_centers" in sd else "class_centers"
    centers = sd[key].detach().cpu().numpy()
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
    # centers[0] = QCD, centers[1] = Top (verified)

    # 5. Angular distances to both prototypes
    dist = angular_distance(X_norm, centers)  # (n_events, 2)
    dist_to_qcd = dist[:, 0]
    dist_to_top = dist[:, 1]

    # Per-event: distance to OWN predicted center, distance to OTHER center, margin
    pred_top = (preds == 1)
    pred_qcd = (preds == 0)

    dist_to_own = np.where(pred_top, dist_to_top, dist_to_qcd)
    dist_to_other = np.where(pred_top, dist_to_qcd, dist_to_top)
    margin = dist_to_other - dist_to_own  # >0 => correctly classified side

    # 6. Load physics features from test.h5 (same event order as embeddings.npz)
    print("\nLoading test.h5 features...")
    events, labels_h5, _ = load_awkward(str(TEST_H5), max_events=max_events, lazy=False)
    features = load_cached_features(events)
    mass_all = features[:, IDX_MASS].astype(np.float64)
    tau32_all = features[:, IDX_TAU32].astype(np.float64)
    zg_all = features[:, IDX_ZG].astype(np.float64)
    theta_g_all = features[:, IDX_THETA_G].astype(np.float64)
    ncons_all = features[:, IDX_NCONS].astype(np.float64)
    print(f"Features: {features.shape}")

    def stats(arr, mask):
        return {
            "mean": float(np.mean(arr[mask])),
            "median": float(np.median(arr[mask])),
            "std": float(np.std(arr[mask])),
            "min": float(np.min(arr[mask])),
            "max": float(np.max(arr[mask])),
        }

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    plot_data = {}
    for cls, tag in [(1, "top"), (0, "qcd")]:
        print(f"\n=== CLASS: {tag.upper()} ===")
        mask = y_true == cls
        n_class = int(mask.sum())
        print(f"Events: {n_class:,}")

        # 7. Bin by quantile distance to own prototype (per class)
        d_own_c = dist_to_own[mask]
        edges, bin_idx_c = bin_by_normalized_distance(d_own_c, n_bins=N_BINS)
        n_bins_actual = len(edges) - 1
        print(f"  bins: {n_bins_actual}, edges: {np.round(edges, 3)}")

        # 8. Compute stats per bin (per class)
        bin_stats = []
        for b in range(n_bins_actual):
            bm = bin_idx_c == b
            n = int(bm.sum())
            if n < 10:
                bin_stats.append({"bin": b, "n": n, "skipped": True})
                continue
            # Mask back to full arrays for stats on the class subset
            full_mask = np.where(mask)[0][bm]
            frac_correct = float(correct[full_mask].mean())
            bin_stats.append({
                "bin": b,
                "n": n,
                "skipped": False,
                "dist_to_own": stats(dist_to_own, full_mask),
                "dist_to_other": stats(dist_to_other, full_mask),
                "margin": stats(margin, full_mask),
                "frac_correct": frac_correct,
                "mass": stats(mass_all, full_mask),
                "tau32": stats(tau32_all, full_mask),
                "zg": stats(zg_all, full_mask),
                "theta_g": stats(theta_g_all, full_mask),
                "n_constituents": stats(ncons_all, full_mask),
            })

        # 9. Correlations (class subset)
        correlations = {
            "dist_to_own_vs_mass": pearson(d_own_c, mass_all[mask]),
            "dist_to_own_vs_tau32": pearson(d_own_c, tau32_all[mask]),
            "dist_to_own_vs_zg": pearson(d_own_c, zg_all[mask]),
            "dist_to_own_vs_theta_g": pearson(d_own_c, theta_g_all[mask]),
            "dist_to_own_vs_n_cons": pearson(d_own_c, ncons_all[mask]),
            "margin_vs_mass": pearson(margin[mask], mass_all[mask]),
            "margin_vs_tau32": pearson(margin[mask], tau32_all[mask]),
            "margin_vs_zg": pearson(margin[mask], zg_all[mask]),
            "margin_vs_theta_g": pearson(margin[mask], theta_g_all[mask]),
        }

        results = {
            "meta": {
                "total_events": int(max_events),
                "class": cls,
                "class_name": tag,
                "n_events_class": n_class,
                "n_bins": n_bins_actual,
                "accuracy": float(correct.mean()),
                "centers_0_is_qcd": True,
                "centers_1_is_top": True,
                "runtime_seconds": time.time() - t0,
            },
            "bin_edges": [float(e) for e in edges],
            "bins": bin_stats,
            "correlations": correlations,
            "summary": {
                "dist_to_own_mean": float(d_own_c.mean()),
                "dist_to_own_std": float(d_own_c.std()),
                "margin_mean": float(margin[mask].mean()),
                "margin_std": float(margin[mask].std()),
                "correct_fraction": float(correct[mask].mean()),
            },
        }

        out_json_c = OUT_DIR / f"prototype_distance_evolution_{tag}.json"
        out_npz_c = OUT_DIR / f"prototype_distance_evolution_{tag}.npz"
        with open(out_json_c, "w") as f:
            json.dump(results, f, indent=2)
        np.savez(out_npz_c,
                 X_norm=X_norm[mask], labels=y_true[mask], preds=preds[mask],
                 correct=correct[mask], dist_to_own=d_own_c,
                 dist_to_other=dist_to_other[mask], margin=margin[mask],
                 mass=mass_all[mask], tau32=tau32_all[mask], zg=zg_all[mask],
                 theta_g=theta_g_all[mask], n_constituents=ncons_all[mask],
                 bin_edges=edges, bin_idx=bin_idx_c)
        print(f"Saved: {out_json_c}")
        print(f"Saved: {out_npz_c}")

        # 11. Print summary
        print(f"Accuracy (class {tag}): {correct[mask].mean():.4f}")
        print(f"Dist to own prototype: mean={d_own_c.mean():.4f} rad, std={d_own_c.std():.4f}")
        print(f"Margin: mean={margin[mask].mean():.4f} rad, std={margin[mask].std():.4f}")
        print("Bin stats (bin, n, dist_own_mean, mass_mean, tau32_mean, frac_correct):")
        for bs in bin_stats:
            if bs.get("skipped"):
                continue
            print(f"  bin={bs['bin']}  n={bs['n']:6d}  "
                  f"dist_own={bs['dist_to_own']['mean']:.4f}  "
                  f"mass={bs['mass']['mean']:7.1f}  "
                  f"tau32={bs['tau32']['mean']:.3f}  "
                  f"zg={bs['zg']['mean']:.3f}  "
                  f"frac_correct={bs['frac_correct']:.3f}")
        print("Correlations (dist_to_own vs observable):")
        for k, v in correlations.items():
            print(f"  {k}: {v:+.4f}")

        plot_data[tag] = {
            "edges": edges,
            "bins": bin_stats,
            "d_own": d_own_c,
            "mass": mass_all[mask],
            "tau32": tau32_all[mask],
        }

    # 12. CERN/HEP-style plots (two panels per class: distance vs mass evolution)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from arcefn.utils.figure_style import apply_style
    apply_style()
    OUT_PLOTS = OUT_DIR / "plots"
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    for tag in ["top", "qcd"]:
        pd_c = plot_data[tag]
        edges = pd_c["edges"]
        bins = [b for b in pd_c["bins"] if not b.get("skipped")]
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        x = np.array([(edges[b["bin"] + 1] + edges[b["bin"]]) / 2 for b in bins])
        # Panel 1: mean distance to own prototype
        axes[0].plot(x, [b["dist_to_own"]["mean"] for b in bins], marker="o",
                     color=CLASS_COLORS[1 if tag == "top" else 0], lw=2, ms=4)
        axes[0].set_ylabel("Mean dist to own prototype [rad]")
        axes[0].set_title(f"Prototype-distance evolution: {tag.upper()}")
        # Panel 2: mean jet mass
        axes[1].plot(x, [b["mass"]["mean"] for b in bins], marker="s",
                     color=CLASS_COLORS[1 if tag == "top" else 0], lw=2, ms=4)
        axes[1].set_xlabel("Quantile of distance to own prototype")
        axes[1].set_ylabel("Mean jet mass [GeV]")
        fig.tight_layout()
        out_png = OUT_PLOTS / f"prototype_distance_evolution_{tag}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_png}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
