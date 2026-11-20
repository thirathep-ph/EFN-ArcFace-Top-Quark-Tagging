#!/usr/bin/env python
"""Plot T/F clustering investigation results.

Generates figures showing:
1. PCA 2-D scatter colored by T (correct) / F (incorrect)
2. Distance-to-prototypes scatter colored by T/F (correct prototype labels)
3. |margin| distribution histograms for T vs F — key separation metric
4. 2-D histogram of margins overlaid

Outputs to paper/figures/review/ (never overwrites originals).

Prototype mapping (verified from numbers): centers[0] = QCD, centers[1] = Top.
margin = dist_to_QCD - dist_to_Top
  margin > 0  => closer to QCD prototype
  margin < 0  => closer to Top prototype
T (correct): margin sign matches true class; F (incorrect): sign mismatches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.figure_style import apply_style
apply_style()
from sklearn.decomposition import PCA

from arcefn.utils.paths import EXPERIMENTS

NPZ = EXPERIMENTS / "robustness_s16_m05" / "tf_clustering.npz"
JSON = EXPERIMENTS / "robustness_s16_m05" / "tf_clustering.json"
OUT_DIR = Path("D:/research/ArcEFN-paper/paper/figures/review")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe palette: viridis-based (T = deep purple, F = yellow-green)
# Avoids red-green pair which is inaccessible to ~8% of males
COLOR_T = "#430153"  # viridis start (deep purple) = correct
COLOR_F = "#fde725"  # viridis end (yellow-green) = incorrect


def main() -> None:
    d = np.load(NPZ)
    X_norm = d["X_norm"]       # (404000, 64)
    y_true = d["labels"]       # 0=QCD, 1=Top
    preds = d["preds"]
    X_pca = d["X_pca"]         # (404000, 2)
    centers = d["centers"]     # (2, 64): centers[0]=QCD, centers[1]=Top

    correct = preds == y_true
    T_mask = correct
    F_mask = ~correct

    n_T = int(T_mask.sum())
    n_F = int(F_mask.sum())
    acc = n_T / (n_T + n_F)

    # Distances to each prototype
    dist_to_qcd = np.linalg.norm(X_norm - centers[0], axis=1)   # centers[0] = QCD
    dist_to_top = np.linalg.norm(X_norm - centers[1], axis=1)   # centers[1] = Top
    margin = dist_to_qcd - dist_to_top                          # >0 => closer to QCD

    margin_T = margin[T_mask]
    margin_F = margin[F_mask]
    abs_margin_T = np.abs(margin_T)
    abs_margin_F = np.abs(margin_F)

    # Confusion matrix
    cm_qcd_qcd = int(((y_true == 0) & (preds == 0)).sum())
    cm_qcd_top = int(((y_true == 0) & (preds == 1)).sum())
    cm_top_qcd = int(((y_true == 1) & (preds == 0)).sum())
    cm_top_top = int(((y_true == 1) & (preds == 1)).sum())

    # Mean |margin| comparison
    mean_abs_T = float(abs_margin_T.mean())
    mean_abs_F = float(abs_margin_F.mean())
    med_abs_T = float(np.median(abs_margin_T))
    med_abs_F = float(np.median(abs_margin_F))

    # Fraction of F with |margin| below T's median
    f_below_t_med = int((abs_margin_F < med_abs_T).sum())
    t_above_f_med = int((abs_margin_T > med_abs_F).sum())

    # --- Figure 1: PCA 2-D colored by T/F ---
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(X_pca[F_mask, 0], X_pca[F_mask, 1], c=COLOR_F, s=1, alpha=0.25,
               label=f"F (incorrect, n={n_F:,})")
    ax.scatter(X_pca[T_mask, 0], X_pca[T_mask, 1], c=COLOR_T, s=1, alpha=0.25,
               label=f"T (correct, n={n_T:,})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7,
               label="PC1 = 0 (approx. boundary)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA 2-D: T={n_T:,} (green) vs F={n_F:,} (red), acc={acc:.4f}")
    ax.legend(loc="best", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tf_pca_2d.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Distance-to-prototypes scatter ---
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(dist_to_qcd[F_mask], dist_to_top[F_mask], c=COLOR_F, s=1, alpha=0.25,
               label=f"F (incorrect, n={n_F:,})")
    ax.scatter(dist_to_qcd[T_mask], dist_to_top[T_mask], c=COLOR_T, s=1, alpha=0.25,
               label=f"T (correct, n={n_T:,})")
    lo = min(dist_to_qcd.min(), dist_to_top.min())
    hi = max(dist_to_qcd.max(), dist_to_top.max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, alpha=0.7,
            label="equidistant (margin = 0)")
    ax.set_xlabel("Distance to QCD prototype (center[0])")
    ax.set_ylabel("Distance to Top prototype (center[1])")
    ax.set_title("Distance to each prototype (colored by T/F)")
    ax.legend(loc="best", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tf_distance_scatter.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: |margin| histogram T vs F (KEY separation metric) ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(0, max(abs_margin_T.max(), abs_margin_F.max()), 60)
    axes[0].hist(abs_margin_T, bins=bins, color=COLOR_T, alpha=0.7,
                 label=f"T (correct, n={n_T:,})")
    axes[0].axvline(med_abs_T, color=COLOR_T, linestyle="--", linewidth=1.5,
                    label=f"median={med_abs_T:.3f}")
    axes[0].set_xlabel("|margin| = |dist_QCD − dist_Top|")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"T: mean={mean_abs_T:.4f}, median={med_abs_T:.4f}")
    axes[0].legend(fontsize=8)
    axes[1].hist(abs_margin_F, bins=bins, color=COLOR_F, alpha=0.7,
                 label=f"F (incorrect, n={n_F:,})")
    axes[1].axvline(med_abs_F, color=COLOR_F, linestyle="--", linewidth=1.5,
                    label=f"median={med_abs_F:.3f}")
    axes[1].set_xlabel("|margin| = |dist_QCD − dist_Top|")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"F: mean={mean_abs_F:.4f}, median={med_abs_F:.4f}")
    axes[1].legend(fontsize=8)
    fig.suptitle("|margin| distribution: T vs F (larger = more confident)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT_DIR / "tf_abs_margin_hist.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: Overlaid |margin| density ---
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hist(abs_margin_T, bins=bins, color=COLOR_T, alpha=0.5, label=f"T (correct)", density=True)
    ax.hist(abs_margin_F, bins=bins, color=COLOR_F, alpha=0.5, label=f"F (incorrect)", density=True)
    ax.axvline(med_abs_T, color=COLOR_T, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"T median={med_abs_T:.3f}")
    ax.axvline(med_abs_F, color=COLOR_F, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"F median={med_abs_F:.3f}")
    ax.set_xlabel("|margin| = |dist_QCD − dist_Top|")
    ax.set_ylabel("Density")
    ax.set_title("Overlaid |margin| density: T vs F")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tf_abs_margin_overlay.png", dpi=150)
    plt.close(fig)

    # --- Print summary ---
    print("=== T/F MARGIN SUMMARY (corrected labels) ===")
    print(f"Confusion matrix: QCD->QCD={cm_qcd_qcd:,}, QCD->Top={cm_qcd_top:,}, "
          f"Top->QCD={cm_top_qcd:,}, Top->Top={cm_top_top:,}")
    print(f"T (correct): n={n_T:,}, mean |margin|={mean_abs_T:.4f}, median |margin|={med_abs_T:.4f}")
    print(f"F (incorrect): n={n_F:,}, mean |margin|={mean_abs_F:.4f}, median |margin|={med_abs_F:.4f}")
    print(f"F with |margin| < T median: {f_below_t_med}/{n_F} ({100*f_below_t_med/n_F:.1f}%)")
    print(f"T with |margin| > F median: {t_above_f_med}/{n_T} ({100*t_above_f_med/n_T:.1f}%)")
    print(f"\nKey insight: T/F CAN overlap on 1D decision axis projection.")
    print(f"But in full 64-D space, correctness = sign of margin = nearest-prototype.")
    print(f"|margin| separates confident correct from confident incorrect.")
    print(f"\nFigures saved to: {OUT_DIR}")
    for f in ["tf_pca_2d.png", "tf_distance_scatter.png", "tf_abs_margin_hist.png", "tf_abs_margin_overlay.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
