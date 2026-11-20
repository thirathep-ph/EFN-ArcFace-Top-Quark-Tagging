"""Latent-space (PCA 2D) visualization of physics evolution by distance to class prototype.

Reads the per-class NPZ outputs of compute_prototype_distance_evolution.py:
    experiments/robustness_s16_m05/prototype_distance_evolution_{top,qcd}.npz
and draws, per class: PCA-2D scatter colored by dist_to_own, class-prototype
proxy marker, and quiver arrows showing mean displacement between quantile
bins.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Apply viridis-based HEP theme
from arcefn.utils.figure_style import apply_style, COLORS, GRID_KWARGS
apply_style()

OUT_DIR = Path("experiments/robustness_s16_m05")
OUT_PLOTS = OUT_DIR / "plots"


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def plot_class(tag: str) -> None:
    d = np.load(OUT_DIR / f"prototype_distance_evolution_{tag}.npz")
    X = d["X_norm"]
    dist = d["dist_to_own"]
    bin_idx = d["bin_idx"]
    n_bins = len(d["bin_edges"]) - 1

    mass = d["mass"]
    tau32 = d["tau32"]

    pts = pca2(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.scatter(pts[:, 0], pts[:, 1], c=dist, s=8, alpha=0.5, cmap="viridis")

    # Bin mean path (solid, connecting bin means) + per-bin physics labels
    bin_pts = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 10:
            continue
        bin_pts.append((b, pts[m].mean(axis=0), m))
    if len(bin_pts) > 1:
        bs = np.array([p[1] for p in bin_pts])
        ax.plot(bs[:, 0], bs[:, 1], "->", color="black", lw=1.8, ms=5, alpha=0.9)
        for b, p, m in bin_pts:
            mm = mass[m].mean()
            tt = tau32[m].mean()
            ax.annotate(f"b{b}\n  m={mm:.0f}\n  Tau32={tt:.2f}",
                        (p[0], p[1]), textcoords="offset points", xytext=(6, 8),
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # Prototype proxy: mean of the bin with smallest mean dist_to_own
    best_b = min(range(n_bins), key=lambda b: dist[bin_idx == b].mean())
    proto = pts[bin_idx == best_b].mean(axis=0)
    ax.scatter(*proto, marker="*", s=400, color="red", edgecolors="black", zorder=5)
    ax.annotate("Prototype", proto, textcoords="offset points", xytext=(0, 12),
                fontsize=9, ha="center", color="red")

    # Quiver: mean displacement per consecutive bin pair
    for b in range(n_bins - 1):
        m0, m1 = bin_idx == b, bin_idx == b + 1
        if not (m0.any() and m1.any()):
            continue
        p0 = pts[m0].mean(axis=0)
        p1 = pts[m1].mean(axis=0)
        v = p1 - p0
        ax.quiver(p0[0], p0[1], v[0], v[1], angles="xy", scale_units="xy",
                  scale=1, color="black", alpha=0.7, width=0.003)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Latent space evolution — {tag.upper()}")
    fig.colorbar(im, label="dist to own prototype [rad]")
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    out = OUT_PLOTS / f"prototype_distance_evolution_latent_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_plane(tag):
    """Physics-feature plane per class: mass vs tau32 scatter colored by dist_to_own,
    with per-quantile-bin mean path + bin labels."""
    d = np.load(OUT_DIR / f"prototype_distance_evolution_{tag}.npz")
    mass = d["mass"]
    tau32 = d["tau32"]
    dist = d["dist_to_own"]
    bin_idx = d["bin_idx"]
    n_bins = len(d["bin_edges"]) - 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(tau32, mass, c=dist, s=5, alpha=0.5, cmap="viridis")
    fig.colorbar(sc, label="dist to own prototype [rad]")

    pts = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() >= 10:
            pts.append((b, float(tau32[m].mean()), float(mass[m].mean())))
    if len(pts) > 1:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, "->", color="black", lw=1.5, ms=6)
        for b, t, m_ in pts:
            ax.annotate(f"b{b}", (t, m_), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, color="black")

    ax.set_xlabel("Tau32")
    ax.set_ylabel("Jet mass [GeV]")
    ax.set_title(f"Physics-feature plane — {tag.upper()}")
    out = OUT_DIR / "plots" / f"prototype_distance_evolution_feature_plane_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for tag in ["top", "qcd"]:
    plot_class(tag)
    plot_feature_plane(tag)
print("DONE")
