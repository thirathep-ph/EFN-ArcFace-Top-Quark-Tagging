"""
Generate a publication-quality 3D visualization of the ArcFace hypersphere.

Projects the 64-dim unit-hypersphere embeddings onto S^2 via uncentered SVD
(top-3 singular vectors), then re-normalizes to the unit sphere. The data
collapses to a great circle S^1, confirming effective dimension ~ 2.

Panel (a): embeddings colored by jet class (QCD / top).
Panel (b): embeddings colored by spectral subclass (core / edge per class).
Panel (c): singular-value spectrum of the 64-dim embedding matrix.

Saves: ``EXPERIMENTS / 'hypersphere_3d.png'``
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from arcefn.utils.device import get_device
from arcefn.utils.paths import EXPERIMENTS

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
SUBCLASS_LABELS = EXPERIMENTS / "physics_sr" / "subclass_labels.npz"
N_POINTS = 50000
SEED = 42

SUBCLASS_NAMES = {0: "QCD core", 1: "QCD edge", 2: "Top core", 3: "Top edge"}
SUBCLASS_COLORS = {0: "#377eb8", 1: "#9ecae1", 2: "#e41a1c", 3: "#fdae6b"}
CLASS_COLORS = {0: "#377eb8", 1: "#e41a1c"}
CLASS_NAMES = {0: "QCD", 1: "Top"}


def draw_sphere(ax, r=1.0, alpha=0.18, color="0.45"):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.35,
                      rstride=4, cstride=4)


def slerp(p0, p1, n=300):
    t = np.linspace(0, 1, n)[:, None]
    om = np.arccos(np.clip(p0 @ p1, -1.0, 1.0))
    return (np.sin((1 - t) * om) * p0 + np.sin(t * om) * p1) / np.sin(om)


def panel_title(ax, letter, title):
    ax.set_title(r"$\mathbf{(%s)}$ %s" % (letter, title), loc="left",
                 fontsize=10.5, pad=6)


def main():
    emb_data = np.load(str(EMBEDDINGS))
    emb = emb_data["embeddings"]
    labels = emb_data["labels"]
    sub_cls = np.load(str(SUBCLASS_LABELS))
    sub = sub_cls["sub"] + 2 * sub_cls["cls"]

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    device, _ = get_device()
    sd = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    if "arcface_head.class_centers" not in sd:
        sd = sd.get("model_state_dict", sd)
    centers = sd["arcface_head.class_centers"].cpu().numpy()
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10

    rng = np.random.default_rng(SEED)
    n_per_class = N_POINTS // 2
    idx = []
    for cls in [0, 1]:
        cls_idx = np.where(labels == cls)[0]
        idx.append(rng.choice(cls_idx, n_per_class, replace=False))
    idx = np.concatenate(idx)
    idx.sort()

    sv_full = np.linalg.svd(emb[idx], full_matrices=False, compute_uv=False)
    sv_ratio = sv_full[:6] / sv_full[0]
    var_frac = np.cumsum(sv_full**2) / np.sum(sv_full**2)

    _, _, vt = np.linalg.svd(emb[idx], full_matrices=False)
    v3 = vt[:3].T
    pts = emb[idx] @ v3
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-10
    c3 = centers @ v3
    c3 /= np.linalg.norm(c3, axis=1, keepdims=True) + 1e-10

    lab = labels[idx]
    sub_ = sub[idx]

    geo_rad = np.arccos(np.clip(c3[0] @ c3[1], -1.0, 1.0))
    geo_deg = np.degrees(geo_rad)

    arc = slerp(c3[0], c3[1])

    plt.rcParams.update({"font.family": "DejaVu Serif",
                         "font.size": 10,
                         "axes.linewidth": 0.6})

    fig = plt.figure(figsize=(14, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.7],
                          wspace=0.04, left=0.01, right=0.99,
                          top=0.92, bottom=0.02)
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1], projection="3d")
    ax3 = fig.add_subplot(gs[2])

    for ax in (ax1, ax2):
        draw_sphere(ax)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=40, azim=-53)
        ax.set_axis_off()

    proto_handle = Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="k", markersize=6,
                          label="Prototypes")

    for cls, color in CLASS_COLORS.items():
        m = lab == cls
        ax1.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=1, alpha=0.4,
                    c=color, rasterized=True)
    ax1.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=60, c="k", zorder=6)
    ax1.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="0.25",
             linewidth=0.9, alpha=0.7, linestyle=(0, (4, 3)))

    mid = arc[150]
    ax1.text(mid[0], mid[1], mid[2] + 0.17,
             f"${geo_deg:.1f}^\\circ$", fontsize=9, ha="center", va="bottom")

    cls_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=6,
                          label=CLASS_NAMES[i])
                   for i, c in CLASS_COLORS.items()]
    cls_handles.append(proto_handle)
    ax1.legend(handles=cls_handles, fontsize=8, frameon=True,
               edgecolor="0.45", facecolor="white", framealpha=0.92,
               ncol=3, columnspacing=1.2, handletextpad=0.4,
               borderpad=0.4, loc="upper right",
               borderaxespad=0)
    panel_title(ax1, "a", "QCD and top jets")

    for s in range(4):
        m = sub_ == s
        ax2.scatter(pts[m, 0], pts[m, 1], pts[m, 2], s=1, alpha=0.4,
                    c=SUBCLASS_COLORS[s], rasterized=True)
    ax2.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=60, c="k", zorder=6)
    ax2.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="0.25",
             linewidth=0.9, alpha=0.7, linestyle=(0, (4, 3)))

    sub_handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=SUBCLASS_COLORS[s], markersize=6,
                          label=f"{SUBCLASS_NAMES[s]} ({(sub_==s).sum()/len(sub_)*100:.0f}%)")
                   for s in range(4)]
    sub_handles.append(proto_handle)
    ax2.legend(handles=sub_handles, fontsize=7.5, frameon=True,
               edgecolor="0.45", facecolor="white", framealpha=0.92,
               ncol=3, columnspacing=1.0, handletextpad=0.4,
               borderpad=0.4, loc="upper right",
               borderaxespad=0)
    panel_title(ax2, "b", "Spectral subclasses")

    comps = np.arange(1, 7)
    bars = ax3.bar(comps, sv_ratio[:6], width=0.6,
                   color=["#377eb8", "#377eb8", "#b0b0b0", "#b0b0b0",
                          "#cccccc", "#cccccc"],
                   edgecolor="white", linewidth=0.5)
    ax3.axhline(sv_ratio[1], color="0.6", ls=":", lw=0.8)
    ax3.annotate(f"{sv_ratio[1]:.2f}", xy=(2, sv_ratio[1]),
                 xytext=(4.2, 0.62), fontsize=8, color="0.35",
                 arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax3.set_xlabel("Component", fontsize=9)
    ax3.set_ylabel(r"Singular value / $\sigma_1$", fontsize=9)
    ax3.set_xticks(comps)
    ax3.set_ylim(0, 1.1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    panel_title(ax3, "c", "Singular-value spectrum")

    out = EXPERIMENTS / "hypersphere_3d.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Singular value ratios: {sv_ratio[:6].round(4)}")
    print(f"Top-2 variance fraction: {var_frac[1]*100:.1f}%")
    print(f"Geodesic angle: {geo_deg:.2f}°")
    print(f"Mean embedding norm: {np.linalg.norm(emb[idx], axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
