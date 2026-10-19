import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from arcefn.utils.figure_style import apply_style, add_grid, save_fig, COLORS

inp = pathlib.Path("experiments/dim_3way_50k.json")
dat = json.loads(inp.read_text())

heads = ["ArcFace", "CosLinear", "Linear"]
labels = [r"ArcFace ($S^{63},\,m=0.5$)", r"CosLinear ($S^{63},\,m=0$)", r"Linear (Euclid.)"]
styles = [dict(color=COLORS["top"], ls="-", marker="o"),
          dict(color=COLORS["gray"], ls="--", marker="s"),
          dict(color=COLORS["gray"], ls=":", marker="^")]

apply_style()
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
fig.subplots_adjust(wspace=0.14, left=0.085, right=0.985, top=0.87, bottom=0.19)

x = np.arange(1, 7)
for i, (ax, h, lab) in enumerate(zip(axes, heads, labels)):
    cum = np.cumsum(dat[h]["pca_top6"]) * 100
    st = styles[i]
    ax.plot(x, cum, lw=1.8, ms=5.5, mec="black", mew=0.6, **st, zorder=3)
    ax.plot([2], [cum[1]], marker=st["marker"], ms=10, mfc=st["color"],
            mec="black", mew=0.8, ls="none", zorder=4)
    ax.annotate(f"2 PCs: {cum[1]:.1f}%", xy=(2, cum[1]),
                xytext=(0.06, 0.05), textcoords="axes fraction",
                ha="left", va="bottom", fontsize=9)
    ax.set_xlim(0.8, 6.2)
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    add_grid(ax, axis="y")
    ax.set_title(f"({chr(97+i)}) {lab}", fontsize=9.5, pad=5)

axes[1].set_xlabel("Principal component rank", fontsize=10)
axes[0].set_ylabel("Cumulative variance (%)", fontsize=10)
# no legend: every panel is titled with its head; ArcFace is the solid red subject curve

out = pathlib.Path("paper/figures/dimensionality_3way.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved to {out} {out.stat().st_size} bytes")
plt.close(fig)
