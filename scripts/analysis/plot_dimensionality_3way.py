import json, pathlib
import matplotlib.pyplot as plt
import numpy as np

inp = pathlib.Path("experiments/dim_3way_50k.json")
dat = json.loads(inp.read_text())
heads = ["ArcFace", "CosLinear", "Linear"]
labels = [r"ArcFace ($S^{63},m=0.5$)", r"CosLinear ($S^{63},m=0$)", r"Linear (Euclid.)"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
fig.subplots_adjust(wspace=0.25, left=0.08, right=0.97, top=0.88, bottom=0.22)

for ax, h, lab, col in zip(axes, heads, labels, colors):
    pca = dat[h]["pca_top6"]
    x = np.arange(1, 7)
    bars = ax.bar(x, pca, color=col, edgecolor="black", linewidth=0.6, width=0.65)
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("PC rank", fontsize=8)
    ax.set_xticks(x)
    ax.set_ylim(0, 0.90)
    ax.tick_params(labelsize=7)
    # annotate cum2 and metrics
    tw = dat[h]["twonn"]
    pr = dat[h]["pr"]
    cum2 = dat[h]["pca_cum2"]*100
    ax.text(0.98, 0.96, f"TwoNN {tw:.2f}\nPR {pr:.2f}\n2 PCs {cum2:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    # value labels on first 2 bars if significant
    for i, v in enumerate(pca[:2]):
        if v > 0.01:
            ax.text(x[i], v+0.02, f"{v*100:.1f}%", ha="center", fontsize=6)

axes[0].set_ylabel("Variance fraction", fontsize=9)
fig.suptitle("Dimensionality by head — same 50k test, paired (seed 42)", fontsize=10, y=0.99)

# add overall note
fig.text(0.5, 0.02, "Only ArcFace collapses to 2D (>99.9% in 2 PCs); CosLinear 48.8% and Linear 89.9% — margin $m=0.5$ makes 2D", 
         ha="center", fontsize=7, style="italic")

out = pathlib.Path("paper/figures/dimensionality_3way.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved to {out} {out.stat().st_size} bytes")
# also copy to arxiv
import shutil
shutil.copy(out, pathlib.Path("paper_arxiv/figures/dimensionality_3way.png"))
print("copied to arxiv")
plt.close(fig)
