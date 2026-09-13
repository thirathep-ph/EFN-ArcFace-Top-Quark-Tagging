"""Regenerate subclass_physics_distributions.png on the PREDICTED partition.

Layout mirrors the original figure: rows = mass, tau32, sqrt(d12);
cols = (filled) discovered subclasses per predicted class, TN-vs-FP
(QCD status split), TP-vs-FN (top status split). All from cached
artifacts (embeddings.npz + rf_cache_test.npz); no model forward pass.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import (apply_style, save_fig, panel_label,
                                         add_grid, SUBCLASS_COLORS, LEGEND_KWARGS)
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

apply_style()
np.random.seed(42)

npz = np.load(EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz")
E, y = npz["embeddings"], npz["labels"]
feat = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"]
assert len(E) == len(y) == len(feat) == 404000

preds = arcface_preds_from_centers(E, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
check_cm(preds, y, "physics_fig")
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

# predicted subclasses, larger cluster -> Core
sub = np.full(len(y), -1, dtype=int)
info = {}
for cls, offset in [(0, 0), (1, 2)]:
    mask = preds == cls
    sl, k = spectral_clustering_subclass(En[mask], max_k=5)
    cnt = Counter(sl)
    order = sorted(cnt.keys(), key=lambda c: -cnt[c])
    mapped = np.array([order.index(v) for v in sl])
    sub[mask] = mapped + offset
    info[cls] = {"k": int(k), "counts": [int((mapped == i).sum()) for i in range(2)]}
print("partition:", info)

tp = (preds == 1) & (y == 1)
fp = (preds == 1) & (y == 0)
tn = (preds == 0) & (y == 0)
fn = (preds == 0) & (y == 1)

mass, tau32 = feat[:, 0], feat[:, 7]
d12 = np.sqrt(np.clip(feat[:, 4], 0, None))

fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex=False)
rows = [("Mass [GeV]", mass, np.linspace(0, 300, 60)),
        ("Tau32", tau32, np.linspace(0.2, 1.0, 60)),
        ("sqrt(d12)", d12, np.linspace(0, 1.2, 60))]
cols = [("Top: Core vs Edge", [(sub == 2, "Top-Core", SUBCLASS_COLORS["Top core"]),
                               (sub == 3, "Top-Edge", SUBCLASS_COLORS["Top edge"])]),
        ("QCD: TN vs FP", [(tn, "TN (true QCD)", SUBCLASS_COLORS["QCD core"]),
                           (fp, "FP (QCD as top)", SUBCLASS_COLORS["Top core"])]),
        ("Top: TP vs FN", [(tp, "TP (true top)", SUBCLASS_COLORS["Top core"]),
                           (fn, "FN (top as QCD)", SUBCLASS_COLORS["QCD core"])])]

for r, (ylabel, vals, bins) in enumerate(rows):
    for c, (ctitle, groups) in enumerate(cols):
        ax = axes[r, c]
        for mask, lab, col in groups:
            v = vals[mask]
            ax.hist(v, bins=bins, density=True, histtype="step",
                    linewidth=1.4, color=col, label=f"{lab} (n={mask.sum():,})")
        panel_label(ax, "(%s)" % chr(97 + r * 3 + c))
        if r == 0:
            ax.set_title(ctitle, fontsize=9)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if r == 2:
            ax.set_xlabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        add_grid(ax)
        if r == 0:
            ax.legend(loc="upper right",
                      **{**LEGEND_KWARGS, "fontsize": 6})

plt.tight_layout(pad=0.6, w_pad=1.0)
save_fig(fig, "subclass_physics_distributions.png")
print("saved subclass_physics_distributions.png")
plt.close(fig)
