"""Regenerate the subclass composition heatmap with canonical Core/Edge labels.

The paper's headline subclass partition clusters on PREDICTED labels over
the full 404k test set (what the model sees) and maps the largest cluster to
Core, the second to Edge.  Use --partition true to reproduce the GT check.
The composition heatmap must use the SAME partition and naming so its caption
is consistent with Table 2.  This script recomputes the per-subclass breakdown
by prediction status from the saved embeddings and cached class centers
(no forward pass, no UMAP).

Artifacts read:
  * experiments/robustness_s16_m05/embeddings.npz  -> 404k embeddings + labels
  * experiments/robustness_s16_m05/checkpoint.pt   -> class centers (predictions)

Output:
  * experiments/subclass_composition.json
  * paper/figures/subclass_composition_heatmap.png  (and experiments copy)
"""
from __future__ import annotations

import json
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import apply_style, save_fig
from arcefn.utils.paths import EXPERIMENTS

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT_JSON = EXPERIMENTS / "subclass_composition.json"
MAX_K = 5

SUBCLASS_NAMES = {0: "QCD-Core", 1: "QCD-Edge", 2: "Top-Core", 3: "Top-Edge"}
STATUS = ["True Positive", "False Positive", "True Negative", "False Negative"]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def load_centers() -> np.ndarray:
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 so the checkpoint loads)
    except ImportError:
        pass
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return normalize(sd["arcface_head.class_centers"].detach().cpu().numpy())


_DISPLAY = {"QCD-Core": "QCD-Dense", "QCD-Edge": "QCD-Dilute",
           "Top-Core": "Top-Dilute", "Top-Edge": "Top-Dense"}


def render(matrix: np.ndarray, row_labels: list, partition: str) -> None:
    """Draw the composition figure: two 2x2 panels (Top | QCD), print-sized.

    Display-only: the matrix is block-diagonal (Top rows live in TP/FP,
    QCD rows in TN/FN), so one 4x4 heatmap wastes half its area. Values
    come from the caller unchanged.
    """
    order = ["QCD-Core", "QCD-Edge", "Top-Core", "Top-Edge"]
    idx = {name: row_labels.index(next(l for l in row_labels if l.startswith(_DISPLAY[name])))
           for name in order}
    top_mat = matrix[[idx["Top-Core"], idx["Top-Edge"]]][:, [0, 1]]      # TP, FP
    qcd_mat = matrix[[idx["QCD-Core"], idx["QCD-Edge"]]][:, [2, 3]]      # TN, FN
    top_rows = [row_labels[idx["Top-Core"]], row_labels[idx["Top-Edge"]]]
    qcd_rows = [row_labels[idx["QCD-Core"]], row_labels[idx["QCD-Edge"]]]

    apply_style()
    fig, (axT, axQ) = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, mat, rows, cols, title in [
        (axT, top_mat, top_rows, ["True Positive", "False Positive"], "Top subclasses"),
        (axQ, qcd_mat, qcd_rows, ["True Negative", "False Negative"], "QCD subclasses"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks([0, 1], cols, fontsize=15)
        ax.set_yticks([0, 1], [r.replace(" (N=", "\n(N=") for r in rows], fontsize=15)
        ax.set_title(title, fontsize=17, pad=8)
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                if v >= 0.05:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=20, color="white" if v > 50 else "black")
    axT.set_ylabel(f"Discovered subclass ({partition}-label partition)", fontsize=15)
    cbar = fig.colorbar(im, ax=[axT, axQ], shrink=0.95, pad=0.02)
    cbar.set_label("Percentage (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=15)
    save_fig(fig, "subclass_composition_heatmap.png", to_paper=False)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["predicted", "true"], default="predicted",
                        help="predicted (model view) or true (GT consistency check)")
    parser.add_argument("--replot", action="store_true",
                        help="re-render PNG from cached OUT_JSON only (no recompute, no JSON write)")
    args = parser.parse_args()

    if args.replot:
        with open(OUT_JSON) as f:
            composition = json.load(f)["composition"]
        matrix, row_labels = [], []
        for sc in range(4):
            entry = composition[SUBCLASS_NAMES[sc]]
            matrix.append([entry["status_pct"][cat] for cat in STATUS])
            row_labels.append(f"{_DISPLAY[SUBCLASS_NAMES[sc]]} (N={entry['n']})")
        render(np.array(matrix), row_labels, args.partition)
        print(f"\nReplotted from: {OUT_JSON}  [COMPLETE]")
        return

    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    centers = load_centers()
    emb_norm = normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)
    partition_labels = preds if args.partition == "predicted" else labels

    true_global = np.full(len(labels), -1, dtype=int)
    for cls in (0, 1):
        mask = partition_labels == cls
        sub, _ = spectral_clustering_subclass(emb_norm[mask], max_k=MAX_K)
        counts = Counter(sub)
        order = sorted(counts, key=lambda c: -counts[c])  # largest -> Core (0)
        mapped = np.array([order.index(s) for s in sub])
        true_global[mask] = 2 * cls + mapped

    tp = (preds == 1) & (labels == 1)
    fp = (preds == 1) & (labels == 0)
    tn = (preds == 0) & (labels == 0)
    fn = (preds == 0) & (labels == 1)
    status = np.empty(len(labels), dtype=object)
    status[tp], status[fp], status[tn], status[fn] = STATUS

    composition: dict = {}
    matrix = []
    row_labels = []
    for sc in range(4):
        sc_mask = true_global == sc
        n_tot = int(sc_mask.sum())
        if n_tot == 0:
            continue
        row = []
        entry = {"subclass": SUBCLASS_NAMES[sc], "n": n_tot, "status_pct": {}}
        for cat in STATUS:
            pct = float(((status == cat) & sc_mask).sum() / n_tot * 100.0)
            row.append(pct)
            entry["status_pct"][cat] = pct
        composition[SUBCLASS_NAMES[sc]] = entry
        matrix.append(row)
        _display = {"QCD-Core": "QCD-Dense", "QCD-Edge": "QCD-Dilute",
                    "Top-Core": "Top-Dilute", "Top-Edge": "Top-Dense"}
        row_labels.append(f"{_display[SUBCLASS_NAMES[sc]]} (N={n_tot})")

    render(np.array(matrix), row_labels, args.partition)

    with open(OUT_JSON, "w") as f:
        json.dump({"n_events": int(len(labels)), "composition": composition}, f, indent=2)
    print(f"\nSaved: {OUT_JSON}  [COMPLETE] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
