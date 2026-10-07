"""Distance-to-prototype (dilute) analysis: where do misses live?

For every test event, the signed boundary margin
``margin = cos(emb, w_Top) - cos(emb, w_QCD)`` measures distance to the
decision rule (nearest prototype, margin-free inference). FP/FN events
should pile up at small |margin| (dilute = far from the predicted
prototype, near the boundary); TP/TN should sit at large |margin|.

Also records boundary medoids: the K FP and K FN events with the
smallest |margin| (the most boundary-like misses) for Lund/deta
follow-up plots.

Artifacts read:
  * experiments/robustness_s16_m05/embeddings.npz  -> 404k embeddings + labels
  * experiments/robustness_s16_m05/checkpoint.pt   -> class centers (predictions)

Output:
  * experiments/distance_dilute.json
  * paper/figures/distance_dilute.png  (and experiments copy)
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arcefn.utils.figure_style import apply_style, save_fig
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm, normalize

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT_JSON = EXPERIMENTS / "distance_dilute.json"
FIG_NAME = "distance_dilute.png"
K_MEDOIDS = 10


def summarize(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "q95": float(q[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-medoids", type=int, default=K_MEDOIDS)
    args = parser.parse_args()
    t0 = time.time()

    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"].astype(np.float64), emb_np["labels"].astype(int)
    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, tag="distance_dilute")

    import torch
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        pass
    sd = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    centers = normalize(sd["arcface_head.class_centers"].detach().cpu().numpy().astype(np.float64))

    cos = normalize(embeddings) @ centers.T  # (N, 2), cols = QCD, Top
    margin = cos[:, 1] - cos[:, 0]  # signed distance to boundary; >0 -> Top
    angle_pred = np.degrees(np.arccos(np.clip(cos[np.arange(len(preds)), preds], -1.0, 1.0)))

    groups = {
        "TP": (preds == 1) & (labels == 1),
        "TN": (preds == 0) & (labels == 0),
        "FP": (preds == 1) & (labels == 0),
        "FN": (preds == 0) & (labels == 1),
    }
    out = {"margin": {}, "angle_to_predicted_deg": {}, "boundary_medoids": {}}
    for name, mask in groups.items():
        out["margin"][name] = summarize(margin[mask])
        out["angle_to_predicted_deg"][name] = summarize(angle_pred[mask])

    for name in ("FP", "FN"):
        idx = np.flatnonzero(groups[name])
        order = idx[np.argsort(np.abs(margin[idx]))[: args.k_medoids]]
        out["boundary_medoids"][name] = [
            {"index": int(i), "margin": float(margin[i]),
             "angle_to_predicted_deg": float(angle_pred[i])}
            for i in order
        ]

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON} in {time.time() - t0:.1f}s")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    bins = np.linspace(-1.0, 1.0, 81)
    for ax, name in zip(axes.flat, ("TP", "TN", "FP", "FN")):
        m = margin[groups[name]]
        ax.hist(m, bins=bins, alpha=0.8)
        ax.axvline(0.0, color="k", lw=1)
        ax.axvline(float(np.median(m)), color="r", lw=1.5,
                   label=f"median {np.median(m):+.2f}")
        ax.set_title(f"{name} (n={len(m)})")
        ax.set_xlabel("margin = cos(Top) - cos(QCD)")
        ax.legend(fontsize=9)
    axes[0, 0].set_ylabel("events")
    axes[1, 0].set_ylabel("events")
    fig.suptitle("Boundary margin per prediction status (dilute = near 0)")
    fig.tight_layout()
    save_fig(fig, FIG_NAME, to_paper=True)
    print(f"saved figure {FIG_NAME}")


if __name__ == "__main__":
    main()
