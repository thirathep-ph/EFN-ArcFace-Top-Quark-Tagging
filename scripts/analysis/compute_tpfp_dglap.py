#!/usr/bin/env python
"""TP/TN/FP/FN DGLAP analysis — group events by prediction correctness per class,
fit bounded DGLAP power law to zg distribution of each group.

Groups:
- TP: Top→Top   (correct top,  label=1 pred=1)
- TN: QCD→QCD   (correct QCD, label=0 pred=0)
- FP: QCD→Top   (false positive, label=0 pred=1)
- FN: Top→QCD   (false negative, label=1 pred=0)

Goal: check whether model errors (FP/FN) have different zg / DGLAP alpha
than correct predictions (TP/TN). If errors cluster in a different zg regime,
that tells us *where* the model is uncertain — physics insight without needing
discrete clusters.

Usage:
    python scripts/analysis/compute_tpfp_dglap.py [--max-events N]

Default max-events=50000 (loads first N events from embeddings + test.h5).
Full 404k test set takes ~15 min due to FastJet feature computation.

Outputs:
- experiments/robustness_s16_m05/tpfp_dglap.json
- experiments/robustness_s16_m05/tpfp_dglap.png
- experiments/robustness_s16_m05/tpfp_dglap.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from arcefn.data.loader import load_awkward
from arcefn.utils.physics import load_cached_features
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
TEST_H5 = EXPERIMENTS.parent / "data" / "top_tagging" / "test.h5"
OUT_DIR = EXPERIMENTS / "robustness_s16_m05"
OUT_JSON = OUT_DIR / "tpfp_dglap.json"
OUT_PNG = OUT_DIR / "tpfp_dglap.png"
OUT_NPZ = OUT_DIR / "tpfp_dglap.npz"

ZG_CUT = 0.1
ZG_MAX = 0.5
N_BINS = 30
RANDOM_STATE = 42


def ap_power_law(zg, A, alpha, beta):
    return (A
            * np.power(np.clip(zg, 1e-10, 1 - 1e-10), alpha)
            * np.power(np.clip(1 - zg, 1e-10, 1 - 1e-10), beta))


def fit_bounded(zg_values, z_cut=ZG_CUT, z_max=ZG_MAX, n_bins=N_BINS):
    """Fit AP power law P(z_g) = A * z_g^alpha * (1-z_g)^beta, alpha in [-2, 0]."""
    valid = zg_values[(zg_values > z_cut) & (zg_values < z_max)]
    if len(valid) < 50:
        return {"A": None, "alpha": None, "beta": None, "r2": None, "n": int(len(valid))}

    hist, bins = np.histogram(valid, bins=n_bins, density=True)
    bc = 0.5 * (bins[:-1] + bins[1:])
    mask = hist > 0
    if mask.sum() < 5:
        return {"A": None, "alpha": None, "beta": None, "r2": None, "n": int(len(valid))}

    popt, _ = curve_fit(
        ap_power_law, bc[mask], hist[mask],
        p0=[1.0, -1.0, 0.0],
        bounds=([0.0, -2.0, -5.0], [100.0, 0.0, 10.0]),
        maxfev=10000,
    )
    A, alpha, beta = popt
    resid = hist[mask] - ap_power_law(bc[mask], *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((hist[mask] - hist[mask].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"A": float(A), "alpha": float(alpha), "beta": float(beta),
            "r2": float(r2), "n": int(len(valid))}


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=50000,
                        help="Max events to load from embeddings + test.h5 (default: 50000)")
    args = parser.parse_args()
    max_events = args.max_events

    # 1. Load embeddings + labels (subset)
    d = np.load(EMBEDDINGS)
    X = d["embeddings"][:max_events]
    y_true = d["labels"][:max_events]
    print(f"Loaded embeddings: {X.shape}, labels: {y_true.shape} (max_events={max_events})")

    # 2. Predictions from checkpoint
    preds = arcface_preds_from_centers(X, CHECKPOINT)
    correct = (preds == y_true)

    # 3. Masks for TP/TN/FP/FN
    # label: 0=QCD, 1=Top ; pred: 0=QCD, 1=Top
    is_top = (y_true == 1)
    is_qcd = (y_true == 0)
    pred_top = (preds == 1)
    pred_qcd = (preds == 0)

    masks = {
        "TP": is_top & pred_top,   # correct Top
        "TN": is_qcd & pred_qcd,   # correct QCD
        "FP": is_qcd & pred_top,   # QCD misclassified as Top
        "FN": is_top & pred_qcd,   # Top misclassified as QCD
    }

    counts = {k: int(m.sum()) for k, m in masks.items()}
    print(f"\nGroup counts: {counts}")
    total = sum(counts.values())
    print(f"Total: {total}")

    # 4. Load test features (zg at index 8) — same event order as embeddings.npz
    print("\nLoading test.h5 features...")
    events, labels_h5, _ = load_awkward(str(TEST_H5), max_events=max_events, lazy=False)
    features = load_cached_features(events)
    zg_all = features[:, 8].astype(np.float64)
    mass_all = features[:, 0].astype(np.float64)
    print(f"Features: {features.shape}, zg range [{zg_all.min():.3f}, {zg_all.max():.3f}]")

    # 5. Fit DGLAP for each group
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("DGLAP zg fit by TP/TN/FP/FN group (bounded alpha ∈ [-2,0])", fontsize=12)

    colors = {"TP": "#0053a6", "TN": "#7aa6d9", "FP": "#c8171f", "FN": "#e88a8f"}
    titles = {"TP": "TP: Top→Top (correct)", "TN": "TN: QCD→QCD (correct)",
              "FP": "FP: QCD→Top (error)", "FN": "FN: Top→QCD (error)"}

    for ax, (name, mask) in zip(axes.flat, masks.items()):
        zg_sub = zg_all[mask]
        mass_sub = mass_all[mask]

        fit = fit_bounded(zg_sub)

        # Histogram
        zg_plot = zg_sub[(zg_sub > ZG_CUT) & (zg_sub < ZG_MAX)]
        ax.hist(zg_plot, bins=N_BINS, density=True, alpha=0.45,
                color=colors[name], label=f"{name} (n={counts[name]})")

        # Fit overlay
        if fit["alpha"] is not None:
            zg_range = np.linspace(ZG_CUT, ZG_MAX, 200)
            ax.plot(zg_range, ap_power_law(zg_range, fit["A"], fit["alpha"], fit["beta"]),
                    color=colors[name], lw=2,
                    label=f"α={fit['alpha']:.3f}, β={fit['beta']:.3f}, R²={fit['r2']:.3f}")

        ax.set_xlabel("$z_g$")
        ax.set_ylabel("$P(z_g)$")
        ax.set_title(titles[name], fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        results[name] = {
            "n": counts[name],
            "fraction": float(counts[name] / total),
            "zg_mean": float(zg_sub.mean()),
            "zg_median": float(np.median(zg_sub)),
            "mass_mean": float(mass_sub.mean()),
            "mass_median": float(np.median(mass_sub)),
            "dglap": fit,
        }

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {OUT_PNG}")

    # 6. Save JSON + NPZ
    results["meta"] = {
        "total_events": total,
        "accuracy": float(counts["TP"] + counts["TN"]) / total,
        "zg_cut": ZG_CUT,
        "zg_max": ZG_MAX,
        "alpha_bound": [-2.0, 0.0],
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {OUT_JSON}")

    np.savez(OUT_NPZ,
             masks={k: v for k, v in masks.items()},
             y_true=y_true,
             preds=preds,
             zg_all=zg_all,
             mass_all=mass_all)
    print(f"Saved NPZ: {OUT_NPZ}")

    # 7. Print summary
    print("\n=== TP/TN/FP/FN DGLAP SUMMARY ===")
    for name in ["TP", "TN", "FP", "FN"]:
        r = results[name]
        dglap = r["dglap"]
        alpha_str = f"{dglap['alpha']:.4f}" if dglap["alpha"] is not None else "N/A"
        beta_str = f"{dglap['beta']:.4f}" if dglap["beta"] is not None else "N/A"
        r2_str = f"{dglap['r2']:.4f}" if dglap["r2"] is not None else "N/A"
        print(f"{name}: n={r['n']:6d} ({100*r['fraction']:5.2f}%)  "
              f"zg_mean={r['zg_mean']:.3f}  mass_mean={r['mass_mean']:7.1f}  "
              f"α={alpha_str}  β={beta_str}  R²={r2_str}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
