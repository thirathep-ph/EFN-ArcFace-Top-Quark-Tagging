#!/usr/bin/env python
"""Plot DGLAP fit residuals for all subgroups."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Load comparison results
comp_path = r"D:\research\ArcEFN\experiments\robustness_s16_m05\dglap_subgroup_comparison.json"
with open(comp_path) as f:
    comp = json.load(f)

# Load cached embeddings and labels
emb_path = r"D:\research\ArcEFN\experiments\robustness_s16_m05\embeddings.npz"
emb_np = np.load(emb_path)
embeddings = emb_np["embeddings"]
labels = emb_np["labels"]
print(f"Loaded {len(embeddings):,} embeddings")

# Use subset for faster residual plotting
MAX_EVENTS = 20000
if len(embeddings) > MAX_EVENTS:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(embeddings), MAX_EVENTS, replace=False)
    embeddings = embeddings[idx]
    labels = labels[idx]
print(f"Using {len(embeddings):,} events for residuals")

# Load test data and compute features
import sys
sys.path.insert(0, r"D:\research\ArcEFN")
from arcefn.data.loader import load_awkward
from arcefn.utils.physics import compute_features
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
from arcefn.utils.paths import DATA_DIR, CHECKPOINT

test_h5 = str(DATA_DIR / "test.h5")
print("Loading test data...")
events, _, _ = load_awkward(test_h5, max_events=len(embeddings), lazy=False)
print(f"Loaded {len(events)} events")

print("Computing features...")
features = compute_features(events)
# Column order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
mass_all = features[:, 0].astype(np.float64)
zg_all = features[:, 8].astype(np.float64)
tau32_all = features[:, 7].astype(np.float64)
theta_g_all = features[:, 9].astype(np.float64)
sqrt_d12_all = features[:, 4].astype(np.float64)

# Subclass discovery (same as comparison script)
print("Discovering subclasses...")
SUBCLASS_NAMES = {0: "QCD_Dense", 1: "QCD_Dilute", 2: "Top_Dilute", 3: "Top_Dense"}

preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
check_cm(preds, labels, "dglap_residuals")

subclass_labels = np.full(len(labels), -1)
for cls in [0, 1]:
    mask = preds == cls
    sc_labels, n_clusters = spectral_clustering_subclass(embeddings[mask], max_k=5)
    if n_clusters >= 2:
        from collections import Counter
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda k: -counts[k])
        label_map = {order[0]: 0, order[1]: 1}
        sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
    sc_labels = sc_labels % 2
    subclass_labels[mask] = sc_labels + (2 * cls)

for sc in range(4):
    n = (subclass_labels == sc).sum()
    print(f"  {SUBCLASS_NAMES[sc]}: {n:,} ({100 * n / len(labels):.1f}%)")

# Get bounded fit parameters from comparison
bounded_params = {}
for sg_name, sg_data in comp.items():
    b = sg_data["methods"]["bounded"]
    bounded_params[sg_name] = {"A": b["A"], "alpha": b["alpha"], "beta": b["beta"]}

# AP power law function
def ap_power_law(zg, A, alpha, beta):
    return A * np.power(np.clip(zg, 1e-10, 1 - 1e-10), alpha) * np.power(
        np.clip(1 - zg, 1e-10, 1 - 1e-10), beta
    )

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

ZG_CUT = 0.1
ZG_MAX = 0.5
N_BINS = 30

for idx, (sg_id, sg_name) in enumerate(SUBCLASS_NAMES.items()):
    ax = axes[idx]
    mask = subclass_labels == sg_id
    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"{sg_name}\n(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(sg_name)
        continue
    
    zg_sub = zg_all[mask]
    zg_sub = zg_sub[(zg_sub > ZG_CUT) & (zg_sub < ZG_MAX)]
    
    if len(zg_sub) < 10:
        ax.text(0.5, 0.5, f"{sg_name}\n(insufficient data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(sg_name)
        continue
    
    # Histogram
    hist, bins = np.histogram(zg_sub, bins=N_BINS, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid = hist > 0
    
    if valid.sum() < 3:
        ax.text(0.5, 0.5, f"{sg_name}\n(insufficient bins)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(sg_name)
        continue
    
    # Model prediction
    p = bounded_params[sg_name]
    model_vals = ap_power_law(bin_centers[valid], p["A"], p["alpha"], p["beta"])
    
    # Residuals
    residuals = hist[valid] - model_vals
    rel_residuals = residuals / (hist[valid] + 1e-10)
    
    # Plot
    ax.scatter(bin_centers[valid], rel_residuals, s=10, alpha=0.6, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5)
    ax.axhline(-0.1, color="gray", linestyle=":", linewidth=0.5)
    ax.set_xlabel(r"$z_g$")
    ax.set_ylabel("Relative residual")
    r2 = comp[sg_name]["methods"]["bounded"]["r2"]
    ax.set_title(f"{sg_name} (α={p['alpha']:.3f}, R²={r2:.3f})")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path(r"D:\research\ArcEFN\experiments\robustness_s16_m05\residual_plots")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "dglap_residuals_bounded.png", dpi=150)
plt.close()
print(f"Saved to {out_dir / 'dglap_residuals_bounded.png'}")

# Also make a combined fit overlay plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (sg_id, sg_name) in enumerate(SUBCLASS_NAMES.items()):
    ax = axes[idx]
    mask = subclass_labels == sg_id
    if mask.sum() == 0:
        continue
    
    zg_sub = zg_all[mask]
    zg_sub = zg_sub[(zg_sub > ZG_CUT) & (zg_sub < ZG_MAX)]
    
    if len(zg_sub) < 10:
        continue
    
    hist, bins = np.histogram(zg_sub, bins=N_BINS, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    valid = hist > 0
    
    if valid.sum() < 3:
        continue
    
    p = bounded_params[sg_name]
    model_vals = ap_power_law(bin_centers[valid], p["A"], p["alpha"], p["beta"])
    
    ax.bar(bin_centers[valid], hist[valid], width=np.diff(bins)[valid], alpha=0.5, label="Data", align="edge")
    ax.plot(bin_centers[valid], model_vals, "r-", label=f"Fit (α={p['alpha']:.3f})")
    ax.set_xlabel(r"$z_g$")
    ax.set_ylabel("Density")
    ax.set_title(sg_name)
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "dglap_fit_overlay.png", dpi=150)
plt.close()
print(f"Saved to {out_dir / 'dglap_fit_overlay.png'}")

print("Done!")