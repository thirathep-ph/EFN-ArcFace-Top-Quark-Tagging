"""Regenerate prototype-distance evolution with 10 bins, PER TRUE CLASS.

The paper's Table tab:phys-evol claims 5 bins and per-class paragraphs whose
numbers (2.0773 +- 0.3988, 20086 Top / 201914 QCD, rho=+0.7209/+0.8658) have no
reproducible artifact. bins shows those came from a 100-event smoke run.
This script computes the real thing and writes a fresh artifact.

Binning is by distance to own prototype, within each true class, 10
equal-frequency bins (user's requested protocol).
"""
import json

import numpy as np
from scipy.stats import spearmanr

D = "experiments/robustness_s16_m05"
z = np.load(f"{D}/prototype_distance_evolution.npz")
d_own = z["dist_to_own"].astype(float)
labels = z["labels"].astype(int)
d_top = z["dist_to_top"].astype(float)
d_qcd = z["dist_to_qcd"].astype(float)

import pandas as pd

feat = pd.read_parquet(f"{D}/test_features.parquet")
mpt = feat["Mass"].to_numpy() / np.maximum(feat["mSD"].to_numpy(), 1e-6)

VARS = {
    "mass": feat["Mass"].to_numpy(float),
    "tau32": feat["Tau32"].to_numpy(float),
    "zg": feat["zg"].to_numpy(float),
    "theta_g": feat["theta_g"].to_numpy(float),
    "n_const": feat["Mult"].to_numpy(float),
    "mSD": feat["mSD"].to_numpy(float),
}

N_BINS = 10
out = {"protocol": {"n_bins": N_BINS, "binning": "equal-frequency of "
                    "distance to own prototype, computed within each true "
                    "class", "n_events": int(len(labels))},
       "classes": {}}

for cname, cval in (("top", 1), ("qcd", 0)):
    sel = labels == cval
    d = d_own[sel]
    # own prototype distance for this class = dist_to_top for top etc.
    own = d_top[sel] if cval == 1 else d_qcd[sel]
    other = d_qcd[sel] if cval == 1 else d_top[sel]
    correct = z["preds"][sel] == cval
    edges = np.quantile(own, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(own, edges) - 1, 0, N_BINS - 1)

    rows = []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() < 10:
            continue
        row = {"bin": b, "n": int(m.sum()),
               "dist_own": float(own[m].mean()),
               "margin": float((own[m] - other[m]).mean()),
               "frac_correct": float(correct[m].mean())}
        for k, v in VARS.items():
            row[k] = float(v[sel][m].mean())
        rows.append(row)

    # Spearman between bin index and each observable across bins
    bidx = np.array([r["bin"] for r in rows], float)
    rho = {k: float(spearmanr(bidx, [r[k] for r in rows]).statistic)
           for k in ("mass", "tau32", "zg", "theta_g", "n_const")}
    rho["frac_correct"] = float(
        spearmanr(bidx, [r["frac_correct"] for r in rows]).statistic)

    out["classes"][cname] = {
        "n_total": int(sel.sum()),
        "accuracy": float(correct.mean()),
        "dist_own_mean": float(own.mean()),
        "dist_own_std": float(own.std()),
        "bin_edges": [float(e) for e in edges],
        "bins": rows,
        "spearman_across_bins": rho,
    }

with open(f"{D}/prototype_distance_evolution_10bin.json", "w",
          encoding="utf-8") as f:
    json.dump(out, f, indent=2)

for cname, c in out["classes"].items():
    print(f"== {cname}: n={c['n_total']} acc={c['accuracy']:.4f} "
          f"<d>={c['dist_own_mean']:.4f}+-{c['dist_own_std']:.4f}")
    print("  bin      n   <d>     mass   mSD   tau32   zg    thg   nC  "
          "correct")
    for r in c["bins"]:
        print(f"  {r['bin']:>3} {r['n']:>7} {r['dist_own']:6.3f} "
              f"{r['mass']:7.1f} {r['mSD']:6.1f} {r['tau32']:6.3f} "
              f"{r['zg']:.3f} {r['theta_g']:.3f} {r['n_const']:5.1f} "
              f"{r['frac_correct']:.3f}")
    print("  rho:", {k: round(v, 4) for k, v in
                     c["spearman_across_bins"].items()})
print("\nsaved:", f"{D}/prototype_distance_evolution_10bin.json")
