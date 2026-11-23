"""Canonical producer for the compact-form weighting ratio.

Writes experiments/physics_sr/compact_form_ratio.json. Every number the paper
quotes about k must come from this file (SSOT discipline), not from stdout.

Fits s = tanh(m/pT) - k*tau32 on the canonical 404k log-odds with four
parameterisations to expose which quantity is gauge-invariant.
"""
import json

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score

R = "experiments/robustness_s16_m05"
OUT = "experiments/physics_sr/compact_form_ratio.json"

L = np.load("experiments/physics_sr/logits.npz")
C = np.load(f"{R}/jet_pt_mpt_cache.npz")
y = L["logitdiff"].astype(np.float64)
lab = L["labels"].astype(int)
mpt = C["mpt"].astype(np.float64)
t32 = C["tau32"].astype(np.float64)


def r2(pred):
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


fA = lambda X, k, e: np.tanh(X[0]) - k * X[1] + e
fB = lambda X, a, k, e: a * np.tanh(X[0]) - k * X[1] + e
fC = lambda X, a, k, e: a * X[0] - k * X[1] + e
fD = lambda X, a, mu, sd, k, e: a * np.tanh((X[0] - mu) / sd) - k * X[1] + e

pA, _ = curve_fit(fA, (mpt, t32), y, p0=[0.35, 0.0], maxfev=20000)
pB, _ = curve_fit(fB, (mpt, t32), y, p0=[10, 3.5, 0.0], maxfev=20000)
pC, _ = curve_fit(fC, (mpt, t32), y, p0=[10, 3.5, 0.0], maxfev=20000)
pD, _ = curve_fit(fD, (mpt, t32), y, p0=[9, 0.21, 0.04, 4.6, 0.0], maxfev=40000)

# B with the offset forced to zero: the offset is what shifts the ratio most
fB0 = lambda X, a, k: a * np.tanh(X[0]) - k * X[1]
pB0, _ = curve_fit(fB0, (mpt, t32), y, p0=[10, 3.5], maxfev=20000)

# Bootstrap the nominal ratio (B) to get its sampling error
rng = np.random.default_rng(42)
n = len(y)
ratios = []
for _ in range(200):
    idx = rng.integers(0, n, n)
    try:
        pb, _ = curve_fit(fB, (mpt[idx], t32[idx]), y[idx], p0=pB, maxfev=4000)
        ratios.append(pb[1] / pb[0])
    except RuntimeError:
        pass
ratios = np.array(ratios)

# Saturated-region sensitivity: drop |logit| > 11 (34.6% of jets)
keep = np.abs(y) <= 11.0
pBk, _ = curve_fit(fB, (mpt[keep], t32[keep]), y[keep], p0=pB, maxfev=20000)

res = {
    "n_events": int(n),
    "target": "logitdiff (log-odds of the 2-class softmax)",
    "units": "raw physical units (m/pT and tau32, not standardised)",
    "A_amplitude_pinned": {
        "k": float(pA[0]), "offset": float(pA[1]),
        "r2": r2(fA((mpt, t32), *pA)), "gauge_invariant": False,
    },
    "B_free_amplitude_tanh": {
        "a": float(pB[0]), "k": float(pB[1]), "offset": float(pB[2]),
        "ratio_k_over_a": float(pB[1] / pB[0]),
        "r2": r2(fB((mpt, t32), *pB)), "gauge_invariant": True,
    },
    "C_linear_mass": {
        "a": float(pC[0]), "k": float(pC[1]),
        "ratio_k_over_a": float(pC[1] / pC[0]),
        "r2": r2(fC((mpt, t32), *pC)),
    },
    "D_free_step": {
        "a": float(pD[0]), "mu": float(pD[1]), "sd": float(pD[2]),
        "k": float(pD[3]), "offset": float(pD[4]),
        "ratio_k_over_a": float(pD[3] / pD[0]),
        "r2": r2(fD((mpt, t32), *pD)),
    },
    "nominal_ratio": float(pB[1] / pB[0]),
    "bootstrap_ratio_mean": float(ratios.mean()),
    "bootstrap_ratio_std": float(ratios.std()),
    "bootstrap_ratio_ci95": [float(np.percentile(ratios, 2.5)),
                             float(np.percentile(ratios, 97.5))],
    "bootstrap_n": int(len(ratios)),
    "sensitivity": {
        "offset_forced_zero": float(pB0[1] / pB0[0]),
        "trim_abs_logit_le_11": float(pBk[1] / pBk[0]),
        "trim_kept_fraction": float(keep.mean()),
        "free_step_profile": float(pD[3] / pD[0]),
    },
    "ratio_range_over_choices": [
        float(min(pB[1] / pB[0], pC[1] / pC[0], pB0[1] / pB0[0],
                  pBk[1] / pBk[0], pD[3] / pD[0])),
        float(max(pB[1] / pB[0], pC[1] / pC[0], pB0[1] / pB0[0],
                  pBk[1] / pBk[0], pD[3] / pD[0])),
    ],
    "one_third_distance_in_bootstrap_sd": float(
        abs(1.0 / 3.0 - pB[1] / pB[0]) / ratios.std()),
    "auc_of_nominal_form": float(roc_auc_score(lab, fB((mpt, t32), *pB))),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
print(f"\nsaved: {OUT}")
