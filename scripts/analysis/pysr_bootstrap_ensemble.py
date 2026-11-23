"""Bootstrap ensemble uncertainty for PySR symbolic regression.

Follows Paper 1 (Bayesian SR, arXiv:2603.14918) approach: quantifies
uncertainty in recovered model structure via resampling.

Fits the compact form s = tanh(a * m/pT + b) + c * tau32 to the model's
logit scores (from paired_scores_cache.npz) across 10 bootstrap resamples.

Also checks asymptotic behavior (Paper 2 emerging direction):
  m/pT -> 0: s -> tanh(b) + c * tau32
  m/pT -> inf: s -> 1 + c * tau32

Usage:
    python scripts/analysis/pysr_bootstrap_ensemble.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.physics import load_cached_features

OUT_DIR = EXPERIMENTS / "robustness_s16_m05"
OUT_JSON = OUT_DIR / "pysr_bootstrap_ensemble.json"
STAGING = Path("C:/Users/Thirathep/AppData/Local/Temp/opencode/staging")


def compact_form(x, a, b, c):
    """s = tanh(a * m/pT + b) + c * tau32"""
    mpt = x[:, 0]
    tau32 = x[:, 1]
    return np.tanh(a * mpt + b) + c * tau32


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Load features from parquet cache
    import pyarrow.parquet as pq
    t = pq.read_table(OUT_DIR / "test_features.parquet",
                      columns=["event_idx", "Mass", "Tau32"])
    mass = t.column("Mass").to_numpy(zero_copy_only=False)
    tau32 = t.column("Tau32").to_numpy(zero_copy_only=False)
    idx_all = t.column("event_idx").to_numpy(zero_copy_only=False)
    order = np.argsort(idx_all, kind="stable")
    mass = mass[order]
    tau32 = tau32[order]

    # Load model scores
    d = np.load(EXPERIMENTS / "paired_scores_cache.npz")
    scores = d["scores_arcface_s16_m05"]
    labels = d["labels"]

    # Compute m/pT (need jet_pt from logits.npz or approximate)
    # Use mass/pT approximation: pT ~ 500 GeV (typical for this dataset)
    # Actually, let's use the cached pt if available
    # From the paper: features are [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
    # m/pT = mass / pt. We need pt. Let's check if there's a pt cache.
    pt_cache = EXPERIMENTS / "pt_cache_test.npz"
    if pt_cache.exists():
        pt_data = np.load(pt_cache)
        jet_pt = pt_data["jet_pt"][:len(mass)]
    else:
        # Approximate: use mass / 0.3 as typical m/pT ratio for top jets
        # Better: load from test.h5
        from arcefn.data.loader import load_awkward
        from arcefn.utils.paths import DATA_DIR
        events, labels_h5, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=len(mass), lazy=False)
        # Compute jet pt from constituents using awkward
        # jet_pt = magnitude of total transverse momentum vector
        import awkward as ak
        if hasattr(events, 'px'):
            px_total = ak.sum(events.px, axis=1)
            py_total = ak.sum(events.py, axis=1)
            jet_pt = np.sqrt(np.asarray(px_total)**2 + np.asarray(py_total)**2)
            jet_pt = np.asarray(jet_pt, dtype=np.float64)
        else:
            jet_pt = mass / 0.3

    mpt = mass / jet_pt

    # Standardize (z-score) as in original PySR
    x_mean = np.array([mpt.mean(), tau32.mean()])
    x_std = np.array([mpt.std(), tau32.std()])
    Xz_mpt = (mpt - x_mean[0]) / x_std[0]
    Xz_tau32 = (tau32 - x_mean[1]) / x_std[1]
    X = np.column_stack([Xz_mpt, Xz_tau32])

    # y: model scores (already in logit space)
    y = scores.astype(np.float64)

    # y_mean, y_std for inverse transform
    y_mean = y.mean()
    y_std = y.std()

    print(f"Data: {len(X)} events")
    print(f"m/pT: mean={mpt.mean():.4f}, std={mpt.std():.4f}")
    print(f"tau32: mean={tau32.mean():.4f}, std={tau32.std():.4f}")
    print(f"scores: mean={y.mean():.4f}, std={y.std():.4f}")

    # ── Bootstrap ensemble ──────────────────────────────────────────────────
    n_bootstrap = 10
    n_events = len(X)
    bootstrap_size = n_events  # full bootstrap (resample with replacement)

    coefficients = []
    metrics = []

    for b in range(n_bootstrap):
        rng_b = np.random.default_rng(42 + b)
        idx = rng_b.choice(n_events, size=bootstrap_size, replace=True)
        X_b = X[idx]
        y_b = y[idx]

        try:
            popt, pcov = curve_fit(compact_form, X_b, y_b, p0=[1.0, 0.0, -0.3], maxfev=5000)
            a, bb, c = popt
            coefficients.append({"a": float(a), "b": float(bb), "c": float(c)})
            # Predict on full data
            y_pred = compact_form(X, *popt)
            acc = float(np.mean((y_pred > 0.5) == (y > 0.5)))
            metrics.append({"acc": acc})
            print(f"  Bootstrap {b+1}/{n_bootstrap}: a={a:.4f}, b={bb:.4f}, c={c:.4f}, acc={acc:.4f}")
        except Exception as e:
            print(f"  Bootstrap {b+1}/{n_bootstrap}: FAILED ({e})")

    # ── Summary statistics ──────────────────────────────────────────────────
    a_vals = np.array([c["a"] for c in coefficients])
    b_vals = np.array([c["b"] for c in coefficients])
    c_vals = np.array([c["c"] for c in coefficients])
    acc_vals = np.array([m["acc"] for m in metrics])

    print(f"\n=== Bootstrap Ensemble Summary (n={n_bootstrap}) ===")
    print(f"  a (tanh slope):  {a_vals.mean():.4f} ± {a_vals.std():.4f}")
    print(f"  b (tanh bias):   {b_vals.mean():.4f} ± {b_vals.std():.4f}")
    print(f"  c (tau32 coeff): {c_vals.mean():.4f} ± {c_vals.std():.4f}")
    print(f"  accuracy:        {acc_vals.mean():.4f} ± {acc_vals.std():.4f}")

    # ── Asymptotic behavior check (Paper 2 emerging direction) ─────────────
    print("\n=== Asymptotic Behavior Check ===")
    a_mean = a_vals.mean()
    b_mean = b_vals.mean()
    c_mean = c_vals.mean()
    print(f"  At m/pT -> 0:  s -> tanh({b_mean:.4f}) + {c_mean:.4f} * tau32 = {np.tanh(b_mean):.4f} + {c_mean:.4f} * tau32")
    print(f"  At m/pT -> inf: s -> 1.0 + {c_mean:.4f} * tau32")
    print(f"  tau32 range: [0, 1] → s range at m/pT->0: [{np.tanh(b_mean):.4f}, {np.tanh(b_mean)+c_mean:.4f}]")
    print(f"  tau32 range: [0, 1] → s range at m/pT->inf: [1.0, {1.0+c_mean:.4f}]")

    # ── Compare with old equation ───────────────────────────────────────────
    print("\n=== Comparison with old equation ===")
    print(f"  Old: s = tanh(m/pT) - 0.348 * tau32  (AUC 0.950, acc 89.4%)")
    print(f"  New: s = tanh({a_mean:.4f} * m/pT + {b_mean:.4f}) + {c_mean:.4f} * tau32")
    print(f"  New acc (mean): {acc_vals.mean():.4f}")

    # ── Save results ────────────────────────────────────────────────────────
    result = {
        "n_bootstrap": n_bootstrap,
        "n_events": int(n_events),
        "coefficients": {
            "a": {"mean": float(a_vals.mean()), "std": float(a_vals.std()), "ci95": [float(a_vals.mean() - 1.96*a_vals.std()), float(a_vals.mean() + 1.96*a_vals.std())]},
            "b": {"mean": float(b_vals.mean()), "std": float(b_vals.std()), "ci95": [float(b_vals.mean() - 1.96*b_vals.std()), float(b_vals.mean() + 1.96*b_vals.std())]},
            "c": {"mean": float(c_vals.mean()), "std": float(c_vals.std()), "ci95": [float(c_vals.mean() - 1.96*c_vals.std()), float(c_vals.mean() + 1.96*c_vals.std())]},
        },
        "metrics": {
            "accuracy": {"mean": float(acc_vals.mean()), "std": float(acc_vals.std())},
        },
        "asymptotic": {
            "mpt_zero": {"s_low": float(np.tanh(b_mean)), "s_high": float(np.tanh(b_mean) + c_mean)},
            "mpt_inf": {"s_low": 1.0, "s_high": float(1.0 + c_mean)},
        },
        "old_equation": {
            "form": "tanh(m/pT) - 0.348 * tau32",
            "acc": 0.894,
            "auc": 0.950,
        },
        "normalization": "z-score (X0 = dimless features, Xz = (X0 - x_mean)/x_std)",
        "runtime_seconds": time.time() - t0,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    import matplotlib.pyplot as plt
    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(a_vals, bins=8, color="#0053a6", alpha=0.7, edgecolor="white")
    axes[0].axvline(a_vals.mean(), color="#c8171f", linestyle="--", lw=2)
    axes[0].set_title(f"a (tanh slope): {a_vals.mean():.4f} ± {a_vals.std():.4f}")
    axes[0].set_xlabel("a")
    axes[1].hist(b_vals, bins=8, color="#0053a6", alpha=0.7, edgecolor="white")
    axes[1].axvline(b_vals.mean(), color="#c8171f", linestyle="--", lw=2)
    axes[1].set_title(f"b (tanh bias): {b_vals.mean():.4f} ± {b_vals.std():.4f}")
    axes[1].set_xlabel("b")
    axes[2].hist(c_vals, bins=8, color="#0053a6", alpha=0.7, edgecolor="white")
    axes[2].axvline(c_vals.mean(), color="#c8171f", linestyle="--", lw=2)
    axes[2].set_title(f"c (tau32 coeff): {c_vals.mean():.4f} ± {c_vals.std():.4f}")
    axes[2].set_xlabel("c")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Bootstrap Ensemble: PySR Compact Equation Coefficients (10 resamples)")
    fig.tight_layout()
    fig.savefig(STAGING / "pysr_bootstrap_ensemble.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {STAGING / 'pysr_bootstrap_ensemble.png'}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
