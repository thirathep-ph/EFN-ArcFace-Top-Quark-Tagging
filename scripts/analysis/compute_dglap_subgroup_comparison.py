"""
DGLAP Subgroup Fit Comparison
=============================

Compare 5 fitting methods across all 4 subclasses (QCD_Dense, QCD_Dilute,
Top_Dilute, Top_Dense) to resolve the Top-Dense non-identifiability
(alpha hitting physical bounds, A -> 100, beta -> 6.6).

Methods:
  A. Bounded fit      - constrain alpha to [-2, 0] (DGLAP physical range)
  B. Mass-decorrelated - bin by mass quantiles, fit within each bin
  C. Alt observables   - fit tau32 / theta_g / sqrt(d12) instead of zg
  D. Bootstrap CI      - report alpha distribution, not just point estimate
  E. Increased stats   - loosen subclass cut / merge adjacent mass bins

Outputs:
  experiments/robustness_s16_m05/dglap_subgroup_comparison.json
  experiments/robustness_s16_m05/dglap_subgroup_comparison.png

Run:
  python -m scripts.analysis.compute_dglap_subgroup_comparison [--max-events N]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS, CHECKPOINT
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
from arcefn.utils.physics import compute_features
from arcefn.utils.figure_style import apply_style, save_fig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution

SUBCLASS_NAMES = {0: "QCD_Dense", 1: "QCD_Dilute", 2: "Top_Dilute", 3: "Top_Dense"}
SUBCLASS_COLORS = {
    "QCD_Dense": "#1f77b4",
    "QCD_Dilute": "#ff7f0e",
    "Top_Dilute": "#2ca02c",
    "Top_Dense": "#d62728",
}
REFERENCE_ALPHA = {0: -1.0, 1: -0.8, 2: 0.0, 3: -0.3}

EMBEDDINGS_PATH = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
OUT_DIR = EXPERIMENTS / "robustness_s16_m05"
OUT_JSON = OUT_DIR / "dglap_subgroup_comparison.json"
OUT_PNG = OUT_DIR / "dglap_subgroup_comparison.png"

MAX_EVENTS_DEFAULT = 100_000
N_BOOTSTRAP = 500
ZG_CUT = 0.1
ZG_MAX = 0.5


# ---------------------------------------------------------------------------
# Power-law models
# ---------------------------------------------------------------------------

def ap_power_law(zg, A, alpha, beta):
    return A * np.power(np.clip(zg, 1e-10, 1 - 1e-10), alpha) * np.power(
        np.clip(1 - zg, 1e-10, 1 - 1e-10), beta
    )


def power_law_1d(x, A, alpha):
    """Simple power law y = A * x^alpha (for tau32 / theta_g style)."""
    return A * np.power(np.clip(x, 1e-10, None), alpha)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Method A: bounded fit (alpha constrained to physical DGLAP range)
# ---------------------------------------------------------------------------

def fit_bounded(zg_values, z_cut=ZG_CUT, z_max=ZG_MAX, n_bins=30):
    """Fit AP power law with alpha bounded to [-2, 0]."""
    valid = zg_values[(zg_values > z_cut) & (zg_values < z_max)]
    if len(valid) < 10:
        return None
    hist, edges = np.histogram(valid, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    centers, hist = centers[mask], hist[mask]
    if len(centers) < 3:
        return None
    try:
        popt, _ = curve_fit(
            ap_power_law, centers, hist,
            p0=[1.0, -0.8, 0.5],
            bounds=([0, -2.0, -5.0], [100.0, 0.0, 10.0]),
            maxfev=20000,
        )
        pred = ap_power_law(centers, *popt)
        return {
            "A": float(popt[0]),
            "alpha": float(popt[1]),
            "beta": float(popt[2]),
            "r2": r2_score(hist, pred),
            "n": int(len(valid)),
            "method": "bounded",
            "alpha_at_bound": bool(abs(popt[1] - (-2.0)) < 1e-3 or abs(popt[1] - 0.0) < 1e-3),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Method B: mass-decorrelated (fit within mass bins)
# ---------------------------------------------------------------------------

def fit_mass_decorrelated(zg_values, mass_values, n_bins=5, min_per_bin=500):
    """Fit AP power law within mass quantile bins; report per-bin alpha."""
    if len(zg_values) < min_per_bin * 2:
        return None
    edges = np.quantile(mass_values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_results = []
    for b in range(n_bins):
        bm = (mass_values >= edges[b]) & (mass_values < edges[b + 1])
        zg_sub = zg_values[bm]
        if zg_sub.sum() < min_per_bin:
            continue
        f = fit_bounded(zg_sub)
        if f:
            bin_results.append({
                "bin": b,
                "mass_lo": float(edges[b]),
                "mass_hi": float(edges[b + 1]),
                "n": int(zg_sub.sum()),
                **{k: v for k, v in f.items() if k != "method"},
            })
    if not bin_results:
        return None
    alphas = [r["alpha"] for r in bin_results]
    return {
        "method": "mass_decorrelated",
        "n_bins": len(bin_results),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "alpha_min": float(np.min(alphas)),
        "alpha_max": float(np.max(alphas)),
        "bin_results": bin_results,
        "total_n": int(sum(r["n"] for r in bin_results)),
    }


# ---------------------------------------------------------------------------
# Method C: alternative observables (tau32, theta_g, sqrt(d12))
# ---------------------------------------------------------------------------

def fit_alt_observable(values, observable="tau32", n_bins=30):
    """Fit simple power law y = A * x^alpha to alternative observable."""
    valid = values[(values > 1e-4) & (values < 10.0)]
    if len(valid) < 10:
        return None
    hist, edges = np.histogram(valid, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    centers, hist = centers[mask], hist[mask]
    if len(centers) < 3:
        return None
    try:
        popt, _ = curve_fit(
            power_law_1d, centers, hist,
            p0=[1.0, -1.0],
            bounds=([0, -5.0], [100.0, 5.0]),
            maxfev=20000,
        )
        pred = power_law_1d(centers, *popt)
        return {
            "observable": observable,
            "A": float(popt[0]),
            "alpha": float(popt[1]),
            "r2": r2_score(hist, pred),
            "n": int(len(valid)),
            "method": f"alt_{observable}",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Method D: bootstrap uncertainty
# ---------------------------------------------------------------------------

def bootstrap_alpha(zg_values, n_boot=N_BOOTSTRAP, seed=42, z_cut=ZG_CUT, z_max=ZG_MAX):
    """Bootstrap distribution of alpha (bounded fit)."""
    rng = np.random.default_rng(seed)
    valid = zg_values[(zg_values > z_cut) & (zg_values < z_max)]
    if len(valid) < 50:
        return None
    alphas = []
    for _ in range(n_boot):
        sample = rng.choice(valid, size=len(valid), replace=True)
        f = fit_bounded(sample)
        if f:
            alphas.append(f["alpha"])
    if len(alphas) < 100:
        return None
    arr = np.array(alphas)
    return {
        "method": "bootstrap",
        "n_boot": len(alphas),
        "alpha_median": float(np.median(arr)),
        "alpha_std": float(np.std(arr)),
        "alpha_ci95_lo": float(np.percentile(arr, 2.5)),
        "alpha_ci95_hi": float(np.percentile(arr, 97.5)),
        "alpha_pct_bounded": float(np.mean(np.array(alphas) <= -0.1)) * 100,
    }


# ---------------------------------------------------------------------------
# Method E: increased statistics (relaxed cut / merged bins)
# ---------------------------------------------------------------------------

def fit_increased_stats(zg_values, z_cut=0.05, z_max=0.6, n_bins=40):
    """Looser zg range + more bins to stabilize fit."""
    valid = zg_values[(zg_values > z_cut) & (zg_values < z_max)]
    if len(valid) < 20:
        return None
    hist, edges = np.histogram(valid, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    centers, hist = centers[mask], hist[mask]
    if len(centers) < 3:
        return None
    try:
        popt, _ = curve_fit(
            ap_power_law, centers, hist,
            p0=[1.0, -0.8, 0.5],
            bounds=([0, -2.0, -5.0], [100.0, 0.0, 10.0]),
            maxfev=20000,
        )
        pred = ap_power_law(centers, *popt)
        return {
            "A": float(popt[0]),
            "alpha": float(popt[1]),
            "beta": float(popt[2]),
            "r2": r2_score(hist, pred),
            "n": int(len(valid)),
            "method": "increased_stats",
            "z_cut": z_cut,
            "z_max": z_max,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subclass discovery (reuse existing pipeline logic)
# ---------------------------------------------------------------------------

def discover_subclasses(embeddings: np.ndarray, labels: np.ndarray):
    """Return per-predicted-class subclass labels (0..3)."""
    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, "dglap_comparison")
    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = preds == cls
        sc_labels, n_clusters = spectral_clustering_subclass(embeddings[mask], max_k=5)
        if n_clusters >= 2:
            counts = Counter(sc_labels)
            order = sorted(counts.keys(), key=lambda k: -counts[k])
            label_map = {order[0]: 0, order[1]: 1}
            sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
        sc_labels = sc_labels % 2
        subclass_labels[mask] = sc_labels + (2 * cls)
    return subclass_labels, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DGLAP subgroup fit comparison")
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS_DEFAULT,
                        help=f"Max test events (default {MAX_EVENTS_DEFAULT:,})")
    parser.add_argument("--use-cached-features", action="store_true",
                        help="Use cached features npz if available")
    args = parser.parse_args()

    apply_style()

    t0 = time.time()
    print("=" * 60)
    print("DGLAP Subgroup Fit Comparison (all 4 subgroups x 5 methods)")
    print("=" * 60)

    # ---- Load embeddings (cached, skip model forward pass) ----
    print(f"\n[1/4] Loading cached embeddings from {EMBEDDINGS_PATH}")
    emb_np = np.load(EMBEDDINGS_PATH)
    embeddings = emb_np["embeddings"]
    labels = emb_np["labels"]
    if len(embeddings) > args.max_events:
        idx = np.random.default_rng(42).choice(len(embeddings), args.max_events, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
    print(f"  Loaded {len(embeddings):,} events")

    # ---- Subclass discovery ----
    print("\n[2/4] Subclass discovery (spectral clustering)...")
    subclass_labels, preds = discover_subclasses(embeddings, labels)
    for sc in range(4):
        n = (subclass_labels == sc).sum()
        print(f"  {SUBCLASS_NAMES[sc]}: {n:,} ({100 * n / len(labels):.1f}%)")

    # ---- Compute features (zg, mass, tau32, theta_g, sqrt(d12)) ----
    print("\n[3/4] Loading test data and computing features...")
    test_h5 = str(DATA_DIR / "test.h5")
    from arcefn.data.loader import load_awkward
    events, _, _ = load_awkward(test_h5, max_events=args.max_events, lazy=False)
    # Align events with the subsampled embeddings/labels
    if len(events) != len(embeddings):
        # events loaded full; subset to match embeddings order
        events = events[: len(embeddings)]
    features = compute_features(events)
    # Column order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
    mass_all = features[:, 0].astype(np.float64)
    zg_all = features[:, 8].astype(np.float64)
    tau32_all = features[:, 7].astype(np.float64)
    theta_g_all = features[:, 9].astype(np.float64)
    sqrt_d12_all = features[:, 4].astype(np.float64)
    print(f"  Features computed: zg, mass, tau32, theta_g, sqrt(d12)")

    # ---- Fit all methods for all subgroups ----
    print("\n[4/4] Fitting all methods x subgroups...")
    results = {}

    for sc in range(4):
        name = SUBCLASS_NAMES[sc]
        mask = subclass_labels == sc
        zg_sub = zg_all[mask]
        mass_sub = mass_all[mask]
        tau32_sub = tau32_all[mask]
        theta_g_sub = theta_g_all[mask]
        sqrt_d12_sub = sqrt_d12_all[mask]

        sub_result = {"n_total": int(mask.sum()), "methods": {}}

        # Method A: bounded fit
        f = fit_bounded(zg_sub)
        sub_result["methods"]["bounded"] = f
        if f:
            print(f"  {name} [A bounded]: alpha={f['alpha']:+.3f} R2={f['r2']:.3f} N={f['n']:,} bound={f['alpha_at_bound']}")

        # Method B: mass-decorrelated
        f = fit_mass_decorrelated(zg_sub, mass_sub)
        sub_result["methods"]["mass_decorrelated"] = f
        if f:
            print(f"  {name} [B mass_dec]: alpha={f['alpha_mean']:+.3f}±{f['alpha_std']:.3f} "
                  f"range[{f['alpha_min']:+.3f},{f['alpha_max']:+.3f}] bins={f['n_bins']}")

        # Method C: alt observables (tau32, theta_g, sqrt(d12))
        for obs_name, obs_vals in [("tau32", tau32_sub), ("theta_g", theta_g_sub), ("sqrt_d12", sqrt_d12_sub)]:
            f = fit_alt_observable(obs_vals, observable=obs_name)
            key = f"alt_{obs_name}"
            sub_result["methods"][key] = f
            if f:
                print(f"  {name} [C {obs_name}]: alpha={f['alpha']:+.3f} R2={f['r2']:.3f} N={f['n']:,}")

        # Method D: bootstrap
        f = bootstrap_alpha(zg_sub)
        sub_result["methods"]["bootstrap"] = f
        if f:
            print(f"  {name} [D bootstrap]: alpha={f['alpha_median']:+.3f} "
                  f"CI[{f['alpha_ci95_lo']:+.3f},{f['alpha_ci95_hi']:+.3f}] "
                  f"pct<=-0.1={f['alpha_pct_bounded']:.1f}%")

        # Method E: increased stats
        f = fit_increased_stats(zg_sub)
        sub_result["methods"]["increased_stats"] = f
        if f:
            print(f"  {name} [E inc_stats]: alpha={f['alpha']:+.3f} R2={f['r2']:.3f} "
                  f"N={f['n']:,} z_cut={f['z_cut']} z_max={f['z_max']}")

        results[name] = sub_result

    # ---- Save JSON ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Saved: {OUT_JSON}")

    # ---- Print summary table ----
    print("\n" + "=" * 60)
    print("SUMMARY TABLE (alpha per method per subgroup)")
    print("=" * 60)
    print(f"  {'Subgroup':14s} {'Method':18s} {'alpha':>9s} {'R2':>7s} {'N':>8s}  Notes")
    print(f"  {'-' * 14} {'-' * 18} {'-' * 9} {'-' * 7} {'-' * 8}")
    for sc in range(4):
        name = SUBCLASS_NAMES[sc]
        for mname, f in results[name]["methods"].items():
            if f is None:
                print(f"  {name:14s} {mname:18s} {'FAILED':>9s} {'-':>7s} {'-':>8s}")
                continue
            alpha = f.get("alpha", f.get("alpha_mean", f.get("alpha_median", float("nan"))))
            r2 = f.get("r2", f.get("alpha_std", float("nan")))
            n = f.get("n", f.get("total_n", f.get("n_boot", 0)))
            note = ""
            if mname == "bounded" and f.get("alpha_at_bound"):
                note = "AT_BOUND"
            if mname == "bootstrap":
                note = f"CI[{f['alpha_ci95_lo']:+.2f},{f['alpha_ci95_hi']:+.2f}]"
            print(f"  {name:14s} {mname:18s} {alpha:>+9.3f} {r2:>7.3f} {n:>8,}  {note}")

    # ---- Recommendation ----
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    for sc in range(4):
        name = SUBCLASS_NAMES[sc]
        bounded = results[name]["methods"]["bounded"]
        bootstrap = results[name]["methods"]["bootstrap"]
        mass_dec = results[name]["methods"]["mass_decorrelated"]
        rec = "Use bounded fit (alpha within DGLAP range)"
        if bounded and bounded.get("alpha_at_bound"):
            rec = "Alpha hits physical bound — consider bootstrap or mass-decorrelated"
        if bootstrap and abs(bootstrap["alpha_ci95_hi"] - bootstrap["alpha_ci95_lo"]) > 0.5:
            rec = "Wide bootstrap CI — non-identifiable even with resampling"
        if mass_dec and mass_dec["alpha_std"] > 0.3:
            rec = "Alpha varies across mass bins — mass echo likely"
        print(f"  {name}: {rec}")

    print(f"\n[COMPLETE] Elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
