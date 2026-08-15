"""
AP Splitting Fit vs DGLAP Reference
===================================
Fits P(zg) = A * zg^alpha * (1-zg)^beta per subclass and compares
alpha against rough reference values (heuristics for display only, NOT
DGLAP predictions; the LO 1/z soft divergence is an asymptotic statement,
not a prediction of the fitted global exponent under grooming/Sudakov):
  - QCD Core:  alpha ~ -1.0  (g->gg soft divergence P(z) propto 1/z)
  - QCD Edge:  alpha ~ -0.8  (mixed perturbative / non-perturbative)
  - Top Core:  alpha ~  0.0  (W->qqbar flat, no soft divergence)
  - Top Edge:  alpha ~ -0.3  (QCD-like contamination)

Saves figure to ``EXPERIMENTS / 'dglap_fit.png'`` and fit parameters
to ``EXPERIMENTS / 'dglap_fit_params.json'``.

Examples
--------
Run from the project root::

    python scripts/analysis/compute_dglap_fit.py
"""
import time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from collections import Counter

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS, MODEL_DIR, CHECKPOINT
from arcefn.utils.device import get_device
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.physics import compute_features
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import (
    apply_style, panel_label, add_grid, save_fig,
    SUBCLASS_COLORS, LEGEND_KWARGS,
)
apply_style()

MAX_EVENTS = 404000
BATCH_SIZE = 2048

SUBCLASS_NAMES = {0: "QCD_Core", 1: "QCD_Edge", 2: "Top_Core", 3: "Top_Edge"}
REFERENCE_ALPHA = {0: -1.0, 1: -0.8, 2: 0.0, 3: -0.3}
REFERENCE_DESC = {
    0: r"g->gg P(z) $\propto$ 1/z (soft divergence)",
    1: r"Mixed QCD",
    2: r"W->qq (no soft divergence)",
    3: r"QCD-like contamination",
}


def ap_power_law(zg, A, alpha, beta):
    """
    Power-law parametrisation of the Altarelli-Parisi splitting function.

    .. math::

        P(z_g) = A \\cdot z_g^{\\alpha} \\cdot (1 - z_g)^{\\beta}

    Parameters
    ----------
    zg : np.ndarray
        Momentum fraction values.
    A : float
        Normalisation constant.
    alpha : float
        Small-z exponent.
    beta : float
        Large-(1-z) exponent.

    Returns
    -------
    np.ndarray
        Evaluated power law at ``zg``.
    """
    return A * np.power(np.clip(zg, 1e-10, 1 - 1e-10), alpha) * np.power(np.clip(1 - zg, 1e-10, 1 - 1e-10), beta)


def fit_zg(zg_values, z_cut=0.1, n_bins=30):
    """
    Fit the AP power law to a histogram of zg values.

    Parameters
    ----------
    zg_values : np.ndarray
        Raw zg values for one subclass.
    z_cut : float
        Lower cut on zg (exclude non-perturbative region).
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict or None
        Fit parameters ``{'A', 'alpha', 'beta', 'r2', 'n'}``, or
        ``None`` if the fit fails.
    """
    valid = zg_values[(zg_values > z_cut) & (zg_values < 0.5)]
    if len(valid) < 10:
        return None
    hist, edges = np.histogram(valid, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = hist > 0
    centers, hist = centers[mask], hist[mask]
    if len(centers) < 3:
        return None
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(ap_power_law, centers, hist, p0=[1.0, -0.5, 0.5],
                            bounds=([0, -5, -5], [100, 10, 10]), maxfev=10000)
        pred = ap_power_law(centers, *popt)
        ss_res = np.sum((hist - pred) ** 2)
        ss_tot = np.sum((hist - hist.mean()) ** 2)
        return {"A": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2]),
                "r2": float(1 - ss_res / (ss_tot + 1e-10)), "n": int(len(valid))}
    except Exception:
        return None


def main():
    """Run DGLAP fit analysis: load model, discover subclasses, fit and plot."""
    print("=" * 60)
    print("AP Splitting Fit vs DGLAP Theory")
    print("=" * 60)

    device, _ = get_device()
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded from {CHECKPOINT}")

    test_h5 = DATA_DIR / 'test.h5'
    print(f"\nStep 1: Loading test data ({MAX_EVENTS:,} events)...")
    t0 = time.time()
    events, labels, weights = load_awkward(test_h5, max_events=MAX_EVENTS, lazy=False)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nStep 2: Computing embeddings...")
    t0 = time.time()
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_emb = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu().numpy())
    embeddings = np.vstack(all_emb)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nStep 3: Subclass discovery via Spectral Clustering...")
    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = labels == cls
        if mask.sum() > 10:
            sc_labels, n_clusters = spectral_clustering_subclass(embeddings[mask], max_k=5)
            if n_clusters >= 2:
                counts = Counter(sc_labels)
                sorted_clusters = sorted(counts.keys(), key=lambda k: -counts[k])
                label_map = {sorted_clusters[0]: 0, sorted_clusters[1]: 1}
                sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
            sc_labels = sc_labels % 2
            subclass_labels[mask] = sc_labels + (2 * cls)

    for sc in range(4):
        n = (subclass_labels == sc).sum()
        print(f"  {SUBCLASS_NAMES[sc]}: {n:,} ({100*n/len(labels):.1f}%)")

    print("\nStep 4: Computing zg via FastJet...")
    t0 = time.time()
    features = compute_features(events)
    zg_all = features[:, 8]
    print(f"  Done in {time.time()-t0:.0f}s")

    del model, embeddings
    import gc; gc.collect()

    print("\nStep 5: Fitting AP splitting functions per subclass...")
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    sc_colors = [
        SUBCLASS_COLORS["QCD core"],
        SUBCLASS_COLORS["QCD edge"],
        SUBCLASS_COLORS["Top core"],
        SUBCLASS_COLORS["Top edge"],
    ]

    for sc in range(4):
        mask = subclass_labels == sc
        zg_sub = zg_all[mask]
        fit = fit_zg(zg_sub)
        name = SUBCLASS_NAMES[sc]

        if fit:
            results[name] = fit
            delta = fit['alpha'] - REFERENCE_ALPHA[sc]
            print(f"\n  {name}:")
            print(f"    alpha_fit = {fit['alpha']:.3f}  |  alpha_ref = {REFERENCE_ALPHA[sc]:+.1f}  |  Delta = {delta:+.3f}")
            print(f"    beta = {fit['beta']:.3f}  |  R^2 = {fit['r2']:.4f}  |  N = {fit['n']:,}")
            print(f"    Reference: {REFERENCE_DESC[sc]}")
        else:
            print(f"\n  {name}: Fit failed (N={mask.sum():,})")

        ax = axes[sc // 2, sc % 2]
        zg_plot = zg_sub[(zg_sub > 0.1) & (zg_sub < 0.5)]
        ax.hist(zg_plot, bins=30, density=True, alpha=0.45, color=sc_colors[sc],
                edgecolor="white", linewidth=0.3, label='Data')
        if fit:
            zg_range = np.linspace(0.1, 0.5, 100)
            ax.plot(zg_range, ap_power_law(zg_range, fit['A'], fit['alpha'], fit['beta']),
                    color="0.2", linewidth=1.2,
                    label=f'Fit: $\\alpha$={fit["alpha"]:.2f}, $R^2$={fit["r2"]:.3f}')
        ax.set_xlabel('$z_g$')
        ax.set_ylabel('$P(z_g)$')
        panel_label(ax, panel_labels[sc])
        display_name = name.replace("_", " ")
        ax.text(0.97, 0.95, display_name, transform=ax.transAxes,
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='0.7'))
        ax.legend(loc="upper right", **{k: v for k, v in LEGEND_KWARGS.items() if k not in ("ncol",)})
        add_grid(ax)
        ax.set_xlim(0.08, 0.52)

    plt.tight_layout(w_pad=1.5, h_pad=1.2)
    save_fig(fig, "dglap_fit.png")

    print(f"\n{'='*60}\nDGLAP COMPARISON TABLE\n{'='*60}")
    print(f"  {'Subclass':12s}  {'alpha_fit':>9s}  {'alpha_ref':>11s}  {'Delta':>7s}  {'beta':>7s}  {'R^2':>6s}  {'N':>8s}")
    for sc in range(4):
        name = SUBCLASS_NAMES[sc]
        if name in results:
            r = results[name]
            d = r['alpha'] - REFERENCE_ALPHA[sc]
            print(f"  {name:12s}  {r['alpha']:>9.3f}  {REFERENCE_ALPHA[sc]:>+11.1f}  {d:>+7.3f}  {r['beta']:>7.3f}  {r['r2']:>6.4f}  {r['n']:>8,}")
        else:
            print(f"  {name:12s}  {'FAILED':>9s}")

    json_path = EXPERIMENTS / 'dglap_fit_params.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
