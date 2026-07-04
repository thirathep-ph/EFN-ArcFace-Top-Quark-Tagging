"""
AP Splitting — CHUNKED FINAL VERSION
======================================
Processes ALL 404K events in batches:
1. Load data with lazy=False (saves memory)
2. Slice each chunk → convert to proper Awkward via load_events_by_indices
3. Compute features with FastJet on each chunk
4. Combine results
5. Subclass discovery via Spectral Clustering (eigengap heuristic)
6. Fit AP splitting function P(zg) ∝ zg^α (1-zg)^β per subclass
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.data.loader import load_awkward, load_events_by_indices, LazyAwkwardEvents
from arcjet.utils.physics import compute_features
from arcjet.utils.clustering import spectral_clustering_subclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "ap_splitting_results.json")


def ap_func(zg, A, alpha, beta):
    return A * np.power(zg + 1e-10, alpha) * np.power(1 - zg + 1e-10, beta)


def fit_ap(zg_values, z_cut=0.1, n_bins=30):
    zg_valid = zg_values[(zg_values > z_cut) & (zg_values < 0.5)]
    if len(zg_valid) < 10:
        return None
    
    hist, bin_edges = np.histogram(zg_valid, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    bin_centers = bin_centers[mask]
    hist = hist[mask]
    
    if len(bin_centers) < 3:
        return None
    
    try:
        popt, _ = curve_fit(ap_func, bin_centers, hist, p0=[1.0, -0.5, 0.5],
                           bounds=([0, -5, -5], [100, 5, 5]), maxfev=10000)
        A, alpha, beta = popt
        predicted = ap_func(bin_centers, *popt)
        ss_res = np.sum((hist - predicted) ** 2)
        ss_tot = np.sum((hist - hist.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        return {"A": float(A), "alpha": float(alpha), "beta": float(beta), "r2": float(r2), "n": len(zg_valid)}
    except:
        return None


def compute_features_chunked(lazy_events, chunk_size=10000):
    """Compute features in chunks, converting lazy→real awkard per chunk."""
    n = len(lazy_events)
    all_features = []
    
    n_chunks = (n + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        
        # Slice first → get real events via __getitem__
        try:
            # Direct slice on lazy
            chunk = lazy_events[start:end]
            feats = compute_features(chunk)
        except:
            # Fallback: load by indices  
            indices = list(range(start, end))
            chunk = load_events_by_indices(TEST_H5, indices)
            feats = compute_features(chunk)
        
        all_features.append(feats)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Chunk {i+1}/{n_chunks} ({end:,}/{n:,})")
    
    return np.vstack(all_features)


def main():
    print("=" * 60)
    print("AP Splitting — CHUNKED FINAL")
    print("=" * 60)
    
    # Step 1: Load ALL data (lazy=False to save memory)
    print("\nStep 1: Loading ALL test data (lazy)...")
    t0 = time.time()
    
    # Use lazy=False to save memory, chunk internally converts to real arrays
    max_ev = 404000
    events, labels, weights = load_awkward(TEST_H5, max_events=max_ev, lazy=False)
    print(f"  Events: {len(labels):,}, Time: {time.time()-t0:.0f}s")
    
    # Step 2: Compute features chunked
    print(f"\nStep 2: Computing features (chunked)...")
    t0 = time.time()
    
    features = compute_features_chunked(events, chunk_size=10000)
    
    print(f"  Total: {features.shape}, Time: {time.time()-t0:.0f}s")
    zg_all = features[:, 8]
    print(f"  zg: {np.isnan(zg_all).sum()} NaN, valid range: [{zg_all[zg_all>0].min():.4f}, {zg_all.max():.4f}]")
    
    # Step 3: Get subclass labels from model via Spectral Clustering
    print(f"\nStep 3: Subclass discovery (Spectral Clustering)...")
    
    import torch
    from torch.utils.data import DataLoader
    from arcjet.models.efn_arcface import TopTaggingModel
    from arcjet.data.loader import JetTaggingDataset
    from arcjet.utils.device import get_device
    
    device, _ = get_device()
    model_path = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    # Use JetTaggingDataset which handles lazy properly
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    
    all_emb = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu().numpy())
    
    embeddings = np.vstack(all_emb)
    
    # Use Spectral Clustering per class (the project-standard approach)
    subclass_names = {0: "QCD_Core", 1: "QCD_Edge", 2: "Top_Core", 3: "Top_Edge"}
    subclass_labels = np.full(len(labels), -1)
    
    for cls in [0, 1]:
        mask = labels == cls
        if mask.sum() > 10:
            sc_labels, n_clusters = spectral_clustering_subclass(embeddings[mask], max_k=5)
            # Map clusters to Core (0) / Edge (1) by size (Core = larger cluster)
            if n_clusters >= 2:
                from collections import Counter
                counts = Counter(sc_labels)
                sorted_clusters = sorted(counts.keys(), key=lambda k: -counts[k])
                label_map = {sorted_clusters[0]: 0, sorted_clusters[1]: 1}
                sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
            sc_labels = sc_labels % 2  # ensure 0/1
            subclass_labels[mask] = sc_labels + (2 * cls)
            print(f"  {subclass_names[2*cls]:10s} + {subclass_names[2*cls+1]:10s}: found {n_clusters} clusters (using top 2)")
    
    print(f"  Subclass distribution:")
    for sc in range(4):
        mask = subclass_labels == sc
        print(f"    {subclass_names[sc]}: {mask.sum()} ({100*mask.sum()/len(labels):.1f}%)")
    
    del model
    import gc; gc.collect()
    
    # Step 4: Fit AP splitting for each subclass
    print(f"\n{'='*60}")
    print("Step 4: Fitting AP splitting functions")
    print("=" * 60)
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'cyan', 'red', 'orange']
    theory_alphas = {0: -1.0, 1: -0.8, 2: 0.0, 3: -0.2}
    
    for sc in range(4):
        mask = subclass_labels == sc
        zg_sub = zg_all[mask]
        
        fit = fit_ap(zg_sub)
        if fit:
            results[subclass_names[sc]] = fit
            print(f"\n  {subclass_names[sc]}:")
            print(f"    α = {fit['alpha']:.3f} (theory: {theory_alphas[sc]:.1f})")
            print(f"    β = {fit['beta']:.3f}")
            print(f"    R² = {fit['r2']:.4f}, N = {fit['n']:,}")
        else:
            print(f"\n  {subclass_names[sc]}: Fit failed")
        
        # Plot
        ax = axes[sc // 2, sc % 2]
        zg_plot = zg_sub[(zg_sub > 0.1) & (zg_sub < 0.5)]
        ax.hist(zg_plot, bins=30, density=True, alpha=0.5, color=colors[sc], label='Data')
        
        if fit:
            zg_range = np.linspace(0.1, 0.5, 100)
            ax.plot(zg_range, ap_func(zg_range, fit['A'], fit['alpha'], fit['beta']), 'k-', lw=2,
                   label=f'Fit: α={fit["alpha"]:.2f}, β={fit["beta"]:.2f}, R²={fit["r2"]:.3f}')
        
        ax.set_xlabel('zg')
        ax.set_ylabel('P(zg)')
        ax.set_title(f'{subclass_names[sc]} (N={mask.sum():,})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "ap_splitting_fit.png"), dpi=150)
    plt.close()
    print("\n  Saved: ap_splitting_fit.png")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Subclass':15s} {'α (fit)':>10s} {'α (theory)':>12s} {'Δα':>10s} {'R²':>8s}")
    for sc in range(4):
        name = subclass_names[sc]
        if name in results:
            a = results[name]['alpha']
            t = theory_alphas[sc]
            print(f"  {name:15s} {a:>10.3f} {t:>12.1f} {a-t:>10.3f} {results[name]['r2']:>8.4f}")
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")
    print("[COMPLETE]")


if __name__ == '__main__':
    main()