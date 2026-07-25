"""
Compare subclass physics: ArcFace vs Linear head
=================================================
Hypothesis: ArcFace (hypersphere S^63) produces subclasses
that match DGLAP theory; Linear (unconstrained) does not.

Output
------
* experiments/head_comparison_params.json — fit results
* experiments/head_comparison.png         — 2×4 comparison figure
"""

import json, gc, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcjet.utils.paths import EXPERIMENTS, DATA_DIR
from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.models.efn_linear import TopTaggingLinear
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device
from arcjet.utils.clustering import spectral_clustering_subclass
from arcjet.utils.physics import compute_features

TEST_H5 = str(DATA_DIR / 'test.h5')
N_EVENTS = 50000
BATCH_SIZE = 1024

NAME_MAP = {0: 'QCD_Core', 1: 'QCD_Edge', 2: 'Top_Core', 3: 'Top_Edge'}

def dglap_power_law(z, alpha, beta):
    z = np.clip(z, 1e-10, 1 - 1e-10)
    return z**alpha * (1 - z)**beta

def get_embeddings(model, loader, device):
    model.eval()
    all_emb, all_labels = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            out = model(x, mask=m)
            emb = out[-1]
            all_emb.append(emb.cpu().numpy())
            all_labels.append(y.numpy())
    return np.vstack(all_emb), np.concatenate(all_labels)

def assign_subclasses(emb, labels):
    sub = np.full(len(labels), -1, dtype=int)
    for cls, offset in [(0, 0), (1, 2)]:
        mask = labels == cls
        if mask.sum() < 10:
            continue
        sl, k = spectral_clustering_subclass(emb[mask], max_k=5)
        sub[mask] = sl + offset
    return sub

def compute_zg_safe(events):
    n = len(events)
    zg = np.full(n, -1.0)
    chunk = 5000
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        try:
            feats = compute_features(events[i:end])
            zg[i:end] = feats[:, 8]
        except Exception:
            pass
    return zg

def fit_subclass(zg):
    valid = (zg > 0.1) & (zg < 0.5)
    v = zg[valid]
    if len(v) < 50:
        return None
    hist, edges = np.histogram(v, bins=30, density=True)
    bc = (edges[:-1] + edges[1:]) / 2
    try:
        popt, _ = curve_fit(dglap_power_law, bc, hist, p0=[-1.0, 0.0],
                           bounds=[[-5, -10], [10, 10]], maxfev=10000)
        res = hist - dglap_power_law(bc, *popt)
        r2 = 1 - np.sum(res**2) / (np.sum((hist - np.mean(hist))**2) + 1e-12)
        return {'alpha': float(popt[0]), 'beta': float(popt[1]),
                'R2': float(r2), 'N': int(len(v))}
    except Exception:
        return None

def main():
    device, _ = get_device()
    print(f"Device: {device}")

    print(f"Loading {N_EVENTS:,} events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Computing zg (once, shared)...")
    t0 = time.time()
    zg = compute_zg_safe(events)
    print(f"  Done in {time.time()-t0:.0f}s")

    models = [
        ('ArcFace', TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device),
         EXPERIMENTS / 'robustness_s16_m05' / 'checkpoint.pt'),
        ('Linear', TopTaggingLinear(embedding_dim=64, particle_dim=128).to(device),
         EXPERIMENTS / 'baselines_linear_seed_42' / 'checkpoint.pt'),
    ]

    all_results = {}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for row, (name, model, ckpt_path) in enumerate(models):
        print(f"\n--- {name} ---")
        t0 = time.time()
        sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.eval()
        emb, y = get_embeddings(model, loader, device)
        del model; gc.collect()
        print(f"  Emb: {emb.shape} in {time.time()-t0:.0f}s")

        sub = assign_subclasses(emb, y)
        n_sc = len(set(sub[sub >= 0]))
        print(f"  Subclasses: {n_sc}")

        results = {}
        for sc_id in range(4):
            mask = sub == sc_id
            n = mask.sum()
            if n < 10:
                continue
            fit = fit_subclass(zg[mask])
            sc_name = NAME_MAP[sc_id]
            if fit:
                results[sc_name] = fit
                print(f"  {sc_name}: alpha={fit['alpha']:.3f}, beta={fit['beta']:.3f}, R2={fit['R2']:.4f}, N={n}")
            else:
                print(f"  {sc_name}: fit failed, N={n}")

            # Plot
            ax = axes[row, sc_id]
            v = zg[mask]
            valid = (v > 0.1) & (v < 0.5)
            v = v[valid]
            if len(v) < 10:
                ax.text(0.5, 0.5, 'N<10', ha='center', va='center')
                continue
            ax.hist(v, bins=30, density=True, alpha=0.4, color=colors[sc_id],
                   label=f'{sc_name} (N={len(v)})')
            if fit:
                z_grid = np.linspace(0.1, 0.5, 100)
                ax.plot(z_grid, dglap_power_law(z_grid, fit['alpha'], fit['beta']),
                       '-', color=colors[sc_id], linewidth=2,
                       label=f'a={fit["alpha"]:.2f}, R2={fit["R2"]:.3f}')
            ax.set_xlabel('$z_g$')
            if sc_id == 0:
                ax.set_ylabel('P($z_g$)')
            ax.set_title(f'{name}: {sc_name}')
            ax.legend(fontsize=6)

        all_results[name] = results

    plt.tight_layout()
    out_png = str(EXPERIMENTS / 'head_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    with open(str(EXPERIMENTS / 'head_comparison_params.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Saved: head_comparison_params.json")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Subclass':<12} {'ArcFace a':<12} {'ArcFace R2':<12} {'Linear a':<12} {'Linear R2':<12}")
    print(f"{'-'*60}")
    for sc_name in NAME_MAP.values():
        af = all_results.get('ArcFace', {}).get(sc_name, {})
        ln = all_results.get('Linear', {}).get(sc_name, {})
        a_af = f"{af.get('alpha', -1):.3f}" if af else 'N/A'
        r_af = f"{af.get('R2', 0):.4f}" if af else 'N/A'
        a_ln = f"{ln.get('alpha', -1):.3f}" if ln else 'N/A'
        r_ln = f"{ln.get('R2', 0):.4f}" if ln else 'N/A'
        print(f"{sc_name:<12} {a_af:<12} {r_af:<12} {a_ln:<12} {r_ln:<12}")
    print(f"{'='*70}")

    print("\n[DONE]")

if __name__ == '__main__':
    main()
