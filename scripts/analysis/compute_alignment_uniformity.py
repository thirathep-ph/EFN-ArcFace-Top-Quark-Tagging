"""
Compare embedding geometry: ArcFace vs Linear head.
====================================================
Key insight: ArcFace forces all information into angular (directional)
degrees of freedom on S^63. Linear head can use norm as a proxy for
confidence, which trivializes clustering.

Output
------
* experiments/embedding_geometry_results.json
* experiments/embedding_geometry.png    — 2-panel figure
"""

import json, gc, time
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcjet.utils.paths import EXPERIMENTS, DATA_DIR
from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.models.efn_linear import TopTaggingLinear
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device
from arcjet.utils.clustering import spectral_clustering_subclass

TEST_H5 = str(DATA_DIR / 'test.h5')
N_EVENTS = 50000
BATCH_SIZE = 1024


def get_raw_embeddings(model, loader, device):
    """Get embeddings WITHOUT L2 normalization."""
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


def main():
    device, _ = get_device()
    print(f"Device: {device}")
    print(f"Loading {N_EVENTS:,} events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    models = [
        ('ArcFace\n(s=16, m=0.5)', TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device),
         EXPERIMENTS / 'robustness_s16_m05' / 'checkpoint.pt'),
        ('Linear\n(cross-entropy)', TopTaggingLinear(embedding_dim=64, particle_dim=128).to(device),
         EXPERIMENTS / 'baselines_linear_seed_42' / 'checkpoint.pt'),
    ]

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ['#e41a1c', '#377eb8']

    for idx, (name, model, ckpt_path) in enumerate(models):
        print(f"\n--- {name} ---")
        t0 = time.time()
        sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.eval()
        emb_raw, y = get_raw_embeddings(model, loader, device)
        del model; gc.collect()
        norms = np.linalg.norm(emb_raw, axis=1)
        print(f"  Emb: {emb_raw.shape}, norms: min={norms.min():.4f}, median={np.median(norms):.4f}, max={norms.max():.4f}")

        # Pane 1: Norm histogram
        ax = axes[0]
        label_clean = name.replace('\n', ' ')
        if idx == 0:
            # ArcFace: all norms = 1.0 (tiny residual from numerics)
            ax.axvline(1.0, color=colors[idx], linestyle='--', linewidth=2, alpha=0.8, label=label_clean)
            ax.text(1.02, 0.9, f'{label_clean}\n(all = 1.0)', color=colors[idx], fontsize=8,
                    transform=ax.get_xaxis_transform(), va='top')
        else:
            ax.hist(norms, bins=80, color=colors[idx], alpha=0.6, density=True, label=label_clean)

        # Pane 2: Subclass balance via spectral clustering
        # Normalize for spectral clustering (standard practice)
        emb_norm = emb_raw / (norms[:, None] + 1e-12)
        sub = np.full(len(y), -1, dtype=int)
        for cls, offset in [(0, 0), (1, 2)]:
            mask = y == cls
            if mask.sum() < 10: continue
            sl, k = spectral_clustering_subclass(emb_norm[mask], max_k=5)
            sub[mask] = sl + offset

        counts = {}
        for sc_id, sc_name in [(0, 'QCD_Core'), (1, 'QCD_Edge'), (2, 'Top_Core'), (3, 'Top_Edge')]:
            n = int(np.sum(sub == sc_id))
            counts[sc_name] = n
            results[f'{name.replace(chr(10), " ")}_{sc_name}_N'] = n

        results[f'{name.replace(chr(10), " ")}_norms_median'] = float(np.median(norms))
        results[f'{name.replace(chr(10), " ")}_norms_std'] = float(np.std(norms))

        ax = axes[1]
        sc_pos = np.arange(4)
        sc_vals = [counts['QCD_Core'], counts['QCD_Edge'],
                   counts['Top_Core'], counts['Top_Edge']]
        offset = idx * 0.3
        bars = ax.bar(sc_pos + offset, sc_vals, 0.3, color=colors[idx], alpha=0.8, label=label_clean)
        # Annotate percentages
        total = sum(sc_vals)
        for bar, val in zip(bars, sc_vals):
            pct = val / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                    f'{pct:.0f}%', ha='center', fontsize=7, rotation=90)

    # Format panel 1
    axes[0].set_title('Embedding Norm Distribution\n(ArcFace: all = 1.0, fixed)', fontsize=10)
    axes[0].set_xlabel('||e||₂', fontsize=9)
    axes[0].set_ylabel('Density', fontsize=9)
    axes[0].set_xlim(0, 6)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.15)

    # Format panel 2
    axes[1].set_title('Subclass Balance\n(ArcFace: balanced; Linear: unbalanced)', fontsize=10)
    axes[1].set_ylabel('N events (50k total)', fontsize=9)
    axes[1].set_xticks(np.arange(4) + 0.15)
    axes[1].set_xticklabels(['QCD\nCore', 'QCD\nEdge', 'Top\nCore', 'Top\nEdge'], fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    out_png = str(EXPERIMENTS / 'embedding_geometry.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    results_json = str(EXPERIMENTS / 'embedding_geometry_results.json')
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_json}")

    print("\n=== Key findings ===")
    for name_key in ['ArcFace (s=16, m=0.5)', 'Linear (cross-entropy)']:
        mn = results.get(f'{name_key}_norms_median', '?')
        sd = results.get(f'{name_key}_norms_std', '?')
        print(f"  {name_key}: norm median={mn}, std={sd}")
        for sc in ['QCD_Core', 'QCD_Edge', 'Top_Core', 'Top_Edge']:
            n = results.get(f'{name_key}_{sc}_N', 0)
            print(f"    {sc}: {n}")


if __name__ == '__main__':
    main()
