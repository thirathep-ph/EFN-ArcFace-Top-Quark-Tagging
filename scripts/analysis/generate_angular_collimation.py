"""
Generate Angular Collimation Figure: ΔR ≈ 2M/pT scaling per subclass
====================================================================
Uses a single 15k-event sample for both physics features + clustering.
"""
import os, gc, time, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from arcjet.utils.paths import DATA_DIR, MODEL_DIR
from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device
from arcjet.utils.physics import compute_features
from arcjet.utils.clustering import spectral_clustering_subclass

TEST_H5 = str(DATA_DIR / 'test.h5')
N_EVENTS = 404000  # full test set
BATCH_SIZE = 1024

def main():
    device = get_device()[0]
    print(f"Device: {device}")

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    ckpt = str(MODEL_DIR / 'checkpoint.pt')
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    centers = sd['arcface_head.class_centers'].cpu().numpy()
    centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
    del sd; gc.collect()

    print(f"Loading {N_EVENTS} test events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS, lazy=False)
    labels = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    print(f"  Loaded {len(events)} events")

    print("Computing physics features (FastJet)...")
    t0 = time.time()
    phys = compute_features(events)
    print(f"  Done in {time.time()-t0:.0f}s, shape={phys.shape}")

    mass = phys[:, 0]
    theta_g = phys[:, 9]
    pt = np.sqrt(np.sum(events.px, axis=1)**2 + np.sum(events.py, axis=1)**2)
    pt = np.array(pt)
    two_m_pt = 2.0 * mass / np.clip(pt, 1.0, None)

    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_emb, all_preds = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            logits, emb = out[0], out[2]
            all_emb.append(emb.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    embeddings = np.vstack(all_emb)
    predictions = np.concatenate(all_preds)

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    print("Spectral clustering per predicted class...")
    subclass_labels = np.full(len(predictions), -1, dtype=int)
    top_mask = predictions == 1
    if top_mask.sum() > 0:
        top_sc, k_top = spectral_clustering_subclass(emb_norm[top_mask], max_k=10)
        active_top, counts_top = np.unique(top_sc, return_counts=True)
        subclass_labels[top_mask] = top_sc + 1
        print(f"  Top subclasses: {dict(zip(active_top, counts_top))} k={k_top}")
    qcd_mask = predictions == 0
    if qcd_mask.sum() > 0:
        qcd_sc, k_qcd = spectral_clustering_subclass(emb_norm[qcd_mask], max_k=10)
        active_qcd, counts_qcd = np.unique(qcd_sc, return_counts=True)
        subclass_labels[qcd_mask] = -(qcd_sc + 1)
        print(f"  QCD subclasses: {dict(zip(active_qcd, counts_qcd))} k={k_qcd}")

    offset = 0
    sc_map = np.full(len(predictions), -1, dtype=int)
    if len(active_qcd) > 0:
        for i, sc in enumerate(active_qcd):
            sc_map[qcd_mask & (subclass_labels == -(sc + 1))] = offset + i
        offset += len(active_qcd)
    if len(active_top) > 0:
        for i, sc in enumerate(active_top):
            sc_map[top_mask & (subclass_labels == (sc + 1))] = offset + i
    sc_map_final = sc_map

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45']
    top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5']

    max_plot = 25000  # subsample plot for clarity, full N used in label
    rng = np.random.RandomState(42)
    for i, sc_idx in enumerate(active_qcd):
        mask = (sc_map_final == (i)) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx_plot = rng.choice(np.where(mask)[0], min(max_plot, mask.sum()), replace=False)
        label = f"QCD Subclass {sc_idx} (N={mask.sum()})"
        axes[0].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                       color=qcd_colors[i % len(qcd_colors)], label=label)

    for i, sc_idx in enumerate(active_top):
        idx = len(active_qcd) + i
        mask = (sc_map_final == idx) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx_plot = rng.choice(np.where(mask)[0], min(max_plot, mask.sum()), replace=False)
        label = f"Top Subclass {sc_idx} (N={mask.sum()})"
        axes[1].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                       color=top_colors[i % len(top_colors)], label=label)

    for ax, lbl in [(axes[0], "QCD Subclasses"), (axes[1], "Top Subclasses")]:
        ax.plot([0, 1.5], [0, 1.5], 'k--', lw=1.5, label=r'$\Delta R = 2M/p_T$')
        ax.set_xlabel(r"$2M/p_T$ (predicted $\Delta R$)")
        ax.set_ylabel(r"$\theta_g$ (measured)")
        ax.set_title(lbl)
        ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.5)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15)

    plt.suptitle(r"Relativistic Angular Scaling: $\theta_g \approx 2M/p_T$", fontsize=13, y=1.02)
    plt.tight_layout()
    out = MODEL_DIR / 'angular_collimation.png'
    plt.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    paper_fig = os.path.join(os.path.dirname(os.path.dirname(str(MODEL_DIR))), 'paper', 'figures', 'angular_collimation.png')
    from shutil import copy2
    copy2(str(out), paper_fig)
    print(f"Copied: {paper_fig}")

if __name__ == '__main__':
    main()
