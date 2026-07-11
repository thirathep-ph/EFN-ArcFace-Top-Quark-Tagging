"""
POC: OOD Generalization Test
=============================
Tests whether the trained ArcFace model distinguishes OOD physics processes
(W, Z, gluon, light quark) in its embedding space, even though it was
trained only on binary Top vs QCD.

Key question: Does the model's hyperspherical embedding space encode
physically meaningful structure beyond the binary decision boundary?
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# Fix Windows CP1252 encoding for jetnet download progress bar
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcjet.utils.paths import EXPERIMENTS, CHECKPOINT, MODEL_DIR, DATA_DIR
from arcjet.utils.device import get_device
from arcjet.models.efn_arcface import TopTaggingModel

dev, backend = get_device()
device = dev
print(f"Device: {device} ({backend})")

# Load trained model
model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Model loaded from {CHECKPOINT}")

# Extract class centers
w_top = model.arcface_head.class_centers.detach().cpu().numpy()  # (2, 64)
w_top_norm = w_top / np.linalg.norm(w_top, axis=1, keepdims=True)
print(f"Class centers: QCD={w_top_norm[0,:3]}..., Top={w_top_norm[1,:3]}...")
print(f"Angular distance: {np.arccos(np.clip(np.dot(w_top_norm[0], w_top_norm[1]), -1, 1)):.4f} rad")

# Load JetNet 30 data (5 classes: g, q, W, Z, t)
print("\nLoading JetNet 30 data...")
import jetnet.datasets as jn
jets_per_class = 2000  # 2000 each = 10K total

class_names = ['gluon', 'light quark', 'W boson', 'Z boson', 'top quark']
all_embeddings = {}
all_cos_sim = {}
total_time = 0

for i, jtype in enumerate(['g', 'q', 'w', 'z', 't']):
    t0 = time.time()
    ds = jn.JetNet(jet_type=jtype, split='test', num_particles=30, download=True)
    
    # Take up to jets_per_class
    n = min(jets_per_class, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)
    
    all_embs = []
    all_cos = []
    for idx in indices:
        parts, label = ds[idx]
        parts = np.asarray(parts, dtype=np.float64)
        # parts shape: (30, 4) = (px, py, pz, E)
        # Convert to our format: need E, px, py, pz
        px, py, pz, E = parts[:, 0], parts[:, 1], parts[:, 2], parts[:, 3]
        
        # Pad to 200 constituents
        N = 200
        E_pad = np.zeros(N)
        px_pad = np.zeros(N)
        py_pad = np.zeros(N)
        pz_pad = np.zeros(N)
        mask = np.zeros(N, dtype=np.float32)
        
        nparts = len(parts)
        E_pad[:nparts] = E.copy()
        px_pad[:nparts] = px.copy()
        py_pad[:nparts] = py.copy()
        pz_pad[:nparts] = pz.copy()
        mask[:nparts] = 1.0
        
        # Stack as (200, 4)
        x = np.stack([E_pad, px_pad, py_pad, pz_pad], axis=1)  # (200, 4)
        
        # Move to device
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, similarities, emb = model(x_t, mask=mask_t)
        
        emb_np = emb.cpu().numpy()[0]  # (64,)
        sim_np = similarities.cpu().numpy()[0]  # (2,)
        
        all_embs.append(emb_np)
        all_cos.append(sim_np)
    
    all_embeddings[class_names[i]] = np.array(all_embs)
    all_cos_sim[class_names[i]] = np.array(all_cos)
    elapsed = time.time() - t0
    total_time += elapsed
    print(f"  {jtype} ({class_names[i]}): {len(all_embs)} jets in {elapsed:.0f}s")

print(f"\nTotal: {sum(len(v) for v in all_embeddings.values())} jets in {total_time:.0f}s")

# Plot: 2D scatter of cos_sim_QCD vs cos_sim_Top
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']
legend_entries = []

for i, name in enumerate(class_names):
    cs = all_cos_sim[name]
    ax.scatter(cs[:, 0], cs[:, 1], c=colors[i], marker=markers[i],
               s=5, alpha=0.4, label=name, edgecolors='none')

# Add class centers
ax.plot(1.0, 0.0, 'k*', markersize=20, label='w_QCD (prototype)', zorder=5)
ax.plot(0.0, 1.0, 'k*', markersize=20, label='w_Top (prototype)', zorder=5)

ax.set_xlabel('cos similarity to w_QCD', fontsize=14)
ax.set_ylabel('cos similarity to w_Top', fontsize=14)
ax.set_title('OOD Generalization: ArcFace Embedding Similarity Space', fontsize=15)
ax.legend(fontsize=11, markerscale=3, loc='upper left')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(str(EXPERIMENTS / 'poc_ood_similarity.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {EXPERIMENTS / 'poc_ood_similarity.png'}")

# Compute mean+std per class
print("\n=== OOD Similarity Summary ===")
print(f"{'Class':<15} {'N':<6} {'cos_QCD_mean':<14} {'cos_QCD_std':<14} {'cos_Top_mean':<14} {'cos_Top_std':<14} {'margin':<10}")
print("-" * 80)
for name in class_names:
    cs = all_cos_sim[name]
    margin = cs[:, 1] - cs[:, 0]
    print(f"{name:<15} {len(cs):<6} {cs[:,0].mean():<14.4f} {cs[:,0].std():<14.4f} "
          f"{cs[:,1].mean():<14.4f} {cs[:,1].std():<14.4f} {margin.mean():<10.4f}")

# Summary statistics
print("\n=== Key OOD Metrics ===")
metric_data = {}
for name in class_names:
    cs = all_cos_sim[name]
    # Fraction classified as 'top' (= cosine to w_Top > cosine to w_QCD)
    frac_top = (cs[:, 1] > cs[:, 0]).mean()
    # Mean confidence (abs difference)
    avg_conf = np.abs(cs[:, 1] - cs[:, 0]).mean()
    metric_data[name] = {'frac_top': float(frac_top), 'avg_confidence': float(avg_conf)}
    print(f"  {name:<15}: P(pred=top)={frac_top:.3f}, avg|margin|={avg_conf:.4f}")

# Save results
results = {
    'n_jets_per_class': jets_per_class,
    'similarity_by_class': {name: {'mean_qcd': float(all_cos_sim[name][:,0].mean()),
                                   'std_qcd': float(all_cos_sim[name][:,0].std()),
                                   'mean_top': float(all_cos_sim[name][:,1].mean()),
                                   'std_top': float(all_cos_sim[name][:,1].std())}
                           for name in class_names},
    'metrics': metric_data
}
with open(str(EXPERIMENTS / 'poc_ood_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {EXPERIMENTS / 'poc_ood_results.json'}")
print("\n[POC COMPLETE]")
