"""
POC: OOD Generalization Test (Fixed JetNet Conversion)
=======================================================
Tests whether the trained ArcFace model distinguishes OOD physics processes
(W, Z, gluon, light quark, top) in its embedding space.

FIX: JetNet stores particle features as [etarel, phirel, ptrel, mask] —
NOT (px, py, pz, E). We read H5 directly and reconstruct absolute
4-momentum from relative coordinates.

Reconstruction (per particle):
    pt_part = ptrel * jet_pt
    eta_part = jet_eta + etarel
    phi_part = phirel (phi_jet=0, rotation invariant)
    px = pt_part * cos(phi_part)
    py = pt_part * sin(phi_part)
    pz = pt_part * sinh(eta_part)
    E  = pt_part * cosh(eta_part)  (massless approximation)
"""
import os, sys, time, json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcefn.utils.paths import EXPERIMENTS, CHECKPOINT, DATA_DIR, get_project_root
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel

dev, backend = get_device()
device = dev
print(f"Device: {device} ({backend})")

model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Model loaded from {CHECKPOINT}")

N_PAD = 200
JETS_PER_CLASS = 50000

JETNET_DIR = get_project_root() / 'data' / 'jetnet'
class_config = [
    ('g', 'gluon'),
    ('q', 'light quark'),
    ('w', 'W boson'),
    ('z', 'Z boson'),
    ('t', 'top quark'),
]
REF_JETNET_TO_TT = {
    'g': ('gluon',    0.0, 0.0),
    'q': ('light_q',  0.0, 0.0),
    'w': ('W boson',  0.0, 0.0),
    'z': ('Z boson',  0.0, 0.0),
    't': ('top',      0.0, 0.0),
}

def reconstruct_4momenta(jets, parts):
    """Reconstruct (E, px, py, pz) from JetNet relative coordinates.

    Parameters
    ----------
    jets : ndarray (N, 4)
        Jet features: [pt, eta, mass, num_particles]
    parts : ndarray (N, 30, 4)
        Particle features: [etarel, phirel, ptrel, mask]

    Returns
    -------
    ndarray (N, N_PAD, 4)
        Padded 4-momentum: (E, px, py, pz) per particle
    ndarray (N, N_PAD)
        Mask: 1 = real particle, 0 = padding
    """
    N = len(jets)
    jet_pt = jets[:, 0:1]          # (N, 1)
    jet_eta = jets[:, 1:2]         # (N, 1)

    pt_part = parts[:, :, 2] * jet_pt         # (N, 30)  ptrel * jet_pt
    eta_part = jet_eta + parts[:, :, 0]        # (N, 30)  jet_eta + etarel
    phi_part = parts[:, :, 1]                  # (N, 30)  phirel (phi_jet=0)
    mask_30 = parts[:, :, 3]                   # (N, 30)

    px = pt_part * np.cos(phi_part)
    py = pt_part * np.sin(phi_part)
    pz = pt_part * np.sinh(eta_part)
    E  = pt_part * np.cosh(eta_part)

    result = np.zeros((N, N_PAD, 4), dtype=np.float32)
    mask = np.zeros((N, N_PAD), dtype=np.float32)
    result[:, :30, 0] = E
    result[:, :30, 1] = px
    result[:, :30, 2] = py
    result[:, :30, 3] = pz
    mask[:, :30] = mask_30
    return result, mask

all_embeddings = {}
all_cos_sim = {}
total_time = 0

for jtype, cname in class_config:
    t0 = time.time()
    h5_path = JETNET_DIR / f'{jtype}.hdf5'
    if not h5_path.exists():
        print(f"  SKIP {jtype}/{cname}: {h5_path} not found")
        continue

    with h5py.File(str(h5_path), 'r') as f:
        jets = f['jet_features'][:]          # (N, 4)
        parts = f['particle_features'][:]    # (N, 30, 4)

    n_total = len(jets)
    indices = np.random.choice(n_total, min(JETS_PER_CLASS, n_total), replace=False)
    jets_sel = jets[indices]
    parts_sel = parts[indices]

    x_np, mask_np = reconstruct_4momenta(jets_sel, parts_sel)

    # Find actual particle counts per jet
    n_parts = parts_sel[:, :, 3].sum(axis=1)
    print(f"  {jtype} ({cname}): {len(indices)} jets, avg particles={n_parts.mean():.0f}")

    x_t = torch.from_numpy(x_np).to(device)
    mask_t = torch.from_numpy(mask_np).to(device)

    with torch.no_grad():
        logits, similarities, emb = model(x_t, mask=mask_t)

    all_embeddings[cname] = emb.cpu().numpy()
    all_cos_sim[cname] = similarities.cpu().numpy()

    elapsed = time.time() - t0
    total_time += elapsed
    print(f"    Done in {elapsed:.0f}s")

print(f"\nTotal: {sum(len(v) for v in all_embeddings.values())} jets in {total_time:.0f}s")

# Plot: cos_sim_QCD vs cos_sim_Top
fig, ax = plt.subplots(figsize=(10, 8))
colors = {'gluon': '#1f77b4', 'light quark': '#ff7f0e', 'W boson': '#2ca02c',
          'Z boson': '#d62728', 'top quark': '#9467bd'}
markers = {'gluon': 'o', 'light quark': 's', 'W boson': '^',
           'Z boson': 'D', 'top quark': 'v'}

for cname in all_cos_sim:
    cs = all_cos_sim[cname]
    ax.scatter(cs[:, 0], cs[:, 1], c=colors[cname], marker=markers[cname],
               s=5, alpha=0.4, label=cname, edgecolors='none')

ax.plot(1.0, 0.0, 'k*', markersize=12, label='$W_{\\mathrm{QCD}}$', zorder=5)
ax.plot(0.0, 1.0, 'k*', markersize=12, label='$W_{\\mathrm{Top}}$', zorder=5)
ax.set_xlabel('Cosine similarity to $W_{\\mathrm{QCD}}$', fontsize=14)
ax.set_ylabel('Cosine similarity to $W_{\\mathrm{Top}}$', fontsize=14)
ax.set_title('OOD Generalization: ArcFace Cosine Similarity Space', fontsize=15)
ax.legend(fontsize=11, markerscale=1.5, loc='center right',
          framealpha=0.9, edgecolor='gray')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.15, 1.05)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(str(EXPERIMENTS / 'poc_ood_similarity.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {EXPERIMENTS / 'poc_ood_similarity.png'}")

# Summary table
print(f"\n{'='*80}")
print(f"{'Class':<15} {'N':<6} {'cos_QCD_mean':<14} {'cos_QCD_std':<14} {'cos_Top_mean':<14} {'cos_Top_std':<14} {'margin':<10}")
print('-'*80)
for cname in class_config:
    cn = cname[1]
    if cn not in all_cos_sim: continue
    cs = all_cos_sim[cn]
    margin = cs[:, 1] - cs[:, 0]
    print(f"{cn:<15} {len(cs):<6} {cs[:,0].mean():<14.4f} {cs[:,0].std():<14.4f} "
          f"{cs[:,1].mean():<14.4f} {cs[:,1].std():<14.4f} {margin.mean():<10.4f}")

# Save results
results = {}
for cn in all_cos_sim:
    cs = all_cos_sim[cn]
    margin = cs[:, 1] - cs[:, 0]
    results[cn] = {
        'n': int(len(cs)),
        'cos_qcd_mean': float(cs[:, 0].mean()),
        'cos_qcd_std': float(cs[:, 0].std()),
        'cos_top_mean': float(cs[:, 1].mean()),
        'cos_top_std': float(cs[:, 1].std()),
        'margin_mean': float(margin.mean()),
        'margin_std': float(margin.std()),
        'frac_top': float((cs[:, 1] > cs[:, 0]).mean()),
    }

with open(str(EXPERIMENTS / 'poc_ood_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {EXPERIMENTS / 'poc_ood_results.json'}")
print("\n[POC COMPLETE]")
