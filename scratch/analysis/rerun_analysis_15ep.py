"""
Re-run full latent space analysis using checkpoints/test_run_15ep/best_model_efn.pt
Architecture: embedding_dim=64, particle_dim=128
Handles old `prototype_class` key with strict=False
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import awkward as ak
import umap
from sklearn.mixture import BayesianGaussianMixture

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcjet.models.efn_arcface import TopTaggingModel, get_efn_features
from arcjet.utils.physics import compute_features
from arcjet.data.loader import load_awkward, JetTaggingDataset
from torch.utils.data import DataLoader

# ============================================================================
# Config
# ============================================================================
CHECKPOINT    = 'checkpoints/test_run_15ep/best_model_efn.pt'
DATA_PATH     = 'data/top_tagging/test.h5'
EMB_SAVE_PATH = 'results/embeddings_15ep.npz'
RESULTS_DIR   = 'results/analysis_15ep'
LUND_DIR      = os.path.join(RESULTS_DIR, 'lund_planes')
MAX_EVENTS    = 20000  # limit for speed on CPU

EMBEDDING_DIM = 64
PARTICLE_DIM  = 128

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LUND_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. Load model with architecture matching test_run_15ep
# ============================================================================
print(f"\nLoading checkpoint: {CHECKPOINT}")
model = TopTaggingModel(
    n_proto_per_class=1,
    s=16.0, m=0.0,
    embedding_dim=EMBEDDING_DIM,
    particle_dim=PARTICLE_DIM
).to(device)

state_dict = torch.load(CHECKPOINT, map_location=device)
# Remove legacy key that no longer exists in the new ArcFacePrototypeLayer
state_dict.pop('prototype_layer.prototype_class', None)
model.load_state_dict(state_dict, strict=True)
model.eval()
print("Model loaded successfully.")

# Load class prototypes
prototypes = model.prototype_layer.prototypes.detach().cpu().numpy()  # (2, EMBEDDING_DIM)
prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10)

# ============================================================================
# 2. Generate embeddings from test set
# ============================================================================
print(f"\nLoading test data from: {DATA_PATH}")
te_ev, te_lab, te_w = load_awkward(DATA_PATH, max_events=MAX_EVENTS)
test_ds = JetTaggingDataset(te_ev, te_lab, te_w)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
N = len(test_ds)
print(f"Test set size: {N}")

print("Generating embeddings...")
all_emb_list = []
all_labels_list = []
with torch.no_grad():
    for x, y, w, m, idx in test_loader:
        x, m = x.to(device), m.to(device)
        coords, z = get_efn_features(x, m)
        jet_emb = model.efn(coords, z, m)
        all_emb_list.append(jet_emb.cpu().numpy())
        all_labels_list.append(y.numpy())

all_emb = np.vstack(all_emb_list)
labels   = np.concatenate(all_labels_list)
print(f"Embeddings shape: {all_emb.shape}")

np.savez(EMB_SAVE_PATH, embeddings=all_emb, labels=labels)
print(f"Saved embeddings to {EMB_SAVE_PATH}")

# ============================================================================
# 3. Normalize & compute predictions
# ============================================================================
all_emb_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10)

cosine_similarities = np.dot(all_emb_norm, prototypes_norm.T)  # (N, 2)
preds = np.argmax(cosine_similarities, axis=1)

tp = (preds == 1) & (labels == 1)
tn = (preds == 0) & (labels == 0)
fp = (preds == 1) & (labels == 0)
fn = (preds == 0) & (labels == 1)

status = np.empty(N, dtype=object)
status[tp] = 'True Positive'
status[tn] = 'True Negative'
status[fp] = 'False Positive'
status[fn] = 'False Negative'

acc = (tp.sum() + tn.sum()) / N
print(f"\nDataset stats: TP={tp.sum()}, TN={tn.sum()}, FP={fp.sum()}, FN={fn.sum()}")
print(f"Test Accuracy: {acc:.4f}")

# ============================================================================
# 4. UMAP projection
# ============================================================================
print("\nFitting UMAP (metric=cosine)...")
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
umap_coords = reducer.fit_transform(all_emb_norm)
proto_umap = reducer.transform(prototypes_norm)
print("UMAP done.")

# ============================================================================
# 5. DPGMM subclass discovery (in 64D normalized latent space)
# ============================================================================
subclass_labels = np.full(N, -1, dtype=int)
min_samples = 10

# Top (preds==1)
top_mask = (preds == 1)
if top_mask.sum() >= min_samples:
    print(f"Fitting DPGMM on {top_mask.sum()} Top predictions...")
    top_latent = all_emb_norm[top_mask]
    dpgmm_top = BayesianGaussianMixture(
        n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2
    )
    top_subclasses = dpgmm_top.fit_predict(top_latent)
    active_top_sc, counts_top = np.unique(top_subclasses, return_counts=True)
    print(f"  Active Top subclasses: {active_top_sc}, counts: {counts_top}")
    subclass_labels[top_mask] = top_subclasses
else:
    print(f"Too few Top predictions ({top_mask.sum()}), skipping DPGMM.")
    active_top_sc = np.array([])
    dpgmm_top = None

# QCD (preds==0)
qcd_mask = (preds == 0)
if qcd_mask.sum() >= min_samples:
    print(f"Fitting DPGMM on {qcd_mask.sum()} QCD predictions...")
    qcd_latent = all_emb_norm[qcd_mask]
    dpgmm_qcd = BayesianGaussianMixture(
        n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2
    )
    qcd_subclasses = dpgmm_qcd.fit_predict(qcd_latent)
    active_qcd_sc, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
    print(f"  Active QCD subclasses: {active_qcd_sc}, counts: {counts_qcd}")
    subclass_labels[qcd_mask] = qcd_subclasses
else:
    print(f"Too few QCD predictions ({qcd_mask.sum()}), skipping DPGMM.")
    active_qcd_sc = np.array([])
    dpgmm_qcd = None

# ============================================================================
# 6. Load physical events & compute physics features
# ============================================================================
print("\nLoading physical events for FastJet...")
all_ev, _, _ = load_awkward(DATA_PATH, max_events=N)
all_ev = all_ev[:N]

# Stratified sampling
sampled_indices = []
for sc in active_top_sc:
    idx = np.where((preds == 1) & (subclass_labels == sc))[0]
    sampled_indices.extend(np.random.choice(idx, min(1000, len(idx)), replace=False))
for sc in active_qcd_sc:
    idx = np.where((preds == 0) & (subclass_labels == sc))[0]
    sampled_indices.extend(np.random.choice(idx, min(1000, len(idx)), replace=False))
for cat in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
    idx = np.where(status == cat)[0]
    if len(idx) > 0:
        sampled_indices.extend(np.random.choice(idx, min(1000, len(idx)), replace=False))

test_idx = np.unique(sampled_indices) if sampled_indices else np.random.choice(N, min(5000, N), replace=False)
print(f"Stratified sampling: {len(test_idx)} jets selected for FastJet")

sub_ev       = all_ev[test_idx]
sub_preds    = preds[test_idx]
sub_subclass = subclass_labels[test_idx]
sub_status   = status[test_idx]
sub_labels   = labels[test_idx]

print("Running FastJet feature calculations...")
features = compute_features(sub_ev)
# Feature order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
mass     = features[:, 0]   # Jet mass (GeV)
tau32    = features[:, 7]   # N-subjettiness ratio tau3/tau2
d12      = features[:, 4]   # sqrt(d12) splitting scale (GeV)
zg       = features[:, 8]   # Groomed momentum fraction (AP splitting variable)
theta_g  = features[:, 9]   # Delta_R of leading SoftDrop split (angular collimation)
print(f"  Features extracted: mass mean={mass.mean():.1f} GeV, zg mean={zg.mean():.3f}, theta_g mean={theta_g.mean():.3f}")

# ============================================================================
# Plots
# ============================================================================

# --- Plot 1: UMAP Prediction Status ---
print("\nPlot 1: UMAP Prediction Status")
plt.figure(figsize=(10, 8))
colors  = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4',
           'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
markers = {'True Negative': 'o', 'True Positive': 'o',
           'False Positive': 'x', 'False Negative': 'x'}
sizes   = {'True Negative': 10, 'True Positive': 10,
           'False Positive': 60, 'False Negative': 60}
alphas  = {'True Negative': 0.25, 'True Positive': 0.25,
           'False Positive': 0.9, 'False Negative': 0.9}

for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
    m = (status == cat)
    if m.sum() > 0:
        plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                    color=colors[cat], marker=markers[cat], s=sizes[cat],
                    alpha=alphas[cat], label=f"{cat} ({m.sum()})")

plt.scatter(proto_umap[0, 0], proto_umap[0, 1], marker='*', s=500,
            color='yellow', edgecolors='black', lw=1.5,
            label='W_QCD (Class Center)', zorder=10)
plt.scatter(proto_umap[1, 0], proto_umap[1, 1], marker='*', s=500,
            color='magenta', edgecolors='black', lw=1.5,
            label='W_Top (Class Center)', zorder=10)

try:
    for cat in ['True Negative', 'True Positive']:
        m = (status == cat)
        if m.sum() > 50:
            sns.kdeplot(x=umap_coords[m, 0], y=umap_coords[m, 1],
                        levels=4, color=colors[cat], alpha=0.3, linewidths=1.2)
except Exception as e:
    print(f"  KDE skipped: {e}")

plt.title("EFN + ArcFace Latent Space (UMAP) & Prediction Status [15ep]", fontsize=13)
plt.xlabel("UMAP Dimension 1"); plt.ylabel("UMAP Dimension 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.grid(True, alpha=0.15)
save = os.path.join(RESULTS_DIR, 'umap_prediction_status.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plot 2: UMAP Subclasses ---
print("Plot 2: UMAP Subclasses")
plt.figure(figsize=(10, 8))
qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']

for i, sc in enumerate(active_qcd_sc):
    m = (preds == 0) & (subclass_labels == sc)
    if m.sum() > 0:
        plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                    color=qcd_colors[i % 5], s=10, alpha=0.4,
                    label=f"QCD Subclass {i} (N={m.sum()})")
for i, sc in enumerate(active_top_sc):
    m = (preds == 1) & (subclass_labels == sc)
    if m.sum() > 0:
        plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                    color=top_colors[i % 5], s=10, alpha=0.4,
                    label=f"Top Subclass {i} (N={m.sum()})")

plt.scatter(proto_umap[0, 0], proto_umap[0, 1], marker='*', s=500,
            color='yellow', edgecolors='black', lw=1.5, label='W_QCD', zorder=10)
plt.scatter(proto_umap[1, 0], proto_umap[1, 1], marker='*', s=500,
            color='magenta', edgecolors='black', lw=1.5, label='W_Top', zorder=10)

plt.title("Discovered Subclasses in Latent Space (UMAP + DPGMM) [15ep]", fontsize=13)
plt.xlabel("UMAP Dimension 1"); plt.ylabel("UMAP Dimension 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.grid(True, alpha=0.15)
save = os.path.join(RESULTS_DIR, 'umap_subclasses.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plot 3: Physics Distributions ---
print("Plot 3: Physics Distributions")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Col 0: Top Subclasses
for i, sc in enumerate(active_top_sc):
    m = (sub_preds == 1) & (sub_subclass == sc)
    if m.sum() > 5:
        sns.kdeplot(mass[m],  ax=axes[0, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
        sns.kdeplot(tau32[m], ax=axes[1, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
        sns.kdeplot(d12[m],   ax=axes[2, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
axes[0, 0].set_title("Discovered Top Subclasses: Jet Mass"); axes[0, 0].set_xlabel("Mass (GeV)"); axes[0, 0].legend()
axes[1, 0].set_title("Discovered Top Subclasses: Tau32");   axes[1, 0].set_xlabel("Tau32");      axes[1, 0].legend()
axes[2, 0].set_title("Discovered Top Subclasses: sqrt(d12)");axes[2, 0].set_xlabel("sqrt(d12) (GeV)"); axes[2, 0].legend()

# Col 1: QCD TN vs FP
s_tn = (sub_status == 'True Negative'); s_fp = (sub_status == 'False Positive')
for feat, row, title, xlabel in [(mass, 0, "QCD Mistakes: Jet Mass", "Mass (GeV)"),
                                  (tau32, 1, "QCD Mistakes: Tau32", "Tau32"),
                                  (d12, 2, "QCD Mistakes: sqrt(d12)", "sqrt(d12) (GeV)")]:
    if s_tn.sum() > 5:
        sns.kdeplot(feat[s_tn], ax=axes[row, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.15)
    if s_fp.sum() > 5:
        sns.kdeplot(feat[s_fp], ax=axes[row, 1], label="Mistaken QCD (FP)", color='red', lw=2)
    axes[row, 1].set_title(title); axes[row, 1].set_xlabel(xlabel); axes[row, 1].legend()

# Col 2: Top TP vs FN
s_tp = (sub_status == 'True Positive'); s_fn = (sub_status == 'False Negative')
for feat, row, title, xlabel in [(mass, 0, "Top Mistakes: Jet Mass", "Mass (GeV)"),
                                  (tau32, 1, "Top Mistakes: Tau32", "Tau32"),
                                  (d12, 2, "Top Mistakes: sqrt(d12)", "sqrt(d12) (GeV)")]:
    if s_tp.sum() > 5:
        sns.kdeplot(feat[s_tp], ax=axes[row, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.15)
    if s_fn.sum() > 5:
        sns.kdeplot(feat[s_fn], ax=axes[row, 2], label="Mistaken Top (FN)", color='orange', lw=2)
    axes[row, 2].set_title(title); axes[row, 2].set_xlabel(xlabel); axes[row, 2].legend()

plt.tight_layout()
save = os.path.join(RESULTS_DIR, 'subclass_physics_distributions.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plot 4: Composition Heatmap ---
print("Plot 4: Subclass Composition Heatmap")
categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
comp_matrix = []
row_labels  = []

for i, sc in enumerate(active_top_sc):
    m = (preds == 1) & (subclass_labels == sc)
    n_tot = m.sum()
    if n_tot > 0:
        row = [((status == cat) & m).sum() / n_tot * 100 for cat in categories]
        comp_matrix.append(row); row_labels.append(f"Top SC {i} (N={n_tot})")
for i, sc in enumerate(active_qcd_sc):
    m = (preds == 0) & (subclass_labels == sc)
    n_tot = m.sum()
    if n_tot > 0:
        row = [((status == cat) & m).sum() / n_tot * 100 for cat in categories]
        comp_matrix.append(row); row_labels.append(f"QCD SC {i} (N={n_tot})")

if comp_matrix:
    plt.figure(figsize=(10, max(4, len(row_labels) * 0.7)))
    sns.heatmap(comp_matrix, annot=True, fmt=".1f",
                xticklabels=categories, yticklabels=row_labels,
                cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title("Subclass Classification Purity [15ep]", fontsize=13)
    plt.xlabel("Prediction Status"); plt.ylabel("Subclass")
    save = os.path.join(RESULTS_DIR, 'subclass_composition_heatmap.png')
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save}")

# --- Plot 5: Representative Jets (QCD & Top) ---
print("Plot 5: Representative Jets")
for cls_name, active_sc, cls_subclasses, cls_mask, dpgmm in [
    ('QCD', active_qcd_sc, qcd_subclasses if qcd_mask.sum() >= min_samples else None,
     qcd_mask, dpgmm_qcd),
    ('Top', active_top_sc, top_subclasses if top_mask.sum() >= min_samples else None,
     top_mask, dpgmm_top),
]:
    if dpgmm is None or cls_subclasses is None:
        continue
    cls_indices = np.where(cls_mask)[0]
    probs = dpgmm.predict_proba(all_emb_norm[cls_mask])
    for i, sc in enumerate(active_sc):
        assigned = (cls_subclasses == sc)
        if assigned.sum() == 0:
            continue
        p = probs[:, sc].copy()
        p[~assigned] = -1.0
        top5_rel = np.argsort(p)[::-1][:5]
        top5_abs = cls_indices[top5_rel]

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for j, t_idx in enumerate(top5_abs):
            ev = all_ev[t_idx]
            px = ak.to_numpy(ev.px); py = ak.to_numpy(ev.py)
            pz = ak.to_numpy(ev.pz); e  = ak.to_numpy(ev.E)
            pt  = np.sqrt(px**2 + py**2)
            eta = np.arcsinh(pz / (pt + 1e-10))
            phi = np.arctan2(py, px)
            z_e = e / (e.sum() + 1e-10)
            axs[j].scatter(eta, phi, s=pt*5, alpha=0.7, c=z_e,
                           cmap='plasma', edgecolors='black', lw=0.3)
            axs[j].set_xlabel(r'$\eta$'); axs[j].set_ylabel(r'$\phi$')
            axs[j].set_title(f"Jet {j+1} (p={p[top5_rel[j]]:.2f})", fontsize=9)
            axs[j].grid(True, alpha=0.15)
        plt.suptitle(f"{cls_name} Subclass {i} - Top 5 Representative Jets [15ep]",
                     fontsize=13, y=1.05)
        plt.tight_layout()
        save = os.path.join(RESULTS_DIR, f'{cls_name.lower()}_subclass_{i}_representative_jets.png')
        plt.savefig(save, dpi=120, bbox_inches='tight'); plt.close()
        print(f"  Saved: {save}")

# --- Plot 6: Cosine Similarity Distributions ---
print("Plot 6: Cosine Similarity Distributions")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, sc in enumerate(active_qcd_sc):
    m = (preds == 0) & (subclass_labels == sc)
    if m.sum() > 5:
        sns.kdeplot(cosine_similarities[m, 0], ax=axes[0],
                    label=f"Subclass {i}", fill=True, alpha=0.1)
axes[0].set_title("QCD Subclasses: Cosine Similarity to W_QCD")
axes[0].set_xlabel("Cosine Similarity"); axes[0].legend(); axes[0].grid(True, alpha=0.15)

for i, sc in enumerate(active_top_sc):
    m = (preds == 1) & (subclass_labels == sc)
    if m.sum() > 5:
        sns.kdeplot(cosine_similarities[m, 1], ax=axes[1],
                    label=f"Subclass {i}", fill=True, alpha=0.1)
axes[1].set_title("Top Subclasses: Cosine Similarity to W_Top")
axes[1].set_xlabel("Cosine Similarity"); axes[1].legend(); axes[1].grid(True, alpha=0.15)

plt.tight_layout()
save = os.path.join(RESULTS_DIR, 'subclass_similarity_distributions.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plot 7: Physics Correlation Matrices (now includes zg) ---
print("Plot 7: Physics Correlation Matrices")
import pandas as pd
n_top = len(active_top_sc); n_qcd = len(active_qcd_sc)
if n_top > 0 or n_qcd > 0:
    cols_n = max(n_top, n_qcd, 1)
    fig, axes = plt.subplots(2, cols_n, figsize=(4 * cols_n, 8))
    fnames = ["Mass", "Tau32", "sqrt(d12)", "zg"]
    for i, sc in enumerate(active_top_sc):
        m = (sub_preds == 1) & (sub_subclass == sc)
        ax = axes[0, i] if cols_n > 1 else axes[0]
        if m.sum() > 5:
            df = pd.DataFrame(np.stack([mass[m], tau32[m], d12[m], zg[m]], axis=1), columns=fnames)
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
        ax.set_title(f"Top SC {i}")
    for i in range(len(active_top_sc), cols_n):
        ax = axes[0, i] if cols_n > 1 else axes[0]
        ax.axis('off')
    for i, sc in enumerate(active_qcd_sc):
        m = (sub_preds == 0) & (sub_subclass == sc)
        ax = axes[1, i] if cols_n > 1 else axes[1]
        if m.sum() > 5:
            df = pd.DataFrame(np.stack([mass[m], tau32[m], d12[m], zg[m]], axis=1), columns=fnames)
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
        ax.set_title(f"QCD SC {i}")
    for i in range(len(active_qcd_sc), cols_n):
        ax = axes[1, i] if cols_n > 1 else axes[1]
        ax.axis('off')
    plt.suptitle("Physics Feature Correlation per Subclass (incl. zg) [15ep]", fontsize=13, y=1.02)
    plt.tight_layout()
    save = os.path.join(RESULTS_DIR, 'subclass_physics_correlations.png')
    plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {save}")

# --- Plot 10: zg (Altarelli-Parisi Splitting Variable) per Subclass ---
print("Plot 10: zg distributions (AP splitting verification)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: QCD subclasses — expect soft divergence: zg -> 0
for i, sc in enumerate(active_qcd_sc):
    m = (sub_preds == 0) & (sub_subclass == sc)
    if m.sum() > 5:
        zg_sc = zg[m]
        zg_sc = zg_sc[(zg_sc >= 0.1) & (zg_sc < 0.5)]  # physical SoftDrop range
        if len(zg_sc) > 5:
            sns.kdeplot(zg_sc, ax=axes[0], label=f"QCD SC {i} (N={m.sum()})",
                        fill=True, alpha=0.15)
axes[0].axvline(0.1, ls='--', color='gray', lw=1, label='$z_{cut}=0.1$')
axes[0].set_title(r"QCD Subclasses: Groomed Momentum Fraction $z_g$", fontsize=12)
axes[0].set_xlabel(r"$z_g$ (SoftDrop symmetry measure)")
axes[0].set_ylabel("Density")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.15)
axes[0].text(0.97, 0.95, r"QCD: $P(z_g) \propto 1/z_g$" "\n" r"(soft/collinear divergence)", transform=axes[0].transAxes,
             ha='right', va='top', fontsize=9, color='gray',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Right: Top subclasses — expect symmetric splitting: zg ~ 0.5 (W->qq)
for i, sc in enumerate(active_top_sc):
    m = (sub_preds == 1) & (sub_subclass == sc)
    if m.sum() > 5:
        zg_sc = zg[m]
        zg_sc = zg_sc[(zg_sc >= 0.1) & (zg_sc < 0.5)]
        if len(zg_sc) > 5:
            sns.kdeplot(zg_sc, ax=axes[1], label=f"Top SC {i} (N={m.sum()})",
                        fill=True, alpha=0.15)
axes[1].axvline(0.1, ls='--', color='gray', lw=1, label='$z_{cut}=0.1$')
axes[1].set_title(r"Top Subclasses: Groomed Momentum Fraction $z_g$", fontsize=12)
axes[1].set_xlabel(r"$z_g$ (SoftDrop symmetry measure)")
axes[1].set_ylabel("Density")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.15)
axes[1].text(0.97, 0.95, r"W-like (high-purity): flat $z_g$" "\n" r"QCD-like (background): peaks at $z_{cut}$", transform=axes[1].transAxes,
             ha='right', va='top', fontsize=9, color='blue',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.suptitle(r"Altarelli-Parisi Splitting Verification via $z_g$ [15ep]", fontsize=13, y=1.02)
plt.tight_layout()
save = os.path.join(RESULTS_DIR, 'zg_ap_splitting.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plot 11: Delta_R12 vs 2M/pT (Angular Collimation Verification) ---
print("Plot 11: Delta_R12 vs 2M/pT (relativistic angular collimation)")

# Compute jet pT from constituent 4-vectors for the sampled subset
print("  Computing jet pT from constituents...")
jets_pt = np.zeros(len(sub_ev))
for k in range(len(sub_ev)):
    ev = sub_ev[k]
    px_sum = float(ak.sum(ev.px))
    py_sum = float(ak.sum(ev.py))
    jets_pt[k] = np.sqrt(px_sum**2 + py_sum**2)

# 2M/pT predicted angular scale
two_m_over_pt = 2.0 * mass / (jets_pt + 1e-10)

# Filter to physically valid region
valid = (theta_g > 0.01) & (theta_g < 1.5) & (two_m_over_pt > 0.01) & (two_m_over_pt < 1.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter with 1:1 line (QCD)
qcd_v = valid & (sub_preds == 0)
if qcd_v.sum() > 10:
    axes[0].scatter(two_m_over_pt[qcd_v], theta_g[qcd_v],
                    alpha=0.2, s=8, color='green', label=f"QCD (N={qcd_v.sum()})")
axes[0].plot([0, 1.5], [0, 1.5], 'k--', lw=1.5, label=r'$\Delta R_{12} = 2M/p_T$')
axes[0].set_xlabel(r"$2M/p_T$ (predicted $\Delta R$)")
axes[0].set_ylabel(r"$\Delta R_{12}$ (measured from SoftDrop)")
axes[0].set_title(r"QCD: Angular Collimation $\Delta R_{12}$ vs $2M/p_T$", fontsize=12)
axes[0].set_xlim(0, 1.2); axes[0].set_ylim(0, 1.5)
axes[0].legend(); axes[0].grid(True, alpha=0.15)

# Right: scatter (Top)
top_v = valid & (sub_preds == 1)
if top_v.sum() > 10:
    axes[1].scatter(two_m_over_pt[top_v], theta_g[top_v],
                    alpha=0.3, s=8, color='blue', label=f"Top (N={top_v.sum()})")
axes[1].plot([0, 1.5], [0, 1.5], 'k--', lw=1.5, label=r'$\Delta R_{12} = 2M/p_T$')
axes[1].set_xlabel(r"$2M/p_T$ (predicted $\Delta R$)")
axes[1].set_ylabel(r"$\Delta R_{12}$ (measured from SoftDrop)")
axes[1].set_title(r"Top: Angular Collimation $\Delta R_{12}$ vs $2M/p_T$", fontsize=12)
axes[1].set_xlim(0, 1.2); axes[1].set_ylim(0, 1.5)
axes[1].legend(); axes[1].grid(True, alpha=0.15)

plt.suptitle(r"Relativistic Angular Scaling: $\Delta R_{12} \approx 2M/p_T$ [15ep]", fontsize=13, y=1.02)
plt.tight_layout()
save = os.path.join(RESULTS_DIR, 'angular_collimation.png')
plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {save}")

# --- Plots 8 & 9: Lund Planes ---
print("Plots 8 & 9: Lund Planes")
from arcjet.utils.plotting import plot_lund_plane

plot_lund_plane(sub_ev[sub_preds == 0], label="Predicted QCD",
                save_path=os.path.join(LUND_DIR, 'class_predicted_qcd.png'))
if (sub_preds == 1).sum() > 0:
    plot_lund_plane(sub_ev[sub_preds == 1], label="Predicted Top",
                    save_path=os.path.join(LUND_DIR, 'class_predicted_top.png'))

for cat, fname in [('True Negative', 'status_tn'), ('False Negative', 'status_fn'),
                    ('True Positive', 'status_tp'), ('False Positive', 'status_fp')]:
    m = (sub_status == cat)
    if m.sum() > 0:
        plot_lund_plane(sub_ev[m], label=cat,
                        save_path=os.path.join(LUND_DIR, f'{fname}.png'))

for i, sc in enumerate(active_qcd_sc):
    m = (sub_preds == 0) & (sub_subclass == sc)
    if m.sum() > 0:
        plot_lund_plane(sub_ev[m], label=f"QCD SC {i}",
                        save_path=os.path.join(LUND_DIR, f'qcd_subclass_{i}.png'))
        m_tn = m & (sub_labels == 0)
        m_fn = m & (sub_labels == 1)
        if m_tn.sum() > 5:
            plot_lund_plane(sub_ev[m_tn], label=f"QCD SC {i} (TN)",
                            save_path=os.path.join(LUND_DIR, f'qcd_subclass_{i}_tn.png'))
        if m_fn.sum() > 5:
            plot_lund_plane(sub_ev[m_fn], label=f"QCD SC {i} (FN)",
                            save_path=os.path.join(LUND_DIR, f'qcd_subclass_{i}_fn.png'))

for i, sc in enumerate(active_top_sc):
    m = (sub_preds == 1) & (sub_subclass == sc)
    if m.sum() > 0:
        plot_lund_plane(sub_ev[m], label=f"Top SC {i}",
                        save_path=os.path.join(LUND_DIR, f'top_subclass_{i}.png'))
        m_tp = m & (sub_labels == 1)
        m_fp = m & (sub_labels == 0)
        if m_tp.sum() > 5:
            plot_lund_plane(sub_ev[m_tp], label=f"Top SC {i} (TP)",
                            save_path=os.path.join(LUND_DIR, f'top_subclass_{i}_tp.png'))
        if m_fp.sum() > 5:
            plot_lund_plane(sub_ev[m_fp], label=f"Top SC {i} (FP)",
                            save_path=os.path.join(LUND_DIR, f'top_subclass_{i}_fp.png'))

print(f"\n[COMPLETE] All results saved to {RESULTS_DIR}/")
