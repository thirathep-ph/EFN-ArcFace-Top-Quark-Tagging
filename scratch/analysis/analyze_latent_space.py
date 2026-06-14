import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import awkward as ak
import umap
from sklearn.mixture import BayesianGaussianMixture
from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.utils.physics import compute_features
from arcjet.data.loader import load_awkward, JetTaggingDataset

# 1. Load trained model & weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 128
particle_dim = 256

print("Loading trained model from checkpoints/best_model_efn.pt...")
model = TopTaggingModel(n_proto_per_class=1, s=16.0, m=0.0, embedding_dim=embedding_dim, particle_dim=particle_dim).to(device)
model.load_state_dict(torch.load('checkpoints/best_model_efn.pt', map_location=device))
model.eval()

# Load class prototypes (centers) from model
prototypes = model.prototype_layer.prototypes.detach().cpu().numpy()  # (2, 128)
prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10)

# 2. Load embeddings & labels
print("Loading embeddings from results/embeddings.npz...")
data = np.load('results/embeddings.npz')
all_emb = data['embeddings']
labels = data['labels']
N = len(all_emb)

# Normalize embeddings
all_emb_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10)

# 3. Calculate similarities & prediction status
cosine_similarities = np.dot(all_emb_norm, prototypes_norm.T)  # (N, 2)
preds = np.argmax(cosine_similarities, axis=1)

# Define TP, TN, FP, FN
tp = (preds == 1) & (labels == 1)
tn = (preds == 0) & (labels == 0)
fp = (preds == 1) & (labels == 0)
fn = (preds == 0) & (labels == 1)

status = np.zeros(N, dtype=object)
status[tp] = 'True Positive'
status[tn] = 'True Negative'
status[fp] = 'False Positive'
status[fn] = 'False Negative'

print(f"Dataset stats: TP={tp.sum()}, TN={tn.sum()}, FP={fp.sum()}, FN={fn.sum()}")
acc = (tp.sum() + tn.sum()) / N
print(f"Calculated Test Accuracy: {acc:.4f}")

# 4. Project to 2D UMAP space using Cosine Metric
print("Fitting UMAP on normalized embeddings (metric='cosine')...")
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
umap_coords = reducer.fit_transform(all_emb_norm)
print("UMAP fitting complete.")

# Project prototypes to same UMAP space
proto_umap = reducer.transform(prototypes_norm)

# 5. Fit DPGMM for unsupervised subclass discovery on UMAP coordinates
subclass_labels = np.zeros(N, dtype=int) - 1
min_samples = 10

# Top subclasses
top_mask = (preds == 1)
if top_mask.sum() >= min_samples:
    print("Fitting DPGMM on Top latent space embeddings...")
    top_latent = all_emb_norm[top_mask]
    dpgmm_top = BayesianGaussianMixture(
        n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2
    )
    top_subclasses = dpgmm_top.fit_predict(top_latent)
    active_top_subclasses, counts_top = np.unique(top_subclasses, return_counts=True)
    print(f"Top Active Subclasses (DPGMM indices): {active_top_subclasses} with counts {counts_top}")
    subclass_labels[top_mask] = top_subclasses
else:
    print(f"Warning: Too few Top predictions ({top_mask.sum()}) to run DPGMM. Skipping Top subclass discovery.")
    active_top_subclasses = np.array([])

# QCD subclasses
qcd_mask = (preds == 0)
if qcd_mask.sum() >= min_samples:
    print("Fitting DPGMM on QCD latent space embeddings...")
    qcd_latent = all_emb_norm[qcd_mask]
    dpgmm_qcd = BayesianGaussianMixture(
        n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2
    )
    qcd_subclasses = dpgmm_qcd.fit_predict(qcd_latent)
    active_qcd_subclasses, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
    print(f"QCD Active Subclasses (DPGMM indices): {active_qcd_subclasses} with counts {counts_qcd}")
    subclass_labels[qcd_mask] = qcd_subclasses
else:
    print(f"Warning: Too few QCD predictions ({qcd_mask.sum()}) to run DPGMM. Skipping QCD subclass discovery.")
    active_qcd_subclasses = np.array([])

# 6. Load physical events from H5 test set
print("Loading physical events for FastJet feature calculations...")
test_ev, test_lab, _ = load_awkward('data/top_tagging/test.h5', max_events=N)
test_ev = test_ev[:N]
test_lab = test_lab[:N]

# Stratified sampling based on latent space subclass and prediction status distribution
sampled_indices = []

# A. Sample from each active Top subclass (predicted Top)
for sc in active_top_subclasses:
    sc_indices = np.where((preds == 1) & (subclass_labels == sc))[0]
    if len(sc_indices) > 0:
        n_sample = min(1000, len(sc_indices))
        sampled_indices.extend(np.random.choice(sc_indices, n_sample, replace=False))
        
# B. Sample from each active QCD subclass (predicted QCD)
for sc in active_qcd_subclasses:
    sc_indices = np.where((preds == 0) & (subclass_labels == sc))[0]
    if len(sc_indices) > 0:
        n_sample = min(1000, len(sc_indices))
        sampled_indices.extend(np.random.choice(sc_indices, n_sample, replace=False))
        
# C. Sample from each prediction status (TP, TN, FP, FN) to ensure rare errors are well-represented
for cat in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
    cat_indices = np.where(status == cat)[0]
    if len(cat_indices) > 0:
        n_sample = min(1000, len(cat_indices))
        sampled_indices.extend(np.random.choice(cat_indices, n_sample, replace=False))
        
# D. Fallback if no indices are selected
if len(sampled_indices) > 0:
    test_idx = np.unique(sampled_indices)
else:
    test_idx = np.random.choice(N, min(5000, N), replace=False)
    
print(f"Stratified sampling complete: selected {len(test_idx)} representative jets for FastJet physics analysis.")

# Filter subset arrays for Plot 3
sub_ev = test_ev[test_idx]
sub_preds = preds[test_idx]
sub_subclass = subclass_labels[test_idx]
sub_status = status[test_idx]
sub_labels = labels[test_idx]


# Compute physical features
print("Running FastJet feature calculations on subset...")
features = compute_features(sub_ev)
mass = features[:, 0]
tau32 = features[:, 7]
d12 = features[:, 4]  # sqrt(d12)


# ============================================================================
# Generating Plots
# ============================================================================
os.makedirs('results', exist_ok=True)

# Plot 1: umap_prediction_status.png
print("Generating Plot 1: results/umap_prediction_status.png...")
plt.figure(figsize=(10, 8))
colors = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4', 'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
markers = {'True Negative': 'o', 'True Positive': 'o', 'False Positive': 'x', 'False Negative': 'x'}
sizes = {'True Negative': 15, 'True Positive': 15, 'False Positive': 40, 'False Negative': 40}
alphas = {'True Negative': 0.3, 'True Positive': 0.3, 'False Positive': 0.8, 'False Negative': 0.8}

for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
    mask_cat = (status == cat)
    if mask_cat.sum() > 0:
        plt.scatter(
            umap_coords[mask_cat, 0], umap_coords[mask_cat, 1],
            color=colors[cat], marker=markers[cat], s=sizes[cat],
            alpha=alphas[cat], label=f"{cat} ({mask_cat.sum()})"
        )

# Plot class prototypes (centers)
plt.scatter(
    proto_umap[0, 0], proto_umap[0, 1],
    marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5,
    label='W_QCD (Class Center)', zorder=10
)
plt.scatter(
    proto_umap[1, 0], proto_umap[1, 1],
    marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5,
    label='W_Top (Class Center)', zorder=10
)

# Add 2D density contour lines for TP and TN to outline their distribution shapes
try:
    for cat in ['True Negative', 'True Positive']:
        mask_cat = (status == cat)
        if mask_cat.sum() > 50:
            sns.kdeplot(
                x=umap_coords[mask_cat, 0], y=umap_coords[mask_cat, 1],
                levels=4, color=colors[cat], alpha=0.3, linewidths=1.2
            )
except Exception as e:
    print(f"Skipping density contours: {e}")

plt.title("EFN + ArcFace Latent Space (UMAP) & Prediction Status", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.grid(True, alpha=0.15)
plt.savefig('results/umap_prediction_status.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Plot 1 to results/umap_prediction_status.png")

# Plot 2: umap_subclasses.png
print("Generating Plot 2: results/umap_subclasses.png...")
plt.figure(figsize=(10, 8))

# Plot QCD Subclasses
qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
for i, sc in enumerate(active_qcd_subclasses):
    sc_mask = (preds == 0) & (subclass_labels == sc)
    if sc_mask.sum() > 0:
        plt.scatter(
            umap_coords[sc_mask, 0], umap_coords[sc_mask, 1],
            color=qcd_colors[i % len(qcd_colors)], s=15, alpha=0.5,
            label=f"QCD Subclass {i} (N={sc_mask.sum()})"
        )

# Plot Top Subclasses
top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
for i, sc in enumerate(active_top_subclasses):
    sc_mask = (preds == 1) & (subclass_labels == sc)
    if sc_mask.sum() > 0:
        plt.scatter(
            umap_coords[sc_mask, 0], umap_coords[sc_mask, 1],
            color=top_colors[i % len(top_colors)], s=15, alpha=0.5,
            label=f"Top Subclass {i} (N={sc_mask.sum()})"
        )

# Plot Class Centers
plt.scatter(
    proto_umap[0, 0], proto_umap[0, 1],
    marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5,
    label='W_QCD (Center)', zorder=10
)
plt.scatter(
    proto_umap[1, 0], proto_umap[1, 1],
    marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5,
    label='W_Top (Center)', zorder=10
)

plt.title("Discovered Subclasses in Latent Space (UMAP + DPGMM)", fontsize=14)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.grid(True, alpha=0.15)
plt.savefig('results/umap_subclasses.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Plot 2 to results/umap_subclasses.png")

# Plot 3: subclass_physics_distributions.png
print("Generating Plot 3: results/subclass_physics_distributions.png...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Column 1: Discovered Top Subclasses
for i, sc in enumerate(active_top_subclasses):
    sc_mask = (preds == 1) & (subclass_labels == sc)
    if sc_mask.sum() > 5:
        sns.kdeplot(mass[sc_mask], ax=axes[0, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
        sns.kdeplot(tau32[sc_mask], ax=axes[1, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
        sns.kdeplot(d12[sc_mask], ax=axes[2, 0], label=f"Subclass {i}", fill=True, alpha=0.1)

axes[0, 0].set_title("Discovered Top Subclasses: Jet Mass", fontsize=12)
axes[0, 0].set_xlabel("Mass (GeV)")
axes[0, 0].legend()

axes[1, 0].set_title("Discovered Top Subclasses: N-subjettiness Tau32", fontsize=12)
axes[1, 0].set_xlabel("Tau32")
axes[1, 0].legend()

axes[2, 0].set_title("Discovered Top Subclasses: Splitting Scale sqrt(d12)", fontsize=12)
axes[2, 0].set_xlabel("sqrt(d12) (GeV)")
axes[2, 0].legend()


# Column 2: QCD TN vs QCD FP (False Positives Analysis)
sub_tn = (sub_status == 'True Negative')
sub_fp = (sub_status == 'False Positive')
if sub_tn.sum() > 5:
    sns.kdeplot(mass[sub_tn], ax=axes[0, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
    sns.kdeplot(tau32[sub_tn], ax=axes[1, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
    sns.kdeplot(d12[sub_tn], ax=axes[2, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
if sub_fp.sum() > 5:
    sns.kdeplot(mass[sub_fp], ax=axes[0, 1], label="Mistaken QCD (FP)", color='red', lw=2)
    sns.kdeplot(tau32[sub_fp], ax=axes[1, 1], label="Mistaken QCD (FP)", color='red', lw=2)
    sns.kdeplot(d12[sub_fp], ax=axes[2, 1], label="Mistaken QCD (FP)", color='red', lw=2)

axes[0, 1].set_title("QCD Mistakes: Jet Mass", fontsize=12)
axes[0, 1].set_xlabel("Mass (GeV)")
axes[0, 1].legend()

axes[1, 1].set_title("QCD Mistakes: N-subjettiness Tau32", fontsize=12)
axes[1, 1].set_xlabel("Tau32")
axes[1, 1].legend()

axes[2, 1].set_title("QCD Mistakes: Splitting Scale sqrt(d12)", fontsize=12)
axes[2, 1].set_xlabel("sqrt(d12) (GeV)")
axes[2, 1].legend()


# Column 3: Top TP vs Top FN (False Negatives Analysis)
sub_tp = (sub_status == 'True Positive')
sub_fn = (sub_status == 'False Negative')
if sub_tp.sum() > 5:
    sns.kdeplot(mass[sub_tp], ax=axes[0, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
    sns.kdeplot(tau32[sub_tp], ax=axes[1, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
    sns.kdeplot(d12[sub_tp], ax=axes[2, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
if sub_fn.sum() > 5:
    sns.kdeplot(mass[sub_fn], ax=axes[0, 2], label="Mistaken Top (FN)", color='orange', lw=2)
    sns.kdeplot(tau32[sub_fn], ax=axes[1, 2], label="Mistaken Top (FN)", color='orange', lw=2)
    sns.kdeplot(d12[sub_fn], ax=axes[2, 2], label="Mistaken Top (FN)", color='orange', lw=2)

axes[0, 2].set_title("Top Mistakes: Jet Mass", fontsize=12)
axes[0, 2].set_xlabel("Mass (GeV)")
axes[0, 2].legend()

axes[1, 2].set_title("Top Mistakes: N-subjettiness Tau32", fontsize=12)
axes[1, 2].set_xlabel("Tau32")
axes[1, 2].legend()

axes[2, 2].set_title("Top Mistakes: Splitting Scale sqrt(d12)", fontsize=12)
axes[2, 2].set_xlabel("sqrt(d12) (GeV)")
axes[2, 2].legend()

plt.tight_layout()
plt.savefig('results/subclass_physics_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Plot 3 to results/subclass_physics_distributions.png")

# ============================================================================
# Plot 4: Subclass Composition Heatmap (Purity / Level 3 T/F Analysis)
# ============================================================================
print("Generating Plot 4: results/subclass_composition_heatmap.png...")
categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
composition_matrix = []
row_labels = []

if len(active_top_subclasses) > 0:
    top_probs = dpgmm_top.predict_proba(top_latent)
    top_hard_subclasses = np.argmax(top_probs, axis=1)
    for i, sc in enumerate(active_top_subclasses):
        sc_mask = (preds == 1) & (subclass_labels == sc)
        n_tot = sc_mask.sum()
        if n_tot > 0:
            row_counts = []
            for cat in categories:
                cat_count = ((status == cat) & sc_mask).sum()
                row_counts.append(cat_count / n_tot * 100.0)
            composition_matrix.append(row_counts)
            row_labels.append(f"Top Subclass {i} (N={n_tot})")

if len(active_qcd_subclasses) > 0:
    qcd_probs = dpgmm_qcd.predict_proba(qcd_latent)
    qcd_hard_subclasses = np.argmax(qcd_probs, axis=1)
    for i, sc in enumerate(active_qcd_subclasses):
        sc_mask = (preds == 0) & (subclass_labels == sc)
        n_tot = sc_mask.sum()
        if n_tot > 0:
            row_counts = []
            for cat in categories:
                cat_count = ((status == cat) & sc_mask).sum()
                row_counts.append(cat_count / n_tot * 100.0)
            composition_matrix.append(row_counts)
            row_labels.append(f"QCD Subclass {i} (N={n_tot})")

if len(composition_matrix) > 0:
    plt.figure(figsize=(10, 6))
    sns.heatmap(composition_matrix, annot=True, fmt=".1f", xticklabels=categories, yticklabels=row_labels, cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title("Subclass Classification Purity / Composition", fontsize=14)
    plt.xlabel("Prediction Status")
    plt.ylabel("Discovered Subclasses")
    plt.savefig('results/subclass_composition_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved Plot 4 to results/subclass_composition_heatmap.png")

# ============================================================================
# Plot 5: Representative Event Displays (Ensemble of Top 5)
# ============================================================================
print("Generating Plot 5: results/subclass_X_representative_jets.png...")
# Top Subclasses
if len(active_top_subclasses) > 0:
    top_indices_in_test = np.where(preds == 1)[0]
    top_probs = dpgmm_top.predict_proba(top_latent)
    for i, sc in enumerate(active_top_subclasses):
        assigned_jets_in_top = (top_subclasses == sc)
        if assigned_jets_in_top.sum() > 0:
            probs_for_sc = top_probs[:, sc].copy()
            probs_for_sc[~assigned_jets_in_top] = -1.0
            top5_relative_indices = np.argsort(probs_for_sc)[::-1][:5]
            top5_test_indices = top_indices_in_test[top5_relative_indices]
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for idx_fig, t_idx in enumerate(top5_test_indices):
                event = test_ev[t_idx]
                px = ak.to_numpy(event.px)
                py = ak.to_numpy(event.py)
                pz = ak.to_numpy(event.pz)
                e = ak.to_numpy(event.E)
                pt = np.sqrt(px**2 + py**2)
                eta = np.arcsinh(pz / (pt + 1e-10))
                phi = np.arctan2(py, px)
                z_e = e / (np.sum(e) + 1e-10)
                
                ax = axes[idx_fig]
                ax.scatter(eta, phi, s=pt*5, alpha=0.7, c=z_e, cmap='plasma', edgecolors='black', linewidth=0.3)
                ax.set_xlabel(r'$\eta$ (rel)')
                ax.set_ylabel(r'$\phi$ (rel)')
                ax.set_title(f"Jet {idx_fig+1} (Prob: {probs_for_sc[top5_relative_indices[idx_fig]]:.2f})", fontsize=10)
                ax.grid(True, alpha=0.15)
                
            plt.suptitle(f"Top Subclass {i} - Top 5 Representative Jets", fontsize=14, y=1.05)
            plt.tight_layout()
            plt.savefig(f'results/top_subclass_{i}_representative_jets.png', dpi=150, bbox_inches='tight')
            plt.close()

# QCD Subclasses
if len(active_qcd_subclasses) > 0:
    qcd_indices_in_test = np.where(preds == 0)[0]
    qcd_probs = dpgmm_qcd.predict_proba(qcd_latent)
    for i, sc in enumerate(active_qcd_subclasses):
        assigned_jets_in_qcd = (qcd_subclasses == sc)
        if assigned_jets_in_qcd.sum() > 0:
            probs_for_sc = qcd_probs[:, sc].copy()
            probs_for_sc[~assigned_jets_in_qcd] = -1.0
            top5_relative_indices = np.argsort(probs_for_sc)[::-1][:5]
            top5_test_indices = qcd_indices_in_test[top5_relative_indices]
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for idx_fig, t_idx in enumerate(top5_test_indices):
                event = test_ev[t_idx]
                px = ak.to_numpy(event.px)
                py = ak.to_numpy(event.py)
                pz = ak.to_numpy(event.pz)
                e = ak.to_numpy(event.E)
                pt = np.sqrt(px**2 + py**2)
                eta = np.arcsinh(pz / (pt + 1e-10))
                phi = np.arctan2(py, px)
                z_e = e / (np.sum(e) + 1e-10)
                
                ax = axes[idx_fig]
                ax.scatter(eta, phi, s=pt*5, alpha=0.7, c=z_e, cmap='plasma', edgecolors='black', linewidth=0.3)
                ax.set_xlabel(r'$\eta$ (rel)')
                ax.set_ylabel(r'$\phi$ (rel)')
                ax.set_title(f"Jet {idx_fig+1} (Prob: {probs_for_sc[top5_relative_indices[idx_fig]]:.2f})", fontsize=10)
                ax.grid(True, alpha=0.15)
                
            plt.suptitle(f"QCD Subclass {i} - Top 5 Representative Jets", fontsize=14, y=1.05)
            plt.tight_layout()
            plt.savefig(f'results/qcd_subclass_{i}_representative_jets.png', dpi=150, bbox_inches='tight')
            plt.close()

# ============================================================================
# Plot 6: Cosine Similarity Distributions
# ============================================================================
print("Generating Plot 6: results/subclass_similarity_distributions.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: QCD subclasses similarity to W_QCD
for i, sc in enumerate(active_qcd_subclasses):
    sc_mask = (preds == 0) & (subclass_labels == sc)
    if sc_mask.sum() > 5:
        sns.kdeplot(cosine_similarities[sc_mask, 0], ax=axes[0], label=f"Subclass {i}", fill=True, alpha=0.1)
axes[0].set_title("QCD Subclasses: Cosine Similarity to W_QCD Center", fontsize=12)
axes[0].set_xlabel("Cosine Similarity")
axes[0].legend()
axes[0].grid(True, alpha=0.15)

# Right: Top subclasses similarity to W_Top
for i, sc in enumerate(active_top_subclasses):
    sc_mask = (preds == 1) & (subclass_labels == sc)
    if sc_mask.sum() > 5:
        sns.kdeplot(cosine_similarities[sc_mask, 1], ax=axes[1], label=f"Subclass {i}", fill=True, alpha=0.1)
axes[1].set_title("Top Subclasses: Cosine Similarity to W_Top Center", fontsize=12)
axes[1].set_xlabel("Cosine Similarity")
axes[1].legend()
axes[1].grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('results/subclass_similarity_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Plot 6 to results/subclass_similarity_distributions.png")

# ============================================================================
# Plot 7: Physics Feature Correlation Matrices per Subclass
# ============================================================================
print("Generating Plot 7: results/subclass_physics_correlations.png...")
import pandas as pd
n_top_sc = len(active_top_subclasses)
n_qcd_sc = len(active_qcd_subclasses)

if n_top_sc > 0 or n_qcd_sc > 0:
    fig, axes = plt.subplots(2, max(n_top_sc, n_qcd_sc), figsize=(4 * max(n_top_sc, n_qcd_sc), 8))
    feature_subnames = ["Mass", "Tau32", "sqrt(d12)"]
    
    # Row 0: Top subclasses
    for i, sc in enumerate(active_top_subclasses):
        sc_mask = (sub_preds == 1) & (sub_subclass == sc)
        ax = axes[0, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[0]
        if sc_mask.sum() > 5:
            sub_df = pd.DataFrame(np.stack([mass[sc_mask], tau32[sc_mask], d12[sc_mask]], axis=1), columns=feature_subnames)
            sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
        ax.set_title(f"Top Subclass {i}", fontsize=11)
        
    # Row 1: QCD subclasses
    for i, sc in enumerate(active_qcd_subclasses):
        sc_mask = (sub_preds == 0) & (sub_subclass == sc)
        ax = axes[1, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[1]
        if sc_mask.sum() > 5:
            sub_df = pd.DataFrame(np.stack([mass[sc_mask], tau32[sc_mask], d12[sc_mask]], axis=1), columns=feature_subnames)
            sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
        ax.set_title(f"QCD Subclass {i}", fontsize=11)
        
    plt.suptitle("Physics Feature Correlation per Subclass", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/subclass_physics_correlations.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved Plot 7 to results/subclass_physics_correlations.png")

# ============================================================================
# Plot 8 & 9: Hierarchical Lund Planes (Level 2 Class & Level 3 Subclass + T/F)
# ============================================================================
print("Generating Plot 8 & 9: results/lund_planes/...")
lund_dir = 'results/lund_planes'
os.makedirs(lund_dir, exist_ok=True)
from arcjet.utils.plotting import plot_lund_plane

# A. Class-level Lund planes
plot_lund_plane(sub_ev[sub_preds == 1], label="Predicted Top (Class-level)", save_path=os.path.join(lund_dir, 'class_predicted_top.png'))
plot_lund_plane(sub_ev[sub_preds == 0], label="Predicted QCD (Class-level)", save_path=os.path.join(lund_dir, 'class_predicted_qcd.png'))

# B. Prediction Status Lund planes (Level 1 T/F)
sub_tp = (sub_status == 'True Positive')
sub_tn = (sub_status == 'True Negative')
sub_fp = (sub_status == 'False Positive')
sub_fn = (sub_status == 'False Negative')

if sub_tp.sum() > 0:
    plot_lund_plane(sub_ev[sub_tp], label="True Positives (TP)", save_path=os.path.join(lund_dir, 'status_tp.png'))
if sub_tn.sum() > 0:
    plot_lund_plane(sub_ev[sub_tn], label="True Negatives (TN)", save_path=os.path.join(lund_dir, 'status_tn.png'))
if sub_fp.sum() > 0:
    plot_lund_plane(sub_ev[sub_fp], label="False Positives (FP)", save_path=os.path.join(lund_dir, 'status_fp.png'))
if sub_fn.sum() > 0:
    plot_lund_plane(sub_ev[sub_fn], label="False Negatives (FN)", save_path=os.path.join(lund_dir, 'status_fn.png'))

# C. Subclass level Lund planes with T/F splits (Level 2 & 3 T/F)
for i, sc in enumerate(active_top_subclasses):
    sc_mask_sub = (sub_preds == 1) & (sub_subclass == sc)
    if sc_mask_sub.sum() > 0:
        plot_lund_plane(sub_ev[sc_mask_sub], label=f"Top Subclass {i}", save_path=os.path.join(lund_dir, f'top_subclass_{i}.png'))
        # T/F Split within this subclass
        sc_tp = sc_mask_sub & (sub_labels == 1)
        sc_fp = sc_mask_sub & (sub_labels == 0)
        if sc_tp.sum() > 5:
            plot_lund_plane(sub_ev[sc_tp], label=f"Top Subclass {i} (TP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_tp.png'))
        if sc_fp.sum() > 5:
            plot_lund_plane(sub_ev[sc_fp], label=f"Top Subclass {i} (FP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_fp.png'))
            
for i, sc in enumerate(active_qcd_subclasses):
    sc_mask_sub = (sub_preds == 0) & (sub_subclass == sc)
    if sc_mask_sub.sum() > 0:
        plot_lund_plane(sub_ev[sc_mask_sub], label=f"QCD Subclass {i}", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}.png'))
        # T/F Split within this subclass
        sc_tn = sc_mask_sub & (sub_labels == 0)
        sc_fn = sc_mask_sub & (sub_labels == 1)
        if sc_tn.sum() > 5:
            plot_lund_plane(sub_ev[sc_tn], label=f"QCD Subclass {i} (TN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_tn.png'))
        if sc_fn.sum() > 5:
            plot_lund_plane(sub_ev[sc_fn], label=f"QCD Subclass {i} (FN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_fn.png'))

print("Saved Lund planes to results/lund_planes/")
print("[COMPLETE] Analysis finished successfully.")
