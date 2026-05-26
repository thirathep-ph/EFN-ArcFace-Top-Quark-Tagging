"""
run_full_analysis.py — Comprehensive EFN + ArcFace Interpretability Analysis
=============================================================================
TIERED POPULATION STRATEGY (solves VRAM + time constraints):

  Tier 1 — Embeddings (ALL events in test.h5, up to MAX_EVENTS_EMB):
    GPU forward pass in batches of BATCH_SIZE.
    Immediately moved to CPU after each batch.
    Model freed from GPU after extraction.
    Cached to disk: embeddings_cache.npz (survives session restart).

  Tier 2 — DPGMM + Latent analysis (EMB_EVENTS events):
    Stratified subsample of Tier 1 embeddings.
    Used for DPGMM fitting, cosine similarity plots, latent T/F analysis.
    Subclass labels are assigned via predict() to ALL Tier 1 events.

  Tier 3 — UMAP visualization (UMAP_EVENTS events, stratified by class):
    Fit UMAP on this subset only (UMAP is O(N^1.5) in memory).
    Prototype positions computed via transform().

  Tier 4 — FastJet physics features (FASTJET_EVENTS events):
    Chunked processing (FASTJET_CHUNK_SIZE events per batch).
    Stratified by class + subclass + T/F status.
    Covers all physics plots: distributions, correlations, zg, angular.

  Tier 5 — Lund planes (LUND_SAMPLE per group):
    Slowest per-event operation; sampled from Tier 4 pool.

Output structure:
  results/analysis_15ep/
    ├── embeddings_cache.npz         ← Tier 1 embeddings (all events)
    ├── prototypes.npy               ← class prototype vectors
    ├── analysis_metrics.json        ← accuracy, T/F counts, NMI, ROC AUC
    ├── roc_curve.png                ← ROC + AUC (robustness proof)
    ├── umap_prediction_status.png   ← Tier 3 UMAP
    ├── umap_subclasses.png
    ├── subclass_physics_distributions.png   ← Tier 4 FastJet
    ├── subclass_composition_heatmap.png     ← Tier 2 DPGMM
    ├── *_representative_jets.png
    ├── subclass_similarity_distributions.png
    ├── subclass_physics_correlations.png
    ├── zg_ap_splitting.png
    ├── angular_collimation.png
    ├── lund_planes/                 ← Tier 5 Lund
    ├── tf_analysis/
    └── edge_analysis/
"""

import os
import sys
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import awkward as ak
import umap
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jet_tagging.models.efn_arcface import TopTaggingModel, get_efn_features
from jet_tagging.utils.physics import compute_features
from jet_tagging.data.loader import load_awkward, JetTaggingDataset
from jet_tagging.utils.plotting import plot_lund_plane

# ============================================================
# CONFIG — edit these to match your environment
# ============================================================
CHECKPOINT     = 'checkpoints/test_run_15ep/best_model_efn.pt'
DATA_PATH      = 'data/top_tagging/test.h5'
RESULTS_DIR    = 'results/analysis_15ep'
TF_DIR         = os.path.join(RESULTS_DIR, 'tf_analysis')
LUND_DIR       = os.path.join(RESULTS_DIR, 'lund_planes')
EDGE_DIR       = os.path.join(RESULTS_DIR, 'edge_analysis')
EMB_CACHE      = os.path.join(RESULTS_DIR, 'embeddings_cache.npz')
PROTO_CACHE    = os.path.join(RESULTS_DIR, 'prototypes.npy')

# --- Tiered population sizes ---
MAX_EVENTS_EMB   = None    # None = use ALL events in test.h5 (Tier 1)
                            # Set e.g. 200_000 if RAM is limited
EMB_EVENTS       = 100_000  # Tier 2: DPGMM + cosine sim + T/F latent plots
UMAP_EVENTS      = 50_000   # Tier 3: UMAP fitting (stratified)
FASTJET_EVENTS   = 100_000  # Tier 4: FastJet physics features
FASTJET_CHUNK    = 2_000    # events per FastJet chunk (controls RAM)
LUND_SAMPLE      = 3_000    # Tier 5: max events per Lund plane group

EMBEDDING_DIM   = 64
PARTICLE_DIM    = 128
BATCH_SIZE      = 256      # GPU batch — conservative for VRAM safety
EDGE_PERCENTILE = 10       # Bottom 10% of |margin| = "edge jets"
DPGMM_SEEDS     = [42, 123, 7]

for d in [RESULTS_DIR, TF_DIR, LUND_DIR, EDGE_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# STEP 1: Load ALL test data (events + labels) into CPU RAM
#   awkward arrays are lazy — this is mostly just reading metadata
#   until actual field access. Should fit in ~5-8 GB RAM.
# ============================================================
from sklearn.metrics import roc_curve, auc as sklearn_auc

print(f"\n[1] Loading test data: {DATA_PATH}")
all_ev, all_lab, all_w = load_awkward(DATA_PATH, max_events=MAX_EVENTS_EMB)
N_TOTAL = len(all_lab)
labels_full = all_lab.astype(np.int64)
print(f"    Loaded {N_TOTAL:,} events total")

# ============================================================
# STEP 2: Memory-efficient embedding extraction
#   GPU is used ONLY in this block.
#   After extraction we free all GPU memory before continuing.
# ============================================================
if os.path.exists(EMB_CACHE) and os.path.exists(PROTO_CACHE):
    print(f"\n[2] Loading cached embeddings from {EMB_CACHE}")
    cache   = np.load(EMB_CACHE)
    all_emb = cache['embeddings']
    prototypes_raw = np.load(PROTO_CACHE)
    print(f"    Embeddings shape: {all_emb.shape} (loaded from cache)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] Extracting embeddings on {device}")
    print(f"    Checkpoint: {CHECKPOINT}")

    model = TopTaggingModel(
        n_proto_per_class=1, s=16.0, m=0.0,
        embedding_dim=EMBEDDING_DIM, particle_dim=PARTICLE_DIM
    ).to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    state_dict.pop('prototype_layer.prototype_class', None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Save prototypes NOW before freeing the model
    prototypes_raw = model.prototype_layer.prototypes.detach().cpu().numpy()
    np.save(PROTO_CACHE, prototypes_raw)
    print(f"    Saved prototypes to {PROTO_CACHE}")

    ds     = JetTaggingDataset(all_ev, all_lab, all_w)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    emb_list, lab_list = [], []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_i, (x, y, w, m, _) in enumerate(loader):
            x, m = x.to(device), m.to(device)
            coords, z = get_efn_features(x, m)
            jet_emb = model.efn(coords, z, m)       # (B, EMBEDDING_DIM)
            emb_list.append(jet_emb.cpu().numpy())   # move to CPU immediately
            lab_list.append(y.numpy())
            if (batch_i + 1) % 5 == 0:
                print(f"    Batch {batch_i + 1}/{total_batches}...", flush=True)

    all_emb = np.vstack(emb_list)
    # Note: lab_list may differ from labels_full if Colab session was truncated;
    # always use labels_full (loaded fresh from disk in Step 1).
    del emb_list, lab_list

    # ---- FREE ALL GPU MEMORY ----
    del model, ds, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("    GPU memory freed.")

    np.savez(EMB_CACHE, embeddings=all_emb, labels=labels_full)
    print(f"    Embeddings cached to {EMB_CACHE}")

# ============================================================
# Tier 1 embeddings are stored in all_emb / labels_full (N_TOTAL)
# ============================================================
print(f"    Embeddings shape: {all_emb.shape}  |  Labels: {labels_full.shape}")

# ============================================================
# STEP 3: Normalize embeddings (full Tier 1)
# ============================================================
print("\n[3] Normalizing embeddings and computing cosine similarities")
all_emb_norm_full = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10)
prototypes_norm   = prototypes_raw / (np.linalg.norm(prototypes_raw, axis=1, keepdims=True) + 1e-10)

cosine_sim_full = np.dot(all_emb_norm_full, prototypes_norm.T)  # (N_TOTAL, 2)
preds_full      = np.argmax(cosine_sim_full, axis=1)
margin_full     = cosine_sim_full[:, 1] - cosine_sim_full[:, 0]

tp_full = (preds_full == 1) & (labels_full == 1)
tn_full = (preds_full == 0) & (labels_full == 0)
fp_full = (preds_full == 1) & (labels_full == 0)
fn_full = (preds_full == 0) & (labels_full == 1)

status_full = np.empty(N_TOTAL, dtype=object)
status_full[tp_full] = 'TP'; status_full[tn_full] = 'TN'
status_full[fp_full] = 'FP'; status_full[fn_full] = 'FN'

acc = (tp_full.sum() + tn_full.sum()) / N_TOTAL
print(f"    Accuracy (all {N_TOTAL:,} events): {acc:.4f}")
print(f"    TP={tp_full.sum():7,}  TN={tn_full.sum():7,}  FP={fp_full.sum():7,}  FN={fn_full.sum():7,}")

# ============================================================
# STEP 3b: ROC Curve (uses full Tier 1 — most statistically powerful)
# ============================================================
print("\n[3b] Computing ROC curve (full Tier 1)")
fpr, tpr, _ = roc_curve(labels_full, margin_full)  # margin = score for Top class
roc_auc = sklearn_auc(fpr, tpr)
print(f"    ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'EFN+ArcFace (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.fill_between(fpr, tpr, alpha=0.08, color='#1f77b4')
plt.xlabel('False Positive Rate (QCD misidentified as Top)', fontsize=12)
plt.ylabel('True Positive Rate (Top identification efficiency)', fontsize=12)
plt.title('ROC Curve: EFN + ArcFace Top Tagger', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {os.path.join(RESULTS_DIR, 'roc_curve.png')}")

# ============================================================
# STEP 3c: Tier 2 — stratified subsample for DPGMM + latent analysis
#   Stratify by (predicted class) × (T/F status) to preserve all categories
# ============================================================
print(f"\n[3c] Tier 2 subsample: {EMB_EVENTS:,} events for DPGMM + latent analysis")
rng = np.random.default_rng(42)
per_stratum = EMB_EVENTS // 4   # TP, TN, FP, FN
t2_idx = []
for mask_s in [tp_full, tn_full, fp_full, fn_full]:
    idx_s = np.where(mask_s)[0]
    n_s = min(per_stratum, len(idx_s))
    t2_idx.extend(rng.choice(idx_s, n_s, replace=False).tolist())
t2_idx = np.array(sorted(set(t2_idx)))
print(f"    Tier 2 index size: {len(t2_idx):,}")

# Tier 2 arrays
all_emb_norm = all_emb_norm_full[t2_idx]    # (T2, 64)
cosine_sim   = cosine_sim_full[t2_idx]       # (T2, 2)
preds        = preds_full[t2_idx]
margin       = margin_full[t2_idx]
labels       = labels_full[t2_idx]
N            = len(t2_idx)

tp = (preds == 1) & (labels == 1)
tn = (preds == 0) & (labels == 0)
fp = (preds == 1) & (labels == 0)
fn = (preds == 0) & (labels == 1)

status = np.empty(N, dtype=object)
status[tp] = 'TP'; status[tn] = 'TN'
status[fp] = 'FP'; status[fn] = 'FN'
print(f"    Tier 2 — TP={tp.sum()} TN={tn.sum()} FP={fp.sum()} FN={fn.sum()}")

# ============================================================
# STEP 4: Edge / decision-boundary jets (on Tier 2)
# ============================================================
print(f"\n[4] Edge analysis (bottom {EDGE_PERCENTILE}% of |margin| in Tier 2)")
abs_margin     = np.abs(margin)
edge_threshold = np.percentile(abs_margin, EDGE_PERCENTILE)
edge_mask      = abs_margin < edge_threshold
print(f"    Edge threshold |margin| < {edge_threshold:.4f}  →  {edge_mask.sum()} edge jets in Tier 2")

# ============================================================
# STEP 5: UMAP on Tier 3 stratified subsample (not full Tier 1)
#   Fit on UMAP_EVENTS points, then transform prototypes only.
#   Tier 2 embeddings are transformed for plotting.
# ============================================================
print(f"\n[5] UMAP on Tier 3 subsample ({UMAP_EVENTS:,} events, stratified)")

# Build Tier 3: balance QCD/Top from Tier 2
qcd_t2 = np.where(preds == 0)[0]; top_t2 = np.where(preds == 1)[0]
n_each = UMAP_EVENTS // 2
t3_local = np.concatenate([
    rng.choice(qcd_t2, min(n_each, len(qcd_t2)), replace=False),
    rng.choice(top_t2, min(n_each, len(top_t2)), replace=False)
])
t3_local = np.sort(t3_local)
print(f"    Tier 3 (UMAP fit) size: {len(t3_local):,}")

reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42,
                    n_neighbors=30, min_dist=0.1)
reducer.fit(all_emb_norm[t3_local])       # fit on Tier 3

# Transform all Tier 2 points (needed for all UMAP plots)
print("    Transforming Tier 2 embeddings into UMAP space...")
umap_coords = reducer.transform(all_emb_norm)   # (N_T2, 2)
proto_umap  = reducer.transform(prototypes_norm)
print("    UMAP done.")

# ============================================================
# STEP 6: DPGMM subclass discovery on Tier 2
# ============================================================
print("\n[6] DPGMM subclass discovery (Tier 2)")
subclass_labels = np.full(N, -1, dtype=int)

qcd_mask = (preds == 0)
top_mask  = (preds == 1)

print(f"    Fitting DPGMM on QCD ({qcd_mask.sum():,} jets in Tier 2)...")
qcd_latent  = all_emb_norm[qcd_mask]
dpgmm_qcd   = BayesianGaussianMixture(
    n_components=5, weight_concentration_prior=1e-3,
    random_state=DPGMM_SEEDS[0], n_init=3
)
qcd_subclasses         = dpgmm_qcd.fit_predict(qcd_latent)
active_qcd_sc, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
subclass_labels[qcd_mask] = qcd_subclasses
print(f"    QCD subclasses: {active_qcd_sc}  counts: {counts_qcd}")

print(f"    Fitting DPGMM on Top ({top_mask.sum():,} jets in Tier 2)...")
top_latent  = all_emb_norm[top_mask]
dpgmm_top   = BayesianGaussianMixture(
    n_components=5, weight_concentration_prior=1e-3,
    random_state=DPGMM_SEEDS[0], n_init=3
)
top_subclasses         = dpgmm_top.fit_predict(top_latent)
active_top_sc, counts_top = np.unique(top_subclasses, return_counts=True)
subclass_labels[top_mask] = top_subclasses
print(f"    Top subclasses: {active_top_sc}  counts: {counts_top}")

# ============================================================
# STEP 7: DPGMM robustness — NMI across 3 seeds (Tier 2)
# ============================================================
print("\n[7] DPGMM robustness check (NMI across seeds)")
robustness = {}
for cls_name, latent, ref_pred in [('QCD', qcd_latent, qcd_subclasses),
                                    ('Top', top_latent, top_subclasses)]:
    nmis = []
    for seed in DPGMM_SEEDS[1:]:
        dpg = BayesianGaussianMixture(
            n_components=5, weight_concentration_prior=1e-3,
            random_state=seed, n_init=2
        )
        alt = dpg.fit_predict(latent)
        nmis.append(float(normalized_mutual_info_score(ref_pred, alt)))
    robustness[cls_name] = {
        'nmi_mean': float(np.mean(nmis)),
        'nmi_std':  float(np.std(nmis)),
        'nmis':     nmis
    }
    print(f"    {cls_name} NMI: {np.mean(nmis):.4f} ± {np.std(nmis):.4f}")

# ============================================================
# STEP 8: FastJet physics — Tier 4 (stratified, chunked)
#   Stratify by class × subclass to ensure all groups are covered.
#   Chunked at FASTJET_CHUNK to avoid loading all events at once.
# ============================================================
print(f"\n[8] Building Tier 4 for FastJet ({FASTJET_EVENTS:,} events, stratified + chunked)")
n_groups = len(active_qcd_sc) + len(active_top_sc) + 4
per_group = FASTJET_EVENTS // n_groups
t4_local_idx = []
for sc in active_qcd_sc:
    idx = np.where(qcd_mask & (subclass_labels == sc))[0]
    t4_local_idx.extend(rng.choice(idx, min(per_group, len(idx)), replace=False).tolist())
for sc in active_top_sc:
    idx = np.where(top_mask & (subclass_labels == sc))[0]
    t4_local_idx.extend(rng.choice(idx, min(per_group, len(idx)), replace=False).tolist())
for mask_s in [tp, tn, fp, fn]:   # ensure T/F minorities are represented
    idx = np.where(mask_s)[0]
    n_s = min(per_group // 2, len(idx))
    if n_s > 0:
        t4_local_idx.extend(rng.choice(idx, n_s, replace=False).tolist())
t4_local_idx  = np.array(sorted(set(t4_local_idx)))   # local indices into Tier 2
t4_global_idx = t2_idx[t4_local_idx]                   # global indices into all_ev
print(f"    Tier 4 size: {len(t4_global_idx):,} events")

print(f"    Running FastJet in chunks of {FASTJET_CHUNK:,}...")
features_list = []
n_chunks = (len(t4_global_idx) - 1) // FASTJET_CHUNK + 1
for ci, chunk_start in enumerate(range(0, len(t4_global_idx), FASTJET_CHUNK)):
    chunk_end  = min(chunk_start + FASTJET_CHUNK, len(t4_global_idx))
    chunk_ev   = all_ev[t4_global_idx[chunk_start:chunk_end]]
    features_list.append(compute_features(chunk_ev))
    print(f"    Chunk {ci+1}/{n_chunks} ({chunk_end:,}/{len(t4_global_idx):,})...", flush=True)

features_t4 = np.vstack(features_list)
# Feature order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
mass    = features_t4[:, 0]   # Jet mass (GeV)
tau32   = features_t4[:, 7]   # τ₃/τ₂
d12     = features_t4[:, 4]   # √d12
zg      = features_t4[:, 8]   # SoftDrop zg
theta_g = features_t4[:, 9]   # SoftDrop ΔR
print(f"    mass={mass.mean():.1f} GeV | tau32={tau32.mean():.3f} | zg={zg.mean():.4f}")

# Re-map T/F masks into Tier 4 local space
t4_preds    = preds[t4_local_idx]
t4_labels   = labels[t4_local_idx]
t4_status   = status[t4_local_idx]
t4_subclass = subclass_labels[t4_local_idx]
t4_qcd      = (t4_preds == 0)
t4_top      = (t4_preds == 1)
t4_tp = (t4_preds == 1) & (t4_labels == 1)
t4_tn = (t4_preds == 0) & (t4_labels == 0)
t4_fp = (t4_preds == 1) & (t4_labels == 0)
t4_fn = (t4_preds == 0) & (t4_labels == 1)

# Jet pT from constituents (Tier 4 only)
print("    Computing jet pT from constituents (Tier 4)...")
jets_pt = np.zeros(len(t4_global_idx))
for k, gidx in enumerate(t4_global_idx):
    ev = all_ev[gidx]
    px_s = float(ak.sum(ev.px)); py_s = float(ak.sum(ev.py))
    jets_pt[k] = np.sqrt(px_s ** 2 + py_s ** 2)
two_m_pt = 2.0 * mass / (jets_pt + 1e-10)

# ============================================================
# =================== HELPER FUNCTIONS =======================
# ============================================================
# Note: feats / feat_names / tf_* use Tier 4 variables.
#       subclass / edge / umap plots use Tier 2 variables (preds, status, etc.)

feats      = [mass, tau32, d12, zg, theta_g]
feat_names = ['Mass (GeV)', r'$\tau_{32}$', r'$\sqrt{d_{12}}$ (GeV)', r'$z_g$', r'$\Delta R_{12}$']

tf_colors = {'TP': '#1f77b4', 'TN': '#2ca02c', 'FP': '#d62728', 'FN': '#ff7f0e'}


def lund_subsample_global(events, global_indices, n_max):
    """Subsample from a pre-computed array of global indices for Lund planes."""
    if len(global_indices) == 0:
        return None
    chosen = np.random.choice(global_indices, min(n_max, len(global_indices)), replace=False)
    return events[chosen]


def lund_from_t2_mask(events, t2_mask, t2_global_indices, n_max):
    """Get Lund sample using a Tier-2 boolean mask, mapping to global event indices."""
    local = np.where(t2_mask)[0]
    if len(local) == 0:
        return None
    chosen_local = np.random.choice(local, min(n_max, len(local)), replace=False)
    return events[t2_global_indices[chosen_local]]


def save(path, dpi=150):
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ============================================================
# =================== ORIGINAL PLOTS ========================
# ============================================================
print("\n[PLOTS] Generating original plots...")

# ------------------------------------------------------------------
# Plot 1: UMAP + Prediction Status (+ edge overlay)
# ------------------------------------------------------------------
print("  Plot 1: UMAP Prediction Status")
plt.figure(figsize=(11, 8))
cat_cfg = {
    'TN': dict(color='#2ca02c', marker='o', s=8,  alpha=0.2),
    'TP': dict(color='#1f77b4', marker='o', s=8,  alpha=0.2),
    'FP': dict(color='#d62728', marker='x', s=50, alpha=0.85),
    'FN': dict(color='#ff7f0e', marker='x', s=50, alpha=0.85),
}
for cat, cfg in cat_cfg.items():
    m = (status == cat)
    if m.sum() > 0:
        plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                    label=f"{cat} ({m.sum()})", **cfg)
plt.scatter(umap_coords[edge_mask, 0], umap_coords[edge_mask, 1],
            marker='D', s=12, color='purple', alpha=0.45,
            label=f'Edge |m|<{edge_threshold:.3f} ({edge_mask.sum()})')
plt.scatter(*proto_umap[0], marker='*', s=500, color='yellow',
            edgecolors='black', lw=1.5, label='$W_{QCD}$', zorder=10)
plt.scatter(*proto_umap[1], marker='*', s=500, color='magenta',
            edgecolors='black', lw=1.5, label='$W_{Top}$', zorder=10)
try:
    for cat in ['TN', 'TP']:
        m = (status == cat)
        if m.sum() > 50:
            sns.kdeplot(x=umap_coords[m, 0], y=umap_coords[m, 1],
                        levels=4, color=cat_cfg[cat]['color'], alpha=0.3, linewidths=1.2)
except Exception as e:
    print(f"    KDE skipped: {e}")
plt.title("EFN + ArcFace Latent Space: Prediction Status + Edge Jets", fontsize=13)
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.grid(True, alpha=0.15)
save(os.path.join(RESULTS_DIR, 'umap_prediction_status.png'))

# ------------------------------------------------------------------
# Plot 2: UMAP + Subclasses
# ------------------------------------------------------------------
print("  Plot 2: UMAP Subclasses")
plt.figure(figsize=(11, 8))
qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
for i, sc in enumerate(active_qcd_sc):
    m = qcd_mask & (subclass_labels == sc)
    plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                color=qcd_colors[i % 5], s=8, alpha=0.35,
                label=f"QCD SC {i} (N={m.sum()})")
for i, sc in enumerate(active_top_sc):
    m = top_mask & (subclass_labels == sc)
    plt.scatter(umap_coords[m, 0], umap_coords[m, 1],
                color=top_colors[i % 5], s=8, alpha=0.35,
                label=f"Top SC {i} (N={m.sum()})")
plt.scatter(*proto_umap[0], marker='*', s=500, color='yellow',
            edgecolors='black', lw=1.5, label='$W_{QCD}$', zorder=10)
plt.scatter(*proto_umap[1], marker='*', s=500, color='magenta',
            edgecolors='black', lw=1.5, label='$W_{Top}$', zorder=10)
plt.title("Discovered Subclasses in Latent Space (DPGMM + UMAP)", fontsize=13)
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.grid(True, alpha=0.15)
save(os.path.join(RESULTS_DIR, 'umap_subclasses.png'))

# ------------------------------------------------------------------
# Plot 3: Physics Distributions (5 features × 3 columns)
# ------------------------------------------------------------------
print("  Plot 3: Subclass Physics Distributions")
fig, axes = plt.subplots(5, 3, figsize=(18, 22))

# Col 0: Top subclasses (Tier 4)
for i, sc in enumerate(active_top_sc):
    m = t4_top & (t4_subclass == sc)
    if m.sum() > 5:
        for row, f in enumerate(feats):
            sns.kdeplot(f[m], ax=axes[row, 0], label=f"SC {i}", fill=True, alpha=0.1)
for row, nm in enumerate(feat_names):
    axes[row, 0].set_title(f"Top Subclasses: {nm}", fontsize=11)
    axes[row, 0].set_xlabel(nm); axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.15)

# Col 1: QCD TN vs FP (Tier 4)
for row, f in enumerate(feats):
    if t4_tn.sum() > 5:
        sns.kdeplot(f[t4_tn], ax=axes[row, 1], label=f"TN ({t4_tn.sum()})",
                    color='#2ca02c', fill=True, alpha=0.15)
    if t4_fp.sum() > 5:
        sns.kdeplot(f[t4_fp], ax=axes[row, 1], label=f"FP ({t4_fp.sum()})",
                    color='#d62728', lw=2, linestyle='--')
    axes[row, 1].set_title(f"QCD: TN vs FP — {feat_names[row]}", fontsize=11)
    axes[row, 1].set_xlabel(feat_names[row]); axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.15)

# Col 2: Top TP vs FN (Tier 4)
for row, f in enumerate(feats):
    if t4_tp.sum() > 5:
        sns.kdeplot(f[t4_tp], ax=axes[row, 2], label=f"TP ({t4_tp.sum()})",
                    color='#1f77b4', fill=True, alpha=0.15)
    if t4_fn.sum() > 5:
        sns.kdeplot(f[t4_fn], ax=axes[row, 2], label=f"FN ({t4_fn.sum()})",
                    color='#ff7f0e', lw=2, linestyle='--')
    axes[row, 2].set_title(f"Top: TP vs FN — {feat_names[row]}", fontsize=11)
    axes[row, 2].set_xlabel(feat_names[row]); axes[row, 2].legend(fontsize=8); axes[row, 2].grid(True, alpha=0.15)

plt.tight_layout()
save(os.path.join(RESULTS_DIR, 'subclass_physics_distributions.png'), dpi=120)

# ------------------------------------------------------------------
# Plot 4: Subclass Composition Heatmap
# ------------------------------------------------------------------
print("  Plot 4: Composition Heatmap (Tier 2)")
categories = ['TP', 'FP', 'TN', 'FN']
comp_matrix, row_labels = [], []
for i, sc in enumerate(active_top_sc):
    m = top_mask & (subclass_labels == sc); n_tot = m.sum()
    if n_tot > 0:
        row = [((status == c) & m).sum() / n_tot * 100 for c in categories]
        comp_matrix.append(row); row_labels.append(f"Top SC {i} (N={n_tot})")
for i, sc in enumerate(active_qcd_sc):
    m = qcd_mask & (subclass_labels == sc); n_tot = m.sum()
    if n_tot > 0:
        row = [((status == c) & m).sum() / n_tot * 100 for c in categories]
        comp_matrix.append(row); row_labels.append(f"QCD SC {i} (N={n_tot})")
if comp_matrix:
    plt.figure(figsize=(10, max(4, len(row_labels) * 0.7)))
    sns.heatmap(comp_matrix, annot=True, fmt=".1f",
                xticklabels=categories, yticklabels=row_labels,
                cmap="YlGnBu", cbar_kws={'label': '%'})
    plt.title("Subclass Classification Purity / Composition", fontsize=13)
    plt.xlabel("Prediction Status"); plt.ylabel("Subclass")
    save(os.path.join(RESULTS_DIR, 'subclass_composition_heatmap.png'))

# ------------------------------------------------------------------
# Plot 5: Representative Jets (QCD & Top, top-5 per subclass)
# ------------------------------------------------------------------
print("  Plot 5: Representative Jets")
# Representative jets use Tier 2 indices (cls_indices) mapped to global via t2_idx
for cls_name, active_sc, cls_subclasses, cls_mask_t2, dpgmm in [
    ('qcd', active_qcd_sc, qcd_subclasses, qcd_mask, dpgmm_qcd),
    ('top', active_top_sc, top_subclasses, top_mask, dpgmm_top),
]:
    cls_local = np.where(cls_mask_t2)[0]              # local indices in Tier 2
    cls_global = t2_idx[cls_local]                    # global indices in all_ev
    probs = dpgmm.predict_proba(all_emb_norm[cls_mask_t2])
    for i, sc in enumerate(active_sc):
        assigned = (cls_subclasses == sc)
        if assigned.sum() == 0:
            continue
        p = probs[:, sc].copy(); p[~assigned] = -1.0
        top5_rel = np.argsort(p)[::-1][:5]
        top5_abs = cls_global[top5_rel]               # global indices for all_ev
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for j, t_idx in enumerate(top5_abs):
            ev = all_ev[t_idx]
            px_a = ak.to_numpy(ev.px); py_a = ak.to_numpy(ev.py)
            pz_a = ak.to_numpy(ev.pz); e_a  = ak.to_numpy(ev.E)
            pt_a = np.sqrt(px_a ** 2 + py_a ** 2)
            eta_a = np.arcsinh(pz_a / (pt_a + 1e-10))
            phi_a = np.arctan2(py_a, px_a)
            z_e   = e_a / (e_a.sum() + 1e-10)
            axs[j].scatter(eta_a, phi_a, s=pt_a * 5, alpha=0.7,
                           c=z_e, cmap='plasma', edgecolors='black', lw=0.3)
            axs[j].set_xlabel(r'$\eta$'); axs[j].set_ylabel(r'$\phi$')
            axs[j].set_title(f"p={p[top5_rel[j]]:.2f}", fontsize=9)
            axs[j].grid(True, alpha=0.15)
        plt.suptitle(f"{cls_name.upper()} SC {i} – Top 5 Representative Jets", fontsize=13, y=1.05)
        plt.tight_layout()
        save(os.path.join(RESULTS_DIR, f'{cls_name}_subclass_{i}_representative_jets.png'), dpi=120)

# ------------------------------------------------------------------
# Plot 6: Cosine Similarity Distributions (per subclass)
# ------------------------------------------------------------------
print("  Plot 6: Cosine Similarity Distributions")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, sc in enumerate(active_qcd_sc):
    m = qcd_mask & (subclass_labels == sc)
    if m.sum() > 5:
        sns.kdeplot(cosine_sim[m, 0], ax=axes[0], label=f"SC {i}", fill=True, alpha=0.12)
axes[0].set_title(r"QCD Subclasses: Cosine Sim to $W_{QCD}$")
axes[0].set_xlabel("Cosine Similarity"); axes[0].legend(); axes[0].grid(True, alpha=0.15)
for i, sc in enumerate(active_top_sc):
    m = top_mask & (subclass_labels == sc)
    if m.sum() > 5:
        sns.kdeplot(cosine_sim[m, 1], ax=axes[1], label=f"SC {i}", fill=True, alpha=0.12)
axes[1].set_title(r"Top Subclasses: Cosine Sim to $W_{Top}$")
axes[1].set_xlabel("Cosine Similarity"); axes[1].legend(); axes[1].grid(True, alpha=0.15)
plt.tight_layout()
save(os.path.join(RESULTS_DIR, 'subclass_similarity_distributions.png'))

# ------------------------------------------------------------------
# Plot 7: Physics Correlation Matrices per subclass
# ------------------------------------------------------------------
print("  Plot 7: Physics Correlations")
n_top = len(active_top_sc); n_qcd = len(active_qcd_sc)
cols_n = max(n_top, n_qcd, 1)
fig, axes = plt.subplots(2, cols_n, figsize=(4 * cols_n, 8))
fnames_corr = ['Mass', r'$\tau_{32}$', r'$\sqrt{d_{12}}$', r'$z_g$']
for i, sc in enumerate(active_top_sc):
    m = t4_top & (t4_subclass == sc)
    ax = axes[0, i] if cols_n > 1 else axes[0]
    if m.sum() > 5:
        df = pd.DataFrame(np.stack([mass[m], tau32[m], d12[m], zg[m]], axis=1), columns=fnames_corr)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=False, ax=ax)
    ax.set_title(f"Top SC {i}", fontsize=10)
for i in range(len(active_top_sc), cols_n):
    (axes[0, i] if cols_n > 1 else axes[0]).axis('off')
for i, sc in enumerate(active_qcd_sc):
    m = t4_qcd & (t4_subclass == sc)
    ax = axes[1, i] if cols_n > 1 else axes[1]
    if m.sum() > 5:
        df = pd.DataFrame(np.stack([mass[m], tau32[m], d12[m], zg[m]], axis=1), columns=fnames_corr)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=False, ax=ax)
    ax.set_title(f"QCD SC {i}", fontsize=10)
for i in range(len(active_qcd_sc), cols_n):
    (axes[1, i] if cols_n > 1 else axes[1]).axis('off')
plt.suptitle(r"Physics Feature Correlations per Subclass (incl. $z_g$)", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(RESULTS_DIR, 'subclass_physics_correlations.png'))

# ------------------------------------------------------------------
# Plot 10: zg (Altarelli-Parisi splitting variable)
# ------------------------------------------------------------------
print("  Plot 10: zg AP Splitting")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, sc in enumerate(active_qcd_sc):
    m = t4_qcd & (t4_subclass == sc)
    if m.sum() > 5:
        zg_sc = zg[m]; zg_sc = zg_sc[(zg_sc > 0.05) & (zg_sc < 0.5)]
        if len(zg_sc) > 5:
            sns.kdeplot(zg_sc, ax=axes[0], label=f"QCD SC {i} (N={m.sum()})", fill=True, alpha=0.15)
axes[0].axvline(0.1, ls='--', color='gray', lw=1, label='$z_{cut}=0.1$')
axes[0].set_title(r"QCD: $z_g$ (Altarelli-Parisi Splitting)", fontsize=12)
axes[0].set_xlabel(r"$z_g$"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)
axes[0].text(0.97, 0.95, r"QCD: $z_g \to 0$ (soft divergence)",
             transform=axes[0].transAxes, ha='right', va='top', fontsize=9, color='gray',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

for i, sc in enumerate(active_top_sc):
    m = t4_top & (t4_subclass == sc)
    if m.sum() > 5:
        zg_sc = zg[m]; zg_sc = zg_sc[(zg_sc > 0.05) & (zg_sc < 0.5)]
        if len(zg_sc) > 5:
            sns.kdeplot(zg_sc, ax=axes[1], label=f"Top SC {i} (N={m.sum()})", fill=True, alpha=0.15)
axes[1].axvline(0.5, ls='--', color='blue', lw=1, label=r'$z_g=0.5$ (symmetric)')
axes[1].set_title(r"Top: $z_g$ ($W \to q\bar{q}$ decay)", fontsize=12)
axes[1].set_xlabel(r"$z_g$"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.15)
axes[1].text(0.97, 0.95, r"Top: $z_g \to 0.5$ ($W \to q\bar{q}$)",
             transform=axes[1].transAxes, ha='right', va='top', fontsize=9, color='blue',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.suptitle(r"Altarelli-Parisi Splitting Verification via $z_g$", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(RESULTS_DIR, 'zg_ap_splitting.png'))

# ------------------------------------------------------------------
# Plot 11: Angular Collimation ΔR₁₂ ≈ 2M/pT
# ------------------------------------------------------------------
print("  Plot 11: Angular Collimation")
valid = (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, t4_mask_c, label_c, col in [
    (axes[0], t4_qcd, 'QCD', '#2ca02c'),
    (axes[1], t4_top, 'Top', '#1f77b4'),
]:
    v = valid & t4_mask_c
    if v.sum() > 10:
        ax.scatter(two_m_pt[v], theta_g[v],
                   alpha=0.15, s=5, color=col, label=f"{label_c} (N={v.sum()})")
    ax.plot([0, 1.5], [0, 1.5], 'k--', lw=1.5, label=r'$\Delta R_{12} = 2M/p_T$')
    ax.set_xlabel(r"$2M/p_T$ (predicted $\Delta R$)")
    ax.set_ylabel(r"$\Delta R_{12}$ (SoftDrop measured)")
    ax.set_title(f"{label_c}: Angular Collimation", fontsize=12)
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.5)
    ax.legend(); ax.grid(True, alpha=0.15)
plt.suptitle(r"Relativistic Angular Scaling: $\Delta R_{12} \approx 2M/p_T$", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(RESULTS_DIR, 'angular_collimation.png'))

# ============================================================
# =================== T/F ANALYSIS ==========================
# ============================================================
print("\n[T/F ANALYSIS] Generating True/False analysis plots...")

# ------------------------------------------------------------------
# TF-1: Latent Cosine Similarity T/F at Class Level
# ------------------------------------------------------------------
print("  TF-1: Class-Level Latent T/F Similarity")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Predicted QCD — TN (correct) vs FN (missed top)
for label_str, mask_now, color, kws in [
    (f'TN: True QCD ({tn.sum()})', tn, '#2ca02c', dict(fill=True, alpha=0.2)),
    (f'FN: Missed Top ({fn.sum()})', fn, '#ff7f0e', dict(lw=2, linestyle='--')),
]:
    if mask_now.sum() > 5:
        sns.kdeplot(cosine_sim[mask_now, 0], ax=axes[0],
                    label=label_str, color=color, **kws)
axes[0].set_title(r"Predicted QCD: Sim to $W_{QCD}$ — TN vs FN", fontsize=12)
axes[0].set_xlabel(r"Cosine Similarity to $W_{QCD}$")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)

# Right: Predicted Top — TP (correct) vs FP (fake top)
for label_str, mask_now, color, kws in [
    (f'TP: True Top ({tp.sum()})', tp, '#1f77b4', dict(fill=True, alpha=0.2)),
    (f'FP: Fake Top ({fp.sum()})', fp, '#d62728', dict(lw=2, linestyle='--')),
]:
    if mask_now.sum() > 5:
        sns.kdeplot(cosine_sim[mask_now, 1], ax=axes[1],
                    label=label_str, color=color, **kws)
axes[1].set_title(r"Predicted Top: Sim to $W_{Top}$ — TP vs FP", fontsize=12)
axes[1].set_xlabel(r"Cosine Similarity to $W_{Top}$")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.15)

plt.suptitle("Latent Space Cosine Similarity: True vs False Predictions (Class Level)", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(TF_DIR, 'class_latent_tf_similarity.png'))

# ------------------------------------------------------------------
# TF-2: Latent Cosine Similarity T/F at Subclass Level
# ------------------------------------------------------------------
print("  TF-2: Subclass-Level Latent T/F Similarity")
n_cols = max(len(active_qcd_sc), len(active_top_sc))
fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

# Row 0: QCD subclasses — TN vs FN
for i, sc in enumerate(active_qcd_sc):
    m_sc = qcd_mask & (subclass_labels == sc)
    ax = axes[0, i] if n_cols > 1 else axes[0]
    m_tn_sc = m_sc & tn; m_fn_sc = m_sc & fn
    if m_tn_sc.sum() > 5:
        sns.kdeplot(cosine_sim[m_tn_sc, 0], ax=ax,
                    label=f"TN ({m_tn_sc.sum()})", color='#2ca02c', fill=True, alpha=0.2)
    if m_fn_sc.sum() > 5:
        sns.kdeplot(cosine_sim[m_fn_sc, 0], ax=ax,
                    label=f"FN ({m_fn_sc.sum()})", color='#ff7f0e', lw=2, linestyle='--')
    ax.set_title(f"QCD SC {i}: Sim to $W_{{QCD}}$\n(TN vs FN)", fontsize=10)
    ax.set_xlabel("Cosine Sim"); ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
for j in range(len(active_qcd_sc), n_cols):
    (axes[0, j] if n_cols > 1 else axes[0]).axis('off')

# Row 1: Top subclasses — TP vs FP
for i, sc in enumerate(active_top_sc):
    m_sc = top_mask & (subclass_labels == sc)
    ax = axes[1, i] if n_cols > 1 else axes[1]
    m_tp_sc = m_sc & tp; m_fp_sc = m_sc & fp
    if m_tp_sc.sum() > 5:
        sns.kdeplot(cosine_sim[m_tp_sc, 1], ax=ax,
                    label=f"TP ({m_tp_sc.sum()})", color='#1f77b4', fill=True, alpha=0.2)
    if m_fp_sc.sum() > 5:
        sns.kdeplot(cosine_sim[m_fp_sc, 1], ax=ax,
                    label=f"FP ({m_fp_sc.sum()})", color='#d62728', lw=2, linestyle='--')
    ax.set_title(f"Top SC {i}: Sim to $W_{{Top}}$\n(TP vs FP)", fontsize=10)
    ax.set_xlabel("Cosine Sim"); ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
for j in range(len(active_top_sc), n_cols):
    (axes[1, j] if n_cols > 1 else axes[1]).axis('off')

plt.suptitle("Subclass-Level Latent Similarity: T/F Splits", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(TF_DIR, 'subclass_latent_tf_similarity.png'))

# ------------------------------------------------------------------
# TF-3: Physics T/F at Class Level (5 features × 2 columns)
# ------------------------------------------------------------------
print("  TF-3: Class-Level Physics T/F Distributions (Tier 4)")
fig, axes = plt.subplots(5, 2, figsize=(14, 20))
for row, (feat, fname) in enumerate(zip(feats, feat_names)):
    # Col 0: QCD TN vs FN
    if t4_tn.sum() > 5:
        sns.kdeplot(feat[t4_tn], ax=axes[row, 0],
                    label=f"TN ({t4_tn.sum()})", color='#2ca02c', fill=True, alpha=0.2)
    if t4_fn.sum() > 5:
        sns.kdeplot(feat[t4_fn], ax=axes[row, 0],
                    label=f"FN ({t4_fn.sum()})", color='#ff7f0e', lw=2, linestyle='--')
    axes[row, 0].set_title(f"QCD: TN vs FN — {fname}", fontsize=11)
    axes[row, 0].set_xlabel(fname); axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.15)
    # Col 1: Top TP vs FP
    if t4_tp.sum() > 5:
        sns.kdeplot(feat[t4_tp], ax=axes[row, 1],
                    label=f"TP ({t4_tp.sum()})", color='#1f77b4', fill=True, alpha=0.2)
    if t4_fp.sum() > 5:
        sns.kdeplot(feat[t4_fp], ax=axes[row, 1],
                    label=f"FP ({t4_fp.sum()})", color='#d62728', lw=2, linestyle='--')
    axes[row, 1].set_title(f"Top: TP vs FP — {fname}", fontsize=11)
    axes[row, 1].set_xlabel(fname); axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.15)
plt.suptitle("FastJet Physics Features: True vs False (Class Level, Tier 4)", fontsize=13, y=1.005)
plt.tight_layout()
save(os.path.join(TF_DIR, 'class_physics_tf_distributions.png'), dpi=120)

# ------------------------------------------------------------------
# TF-4 & TF-5: Physics T/F at Subclass Level
#   QCD: TN vs FN  |  Top: TP vs FP
# ------------------------------------------------------------------
# TF-4 & TF-5: Physics T/F at Subclass Level (Tier 4)
for cls_name, active_sc, t4_cls_mask, t4_true_mask, t4_err_mask, true_cat, err_cat, tc, ec in [
    ('QCD', active_qcd_sc, t4_qcd, t4_tn, t4_fn, 'TN', 'FN', '#2ca02c', '#ff7f0e'),
    ('Top', active_top_sc, t4_top, t4_tp, t4_fp, 'TP', 'FP', '#1f77b4', '#d62728'),
]:
    n_sc = len(active_sc)
    print(f"  TF-{'4' if cls_name=='QCD' else '5'}: {cls_name} Subclass Physics T/F")
    fig, axes = plt.subplots(n_sc, 5, figsize=(20, 4 * n_sc))
    if n_sc == 1:
        axes = axes[np.newaxis, :]
    for row_i, sc in enumerate(active_sc):
        m_sc   = t4_cls_mask & (t4_subclass == sc)
        m_true = m_sc & t4_true_mask
        m_err  = m_sc & t4_err_mask
        for col_j, (feat, fname) in enumerate(zip(feats, feat_names)):
            ax = axes[row_i, col_j]
            if m_true.sum() > 5:
                sns.kdeplot(feat[m_true], ax=ax, label=f"{true_cat} ({m_true.sum()})",
                            color=tc, fill=True, alpha=0.2)
            if m_err.sum() > 5:
                sns.kdeplot(feat[m_err], ax=ax, label=f"{err_cat} ({m_err.sum()})",
                            color=ec, lw=2, linestyle='--')
            ax.set_title(f"{cls_name} SC {row_i}: {fname}", fontsize=9)
            ax.set_xlabel(fname, fontsize=8); ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
    plt.suptitle(f"{cls_name} Subclasses — {true_cat} vs {err_cat} Physics Distributions",
                 fontsize=13, y=1.005)
    plt.tight_layout()
    fname_out = f'qcd_subclass_physics_tf_distributions.png' if cls_name == 'QCD' \
                else f'top_subclass_physics_tf_distributions.png'
    save(os.path.join(TF_DIR, fname_out), dpi=120)

# ------------------------------------------------------------------
# TF-6: UMAP T/F distribution within each subclass
# ------------------------------------------------------------------
print("  TF-6: UMAP T/F within Subclasses")
n_cols_umap = max(len(active_qcd_sc), len(active_top_sc))
fig, axes = plt.subplots(2, n_cols_umap, figsize=(4 * n_cols_umap, 9))

for row_i, (cls_name, active_sc, cls_mask_now, true_cat, err_cat, tc, ec) in enumerate([
    ('QCD', active_qcd_sc, qcd_mask, 'TN', 'FN', '#2ca02c', '#ff7f0e'),
    ('Top', active_top_sc, top_mask,  'TP', 'FP', '#1f77b4', '#d62728'),
]):
    for i, sc in enumerate(active_sc):
        ax = axes[row_i, i] if n_cols_umap > 1 else axes[row_i]
        m_sc   = cls_mask_now & (subclass_labels == sc)
        m_true = m_sc & (status == true_cat)
        m_err  = m_sc & (status == err_cat)
        if m_true.sum() > 5:
            ax.scatter(umap_coords[m_true, 0], umap_coords[m_true, 1],
                       color=tc, s=8, alpha=0.4, label=f"{true_cat} ({m_true.sum()})")
        if m_err.sum() > 5:
            ax.scatter(umap_coords[m_err, 0], umap_coords[m_err, 1],
                       color=ec, marker='x', s=30, alpha=0.85,
                       label=f"{err_cat} ({m_err.sum()})")
        ax.set_title(f"{cls_name} SC {i}: {true_cat} vs {err_cat}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    for j in range(len(active_sc), n_cols_umap):
        (axes[row_i, j] if n_cols_umap > 1 else axes[row_i]).axis('off')

plt.suptitle("UMAP: T/F Distribution within Subclasses", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(TF_DIR, 'umap_subclass_tf_splits.png'))

# ============================================================
# =================== EDGE ANALYSIS =========================
# ============================================================
print("\n[EDGE ANALYSIS] Decision boundary jets...")

core_mask = ~edge_mask

# ------------------------------------------------------------------
# Edge-1: Margin distribution + composition bar
# ------------------------------------------------------------------
print("  Edge-1: Margin Distribution & Composition")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(margin, ax=axes[0], bins=100, color='steelblue', stat='density', alpha=0.7)
axes[0].axvline(0, color='black', lw=1.5, ls='--', label='Decision boundary (margin=0)')
axes[0].axvline( edge_threshold, color='red', lw=1.5, ls=':', label=f'Edge threshold ±{edge_threshold:.3f}')
axes[0].axvline(-edge_threshold, color='red', lw=1.5, ls=':')
axes[0].fill_betweenx([0, axes[0].get_ylim()[1] if len(axes[0].get_ylim()) > 0 else 1],
                       -edge_threshold, edge_threshold, alpha=0.08, color='purple',
                       label=f'Edge zone ({edge_mask.sum()} jets)')
axes[0].set_title("Cosine Similarity Margin Distribution")
axes[0].set_xlabel("Margin: $\\text{sim}(W_{Top}) - \\text{sim}(W_{QCD})$")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.15)

edge_comp = {cat: int(((status == cat) & edge_mask).sum()) for cat in ['TP', 'TN', 'FP', 'FN']}
bars = axes[1].bar(edge_comp.keys(), edge_comp.values(),
                   color=[tf_colors[c] for c in edge_comp.keys()], edgecolor='black', lw=0.5)
axes[1].set_title(f"Edge Jets Composition ({edge_mask.sum()} jets)")
axes[1].set_ylabel("Count"); axes[1].grid(True, alpha=0.15, axis='y')
for bar, (k, v) in zip(bars, edge_comp.items()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{v}\n({v/edge_mask.sum()*100:.1f}%)", ha='center', fontsize=9)
plt.suptitle("Decision Boundary: Edge Jet Analysis", fontsize=13, y=1.02)
plt.tight_layout()
save(os.path.join(EDGE_DIR, 'edge_margin_distribution.png'))

# ------------------------------------------------------------------
# Edge-2: Edge vs Core physics comparison
# ------------------------------------------------------------------
print("  Edge-2: Edge vs Core Physics (Tier 4)")
# Need edge/core masks in Tier 4 space
t4_edge_mask = np.abs(margin[t4_local_idx]) < edge_threshold
t4_core_mask = ~t4_edge_mask
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for row_i, (t4_cls, cls_name) in enumerate([(t4_qcd, 'QCD'), (t4_top, 'Top')]):
    for col_j, (feat, fname) in enumerate(zip(feats, feat_names)):
        ax = axes[row_i, col_j]
        m_core = t4_cls & t4_core_mask
        m_edge = t4_cls & t4_edge_mask
        if m_core.sum() > 5:
            sns.kdeplot(feat[m_core], ax=ax, label=f"Core ({m_core.sum()})",
                        color='steelblue', fill=True, alpha=0.2)
        if m_edge.sum() > 5:
            sns.kdeplot(feat[m_edge], ax=ax, label=f"Edge ({m_edge.sum()})",
                        color='purple', lw=2, linestyle='--')
        ax.set_title(f"{cls_name}: {fname}", fontsize=9)
        ax.set_xlabel(fname, fontsize=8); ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
plt.suptitle("Edge vs Core Jets: Physics Feature Comparison", fontsize=13, y=1.005)
plt.tight_layout()
save(os.path.join(EDGE_DIR, 'edge_vs_core_physics.png'), dpi=120)

# ------------------------------------------------------------------
# Edge-3: UMAP location of edge jets
# ------------------------------------------------------------------
print("  Edge-3: Edge Jets UMAP Location")
plt.figure(figsize=(11, 8))
plt.scatter(umap_coords[qcd_mask & core_mask, 0], umap_coords[qcd_mask & core_mask, 1],
            color='#74c476', s=6, alpha=0.2, label=f"QCD Core ({(qcd_mask & core_mask).sum()})")
plt.scatter(umap_coords[top_mask & core_mask, 0], umap_coords[top_mask & core_mask, 1],
            color='#6baed6', s=6, alpha=0.2, label=f"Top Core ({(top_mask & core_mask).sum()})")
plt.scatter(umap_coords[qcd_mask & edge_mask, 0], umap_coords[qcd_mask & edge_mask, 1],
            color='darkgreen', s=30, alpha=0.9, marker='D',
            label=f"QCD Edge ({(qcd_mask & edge_mask).sum()})")
plt.scatter(umap_coords[top_mask & edge_mask, 0], umap_coords[top_mask & edge_mask, 1],
            color='navy', s=30, alpha=0.9, marker='D',
            label=f"Top Edge ({(top_mask & edge_mask).sum()})")
plt.scatter(*proto_umap[0], marker='*', s=500, color='yellow',
            edgecolors='black', lw=1.5, label='$W_{QCD}$', zorder=10)
plt.scatter(*proto_umap[1], marker='*', s=500, color='magenta',
            edgecolors='black', lw=1.5, label='$W_{Top}$', zorder=10)
plt.title("Edge Jets Location in Latent Space", fontsize=13)
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.grid(True, alpha=0.15)
save(os.path.join(EDGE_DIR, 'edge_umap_location.png'))

# ------------------------------------------------------------------
# Edge-4 & 5: Lund Planes for edge jets (sampled)
# ------------------------------------------------------------------
print("  Edge Lund Planes (Tier 2 → global indices)")
for label_str, t2_mask_e, fname_out in [
    (f"QCD Edge Jets (|margin|<{edge_threshold:.3f})", qcd_mask & edge_mask, 'edge_qcd_lund.png'),
    (f"Top Edge Jets (|margin|<{edge_threshold:.3f})", top_mask & edge_mask, 'edge_top_lund.png'),
    (f"QCD Core Jets",                                 qcd_mask & core_mask, 'core_qcd_lund.png'),
    (f"Top Core Jets",                                 top_mask & core_mask, 'core_top_lund.png'),
]:
    ev_sub = lund_from_t2_mask(all_ev, t2_mask_e, t2_idx, LUND_SAMPLE)
    if ev_sub is not None:
        plot_lund_plane(ev_sub, label=label_str,
                        save_path=os.path.join(EDGE_DIR, fname_out))

# ============================================================
# =================== LUND PLANES ===========================
# ============================================================
print("\n[LUND PLANES] Generating hierarchical Lund planes (sampled)...")

# Class + Status level
for label_str, t2_mask_l, fname_out in [
    ('Predicted QCD',  qcd_mask,       'class_predicted_qcd.png'),
    ('Predicted Top',  top_mask,        'class_predicted_top.png'),
    ('True Negative',  tn,              'status_tn.png'),
    ('True Positive',  tp,              'status_tp.png'),
    ('False Positive', fp,              'status_fp.png'),
    ('False Negative', fn,              'status_fn.png'),
]:
    ev_sub = lund_from_t2_mask(all_ev, t2_mask_l, t2_idx, LUND_SAMPLE)
    if ev_sub is not None:
        plot_lund_plane(ev_sub, label=label_str,
                        save_path=os.path.join(LUND_DIR, fname_out))

# Subclass + T/F level
for cls_name, active_sc, cls_mask_t2, true_cat, err_cat in [
    ('qcd', active_qcd_sc, qcd_mask, 'TN', 'FN'),
    ('top', active_top_sc, top_mask,  'TP', 'FP'),
]:
    for i, sc in enumerate(active_sc):
        m_sc   = cls_mask_t2 & (subclass_labels == sc)
        m_true = m_sc & (status == true_cat)
        m_err  = m_sc & (status == err_cat)
        for lbl, msk_t2, suffix in [
            (f"{cls_name.upper()} SC {i}",           m_sc,   ''),
            (f"{cls_name.upper()} SC {i} ({true_cat})", m_true, f'_{true_cat.lower()}'),
            (f"{cls_name.upper()} SC {i} ({err_cat})",  m_err,  f'_{err_cat.lower()}'),
        ]:
            ev_sub = lund_from_t2_mask(all_ev, msk_t2, t2_idx, LUND_SAMPLE)
            if ev_sub is not None:
                plot_lund_plane(ev_sub, label=lbl,
                                save_path=os.path.join(LUND_DIR, f'{cls_name}_subclass_{i}{suffix}.png'))

# ============================================================
# =================== SAVE METRICS ==========================
# ============================================================
metrics = {
    # Full dataset statistics (Tier 1)
    'n_total':          N_TOTAL,
    'accuracy_full':    float(acc),       # computed on all N_TOTAL
    'roc_auc':          float(roc_auc),
    'tp_full': int(tp_full.sum()), 'tn_full': int(tn_full.sum()),
    'fp_full': int(fp_full.sum()), 'fn_full': int(fn_full.sum()),
    # Tier 2 analysis (100k stratified)
    'n_tier2':          N,
    'tp_t2': int(tp.sum()), 'tn_t2': int(tn.sum()),
    'fp_t2': int(fp.sum()), 'fn_t2': int(fn.sum()),
    # Tier 4 FastJet (100k)
    'n_tier4':          int(len(t4_global_idx)),
    # Edge analysis (Tier 2)
    'edge_count':       int(edge_mask.sum()),
    'edge_threshold':   float(edge_threshold),
    # DPGMM (Tier 2)
    'qcd_subclasses':   int(len(active_qcd_sc)),
    'top_subclasses':   int(len(active_top_sc)),
    'qcd_sc_counts':    [int(c) for c in counts_qcd],
    'top_sc_counts':    [int(c) for c in counts_top],
    'dpgmm_robustness': {k: {'nmi_mean': v['nmi_mean'], 'nmi_std': v['nmi_std']}
                         for k, v in robustness.items()}
}
with open(os.path.join(RESULTS_DIR, 'analysis_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*60}")
print(f"[COMPLETE] All results saved to {RESULTS_DIR}/")
print(f"  Accuracy (full {N_TOTAL:,}): {acc:.4f}")
print(f"  ROC AUC:                   {roc_auc:.4f}")
print(f"  T/F (Tier 2 {N:,}): TP={tp.sum()} TN={tn.sum()} FP={fp.sum()} FN={fn.sum()}")
print(f"  Edge jets (Tier 2):        {edge_mask.sum()} (|margin| < {edge_threshold:.4f})")
print(f"  QCD DPGMM NMI: {robustness['QCD']['nmi_mean']:.4f} ± {robustness['QCD']['nmi_std']:.4f}")
print(f"  Top DPGMM NMI: {robustness['Top']['nmi_mean']:.4f} ± {robustness['Top']['nmi_std']:.4f}")
print(f"  FastJet Tier 4: {len(t4_global_idx):,} events processed")
print(f"\n  Output: {RESULTS_DIR}/")
print(f"  T/F:     {TF_DIR}/")
print(f"  Edge:    {EDGE_DIR}/")
print(f"  Lund:    {LUND_DIR}/")
print('='*60)
