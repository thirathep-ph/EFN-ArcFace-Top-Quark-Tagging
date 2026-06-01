import os
import sys
import json
import time
import gc
import numpy as np
import awkward as ak
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.models.efn_arcface import count_parameters
from jet_tagging.utils.physics import compute_features
from jet_tagging.data.loader import get_h5_len, load_events_by_indices, JetTaggingDataset
from jet_tagging.utils.plotting import (
    plot_latent_space,
    plot_performance,
    plot_representative_jet,
    plot_physics_correlations,
    plot_lund_plane,
    plot_training_results,
    plot_decision_boundary,
    plot_feature_distributions
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Offline Local Analysis for EFN+ArcFace")
    parser.add_argument('--embeddings_path', type=str, default='main_arcface/embeddings.npz',
                        help='Path to saved test embeddings (.npz)')
    parser.add_argument('--checkpoint_path', type=str, default='main_arcface/best_model_efn.pt',
                        help='Path to EFN model checkpoint (.pt)')
    parser.add_argument('--test_h5_path', type=str, default='data/top_tagging/test.h5',
                        help='Path to test.h5 data file')
    parser.add_argument('--output_dir', type=str, default='main_arcface',
                        help='Directory to save output plots and metrics')
    parser.add_argument('--sample_size', type=int, default=50000,
                        help='Sample size for local UMAP, Spectral Clustering & Lund Planes')
    return parser.parse_args()

def estimate_clusters_eigengap(X_norm, max_k=10):
    from scipy.sparse.linalg import eigsh, LinearOperator
    n_samples = X_norm.shape[0]
    if n_samples < 3:
        return 2, None
    sum_X = np.sum(X_norm, axis=0)
    d = (n_samples + X_norm @ sum_X) / 2.0
    d = np.clip(d, 1e-10, None)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    
    def matvec(v):
        y = d_inv_sqrt * v
        Ay = (np.sum(y) + X_norm @ (X_norm.T @ y)) / 2.0
        return d_inv_sqrt * Ay
        
    M_op = LinearOperator((n_samples, n_samples), matvec=matvec)
    try:
        k_eigen = min(max_k + 2, n_samples - 2)
        vals, _ = eigsh(M_op, k=k_eigen, which='LM')
        eigenvalues = np.sort(1.0 - vals)
        max_k_checked = min(max_k, len(eigenvalues) - 2)
        if max_k_checked < 2:
            return 2, eigenvalues
        gaps = np.diff(eigenvalues[1:max_k_checked + 2])
        optimal_k = np.argmax(gaps) + 2
        return optimal_k, eigenvalues
    except Exception as e:
        print(f"Warning in eigengap calculation: {e}. Falling back to default k=5.")
        return 5, None

def spectral_clustering_subclass(X, max_k=10):
    from scipy.sparse.linalg import eigsh, LinearOperator
    from sklearn.cluster import KMeans
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), 1
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    optimal_k, _ = estimate_clusters_eigengap(X_norm, max_k=max_k)
    print(f"  Eigengap heuristic suggests optimal subclasses k = {optimal_k}")
    sum_X = np.sum(X_norm, axis=0)
    d = (n_samples + X_norm @ sum_X) / 2.0
    d = np.clip(d, 1e-10, None)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    
    def matvec(v):
        y = d_inv_sqrt * v
        Ay = (np.sum(y) + X_norm @ (X_norm.T @ y)) / 2.0
        return d_inv_sqrt * Ay
        
    M_op = LinearOperator((n_samples, n_samples), matvec=matvec)
    try:
        vals, vecs = eigsh(M_op, k=optimal_k, which='LM')
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        U = vecs / (row_norms + 1e-10)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(U)
        return labels, optimal_k
    except Exception as e:
        print(f"Warning in spectral clustering: {e}. Falling back to standard K-means.")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)
        return labels, optimal_k

def main():
    args = parse_args()
    
    # 1. Verify existence of files
    if not os.path.exists(args.embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at: {args.embeddings_path}")
    
    if not os.path.exists(args.test_h5_path):
        raise FileNotFoundError(f"Test data file not found at: {args.test_h5_path}")
        
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        fallback_path = "checkpoints/best_model_efn.pt"
        if os.path.exists(fallback_path):
            print(f"Warning: {checkpoint_path} not found. Falling back to fallback path: {fallback_path}")
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found. Please download best_model_efn.pt and put it at {checkpoint_path}")
            
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STARTING OFFLINE LOCAL ANALYSIS (Stratified Sampling)")
    print("=" * 60)
    
    # 2. Load Checkpoint and Extract Class Centers
    print(f"Loading checkpoint weights from: '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if checkpoint_path == "checkpoints/best_model_efn.pt" or "main_arcface" not in checkpoint_path:
        print("\n" + "!" * 65)
        print("WARNING: Using a fallback checkpoint.")
        print("If these weights are from an older run, they will not match your new embeddings.npz!")
        print("Please download the new best_model_efn.pt from your Colab run and place it in main_arcface/")
        print("!" * 65 + "\n")
        
    if 'arcface_head.class_centers' in state_dict:
        class_centers = state_dict['arcface_head.class_centers'].cpu().numpy()
    elif 'prototype_layer.prototypes' in state_dict:
        class_centers = state_dict['prototype_layer.prototypes'].cpu().numpy()
    else:
        raise KeyError("Could not find 'arcface_head.class_centers' or 'prototype_layer.prototypes' in the state_dict keys.")
    print(f"Extracted Class Centers from state_dict. Shape: {class_centers.shape}")
    
    # 3. Load Embeddings & Labels
    print(f"Loading test embeddings from: '{args.embeddings_path}'...")
    data = np.load(args.embeddings_path)
    embeddings = data['embeddings'] # shape (404000, 64)
    labels = data['labels'] # shape (404000,)
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # 4. Compute Cosine Similarities on FULL Test Set
    print("\nComputing Cosine Similarities on FULL Test Set (404k jets)...")
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    
    # Compute Cosine Similarity to each center
    t_sims = np.dot(embeddings_norm, centers_norm.T) # shape (404000, 2)
    
    # Predictions are based on argmax (highest similarity)
    preds = np.argmax(t_sims, axis=1)
    
    # CORRECTED ROC DECISION SCORE: Top Similarity - QCD Similarity (forces diagonal separation)
    decision_scores = t_sims[:, 1] - t_sims[:, 0]
    
    # 5. Compute Full Confusion Matrix & Corrected Metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(labels, preds)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    print(f"Full Test Accuracy (Calculated): {acc:.5f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Load history file if available to update metrics plots
    history_path = os.path.join(args.output_dir, 'history.json')
    history = {'train_loss': [0.0], 'train_acc': [0.0], 'val_loss': [0.0], 'val_acc': [0.0], 
               'sep_qcd': [0.0], 'sep_top': [0.0]}
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            
    # Plot updated training results with corrected Confusion Matrix
    plot_training_results(history, cm, save_path=os.path.join(args.output_dir, 'training_results_efn.png'))
    
    # Save/Update advanced_metrics.json
    advanced_metrics = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
        'num_params': len(state_dict.keys()) # placeholder for stats
    }
    with open(os.path.join(args.output_dir, 'advanced_metrics.json'), 'w') as f:
        json.dump(advanced_metrics, f, indent=4)
        
    # Plot Corrected Performance (Passing 1D decision_scores avoids plotting.py len>1 branch)
    print("Plotting Corrected ROC Curve (using Top - QCD Cosine Similarity difference)...")
    plot_performance(labels, decision_scores, save_path=os.path.join(args.output_dir, 'performance.png'))
    
    # Plot Decision Boundary in similarity space
    plot_decision_boundary(labels, t_sims, save_path=os.path.join(args.output_dir, 'decision_boundary.png'))
    
    # 6. Perform Stratified Sample Selection for UMAP and Spectral Clustering (50k sample)
    print(f"\nPerforming Stratified Sampling (N = {args.sample_size:,} jets)...")
    n_sample_per_class = args.sample_size // 2
    qcd_indices_all = np.where(labels == 0)[0]
    top_indices_all = np.where(labels == 1)[0]
    
    np.random.seed(42)
    sampled_qcd_idx = np.random.choice(qcd_indices_all, min(len(qcd_indices_all), n_sample_per_class), replace=False)
    sampled_top_idx = np.random.choice(top_indices_all, min(len(top_indices_all), n_sample_per_class), replace=False)
    
    # Combined, shuffled sampled indices (in terms of test set)
    sampled_test_idx = np.concatenate([sampled_qcd_idx, sampled_top_idx])
    np.random.shuffle(sampled_test_idx)
    
    sub_emb = embeddings[sampled_test_idx]
    sub_labels = labels[sampled_test_idx]
    sub_preds = preds[sampled_test_idx]
    sub_sims = t_sims[sampled_test_idx]
    sub_scores = decision_scores[sampled_test_idx]
    
    sub_emb_norm = sub_emb / (np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-10)
    print(f"Sampled subset: {len(sub_emb_norm)} embeddings (QCD={len(sampled_qcd_idx)}, Top={len(sampled_top_idx)})")
    
    # 7. Fit UMAP on 50k normalized embeddings
    import umap
    print("Fitting UMAP (metric='cosine') in parallel on all CPU cores...")
    t_umap_start = time.time()
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=None, n_jobs=-1)
    umap_coords = reducer.fit_transform(sub_emb_norm)
    centers_umap = reducer.transform(centers_norm)
    print(f"UMAP fit complete in {time.time() - t_umap_start:.1f}s.")
    
    # 8. Spectral Clustering Subclass Discovery
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    top_mask = (sub_preds == 1)
    print("\nFitting Spectral Clustering on predicted Top embeddings in subset...")
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, k_top = spectral_clustering_subclass(top_latent, max_k=10)
    active_top, counts_top = np.unique(top_subclasses, return_counts=True)
    print(f"Top Active Subclasses: {active_top} with counts {counts_top}")
    subclass_labels[top_mask] = top_subclasses
    
    qcd_mask = (sub_preds == 0)
    print("\nFitting Spectral Clustering on predicted QCD embeddings in subset...")
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, k_qcd = spectral_clustering_subclass(qcd_latent, max_k=10)
    active_qcd, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
    print(f"QCD Active Subclasses: {active_qcd} with counts {counts_qcd}")
    subclass_labels[qcd_mask] = qcd_subclasses
    
    # Calculate Prediction Status for 50k subset
    tp_mask = (sub_preds == 1) & (sub_labels == 1)
    tn_mask = (sub_preds == 0) & (sub_labels == 0)
    fp_mask = (sub_preds == 1) & (sub_labels == 0)
    fn_mask = (sub_preds == 0) & (sub_labels == 1)
    
    status = np.zeros(len(sub_emb), dtype=object)
    status[tp_mask] = 'True Positive'
    status[tn_mask] = 'True Negative'
    status[fp_mask] = 'False Positive'
    status[fn_mask] = 'False Negative'
    
    # Plot 1: umap_prediction_status.png
    print("\nGenerating UMAP Prediction Status plot...")
    plt.figure(figsize=(10, 8))
    colors = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4', 'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
    markers = {'True Negative': 'o', 'True Positive': 'o', 'False Positive': 'x', 'False Negative': 'x'}
    sizes = {'True Negative': 15, 'True Positive': 15, 'False Positive': 40, 'False Negative': 40}
    alphas = {'True Negative': 0.3, 'True Positive': 0.3, 'False Positive': 0.8, 'False Negative': 0.8}
    
    for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
        mask_cat = (status == cat)
        if mask_cat.sum() > 0:
            plt.scatter(umap_coords[mask_cat, 0], umap_coords[mask_cat, 1],
                        color=colors[cat], marker=markers[cat], s=sizes[cat],
                        alpha=alphas[cat], label=f"{cat} ({mask_cat.sum()})")
                        
    plt.scatter(centers_umap[0, 0], centers_umap[0, 1], marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
    plt.scatter(centers_umap[1, 0], centers_umap[1, 1], marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
    
    try:
        for cat in ['True Negative', 'True Positive']:
            mask_cat = (status == cat)
            if mask_cat.sum() > 50:
                sns.kdeplot(x=umap_coords[mask_cat, 0], y=umap_coords[mask_cat, 1], levels=4, color=colors[cat], alpha=0.3, linewidths=1.2)
    except:
        pass
        
    plt.title("EFN + ArcFace Latent Space (UMAP) & Prediction Status", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.15)
    plt.savefig(os.path.join(args.output_dir, 'umap_prediction_status.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: umap_subclasses.png
    print("Generating UMAP Subclasses plot...")
    plt.figure(figsize=(10, 8))
    qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
    for i, sc in enumerate(active_qcd):
        sc_mask = (sub_preds == 0) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            plt.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], color=qcd_colors[i % len(qcd_colors)], s=15, alpha=0.5, label=f"QCD Subclass {i} (N={sc_mask.sum()})")
            
    top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    for i, sc in enumerate(active_top):
        sc_mask = (sub_preds == 1) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            plt.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], color=top_colors[i % len(top_colors)], s=15, alpha=0.5, label=f"Top Subclass {i} (N={sc_mask.sum()})")
            
    plt.scatter(centers_umap[0, 0], centers_umap[0, 1], marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
    plt.scatter(centers_umap[1, 0], centers_umap[1, 1], marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
    plt.title("Discovered Subclasses in Latent Space (UMAP + KASP)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.15)
    plt.savefig(os.path.join(args.output_dir, 'umap_subclasses.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 9. Stratified sampling within 50k subset to compute FastJet physics and Lund Planes
    print("\nStratified sampling within subset for FastJet features and Lund Planes...")
    sampled_indices_phys = []
    
    # A. Sample from each active Top subclass (predicted Top)
    for sc in active_top:
        sc_indices = np.where((sub_preds == 1) & (subclass_labels == sc))[0]
        if len(sc_indices) > 0:
            n_sample = min(1000, len(sc_indices))
            sampled_indices_phys.extend(np.random.choice(sc_indices, n_sample, replace=False))
            
    # B. Sample from each active QCD subclass (predicted QCD)
    for sc in active_qcd:
        sc_indices = np.where((sub_preds == 0) & (subclass_labels == sc))[0]
        if len(sc_indices) > 0:
            n_sample = min(1000, len(sc_indices))
            sampled_indices_phys.extend(np.random.choice(sc_indices, n_sample, replace=False))
            
    # C. Sample from each prediction status (TP, TN, FP, FN)
    for cat in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
        cat_indices = np.where(status == cat)[0]
        if len(cat_indices) > 0:
            n_sample = min(1000, len(cat_indices))
            sampled_indices_phys.extend(np.random.choice(cat_indices, n_sample, replace=False))
            
    # Unique indices relative to the 50k subset
    sampled_indices_phys = np.unique(sampled_indices_phys)
    print(f"Selected {len(sampled_indices_phys)} physics sampled jets from the 50k subset.")
    
    # Map back to global test set indices to load H5 files
    global_phys_idx = sampled_test_idx[sampled_indices_phys]
    
    # Load dynamically from test.h5
    print(f"Loading awkward arrays for {len(global_phys_idx)} jets from '{args.test_h5_path}'...")
    phys_events = load_events_by_indices(args.test_h5_path, global_phys_idx)
    
    # Extract subsets corresponding to these physics events
    phys_labels = sub_labels[sampled_indices_phys]
    phys_scores = sub_scores[sampled_indices_phys]
    phys_subclass = subclass_labels[sampled_indices_phys]
    phys_status = status[sampled_indices_phys]
    phys_preds = sub_preds[sampled_indices_phys]
    
    # Run compute_features (includes FastJet C/A radius R=2.0)
    print("Computing FastJet jet substructure features (R=2.0, SoftDrop R0=2.0)...")
    phys_features = compute_features(phys_events)
    feature_names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    
    # Plot 3: subclass_physics_distributions.png
    print("Generating subclass physics distributions KDE plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    mass_all = phys_features[:, 0]
    tau32_all = phys_features[:, 7]
    d12_all = phys_features[:, 4]
    
    # Col 1: Discovered Top Subclasses
    for i, sc in enumerate(active_top):
        sc_mask = (phys_preds == 1) & (phys_subclass == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(mass_all[sc_mask], ax=axes[0, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
            sns.kdeplot(tau32_all[sc_mask], ax=axes[1, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
            sns.kdeplot(d12_all[sc_mask], ax=axes[2, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[0, 0].set_title("Discovered Top Subclasses: Jet Mass", fontsize=12)
    axes[0, 0].set_xlabel("Mass (GeV)")
    axes[0, 0].legend()
    axes[1, 0].set_title("Discovered Top Subclasses: N-subjettiness Tau32", fontsize=12)
    axes[1, 0].set_xlabel("Tau32")
    axes[1, 0].legend()
    axes[2, 0].set_title("Discovered Top Subclasses: Splitting Scale sqrt(d12)", fontsize=12)
    axes[2, 0].set_xlabel("sqrt(d12) (GeV)")
    axes[2, 0].legend()
    
    # Col 2: QCD TN vs QCD FP
    tn_sub = (phys_status == 'True Negative')
    fp_sub = (phys_status == 'False Positive')
    if tn_sub.sum() > 5:
        sns.kdeplot(mass_all[tn_sub], ax=axes[0, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
        sns.kdeplot(tau32_all[tn_sub], ax=axes[1, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
        sns.kdeplot(d12_all[tn_sub], ax=axes[2, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
    if fp_sub.sum() > 5:
        sns.kdeplot(mass_all[fp_sub], ax=axes[0, 1], label="Mistaken QCD (FP)", color='red', lw=2)
        sns.kdeplot(tau32_all[fp_sub], ax=axes[1, 1], label="Mistaken QCD (FP)", color='red', lw=2)
        sns.kdeplot(d12_all[fp_sub], ax=axes[2, 1], label="Mistaken QCD (FP)", color='red', lw=2)
    axes[0, 1].set_title("QCD Mistakes: Jet Mass", fontsize=12)
    axes[0, 1].set_xlabel("Mass (GeV)")
    axes[0, 1].legend()
    axes[1, 1].set_title("QCD Mistakes: N-subjettiness Tau32", fontsize=12)
    axes[1, 1].set_xlabel("Tau32")
    axes[1, 1].legend()
    axes[2, 1].set_title("QCD Mistakes: Splitting Scale sqrt(d12)", fontsize=12)
    axes[2, 1].set_xlabel("sqrt(d12) (GeV)")
    axes[2, 1].legend()
    
    # Col 3: Top TP vs Top FN
    tp_sub = (phys_status == 'True Positive')
    fn_sub = (phys_status == 'False Negative')
    if tp_sub.sum() > 5:
        sns.kdeplot(mass_all[tp_sub], ax=axes[0, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
        sns.kdeplot(tau32_all[tp_sub], ax=axes[1, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
        sns.kdeplot(d12_all[tp_sub], ax=axes[2, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
    if fn_sub.sum() > 5:
        sns.kdeplot(mass_all[fn_sub], ax=axes[0, 2], label="Mistaken Top (FN)", color='orange', lw=2)
        sns.kdeplot(tau32_all[fn_sub], ax=axes[1, 2], label="Mistaken Top (FN)", color='orange', lw=2)
        sns.kdeplot(d12_all[fn_sub], ax=axes[2, 2], label="Mistaken Top (FN)", color='orange', lw=2)
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
    plt.savefig(os.path.join(args.output_dir, 'subclass_physics_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 4: subclass_composition_heatmap.png
    print("Generating subclass composition heatmap...")
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    composition_matrix = []
    row_labels = []

    if len(active_top) > 0:
        for i, sc in enumerate(active_top):
            sc_mask = (sub_preds == 1) & (subclass_labels == sc)
            n_tot = sc_mask.sum()
            if n_tot > 0:
                row_counts = []
                for cat in categories:
                    cat_count = ((status == cat) & sc_mask).sum()
                    row_counts.append(cat_count / n_tot * 100.0)
                composition_matrix.append(row_counts)
                row_labels.append(f"Top Subclass {i} (N={n_tot})")

    if len(active_qcd) > 0:
        for i, sc in enumerate(active_qcd):
            sc_mask = (sub_preds == 0) & (subclass_labels == sc)
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
        plt.savefig(os.path.join(args.output_dir, 'subclass_composition_heatmap.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # Plot 5: Representative Event Displays (Centroids)
    print("Generating representative subclass event displays...")
    if len(active_top) > 0:
        for i, sc in enumerate(active_top):
            assigned_jets_in_top = (phys_preds == 1) & (phys_subclass == sc)
            if assigned_jets_in_top.sum() > 0:
                sc_emb = phys_features[assigned_jets_in_top] # use features for simple centroid selection
                centroid_idx = np.argmin(np.linalg.norm(sc_emb - sc_emb.mean(axis=0), axis=1))
                global_idx = global_phys_idx[assigned_jets_in_top][centroid_idx]
                single_event = load_events_by_indices(args.test_h5_path, [global_idx])[0]
                plot_representative_jet(single_event, title=f"Top Subclass {i} Centroid (Jet {global_idx})", 
                                        save_path=os.path.join(args.output_dir, f"top_subclass_{i}_centroid.png"))

    if len(active_qcd) > 0:
        for i, sc in enumerate(active_qcd):
            assigned_jets_in_qcd = (phys_preds == 0) & (phys_subclass == sc)
            if assigned_jets_in_qcd.sum() > 0:
                sc_emb = phys_features[assigned_jets_in_qcd]
                centroid_idx = np.argmin(np.linalg.norm(sc_emb - sc_emb.mean(axis=0), axis=1))
                global_idx = global_phys_idx[assigned_jets_in_qcd][centroid_idx]
                single_event = load_events_by_indices(args.test_h5_path, [global_idx])[0]
                plot_representative_jet(single_event, title=f"QCD Subclass {i} Centroid (Jet {global_idx})", 
                                        save_path=os.path.join(args.output_dir, f"qcd_subclass_{i}_centroid.png"))

    # Plot 6: Cosine Similarity Distributions
    print("Generating similarity distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, sc in enumerate(active_qcd):
        sc_mask = (sub_preds == 0) & (subclass_labels == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(sub_sims[sc_mask, 0], ax=axes[0], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[0].set_title("QCD Subclasses: Cosine Similarity to W_QCD Center", fontsize=12)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15)

    for i, sc in enumerate(active_top):
        sc_mask = (sub_preds == 1) & (subclass_labels == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(sub_sims[sc_mask, 1], ax=axes[1], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[1].set_title("Top Subclasses: Cosine Similarity to W_Top Center", fontsize=12)
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'subclass_similarity_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Plot 7: Physics Feature Correlation Matrices
    print("Generating subclass physics correlations heatmaps...")
    import pandas as pd
    n_top_sc = len(active_top)
    n_qcd_sc = len(active_qcd)
    if n_top_sc > 0 or n_qcd_sc > 0:
        fig, axes = plt.subplots(2, max(n_top_sc, n_qcd_sc), figsize=(4 * max(n_top_sc, n_qcd_sc), 8))
        feature_subnames = ["Mass", "Tau32", "sqrt(d12)"]
        
        for i, sc in enumerate(active_top):
            sc_mask = (phys_preds == 1) & (phys_subclass == sc)
            ax = axes[0, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[0]
            if sc_mask.sum() > 5:
                sub_df = pd.DataFrame(np.stack([mass_all[sc_mask], tau32_all[sc_mask], d12_all[sc_mask]], axis=1), columns=feature_subnames)
                sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
            ax.set_title(f"Top Subclass {i}", fontsize=11)
            
        for i, sc in enumerate(active_qcd):
            sc_mask = (phys_preds == 0) & (phys_subclass == sc)
            ax = axes[1, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[1]
            if sc_mask.sum() > 5:
                sub_df = pd.DataFrame(np.stack([mass_all[sc_mask], tau32_all[sc_mask], d12_all[sc_mask]], axis=1), columns=feature_subnames)
                sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
            ax.set_title(f"QCD Subclass {i}", fontsize=11)
            
        plt.suptitle("Physics Feature Correlation per Subclass", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'subclass_physics_correlations.png'), dpi=200, bbox_inches='tight')
        plt.close()

    # Plot 8: Hierarchical Lund Planes
    print("Generating Lund Plane plots...")
    lund_dir = os.path.join(args.output_dir, 'lund_planes')
    os.makedirs(lund_dir, exist_ok=True)
    
    # Class-level Lund planes
    plot_lund_plane(phys_events, label="Overall Test Sample (Non-split)", save_path=os.path.join(lund_dir, 'class_overall_non_split.png'))
    
    # Prediction Status Lund planes
    for cat, fname in [('True Positive', 'status_tp.png'), ('True Negative', 'status_tn.png'), 
                       ('False Positive', 'status_fp.png'), ('False Negative', 'status_fn.png')]:
        cat_mask = phys_status == cat
        if cat_mask.sum() > 5:
            plot_lund_plane(phys_events[cat_mask], label=f"{cat} Jets", save_path=os.path.join(lund_dir, fname))

    # Subclass level Lund planes with T/F splits
    for i, sc in enumerate(active_top):
        sc_mask_sub = (phys_preds == 1) & (phys_subclass == sc)
        if sc_mask_sub.sum() > 5:
            plot_lund_plane(phys_events[sc_mask_sub], label=f"Top Subclass {i}", save_path=os.path.join(lund_dir, f'top_subclass_{i}.png'))
            sc_tp = sc_mask_sub & (phys_labels == 1)
            sc_fp = sc_mask_sub & (phys_labels == 0)
            if sc_tp.sum() > 5:
                plot_lund_plane(phys_events[sc_tp], label=f"Top Subclass {i} (TP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_tp.png'))
            if sc_fp.sum() > 5:
                plot_lund_plane(phys_events[sc_fp], label=f"Top Subclass {i} (FP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_fp.png'))
                
    for i, sc in enumerate(active_qcd):
        sc_mask_sub = (phys_preds == 0) & (phys_subclass == sc)
        if sc_mask_sub.sum() > 5:
            plot_lund_plane(phys_events[sc_mask_sub], label=f"QCD Subclass {i}", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}.png'))
            sc_tn = sc_mask_sub & (phys_labels == 0)
            sc_fn = sc_mask_sub & (phys_labels == 1)
            if sc_tn.sum() > 5:
                plot_lund_plane(phys_events[sc_tn], label=f"QCD Subclass {i} (TN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_tn.png'))
            if sc_fn.sum() > 5:
                plot_lund_plane(phys_events[sc_fn], label=f"QCD Subclass {i} (FN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_fn.png'))

    # Feature distribution & correlations (class level)
    plot_physics_correlations(phys_labels, phys_scores, phys_features, feature_names,
                               save_path=os.path.join(args.output_dir, 'correlations.png'))
    plot_feature_distributions(phys_labels, phys_features, feature_names,
                                save_path=os.path.join(args.output_dir, 'feature_distributions.png'))
    
    # Class-level global Lund Planes
    qcd_mask_glob = phys_labels == 0
    top_mask_glob = phys_labels == 1
    if qcd_mask_glob.sum() > 5:
        plot_lund_plane(phys_events[qcd_mask_glob], label="QCD", save_path=os.path.join(args.output_dir, 'lund_plane_qcd.png'))
    if top_mask_glob.sum() > 5:
        plot_lund_plane(phys_events[top_mask_glob], label="Top", save_path=os.path.join(args.output_dir, 'lund_plane_top.png'))
        
    # Latent space scatter (compatibility)
    plot_latent_space(sub_emb[:5000], sub_labels[:5000], class_centers=class_centers,
                      save_path=os.path.join(args.output_dir, 'latent_space.png'))
                      
    print("\n" + "=" * 60)
    print(f"[COMPLETE] Local analysis finished. Results saved to '{args.output_dir}'")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
