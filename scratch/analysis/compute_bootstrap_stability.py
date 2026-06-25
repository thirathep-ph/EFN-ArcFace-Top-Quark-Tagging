import os
import sys
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analyze_offline import spectral_clustering_subclass

def run_stability_bootstrap(latent_data, n_iterations=20, sub_fraction=0.9):
    n_samples = len(latent_data)
    # 1. Run baseline clustering on all samples
    print(f"  Running reference clustering on all {n_samples} samples...")
    ref_labels, k = spectral_clustering_subclass(latent_data, max_k=5)
    
    aris = []
    
    # 2. Bootstrap iterations
    for i in range(n_iterations):
        # Select random 90% subsample
        np.random.seed(i)
        sub_size = int(n_samples * sub_fraction)
        indices = np.random.choice(n_samples, sub_size, replace=False)
        
        # Run clustering on subsample
        sub_latent = latent_data[indices]
        # Set max_k same as baseline k to keep consistency
        sub_labels, _ = spectral_clustering_subclass(sub_latent, max_k=k)
        
        # Compare overlapping labels using ARI
        ref_sub_labels = ref_labels[indices]
        ari = adjusted_rand_score(ref_sub_labels, sub_labels)
        aris.append(ari)
        
    return np.mean(aris), np.std(aris), k

def main():
    embeddings_path = 'experiments/arcface_100ep/embeddings.npz'
    checkpoint_path = 'experiments/arcface_100ep/best_model_efn.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/best_model_efn.pt'
    
    print("1. Loading embeddings & checkpoints...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'arcface_head.class_centers' in state_dict:
        class_centers = state_dict['arcface_head.class_centers'].cpu().numpy()
    elif 'prototype_layer.prototypes' in state_dict:
        class_centers = state_dict['prototype_layer.prototypes'].cpu().numpy()
        
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    
    t_sims = np.dot(embeddings_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    
    # Sample 2,000 predicted QCD and 2,000 predicted Top
    print("2. Sampling 2,000 jets per class for Bootstrap...")
    np.random.seed(42)
    qcd_indices_all = np.where(preds == 0)[0]
    top_indices_all = np.where(preds == 1)[0]
    
    sampled_qcd_idx = np.random.choice(qcd_indices_all, 2000, replace=False)
    sampled_top_idx = np.random.choice(top_indices_all, 2000, replace=False)
    
    qcd_latent = embeddings_norm[sampled_qcd_idx]
    top_latent = embeddings_norm[sampled_top_idx]
    
    # Run Bootstrap for predicted Top
    print("\n--- Running Bootstrap on predicted Top ---")
    mean_ari_top, std_ari_top, k_top = run_stability_bootstrap(top_latent, n_iterations=20)
    
    # Run Bootstrap for predicted QCD
    print("\n--- Running Bootstrap on predicted QCD ---")
    mean_ari_qcd, std_ari_qcd, k_qcd = run_stability_bootstrap(qcd_latent, n_iterations=20)
    
    print("\n" + "="*50)
    print("PART 4: CLUSTER STABILITY ANALYSIS (BOOTSTRAP)")
    print("="*50)
    print(f"predicted Top Subclasses (k={k_top}):")
    print(f"  Mean Bootstrap ARI                        : {mean_ari_top:.4f}")
    print(f"  Bootstrap ARI Std Dev                     : {std_ari_top:.4f}")
    print(f"predicted QCD Subclasses (k={k_qcd}):")
    print(f"  Mean Bootstrap ARI                        : {mean_ari_qcd:.4f}")
    print(f"  Bootstrap ARI Std Dev                     : {std_ari_qcd:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
