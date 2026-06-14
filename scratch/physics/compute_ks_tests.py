import os
import sys
import numpy as np
import torch
from scipy.stats import ks_2samp

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.utils.physics import compute_features
from arcjet.data.loader import load_events_by_indices
from scripts.analyze_offline import spectral_clustering_subclass

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    test_h5_path = 'data/top_tagging/test.h5'
    
    print("1. Loading embeddings & checkpoints...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'arcface_head.class_centers' in state_dict:
        class_centers = state_dict['arcface_head.class_centers'].cpu().numpy()
    elif 'prototype_layer.prototypes' in state_dict:
        class_centers = state_dict['prototype_layer.prototypes'].cpu().numpy()
        
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    
    t_sims = np.dot(embeddings_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    
    # 2. Stratified sampling of 3,000 jets
    print("2. Sampling 3,000 jets...")
    n_sample = 1500
    qcd_indices_all = np.where(labels == 0)[0]
    top_indices_all = np.where(labels == 1)[0]
    
    np.random.seed(42)
    sampled_qcd_idx = np.random.choice(qcd_indices_all, min(len(qcd_indices_all), n_sample), replace=False)
    sampled_top_idx = np.random.choice(top_indices_all, min(len(top_indices_all), n_sample), replace=False)
    
    sampled_test_idx = np.concatenate([sampled_qcd_idx, sampled_top_idx])
    np.random.shuffle(sampled_test_idx)
    
    sub_emb = embeddings[sampled_test_idx]
    sub_preds = preds[sampled_test_idx]
    sub_emb_norm = sub_emb / (np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-10)
    
    # 3. Spectral Clustering
    print("3. Running Spectral Clustering for Subclasses...")
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, _ = spectral_clustering_subclass(top_latent, max_k=10)
    subclass_labels[top_mask] = top_subclasses
    
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, _ = spectral_clustering_subclass(qcd_latent, max_k=10)
    subclass_labels[qcd_mask] = qcd_subclasses
    
    # 4. Load physical features
    print("4. Loading physical events & computing features...")
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    phys_features = compute_features(phys_events)
    # Observable features: Mass (0), Tau32 (7), d12 (4)
    obs_names = ["Jet Mass", "N-subjettiness Tau32", "Splitting Scale sqrt(d12)"]
    obs_indices = [0, 7, 4]
    
    # 5. Run KS tests
    print("5. Performing Kolmogorov-Smirnov (KS) tests between subclasses...")
    
    top_ks = {}
    qcd_ks = {}
    
    for name, idx in zip(obs_names, obs_indices):
        # Top subclass 0 vs 1
        c0_top = phys_features[top_mask & (subclass_labels == 0), idx]
        c1_top = phys_features[top_mask & (subclass_labels == 1), idx]
        stat, p_val = ks_2samp(c0_top, c1_top)
        top_ks[name] = (stat, p_val)
        
        # QCD subclass 0 vs 1
        c0_qcd = phys_features[qcd_mask & (subclass_labels == 0), idx]
        c1_qcd = phys_features[qcd_mask & (subclass_labels == 1), idx]
        stat, p_val = ks_2samp(c0_qcd, c1_qcd)
        qcd_ks[name] = (stat, p_val)
        
    print("\n" + "="*55)
    print("PART 3: KOLMOGOROV-SMIRNOV (KS) PHYSICS DISTINCTION RESULTS")
    print("="*55)
    print("predicted Top Subclass 0 vs. Subclass 1:")
    for name in obs_names:
        stat, p_val = top_ks[name]
        distinct = "Yes (p < 0.05)" if p_val < 0.05 else "No"
        print(f"  {name:28s}: KS Stat = {stat:.4f} | p-value = {p_val:.4e} | Distinct = {distinct}")
        
    print("\npredicted QCD Subclass 0 vs. Subclass 1:")
    for name in obs_names:
        stat, p_val = qcd_ks[name]
        distinct = "Yes (p < 0.05)" if p_val < 0.05 else "No"
        print(f"  {name:28s}: KS Stat = {stat:.4f} | p-value = {p_val:.4e} | Distinct = {distinct}")
    print("="*55 + "\n")

if __name__ == '__main__':
    main()
