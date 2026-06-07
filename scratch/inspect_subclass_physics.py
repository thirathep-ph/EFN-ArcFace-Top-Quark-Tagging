import os
import sys
import numpy as np
import awkward as ak
import torch

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.utils.physics import compute_features, print_feature_stats
from jet_tagging.data.loader import load_events_by_indices
from scripts.analyze_offline import spectral_clustering_subclass

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    test_h5_path = 'data/top_tagging/test.h5'
    
    print("Loading embeddings...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    # 2. Estimate similarities to identify predictions
    print("Loading model checkpoints to get centers...")
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'arcface_head.class_centers' in state_dict:
        class_centers = state_dict['arcface_head.class_centers'].cpu().numpy()
    elif 'prototype_layer.prototypes' in state_dict:
        class_centers = state_dict['prototype_layer.prototypes'].cpu().numpy()
    
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    
    t_sims = np.dot(embeddings_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    
    # Sample 5,000 QCD and 5,000 Top
    print("Sampling jets for analysis...")
    n_sample = 5000
    qcd_indices_all = np.where(labels == 0)[0]
    top_indices_all = np.where(labels == 1)[0]
    
    np.random.seed(42)
    sampled_qcd_idx = np.random.choice(qcd_indices_all, min(len(qcd_indices_all), n_sample), replace=False)
    sampled_top_idx = np.random.choice(top_indices_all, min(len(top_indices_all), n_sample), replace=False)
    
    sampled_test_idx = np.concatenate([sampled_qcd_idx, sampled_top_idx])
    np.random.shuffle(sampled_test_idx)
    
    sub_emb = embeddings[sampled_test_idx]
    sub_labels = labels[sampled_test_idx]
    sub_preds = preds[sampled_test_idx]
    sub_emb_norm = sub_emb / (np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-10)
    
    # Run spectral clustering
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    print("Clustering predicted Top...")
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, k_top = spectral_clustering_subclass(top_latent, max_k=10)
    subclass_labels[top_mask] = top_subclasses
    
    print("Clustering predicted QCD...")
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, k_qcd = spectral_clustering_subclass(qcd_latent, max_k=10)
    subclass_labels[qcd_mask] = qcd_subclasses
    
    # Load physical events
    print("Loading event data...")
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    print("Computing features...")
    phys_features = compute_features(phys_events)
    
    feature_names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    
    # Print stats for each subclass
    print("\n" + "="*50)
    print("SUBCLASS ANALYSIS RESULTS")
    print("="*50)
    
    print("\n--- predicted QCD Subclasses ---")
    active_qcd = np.unique(qcd_subclasses)
    for i, sc in enumerate(active_qcd):
        sc_mask = (sub_preds == 0) & (subclass_labels == sc)
        print(f"\nQCD Subclass {i} (N = {sc_mask.sum()}):")
        # Print means of features
        for f_idx, f_name in enumerate(feature_names):
            vals = phys_features[sc_mask, f_idx]
            print(f"  {f_name:10s}: mean = {np.mean(vals):.4f}, std = {np.std(vals):.4f}")
            
    print("\n--- predicted Top Subclasses ---")
    active_top = np.unique(top_subclasses)
    for i, sc in enumerate(active_top):
        sc_mask = (sub_preds == 1) & (subclass_labels == sc)
        print(f"\nTop Subclass {i} (N = {sc_mask.sum()}):")
        for f_idx, f_name in enumerate(feature_names):
            vals = phys_features[sc_mask, f_idx]
            print(f"  {f_name:10s}: mean = {np.mean(vals):.4f}, std = {np.std(vals):.4f}")

if __name__ == '__main__':
    main()
