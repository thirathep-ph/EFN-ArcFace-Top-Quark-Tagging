import os
import sys
import numpy as np
import torch
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.utils.physics import compute_features
from jet_tagging.data.loader import load_events_by_indices
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
    
    # 2. Stratified sampling of 3,000 jets for FastJet physics & clustering
    print("2. Sampling 3,000 jets (1,500 QCD, 1,500 Top)...")
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
    
    # 4. Cosine Silhouette Score
    print("4. Calculating Cosine Silhouette Scores...")
    sil_top = silhouette_score(top_latent, top_subclasses, metric='cosine')
    sil_qcd = silhouette_score(qcd_latent, qcd_subclasses, metric='cosine')
    
    # 5. Load physical variables
    print("5. Loading physical events & computing features...")
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    phys_features = compute_features(phys_events)
    # Observable features: Mass (0), Tau32 (7), sqrt(d12) (4)
    obs_top = phys_features[top_mask][:, [0, 7, 4]]
    obs_qcd = phys_features[qcd_mask][:, [0, 7, 4]]
    
    # Standardize physics features for K-means
    obs_top_norm = (obs_top - obs_top.mean(axis=0)) / (obs_top.std(axis=0) + 1e-10)
    obs_qcd_norm = (obs_qcd - obs_qcd.mean(axis=0)) / (obs_qcd.std(axis=0) + 1e-10)
    
    # 6. Fit K-means on physical variables
    print("6. Clustering physical variables with K-means...")
    kmeans_top = KMeans(n_clusters=len(np.unique(top_subclasses)), random_state=42, n_init=10).fit_predict(obs_top_norm)
    kmeans_qcd = KMeans(n_clusters=len(np.unique(qcd_subclasses)), random_state=42, n_init=10).fit_predict(obs_qcd_norm)
    
    # Calculate Adjusted Rand Index (ARI)
    ari_top = adjusted_rand_score(top_subclasses, kmeans_top)
    ari_qcd = adjusted_rand_score(qcd_subclasses, kmeans_qcd)
    
    print("\n" + "="*50)
    print("PART 2: CLUSTERING QUALITY & ALIGNMENT RESULTS")
    print("="*50)
    print(f"predicted Top Subclasses:")
    print(f"  Cosine Silhouette Score                   : {sil_top:.4f}")
    print(f"  ARI (AI Subclasses vs. Obs $K$-means)      : {ari_top:.4f}")
    print(f"predicted QCD Subclasses:")
    print(f"  Cosine Silhouette Score                   : {sil_qcd:.4f}")
    print(f"  ARI (AI Subclasses vs. Obs $K$-means)      : {ari_qcd:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
