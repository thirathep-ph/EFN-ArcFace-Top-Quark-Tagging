import os
import sys
import numpy as np
import torch
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
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
    
    # Sample 3,000 jets
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
    
    # 3. Spectral Clustering (SC) Subclasses
    print("3. Running Spectral Clustering (Manifold-aware)...")
    sc_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_sc, _ = spectral_clustering_subclass(top_latent, max_k=10)
    sc_labels[top_mask] = top_sc
    
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_sc, _ = spectral_clustering_subclass(qcd_latent, max_k=10)
    sc_labels[qcd_mask] = qcd_sc
    
    # 4. Spherical K-means (vMF Proxy) Subclasses
    print("4. Running Spherical K-means (Spherical constraint)...")
    skm_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    kmeans_top = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(top_latent)
    skm_labels[top_mask] = kmeans_top
    
    kmeans_qcd = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(qcd_latent)
    skm_labels[qcd_mask] = kmeans_qcd
    
    # 5. Compare overlap using ARI
    ari_top = adjusted_rand_score(top_sc, kmeans_top)
    ari_qcd = adjusted_rand_score(qcd_sc, kmeans_qcd)
    
    # 6. Compare Cosine Silhouette Scores
    sil_sc_top = silhouette_score(top_latent, top_sc, metric='cosine')
    sil_skm_top = silhouette_score(top_latent, kmeans_top, metric='cosine')
    
    sil_sc_qcd = silhouette_score(qcd_latent, qcd_sc, metric='cosine')
    sil_skm_qcd = silhouette_score(qcd_latent, kmeans_qcd, metric='cosine')
    
    # 7. Compute physical features and run KS tests to see physical distinctness
    print("7. Loading physical features & computing KS statistics...")
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    phys_features = compute_features(phys_events)
    # Mass index = 0, Tau32 index = 7
    mass_top = phys_features[top_mask][:, 0]
    tau32_top = phys_features[top_mask][:, 7]
    
    # KS for Spectral Clustering Top
    stat_mass_sc, p_mass_sc = ks_2samp(mass_top[top_sc == 0], mass_top[top_sc == 1])
    stat_tau_sc, p_tau_sc = ks_2samp(tau32_top[top_sc == 0], tau32_top[top_sc == 1])
    
    # KS for Spherical K-means Top
    stat_mass_skm, p_mass_skm = ks_2samp(mass_top[kmeans_top == 0], mass_top[kmeans_top == 1])
    stat_tau_skm, p_tau_skm = ks_2samp(tau32_top[kmeans_top == 0], tau32_top[kmeans_top == 1])
    
    print("\n" + "="*60)
    print("COMPARISON: SPECTRAL CLUSTERING VS. SPHERICAL K-MEANS")
    print("="*60)
    print(f"Overlap between algorithms (Adjusted Rand Index):")
    print(f"  Top Class ARI                             : {ari_top:.4f}")
    print(f"  QCD Class ARI                             : {ari_qcd:.4f}")
    print("\nCosine Silhouette Score comparison (Higher is better):")
    print(f"  Top Class - Spectral Clustering           : {sil_sc_top:.4f}")
    print(f"  Top Class - Spherical K-means             : {sil_skm_top:.4f}")
    print(f"  QCD Class - Spectral Clustering           : {sil_sc_qcd:.4f}")
    print(f"  QCD Class - Spherical K-means             : {sil_skm_qcd:.4f}")
    print("\nPhysical Observables Distinction (predicted Top):")
    print(f"  Spectral Clustering (SC) Subclasses:")
    print(f"    Jet Mass: KS Stat = {stat_mass_sc:.4f} | p-value = {p_mass_sc:.4e}")
    print(f"    Tau32   : KS Stat = {stat_tau_sc:.4f} | p-value = {p_tau_sc:.4e}")
    print(f"  Spherical K-means (vMF) Subclasses:")
    print(f"    Jet Mass: KS Stat = {stat_mass_skm:.4f} | p-value = {p_mass_skm:.4e}")
    print(f"    Tau32   : KS Stat = {stat_tau_skm:.4f} | p-value = {p_tau_skm:.4e}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
