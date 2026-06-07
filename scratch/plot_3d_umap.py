import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_offline import spectral_clustering_subclass

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    artifact_dir = r'C:\Users\NBODT\.gemini\antigravity\brain\ad9c1e9b-859a-4f7b-b494-066dd7abf61d'
    
    print("1. Loading embeddings & model centers...")
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
    
    # 2. Cosine similarities & predictions
    t_sims = np.dot(embeddings_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    
    # 3. Stratified sampling (3,000 jets for fast execution)
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
    sub_labels = labels[sampled_test_idx]
    sub_preds = preds[sampled_test_idx]
    sub_emb_norm = sub_emb / (np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-10)
    
    # Run spectral clustering to find subclasses for 3D plot
    print("3. Performing Spectral Clustering...")
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, k_top = spectral_clustering_subclass(top_latent, max_k=10)
    subclass_labels[top_mask] = top_subclasses
    
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, k_qcd = spectral_clustering_subclass(qcd_latent, max_k=10)
    subclass_labels[qcd_mask] = qcd_subclasses
    
    tp_mask = (sub_preds == 1) & (sub_labels == 1)
    tn_mask = (sub_preds == 0) & (sub_labels == 0)
    fp_mask = (sub_preds == 1) & (sub_labels == 0)
    fn_mask = (sub_preds == 0) & (sub_labels == 1)
    
    status = np.zeros(len(sub_emb), dtype=object)
    status[tp_mask] = 'True Positive'
    status[tn_mask] = 'True Negative'
    status[fp_mask] = 'False Positive'
    status[fn_mask] = 'False Negative'
    
    # 4. Fit 3D UMAP
    print("4. Fitting 3D UMAP in parallel on CPU...")
    reducer = umap.UMAP(n_components=3, metric='cosine', random_state=None, n_jobs=-1)
    umap_coords_3d = reducer.fit_transform(sub_emb_norm)
    centers_umap_3d = reducer.transform(centers_norm)
    print("   UMAP projection finished.")
    
    # 5. Plotting 3D UMAP views
    print("5. Generating 3D plots...")
    colors = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4', 'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
    markers = {'True Negative': 'o', 'True Positive': 'o', 'False Positive': 'x', 'False Negative': 'x'}
    sizes = {'True Negative': 12, 'True Positive': 12, 'False Positive': 35, 'False Negative': 35}
    alphas = {'True Negative': 0.4, 'True Positive': 0.4, 'False Positive': 0.8, 'False Negative': 0.8}
    
    view_angles = [(20, 45), (20, 135), (20, 225)]
    
    for idx, (elev, azim) in enumerate(view_angles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points
        for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
            mask_cat = (status == cat)
            if mask_cat.sum() > 0:
                ax.scatter(umap_coords_3d[mask_cat, 0], 
                           umap_coords_3d[mask_cat, 1], 
                           umap_coords_3d[mask_cat, 2],
                           color=colors[cat], marker=markers[cat], s=sizes[cat],
                           alpha=alphas[cat], label=f"{cat} ({mask_cat.sum()})")
                
        # Plot class centers
        ax.scatter(centers_umap_3d[0, 0], centers_umap_3d[0, 1], centers_umap_3d[0, 2],
                   marker='*', s=400, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
        ax.scatter(centers_umap_3d[1, 0], centers_umap_3d[1, 1], centers_umap_3d[1, 2],
                   marker='*', s=400, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
        
        ax.set_title(f"3D UMAP Space - View {idx+1} (elev={elev}, azim={azim})", fontsize=14)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc='upper left', bbox_to_anchor=(0.95, 0.9))
        
        output_name = f'umap_3d_angle{idx+1}.png'
        plt.savefig(os.path.join(artifact_dir, output_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved {output_name} to artifacts.")
        
    # Plot subclasses in 3D (Angle 1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    qcd_colors = ['#a1d99b', '#41ab5d', '#005a32']
    for i, sc in enumerate(np.unique(qcd_subclasses)):
        sc_mask = (sub_preds == 0) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            ax.scatter(umap_coords_3d[sc_mask, 0], umap_coords_3d[sc_mask, 1], umap_coords_3d[sc_mask, 2],
                       color=qcd_colors[i % len(qcd_colors)], s=12, alpha=0.5, label=f"QCD Subclass {i} (N={sc_mask.sum()})")
            
    top_colors = ['#9ecae1', '#4292c6', '#084594']
    for i, sc in enumerate(np.unique(top_subclasses)):
        sc_mask = (sub_preds == 1) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            ax.scatter(umap_coords_3d[sc_mask, 0], umap_coords_3d[sc_mask, 1], umap_coords_3d[sc_mask, 2],
                       color=top_colors[i % len(top_colors)], s=12, alpha=0.5, label=f"Top Subclass {i} (N={sc_mask.sum()})")
            
    # Class centers
    ax.scatter(centers_umap_3d[0, 0], centers_umap_3d[0, 1], centers_umap_3d[0, 2],
               marker='*', s=400, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
    ax.scatter(centers_umap_3d[1, 0], centers_umap_3d[1, 1], centers_umap_3d[1, 2],
               marker='*', s=400, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
    
    ax.set_title("3D UMAP - Discovered Subclasses", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', bbox_to_anchor=(0.95, 0.9))
    
    output_name = 'umap_3d_subclasses.png'
    plt.savefig(os.path.join(artifact_dir, output_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved {output_name} to artifacts.")
    
    print("\n[COMPLETE] 3D UMAP plots generated successfully.")

if __name__ == '__main__':
    main()
