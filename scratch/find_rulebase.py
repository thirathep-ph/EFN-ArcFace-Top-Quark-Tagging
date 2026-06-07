import os
import sys
import numpy as np
import torch

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.utils.physics import compute_features
from jet_tagging.data.loader import load_events_by_indices

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    test_h5_path = 'data/top_tagging/test.h5'
    
    print("1. Loading test labels & indices...")
    data = np.load(embeddings_path)
    labels = data['labels']
    
    # Stratified sampling of 3,000 jets (to match our physical features sample)
    n_sample = 1500
    qcd_indices_all = np.where(labels == 0)[0]
    top_indices_all = np.where(labels == 1)[0]
    
    np.random.seed(42)
    sampled_qcd_idx = np.random.choice(qcd_indices_all, min(len(qcd_indices_all), n_sample), replace=False)
    sampled_top_idx = np.random.choice(top_indices_all, min(len(top_indices_all), n_sample), replace=False)
    
    sampled_test_idx = np.concatenate([sampled_qcd_idx, sampled_top_idx])
    np.random.shuffle(sampled_test_idx)
    
    sub_labels = labels[sampled_test_idx]
    
    # Load physical features: Mass (0), Tau32 (7), zg (8)
    print("2. Loading physical events & computing features...")
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    phys_features = compute_features(phys_events)
    
    mass = phys_features[:, 0]
    tau32 = phys_features[:, 7]
    zg = phys_features[:, 8]
    
    # 3. Apply Theorist's Rule-Base for Class Separation:
    # Rule: Class is Top if Mass is in the Top window [140, 200] GeV AND jet is 3-prong (Tau32 < 0.55)
    print("3. Evaluating Theorist's Rule-Base...")
    rule_top_preds = (mass >= 140) & (mass <= 200) & (tau32 < 0.55)
    rule_preds = np.where(rule_top_preds, 1, 0)
    
    # Calculate accuracy
    correct = np.sum(rule_preds == sub_labels)
    rule_accuracy = correct / len(sub_labels)
    
    # Let's optimize the cuts via grid search to find the absolute best physical rule-base
    best_acc = 0.0
    best_cuts = None
    
    for mass_min in [130, 140, 150]:
        for mass_max in [190, 200, 210]:
            for t32_cut in [0.50, 0.55, 0.60, 0.65]:
                opt_preds = (mass >= mass_min) & (mass <= mass_max) & (tau32 < t32_cut)
                acc = np.sum(opt_preds == sub_labels) / len(sub_labels)
                if acc > best_acc:
                    best_acc = acc
                    best_cuts = (mass_min, mass_max, t32_cut)
                    
    # 4. Altarelli-Parisi (AP) Splitting function analysis
    # Let's count how many jets in each subclass follow the QCD 1/zg divergence vs Top flat decay
    # We define AP-like as zg < 0.2 (peaking near threshold) vs isotropic as zg > 0.2 (flat)
    print("4. Analyzing Altarelli-Parisi splitting distributions...")
    
    # For Top (using the AI predicted Top subclasses)
    # We need subclass labels to print the AP metrics
    from scripts.analyze_offline import spectral_clustering_subclass
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'arcface_head.class_centers' in state_dict:
        class_centers = state_dict['arcface_head.class_centers'].cpu().numpy()
    elif 'prototype_layer.prototypes' in state_dict:
        class_centers = state_dict['prototype_layer.prototypes'].cpu().numpy()
        
    embeddings_norm = data['embeddings'] / (np.linalg.norm(data['embeddings'], axis=1, keepdims=True) + 1e-10)
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    t_sims = np.dot(embeddings_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    
    sub_preds = preds[sampled_test_idx]
    sub_emb = data['embeddings'][sampled_test_idx]
    sub_emb_norm = sub_emb / (np.linalg.norm(sub_emb, axis=1, keepdims=True) + 1e-10)
    
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, _ = spectral_clustering_subclass(top_latent, max_k=10)
    subclass_labels[top_mask] = top_subclasses
    
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, _ = spectral_clustering_subclass(qcd_latent, max_k=10)
    subclass_labels[qcd_mask] = qcd_subclasses
    
    print("\n" + "="*60)
    print("THEORIST RULE-BASE & ALTARELLI-PARISI (AP) SPLITTING ANALYSIS")
    print("="*60)
    print(f"Standard Theorist Rule-Base [140 < Mass < 200 GeV AND Tau32 < 0.55]:")
    print(f"  Rule-Base Accuracy                        : {rule_accuracy*100:.2f}%")
    print(f"  AI Model Accuracy (Reference)             : 91.54%")
    print(f"\nOptimized Theorist Rule-Base [ {best_cuts[0]} < Mass < {best_cuts[1]} GeV AND Tau32 < {best_cuts[2]} ]:")
    print(f"  Optimized Rule-Base Accuracy              : {best_acc*100:.2f}%")
    
    print(f"\nAP Splitting Function ($z_g$) Fraction by Subclass:")
    print(f"  (Isotropic flat decays have higher zg; collinear splittings peak near zg=0.1)")
    
    for i in np.unique(top_subclasses):
        sc_mask = top_mask & (subclass_labels == i)
        zg_vals = zg[sc_mask]
        pct_isotropic = np.sum(zg_vals > 0.2) / len(zg_vals) * 100
        print(f"  Top Subclass {i} (Mass={mass[sc_mask].mean():.1f} GeV): {pct_isotropic:.1f}% jets with zg > 0.2 (Isotropic/Flat)")
        
    for i in np.unique(qcd_subclasses):
        sc_mask = qcd_mask & (subclass_labels == i)
        zg_vals = zg[sc_mask]
        pct_isotropic = np.sum(zg_vals > 0.2) / len(zg_vals) * 100
        print(f"  QCD Subclass {i} (Mass={mass[sc_mask].mean():.1f} GeV): {pct_isotropic:.1f}% jets with zg > 0.2 (Collinear/Peaked)")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
