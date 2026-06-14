import os
import sys
import numpy as np
import torch
import awkward as ak

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.data.loader import load_events_by_indices
from scripts.analyze_offline import spectral_clustering_subclass
from arcjet.utils.physics import compute_features

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    test_h5_path = 'data/top_tagging/test.h5'
    output_path = 'main_arcface/centroid_constituents_report.md'
    artifact_dir = r'C:\Users\NBODT\.gemini\antigravity\brain\ad9c1e9b-859a-4f7b-b494-066dd7abf61d'
    
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
    
    # Stratified sample of 3,000 jets to run the Spectral Clustering
    print("2. Sampling 3,000 jets for subclass matching...")
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
    
    subclass_labels = np.zeros(len(sub_emb), dtype=int) - 1
    
    top_mask = (sub_preds == 1)
    top_latent = sub_emb_norm[top_mask]
    top_subclasses, k_top = spectral_clustering_subclass(top_latent, max_k=10)
    subclass_labels[top_mask] = top_subclasses
    
    qcd_mask = (sub_preds == 0)
    qcd_latent = sub_emb_norm[qcd_mask]
    qcd_subclasses, k_qcd = spectral_clustering_subclass(qcd_latent, max_k=10)
    subclass_labels[qcd_mask] = qcd_subclasses
    
    # 3. Find Representative Jet (Centroid) in physical features space for each subclass
    print("3. Finding representative centroids and loading constituent details...")
    
    # Compute features for physics matching
    phys_events = load_events_by_indices(test_h5_path, sampled_test_idx)
    phys_features = compute_features(phys_events)
    
    centroids_info = []
    
    def extract_subclass_centroid(class_name, class_mask, subclass_id, subclass_labels_class, active_subclasses):
        # Filter features of this subclass
        subclass_mask = class_mask & (subclass_labels == subclass_id)
        sc_features = phys_features[subclass_mask]
        sc_global_indices = sampled_test_idx[subclass_mask]
        
        # Centroid is the sample closest to the mean feature vector
        mean_feat = sc_features.mean(axis=0)
        centroid_local_idx = np.argmin(np.linalg.norm(sc_features - mean_feat, axis=1))
        centroid_global_idx = sc_global_indices[centroid_local_idx]
        
        # Load single event constituents
        event_p4 = load_events_by_indices(test_h5_path, [centroid_global_idx])[0]
        
        # Sort constituents by energy E
        px = np.array(event_p4.px)
        py = np.array(event_p4.py)
        pz = np.array(event_p4.pz)
        E = np.array(event_p4.E)
        
        pt = np.sqrt(px**2 + py**2)
        # Calculate eta and phi relative to jet axis (approximate using sum of px, py, pz)
        jet_px, jet_py, jet_pz, jet_E = px.sum(), py.sum(), pz.sum(), E.sum()
        jet_pt = np.sqrt(jet_px**2 + jet_py**2)
        jet_theta = np.arccos(jet_pz / np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2))
        jet_eta = -np.log(np.tan(jet_theta / 2.0))
        jet_phi = np.arctan2(jet_py, jet_px)
        
        # Particle etas and phis
        p_theta = np.arccos(pz / np.sqrt(px**2 + py**2 + pz**2 + 1e-10))
        p_eta = -np.log(np.tan(p_theta / 2.0) + 1e-10)
        p_phi = np.arctan2(py, px)
        
        d_eta = p_eta - jet_eta
        d_phi = p_phi - jet_phi
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi # Wrap phi
        
        z = E / jet_E
        
        # Sort by energy fraction descending
        sort_order = np.argsort(z)[::-1]
        
        constituents_list = []
        for rank, idx in enumerate(sort_order[:5]):
            constituents_list.append({
                "rank": rank + 1,
                "d_eta": d_eta[idx],
                "d_phi": d_phi[idx],
                "z": z[idx],
                "E_GeV": E[idx],
                "pt_GeV": pt[idx]
            })
            
        centroids_info.append({
            "class": class_name,
            "subclass": subclass_id,
            "global_idx": int(centroid_global_idx),
            "jet_mass": sc_features[centroid_local_idx, 0],
            "jet_tau32": sc_features[centroid_local_idx, 7],
            "jet_d12": sc_features[centroid_local_idx, 4],
            "constituents": constituents_list
        })
        
    for sc in np.unique(top_subclasses):
        extract_subclass_centroid("Top", top_mask, sc, top_subclasses, np.unique(top_subclasses))
    for sc in np.unique(qcd_subclasses):
        extract_subclass_centroid("QCD", qcd_mask, sc, qcd_subclasses, np.unique(qcd_subclasses))
        
    # Write Markdown Report
    print("4. Writing report to files...")
    report = """# Representative Centroid Jet Constituents Details

This report displays the exact constituent-level kinematics of the top 5 hardest particles for the centroid jets representing each discovered subclass.

---
"""
    for entry in centroids_info:
        report += f"""
## 🛩️ {entry['class']} Subclass {entry['subclass']} Centroid (Jet Index #{entry['global_idx']})
- **Physical Characteristics:** Jet Mass = **{entry['jet_mass']:.2f} GeV** | $\\tau_{{32}}$ = **{entry['jet_tau32']:.4f}** | $\\sqrt{{d_{{12}}}}$ = **{entry['jet_d12']:.2f} GeV**

### Top 5 Hardest Particle Constituents:
| Rank | Relative Angular $\\Delta\\eta_i$ | Relative Angular $\\Delta\\phi_i$ | Energy Fraction $z_i$ | Particle Energy $E_i$ (GeV) | Particle $p_{{T, i}}$ (GeV) |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for p in entry['constituents']:
            report += f"| {p['rank']} | {p['d_eta']:.4f} | {p['d_phi']:.4f} | {p['z']:.4f} | {p['E_GeV']:.2f} | {p['pt_GeV']:.2f} |\n"
            
    # Save locally (specify utf-8)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    # Save to artifacts directory (specify utf-8)
    with open(os.path.join(artifact_dir, 'centroid_constituents_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
        
    print("\n[COMPLETE] Centroid Report saved successfully to artifacts.")

if __name__ == '__main__':
    main()
