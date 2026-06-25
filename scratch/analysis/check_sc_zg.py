import torch
import os
import sys
import numpy as np
import awkward as ak
from torch.utils.data import DataLoader
from sklearn.mixture import BayesianGaussianMixture

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.physics import compute_features
from arcjet.utils.device import get_device

def main():
    device, _ = get_device()
    
    # Load model
    model = TopTaggingModel(embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(torch.load("checkpoints/test_run_15ep/best_model_efn.pt", map_location=device))
    model.eval()
    
    # Load 20,000 events
    events, labels, weights = load_awkward("data/top_tagging/test.h5", max_events=20000, lazy=True)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Extract embeddings and predictions
    all_emb = []
    all_sims = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            logits, sims, emb = model(x.to(device), mask=m.to(device))
            all_emb.append(emb.cpu().numpy())
            all_sims.append(sims.cpu().numpy())
            
    all_emb = np.concatenate(all_emb, axis=0)
    all_sims = np.concatenate(all_sims, axis=0)
    
    # Class scores
    qcd_scores = all_sims[:, 0]
    top_scores = all_sims[:, 1]
    preds = (top_scores > qcd_scores).astype(int)
    
    # Normalize embeddings
    all_emb_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10)
    
    # Fit DPGMM on Top predicted
    top_mask = (preds == 1)
    top_latent = all_emb_norm[top_mask]
    
    dpgmm_top = BayesianGaussianMixture(
        n_components=5, weight_concentration_prior=1e-3,
        random_state=42, n_init=3
    )
    top_subclasses = dpgmm_top.fit_predict(top_latent)
    active_top_sc, counts_top = np.unique(top_subclasses, return_counts=True)
    print(f"Top Subclasses: {active_top_sc} | Counts: {counts_top}")
    
    # Sliced events for Top predicted subclasses
    # We want to calculate the physical features of the events in each Top subclass
    # Top predicted events are at indices:
    top_global_indices = np.where(top_mask)[0]
    
    # Let's compute physics features for all Top predicted events
    print("Computing features for Top predicted events...")
    top_events = dataset.events[top_global_indices]
    features = compute_features(top_events)
    
    # Feature order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
    masses = features[:, 0]
    msds = features[:, 1]
    mults = features[:, 2]
    nsds = features[:, 3]
    d12s = features[:, 4]
    tau32s = features[:, 7]
    zgs = features[:, 8]
    
    print("\n--- Physical Properties of Top Subclasses ---")
    for i, sc in enumerate(active_top_sc):
        sc_mask = (top_subclasses == sc)
        sc_mass = masses[sc_mask]
        sc_zg = zgs[sc_mask]
        sc_tau32 = tau32s[sc_mask]
        sc_d12 = d12s[sc_mask]
        
        # Purity
        sc_labels = labels[top_global_indices][sc_mask]
        tp_frac = np.mean(sc_labels == 1) * 100
        
        print(f"Top Subclass {i} (N={sc_mask.sum()}):")
        print(f"  Purity (TP %): {tp_frac:.1f}%")
        print(f"  Mass:          mean={sc_mass.mean():.1f} GeV, std={sc_mass.std():.1f}")
        print(f"  zg:            mean={sc_zg.mean():.3f}, std={sc_zg.std():.3f}")
        print(f"  Tau32:         mean={sc_tau32.mean():.3f}, std={sc_tau32.std():.3f}")
        print(f"  sqrt(d12):     mean={sc_d12.mean():.1f} GeV, std={sc_d12.std():.1f}")
        print("-" * 40)

if __name__ == '__main__':
    main()
