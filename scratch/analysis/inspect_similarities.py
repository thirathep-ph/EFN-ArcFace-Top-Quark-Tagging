import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device

def inspect_similarities(model_path, data_dir="data/top_tagging", max_events=1000):
    print(f"\n=== Inspecting Similarities for {model_path} ===")
    if not os.path.exists(model_path):
        print("Checkpoint file does not exist")
        return
        
    device, _ = get_device()
    
    # Load model
    model = TopTaggingModel(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load dataset
    events, labels, weights = load_awkward(os.path.join(data_dir, "train.h5"), max_events=max_events)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # Check prototypes
    protos = model.arcface_head.class_centers.data
    norm_qcd = torch.norm(protos[0]).item()
    norm_top = torch.norm(protos[1]).item()
    dot_product = torch.dot(protos[0], protos[1]).item()
    print(f"Prototype QCD Norm: {norm_qcd:.6f}, Top Norm: {norm_top:.6f}")
    print(f"Prototype Dot Product: {dot_product:.6f}")
    
    for mode in ['eval', 'train']:
        if mode == 'train':
            model.train()
        else:
            model.eval()
            
        all_sims = []
        all_labels = []
        
        with torch.no_grad():
            for x, y, w, m, _ in loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits, sims, _ = model(x, mask=m)
                all_sims.append(sims.cpu())
                all_labels.append(y.cpu())
                
        all_sims = torch.cat(all_sims)
        all_labels = torch.cat(all_labels)
        
        qcd_mask = (all_labels == 0)
        top_mask = (all_labels == 1)
        
        # Mean similarity to correct prototype
        mean_sim_qcd = all_sims[qcd_mask][:, 0].mean().item()
        mean_sim_top = all_sims[top_mask][:, 1].mean().item()
        
        # Mean similarity to incorrect prototype
        mean_sim_qcd_to_top = all_sims[qcd_mask][:, 1].mean().item()
        mean_sim_top_to_qcd = all_sims[top_mask][:, 0].mean().item()
        
        print(f"\n[BN Mode: {mode.upper()}]")
        print(f"  QCD events to QCD prototype: {mean_sim_qcd:.4f}")
        print(f"  QCD events to Top prototype: {mean_sim_qcd_to_top:.4f}")
        print(f"  Top events to Top prototype: {mean_sim_top:.4f}")
        print(f"  Top events to QCD prototype: {mean_sim_top_to_qcd:.4f}")

if __name__ == "__main__":
    inspect_similarities("checkpoints/test_run_15ep/best_model_efn.pt")
