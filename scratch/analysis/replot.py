import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.plotting import plot_latent_space
from arcjet.utils.device import get_device

def replot(model_path, data_dir="data/top_tagging", max_events=5000, output_path="results/test_run_15ep/latent_space_fixed.png"):
    print(f"\n=== Replottng Latent Space for {model_path} ===")
    if not os.path.exists(model_path):
        print("Checkpoint file does not exist")
        return
        
    device, _ = get_device()
    
    # Load model
    model = TopTaggingModel(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load dataset
    events, labels, weights = load_awkward(os.path.join(data_dir, "test.h5"), max_events=max_events)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    all_emb = []
    all_labels = []
    
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu())
            all_labels.append(y.cpu())
            
    all_emb = torch.cat(all_emb).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    prototypes = model.arcface_head.class_centers.detach().cpu().numpy()
    
    # Run plot_latent_space
    plot_latent_space(all_emb, all_labels, prototypes=prototypes, save_path=output_path)
    print("Done replotting!")

if __name__ == "__main__":
    replot("checkpoints/test_run_15ep/best_model_efn.pt")
