import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.models.efn_arcface import TopTaggingModel

def inspect(model_path):
    print(f"\n=== Inspecting {model_path} ===")
    if not os.path.exists(model_path):
        print("File does not exist")
        return
    
    state_dict = torch.load(model_path, map_location='cpu')
    n_protos = state_dict['prototype_layer.prototypes'].shape[0]
    n_proto_per_class = n_protos // 2
    
    # Load model
    model = TopTaggingModel(n_proto_per_class=n_proto_per_class, embedding_dim=64)
    model.load_state_dict(state_dict)
    
    # Get prototypes
    protos = model.prototype_layer.prototypes.data # Shape: (2, 64)
    print("Prototype shape:", protos.shape)
    
    # Calculate norms
    norm_qcd = torch.norm(protos[0]).item()
    norm_top = torch.norm(protos[1]).item()
    print(f"Norm of QCD Prototype: {norm_qcd:.6f}")
    print(f"Norm of Top Prototype: {norm_top:.6f}")
    
    # Dot product
    dot_product = torch.dot(protos[0], protos[1]).item()
    print(f"Dot product (QCD, Top): {dot_product:.6f}")
    
    # Cosine similarity
    cos_sim = dot_product / (norm_qcd * norm_top + 1e-10)
    print(f"Cosine Similarity (QCD, Top): {cos_sim:.6f}")
    
    # Check if they are unit vectors in the model state
    # Wait, does the model normalize them in forward? Yes: F.normalize(self.prototypes, dim=1)
    # But does the optimizer update the unnormalized parameters? Yes.
    # Let's see the raw values
    print("Raw QCD Prototype values (first 10):", protos[0][:10].tolist())
    print("Raw Top Prototype values (first 10):", protos[1][:10].tolist())

if __name__ == "__main__":
    inspect("checkpoints/baseline/best_model_efn.pt")
    inspect("checkpoints/decorrelated/best_model_efn.pt")
    inspect("checkpoints/best_model_efn.pt")
