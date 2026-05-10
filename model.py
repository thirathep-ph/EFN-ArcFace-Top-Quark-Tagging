"""
DeepSets + Prototype Layer Model for Top Tagging
=================================================
Architecture:
    Constituents (E, px, py, pz) -> DeepSets -> jet_embedding
    Constituents -> FastJet -> 6 features
    Concat [jet_embedding, fastjet_features] -> Prototype Layer -> Classification

Prototype Layer uses Energy Distance (EDM) similarity metric.
20 prototypes: 10 for Top class, 10 for QCD class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# DeepSets
# ============================================================================

class ParticleEncoder(nn.Module):
    """
    Per-particle MLP: (E, px, py, pz) -> particle_embedding
    """
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (batch_size, n_constituents, 4) - (E, px, py, pz)
        Returns: (batch_size, n_constituents, output_dim)
        """
        b, n, d = x.shape
        # Flatten particles for batchnorm
        x_flat = x.reshape(b * n, d)
        out_flat = self.net(x_flat)
        return out_flat.reshape(b, n, -1)


class DeepSets(nn.Module):
    """
    DeepSets: permutation-invariant set processing.
    
    phi: per-particle MLP
    rho: set-level MLP (after sum pooling)
    """
    def __init__(self, input_dim=4, particle_dim=128, jet_dim=128):
        super().__init__()
        self.phi = ParticleEncoder(input_dim, hidden_dim=particle_dim, output_dim=particle_dim)
        self.rho = nn.Sequential(
            nn.Linear(particle_dim, jet_dim),
            nn.BatchNorm1d(jet_dim),
            nn.ReLU(),
            nn.Linear(jet_dim, jet_dim),
        )

    def forward(self, x, mask=None):
        """
        x: (batch_size, n_constituents, 4) - (E, px, py, pz)
        mask: (batch_size, n_constituents) - True for valid particles, False for padding
        
        Returns: (batch_size, jet_dim) - jet embedding
        """
        # Per-particle encoding
        particle_emb = self.phi(x)  # (B, N, particle_dim)
        
        # Sum pooling (permutation invariant)
        if mask is not None:
            # Zero out padded particles
            particle_emb = particle_emb * mask.unsqueeze(-1).float()
            jet_emb = particle_emb.sum(dim=1)  # (B, particle_dim)
            # Normalize by number of valid particles
            n_valid = mask.sum(dim=1, keepdim=True).float()
            n_valid = n_valid.clamp(min=1.0)
            jet_emb = jet_emb / n_valid
        else:
            jet_emb = particle_emb.mean(dim=1)  # (B, particle_dim)
        
        # Jet-level encoding
        jet_emb = self.rho(jet_emb)  # (B, jet_dim)
        return jet_emb


# ============================================================================
# Prototype Layer (xAI)
# ============================================================================

class PrototypeLayer(nn.Module):
    """
    Prototype layer with Energy Distance (EDM) similarity.
    
    EDM(z, p) = sqrt( E[z] + E[p] - 2 * E[sqrt(z * p)] )
    
    For deterministic embeddings (not distributions), this simplifies to:
    EDM(z, p) = sqrt( z^2 + p^2 - 2 * sqrt(z * p) )
    
    Actually, for deterministic vectors, the standard EDM is:
    EDM(z, p) = ||z - p||_2  (Euclidean distance)
    
    We use a similarity transformation:
    sim(z, p) = exp(-EDM(z, p) / temperature)
    """
    def __init__(self, input_dim, n_prototypes=20, n_classes=2, temperature=1.0):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.temperature = temperature
        
        # Prototype vectors: (n_prototypes, input_dim)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim) * 0.1)
        
        # Class mapping: which prototypes belong to which class
        # First n_prototypes//2 for class 0 (QCD), rest for class 1 (Top)
        self.register_buffer(
            'prototype_class',
            torch.cat([
                torch.zeros(n_prototypes // 2, dtype=torch.long),
                torch.ones(n_prototypes - n_prototypes // 2, dtype=torch.long),
            ])
        )
        
        # Classification head: prototype similarities -> class scores
        self.classifier = nn.Linear(n_prototypes, n_classes)

    def forward(self, x):
        """
        x: (batch_size, input_dim) - concatenated [jet_embedding, fastjet_features]
        
        Returns:
            similarities: (batch_size, n_prototypes) - EDM-based similarity scores
            logits: (batch_size, n_classes) - class scores
        """
        # Compute EDM (Euclidean distance) between each input and each prototype
        # x: (B, D), prototypes: (P, D)
        # dist: (B, P)
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B, P, D)
        dist = torch.norm(diff, dim=2)  # (B, P)
        
        # Convert distance to similarity: exp(-dist / temperature)
        similarities = torch.exp(-dist / self.temperature)  # (B, P)
        
        # Classification from prototype similarities
        logits = self.classifier(similarities)  # (B, n_classes)
        
        return similarities, logits

    def get_prototype_labels(self):
        """Return class label for each prototype."""
        return self.prototype_class

    def get_prototype_activations(self, x):
        """
        Get per-prototype similarity for interpretability.
        Returns: (batch_size, n_prototypes)
        """
        with torch.no_grad():
            sim, _ = self.forward(x)
        return sim


# ============================================================================
# Full Model
# ============================================================================

class TopTaggingModel(nn.Module):
    """
    Complete model: DeepSets + FastJet features + Prototype Layer.
    
    Input: (batch_size, 200, 4) constituent 4-momenta (E, px, py, pz)
    + (batch_size, 6) FastJet features [mass, pT, eta, tau21, tau32, D2]
    
    Output: (batch_size, 2) class scores [P(QCD), P(Top)]
    """
    def __init__(self, 
                 constituent_dim=4,
                 n_constituents=200,
                 particle_dim=128,
                 jet_dim=128,
                 n_fastjet_features=6,
                 n_prototypes=20,
                 n_classes=2,
                 temperature=1.0):
        super().__init__()
        
        self.n_constituents = n_constituents
        self.jet_dim = jet_dim
        self.n_fastjet_features = n_fastjet_features
        
        # DeepSets
        self.deepsets = DeepSets(
            input_dim=constituent_dim,
            particle_dim=particle_dim,
            jet_dim=jet_dim,
        )
        
        # Combine jet embedding + FastJet features
        combined_dim = jet_dim + n_fastjet_features
        
        # Project combined features
        self.combined_proj = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.BatchNorm1d(combined_dim),
            nn.ReLU(),
        )
        
        # Prototype Layer
        self.prototype_layer = PrototypeLayer(
            input_dim=combined_dim,
            n_prototypes=n_prototypes,
            n_classes=n_classes,
            temperature=temperature,
        )

    def forward(self, constituents, fastjet_features, mask=None):
        """
        constituents: (batch_size, 200, 4) - (E, px, py, pz)
        fastjet_features: (batch_size, 6) - [mass, pT, eta, tau21, tau32, D2]
        mask: (batch_size, 200) - True for valid particles
        
        Returns:
            logits: (batch_size, 2) - [P(QCD), P(Top)]
            similarities: (batch_size, 20) - prototype similarity scores
            jet_emb: (batch_size, jet_dim) - jet embedding from DeepSets
        """
        # DeepSets jet embedding
        jet_emb = self.deepsets(constituents, mask)  # (B, jet_dim)
        
        # Concatenate with FastJet features
        combined = torch.cat([jet_emb, fastjet_features], dim=1)  # (B, jet_dim + 6)
        combined = self.combined_proj(combined)
        
        # Prototype layer
        similarities, logits = self.prototype_layer(combined)
        
        return logits, similarities, jet_emb

    def predict(self, constituents, fastjet_features, mask=None):
        """Get class probabilities."""
        logits, similarities, jet_emb = self.forward(constituents, fastjet_features, mask)
        probs = F.softmax(logits, dim=1)
        return probs, similarities, jet_emb


# ============================================================================
# Loss Functions
# ============================================================================

def prototype_loss(logits, labels, similarities, prototype_layer, 
                   lambda_clf=1.0, lambda_clst=0.1, lambda_sep=0.1):
    """
    Combined loss: classification + clustering + separation.
    
    L = lambda_clf * L_clf + lambda_clst * L_clst + lambda_sep * L_sep
    
    L_clf: Cross-entropy classification loss
    L_clst: Clustering loss - pull inputs toward their class prototypes
    L_sep: Separation loss - push prototypes of different classes apart
    """
    # Classification loss
    L_clf = F.cross_entropy(logits, labels)
    
    # Clustering loss: minimize distance to same-class prototypes
    proto_labels = prototype_layer.get_prototype_labels()  # (P,)
    n_protos = len(proto_labels)
    
    # For each input, find the closest same-class prototype
    diff = similarities.unsqueeze(2) - proto_labels.unsqueeze(0).unsqueeze(0).float()
    # Actually, we want: for each input, maximize similarity to same-class prototypes
    # L_clst = -mean over inputs of mean similarity to same-class prototypes
    
    # Compute per-prototype similarity masked by class
    # similarities: (B, P), proto_labels: (P,)
    # For each input, gather similarities to prototypes of its class
    batch_size = similarities.shape[0]
    device = similarities.device
    
    L_clst = 0.0
    L_sep = 0.0
    
    for c in [0, 1]:
        # Prototypes of this class
        class_proto_mask = (proto_labels == c)  # (P,)
        class_proto_idx = torch.where(class_proto_mask)[0]
        
        if len(class_proto_idx) == 0:
            continue
        
        # Inputs of this class
        class_input_mask = (labels == c)  # (B,)
        if class_input_mask.sum() == 0:
            continue
        
        # Similarities of class-c inputs to class-c prototypes
        class_sims = similarities[class_input_mask][:, class_proto_idx]  # (N_c, P_c)
        
        # Clustering: maximize similarity to same-class prototypes
        # We want high similarity -> minimize -log(similarity)
        L_clst += -torch.mean(torch.log(class_sims.clamp(min=1e-10)))
        
        # Separation: minimize similarity to opposite-class prototypes
        other_proto_idx = torch.where(~class_proto_mask)[0]
        if len(other_proto_idx) > 0:
            other_sims = similarities[class_input_mask][:, other_proto_idx]
            L_sep += torch.mean(other_sims)
    
    L_clst = L_clst / 2.0  # average over classes
    L_sep = L_sep / 2.0
    
    total_loss = lambda_clf * L_clf + lambda_clst * L_clst + lambda_sep * L_sep
    
    return total_loss, {
        'loss_clf': L_clf.item(),
        'loss_clst': L_clst.item() if isinstance(L_clst, torch.Tensor) else L_clst,
        'loss_sep': L_sep.item() if isinstance(L_sep, torch.Tensor) else L_sep,
    }


# ============================================================================
# Utility
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick test
if __name__ == '__main__':
    print("=" * 60)
    print("Model Architecture Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = TopTaggingModel(
        constituent_dim=4,
        n_constituents=200,
        particle_dim=128,
        jet_dim=128,
        n_fastjet_features=6,
        n_prototypes=20,
        n_classes=2,
        temperature=1.0,
    ).to(device)
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    constituents = torch.randn(batch_size, 200, 4).to(device)
    fastjet_features = torch.randn(batch_size, 6).to(device)
    mask = torch.ones(batch_size, 200, dtype=torch.bool).to(device)
    # Simulate some padding
    mask[:, 150:] = False
    constituents[:, 150:] = 0.0
    
    logits, similarities, jet_emb = model(constituents, fastjet_features, mask)
    
    print(f"\nForward pass shapes:")
    print(f"  constituents: {constituents.shape}")
    print(f"  fastjet_features: {fastjet_features.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  similarities: {similarities.shape}")
    print(f"  jet_emb: {jet_emb.shape}")
    
    # Test loss
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    loss, loss_components = prototype_loss(logits, labels, similarities, model.prototype_layer)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"  L_clf: {loss_components['loss_clf']:.4f}")
    print(f"  L_clst: {loss_components['loss_clst']:.4f}")
    print(f"  L_sep: {loss_components['loss_sep']:.4f}")
    
    # Test prediction
    probs, sims, emb = model.predict(constituents, fastjet_features, mask)
    print(f"\nPredictions: {probs}")
    print(f"  Predicted classes: {probs.argmax(dim=1)}")
    print(f"  True labels: {labels}")
    
    print("\n" + "=" * 60)
    print("  Model architecture ready!")
    print("=" * 60)
