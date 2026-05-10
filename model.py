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
# Prototype Layer (xAI) — Energy-Based Model
# ============================================================================

class PrototypeLayer(nn.Module):
    """
    Prototype layer with cosine similarity (Energy-Based Model).
    
    Architecture:
        - Normalize inputs and prototypes to unit sphere
        - Compute cosine similarity: sim = z·p  (range [-1, 1])
        - Convert to distance: d = 1 - sim  (range [0, 2])
        - Per-class energy: min distance to that class's prototypes
        - Classification via softmax over negative energies
    
    20 prototypes: 10 for Top class, 10 for QCD class.
    No learned classifier layer — pure prototype-based decision.
    """
    def __init__(self, input_dim, n_prototypes=20, n_classes=2):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        
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

    def forward(self, x):
        """
        x: (batch_size, input_dim) - concatenated [jet_embedding, fastjet_features]
        
        Returns:
            d_qcd: (batch_size,) - min cosine distance to QCD prototypes
            d_top: (batch_size,) - min cosine distance to Top prototypes
            similarities: (batch_size, n_prototypes) - cosine similarities to all prototypes
        """
        # Normalize to unit sphere
        x_norm = F.normalize(x, dim=1)  # (B, D)
        p_norm = F.normalize(self.prototypes, dim=1)  # (P, D)
        
        # Cosine similarity: (B, P), range [-1, 1]
        similarities = x_norm @ p_norm.T
        
        # Convert to distance: d = 1 - sim, range [0, 2]
        # d = 0 means identical direction, d = 2 means opposite direction
        distances = 1.0 - similarities  # (B, P)
        
        # Per-class min distance (energy)
        n_qcd = self.n_prototypes // 2
        d_qcd = distances[:, :n_qcd].min(dim=1)[0]  # (B,)
        d_top = distances[:, n_qcd:].min(dim=1)[0]  # (B,)
        
        return d_qcd, d_top, similarities

    def get_prototype_labels(self):
        """Return class label for each prototype."""
        return self.prototype_class

    def get_prototype_activations(self, x):
        """
        Get per-prototype similarity for interpretability.
        Returns: (batch_size, n_prototypes)
        """
        with torch.no_grad():
            _, _, similarities = self.forward(x)
        return similarities


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
        )

    def forward(self, constituents, fastjet_features, mask=None):
        """
        constituents: (batch_size, 200, 4) - (E, px, py, pz)
        fastjet_features: (batch_size, 6) - [mass, pT, eta, tau21, tau32, D2]
        mask: (batch_size, 200) - True for valid particles
        
        Returns:
            d_qcd: (batch_size,) - min cosine distance to QCD prototypes
            d_top: (batch_size,) - min cosine distance to Top prototypes
            similarities: (batch_size, 20) - cosine similarities to all prototypes
            jet_emb: (batch_size, jet_dim) - jet embedding from DeepSets
        """
        # DeepSets jet embedding
        jet_emb = self.deepsets(constituents, mask)  # (B, jet_dim)
        
        # Concatenate with FastJet features
        combined = torch.cat([jet_emb, fastjet_features], dim=1)  # (B, jet_dim + 6)
        combined = self.combined_proj(combined)
        
        # Prototype layer
        d_qcd, d_top, similarities = self.prototype_layer(combined)
        
        return d_qcd, d_top, similarities, jet_emb

    def predict(self, constituents, fastjet_features, mask=None):
        """Get class probabilities from energy-based model."""
        d_qcd, d_top, similarities, jet_emb = self.forward(constituents, fastjet_features, mask)
        # P = softmax(-E) where E = min distance to class prototypes
        energies = torch.stack([d_qcd, d_top], dim=1)  # (B, 2)
        probs = F.softmax(-energies, dim=1)  # lower energy = higher probability
        return probs, similarities, jet_emb


# ============================================================================
# Loss Functions — Energy-Based Model
# ============================================================================

def energy_loss(d_qcd, d_top, labels, class_weight=None):
    """
    Energy-Based Model loss function.
    
    Energy for each class = min cosine distance to that class's prototypes.
    Classification via softmax over negative energies:
        P(class=c) = exp(-E_c) / (exp(-E_qcd) + exp(-E_top))
    
    Lower energy = higher probability = better match to class prototypes.
    
    Args:
        d_qcd: (batch_size,) - min cosine distance to QCD prototypes [0, 2]
        d_top: (batch_size,) - min cosine distance to Top prototypes [0, 2]
        labels: (batch_size,) - 0=QCD, 1=Top
        class_weight: (2,) tensor - weight for each class (handles imbalance)
    
    Returns:
        loss: scalar tensor
    """
    # Stack energies: (B, 2) where col 0 = E_qcd, col 1 = E_top
    energies = torch.stack([d_qcd, d_top], dim=1)
    
    # Softmax over negative energies: P = softmax(-E)
    # Lower energy -> higher probability
    log_probs = F.log_softmax(-energies, dim=1)
    
    # Negative log-likelihood with optional class weighting
    if class_weight is not None:
        # class_weight: (2,) -> weight per class
        # For each sample, weight = class_weight[label]
        sample_weights = class_weight[labels]
        loss = -log_probs[range(len(labels)), labels] * sample_weights
        loss = loss.mean()
    else:
        loss = F.nll_loss(log_probs, labels)
    
    return loss



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
    
    d_qcd, d_top, similarities, jet_emb = model(constituents, fastjet_features, mask)
    
    print(f"\nForward pass shapes:")
    print(f"  constituents: {constituents.shape}")
    print(f"  fastjet_features: {fastjet_features.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  d_qcd: {d_qcd.shape}")
    print(f"  d_top: {d_top.shape}")
    print(f"  similarities: {similarities.shape}")
    print(f"  jet_emb: {jet_emb.shape}")
    
    # Test energy loss
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    loss = energy_loss(d_qcd, d_top, labels)
    
    print(f"\nEnergy loss: {loss.item():.4f}")
    
    # Test with class weights
    class_weight = torch.tensor([1.0, 1.0]).to(device)
    loss_w = energy_loss(d_qcd, d_top, labels, class_weight=class_weight)
    print(f"Energy loss (weighted): {loss_w.item():.4f}")
    
    # Test prediction
    probs, sims, emb = model.predict(constituents, fastjet_features, mask)
    print(f"\nPredictions: {probs}")
    print(f"  Predicted classes: {probs.argmax(dim=1)}")
    print(f"  True labels: {labels}")
    print(f"  Energies: d_qcd={d_qcd}, d_top={d_top}")

    
    print("\n" + "=" * 60)
    print("  Model architecture ready!")
    print("=" * 60)
