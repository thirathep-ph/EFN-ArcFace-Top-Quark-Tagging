"""
PFN (Particle Flow Network) + ArcFace Prototype Layer for Top Tagging
=====================================================================
Ante-hoc (Intrinsic) Interpretability Architecture:
    Constituents (E, px, py, pz) -> Relative Features -> PFN -> jet_embedding
    jet_embedding -> ArcFace Prototype Layer -> Classification

Interpretability:
    The model is inherently interpretable because classification is based
    on similarity to learned 'Prototypes' in the latent space.
    FastJet features are used post-training to GROUND these prototypes
    in physical observables (mass, tau32, etc.).

ArcFace Loss:
    Used to enhance class separation and prototype clustering.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# PFN (Particle Flow Network)
# ============================================================================

def get_relative_features(constituents, mask=None):
    """
    Transform (E, px, py, pz) to PFN-style relative features:
    (ln pT, ln E, Delta eta, Delta phi, pT_rel, E_rel)

    Args:
        constituents: (B, N, 4) - (E, px, py, pz)
        mask: (B, N) - True for valid particles
    """
    E = constituents[:, :, 0]
    px = constituents[:, :, 1]
    py = constituents[:, :, 2]
    pz = constituents[:, :, 3]

    # Particle-level pT and eta/phi
    pt = torch.sqrt(px**2 + py**2 + 1e-10)
    # Rapidity y = 0.5 * ln((E+pz)/(E-pz))
    y = 0.5 * torch.log((E + pz + 1e-10) / (E - pz + 1e-10))
    phi = torch.atan2(py, px)

    # Jet-level 4-momentum (sum of constituents)
    if mask is not None:
        # Zero out masked particles
        m = mask.unsqueeze(-1).float()
        jet_p4 = (constituents * m).sum(dim=1)
    else:
        jet_p4 = constituents.sum(dim=1)

    jet_E = jet_p4[:, 0]
    jet_px = jet_p4[:, 1]
    jet_py = jet_p4[:, 2]
    jet_pz = jet_p4[:, 3]

    jet_pt = torch.sqrt(jet_px**2 + jet_py**2 + 1e-10)
    jet_y = 0.5 * torch.log((jet_E + jet_pz + 1e-10) / (jet_E - jet_pz + 1e-10))
    jet_phi = torch.atan2(jet_py, jet_px)

    # Relative coordinates
    # Note: Wrap phi to [-pi, pi]
    d_phi = phi - jet_phi.unsqueeze(1)
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    d_y = y - jet_y.unsqueeze(1)

    # Normalized pT and E
    pt_rel = pt / jet_pt.unsqueeze(1)
    E_rel = E / jet_E.unsqueeze(1)

    # Log features (standard in PFN)
    ln_pt = torch.log(pt + 1e-10)
    ln_E = torch.log(E + 1e-10)

    # Stack features: (B, N, 6)
    # [ln_pt, ln_E, d_y, d_phi, pt_rel, E_rel]
    features = torch.stack([ln_pt, ln_E, d_y, d_phi, pt_rel, E_rel], dim=2)

    # Apply mask again to be safe
    if mask is not None:
        features = features * mask.unsqueeze(-1).float()

    return features


class PFNEncoder(nn.Module):
    """
    Per-particle MLP: particle_features -> particle_embedding
    """
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (batch_size, n_constituents, input_dim)
        Returns: (batch_size, n_constituents, output_dim)
        """
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        out_flat = self.net(x_flat)
        return out_flat.reshape(b, n, -1)


class PFN(nn.Module):
    """
    Particle Flow Network: permutation-invariant set processing.
    """
    def __init__(self, input_dim=6, particle_dim=256, jet_dim=256):
        super().__init__()
        self.phi = PFNEncoder(input_dim, hidden_dim=particle_dim, output_dim=particle_dim)
        self.rho = nn.Sequential(
            nn.Linear(particle_dim, jet_dim),
            nn.BatchNorm1d(jet_dim),
            nn.ReLU(),
            nn.Linear(jet_dim, jet_dim),
            nn.BatchNorm1d(jet_dim),
            nn.ReLU(),
            nn.Linear(jet_dim, jet_dim),
        )

    def forward(self, x, mask=None):
        """
        x: (batch_size, n_constituents, input_dim)
        """
        # Per-particle encoding
        particle_emb = self.phi(x)  # (B, N, particle_dim)

        # Sum pooling
        if mask is not None:
            particle_emb = particle_emb * mask.unsqueeze(-1).float()
            jet_emb = particle_emb.sum(dim=1)
        else:
            jet_emb = particle_emb.mean(dim=1)

        # Jet-level encoding
        jet_emb = self.rho(jet_emb)
        return jet_emb


# ============================================================================
# ArcFace Prototype Layer (Ante-hoc Interpretability)
# ============================================================================

class ArcFacePrototypeLayer(nn.Module):
    """
    Ante-hoc interpretability layer using ArcFace (Additive Angular Margin).

    20 Prototypes: 10 for QCD (class 0), 10 for Top (class 1).
    During training: Adds angular margin to target class prototypes.
    """
    def __init__(self, input_dim, n_prototypes=20, n_classes=2, s=32.0, m=0.5):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.s = s  # Scale factor
        self.m = m  # Angular margin

        # Prototypes: (n_prototypes, input_dim)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Prototype labels: 0 for first half, 1 for second half
        self.register_buffer(
            'prototype_class',
            torch.cat([
                torch.zeros(n_prototypes // 2, dtype=torch.long),
                torch.ones(n_prototypes - n_prototypes // 2, dtype=torch.long),
            ])
        )

    def forward(self, x, labels=None):
        """
        x: (batch_size, input_dim) - jet embedding
        labels: (batch_size,) - ground truth class (only for training)

        Returns:
            logits: (batch_size, 2) - class scores for ArcFace
            similarities: (batch_size, n_prototypes) - raw cosine similarities
        """
        # Normalize to hypersphere
        x_norm = F.normalize(x, dim=1)  # (B, D)
        p_norm = F.normalize(self.prototypes, dim=1)  # (P, D)

        # Cosine similarity: (B, P)
        similarities = x_norm @ p_norm.T

        # --- Ante-hoc Decision Logic ---
        # Get max similarity for each class
        # (B, 10) -> (B,)
        n_qcd = self.n_prototypes // 2
        sim_qcd = similarities[:, :n_qcd].max(dim=1)[0]
        sim_top = similarities[:, n_qcd:].max(dim=1)[0]

        # Combine into class-level logits: (B, 2)
        logits = torch.stack([sim_qcd, sim_top], dim=1)

        # --- ArcFace Margin (Training only) ---
        if self.training and labels is not None:
            # Clamp for numerical stability in acos
            logits_clamped = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(logits_clamped)

            # Target class index
            target_mask = F.one_hot(labels, num_classes=self.n_classes).bool()

            # Apply additive angular margin: cos(theta + m)
            # This makes it harder for the embedding to belong to the target class
            # forcing them to be even closer to the prototypes.
            m_logits = torch.cos(theta + self.m)

            # Replace target logits with margin-adjusted versions
            logits = torch.where(target_mask, m_logits, logits)

        # Scale for softmax
        logits = logits * self.s

        return logits, similarities


# ============================================================================
# Full Model
# ============================================================================

class TopTaggingModel(nn.Module):
    """
    Complete Model: PFN + ArcFace Prototype Layer.
    Ante-hoc architecture with physical feature interpretation.
    """
    def __init__(self,
                 constituent_dim=6,  # PFN features
                 n_constituents=200,
                 particle_dim=256,
                 jet_dim=256,
                 n_prototypes=20,
                 n_classes=2,
                 s=32.0,
                 m=0.5):
        super().__init__()

        self.pfn = PFN(
            input_dim=constituent_dim,
            particle_dim=particle_dim,
            jet_dim=jet_dim,
        )

        self.prototype_layer = ArcFacePrototypeLayer(
            input_dim=jet_dim,
            n_prototypes=n_prototypes,
            n_classes=n_classes,
            s=s,
            m=m,
        )

    def forward(self, constituents, labels=None, mask=None):
        """
        Input: Raw (E, px, py, pz)
        """
        # 1. Transform to PFN relative features on-the-fly
        pfn_features = get_relative_features(constituents, mask)

        # 2. Extract jet embedding
        jet_emb = self.pfn(pfn_features, mask)

        # 3. Prototype layer (ArcFace)
        logits, similarities = self.prototype_layer(jet_emb, labels)

        return logits, similarities, jet_emb

    @torch.no_grad()
    def predict(self, constituents, mask=None):
        """Standard prediction (no margin)."""
        self.eval()
        logits, similarities, jet_emb = self.forward(constituents, labels=None, mask=mask)
        probs = F.softmax(logits, dim=1)
        return probs, similarities, jet_emb


# ============================================================================
# Loss & Utility
# ============================================================================

def arcface_loss(logits, labels, class_weight=None):
    """Simple CrossEntropy on ArcFace logits."""
    return F.cross_entropy(logits, labels, weight=class_weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("PFN + ArcFace Prototype Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TopTaggingModel().to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward
    batch_size = 4
    x = torch.randn(batch_size, 200, 4).to(device) # E, px, py, pz
    y = torch.randint(0, 2, (batch_size,)).to(device)
    mask = torch.ones(batch_size, 200, dtype=torch.bool).to(device)

    logits, sims, emb = model(x, y, mask)
    loss = arcface_loss(logits, y)

    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("Success!")

    
    print("\n" + "=" * 60)
    print("  Model architecture ready!")
    print("=" * 60)
