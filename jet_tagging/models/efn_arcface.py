"""
EFN (Energy Flow Network) + ArcFace Prototype Layer for Top Tagging
=====================================================================
Ante-hoc (Intrinsic) Interpretability Architecture:
    Constituents (E, px, py, pz) -> Centering/Rotation -> EFN -> jet_embedding
    jet_embedding -> ArcFace Prototype Layer -> Classification

EFN Logic:
    - MLP (Phi) sees ONLY angular coordinates (delta_eta, delta_phi).
    - Energy (E) is used as a weight for the sum pooling: Sum(E_i * Phi(eta_i, phi_i)).
    - This ensures IRC-safety and focuses interpretability on energy patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vector

# Register vector for torch (if using vector-torch integration)
# Note: For pure speed in forward passes, we use torch operations that mimic vector logic.

def get_efn_features(constituents, mask=None):
    """
    Transform raw Cartesian (E, px, py, pz) into EFN-ready features.
    Ensures centering and rotation for better interpretability.
    """
    E = constituents[:, :, 0]
    px = constituents[:, :, 1]
    py = constituents[:, :, 2]
    pz = constituents[:, :, 3]

    # 1. Basic Kinematics
    pt = torch.sqrt(px**2 + py**2 + 1e-10)
    eta = torch.asinh(pz / (pt + 1e-10))
    phi = torch.atan2(py, px)

    # 2. Jet-level 4-momentum for centering
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        jet_p4 = (constituents * m).sum(dim=1)
    else:
        jet_p4 = constituents.sum(dim=1)

    # Use vector-like logic for centering
    j_px, j_py, j_pz = jet_p4[:, 1], jet_p4[:, 2], jet_p4[:, 3]
    j_pt = torch.sqrt(j_px**2 + j_py**2 + 1e-10)
    j_eta = torch.asinh(j_pz / (j_pt + 1e-10))
    j_phi = torch.atan2(j_py, j_px)

    # 3. Relative Coordinates (Angular Only for EFN)
    d_eta = eta - j_eta.unsqueeze(1)
    d_phi = phi - j_phi.unsqueeze(1)
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi # Wrap phi

    # EFN input is JUST geometry
    features = torch.stack([d_eta, d_phi], dim=2) # (B, N, 2)

    # Normalized energy weights for pooling
    z = E / (E.sum(dim=1, keepdim=True) + 1e-10)

    if mask is not None:
        features = features * mask.unsqueeze(-1).float()
        z = z * mask.float()

    return features, z


class EFN(nn.Module):
    """
    Energy Flow Network: IRC-safe set processing.
    """
    def __init__(self, input_dim=2, particle_dim=256, jet_dim=256):
        super().__init__()
        # Phi: Per-particle MLP (sees only geometry)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, particle_dim),
        )
        
        # Rho: Jet-level MLP
        self.rho = nn.Sequential(
            nn.Linear(particle_dim, jet_dim),
            nn.BatchNorm1d(jet_dim),
            nn.ReLU(),
            nn.Linear(jet_dim, jet_dim),
        )

    def forward(self, x, z, mask=None):
        """
        x: (B, N, 2) - [d_eta, d_phi]
        z: (B, N) - [energy weights]
        """
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        phi_out_flat = self.phi(x_flat)
        phi_out = phi_out_flat.reshape(b, n, -1) # (B, N, particle_dim)

        # IRC-Safe Energy Weighted Pooling
        # Sum(z_i * Phi(x_i))
        weighted_phi = phi_out * z.unsqueeze(-1)
        jet_emb = weighted_phi.sum(dim=1) # (B, particle_dim)

        # Jet-level processing
        jet_emb = self.rho(jet_emb)
        return jet_emb


class ArcFacePrototypeLayer(nn.Module):
    def __init__(self, input_dim, n_prototypes=20, n_classes=2, s=32.0, m=0.5, fixed=True):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.fixed = fixed

        # 1. Initialize Prototypes in opposite hemispheres
        # Class 0 (QCD) will cluster around [1, 0, 0, ...]
        # Class 1 (Top) will cluster around [-1, 0, 0, ...]
        n_per_class = n_prototypes // 2
        protos = torch.zeros(n_prototypes, input_dim)
        
        # Anchor vectors
        anchor_qcd = torch.zeros(input_dim)
        anchor_qcd[0] = 1.0
        
        anchor_top = torch.zeros(input_dim)
        anchor_top[0] = -1.0
        
        # Add small random noise to separate sub-centers within the same class
        # This allows prototypes to act as distinct "bins" even if fixed
        noise_scale = 0.1
        
        # QCD Prototypes
        protos[:n_per_class] = anchor_qcd + torch.randn(n_per_class, input_dim) * noise_scale
        # Top Prototypes
        protos[n_per_class:] = anchor_top + torch.randn(n_per_class, input_dim) * noise_scale
        
        # Normalize to hypersphere
        protos = F.normalize(protos, dim=1)
        
        self.prototypes = nn.Parameter(protos, requires_grad=not fixed)

        self.register_buffer('prototype_class', torch.cat([
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_prototypes - n_per_class, dtype=torch.long),
        ]))

    def set_margin(self, m):
        """Dynamic update for margin warm-up"""
        self.m = m

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        similarities = x_norm @ p_norm.T

        n_qcd = self.n_prototypes // 2
        # Max similarity to sub-centers of each class
        sim_qcd = similarities[:, :n_qcd].max(dim=1)[0]
        sim_top = similarities[:, n_qcd:].max(dim=1)[0]
        logits = torch.stack([sim_qcd, sim_top], dim=1)

        if self.training and labels is not None:
            # Apply ArcFace margin penalty only to the ground-truth class
            logits_clamped = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(logits_clamped)
            
            # target_mask identifies which class's logit should receive the margin
            target_mask = F.one_hot(labels, num_classes=self.n_classes).bool()
            
            # cos(theta + m)
            m_logits = torch.cos(theta + self.m)
            
            # Replace ground truth logit with margin-penalized version
            logits = torch.where(target_mask, m_logits, logits)

        return logits * self.s, similarities


class TopTaggingModel(nn.Module):
    def __init__(self, n_prototypes=20, s=32.0, m=0.5, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.efn = EFN(input_dim=2, particle_dim=128, jet_dim=embedding_dim)
        self.prototype_layer = ArcFacePrototypeLayer(
            input_dim=embedding_dim, n_prototypes=n_prototypes, s=s, m=m
        )

    def forward(self, constituents, labels=None, mask=None):
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)
        logits, similarities = self.prototype_layer(jet_emb, labels)
        return logits, similarities, jet_emb

def arcface_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("EFN + ArcFace Prototype Model (xAI)")
    model = TopTaggingModel(embedding_dim=64)
    x = torch.randn(4, 200, 4)
    y = torch.randint(0, 2, (4,))
    logits, sims, emb = model(x, y)
    print(f"Logits: {logits.shape}, Sims: {sims.shape}, Emb: {emb.shape}")
    print("Success!")
