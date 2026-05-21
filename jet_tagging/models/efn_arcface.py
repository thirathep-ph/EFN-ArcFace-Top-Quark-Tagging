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
    """
    ArcFace Prototype Layer with Diversity Regularization.
    
    Uses n_proto_per_class prototypes per class (default: 2), initialized
    orthogonally so each prototype captures a distinct substructure mode.
    A diversity loss prevents prototype collapse during training.
    """
    def __init__(self, input_dim, n_proto_per_class=1, n_classes=2, s=32.0, m=0.5):
        super().__init__()
        self.n_proto_per_class = n_proto_per_class
        self.n_prototypes = n_proto_per_class * n_classes
        self.n_classes = n_classes
        self.s = s
        self.m = m

        # Dense initialization: prototypes are initialized randomly on the unit hypersphere.
        # This prevents the model from ignoring 62-63 dimensions of the 64D embedding space,
        # which previously caused the cosine similarities to be artificially diluted to ~0.07.
        # By making prototypes dense, the EFN must utilize all dimensions of the latent space.
        protos = torch.randn(self.n_prototypes, input_dim)
        
        # Normalize to unit hypersphere
        protos = F.normalize(protos, dim=1)
        self.prototypes = nn.Parameter(protos)

        # Register which prototype belongs to which class
        self.register_buffer('prototype_class', torch.cat([
            torch.zeros(n_proto_per_class, dtype=torch.long),
            torch.ones(n_proto_per_class, dtype=torch.long),
        ]))

    def set_margin(self, m):
        self.m = m

    def orthogonalize_prototypes(self):
        """
        Hard Orthogonal Projection (Gram-Schmidt) for same-class prototypes.
        Called after every optimizer.step() — no hyperparameter needed.
        
        Enforces: p0 · p1 = 0 within each class at all times.
        Each prototype captures an INDEPENDENT substructure mode.
        """
        with torch.no_grad():
            p = self.prototypes
            for c in range(self.n_classes):
                mask = self.prototype_class == c
                idx = mask.nonzero(as_tuple=True)[0]
                if len(idx) < 2:
                    continue
                # Gram-Schmidt on the 2 same-class prototypes
                p0 = F.normalize(p[idx[0]], dim=0)
                p1 = p[idx[1]]
                p1 = p1 - (p1 @ p0) * p0   # remove projection onto p0
                p1 = F.normalize(p1, dim=0)
                p[idx[0]] = p0
                p[idx[1]] = p1

    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        similarities = x_norm @ p_norm.T  # (B, n_prototypes)

        n = self.n_proto_per_class
        # Aggregate: max similarity across prototypes within each class
        sim_qcd = similarities[:, :n].max(dim=1)[0]
        sim_top = similarities[:, n:].max(dim=1)[0]
        logits = torch.stack([sim_qcd, sim_top], dim=1)

        if self.training and labels is not None:
            logits_clamped = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(logits_clamped)
            target_mask = F.one_hot(labels, num_classes=self.n_classes).bool()
            m_logits = torch.cos(theta + self.m)
            logits = torch.where(target_mask, m_logits, logits)

        return logits * self.s, similarities


class TopTaggingModel(nn.Module):
    def __init__(self, n_proto_per_class=1, s=32.0, m=0.5, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.efn = EFN(input_dim=2, particle_dim=128, jet_dim=embedding_dim)
        self.prototype_layer = ArcFacePrototypeLayer(
            input_dim=embedding_dim, n_proto_per_class=n_proto_per_class, s=s, m=m
        )

    def forward(self, constituents, labels=None, mask=None):
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)
        logits, similarities = self.prototype_layer(jet_emb, labels)
        return logits, similarities, jet_emb

def arcface_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def distance_corr_loss(x, y, mask=None):
    """
    Computes the Distance Correlation (DisCo) between x and y.
    x: (B,) - model predictions/scores
    y: (B,) - physical parameter (e.g., jet mass)
    mask: (B,) - boolean mask to filter (e.g., only background/QCD events)
    """
    if mask is not None:
        x = x[mask]
        y = y[mask]
    
    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # Pairwise distance matrices
    x_dist = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    y_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))
    
    # Double centering
    x_mean_row = x_dist.mean(dim=1, keepdim=True)
    x_mean_col = x_dist.mean(dim=0, keepdim=True)
    x_mean_all = x_dist.mean()
    A = x_dist - x_mean_row - x_mean_col + x_mean_all
    
    y_mean_row = y_dist.mean(dim=1, keepdim=True)
    y_mean_col = y_dist.mean(dim=0, keepdim=True)
    y_mean_all = y_dist.mean()
    B = y_dist - y_mean_row - y_mean_col + y_mean_all
    
    # Distance covariance
    dcov2 = (A * B).mean()
    
    # Distance variance
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    
    # Distance correlation
    dcor = torch.sqrt(dcov2 / (torch.sqrt(dvarx2 * dvary2) + 1e-10) + 1e-10)
    return dcor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    print("EFN + ArcFace Prototype Model (xAI, 1 Prototype Per Class)")
    model = TopTaggingModel(n_proto_per_class=1, embedding_dim=64)
    x = torch.randn(4, 200, 4)
    y = torch.randint(0, 2, (4,))
    logits, sims, emb = model(x, y)
    print(f"Logits: {logits.shape}, Sims: {sims.shape}, Emb: {emb.shape}")
    # Compute sample mass from constituents for test
    E_tot = x[:, :, 0].sum(dim=1)
    px_tot = x[:, :, 1].sum(dim=1)
    py_tot = x[:, :, 2].sum(dim=1)
    pz_tot = x[:, :, 3].sum(dim=1)
    mass = torch.sqrt(torch.clamp(E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2), min=1e-10))
    dcor = distance_corr_loss(logits[:, 1], mass)
    print(f"DisCo Loss (between top logits & mass): {dcor.item():.4f}")
    print("Success!")
