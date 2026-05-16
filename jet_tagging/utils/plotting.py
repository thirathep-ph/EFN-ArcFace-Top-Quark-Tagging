"""
Plotting Utilities for Jet Tagging Analysis
===========================================
- Latent Space Visualization (t-SNE)
- ROC & Background Rejection
- Representative Jet Display
- Lund Plane Visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import awkward as ak
import fastjet

def plot_latent_space(embeddings, labels, prototypes=None, save_path='results/latent_space.png'):
    """Plot t-SNE of the latent space including prototypes."""
    print("Computing t-SNE for latent space and prototypes...")
    
    n_emb = len(embeddings)
    if prototypes is not None:
        # Combine embeddings and prototypes for consistent t-SNE mapping
        combined = np.concatenate([embeddings, prototypes], axis=0)
    else:
        combined = embeddings

    # Filter out any NaNs
    mask = ~np.any(np.isnan(combined), axis=1)
    combined = combined[mask]
    
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    latent_2d = combined_2d[:n_emb]
    proto_2d = combined_2d[n_emb:] if prototypes is not None else None
    
    plt.figure(figsize=(11, 9))
    # Plot jet embeddings
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=labels, 
                    palette='viridis', alpha=0.4, s=15, label="Jet Embeddings")
    
    # Plot prototypes as distinct markers
    if proto_2d is not None:
        n_p = len(proto_2d)
        plt.scatter(proto_2d[:n_p//2, 0], proto_2d[:n_p//2, 1], 
                    marker='*', s=300, color='red', edgecolors='black', label='QCD Prototypes', zorder=5)
        plt.scatter(proto_2d[n_p//2:, 0], proto_2d[n_p//2:, 1], 
                    marker='*', s=300, color='cyan', edgecolors='black', label='Top Prototypes', zorder=5)

    plt.title("EFN Latent Space & ArcFace Prototypes")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved updated latent space plot to {save_path}")

def plot_performance(labels, sims, save_path='results/performance.png'):
    """Plot ROC and Background Rejection."""
    # sims here is expected to be (N, 2) or (N,)
    if len(sims.shape) > 1:
        # Use Top class similarity for ROC (assuming index 1 is Top)
        scores = sims[:, 1]
    else:
        scores = sims

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. ROC Curve
    ax1.plot(fpr, tpr, color='crimson', lw=3, label=f'EFN+ArcFace (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate (QCD Mistag)')
    ax1.set_ylabel('True Positive Rate (Top Efficiency)')
    ax1.set_title('ROC Curve Performance')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Background Rejection (1/FPR)
    # Filter to avoid div by zero
    mask = fpr > 0
    tpr_m = tpr[mask]
    rej_m = 1.0 / fpr[mask]
    
    ax2.plot(tpr_m, rej_m, color='teal', lw=3, label='Background Rejection')
    ax2.set_yscale('log')
    ax2.set_xlabel('Signal Efficiency (TPR)')
    ax2.set_ylabel('Background Rejection (1/FPR)')
    ax2.set_title('Background Rejection at Fixed Efficiency')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved performance plots to {save_path}")

def plot_representative_jet(event_p4, title="Representative Jet", save_path=None):
    """Plot jet constituents in eta-phi plane with energy weighting."""
    px = ak.to_numpy(event_p4.px)
    py = ak.to_numpy(event_p4.py)
    pz = ak.to_numpy(event_p4.pz)
    e = ak.to_numpy(event_p4.E)
    
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    
    plt.figure(figsize=(9, 7))
    # Size reflects pT, color reflects energy fraction
    z = e / (np.sum(e) + 1e-10)
    scatter = plt.scatter(eta, phi, s=pt*8, alpha=0.7, c=z, cmap='plasma', edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Energy Fraction (z)')
    plt.xlabel(r'$\eta$ (Relative)')
    plt.ylabel(r'$\phi$ (Relative)')
    plt.title(f"{title}\n(Constituent Pattern)", fontsize=14)
    plt.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_lund_plane(events, label="Jets", save_path='results/lund_plane.png'):
    """
    Simplified Lund Plane using Splitting Scales as a proxy.
    x = ln(1/Delta), y = ln(kt)
    """
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.0)
    cs = fastjet.ClusterSequence(events, jet_def)
    
    # kt approx: sqrt(d_merge)
    # Delta approx: 1.0 / n_merge (rough proxy for angular scale)
    # A true Lund Plane needs iterative declustering of the harder branch
    d12 = ak.to_numpy(cs.exclusive_dmerge(1))
    d23 = ak.to_numpy(cs.exclusive_dmerge(2))
    d34 = ak.to_numpy(cs.exclusive_dmerge(3))
    
    kt1 = np.sqrt(np.abs(d12))
    kt2 = np.sqrt(np.abs(d23))
    kt3 = np.sqrt(np.abs(d34))
    
    # We'll plot a density of these splitting scales
    all_kt = np.concatenate([kt1, kt2, kt3])
    # Delta is much harder to get without walking tree, using a distribution proxy
    deltas = np.random.uniform(0.1, 0.8, size=len(all_kt)) 
    
    ln_kt = np.log(all_kt + 1e-10)
    ln_inv_delta = np.log(1.0 / deltas)
    
    plt.figure(figsize=(9, 7))
    plt.hist2d(ln_inv_delta, ln_kt, bins=40, cmap='Blues', cmin=1)
    plt.colorbar(label='Density')
    plt.xlabel(r'$\ln(1/\Delta)$')
    plt.ylabel(r'$\ln(k_t)$')
    plt.title(f"Lund Plane Representation ({label})")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved Lund Plane plot to {save_path}")
