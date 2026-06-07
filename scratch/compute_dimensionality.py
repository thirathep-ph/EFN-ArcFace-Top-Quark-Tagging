import os
import sys
import numpy as np

def estimate_twonn_dimension(X, fraction=0.9):
    from sklearn.neighbors import NearestNeighbors
    # Use Cosine distance to match our ArcFace spherical metric
    nbrs = NearestNeighbors(n_neighbors=3, metric='cosine', n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filter zeros and non-physical values
    mask = (r1 > 1e-10) & (r2 > r1)
    r1 = r1[mask]
    r2 = r2[mask]
    
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    N_clean = len(mu_sorted)
    
    y = -np.log(1.0 - np.arange(1, N_clean + 1) / (N_clean + 1))
    x = np.log(mu_sorted)
    
    cutoff = int(N_clean * fraction)
    slope, _ = np.polyfit(x[:cutoff], y[:cutoff], 1)
    return slope

def main():
    embeddings_path = 'main_arcface/embeddings.npz'
    
    print("Loading test embeddings...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    
    # Normalize embeddings (ArcFace projection)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    print(f"Loaded {len(embeddings_norm)} embeddings of dimension {embeddings_norm.shape[1]}")
    
    # 1. Calculate Participation Ratio (PR) on full test set
    print("\nCalculating Participation Ratio...")
    cov = np.cov(embeddings_norm.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    
    # 2. Calculate Intrinsic Dimension (TwoNN) on a sample of 10,000 jets
    print("Calculating Intrinsic Dimension (TwoNN) on 10,000 jets...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(embeddings_norm), 10000, replace=False)
    intrinsic_dim = estimate_twonn_dimension(embeddings_norm[sample_idx])
    
    print("\n" + "="*50)
    print("PART 1: LATENT SPACE DIMENSIONALITY RESULTS")
    print("="*50)
    print(f"Embedding Space Dimension (Nominal)  : 64")
    print(f"Participation Ratio (Effective Lin)  : {pr:.2f}")
    print(f"TwoNN Intrinsic Dimension (Topology) : {intrinsic_dim:.2f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
