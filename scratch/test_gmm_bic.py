import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def main():
    # Load saved embeddings
    data = np.load("results/embeddings_15ep.npz")
    embeddings = data["embeddings"]
    labels = data["labels"]
    
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Analyze Top
    top_mask = (labels == 1)
    top_latent = emb_norm[top_mask]
    
    print(f"Top samples: {len(top_latent)}")
    
    bics = []
    aics = []
    ks = list(range(2, 11))
    
    for k in ks:
        # Fit standard GMM to compute BIC/AIC
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=1)
        gmm.fit(top_latent)
        bic = gmm.bic(top_latent)
        aic = gmm.aic(top_latent)
        bics.append(bic)
        aics.append(aic)
        print(f"K={k:2d} | BIC={bic:.1f} | AIC={aic:.1f}")
        
    # Find optimal K
    opt_k_bic = ks[np.argmin(bics)]
    opt_k_aic = ks[np.argmin(aics)]
    print(f"\nOptimal K according to BIC: {opt_k_bic}")
    print(f"Optimal K according to AIC: {opt_k_aic}")

if __name__ == '__main__':
    main()
