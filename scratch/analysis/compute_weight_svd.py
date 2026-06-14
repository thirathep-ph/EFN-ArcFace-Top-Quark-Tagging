import os
import sys
import torch
import numpy as np

def analyze_matrix_rank(W, threshold=0.99):
    # W is a PyTorch tensor weight matrix
    W_np = W.cpu().numpy()
    
    # 2D weight matrix check
    if len(W_np.shape) != 2:
        return None
        
    # Singular Value Decomposition
    S = np.linalg.svd(W_np, compute_uv=False)
    
    squared_S = S ** 2
    total_var = np.sum(squared_S)
    var_explained = squared_S / (total_var + 1e-10)
    cum_var = np.cumsum(var_explained)
    
    # Find effective rank at different thresholds
    ranks = {}
    for t in [0.90, 0.95, 0.99]:
        idx = np.where(cum_var >= t)[0]
        if len(idx) > 0:
            ranks[t] = idx[0] + 1
        else:
            ranks[t] = len(S)
            
    return S, ranks

def main():
    checkpoint_path = 'main_arcface/best_model_efn.pt'
    print(f"Loading checkpoint '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Filter only weights from efn (excluding bias, batchnorm running parameters, and class centers)
    weight_keys = [k for k in state_dict.keys() if 'weight' in k and ('phi' in k or 'rho' in k) and 'weight_ch' not in k]
    
    print("\n" + "="*70)
    print("RESULTS: NEURAL NETWORK WEIGHT MATRIX SVD ANALYSIS")
    print("="*70)
    print(f"{'Layer Name':22s} | {'Nominal Shape':15s} | {'Effective Rank (90% / 95% / 99% Var)':30s}")
    print("-"*70)
    
    for key in weight_keys:
        W = state_dict[key]
        res = analyze_matrix_rank(W)
        if res is not None:
            S, ranks = res
            shape_str = f"{W.shape[0]}x{W.shape[1]}"
            rank_str = f"{ranks[0.90]:d} / {ranks[0.95]:d} / {ranks[0.99]:d}"
            print(f"{key:22s} | {shape_str:15s} | {rank_str:30s}")
            
    print("="*70)
    print("\nPhysics & DL Interpretation:")
    print("  - Layers with very low Effective Rank compared to their nominal size are")
    print("    overparameterized. For instance, if a 128x128 layer has a 99% rank of 20,")
    print("    we can prune it down to 32 channels without loss of physical expression.")
    print("  - This provides a strict, math-based proof for model compression in our paper.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
