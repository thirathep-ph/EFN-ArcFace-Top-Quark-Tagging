"""
Decision Tree Rules Extracted from Model Behavior
===================================================
Trains a decision tree on the MODEL's latent space (cosine similarities)
to extract interpretable rules for subclass classification.

This answers: "What conditions does the model use to classify subclasses?"
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "dt_rules_results.json")


def get_model_predictions(model, h5_path, device, max_events=50000):
    """Get model's cosine similarities and predictions."""
    events, labels, weights = load_awkward(h5_path, max_events=max_events)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    
    all_sims, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, sims, emb = model(x, mask=m)
            preds = logits.argmax(dim=1)
            
            all_sims.append(sims.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
    
    return (np.vstack(all_sims), np.concatenate(all_preds), 
            np.concatenate(all_labels))


def get_subclass_labels(embeddings, labels):
    """Get subclass labels using KMeans on model embeddings."""
    from sklearn.cluster import KMeans
    
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    subclass_labels = np.full(len(labels), -1)
    
    for cls in [0, 1]:
        mask = labels == cls
        if mask.sum() > 10:
            km = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(emb_norm[mask])
            subclass_labels[mask] = km.labels_ + (2 * cls)
    
    return subclass_labels


def main():
    device, _ = get_device()
    print(f"Device: {device}\n")
    
    # Step 1: Load model and get predictions
    print("=" * 60)
    print("Step 1: Loading model predictions")
    print("=" * 60)
    
    model_path = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    t0 = time.time()
    sims, preds, labels = get_model_predictions(model, TEST_H5, device, max_events=50000)
    print(f"  Predictions: {sims.shape}, Time: {time.time()-t0:.0f}s")
    print(f"  Accuracy: {np.mean(preds == labels):.4f}")
    
    # Step 2: Get embeddings for subclass discovery
    print(f"\n{'='*60}")
    print("Step 2: Subclass discovery")
    print("=" * 60)
    
    # Get embeddings
    events, labels_raw, weights = load_awkward(TEST_H5, max_events=50000)
    ds = JetTaggingDataset(events, labels_raw, weights)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    
    all_emb = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu().numpy())
    
    embeddings = np.vstack(all_emb)
    subclass_labels = get_subclass_labels(embeddings, labels)
    
    # Subclass names
    subclass_names = {0: "QCD_Core", 1: "QCD_Edge", 2: "Top_Core", 3: "Top_Edge"}
    print(f"  Subclass distribution:")
    for sc in range(4):
        mask = subclass_labels == sc
        print(f"    {subclass_names[sc]}: {mask.sum()} ({100*mask.sum()/len(labels):.1f}%)")
    
    # Step 3: Train decision tree on model's cosine similarities
    print(f"\n{'='*60}")
    print("Step 3: Training decision tree on model's latent space")
    print("=" * 60)
    
    feature_names = ["cos_sim_to_QCD", "cos_sim_to_Top"]
    
    # Train DT with different depths
    dt_results = {}
    for depth in [2, 3, 5, 10]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(sims, subclass_labels)
        
        dt_preds = dt.predict(sims)
        dt_acc = accuracy_score(subclass_labels, dt_preds)
        
        dt_results[depth] = {
            "accuracy": float(dt_acc),
            "tree_text": export_text(dt, feature_names=feature_names),
        }
        
        print(f"\n  Depth={depth}: Accuracy={dt_acc:.4f}")
        print(f"  Tree rules:")
        print(export_text(dt, feature_names=feature_names))
    
    # Step 4: Extract rules from best depth
    print(f"\n{'='*60}")
    print("Step 4: Extracting interpretable rules")
    print("=" * 60)
    
    best_depth = max(dt_results.keys(), key=lambda d: dt_results[d]["accuracy"])
    best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_tree.fit(sims, subclass_labels)
    
    # Get feature importance
    feature_importance = best_tree.feature_importances_
    print(f"\n  Feature importance (depth={best_depth}):")
    for name, imp in zip(feature_names, feature_importance):
        print(f"    {name}: {imp:.4f}")
    
    # Get tree structure
    tree_text = export_text(best_tree, feature_names=feature_names)
    print(f"\n  Decision tree rules:\n{tree_text}")
    
    # Step 5: Visualize decision boundary
    print(f"\n{'='*60}")
    print("Step 5: Visualizing decision boundary")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Model's cosine similarity space
    ax = axes[0]
    colors = ['blue', 'cyan', 'red', 'orange']
    for sc in range(4):
        mask = subclass_labels == sc
        ax.scatter(sims[mask, 0], sims[mask, 1], c=colors[sc], alpha=0.3, s=5, label=subclass_names[sc])
    ax.set_xlabel('Cosine Sim to QCD')
    ax.set_ylabel('Cosine Sim to Top')
    ax.set_title('Model Latent Space (Cosine Similarities)')
    ax.legend()
    ax.grid(True, alpha=0.15)
    
    # Right: Decision tree boundary
    ax = axes[1]
    x_min, x_max = sims[:, 0].min() - 0.1, sims[:, 0].max() + 0.1
    y_min, y_max = sims[:, 1].min() - 0.1, sims[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    Z = best_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['blue', 'cyan', 'red', 'orange'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    for sc in range(4):
        mask = subclass_labels == sc
        ax.scatter(sims[mask, 0], sims[mask, 1], c=colors[sc], alpha=0.5, s=5, label=subclass_names[sc])
    ax.set_xlabel('Cosine Sim to QCD')
    ax.set_ylabel('Cosine Sim to Top')
    ax.set_title(f'Decision Tree Boundary (depth={best_depth})')
    ax.legend()
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "dt_rules_boundary.png"), dpi=150)
    plt.close()
    print("  Saved: dt_rules_boundary.png")
    
    # Save results
    results = {
        "n_events": len(labels),
        "accuracy": float(np.mean(preds == labels)),
        "dt_results": {str(d): dt_results[d] for d in dt_results},
        "best_depth": best_depth,
        "feature_importance": {name: float(imp) for name, imp in zip(feature_names, feature_importance)},
        "tree_rules": tree_text,
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {RESULTS_PATH}")
    
    print(f"\n{'='*60}")
    print("DECISION TREE RULES SUMMARY")
    print("=" * 60)
    print(f"  Best depth: {best_depth}")
    print(f"  DT accuracy: {dt_results[best_depth]['accuracy']:.4f}")
    print(f"  Rules:\n{tree_text}")
    
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
