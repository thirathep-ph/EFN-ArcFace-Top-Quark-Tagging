"""
Decision Tree Rules from FastJet Physics Features
===================================================
Trains decision tree on physics features (Mass, Tau32, zg, etc.)
to extract interpretable rules for subclass classification.

This answers: "What physics conditions define each subclass?"
"""
import os, sys, json, time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.data.loader import load_awkward
from arcjet.utils.physics import compute_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "dt_features_results.json")

FEATURE_NAMES = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]


def get_subclass_labels_from_model():
    """Get subclass labels using the main ArcFace model."""
    import torch
    from torch.utils.data import DataLoader
    from arcjet.models.efn_arcface import TopTaggingModel
    from arcjet.data.loader import JetTaggingDataset
    from arcjet.utils.device import get_device
    from sklearn.cluster import KMeans
    
    device, _ = get_device()
    model_path = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    
    events, labels, weights = load_awkward(TEST_H5, max_events=50000, lazy=False)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    
    all_emb = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu().numpy())
    
    embeddings = np.vstack(all_emb)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Subclass discovery
    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = labels == cls
        if mask.sum() > 10:
            km = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(emb_norm[mask])
            subclass_labels[mask] = km.labels_ + (2 * cls)
    
    return events, labels, subclass_labels


def main():
    print("=" * 60)
    print("Decision Tree Rules from Physics Features")
    print("=" * 60)
    
    # Step 1: Get subclass labels from model
    print("\nStep 1: Getting subclass labels from model...")
    events, labels, subclass_labels = get_subclass_labels_from_model()
    
    subclass_names = {0: "QCD_Core", 1: "QCD_Edge", 2: "Top_Core", 3: "Top_Edge"}
    print(f"  Subclass distribution:")
    for sc in range(4):
        mask = subclass_labels == sc
        print(f"    {subclass_names[sc]}: {mask.sum()} ({100*mask.sum()/len(labels):.1f}%)")
    
    # Step 2: Compute physics features
    print(f"\nStep 2: Computing physics features...")
    t0 = time.time()
    features = compute_features(events)
    print(f"  Features: {features.shape}, Time: {time.time()-t0:.0f}s")
    
    # Step 3: Train decision tree on physics features
    print(f"\n{'='*60}")
    print("Step 3: Training decision tree on physics features")
    print("=" * 60)
    
    dt_results = {}
    for depth in [2, 3, 5, 10]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(features, subclass_labels)
        
        dt_preds = dt.predict(features)
        dt_acc = accuracy_score(subclass_labels, dt_preds)
        
        dt_results[depth] = {
            "accuracy": float(dt_acc),
            "tree_text": export_text(dt, feature_names=FEATURE_NAMES),
            "feature_importance": {name: float(imp) for name, imp in zip(FEATURE_NAMES, dt.feature_importances_)},
        }
        
        print(f"\n  Depth={depth}: Accuracy={dt_acc:.4f}")
        print(f"  Tree rules:")
        print(export_text(dt, feature_names=FEATURE_NAMES))
    
    # Step 4: Extract rules from best depth
    print(f"\n{'='*60}")
    print("Step 4: Best rules summary")
    print("=" * 60)
    
    best_depth = max(dt_results.keys(), key=lambda d: dt_results[d]["accuracy"])
    best = dt_results[best_depth]
    
    print(f"\n  Best depth: {best_depth}")
  Accuracy: {best['accuracy']:.4f}")
    print(f"\n  Feature importance:")
    sorted_imp = sorted(best'].items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp:
        if imp > 0.01:
            print(f"    {name:15s}: {imp:.4f}")
    
    print(f"\n  Decision tree rules:\n{best['tree_text']}")
    
    # Step 5: Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature importance
    ax = axes[0]
    names = [n for n, _ in sorted_imp if _ > 0.01]
    values = [v for _, v in sorted_imp if v > 0.01]
    ax.barh(names, values, color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance (depth={best_depth})')
    ax.grid(True, alpha=0.15, axis='x')
    
    # Decision boundary (top 2 features)
    ax = axes[1]
    top2 = [n for n, _ in sorted_imp[:2]]
    if len(top2) >= 2:
        idx1 = FEATURE_NAMES.index(top2[0])
        idx2 = FEATURE_NAMES.index(top2[1])
        
        colors = ['blue', 'cyan', 'red', 'orange']
        for sc in range(4):
            mask = subclass_labels == sc
            ax.scatter(features[mask, idx1], features[mask, idx2], 
                      c=colors[sc], alpha=0.3, s=5, label=subclass_names[sc])
        ax.set_xlabel(top2[0])
        ax.set_ylabel(top2[1])
        ax.set_title('Subclass Separation (Top 2 Features)')
        ax.legend()
        ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "dt_features_rules.png"), dpi=150)
    plt.close()
    print("\n  Saved: dt_features_rules.png")
    
    # Save
    results = {
        "n_events": len(labels),
        "best_depth": best_depth,
        "best_accuracy": best['accuracy'],
        "feature_importance": best['feature_importance'],
        "tree_rules": best['tree_text'],
        "all_depths": {str(d): dt_results[d] for d in dt_results},
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {RESULTS_PATH}")
    
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
