"""
Decision Tree Baseline: Interpretable Physics-Based Classifier
==============================================================
Trains decision tree classifiers on physics features (Mass, Tau32, d12, zg, etc.)
for 20K events. Reports accuracy and feature importance at max_depth=5, 10, 15.
Compares with ArcFace accuracy (91.75%).
Saves plot to experiments/decision_tree.png
"""
import os, sys, json, time, gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device
from arcjet.utils.physics import compute_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
TRAIN_H5 = os.path.join(DATA_DIR, "train.h5")

N_EVENTS = 20000
ARC_FACE_ACCURACY = 0.9175  # reference accuracy
FEATURE_NAMES = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]


def compute_physics_features_chunked(h5_path, max_events=20000, chunk_size=2000):
    """Compute physics features in chunks to avoid memory issues."""
    events, labels, weights = load_awkward(h5_path, max_events=max_events, lazy=False)

    all_features = []
    n_chunks = (len(events) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(events))
        chunk = events[start:end]

        try:
            features = compute_features(chunk)
            all_features.append(features)
        except Exception as e:
            print(f"    Chunk {i+1}/{n_chunks} FastJet failed: {e}")
            # Fallback: basic features
            basic = np.zeros((len(chunk), 10))
            for j in range(len(chunk)):
                E = chunk[j][:, 0] if chunk[j].ndim == 2 else np.array([c[0] for c in chunk[j]])
                px = chunk[j][:, 1] if chunk[j].ndim == 2 else np.array([c[1] for c in chunk[j]])
                py = chunk[j][:, 2] if chunk[j].ndim == 2 else np.array([c[2] for c in chunk[j]])
                pz = chunk[j][:, 3] if chunk[j].ndim == 2 else np.array([c[3] for c in chunk[j]])

                jet_E = np.sum(E)
                jet_px = np.sum(px)
                jet_py = np.sum(py)
                jet_pz = np.sum(pz)
                mass = np.sqrt(max(0, jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2))
                basic[j, 0] = mass
                basic[j, 2] = np.sum(E > 0)
            all_features.append(basic)

    return np.vstack(all_features), labels


def main():
    print("=" * 60)
    print("DECISION TREE BASELINE")
    print("=" * 60)

    # Step 1: Compute physics features for training set
    print(f"\nStep 1: Computing physics features for {N_EVENTS} training events...")
    t0 = time.time()
    X_train, y_train = compute_physics_features_chunked(TRAIN_H5, max_events=N_EVENTS, chunk_size=2000)
    print(f"  Train features: {X_train.shape} ({time.time()-t0:.1f}s)")

    # Step 2: Compute physics features for test set
    print(f"\nStep 2: Computing physics features for {N_EVENTS} test events...")
    t0 = time.time()
    X_test, y_test = compute_physics_features_chunked(TEST_H5, max_events=N_EVENTS, chunk_size=2000)
    print(f"  Test features: {X_test.shape} ({time.time()-t0:.1f}s)")

    # Replace NaN/inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 3: Train decision trees with different depths
    depths = [5, 10, 15]
    results = {}

    for depth in depths:
        print(f"\nStep 3: Training Decision Tree (max_depth={depth})...")
        t0 = time.time()
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate
        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        report = classification_report(y_test, dt.predict(X_test), output_dict=True)

        # Feature importance
        importance = dt.feature_importances_

        results[depth] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "feature_importance": {FEATURE_NAMES[i]: float(importance[i]) for i in range(len(FEATURE_NAMES))},
            "train_time_seconds": float(train_time),
            "precision": float(report['1']['precision']),
            "recall": float(report['1']['recall']),
            "f1": float(report['1']['f1-score']),
        }

        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        print(f"  F1 score:       {report['1']['f1-score']:.4f}")
        print(f"  Training time:  {train_time:.2f}s")

        print(f"\n  Feature Importance (depth={depth}):")
        sorted_imp = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)
        for rank, (idx, imp) in enumerate(sorted_imp):
            print(f"    #{rank+1}: {FEATURE_NAMES[idx]:15s} = {imp:.4f}")

    # Step 4: Comparison with ArcFace
    print(f"\n{'='*60}")
    print("COMPARISON WITH ARC FACE")
    print("=" * 60)
    print(f"  ArcFace Accuracy: {ARC_FACE_ACCURACY:.4f}")
    for depth in depths:
        acc = results[depth]['test_accuracy']
        gap = ARC_FACE_ACCURACY - acc
        print(f"  DT (depth={depth:2d}):  {acc:.4f}  (gap: {gap:+.4f})")

    # Step 5: Generate plots
    print(f"\nStep 5: Generating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, depth in enumerate(depths):
        ax = axes[idx]
        importance = list(results[depth]['feature_importance'].values())
        names = list(results[depth]['feature_importance'].keys())

        # Sort by importance
        sorted_pairs = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
        names_sorted = [p[0] for p in sorted_pairs]
        imp_sorted = [p[1] for p in sorted_pairs]

        colors = ['darkgreen' if imp > 0.15 else 'steelblue' if imp > 0.05 else 'gray' for imp in imp_sorted]
        bars = ax.barh(range(len(names_sorted)), imp_sorted, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Decision Tree (max_depth={depth})\nAcc={results[depth]["test_accuracy"]:.4f}')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, axis='x')

        # Add value labels
        for i, v in enumerate(imp_sorted):
            if v > 0.01:
                ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "decision_tree.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: decision_tree.png")

    # Save results
    out = {
        "n_events": N_EVENTS,
        "arc_face_accuracy": ARC_FACE_ACCURACY,
        "depths": {str(d): results[d] for d in depths},
    }
    with open(os.path.join(EXPERIMENTS_DIR, "decision_tree_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: decision_tree_results.json")

    print(f"\n{'='*60}")
    print("DECISION TREE BASELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
