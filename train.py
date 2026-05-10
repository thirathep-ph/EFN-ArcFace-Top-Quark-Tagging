"""
Training Pipeline for Top Tagging with DeepSets + Prototype Layer
=================================================================
Trains the model on train.h5, validates on val.h5, tests on test.h5.

Optimized version: pre-computes all numpy arrays in __init__ so that
__getitem__ does O(1) numpy slicing — NO per-particle Python loop.

Usage:
    python train.py                          # Full training
    python train.py --quick                   # Quick test (2,000 events)
    python train.py --quick --n_events 5000   # Quick test with 5,000 events
    python train.py --epochs 20 --batch_size 256

Output:
    - best_model.pt:  Best model checkpoint
    - training_history.json:  Loss and accuracy history
    - training_results.png:  9-panel figure (loss, accuracy, energy, decision
      boundary, prototype activations, ROC, confusion matrix, PR curve, scores)
"""
import json
import time
import sys
import argparse
import numpy as np
import awkward as ak
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import TopTaggingModel, energy_loss, count_parameters

from fastjet_features import compute_features
from to_awkward import load_awkward


# ============================================================================
# Dataset Wrapper for PyTorch (OPTIMIZED — Vectorized __getitem__)
# ============================================================================

class JetTaggingDataset(Dataset):
    """
    PyTorch Dataset that loads H5 data and computes FastJet features.

    OPTIMIZATION: Pre-computes ALL numpy arrays in __init__ using vectorized
    Awkward -> numpy conversion. __getitem__ does O(1) numpy slicing --
    NO per-particle Python loop. This eliminates the main training bottleneck.
    """
    def __init__(self, h5_path, max_events=None, precompute_fastjet=True, max_constituents=200):
        self.h5_path = h5_path
        self.max_constituents = max_constituents

        # Load constituents as awkward arrays
        print(f"Loading {h5_path}...")
        self.events, self.labels, self.weights = load_awkward(
            h5_path, max_events=max_events, trim_padding=True
        )
        self.n = len(self.events)

        # Pre-compute FastJet features
        if precompute_fastjet:
            print(f"Computing FastJet features for {self.n} events...")
            t0 = time.time()
            self.fastjet_features = compute_features(self.events)
            t1 = time.time()
            print(f"  Done in {t1-t0:.1f}s ({self.n/(t1-t0):.0f} events/s)")
        else:
            self.fastjet_features = None

        # ====================================================================
        # OPTIMIZATION: Pre-compute padded constituent array ONCE
        # ====================================================================
        # Convert awkward arrays to flat numpy arrays (vectorized -- NO Python loop)
        print(f"  Pre-computing padded constituent arrays ({self.n} events)...")
        t0 = time.time()

        # Get counts per event
        counts = ak.num(self.events)
        self.counts_np = ak.to_numpy(counts).astype(np.int32)

        # Flatten all particles into 1D arrays
        E_flat = ak.to_numpy(ak.flatten(self.events.E)).astype(np.float32)
        px_flat = ak.to_numpy(ak.flatten(self.events.px)).astype(np.float32)
        py_flat = ak.to_numpy(ak.flatten(self.events.py)).astype(np.float32)
        pz_flat = ak.to_numpy(ak.flatten(self.events.pz)).astype(np.float32)

        # Build padded array: (N, max_constituents, 4)
        self.constituents = np.zeros((self.n, max_constituents, 4), dtype=np.float32)
        self.mask = np.zeros((self.n, max_constituents), dtype=bool)

        # Fill using numpy slicing (still per-event, but each iteration is
        # a vectorized slice assignment -- MUCH faster than per-particle loop)
        start = 0
        for i in range(self.n):
            n_p = self.counts_np[i]
            n_use = min(n_p, max_constituents)
            if n_use > 0:
                end = start + n_use
                self.constituents[i, :n_use, 0] = E_flat[start:end]
                self.constituents[i, :n_use, 1] = px_flat[start:end]
                self.constituents[i, :n_use, 2] = py_flat[start:end]
                self.constituents[i, :n_use, 3] = pz_flat[start:end]
                self.mask[i, :n_use] = True
                start = end

        t1 = time.time()
        print(f"  Done in {t1-t0:.2f}s")

        # Convert labels and weights to numpy
        self.labels_np = np.asarray(self.labels, dtype=np.int64)
        self.weights_np = np.asarray(self.weights, dtype=np.float32)

        n_top = int(self.labels_np.sum())
        n_qcd = self.n - n_top
        print(f"  {self.n:,} events ({n_top:,} Top, {n_qcd:,} QCD)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        O(1) numpy slicing -- NO per-particle Python loop!

        Returns:
            constituents: (200, 4) numpy array - (E, px, py, pz)
            fastjet_features: (6,) numpy array - [mass, pT, eta, tau21, tau32, D2]
            label: int - 0=QCD, 1=Top
            weight: float
            mask: (200,) bool - True for valid particles
        """
        constituents = self.constituents[idx]  # (200, 4) -- O(1) slice
        mask = self.mask[idx]  # (200,) -- O(1) slice

        fj = self.fastjet_features[idx] if self.fastjet_features is not None else np.zeros(6, dtype=np.float32)

        label = self.labels_np[idx]
        weight = self.weights_np[idx]

        return constituents, fj, label, weight, mask


def collate_fn(batch):
    """Collate batch of (constituents, fj, label, weight, mask) into tensors."""
    constituents = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    fj = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    mask = torch.from_numpy(np.stack([b[4] for b in batch])).bool()
    return constituents, fj, labels, weights, mask


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, loader, optimizer, device, class_weight=None):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for constituents, fj, labels, weights, mask in loader:
        constituents = constituents.to(device)
        fj = fj.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        d_qcd, d_top, similarities, jet_emb = model(constituents, fj, mask)

        loss = energy_loss(d_qcd, d_top, labels, class_weight=class_weight)

        loss.backward()
        optimizer.step()

        # Accuracy from energy-based decision: lower energy = predicted class
        preds = (d_top < d_qcd).long()  # 1 if Top (lower d_top), 0 if QCD (lower d_qcd)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples



@torch.no_grad()
def evaluate(model, loader, device, class_weight=None):
    """Evaluate model. Returns loss, accuracy, and all predictions."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_d_qcd = []
    all_d_top = []
    all_labels = []
    all_similarities = []

    for constituents, fj, labels, weights, mask in loader:
        constituents = constituents.to(device)
        fj = fj.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        d_qcd, d_top, similarities, jet_emb = model(constituents, fj, mask)

        loss = energy_loss(d_qcd, d_top, labels, class_weight=class_weight)

        preds = (d_top < d_qcd).long()
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        all_d_qcd.append(d_qcd.cpu())
        all_d_top.append(d_top.cpu())
        all_labels.append(labels.cpu())
        all_similarities.append(similarities.cpu())

    all_d_qcd = torch.cat(all_d_qcd)
    all_d_top = torch.cat(all_d_top)
    all_labels = torch.cat(all_labels)
    all_similarities = torch.cat(all_similarities)

    return (total_loss / total_samples,
            total_correct / total_samples,
            all_d_qcd, all_d_top, all_labels, all_similarities)


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def comprehensive_evaluation(test_labels, test_d_qcd, test_d_top, test_similarities,
                              proto_labels_arr=None):
    """
    Compute comprehensive classification metrics.

    Returns dict with: accuracy, precision, recall, f1, confusion matrix,
    classification report, ROC AUC, and score arrays.
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report,
                                 roc_curve, auc, precision_recall_curve,
                                 average_precision_score)

    labels_np = test_labels.numpy()
    d_qcd_np = test_d_qcd.numpy()
    d_top_np = test_d_top.numpy()

    # Predictions: lower energy = predicted class
    preds = (d_top_np < d_qcd_np).astype(np.int64)

    # Score: d_qcd - d_top (positive = more Top-like)
    scores = d_qcd_np - d_top_np

    # Confusion matrix
    cm = confusion_matrix(labels_np, preds)

    # Metrics
    acc = accuracy_score(labels_np, preds)
    prec = precision_score(labels_np, preds, zero_division=0)
    rec = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)

    # ROC
    fpr, tpr, _ = roc_curve(labels_np, scores)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision_curve, recall_curve, _ = precision_recall_curve(labels_np, scores)
    avg_prec = average_precision_score(labels_np, scores)

    # Classification report as dict
    report = classification_report(labels_np, preds, target_names=['QCD', 'Top'],
                                   output_dict=True, zero_division=0)

    # Per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall for Top
    tnr_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity for QCD
    ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision for Top
    npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

    result = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_prec,
        'true_positive_rate': tpr_val,
        'true_negative_rate': tnr_val,
        'positive_predictive_value': ppv_val,
        'negative_predictive_value': npv_val,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': preds,
        'scores': scores,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
    }
    return result


def print_evaluation_report(metrics):
    """Print a formatted evaluation report."""
    cm = metrics['confusion_matrix']
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)

    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  QCD    Top")
    print(f"  Actual QCD    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"         Top    {cm[1,0]:6d}  {cm[1,1]:6d}")

    print(f"\nMetrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Top):    {metrics['precision']:.4f}")
    print(f"  Recall (Top):       {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:      {metrics['avg_precision']:.4f}")
    print(f"  TPR (Sensitivity):  {metrics['true_positive_rate']:.4f}")
    print(f"  TNR (Specificity):  {metrics['true_negative_rate']:.4f}")
    print(f"  PPV (Precision):    {metrics['positive_predictive_value']:.4f}")
    print(f"  NPV:                {metrics['negative_predictive_value']:.4f}")

    print(f"\nPer-class breakdown:")
    for cls_name in ['QCD', 'Top']:
        cr = metrics['classification_report'][cls_name]
        print(f"  {cls_name}: precision={cr['precision']:.4f}, "
              f"recall={cr['recall']:.4f}, f1={cr['f1-score']:.4f}, "
              f"support={int(cr['support'])}")


# ============================================================================
# Plotting
# ============================================================================

def plot_results(history, test_d_qcd, test_d_top, test_labels, test_similarities,
                 save_path='training_results.png'):
    """
    Generate a 9-panel figure summarizing training results.

    Panels:
        1. Loss curves (train + val)
        2. Accuracy curves (train + val)
        3. Energy distribution (d_qcd vs d_top by true class)
        4. Decision boundary scatter (d_qcd vs d_top)
        5. Prototype activation bar chart
        6. ROC curve with AUC
        7. Confusion matrix heatmap
        8. Precision-Recall curve
        9. Score distribution by class
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
        import seaborn as sns
    except ImportError:
        print("  [plot_results] matplotlib or sklearn not available. Skipping plot.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('Top Tagging -- DeepSets + Prototype Layer (EBM)', fontsize=16, y=1.02)

    d_qcd_np = test_d_qcd.numpy()
    d_top_np = test_d_top.numpy()
    labels_np = test_labels.numpy()
    scores = d_qcd_np - d_top_np  # Higher = more Top-like
    preds = (d_top_np < d_qcd_np).astype(np.int64)

    qcd_mask = labels_np == 0
    top_mask = labels_np == 1

    # --- Panel 1: Loss curves ---
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Accuracy curves ---
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    ax.axhline(y=history.get('test_acc', 0), color='g', linestyle='--',
               label=f"Test: {history.get('test_acc', 0)*100:.1f}%")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Energy distribution ---
    ax = axes[0, 2]
    if qcd_mask.sum() > 0:
        ax.hist(d_qcd_np[qcd_mask], bins=50, alpha=0.5, label='QCD -> d_qcd', color='blue', density=True)
        ax.hist(d_top_np[qcd_mask], bins=50, alpha=0.5, label='QCD -> d_top', color='cyan', density=True)
    if top_mask.sum() > 0:
        ax.hist(d_qcd_np[top_mask], bins=50, alpha=0.5, label='Top -> d_qcd', color='red', density=True)
        ax.hist(d_top_np[top_mask], bins=50, alpha=0.5, label='Top -> d_top', color='orange', density=True)
    ax.set_xlabel('Cosine Distance (1 - similarity)')
    ax.set_ylabel('Density')
    ax.set_title('Energy Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Decision boundary scatter ---
    ax = axes[1, 0]
    if qcd_mask.sum() > 0:
        ax.scatter(d_qcd_np[qcd_mask], d_top_np[qcd_mask],
                   c='blue', alpha=0.3, s=5, label='QCD', edgecolors='none')
    if top_mask.sum() > 0:
        ax.scatter(d_qcd_np[top_mask], d_top_np[top_mask],
                   c='red', alpha=0.3, s=5, label='Top', edgecolors='none')
    lim_max = max(d_qcd_np.max(), d_top_np.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('d_qcd (distance to QCD prototypes)')
    ax.set_ylabel('d_top (distance to Top prototypes)')
    ax.set_title('Decision Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    # --- Panel 5: Prototype activation ---
    ax = axes[1, 1]
    proto_labels_arr = np.array([0] * 10 + [1] * 10)  # 10 QCD + 10 Top
    n_proto = test_similarities.shape[1]

    mean_sims = []
    proto_names = []
    colors = []
    for p in range(n_proto):
        pclass = 'Top' if proto_labels_arr[p] == 1 else 'QCD'
        mean_sim = test_similarities[:, p].mean().item()
        mean_sims.append(mean_sim)
        proto_names.append(f'P{p}\n({pclass})')
        colors.append('red' if proto_labels_arr[p] == 1 else 'blue')

    bars = ax.bar(range(n_proto), mean_sims, color=colors, alpha=0.7)
    ax.set_xticks(range(n_proto))
    ax.set_xticklabels(proto_names, fontsize=7)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Prototype Activations')
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='QCD prototype'),
                       Patch(facecolor='red', alpha=0.7, label='Top prototype')]
    ax.legend(handles=legend_elements, fontsize=8)

    # --- Panel 6: ROC curve ---
    ax = axes[1, 2]
    fpr, tpr, _ = roc_curve(labels_np, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Panel 7: Confusion Matrix ---
    ax = axes[2, 0]
    cm = confusion_matrix(labels_np, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['QCD', 'Top'], yticklabels=['QCD', 'Top'],
                cbar=False, annot_kws={'size': 16})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix\n(Acc={history.get("test_acc", 0)*100:.1f}%)')

    # Add percentages in cells
    cm_sum = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm_sum * 100
            ax.text(j + 0.5, i + 0.65, f'{pct:.1f}%',
                    ha='center', va='center', fontsize=10, color='gray')

    # --- Panel 8: Precision-Recall curve ---
    ax = axes[2, 1]
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision_curve, recall_curve, _ = precision_recall_curve(labels_np, scores)
    avg_prec = average_precision_score(labels_np, scores)
    ax.plot(recall_curve, precision_curve, 'b-', linewidth=2,
            label=f'PR (AP = {avg_prec:.4f})')
    ax.axhline(y=labels_np.mean(), color='k', linestyle='--', alpha=0.5,
               label=f'Baseline ({labels_np.mean():.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Panel 9: Score distribution ---
    ax = axes[2, 2]
    if qcd_mask.sum() > 0:
        ax.hist(scores[qcd_mask], bins=50, alpha=0.5, label='QCD', color='blue', density=True)
    if top_mask.sum() > 0:
        ax.hist(scores[top_mask], bins=50, alpha=0.5, label='Top', color='red', density=True)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Decision boundary')
    ax.set_xlabel('Score (d_qcd - d_top)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Top Tagging model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test with limited events')
    parser.add_argument('--n_events', type=int, default=2000, help='Events to use in --quick mode')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for GPU transfer')
    parser.add_argument('--particle_dim', type=int, default=128, help='Particle embedding dimension')
    parser.add_argument('--jet_dim', type=int, default=128, help='Jet embedding dimension')
    parser.add_argument('--n_prototypes', type=int, default=20, help='Number of prototypes')
    parser.add_argument('--temperature', type=float, default=1.0, help='Prototype temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Limit events for quick test
    max_events = args.n_events if args.quick else None

    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    print("[Progress] Loading train.h5...")
    sys.stdout.flush()
    train_ds = JetTaggingDataset('train.h5', max_events=max_events)
    print(f"[Progress] train.h5 loaded: {len(train_ds)} events")
    sys.stdout.flush()

    print("[Progress] Loading val.h5...")
    sys.stdout.flush()
    val_ds = JetTaggingDataset('val.h5', max_events=max_events)
    print(f"[Progress] val.h5 loaded: {len(val_ds)} events")
    sys.stdout.flush()

    print("[Progress] Loading test.h5...")
    sys.stdout.flush()
    test_ds = JetTaggingDataset('test.h5', max_events=max_events)
    print(f"[Progress] test.h5 loaded: {len(test_ds)} events")
    sys.stdout.flush()

    # Compute class weights for imbalance
    n_total = len(train_ds)
    n_top = int(train_ds.labels_np.sum())
    n_qcd = n_total - n_top
    # weight = n_total / (n_classes * n_class_samples)
    class_weight = torch.tensor([
        n_total / (2 * n_qcd),
        n_total / (2 * n_top),
    ], dtype=torch.float32).to(device)
    print(f"Class weights: QCD={class_weight[0]:.4f}, Top={class_weight[1]:.4f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    model = TopTaggingModel(
        constituent_dim=4,
        n_constituents=200,
        particle_dim=args.particle_dim,
        jet_dim=args.jet_dim,
        n_fastjet_features=6,
        n_prototypes=args.n_prototypes,
        n_classes=2,
        temperature=args.temperature,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'best_val_acc': 0.0,
    }

    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            class_weight=class_weight,
        )

        val_loss, val_acc, _, _, _, _ = evaluate(
            model, val_loader, device,
            class_weight=class_weight,
        )

        scheduler.step(val_acc)

        t1 = time.time()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"time={t1-t0:.1f}s | NEW BEST!")
        else:
            print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"time={t1-t0:.1f}s")

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    model.load_state_dict(torch.load('best_model.pt'))

    test_loss, test_acc, test_d_qcd, test_d_top, test_labels, test_similarities = evaluate(
        model, test_loader, device,
        class_weight=class_weight,
    )

    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test loss: {test_loss:.4f}")

    # Save history
    history['test_acc'] = test_acc
    history['test_loss'] = test_loss

    # ====================================================================
    # Comprehensive Evaluation
    # ====================================================================
    print("\n" + "=" * 60)
    print("Comprehensive Evaluation...")
    print("=" * 60)

    proto_labels_arr = model.prototype_layer.get_prototype_labels().cpu().numpy()
    metrics = comprehensive_evaluation(
        test_labels, test_d_qcd, test_d_top, test_similarities,
        proto_labels_arr=proto_labels_arr,
    )
    print_evaluation_report(metrics)

    # Save metrics to JSON
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                    if k not in ['predictions', 'scores', 'fpr', 'tpr',
                                 'precision_curve', 'recall_curve']}
    history['test_metrics'] = metrics_json
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Print prototype analysis
    print("\n" + "=" * 60)
    print("Prototype Analysis")
    print("=" * 60)

    for i in range(args.n_prototypes):
        class_name = "Top" if proto_labels_arr[i] == 1 else "QCD"
        sim_to_proto = test_similarities[:, i].numpy()
        print(f"  Prototype {i:2d} ({class_name}): "
              f"mean_sim={sim_to_proto.mean():.4f} +/- {sim_to_proto.std():.4f}")

    # Per-class prototype activations
    for cls, cls_name in [(0, 'QCD'), (1, 'Top')]:
        cls_mask = (test_labels.numpy() == cls)
        if cls_mask.sum() == 0:
            continue
        cls_sims = test_similarities[cls_mask].numpy()
        print(f"\n  {cls_name} events - prototype activations:")
        for i in range(args.n_prototypes):
            print(f"    Proto {i:2d}: {cls_sims[:, i].mean():.4f}")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    plot_results(history, test_d_qcd, test_d_top, test_labels, test_similarities)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
