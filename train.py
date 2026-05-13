"""
Training Pipeline for EFN + ArcFace Prototype Layer
===================================================
Ante-hoc interpretability: FastJet is used ONLY for interpreting
learned prototypes after training.
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

from model import TopTaggingModel, arcface_loss, count_parameters
from fastjet_features import compute_features
from to_awkward import load_awkward


# ============================================================================
# Dataset Wrapper for PyTorch (EFN Version)
# ============================================================================

class JetTaggingDataset(Dataset):
    """
    Optimized EFN Dataset. Pre-computes padded constituent arrays.
    """
    def __init__(self, h5_path, max_events=None, max_constituents=200):
        self.h5_path = h5_path
        self.max_constituents = max_constituents

        print(f"Loading {h5_path}...")
        self.events, self.labels, self.weights = load_awkward(
            h5_path, max_events=max_events, trim_padding=True
        )
        self.n = len(self.events)

        print(f"  Pre-computing padded constituent arrays ({self.n} events)...")
        t0 = time.time()

        counts = ak.num(self.events)
        self.counts_np = ak.to_numpy(counts).astype(np.int32)

        # Flatten for fast processing
        E_flat = ak.to_numpy(ak.flatten(self.events.E)).astype(np.float32)
        px_flat = ak.to_numpy(ak.flatten(self.events.px)).astype(np.float32)
        py_flat = ak.to_numpy(ak.flatten(self.events.py)).astype(np.float32)
        pz_flat = ak.to_numpy(ak.flatten(self.events.pz)).astype(np.float32)

        self.constituents = np.zeros((self.n, max_constituents, 4), dtype=np.float32)
        self.mask = np.zeros((self.n, max_constituents), dtype=bool)

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

        self.labels_np = np.asarray(self.labels, dtype=np.int64)
        self.weights_np = np.asarray(self.weights, dtype=np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.constituents[idx], 
                self.labels_np[idx], 
                self.weights_np[idx], 
                self.mask[idx],
                idx)


def collate_fn(batch):
    constituents = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    mask = torch.from_numpy(np.stack([b[3] for b in batch])).bool()
    indices = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return constituents, labels, weights, mask, indices


# ============================================================================
# Training & Evaluation Loops
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for constituents, labels, weights, mask, _ in loader:
        constituents, labels, mask = constituents.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()
        # Training: ArcFace margin applied
        logits, sims, emb = model(constituents, labels=labels, mask=mask)
        loss = arcface_loss(logits, labels)
        
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_sims = []
    all_labels = []
    all_indices = []

    for constituents, labels, weights, mask, indices in loader:
        constituents, labels, mask = constituents.to(device), labels.to(device), mask.to(device)

        # Eval: No margin
        logits, similarities, emb = model(constituents, mask=mask)
        loss = arcface_loss(logits, labels)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        all_sims.append(similarities.cpu())
        all_labels.append(labels.cpu())
        all_indices.append(indices.cpu())

    return (total_loss / total_samples, 
            total_correct / total_samples,
            torch.cat(all_sims), 
            torch.cat(all_labels),
            torch.cat(all_indices))


# ============================================================================
# Evaluation & Plotting
# ============================================================================

def comprehensive_evaluation(test_labels, test_sims):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_curve, auc, 
                                 precision_recall_curve, average_precision_score)
    
    labels_np = test_labels.numpy()
    sims_np = test_sims.numpy()
    
    n_qcd = sims_np.shape[1] // 2
    sim_qcd = sims_np[:, :n_qcd].max(axis=1)
    sim_top = sims_np[:, n_qcd:].max(axis=1)
    
    preds = (sim_top > sim_qcd).astype(np.int64)
    scores = sim_top - sim_qcd
    
    cm = confusion_matrix(labels_np, preds)
    acc = accuracy_score(labels_np, preds)
    fpr, tpr, _ = roc_curve(labels_np, scores)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels_np, scores)
    avg_prec = average_precision_score(labels_np, scores)
    
    return {
        'accuracy': acc, 'cm': cm, 'roc_auc': roc_auc, 'avg_prec': avg_prec,
        'fpr': fpr, 'tpr': tpr, 'precision_curve': precision_curve, 'recall_curve': recall_curve,
        'scores': scores, 'sim_qcd': sim_qcd, 'sim_top': sim_top, 'preds': preds
    }


def plot_results(history, metrics, test_sims, save_path='training_results_efn.png'):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('EFN + ArcFace Prototype Model (IRC-Safe xAI)', fontsize=16, fontweight='bold', y=1.02)

    epochs = range(1, len(history['train_loss']) + 1)
    labels_np = metrics['labels_np']
    qcd_mask = labels_np == 0
    top_mask = labels_np == 1

    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss (EFN)')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['train_acc'], label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val')
    axes[0, 1].set_title(f'Accuracy (Test: {metrics["accuracy"]*100:.2f}%)')
    axes[0, 1].legend()

    ax = axes[0, 2]
    bins = np.linspace(-1, 1, 50)
    ax.hist(metrics['sim_qcd'][qcd_mask], bins=bins, alpha=0.5, label='QCD Sim', color='blue', density=True)
    ax.hist(metrics['sim_top'][top_mask], bins=bins, alpha=0.5, label='Top Sim', color='red', density=True)
    ax.set_title('Prototype Similarities')
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(metrics['sim_qcd'][qcd_mask], metrics['sim_top'][qcd_mask], c='blue', alpha=0.1, s=2)
    ax.scatter(metrics['sim_qcd'][top_mask], metrics['sim_top'][top_mask], c='red', alpha=0.1, s=2)
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    ax.set_title('Similarity Space')

    ax = axes[1, 1]
    mean_sims = test_sims.mean(dim=0).numpy()
    n_proto = len(mean_sims)
    ax.bar(range(n_proto), mean_sims, color=['blue']*10 + ['red']*10, alpha=0.7)
    ax.set_title('Mean Prototype Activations')

    ax = axes[1, 2]
    ax.plot(metrics['fpr'], metrics['tpr'], color='green', label=f'AUC={metrics["roc_auc"]:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title('ROC Curve (EFN)')
    ax.legend()

    ax = axes[2, 0]
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Confusion Matrix')

    ax = axes[2, 1]
    ax.plot(metrics['recall_curve'], metrics['precision_curve'], color='purple', label=f'AP={metrics["avg_prec"]:.4f}')
    ax.set_title('PR Curve')
    ax.legend()

    ax = axes[2, 2]
    ax.hist(metrics['scores'][qcd_mask], bins=50, alpha=0.5, label='QCD', color='blue', density=True)
    ax.hist(metrics['scores'][top_mask], bins=50, alpha=0.5, label='Top', color='red', density=True)
    ax.set_title('Decision Score (EFN)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved EFN summary to {save_path}")


# ============================================================================
# Interpretation
# ============================================================================

def interpret_prototypes(model, loader, dataset, device, n_prototypes=20):
    print("\n" + "=" * 60)
    print("EFN PROTOTYPE INTERPRETATION")
    print("=" * 60)
    model.eval()
    _, _, all_sims, all_labels, all_indices = evaluate(model, loader, device)
    best_match_indices = all_sims.argmax(dim=0)
    
    proto_physics_data = []
    for i in range(n_prototypes):
        match_idx = best_match_indices[i].item()
        real_idx = all_indices[match_idx].item()
        event_p4 = dataset.events[real_idx]
        feats = compute_features(ak.Array([event_p4]))[0]
        
        proto_class = "Top" if i >= (n_prototypes // 2) else "QCD"
        print(f"Proto {i:2d} ({proto_class}): Match #{real_idx}, Mass={feats[0]:.2f}, pT={feats[1]:.2f}, tau32={feats[4]:.3f}")
        
        proto_physics_data.append({
            'proto_id': i, 'proto_class': proto_class, 'event_idx': real_idx, 'features': feats.tolist()
        })
    return proto_physics_data, all_labels, all_sims


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_events = 5000 if args.quick else None

    train_ds = JetTaggingDataset('train.h5', max_events=max_events)
    val_ds = JetTaggingDataset('val.h5', max_events=max_events)
    test_ds = JetTaggingDataset('test.h5', max_events=max_events)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1024, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1024, collate_fn=collate_fn)

    model = TopTaggingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nTraining EFN + ArcFace (Params: {count_parameters(model):,})")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_acc, _, _, _ = evaluate(model, val_loader, device)
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)

        print(f"Epoch {epoch+1:2d}: Train Loss={t_loss:.4f} Acc={t_acc:.4f} | Val Loss={v_loss:.4f} Acc={v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), 'best_model_efn.pt')

    model.load_state_dict(torch.load('best_model_efn.pt'))
    proto_data, test_labels, test_sims = interpret_prototypes(model, test_loader, test_ds, device)
    eval_metrics = comprehensive_evaluation(test_labels, test_sims)
    eval_metrics['labels_np'] = test_labels.numpy()
    plot_results(history, eval_metrics, test_sims)
    
    with open('efn_results.json', 'w') as f:
        json.dump({'history': history, 'prototypes': proto_data}, f, indent=2)

    print("\nTraining Complete: EFN + ArcFace ready.")

if __name__ == '__main__':
    main()
