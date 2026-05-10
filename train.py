"""
Training Pipeline for Top Tagging with DeepSets + Prototype Layer
=================================================================
Trains the model on train.h5, validates on val.h5, tests on test.h5.

Usage:
    python train.py                          # Full training
    python train.py --quick                   # Quick test (1000 events)
    python train.py --epochs 20 --batch_size 256

Output:
    - best_model.pt:  Best model checkpoint
    - training_history.json:  Loss and accuracy history
    - prototype_analysis.png:  Prototype similarity visualization
"""
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import TopTaggingModel, prototype_loss, count_parameters
from fastjet_features import compute_features
from to_awkward import load_awkward


# ============================================================================
# Dataset Wrapper for PyTorch
# ============================================================================

class JetTaggingDataset(Dataset):
    """
    PyTorch Dataset that loads H5 data and computes FastJet features on-the-fly.
    
    For efficiency, we pre-compute FastJet features and cache them.
    """
    def __init__(self, h5_path, max_events=None, precompute_fastjet=True):
        self.h5_path = h5_path
        
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
        
        # Build mask for padding (all valid since we trimmed)
        # After trim_padding, all constituents are valid
        self.mask = None  # Will be built per-batch from constituent data
        
        n_top = int(self.labels.sum())
        n_qcd = self.n - n_top
        print(f"  {self.n:,} events ({n_top:,} Top, {n_qcd:,} QCD)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
        Returns:
            constituents: (200, 4) numpy array - (E, px, py, pz)
            fastjet_features: (6,) numpy array - [mass, pT, eta, tau21, tau32, D2]
            label: int - 0=QCD, 1=Top
            weight: float
            mask: (200,) bool - True for valid particles
        """
        # Get constituents from awkward array
        evt = self.events[idx]
        n_particles = len(evt)
        
        # Build padded constituent array
        constituents = np.zeros((200, 4), dtype=np.float32)
        mask = np.zeros(200, dtype=bool)
        
        for i in range(min(n_particles, 200)):
            p = evt[i]
            constituents[i, 0] = float(p.E)
            constituents[i, 1] = float(p.px)
            constituents[i, 2] = float(p.py)
            constituents[i, 3] = float(p.pz)
            mask[i] = True
        
        # FastJet features
        fj = self.fastjet_features[idx] if self.fastjet_features is not None else np.zeros(6, dtype=np.float32)
        
        label = int(self.labels[idx])
        weight = float(self.weights[idx])
        
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

def train_epoch(model, loader, optimizer, device, lambda_clf=1.0, lambda_clst=0.1, lambda_sep=0.1):
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
        
        logits, similarities, jet_emb = model(constituents, fj, mask)
        
        loss, components = prototype_loss(
            logits, labels, similarities, model.prototype_layer,
            lambda_clf=lambda_clf, lambda_clst=lambda_clst, lambda_sep=lambda_sep,
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, lambda_clf=1.0, lambda_clst=0.1, lambda_sep=0.1):
    """Evaluate model. Returns loss, accuracy, and all predictions."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_logits = []
    all_labels = []
    all_similarities = []
    
    for constituents, fj, labels, weights, mask in loader:
        constituents = constituents.to(device)
        fj = fj.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        logits, similarities, jet_emb = model(constituents, fj, mask)
        
        loss, components = prototype_loss(
            logits, labels, similarities, model.prototype_layer,
            lambda_clf=lambda_clf, lambda_clst=lambda_clst, lambda_sep=lambda_sep,
        )
        
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += len(labels)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_similarities.append(similarities.cpu())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_similarities = torch.cat(all_similarities)
    
    return (total_loss / total_samples, 
            total_correct / total_samples,
            all_logits, all_labels, all_similarities)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Top Tagging model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--quick', action='store_true', help='Quick test with 1000 events')
    parser.add_argument('--lambda_clf', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--lambda_clst', type=float, default=0.1, help='Clustering loss weight')
    parser.add_argument('--lambda_sep', type=float, default=0.1, help='Separation loss weight')
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
    max_events = 1000 if args.quick else None
    
    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    train_ds = JetTaggingDataset('train.h5', max_events=max_events)
    val_ds = JetTaggingDataset('val.h5', max_events=max_events)
    test_ds = JetTaggingDataset('test.h5', max_events=max_events)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
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
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
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
            lambda_clf=args.lambda_clf,
            lambda_clst=args.lambda_clst,
            lambda_sep=args.lambda_sep,
        )
        
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, device,
            lambda_clf=args.lambda_clf,
            lambda_clst=args.lambda_clst,
            lambda_sep=args.lambda_sep,
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
    
    test_loss, test_acc, test_logits, test_labels, test_similarities = evaluate(
        model, test_loader, device,
        lambda_clf=args.lambda_clf,
        lambda_clst=args.lambda_clst,
        lambda_sep=args.lambda_sep,
    )
    
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save history
    history['test_acc'] = test_acc
    history['test_loss'] = test_loss
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Print prototype analysis
    print("\n" + "=" * 60)
    print("Prototype Analysis")
    print("=" * 60)
    
    proto_labels = model.prototype_layer.get_prototype_labels().cpu().numpy()
    proto_vectors = model.prototype_layer.prototypes.detach().cpu().numpy()
    
    for i in range(args.n_prototypes):
        class_name = "Top" if proto_labels[i] == 1 else "QCD"
        # Average similarity of test inputs to this prototype
        sim_to_proto = test_similarities[:, i].numpy()
        print(f"  Prototype {i:2d} ({class_name}): "
              f"mean_sim={sim_to_proto.mean():.4f} ± {sim_to_proto.std():.4f}")
    
    # Per-class prototype activations
    for cls, cls_name in [(0, 'QCD'), (1, 'Top')]:
        cls_mask = (test_labels.numpy() == cls)
        if cls_mask.sum() == 0:
            continue
        cls_sims = test_similarities[cls_mask].numpy()
        print(f"\n  {cls_name} events - prototype activations:")
        for i in range(args.n_prototypes):
            print(f"    Proto {i:2d}: {cls_sims[:, i].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
