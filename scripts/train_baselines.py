"""
Training Script: EFN Baselines (Linear + CosLinear)
====================================================
Trains and evaluates two ablation baselines:
    A: EFN + Linear + CE  (standard classifier)
    B: EFN + CosLinear + CE  (ArcFace without margin)

Usage:
    # Train Linear baseline (30 epochs)
    python scripts/train_baselines.py --model linear --epochs 30

    # Train CosLinear baseline (30 epochs)
    python scripts/train_baselines.py --model coslinear --epochs 30

    # Train both with 3 seeds each
    python scripts/train_baselines.py --model both --epochs 30 --seeds 42 123 7
"""
import json
import time
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.data.loader import load_awkward, JetTaggingDataset, get_h5_len
from jet_tagging.models.efn_linear import TopTaggingLinear, count_parameters
from jet_tagging.models.efn_coslinear import TopTaggingCosLinear
import gc


def train_epoch(model, loader, optimizer, device, accumulation_steps=8):
    model.train()
    total_loss, total_acc = 0, 0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, (x, y, w, m, _) in enumerate(loader):
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits, _ = model(x, mask=m)
        loss = F.cross_entropy(logits, y)

        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, file_path, device, batch_size=1024):
    model.eval()
    total_len = get_h5_len(file_path)
    events, labels, weights = load_awkward(file_path, max_events=min(total_len, 100000))
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _ = model(x, mask=m)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits).numpy()

    acc = np.mean(all_preds == all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_logits[:, 1])
    except Exception:
        auc = 0.0

    return {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1_score': float(f1),
        'roc_auc': float(auc), 'loss': total_loss / n_batches,
        'n_events': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'coslinear', 'both'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds (e.g., 42 123 7)')
    parser.add_argument('--data_dir', type=str, default='data/top_tagging')
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/baselines')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--particle_dim', type=int, default=128)
    parser.add_argument('--scale', type=float, default=16.0,
                        help='Scale factor (only for coslinear)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models_to_train = ['linear', 'coslinear'] if args.model == 'both' else [args.model]

    for model_name in models_to_train:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Training: {model_name} | Seed: {seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create model
            if model_name == 'linear':
                model = TopTaggingLinear(
                    embedding_dim=args.embedding_dim,
                    particle_dim=args.particle_dim
                ).to(device)
            else:
                model = TopTaggingCosLinear(
                    embedding_dim=args.embedding_dim,
                    particle_dim=args.particle_dim,
                    s=args.scale
                ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            print(f"Params: {count_parameters(model):,}")

            # Training
            train_path = os.path.join(args.data_dir, 'train.h5')
            val_path = os.path.join(args.data_dir, 'val.h5')

            best_val_acc = 0.0
            history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(args.epochs):
                # Load training data (lazy)
                tr_ev, tr_lab, tr_w = load_awkward(
                    train_path, max_events=50000, lazy=True
                )
                tr_ds = JetTaggingDataset(tr_ev, tr_lab, tr_w)
                tr_loader = DataLoader(
                    tr_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True
                )

                t_loss, t_acc = train_epoch(
                    model, tr_loader, optimizer, device,
                    accumulation_steps=args.accumulation_steps
                )

                # Evaluate on validation set
                val_metrics = evaluate(model, val_path, device)

                scheduler.step()

                history['train_loss'].append(t_loss)
                history['train_acc'].append(t_acc)
                history['val_acc'].append(val_metrics['accuracy'])

                is_best = val_metrics['accuracy'] > best_val_acc
                if is_best:
                    best_val_acc = val_metrics['accuracy']
                    ckpt_dir = os.path.join(args.checkpoint_dir, model_name, f'seed_{seed}')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, 'best_model.pt'))

                print(f"Epoch {epoch+1:2d}/{args.epochs}: "
                      f"Train Loss={t_loss:.4f} Acc={t_acc:.4f} | "
                      f"Val Acc={val_metrics['accuracy']:.4f}"
                      f"{' [NEW BEST]' if is_best else ''}")

                # Cleanup
                del tr_loader, tr_ds, tr_ev, tr_lab, tr_w
                gc.collect()
                torch.cuda.empty_cache()

            # Final evaluation on test set
            test_path = os.path.join(args.data_dir, 'test.h5')
            test_metrics = evaluate(model, test_path, device)
            test_metrics['params'] = count_parameters(model)

            # Save results
            out_dir = os.path.join(args.output_dir, model_name, f'seed_{seed}')
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)
            with open(os.path.join(out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            print(f"\nTest: Acc={test_metrics['accuracy']:.4f} "
                  f"Prec={test_metrics['precision']:.4f} "
                  f"Rec={test_metrics['recall']:.4f} "
                  f"F1={test_metrics['f1_score']:.4f} "
                  f"AUC={test_metrics['roc_auc']:.4f}")
            print(f"Saved to {out_dir}")

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
