"""
Embedding Dimension Ablation Study
===================================
Trains EFN+ArcFace models with embedding_dim=16, 32, 128 (50 epochs each).
Saves checkpoints to experiments/ablation_dim_{16,32,128}/.
Reports accuracy for each configuration.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel, arcface_loss, count_parameters
from arcjet.data.loader import load_awkward, JetTaggingDataset, get_h5_len
from arcjet.utils.device import get_device, empty_cache

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TRAIN_H5 = os.path.join(DATA_DIR, "train.h5")
VAL_H5 = os.path.join(DATA_DIR, "val.h5")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")

EPOCHS = 50
BATCH_SIZE = 128
ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-3
PARTICLE_DIM = 128
ABLATION_DIMS = [16, 32, 128]
REFERENCE_DIM = 64
REFERENCE_ACC = 0.9175


def train_epoch(model, loader, optimizer, device, scheduler=None):
    model.train()
    total_loss, total_acc, n_batches = 0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (x, y, w, m, _) in enumerate(loader):
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits, sims, _ = model(x, labels=y, mask=m)
        loss = arcface_loss(logits, y) / ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * ACCUMULATION_STEPS
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, h5_path, device, max_events=None, batch_size=1024):
    model.eval()
    total_loss, total_acc, n_batches = 0, 0, 0

    total_len = get_h5_len(h5_path)
    if max_events:
        total_len = min(total_len, max_events)
    chunk_size = 100000
    num_chunks = int(np.ceil(total_len / chunk_size))

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        n_events = min(chunk_size, total_len - start)
        ev, lab, w = load_awkward(h5_path, max_events=n_events, start_event=start, lazy=True)
        ds = JetTaggingDataset(ev, lab, w)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        for x, y, w_b, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, sims, _ = model(x, mask=m)
            loss = arcface_loss(logits, y)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == y).float().mean().item()
            n_batches += 1

        del loader, ds, ev, lab, w
        gc.collect()

    return total_loss / n_batches, total_acc / n_batches


def train_ablation_dim(embedding_dim, device, backend_type):
    """Train a single ablation model."""
    output_dir = os.path.join(EXPERIMENTS_DIR, f"ablation_dim_{embedding_dim}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")

    print(f"\n  Creating model: embedding_dim={embedding_dim}, particle_dim={PARTICLE_DIM}")
    model = TopTaggingModel(
        s=16.0, m=0.0, embedding_dim=embedding_dim, particle_dim=PARTICLE_DIM
    ).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load full training data
    print(f"  Loading training data...")
    train_events, train_labels, train_weights = load_awkward(TRAIN_H5, lazy=True)
    train_ds = JetTaggingDataset(train_events, train_labels, train_weights)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    print(f"  Train events: {len(train_ds):,}")

    # Scheduler
    steps_per_epoch = int(np.ceil(len(train_loader) / ACCUMULATION_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=total_steps)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    warmup_epochs = 10
    final_margin = 0.5

    for epoch in range(EPOCHS):
        current_m = final_margin * min(1.0, (epoch + 1) / warmup_epochs)
        model.arcface_head.set_margin(current_m)

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc = evaluate(model, VAL_H5, device, max_events=50000)
        epoch_time = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        tag = " [NEW BEST]" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}: m={current_m:.2f} | "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Acc={val_acc:.4f}{tag} | {epoch_time:.1f}s")

    # Final test evaluation
    print(f"  Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    test_loss, test_acc = evaluate(model, TEST_H5, device, max_events=100000)
    print(f"  Test accuracy: {test_acc:.4f}")

    # Save history
    history['test_accuracy'] = float(test_acc)
    history['best_val_accuracy'] = float(best_val_acc)
    history['n_parameters'] = n_params
    history['embedding_dim'] = embedding_dim
    with open(os.path.join(output_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return {
        "embedding_dim": embedding_dim,
        "n_parameters": n_params,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "history": history,
    }


def main():
    device, backend_type = get_device()
    print(f"Device: {device}\n")
    print("=" * 60)
    print("EMBEDDING DIMENSION ABLATION STUDY")
    print("=" * 60)
    print(f"  Ablation dims: {ABLATION_DIMS}")
    print(f"  Reference (dim={REFERENCE_DIM}): {REFERENCE_ACC:.4f}")
    print(f"  Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"  Particle dim: {PARTICLE_DIM}")

    all_results = {}

    for dim in ABLATION_DIMS:
        print(f"\n{'='*60}")
        print(f"TRAINING: embedding_dim={dim}")
        print("=" * 60)
        t0 = time.time()
        result = train_ablation_dim(dim, device, backend_type)
        elapsed = time.time() - t0
        result['training_time_seconds'] = elapsed
        all_results[dim] = result
        print(f"\n  dim={dim} complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")
        gc.collect()
        empty_cache(backend_type)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"  {'Dim':>6s} {'Params':>10s} {'Val Acc':>10s} {'Test Acc':>10s} {'Gap vs Ref':>12s} {'Time':>10s}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    print(f"  {REFERENCE_DIM:>6d} {'—':>10s} {'—':>10s} {REFERENCE_ACC:>10.4f} {'(reference)':>12s} {'—':>10s}")
    for dim in ABLATION_DIMS:
        r = all_results[dim]
        gap = r['test_accuracy'] - REFERENCE_ACC
        time_str = f"{r['training_time_seconds']/60:.1f}m"
        print(f"  {dim:>6d} {r['n_parameters']:>10,d} {r['best_val_accuracy']:>10.4f} "
              f"{r['test_accuracy']:>10.4f} {gap:>+12.4f} {time_str:>10s}")

    # Save combined results
    out = {
        "reference": {"embedding_dim": REFERENCE_DIM, "test_accuracy": REFERENCE_ACC},
        "ablations": {str(dim): all_results[dim] for dim in ABLATION_DIMS},
    }
    with open(os.path.join(EXPERIMENTS_DIR, "ablation_dim_results.json"), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: ablation_dim_results.json")

    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
