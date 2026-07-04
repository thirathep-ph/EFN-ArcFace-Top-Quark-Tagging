"""
PGD Adversarial Attack: Projected Gradient Descent
===================================================
Stronger than FGSM — multi-step iterative attack.
Parameters: 10 steps, epsilon=0.01, alpha=0.002
Reports clean accuracy, adversarial accuracy, and accuracy drop.
Saves plot to experiments/pgd_attack.png
"""
import os, sys, gc, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
MODEL_PATH = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")

N_EVENTS = 5000
BATCH_SIZE = 128
EPSILON = 0.01
ALPHA = 0.002
N_STEPS = 10


def pgd_attack(model, x, y, mask, epsilon, alpha, n_steps):
    """
    PGD attack on the constituents.
    x: (B, N, 4) — [E, px, py, pz]
    We perturb only the energy channel (index 0) since it's the most physically meaningful.
    """
    x_adv = x.clone().detach()

    for step in range(n_steps):
        x_adv.requires_grad_(True)

        logits, sims, emb = model(x_adv, labels=y, mask=mask)
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # Gradient ascent on the loss (to maximize loss = minimize accuracy)
        # Only perturb energy channel
        perturbation = torch.zeros_like(x_adv)
        perturbation[:, :, 0] = alpha * grad[:, :, 0].sign()

        x_adv = x_adv + perturbation

        # Project back to epsilon ball (L-infinity)
        delta = x_adv - x
        delta[:, :, 0] = torch.clamp(delta[:, :, 0], -epsilon, epsilon)
        x_adv = x + delta

        # Ensure energy stays non-negative
        x_adv[:, :, 0] = torch.clamp(x_adv[:, :, 0], min=0.0)

        # Re-apply mask (don't perturb padding)
        x_adv = x_adv * mask.unsqueeze(-1).float() + x * (1 - mask.unsqueeze(-1).float())

        x_adv = x_adv.detach()

    return x_adv


def evaluate_clean(model, loader, device, n_events):
    """Evaluate on clean data."""
    model.eval()
    all_preds = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            if count >= n_events:
                break
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            count += len(y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def evaluate_pgd(model, loader, device, n_events, epsilon, alpha, n_steps):
    """Evaluate on PGD-perturbed data."""
    model.eval()
    all_preds = []
    all_labels = []
    count = 0

    for x, y, w, m, _ in loader:
        if count >= n_events:
            break
        x, y, m = x.to(device), y.to(device), m.to(device)

        x_adv = pgd_attack(model, x, y, m, epsilon, alpha, n_steps)

        with torch.no_grad():
            logits, _, _ = model(x_adv, mask=m)
            preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        count += len(y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def main():
    device, backend_type = get_device()
    print(f"Device: {device}\n")
    print("=" * 60)
    print("PGD ADVERSARIAL ATTACK")
    print("=" * 60)
    print(f"  Parameters: {N_STEPS} steps, epsilon={EPSILON}, alpha={ALPHA}")

    # Load model
    print("\nStep 1: Loading model...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  Model loaded from {MODEL_PATH}")

    # Load data
    print(f"\nStep 2: Loading {N_EVENTS} test events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Loaded {len(ds)} events")

    # Clean evaluation
    print(f"\nStep 3: Evaluating on clean data...")
    t0 = time.time()
    clean_preds, clean_labels = evaluate_clean(model, loader, device, N_EVENTS)
    clean_acc = np.mean(clean_preds == clean_labels)
    print(f"  Clean accuracy: {clean_acc:.4f} ({time.time()-t0:.1f}s)")

    # PGD attack
    print(f"\nStep 4: Running PGD attack ({N_STEPS} steps)...")
    t0 = time.time()
    pgd_preds, pgd_labels = evaluate_pgd(model, loader, device, N_EVENTS, EPSILON, ALPHA, N_STEPS)
    pgd_acc = np.mean(pgd_preds == pgd_labels)
    print(f"  PGD accuracy: {pgd_acc:.4f} ({time.time()-t0:.1f}s)")

    # Results
    drop = clean_acc - pgd_acc
    flip_rate = np.mean(clean_preds != pgd_preds)

    print(f"\n{'='*60}")
    print("PGD ATTACK RESULTS")
    print("=" * 60)
    print(f"  Clean accuracy:        {clean_acc:.4f}")
    print(f"  Adversarial accuracy:  {pgd_acc:.4f}")
    print(f"  Accuracy drop:         {drop:.4f}")
    print(f"  Prediction flip rate:  {flip_rate:.4f}")

    # Per-class breakdown
    for cls, name in [(0, "QCD"), (1, "Top")]:
        mask = clean_labels == cls
        clean_cls_acc = np.mean(clean_preds[mask] == clean_labels[mask])
        pgd_cls_acc = np.mean(pgd_preds[mask] == pgd_labels[mask])
        print(f"  {name:5s} — Clean: {clean_cls_acc:.4f}, PGD: {pgd_cls_acc:.4f}, Drop: {clean_cls_acc - pgd_cls_acc:.4f}")

    # Also test with different epsilon values
    print(f"\nStep 5: Epsilon sweep...")
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
    sweep_results = []
    for eps in epsilons:
        print(f"  Testing epsilon={eps:.3f}...")
        adv_preds, adv_labels = evaluate_pgd(model, loader, device, N_EVENTS, eps, eps / N_STEPS, N_STEPS)
        adv_acc = np.mean(adv_preds == adv_labels)
        sweep_results.append({"epsilon": eps, "accuracy": float(adv_acc)})
        print(f"    Accuracy: {adv_acc:.4f}")

    # Plot
    print(f"\nStep 6: Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Clean vs PGD comparison
    ax = axes[0]
    categories = ['Clean', 'PGD\n(eps=0.01)']
    accuracies = [clean_acc, pgd_acc]
    colors = ['steelblue', 'crimson']
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Clean vs PGD Adversarial Accuracy')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.15, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    # Add drop annotation
    ax.annotate(f'Drop: {drop:.4f}', xy=(0.5, (clean_acc + pgd_acc) / 2),
                xytext=(0.5, (clean_acc + pgd_acc) / 2 - 0.15),
                ha='center', fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Plot 2: Epsilon sweep
    ax = axes[1]
    eps_vals = [r['epsilon'] for r in sweep_results]
    acc_vals = [r['accuracy'] for r in sweep_results]
    ax.plot(eps_vals, acc_vals, 'o-', color='crimson', linewidth=2, markersize=8)
    ax.axhline(clean_acc, color='steelblue', linestyle='--', label=f'Clean ({clean_acc:.4f})')
    ax.set_xlabel('Epsilon (perturbation magnitude)')
    ax.set_ylabel('Adversarial Accuracy')
    ax.set_title('PGD Robustness vs Perturbation Strength')
    ax.legend()
    ax.grid(True, alpha=0.15)
    for eps, acc in zip(eps_vals, acc_vals):
        ax.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "pgd_attack.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pgd_attack.png")

    # Save results
    import json
    out = {
        "n_events": N_EVENTS,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "n_steps": N_STEPS,
        "clean_accuracy": float(clean_acc),
        "pgd_accuracy": float(pgd_acc),
        "accuracy_drop": float(drop),
        "flip_rate": float(flip_rate),
        "epsilon_sweep": sweep_results,
    }
    with open(os.path.join(EXPERIMENTS_DIR, "pgd_attack_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: pgd_attack_results.json")

    print(f"\n{'='*60}")
    print("PGD ATTACK COMPLETE")
    print("=" * 60)

    del model
    gc.collect()


if __name__ == '__main__':
    main()
