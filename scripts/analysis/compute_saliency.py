"""
Saliency Analysis (Fixed): Gradient-based Feature Importance
============================================================
Fixes the issue where saliency was looking at padding constituents.
Masks out padding (E=0) before computing gradients.

Reports top 10 most important constituents by saliency magnitude.
Saves plot to ``EXPERIMENTS / 'saliency_fixed.png'`` and results JSON
to ``EXPERIMENTS / 'saliency_fixed_results.json'``.

Examples
--------
Run from the project root::

    python scripts/analysis/compute_saliency.py
"""
import os, sys, gc, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS, MODEL_DIR, CHECKPOINT
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset

N_EVENTS = 5000
BATCH_SIZE = 64


def compute_saliency(model, loader, device, n_events):
    """
    Compute gradient saliency per constituent.

    For each event, compute d(logit_correct) / d(E_i) for each
    constituent *i*.  Padding constituents (E=0) are masked out
    before gradient computation.

    Parameters
    ----------
    model : torch.nn.Module
        The trained TopTaggingModel.
    loader : DataLoader
        DataLoader yielding (x, y, w, m, _) batches.
    device : torch.device
        Device to run computation on.
    n_events : int
        Maximum number of events to process.

    Returns
    -------
    saliency_all : np.ndarray, shape (n_events, N)
        Saliency magnitude per constituent for each event.
    labels_all : np.ndarray, shape (n_events,)
        Ground-truth labels.
    preds_all : np.ndarray, shape (n_events,)
        Model predictions.
    """
    model.eval()
    all_saliencies = []
    all_labels = []
    all_preds = []
    count = 0

    for x, y, w, m, _ in loader:
        if count >= n_events:
            break
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)

        x_masked = x * m.unsqueeze(-1)
        x_masked.requires_grad_(True)

        logits, sims, emb = model(x_masked, labels=y, mask=m)

        pred_classes = logits.argmax(dim=1)
        selected_logits = logits[torch.arange(len(y)), pred_classes]

        grads = torch.autograd.grad(
            selected_logits.sum(),
            x_masked,
            retain_graph=False,
            create_graph=False,
        )[0]

        saliency = grads.abs().sum(dim=-1)
        saliency = saliency * m

        all_saliencies.append(saliency.detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_preds.append(pred_classes.detach().cpu().numpy())
        count += len(y)

    saliency_all = np.concatenate(all_saliencies, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    preds_all = np.concatenate(all_preds, axis=0)

    return saliency_all, labels_all, preds_all


def analyze_saliency(saliencies, labels, preds):
    """
    Analyze saliency patterns across events.

    Computes:
    1. Mean saliency per constituent rank (sorted by importance)
    2. Top 10 most important constituents
    3. Saliency vs energy fraction

    Parameters
    ----------
    saliencies : np.ndarray, shape (n_events, N)
        Saliency per constituent per event.
    labels : np.ndarray, shape (n_events,)
        Ground-truth labels.
    preds : np.ndarray, shape (n_events,)
        Model predictions.

    Returns
    -------
    dict
        Contains keys: mean_rank_saliency, std_rank_saliency,
        mean_by_position, mean_cumulative_fraction, top_10_mean,
        top_10_std.
    """
    n_events = saliencies.shape[0]
    max_constituents = saliencies.shape[1]

    rank_saliency = np.zeros((n_events, max_constituents))
    for i in range(n_events):
        row = saliencies[i]
        nonzero_mask = row > 0
        nonzero_sal = row[nonzero_mask]
        sorted_sal = np.sort(nonzero_sal)[::-1]
        rank_saliency[i, :len(sorted_sal)] = sorted_sal

    mean_rank_saliency = rank_saliency.mean(axis=0)
    std_rank_saliency = rank_saliency.std(axis=0)

    top_10_mean = mean_rank_saliency[:10]
    top_10_std = std_rank_saliency[:10]

    mean_by_position = saliencies.mean(axis=0)

    total_sal = saliencies.sum(axis=1, keepdims=True) + 1e-10
    sorted_sal_matrix = np.sort(saliencies, axis=1)[:, ::-1]
    cumsum_sal = np.cumsum(sorted_sal_matrix, axis=1) / total_sal
    mean_cumsum = cumsum_sal.mean(axis=0)

    return {
        "mean_rank_saliency": mean_rank_saliency,
        "std_rank_saliency": std_rank_saliency,
        "mean_by_position": mean_by_position,
        "mean_cumulative_fraction": mean_cumsum,
        "top_10_mean": top_10_mean,
        "top_10_std": top_10_std,
    }


def plot_saliency(results, save_path):
    """
    Generate saliency analysis plots.

    Creates a 3-panel figure:
    1. Mean saliency by constituent rank (sorted by importance)
    2. Cumulative fraction of total saliency
    3. Top 10 most important constituents (bar chart)

    Parameters
    ----------
    results : dict
        Output from ``analyze_saliency``.
    save_path : Path
        Destination path for the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    ax = axes[0]
    n_show = 50
    ranks = np.arange(n_show)
    mean_sal = results["mean_rank_saliency"][:n_show]
    std_sal = results["std_rank_saliency"][:n_show]
    ax.fill_between(ranks, mean_sal - std_sal, mean_sal + std_sal, alpha=0.3, color='steelblue')
    ax.plot(ranks, mean_sal, 'o-', color='steelblue', markersize=3)
    ax.set_xlabel("Constituent Rank (by saliency)")
    ax.set_ylabel("Mean Saliency (|d logit/d x|)")
    ax.set_title("Saliency by Constituent Rank")
    ax.grid(True, alpha=0.15)
    ax.set_yscale('log')

    ax = axes[1]
    n_show_cum = 100
    ax.plot(np.arange(n_show_cum), results["mean_cumulative_fraction"][:n_show_cum], 'o-', color='darkorange', markersize=3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='80%')
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax.set_xlabel("Number of Top Constituents")
    ax.set_ylabel("Cumulative Fraction of Total Saliency")
    ax.set_title("Saliency Concentration")
    ax.legend()
    ax.grid(True, alpha=0.15)

    ax = axes[2]
    top_10_mean = results["top_10_mean"]
    top_10_std = results["top_10_std"]
    bars = ax.bar(range(10), top_10_mean, yerr=top_10_std, capsize=3,
                  color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Constituent Rank")
    ax.set_ylabel("Mean Saliency")
    ax.set_title("Top 10 Most Important Constituents")
    ax.grid(True, alpha=0.15, axis='y')
    for i, (v, s) in enumerate(zip(top_10_mean, top_10_std)):
        ax.text(i, v + s + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Run saliency analysis: load model, compute gradients, plot and save."""
    device, backend_type = get_device()
    print(f"Device: {device}\n")
    print("=" * 60)
    print("SALIENCY ANALYSIS (FIXED -- PADDING MASKED)")
    print("=" * 60)

    print("\nStep 1: Loading model...")
    sd = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  Model loaded from {CHECKPOINT}")

    test_h5 = DATA_DIR / 'test.h5'
    print(f"\nStep 2: Loading {N_EVENTS} test events...")
    events, labels, weights = load_awkward(test_h5, max_events=N_EVENTS)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Loaded {len(ds)} events")

    print(f"\nStep 3: Computing gradient saliency (padding masked)...")
    t0 = time.time()
    saliencies, labels_arr, preds_arr = compute_saliency(model, loader, device, N_EVENTS)
    acc = np.mean(preds_arr == labels_arr)
    print(f"  Saliency computed in {time.time()-t0:.1f}s")
    print(f"  Accuracy on these events: {acc:.4f}")
    print(f"  Saliency shape: {saliencies.shape}")

    print(f"\nStep 4: Analyzing saliency patterns...")
    results = analyze_saliency(saliencies, labels_arr, preds_arr)

    print(f"\n  Top 10 Most Important Constituents (by saliency rank):")
    print(f"  {'Rank':>6s} {'Mean Saliency':>15s} {'Std Saliency':>15s}")
    for i in range(10):
        print(f"  {i:>6d} {results['top_10_mean'][i]:>15.6f} {results['top_10_std'][i]:>15.6f}")

    cumfrac = results["mean_cumulative_fraction"]
    n_50 = np.argmax(cumfrac >= 0.5) + 1
    n_80 = np.argmax(cumfrac >= 0.8) + 1
    n_90 = np.argmax(cumfrac >= 0.9) + 1
    print(f"\n  Saliency Concentration:")
    print(f"    Top {n_50} constituents capture 50% of total saliency")
    print(f"    Top {n_80} constituents capture 80% of total saliency")
    print(f"    Top {n_90} constituents capture 90% of total saliency")

    print(f"\nStep 5: Generating plots...")
    plot_saliency(results, EXPERIMENTS / "saliency_fixed.png")

    out = {
        "n_events": int(N_EVENTS),
        "accuracy": float(acc),
        "top_10_mean_saliency": results["top_10_mean"].tolist(),
        "top_10_std_saliency": results["top_10_std"].tolist(),
        "n_for_50pct": int(n_50),
        "n_for_80pct": int(n_80),
        "n_for_90pct": int(n_90),
    }
    json_path = EXPERIMENTS / "saliency_fixed_results.json"
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*60}")
    print("SALIENCY ANALYSIS COMPLETE")
    print("=" * 60)

    del model
    gc.collect()


if __name__ == '__main__':
    main()
