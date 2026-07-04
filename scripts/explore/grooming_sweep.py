"""
Grooming Parameter Sweep: SoftDrop beta and z_cut
==================================================
Evaluates the model on test set with different grooming parameters:
  beta ∈ {0, 1, 2}
  z_cut ∈ {0.05, 0.1, 0.2}
Reports accuracy for each combination.
Saves plot to experiments/grooming_sweep.png

Note: The model was trained with ungroomed constituents. This sweep tests
how sensitive the model is to grooming choices at inference time.
"""
import os, sys, gc, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel, get_efn_features
from arcjet.data.loader import load_awkward, JetTaggingDataset, load_events_by_indices
from arcjet.utils.device import get_device

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "top_tagging")
TEST_H5 = os.path.join(DATA_DIR, "test.h5")
MODEL_PATH = os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt")

N_EVENTS = 2000
BATCH_SIZE = 256

# SoftDrop parameters to sweep
BETA_VALUES = [0, 1, 2]
Z_CUT_VALUES = [0.05, 0.1, 0.2]


def apply_softdrop_numpy(events_4vec, beta, z_cut):
    """
    Apply SoftDrop grooming to events using FastJet.
    events_4vec: awkward array of Momentum4D
    Returns: list of groomed constituent arrays (N, 4) for each event

    Processes events individually to avoid FastJet multi-event bugs.
    """
    import awkward as ak
    import fastjet

    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 2.0)
    groomed_constituents = []

    for i in range(len(events_4vec)):
        try:
            # Get single event constituents
            ev = events_4vec[i]
            px = ak.values_astype(ev.px, np.float64)
            py = ak.values_astype(ev.py, np.float64)
            pz = ak.values_astype(ev.pz, np.float64)
            E = ak.values_astype(ev.E, np.float64)
            cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})

            cs = fastjet.ClusterSequence(cons, jet_def)

            # Apply SoftDrop grooming
            sd_jets = cs.exclusive_jets_softdrop_grooming(
                njets=1,
                beta=float(beta),
                symmetry_cut=float(z_cut),
                R0=2.0
            )

            if len(sd_jets) > 0:
                sd_cons = sd_jets[0].constituents
                n_cons = len(sd_cons)
                if n_cons > 0:
                    cons_E = ak.to_numpy(sd_cons.E)
                    cons_px = ak.to_numpy(sd_cons.px)
                    cons_py = ak.to_numpy(sd_cons.py)
                    cons_pz = ak.to_numpy(sd_cons.pz)
                    groomed_constituents.append(np.column_stack([cons_E, cons_px, cons_py, cons_pz]))
                else:
                    # Grooming removed all constituents, use original
                    groomed_constituents.append(np.column_stack([
                        ak.to_numpy(ev.E), ak.to_numpy(ev.px),
                        ak.to_numpy(ev.py), ak.to_numpy(ev.pz)
                    ]))
            else:
                groomed_constituents.append(np.column_stack([
                    ak.to_numpy(ev.E), ak.to_numpy(ev.px),
                    ak.to_numpy(ev.py), ak.to_numpy(ev.pz)
                ]))
        except Exception:
            # Fallback: use original constituents
            ev = events_4vec[i]
            groomed_constituents.append(np.column_stack([
                ak.to_numpy(ev.E), ak.to_numpy(ev.px),
                ak.to_numpy(ev.py), ak.to_numpy(ev.pz)
            ]))

    return groomed_constituents


def pad_constituents(groomed_list, max_n=200):
    """Pad groomed constituents to fixed size for batching."""
    batch_size = len(groomed_list)
    padded = np.zeros((batch_size, max_n, 4), dtype=np.float32)
    mask = np.zeros((batch_size, max_n), dtype=np.float32)

    for i, cons in enumerate(groomed_list):
        n = min(len(cons), max_n)
        if n > 0:
            # Sort by energy descending (most energetic first)
            sorted_idx = np.argsort(cons[:, 0])[::-1]
            padded[i, :n] = cons[sorted_idx[:n]]
            mask[i, :n] = 1.0

    return padded, mask


def evaluate_with_grooming(model, events, labels, weights, device, beta, z_cut, batch_size=BATCH_SIZE):
    """Evaluate model with specific grooming parameters."""
    model.eval()

    # Apply grooming to all events
    groomed = apply_softdrop_numpy(events, beta, z_cut)

    # Pad and create batches
    all_preds = []
    all_labels_list = []

    n_events = len(groomed)
    for start in range(0, n_events, batch_size):
        end = min(start + batch_size, n_events)
        batch_groomed = groomed[start:end]
        batch_labels = labels[start:end]

        padded, mask = pad_constituents(batch_groomed)

        x = torch.from_numpy(padded).to(device)
        m = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            logits, sims, emb = model(x, mask=m)
            preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels_list.append(batch_labels)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels_list)

    acc = np.mean(all_preds == all_labels)
    return acc, all_preds, all_labels


def main():
    device, backend_type = get_device()
    print(f"Device: {device}\n")
    print("=" * 60)
    print("GROOMING PARAMETER SWEEP")
    print("=" * 60)
    print(f"  Beta values: {BETA_VALUES}")
    print(f"  z_cut values: {Z_CUT_VALUES}")
    print(f"  Events: {N_EVENTS}")

    # Load model
    print("\nStep 1: Loading model...")
    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  Model loaded from {MODEL_PATH}")

    # Load test data
    print(f"\nStep 2: Loading {N_EVENTS} test events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=N_EVENTS)
    print(f"  Loaded {len(events)} events")

    # Baseline: no grooming (use model directly on ungroomed data)
    print(f"\nStep 3: Baseline (no grooming)...")
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds_base = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            all_preds_base.append(logits.argmax(dim=1).cpu().numpy())
    baseline_acc = np.mean(np.concatenate(all_preds_base) == labels)
    print(f"  Baseline accuracy (no grooming): {baseline_acc:.4f}")

    # Sweep
    print(f"\nStep 4: Running grooming sweep...")
    results = {}
    for beta in BETA_VALUES:
        for z_cut in Z_CUT_VALUES:
            key = f"beta={beta}, z_cut={z_cut}"
            print(f"  Testing {key}...")
            t0 = time.time()
            acc, _, _ = evaluate_with_grooming(model, events, labels, weights, device, beta, z_cut)
            elapsed = time.time() - t0
            results[key] = {"beta": beta, "z_cut": z_cut, "accuracy": float(acc), "time": float(elapsed)}
            print(f"    Accuracy: {acc:.4f} ({elapsed:.1f}s)")
            gc.collect()

    # Print results table
    print(f"\n{'='*60}")
    print("GROOMING SWEEP RESULTS")
    print("=" * 60)
    print(f"  Baseline (no grooming): {baseline_acc:.4f}")
    print(f"\n  {'beta':>6s} {'z_cut':>8s} {'Accuracy':>10s} {'Drop':>10s}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for beta in BETA_VALUES:
        for z_cut in Z_CUT_VALUES:
            key = f"beta={beta}, z_cut={z_cut}"
            acc = results[key]['accuracy']
            drop = baseline_acc - acc
            print(f"  {beta:>6d} {z_cut:>8.2f} {acc:>10.4f} {drop:>+10.4f}")

    # Plot
    print(f"\nStep 5: Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Heatmap of accuracy
    ax = axes[0]
    acc_matrix = np.zeros((len(BETA_VALUES), len(Z_CUT_VALUES)))
    for i, beta in enumerate(BETA_VALUES):
        for j, z_cut in enumerate(Z_CUT_VALUES):
            key = f"beta={beta}, z_cut={z_cut}"
            acc_matrix[i, j] = results[key]['accuracy']

    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=acc_matrix.min() - 0.01,
                   vmax=baseline_acc + 0.005)
    ax.set_xticks(range(len(Z_CUT_VALUES)))
    ax.set_xticklabels([f'{zc:.2f}' for zc in Z_CUT_VALUES])
    ax.set_yticks(range(len(BETA_VALUES)))
    ax.set_yticklabels([f'{b}' for b in BETA_VALUES])
    ax.set_xlabel('z_cut (symmetry cut)')
    ax.set_ylabel('beta (SoftDrop)')
    ax.set_title('Accuracy vs Grooming Parameters')
    plt.colorbar(im, ax=ax, label='Accuracy')

    # Add text annotations
    for i in range(len(BETA_VALUES)):
        for j in range(len(Z_CUT_VALUES)):
            ax.text(j, i, f'{acc_matrix[i, j]:.4f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if acc_matrix[i, j] < (acc_matrix.min() + acc_matrix.max()) / 2 else 'black')

    # Plot 2: Line plot
    ax = axes[1]
    for beta in BETA_VALUES:
        accs = [results[f"beta={beta}, z_cut={zc}"]['accuracy'] for zc in Z_CUT_VALUES]
        ax.plot(Z_CUT_VALUES, accs, 'o-', label=f'beta={beta}', linewidth=2, markersize=8)
    ax.axhline(baseline_acc, color='black', linestyle='--', label=f'No grooming ({baseline_acc:.4f})')
    ax.set_xlabel('z_cut')
    ax.set_ylabel('Accuracy')
    ax.set_title('Grooming Sweep: Accuracy vs z_cut')
    ax.legend()
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "grooming_sweep.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: grooming_sweep.png")

    # Save results
    out = {
        "baseline_accuracy": float(baseline_acc),
        "n_events": N_EVENTS,
        "results": results,
    }
    with open(os.path.join(EXPERIMENTS_DIR, "grooming_sweep_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: grooming_sweep_results.json")

    print(f"\n{'='*60}")
    print("GROOMING SWEEP COMPLETE")
    print("=" * 60)

    del model
    gc.collect()


if __name__ == '__main__':
    main()
