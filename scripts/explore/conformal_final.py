"""
Conformal Prediction Analysis (Final Version)
==============================================
Clean implementation that handles all model types correctly.
"""
import os, sys, json, time, gc
import numpy as np
import torch
import torch.nn.functional as F
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
VAL_H5 = os.path.join(DATA_DIR, "val.h5")
RESULTS_PATH = os.path.join(EXPERIMENTS_DIR, "conformal_results.json")


def load_model(path, device):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'arcface_head.class_centers' in sd:
        model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    elif 'classifier.weight' in sd:
        from arcjet.models.efn_linear import TopTaggingLinear
        model = TopTaggingLinear(embedding_dim=64, particle_dim=128).to(device)
    else:
        from arcjet.models.efn_coslinear import TopTaggingCosLinear
        model = TopTaggingCosLinear(embedding_dim=64, particle_dim=128, s=16.0).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def predict(model, h5_path, device, max_events=50000):
    events, labels, weights = load_awkward(h5_path, max_events=max_events)
    ds = JetTaggingDataset(events, labels, weights)
    loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            # Handle different model output formats
            if isinstance(out, (tuple, list)):
                logits = out[0]  # ArcFace returns (logits, sims, emb)
            else:
                logits = out     # Linear/CosLinear returns logits only
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    
    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = logits.argmax(axis=1)
    return probs, preds, labels


def main():
    device, _ = get_device()
    print(f"Device: {device}\n")
    
    configs = {
        "arcface_s16_m05": os.path.join(EXPERIMENTS_DIR, "robustness_s16_m05", "checkpoint.pt"),
        "arcface_s8_m00": os.path.join(EXPERIMENTS_DIR, "robustness_s8_m00", "checkpoint.pt"),
        "arcface_s32_m00": os.path.join(EXPERIMENTS_DIR, "robustness_s32_m00", "checkpoint.pt"),
        "linear_seed42": os.path.join(EXPERIMENTS_DIR, "baselines_linear_seed_42", "checkpoint.pt"),
        "coslinear_seed42": os.path.join(EXPERIMENTS_DIR, "baselines_coslinear_seed_42", "checkpoint.pt"),
    }
    
    # Load predictions
    print("Loading predictions...")
    val_probs, val_labels = {}, {}
    test_probs, test_preds, test_labels = {}, {}, {}
    
    for name, path in configs.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}"); continue
        
        t0 = time.time()
        model = load_model(path, device)
        
        vp, _, vl = predict(model, VAL_H5, device, 20000)
        tp, tpr, tl = predict(model, TEST_H5, device, 50000)
        
        val_probs[name], val_labels[name] = vp, vl
        test_probs[name], test_preds[name], test_labels[name] = tp, tpr, tl
        
        acc = np.mean(tpr == tl)
        print(f"  {name}: Acc={acc:.4f} ({time.time()-t0:.0f}s)")
        
        del model
        gc.collect()
    
    # Conformal calibration
    print(f"\nConformal calibration (alpha=0.1)...")
    alpha = 0.1
    results = {}
    
    for name in test_probs:
        t0 = time.time()
        
        # Calibrate on val set
        scores = 1.0 - val_probs[name][np.arange(len(val_labels[name])), val_labels[name]]
        q_hat = np.quantile(scores, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        
        # Predict on test set
        pred_sets = []
        for i in range(len(test_probs[name])):
            classes = np.where(1.0 - test_probs[name][i] < q_hat)[0]
            pred_sets.append(classes)
        
        # Metrics
        covered = sum(1 for s, y in zip(pred_sets, test_labels[name]) if y in s)
        coverage = covered / len(test_labels[name])
        set_sizes = np.array([len(s) for s in pred_sets])
        acc = np.mean(test_preds[name] == test_labels[name])
        
        results[name] = {
            "accuracy": float(acc),
            "q_hat": float(q_hat),
            "target_coverage": 1 - alpha,
            "empirical_coverage": float(coverage),
            "mean_set_size": float(np.mean(set_sizes)),
            "frac_size_1": float(np.mean(set_sizes == 1)),
        }
        
        print(f"  {name}: Coverage={coverage:.4f}, MeanSize={np.mean(set_sizes):.2f} ({time.time()-t0:.0f}s)")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Model Comparison")
    print("=" * 60)
    print(f"  {'Model':20s} {'Acc':>8s} {'Cov':>8s} {'Size':>8s} {'%Def':>8s}")
    for name in results:
        r = results[name]
        print(f"  {name:20s} {r['accuracy']:>8.4f} {r['empirical_coverage']:>8.4f} "
              f"{r['mean_set_size']:>8.2f} {r['frac_size_1']:>7.1%}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    
    ax = axes[0]
    for name in names:
        ax.scatter(results[name]["accuracy"], results[name]["empirical_coverage"], s=100, label=name[:15])
    ax.axhline(0.9, color='red', ls='--', label='Target 0.9')
    ax.set_xlabel('Accuracy'); ax.set_ylabel('Coverage')
    ax.set_title('Conformal Prediction'); ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
    
    ax = axes[1]
    for name in names:
        ax.scatter(results[name]["accuracy"], results[name]["mean_set_size"], s=100, label=name[:15])
    ax.set_xlabel('Accuracy'); ax.set_ylabel('Mean Set Size')
    ax.set_title('Efficiency'); ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "conformal_analysis.png"), dpi=150)
    plt.close()
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {RESULTS_PATH}")
    print("[COMPLETE]")


if __name__ == '__main__':
    main()
