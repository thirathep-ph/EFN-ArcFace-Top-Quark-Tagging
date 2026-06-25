"""
Evaluate all 9 robustness checkpoints and compile results table.

Usage:
    python scripts/eval_robustness.py

Output:
    experiments/robustness_results.json
    experiments/robustness_results_table.md
"""
import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcjet.models.efn_arcface import TopTaggingModel
from arcjet.data.loader import load_awkward, JetTaggingDataset
from arcjet.utils.device import get_device

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
TEST_H5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "top_tagging", "test.h5")

ROBUSTNESS_CONFIGS = [
    (8, 0.0), (8, 0.3), (8, 0.5),
    (16, 0.0), (16, 0.3), (16, 0.5),
    (32, 0.0), (32, 0.3), (32, 0.5),
]


def evaluate_checkpoint(model_path, device, max_events=404000):
    """Evaluate a single checkpoint on test set."""
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    events, labels, weights = load_awkward(TEST_H5, max_events=max_events)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, sims, _ = model(x, mask=m)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)

    acc = np.mean(all_preds == all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    # ROC AUC
    try:
        auc = roc_auc_score(all_labels, all_logits[:, 1])
    except Exception:
        auc = 0.0

    # Confusion matrix
    tp = int(np.sum((all_preds == 1) & (all_labels == 1)))
    fp = int(np.sum((all_preds == 1) & (all_labels == 0)))
    fn = int(np.sum((all_preds == 0) & (all_labels == 1)))
    tn = int(np.sum((all_preds == 0) & (all_labels == 0)))

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n_events': len(all_labels),
    }


def main():
    device, backend_type = get_device()
    print(f"Device: {device}")
    print(f"Test set: {TEST_H5}")
    print()

    results = []

    for s, m in ROBUSTNESS_CONFIGS:
        m_str = str(m).replace(".", "0")
        exp_dir = os.path.join(EXPERIMENTS_DIR, f"robustness_s{s}_m{m_str}")
        ckpt = os.path.join(exp_dir, "checkpoint.pt")

        if not os.path.exists(ckpt):
            print(f"  [SKIP] s={s}, m={m} - checkpoint not found at {ckpt}")
            continue

        print(f"  [EVAL] s={s}, m={m} ...")
        t0 = time.time()
        res = evaluate_checkpoint(ckpt, device)
        elapsed = time.time() - t0

        res['scale'] = s
        res['margin'] = m
        res['time_seconds'] = elapsed
        results.append(res)

        print(f"    Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}  ({elapsed:.0f}s)")

    # Sort by scale then margin
    results.sort(key=lambda x: (x['scale'], x['margin']))

    # Save JSON
    out_json = os.path.join(EXPERIMENTS_DIR, "robustness_results.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_json}")

    # Generate Markdown table
    out_md = os.path.join(EXPERIMENTS_DIR, "robustness_results_table.md")
    with open(out_md, 'w') as f:
        f.write("# Robustness Analysis Results\n\n")
        f.write("| Scale (s) | Margin (m) | Accuracy | Precision | Recall | F1 | ROC AUC | TP | FP | FN | TN |\n")
        f.write("|:---------:|:----------:|:--------:|:---------:|:------:|:--:|:-------:|:--:|:--:|:--:|:--:|\n")
        for r in results:
            f.write(f"| {r['scale']} | {r['margin']:.1f} | {r['accuracy']:.4f} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                    f"{r['roc_auc']:.4f} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | {r['tn']:,} |\n")

        # Summary stats
        accs = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]
        aucs = [r['roc_auc'] for r in results]

        f.write(f"\n## Summary\n\n")
        f.write(f"| Metric | Mean | Std | Min | Max |\n")
        f.write(f"|:-------|:----:|:---:|:---:|:---:|\n")
        f.write(f"| Accuracy | {np.mean(accs):.4f} | {np.std(accs):.4f} | {np.min(accs):.4f} | {np.max(accs):.4f} |\n")
        f.write(f"| F1 | {np.mean(f1s):.4f} | {np.std(f1s):.4f} | {np.min(f1s):.4f} | {np.max(f1s):.4f} |\n")
        f.write(f"| ROC AUC | {np.mean(aucs):.4f} | {np.std(aucs):.4f} | {np.min(aucs):.4f} | {np.max(aucs):.4f} |\n")

    print(f"Saved to {out_md}")
    print("\nDone!")


if __name__ == '__main__':
    main()
