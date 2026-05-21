"""
Model Evaluation Script for EFN + ArcFace (1 Prototype Per Class)
================================================================
This script evaluates a trained model checkpoint on the specified dataset split (train, val, or test).
It allows checking classification performance under both:
  1. 'eval' mode: using Batch Normalization running statistics (standard).
  2. 'train' mode: using batch-level statistics (useful if running stats have not converged).
"""

import argparse
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jet_tagging.models.efn_arcface import TopTaggingModel
from jet_tagging.data.loader import load_awkward, JetTaggingDataset


def evaluate_model(model_path, data_dir, split, max_events, mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model Checkpoint
    # We use n_proto_per_class=1 (1 QCD + 1 Top prototype)
    model = TopTaggingModel(n_proto_per_class=1, s=32.0, m=0.0, embedding_dim=64).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file '{model_path}' not found.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded checkpoint: {model_path}")
    
    # 2. Load Dataset Split
    file_path = os.path.join(data_dir, f"{split}.h5")
    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        sys.exit(1)
        
    events, labels, weights = load_awkward(file_path, max_events=max_events)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    # 3. Helper function to run evaluation
    def run_eval(bn_mode):
        if bn_mode == 'train':
            model.train()
        else:
            model.eval()
            
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y, w, m, _ in loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits, _, _ = model(x, mask=m)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = np.mean(all_preds == all_labels)
        
        # Calculate confusion matrix components
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        
        # Advanced Metrics
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        
        # Prediction distribution
        unique, counts = np.unique(all_preds, return_counts=True)
        pred_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'pred_dist': pred_dist
        }

    # 4. Perform Evaluation
    modes_to_test = ['eval', 'train'] if mode == 'both' else [mode]
    
    print("\n" + "=" * 55)
    print(f"EVALUATION REPORT: {split.upper()} split ({len(dataset):,} events)")
    print("=" * 55)
    
    for m in modes_to_test:
        res = run_eval(m)
        print(f"\n[BN Mode: {m.upper()}]")
        print(f"  Accuracy:  {res['accuracy']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall:    {res['recall']:.4f}")
        print(f"  F1-Score:  {res['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True QCD (0):  TN={res['tn']:5d} | FP={res['fp']:5d}")
        print(f"    True Top (1):  FN={res['fn']:5d} | TP={res['tp']:5d}")
        print(f"  Prediction Distribution: {res['pred_dist']}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EFN + ArcFace model checkpoints")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the saved PyTorch model checkpoint (.pt)")
    parser.add_argument('--data_dir', type=str, default='data/top_tagging',
                        help="Directory containing the HDF5 data splits (default: data/top_tagging)")
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help="Dataset split to evaluate (default: val)")
    parser.add_argument('--max_events', type=int, default=5000,
                        help="Maximum number of events to load for evaluation (default: 5000)")
    parser.add_argument('--mode', type=str, default='both', choices=['eval', 'train', 'both'],
                        help="BN evaluation mode: 'eval', 'train', or 'both' (default: both)")
                        
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.split, args.max_events, args.mode)


if __name__ == '__main__':
    main()
