"""
Training Pipeline for EFN + ArcFace Prototype Layer
===================================================
Ante-hoc interpretability: FastJet is used ONLY for interpreting
learned prototypes after training.
"""
import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import awkward as ak
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from jet_tagging.models.efn_arcface import TopTaggingModel, arcface_loss, count_parameters, distance_corr_loss
from jet_tagging.utils.physics import compute_features
from jet_tagging.data.loader import load_awkward, JetTaggingDataset
from jet_tagging.utils.plotting import (plot_latent_space, plot_performance, 
                                      plot_representative_jet, plot_physics_correlations, 
                                      plot_lund_plane, plot_training_results)


# ============================================================================
# Training & Evaluation Loops
# ============================================================================

def train_epoch(model, loader, optimizer, device, scheduler=None, decorrelate=False, lambda_disco=10.0):
    model.train()
    total_loss, total_acc = 0, 0
    total_disco = 0.0
    criterion_cls = arcface_loss
    
    for x, y, w, m, _ in loader:
        x, y, w, m = x.to(device), y.to(device), w.to(device), m.to(device)
        
        optimizer.zero_grad()
        logits, sims, _ = model(x, mask=m)
        cls_loss = criterion_cls(logits, y)
        
        if decorrelate:
            # Calculate analytical jet mass in PyTorch from constituent 4-vectors
            E_tot = x[:, :, 0].sum(dim=1)
            px_tot = x[:, :, 1].sum(dim=1)
            py_tot = x[:, :, 2].sum(dim=1)
            pz_tot = x[:, :, 3].sum(dim=1)
            mass = torch.sqrt(torch.clamp(E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2), min=1e-10))
            
            # Compute DisCo loss only on QCD (background) events
            qcd_mask = (y == 0)
            scores = logits[:, 1] - logits[:, 0]
            disco_val = distance_corr_loss(scores, mass, mask=qcd_mask)
            
            loss = cls_loss + lambda_disco * disco_val
            total_disco += disco_val.item()
        else:
            loss = cls_loss
            
        loss.backward()
        optimizer.step()
        
        # Hard Orthogonal Projection
        model.prototype_layer.orthogonalize_prototypes()
        
        if scheduler: scheduler.step()
        
        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()
        
    return total_loss / len(loader), total_acc / len(loader), total_disco / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc = 0, 0
    all_sims, all_labels, all_indices, all_logits = [], [], [], []
    criterion_cls = arcface_loss
    
    with torch.no_grad():
        for x, y, w, m, idx in loader:
            x, y, w, m = x.to(device), y.to(device), w.to(device), m.to(device)
            logits, sims, _ = model(x, mask=m)
            
            loss = criterion_cls(logits, y)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == y).float().mean().item()
            
            all_logits.append(logits.cpu())
            all_sims.append(sims.cpu())
            all_labels.append(y.cpu())
            all_indices.append(idx.cpu())
            
    return total_loss / len(loader), total_acc / len(loader), \
           torch.cat(all_sims), torch.cat(all_labels), torch.cat(all_indices), torch.cat(all_logits)


# ============================================================================
# Interpretation
# ============================================================================

def interpret_prototypes(model, loader, dataset, device, n_prototypes=2, output_dir='results'):
    print("\n" + "=" * 60)
    print("EFN PROTOTYPE INTERPRETATION (Advanced Physics)")
    print("=" * 60)
    
    # 6 values returned from fixed evaluate
    _, _, all_sims, all_labels, all_indices, all_logits = evaluate(model, loader, device)
    
    # Find best match for each prototype (dim=0 is samples)
    best_match_indices = all_sims.argmax(dim=0)
    
    proto_physics_data = []
    for i in range(n_prototypes):
        match_idx = best_match_indices[i].item()
        real_idx = all_indices[match_idx].item()
        event_p4 = dataset.events[real_idx]
        feats = compute_features(ak.Array([event_p4]))[0]
        
        proto_class = "Top" if i >= (n_prototypes // 2) else "QCD"
        print(f"Proto {i:2d} ({proto_class}): Match #{real_idx}, Mass={feats[0]:.1f}, nSD={feats[3]:.0f}, Tau32={feats[7]:.3f}, zg={feats[8]:.3f}")
        
        proto_physics_data.append({
            'proto_id': i, 'proto_class': proto_class, 'event_idx': real_idx, 'features': feats.tolist()
        })
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'efn_results.json'), 'w') as f:
        json.dump(proto_physics_data, f, indent=2)
    return proto_physics_data, all_labels, all_sims


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max_events', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/top_tagging')
    parser.add_argument('--n_proto_per_class', type=int, default=1)
    parser.add_argument('--decorrelate', action='store_true')
    parser.add_argument('--lambda_disco', type=float, default=10.0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.max_events:
        max_events = args.max_events
    else:
        max_events = 5000 if args.quick else None

    print("Loading datasets...")
    import os
    tr_ev, tr_lab, tr_w = load_awkward(os.path.join(args.data_dir, 'train.h5'), max_events=max_events)
    val_ev, val_lab, val_w = load_awkward(os.path.join(args.data_dir, 'val.h5'), max_events=max_events)
    te_ev, te_lab, te_w = load_awkward(os.path.join(args.data_dir, 'test.h5'), max_events=max_events)

    train_ds = JetTaggingDataset(tr_ev, tr_lab, tr_w)
    val_ds = JetTaggingDataset(val_ev, val_lab, val_w)
    test_ds = JetTaggingDataset(te_ev, te_lab, te_w)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    test_loader = DataLoader(test_ds, batch_size=1024)

    model = TopTaggingModel(n_proto_per_class=args.n_proto_per_class, s=32.0, m=0.0, embedding_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=len(train_loader)*args.epochs)
    
    warmup_epochs, final_margin = 10, 0.5
    print(f"\nTraining EFN + ArcFace (xAI Prototype Model, Params: {count_parameters(model):,})")
    
    history = {'train_loss': [], 'train_acc': [], 'train_disco': [], 'val_loss': [], 'val_acc': [], 
               'sep_qcd': [], 'sep_top': []}
    best_acc = 0.0
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model_efn.pt')

    epoch_pbar = tqdm(range(args.epochs), desc="Overall Progress")
    for epoch in epoch_pbar:
        current_m = final_margin * min(1.0, (epoch + 1) / warmup_epochs)
        model.prototype_layer.set_margin(current_m)

        t0 = time.time()
        t_loss, t_acc, t_disco = train_epoch(model, train_loader, optimizer, device, scheduler=scheduler,
                                             decorrelate=args.decorrelate, lambda_disco=args.lambda_disco)
        v_loss, v_acc, v_sims, v_labels, _, v_logits = evaluate(model, val_loader, device)
        t1 = time.time()
        
        n = model.prototype_layer.n_proto_per_class
        qcd_mask, top_mask = v_labels.numpy() == 0, v_labels.numpy() == 1
        sep_qcd = v_sims[qcd_mask][:, :n].max(dim=1)[0].mean().item()
        sep_top = v_sims[top_mask][:, n:].max(dim=1)[0].mean().item()

        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['train_disco'].append(t_disco)
        history['val_loss'].append(v_loss); history['val_acc'].append(v_acc)
        history['sep_qcd'].append(sep_qcd); history['sep_top'].append(sep_top)
        
        curr_lr = optimizer.param_groups[0]['lr']
        tag = " [NEW BEST]" if v_acc > best_acc else ""
        if v_acc > best_acc:
            best_acc = v_acc
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1:2d}/{args.epochs}: m={current_m:.2f} | LR={curr_lr:.6f} | Time={t1-t0:.1f}s")
        if args.decorrelate:
            print(f"  Train: Loss={t_loss:.4f}, Acc={t_acc:.4f}, DisCo={t_disco:.4f} | Val: Acc={v_acc:.4f}{tag}")
        else:
            print(f"  Train: Loss={t_loss:.4f}, Acc={t_acc:.4f} | Val: Acc={v_acc:.4f}{tag}")
        print(f"  Sep:   QCD_Sim={sep_qcd:.4f}, Top_Sim={sep_top:.4f}")
        print("-" * 65)

    # Save History JSON
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # FINAL ANALYSIS
    model.load_state_dict(torch.load(best_model_path))
    proto_data, t_labels, t_sims = interpret_prototypes(
        model, test_loader, test_ds, device, 
        n_prototypes=model.prototype_layer.n_prototypes, 
        output_dir=args.output_dir
    )
    
    # 0. Global Metrics & Training Results
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from jet_tagging.utils.plotting import plot_decision_boundary
    
    # Correctly aggregate prototype similarities into class scores
    n = model.prototype_layer.n_proto_per_class
    qcd_scores = t_sims[:, :n].max(dim=1)[0]
    top_scores = t_sims[:, n:].max(dim=1)[0]
    class_scores = torch.stack([qcd_scores, top_scores], dim=1)
    
    test_preds = class_scores.argmax(dim=1).numpy()
    cm = confusion_matrix(t_labels.numpy(), test_preds)
    plot_training_results(history, cm, save_path=os.path.join(args.output_dir, 'training_results_efn.png'))
    
    # Advanced Metrics
    prec, rec, f1, _ = precision_recall_fscore_support(t_labels.numpy(), test_preds, average='binary')
    
    # Benchmark Inference Time (Com-Sci)
    model.eval()
    t_start = time.time()
    with torch.no_grad():
        for x, _, _, m, _ in test_loader:
            _ = model(x.to(device), mask=m.to(device))
    inf_time = (time.time() - t_start) / len(test_ds) * 1000 # ms/jet
    
    # Save Metrics
    advanced_metrics = {
        'precision': prec, 'recall': rec, 'f1_score': f1,
        'inference_time_ms': inf_time,
        'num_params': count_parameters(model)
    }
    with open(os.path.join(args.output_dir, 'advanced_metrics.json'), 'w') as f:
        json.dump(advanced_metrics, f, indent=4)
        
    # 1. Performance & Boundary
    plot_performance(t_labels.numpy(), class_scores.numpy(), save_path=os.path.join(args.output_dir, 'performance.png'))
    plot_decision_boundary(t_labels.numpy(), class_scores.numpy(), save_path=os.path.join(args.output_dir, 'decision_boundary.png'))
    
    # 2. Latent Space (Get embeddings)
    all_emb = []
    model.eval()
    with torch.no_grad():
        for x, _, _, m, _ in test_loader:
            _, _, emb = model(x.to(device), mask=m.to(device))
            all_emb.append(emb.cpu())
    all_emb = torch.cat(all_emb).numpy()
    plot_latent_space(all_emb[:5000], t_labels.numpy()[:5000], 
                      prototypes=model.prototype_layer.prototypes.detach().cpu().numpy(),
                      save_path=os.path.join(args.output_dir, 'latent_space.png'))
    
    # 3. Representative Jets
    for entry in proto_data:
        plot_representative_jet(test_ds.events[entry['event_idx']], title=f"Proto {entry['proto_id']} ({entry['proto_class']})", 
                                 save_path=os.path.join(args.output_dir, f"proto_{entry['proto_id']}_jet.png"))

    # 4. Physics Analysis
    print("  Running Substructure Correlation & Lund Plane...")
    test_idx = np.random.choice(len(test_ds), min(5000, len(test_ds)), replace=False)
    sub_events = test_ds.events[test_idx]
    sub_labels = t_labels.numpy()[test_idx]
    
    # Correctly aggregate scores for correlation analysis
    sub_class_scores = class_scores.numpy()[test_idx]
    sub_scores = sub_class_scores[:, 1] - sub_class_scores[:, 0]
    
    sub_features = compute_features(sub_events)
    feature_names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    
    # Correlation & Distribution Analysis
    from jet_tagging.utils.plotting import plot_feature_distributions
    plot_physics_correlations(sub_labels, sub_scores, sub_features, feature_names,
                               save_path=os.path.join(args.output_dir, 'correlations.png'))
    plot_feature_distributions(sub_labels, sub_features, feature_names,
                                save_path=os.path.join(args.output_dir, 'feature_distributions.png'))
    plot_lund_plane(sub_events[sub_labels==0], label="QCD", save_path=os.path.join(args.output_dir, 'lund_plane_qcd.png'))
    plot_lund_plane(sub_events[sub_labels==1], label="Top", save_path=os.path.join(args.output_dir, 'lund_plane_top.png'))

    print(f"\n[COMPLETE] All research products saved to '{args.output_dir}'.")

if __name__ == '__main__':
    main()
