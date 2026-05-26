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
import matplotlib.pyplot as plt

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
        logits, sims, _ = model(x, labels=y, mask=m)
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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max_events', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--data_dir', type=str, default='data/top_tagging')
    parser.add_argument('--n_proto_per_class', type=int, default=1)
    parser.add_argument('--decorrelate', action='store_true')
    parser.add_argument('--lambda_disco', type=float, default=10.0)
    parser.add_argument('--scale', type=float, default=16.0, help='ArcFace scale parameter s')
    parser.add_argument('--embedding_dim', type=int, default=128, help='EFN embedding/latent dimension')
    parser.add_argument('--particle_dim', type=int, default=256, help='EFN particle/phi hidden dimension')
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

    model = TopTaggingModel(
        n_proto_per_class=args.n_proto_per_class,
        s=args.scale,
        m=0.0,
        embedding_dim=args.embedding_dim,
        particle_dim=args.particle_dim
    ).to(device)
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
    
    # Export jet embeddings and labels post-training
    emb_path = os.path.join(args.output_dir, 'embeddings.npz')
    np.savez(emb_path, embeddings=all_emb, labels=t_labels.numpy())
    print(f"Exported {len(all_emb)} jet embeddings to '{emb_path}'")

    # --- ADVANCED UMAP & DPGMM ANALYSIS ---
    import umap
    from sklearn.mixture import BayesianGaussianMixture
    import seaborn as sns
    
    print("\n--- Running Advanced UMAP & DPGMM Subclass Discovery ---")
    all_emb_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-10)
    prototypes = model.prototype_layer.prototypes.detach().cpu().numpy()
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-10)
    
    # UMAP Projection
    print("Fitting UMAP (metric='cosine')...")
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    umap_coords = reducer.fit_transform(all_emb_norm)
    proto_umap = reducer.transform(prototypes_norm)
    
    # DPGMM Subclass Discovery
    subclass_labels = np.zeros(len(all_emb), dtype=int) - 1
    min_samples = 10
    
    top_mask = (test_preds == 1)
    if top_mask.sum() >= min_samples:
        print("Fitting DPGMM on Top latent space embeddings...")
        top_latent = all_emb_norm[top_mask]
        dpgmm_top = BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2)
        top_subclasses = dpgmm_top.fit_predict(top_latent)
        active_top, counts_top = np.unique(top_subclasses, return_counts=True)
        print(f"Top Active Subclasses: {active_top} with counts {counts_top}")
        subclass_labels[top_mask] = top_subclasses
    else:
        print(f"Warning: Too few Top predictions ({top_mask.sum()}) to run DPGMM. Skipping Top subclass discovery.")
        active_top = np.array([])
        
    qcd_mask = (test_preds == 0)
    if qcd_mask.sum() >= min_samples:
        print("Fitting DPGMM on QCD latent space embeddings...")
        qcd_latent = all_emb_norm[qcd_mask]
        dpgmm_qcd = BayesianGaussianMixture(n_components=5, weight_concentration_prior=1e-3, random_state=42, n_init=2)
        qcd_subclasses = dpgmm_qcd.fit_predict(qcd_latent)
        active_qcd, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
        print(f"QCD Active Subclasses: {active_qcd} with counts {counts_qcd}")
        subclass_labels[qcd_mask] = qcd_subclasses
    else:
        print(f"Warning: Too few QCD predictions ({qcd_mask.sum()}) to run DPGMM. Skipping QCD subclass discovery.")
        active_qcd = np.array([])
    
    # Calculate Prediction Status
    preds = class_scores.argmax(dim=1).numpy()
    tp_mask = (preds == 1) & (t_labels.numpy() == 1)
    tn_mask = (preds == 0) & (t_labels.numpy() == 0)
    fp_mask = (preds == 1) & (t_labels.numpy() == 0)
    fn_mask = (preds == 0) & (t_labels.numpy() == 1)
    
    status = np.zeros(len(all_emb), dtype=object)
    status[tp_mask] = 'True Positive'
    status[tn_mask] = 'True Negative'
    status[fp_mask] = 'False Positive'
    status[fn_mask] = 'False Negative'
    
    # Plot 1: umap_prediction_status.png
    plt.figure(figsize=(10, 8))
    colors = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4', 'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
    markers = {'True Negative': 'o', 'True Positive': 'o', 'False Positive': 'x', 'False Negative': 'x'}
    sizes = {'True Negative': 15, 'True Positive': 15, 'False Positive': 40, 'False Negative': 40}
    alphas = {'True Negative': 0.3, 'True Positive': 0.3, 'False Positive': 0.8, 'False Negative': 0.8}
    
    for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
        mask_cat = (status == cat)
        if mask_cat.sum() > 0:
            plt.scatter(umap_coords[mask_cat, 0], umap_coords[mask_cat, 1],
                        color=colors[cat], marker=markers[cat], s=sizes[cat],
                        alpha=alphas[cat], label=f"{cat} ({mask_cat.sum()})")
                        
    plt.scatter(proto_umap[0, 0], proto_umap[0, 1], marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
    plt.scatter(proto_umap[1, 0], proto_umap[1, 1], marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
    
    try:
        for cat in ['True Negative', 'True Positive']:
            mask_cat = (status == cat)
            if mask_cat.sum() > 50:
                sns.kdeplot(x=umap_coords[mask_cat, 0], y=umap_coords[mask_cat, 1], levels=4, color=colors[cat], alpha=0.3, linewidths=1.2)
    except Exception as e:
        pass
        
    plt.title("EFN + ArcFace Latent Space (UMAP) & Prediction Status", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.15)
    plt.savefig(os.path.join(args.output_dir, 'umap_prediction_status.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: umap_subclasses.png
    plt.figure(figsize=(10, 8))
    qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
    for i, sc in enumerate(active_qcd):
        sc_mask = (test_preds == 0) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            plt.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], color=qcd_colors[i % len(qcd_colors)], s=15, alpha=0.5, label=f"QCD Subclass {i} (N={sc_mask.sum()})")
            
    top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    for i, sc in enumerate(active_top):
        sc_mask = (test_preds == 1) & (subclass_labels == sc)
        if sc_mask.sum() > 0:
            plt.scatter(umap_coords[sc_mask, 0], umap_coords[sc_mask, 1], color=top_colors[i % len(top_colors)], s=15, alpha=0.5, label=f"Top Subclass {i} (N={sc_mask.sum()})")
            
    plt.scatter(proto_umap[0, 0], proto_umap[0, 1], marker='*', s=450, color='yellow', edgecolors='black', linewidth=1.5, label='W_QCD (Center)', zorder=10)
    plt.scatter(proto_umap[1, 0], proto_umap[1, 1], marker='*', s=450, color='magenta', edgecolors='black', linewidth=1.5, label='W_Top (Center)', zorder=10)
    plt.title("Discovered Subclasses in Latent Space (UMAP + DPGMM)", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.15)
    plt.savefig(os.path.join(args.output_dir, 'umap_subclasses.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Representative Jets
    for entry in proto_data:
        plot_representative_jet(test_ds.events[entry['event_idx']], title=f"Proto {entry['proto_id']} ({entry['proto_class']})", 
                                 save_path=os.path.join(args.output_dir, f"proto_{entry['proto_id']}_jet.png"))
                                 
    # 4. Physics Analysis
    print("  Running Substructure Correlation & Lund Plane...")
    
    # Stratified sampling based on latent space subclass and prediction status distribution
    sampled_indices = []
    
    # A. Sample from each active Top subclass (predicted Top)
    for sc in active_top:
        sc_indices = np.where((test_preds == 1) & (subclass_labels == sc))[0]
        if len(sc_indices) > 0:
            n_sample = min(1000, len(sc_indices))
            sampled_indices.extend(np.random.choice(sc_indices, n_sample, replace=False))
            
    # B. Sample from each active QCD subclass (predicted QCD)
    for sc in active_qcd:
        sc_indices = np.where((test_preds == 0) & (subclass_labels == sc))[0]
        if len(sc_indices) > 0:
            n_sample = min(1000, len(sc_indices))
            sampled_indices.extend(np.random.choice(sc_indices, n_sample, replace=False))
            
    # C. Sample from each prediction status (TP, TN, FP, FN) to ensure rare errors are well-represented
    for cat in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
        cat_indices = np.where(status == cat)[0]
        if len(cat_indices) > 0:
            n_sample = min(1000, len(cat_indices))
            sampled_indices.extend(np.random.choice(cat_indices, n_sample, replace=False))
            
    # D. Fallback if no indices are selected
    if len(sampled_indices) > 0:
        test_idx = np.unique(sampled_indices)
    else:
        test_idx = np.random.choice(len(test_ds), min(5000, len(test_ds)), replace=False)
        
    print(f"  Stratified sampling complete: selected {len(test_idx)} representative jets based on latent space distributions.")
    sub_events = test_ds.events[test_idx]
    sub_labels = t_labels.numpy()[test_idx]
    
    # Correctly aggregate scores for correlation analysis
    sub_class_scores = class_scores.numpy()[test_idx]
    sub_scores = sub_class_scores[:, 1] - sub_class_scores[:, 0]
    
    sub_features = compute_features(sub_events)
    feature_names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    
    # Plot 3: subclass_physics_distributions.png (integrated)
    print("Generating subclass physics distributions...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    mass_all = sub_features[:, 0]
    tau32_all = sub_features[:, 7]
    d12_all = sub_features[:, 4]
    
    # We map back sub_idx to active_top and active_qcd subclass labels
    sub_subclass = subclass_labels[test_idx]
    sub_status = status[test_idx]
    sub_preds = test_preds[test_idx]
    
    # Col 1: Discovered Top Subclasses
    for i, sc in enumerate(active_top):
        sc_mask = (sub_preds == 1) & (sub_subclass == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(mass_all[sc_mask], ax=axes[0, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
            sns.kdeplot(tau32_all[sc_mask], ax=axes[1, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
            sns.kdeplot(d12_all[sc_mask], ax=axes[2, 0], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[0, 0].set_title("Discovered Top Subclasses: Jet Mass", fontsize=12)
    axes[0, 0].set_xlabel("Mass (GeV)")
    axes[0, 0].legend()
    axes[1, 0].set_title("Discovered Top Subclasses: N-subjettiness Tau32", fontsize=12)
    axes[1, 0].set_xlabel("Tau32")
    axes[1, 0].legend()
    axes[2, 0].set_title("Discovered Top Subclasses: Splitting Scale sqrt(d12)", fontsize=12)
    axes[2, 0].set_xlabel("sqrt(d12) (GeV)")
    axes[2, 0].legend()
    
    # Col 2: QCD TN vs QCD FP
    tn_sub = (sub_status == 'True Negative')
    fp_sub = (sub_status == 'False Positive')
    if tn_sub.sum() > 5:
        sns.kdeplot(mass_all[tn_sub], ax=axes[0, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
        sns.kdeplot(tau32_all[tn_sub], ax=axes[1, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
        sns.kdeplot(d12_all[tn_sub], ax=axes[2, 1], label="True QCD (TN)", color='green', fill=True, alpha=0.1)
    if fp_sub.sum() > 5:
        sns.kdeplot(mass_all[fp_sub], ax=axes[0, 1], label="Mistaken QCD (FP)", color='red', lw=2)
        sns.kdeplot(tau32_all[fp_sub], ax=axes[1, 1], label="Mistaken QCD (FP)", color='red', lw=2)
        sns.kdeplot(d12_all[fp_sub], ax=axes[2, 1], label="Mistaken QCD (FP)", color='red', lw=2)
    axes[0, 1].set_title("QCD Mistakes: Jet Mass", fontsize=12)
    axes[0, 1].set_xlabel("Mass (GeV)")
    axes[0, 1].legend()
    axes[1, 1].set_title("QCD Mistakes: N-subjettiness Tau32", fontsize=12)
    axes[1, 1].set_xlabel("Tau32")
    axes[1, 1].legend()
    axes[2, 1].set_title("QCD Mistakes: Splitting Scale sqrt(d12)", fontsize=12)
    axes[2, 1].set_xlabel("sqrt(d12) (GeV)")
    axes[2, 1].legend()
    
    # Col 3: Top TP vs Top FN
    tp_sub = (sub_status == 'True Positive')
    fn_sub = (sub_status == 'False Negative')
    if tp_sub.sum() > 5:
        sns.kdeplot(mass_all[tp_sub], ax=axes[0, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
        sns.kdeplot(tau32_all[tp_sub], ax=axes[1, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
        sns.kdeplot(d12_all[tp_sub], ax=axes[2, 2], label="True Top (TP)", color='blue', fill=True, alpha=0.1)
    if fn_sub.sum() > 5:
        sns.kdeplot(mass_all[fn_sub], ax=axes[0, 2], label="Mistaken Top (FN)", color='orange', lw=2)
        sns.kdeplot(tau32_all[fn_sub], ax=axes[1, 2], label="Mistaken Top (FN)", color='orange', lw=2)
        sns.kdeplot(d12_all[fn_sub], ax=axes[2, 2], label="Mistaken Top (FN)", color='orange', lw=2)
    axes[0, 2].set_title("Top Mistakes: Jet Mass", fontsize=12)
    axes[0, 2].set_xlabel("Mass (GeV)")
    axes[0, 2].legend()
    axes[1, 2].set_title("Top Mistakes: N-subjettiness Tau32", fontsize=12)
    axes[1, 2].set_xlabel("Tau32")
    axes[1, 2].legend()
    axes[2, 2].set_title("Top Mistakes: Splitting Scale sqrt(d12)", fontsize=12)
    axes[2, 2].set_xlabel("sqrt(d12) (GeV)")
    axes[2, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'subclass_physics_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot 3 to {os.path.join(args.output_dir, 'subclass_physics_distributions.png')}")

    # ============================================================================
    # Plot 4: Subclass Composition Heatmap (Purity / Level 3 T/F Analysis)
    # ============================================================================
    print("Generating Plot 4: subclass_composition_heatmap.png...")
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    composition_matrix = []
    row_labels = []

    if len(active_top) > 0:
        top_probs = dpgmm_top.predict_proba(top_latent)
        for i, sc in enumerate(active_top):
            sc_mask = (test_preds == 1) & (subclass_labels == sc)
            n_tot = sc_mask.sum()
            if n_tot > 0:
                row_counts = []
                for cat in categories:
                    cat_count = ((status == cat) & sc_mask).sum()
                    row_counts.append(cat_count / n_tot * 100.0)
                composition_matrix.append(row_counts)
                row_labels.append(f"Top Subclass {i} (N={n_tot})")

    if len(active_qcd) > 0:
        qcd_probs = dpgmm_qcd.predict_proba(qcd_latent)
        for i, sc in enumerate(active_qcd):
            sc_mask = (test_preds == 0) & (subclass_labels == sc)
            n_tot = sc_mask.sum()
            if n_tot > 0:
                row_counts = []
                for cat in categories:
                    cat_count = ((status == cat) & sc_mask).sum()
                    row_counts.append(cat_count / n_tot * 100.0)
                composition_matrix.append(row_counts)
                row_labels.append(f"QCD Subclass {i} (N={n_tot})")

    if len(composition_matrix) > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(composition_matrix, annot=True, fmt=".1f", xticklabels=categories, yticklabels=row_labels, cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
        plt.title("Subclass Classification Purity / Composition", fontsize=14)
        plt.xlabel("Prediction Status")
        plt.ylabel("Discovered Subclasses")
        plt.savefig(os.path.join(args.output_dir, 'subclass_composition_heatmap.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved Plot 4 to {os.path.join(args.output_dir, 'subclass_composition_heatmap.png')}")

    # ============================================================================
    # Plot 5: Representative Event Displays (Ensemble of Top 5)
    # ============================================================================
    print("Generating Plot 5: subclass_X_representative_jets.png...")
    # Top Subclasses
    if len(active_top) > 0:
        top_indices_in_test = np.where(test_preds == 1)[0]
        top_probs = dpgmm_top.predict_proba(top_latent)
        for i, sc in enumerate(active_top):
            assigned_jets_in_top = (top_subclasses == sc)
            if assigned_jets_in_top.sum() > 0:
                probs_for_sc = top_probs[:, sc].copy()
                probs_for_sc[~assigned_jets_in_top] = -1.0
                top5_relative_indices = np.argsort(probs_for_sc)[::-1][:5]
                top5_test_indices = top_indices_in_test[top5_relative_indices]
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                for idx_fig, t_idx in enumerate(top5_test_indices):
                    event = test_ds.events[t_idx]
                    px = ak.to_numpy(event.px)
                    py = ak.to_numpy(event.py)
                    pz = ak.to_numpy(event.pz)
                    e = ak.to_numpy(event.E)
                    pt = np.sqrt(px**2 + py**2)
                    eta = np.arcsinh(pz / (pt + 1e-10))
                    phi = np.arctan2(py, px)
                    z_e = e / (np.sum(e) + 1e-10)
                    
                    ax = axes[idx_fig]
                    ax.scatter(eta, phi, s=pt*5, alpha=0.7, c=z_e, cmap='plasma', edgecolors='black', linewidth=0.3)
                    ax.set_xlabel(r'$\eta$ (rel)')
                    ax.set_ylabel(r'$\phi$ (rel)')
                    ax.set_title(f"Jet {idx_fig+1} (Prob: {probs_for_sc[top5_relative_indices[idx_fig]]:.2f})", fontsize=10)
                    ax.grid(True, alpha=0.15)
                    
                plt.suptitle(f"Top Subclass {i} - Top 5 Representative Jets", fontsize=14, y=1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'top_subclass_{i}_representative_jets.png'), dpi=150, bbox_inches='tight')
                plt.close()

    # QCD Subclasses
    if len(active_qcd) > 0:
        qcd_indices_in_test = np.where(test_preds == 0)[0]
        qcd_probs = dpgmm_qcd.predict_proba(qcd_latent)
        for i, sc in enumerate(active_qcd):
            assigned_jets_in_qcd = (qcd_subclasses == sc)
            if assigned_jets_in_qcd.sum() > 0:
                probs_for_sc = qcd_probs[:, sc].copy()
                probs_for_sc[~assigned_jets_in_qcd] = -1.0
                top5_relative_indices = np.argsort(probs_for_sc)[::-1][:5]
                top5_test_indices = qcd_indices_in_test[top5_relative_indices]
                
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                for idx_fig, t_idx in enumerate(top5_test_indices):
                    event = test_ds.events[t_idx]
                    px = ak.to_numpy(event.px)
                    py = ak.to_numpy(event.py)
                    pz = ak.to_numpy(event.pz)
                    e = ak.to_numpy(event.E)
                    pt = np.sqrt(px**2 + py**2)
                    eta = np.arcsinh(pz / (pt + 1e-10))
                    phi = np.arctan2(py, px)
                    z_e = e / (np.sum(e) + 1e-10)
                    
                    ax = axes[idx_fig]
                    ax.scatter(eta, phi, s=pt*5, alpha=0.7, c=z_e, cmap='plasma', edgecolors='black', linewidth=0.3)
                    ax.set_xlabel(r'$\eta$ (rel)')
                    ax.set_ylabel(r'$\phi$ (rel)')
                    ax.set_title(f"Jet {idx_fig+1} (Prob: {probs_for_sc[top5_relative_indices[idx_fig]]:.2f})", fontsize=10)
                    ax.grid(True, alpha=0.15)
                    
                plt.suptitle(f"QCD Subclass {i} - Top 5 Representative Jets", fontsize=14, y=1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f'qcd_subclass_{i}_representative_jets.png'), dpi=150, bbox_inches='tight')
                plt.close()

    # ============================================================================
    # Plot 6: Cosine Similarity Distributions
    # ============================================================================
    print("Generating Plot 6: subclass_similarity_distributions.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: QCD subclasses similarity to W_QCD
    for i, sc in enumerate(active_qcd):
        sc_mask = (test_preds == 0) & (subclass_labels == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(class_scores.numpy()[sc_mask, 0], ax=axes[0], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[0].set_title("QCD Subclasses: Cosine Similarity to W_QCD Center", fontsize=12)
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15)

    # Right: Top subclasses similarity to W_Top
    for i, sc in enumerate(active_top):
        sc_mask = (test_preds == 1) & (subclass_labels == sc)
        if sc_mask.sum() > 5:
            sns.kdeplot(class_scores.numpy()[sc_mask, 1], ax=axes[1], label=f"Subclass {i}", fill=True, alpha=0.1)
    axes[1].set_title("Top Subclasses: Cosine Similarity to W_Top Center", fontsize=12)
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'subclass_similarity_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot 6 to {os.path.join(args.output_dir, 'subclass_similarity_distributions.png')}")

    # ============================================================================
    # Plot 7: Physics Feature Correlation Matrices per Subclass
    # ============================================================================
    print("Generating Plot 7: subclass_physics_correlations.png...")
    import pandas as pd
    n_top_sc = len(active_top)
    n_qcd_sc = len(active_qcd)

    if n_top_sc > 0 or n_qcd_sc > 0:
        fig, axes = plt.subplots(2, max(n_top_sc, n_qcd_sc), figsize=(4 * max(n_top_sc, n_qcd_sc), 8))
        feature_subnames = ["Mass", "Tau32", "sqrt(d12)"]
        
        # Row 0: Top subclasses
        for i, sc in enumerate(active_top):
            sc_mask = (sub_preds == 1) & (sub_subclass == sc)
            ax = axes[0, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[0]
            if sc_mask.sum() > 5:
                sub_df = pd.DataFrame(np.stack([mass_all[sc_mask], tau32_all[sc_mask], d12_all[sc_mask]], axis=1), columns=feature_subnames)
                sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
            ax.set_title(f"Top Subclass {i}", fontsize=11)
            
        # Row 1: QCD subclasses
        for i, sc in enumerate(active_qcd):
            sc_mask = (sub_preds == 0) & (sub_subclass == sc)
            ax = axes[1, i] if max(n_top_sc, n_qcd_sc) > 1 else axes[1]
            if sc_mask.sum() > 5:
                sub_df = pd.DataFrame(np.stack([mass_all[sc_mask], tau32_all[sc_mask], d12_all[sc_mask]], axis=1), columns=feature_subnames)
                sns.heatmap(sub_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
            ax.set_title(f"QCD Subclass {i}", fontsize=11)
            
        plt.suptitle("Physics Feature Correlation per Subclass", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'subclass_physics_correlations.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved Plot 7 to {os.path.join(args.output_dir, 'subclass_physics_correlations.png')}")

    # ============================================================================
    # Plot 8 & 9: Hierarchical Lund Planes (Level 2 Class & Level 3 Subclass + T/F)
    # ============================================================================
    print("Generating Plot 8 & 9: lund_planes/...")
    lund_dir = os.path.join(args.output_dir, 'lund_planes')
    os.makedirs(lund_dir, exist_ok=True)
    from jet_tagging.utils.plotting import plot_lund_plane

    # A. Class-level Lund planes
    plot_lund_plane(sub_events[sub_preds == 1], label="Predicted Top (Class-level)", save_path=os.path.join(lund_dir, 'class_predicted_top.png'))
    plot_lund_plane(sub_events[sub_preds == 0], label="Predicted QCD (Class-level)", save_path=os.path.join(lund_dir, 'class_predicted_qcd.png'))

    # B. Prediction Status Lund planes (Level 1 T/F)
    sub_tp = (sub_status == 'True Positive')
    sub_tn = (sub_status == 'True Negative')
    sub_fp = (sub_status == 'False Positive')
    sub_fn = (sub_status == 'False Negative')

    if sub_tp.sum() > 0:
        plot_lund_plane(sub_events[sub_tp], label="True Positives (TP)", save_path=os.path.join(lund_dir, 'status_tp.png'))
    if sub_tn.sum() > 0:
        plot_lund_plane(sub_events[sub_tn], label="True Negatives (TN)", save_path=os.path.join(lund_dir, 'status_tn.png'))
    if sub_fp.sum() > 0:
        plot_lund_plane(sub_events[sub_fp], label="False Positives (FP)", save_path=os.path.join(lund_dir, 'status_fp.png'))
    if sub_fn.sum() > 0:
        plot_lund_plane(sub_events[sub_fn], label="False Negatives (FN)", save_path=os.path.join(lund_dir, 'status_fn.png'))

    # C. Subclass level Lund planes with T/F splits (Level 2 & 3 T/F)
    for i, sc in enumerate(active_top):
        sc_mask_sub = (sub_preds == 1) & (sub_subclass == sc)
        if sc_mask_sub.sum() > 0:
            plot_lund_plane(sub_events[sc_mask_sub], label=f"Top Subclass {i}", save_path=os.path.join(lund_dir, f'top_subclass_{i}.png'))
            # T/F Split within this subclass
            sc_tp = sc_mask_sub & (sub_labels == 1)
            sc_fp = sc_mask_sub & (sub_labels == 0)
            if sc_tp.sum() > 5:
                plot_lund_plane(sub_events[sc_tp], label=f"Top Subclass {i} (TP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_tp.png'))
            if sc_fp.sum() > 5:
                plot_lund_plane(sub_events[sc_fp], label=f"Top Subclass {i} (FP)", save_path=os.path.join(lund_dir, f'top_subclass_{i}_fp.png'))
                
    for i, sc in enumerate(active_qcd):
        sc_mask_sub = (sub_preds == 0) & (sub_subclass == sc)
        if sc_mask_sub.sum() > 0:
            plot_lund_plane(sub_events[sc_mask_sub], label=f"QCD Subclass {i}", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}.png'))
            # T/F Split within this subclass
            sc_tn = sc_mask_sub & (sub_labels == 0)
            sc_fn = sc_mask_sub & (sub_labels == 1)
            if sc_tn.sum() > 5:
                plot_lund_plane(sub_events[sc_tn], label=f"QCD Subclass {i} (TN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_tn.png'))
            if sc_fn.sum() > 5:
                plot_lund_plane(sub_events[sc_fn], label=f"QCD Subclass {i} (FN)", save_path=os.path.join(lund_dir, f'qcd_subclass_{i}_fn.png'))

    print(f"Saved Lund planes to {lund_dir}")
    
    # Correlation & Distribution Analysis
    from jet_tagging.utils.plotting import plot_feature_distributions
    plot_physics_correlations(sub_labels, sub_scores, sub_features, feature_names,
                               save_path=os.path.join(args.output_dir, 'correlations.png'))
    plot_feature_distributions(sub_labels, sub_features, feature_names,
                                save_path=os.path.join(args.output_dir, 'feature_distributions.png'))
    plot_lund_plane(sub_events[sub_labels==0], label="QCD", save_path=os.path.join(args.output_dir, 'lund_plane_qcd.png'))
    plot_lund_plane(sub_events[sub_labels==1], label="Top", save_path=os.path.join(args.output_dir, 'lund_plane_top.png'))
    
    # (Backward compatible latent space plot)
    plot_latent_space(all_emb[:5000], t_labels.numpy()[:5000], 
                      prototypes=model.prototype_layer.prototypes.detach().cpu().numpy(),
                      save_path=os.path.join(args.output_dir, 'latent_space.png'))

    print(f"\n[COMPLETE] All research products saved to '{args.output_dir}'.")

if __name__ == '__main__':
    main()
