# 🌌 Jet Tagging: Ante-Hoc Interpretable AI for Top Quark Tagging

**EFN (Energy Flow Network) + ArcFace Hyperspherical Prototypes** for **ante-hoc interpretable** jet classification in high-energy physics.

| Component | Status | Details |
|:----------|:------:|:--------|
| **Model Architecture** | ✅ Complete | EFN backbone + ArcFace head (1 prototype per class) |
| **Training Pipeline** | ✅ Complete | 15-epoch trained on 1.2M events, chunk-based memory-safe loading |
| **Classification** | ✅ Complete | Binary: Top quark jets (signal) vs QCD jets (background) |
| **Subclass Discovery** | ✅ Complete | 5 Top subclasses + 5 QCD subclasses via Spectral Clustering / DPGMM |
| **Physics Validation** | ✅ Complete | Lund planes, substructure, $z_g$, angular collimation |
| **SVD Analysis** | ✅ Code ready | Dimension justification via explained variance |
| **Baseline Benchmarking** | ⏳ TODO | ParticleNet / LorentzNet / Plain EFN comparison |

---

## 🧠 Architecture Overview

```
Constituents (p_T, eta, phi, E per particle)
    |
    +---> Relativization & Centering: DeltaR_i = (Delta_eta_i, Delta_phi_i)
    |
    +---> EFN (Energy Flow Network) - IRC-Safe
    |     +-- Phi MLP:  [2 -> 128 -> 128 -> 128]  (sees ONLY angular geometry)
    |     +-- Sum Pooling weighted by energy fraction z_i = E_i / Sum E
    |     +-- Rho MLP:   [128 -> 128 -> 64]       (jet-level embedding)
    |
    +---> Jet Latent Embedding  R^64  (L2-normalized to hypersphere S^63)
    |
    +---> ArcFace Classification Head
          +-- W_QCD (class center on hypersphere)
          +-- W_Top (class center on hypersphere)
          +-- Cosine similarity -> geodesic margin m = 0.5, scale s = 16.0
```

### Key Innovation: Ante-Hoc Interpretability

Unlike **post-hoc** interpretability methods (Grad-CAM, LIME, attention weights) that explain a model *after* training, ArcFace prototypes provide **ante-hoc (intrinsic) interpretability**:

- The class centers $W_{QCD}, W_{Top}$ live on the unit hypersphere $S^{63}$
- Each jet's embedding cosine similarity to these centers *is* the classification decision
- You can inspect *which jets are closest to each center* to understand "what the model thinks a Top jet looks like"
- **No separate explanation step needed** - the model's decision is inherently interpretable

---

## 📦 Project Structure

```
jet_tagging/
+-- jet_tagging/                    # Core library (editable package)
|   +-- models/
|   |   +-- efn_arcface.py          # EFN + ArcFace model definition
|   +-- data/
|   |   +-- loader.py               # H5 -> Awkward -> PyTorch data pipeline
|   +-- utils/
|       +-- physics.py              # FastJet substructure: mass, tau32, zg, Lund coordinates
|       +-- plotting.py             # Visualization: latent space, Lund planes, ROC, distributions
|
+-- scripts/
|   +-- train_top.py                # Full training + subclass discovery + Lund plane analysis
|   +-- evaluate_models.py          # Standalone model evaluation (eval/train BN modes)
|
+-- notebooks/
|   +-- JetTagging_EFN_ArcFace.ipynb  # Google Colab-ready training notebook
|
+-- main_arcface/                   # Primary analysis outputs
|   +-- centroid_constituents_report.md  # Detailed physics of centroid jets
|   +-- embeddings.npz              # Jet embeddings (404K test events)
|   +-- lund_planes/                # Class + subclass + T/F Lund plane visualizations
|       +-- class_overall_non_split.png
|       +-- status_tp.png / status_tn.png / status_fp.png / status_fn.png
|       +-- top_subclass_0..1 / qcd_subclass_0..1  (with T/F splits)
|
+-- results/
|   +-- analysis_15ep/              # Comprehensive 15-epoch model analysis
|       +-- umap_prediction_status.png       # UMAP: T/F prediction distribution
|       +-- umap_subclasses.png             # UMAP: 5 Top + 5 QCD subclasses
|       +-- subclass_physics_distributions.png  # Physics per subclass
|       +-- subclass_composition_heatmap.png    # Purity matrix
|       +-- subclass_physics_correlations.png   # Physics feature correlations
|       +-- subclass_similarity_distributions.png  # Cosine similarity per subclass
|       +-- zg_ap_splitting.png             # Altarelli-Parisi splitting validation
|       +-- angular_collimation.png         # Relativistic angular scaling
|       +-- top_subclass_[0-4]_representative_jets.png
|       +-- qcd_subclass_[0-4]_representative_jets.png
|       +-- lund_planes/                    # Full hierarchical Lund planes
|
+-- checkpoints/
|   +-- test_run_15ep/best_model_efn.pt     # Trained checkpoint (15 epochs)
|   +-- best_model_efn.pt                   # Latest best model
|   +-- baseline/                           # Placeholder for baseline models
|   +-- ...                                 # Previous experiment checkpoints
|
+-- scratch/                          # Standalone analysis scripts
|   +-- run_full_analysis.py          # Tiered full-scale analysis (Tier 1-5)
|   +-- compute_svd_variance.py       # SVD-based latent dimension justification
|   +-- compute_clustering_metrics.py # Silhouette score + ARI
|   +-- compute_bootstrap_stability.py
|   +-- ...
|
+-- data/top_tagging/                 # H5 datasets (train.h5, val.h5, test.h5)
|
+-- docs/                             # Documentation (in progress)
+-- Dockerfile / docker-compose.yml   # GPU-ready containerization
+-- requirements.txt                  # Python dependencies
+-- setup.py                          # Editable package
```

---

## 🚀 Quick Start

### Installation

```bash
# Editable install
pip install -e .

# Docker (GPU)
docker-compose up --build
```

### Train from scratch

```bash
# Quick test (5K events, 10 epochs)
python scripts/train_top.py --quick --epochs 10

# Full training (1.2M events, 30 epochs)
python scripts/train_top.py --epochs 30
```

### Evaluate trained model

```bash
python scripts/evaluate_models.py \
    --model_path checkpoints/test_run_15ep/best_model_efn.pt \
    --split test \
    --max_events 5000
```

### Run full analysis pipeline

```bash
python scripts/train_top.py --analysis_only \
    --checkpoint_dir checkpoints/test_run_15ep
```

Or use the standalone tiered analysis:

```bash
python scratch/run_full_analysis.py
```

---

## 📊 Completed Analysis Results

### Classification Performance
The model achieves strong separation between Top and QCD jets via cosine similarity margin on the hypersphere:

| Metric | Value |
|:-------|:-----:|
| Architecture | EFN (128-dim Phi, 64-dim embedding) |
| ArcFace params | s = 16.0, m = 0.5, warmup 10 epochs |
| Training | 15 epochs, ~1.2M events |
| Subclasses (Top) | **5** discovered via Spectral Clustering / DPGMM |
| Subclasses (QCD) | **5** discovered via Spectral Clustering / DPGMM |

### Subclass Discovery

The model discovers **physically meaningful subclasses** without supervision:

- **Top Subclass 0**: High-mass (~171 GeV), moderate 3-prong (tau32 ~ 0.45) - clean hadronic top decay
- **Top Subclass 1**: Higher tau32 ~ 0.68 - more QCD-like, closer to decision boundary
- **QCD Subclass 0**: High-mass (~131 GeV), moderate tau32 - hard gluon radiation mimicking top
- **QCD Subclass 1**: Low-mass (~62 GeV), high tau32 - ordinary single-prong QCD

### Physics Validation

- **Altarelli-Parisi Splitting (zg)**: Top subclasses show flatter zg distribution (W-boson decay), QCD shows 1/zg divergence - **physically correct**
- **Angular Collimation**: Delta R12 ~ 2M/pT - **relativistic scaling confirmed**
- **Lund Planes**: Clear topological differences between TP/TN/FP/FN categories

### Example: Centroid Jets

| Class | Jet Mass | tau32 | sqrt(d12) | Interpretation |
|:------|:--------:|:-----:|:----------:|:---------------|
| Top SC 0 | 171.2 GeV | 0.45 | 0.36 GeV | Clean 3-prong top decay |
| Top SC 1 | 176.9 GeV | 0.68 | 0.36 GeV | Diffuse top (near boundary) |
| QCD SC 0 | 131.2 GeV | 0.59 | 0.34 GeV | Hard radiation mimicking top |
| QCD SC 1 | 62.0 GeV | 0.81 | 0.37 GeV | Soft single-prong QCD |

---

## 🗺️ Publication Roadmap

### Current Status: Ready for paper writing (~70% complete)

| Component | Status | Priority |
|:----------|:------:|:--------:|
| Novel architecture (EFN+ArcFace) | ✅ Done | Core contribution |
| Training pipeline | ✅ Done | |
| Subclass discovery (Spectral Clustering) | ✅ Done | |
| Physics validation (Lund, zg, collimation) | ✅ Done | |
| Edge/decision boundary analysis | ✅ Done | |
| SVD dimension analysis | ✅ Script ready | Run and include |
| **Baseline comparison** | ❌ TODO | Highest priority |
| Ablation study (ArcFace vs Softmax) | ❌ TODO | Medium |
| Paper writing | ❌ TODO | |

### TODO: Baseline Models to Implement

1. **Plain EFN + Softmax** (ablated ArcFace) - quantifies interpretability cost
2. **ParticleNet** (Qu & Gouskos, 2020) - current SOTA for top tagging
3. **PFN / Deep Sets** (Komiske et al., 2019) - comparable paradigm
4. **LorentzNet** (Gong et al., 2022) - recent SOTA with Lorentz equivariance

Target metrics for comparison: ROC AUC, Accuracy, Background Rejection at 30% / 50% efficiency.

---

## 🐳 Docker & Reproducibility

```bash
# Build and run with GPU support
docker-compose up --build

# Or build manually
docker build -t jet-tagging .
docker run --gpus all -it jet-tagging
```

---

## 📚 Dependencies

- **Core ML**: torch >= 2.1.0, scikit-learn
- **HEP Physics**: fastjet, awkward, vector, h5py, hdf5plugin
- **Visualization**: matplotlib, seaborn, umap-learn

See [requirements.txt](requirements.txt) for full details.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{jet-tagging-efn-arcface,
  author = {Thirathep Phongpreecha},
  title = {Jet Tagging: Ante-Hoc Interpretable AI with EFN and ArcFace Prototypes},
  year = {2025},
  howpublished = {\url{https://github.com/thirathep-ph/Jet-Tagging}}
}
```

---

## 📄 License

This project is licensed under the MIT License.
