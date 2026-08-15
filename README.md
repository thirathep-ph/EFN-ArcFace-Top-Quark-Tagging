# 🌌 Jet Tagging: Ante-Hoc Interpretable AI for Top Quark Tagging

[![CI](https://github.com/thirathep-ph/ArcEFN/actions/workflows/ci.yml/badge.svg)](https://github.com/thirathep-ph/ArcEFN/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EFN (Energy Flow Network) + ArcFace Hyperspherical Prototypes** for **ante-hoc interpretable** jet classification in high-energy physics.

Reference paper: *Ante-Hoc Interpretable Top Quark Tagging with Energy Flow Networks and Hyperspherical Prototypes* (Thirathep N. Phiankham).

| Component | Status | Details |
|:----------|:------:|:--------|
| **Model Architecture** | ✅ Complete | EFN backbone + ArcFace head (1 prototype per class) |
| **Training Pipeline** | ✅ Complete | 50-epoch trained on 1.2M events, chunk-based memory-safe loading |
| **Classification** | ✅ Complete | Binary: Top quark jets (signal) vs QCD jets (background) |
| **Baseline Comparison** | ✅ Complete | EFN+Linear & EFN+CosLinear × 3 seeds with error bars |
| **Robustness Analysis** | ✅ Complete | s×m grid (9 configs), all 91–92% accuracy |
| **Subclass Discovery** | ✅ Complete | 2 Top subclasses (Core/Edge) + 2 QCD subclasses (Core/Edge) via Spectral Clustering |
| **Physics Validation** | ✅ Complete | Lund planes, substructure, zg, angular collimation |
| **Dimensionality Analysis** | ✅ Complete | SVD, TwoNN, Participation Ratio — effective dim ≈ 2 |

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

### Pipeline Flowchart

```
Data Flow (training):
  H5 Dataset ──> load_awkward() ──> JetTaggingDataset ──> DataLoader
       (Top Tagging Ref Dataset)        (chunked, 200 cons.)    (batch)
                                                                    │
                                                                    ▼
Training Pipeline:
  DataLoader ──> EFN (Phi MLP + Sum Pool + Rho MLP) ──> Jet Embeddings
                                                              │
                                                              ▼
                              L2 Normalize ──> Hypersphere S^63 ──> ArcFace Loss
                                                                       │
                                                                       ▼
                                                            Backpropagation

Evaluation + Analysis:
  Checkpoint ──> EFN + ArcFace ──> Logits               Jet Embeddings
                                       │                       │
                                       ▼                       ├──> UMAP (Fig 5-6)
                                 Confusion Matrix              ├──> Eigengap (Fig 2)
                                       │                       ├──> Spectral Clustering
                                       ▼                       ├──> DGLAP Fit (Fig 8)
                                 ROC Curve (Fig 4)             ├──> Shenanigans (Fig 10)
                                       │                       └──> MI Analysis (Fig 13)
                                       ▼
                                 Rejection (Fig 12)

Paper Figures (%) ──> LaTeX (paper/main.tex) ──> PDF
```

### Key Innovation: Ante-Hoc Interpretability

Unlike **post-hoc** interpretability methods (Grad-CAM, LIME, attention weights) that explain a model *after* training, ArcFace prototypes provide **ante-hoc (intrinsic) interpretability**:

- The class centers $W_{QCD}, W_{Top}$ live on the unit hypersphere $S^{63}$
- Each jet's embedding cosine similarity to these centers *is* the classification decision
- You can inspect *which jets are closest to each center* to understand "what the model thinks a Top jet looks like"
- **No separate explanation step needed** — the model's decision is inherently interpretable

---

## 📦 Project Structure

```
arcefn/                            # Core library (editable package)
+-- models/
|   +-- efn_arcface.py                  # EFN + ArcFace model (main)
|   +-- efn_linear.py                   # EFN + Linear baseline
|   +-- efn_coslinear.py                # EFN + Cosine Linear baseline
+-- data/
|   +-- loader.py                       # H5 -> Awkward -> PyTorch data pipeline
+-- utils/
|   +-- clustering.py                   # Spectral clustering + eigengap
|   +-- physics.py                      # FastJet substructure: mass, tau32, zg, Lund
|   +-- plotting.py                     # Visualization: latent space, Lund planes, ROC
|   +-- device.py                       # Device utilities
|   +-- paths.py                        # Cross-platform path resolution
+-- tests/
    +-- test_model.py                   # Model forward-pass smoke tests
    +-- test_loader.py                  # Data loading tests
    +-- test_clustering.py              # Spectral clustering tests
    +-- test_physics.py                 # FastJet physics feature tests
    +-- test_utils.py                   # Device + path utility tests

scripts/                            # Training & evaluation
+-- train_arcface.py                # EFN + ArcFace training (CLI)
+-- train_baselines.py              # Linear + CosLinear baseline training
+-- evaluate_model.py               # Standalone model evaluation
+-- evaluate_robustness.py          # Robustness sweep (s × m grid)
+-- run_pipeline.py                 # Orchestrator — run full pipeline

scripts/analysis/                   # Paper figure & analysis scripts (all results)
+-- compute_mass_vs_score.py            # Fig: mass vs classification score
+-- compute_dglap_fit.py                # Fig: Altarelli-Parisi / DGLAP fits
+-- compute_bootstrap_ci.py             # Fig: ROC, rejection rates, bootstrap CI
+-- compute_interpretation.py           # Fig: MI heatmap, effect sizes, PCA
+-- compute_saliency.py                 # Fig: constituent saliency analysis
+-- compute_eigengap.py                 # Fig: eigengap spectrum
+-- compute_dimensionality.py           # Tab: TwoNN, Participation Ratio, angular distance
+-- compute_robustness_tests.py         # Tab: pT rescaling, rotation, dropout
+-- generate_paper_figures.py           # Fig: UMAP, Lund planes, subclass physics
+-- compute_eda.py                      # Appendix: EDA figures (kinematics, substructure)
+-- compute_svd_variance.py             # SVD dimensionality analysis
+-- compute_head_geometry.py            # ArcFace vs Linear head comparison
+-- compute_alignment_uniformity.py     # Alignment & uniformity analysis
+-- generate_angular_collimation.py     # Angular collimation ΔR ≈ 2M/pT
+-- poc_ood_generalization.py           # OOD generalization (JetNet dataset)
+-- poc_mass_decoupled_accuracy.py      # Mass-decoupled accuracy test
+-- poc_injection_scan.py               # Injection scan (signal fraction sweep)
+-- poc_embedding_physics_manifold.py   # Embedding-physics manifold analysis

data/                               # Datasets (not tracked in git)
+-- top_tagging/                    # Top Tagging Reference Dataset (H5)

experiments/                        # Results (not tracked in git)
+-- robustness_s16_m05/             # Main model checkpoint + figures
+-- baselines_linear_seed_*/        # Linear baselines
+-- baselines_coslinear_seed_*/     # CosLinear baselines
+-- robustness_s*_m*/               # Robustness sweep

Dockerfile / docker-compose.yml     # GPU-ready containerization
requirements.txt                    # Python dependencies
setup.py                            # Editable package install
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Download the Top Tagging Reference Dataset (3.4 GB)
python scripts/download_data.py

# 2. Editable install
pip install -e .

# Docker (GPU)
docker-compose up --build
```

### Reproduce paper results

```bash
# 0. Data and dependencies (once)
python scripts/download_data.py
pip install -e .

# 1. Train models
python scripts/train_arcface.py                       # EFN + ArcFace (full)
python scripts/train_baselines.py --n_seeds 3          # Linear + CosLinear baselines

# 2. Evaluate
python scripts/evaluate_model.py --split test           # evaluation metrics
python scripts/evaluate_robustness.py                  # robustness sweep

# 3. Generate all paper figures & tables
python scripts/run_pipeline.py --figures

# Or run individual analysis scripts
python scripts/analysis/compute_bootstrap_ci.py        # Table 1, ROC curves
python scripts/analysis/compute_eigengap.py            # Fig: eigengap spectrum
python scripts/analysis/compute_dglap_fit.py           # Fig: AP/DGLAP fits
python scripts/analysis/compute_mass_vs_score.py       # Fig: mass vs score
python scripts/analysis/compute_interpretation.py      # Fig: MI heatmap
python scripts/analysis/compute_saliency.py            # Fig: saliency
python scripts/analysis/compute_dimensionality.py      # Table: TwoNN, PR
python scripts/analysis/compute_robustness_tests.py    # Table: robustness
python scripts/analysis/generate_paper_figures.py      # UMAP, Lund, subclass
python scripts/analysis/compute_eda.py                 # Appendix: EDA figures
python scripts/analysis/compute_head_geometry.py       # ArcFace vs Linear comparison
python scripts/analysis/compute_alignment_uniformity.py# Embedding quality metrics
python scripts/analysis/poc_ood_generalization.py      # OOD generalization
```

### Quick test

```bash
# Quick test (5K events, 10 epochs)
python scripts/train_arcface.py --quick --epochs 10
```

### Evaluate trained model

```bash
python scripts/evaluate_model.py --split test --max_events 50000
```

---

## 📊 Key Results

### Classification Performance

| Model | Accuracy | ROC AUC | Rejection@50%TPR |
|:------|:--------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | 91.84% [91.76, 91.92] | 0.973 | 120× |
| EFN + Linear (3-seed mean) | 91.94% | 0.974 | 117× |
| EFN + CosLinear (3-seed mean) | 92.00% | 0.974 | 138× |

### Subclass Discovery

The model discovers **physically meaningful subclasses** without supervision:

- **Top Core**: High-mass (~171 GeV), moderate 3-prong (τ32 ~ 0.45) — clean hadronic top decay
- **Top Edge**: Higher τ32 ~ 0.68 — more QCD-like, closer to decision boundary
- **QCD Core**: High-mass (~131 GeV), moderate τ32 — hard gluon radiation mimicking top
- **QCD Edge**: Low-mass (~62 GeV), high τ32 — ordinary single-prong QCD

### Physics Validation

- **Altarelli-Parisi Splitting (zg)**: Top subclasses show flatter zg distribution (W-boson decay), QCD shows 1/zg divergence — **physically correct**
- **Angular Collimation**: ΔR12 ~ 2M/pT — **relativistic scaling confirmed**
- **Lund Planes**: Clear topological differences between TP/TN/FP/FN categories

### Head Comparison: ArcFace vs Linear

The hyperspherical constraint is essential for clean subclass discovery:

| Metric | ArcFace (hypersphere) | Linear (Euclidean) |
|:-------|:---------------------:|:------------------:|
| Subclass balance | Balanced ~20-31% per subclass | Edge subclasses only 5-7% |
| DGLAP fit quality | R² = 0.97-0.99 for all 4 subclasses | Poor fits on small Edge subclasses |
| Embedding norms | Fixed at $\|\mathbf{e}\|_2 = 1.0$ (hypersphere) | Varies 0.5-6.0 (Euclidean) |

The balanced subclasses and clean DGLAP fits under ArcFace confirm that the hypersphere geometry is necessary — the Euclidean head does not produce physically meaningful subclasses.

### Robustness

All s × m configurations maintain 91–92% accuracy (except s=8, m=0.5 at 89.5%), confirming robustness to ArcFace hyperparameter choices.

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
  author = {Thirathep N. Phiankham},
  title = {Jet Tagging: Ante-Hoc Interpretable AI with EFN and ArcFace Prototypes},
  year = {2025},
  howpublished = {\url{https://github.com/thirathep-ph/ArcEFN}}
}
```

---

## 📄 License

This project is licensed under the MIT License.
