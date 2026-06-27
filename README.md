# 🌌 Jet Tagging: Ante-Hoc Interpretable AI for Top Quark Tagging

**EFN (Energy Flow Network) + ArcFace Hyperspherical Prototypes** for **ante-hoc interpretable** jet classification in high-energy physics.

| Component | Status | Details |
|:----------|:------:|:--------|
| **Model Architecture** | ✅ Complete | EFN backbone + ArcFace head (1 prototype per class) |
| **Training Pipeline** | ✅ Complete | 100-epoch trained on 1.2M events, chunk-based memory-safe loading |
| **Classification** | ✅ Complete | Binary: Top quark jets (signal) vs QCD jets (background) |
| **Baseline Comparison** | ✅ Complete | EFN+Linear & EFN+CosLinear × 3 seeds with error bars |
| **Robustness Analysis** | ✅ Complete | s×m grid (9 configs), all 91–92% accuracy |
| **Subclass Discovery** | ✅ Complete | 2 Top subclasses (Core/Edge) + 2 QCD subclasses (Core/Edge) via Spectral Clustering |
| **Physics Validation** | ✅ Complete | Lund planes, substructure, zg, angular collimation |
| **Dimensionality Analysis** | ✅ Complete | SVD, TwoNN, Participation Ratio — effective dim ≈ 2 |
| **Paper Writing** | ⏳ In Progress | Target: SciPost Physics |

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
- **No separate explanation step needed** — the model's decision is inherently interpretable

---

## 📦 Project Structure

```
arcjet/                        # Core library (editable package)
+-- models/
|   +-- efn_arcface.py              # EFN + ArcFace model (main)
|   +-- efn_linear.py               # EFN + Linear baseline
|   +-- efn_coslinear.py            # EFN + Cosine Linear baseline
+-- data/
|   +-- loader.py                   # H5 -> Awkward -> PyTorch data pipeline
+-- utils/
    +-- clustering.py               # Spectral clustering + eigengap
    +-- physics.py                  # FastJet substructure: mass, tau32, zg, Lund
    +-- plotting.py                 # Visualization: latent space, Lund planes, ROC
    +-- device.py                   # Device utilities

scripts/                            # Training & evaluation pipeline
+-- train_top.py                    # Full training + subclass discovery + analysis
+-- train_baselines.py              # Baseline model training (Linear + CosLinear)
+-- evaluate_models.py              # Standalone model evaluation
+-- eval_robustness.py              # Robustness sweep (s × m grid)
+-- analyze_offline.py              # Offline analysis of saved results
+-- run_all.py                      # Run full pipeline

scratch/                            # Standalone analysis scripts (reproducible)
+-- analysis/
|   +-- run_full_analysis.py            # Full analysis pipeline
|   +-- compute_svd_variance.py         # SVD dimensionality analysis
+-- clustering/
|   +-- compute_clustering_metrics.py   # Clustering quality metrics
+-- physics/
    +-- compute_ks_tests.py             # Kolmogorov-Smirnov tests
    +-- compute_jsd.py                  # Jensen-Shannon divergence

notebooks/
+-- JetTagging_EFN_ArcFace.ipynb    # Google Colab-ready training notebook

plans/
+-- scipost_complete_plan.md        # Publication plan & results tables

data/                               # Datasets (not tracked in git)
+-- top_tagging/                    # Top Tagging Reference Dataset (H5)
+-- jetnet/                         # JetNet datasets (H5)

experiments/                        # Trained model results (not tracked in git)
+-- arcface_100ep/                  # Primary model (100 epochs)
+-- baselines_linear_seed_*/        # Linear baselines (3 seeds)
+-- baselines_coslinear_seed_*/     # CosLinear baselines (3 seeds)
+-- robustness_s*_m*/               # Robustness sweep (9 configs)
+-- svd_analysis/                   # SVD analysis outputs

Dockerfile / docker-compose.yml     # GPU-ready containerization
requirements.txt                    # Python dependencies
setup.py                            # Editable package install
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

# Full training (1.2M events, 30 epochs, chunked loading)
python scripts/train_top.py --epochs 30

# Fast training with no chunking (loads all data into RAM, ~2.4GB)
python scripts/train_top.py --epochs 30 --no-chunk

# Fast training with mixed precision (GPU only, ~2x speedup)
python scripts/train_top.py --epochs 30 --no-chunk --amp
```

### Evaluate trained model

```bash
python scripts/evaluate_models.py \
    --model_path experiments/arcface_100ep/best_model_efn.pt \
    --split test \
    --max_events 50000
```

### Run robustness sweep

```bash
python scripts/eval_robustness.py
```

### Run full analysis pipeline

```bash
python scripts/analyze_offline.py
```

---

## 📊 Key Results

### Classification Performance

| Model | Accuracy | F1 |
|:------|:--------:|:--:|
| EFN + ArcFace (s=16, m=0.5) | 91.75% | 91.89% |
| EFN + Linear (3-seed mean) | 91.94% | — |
| EFN + CosLinear (3-seed mean) | 92.01% | — |

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
  author = {Thirathep Phongpreecha},
  title = {Jet Tagging: Ante-Hoc Interpretable AI with EFN and ArcFace Prototypes},
  year = {2025},
  howpublished = {\url{https://github.com/thirathep-ph/ArcJet}}
}
```

---

## 📄 License

This project is licensed under the MIT License.
