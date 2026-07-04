# 🌌 Jet Tagging: Ante-Hoc Interpretable AI for Top Quark Tagging

**EFN (Energy Flow Network) + ArcFace Hyperspherical Prototypes** for **ante-hoc interpretable** jet classification in high-energy physics.

Reference paper: *Ante-Hoc Interpretable Top Quark Tagging with Energy Flow Networks and Hyperspherical Prototypes* (Thirathep Phongpreecha).

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
arcjet/                            # Core library (editable package)
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

scripts/analysis/                   # Paper figure generation (all figures)
+-- compute_mass_vs_score.py        # Fig: mass vs classification score
+-- compute_dglap_fit.py            # Fig: Altarelli-Parisi / DGLAP fits
+-- compute_bootstrap_ci.py         # Fig: ROC, rejection rates, bootstrap CI
+-- compute_interpretation.py       # Fig: MI heatmap, effect sizes, PCA
+-- compute_saliency.py             # Fig: constituent saliency analysis
+-- compute_eigengap.py             # Fig: eigengap spectrum
+-- compute_dimensionality.py       # Tab: TwoNN, Participation Ratio, angular distance
+-- compute_robustness_tests.py     # Tab: pT rescaling, rotation, dropout
+-- generate_paper_figures.py       # Fig: UMAP, Lund planes, subclass physics

scripts/explore/                    # Additional analyses (not paper figures)
+-- bayesian_v3.py                  # Bayesian uncertainty quantification
+-- conformal_final.py              # Conformal prediction coverage
+-- pgd_attack.py                   # Adversarial PGD robustness
+-- onnx_export.py                  # ONNX model export
+-- ...

scratch/                            # Early-stage analysis scripts
+-- analysis/run_full_analysis.py
+-- clustering/compute_clustering_metrics.py
+-- physics/

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
# Editable install
pip install -e .

# Docker (GPU)
docker-compose up --build
```

### Reproduce paper results

```bash
# Train models
python scripts/train_arcface.py                       # EFN + ArcFace (full)
python scripts/train_baselines.py --n_seeds 3          # Linear + CosLinear baselines

# Generate all paper figures & tables
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
