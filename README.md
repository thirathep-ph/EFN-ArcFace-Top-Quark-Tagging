# 🌌 Jet Tagging: Ante-Hoc Interpretable AI for Top Quark Tagging

**EFN (Energy Flow Network) + ArcFace Hyperspherical Prototypes** for **ante-hoc interpretable** jet classification in high-energy physics.

| Component | Status | Details |
|:----------|:------:|:--------|
| **Model Architecture** | ✅ Complete | EFN backbone + ArcFace head (1 prototype per class) |
| **Training Pipeline** | ✅ Complete | 100-epoch trained on 1.2M events, chunk-based memory-safe loading |
| **Classification** | ✅ Complete | Binary: Top quark jets (signal) vs QCD jets (background) |
| **Subclass Discovery** | ✅ Complete | 2 Top subclasses (Core/Edge) + 2 QCD subclasses (Core/Edge) via Spectral Clustering |
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
    |     +-- Rho MLP:   [128 -> 128 -> 64]       (jet-level embedding: transform then compress)
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
jet_tagging/                        # Core library (editable package)
+-- models/
|   +-- efn_arcface.py              # EFN + ArcFace model (main)
|   +-- efn_linear.py               # EFN + Linear baseline
|   +-- efn_coslinear.py            # EFN + Cosine Linear baseline
+-- data/
|   +-- loader.py                   # H5 -> Awkward -> PyTorch data pipeline
+-- utils/
    +-- physics.py                  # FastJet substructure: mass, tau32, zg, Lund
    +-- plotting.py                 # Visualization: latent space, Lund planes, ROC

scripts/                            # Training & evaluation
+-- train_top.py                    # Full training + subclass discovery + analysis
+-- train_baselines.py              # Baseline model training (Linear + CosLinear)
+-- evaluate_models.py              # Standalone model evaluation

notebooks/
+-- JetTagging_EFN_ArcFace.ipynb    # Google Colab-ready training notebook

main_arcface/                       # PRIMARY RESULTS (100-epoch model)
+-- best_model_efn.pt               # Trained checkpoint
+-- advanced_metrics.json           # Acc 91.74%, F1 91.89%
+-- embeddings.npz                  # 404K test embeddings
+-- centroid_constituents_report.md # Physics of centroid jets
+-- svd_scree_plot.png              # SVD dimension analysis
+-- lund_planes/                    # Lund plane visualizations

results/
+-- analysis_15ep/                  # 15-epoch model analysis (subclass discovery)
+-- test_run_50k/                   # 50K-event experiment with metrics

checkpoints/
+-- best_model_efn.pt               # Root-level checkpoint (with optimizer state)
+-- test_run_15ep/                  # 15-epoch checkpoint
+-- test_run_50k/                   # 50K-event checkpoint
+-- test_chunk/                     # Chunked training test
+-- baseline/                       # Baseline model checkpoints (future)

scratch/                            # Standalone analysis scripts
+-- clustering/                     # Clustering comparisons (SC, K-means, GMM, etc.)
+-- physics/                        # Physics analysis (KS tests, JSD, constituents)
+-- analysis/                       # Dimensionality, SVD, bootstrap, UMAP, etc.
+-- archive/                        # One-time-use scripts

plans/                              # Planning & strategy documents
+-- research_master_plan.md         # Master blueprint
+-- parameter_catalog.md            # Complete parameter catalog with reviewer defenses
+-- strategy_rethink.md             # Current strategy & robustness plan
+-- audit_fixes.md                  # Workspace audit & fixes
+-- reviewer_simulation.md          # 7 simulated reviewer questions
+-- reviewer_responses.md           # Response strategies
+-- multi_perspective_review.md     # 6 reviewer perspectives
+-- paper_results_table.md          # Paper-ready results tables
+-- publication_roadmap.md          # Target venue & timeline
+-- fitness_assessment.md           # Go/No-Go decision
+-- improvement_ideas.md            # 13 improvement ideas
+-- baseline_implementation_plan.md # Baseline training plan
+-- session_log_*.md                # Session logs
+-- MIGRATION_GUIDE.md              # Machine migration guide
+-- README.md                       # This index

data/top_tagging/                   # H5 datasets (train.h5, val.h5, test.h5)

Dockerfile / docker-compose.yml     # GPU-ready containerization
requirements.txt                    # Python dependencies
setup.py                            # Editable package
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
| Training | 100 epochs, ~1.2M events |
| Subclasses (Top) | **2** (Core/Edge) discovered via Spectral Clustering |
| Subclasses (QCD) | **2** (Core/Edge) discovered via Spectral Clustering |

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
