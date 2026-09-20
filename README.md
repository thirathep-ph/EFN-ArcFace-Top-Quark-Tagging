# 🌌 Hyperspherical Prototype Classification for Interpretable Top Quark Tagging

[![CI](https://github.com/thirathep-ph/ArcEFN/actions/workflows/ci.yml/badge.svg)](https://github.com/thirathep-ph/ArcEFN/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EFN (Energy Flow Network) + ArcFace hyperspherical prototypes** for jet classification — a geometrically transparent decision rule on the unit hypersphere $S^{63}$.

Paper: *EFN–ArcFace Top Quark Tagging and Jet Substructure via Hyperspherical Prototypes* — Thirathep N. Phiankham ([`paper_arxiv/main.pdf`](paper_arxiv/main.pdf)).

![EFN+ArcFace pipeline and key results](docs/infographic.png)

---

## 🧠 Architecture Overview

```
Constituents (p_T, eta, phi, E per particle)
    |
    +---> Relativization & Centering: DeltaR_i = (Delta_eta_i, Delta_phi_i)
    |
    +---> EFN (Energy Flow Network) - IRC-Safe
    |     +-- Phi MLP:  [2 -> 128 -> 128 -> 128]  (angular geometry only)
    |     +-- Sum Pool weighted by energy fraction z_i = E_i / Sum E
    |     +-- Rho MLP:   [128 -> 128 -> 64]       (jet-level embedding)
    |
    +---> Jet Latent Embedding  R^64  (L2-normalized to S^63)
    |
    +---> ArcFace Head
          +-- W_QCD, W_Top (learned prototypes on S^63)
          +-- cos(θ+m)·s  (m=0.5, s=16; s=22 tested within noise)
```

### What the geometry buys you

Complementing post-hoc explanations, the decision rule is **geometrically transparent**:

- $W_{\text{QCD}}, W_{\text{Top}} \in S^{63}$ are learned prototypes
- Classification = nearest prototype by geodesic distance $d = \arccos(\hat e \cdot w_j)$
 - Physical meaning of prototypes is **validated post-hoc** ($z_g$, Lund, $\Delta R$) — geometry is intrinsic, physics is checked (see paper Sec. 4)

---

## 📦 Project Structure

```
arcefn/                          # core library (pip install -e .)
  models/  efn_arcface.py, efn_linear.py, efn_coslinear.py
  data/    loader.py
  utils/   clustering.py, physics.py, plotting.py, device.py, paths.py, model_loading.py
  tests/   test_model.py, test_loader.py, test_clustering.py, ...

scripts/                         # training & evaluation
  train_arcface.py               # EFN+ArcFace (CLI, --scale 16/22, --seed)
  train_baselines.py
  evaluate_model.py
  evaluate_robustness.py

scripts/analysis/                # paper figures (all regenerate experiments/*.json)
  compute_head_geometry.py       # 3-way ArcFace/CosLinear/Linear zg fits
  compute_alignment_uniformity.py# embedding norms + subclass balance (3-way)
  compute_dglap_fit.py, compute_bootstrap_ci.py, ...

paper/                           # SciPost manuscript (32pp, 50 cites verified)
  main.tex, refs.bib, SciPost.cls, arxiv.sty, figures/*.png
  overleaf.zip                   # upload-ready for Overleaf (SciPost)
paper_arxiv/                     # arXiv manuscript (27pp, article+arxiv.sty, synced from paper/ via scripts/sync_arxiv.py)
  main.tex, refs.bib, arxiv.sty, figures/*.png
  paper_arxiv.tar.gz             # upload-ready for arXiv (texlive 2025)

data/top_tagging/                # Top Tagging Reference Dataset (H5, not tracked)
experiments/                     # results (*.json tracked, *.pt ignored)
  robustness_s16_m05/            # main ArcFace s=16
  robustness_s22_m05 + _seed_123/_seed_7/  # s=22 3-seed
  baselines_linear_seed_*/ baselines_coslinear_seed_*/
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Dataset (3.4 GB)
python scripts/download_data.py

# 2. Editable install
pip install -e .

# Docker (GPU)
docker-compose up --build
```

### Reproduce paper numbers (404k test)

```bash
# Train (full: 50 epochs, 1.2M train / 404k test; --quick for 5k/10 epochs)
python scripts/train_arcface.py --scale 16 --seed 42
python scripts/train_arcface.py --scale 22 --seed 42  # within-noise check

# Baselines (3 seeds each)
python scripts/train_baselines.py --n_seeds 3

# Evaluate on full test
python -c "
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.device import get_device
# load_model_from_dir('experiments/robustness_s16_m05', device=get_device()[0])
"

# All paper figures
python scripts/analysis/compute_head_geometry.py --model_dir experiments/robustness_s16_m05 \
  --compare_dir experiments/baselines_coslinear_seed_42 experiments/baselines_linear_seed_42 --out_tag _3way --max_events 50000
python scripts/analysis/compute_alignment_uniformity.py --out_tag _3way --max_events 50000
```

### Paper → PDF / Overleaf / arXiv

```bash
# SciPost (32pp, 0 undefined)
cd paper
pdflatex main; bibtex main; pdflatex main; pdflatex main  # -> main.pdf (32pp)
# or upload paper/overleaf.zip to Overleaf — builds standalone (TeX Live 2025)

# arXiv (27pp, article+arxiv.sty; regenerate with: python scripts/sync_arxiv.py)
cd ../paper_arxiv
pdflatex main; bibtex main; pdflatex main; pdflatex main  # -> main.pdf (27pp)
# or upload paper_arxiv.tar.gz to arXiv — TeX Live 2025 pdflatex
```

---

## 📊 Key Results

### Classification (404k test, Table 1 in paper)

| Model | Accuracy | ROC AUC | Rejection@50%TPR |
|:------|:--------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | **91.84% [91.76, 91.92]** | 0.974 | **124×** |
| EFN + Linear (3-seed mean) | 91.94% ±0.01% | 0.974 | 122× |
| EFN + CosLinear (3-seed mean) | 92.00% ±0.03% | 0.975 | 132× |
| RF (10 substructure features) | 92.37% | 0.977 | 150× |

Scale check (3 seeds, full test): **s=22 → 91.81% mean (91.82/91.80/91.82)** vs s=16 91.84% — within 95% binomial ±0.08pp, retained s=16 for training stability (paper Sec. 2.3).

### Subclass Discovery

Unsupervised spectral clustering on $S^{63}$ **per predicted class** finds 4 subclasses:

- **Top Core**: ~181 GeV, τ32 ~0.63 — boundary mixture (absorbs 93% of false positives)
- **Top Edge**: ~174 GeV, τ32 ~0.46 — clean 3-prong tops (98.5% pure)
- **QCD Core**: ~62 GeV — ordinary single-prong
- **QCD Edge**: ~134 GeV — hard gluon mimicking top (holds 99% of false negatives)

Predicted vs ground-truth partitions agree: ARI 0.79 global (0.95 Top, 0.88 QCD), same $k=2$.

### Physics Validation (Sec. 4)

- **$z_g$ DGLAP fits**: QCD $\alpha \approx -1.2$, Top flatter/positive — consistent with splitting hierarchy
- **Angular collimation**: $\Delta R \sim 2m/p_T$
- **Lund planes**: distinct topologies per subclass

### Head Comparison — why the hypersphere matters (Sec. 4.3, 3-way)

| Metric | ArcFace ($S^{63}$, $m=0.5$) | CosLinear ($S^{63}$, $m=0$) | Linear (Euclidean) |
|:-------|:---------------------------:|:---------------------------:|:------------------:|
| Subclass balance | **29/19/28/24%** | **24/24/31/21%** | **37/12/42/10%** (Edge 10–12%) |
| DGLAP $R^2$ | 0.92–0.99 all 4 | QCD clean, Top $R^2=0.67$ (Edge) | Top-Edge collapses ($R^2=0.30$) |
| Raw norm median | 7.73 ±4.94 | 5.47 ±3.87 | 3.69 ±2.40 |
| What it isolates | hypersphere + margin | **hypersphere alone** | neither |

> **Takeaway:** balanced subclasses + clean QCD splitting come from the **hypersphere** (CosLinear keeps them); **margin $m=0.5$** specifically resolves Top Core vs Edge. Figure: `head_geometry_3way.png` + `embedding_geometry_3way.png`.

---

## 🐳 Docker & Reproducibility

```bash
docker-compose up --build
# or
docker build -t jet-tagging .; docker run --gpus all -it jet-tagging
```

All paper numbers are traceable to `experiments/*.json` (tracked) — see paper Sec. 8 (statistical considerations) and each figure caption for sample sizes.

---

## 📚 Dependencies

- **Core ML**: torch 2.3.1 (torch-directml on AMD RX 5700 XT; no CUDA/AMP needed), scikit-learn
- **HEP**: fastjet, awkward, vector, h5py, hdf5plugin
- **Viz**: matplotlib, seaborn, umap-learn

See `requirements.txt` for full list.

---

## 📖 Citation

```bibtex
@misc{jet-tagging-hyperspherical,
  author = {Thirathep N. Phiankham},
  title = {Hyperspherical Prototype Classification for Interpretable Top Quark Tagging with Energy Flow Networks},
  year = {2026},
  howpublished = {\url{https://github.com/thirathep-ph/ArcEFN}},
   note = {SciPost submission}
}
```

---

## 📄 License

MIT
