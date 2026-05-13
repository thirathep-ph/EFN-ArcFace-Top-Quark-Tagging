# Jet Tagging: Interpretable Physics with AI

A research-oriented library for Jet Tagging using Energy Flow Networks (EFN) and ArcFace Prototype layers for **ante-hoc interpretability**.

## 🚀 Structure
- `jet_tagging/`: Core library containing models, data loaders, and physics utilities.
- `scripts/`: Production-ready training and evaluation scripts.
- `notebooks/`: Exploration and Google Colab-ready notebooks.
- `data/`: Local storage for H5 datasets.

## 🛠️ Installation
To install as an editable package:
```bash
pip install -e .
```

## 🐳 Docker Support
To build and run using Docker (requires NVIDIA Container Toolkit for GPU):
```bash
docker-compose up --build
```

## 📈 Usage
To train the Top Tagging model:
```bash
python scripts/train_top.py --quick --epochs 10
```

## 🧠 Features
- **EFN Backbone**: IRC-safe architecture using angular geometry and energy weighting.
- **ArcFace Prototypes**: Interpretable classification head that matches jets to learned physics patterns.
- **Vector Integration**: Robust kinematic transformations using the `vector` library.
