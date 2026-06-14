# Blueprint: EFN + ArcFace for Interpretative Jet Tagging

This document serves as the single, consolidated master blueprint for the research project. It integrates the data structure, core architecture, memory-safe training pipeline, hyperparameter sweep plan, SVD width analysis, and the two-round hierarchical T/F physical analysis.

---

## 1. Dataset & Data Pipeline Structure

The dataset contains simulated collision events categorized into signal (**Top quark jets**) and background (**QCD jets**).

### A. Dataset Splits
- `train.h5` (~1.21M events) - Model training
- `val.h5` (~403K events) - Validation & checkpointing (Top signal is binarized from raw label 2)
- `test.h5` (~404K events) - Final physics analysis & evaluation

### B. HDF5 Feature Columns (`values_block_0`)
Each event contains a padded constituent list of up to $N = 200$ particles per jet.
- **Columns 0, 4, 8, ... (every 4th):** Constituent $E$ (energy)
- **Columns 1, 5, 9, ... (every 4th):** Constituent $p_x$ (x-momentum)
- **Columns 2, 6, 10, ... (every 4th):** Constituent $p_y$ (y-momentum)
- **Columns 3, 7, 11, ... (every 4th):** Constituent $p_z$ (z-momentum)
- **Columns 800–803:** Truth jet-level momentum $(p_x, p_y, p_z, E)$

`values_block_1` contains the event weight (Col 0) and target label (Col 1: $0 = \text{QCD}$, $1 = \text{Top}$).

### C. Memory-Safe Chunking & On-Demand Data Loading
To run the full 1.2M events dataset on standard Google Colab without RAM overflow:
1. **Chunk-based Training:** Train on the full dataset by loading slice blocks of 100,000 events sequentially into CPU memory. The data is shuffled within each chunk, and chunk order is randomized across epochs.
2. **On-Demand Loading:** For post-training visualizations (representative jets, subclass plots), retrieve only the specific events by index using `load_events_by_indices`, keeping RAM usage constant.

---

## 2. EFN + ArcFace Core Architecture

The model combines a permutation-invariant backbone for representation learning with a hyperspherical classification head for intrinsic interpretability.

```
Constituents (E, px, py, pz) 
    │
    ├──→ Relativization & Centering: ΔR_i = (Δη_i, Δφ_i)
    │
    ├──→ EFN (Energy Flow Network)
    │     ├── Phi MLP (sees ONLY geometry to ensure IRC-safety)
    │     ├── Sum Pooling (weighted by Energy fraction z_i)
    │     └── Rho MLP (maps to jet representation)
    │
    ├──→ Jet Latent Embedding (d_embedding)
    │
    └──→ ArcFace Head (Standard cosine classifier with geodesic margin)
          ├── QCD weight vector (W_QCD center on sphere)
          └── Top weight vector (W_Top center on sphere)
```

### A. Energy Flow Network (EFN) Backbone
- **IRC-Safety:** Enforced by feeding only angular coordinates $(\Delta\eta, \Delta\phi)$ to the particle-level MLP ($\Phi$) and using normalized energy fractions ($z_i = E_i / \sum E$) as linear weights in sum pooling.
- **Backbone Dimensions:**
  - Particle MLP $\Phi$: $[2 \to 128 \to 128 \to 128]$ with ReLU
  - Jet MLP $\rho$: $[128 \to 64 \to 64]$ (embedding dimension $d = 64$)

### B. Standard ArcFace Classification Head
- Represents class centroids as weight vectors ($W_{\text{QCD}}$, $W_{\text{Top}}$) on a unit hypersphere ($S^{D-1}$).
- Uses standard cosine similarity mapping during inference.
- Scale parameter $s = 16.0$ to prevent logit explosion in binary classification.
- Margin $m = 0.5$ (in radians) is applied geodesically during training, with a linear margin warmup from $0.0 \to 0.5$ over the first 10 epochs to stabilize early training.

---

## 3. Two-Round Hierarchical T/F Physics Analysis

Downstream analysis is structured into two sequential rounds of classification purity checks and phase-space visualizations (Lund planes) using FastJet.

```
[Test Set Predictions]
   │
   ├──→ Round 1: Class-level T/F Analysis
   │     ├── Group into TP, TN, FP, FN
   │     ├── Plot 4 Lund Planes
   │     └── FastJet Substructure Diagnosis (Why did errors happen?)
   │
   └──→ Round 2: Subclass-level T/F Analysis
         ├── Group predicted Top (TP vs FP) & predicted QCD (TN vs FN)
         ├── Run Spectral Clustering (Manifold-aware, cosine affinity)
         └── Plot Lund Planes per Subclass (Non-split & Split)
```

### Round 1: Class-Level T/F Analysis
- **Goal:** Analyze overall classification success and diagnose error regions.
- **Process:** Split the entire test set into categories:
  - **Overall (Non-split):** The complete radiation profile.
  - **True Positive (TP):** Correct signal prediction.
  - **True Negative (TN):** Correct background prediction.
  - **False Positive (FP):** Background misclassified as Signal.
  - **False Negative (FN):** Signal misclassified as Background.
- **Visualizations:** Plot 5 independent Lund Planes (one non-split and four T/F splits).
- **FastJet Diagnosis:** Compute substructure variables (Mass, $\tau_{32}$, $z_g$, $\sqrt{d_{12}}$) for each group:
  - Analyze whether FPs (misclassified QCD) mimic Top jets due to hard gluon radiation forming high-mass, multi-prong structures.
  - Analyze whether FNs (misclassified Tops) lost constituents due to high collimation or leptonic decay, mimicking single-prong QCD.

### Round 2: Subclass-Level T/F Analysis (Subclass Discovery)
- **Goal:** Uncover the manifold topology of learned physical subclasses.
- **Process:** Partition the test dataset based on predictions:
  - **Predicted Top Group (`preds == 1`)**
  - **Predicted QCD Group (`preds == 0`)**
- **Action:** Run **Spectral Clustering** (cosine affinity) on the 64D embeddings of each group separately to discover subclasses.
- **Subclass Visualizations:** For every discovered subclass, plot **three** Lund Planes:
  1. **Non-split:** Overall subclass radiation profile (always plotted).
  2. **Correct split:** TP (for Top subclasses) or TN (for QCD subclasses) — plotted if ≥5 events.
  3. **Incorrect split:** FP (for Top subclasses) or FN (for QCD subclasses) — plotted if ≥5 events.
  - *Physics Insight:* Diagnoses whether a subclass represents a stable, clean physical topology or is highly contaminated by specific error modes.

---

## 4. Hyperparameter Sweeps & SVD Optimization Plan

We perform a hyperparameter sweep to identify the optimal model, followed by SVD analysis to mathematically verify the network and latent space size.

### A. Grid Sweep Matrix (8 Trials on Colab Pro) — *Not Yet Executed*
Using V100/A100 GPUs, we plan to run 8 trials of 15 epochs each (~10 minutes per trial) over the following parameters:
- **Embedding Dim ($d$):** `[64, 128]`
- **ArcFace Scale ($s$):** `[16, 32]`
- **ArcFace Margin ($m$):** `[0.3, 0.5]`

The best trial would be selected based on validation ROC AUC, accuracy, and class separation gap, then trained for **30 epochs** in a single final run.

### B. SVD-based Latent Dimension Selection
Perform SVD on the final test embedding matrix $X \in \mathbb{R}^{N \times d}$:
$$X = U \Sigma V^T$$
Plot the Scree plot of singular values $\sigma_i$. Calculate the dimension $k$ needed to capture $\ge 99\%$ of the explained variance. This mathematically justifies the minimum required latent dimension for the jet representation.

### C. SVD-based Network Width Analysis
Perform SVD on the EFN's hidden layer weight matrices $W_l$:
$$W_l = U_l \Sigma_l V_l^T$$
Analyze the rank deficiency of the $128 \times 128$ MLPs. If the singular values decay rapidly, it mathematically proves overparameterization, showing that the network width can be compressed (e.g. to 64) in future studies.

---

## 5. Mathematical & Theoretical Consistency

1. **Hyperspherical Constraint:** ArcFace forces embeddings to unit length. Thus, the embedding space is a hypersphere $S^{D-1}$.
2. **Spectral Clustering Cosine Affinity:** We use Cosine Similarity ($1 - d_{\text{cosine}}$) for Spectral Clustering. This matches the hyperspherical geometry of ArcFace, preventing metric distortion.
3. **Spectral Clustering (SC) vs. von Mises-Fisher (vMF):**
   - **vMF** assumes subclasses are isotropic spherical caps on the sphere.
   - **SC** is non-parametric and manifold-aware, allowing it to capture continuous physical transitions (curved, non-convex shapes) in the latent space.
   - Therefore, SC is chosen to represent the continuous nature of QCD/Top jet phase space.
4. **Cosine $k$-NN Proxy:** *Not yet implemented.* Planned for out-of-sample subclass assignment based on Cosine Distance, maintaining 100% metric alignment with the rest of the pipeline.
