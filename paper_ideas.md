# Paper Ideas & Design Justifications

This document outlines the conceptual justifications, hyperparameter settings, hierarchical analysis workflows, and plotting roadmaps for the EFN + ArcFace interpretability paper. It serves as a draft reference for writing the publication.

---

## 1. Table of Hyperparameters

The following table summarizes all model, training, and clustering hyperparameters used in the pipeline.

| Category | Parameter | Value | Justification |
| :--- | :--- | :--- | :--- |
| **EFN Architecture** | Input Dimension | 2 | Relative coordinates $(\Delta\eta, \Delta\phi)$ to enforce IRC-safety. |
| | Particle MLP ($\Phi$) Hidden Dimensions | $[256, 256, 256]$ | Captures complex geometry mappings of jet constituents. |
| | Jet MLP ($\rho$) Hidden Dimensions | $[256, 128]$ | Compresses aggregated particle features to latent space. |
| | Embedding (Latent) Dimension | 128 | Standard compact dimension for representation learning. |
| **ArcFace Head** | Scale ($s$) | 16.0 | Adjusts angular boundary sharpness; prevents gradient saturation. |
| | Target Margin ($m$) | 0.5 | Geodesic angular margin on unit hypersphere. |
| | Margin Warmup Epochs | 10 | Linearly scales margin from $0 \to 0.5$ to ease early convergence. |
| **Training** | Optimizer | Adam ($lr=10^{-3}$) | Standard robust optimizer for Deep Learning. |
| | Scheduler | OneCycleLR ($max\_lr=2\times 10^{-3}$) | Accelerates training convergence. |
| | Batch Size | 1024 | Standard large batch size for stable EFN updates. |
| **DPGMM** | Max Components ($K_{max}$) | 5 | Upper bound for subclasses; pruned by Dirichlet Process. |
| | Weight Concentration Prior ($\alpha$) | $10^{-3}$ | Low value forces model to prune inactive components. |
| | Covariance Type | `full` | Allows subclasses to have arbitrary elliptical shapes. |
| **UMAP** | Metric | `cosine` | Matches the angular distance metric optimized by ArcFace. |
| | $n\_neighbors$ | 15 | Preserves local and global neighbor structures. |
| | $min\_dist$ | 0.1 | Allows tight visual clustering of similar points. |
| | Random Seed | 42 | Ensures visualization reproducibility. |

---

## 2. Design Justifications (The "Why")

Use these arguments in the *Methodology* and *Discussion* sections of the paper:

### A. Why EFN instead of PFN?
* **IRC-Safety:** EFN enforces Infrared and Collinear (IRC) safety because its particle MLP ($\Phi$) does not process momentum/energy directly; energy is used only as a linear weight in the sum pooling. PFN includes momentum inside non-linear operations, violating IRC-safety. In perturbative QCD, IRC-safety is a requirement for calculability.

### B. Why Standard ArcFace instead of Multi-Prototype?
* **Gradient Discontinuity:** Multi-prototype architectures use max-pooling over prototypes. This creates non-differentiable boundaries and gradient discontinuities during training. Standard ArcFace is fully differentiable and represents each class as a single, clean center vector ($W_{QCD}, W_{Top}$) on the hypersphere, serving as a stable physical reference.

### C. Why DPGMM instead of K-Means / Standard GMM?
* **Non-parametric Nature:** Standard clustering requires pre-defining $K$. We do not know a priori how many physical topologies are learned in the latent space. DPGMM uses a Dirichlet Process prior to dynamically determine the natural number of subclasses, letting the data speak for itself.

### D. Why DPGMM in 128D instead of 2D UMAP space?
* **No Projection Artifacts:** UMAP projects 128D to 2D using a non-linear mapping that distorts metric distances and density distributions (e.g. flattening cluster sizes). Performing clustering directly in the 128D normalized latent space respects the true metric space optimized by ArcFace. UMAP is used strictly for 2D visualization of these 128D clusters.

### E. Why not HDBSCAN?
* **Interesting Boundary Regions:** HDBSCAN is a density-based algorithm that flags low-density points as "noise" (outliers). In xAI, these low-density boundary regions contain the most physically interesting jets: the False Positives and False Negatives where the model is confused. DPGMM assigns every point a probability membership, allowing full analysis of boundary regions rather than discarding them.

---

## 3. Hierarchical 3-Level Analysis Pipeline

The interpretability analysis is organized into a clean, hierarchical three-level framework:

### Level 1: Global Classification & T/F Analysis
* **Goal:** Evaluate the model's global prediction performance.
* **Process:** Compute predictions on the full test set. Compare predictions (`preds == 0` or `1`) against ground truth labels (`labels == 0` or `1`) to categorize all events into:
  - **True Positive (TP)**
  - **True Negative (TN)**
  - **False Positive (FP)**
  - **False Negative (FN)**
* **Output:** Global metrics (Accuracy, F1, ROC, confusion matrix) and Plot 1 (Latent Space Error Map).

### Level 2: Predicted Class-Based Clustering (Subclass Discovery)
* **Goal:** Uncover the latent topologies (subclasses) learned by the model.
* **Process:** Partition the test set into two groups based *strictly* on model predictions (what the model "says" the jet is, blind to whether it is correct or incorrect):
  - **Predicted Top Group (`preds == 1`)**
  - **Predicted QCD Group (`preds == 0`)**
* **Action:** Perform DPGMM clustering on the 128D normalized embeddings of each group *independently* to discover subclasses within Predicted Top and Predicted QCD.
* **Output:** Subclass assignments for all jets, Plot 2 (Discovered Subclasses Map).

### Level 3: Subclass-Level T/F Analysis
* **Goal:** Diagnose the composition and stability of each discovered subclass.
* **Process:** For each discovered subclass, analyze its classification composition and physical distributions:
  - For **Predicted Top subclasses**, evaluate what fraction of the subclass is **True Positive (TP)** vs. **False Negative (FN)** (i.e. identify which subclasses represent highly confident true tops vs. ambiguous regions prone to misclassifying tops as QCD).
  - For **Predicted QCD subclasses**, evaluate what fraction of the subclass is **True Negative (TN)** vs. **False Positive (FP)** (i.e. identify which subclasses are pure backgrounds vs. background regions that mimic top signatures).
* **Output:** Plot 4 (Subclass Composition Heatmap), Plot 3 (Subclass Physics Distributions), Plot 8 (Lund Planes per Subclass), and Plot 7 (Physics Feature Correlation Matrices).

---

## 4. Deep Hierarchical Lund Plane & T/F Analysis

To capture the physical radiation phase-space, the paper will include a comprehensive hierarchical Lund Plane analysis at **both Level 2 (Class) and Level 3 (Subclass) levels**, each fully split by T/F classification results:

### A. Class-Level Lund Planes & T/F Analysis
* **Predicted Class Lund Planes:** Plot the Lund Plane for Predicted Top (`preds == 1`) and Predicted QCD (`preds == 0`) to show the overall radiation profiles.
* **Class T/F Splits:** Compare correct predictions against errors:
  - **True Positives (TP)** vs. **False Negatives (FN)** (Top jets).
  - **True Negatives (TN)** vs. **False Positives (FP)** (QCD jets).
  - *Physics Insight:* Diagnoses which radiation patterns mislead the model (e.g. showing that FPs have Top-like splitting densities).

### B. Subclass-Level Lund Planes & T/F Analysis
* **Subclass Lund Planes:** Plot the Lund Plane for *each* discovered subclass (e.g., Top Subclass 0, 1, 2... and QCD Subclass 0, 1, 2...) to show the radiation patterns of the discovered topologies.
* **Subclass T/F Splits:** For each subclass, split and compare:
  - TP vs. FN inside Top Subclasses.
  - TN vs. FP inside QCD Subclasses.
  - *Physics Insight:* Shows whether a subclass represents a stable physical process or is highly contaminated by classification errors.

---

## 5. Ablation Studies & Parameter Stability

To prove that every component of the EFN + ArcFace model is necessary and optimal, the paper will include:

### A. Ablation Analysis (Component-by-Component)
Evaluate the impact of removing or replacing core elements:
1. **Remove EFN Relative Coordinates (Use Absolute Coordinates):** Shows how the model loses IRC-safety and degrades in physics generalization.
2. **Remove ArcFace Margin ($m=0$):** Shows that the latent space becomes disorganized, clusters overlap, and subclass separation degrades.
3. **Multi-Prototype vs. Standard ArcFace:** Shows that multi-prototypes degrade classification scores and lead to gradient training instabilities.
4. **UMAP-space vs. 128D-space Clustering:** Compares DPGMM clustering quality to prove that 2D clustering creates artificial shape distortions.

### B. Parameter Study
* **ArcFace Margin Scan ($m \in [0.1, 0.3, 0.5]$):** Shows how margin size controls cluster tightness and class separability.
* **Scale Parameter Scan ($s \in [8.0, 16.0, 32.0]$):** Shows how $s$ scales the logits and stabilizes the cross-entropy loss.

---

## 6. Distribution-Based Representatives (Rather than Static Vectors)

Standard prototype models present a single prototype vector (which is a static center point). This paper improves this by presenting a **distribution-based representative display** for each subclass:
* **The Problem with Single Vectors:** A single vector represents an isolated, idealized point on the hypersphere, which may not capture the full physical width and variation of the subclass.
* **Our Solution:** For each subclass, we construct:
  - **Event Display Distribution:** Show a small ensemble (or average display) of the top 5 most representative jets (highest DPGMM responsibility).
  - **Kinematic Observables Distribution:** Plot the physical feature distributions (Mass, $\tau_{32}$, splitting scales) *within* each subclass to show the full shape of the topology rather than a single number.

---

## 7. Data Science & Machine Learning Analysis

Provide standard data science metrics for validation:
* **Classification Performance:** Accuracy, Precision, Recall, F1-Score, and ROC AUC curves.
* **Confusion Matrices:** Detailed layout of TP, TN, FP, FN ratios.
* **Inference Time & Efficiency:** Inference speed (ms/jet) and parameter counts to prove EFN efficiency compared to heavy networks like Transformers.
* **Data Science Representation Correlation Heatmaps:**
  - Correlation between latent space dimension activations (or their top Principal Components via PCA) and the final ArcFace classification logits. This identifies which dimensions of the 128D latent space are heavily weighted or responsible for driving the final classification score.

---

## 8. Theorist & Phenomenology Analysis

Focus on the physical and perturbative QCD properties of the subclasses:
* **Physical Peak Table:** Mode values of Mass, $\tau_{32}$, and $\sqrt{d_{12}}$ mapping directly to the Top quark mass ($172.5\text{ GeV}$) and W boson mass ($80.4\text{ GeV}$).
* **Relativistic Angular Scaling:** Verification of the subjet collimation relation $\Delta R_{12} \approx 2M/p_T$.
* **Altarelli-Parisi Splitting Variables ($z_g$):** Verification that QCD subclasses exhibit soft divergence ($z_g \to 0$) while Top subclasses exhibit symmetric splitting ($z_g \to 0.5$).
* **Connecting AP Splitting to Symbolic Regression (Conceptual Discussion):**
  - Discuss how the analytical shapes of Altarelli-Parisi splitting kernels (e.g. $P_{qq}(z) = C_F \frac{1+z^2}{1-z}$) correspond to the numeric distributions learned within the QCD subclasses. Explain that while the neural network learns these shapes numerically, they can be mapped to analytic equations.
* **Physics Observable & Model Correlation Heatmaps (per Subclass):**
  - Plot the 2D correlation matrix of all physical features (Mass, $m_{SD}$, Multiplicity, $\tau_{32}$, $\sqrt{d_{12}}$, $z_g$, etc.) *separately for each discovered subclass* compared to the overall class. This demonstrates whether the model's subclasses correspond to distinct physical correlation regions (e.g. showing that one subclass has a high mass-multiplicity correlation typical of soft gluons, while another has mass decoupled from multiplicity typical of hard electroweak decays).
  - Plot the correlation between the model's classification logits/similarities and these physical observables to clearly show which physical features the model relies on most (e.g. mass vs. shape observables).

---

## 9. Future Work (Roadmap for Next Paper)

* **Analytical Reconstruction via Symbolic Regression (SR):** Propose using Symbolic Regression algorithms (e.g. PySR) to analytically fit the discovered subclass distributions (like $z_g$ and $\sqrt{d_{12}}$) to reconstruct the exact mathematical formulas of Altarelli-Parisi splitting kernels and angular collimation scales directly from the latent space.
* **Multiclass Extension:** Moving from binary (Top/QCD) classification to multiclass (Top, W, Z, Higgs, QCD) to study more complex overlapping topologies.
* **Mass-decorrelation via DisCo Loss:** Activating the Distance Correlation (DisCo) loss during training to force the latent space to be independent of jet mass, comparing how mass-decorrelation alters the discovered subclasses.
* **Supervised Contrastive Learning (SupCon):** Investigating alternative metric learning objectives to compare the tightness and physics of the learned manifolds.
* **Anomaly Detection:** Using the structured hyperspherical space to identify out-of-distribution physics events (e.g., new heavy resonances).
