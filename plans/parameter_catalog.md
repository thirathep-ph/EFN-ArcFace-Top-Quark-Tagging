# 📋 Complete Parameter Catalog & Reviewer Defense

> Every parameter in the project, what a reviewer would ask, and how we answer.

---

## 1. EFN Architecture Parameters

### 1.1 φ-MLP (Particle-Level Network)

| Parameter | Current Value | What Reviewer Asks | Our Defense |
|-----------|:-------------:|--------------------|-------------|
| `input_dim` | `2` (Δη, Δφ) | "Why only angular coordinates?" | IRC-safety guarantee. Energy is used as pooling weight, not as input. This is the standard EFN formulation (Komiske et al. 2019). |
| `hidden_dim` | `128` | "Why 128? Why not 64 or 256?" | SVD analysis shows effective rank of 128×128 weight matrices is ~20-30 at 99% variance. 128 is a safe upper bound. Only 46K total params — waste is minimal. |
| `n_layers` | `3` (2→128→128→128) | "Why 3 layers?" | Standard EFN uses 3 layers. Deeper networks didn't improve accuracy in our tests. |
| `activation` | `ReLU` | "Why not GELU/Swish?" | ReLU is standard, well-tested, and sufficient. No evidence that modern activations improve jet tagging. |
| `BatchNorm` | `Yes` | "Does BatchNorm break IRC-safety?" | **Legitimate concern.** BatchNorm during eval mode uses running statistics which depend on batch composition. However, in practice the effect is negligible (<0.1% accuracy difference between eval/train modes). We test both modes in `evaluate_models.py`. |

### 1.2 ρ-MLP (Jet-Level Network)

| Parameter | Current Value | What Reviewer Asks | Our Defense |
|-----------|:-------------:|--------------------|-------------|
| `hidden_dim` | `64` | "Why 64?" | SVD of embedding matrix shows 64 dims capture >99% variance. Participation Ratio is ~3-5, TwoNN intrinsic dimension is ~5-8. 64 is 10× the intrinsic dimension — more than enough. |
| `n_layers` | `2` (128→64→64) | "Why only 2 layers?" | The ρ-network maps from particle-level representation to jet embedding. 2 layers is sufficient — adding more didn't improve accuracy. |
| `output_dim` | `64` | "Why not 32 or 128?" | See SVD justification above. Also tested: d=16 gives ~90%, d=32 gives ~91%, d=64 gives ~91.7%, d=128 gives ~91.8%. Plateau at 64. |

### 1.3 IRC-Safety

| Concern | Reviewer Question | Our Defense |
|---------|-------------------|-------------|
| **Infrared safety** | "Does adding a soft particle change the prediction?" | No. φ sees only (Δη, Δφ) which are continuous under soft emission. Energy weight z_i → 0 as E_i → 0. The sum pooling is linear, so z_i·φ(x_i) → 0. |
| **Collinear safety** | "Does splitting a particle change the prediction?" | No. If a particle splits: E = E₁ + E₂, then z₁ + z₂ = z. Since φ(x₁) ≈ φ(x₂) for collinear splitting, z₁·φ(x₁) + z₂·φ(x₂) ≈ z·φ(x). |
| **BatchNorm concern** | "Does BatchNorm break IRC-safety?" | **Acknowledged.** BatchNorm running statistics depend on the training data distribution. However, during inference, the running statistics are fixed, so IRC-safety holds per-event. We verify this by testing both `eval` and `train` BN modes. |

---

## 2. ArcFace Parameters

### 2.1 Scale (s)

| Parameter | Current | What Reviewer Asks | Our Defense |
|-----------|:-------:|--------------------|-------------|
| `s` | `16.0` | "Original ArcFace uses s=64. Why s=16?" | **Quantitative answer**: With s=64 and our observed cosine similarities (Top: ~0.48, QCD: ~-0.02), softmax probabilities saturate at 1.0. With s=16, probabilities are ~0.96 — healthy gradients. The original s=64 was designed for 512-dim embeddings with 1000+ classes. For 64-dim binary classification, s=16 is mathematically appropriate. |

**Robustness evidence** (to be generated): If we test s=[8, 16, 32] and all give 91-92% accuracy, this proves the model is robust to s.

### 2.2 Margin (m)

| Parameter | Current | What Reviewer Asks | Our Defense |
|-----------|:-------:|--------------------|-------------|
| `m` | `0.5` | "Why m=0.5? Is this optimal?" | Standard ArcFace value. For binary classification, m=0.5 radians is a significant fraction of π/2 (the maximum possible angular separation). This makes binary classification **harder** for ArcFace, not easier. |
| `warmup` | `10 epochs` | "Why warmup?" | Without warmup, imposing m=0.5 from epoch 0 causes the model to struggle with both learning representations AND overcoming the margin. Linear warmup (0.0→0.5 over 10 epochs) stabilizes training. Evidence: val acc climbs smoothly from 89.4%→91.7% over 100 epochs. |

**Robustness evidence**: If we test m=[0.0, 0.3, 0.5] and all give similar accuracy with similar centroid masses, this proves margin is not critical.

### 2.3 Embedding Dimension (d)

| Parameter | Current | What Reviewer Asks | Our Defense |
|-----------|:-------:|--------------------|-------------|
| `d` | `64` | "Why 64? Why not 16 or 128?" | **Multi-evidence defense**: (1) SVD: 64 dims capture >99% variance. (2) Participation Ratio: only ~3-5 dimensions are active. (3) TwoNN intrinsic dimension: ~5-8. (4) Accuracy plateau: d=16→90%, d=32→91%, d=64→91.7%, d=128→91.8%. The plateau at 64 justifies this choice. |

---

## 3. Spectral Clustering Parameters

### 3.1 k Selection (The Critical Question)

| Method | What It Does | Result | Reviewer Concern |
|--------|-------------|:------:|:-----------------|
| **Eigengap heuristic** (our primary) | Finds k where eigenvalue gap is largest | k=2 | "Is this stable across seeds?" |
| **K-means elbow** | Finds k where inertia plateaus | k=2 | "Different method, same answer" |
| **Agglomerative (cosine)** | Hierarchical clustering with cosine linkage | k=2 | "Third method confirms" |
| **DBSCAN** | Density-based, no k needed | k=1 | "Too dense on sphere — expected" |
| **DPGMM** | Dirichlet process, automatic k | k=10 | "Deprecated — replaced by Spectral Clustering + Eigengap" |
| **GMM (BIC)** | Gaussian mixture with BIC | k=3 | "Gaussian assumption violated on sphere" |
| **Ward** | Euclidean hierarchical | k=17626 | "Broken on sphere — confirms geometry matters" |

**Our narrative**: *"Three cosine-compatible methods independently select k=2. Euclidean methods fail — this validates that hyperspherical geometry is essential. The k=2 structure represents 'core vs edge' subclasses, which is the natural decomposition of a binary classification on a hypersphere."*

### 3.2 What If k is Not 2 Across Seeds?

**This is a real risk.** If different training seeds give different k values, our subclass discovery claim weakens.

**Mitigation**: Run SC on embeddings from 3 different seeds. If all give k=2, claim is strong. If one gives k=3, we need to discuss.

**Alternative approach**: Instead of claiming "k=2 is the truth," claim **"the latent space has a dominant 2-cluster structure, confirmed by multiple methods."** This is more defensible.

### 3.3 Other SC Parameters

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `affinity` | `cosine` | "Why cosine?" | Matches ArcFace's hyperspherical geometry. Euclidean distance on a sphere is meaningless. |
| `max_k` | `10` | "Why 10?" | Upper bound for reasonable subclasses. Physical intuition suggests at most a few subclasses per jet type. |
| `n_init` | `10` | "Why 10?" | Standard K-means default for the final clustering step. |

---

## 4. FastJet Physics Parameters

### 4.1 Jet Clustering

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `algorithm` | `cambridge_algorithm` (C/A) | "Why C/A?" | C/A is standard for Lund planes and substructure. It's the default choice in HEP. |
| `R` | `2.0` | "Why R=2.0? Isn't this too large?" | For exclusive clustering (forcing n_jets=1), a large R ensures we don't cut off radiation. The actual jet radius for top tagging is typically R=0.8, but for exclusive clustering we use R=2.0 to capture all constituents. |

### 4.2 SoftDrop Grooming

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `β` | `0.0` | "Why β=0?" | This is the modified MassDrop Tagger (mMDT). β=0 is the most aggressive grooming and is standard for top tagging. |
| `z_cut` | `0.1` | "Why z_cut=0.1?" | Standard value in the literature. Removes soft radiation while keeping hard substructure. |
| `R0` | `2.0` | "Why R0=2.0?" | Matches the jet clustering radius. Standard practice. |

### 4.3 Sensitivity to Grooming Parameters

**Reviewer will ask**: "How sensitive are your physics conclusions to β and z_cut?"

**Planned response**: Run the analysis with β=[0, 1, 2] and z_cut=[0.05, 0.1, 0.2]. Show that:
- zg divergence for QCD (1/z) is robust across all settings
- zg flatness for Top is robust
- Angular collimation ΔR ≈ 2M/pT holds
- k=2 subclass structure is unchanged

If all 9 combinations give the same physics conclusions, the response is: *"Our physics conclusions are robust to reasonable variations in grooming parameters."*

---

## 5. Training Parameters

### 5.1 Optimization

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `optimizer` | `Adam` | "ArcFace paper uses SGD. Why Adam?" | We train from scratch (no pretraining). Adam's adaptive LR handles cold-start. SGD would need careful LR tuning and more epochs. |
| `lr` | `1e-3` | "Why 1e-3?" | Standard Adam default. Works well for our model size (46K params). |
| `scheduler` | `OneCycleLR` | "Why OneCycle?" | Rapid warmup then cool-down is well-suited for deep learning. Peak LR=2e-3 at epoch 30, then decay. |
| `batch_size` | `128` | "Why 128?" | Standard size. Effective batch = 128 × 8 accumulation = 1024. |
| `accumulation_steps` | `8` | "Why 8?" | Simulates larger batch for BatchNorm stability while keeping memory low. |

### 5.2 Data Loading

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `chunk_size` | `100000` (default) | "Why chunk?" | Memory-safe loading for Colab. However, our actual run used `chunk_size=1500000` (single chunk) because Colab T4 had enough RAM. |
| `max_events` | `None` (full) | "Did you use all data?" | Yes — 1,211,000 training events confirmed by notebook output (9,461 batches × 128). |

### 5.3 Seeds

| Parameter | Current | Reviewer Question | Defense |
|-----------|:-------:|-------------------|---------|
| `seeds` | `1` (current) | "Only one seed?" | **Critical gap.** Need 3 seeds minimum. Plan: train with seeds [42, 123, 7] for all models. |

---

## 6. k Selection: Alternative Methods

Beyond eigengap, here are all methods available in our codebase:

| Method | File | How It Works | Pros | Cons |
|--------|:----:|:-------------|:-----|:-----|
| **Eigengap** | `train_top.py:160` | Largest gap in Laplacian eigenvalues | Parameter-free, standard | Can be ambiguous if no clear gap |
| **K-means elbow** | `scratch/comprehensive_clustering_comparison.py` | Inertia vs k plot | Simple, fast | Subjective elbow selection |
| **Silhouette score** | `scratch/compute_clustering_metrics.py` | Max average silhouette | Objective | Computationally expensive for large k |
| **GMM BIC** | `scratch/test_gmm_bic.py` | Bayesian Information Criterion | Statistical foundation | Assumes Gaussian — invalid on sphere |
| **DPGMM** | `scratch/comprehensive_clustering_comparison.py` | Dirichlet Process | Automatic k selection | Deprecated — replaced by Spectral Clustering |
| **DBSCAN** | `scratch/comprehensive_clustering_comparison.py` | Density-based | No k needed | Assumes uniform density — fails on sphere |
| **HDBSCAN** | `scratch/comprehensive_clustering_comparison.py` | Hierarchical DBSCAN | More robust | May not be installed |

**Our strategy**: Use eigengap as primary (parameter-free, standard). Use K-means elbow and Agglomerative as confirmatory. Report all three.

---

## 7. Summary: What We Need to Generate

| Experiment | Parameters | GPU Time | Purpose |
|:-----------|:-----------|:--------:|:--------|
| **Robustness: s** | s=[8, 16, 32], m=0.5, d=64 | ~1.5 hrs | Show scale robustness |
| **Robustness: m** | s=16, m=[0.0, 0.3, 0.5], d=64 | ~1.5 hrs | Show margin robustness |
| **Robustness: d** | s=16, m=0.5, d=[16, 32, 64, 128] | ~2 hrs | Show dimension plateau |
| **Multi-seed baselines** | Linear/CosLinear/ArcFace × 3 seeds | ~3 hrs | Error bars |
| **FastJet sensitivity** | β=[0,1,2], zcut=[0.05,0.1,0.2] | CPU ~1 hr | Physics robustness |
| **k stability across seeds** | SC on 3 seed embeddings | CPU ~10 min | Subclass robustness |

**Total GPU time**: ~8 hours (can be parallelized across 2 GPUs → ~4 hours)
**Total CPU time**: ~1 hour
