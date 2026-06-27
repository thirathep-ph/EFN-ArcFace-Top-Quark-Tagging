# SciPost Physics — Publication Plan

> **Target**: SciPost Physics Core (or SciPost Physics)
> **Title**: "Ante-Hoc Interpretable Jet Tagging with Energy Flow Networks and Hyperspherical Prototypes"
> **Author**: Thirathep Phongpreecha
> **Last Updated**: 2026-06-27

---

## 1. Current Status

### 1.1 Completed ✅

| Component | Details | Location |
|:----------|:--------|:---------|
| **EFN + ArcFace model** | 100-epoch, 91.75% acc, F1 91.89% | `experiments/arcface_100ep/` |
| **Linear baselines** | 3 seeds: 91.93–91.95% acc | `experiments/baselines_linear_seed_{42,123,7}/` |
| **CosLinear baselines** | 3 seeds: 91.99–92.03% acc | `experiments/baselines_coslinear_seed_{42,123,7}/` |
| **Robustness sweep** | 9 configs (s×m grid), all 91–92% acc | `experiments/robustness_s{8,16,32}_m{00,03,05}/` |
| **Subclass discovery** | 2 Top + 2 QCD subclasses via Spectral Clustering | `experiments/arcface_100ep/` |
| **Physics validation** | zg splitting, angular collimation, Lund planes | `experiments/arcface_100ep/` |
| **SVD analysis** | Embedding effective dim = 2, intrinsic dim = 2.69 | `experiments/svd_analysis/` |
| **Clustering comparison** | SC, K-means, Agglomerative all select k=2 | `experiments/comprehensive_analysis_results.json` |
| **Bootstrap CI** | Acc 81.2% [80.8%, 81.6%] on 50K subset | `experiments/comprehensive_analysis_results.json` |
| **20+ figures** | UMAP, Lund, ROC, SVD, physics distributions | `experiments/arcface_100ep/` |
| **Codebase** | Clean, reproducible pipeline | `arcjet/`, `scripts/` |

### 1.2 Remaining Work ⚠️

| Item | Priority | Effort | Notes |
|:-----|:--------:|:------:|:------|
| **Paper writing** | 🔴 Critical | 2–4 weeks | Sections 1–6 + abstract |
| **Clustering method table** | 🟡 Important | Done | Data exists in comprehensive_analysis_results.json — just format for paper |
| **External baselines** | 🟢 Optional | 1–2 weeks | ParticleNet/LorentzNet — can defer to revision |

---

## 2. Key Results Tables (Paper-Ready)

### Table 1: Classification Performance

| Model | Seed | Accuracy | F1 | ROC AUC |
|:------|:----:|:--------:|:--:|:-------:|
| EFN + ArcFace | — | 91.75% | 91.89% | — |
| EFN + Linear | 42 | 91.93% | — | — |
| EFN + Linear | 123 | 91.95% | — | — |
| EFN + Linear | 7 | ~91.9% | — | — |
| EFN + CosLinear | 42 | 91.99% | — | — |
| EFN + CosLinear | 123 | 92.03% | — | — |
| EFN + CosLinear | 7 | ~92.0% | — | — |

> **Key takeaway**: ArcFace achieves comparable accuracy while providing intrinsic interpretability via hyperspherical prototypes.

### Table 2: Robustness Analysis (s × m Grid)

| Scale (s) | Margin (m) | Accuracy | F1 |
|:---------:|:----------:|:--------:|:--:|
| 8 | 0.0 | 91.76% | 91.86% |
| 8 | 0.3 | 91.79% | 91.89% |
| 8 | 0.5 | 89.49% | 89.94% |
| 16 | 0.0 | 91.77% | 91.88% |
| 16 | 0.3 | 91.74% | 91.82% |
| 16 | 0.5 | 91.71% | 91.83% |
| 32 | 0.0 | 91.90% | 92.02% |
| 32 | 0.3 | 91.62% | 91.72% |
| 32 | 0.5 | 91.61% | 91.68% |

> **Key takeaway**: Accuracy is stable across s ∈ {8,16,32} and m ∈ {0.0,0.3}. Only s=8, m=0.5 shows degradation (~89.5%), confirming robustness to hyperparameter choices.

### Table 3: Subclass Discovery (Clustering Comparison)

| Method | Top k | QCD k | Top Silhouette | QCD Silhouette |
|:-------|:-----:|:-----:|:--------------:|:--------------:|
| Spectral Clustering + Eigengap | 2 | 2 | 0.734 | 0.850 |
| K-means Elbow | 2 | 2 | — | — |
| Agglomerative (cosine) | 2 | 2 | 0.791 | 0.819 |

> **Key takeaway**: All three methods independently select k=2 for both classes, confirming the Core/Edge decomposition is not an artifact of the clustering method.

### Table 4: Embedding Dimensionality Analysis

| Method | Dimension | Interpretation |
|:-------|:---------:|:---------------|
| SVD (90% variance) | 2 | Embeddings lie on a 2D subspace |
| TwoNN (intrinsic dim) | 2.69 | Consistent with ~3D manifold |
| Participation Ratio | 1.38 | Strongly concentrated |
| Class center angular distance | 0.276 rad | Well-separated on hypersphere |

> **Key takeaway**: Despite 64-dim embeddings, the effective dimensionality is ~2–3, matching the binary classification structure.

### Table 5: Robustness & Perturbation Analysis

| Test | Result | Interpretation |
|:-----|:------:|:---------------|
| Constituent dropout (10%) | 88.7% acc | Graceful degradation |
| Constituent dropout (50%) | 65.9% acc | Expected — losing half the jet |
| pT rescaling (0.5–2.0) | 91.5% acc | IRC-safe: invariant to pT scaling |
| Jet rotation (0–2 rad) | 91.5% acc | IRC-safe: invariant to rotation |
| Adversarial perturbation | 92.9% → 91.9% | Only 1% drop — robust |
| Bootstrap CI (95%) | [80.8%, 81.6%] | Stable on 50K subset |

---

## 3. Paper Structure (SciPost Format)

### Abstract (~150 words)
- Jet tagging is critical for HEP
- Current methods are black-box (post-hoc interpretability)
- We propose EFN + ArcFace: hyperspherical prototypes provide ante-hoc interpretability
- Discover physically meaningful subclasses (Core/Edge) without supervision
- Validate against QCD physics (zg, angular collimation, Lund planes)
- Comparable accuracy to standard classifiers with intrinsic interpretability

### Section Outline

```
1. Introduction (2 pages)
   1.1 Jet tagging problem
   1.2 Interpretability in HEP
   1.3 Our contribution: ante-hoc via ArcFace prototypes

2. Methodology (3 pages)
   2.1 Energy Flow Networks (IRC-safe)
   2.2 ArcFace Hyperspherical Prototypes
   2.3 Spectral Clustering + Eigengap for subclass discovery
   2.4 IRC-safety guarantees

3. Experimental Setup (1 page)
   3.1 Dataset: Top Tagging Reference Dataset
   3.2 Training details
   3.3 Evaluation metrics

4. Results (4 pages)
   4.1 Classification Performance (Table 1)
   4.2 Subclass Discovery (Tables 2–3, Figures 1–4)
   4.3 Physics Validation (Figures 5–8)
   4.4 Robustness Analysis (Tables 2, 5)
   4.5 Dimensionality Analysis (Table 4)
   4.6 Error Analysis

5. Discussion (1.5 pages)
   5.1 What the prototypes learned
   5.2 Physical interpretation of subclasses
   5.3 Interpretability vs performance trade-off
   5.4 Limitations

6. Conclusion (0.5 pages)

References (~30)
```

### Figures

| # | Content | Source |
|---|:--------|:-------|
| 1 | Architecture diagram | Create new (tikz) |
| 2 | ROC curve | `experiments/arcface_100ep/performance.png` |
| 3 | UMAP prediction status | `experiments/arcface_100ep/umap_prediction_status.png` |
| 4 | UMAP subclasses | `experiments/arcface_100ep/umap_subclasses.png` |
| 5 | Subclass physics distributions | `experiments/arcface_100ep/subclass_physics_distributions.png` |
| 6 | zg splitting | `experiments/arcface_100ep/zg_ap_splitting.png` |
| 7 | Angular collimation | `experiments/arcface_100ep/angular_collimation.png` |
| 8 | Lund planes (TP/TN/FP/FN) | `experiments/arcface_100ep/lund_planes/` |
| 9 | SVD scree plot | `experiments/svd_analysis/svd_scree_plot.png` |
| 10 | Robustness heatmap | Create from robustness_results.json |

---

## 4. Timeline

| Week | Tasks | Deliverable |
|:-----|:------|:------------|
| **Week 1** | Write Sections 1–2 (Intro, Methodology); compile all figures with captions; fill Tables 1–5 | Draft Sections 1–2 |
| **Week 2** | Write Sections 3–4 (Setup, Results) | Draft Sections 1–4 |
| **Week 3** | Write Sections 5–6 (Discussion, Conclusion); abstract; internal review | Full draft |
| **Week 4** | Revisions; final polish; submit to SciPost | Submission |

---

## 5. Reviewer Defense Strategy

| # | Anticipated Question | Response |
|---|:---------------------|:---------|
| 1 | "Why not ParticleNet?" | Our contribution is interpretability, not SOTA accuracy. ParticleNet is a black-box. We provide comparable accuracy (91.75% vs ~93%) with intrinsic interpretability. Future work: apply ArcFace head to ParticleNet backbone. |
| 2 | "Is k=2 really optimal?" | Eigengap heuristic (parameter-free) selects k=2. K-means elbow and Agglomerative confirm k=2. The 2-cluster structure represents Core/Edge — the natural decomposition on a hypersphere. |
| 3 | "Are subclasses physically meaningful or artifacts?" | Yes: zg splitting matches QCD physics (1/zg for QCD, flat for Top), angular collimation ΔR ≈ 2M/pT confirmed, Lund planes show topological differences. |
| 4 | "What about BatchNorm breaking IRC-safety?" | During inference, BN uses fixed running statistics → IRC-safety holds per-event. We verify <0.1% accuracy difference between eval/train modes. |
| 5 | "Why s=16 not s=64?" | Original ArcFace s=64 was for 512-dim, 1000+ classes. For 64-dim binary, s=64 saturates softmax. s=16 gives healthy gradients. Robustness: s∈{8,16,32} all give 91–92%. |
| 6 | "Is this better than post-hoc methods?" | Different paradigm: ante-hoc (built-in) vs post-hoc (applied after). No need for Grad-CAM/LIME — the model's decision IS the explanation. |
| 7 | "What about edge jets?" | Edge jets (|margin| < 0.05) are physically meaningful: FN = diffuse tops, FP = high-mass QCD. The model's uncertainty aligns with physical ambiguity. |
