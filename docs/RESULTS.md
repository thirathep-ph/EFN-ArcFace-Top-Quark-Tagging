# EFN–ArcFace Top Tagging: complete results in one page

Paper-like, not a paper: abstract, tables, figures, interpretation, conclusion.
No filler. Every number below is machine-checked against `experiments/numbers.json`
(48 entries) via `scripts/paper_checks/run_all.py`. Figures are generated from
tracked artifacts by `scripts/analysis/`.

## Abstract

Top-vs-QCD jet classification with a geometrically readable decision rule: an
IRC-safe Energy Flow Network backbone (59,072 params) plus an ArcFace prototype
head, so classification is exactly nearest-prototype on the unit hypersphere
S⁶³. On 404k test jets: **91.84% accuracy [91.76, 91.92], AUC 0.974, 124×
rejection at 50% signal efficiency**. Unsupervised clustering on the manifold
recovers 4 subclasses (Core/Edge × Top/QCD); 3 of 4 validate against QCD
predictions (splitting-function exponents, angular scaling, Lund topology).
The Top-Edge fit is non-identifiable and excluded from physics reading.

## Method (one paragraph)

Constituents → EFN Φ [2→128→128→128] on (Δη, Δφ) only, energy-weighted sum
pool, ρ [128→128→64] → L2-normalized 64-D embedding → ArcFace prototypes
(m=0.5, s=16, seed 42). Trained on 1.2M jets (Top Tagging Reference Dataset).
Controls: EFN+Linear and EFN+CosLinear heads (3 seeds each) isolate hypersphere
vs margin effects; random forest on 10 hand-crafted observables as classical
ceiling; null controls (noise jets, untrained encoder) + one label-shuffled
retraining.

![Architecture](../paper/figures/architecture_schematic.png)
EFN ([2→128→128→128] Φ, energy-weighted sum pool, [128→128→64] ρ) → L2-normalized
64-D embedding on S⁶³ → ArcFace prototypes (m=0.5, s=16). 59,072 parameters total.

## Results

### Classification (404k test)

| Model | Accuracy | ROC AUC | Rejection@50%TPR |
|:------|:--------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | **91.84% [91.76, 91.92]** | 0.974 | **124×** |
| EFN + Linear (3-seed mean) | 91.94% ±0.01 | 0.974 | 122× |
| EFN + CosLinear (3-seed mean) | 92.00% ±0.03 | 0.975 | 132× |
| RF (10 substructure features) | 92.37% | 0.977 | 150× |

ArcFace trails the best EFN baseline by 0.16pp (McNemar χ²=32.3, p=1.3e-8 —
real gap, small size); the RF on engineered features leads outright.

![ROC](../paper/figures/final_roc_curves.png)
ROC curves: EFN+ArcFace AUC 0.974, accuracy 91.84% [91.76, 91.92]; RF leads at 92.37%.

![Rejection](../paper/figures/final_rejection_rates.png)
Rejection at 50% signal efficiency: ArcFace 124×, RF 150×, symbolic c=6 distillate 33×.

![POC](../paper/figures/poc_combined.png)
Proof-of-concept: score calibration and injection-scan stability.

![Mass vs score](../paper/figures/mass_vs_score_scatter.png)
Model score vs jet mass — gradual transition, not a sharp mass cutoff; misclassified
events sit in the mass-overlap region (physical ambiguity, not just model error).

### Subclass discovery (spectral clustering per predicted class, k\*=2 × 2)

| Subclass | Mass | τ32 | Reading |
|:---------|:----:|:---:|:--------|
| QCD-Core | ~62 GeV | — | ordinary single-prong |
| QCD-Edge | ~134 GeV | — | hard gluon mimicking top (holds 99% of false negatives) |
| Top-Core | ~181 GeV | ~0.63 | boundary mixture (absorbs 93% of false positives) |
| Top-Edge | ~174 GeV | ~0.46 | 98.5% pure but diffuse tail — fit non-identifiable, excluded |

Predicted-vs-truth partitions agree (ARI 0.79; 0.95 Top, 0.88 QCD). Null
controls also return k\*=2 — so k\*=2 alone proves nothing; the physics case
rests on the validation battery below, which a pure geometry artifact would
not pass.

![Eigengap](../paper/figures/eigengap_spectrum.png)
Eigenvalue spectra + eigengaps: largest gap at k=2 for both QCD and Top partitions.

![Dendrogram](../paper/figures/hc_dendrogram.png)
Ward hierarchical clustering dendrogram supporting the 2-way split.

![Composition](../paper/figures/subclass_composition_heatmap.png)
Subclass composition across prediction categories (TP/FP/FN/TN).

![Subclass physics](../paper/figures/subclass_physics_distributions.png)
Mass / τ32 / zg distributions per subclass: the four groups separate physically.

### DGLAP splitting-function fits (A·z^α(1−z)^β on SoftDrop zg)

| Subclass | α | R² | Reading |
|:---------|:--:|:--:|:--------|
| QCD-Core | −1.242 | 0.998 | soft 1/zg tail, steeper than −1 (running coupling + Sudakov + grooming) |
| QCD-Edge | −1.022 | 0.997 | harder, more symmetric splittings |
| Top-Core | −0.493 | 0.960 | suppressed small-zg (W→qq̄ lacks soft singularity) |
| Top-Edge | +1.105 | 0.988 | amplitude-bound saturated — non-identifiable, excluded |

Quark-initiated jets share the same 1/zg soft tail (q→qg differs only by color
factor CF=4/3 vs CA=3), so the fit measures the common tail slope. Ground-truth
partition agrees within ±0.17.

![DGLAP fits](../paper/figures/dglap_fit.png)
Power-law fits A·z^α(1−z)^β: QCD-Core α=−1.242, QCD-Edge −1.022, Top-Core −0.493,
Top-Edge +1.105 (non-identifiable, excluded from physics reading).

### Angular collimation

![Angular](../paper/figures/angular_collimation.png)
Subclass means follow θg ≈ 2m/pT scaling (narrow pT window: readout of mass, stated as limitation).

### Lund planes (qualitative consistency)

![Lund TP](../paper/figures/lund_status_tp.png)
True-Top Lund plane: clear 3-prong structure.

![Lund TN](../paper/figures/lund_status_tn.png)
True-QCD Lund plane: single-prong radiation pattern.

![Lund FP](../paper/figures/lund_status_fp.png)
False positives: top-like QCD (high-mass tail).

![Lund FN](../paper/figures/lund_status_fn.png)
False negatives: QCD-like tops in the overlap region.

![Lund QCD-Core](../paper/figures/lund_subclass_qcd_core.png)
QCD-Core Lund plane.

![Lund QCD-Edge TN](../paper/figures/lund_subclass_qcd_edge_tn.png)
QCD-Edge (true-negative slice) Lund plane.

![Lund QCD-Edge FP](../paper/figures/lund_subclass_qcd_edge_fp.png)
QCD-Edge (false-positive slice) Lund plane.

![Lund Top-Core TP](../paper/figures/lund_subclass_top_core_tp.png)
Top-Core (true-positive slice) Lund plane.

![Lund Top-Core FN](../paper/figures/lund_subclass_top_core_fn.png)
Top-Core (false-negative slice) Lund plane.

![Lund Top-Edge](../paper/figures/lund_subclass_top_edge.png)
Top-Edge Lund plane (diffuse tail, excluded from physics reading).

### Embedding geometry (3-head comparison)

Hypersphere alone (CosLinear, m=0) keeps balanced subclasses; margin m=0.5
specifically resolves Top Core vs Edge. TwoNN: ArcFace 2.21 vs 4.98/5.19
controls. PCA: PC1 83.6% + PC2 16.4% ≈ 100% (2-D collapse). Prototype
separation 43.6° (0.76 rad) vs margin 0.5 rad.

![Hypersphere](../paper/figures/hypersphere_3d.png)
Embeddings collapse onto a 2-D disk spanned by the two prototypes.

![PCA](../paper/figures/pca_scatter_3way.png)
PCA scatter: PC1 83.6% + PC2 16.4% ≈ 100% (2-D collapse).

![Embedding geometry](../paper/figures/embedding_geometry_3way.png)
Embedding geometry across ArcFace / CosLinear / Linear heads.

![Head geometry](../paper/figures/head_geometry_3way.png)
Head comparison: hypersphere alone (CosLinear) keeps balanced subclasses;
margin m=0.5 specifically resolves Top Core vs Edge.

![Dimensionality](../paper/figures/dimensionality_3way.png)
TwoNN + participation ratio per head: ArcFace TwoNN 2.21 vs 4.98/5.19 controls.

### Δη–Δφ energy-flow maps

![Class QCD](../paper/figures/deta_dphi_class_qcd_rep.png)
Representative QCD energy-flow map.

![Class Top](../paper/figures/deta_dphi_class_top_rep.png)
Representative Top energy-flow map (3 deposits).

![QCD-Core avg](../paper/figures/deta_dphi_subclass_qcd_core_avg.png)
QCD-Core average map.

![QCD-Core rep](../paper/figures/deta_dphi_subclass_qcd_core_rep.png)
QCD-Core representative map.

![QCD-Edge avg](../paper/figures/deta_dphi_subclass_qcd_edge_avg.png)
QCD-Edge average map.

![QCD-Edge rep](../paper/figures/deta_dphi_subclass_qcd_edge_rep.png)
QCD-Edge representative map.

![Top-Core avg](../paper/figures/deta_dphi_subclass_top_core_avg.png)
Top-Core average map.

![Top-Core rep](../paper/figures/deta_dphi_subclass_top_core_rep.png)
Top-Core representative map.

![Top-Edge avg](../paper/figures/deta_dphi_subclass_top_edge_avg.png)
Top-Edge average map.

![Top-Edge rep](../paper/figures/deta_dphi_subclass_top_edge_rep.png)
Top-Edge representative map.

### Symbolic distillation (PySR, leading physics only)

![Pareto](../paper/figures/pareto_front_sr.png)
Pareto front: compact c=6 form tanh(m/pT) − 0.348·τ32 (leading-physics distillate,
not a replacement — rejection 33× vs 124× full model).

### Out-of-distribution (JetNet30, hypothesis-generating)

![OOD](../paper/figures/poc_ood_similarity.png)
Cosine similarity to prototypes per JetNet class: W/Z near QCD prototype,
JetNet top maps to neither (30 vs ≤200 constituents shift — expected, honest low confidence).

### Dataset EDA

![EDA overview](../paper/figures/eda_figA1_dataset_overview.png)
Dataset overview: class balance, kinematics ranges.

![EDA kinematics](../paper/figures/eda_figA2_kinematics.png)
Kinematic distributions (pT, mass, multiplicity).

![EDA substructure](../paper/figures/eda_figA3_substructure.png)
Substructure observables (τ21, τ32, zg, θg, nSD).

![EDA correlation](../paper/figures/eda_figA4_correlation.png)
Observable correlation matrix (mass confounding visible here first).

## Interpretation

- The model rediscovers mass + 3-prong substructure from raw (Δη, Δφ) +
  energy alone — features the RF is handed pre-computed. That is the whole
  point: same physics, no hand-crafted inputs, plus a drawable decision rule.
- The 4-subclass split is recovery, not discovery: QCD Core–Edge separation
  is substantially mass-confounded (mass-matched residual only suggestive);
  what is robust is the ordering (QCD steeper than Top) across methods.
- Prototype geometry is honest under shift: OOD inputs get low confidence to
  both prototypes instead of confident misclassification.

## Conclusion

A 59k-parameter EFN+ArcFace tagger reaches 91.84% with a nearest-prototype
decision rule; its 2-D latent manifold splits into 4 subclasses, 3 of which
check out against QCD predictions. It does not beat a feature-based RF
(92.37%) and does not claim to — the contribution is the readable,
checkable representation. Simulation-only (Pythia 8), binary only, no
detector effects; cross-generator and detector-level validation are open.

## Reproduce

```bash
pip install -e .                                   # core library
python scripts/train_arcface.py --scale 16         # 50 epochs (or --quick)
python scripts/paper_checks/run_all.py             # verify all 48 numbers
```

Data (2 GB H5, not tracked): `python scripts/download_data.py`.
