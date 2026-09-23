# EFN–ArcFace Top Tagging: Results

> Like a paper, but not a paper. Everything below is generated from code in
> `scripts/analysis/` from tracked artifacts in `experiments/*.json`.
> All 48 reported numbers are machine-checked
> (`experiments/numbers.json` + `scripts/paper_checks/run_all.py`).
> Figures are collapsed by section — click any header to expand.

## Abstract

Top-quark vs QCD jet classification with a geometrically readable decision rule:
an IRC-safe Energy Flow Network backbone (Φ [2→128→128→128], energy-weighted
sum pool, ρ [128→128→64]; 59,072 parameters total) plus an ArcFace prototype
head (margin m=0.5, scale s=16, seed 42), so classification is exactly
nearest-prototype on the unit hypersphere S⁶³. Trained on 1.2M jets, tested on
404k (Top Tagging Reference Dataset). Unsupervised clustering on the manifold
recovers 4 subclasses (Core/Edge × Top/QCD), 3 of 4 validated against QCD
predictions (splitting functions, angular scaling, Lund topology); Top-Edge is
non-identifiable and excluded from physics reading.

## Method (one paragraph)

Constituents (E, px, py, pz) → jet-centering → Φ sees only (Δη, Δφ), energy
fraction z = E/ΣE weights the sum pool (IRC safety by construction) → ρ maps to
a 64-D embedding → L2-normalized onto S⁶³ → ArcFace prototypes W_QCD, W_Top
define classification by geodesic distance. Baselines: EFN+Linear and
EFN+CosLinear (3 seeds each) isolate hypersphere vs margin effects; random
forest on 10 hand-crafted substructure observables. Controls: Gaussian-noise
jets, untrained encoder, label-shuffled retraining.

## Results

### Classification (404k test)

| Model | Accuracy | ROC AUC | Rejection@50%TPR |
|:------|:--------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | **91.84% [91.76, 91.92]** | 0.974 | **124×** |
| EFN + Linear (3-seed mean) | 91.94% ±0.01 | 0.974 | 122× |
| EFN + CosLinear (3-seed mean) | 92.00% ±0.03 | 0.975 | 132× |
| RF (10 substructure features) | 92.37% | 0.977 | 150× |

Scale check: s=22 → 91.81% mean (91.82/91.80/91.82) vs s=16 91.84% — within
noise; s=16 kept for stability. McNemar (ArcFace vs CosLinear): χ²=32.3,
p=1.3e-8; seed spread 0.05pp < 0.16pp gap, same direction all seeds.

### Subclasses (spectral clustering per predicted class, eigengap → k=2 × 2)

| Subclass | Mass | τ32 | Reading |
|:---------|:----:|:---:|:--------|
| Top Core | ~181 GeV | ~0.63 | boundary mixture (absorbs 93% of false positives) |
| Top Edge | ~174 GeV | ~0.46 | diffuse tail — fit non-identifiable, **excluded** |
| QCD Core | ~62 GeV | — | ordinary single-prong |
| QCD Edge | ~134 GeV | — | hard gluon mimicking top (holds 99% of false negatives) |

Predicted-vs-truth partitions agree: ARI 0.79 global (0.95 Top, 0.88 QCD).

### DGLAP splitting-function fits (SoftDrop z_g → A·z^α(1−z)^β)

| Subclass | α | R² |
|:---------|:--:|:--:|
| QCD-Core | −1.242 | 0.998 |
| QCD-Edge | −1.022 | 0.997 |
| Top-Core | −0.493 | 0.960 |
| Top-Edge | +1.105 | 0.988 (reported, excluded) |

Ordering QCD-steeper-than-Top matches the soft 1/z_g tail (quark and gluon both
give it after z_g<1/2 folding; differ only by C_F vs C_A). −1.24 steeper than
−1 is consistent with running coupling + Sudakov + grooming (Cal et al.).

### Geometry (3-head comparison)

TwoNN: ArcFace 2.21 vs CosLinear 4.98 vs Linear 5.19 — hypersphere collapses to
~2-D (PCA 83.6% + 16.4%). Prototypes 0.76 rad (43.6°) apart. Margin m=0.5
specifically resolves Top Core vs Edge; hypersphere alone (CosLinear) keeps
balanced subclasses.

## Interpretation

The model rediscovers mass + 3-prong substructure from raw (Δη, Δφ) inputs —
nothing hand-fed (unlike the RF). PySR distills tanh(m/pT) − 0.348·τ32
(leading-physics distillate, not a replacement: rejection 33× vs 124×).
Score-vs-mass is gradual, not a cutoff — the embedding angle carries
complementary substructure. OOD (JetNet30): W/Z sit near the QCD prototype,
JetNet top maps to neither (30 vs ≤200 constituents shift) with honest low
confidence on both prototypes — hypothesis-generating, not a verdict.

## Conclusion

A 59k-parameter tagger at 91.84% whose decision rule can be drawn, audited,
and argued with: 4 subclasses found, 3 validated against QCD, Top-Edge
honestly excluded. Limitations (stated, not hidden): simulation-only, Pythia 8
only, binary task, narrow pT window (angular scaling ≈ mass readout),
single-seed label shuffle, exploratory p-values.

## Reproduce

```bash
pip install -e .                                   # or docker-compose up --build
python scripts/download_data.py                    # 3.4 GB reference dataset
python scripts/train_arcface.py --scale 16 --seed 42
python scripts/paper_checks/run_all.py             # 48/48 numbers, refs, floats, build
```

---

## Figure gallery (click to expand — all 42 figures)

<details>
<summary><b>Pipeline</b> — architecture schematic</summary>

![Architecture](../paper/figures/architecture_schematic.png)
EFN ([2→128→128→128] Φ, energy-weighted sum pool, [128→128→64] ρ) → L2-normalized
64-D embedding on S⁶³ → ArcFace prototypes (m=0.5, s=16). 59,072 parameters total.

</details>

<details>
<summary><b>Classification</b> — ROC, rejection, calibration, mass vs score (4)</summary>

![ROC](../paper/figures/final_roc_curves.png)
ROC curves: EFN+ArcFace AUC 0.974, accuracy 91.84% [91.76, 91.92]; RF leads at 92.37%.

![Rejection](../paper/figures/final_rejection_rates.png)
Rejection at 50% signal efficiency: ArcFace 124×, RF 150×, symbolic c=6 distillate 33×.

![POC](../paper/figures/poc_combined.png)
Proof-of-concept: score calibration and injection-scan stability.

![Mass vs score](../paper/figures/mass_vs_score_scatter.png)
Model score vs jet mass — gradual transition, not a sharp mass cutoff; misclassified
events sit in the mass-overlap region (physical ambiguity, not just model error).

</details>

<details>
<summary><b>Subclass discovery</b> — eigengap, dendrogram, composition, physics (4)</summary>

![Eigengap](../paper/figures/eigengap_spectrum.png)
Eigenvalue spectra + eigengaps: largest gap at k=2 for both QCD and Top partitions.

![Dendrogram](../paper/figures/hc_dendrogram.png)
Ward hierarchical clustering dendrogram supporting the 2-way split.

![Composition](../paper/figures/subclass_composition_heatmap.png)
Subclass composition across prediction categories (TP/FP/FN/TN).

![Subclass physics](../paper/figures/subclass_physics_distributions.png)
Mass / τ32 / zg distributions per subclass: the four groups separate physically.

</details>

<details>
<summary><b>DGLAP fits</b> — splitting-function power laws (1)</summary>

![DGLAP fits](../paper/figures/dglap_fit.png)
Power-law fits A·z^α(1−z)^β: QCD-Core α=−1.242, QCD-Edge −1.022, Top-Core −0.493,
Top-Edge +1.105 (non-identifiable, excluded from physics reading).

</details>

<details>
<summary><b>Angular collimation</b> — θg ≈ 2m/pT scaling (1)</summary>

![Angular](../paper/figures/angular_collimation.png)
Subclass means follow θg ≈ 2m/pT scaling (narrow pT window: readout of mass, stated as limitation).

</details>

<details>
<summary><b>Lund planes</b> — status + per-subclass slices (10)</summary>

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

</details>

<details>
<summary><b>Embedding geometry</b> — 3-head comparison (5)</summary>

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

</details>

<details>
<summary><b>Δη–Δφ energy-flow maps</b> — class + subclass averages (10)</summary>

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

</details>

<details>
<summary><b>Symbolic distillation</b> — Pareto front (1)</summary>

![Pareto](../paper/figures/pareto_front_sr.png)
Pareto front: compact c=6 form tanh(m/pT) − 0.348·τ32 (leading-physics distillate,
not a replacement — rejection 33× vs 124× full model).

</details>

<details>
<summary><b>Out-of-distribution</b> — JetNet30 prototype similarity (1)</summary>

![OOD](../paper/figures/poc_ood_similarity.png)
Cosine similarity to prototypes per JetNet class: W/Z near QCD prototype,
JetNet top maps to neither (30 vs ≤200 constituents shift — expected, honest low confidence).

</details>

<details>
<summary><b>Dataset EDA</b> — appendix plots (4)</summary>

![EDA overview](../paper/figures/eda_figA1_dataset_overview.png)
Dataset overview: class balance, kinematics ranges.

![EDA kinematics](../paper/figures/eda_figA2_kinematics.png)
Kinematic distributions (pT, mass, multiplicity).

![EDA substructure](../paper/figures/eda_figA3_substructure.png)
Substructure observables (τ21, τ32, zg, θg, nSD).

![EDA correlation](../paper/figures/eda_figA4_correlation.png)
Observable correlation matrix (mass confounding visible here first).

</details>
