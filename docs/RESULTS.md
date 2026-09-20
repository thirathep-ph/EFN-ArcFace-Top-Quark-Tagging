# Results Gallery — every experiment figure, no paper required

All figures below are generated from code in `scripts/analysis/` from tracked
artifacts in `experiments/*.json`. One line each. Numbers match
`experiments/numbers.json` (48 entries, machine-checked).

## Pipeline

![Architecture](../paper/figures/architecture_schematic.png)
EFN ([2→128→128→128] Φ, energy-weighted sum pool, [128→128→64] ρ) → L2-normalized
64-D embedding on S⁶³ → ArcFace prototypes (m=0.5, s=16). 59,072 parameters total.

## Classification (404k test)

![ROC](../paper/figures/final_roc_curves.png)
ROC curves: EFN+ArcFace AUC 0.974, accuracy 91.84% [91.76, 91.92]; RF leads at 92.37%.

![Rejection](../paper/figures/final_rejection_rates.png)
Rejection at 50% signal efficiency: ArcFace 124×, RF 150×, symbolic c=6 distillate 33×.

![POC](../paper/figures/poc_combined.png)
Proof-of-concept: score calibration and injection-scan stability.

![Mass vs score](../paper/figures/mass_vs_score_scatter.png)
Model score vs jet mass — gradual transition, not a sharp mass cutoff; misclassified
events sit in the mass-overlap region (physical ambiguity, not just model error).

## Subclass discovery (spectral clustering per predicted class, k*=2 × 2)

![Eigengap](../paper/figures/eigengap_spectrum.png)
Eigenvalue spectra + eigengaps: largest gap at k=2 for both QCD and Top partitions.

![Dendrogram](../paper/figures/hc_dendrogram.png)
Ward hierarchical clustering dendrogram supporting the 2-way split.

![Composition](../paper/figures/subclass_composition_heatmap.png)
Subclass composition across prediction categories (TP/FP/FN/TN).

![Subclass physics](../paper/figures/subclass_physics_distributions.png)
Mass / τ32 / zg distributions per subclass: the four groups separate physically.

## DGLAP splitting-function fits

![DGLAP fits](../paper/figures/dglap_fit.png)
Power-law fits A·z^α(1−z)^β: QCD-Core α=−1.242, QCD-Edge −1.022, Top-Core −0.493,
Top-Edge +1.105 (non-identifiable, excluded from physics reading).

## Angular collimation

![Angular](../paper/figures/angular_collimation.png)
Subclass means follow θg ≈ 2m/pT scaling (narrow pT window: readout of mass, stated as limitation).

## Lund planes

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

## Embedding geometry (3-head comparison)

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

## Δη–Δφ energy-flow maps

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

## Symbolic distillation (PySR)

![Pareto](../paper/figures/pareto_front_sr.png)
Pareto front: compact c=6 form tanh(m/pT) − 0.348·τ32 (leading-physics distillate,
not a replacement — rejection 33× vs 124× full model).

## Out-of-distribution (JetNet30, hypothesis-generating)

![OOD](../paper/figures/poc_ood_similarity.png)
Cosine similarity to prototypes per JetNet class: W/Z near QCD prototype,
JetNet top maps to neither (30 vs ≤200 constituents shift — expected, honest low confidence).

## Dataset EDA (appendix)

![EDA overview](../paper/figures/eda_figA1_dataset_overview.png)
Dataset overview: class balance, kinematics ranges.

![EDA kinematics](../paper/figures/eda_figA2_kinematics.png)
Kinematic distributions (pT, mass, multiplicity).

![EDA substructure](../paper/figures/eda_figA3_substructure.png)
Substructure observables (τ21, τ32, zg, θg, nSD).

![EDA correlation](../paper/figures/eda_figA4_correlation.png)
Observable correlation matrix (mass confounding visible here first).
