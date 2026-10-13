# EFN–ArcFace Top Tagging — Results

## Abstract

Top-quark vs QCD jet classification with a geometrically readable decision rule: an IRC-safe Energy Flow Network backbone (59,072 params) plus an ArcFace head that learns one prototype per class, so classifying = nearest class-prototype on the unit hypersphere S⁶³. Tested on 404k jets: **91.84% accuracy [91.76, 91.92], AUC 0.974, 124× rejection**. Subclass structure (4 groups: Dense/Dilute × Top/QCD) was a secondary finding en route — uncovered by unsupervised clustering on the manifold, 3 of 4 validated against QCD predictions (splitting functions, angular scaling, Lund topology). A feature-based random forest still leads (92.37%) — stated openly, not hidden.

<details>
<summary>Pipeline figure</summary>

![Architecture](figures/architecture_schematic.png)
EFN ([2→128→128→128] Φ, energy-weighted sum pool, [128→128→64] ρ) → L2-normalized 64-D embedding on S⁶³ → ArcFace prototypes (m=0.5, s=16, seed 42). Trained on 1.2M jets.

</details>

## 1. Methodology (paper §2)

Jets → centered (Δη, Δφ) + energy weights → EFN → 64-D embedding → nearest of 2 learned class prototypes (primary goal). Subclass structure found en route by spectral clustering per predicted class (eigengap picks k=2).

<details>
<summary>Method figures</summary>

![Class QCD](figures/deta_dphi_class_qcd_rep.png)
Representative QCD energy-flow map (single deposit).

![Class Top](figures/deta_dphi_class_top_rep.png)
Representative Top energy-flow map (3 deposits: b + W→qq̄).

</details>

## 2. Classification performance (paper §3.1)

**Takeaway:** competitive with plain EFN baselines; RF on hand-crafted features leads — the contribution is the readable representation, not accuracy.

<details>
<summary>Table + figures</summary>

| Model | Accuracy (404k test) | ROC AUC | Rejection@50%TPR |
|:------|:--------------------:|:-------:|:----------------:|
| EFN + ArcFace (s=16, m=0.5) | **91.84% [91.76, 91.92]** | 0.974 | **124×** |
| EFN + Linear (3-seed mean) | 91.94% ±0.01 | 0.974 | 122× |
| EFN + CosLinear (3-seed mean) | 92.00% ±0.03 | 0.975 | 132× |
| RF (10 substructure features) | 92.37% | 0.977 | 150× |

Scale check (3 seeds): s=22 → 91.81% mean (91.82/91.80/91.82) vs s=16 91.84% — within noise; s=16 kept for stability. McNemar ArcFace-vs-CosLinear: χ²=32.3, p=1.3e-8 (same test set, so the gap is model, not data luck).

![ROC](figures/final_roc_curves.png)
ROC curves: EFN+ArcFace AUC 0.974; RF leads at 92.37% accuracy.

![Rejection](figures/final_rejection_rates.png)
Rejection at 50% signal efficiency: ArcFace 124×, RF 150×, symbolic c=6 distillate 33×.

![POC](figures/poc_combined.png)
Proof-of-concept: score calibration and injection-scan stability.

![Mass vs score](figures/mass_vs_score_scatter.png)
Model score vs jet mass — gradual transition, not a sharp mass cutoff; misclassified events sit in the mass-overlap region (physical ambiguity, not just model error).

</details>

## 3. Secondary finding: subclass structure & physics check (paper §3.2)

**Takeaway:** secondary finding en route — 4 subclasses found, 3 of 4 validated against QCD; Top-Dense excluded (non-identifiable fit). Predicted-vs-truth partitions agree (ARI 0.79).

<details>
<summary>Table + figures</summary>

| Subclass | Mass | τ32 | Reading |
|:---------|:----:|:---:|:--------|
| QCD-Dense | ~62 GeV | — | ordinary single-prong |
| QCD-Dilute | ~134 GeV | — | hard gluon mimicking top (holds 99% of false negatives) |
| Top-Dilute | ~181 GeV | ~0.63 | boundary mixture (absorbs 93% of false positives) |
| Top-Dense | ~174 GeV | ~0.46 | diffuse tail, fit non-identifiable — excluded |

DGLAP power-law fits A·z^α(1−z)^β: QCD-Dense α=−1.242, QCD-Dilute −1.022, Top-Dilute −0.493 (ordering matches soft-gluon hierarchy; quark jets share the same 1/zg tail modulo color factor). Angular: subclass means follow θg ≈ 2m/pT (narrow-pT readout, stated limitation). One mass-matched bin keeps the Dense–Dilute gap (suggestive, not fully clean).

![Eigengap](figures/eigengap_spectrum.png)
Eigenvalue spectra + eigengaps: largest gap at k=2 for both QCD and Top partitions.

![Dendrogram](figures/hc_dendrogram.png)
Ward hierarchical clustering dendrogram supporting the 2-way split.

![Composition](figures/subclass_composition_heatmap.png)
Subclass composition across prediction categories (TP/FP/FN/TN).

![Subclass physics](figures/subclass_physics_distributions.png)
Mass / τ32 / zg distributions per subclass: the four groups separate physically.

![DGLAP fits](figures/dglap_fit.png)
Power-law fits: QCD-Dense α=−1.242, QCD-Dilute −1.022, Top-Dilute −0.493, Top-Dense +1.105 (excluded).

![Angular](figures/angular_collimation.png)
Subclass means follow θg ≈ 2m/pT scaling (narrow pT window: readout of mass, stated as limitation).

![Lund TP](figures/lund_status_tp.png)
True-Top Lund plane: clear 3-prong structure.

![Lund TN](figures/lund_status_tn.png)
True-QCD Lund plane: single-prong radiation pattern.

![Lund FP](figures/lund_status_fp.png)
False positives: top-like QCD (high-mass tail).

![Lund FN](figures/lund_status_fn.png)
False negatives: QCD-like tops in the overlap region.

![Lund QCD-Dense](figures/lund_subclass_qcd_dense.png)
QCD-Dense Lund plane.

![Lund QCD-Dilute TN](figures/lund_subclass_qcd_dilute_tn.png)
QCD-Dilute (true-negative slice) Lund plane.

![Lund QCD-Dilute FP](figures/lund_subclass_qcd_dilute_fp.png)
QCD-Dilute (false-positive slice) Lund plane.

![Lund Top-Dilute TP](figures/lund_subclass_top_dilute_tp.png)
Top-Dilute (true-positive slice) Lund plane.

![Lund Top-Dilute FN](figures/lund_subclass_top_dilute_fn.png)
Top-Dilute (false-negative slice) Lund plane.

![Lund Top-Dense](figures/lund_subclass_top_dense.png)
Top-Dense Lund plane (diffuse tail, excluded from physics reading).

![QCD-Dense avg](figures/deta_dphi_subclass_qcd_dense_avg.png)
QCD-Dense average energy-flow map.

![QCD-Dense rep](figures/deta_dphi_subclass_qcd_dense_rep.png)
QCD-Dense representative map.

![QCD-Dilute avg](figures/deta_dphi_subclass_qcd_dilute_avg.png)
QCD-Dilute average map.

![QCD-Dilute rep](figures/deta_dphi_subclass_qcd_dilute_rep.png)
QCD-Dilute representative map.

![Top-Dilute avg](figures/deta_dphi_subclass_top_dilute_avg.png)
Top-Dilute average map.

![Top-Dilute rep](figures/deta_dphi_subclass_top_dilute_rep.png)
Top-Dilute representative map.

![Top-Dense avg](figures/deta_dphi_subclass_top_dense_avg.png)
Top-Dense average map.

![Top-Dense rep](figures/deta_dphi_subclass_top_dense_rep.png)
Top-Dense representative map.

</details>

## 4. Geometry & supporting evidence (paper §3.3)

**Takeaway:** embeddings collapse to 2-D (TwoNN 2.21, PCA 83.6+16.4); prototypes 43.6° apart; hypersphere alone (CosLinear) keeps balanced subclasses, margin m=0.5 resolves Top Dilute vs Dense. Null controls (noise, untrained) also give k*=2 — so k*=2 alone proves nothing; the physics case rests on §3.

<details>
<summary>Figures</summary>

![Hypersphere](figures/hypersphere_3d.png)
Embeddings collapse onto a 2-D disk spanned by the two prototypes.

![PCA](figures/pca_scatter_3way.png)
PCA scatter: PC1 83.6% + PC2 16.4% ≈ 100% (2-D collapse).

![Embedding geometry](figures/embedding_geometry_3way.png)
Embedding geometry across ArcFace / CosLinear / Linear heads.

![Head geometry](figures/head_geometry_3way.png)
Head comparison: hypersphere alone (CosLinear) keeps balanced subclasses; margin m=0.5 specifically resolves Top Dilute vs Dense.

![Dimensionality](figures/dimensionality_3way.png)
TwoNN + participation ratio per head: ArcFace TwoNN 2.21 vs 4.98/5.19 controls.

</details>

## 5. Symbolic distillation (paper §3.4)

**Takeaway:** PySR recovers tanh(m/pT) − 0.348·τ32 — the model rediscovers mass+3-prongness from raw inputs. Leading-physics distillate only (rejection 33× vs 124×), not a replacement.

<details>
<summary>Figure</summary>

![Pareto](figures/pareto_front_sr.png)
Pareto front: compact c=6 form (leading-physics distillate, not a replacement — rejection 33× vs 124× full model).

</details>

### 5.1 Norm-matrix findings (2026-09-28, quick scale 100k, seeds 42/123/7, --no-nest-tanh)

15 cells (5 norms × 3 seeds), tagged `fit_quick_norm_*` + `eval_norm_*_{self,full}` in `experiments/physics_sr/`; published artifacts untouched. Metric = transfer-AUC of each cell's best-AUC equation evaluated on the full 404k set. Re-verified from artifacts 2026-09-28.

| Norm | s42 (c / AUC) | s123 (c / AUC) | s7 (c / AUC) | Mean | Spread |
|:-----|:---:|:---:|:---:|:---:|:---:|
| z-score | 21 / 0.9624 | 19 / 0.9618 | 13 / 0.9591 | **0.9611** | 0.0033 |
| robust (median/IQR) | 14 / 0.9602 | 16 / 0.9602 | 16 / 0.9609 | 0.9605 | **0.0008** |
| minmax | 15 / 0.9582 | 20 / 0.9514 | 12 / 0.9515 | 0.9537 | 0.0068 |
| raw (y rescaled, bug fixed) | 14 / 0.9509 | 25 / 0.9528 | 14 / 0.9537 | 0.9525 | 0.0027 |
| sigma (x/std only) | 20 / 0.9488 | 7 / 0.9409 | 24 / 0.9603 | 0.9500 | 0.0194 |

**Settled claims (evidence-backed):**
- z ≈ robust statistically tied at top (gap 0.0006 < z's own spread 0.0033, n=3). Sayable: "z+robust clearly beat minmax/raw/sigma". NOT sayable: "z is the best".
- Mass_over_pT present in all 15 best equations; cleanly Mass-led in 14/15 (mm-s123 mixes mSD_over_pT lead) — physics claim (mass+3-prongness) is robust to norm choice.
- Transfer quick→full holds (self vs full agree to 3rd decimal); saturation 100k→404k ≈ +0.002 AUC (diminishing returns); 10k seed-unstable (AUC 0.906 vs 0.951 across seeds).
- no-nest-tanh costs ~nothing at full scale (corr 0.9658 vs 0.9688 unconstrained) and recovers the paper's c=6 clean form.
- Singularity flags: sigma-s42 `log(Mass_over_pT)`; raw-s7 `Tau32/Mass_over_pT` division; mm-s123 best-c shifts c15→c20 self-to-full. Sigma most unstable (spread 0.0194).
- Bug found+fixed: `fit_physics_sr.py` raw path forgot y_hat rescale (all pre-fix raw numbers were in yz units, invalid).

**Decisions (binding):**
- D1: paper-wide symbolic seeds = 42/123/7 (consistency over statistics). Paper line ~568 still says 42/123/456 — MUST edit. Seed-456 artifacts are legacy.
- D2: final norm verdict needs full-404k z-vs-robust confirm (quick-scale gap is noise). Tiebreak order: full-scale transfer-AUC → cross-seed spread → z as standard practice with robust noted as equivalent.
- D3: condition-variation screens run on BOTH z and robust (Stage 1 seed-42 screen = 16 cells, not 8).

**OPEN (do not write into paper before closing):**
- O1: z vs robust final winner (needs 2 full-404k confirm runs).
- O2: parsimony / operators / nesting / maxsize effects (Stage B screen).

### 5.2 Stage-B condition screen (2026-09-29, quick scale 100k, seed 42, z+robust)

16/16 cells done, tagged `fit_quick_stgB_*` + `eval_stgB_*_{self,full}`; summary artifact `experiments/physics_sr/stageB_summary.json` (via `scripts/analysis/summarize_sr_cells.py`). Metric = transfer-AUC on full 404k. Baseline for comparison: norm-matrix z_s42 = 0.9624 (c21).

| Config | z (c / AUC) | robust (c / AUC) |
|:---|:---:|:---:|
| pars 1e-5 | 16 / 0.9630 | 23 / 0.9622 |
| pars 3e-4 | 16 / 0.9623 | 18 / 0.9625 |
| pars 1e-3 | 18 / 0.9630 | 24 / 0.9620 |
| no-exp-log | 14 / 0.9628 | 21 / 0.9627 |
| +pow | 16 / 0.9627 | 15 / 0.9586 |
| tanh-only | 19 / 0.9634 | 14 / 0.9606 |
| nesting allowed | 18 / 0.9611 | 19 / 0.9632 |
| maxsize 35 | 27 / 0.9635 | 23 / 0.9638 |

**Findings:** no variant exceeds baseline by more than 0.002 (full range 0.9586–0.9638 vs baseline 0.9624; paper full-scale c21 = 0.9637). +pow is null-to-negative (robust +pow is the worst cell, 0.9586; z +pow keeps 0.9627 with an ugly 0.267^Mass form) → pow DROPPED. Parsimony 1e-5–1e-3 immaterial → 1e-4 kept as "insensitive", not "tuned". Nesting ban costs nothing (nestok z 0.9611 ≈ baseline). Dropping exp/log costs nothing (0.9628/0.9627) and removes the log-singularity class seen in sigma-s42.
**c=6 form recurrence:** clean `tanh(Mass_over_pT)−k·Tau32` recurs in 9 independent cells (norm matrix + Stage B); k = 0.3471–0.3477 in 5 cells, 0.3431 (tanhonly), 0.2786 (z_s123), 0.4290 (nestok). Honest statement: form recurs, k clusters 0.34–0.35 (paper refined 0.348 sits centrally), full span 0.28–0.43. NOT "k=0.347±0.001 universally".

**Decisions (binding):**
- D4 (overrides D2): O1 closed WITHOUT full-confirm runs. Rationale: quick-scale gap z−robust = 0.0007 < z's own spread 0.0033; separating it at full scale would need ≫404k events for a distinction with no physics consequence (both Mass-led, same structural forms). Paper states the tie honestly and adopts z-score as the standard-practice default, robust noted as equivalent.
- D5: Stage-B tables stay OUT of the paper (single-seed screen, all within noise of published full-scale numbers). Paper gets null-result sentences only. C1 cut.
- O2 closed by this section.
- O3: final closed-form equation replacing the published one (Stage 3 full-confirm).
- O4: parsimony 1e-4 is an unjustified magic number until B1 lands.

## 6. Discussion (paper §4)

- **What the class prototypes learned (primary):** high-mass/low-τ32 (Top) vs low-mass/high-τ32 (QCD) — checked, not assumed.
- **Why the hypersphere matters:** 3-head comparison isolates it — CosLinear keeps balanced subclasses without any margin; Linear collapses.
- **Limitations (stated openly):** simulation-only (Pythia 8), binary task, no detector effects, Top-Dense excluded, single-seed label-shuffle residual, QCD mass-confounding stated, narrow pT window, OOD scan is hypothesis-generating only.
- **Future work:** cross-generator check, multi-seed shuffle, full mass-matched table, multi-class.

## 7. Conclusion (paper §5)

A 59k-parameter tagger whose binary rule is nearest-class-prototype on S⁶³ (primary contribution) reaches 91.84% with a 2-D, QCD-checked latent geometry; subclass structure is a secondary finding en route: 4 found, 3 validated, controls run, limits stated. The RF still wins on accuracy — and says so in the abstract.

## Appendix A — dataset EDA

<details>
<summary>Figures</summary>

![EDA overview](figures/eda_figA1_dataset_overview.png)
Dataset overview: class balance, kinematics ranges.

![EDA kinematics](figures/eda_figA2_kinematics.png)
Kinematic distributions (pT, mass, multiplicity).

![EDA substructure](figures/eda_figA3_substructure.png)
Substructure observables (τ21, τ32, zg, θg, nSD).

![EDA correlation](figures/eda_figA4_correlation.png)
Observable correlation matrix (mass confounding visible here first).

</details>

## Appendix B — out-of-distribution (JetNet30, hypothesis-generating)

<details>
<summary>Figure</summary>

![OOD](figures/poc_ood_similarity.png)
Cosine similarity to prototypes per JetNet class: W/Z near QCD prototype; JetNet top maps to neither (30 vs ≤200 constituents shift — expected, honest low confidence).

</details>

## Reproduce

```bash
pip install -e . && python scripts/download_data.py   # 3.4 GB dataset
python scripts/train_arcface.py --scale 16 --seed 42  # 50 epochs
```
