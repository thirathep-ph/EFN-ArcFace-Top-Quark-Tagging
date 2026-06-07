# 📊 Paper Results Table — Compiled from Existing Data

> Source files: `main_arcface/`, `results/test_run_50k/`, `results/analysis_15ep/`

---

## Table 1: Classification Performance

| Model | Acc | Precision | Recall | F1 | Params |
|:------|:---:|:---------:|:------:|:--:|:------:|
| **EFN + ArcFace (ours)** | **91.74%** | **90.37%** | **93.46%** | **91.89%** | 46,592 |
| EFN + Linear + CE | — | — | — | — | 46,338 |
| EFN + CosLinear + CE | — | — | — | — | 46,338 |

*Source: `main_arcface/advanced_metrics.json`* (ArcFace row)
*Note: Linear + CosLinear baselines need GPU training*

---

## Table 2: Subclass Discovery (Spectral Clustering)

| Class | Subclass | N | Mass (GeV) | tau32 | sqrt(d12) | Character |
|:------|:---------|:-:|:----------:|:-----:|:---------:|:----------|
| **Top** | SC 0 (Core) | — | 171.2 ± 23.5 | 0.50 ± 0.13 | 0.34 ± 0.04 | Clean 3-prong |
| | SC 1 (Edge) | — | 179.2 ± 31.8 | 0.63 ± 0.13 | 0.34 ± 0.04 | Diffuse / ambiguous |
| **QCD** | SC 0 (Edge) | — | 111.9 ± 25.5 | 0.72 ± 0.11 | 0.34 ± 0.04 | High-mass mimic Top |
| | SC 1 (Core) | — | 60.5 ± 18.6 | 0.74 ± 0.11 | 0.34 ± 0.05 | Soft single-prong |

*Source: `results/test_run_50k/statistical_metrics.json` → group_stats*

---

## Table 3: Subclass Separation (Statistical Tests)

| Comparison | Mass KS (p) | tau32 KS (p) | Mass JSD | tau32 JSD |
|:-----------|:----------:|:------------:|:--------:|:---------:|
| Top SC 0 vs SC 1 | 0.12 (1.5e-5) | **0.41 (4.8e-59)** | 0.20 | **0.36** |
| QCD SC 0 vs SC 1 | **0.75 (1.2e-222)** | 0.12 (5.0e-5) | **0.65** | 0.15 |
| TP vs FN | **0.68 (1.4e-178)** | **0.36 (4.2e-47)** | **0.57** | 0.34 |
| TN vs FP | **0.90 (0.0)** | 0.19 (2.2e-12) | **0.75** | 0.19 |

*Source: `results/test_run_50k/statistical_metrics.json` → statistical_tests*

---

## Table 4: Clustering Quality

| Metric | Top Subclasses | QCD Subclasses |
|:-------|:--------------:|:--------------:|
| **Cosine Silhouette (SC)** | **0.743** | **0.856** |
| **Cosine Silhouette (K-means)** | 0.749 | 0.865 |
| **ARI (SC vs K-means)** | **0.898** | **0.923** |

*Source: `results/test_run_50k/statistical_metrics.json` → silhouette_scores, adjusted_rand_indices*

---

## Table 5: Error Analysis (T/F Composition)

| Group | Mass (GeV) | tau32 | sqrt(d12) | Interpretation |
|:------|:----------:|:-----:|:---------:|:---------------|
| **TP** (True Top) | 177.1 ± 23.1 | 0.54 ± 0.13 | 0.34 ± 0.04 | Clean top — correct |
| **FN** (Missed Top) | 139.4 ± 20.9 | 0.65 ± 0.12 | 0.34 ± 0.04 | Diffuse top → classified as QCD |
| **TN** (True QCD) | 80.0 ± 31.3 | 0.74 ± 0.11 | 0.34 ± 0.04 | Clean QCD — correct |
| **FP** (Fake Top) | **182.2 ± 37.4** | 0.69 ± 0.12 | 0.35 ± 0.04 | High-mass QCD → mimics top |

*Source: `results/test_run_50k/statistical_metrics.json` → group_stats*

---

## Table 6: Existing Figures Ready for Paper

| Figure | File | Description |
|:-------|:-----|:------------|
| ROC + AUC | `main_arcface/performance.png` | Classification performance |
| Confusion matrix | `main_arcface/confusion_matrix.png` | TP/TN/FP/FN counts |
| Training history | `main_arcface/training_results_efn.png` | Loss + Accuracy + Separation |
| UMAP status | `results/analysis_15ep/umap_prediction_status.png` | Latent space with T/F |
| UMAP subclasses | `results/analysis_15ep/umap_subclasses.png` | 2+2 subclasses |
| Subclass physics | `results/analysis_15ep/subclass_physics_distributions.png` | Mass, tau32, d12 per subclass |
| Composition heatmap | `results/analysis_15ep/subclass_composition_heatmap.png` | Purity matrix |
| Cosine similarities | `results/analysis_15ep/subclass_similarity_distributions.png` | Per-subclass alignment |
| zg splitting | `results/analysis_15ep/zg_ap_splitting.png` | Altarelli-Parisi validation |
| Angular collimation | `results/analysis_15ep/angular_collimation.png` | Relativistic scaling |
| Physics correlations | `results/analysis_15ep/subclass_physics_correlations.png` | Feature relationships |
| Representative jets | `results/analysis_15ep/top_subclass_*_representative_jets.png` | Top-5 per Top subclass |
| Representative jets | `results/analysis_15ep/qcd_subclass_*_representative_jets.png` | Top-5 per QCD subclass |
| Lund planes | `results/analysis_15ep/lund_planes/*.png` | Full hierarchical set |
| SVD scree | `main_arcface/svd_scree_plot.png` | Dimension justification |
| Centroid constituent | `main_arcface/top_subclass_0_centroid.png` | Top prototype visualization |
| Centroid constituent | `main_arcface/qcd_subclass_0_centroid.png` | QCD prototype visualization |

---

## 📝 Notes for Paper Writing

1. **Classification table** (Table 1): Need to train Linear + CosLinear baselines (GPU)
2. **All other tables** (2-5): Data already exists in `test_run_50k/statistical_metrics.json`
3. **Figures**: 16+ figures ready — only need to arrange + caption
4. **Best model**: 100-epoch run in `main_arcface/` (91.74% acc)
5. **SC k=2 determined by eigengap** — key narrative: "core vs edge subclasses"
