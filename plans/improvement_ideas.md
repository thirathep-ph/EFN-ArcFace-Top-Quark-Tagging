# 🔧 Idea Bank: Low-Effort Improvements → Push to 90%+

## Already in Plan

| # | Idea | Type | Time |
|:-:|:-----|:----:|:----:|
| A | **Bootstrap confidence intervals** | CPU run | 5 min |
| B | **Centroid mass across all runs** | CPU lookup | 10 min |
| C | **Null test argument** | Writing | 15 min |
| D | **Literature context table** | Writing | 20 min |
| E | **Interpretability score (I-score)** | Writing | 10 min |

---

## New Ideas

### F. Physics from Prototypes (CPU lookup, 15 min)
Extract physics from ArcFace prototypes (already have centroid constituent data):
- Energy fraction of leading constituent
- Subjet multiplicity (nSD)
- Angular spread of constituents
→ Compare to known top jet properties
→ "The prototype IS a physically valid top jet"

### G. Per-Subclass Purity Quantification (CPU, 10 min)
From existing composition heatmap data:

| Subclass | Purity | Error Rate | Interpretation |
|:---------|:------:|:----------:|:---------------|
| Top Core (SC 0) | ~95% TP | ~5% FN | Safe zone |
| Top Edge (SC 1) | ~60% TP | ~40% FN | Ambiguous zone |
| QCD Core (SC 1) | ~98% TN | ~2% FP | Safe zone |
| QCD Edge (SC 0) | ~55% TN | ~45% FP | Dangerous zone |

→ Quantifies "core vs edge" narrative

### H. Learning Curve (already in history.json, writing 10 min)
Plot accuracy vs epoch:
- ArcFace: smooth convergence, plateaus at ~91.7%
- (After baselines) Linear: does it converge faster/slower?
→ Shows training dynamics

### I. Prototype Evolution During Training (CPU lookup, 15 min)
From saved checkpoints at different epochs:
- Epoch 1: centroid mass = ?
- Epoch 5: centroid mass = ?
- Epoch 15: centroid mass = ?
- Epoch 100: centroid mass = 171.2 GeV
→ Show monotonic convergence to m_top

### J. Embedding Coloring by Physics (CPU, already have embeddings, 20 min)
Re-color UMAP plots by:
- Jet mass (color gradient)
- tau32 (color gradient)
- zg (color gradient)
→ Show latent space preserves physics ordering

### K. Multi-Class Extension Discussion (Writing only, 15 min)
Show that ArcFace naturally extends to N classes:
- W vs Z vs Top vs QCD (4 prototypes on S^63)
- Quark vs Gluon discrimination
- No architecture changes needed
→ Demonstrates generality

### L. Margin Effect Ablation (GPU, 45 min)
One additional training run:
- EFN + ArcFace (m=0.3) — intermediate margin
Compare with:
- EFN + CosLinear (m=0) — no margin
- EFN + ArcFace (m=0.5) — full margin
→ Shows margin matters and optimal value

### M. Simple Grad-CAM Visual (GPU, 30 min, one-time)
For one example jet on the EFN+Linear baseline:
- Run forward pass → get prediction
- Run backward pass → get gradient w.r.t input particles
- Plot: ArcFace prototype interpretation vs Grad-CAM attribution
→ Single compelling figure showing ante-hoc > post-hoc

---

## Prioritized Action List

### Tier 1 — Do NOW (CPU, < 1 hour total)
- [x] Bootstrap CIs (A)
- [x] Centroid mass across runs (B) — ตรวจสอบจาก `results/test_run_*/efn_results.json`
- [x] Per-subclass purity table (G)
- [x] Prototype evolution during training (I) — ถ้ามี checkpoint หลาย epoch
- [x] Literature context table (D)
- [x] Interpretability score (E)

### Tier 2 — Do WITH baselines (GPU needed)
- [ ] Train baselines (already planned)
- [ ] Margin effect sweep (L) — optional, 1 extra run
- [ ] Simple Grad-CAM visual (M) — optional, 1 example

### Tier 3 — Writing only
- [ ] Null test argument (C)
- [ ] Multi-class discussion (K)
- [ ] Learning curve (H)

---

## Summary: What These Add

| Before | After All Improvements |
|:-------|:----------------------|
| 1 seed results | Bootstrap CIs + multi-run centroids |
| Qualitative interpretability | I-score + purity quantification |
| Binary only claim | Multi-class extension discussion |
| ArcFace only | Learning curve + prototype evolution |
| No context | Literature comparison table |
| Core vs Edge narrative | Quantified purity per subclass |

### Impact Estimate:
```
Without improvements:  70%
With Tier 1 only:      80% (+10%)
With Tier 1 + Baselines: 85% (+15% total)
With Tier 1 + Baselines + Tier 2: 90% (+20% total)
```
