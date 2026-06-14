# 🚀 Final Execution Plan: EFN + ArcFace for SciPost Physics Core

> **Author**: thirathep-ph (first-author debut paper)
> **Target**: SciPost Physics Core
> **Backup**: MLST (Machine Learning: Science and Technology)
> **Status**: Code ~90% ready. Experiments ~30% done. Paper 0% written.

---

## 1. Reality Check: Is This Worth Graduate School Admission?

**Pure honesty:**

| Criterion | This Paper | What's Expected |
|:----------|:-----------|:----------------|
| **Novelty** | Combination of existing methods (EFN + ArcFace) | Novel combination is acceptable for methods paper |
| **Technical depth** | Strong — full pipeline from data to physics validation | ✅ Above average for first paper |
| **Experiments** | 1 main model + 2 baselines + robustness (15 runs) | ✅ Thorough |
| **Physics validation** | Lund planes, zg, angular collimation, centroid mass | ✅ Strong — this is the paper's unique strength |
| **Writing** | Not started | ❌ Need 2-3 weeks |
| **Venue** | SciPost Physics Core (~60% acceptance) | Ambitious but achievable |

**Verdict**: This paper + good grades = **competitive for MS/PhD at top-50 universities**. For top-10 (MIT, Stanford, Berkeley), you'd need a second, more novel paper or a conference presentation.

**What this paper proves to admissions committees:**
- You can design and execute a complete ML + physics research project
- You understand both deep learning and domain science (HEP)
- You can write and publish in a peer-reviewed journal
- You're ready for graduate-level research

---

## 2. Complete Experiment Plan

### 2.1 Models to Train

| Model | Description | Seeds | Epochs | Est. Time |
|:------|:------------|:-----:|:------:|:---------:|
| **ArcFace (ours)** | EFN + ArcFace, s=16, m=0.5 | 3 | 50 | ~4.5 hrs |
| **Linear (baseline)** | EFN + Linear + CE | 3 | 50 | ~4.5 hrs |
| **CosLinear (baseline)** | EFN + CosLinear + CE, s=16, m=0 | 3 | 50 | ~4.5 hrs |
| **Robustness** | 9×(s=[8,16,32], m=[0.0,0.3,0.5]) | 1 each | 50 | ~13.5 hrs |

**Total GPU time**: ~27 hours
**Total Colab units**: ~90 (exactly your budget)

### 2.2 Commands

```bash
# ArcFace (3 seeds)
python scripts/train_top.py --epochs 50 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --seed 123 --no-chunk --amp
python scripts/train_top.py --epochs 50 --seed 7 --no-chunk --amp

# Linear (3 seeds)
python scripts/train_baselines.py --model linear --epochs 50 --seeds 42 123 7 --no-chunk

# CosLinear (3 seeds)
python scripts/train_baselines.py --model coslinear --epochs 50 --seeds 42 123 7 --no-chunk

# Robustness (9 combinations)
python scripts/train_top.py --epochs 50 --scale 8 --margin 0.0 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 8 --margin 0.3 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 8 --margin 0.5 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 16 --margin 0.0 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 16 --margin 0.3 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 16 --margin 0.5 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 32 --margin 0.0 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 32 --margin 0.3 --seed 42 --no-chunk --amp
python scripts/train_top.py --epochs 50 --scale 32 --margin 0.5 --seed 42 --no-chunk --amp
```

**Note**: `--margin` argument needs to be added to `train_top.py`. Currently only `--scale` exists.

---

## 3. Paper Structure

### Title
"Ante-Hoc Interpretable Jet Tagging with Hyperspherical Prototypes"

### Authors
Thirathep Phongpreecha (single author)

### Sections

```
1. Introduction (2 pages)
   - Jet tagging is essential for HEP but models are black boxes
   - Post-hoc interpretability (Grad-CAM, SHAP) is fragile
   - Our contribution: first ante-hoc interpretable jet tagger
   
2. Related Work (1.5 pages)
   - Jet tagging architectures (ParticleNet, PFN, LorentzNet)
   - Interpretable ML in HEP (attention, concept-based)
   - ArcFace in computer vision
   
3. Methodology (2 pages)
   - 3.1 EFN backbone (IRC-safety)
   - 3.2 ArcFace head (hyperspherical prototypes)
   - 3.3 Spectral Clustering for subclass discovery
   
4. Experiments (4 pages)
   - 4.1 Dataset: Top Tagging Reference Dataset
   - 4.2 Classification performance (Table 1)
   - 4.3 Ablation study (Table 2: ArcFace vs Linear vs CosLinear)
   - 4.4 Robustness analysis (Table 3: 9×(s,m) heatmap)
   - 4.5 Subclass discovery (Table 4: k=2, Core vs Edge)
   - 4.6 Physics validation (Lund, zg, collimation)
   
5. Discussion (1.5 pages)
   - What prototypes learned (centroid mass ≈ m_top)
   - Core vs Edge subclasses
   - Limitations and future work
   
6. Conclusion (0.5 page)
```

### Tables

| Table | Content | Source |
|:------|:--------|:-------|
| 1 | Classification: ArcFace vs Linear vs CosLinear (mean±std, 3 seeds) | New experiments |
| 2 | Robustness: 3×3 heatmap of (s,m) → accuracy | New experiments |
| 3 | Subclass physics: mass, tau32, sqrt(d12) per subclass | Existing data |
| 4 | Clustering comparison: SC vs K-means vs Agglo | Existing data |
| 5 | Error analysis: TP/TN/FP/FN composition | Existing data |

### Figures

| Figure | Description | Status |
|:-------|:------------|:-------|
| 1 | Architecture diagram | Need to create |
| 2 | Learning curve (accuracy vs epoch) | Can generate from history.json |
| 3 | ROC curve + Background rejection | Already exists |
| 4 | Robustness heatmap (s,m) | Need robustness data |
| 5 | UMAP latent space (2D) | Already exists (15-epoch) |
| 6 | UMAP latent space (3D) | Script exists, needs cleanup |
| 7 | Subclass physics distributions | Already exists |
| 8 | Lund planes (TP/TN/FP/FN) | Already exists |
| 9 | Centroid jet visualization | Already exists |
| 10 | zg Altarelli-Parisi validation | Already exists |

---

## 4. Timeline

| Phase | What | Time | Depends On |
|:------|:-----|:----:|:-----------|
| **1. Code fixes** | ρ-MLP fix, --margin arg, local test | 1 day | Nothing |
| **2. Colab runs** | 15 training runs (90 units) | ~2 days | Phase 1 |
| **3. Analysis** | Generate tables, figures, robustness heatmap | 1 day | Phase 2 |
| **4. Paper writing** | Full draft | 2 weeks | Phase 3 |
| **5. Review & submit** | Internal review, revisions, submit | 1 week | Phase 4 |

**Total**: ~3-4 weeks to submission.

---

## 5. What Success Looks Like

### Best Case: Accepted at SciPost Physics Core
- Open access, free
- Good for graduate school applications
- Citation potential in HEP + ML community

### Good Case: Accepted at MLST (backup)
- Machine Learning: Science and Technology
- IOP journal, good reputation
- ~75% acceptance rate

### Safe Case: Accepted at JINST
- Journal of Instrumentation
- HEP-focused, lower impact
- ~80% acceptance rate

### Worst Case: Rejected everywhere
- Still valuable for graduate school applications
- "Submitted for publication" on CV
- Can present at workshops (ML4Jets, ACAT)

---

## 6. What Makes This Paper Unique

**The centroid mass self-verification is the killer result.**

No previous jet tagging paper has shown that the model's internal representation converges to the correct physical mass (168.8 GeV ≈ 172.5 GeV) without being told the mass. This is:
- **Interpretable**: The prototype literally IS an ideal Top jet
- **Self-verifying**: The model proves itself correct
- **Unique**: No black-box model can offer this

This single result is worth more than 10 baselines. It's the reason this paper can be competitive at SciPost despite being a first-author undergraduate paper.
