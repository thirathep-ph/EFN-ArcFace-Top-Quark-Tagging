# 📋 Session Log: 2025-06-07 — Complete Project Analysis

> **Objective**: Assess publication potential, rewrite README, plan baselines, prepare for SciPost
> **Context**: First-author debut paper, single author, limited compute

---

## 1. ข้อผิดพลาดที่เกิดขึ้น (Mistakes Made)

### Mistake 1: มองไม่เห็น results ที่มีอยู่แล้ว
- ตอนแรกบอก user ว่า "ต้องทำ SC vs K-means" แต่จริงๆ มี script `scratch/compare_sc_vs_kmeans.py` พร้อมรัน
- ตอนแรกบอก "ต้องทำ SVD" แต่จริงๆ `main_arcface/svd_scree_plot.png` มีอยู่แล้ว
- ตอนแรกบอก "ยังไม่มี baseline model" แต่จริงๆ `checkpoints/baseline/best_model_efn.pt` มีอยู่แล้ว
- **Lesson**: ตรวจสอบให้ละเอียดก่อนบอก user ว่าอะไรยังไม่ได้ทำ

### Mistake 2: ใช้ DPGMM (k=5) แทน SC (k=2) ในการอ้างอิงผล
- บอกว่า "5 Top subclasses + 5 QCD subclasses" แต่จริงๆ แล้วนั่นคือผล DPGMM (n_components=5 forced)
- ผล SC ที่ถูกต้องคือ **k=2 ต่อ class** (eigengap heuristic บอกเอง)
- **Lesson**: อย่าสับสนระหว่าง DPGMM (forced k) กับ SC (parameter-free)

### Mistake 3: เรียก "Softmax" แทน "Linear" สำหรับ ablation
- บอก user ว่า "EFN + Softmax" แต่จริงๆ ต้องบอก "EFN + Linear + CE"
- เพราะ CrossEntropyLoss มี softmax รวมอยู่แล้ว
- **Lesson**: ใช้ terminology ที่ถูกต้อง — Softmax ≠ Linear

### Mistake 4: ไม่ได้ตรวจสอบ results directory ก่อน
- มี `results/test_run_50k/statistical_metrics.json` ที่มี Silhouette, ARI, KS tests, JSD ครบ
- มี embeddings หลายรุ่น (`embeddings.npz`, `embeddings_15ep.npz`)
- มี full training history สำหรับหลายรุ่น (5, 10, 15, 30, 100 epochs)
- **Lesson**: ตรวจ `results/`, `main_arcface/`, `checkpoints/` ให้ละเอียดก่อนทุกครั้ง

### Mistake 5: สรุปว่า "SC vs K-means ผล ARI=0.90 ดี" โดยไม่คิดถึง interpretation
- ARI=0.90 แปลว่า SC ไม่ได้ดีกว่า K-means — อาจเป็นจุดอ่อนของ methods paper
- **Lesson**: ดูผลลัพธ์อย่างมีวิจารณญาณ อย่าสรุปทันทีว่าสิ่งที่เจอ "ดี"

---

## 2. สิ่งที่ค้นพบ (Findings)

### 2.1 Existing Results (มีอะไรบ้าง)

| หมวด | รายละเอียด | ไฟล์ |
|:------|:-----------|:------|
| **Model (100 epochs)** | EFN + ArcFace, 46K params | `main_arcface/best_model_efn.pt` |
| **Classification metrics** | Acc=91.74%, Prec=90.37%, Rec=93.46%, F1=91.89% | `main_arcface/advanced_metrics.json` |
| **SVD analysis** | Scree plot + explained variance | `main_arcface/svd_scree_plot.png` |
| **Centroid physics** | Top centroid mass=171.2 GeV, QCD centroid mass=63.5 GeV | `main_arcface/centroid_constituents_report.md` |
| **SC subclasses** | k=2 per class (Top: Core/Edge, QCD: Core/Edge) | `main_arcface/lund_planes/` |
| **Statistical tests** | KS tests (p<0.01), JSD, Silhouette, ARI | `results/test_run_50k/statistical_metrics.json` |
| **Subclass physics** | Mass, tau32, sqrt(d12) per subclass | Same file |
| **Error analysis** | TP: M=177, FN: M=139, TN: M=80, FP: M=182 GeV | Same file |
| **Physics validation** | zg Altarelli-Parisi, angular collimation | `results/analysis_15ep/` |
| **Lund planes** | Full hierarchical (class + subclass + T/F) | `results/analysis_15ep/lund_planes/` |
| **UMAP** | Prediction status + subclasses | `results/analysis_15ep/` |
| **Training history** | หลายรุ่น (5, 10, 15, 30, 50, 100 epochs) | `results/test_run_*/history.json` |
| **Baseline checkpoint** | EFN + prototype (แต่ยังไม่ได้ train จริง?) | `checkpoints/baseline/best_model_efn.pt` |
| **Embeddings** | ~404K test embeddings | `main_arcface/embeddings.npz` |
| **Figures** | 16+ publication-ready plots | ดู Table 6 ใน paper_results_table.md |

### 2.2 SC vs K-means Results (จาก `scratch/compare_sc_vs_kmeans.py`)

```
Spectral Clustering (main method):
├── Top: Silhouette = 0.743
├── QCD: Silhouette = 0.856
├── Eigengap → k=2 (automatic)

Spherical K-means (comparison):
├── Top: Silhouette = 0.749
├── QCD: Silhouette = 0.865

Agreement:
├── Top ARI = 0.898  (SC ≈ K-means)
├── QCD ARI = 0.923  (SC ≈ K-means)
└── Conclusion: Subclasses robust, but SC doesn't add value over K-means
```

### 2.3 Subclass Physics Summary

| Subclass | Mass (GeV) | tau32 | sqrt(d12) | Character |
|:---------|:----------:|:-----:|:---------:|:----------|
| **Top SC 0** (Core) | 171.2 ± 23.5 | 0.50 ± 0.13 | 0.34 ± 0.04 | Clean 3-prong — TP |
| **Top SC 1** (Edge) | 179.2 ± 31.8 | 0.63 ± 0.13 | 0.34 ± 0.04 | Diffuse — FN |
| **QCD SC 0** (Edge) | 111.9 ± 25.5 | 0.72 ± 0.11 | 0.34 ± 0.04 | High-mass — FP |
| **QCD SC 1** (Core) | 60.5 ± 18.6 | 0.74 ± 0.11 | 0.34 ± 0.05 | Soft — TN |

Key: sqrt(d12) ≈ 0.34 for ALL subclasses → splitting scale is NOT the distinguishing feature
Mass and tau32 ARE the distinguishing features → model relies on these

---

## 3. สิ่งที่สร้างขึ้นในวันนี้ (New Artifacts)

### 3.1 Planning Documents (ใน /plans/)

| File | Purpose |
|:-----|:--------|
| `publication_roadmap.md` | SciPost strategy, narrative, timeline |
| `baseline_implementation_plan.md` | A/B/C ablation hierarchy |
| `reviewer_simulation.md` | 7 reviewer questions with severity |
| `reviewer_responses.md` | Response strategies for each question |
| `paper_results_table.md` | 6 tables from existing results |
| `session_log_2025-06-07.md` | **(this file)** Complete session log |

### 3.2 Code Files

| File | Purpose |
|:-----|:--------|
| `jet_tagging/models/efn_linear.py` | Model A: EFN + Linear + CE (standard) |
| `jet_tagging/models/efn_coslinear.py` | Model B: EFN + CosLinear + CE (ArcFace - margin) |
| `scripts/train_baselines.py` | Training script for both (multi-seed, --model both) |

### 3.3 Updated Files

| File | Change |
|:-----|:-------|
| `README.md` | Full rewrite (was outdated) |

---

## 4. Remaining Work

### 4.1 GPU Required (not yet done)

| Task | Commands |
|:-----|:---------|
| Train Linear baseline (3 seeds) | `python scripts/train_baselines.py --model linear --epochs 30 --seeds 42 123 7` |
| Train CosLinear baseline (3 seeds) | `python scripts/train_baselines.py --model coslinear --epochs 30 --seeds 42 123 7` |

### 4.2 CPU Tasks (can be done anytime)

| Task | Status |
|:-----|:-------|
| SC vs K-means comparison | ✅ **DONE** — ran today |
| SVD analysis | ✅ Already exists (`svd_scree_plot.png`) |
| Compile results table | ✅ **DONE** — `paper_results_table.md` |
| Paper writing | ❌ Not started |

### 4.3 Open Questions (need discussion)

1. **SC vs K-means: ARI=0.90 → Strengthens or weakens paper?**
   - PRO: Subclasses are robust (not algorithm artifact)
   - CON: SC doesn't add value over simpler method
   - Solution: Reframe narrative from "SC is better" to "subclasses are robust"

2. **Baseline models: What exactly to compare?**
   - A: EFN + Linear + CE (standard classifier)
   - B: EFN + CosLinear + CE (ArcFace - margin)
   - C: EFN + ArcFace + CE (ours — already have)
   - Question: Is this sufficient for SciPost?

3. **Statistical significance: How many seeds?**
   - 3 seeds minimal, 5 seeds preferred
   - User's computer may not handle 5 seeds (time constraints)

4. **Paper venue: SciPost Physics Core vs MLST vs JINST?**
   - SciPost: Open access, physics focus — user's preference
   - But: First-time author, high selectivity (~50%)
   - Alternative: MLST (~65%) or JINST (~70%) as safer option

---

## 5. Key Insights for Narrative

### What's Strong
- ArcFace prototype centroid mass (171.2 GeV) ≈ top quark mass (172.5 GeV) — **self-verification**
- Classification accuracy 91.74% with only 46K params — **efficient**
- Subclasses are robust (ARI=0.90 between SC and K-means) — **reproducible**
- Physics validation aligns with QCD (zg, collimation) — **physically correct**
- Error composition (FP: high-mass QCD, FN: diffuse Top) — **interpretable errors**

### What's Weak
- SC doesn't outperform K-means (silhouette nearly identical)
- Binary classification may be too simple to demonstrate ArcFace's full benefits
- No comparison with post-hoc interpretability methods (Grad-CAM, LIME)
- No comparison with SOTA architectures (ParticleNet)
- Single seed for main results (no error bars yet)

### What's Missing
- Ablation baselines (A + B) need training (GPU)
- Error bars across seeds
- Post-hoc interpretability comparison (literature-based)
- Multi-class extension discussion

---

## 6. เป้าหมายต่อไป (Next Steps)

1. ตัดสินใจ: SC ≈ K-means → แก้ narrative หรือเปลี่ยน approach?
2. Train baselines (เมื่อมี GPU)
3. เขียน paper — ใช้ narrative "core vs edge subclasses" + robust physics validation
4. ส่ง SciPost Physics Core หรือปรับเป็น MLST/JINST

---

*End of session log. Total working time: ~3 hours.*
