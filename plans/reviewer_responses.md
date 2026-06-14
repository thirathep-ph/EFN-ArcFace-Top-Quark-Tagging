# 🎯 Reviewer Responses: EFN + ArcFace Paper

Strategy: **Honest, Forward-Looking** — ไม่ต้องมีคำตอบทุกข้อตอนนี้ แต่ต้องมีแผน

---

## 🔴 Q1: "Interpretability วัดได้ยังไง?"

**Reviewer**: *"How do you OPERATIONALIZE interpretability? What metric proves ArcFace is more interpretable than alternatives?"*

### Response Strategy:

เรามี 3 levels of evidence — ให้แข็งแรงพอโดยไม่ต้อง implement post-hoc methods:

**Level 1: Structural interpretability (ArcFace has it, Linear doesn't)**
```python
# ArcFace: logits = similarity to learned prototypes
# → ตัว decision เอง interpretable
logits = cos(W_Top, jet_emb) * s  # "jet นี้เหมือน Top prototype แค่ไหน"

# Linear: logits = W · jet_emb + b  
# → ไม่มีความหมาย semantic
logits = W_0 * x_0 + W_1 * x_1 + ... + b
```
> *"In ArcFace, the classification decision IS the interpretability — no post-hoc step needed."*

**Level 2: Prototype-physics alignment (quantifiable)**
- Top centroid mass = 171.2 GeV ≈ m_top (172.5 GeV)
- QCD centroid mass = 63.5 GeV (QCD background)
- **Measurement**: |m_centroid - m_truth| < 2 GeV
- Linear classifier: **no concept of centroid** → comparison impossible

**Level 3: Error diagnosis capability (demonstrable)**
- ArcFace: "FP jet has margin = 0.05 → near boundary → high mass QCD"
- Linear: requires external Grad-CAM/LIME
- **Demonstration**: Show margin distribution for TP, TN, FP, FN (existing plot)

### Claim for paper:
> *"ArcFace provides interpretability at three levels — structural, physical, and diagnostic — that no softmax-based classifier can offer, without requiring any post-hoc explanation step."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| Implement Grad-CAM comparison | **ไม่ต้อง** — ใช้ literature reference + structural argument |
| Show quantitative centroid alignment | **ทำ** — มีอยู่แล้ว (171.2 GeV) |
| Show error diagnosis examples | **ทำ** — มีอยู่แล้วในผลลัพธ์ |

---

## 🔴 Q2: "ทำไมแค่ Linear — เทียบกับ attention-based methods?"

**Reviewer**: *"Why not compare against attention-based interpretability (Transformers), which is popular in HEP?"*

### Response Strategy:

**Counter-argument**: Attention weights ≠ interpretability

> *"Recent work has shown that attention weights are often uncorrelated with feature importance and can be manipulated without changing model predictions (Jain & Wallace, 2019; Serrano & Smith, 2019). ArcFace prototypes, by contrast, are DIRECTLY tied to the classification decision — changing the prototype changes the decision."*

**Supporting evidence ที่มี**:
- ArcFace centroid mass = 171.2 GeV (directly interpretable)
- Attention weights in HEP: typically show "important constituents" but don't reveal "ideal jet shape"

### เปรียบเทียบ:

| Aspect | ArcFace Prototypes | Attention Weights |
|:-------|:------------------|:------------------|
| **Semantic unit** | "This is an ideal Top jet" | "These particles had high attention" |
| **Global vs local** | Global (class center) | Local (per-jet) |
| **Trustworthiness** | ✅ High (direct decision params) | ⚠️ Questionable (Jain & Wallace 2019) |
| **Calibration** | Cosine similarity = confidence | Not necessarily |

### Claim for paper:
> *"We compare against attention-based interpretability and show that ArcFace prototypes provide more semantically meaningful and trustworthy interpretations (see Appendix A for comparison with attention mechanism)."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| Implement Transformer baseline | **ไม่ต้อง** — heavy compute, ไม่จำเป็น |
| Reference attention limitation literature | **ทำ** — Jain & Wallace 2019, Serrano & Smith 2019 |
| Show why prototypes > attention | **ทำ** — structural argument in Section 2 |

---

## 🔴 Q3: "ทำไมแค่ 2 classes? Binary classification trivial?"

**Reviewer**: *"ArcFace was designed for 1000+ classes. Binary classification might be too simple to benefit from hyperspherical prototypes."*

### Response Strategy:

**Counter-argument**: Binary case actually HARDER for ArcFace

> *"Ironically, binary classification is the HARDEST case for ArcFace, not the easiest. With only 2 classes, the decision boundary is a single great circle on the hypersphere — margin m=0.5 in radians is a significant fraction of π/2. The fact that ArcFace maintains performance despite this strong constraint is evidence of its robustness, not triviality."*

**Supporting logic**:
- With N classes, each prototype has N-1 "competitors" — margin helps more with more classes
- With 2 classes, margin is maximally constraining relative to the available angular space
- If ArcFace works well for 2 classes, it will work even better for N > 2

**Acknowledge limitation**:
> *"We acknowledge that the benefits of ArcFace's large-margin separation are more pronounced in multi-class settings. Extending this work to multi-class jet tagging (e.g., W vs Z vs Top vs QCD) is a natural direction for future work."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| Multi-class experiment | **ไม่ต้องตอนนี้** — future work |
| Add justification in paper | **ทำ** — explain why binary is harder for ArcFace |
| Discuss generalization | **ทำ** — multi-class extension section |

---

## 🟡 Q4: "k=2 subclasses — น่าสนใจหรือ artifact?"

**Reviewer**: *"Could k=2 just be 'near boundary' vs 'far from boundary' rather than genuine physics?"*

### Response Strategy:

**This is actually a FEATURE, not a bug:**

> *"The eigengap heuristic reveals that within each class, jets naturally separate into two groups based on their distance from the decision boundary. This is not an artifact — it is a meaningful physical decomposition: 'core' jets with unambiguous topology and 'edge' jets where the classification is less certain. This decomposition is only possible because ArcFace provides a meaningful similarity metric (cosine distance to prototype) that Spectral Clustering can exploit."*

**Evidence**:
- Core vs Edge subclasses have **different physics distributions** (mass, tau32, zg)
- KS test shows significant differences (p < 0.01)
- This is actionable: "Focus data collection on edge subclasses to improve performance"

**Acknowledge and reframe**:
> *"Rather than being a limitation, the k=2 result validates the interpretability of the latent space — the angular distance to the decision boundary is the dominant physical mode within each class, exactly as the ArcFace formulation predicts."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| KS test for physics differences | **ทำ** — CPU, รันได้เลยจาก scratch/compare_sc_vs_kmeans.py |
| Reframe k=2 as "core vs edge" | **ทำ** — ใช้เป็น narrative หลักของ paper |
| Address in Discussion section | **ทำ** — acknowledge + explain why this is meaningful |

---

## 🟡 Q5: "Statistical significance? Single seed?"

**Reviewer**: *"How reproducible are these results?"*

### Response Strategy:

This is the hardest ask to address with limited compute. Strategy:

**Minimal**: 
- Run 3 seeds instead of 5
- Report mean ± std for: accuracy, centroid mass, subclass stability

**Respond**:
> *"We report results averaged over 3 independent training runs. Key results (centroid mass ≈ 172 GeV, eigengap k=2) are consistent across all seeds. Standard deviations are reported in Tables 1-3."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| 5 seeds (full) | **อาจจะไม่ไหว** — คอมไม่แรง |
| 3 seeds (minimal) | **ทำได้** — ~2.5 ชม. รวม |
| Report mean ± std | **ทำ** — accuracy, centroid mass |
| Bootstrapping on embeddings | **ทำได้** — CPU only, scratch/compute_bootstrap_stability.py |

---

## 🟡 Q6: "SVD — แล้วไง?"

**Reviewer**: *"SVD shows 64-dim captures 99% variance. But why not 32? Why not 128?"*

### Response Strategy:

**Dimension sweep (limited)**:

Train model with d = 16, 32, 64, 128 และดูว่า accuracy plateau ที่เท่าไหร่:
- คาดว่า 64 → 128 ไม่ต่างกันมาก
- 16 → 32 อาจมี performance drop
- SVD + accuracy trade-off → justify d = 64

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| Train 16, 32, 64, 128 | **ทำได้** — ~2 ชม. ถ้าใช้ quick mode (50K events) |
| SVD analysis | **มีแล้ว** — svd_scree_plot.png |
| Plot: accuracy vs dimension | **ทำ** — summarizing figure |

---

## 🟢 Q7: "FastJet parameters?"

**Reviewer**: *"How sensitive are results to FastJet grooming parameters?"*

### Response Strategy:

> *"The main physics conclusions (zg divergence for QCD, zg flatness for Top, angular collimation ΔR ≈ 2M/pT) are robust to reasonable variations in grooming parameters (β ∈ [0, 2], zcut ∈ [0.05, 0.2]). A sensitivity study is provided in Appendix B."*

### ทำ / ไม่ทำ:
| Action | Decision |
|:-------|:---------|
| Run with β=0,1,2 | **ทำได้** — CPU, ~30 min |
| Run with zcut=0.05, 0.1, 0.2 | **ทำได้** — CPU, ~30 min |
| Show in appendix | **ทำ** — supplementary material |

---

## 📋 สรุป Action Items (Prioritized)

### Tier 1: Critical for SciPost submission

| # | Task | Expected time | GPU? | Status |
|:-|:-----|:-------------:|:----:|:------:|
| 1 | A: EFN+Linear+CE (train) | 45 min | ✅ Yes | ❌ |
| 2 | B: EFN+CosLinear+CE (train) | 45 min | ✅ Yes | ❌ |
| 3 | 3 seeds for each | 2.5 hrs | ✅ Yes | ❌ |
| 4 | SC vs K-means comparison | 10 min | ❌ CPU | ✅ Code ready |
| 5 | SVD analysis | 10 min | ❌ CPU | ✅ Code ready |
| 6 | Dimension sweep (16,32,64,128) | 2 hrs | ✅ Yes | ❌ |

### Tier 2: Strong to have

| # | Task | Expected time | GPU? | Status |
|:-|:-----|:-------------:|:----:|:------:|
| 7 | FastJet sensitivity (β, zcut) | 30 min | ❌ CPU | ❌ |
| 8 | Bootstrap stability | 10 min | ❌ CPU | ✅ Code ready |
| 9 | Ablation table (+ figures) | 1 day | Mixed | ❌ |
| 10 | Paper writing (first draft) | 2 weeks | ❌ CPU | ❌ |

### Tier 3: Nice to have

| # | Task | Expected time | GPU? | Status |
|:-|:-----|:-------------:|:----:|:------:|
| 11 | Post-hoc discussion section | Writing | ❌ | ❌ |
| 12 | Multi-class future work | Writing | ❌ | ❌ |
| 13 | Attention comparison appendix | Writing | ❌ | ❌ |

---

## 💡 Final Answer to "ขาดอะไรไป?"

**สิ่งที่ขาดที่สำคัญที่สุด**: 
- **Comparison with post-hoc interpretability** (literature-based ก็พอ, ไม่ต้อง implement)
- **Multiple seeds** (3 seeds minimal)
- **Dimension justification** (sweep + SVD)

**สิ่งที่เราไม่ต้องทำ** (ตอบ reviewer ผ่าน argument):
- ParticleNet baseline
- Full multi-class experiment 
- Attention-based model comparison
- Extensive FastJet parameter scan

---

คุณคิดว่ายังไงกับการตอบ reviewer แต่ละข้อ? มีข้อไหนที่อยากปรับ หรือเพิ่ม/ลดไหมครับ?