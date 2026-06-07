# 🎭 Reviewer Simulation: SciPost Physics Core

**Paper**: "Ante-Hoc Interpretable Jet Tagging with EFN and ArcFace Hyperspherical Prototypes"
**Reviewer expertise**: HEP + ML interpretability

---

## Reviewer's Initial Impression

*"The paper combines EFNs with ArcFace for interpretable jet tagging. The analysis is extensive (Lund planes, zg, angular collimation, subclass discovery). However, I have concerns about the interpretability claim and the completeness of comparisons."*

---

## 🔴 Review Question 1: "Interpretability วัดได้ยังไง?"

**Reviewer**: *"You claim ArcFace provides 'interpretability' — but interpretability is an abstract concept. How do you OPERATIONALIZE it? What metric proves something is 'more interpretable'?"*

### ปัญหา:
| Claim ใน paper | Evidence ที่มี | Gap |
|:--------------|:--------------|:----|
| "Prototypes มี semantic meaning" | Centroid mass ≈ 172 GeV | ✅ OK |
| "Subclasses มีความหมายทางฟิสิกส์" | SC → k=2, physics distributions | ✅ OK |
| "ArcFace interpretable กว่า Softmax" | ❌ **ไม่มี direct comparison** | 🔴 **GAP** |
| "Ante-hoc ดีกว่า Post-hoc" | ❌ **ไม่มี comparison กับ post-hoc** | 🔴 **GAP** |

### What reviewer expects:
> *"To substantiate the interpretability claim, the authors should provide a quantitative comparison — e.g., stability of explanations across training seeds, or faithfulness of prototypes to model decisions — between ArcFace and a post-hoc method applied to the same EFN backbone."*

---

## 🔴 Review Question 2: "ทำไมแค่ Linear — ทำไมไม่เทียบกับ interpretable methods อื่น?"

**Reviewer**: *"You compare ArcFace only against a standard linear head. Why not compare against other interpretable classification methods commonly used in HEP, such as attention-based models or concept bottleneck models? The comparison seems cherry-picked."*

### ปัญหา:
- Reviewer ไม่รู้จัก ArcFace
- Reviewer อยากเห็นการเทียบกับ methods ที่ตนรู้จัก
- HEP community กำลังสนใจ attention-based interpretability

### What reviewer expects:
> *"At minimum, the authors should include a comparison with attention-based interpretability (e.g., Transformer encoder with attention weights as explanation) to demonstrate that ArcFace provides superior or complementary interpretability."*

---

## 🔴 Review Question 3: "ทำไมแค่ 2 classes? ArcFace designed for 1000+ classes"

**Reviewer**: *"ArcFace was developed for face recognition with thousands of classes. Here you apply it to binary classification. Is there any benefit to hyperspherical prototypes when there are only 2 classes? The decision boundary for binary cosine classification is just a great circle on the sphere."*

### ปัญหา:
- Binary case might be too simple
- Many benefits of ArcFace (large margin separation) are designed for many classes
- Reviewer might think "this is over-engineering"

### What reviewer expects:
> *"The authors should discuss whether the method generalizes to multi-class jet tagging (e.g., W vs Z vs Top vs QCD) or whether the benefits are specific to binary classification."*

---

## 🟡 Review Question 4: "k=2 subclasses — น่าสนใจหรือ trivial?"

**Reviewer**: *"The eigengap heuristic finds k=2 subclasses for both Top and QCD. Could this be an artifact of the binary classification setup? If the latent space is already separated into 2 hemispheres (one per class), further clustering within each hemisphere might be dominated by distance to the decision boundary, not genuine physical substructure."*

### ปัญหา:
- k=2 might just be "close to boundary" vs "far from boundary"
- This interpretation supports the "core vs edge" narrative BUT also undercuts the "physical subclass discovery" claim

### What reviewer expects:
> *"The authors should clarify: are the subclasses physically distinct (e.g., different decay modes) or just distance-from-boundary groups? A multi-class experiment would help disentangle these."*

---

## 🟡 Review Question 5: "Statistical Significance?"

**Reviewer**: *"Results are reported from a single training run. How stable are these findings across different random seeds? The centroid mass of 171.2 GeV is striking — is this reproducible?"*

### ปัญหา:
- No error bars
- Single seed results could be luck
- SciPost expects robustness analysis

### What reviewer expects:
> *"The authors should report means and standard deviations over at least 5 training seeds for all key results (accuracy, centroid mass, subclass assignments)."*

---

## 🟡 Review Question 6: "SVD analysis — แล้วไง?"

**Reviewer**: *"The SVD analysis shows that 64 dimensions capture 99% of variance. This is presented as a justification for the latent dimension. But what is the ALTERNATIVE? Would performance degrade with 32 dimensions? Why 64 specifically?"*

### ปัญหา:
- SVD shows 64 is sufficient but doesn't show it's OPTIMAL
- Missing: performance vs dimension trade-off curve

### What reviewer expects:
> *"A sweep over embedding dimensions (e.g., 16, 32, 64, 128) with corresponding accuracy and interpretability metrics would strengthen the dimension selection argument."*

---

## 🟢 Review Question 7: "FastJet parameters?"

**Reviewer**: *"The Lund plane and substructure analysis uses specific FastJet parameters (R=2.0, beta=0.0, zcut=0.1). How sensitive are the results to these choices?"*

### ปัญหา:
- FastJet parameters could affect the physics conclusions
- Missing: systematic variation of grooming parameters

### What reviewer expects:
> *"A sensitivity study of the main physics results (zg, tau32, angular collimation) to grooming parameters should be included in supplementary material."*

---

## ✅ Summary: Reviewer Scorecard

| Criteria | Current Status | Target for SciPost |
|:---------|:--------------|:-------------------|
| **Novelty** | ⚠️ Moderate | ✅ Clear |
| **Method soundness** | ✅ Good | ✅ Good |
| **Interpretability evidence** | ❌ **Weak** | ✅ **Strong** |
| **Comparison completeness** | ❌ **Missing** | ✅ **Comprehensive** |
| **Statistical rigor** | ❌ **Single seed** | ✅ **Multi-seed** |
| **Reproducibility** | ⚠️ Partial | ✅ Full |
| **Physics validation** | ✅ Strong | ✅ Strong |
| **Writing** | ❌ Not started | ✅ Clear |

---

## 🎯 Action Items from Reviewer Simulation

### Critical (ต้องทำ)
| # | Item | Effort |
|:-|:-----|:------:|
| 1 | **เปรียบเทียบกับ Post-hoc interpretability** (อย่างน้อย literature-based + stability argument) | 2-3 days |
| 2 | **Multi-seed training** (5 seeds) for statistical significance | ~3 hours |
| 3 | **Interpretability comparison table**: ArcFace vs Linear vs CosLinear | 1 day |
| 4 | **Address k=2 ambiguity**: distance-from-boundary vs genuine physics | Discussion |

### Important (ควรทำ)
| # | Item | Effort |
|:-|:-----|:------:|
| 5 | **Dimension sweep**: 16, 32, 64, 128 | ~2 hours |
| 6 | **Attention-based comparison** (optional — for stronger paper) | 3-5 days |
| 7 | **FastJet parameter sensitivity** | 1 day |

### Nice to have
| # | Item | Effort |
|:-|:-----|:------:|
| 8 | Multi-class extension discussion | Writing only |
| 9 | Comparison with literature values (ParticleNet AUC) | Writing only |

---

## 💡 สิ่งที่ผมคิดว่า "ขาด" ไปจริงๆ เมื่อกี้

เวลาผมตอบคำถามคุณเรื่อง comparison ผม focus ที่ **architectural ablation** (เปลี่ยนแค่ head) แต่ลืมเรื่องที่สำคัญกว่าสำหรับ SciPost:

> **Interpretability is NOT self-evident.** You can't just claim "prototypes are interpretable" and end the story. You need to:
> 1. **Prove** they are interpretable (vs alternatives)
> 2. **Demonstrate** what you can DO with them that you can't do otherwise
> 3. **Quantify** the interpretability

**สิ่งที่ขาดไปในแผนของผม**: การเปรียบเทียบ interpretability **ระหว่าง** methods อย่างเป็นระบบ ไม่ใช่แค่ ablation ของ architecture

---

คุณคิดว่า reviewer questions ข้อไหนสำคัญที่สุดสำหรับ SciPost? หรือมีข้ออื่นที่ผมยังมองไม่เห็นอีก?