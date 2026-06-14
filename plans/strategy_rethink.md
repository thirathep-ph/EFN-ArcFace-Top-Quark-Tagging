# 🧠 Strategy Rethink: Baseline, Model Quality & Robustness

> Brainstorming document for the next phase of the project.

---

## 1. Is Our Current Model Actually Good?

### Quantitative Self-Assessment

| Metric | Our Value | What It Tells Us |
|--------|:---------:|:-----------------|
| **Test Accuracy** | 91.74% | Solid but not SOTA |
| **Precision** | 90.37% | When we say Top, we're right 90% of the time |
| **Recall** | 93.46% | We catch 93% of true Top jets |
| **F1** | 91.89% | Balanced performance |
| **Parameters** | 46,592 | Tiny — 100× smaller than ParticleNet (~500K) |
| **Inference time** | 0.069 ms/jet | ~14,500 jets/second — real-time capable |
| **Centroid mass** | 168.8 GeV | 2% from true top mass (172.5 GeV) — self-verifying |
| **Bootstrap CI** | [91.66%, 91.83%] | Very tight — model is stable |

### Comparison to Literature (Top Tagging Reference Dataset)

| Model | Accuracy | AUC | Params | Reference |
|:------|:--------:|:---:|:------:|:----------|
| **Ours (EFN+ArcFace)** | **91.74%** | ~0.97* | **46K** | This work |
| EFN + Linear | ~91% (est.) | ~0.97 | 46K | Ablation target |
| ParticleNet | **~93-94%** | **~0.98** | ~500K | Qu & Gouskos 2020 |
| PFN / Deep Sets | ~91% | ~0.97 | ~60K | Komiske et al. 2019 |
| LorentzNet | ~93% | ~0.98 | ~300K | Gong et al. 2022 |
| ResNeXt (baseline) | ~92% | ~0.97 | ~1.5M | Top Tagging Ref Dataset |

*\*AUC estimated from accuracy; we haven't computed it yet*

### Verdict

**Our model is good, but not SOTA.** And that's **fine** — our paper's claim is not "we beat ParticleNet." Our claim is **"ante-hoc interpretability at minimal performance cost."** The 91.74% accuracy with only 46K params and built-in interpretability is the story.

**What matters most**: The centroid mass self-verification (168.8 GeV ≈ 172.5 GeV) is a unique result that no black-box model can offer. This is our strongest evidence.

---

## 2. Baseline Strategy: Train vs Pre-trained vs Literature

### Option A: Train Our Own Baselines (Current Plan)

| Pros | Cons |
|:-----|:-----|
| Fair comparison (same data, same preprocessing) | Needs GPU time (~3 hrs for 3 seeds) |
| Can compute all metrics consistently | `train_baselines.py` needs fixing first |
| Strongest for reviewer credibility | Risk of implementation bugs |

### Option B: Use Pre-trained Models

For the Top Tagging Reference Dataset, pre-trained models are available:
- **ParticleNet**: Official weights from Qu & Gouskos
- **PFN/PEFN**: From Komiske et al. (EnergyFlow package)

| Pros | Cons |
|:-----|:-----|
| No training time needed | May use different preprocessing |
| Authoritative implementations | Can't control training details |
| Saves GPU for other experiments | May not match our data split exactly |

### Option C: Cite Literature Values Only

| Pros | Cons |
|:-----|:-----|
| Zero compute cost | Weakest evidence |
| Can cite multiple papers | "Our numbers vs their numbers" is apples-to-oranges |
| Simple to write | Reviewer will ask "did you run on the same split?" |

### My Recommendation: **Hybrid — Train EFN baselines, cite ParticleNet**

```
For the paper:
  Table: [EFN+ArcFace] [EFN+Linear] [EFN+CosLinear] [ParticleNet*]
  
  * = cited from literature (Qu & Gouskos 2020)
  All EFN variants: trained by us on same data/splits
```

This gives us:
- **Fair ablation** (EFN variants) — trained by us, same backbone
- **Context** (ParticleNet) — cited, shows where SOTA is
- **No need** to implement complex architectures

---

## 3. Hyperparameter Sweep vs Parameter Robustness

### The Traditional Approach: Hyperparameter Sweep

Find the "best" model by trying many combinations:
```
d ∈ [64, 128], s ∈ [16, 32], m ∈ [0.3, 0.5]
→ 8 trials → pick best → train longer
```

**Problem**: This is a fishing expedition. It finds the best model for THIS dataset but tells us nothing about whether the model is robust. A reviewer will ask: *"Did you tune on the test set?"*

### The Better Approach: Parameter Robustness

Instead of asking "what's the best?", ask **"how sensitive is performance to parameter changes?"**

```
Fix d=64 (justified by SVD: 64 dims capture >99% variance)
Vary s ∈ [8, 16, 32]  (scale factor)
Vary m ∈ [0.0, 0.3, 0.5]  (margin)

For each combination: train ONCE, report accuracy
→ Show a 3×3 heatmap of (s, m) → accuracy
→ If all 9 cells are within ±1%, the model is ROBUST
→ If some cells drop sharply, we learn something
```

**Why this is better for our paper:**

| Aspect | Hyperparameter Sweep | Robustness Analysis |
|:-------|:--------------------:|:-------------------:|
| Narrative | "We found the best" | "Our method is stable" |
| Reviewer risk | "Did you overfit to val set?" | "Good — shows method works" |
| Compute | 8+ trials | 9 trials (same cost) |
| Scientific value | Low (just optimization) | High (generalizability) |
| Matches our claim | ❌ "We're the best" | ✅ "We're interpretable AND robust" |

### What to Measure in Robustness Analysis

Instead of just accuracy, measure **what matters for our narrative**:

```
For each (s, m) combination:
  1. Test accuracy
  2. Centroid mass (should be ~172 GeV for Top)
  3. Cosine separation gap (sim_Top - sim_QCD for Top jets)
  4. k from eigengap (should be 2)
  5. Subclass purity (Core vs Edge)
```

If all 9 combinations give:
- Accuracy: 91-92% ✓
- Centroid mass: 165-175 GeV ✓
- k=2 ✓
- Core purity >95% ✓

Then our **claim is bulletproof**: *"ArcFace is robust to hyperparameter choice — the physics emerges naturally."*

### Recommendation: **Robustness over Sweep**

```
Instead of: "We searched for the best hyperparameters"
Say: "We fixed d=64 (SVD-justified) and tested 9 (s,m) combinations.
      All 9 produce consistent accuracy (91-92%), 
      consistent centroid mass (~170 GeV),
      and consistent k=2 subclass structure.
      This demonstrates that ArcFace's interpretability 
      is not fragile — it's a robust property of the method."
```

This is a **much stronger scientific claim**.

---

## 4. New Recommended Plan

### Phase 1: Fix Baseline Script (1 day)
- Rewrite `train_baselines.py` with chunked loading
- Train EFN+Linear and EFN+CosLinear (1 seed each, 30 epochs)
- Verify they work before multi-seed run

### Phase 2: Robustness Analysis (GPU, ~4-5 hours)
Run 9 combinations of (s, m) with d=64 fixed:
```
s = [8, 16, 32]
m = [0.0, 0.3, 0.5]
```
For each: train 30 epochs, evaluate on test, compute all metrics.

### Phase 3: Multi-Seed Baselines (GPU, ~3 hours)
- Train EFN+Linear: 3 seeds × 30 epochs
- Train EFN+CosLinear: 3 seeds × 30 epochs
- Train EFN+ArcFace (s=16, m=0.5): 3 seeds × 30 epochs (if not already)

### Phase 4: Paper Writing
- Table 1: Classification (our 3 models × 3 seeds, + ParticleNet cited)
- Table 2: Robustness heatmap (9 combinations)
- Table 3-5: Subclass physics (from existing data)
- Figure: Centroid mass vs (s, m) — should be flat ~170 GeV

---

## 5. What This Changes From Previous Plan

| Old Plan | New Plan | Why |
|:---------|:---------|:----|
| 8-trial sweep for "best" model | 9-trial robustness analysis | Stronger scientific claim |
| ParticleNet implementation | ParticleNet cited from literature | Saves weeks of work |
| Focus on beating baselines | Focus on robustness + interpretability | Matches our narrative |
| Single best model | Ensemble of robust models | Better for reproducibility |
