# 📁 Plans Directory — Current State

> **Target**: SciPost Physics Core
> **Narrative**: "Core vs Edge Subclass Discovery via Parameter-Free Spectral Clustering"
> **Last Updated**: 2026-06-25

---

## 📊 Project Status

| Component | Status | Details |
|:----------|:------:|:--------|
| **Code** | ✅ 90% | All scripts fixed, Device detection working, DirectML compatible |
| **Experiments** | 🔶 70% | Baselines complete, 4/9 robustness trials done, SVD done, bootstrap needs re-run |
| **Paper** | ❌ 0% | Not started |

---

## 📂 Active Documents

### Core References (KEEP — Current)

| File | Purpose | Status |
|:-----|:--------|:------:|
| [`parameter_catalog.md`](parameter_catalog.md) | Complete parameter catalog with reviewer defenses | ✅ Current |
| [`parameter_justification.md`](parameter_justification.md) | Why each model uses its parameters (written this session) | ✅ Current |
| [`output_folder_structure.md`](output_folder_structure.md) | Output directory conventions (written this session) | ✅ Current |
| [`paper_results_table.md`](paper_results_table.md) | All results compiled into paper-ready tables | ✅ Current |
| [`publication_roadmap.md`](publication_roadmap.md) | Target venue strategy, timeline, paper structure | ✅ Current |

### Archived Documents (Historical — Read-only)

| File | Purpose | Reason Archived |
|:-----|:--------|:----------------|
| [`full_ablation_plan.md`](full_ablation_plan.md) | Complete ablation plan (all models & parameters) | Superseded by current experiment state |

---

## 🗑️ Deleted Documents (2026-06-25 Cleanup)

The following 18 files were deleted because they are outdated, redundant, or superseded:

| File | Reason |
|:-----|:-------|
| `audit_fixes.md` | All fixes were applied to code |
| `baseline_implementation_plan.md` | Baselines are done |
| `context_recovery.md` | Old session recovery — no longer needed |
| `directml_impact_analysis.md` | Analysis complete |
| `directml_implementation_plan.md` | Implementation complete |
| `final_execution_plan.md` | Superseded by current work |
| `fitness_assessment.md` | Superseded — project state has advanced |
| `improvement_ideas.md` | Not needed for current phase |
| `MIGRATION_GUIDE.md` | Not needed |
| `multi_perspective_review.md` | Not needed |
| `reproducibility_plan.md` | Not needed |
| `research_master_plan.md` | Superseded by current work |
| `reviewer_priority_map.md` | Not needed |
| `reviewer_responses.md` | Not needed |
| `reviewer_simulation.md` | Not needed |
| `session_log_2025-06-07.md` | Historical record |
| `session_log_2025-06-08.md` | Historical record |
| `strategy_rethink.md` | Superseded by current work |

---

## 🎯 Next Steps

1. **Re-run bootstrap** — `scratch/analysis/compute_bootstrap_stability.py` (CPU, ~10 min)
2. **Complete robustness trials** — 5 remaining (s×m) combinations (GPU, ~10 hrs)
3. **Write paper** — Follow structure in [`publication_roadmap.md`](publication_roadmap.md)
4. **Submit to SciPost Physics Core**

---

## 📈 Key Results Summary

| Metric | Value | Source |
|:-------|:------|:-------|
| Test Accuracy | 91.74% | `main_arcface/advanced_metrics.json` |
| Precision | 90.37% | Same |
| Recall | 93.46% | Same |
| F1 | 91.89% | Same |
| Parameters | 46,592 | Model checkpoint |
| Top Centroid Mass | 168.8 GeV | `main_arcface/centroid_constituents_report.md` |
| QCD Centroid Mass | 63.5 GeV | Same |
| Bootstrap CI | [91.66%, 91.83%] | `scratch/compute_bootstrap_stability.py` |
| Subclass Structure | k=2 per class (Core/Edge) | Spectral Clustering (eigengap) |
| Top Core Purity | 98.5% TP | `scratch/compile_paper_stats.py` |
| Top Edge Purity | 83.2% TP | Same |
| QCD Core Purity | 99.9% TN | Same |
| QCD Edge Purity | 82.9% TN | Same |

---

## 📁 File Count: 6 files (down from 27)

**Before**: 27 files (many outdated, redundant, or superseded)
**After**: 6 files (all current and relevant)
