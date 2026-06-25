# Code Cleanup Plan

**Date:** 2026-06-25
**Status:** In Progress
**Based on:** Code audit findings (10 issues identified)

---

## Executive Summary

This document outlines a systematic cleanup plan to address 10 code quality issues found during the audit. The issues range from duplicated code and wrong model path references to stale documentation. The plan is organized by priority, with the most impactful changes (unifying duplicated code, fixing broken model references) addressed first.

**Key Goals:**
1. Eliminate code duplication by creating a unified clustering module
2. Fix all broken `prototype_layer.prototypes` references to use `arcface_head.class_centers`
3. Remove stale DPGMM references in favor of Spectral Clustering + Eigengap
4. Update documentation to reflect current codebase state

---

## Priority 1: Create Unified Clustering Module

### Issue
`scripts/train_top.py` (lines 194-292) and `scripts/analyze_offline.py` (lines 45-108) contain identical copies of `estimate_clusters_eigengap()` and `spectral_clustering_subclass()`. This violates DRY and risks divergence.

### Action
**Create** [`arcjet/utils/clustering.py`](../arcjet/utils/clustering.py) containing:
- `estimate_clusters_eigengap()` — copied from `scripts/train_top.py` (the canonical version)
- `spectral_clustering_subclass()` — copied from `scripts/train_top.py`

**Update** [`scripts/train_top.py`](../scripts/train_top.py):
- Remove local definitions of both functions (lines 194-292)
- Add import: `from arcjet.utils.clustering import estimate_clusters_eigengap, spectral_clustering_subclass`

**Update** [`scripts/analyze_offline.py`](../scripts/analyze_offline.py):
- Remove local definitions of both functions (lines 45-108)
- Add import: `from arcjet.utils.clustering import estimate_clusters_eigengap, spectral_clustering_subclass`

**Update** [`arcjet/utils/__init__.py`](../arcjet/utils/__init__.py):
- Add export for `clustering` module

---

## Priority 2: Fix `prototype_layer.prototypes` → `arcface_head.class_centers`

### Issue
The actual model ([`arcjet/models/efn_arcface.py`](../arcjet/models/efn_arcface.py)) uses `model.arcface_head.class_centers`, but many scratch scripts reference the old `model.prototype_layer.prototypes`. These references are broken.

### Files to Fix

| File | Lines | Change |
|------|-------|--------|
| [`scratch/analysis/run_full_analysis.py`](../scratch/analysis/run_full_analysis.py) | 131, 135, 140 | Remove `n_proto_per_class=1`, change `model.prototype_layer.prototypes` → `model.arcface_head.class_centers` |
| [`scratch/analysis/analyze_latent_space.py`](../scratch/analysis/analyze_latent_space.py) | 21, 26 | Remove `n_proto_per_class=1`, change `model.prototype_layer.prototypes` → `model.arcface_head.class_centers` |
| [`scratch/analysis/check_sc_zg.py`](../scratch/analysis/check_sc_zg.py) | 21 | Remove `n_proto_per_class=1` |
| [`scratch/analysis/inspect_similarities.py`](../scratch/analysis/inspect_similarities.py) | 23, 32 | Remove `n_proto_per_class=1`, change `model.prototype_layer.prototypes` → `model.arcface_head.class_centers` |
| [`scratch/analysis/replot.py`](../scratch/analysis/replot.py) | 24, 46 | Remove `n_proto_per_class=1`, change `model.prototype_layer.prototypes` → `model.arcface_head.class_centers` |
| [`scratch/analysis/rerun_analysis_15ep.py`](../scratch/analysis/rerun_analysis_15ep.py) | 46-47, 61 | Remove `n_proto_per_class=1`, change `model.prototype_layer.prototypes` → `model.arcface_head.class_centers` |

### Pattern to Apply
Replace:
```python
model = EFNArcFace.load_from_checkpoint(ckpt_path, n_proto_per_class=1, ...)
protos = model.prototype_layer.prototypes
```
With:
```python
model = EFNArcFace.load_from_checkpoint(ckpt_path, ...)
centers = model.arcface_head.class_centers
```

**Note:** Some files already use the correct pattern with backward-compat fallback (e.g., `compute_bootstrap_stability.py`, `compute_immediate_metrics.py`). These can serve as reference implementations.

---

## Priority 3: Fix DPGMM → Spectral Clustering in `run_full_analysis.py`

### Issue
[`scratch/analysis/run_full_analysis.py`](../scratch/analysis/run_full_analysis.py) (lines 60, 294-344) uses `BayesianGaussianMixture` (DPGMM) for subclass discovery, while the rest of the codebase uses Spectral Clustering + Eigengap.

### Action
- Remove `BayesianGaussianMixture` import (line 60)
- Replace DPGMM fitting code (lines 294-344) with calls to:
  - `estimate_clusters_eigengap()` from `arcjet.utils.clustering`
  - `spectral_clustering_subclass()` from `arcjet.utils.clustering`
- Update any downstream code that expects DPGMM output format

---

## Priority 4: Update Plans to Remove DPGMM References

### Files to Update

**[`plans/publication_roadmap.md`](../plans/publication_roadmap.md)** (line 41):
- Change: "DPGMM for subclass discovery" → "Spectral Clustering + Eigengap for subclass discovery"

**[`plans/parameter_catalog.md`](../plans/parameter_catalog.md)** (lines 73-74, 165-166):
- Change: "DPGMM as current approach" → "Spectral Clustering + Eigengap as current approach"

---

## Priority 5: Update README.md

### Issue
README contains stale references and incorrect paths.

### Changes to [`README.md`](../README.md)

| Line | Current | Fix |
|------|---------|-----|
| 10 | "2 Top subclasses (Core/Edge) + 2 QCD subclasses (Core/Edge)" | Update to reflect parameter-free clustering |
| 79 | References `results/analysis_15ep/` (DPGMM) | Update to current analysis directory |
| 83 | References `checkpoints/best_model_efn.pt` | Change to `experiments/arcface_100ep/` |
| 168 | References `scratch/run_full_analysis.py` | Change to `scratch/analysis/run_full_analysis.py` |

---

## Priority 6: Fix Minor Issues

### 6a. Stale DPGMM Comment in `train_top.py`
**File:** [`scripts/train_top.py`](../scripts/train_top.py) (line 765)
- Change plot title from "UMAP + DPGMM" to "UMAP + Spectral Clustering"

### 6b. Add AMP Support to `train_baselines.py`
**File:** [`scripts/train_baselines.py`](../scripts/train_baselines.py)
- Add `--amp` argument (matching `train_top.py`)
- Add `amp_available`/`get_amp_scaler` imports from `arcjet.utils.device`

### 6c. Missing `evaluate_models.py` Reference
- Verify [`scripts/evaluate_models.py`](../scripts/evaluate_models.py) exists and is functional
- Add reference in training scripts or README if needed

### 6d. Experiments Missing `metrics.json`
- Run evaluation on `experiments/robustness_s8_m00/`, `s8_m03/`, `s8_m05/`, `s16_m*`, `s32_m*` to generate missing `metrics.json`
- Investigate `experiments/svd_analysis/` — determine if metrics are applicable

### 6e. `arcjet/utils/__init__.py` Exports
- Ensure all utility functions are properly exported
- Add `clustering` module to exports

---

## Implementation Order

| Step | Task | Dependencies |
|------|------|--------------|
| 1 | Create `arcjet/utils/clustering.py` | None |
| 2 | Update `scripts/train_top.py` to use new module | Step 1 |
| 3 | Update `scripts/analyze_offline.py` to use new module | Step 1 |
| 4 | Fix `scratch/analysis/run_full_analysis.py` (prototypes + DPGMM) | Step 1 |
| 5 | Fix remaining scratch/ files (prototypes references) | None |
| 6 | Update `plans/publication_roadmap.md` | None |
| 7 | Update `plans/parameter_catalog.md` | None |
| 8 | Update `README.md` | Steps 4, 5 |
| 9 | Fix stale DPGMM comment in `train_top.py` | Step 2 |
| 10 | Add AMP support to `train_baselines.py` | None |
| 11 | Generate missing `metrics.json` files | None |
| 12 | Update `arcjet/utils/__init__.py` exports | Step 1 |

---

## Verification Checklist

- [ ] `arcjet/utils/clustering.py` exists and contains both functions
- [ ] `scripts/train_top.py` imports from `arcjet.utils.clustering` (no local definitions)
- [ ] `scripts/analyze_offline.py` imports from `arcjet.utils.clustering` (no local definitions)
- [ ] No remaining references to `model.prototype_layer.prototypes` in scratch/ files
- [ ] No remaining `n_proto_per_class` parameter in scratch/ files
- [ ] `run_full_analysis.py` uses Spectral Clustering (no DPGMM)
- [ ] Plans files reference Spectral Clustering, not DPGMM
- [ ] README paths and descriptions are accurate
- [ ] `train_baselines.py` has `--amp` support
- [ ] All robustness experiments have `metrics.json`
- [ ] `arcjet/utils/__init__.py` exports clustering module

---

## Notes

- The backward-compat fallback pattern (try `arcface_head.class_centers`, fall back to `prototype_layer.prototypes`) used in some files is recommended for any code that may encounter older checkpoints.
- After completing this cleanup, consider adding a pre-commit hook or CI check to prevent reintroduction of `prototype_layer.prototypes` references.
