# 🔍 Workspace Audit — Issues & Fix Plan

> Generated: 2025-06-13 (Final version after exhaustive verification)
> Source: Cross-reference of every file against notebook execution output (ground truth)

---

## 📓 Ground Truth: What Actually Ran on Colab T4

From [`notebooks/JetTagging_EFN_ArcFace.ipynb`](notebooks/JetTagging_EFN_ArcFace.ipynb:110-124):

**Command**: `python3 -u scripts/train_top.py --epochs 100 --chunk_size 1500000 --accumulation_steps 8`

**Confirmed facts** (from notebook output):
- **100 epochs** (4h 13m on Colab T4)
- **1,211,000 training events** (9,461 batches × 128 batch size)
- **1 chunk of 1,500,000** (loaded all data at once, not chunked)
- **Margin warmup**: 0.05→0.50 over first 10 epochs
- **OneCycleLR**: LR 0.000085 → 0.002 (peak at epoch 30) → ~0 (epoch 100)
- **Best val acc**: 91.67% at epoch 92
- **Final test acc**: 91.74% (from advanced_metrics.json)
- **Centroid masses**: QCD=63.5 GeV, Top=168.8 GeV
- **UMAP/SC analysis was INTERRUPTED** (KeyboardInterrupt at line 1158)
- **Model params**: 46,592 (confirmed in notebook output lines 117 and 1306)

---

## 🔴 P0 — Critical (Blocks Paper)

### P0-1: `train_baselines.py` uses 50K events instead of 1.2M

**File**: [`scripts/train_baselines.py`](scripts/train_baselines.py:166)

**Problem**: Line 166 hardcodes `max_events=50000`. Baselines would train on 4% of the data the main model uses. Any accuracy comparison would be invalid.

**Fix**: Rewrite to use chunked loading matching [`train_top.py`](scripts/train_top.py:372-407).

**Also**: Line 66 evaluates on only 100K events — should evaluate on full test set (404K).

---

### P0-2: `evaluate_models.py` hardcodes wrong ArcFace params

**File**: [`scripts/evaluate_models.py`](scripts/evaluate_models.py:31)

**Problem**: `TopTaggingModel(s=32.0, m=0.0)` but trained checkpoint uses `s=16.0, m=0.5`. Model won't match saved weights.

**Fix**: Change to `TopTaggingModel(s=16.0, m=0.5, embedding_dim=64)`.

---

### P0-3: `train_baselines.py` imports `F` inside `if __name__`

**File**: [`scripts/train_baselines.py`](scripts/train_baselines.py:232)

**Problem**: `import torch.nn.functional as F` at line 232 inside `if __name__ == '__main__'`, but `F.cross_entropy` used at line 48. Will crash if imported as module.

**Fix**: Move `import torch.nn.functional as F` to top of file.

---

### P0-4: `paper_results_table.md` has wrong silhouette/ARI values

**File**: [`plans/paper_results_table.md`](plans/paper_results_table.md:48-54)

**Problem**: Table 4 claims Silhouette Top=0.743, QCD=0.856, ARI Top=0.898, QCD=0.923. But [`results/test_run_50k/statistical_metrics.json`](results/test_run_50k/statistical_metrics.json:158-165) says: Silhouette Top=0.710, QCD=0.736, ARI Top=0.116, QCD=0.457. **Completely different numbers.**

**Fix**: Verify which run is correct, update table.

---

### P0-5: `paper_results_table.md` Table 2 mass values don't match

**File**: [`plans/paper_results_table.md`](plans/paper_results_table.md:22-27)

**Problem**: Table 2 says "Top SC 0: Mass=171.2 ± 23.5" but [`results/test_run_50k/statistical_metrics.json`](results/test_run_50k/statistical_metrics.json:4-6) says `mass_mean=179.5`. The 171.2 is the **single centroid jet** mass, not the subclass mean.

**Fix**: Update to use correct subclass mean, or clearly label centroid vs mean.

---

### P0-6: Notebook has hardcoded GitHub token

**File**: [`notebooks/JetTagging_EFN_ArcFace.ipynb`](notebooks/JetTagging_EFN_ArcFace.ipynb:51)

**Problem**: `GIT_TOKEN = "***REMOVED-GITHUB-PAT***"` — security risk.

**Fix**: Replace with `os.environ.get("GIT_TOKEN", "")` or `getpass()`.

---

### P0-7: `baseline_implementation_plan.md` references nonexistent files

**File**: [`plans/baseline_implementation_plan.md`](plans/baseline_implementation_plan.md:20,66)

**Problem**: References `efn_softmax.py` and `pfn.py` — **neither exists**. Actual files are `efn_linear.py` and `efn_coslinear.py`.

**Fix**: Update plan to reference actual files.

---

### P0-8: `main_arcface/advanced_metrics.json` says `num_params: 26` — WRONG

**File**: [`main_arcface/advanced_metrics.json`](main_arcface/advanced_metrics.json:6)

**Problem**: JSON says `"num_params": 26` but actual model has **46,592** (confirmed by notebook output and `results/test_run_50k/advanced_metrics.json`). The 26 is clearly a bug — possibly the JSON was corrupted or overwritten.

**Fix**: Manually correct to `46592` or re-run evaluation.

---

## 🟡 P1 — Important (Weakens Paper)

### P1-1: README says "5 subclasses" but actual SC result is k=2

**File**: [`README.md`](README.md:10,169-170)

**Problem**: README states "5 Top + 5 QCD subclasses" but eigengap heuristic selects **k=2 per class**. The 5-subclass result was from DPGMM with forced `n_components=5`.

**Fix**: Update to "2 Top subclasses (Core/Edge) + 2 QCD subclasses (Core/Edge)".

---

### P1-2: `research_master_plan.md` says "128D embeddings" — actual is 64D

**File**: [`research_master_plan.md`](research_master_plan.md:104)

**Problem**: Says "Run KASP Clustering on the 128D embeddings" but actual embedding is **64D**.

**Fix**: Change all references from 128D to 64D.

---

### P1-3: Rho MLP architecture mismatch

**Files**: [`research_master_plan.md`](research_master_plan.md:58), [`README.md`](README.md:27), [`arcjet/models/efn_arcface.py`](arcjet/models/efn_arcface.py:86-91)

**Problem**: Docs say `Rho MLP: [128 → 128 → 64]` but code has `[128 → 64 → 64]` (2 linear layers, not 3).

**Fix**: Update docs to match code, or add the missing layer.

---

### P1-4: `distance_corr_loss` is dead code

**File**: [`arcjet/models/efn_arcface.py`](arcjet/models/efn_arcface.py:173-212)

**Problem**: 40 lines of code, imported in `train_top.py`, but **never called** anywhere.

**Fix**: Remove or add comment noting it's available for future use.

---

### P1-5: `setup.py` missing `umap-learn` dependency

**File**: [`setup.py`](setup.py:7-17)

**Problem**: `umap-learn` in `requirements.txt` but not in `setup.py`. `pip install -e .` will fail at runtime.

**Fix**: Add `"umap-learn"` to `install_requires`.

---

### P1-6: `physics.py` docstring says 8 features, code returns 10

**File**: [`arcjet/utils/physics.py`](arcjet/utils/physics.py:7,68-70)

**Problem**: Docstring says 8 features but code returns 10 (`zg`, `theta_g`, `n_sd` are undocumented). Also `D2` is computed but **not included** in output.

**Fix**: Update docstring to match actual output.

---

### P1-7: README says "15 epochs" but notebook shows 100 epochs

**File**: [`README.md`](README.md:8,168)

**Problem**: README says "15-epoch trained" and "Training: 15 epochs" but actual run was **100 epochs**.

**Fix**: Update README to reflect 100 epochs.

---

### P1-8: `train_top.py` default `--epochs 30` doesn't match actual run

**File**: [`scripts/train_top.py`](scripts/train_top.py:267)

**Problem**: Default is 30 epochs but actual training used 100. Fine for testing but documentation should clarify.

**Fix**: Either change default to 100 or add clarifying comment.

---

### P1-9: Notebook uses `chunk_size=1500000` (single chunk) — not 100000

**File**: [`notebooks/JetTagging_EFN_ArcFace.ipynb`](notebooks/JetTagging_EFN_ArcFace.ipynb:1180)

**Problem**: Notebook sets `CHUNK_SIZE = 1500000` which loads ALL 1.2M events at once. Defeats the memory-safe chunking described in the research plan.

**Fix**: Update documentation or change notebook to use default 100000.

---

### P1-10: Notebook metrics vs advanced_metrics.json disagree

**File**: [`notebooks/JetTagging_EFN_ArcFace.ipynb`](notebooks/JetTagging_EFN_ArcFace.ipynb:1302-1306) vs [`main_arcface/advanced_metrics.json`](main_arcface/advanced_metrics.json:2-5)

**Problem**: Notebook shows Precision=0.8336, Recall=0.9679, F1=0.8957. But JSON says Precision=0.9037, Recall=0.9346, F1=0.9189. These are from the same run but disagree — likely one is validation and one is test, or different BN modes.

**Fix**: Verify which is correct and ensure consistency.

---

### P1-11: `research_master_plan.md` says H5 columns are pT, eta, phi, E — WRONG

**File**: [`research_master_plan.md`](research_master_plan.md:18-21)

**Problem**: Plan says columns 0-199 are pT, 200-399 are eta, 400-599 are phi, 600-799 are E. But [`loader.py`](arcjet/data/loader.py:116-119) extracts **E, px, py, pz** (interleaved: col 0=E, col 1=px, col 2=py, col 3=pz, col 4=E, ...). The plan describes a different data layout than what the code actually uses.

**Fix**: Update plan to describe the actual interleaved (E, px, py, pz) layout.

---

### P1-12: `research_master_plan.md` says "KASP Clustering" but code uses Spectral Clustering

**File**: [`research_master_plan.md`](research_master_plan.md:82,104,141-144)

**Problem**: Plan repeatedly mentions "KASP Clustering" but the actual code in [`train_top.py`](scripts/train_top.py:206-258) implements **Spectral Clustering with cosine affinity**. KASP (K-means-based Approximate Spectral Clustering) is a specific variant that's not what's implemented.

**Fix**: Update plan to say "Spectral Clustering" or implement actual KASP.

---

### P1-13: `research_master_plan.md` says "Cosine k-NN Proxy" — not implemented

**File**: [`research_master_plan.md`](research_master_plan.md:145)

**Problem**: Plan describes "Cosine k-NN Proxy" for out-of-sample assignment but this is **not implemented** anywhere in the codebase.

**Fix**: Either implement it or remove from plan.

---

### P1-14: `research_master_plan.md` says "Hyperparameter sweep (8 trials)" — never done

**File**: [`research_master_plan.md`](research_master_plan.md:117-123)

**Problem**: Plan describes an 8-trial hyperparameter sweep over d=[64,128], s=[16,32], m=[0.3,0.5]. This was **never executed**.

**Fix**: Either run the sweep or mark as "future work."

---

## 🟢 P2 — Minor (Cleanup)

### P2-1: Empty `__init__.py` files

**Files**: All `__init__.py` in `arcjet/`, `arcjet/models/`, `arcjet/data/`, `arcjet/utils/`

**Problem**: All empty. No exports defined.

**Fix**: Add `__all__` exports.

---

### P2-2: `docs/` directory doesn't exist

**File**: [`README.md`](README.md:105)

**Problem**: README references `docs/` but it doesn't exist.

**Fix**: Create directory or remove reference.

---

### P2-3: `vector` import unused in `efn_arcface.py`

**File**: [`arcjet/models/efn_arcface.py`](arcjet/models/efn_arcface.py:17)

**Problem**: `import vector` never used.

**Fix**: Remove unused import.

---

### P2-4: `setup.py` author is "Your Name"

**File**: [`setup.py`](setup.py:19)

**Problem**: Placeholder text.

**Fix**: Update to actual author name.

---

### P2-5: Notebook UMAP/SC analysis was interrupted

**File**: [`notebooks/JetTagging_EFN_ArcFace.ipynb`](notebooks/JetTagging_EFN_ArcFace.ipynb:1158)

**Problem**: UMAP fitting and Spectral Clustering was **interrupted by KeyboardInterrupt**. Subclass discovery plots were never generated for the 100-epoch model.

**Fix**: Re-run analysis step on the 100-epoch model.

---

### P2-6: `research_master_plan.md` says "Round 1: Plot 4 Lund Planes" but code plots 5

**File**: [`research_master_plan.md`](research_master_plan.md:94) vs [`train_top.py`](scripts/train_top.py:1033-1048)

**Problem**: Plan says 4 T/F Lund planes but code plots 5 (1 non-split + 4 T/F splits).

**Fix**: Update plan to say 5.

---

### P2-7: `research_master_plan.md` says "Round 2: 3 Lund planes per subclass" but code plots 2-3

**File**: [`research_master_plan.md`](research_master_plan.md:105-108) vs [`train_top.py`](scripts/train_top.py:1050-1073)

**Problem**: Plan says 3 Lund planes per subclass (non-split + correct + incorrect) but code only plots 2-3 depending on whether enough samples exist for the split.

**Fix**: Update plan to reflect conditional plotting.

---

## 📋 Summary

| Priority | Count | Description |
|:--------:|:-----:|-------------|
| 🔴 P0 | 8 | Blocks paper: broken scripts, wrong data, security risk, corrupted metrics |
| 🟡 P1 | 14 | Weakens paper: wrong docs, dead code, missing features, inconsistent claims |
| 🟢 P2 | 7 | Cleanup: empty files, unused imports, interrupted analysis |

**Total: 29 issues** (6 more than previous version — found P1-11 through P1-14 and P2-6, P2-7 in the research_master_plan.md on this final pass).
