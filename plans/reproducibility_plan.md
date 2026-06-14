# 📋 Research-Grade Reproducibility Plan

> Goal: Make the codebase ready for public release and paper submission.
> Every experiment must be reproducible with a single command.

---

## Phase 1: Fix Code Bugs (CPU, ~30 min)

### 1.1 `evaluate_models.py` — Wrong ArcFace params ✅ DONE
- Changed `s=32.0, m=0.0` → `s=16.0, m=0.5`

### 1.2 `train_baselines.py` — Rewrite for chunked loading
**Problem**: Hardcodes `max_events=50000` and `F` import inside `if __name__`.
**Fix**: 
- Move `import torch.nn.functional as F` to top
- Replace single 50K load with chunked loop matching `train_top.py`
- Use `get_h5_len`, `chunk_size`, shuffle chunk order
- Evaluate on full test set (404K), not 100K

### 1.3 `train_top.py` — Minor issues
- `distance_corr_loss` imported but unused → add comment or remove import
- Default `--epochs 30` is fine for testing, but document that 100 was used for final model

---

## Phase 2: Make Reproducible (CPU, ~1 hr)

### 2.1 Seed Everything
- All random operations must be seedable: `torch.manual_seed()`, `np.random.seed()`, `random.seed()`
- `train_baselines.py` already has `--seeds` argument ✅
- `train_top.py` needs a `--seed` argument (currently uses default torch seed)

### 2.2 Freeze Dependencies
```bash
.venv_gpu/bin/pip freeze > requirements-locked.txt
```
This creates a file with exact versions (e.g., `torch==2.12.0+rocm6.2`).

### 2.3 Configuration File
Create `config.yaml` or use existing `jet_tagging/config.json` as the single source of truth:
```json
{
  "model": {
    "embedding_dim": 64,
    "particle_dim": 128,
    "s": 16.0,
    "m": 0.5,
    "warmup_epochs": 10
  },
  "training": {
    "epochs": 100,
    "batch_size": 128,
    "accumulation_steps": 8,
    "learning_rate": 1e-3,
    "scheduler": "OneCycleLR",
    "max_lr": 2e-3
  },
  "data": {
    "chunk_size": 100000,
    "max_constituents": 200
  },
  "clustering": {
    "method": "spectral",
    "affinity": "cosine",
    "max_k": 10
  }
}
```

### 2.4 Add `--seed` argument to `train_top.py`
Currently `train_top.py` has no seed control. Add:
```python
parser.add_argument('--seed', type=int, default=42, help='Random seed')
```

---

## Phase 3: Run Baseline Training (GPU, ~3 hrs)

### 3.1 Fix `train_baselines.py` first
Rewrite to use chunked loading. The structure should mirror `train_top.py`:
- Same chunk size (100,000)
- Same optimizer (Adam, lr=1e-3)
- Same scheduler (OneCycleLR, not CosineAnnealingLR)
- Same batch size (128) and accumulation (8)

### 3.2 Run Command
```bash
# After fixing:
python scripts/train_baselines.py --model both --epochs 30 --seeds 42 123 7
```

### 3.3 Expected Output
```
results/baselines/
├── linear/
│   ├── seed_42/metrics.json
│   ├── seed_42/history.json
│   ├── seed_123/metrics.json
│   └── seed_7/metrics.json
└── coslinear/
    ├── seed_42/metrics.json
    └── ...
```

---

## Phase 4: Robustness Analysis (GPU, ~4-5 hrs)

### 4.1 Create `scripts/robustness_sweep.py`
New script that trains with different (s, m) combinations:

```python
# Pseudocode
s_values = [8, 16, 32]
m_values = [0.0, 0.3, 0.5]

for s in s_values:
    for m in m_values:
        model = TopTaggingModel(s=s, m=m, embedding_dim=64)
        train(model, epochs=30)
        evaluate(model)  # accuracy, centroid mass, k from SC
        save_results(s, m, metrics)
```

### 4.2 Output: 3×3 Heatmap
```
        m=0.0   m=0.3   m=0.5
s=8     91.2%   91.4%   91.5%
s=16    91.3%   91.6%   91.7%
s=32    91.1%   91.3%   91.4%
```

If all 9 cells are within ±0.5%, the claim is: **"ArcFace is robust to hyperparameter choice."**

### 4.3 Also Measure Per Combination
- Centroid mass (should be ~170 GeV for all)
- k from eigengap (should be 2 for all)
- Cosine separation gap

---

## Phase 5: Generate Paper Figures & Tables (CPU, ~2 hrs)

### 5.1 Create `scripts/generate_paper_tables.py`
Reads all results and produces:
- Table 1: Classification (ArcFace vs Linear vs CosLinear, mean±std over 3 seeds)
- Table 2: Robustness heatmap (9 combinations)
- Table 3: Subclass physics (from existing data)
- Table 4: Clustering comparison (SC vs K-means vs Agglo)

### 5.2 Create `scripts/generate_paper_figures.py`
Re-generates all figures with consistent styling:
- Font sizes, color schemes, resolution (300 DPI)
- All figures in `paper_figures/` directory

### 5.3 3D UMAP Visualization
**Script**: [`scratch/analysis/plot_3d_umap.py`](scratch/analysis/plot_3d_umap.py) — already exists but needs cleanup:
- [ ] Remove hardcoded Windows path (`artifact_dir`) at line 17
- [ ] Fix `sys.path.append` to work from project root
- [ ] Add command-line arguments for input/output paths
- [ ] Generate both static (rotating GIF) and interactive (plotly) versions
- [ ] Color by: prediction status (TP/TN/FP/FN), subclass, and physics variables (mass, tau32)

**Purpose**: 3D UMAP reveals latent space structure that 2D projections can hide — especially the hyperspherical geometry of ArcFace embeddings.

---

## Phase 6: Write Paper

### 6.1 Structure
```
1. Introduction
   - Problem: Jet taggers are black boxes
   - Solution: Ante-hoc interpretability via ArcFace prototypes
   - Contribution: First demonstration for HEP

2. Related Work
   - Jet tagging architectures
   - Interpretable ML in HEP
   - ArcFace in computer vision

3. Methodology
   - EFN backbone (IRC-safe)
   - ArcFace head (hyperspherical prototypes)
   - Spectral Clustering for subclass discovery

4. Experiments
   - Dataset: Top Tagging Reference Dataset
   - Classification performance (Table 1)
   - Robustness analysis (Table 2)
   - Subclass discovery (Table 3)
   - Physics validation (Lund, zg, collimation)

5. Discussion
   - What prototypes learned (centroid mass ≈ m_top)
   - Core vs Edge subclasses
   - Limitations and future work

6. Conclusion
```

### 6.2 Target Venue
- **Primary**: SciPost Physics Core
- **Backup**: MLST (Machine Learning: Science and Technology)
- **Safe**: JINST

---

## Phase 7: Public Release

### 7.1 Security
- [ ] Remove hardcoded GitHub token from notebook ✅ DONE
- [ ] Check for any other secrets in code

### 7.2 Documentation
- [ ] Update README with full reproduction instructions
- [ ] Add `CITATION.cff` file for citation metadata
- [ ] Add `LICENSE` file (MIT)

### 7.3 Clean Up
- [ ] Remove `amdgpu-install_*.deb` files from repo root
- [ ] Remove `.venv/` and `.venv_gpu/` from git tracking
- [ ] Add `scratch/archive/` to `.gitignore` ✅ DONE

### 7.4 Final Git State
```bash
git add -A
git commit -m "Research-grade release: reproducible experiments + documentation"
git tag v1.0.0
git push --tags
```

---

## Timeline Estimate

| Phase | Time | Requires GPU? | Can Start? |
|:------|:----:|:-------------:|:----------:|
| 1. Fix bugs | ~30 min | ❌ | ✅ Now |
| 2. Reproducibility | ~1 hr | ❌ | ✅ Now |
| 3. Baselines | ~3 hrs | ✅ | After GPU |
| 4. Robustness | ~4-5 hrs | ✅ | After GPU |
| 5. Figures | ~2 hrs | ❌ | After Phase 3-4 |
| 6. Paper | 2-3 weeks | ❌ | Can start anytime |
| 7. Release | ~1 hr | ❌ | After paper |

**Total GPU time**: ~7-8 hours
**Total CPU time**: ~4-5 hours + writing
