# 🚀 Migration Guide — Moving to New Machine

## Current State

**All files live on**: `F:/Jet Tagging/` (SSD being moved)

**What's in Git (code + plans only)**:
- Clean codebase (no checkpoints, embeddings, results, data)
- All planning documents in `plans/`
- Model classes (efn_arcface.py, efn_linear.py, efn_coslinear.py)
- Training scripts (train_top.py, train_baselines.py, evaluate_models.py)
- Scratch analysis scripts

**What's on local only (SSD)**:
- Checkpoints (checkpoints/) — all trained models
- Results (results/, main_arcface/) — all analysis outputs
- Embeddings (main_arcface/embeddings.npz, etc.)
- Data (data/top_tagging/) — H5 datasets
- Plots (results/analysis_15ep/*.png, etc.)

---

## Before Moving

### 1. Verify everything pushed to GitHub

```bash
cd "F:/Jet Tagging"
git status    # Should show: "nothing to commit, working tree clean"
git push      # Push any remaining commits
```

### 2. Check .gitignore is correct

The following should NOT be in git (already excluded):
```
*.h5, *.hdf5, *.pt, *.pth, *.ckpt, *.json, *.png, *.npz, *.npy
results/, .vscode/
```

### 3. Move the SSD

The entire SSD is being physically moved. All files will be preserved.

---

## After Moving — Setup on New Machine

### Step 1: Reconnect SSD

Make sure the SSD is mounted at a path like `F:/Jet Tagging/` or similar.

### Step 2: Reinstall Python environment

```bash
cd "F:/Jet Tagging"
pip install -e .           # Install jet_tagging package
pip install -r requirements.txt  # Install all dependencies
```

### Step 3: Verify everything works

```bash
cd "F:/Jet Tagging"
python -c "from jet_tagging.models.efn_arcface import TopTaggingModel; print('OK')"
```

### Step 4: Pull latest from GitHub (if you pushed from old machine)

```bash
cd "F:/Jet Tagging"
git pull
```

### Step 5: Resume Discussion with AI

When starting a new chat session, share these context files:

**Primary (must read)**:
```
plans/session_log_2025-06-07.md    # Day 1 complete log
plans/session_log_2025-06-08.md    # Day 2 complete log
plans/fitness_assessment.md         # Current state assessment
plans/publication_roadmap.md        # Target: SciPost Physics Core
```

**Key (summarize what we decided)**:
```
- Target: SciPost Physics Core (open access, physics focus, free)
- Narrative: "Core vs Edge" subclasses via parameter-free SC
- Architecture: EFN + ArcFace (ante-hoc interpretable)
- k=2 subclasses confirmed by 3 cosine-compatible methods
- Per-subclass purity: Core 98-99%, Edge 82-83%
- Bootstrap CI: 91.75% [91.66%, 91.83%]
- Remaining: Train Linear + CosLinear baselines (GPU, ~3 hrs, 3 seeds)
- Then: Write paper, submit
```

**Quick start command**:
```bash
# When you have GPU:
python scripts/train_baselines.py --model both --epochs 30 --seeds 42 123 7
```

---

## Quick Reference

| Question | Answer |
|:---------|:-------|
| **What to train?** | `python scripts/train_baselines.py --model both --epochs 30 --seeds 42 123 7` |
| **Where's the best model?** | `main_arcface/best_model_efn.pt` |
| **What's the accuracy?** | 91.75% (95% CI: [91.66, 91.83]) |
| **How many subclasses?** | k=2 per class (eigengap heuristic) |
| **Where to submit?** | SciPost Physics Core |
| **What's remaining?** | Train baselines + write paper |

---

## File Map — What's Where

```
F:/Jet Tagging/
├── README.md                            # Project overview (UPDATED)
├── requirements.txt                     # Dependencies
├── setup.py                             # Package setup
├── .gitignore                           # Excludes checkpoints/data/results
│
├── jet_tagging/                         # Core library
│   ├── models/
│   │   ├── efn_arcface.py               # Main model (EFN + ArcFace)
│   │   ├── efn_linear.py                # Model A: Linear+CE (NEW)
│   │   └── efn_coslinear.py             # Model B: CosLinear+CE (NEW)
│   ├── data/loader.py                   # H5 → PyTorch pipeline
│   └── utils/
│       ├── physics.py                   # FastJet substructure
│       └── plotting.py                  # Visualization
│
├── scripts/
│   ├── train_top.py                     # Main training script
│   ├── train_baselines.py               # Baseline training (NEW)
│   └── evaluate_models.py              # Model evaluation
│
├── notebooks/
│   └── JetTagging_EFN_ArcFace.ipynb    # Colab notebook
│
├── plans/                               # ALL PLANNING DOCS (NEW)
│   ├── session_log_2025-06-07.md       # Day 1 log
│   ├── session_log_2025-06-08.md       # Day 2 log
│   ├── publication_roadmap.md          # SciPost strategy
│   ├── baseline_implementation_plan.md  # A/B/C ablation
│   ├── reviewer_simulation.md          # 7 reviewer questions
│   ├── reviewer_responses.md           # Response strategies
│   ├── paper_results_table.md           # 8 tables ready
│   ├── fitness_assessment.md           # Go/No-Go decision
│   ├── multi_perspective_review.md     # 6 perspectives
│   ├── improvement_ideas.md            # 13 improvement ideas
│   └── MIGRATION_GUIDE.md              # This file
│
├── scratch/                             # Analysis scripts
│   ├── comprehensive_clustering_comparison.py  # 8-method comparison
│   ├── compare_sc_vs_kmeans.py          # SC vs K-means
│   ├── compute_svd_variance.py          # SVD analysis
│   ├── compute_bootstrap_stability.py   # Bootstrap
│   ├── run_full_analysis.py            # Full pipeline
│   └── ...                             # Other tools
│
├── main_arcface/                        # BEST MODEL RESULTS
│   ├── best_model_efn.pt               # Trained checkpoint (100 epochs)
│   ├── advanced_metrics.json           # Acc 91.74%, F1 91.89%
│   ├── svd_scree_plot.png              # SVD dimension analysis
│   ├── centroid_constituents_report.md  # Top centroid = 171.2 GeV
│   ├── embeddings.npz                  # 404K test embeddings
│   └── lund_planes/                    # Hierarchical Lund planes
│
├── results/                             # Previous experiment runs
│   ├── analysis_15ep/                   # Full analysis output
│   ├── test_run_50k/                    # With statistical_metrics.json
│   ├── test_run_15ep/                   # 15-epoch run
│   └── ...                             # Other runs
│
├── checkpoints/                         # Model checkpoints
│   ├── test_run_15ep/                   # 15-epoch model
│   ├── test_run_50k/                    # 50K run
│   ├── baseline/                        # Baseline placeholder
│   └── ...                             # Other checkpoints
│
└── data/
    └── top_tagging/                     # H5 datasets (train/val/test.h5)
```
