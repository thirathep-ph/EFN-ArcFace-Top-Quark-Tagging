# ArcEFN code repo

Work here for models, training, analysis scripts, experiments. Paper lives in `D:\research\ArcEFN-paper`; slides in `D:\research\ArcEFN-talks`. Old mixed repo `D:\research\jet_tagging` is read-only reference.

## Environment

- Python 3.12, `.venv` for analysis/training, `.venv_pysr` for PySR.
- Canonical run: `experiments/robustness_s16_m05/` (404k test events, `checkpoint.pt`, 64-D `embeddings.npz`).
- Data in `data/` (gitignored); download via `scripts/download_data.py`.

## Key entry points

- Train: `scripts/train_arcface.py --scale 16 --seed 42`
- SSOT: `scripts/analysis/create_ssot.py` (ENTRIES) -> `experiments/ssot_paper_numbers.json` (122 entries); gate `scripts/analysis/verify_ssot.py`
- PySR config: `scripts/analysis/pysr_config.py` is the single source of truth for operators, constraints, normalization.

## Conventions

- Code repo and paper repo stay separate; never mix commits.
- Results/checkpoints in `experiments/` or `checkpoints/`; generated plots in `outputs/` or `paper/`.
- Reproducibility: lock deps, keep experiment configs in `configs/`.
- Training can take hours; support `--quick` sanity runs.

## Current paper status

- PySR is removed from `main.tex` for now; restore point is `paper/_removed_pysr.tex` in the paper repo.
- PySR finding: complexity 4 converges across 5 seeds (affine-equivalent, residual ~1e-15) and is the Pareto elbow of every seed. Complexity >=6 splits into two branches (with/without `tau32`); no unique closed form.
- Do not report PySR coefficients (`k` is gauge-dependent: 0.33 raw vs 2.09 step profile). Report form convergence only.
- Canonical test set is 404k (`final_analysis_results.json` at git HEAD). The 100k root file was a stale rerun.
- `tab:phys-evol` must use `prototype_distance_evolution_10bin.json`.

## Cleanup

- Scratch/probe scripts live in `archive/` (gitignored). Live producers and SSOT modules stay in `scripts/analysis/`.
- `archive/` is intentionally not versioned.
