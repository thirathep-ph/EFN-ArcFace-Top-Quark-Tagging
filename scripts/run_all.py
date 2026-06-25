"""
Run Everything: Baselines + Robustness + SVD + Bootstrap
=======================================================
Runs all experiments sequentially so you can leave it overnight.

Directory structure: each experiment gets ONE directory containing
both checkpoints and results:
  experiments/{name}/checkpoint.pt
  experiments/{name}/metrics.json
  experiments/{name}/history.json
  experiments/{name}/embeddings.npz
  experiments/{name}/plots/*.png

Uses --no-chunk mode (loads all data into RAM, ~2.4GB) for faster training.
Uses num_workers=0 for DirectML compatibility on Windows.

Phases:
  1. Baselines: Linear (3 seeds) + CosLinear (3 seeds)
  2. Robustness: 9 trials (s x m)
  3. SVD analysis
  4. Bootstrap
"""
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  [{desc}]")
    print(f"  {cmd}")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, capture_output=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{desc}] {status} in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
    sys.stdout.flush()
    return result.returncode


def checkpoint_exists(exp_dir):
    """Check if an experiment directory has a completed training (checkpoint + history)."""
    ckpt = os.path.join(exp_dir, "checkpoint.pt")
    history = os.path.join(exp_dir, "history.json")
    return os.path.exists(ckpt) and os.path.exists(history)


def metrics_exist(exp_dir):
    """Check if an experiment directory has metrics (either name)."""
    for name in ["metrics.json", "advanced_metrics.json"]:
        if os.path.exists(os.path.join(exp_dir, name)):
            return True
    return False


def main():
    total_t0 = time.time()
    errors = []

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # =========================================================================
    # Phase 1: Baselines
    # =========================================================================
    print("\n" + "="*70)
    print("  PHASE 1: BASELINES")
    print("="*70)
    sys.stdout.flush()

    baseline_configs = [
        ("linear", "42"),
        ("linear", "123"),
        ("linear", "7"),
        ("coslinear", "42"),
        ("coslinear", "123"),
        ("coslinear", "7"),
    ]

    for model_name, seed in baseline_configs:
        exp_dir = os.path.join(EXPERIMENTS_DIR, f"baselines_{model_name}_seed_{seed}")
        if metrics_exist(exp_dir):
            print(f"\n  [SKIP] baselines/{model_name}/seed_{seed} — metrics exist")
            continue

        rc = run(
            f'python scripts/train_baselines.py --model {model_name} '
            f'--epochs 50 --seeds {seed} --no-chunk '
            f'--output_dir "{exp_dir}" --checkpoint_dir "{exp_dir}"',
            f"Baseline: {model_name} seed={seed}"
        )
        if rc != 0:
            errors.append(f"Phase 1 ({model_name} seed={seed}) failed with code {rc}")

    # =========================================================================
    # Phase 2: Robustness
    # =========================================================================
    print("\n" + "="*70)
    print("  PHASE 2: ROBUSTNESS ANALYSIS (9 trials)")
    print("="*70)
    sys.stdout.flush()

    robustness_configs = [
        (8, 0.0), (8, 0.3), (8, 0.5),
        (16, 0.0), (16, 0.3), (16, 0.5),
        (32, 0.0), (32, 0.3), (32, 0.5),
    ]

    for s, m in robustness_configs:
        m_str = str(m).replace(".", "")
        exp_dir = os.path.join(EXPERIMENTS_DIR, f"robustness_s{s}_m{m_str}")

        if checkpoint_exists(exp_dir):
            print(f"\n  [SKIP] robustness/s{s}/m{m_str} — checkpoint exists")
            continue

        rc = run(
            f'python scripts/train_top.py --epochs 50 --scale {s} --margin {m} '
            f'--output_dir "{exp_dir}" --checkpoint_dir "{exp_dir}" --no-chunk --train-only',
            f"Robustness: s={s}, m={m}"
        )
        if rc != 0:
            errors.append(f"Phase 2 (s={s}, m={m}) failed with code {rc}")

    # =========================================================================
    # Phase 3: SVD Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("  PHASE 3: SVD ANALYSIS")
    print("="*70)
    sys.stdout.flush()

    svd_dir = os.path.join(EXPERIMENTS_DIR, "svd_analysis")
    os.makedirs(svd_dir, exist_ok=True)

    rc = run(
        f'python scratch/analysis/compute_svd_variance.py',
        "SVD variance analysis"
    )
    if rc != 0:
        errors.append(f"Phase 3 (SVD variance) failed with code {rc}")

    rc = run(
        f'python scratch/analysis/compute_weight_svd.py',
        "SVD weight analysis"
    )
    if rc != 0:
        errors.append(f"Phase 3 (weight SVD) failed with code {rc}")

    # =========================================================================
    # Phase 4: Bootstrap
    # =========================================================================
    print("\n" + "="*70)
    print("  PHASE 4: BOOTSTRAP")
    print("="*70)
    sys.stdout.flush()

    rc = run(
        f'python scratch/analysis/compute_bootstrap_stability.py',
        "Bootstrap confidence intervals"
    )
    if rc != 0:
        errors.append(f"Phase 4 (bootstrap) failed with code {rc}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.time() - total_t0
    print("\n" + "="*70)
    print("  ALL DONE!")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  All phases completed successfully!")
    print(f"  Experiments saved to: {EXPERIMENTS_DIR}")
    print("="*70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
