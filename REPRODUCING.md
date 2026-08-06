# Reproducing the paper (ArcEFN: top tagging with EFN + ArcFace)

This document maps **every quantitative claim** in `paper/main.tex` to the exact
command, artifact, and expected value needed to reproduce it.

## Environment

- Python 3.12 venv: `.\\.venv\\Scripts\\python.exe` (Windows; equivalent on Linux/macOS).
- Install: `pip install -r requirements-locked.txt` (if present) or the repo's dependency list.
- All analysis scripts are run **from the repository root** as modules:
  ```
  .\\.venv\\Scripts\\python.exe -m scripts.analysis.<name> [args]
  ```
  (plain `python scripts/...` fails because the `arcefn` package needs the repo root on `sys.path`).
- Data: `data/top_tagging/test.h5` (404,000 test events) — download instructions in README.
- Unless stated otherwise: **seed 42**, **no replacement**, and **full 404k test set**.
- Checkpoints live in `experiments/*/checkpoint.pt` (not tracked in git; regenerate with
  `scripts/train_arcface.py`, `scripts/train_baselines.py`). All result JSONs ARE tracked.

## Number -> command -> artifact -> expected value

| Paper claim | Command | Artifact | Expected |
|---|---|---|---|
| Accuracy 91.84% [91.76, 91.92] | `compute_paired_stats.py` | `experiments/paired_model_stats.json` | acc=0.9184; 95% binomial CI [0.9176, 0.9192] |
| Table 1 AUC / Rejection@50%TPR (0.974/0.974/0.975; 124/122/132) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/final_analysis_results.json` | arcface AUC 0.9739, rej50 123.6; linear 0.9743/121.6; coslinear 0.9748/131.7 |
| Table 1 RF baseline (92.37% / 0.977 / 150) | `compute_rf_baseline.py --max-train-events 403000 --max-events 404000 --n-estimators 500` | `experiments/rf_baseline_results.json` | acc 0.92368, AUC 0.97655, rej50 150.01 (train val.h5 -> test.h5, seed 42; features cached in `experiments/rf_cache_{val,test}.npz`) |
| McNemar p-values (4.3e-05 / 1.3e-08 / 0.16) | `compute_paired_stats.py` | `experiments/paired_model_stats.json` | pairwise discordants + continuity-corrected chi2 |
| Robustness pT rescaling 91.84% all scales | `compute_robustness_tests.py` (N_EVENTS=404000) | `experiments/robustness_tests_results.json` | 0.9184 at 0.5/0.8/1.0/1.5/2.0x |
| Robustness azimuthal rotation 91.84% all angles | same | same | 0.9184 at 0/0.1/0.5/1.0/2.0 rad (phys. phi_i -> phi_i + alpha) |
| Robustness dropout 91.84/91.04/88.95/79.23/55.91% | same | same | at 0/10/20/50/80% |
| Table 4 TwoNN 2.17, PR 1.38, center angle 0.76 rad | `compute_dimensionality.py` (N_EVENTS=50000) | `experiments/dimensionality_results.json` | twonn 2.1749; PR 1.3776; angle 0.7611 rad |
| Table 4 PCA 2 comps > 99.9% (83.7/16.3) | `compute_interpretation.py` | `experiments/interpretation_full_results.json` | pca_variance [0.837078, 0.162916, ...] |
| Table 2 DGLAP exponents (Table 2) | `compute_dglap_fit.py` (MAX_EVENTS=404000) | `experiments/ap_dglap_fit_params.json` | QCD_Core -1.267/-1.275/0.999; QCD_Edge -0.859/-0.264/0.995; Top_Core -0.543/0.118/0.967; Top_Edge 1.106/6.600/0.988 |
| Subclass counts 126,514/75,400/100,421/101,665 | `compute_interpretation.py` | same | n_qcd_core/n_qcd_edge/n_top_core/n_top_edge |
| MI top-5 dims > 0.92 with Mass; effect sizes (d=1.75 QCD, 1.11 Top tau32) | `compute_interpretation.py` | same | top_mi_pairs; effect_sizes |
| Saliency top-1 ~15%, top-3 ~40%, 12 -> 50%, 28 -> 80% | `compute_saliency.py` (N_EVENTS=5000) | `experiments/saliency_fixed_results.json` | n_for_50pct=12, n_for_80pct=28 |
| Mass-decorrelation AUC 0.89 [140,170), 90.2% [150,200) | `poc_mass_decoupled_accuracy.py` (50000) | `experiments/poc_mass_decoupled_results.json` | 0.8873 / 0.9024 |
| ~8% of jets > 160 GeV misclassified as QCD | `compute_mass_vs_score.py` (50000) | printed diagnostic | 7.6% QCD-like |
| Injection scan 0.416+-0.004 / -0.516+-0.006 / CE 0.046 | `poc_injection_scan.py` | `experiments/poc_injection_scan_results.json` | margins per 5% step, 20k/step |
| OOD JetNet30 cos values (W/Z 0.58, gluon 0.39, quark 0.45; top < -0.2 / < -0.6) | `poc_ood_generalization.py` (JETS_PER_CLASS=50000) | `experiments/poc_ood_results.json` | per-class cosine similarities |
| Head comparison 30/20/25/25 vs 45/5/43/6, R2, beta | `compare_head_geometry.py` (N_EVENTS=50000) | `experiments/head_comparison_params.json` | N per subclass + fit params |
| Embedding norms 7.73/4.94 (ArcFace), 3.69/2.40 (Linear) | `compute_alignment_uniformity.py` (50000) | `experiments/embedding_geometry_results.json` | medians + std |
| EDA table (train 1,211,000 / val 403,000 / test 404,000; means) | `compute_eda.py` | `experiments/eda/eda_stats.json` | per-split stats |
| EDA supplement sigma(pT)=29, sigma(M)=56, corr 0.94/0.62, tau32 0.55/0.73, mult diff ~11 | `compute_eda_supplement.py` (n=100k, seed 42) | `experiments/eda_supplement_results.json` | pT std 28.738; M std 56.432; corr 0.9436/0.6244; tau32 0.548/0.733; mult 54.70/43.37 |
| Conformal coverage 90.1% | deleted script; see `git show 61120f0^:scripts/explore/conformal_final.py` | `experiments/conformal_results.json` | empirical_coverage 0.90066 |
| PGD drop 0.02% (eps=0.01, n=5000) | deleted script; see `git show 61120f0^:scripts/explore/pgd_attack.py` | `experiments/pgd_attack_results.json` | 0.9148 vs clean 0.915 |
| Figures (ROC, rejection rates, NMI/ARI) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/final_roc_curves.png`, `final_rejection_rates.png`, `final_nmi_ari.png` | copied into `paper/figures/` |
| Lund planes, UMAP, composition, physics distributions | `generate_paper_figures.py` (MAX_EVENTS=100000) | `experiments/robustness_s16_m05/*.png` | figures (100k events — captions say 100k) |
| Angular collimation | `generate_angular_collimation.py` (404000) | `experiments/robustness_s16_m05/angular_collimation.png` | scaling exponents |

## Statistical claims (Section "Statistical Considerations")

- 95% binomial CI at n=404,000, p=0.9184: `p +- 1.96*sqrt(p(1-p)/n)` = +-0.000844 -> [91.76, 91.92] (analytic; matches bootstrap CI [91.75, 91.92]).
- Subsampled analyses use seeded (seed 42) random draws without replacement, shared across models -> strictly paired (McNemar valid).
- Power: unpaired two-proportion test on 404k resolves ~0.17 pp at 80% power; McNemar resolves differences below that; per-group n for Cohen d = 15.7/d^2 (393 for d=0.2; ~5 for d=1.75); all subclass groups >= 75k.

## Superseded / NOT to be used

The following artifacts predate the final analysis and contain outdated or
incorrect numbers. They are tracked for completeness only and must not be
cited:

- `comprehensive_analysis_results.json` — orphaned producer script (never in repo); l4 dropout/rotation numbers WRONG
- `full_analysis_results.json` — old pre-audit numbers
- `ap_splitting_results.json` — superseded by `ap_dglap_fit_params.json` (bounds change)
- `robustness_results.json`, `robustness_results_table.md` — old s x m sweep, different n and method
- `bayesian_results*.json`, `decision_tree_results.json`, `dt_rules_results.json`, `grooming_sweep_results.json`, `onnx_results.json`, `emd_validation_results.json`, `interpretation_results.json` — exploratory, not in paper
- `alignment_uniformity_results.json` — superseded by `embedding_geometry_results.json`
- `arcface_100ep/metrics.json` — final-epoch checkpoint metrics differ from best-val checkpoint used for all paper numbers
- `baselines_coslinear_seed_*` `metrics.json` roc_auc — final-epoch anomaly; paper uses best-val checkpoints via `compute_bootstrap_ci.py`

## Generative-AI disclosure

See the "Use of Generative AI Tools" section at the end of `paper/main.tex`.
When submitting to SciPost, also fill in the "Disclosure of Generative AI use"
field of the submission form with the same statement.

## Git hygiene

- `experiments/*.json` (small result files) ARE tracked; checkpoints, `.npz`, `.png`, `.onnx` are ignored (regenerable).
- Repo: https://github.com/thirathep-ph/ArcEFN (branch `main`).
