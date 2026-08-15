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
- Main-run provenance (`experiments/robustness_s16_m05/`): `checkpoint.pt` (2026-06-24) and
  `embeddings.npz` (2026-07-03) are the authoritative model + embedding artifacts behind every
  test-set number. `history.json` (2026-08-15) is the training/validation curve (regenerable);
  its final val_acc is 0.9175, and the paper's 0.9184 is the disjoint test-set accuracy.
  `arcefn/config.json` was previously populated from the superseded 100-epoch run
  (`arcface_100ep/metrics.json`, 0.91745 acc / 46,592 params) — corrected 2026-08-15 to the
  main run (59,072 params).

## Number -> command -> artifact -> expected value

| Paper claim | Command | Artifact | Expected |
|---|---|---|---|
| Accuracy 91.84% [91.76, 91.92] | `compute_paired_stats.py` | `experiments/paired_model_stats.json` | acc=0.9184; 95% binomial CI [0.9176, 0.9192] |
| Table 1 AUC / Rejection@50%TPR (0.974/0.974/0.975; 124/122/132) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/final_analysis_results.json` | arcface AUC 0.9739, rej50 123.6; linear 0.9743/121.6; coslinear 0.9748/131.7 |
| Table 1 RF baseline (92.37% / 0.977 / 150) | `compute_rf_baseline.py --max-train-events 403000 --max-events 404000 --n-estimators 500` | `experiments/rf_baseline_results.json` | acc 0.92368, AUC 0.97655, rej50 150.01 (train val.h5 -> test.h5, seed 42; features cached in `experiments/rf_cache_{val,test}.npz`) |
| McNemar p-values (4.3e-05 / 1.3e-08 / 0.16) | `compute_paired_stats.py` | `experiments/paired_model_stats.json` | pairwise discordants + continuity-corrected chi2 |
| ArcFace seed variability (91.84 / 91.88 / 91.83% for seeds 42/123/7) | `compute_arcface_seed_metrics.py` | `experiments/arcface_seed_metrics.json` | test acc + confusion matrix + best/final val_acc per seed (full 404k forward pass) |
| Robustness 4-momentum (JES) rescaling 91.84% all scales | `compute_robustness_tests.py` (N_EVENTS=404000) | `experiments/robustness_tests_results.json` | 0.9184 at 0.5/0.8/1.0/1.5/2.0x (full E,px,py,pz scaling) |
| Robustness azimuthal rotation 91.84% all angles | same | same | 0.9184 at 0/0.1/0.5/1.0/2.0 rad (phys. phi_i -> phi_i + alpha) |
| Robustness dropout 91.84/91.04/88.95/79.23/55.91% | same | same | at 0/10/20/50/80% |
| Table 4 TwoNN 2.17, PR 1.38, center angle 0.76 rad | `compute_dimensionality.py` (N_EVENTS=50000) | `experiments/dimensionality_results.json` | twonn 2.1749; PR 1.3776; angle 0.7611 rad |
| Table 4 PCA 2 comps > 99.9% (83.7/16.3) | `compute_interpretation.py` | `experiments/interpretation_full_results.json` | pca_variance [0.837078, 0.162916, ...] |
| Table 2 DGLAP exponents (Table 2) | `compute_dglap_fit.py` (MAX_EVENTS=404000) | `experiments/dglap_fit_params.json` | QCD_Core -1.267/-1.275/0.999; QCD_Edge -0.859/-0.264/0.995; Top_Core -0.543/0.118/0.967; Top_Edge 1.106/6.600/0.988 |
| Subclass counts 123,322/78,592/101,260/100,826 | `compute_interpretation.py` | same | n_qcd_core/n_qcd_edge/n_top_core/n_top_edge (canonical spectral partition; an earlier version of the interpretation analysis used KMeans labels, giving 126,514/75,400/100,421/101,665 on the same embeddings — replaced 2026-08-10 so all subclass analyses share the spectral partition of Table 2; the 2026-08-10 spectral rerun of `compute_interpretation.py` reports 123,320/78,594/101,263/100,823, within +/-2-3 events per class of these counts due to float-level differences between fresh inference and the saved embeddings (k=2, both sum to 404,000)) |
| MI top-5 dims > 0.92 with Mass; effect sizes (d=1.73 QCD, 1.11 Top tau32) | `compute_interpretation.py` | same | top_mi_pairs; effect_sizes |
| Saliency top-1 ~15%, top-3 ~40%, 12 -> 50%, 28 -> 80% | `compute_saliency.py` (N_EVENTS=5000) | `experiments/saliency_fixed_results.json` | n_for_50pct=12, n_for_80pct=28 |
| Mass-decorrelation AUC 0.89 [140,170), 90.2% [150,200) | `poc_mass_decoupled_accuracy.py` (50000) | `experiments/poc_mass_decoupled_results.json` | 0.8873 / 0.9024 |
| ~8% of jets > 160 GeV misclassified as QCD | `compute_mass_vs_score.py` (50000) | printed diagnostic | 7.6% QCD-like |
| Injection scan 0.416+-0.004 / -0.516+-0.006 / CE 0.046 | `poc_injection_scan.py` | `experiments/poc_injection_scan_results.json` | margins per 5% step, 20k/step |
| OOD JetNet30 cos values (W/Z 0.58, gluon 0.39, quark 0.45; top < -0.2 / < -0.6) | `poc_ood_generalization.py` (JETS_PER_CLASS=50000) | `experiments/poc_ood_results.json` | per-class cosine similarities |
| Head comparison 30/20/25/25 vs 45/5/43/6, R2, beta | `compute_head_geometry.py` (N_EVENTS=50000) | `experiments/head_geometry_params.json` | N per subclass + fit params; fit P(zg)=A·zg^α·(1-zg)^β (amplitude A included, matches Table 5) |
| Embedding norms 7.73/4.94 (ArcFace), 3.69/2.40 (Linear) | `compute_alignment_uniformity.py` (50000) | `experiments/embedding_geometry_results.json` | medians + std |
| EDA table (train 1,211,000 / val 403,000 / test 404,000; means) | `compute_eda.py` | `experiments/eda/eda_stats.json` | per-split stats |
| EDA supplement sigma(pT)=29, sigma(M)=56, corr 0.94/0.62, tau32 0.55/0.73, mult diff ~11 | `compute_eda_supplement.py` (n=100k, seed 42) | `experiments/eda_supplement_results.json` | pT std 28.738; M std 56.432; corr 0.9436/0.6244; tau32 0.548/0.733; mult 54.70/43.37 |
| Conformal coverage 90.1% | deleted script; see `git show 61120f0^:scripts/explore/conformal_final.py` | `experiments/conformal_results.json` | empirical_coverage 0.90066 |
| Adversarial FGSM / PGD drop 0.7% / 0.7% (eps=0.01, n=20k, seed 42) | `compute_adversarial_tests.py` | `experiments/adversarial_results.json` | clean 0.9218; FGSM 0.9146 (0.72% drop); PGD 0.91435 (0.745% drop) |
| Mass-cut 83.8% @ 160 GeV / 88.9% @ 140 GeV (optimal) | `compute_mass_cut.py` (reads `experiments/rf_cache_test.npz`) | `experiments/mass_cut_results.json` | acc per cut: 100->0.8165, 120->0.8682, 140->0.8891, 150->0.8775, 160->0.8379, 172->0.7283 |
| IRC injection (200 jets): mean cos > 0.999 at all levels | `compute_irc_injection.py` (N_EVENTS=200) | `experiments/irc_injection_results.json` | soft e0.01 mean 0.99988 / min 0.99523; e0.001 mean 0.999999; collinear mean 0.9999999; masking 1.0 |
| Figures (ROC, rejection rates, NMI/ARI) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/final_roc_curves.png`, `final_rejection_rates.png`, `final_nmi_ari.png` | copied into `paper/figures/`; NMI/ARI from jet embeddings (out[-1]), not logits |
| Lund planes, UMAP, composition, physics distributions | `generate_paper_figures.py` (MAX_EVENTS=100000) | `experiments/robustness_s16_m05/*.png` | figures (100k events — captions say 100k) |
| Angular collimation | `generate_angular_collimation.py` (404000) | `experiments/robustness_s16_m05/angular_collimation.png` | scaling exponents |
| Clustering robustness: k=2 robust (SC vs SKM vs vMF); SKM ARI 0.94 QCD / 0.99 Top | `compute_subclass_clustering.py` | `experiments/subclass_clustering_comparison.json` | QCD SC sil 0.838 vs SKM 0.844; Top all 0.736; vMF degenerate (kappa cap) |
| Subclass angular concentration (12.6/24.5 deg QCD Core/Edge; 14.8/13.5 deg Top) | `compute_subclass_concentration.py` | `experiments/subclass_concentration_results.json` | rbar + kappa_hat + theta_mean per subclass |
| Mass-matched DGLAP control (Top 178/174 GeV tau32 0.63/0.46; QCD 64/145 GeV, [95,133) GeV bin Core -1.89 vs Edge -1.38) | `compute_mass_matched_dglap.py` | `experiments/mass_matched_dglap.json` | per-subclass mean mass/tau32 + per-mass-bin alpha/beta/R2/delta_alpha |
| Subclass composition by prediction status (FP = 100% QCD-Edge; FN = 100% Top-Core; Top-Edge 100% TP) | `compute_subclass_composition.py` | `experiments/subclass_composition.json` | per-subclass % in TP/FP/TN/FN (canonical partition, full 404k) |

## Statistical claims (Section "Statistical Considerations")

- 95% binomial CI at n=404,000, p=0.9184: `p +- 1.96*sqrt(p(1-p)/n)` = +-0.000844 -> [91.76, 91.92] (analytic; matches bootstrap CI [91.75, 91.92]).
- Subsampled analyses use fixed, deterministic subsets of the test set (the first N events, or a seeded random draw with seed 42 where noted), shared across models -> strictly paired (McNemar valid).
- Power: unpaired two-proportion test on 404k resolves ~0.17 pp at 80% power; McNemar resolves differences below that; per-group n for Cohen d = 15.7/d^2 (393 for d=0.2; ~5 for d=1.75); all subclass groups >= 75k.

## Superseded / NOT to be used

The following artifacts predate the final analysis and contain outdated or
incorrect numbers. They are tracked for completeness only and must not be
cited:

- `comprehensive_analysis_results.json` — orphaned producer script (never in repo); l4 dropout/rotation numbers WRONG
- `full_analysis_results.json` — old pre-audit numbers
- `ap_splitting_results.json` — superseded by `dglap_fit_params.json` (bounds change)
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

## Pending / planned extensions (decided, not yet executed)

1. **Full-embedding symbolic distillation with SymTorch** (user's choice A, noted 2026-08-07):
   - Goal: replace the ENTIRE learned mapping (per-particle `phi` 128-d and/or the 64-d embedding) with closed-form symbolic expressions, going BEYOND the 2D decision-plane distillation which is capped at ~85.2% accuracy (info loss from the LS plane projection, R2=0.677; see `experiments/esr/plane_ceiling.json`).
   - Tool: `torch-symbolic` (SymTorch, PyPI) — PySR-powered, torch-native, by E. Tan, A. Soubki, M. Cranmer (arXiv 2602.21307). Depends on `pysr`; requires Python >=3.11 (repo venv is 3.12; install via `pip install torch-symbolic`).
   - Existing POC artifacts to build on: `experiments/esr/{perparticle.npz, jet.npz, phi_sym.json, readout_sym.json, validation_results.json, plane_ceiling.json}`; scripts `compute_esr_targets.py` (exact-plane LS targets), `fit_esr.py` (PySR two-stage), `compute_esr_validate.py` (full-404k closed-form validation), `compute_esr_ceiling.py`.
   - Open questions when we resume: per-output distillation of the 128-d `phi` map vs end-to-end neuro-symbolic training; whether a hybrid model (symbolic surrogates inside EFN) can match 91.84% accuracy while keeping the ~17x speedup measured for the 2D plane.

2. **vMF options not yet done** (from the subclass-clustering comparison study, `experiments/subclass_clustering_comparison.json`):
   - B) soft-membership analysis: vMF membership entropy per jet vs mass — QCD boundary blurry vs Top sharp (physics finding).
   - C) vMF on the intrinsic 2D latent plane (cos_QCD, cos_Top) — connects effective-dim-2 story.
   - Note: plain vMF EM fails on this data (kappa hits 600 cap; BIC monotone; QCD partition diverges ARI 0.619) — needs MAP prior / bounded-kappa if revisited.

3. **2D-latent clustering study** (script WRITTEN but NOT run; user deferred, noted 2026-08-07):
   - Script: `scripts/analysis/compute_subclass_2dlatent.py` (ready to run; output `experiments/subclass_2dlatent_results.json`).
   - Motivation: embeddings lie on a ~2D intrinsic submanifold of S^63 (TwoNN 2.17, PR 1.38, PCA 99.9% in 2 comps); vMF's kappa divergence is a dimension-mismatch symptom (full-S^63 assumption vs actual 2D latent), not a vMF defect. Clustering should move to the real geometry.
   - Plan: per true class — PCA-2D of embeddings, then GMM (k=2..4, BIC/AIC), vMF-on-S^1 (bounded kappa via brentq, soft resp), SKM-2D, HDBSCAN (density-based arbiter for continuum-vs-bimodal), plus soft-membership entropy vs mass (option B); compare partitions vs spectral SC (ARI), silhouette on S^63 (cosine), and DGLAP fits (physics arbiter).
   - Current validated baseline stands: SC k=2 (DGLAP Table 2 exact); SKM ARI 0.937 QCD / 0.994 Top; theta_mean QCD-Core 12.6 deg / QCD-Edge 24.5 deg / Top 14.8 / 13.5 deg.
