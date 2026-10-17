# Reproducing the results (EFN–ArcFace top tagging)

This document maps **every quantitative claim** in `paper/main.tex` to the exact
command, artifact, and expected value needed to reproduce it.

## Environment

- Python 3.12 venv: `.\\.venv\\Scripts\\python.exe` (Windows; equivalent on Linux/macOS).
- Install: `pip install -r requirements-locked.txt` on AMD/Windows (verified) or `pip install -r requirements.txt` + CUDA wheel on NVIDIA/Linux/Mac (`pip install torch --index-url https://download.pytorch.org/whl/cu121`; see `arcefn/utils/device.py` `ARCEFN_DEVICE` override).
- All analysis scripts are run **from the repository root** as modules:
  ```
  .\\.venv\\Scripts\\python.exe -m scripts.analysis.<name> [args]
  ```
  (plain `python scripts/...` fails because the `arcefn` package needs the repo root on `sys.path`).
- Data: `data/top_tagging/test.h5` (404,000 test events) — download instructions in README.
- Unless stated otherwise: **seed 42**, **no replacement**, and **full 404k test set**.
- Checkpoints live in `experiments/*/checkpoint.pt` (not tracked in git; regenerate with
  `scripts/train_arcface.py`, `scripts/train_baselines.py`). All result JSONs ARE tracked.
- Hyperparameters (what was tuned vs fixed): $s=16.0$, $m=0.5$ from face literature (tested 0.3/0.5/0.7 on val, $m=0.5$ stable); $\Phi$ 128-d, $\rho$ 64-d fixed (Komiske et al.); embedding 64-d primary + 2-d ablation ($\Delta+0.05$~pp $p=0.125$)---no 4/8/16-d sweep (resource ceiling, $\sim$18h/train); Adam $10^{-3}$, OneCycleLR, 50 epochs, batch 128$\to$1024 accum, seed 42 (123/7 stability). No extensive swapping: keeps G4 falsifiable without conflating tuning; full config in each `config.json`.
- Two-sided view: primary = predicted partition (what model sees) 117786/75299/111029/99886 $\alpha=-1.242/-1.022/-0.493/+1.105$; GT check 123322/78592/101260/100826 $\alpha=-1.267/-0.859/-0.543/+1.106$ ARI 0.79 in `subclass_label_comparison.json` (Figs. predicted in `paper/figures/`, GT archived `HEAD~1:paper/figures/supplementary_true/`). See App.~\ref{app:gt-vs-pred}.
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
| Table 1 AUC / Rejection@50%TPR (0.974/0.974/0.975; 124/122/132) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/archive/final_analysis_results.json` | arcface AUC 0.9739, rej50 123.6; linear 0.9743/121.6; coslinear 0.9748/131.7 (the archived copy holds the paper numbers; the script rewrites the top-level path on re-run) |
| Table 1 RF baseline (92.37% / 0.977 / 150) | `compute_rf_baseline.py --max-train-events 403000 --max-events 404000 --n-estimators 500` | `experiments/rf_baseline_results.json` | acc 0.92368, AUC 0.97655, rej50 150.01 (train val.h5 -> test.h5, seed 42; features cached in `experiments/rf_cache_{val,test}.npz`) |
| McNemar p-values (4.3e-05 / 1.3e-08 / 0.16) | `compute_paired_stats.py` | `experiments/paired_model_stats.json` | pairwise discordants + continuity-corrected chi2 |
| ArcFace seed variability (91.84 / 91.88 / 91.83% for seeds 42/123/7) | `compute_arcface_seed_metrics.py` | `experiments/arcface_seed_metrics.json` | test acc + confusion matrix + best/final val_acc per seed (full 404k forward pass) |
| Robustness 4-momentum (JES) rescaling 91.84% all scales | `compute_robustness_tests.py` (N_EVENTS=404000) | `experiments/robustness_tests_results.json` | 0.9184 at 0.5/0.8/1.0/1.5/2.0x (full E,px,py,pz scaling) |
| Robustness azimuthal rotation 91.84% all angles | same | same | 0.9184 at 0/0.1/0.5/1.0/2.0 rad (phys. phi_i -> phi_i + alpha) |
| Robustness dropout 91.84/91.04/88.95/79.23/55.91% | same | same | at 0/10/20/50/80% |
| Table 4 TwoNN 2.17, PR 1.38, center angle 0.76 rad | `compute_dimensionality.py` (N_EVENTS=50000) | `experiments/dimensionality_results.json` | twonn 2.1749; PR 1.3776; angle 0.7611 rad |
| Table 4 PCA 2 comps > 99.9% (83.7/16.3) | `compute_interpretation.py` | `experiments/interpretation_full_results.json` | pca_variance [0.837078, 0.162916, ...] |
| Dim-ablation: 2D training parity (91.89% vs 91.84%, Δ=+0.05pp, p=0.125) | `compute_dim_ablation_eval.py` | `experiments/dim_ablation_results.json` | d=2 acc 0.918948 AUC 0.9741 rej50 124.79; d=64 acc 0.918403 AUC 0.9739 rej50 123.65; McNemar chi2=2.359 p=0.125; gate passed (|Δacc|<0.2pp, |ΔAUC|<0.003, |Δrej50|<30) |
| Table 2 DGLAP exponents (primary: predicted-label partition, Table 2; ground-truth gives -1.267/-0.859/-0.543/+1.106, 402055 fitted 99.5% 95% bootstrap CI 1k) | `compute_subclass_label_comparison.py` (pred_partition) + `compute_dglap_fit.py` (ground-truth `dglap_fit_params.json` CI) | `experiments/subclass_label_comparison.json` pred: QCD_Core -1.242/-1.252/0.998, QCD_Edge -1.022/-0.401/0.997, Top_Core -0.493/0.135/0.960, Top_Edge +1.105/6.602/0.988; ground-truth `dglap_fit_params.json` QCD_Core -1.267 [-1.313,-1.216]/-1.275/0.999; QCD_Edge -0.859 [-0.923,-0.803]/-0.264/0.995; Top_Core -0.543 [-0.593,-0.498]/0.118/0.967; Top_Edge 1.106 [1.100,1.112]/6.600/0.988; $n$ pred 117786/75299/111029/99886; ground $n$=121480/78501/101251/100826; fields `alpha_ci95`/`beta_ci95` on ground-truth fit |
| Subclass counts (primary: predicted-label partition 117,786/75,299/111,029/99,886; ground-truth canonical 123,322/78,592/101,260/100,826) | `compute_subclass_label_comparison.py` (pred) / `compute_interpretation.py` (ground-truth) | `experiments/subclass_label_comparison.json` pred + `experiments/subclass_composition.json` ground | n_qcd_core/n_qcd_edge/n_top_core/n_top_edge (predicted-label primary proves model-seen structure; ground-truth canonical in Table 2 note; ARI 0.79; KMeans old 126,514/75,400/100,421/101,665 replaced 2026-08-10; spectral rerun 123,320/78,594/101,263/100,823 within +/-2-3) |
| MI top-5 dims > 0.92 with Mass; effect sizes (d=1.73 QCD, 1.11 Top tau32) | `compute_interpretation.py` | same | top_mi_pairs; effect_sizes |
| Saliency (padding-masked, nan-fixed): 12/28/38 constituents for 50/80/90% | `compute_saliency.py` (N_EVENTS=5000) | `experiments/saliency_results.json` | n_for_50pct=12, n_for_80pct=28, n_for_90pct=38; the earlier 1/1/1 was a nan-propagation bug (fixed 2026-08-21 nan_to_num); the 12/28 before 2026-08-19 was unmasked padding. |
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
| Mass-only null comparison (fig:poc right: mass-only AUC 0.70 [140,170), 0.55 [150,200) vs model 0.89/0.88) | `compute_mass_window_null.py` | `experiments/mass_window_null.json` | mass_only_auc 0.6963 [140,170) / 0.5484 [150,200) (mass-only ranker + prevalence baseline) |
| pT smear 5\% per-constituent Gaussian (404k, seed 42) | `compute_pt_smear.py [--sigma 0.05]` | `experiments/pt_smear_results.json` | clean 0.91840 smeared 0.91833 $\Delta=-0.007$~pp (IRC-safe averaging) |
| IRC injection (200 jets): mean cos > 0.999 at all levels | `compute_irc_injection.py` (N_EVENTS=200) | `experiments/irc_injection_results.json` | soft e0.01 mean 0.99988 / min 0.99523; e0.001 mean 0.999999; collinear mean 0.9999999; masking 1.0 |
| Figures (ROC, rejection rates, NMI/ARI) | `compute_bootstrap_ci.py --max-events 404000` | `experiments/final_roc_curves.png`, `final_rejection_rates.png`, `final_nmi_ari.png` | copied into `paper/figures/`; NMI/ARI from jet embeddings (out[-1]), not logits |
| Lund planes (15k physics subsample), composition/physics (100k) | `generate_paper_figures.py` (subsample table `paper/main.tex:tab:subsample`) + `compute_subclass_lund_planes.py` | `paper/figures/lund_*.png` (15k), `subclass_composition*.png` (100k) | Lund: `compute_subclass_lund_planes.py` uses 15k subsample (`subclass_lund_planes.json`), composition 100k — see `paper/main.tex:398/430` and Table `tab:subsample` (UMAP removed in favor of 2D hypersphere + Δη-Δφ) |
| Subclass-resolved Lund planes (fig:lund-subclass) | `compute_subclass_lund_planes.py` | `experiments/subclass_lund_planes.json` + `experiments/robustness_s16_m05/lund_planes/subclass_*.png` + `paper/figures/lund_subclass_*.png` | 6 panels with counts qcd_core 123,322 / qcd_edge_tn 57,695 / qcd_edge_fp 20,897 / top_core_tp 89,192 / top_core_fn 12,068 / top_edge 100,826 + per-panel stats (mean ln k_T, ln(1/Δ), mass, tau32); canonical spectral partition, saved 404k embeddings, no forward pass |
| Angular collimation | `generate_angular_collimation.py` (404000) | `experiments/robustness_s16_m05/angular_collimation.png` | scaling exponents |
| Clustering robustness: k=2 robust (SC vs SKM vs vMF); SKM ARI 0.94 QCD / 0.99 Top | `compute_subclass_clustering.py` | `experiments/subclass_clustering_comparison.json` | QCD SC sil 0.838 vs SKM 0.844; Top all 0.736; vMF degenerate (kappa cap) |
| Subclass angular concentration (12.6/24.5 deg QCD Dense/Dilute; 14.8/13.5 deg Top) | `compute_subclass_concentration.py` | `experiments/subclass_concentration_results.json` | rbar + kappa_hat + theta_mean per subclass |
| Mass-matched DGLAP control (Top 178/174 GeV tau32 0.63/0.46; QCD 64/145 GeV, [95,133) GeV bin Core -1.89 vs Edge -1.38) | `compute_mass_matched_dglap.py` | `experiments/mass_matched_dglap.json` | per-subclass mean mass/tau32 + per-mass-bin alpha/beta/R2/delta_alpha |
| Subclass composition by prediction status (FP = 100% QCD-Dilute; FN = 100% Top-Dilute; Top-Dense 100% TP) | `compute_subclass_composition.py` | `experiments/subclass_composition.json` | per-subclass % in TP/FP/TN/FN (canonical partition, full 404k) |
| Table 5 symbolic distillation (c=6 89.44/0.950/33; c=21 90.07/0.964/74/0.969; Eq. score corr 0.969) | `fit_esr.py`/`compute_esr_validate.py` (see `scripts/analysis/fit_symbolic.py`) | `experiments/physics_sr/symbolic_eval.json`, `fit_full.json` | c=6 acc 0.8944 AUC 0.9496 rej50 32.85; c=21 acc 0.9007 AUC 0.9637 rej50 73.56 corr 0.9688; Eq. matches `fit_full.json` best_equation |
| Symbolic univariate AUCs (m/pT 0.906, mSD/pT 0.896, theta_g 0.876, tau32 0.147) | `compute_feature_probe.py` | `experiments/physics_sr/probe_features.json` | Mass_over_pT 0.9059; mSD_over_pT 0.8963; theta_g 0.8764; Tau32 0.1471 |
| alpha_SR rediscovery (-1.23/-0.96/-0.54/+0.66 vs -1.27/-0.86/-0.54/+1.11; top-Dense +0.66..+0.69 bracket) | `compute_symbolic_rediscovery.py` | `experiments/physics_sr/subclass_rediscovery.json` | alpha_sr/-fitzg/-lr per subclass; gate found_all_subclasses=true (Top-Dense delta vs fitzg 0.443 explained as beta-spread) |
| Seed stability of subclass partition (Top ARI cross-seed only 0.68, QCD 0.92) | `compute_seed_stability.py` | `experiments/seed_stability_analysis.json` | per-seed counts + alphas + ARI vs seed 42 (uncited in paper; relevant to model-selection argument) |
| Embedding axes correlation (PC1 vs mass pearson -0.887, r2 0.79) | `compute_embedding_axes.py` | `experiments/embedding_axes.json` | pca_variance_ratio + per-feature pearson/r2 (uncited; restored on dev per user decision) |
| tau32-ablation (paper: 'disabling tau32 reduces to tanh(m/pT) alone: 88.2% acc, AUC 0.906; +1.2 pp gain') | `compute_symbolic_ablation.py` | `experiments/physics_sr/symbolic_ablation.json` | ablated (Tau32 input zeroed in c=6, the only place it enters) acc 0.8824, AUC 0.9059, rej50 11.31; zeroing tau32 reproduces c=2 exactly (max_abs_diff 0.0). **RESOLVED 2026-08-18**: the paper's earlier 86.6%/+2.8 pp was never committed; corrected to 88.2%/+1.2 pp in paper line 569 |
| Raw-embedding clustering k=2 (unbiased, full 404k, 64-d) — continuous-arc check | `compute_raw_embedding_clustering.py [--model_dir experiments/robustness_s16_m05]` | `experiments/raw_embedding_clustering.json` | k_raw=2, ARI vs class 0.583, vs true 4-way 0.397 — confirms no label-free class separation; subclass structure is per-class (see Sec. 5a) |
| Raw-embedding clustering k=2 (unbiased, full 404k d=2) — continuous-arc check | `compute_raw_embedding_clustering.py [--model_dir experiments/ablation_dim_2]` | `experiments/ablation_dim_2/raw_embedding_clustering.json` | k_raw=2, ARI vs class labels 0.575, vs true 4-way 0.390 — confirms no label-free class separation; subclass structure is per-class (see Sec. 5a) |
| Δη-Δφ planes (15k physics subsample, avg + medoid rep per class/subclass) | `compute_deta_dphi_planes.py [--model_dir experiments/robustness_s16_m05]` | `experiments/robustness_s16_m05/deta_dphi/*.png` + `experiments/deta_dphi_planes.json` + `paper/figures/deta_dphi_*_avg.png` (canonical) + `_rep.png` supplement | 6 avg panels (2 class +4 subclass) +6 medoid reps; canonical counts e.g. QCD-Dense 123322 plotted 3199 rep 229139; d=2 variant in `experiments/ablation_dim_2/deta_dphi/` |
| Seed stability d=2 (seeds 7/123 vs 42, 404k, d=2) | `compute_seed_stability.py --model_dir experiments/ablation_dim_2` | `experiments/ablation_dim_2/seed_stability_analysis.json` | per-seed subclass counts + DGLAP alphas stable (QCD_Core -1.27..-1.29, Top_ARI 0.68, QCD 0.92); mirrors 64-d pattern (row 70) |
| Subclass label comparison B-primary (true- vs pred-partition, d=2) | `compute_subclass_label_comparison.py [--model_dir experiments/ablation_dim_2]` | `experiments/ablation_dim_2/subclass_label_comparison.json` | pred_partition (B-primary) QCD_Core 116510 α-1.241/β-1.268, QCD_Edge 76423 -1.011/-0.336, Top_Core 111652 -0.501/0.124, Top_Edge 99415 +1.107/6.590; comparison ARI global 0.796, QCD 0.888, Top 0.960; overlap matrix 4×4 |

## Statistical claims (Section "Statistical Considerations")

- 95% binomial CI at n=404,000, p=0.9184: `p +- 1.96*sqrt(p(1-p)/n)` = +-0.000844 -> [91.76, 91.92] (analytic; matches bootstrap CI [91.75, 91.92]).
- Subsampled analyses use fixed, deterministic subsets of the test set (the first N events, or a seeded random draw with seed 42 where noted), shared across models -> strictly paired (McNemar valid).
- Power: unpaired two-proportion test on 404k resolves ~0.17 pp at 80% power; McNemar resolves differences below that; per-group n for Cohen d = 15.7/d^2 (393 for d=0.2; ~5 for d=1.75); all subclass groups >= 75k.

## Superseded / NOT to be used

The following artifacts predate the final analysis and contain outdated or
incorrect numbers. They are tracked for completeness only and must not be
cited. Superseded/exploratory result files have been **moved to
`experiments/archive/`** (2026-08-18 cleanup); a handful are still listed at the
top level because the paper or this document references them:

- `experiments/archive/comprehensive_analysis_results.json` — orphaned producer script (never in repo); l4 dropout/rotation numbers WRONG
- `experiments/archive/full_analysis_results.json` — old pre-audit numbers
- `experiments/archive/ap_splitting_results.json` — superseded by `dglap_fit_params.json` (bounds change)
- `experiments/archive/robustness_results.json`, `robustness_results_table.md` — old s x m sweep, different n and method
- `experiments/archive/bayesian_results*.json`, `decision_tree_results.json`, `dt_rules_results.json`, `grooming_sweep_results.json`, `onnx_results.json`, `emd_validation_results.json`, `interpretation_results.json` — exploratory, not in paper
- `experiments/archive/alignment_uniformity_results.json` — superseded by `embedding_geometry_results.json`
- `experiments/archive/pgd_attack_results.json` — superseded by `adversarial_results.json` (n=20000)
- `experiments/archive/arcface_100ep/metrics.json` — final-epoch checkpoint metrics differ from best-val checkpoint used for all paper numbers
- `experiments/archive/baselines_coslinear_seed_*_metrics.json` — final-epoch roc_auc anomaly; paper uses best-val checkpoints via `compute_bootstrap_ci.py`
- `experiments/archive/jetnet_poc/*`, `lorentznet_subclasses*.json` — POCs, not in the paper

## Generative-AI disclosure

See the "Use of Generative AI Tools" section at the end of `paper/main.tex`.
When submitting to SciPost, also fill in the "Disclosure of Generative AI use"
field of the submission form with the same statement.

## Git hygiene

- `experiments/*.json` (small result files) ARE tracked; checkpoints, `.npz`, `.png`, `.onnx` are ignored (regenerable).
- Repo: https://github.com/thirathep-ph/EFN-ArcFace-Top-Quark-Tagging-and-Jet-Substructure-via-Hyperspherical-Prototypes (branch `main`).

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
   - Current validated baseline stands: SC k=2 (DGLAP Table 2 exact); SKM ARI 0.937 QCD / 0.994 Top; theta_mean QCD-Dense 12.6 deg / QCD-Dilute 24.5 deg / Top 14.8 / 13.5 deg.
