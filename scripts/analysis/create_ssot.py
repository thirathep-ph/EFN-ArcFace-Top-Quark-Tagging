"""Regenerate experiments/ssot_paper_numbers.json: derive every expect from
its artifact pointer instead of hand-typing it.

Every entry is {name, artifact, pointer, scale, tol, paper_text}. The value
written into `expect` is READ FROM the artifact, so a wrong `expect` cannot
survive a regeneration. `paper_text` records what the manuscript currently
prints, so verify_ssot.py can report text-vs-artifact disagreement separately.
"""
import json

ENTRIES = [
    # --- headline classifier performance -----------------------------------
    ("arcface_acc_s16_pct", "experiments/arcface_seed_metrics.json",
     "42.accuracy", 100, 0.005, r"91.84\%"),
    ("arcface_acc_s123_pct", "experiments/arcface_seed_metrics.json",
     "123.accuracy", 100, 0.005, None),
    ("arcface_acc_s7_pct", "experiments/arcface_seed_metrics.json",
     "7.accuracy", 100, 0.005, None),
    ("cm_TN", "experiments/arcface_seed_metrics.json",
     "42.confusion_matrix.0.0", 1, 0, "181017"),
    ("cm_FP", "experiments/arcface_seed_metrics.json",
     "42.confusion_matrix.0.1", 1, 0, "20897"),
    ("cm_FN", "experiments/arcface_seed_metrics.json",
     "42.confusion_matrix.1.0", 1, 0, "12068"),
    ("cm_TP", "experiments/arcface_seed_metrics.json",
     "42.confusion_matrix.1.1", 1, 0, "190018"),
    # CANONICAL: experiments/final_analysis_results.json @ git HEAD = 404k.
    # An uncommitted 100k rerun (2026-11-21) overwrote it once and reported
    # rej50 115.17 vs 123.65 on the full set; restored from HEAD. The archive
    # copy is the same 404k run, differing only in the bootstrap RNG draw.
    ("bootstrap_ci_lower_pct", "experiments/final_analysis_results.json",
     "bootstrap_ci.arcface_s16_m05.ci_lower", 100, 0.01, r"91.76\%"),
    ("bootstrap_ci_upper_pct", "experiments/final_analysis_results.json",
     "bootstrap_ci.arcface_s16_m05.ci_upper", 100, 0.01, r"91.92\%"),
    # Rejection and AUC recomputed on the full 404k set from logits.npz,
    # which reproduces 0.91840 / 0.97387 / 123.65 of that same run.
    ("rejection_tpr50", "experiments/physics_sr/logits.npz",
     "logitdiff:rej50", 1, 0.5, "124"),
    ("auc_full_model", "experiments/physics_sr/logits.npz",
     "logitdiff:auc", 1, 0.0005, "0.974"),
    # The bracket the paper prints in Table 1 / App. A is the binomial
    # interval, not the bootstrap one. Computed from n and accuracy.
    ("binom_ci_lower_pct", "DERIVED", "binom_lower", 100, 0.005,
     r"91.76\%"),
    ("binom_ci_upper_pct", "DERIVED", "binom_upper", 100, 0.005,
     r"91.92\%"),
    ("binom_halfwidth_pp", "DERIVED", "binom_halfwidth", 1, 0.005, "0.08"),
    # --- baselines / seeds --------------------------------------------------
    ("coslinear_3seed_mean_pct", "experiments/seed_accs.json",
     "mean:baselines_coslinear_seed_42.accuracy,"
     "baselines_coslinear_seed_123.accuracy,"
     "baselines_coslinear_seed_7.accuracy", 100, 0.005, "92.00"),
    ("coslinear_seed_42_pct", "experiments/seed_accs.json",
     "baselines_coslinear_seed_42.accuracy", 100, 0.005, None),
    ("s22_3seed_mean_pct", "experiments/seed_accs.json",
     "mean:robustness_s22_m05.accuracy,robustness_s22_seed_123.accuracy,"
     "robustness_s22_seed_7.accuracy", 100, 0.005, None),
    # --- embedding geometry -------------------------------------------------
    ("pca_pc1_pct", "experiments/dim_3way_50k.json",
     "ArcFace.pca_top6.0", 100, 0.05, "83.6"),
    ("pca_pc2_pct", "experiments/dim_3way_50k.json",
     "ArcFace.pca_top6.1", 100, 0.05, "16.4"),
    ("twonn_arcface", "experiments/dim_3way_50k.json", "ArcFace.twonn",
     1, 0.005, "2.21"),
    ("pr_arcface", "experiments/dim_3way_50k.json", "ArcFace.pr", 1, 0.005,
     "1.38"),
    ("twonn_coslinear", "experiments/dim_3way_50k.json",
     "CosLinear.twonn", 1, 0.005, "4.98"),
    ("pr_coslinear", "experiments/dim_3way_50k.json", "CosLinear.pr", 1,
     0.005, "5.63"),
    ("twonn_linear", "experiments/dim_3way_50k.json", "Linear.twonn", 1,
     0.005, "5.19"),
    ("pca_cum2_coslinear_pct", "experiments/dim_3way_50k.json",
     "CosLinear.pca_cum2", 100, 0.05, "48.8"),
    ("pca_cum2_linear_pct", "experiments/dim_3way_50k.json",
     "Linear.pca_cum2", 100, 0.05, "89.9"),
    # --- DGLAP by T/F status ------------------------------------------------
    ("dglap_tf_TP_alpha", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "TP.dglap.alpha", 1, 0.001, "-0.0000"),
    ("dglap_tf_TN_alpha", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "TN.dglap.alpha", 1, 0.001, "-1.1936"),
    ("dglap_tf_FP_alpha", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "FP.dglap.alpha", 1, 0.001, "-0.2477"),
    ("dglap_tf_FN_alpha", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "FN.dglap.alpha", 1, 0.001, "-0.5974"),
    ("dglap_tf_TP_r2", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "TP.dglap.r2", 1, 0.001, "0.9638"),
    ("dglap_tf_TN_r2", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "TN.dglap.r2", 1, 0.001, "0.9993"),
    ("dglap_tf_FP_r2", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "FP.dglap.r2", 1, 0.001, "0.8966"),
    ("dglap_tf_FN_r2", "experiments/robustness_s16_m05/tpfp_dglap.json",
     "FN.dglap.r2", 1, 0.001, "0.9714"),
    # --- clustering / order -------------------------------------------------
    ("hopkins_H", "experiments/robustness_s16_m05/hopkins_silhouette.json",
     "hopkins_statistic", 1, 0.0001, "1.0000"),
    ("silhouette_score",
     "experiments/robustness_s16_m05/hopkins_silhouette.json",
     "silhouette_score", 1, 0.001, "0.6452"),
    # --- prototype-distance evolution (10 bins, per true class) -------------
    ("proto_evol_top_n",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.n_total", 1, 0, "202086"),
    ("proto_evol_top_acc",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.accuracy", 1, 0.001, "0.9403"),
    ("proto_evol_top_dist_mean",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.dist_own_mean", 1, 0.001, "2.0967"),
    ("proto_evol_top_tau32_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.spearman_across_bins.tau32", 1, 0.001, "+1.00"),
    ("proto_evol_top_mass_first",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.bins.0.mass", 1, 0.05, "171.6"),
    ("proto_evol_top_mass_last",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.bins.9.mass", 1, 0.05, "164.4"),
    ("proto_evol_top_correct_first",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.bins.0.frac_correct", 1, 0.001, "1.000"),
    ("proto_evol_top_correct_last",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.bins.9.frac_correct", 1, 0.001, "0.417"),
    ("proto_evol_qcd_n",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.n_total", 1, 0, "201914"),
    ("proto_evol_qcd_acc",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.accuracy", 1, 0.001, "0.8965"),
    ("proto_evol_qcd_dist_mean",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.dist_own_mean", 1, 0.001, "1.6378"),
    ("proto_evol_qcd_mass_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.spearman_across_bins.mass", 1, 0.001, "+0.89"),
    ("proto_evol_qcd_thetag_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.spearman_across_bins.theta_g", 1, 0.001, "+0.83"),
    ("proto_evol_qcd_nconst_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.spearman_across_bins.n_const", 1, 0.01, "+0.96"),
    ("proto_evol_qcd_mass_first",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.bins.0.mass", 1, 0.05, "57.0"),
    ("proto_evol_qcd_mass_last",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.bins.9.mass", 1, 0.05, "178.4"),
    ("proto_evol_qcd_correct_first",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.bins.0.frac_correct", 1, 0.001, "1.000"),
    ("proto_evol_qcd_correct_last",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.bins.9.frac_correct", 1, 0.001, "0.015"),
    ("proto_evol_top_mass_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.spearman_across_bins.mass", 1, 0.001, None),
    ("proto_evol_top_nconst_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.spearman_across_bins.n_const", 1, 0.001, None),
    ("proto_evol_top_dist_std",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.top.dist_own_std", 1, 0.001, None),
    ("proto_evol_qcd_tau32_rho",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.spearman_across_bins.tau32", 1, 0.001, None),
    ("proto_evol_qcd_dist_std",
     "experiments/robustness_s16_m05/prototype_distance_evolution_10bin.json",
     "classes.qcd.dist_own_std", 1, 0.001, None),
    # --- symbolic regression -------------------------------------------------
    ("norm_raw_mean_auc", "experiments/physics_sr/norm_summary.json",
     "mean:cells.norm_raw_s42.auc,cells.norm_raw_s123.auc,"
     "cells.norm_raw_s7.auc", 1, 0.0005, None),
    ("norm_z_mean_auc", "experiments/physics_sr/norm_summary.json",
     "mean:cells.norm_z_s42.auc,cells.norm_z_s123.auc,cells.norm_z_s7.auc",
     1, 0.0005, None),
    ("norm_mm_mean_auc", "experiments/physics_sr/norm_summary.json",
     "mean:cells.norm_mm_s42.auc,cells.norm_mm_s123.auc,"
     "cells.norm_mm_s7.auc", 1, 0.0005, None),
    ("norm_robust_mean_auc", "experiments/physics_sr/norm_summary.json",
     "mean:cells.norm_robust_s42.auc,cells.norm_robust_s123.auc,"
     "cells.norm_robust_s7.auc", 1, 0.0005, None),
    ("norm_sigma_mean_auc", "experiments/physics_sr/norm_summary.json",
     "mean:cells.norm_sigma_s42.auc,cells.norm_sigma_s123.auc,"
     "cells.norm_sigma_s7.auc", 1, 0.0005, None),
    # --- tab:symbolic, one row per complexity ------------------------------
    # Source is symbolic_eval.json, which re-scores every equation of the
    # 404k fit (itself archived in the old repo as fit_full.json).
    ("sym_c2_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.1.metrics.acc", 100, 0.005, "88.24"),
    ("sym_c2_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.1.metrics.auc", 1, 0.0005, "0.906"),
    ("sym_c2_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.1.metrics.rej50", 1, 0.5, "11"),
    ("sym_c5_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.3.metrics.acc", 100, 0.005, "87.98"),
    ("sym_c5_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.3.metrics.auc", 1, 0.0005, "0.943"),
    ("sym_c5_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.3.metrics.rej50", 1, 0.5, "29"),
    ("sym_c6_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.4.metrics.acc", 100, 0.005, "89.44"),
    ("sym_c6_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.4.metrics.auc", 1, 0.0005, "0.950"),
    ("sym_c6_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.4.metrics.rej50", 1, 0.5, "33"),
    ("sym_c9_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.7.metrics.acc", 100, 0.005, "89.64"),
    ("sym_c9_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.7.metrics.auc", 1, 0.0005, "0.958"),
    ("sym_c9_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.7.metrics.rej50", 1, 0.5, "53"),
    ("sym_c11_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.9.metrics.acc", 100, 0.005, "89.44"),
    ("sym_c11_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.9.metrics.auc", 1, 0.0005, "0.960"),
    ("sym_c11_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.9.metrics.rej50", 1, 0.5, "69"),
    ("sym_c16_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.13.metrics.acc", 100, 0.005, "90.02"),
    ("sym_c16_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.13.metrics.auc", 1, 0.0005, "0.963"),
    ("sym_c16_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.13.metrics.rej50", 1, 0.5, "73"),
    ("sym_c19_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.16.metrics.acc", 100, 0.005, "89.71"),
    ("sym_c19_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.16.metrics.auc", 1, 0.0005, "0.963"),
    ("sym_c19_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.16.metrics.rej50", 1, 0.5, "77"),
    ("sym_c21_acc_pct", "experiments/physics_sr/symbolic_eval.json",
     "equations.17.metrics.acc", 100, 0.005, "90.07"),
    ("sym_c21_auc", "experiments/physics_sr/symbolic_eval.json",
     "equations.17.metrics.auc", 1, 0.0005, "0.964"),
    ("sym_c21_rej50", "experiments/physics_sr/symbolic_eval.json",
     "equations.17.metrics.rej50", 1, 0.5, "74"),
    # corr(equation, neural logit) quoted in the distillation paragraph.
    ("sym_c21_corr_nn", "experiments/physics_sr/symbolic_eval.json",
     "equations.17.corr_with_nn_logit", 1, 0.001, "0.969"),

    # --- compact-form weighting ratio (Eq. eq:symbolic-compact) -------------
    ("k_nominal", "experiments/physics_sr/compact_form_ratio.json",
     "nominal_ratio", 1, 0.0005, "0.33"),
    ("k_bootstrap_std", "experiments/physics_sr/compact_form_ratio.json",
     "bootstrap_ratio_std", 1, 0.0005, "0.0011"),
    ("k_ci_lo", "experiments/physics_sr/compact_form_ratio.json",
     "bootstrap_ratio_ci95.0", 1, 0.0005, "0.328"),
    ("k_ci_hi", "experiments/physics_sr/compact_form_ratio.json",
     "bootstrap_ratio_ci95.1", 1, 0.0005, "0.332"),
    ("k_offset_zero", "experiments/physics_sr/compact_form_ratio.json",
     "sensitivity.offset_forced_zero", 1, 0.0005, "0.366"),
    ("k_linear_mass", "experiments/physics_sr/compact_form_ratio.json",
     "C_linear_mass.ratio_k_over_a", 1, 0.0005, "0.356"),
    ("k_trim_saturated", "experiments/physics_sr/compact_form_ratio.json",
     "sensitivity.trim_abs_logit_le_11", 1, 0.0005, "0.467"),
    ("k_free_step_profile", "experiments/physics_sr/compact_form_ratio.json",
     "sensitivity.free_step_profile", 1, 0.005, "2.09"),
    ("k_auc_nominal_form", "experiments/physics_sr/compact_form_ratio.json",
     "auc_of_nominal_form", 1, 0.0005, "0.943"),

    # --- tab:clf baselines (rej50 computed from paired_scores_cache.npz) ----
    ("linear_acc_3seed_mean_pct", "DERIVED", "linear_3seed_mean", 1, 0.005,
     "91.94"),
    ("linear_rej50", "experiments/paired_scores_cache.npz",
     "scores_linear_seed42:rej50", 1, 0.5, "122"),
    ("linear_auc", "experiments/paired_scores_cache.npz",
     "scores_linear_seed42:auc", 1, 0.0005, "0.974"),
    ("coslinear_rej50", "experiments/paired_scores_cache.npz",
     "scores_coslinear_seed42:rej50", 1, 0.5, "132"),
    ("coslinear_auc", "experiments/paired_scores_cache.npz",
     "scores_coslinear_seed42:auc", 1, 0.0005, "0.975"),
    # --- physics paragraphs (50k mass-decoupled subset) ---------------------
    ("qcd_frac_at_mass_gt160_pct", "DERIVED", "qcd_frac_mass160", 100, 0.05,
     "7.6"),
    ("auc_mass_window_140_170", "experiments/mass_window_null.json",
     "windows.0.model_auc", 1, 0.005, "0.89"),
    ("auc_massonly_window_140_170", "experiments/mass_window_null.json",
     "windows.0.mass_only_auc", 1, 0.005, "0.70"),
    ("auc_mass_window_150_200", "experiments/mass_window_null.json",
     "windows.1.model_auc", 1, 0.005, "0.88"),
    ("auc_massonly_window_150_200", "experiments/mass_window_null.json",
     "windows.1.mass_only_auc", 1, 0.005, "0.55"),
    ("acc_mass_window_150_200_pct", "experiments/mass_window_null.json",
     "windows.1.model_acc", 100, 0.05, "90.2"),
    ("majority_acc_window_150_200_pct", "experiments/mass_window_null.json",
     "windows.1.majority_acc", 100, 0.05, "88.8"),
    # --- injection scan (prototypes are fixed points) ----------------------
    ("inject_margin_top", "experiments/poc_injection_scan_results.json",
     "inject:margin_top", 1, 0.005, "0.416"),
    ("inject_margin_qcd", "experiments/poc_injection_scan_results.json",
     "inject:margin_qcd", 1, 0.005, "-0.516"),
    ("inject_calib_err", "experiments/poc_injection_scan_results.json",
     "inject:calib", 1, 0.005, "0.046"),
    # --- embedding geometry extras -----------------------------------------
    ("emb_norm_median", "experiments/embedding_geometry_results.json",
     "ArcFace (s=16, m=0.5)_norms_median", 1, 0.005, "7.73"),
    ("emb_norm_std", "experiments/embedding_geometry_results.json",
     "ArcFace (s=16, m=0.5)_norms_std", 1, 0.005, "4.94"),
    ("tau32_alone_auc_raw", "DERIVED", "tau32_auc", 1, 0.001, "0.147"),
    ("tau32_alone_auc_neg", "DERIVED", "tau32_auc_neg", 1, 0.001, "0.853"),
    ("proto_angle_deg", "experiments/dimensionality_results.json",
     "class_center_angular_distance_deg", 1, 0.05, "43.6"),
    ("conformal_coverage_pct", "experiments/archive/conformal_results.json",
     "arcface_s16_m05.empirical_coverage", 100, 0.05, "90.1"),
    ("eda_mean_mass_train", "experiments/eda/eda_stats.json",
     "dataset_stats.0.mean_mass", 1, 0.05, "135.7"),
    ("eda_mean_mass_val", "experiments/eda/eda_stats.json",
     "dataset_stats.1.mean_mass", 1, 0.05, "135.8"),
    ("eda_mean_ncons_train", "experiments/eda/eda_stats.json",
     "dataset_stats.0.mean_constituents", 1, 0.05, "49.0"),
]


def _derived(pointer):
    """Quantities computed from raw caches, not stored anywhere as a number."""
    import numpy as np

    if pointer in ("binom_lower", "binom_upper", "binom_halfwidth"):
        acc, n = 0.9184034653465346, 404000
        hw = 1.959964 * (acc * (1 - acc) / n) ** 0.5
        return {"binom_lower": acc - hw, "binom_upper": acc + hw,
                "binom_halfwidth": hw * 100}[pointer]

    from sklearn.metrics import roc_auc_score
    import pyarrow.parquet as pq

    lg = np.load("experiments/physics_sr/logits.npz")
    y = lg["labels"].astype(int)
    s = lg["logitdiff"].astype(float)
    mass = np.load("experiments/robustness_s16_m05/jet_pt_mpt_cache.npz")["mass"]

    if pointer == "qcd_frac_mass160":
        m = mass > 160
        return float(np.mean(s[m] < 0))

    for lo, up, tag in ((140, 170, "140_170"), (150, 200, "150_200")):
        if tag in pointer:
            m = (mass >= lo) & (mass < up)
            if pointer.startswith("auc_massonly"):
                return float(roc_auc_score(y[m], mass[m]))
            return float(roc_auc_score(y[m], s[m]))

    if pointer in ("emb_median", "emb_std"):
        e = np.load("experiments/robustness_s16_m05/embeddings.npz")
        norms = np.linalg.norm(e["embeddings"], axis=1)
        return float(np.median(norms) if pointer == "emb_median"
                     else norms.std())

    if pointer == "linear_3seed_mean":
        import json as _json
        vals = [_json.load(open(f"experiments/baselines_linear_seed_{s}/"
                                "metrics.json", encoding="utf-8"))["accuracy"]
                for s in (42, 7, 123)]
        return float(np.mean(vals) * 100)

    if pointer in ("tau32_auc", "tau32_auc_neg"):
        tau = np.load("experiments/robustness_s16_m05/"
                      "jet_pt_mpt_cache.npz")["tau32"]
        return float(roc_auc_score(y, -tau if pointer == "tau32_auc_neg"
                                   else tau))

    raise KeyError(pointer)


def resolve(art, pointer, root):
    import os
    if art == "DERIVED":
        return _derived(pointer)
    if art.endswith(".npz"):
        import numpy as np
        key, _, metric = pointer.partition(":")
        zf = np.load(os.path.join(root, art))
        y = zf[key].astype(float)
        if metric == "acc_eq_half":
            lab = zf["labels"].astype(int)
            return float(((zf["P_top"].reshape(-1) >= 0.5).astype(int)
                          == lab).mean())
        lab = zf["labels"].astype(int)
        if metric == "auc":
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(lab, y))
        if metric == "rej50":
            cut = np.quantile(y[lab == 1], 0.5)
            return float(1.0 / (y[lab == 0] >= cut).mean())
        raise KeyError(metric)
    obj = json.load(open(os.path.join(root, art), encoding="utf-8"))
    # A JSON key can itself contain dots/parens; try it whole before splitting.
    if isinstance(obj, dict) and pointer in obj:
        return obj[pointer]
    if pointer.startswith("inject:"):
        rows = [r for r in obj if 0 < r["true_signal_frac"] < 1]
        which = pointer.split(":")[1]

        def mean_sd(vals):
            m = sum(vals) / len(vals)
            return m, (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

        if which == "margin_top":
            return mean_sd([r["margin_mean_top"] for r in rows])[0]
        if which == "margin_qcd":
            return mean_sd([r["margin_mean_qcd"] for r in rows])[0]
        if which == "calib":
            return sum(abs(r["true_signal_frac"] - r["pred_signal_frac"])
                       for r in obj) / len(obj)
        raise KeyError(which)
    head, _, rest = pointer.partition(":")
    if rest:
        vals = []
        for part in rest.split(","):
            cur = obj
            for tok in part.split("."):
                cur = cur[tok]
            vals.append(float(cur))
        return {"mean": sum(vals) / len(vals), "max": max(vals),
                "min": min(vals)}[head]
    cur = obj
    for tok in pointer.split("."):
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def main():
    ROOT = "."
    out = []
    for name, art, ptr, scale, tol, text in ENTRIES:
        val = float(resolve(art, ptr, ROOT)) * scale
        out.append({"name": name, "artifact": art, "pointer": ptr,
                    "scale": scale, "tol": tol,
                    "expect": round(val, 6), "paper_text": text})

    with open("experiments/ssot_paper_numbers.json", "w", encoding="utf-8") as f:
        json.dump({"entries": out}, f, indent=2)

    print(f"wrote {len(out)} entries")
    for e in out:
        if e["paper_text"]:
            try:
                pv = float(e["paper_text"].replace("\\%", ""))
            except ValueError:
                continue
            if abs(pv - e["expect"]) > e["tol"]:
                print(f'  PAPER MISMATCH {e["name"]}: text={e["paper_text"]} '
                      f'artifact={e["expect"]}')


if __name__ == "__main__":
    main()
