# Output Folder Structure — No Mixing Allowed

> Every model/config combination gets its OWN folder. No shared directories.

---

## Smoke Test Results (for verification)

```
results/smoke_test/
  baselines/
    linear/
      seed_42/
        metrics.json
        history.json
        best_model.pt
    coslinear/
      seed_42/
        metrics.json
        history.json
        best_model.pt
  arcface/
    s16_m05/          ← default ArcFace
      advanced_metrics.json
      embeddings.npz
      history.json
      performance.png
      umap_*.png
      lund_planes/*.png
    s8_m0/
      advanced_metrics.json
      ...
    s32_m05/
      advanced_metrics.json
      ...
```

## Full Run Results

```
results/
  baselines/
    linear/
      seed_42/
        metrics.json       ← accuracy, precision, recall, f1, auc
        history.json       ← loss & acc per epoch
        best_model.pt      ← checkpoint
      seed_123/
        ...
      seed_7/
        ...
    coslinear/
      seed_42/
        ...
      seed_123/
        ...
      seed_7/
        ...

  robustness/
    s8_m00/               ← scale=8,  margin=0.0
      advanced_metrics.json
      embeddings.npz
      history.json
      performance.png
      decision_boundary.png
      training_results_efn.png
      confusion_matrix.png
      correlations.png
      mass_sculpting.png
      feature_distributions.png
      latent_space.png
      umap_prediction_status.png
      umap_subclasses.png
      subclass_physics_distributions.png
      subclass_composition_heatmap.png
      subclass_similarity_distributions.png
      subclass_physics_correlations.png
      top_subclass_*_representative_jets.png
      qcd_subclass_*_representative_jets.png
      top_subclass_*_centroid.png
      qcd_subclass_*_centroid.png
      lund_planes/
        class_overall_non_split.png
        status_tp.png
        status_tn.png
        status_fp.png
        status_fn.png
        top_subclass_*.png
        top_subclass_*_tp.png
        top_subclass_*_fp.png
        qcd_subclass_*.png
        qcd_subclass_*_tn.png
        qcd_subclass_*_fn.png
        lund_plane_qcd.png
        lund_plane_top.png
    s8_m03/
      ...
    s8_m05/
      ...
    s16_m00/
      ...
    s16_m03/
      ...
    s16_m05/
      ...
    s32_m00/
      ...
    s32_m03/
      ...
    s32_m05/
      ...

  svd/
    svd_scree_plot.png
    svd_weight_scree.png

  bootstrap/
    bootstrap_results.json

checkpoints/
  baselines/
    linear/
      seed_42/best_model.pt
      seed_123/best_model.pt
      seed_7/best_model.pt
    coslinear/
      seed_42/best_model.pt
      seed_123/best_model.pt
      seed_7/best_model.pt
  robustness/
    s8_m00/best_model_efn.pt
    s8_m03/best_model_efn.pt
    ...
    s32_m05/best_model_efn.pt
```

## Key Rules

1. **No shared folders** — every model/config gets its own `--output_dir` and `--checkpoint_dir`
2. **Baselines**: `results/baselines/{model_name}/seed_{n}/`
3. **Robustness**: `results/robustness/s{s}_m{m}/` (e.g., `s16_m05`)
4. **SVD**: `results/svd/` (separate from model results)
5. **Bootstrap**: `results/bootstrap/` (separate)
6. **Smoke test**: `results/smoke_test/` (mirrors the same structure but with tiny data)
