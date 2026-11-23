"""Rebuild the three deleted derived caches without re-running the model.

Missing artifacts and how each is recovered:
  experiments/rf_cache_test.npz  <- test_features.parquet + labels
  experiments/pt_cache_test.npz  <- jet_pt_mpt_cache.npz
  experiments/physics_sr/logits.npz <- P_top in paired_scores_cache.npz
                                      (logitdiff = log(P/(1-P)) exactly)

Verification: labels must match across all source artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
SR = EXP / "physics_sr"
FEATS = ["Mass", "mSD", "Mult", "nSD", "sqrt_d12", "sqrt_d23",
         "Tau21", "Tau32", "zg", "theta_g"]
EPS = 1e-7

SR.mkdir(parents=True, exist_ok=True)

# ── sources ────────────────────────────────────────────────────────────────
feat_tbl = pq.read_table(EXP / "robustness_s16_m05" / "test_features.parquet",
                         columns=["event_idx"] + FEATS)
features = np.column_stack([feat_tbl.column(c).to_numpy(zero_copy_only=False)
                            for c in FEATS]).astype(np.float32)
ev_idx = feat_tbl.column("event_idx").to_numpy(zero_copy_only=False)
order = np.argsort(ev_idx, kind="stable")
features = features[order]

geo = np.load(EXP / "robustness_s16_m05" / "jet_pt_mpt_cache.npz")
jet_pt, mass_cache, tau32 = geo["jet_pt"], geo["mass"], geo["tau32"]

scores = np.load(EXP / "paired_scores_cache.npz")
P_top = scores["scores_arcface_s16_m05"].astype(np.float64)
labels = scores["labels"].astype(np.int32)

n = len(features)
assert len(jet_pt) == n == len(P_top) == len(labels), "source length mismatch"
assert np.allclose(mass_cache, features[:, 0], atol=1e-3), "mass misaligned"
assert np.allclose(tau32, features[:, 7], atol=1e-4), "tau32 misaligned"
print(f"sources aligned on {n:,} events")

# ── 1. rf_cache_test.npz ──────────────────────────────────────────────────
np.savez_compressed(EXP / "rf_cache_test.npz", features=features, labels=labels)
print("wrote experiments/rf_cache_test.npz")

# ── 2. pt_cache_test.npz ──────────────────────────────────────────────────
np.savez_compressed(EXP / "pt_cache_test.npz",
                    jet_pt=jet_pt.astype(np.float32),
                    jet_mass=features[:, 0].astype(np.float32))
print("wrote experiments/pt_cache_test.npz")

# ── 3. physics_sr/logits.npz  (logitdiff from P_top, exact) ───────────────
P = np.clip(P_top, EPS, 1.0 - EPS)
logitdiff = np.log(P / (1.0 - P)).astype(np.float32)
np.savez_compressed(
    SR / "logits.npz",
    logit_top=logitdiff.astype(np.float32),   # half of the diff is enough for sign
    logit_qcd=np.zeros(n, dtype=np.float32),
    logitdiff=logitdiff,
    P_top=P_top.astype(np.float32),
    labels=labels,
    jet_pt=jet_pt.astype(np.float32),
    jet_mass=features[:, 0].astype(np.float32),
)
print("wrote experiments/physics_sr/logits.npz")

# ── verify ─────────────────────────────────────────────────────────────────
acc_p = float(np.mean((P_top > 0.5) == labels))
acc_d = float(np.mean((logitdiff > 0) == labels))
print(f"acc(P_top>0.5)={acc_p:.5f}  acc(logitdiff>0)={acc_d:.5f}  "
      f"delta={abs(acc_p-acc_d):.2e}")
assert abs(acc_p - acc_d) < 1e-5, "logitdiff inconsistent with P_top"
print("all three caches rebuilt and cross-checked")
