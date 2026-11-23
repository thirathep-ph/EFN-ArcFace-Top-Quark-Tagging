"""Cache jet_pt and m/pT — CORRECTED layout.

values_block_0 layout per constituent (stride 4, 200 constituents):
    [E, px, py, pz]  x200  ->  columns 0..799
    columns 800..803 = event-level extras (unused here)

Verified by column statistics: c0 (offset 0) is always positive and largest
(energy), c1/c2/c3 are ~50% negative (momentum components).

jet pT = |sum_i (px_i, py_i)|  (vector sum, not sum of magnitudes)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tables

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP = PROJECT_ROOT / "experiments" / "robustness_s16_m05"
OUT = EXP / "jet_pt_mpt_cache.npz"

with tables.open_file(str(PROJECT_ROOT / "data" / "top_tagging" / "test.h5"), "r") as f:
    B = f.get_node("/table/table").read()["values_block_0"]

E = B[:, 0:800:4]
px = B[:, 1:800:4]
py = B[:, 2:800:4]
pz = B[:, 3:800:4]

# validate: E^2 >= p^2 for real particles
p2 = px**2 + py**2 + pz**2
frac_ok = float(np.mean(E**2 >= p2 - 1e-3))
jet_pt = np.sqrt(px.sum(axis=1) ** 2 + py.sum(axis=1) ** 2)

t = pq.read_table(EXP / "test_features.parquet", columns=["event_idx", "Mass", "Tau32"])
mass = t.column("Mass").to_numpy(zero_copy_only=False)
tau32 = t.column("Tau32").to_numpy(zero_copy_only=False)
idx = t.column("event_idx").to_numpy(zero_copy_only=False)
order = np.argsort(idx, kind="stable")
mass = mass[order]
tau32 = tau32[order]

mpt = mass / jet_pt

np.savez(OUT, jet_pt=jet_pt, mass=mass, tau32=tau32, mpt=mpt,
         E_sum=E.sum(axis=1))

print(f"layout check  E^2>=p^2 fraction = {frac_ok:.4f}  (expect ~1.0)")
print(f"jet_pt  median={np.median(jet_pt):.1f}  p5={np.percentile(jet_pt,5):.1f}  "
      f"p95={np.percentile(jet_pt,95):.1f}")
print(f"mass    median={np.median(mass):.1f}")
print(f"m/pT    median={np.median(mpt):.4f}")
print(f"saved   {OUT}")

summary = {
    "layout": "per constituent [E, px, py, pz] stride 4, cols 0..799",
    "E2_ge_p2_fraction": frac_ok,
    "jet_pt_median": float(np.median(jet_pt)),
    "mpt_median": float(np.median(mpt)),
    "n_events": int(len(jet_pt)),
}
with open(EXP / "jet_pt_mpt_cache_meta.json", "w") as f:
    json.dump(summary, f, indent=2)
