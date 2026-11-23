"""Multi-seed Pareto fronts: do all seeds reach the same equation form?

For each seed, PySR returns a whole front (one equation per complexity).
Plotting loss vs complexity with one line per seed shows:
  - the fronts overlap, i.e. no seed finds a better trade-off, and
  - at a given complexity the equations have the same form.

Writes experiments/physics_sr/pysr_multiseed_fronts.json, then the figure
fig13_multiseed_fronts.png (staging dir + paper review dir).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pysr_config import (   # noqa: E402
    BINARY_OPERATORS, BUDGET, COMPLEXITY_OF_OPERATORS, RAW_FEATURES,
    UNARY_OPERATORS, get_constraints, get_nested_constraints, normalize,
)

EXP = ROOT / "experiments"
SR = EXP / "physics_sr"
OUT = SR / "pysr_multiseed_fronts.json"

SEEDS = [42, 123, 7, 2024, 31337]
VARS = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
        "sqrt_d12_over_pT", "sqrt_d23_over_pT",
        "Tau21", "Tau32", "zg", "theta_g"]


def load_data():
    raw = np.column_stack([
        pq.read_table(EXP / "robustness_s16_m05" / "test_features.parquet",
                      columns=[c]).column(c).to_numpy(zero_copy_only=False)
        for c in RAW_FEATURES]).astype(np.float64)
    L = np.load(SR / "logits.npz")
    y = L["logitdiff"].astype(np.float64)
    labels = L["labels"].astype(np.int64)
    pt = np.maximum(L["jet_pt"].astype(np.float64), 1e-3)
    X = raw.copy()
    for i in (0, 1, 4, 5):
        X[:, i] = np.maximum(X[:, i], 0) / pt
    Xn, block = normalize(X, "z")
    return Xn, y, labels, block


def main() -> None:
    from pysr import PySRRegressor
    from sklearn.metrics import roc_auc_score

    Xn, y, labels, block = load_data()
    cfg = BUDGET["quick"]

    runs = {}
    t_all = time.time()
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        t0 = time.time()
        m = PySRRegressor(
            model_selection="best",
            niterations=cfg["niterations"],
            populations=cfg["populations"],
            population_size=cfg["population_size"],
            binary_operators=BINARY_OPERATORS,
            unary_operators=UNARY_OPERATORS,
            maxsize=cfg["maxsize"], maxdepth=cfg["maxdepth"],
            complexity_of_operators=COMPLEXITY_OF_OPERATORS,
            nested_constraints=get_nested_constraints("self-identity"),
            constraints=get_constraints(),
            parsimony=0.01, procs=1, random_state=seed,
            verbosity=0, progress=False, output_jax_format=False,
        )
        m.fit(Xn, y, variable_names=VARS)

        rows, seen = [], set()
        for i in range(len(m.equations_)):
            r = m.equations_.iloc[i]
            key = (int(r["complexity"]), str(r["equation"]))
            if key in seen:
                continue
            seen.add(key)
            try:
                pred = m.predict(Xn, index=i)
                auc = float(roc_auc_score(labels, pred))
                if auc < 0.5:
                    auc = 1.0 - auc
            except Exception:
                auc = None
            rows.append({"complexity": int(r["complexity"]),
                         "loss": float(r["loss"]),
                         "equation": str(r["equation"]),
                         "auc": auc})
        runs[seed] = rows
        print(f"  {len(rows)} front points in {time.time()-t0:.0f}s", flush=True)
        for r in rows:
            print(f"    c={r['complexity']:3d} loss={r['loss']:8.3f}  {r['equation'][:66]}")

    out = {"seeds": SEEDS, "target": "logitdiff",
           "normalization": block, "variable_names": VARS,
           "config": cfg, "runs": {str(k): v for k, v in runs.items()},
           "runtime_seconds": time.time() - t_all}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {OUT}  ({time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
