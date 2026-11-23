"""Judge pass: verify parquet cache + load_cached_features vs compute_features."""
import sys

sys.path.insert(0, ".")

# NOTE: arcefn imports (which load awkward) BEFORE pyarrow —
# importing pyarrow first crashes the awkward native import in this venv.
from arcefn.utils.physics import compute_features, load_cached_features
from arcefn.data.loader import load_awkward
from arcefn.utils.paths import DATA_DIR

import numpy as np
import pyarrow.parquet as pq

PASS = []

def check(name, fn):
    try:
        fn()
        PASS.append((name, "PASS", ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        PASS.append((name, "FAIL", str(e)))
        print(f"  [FAIL] {name}: {e}")


def check1():
    t = pq.read_table("experiments/robustness_s16_m05/test_features.parquet")
    assert t.num_rows == 404000, f"rows={t.num_rows}"
    assert t.num_columns == 12, f"cols={t.num_columns}"
    assert "event_idx" in t.column_names and "label" in t.column_names


def check2():
    f = load_cached_features(None)
    assert f.shape == (404000, 10), f"shape={f.shape}"
    assert np.all(np.isfinite(f)), "non-finite values"
    t = pq.read_table("experiments/robustness_s16_m05/test_features.parquet",
                      columns=["Mass", "mSD", "Mult", "nSD"])
    assert np.isclose(f[0, 0], t.column("Mass")[0].as_py()), "Mass row0 mismatch"


def check3():
    events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=50, lazy=False)
    f_cache = load_cached_features(None)[:50]  # from parquet
    f_ref = compute_features(events)           # fresh recompute on 50 events
    assert f_cache.shape == (50, 10), f"shape={f_cache.shape}"
    np.testing.assert_allclose(f_cache, f_ref, rtol=1e-5, atol=1e-5)
    print("    max abs diff:", np.max(np.abs(f_cache - f_ref)))


print("Judge pass:")
check("parquet rows=404000, cols=12", check1)
check("load_cached_features shape (404000,10) & finite", check2)
check("cross-check compute_features vs load_cached_features (50 events)", check3)

n_fail = sum(1 for _, s, _ in PASS if s == "FAIL")
print(f"\nTotal: {len(PASS)}, Fail: {n_fail}")
sys.exit(1 if n_fail else 0)
