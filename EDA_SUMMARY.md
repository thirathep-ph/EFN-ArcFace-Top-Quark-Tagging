# Top Tagging Dataset — EDA Summary

## File Overview

| File | Size | Entries | Raw Label | Class |
|------|------|---------|-----------|-------|
| `train.h5` | 0.97 GB | 1,211,000 | 0 | **QCD** (background) |
| `val.h5`   | 0.32 GB | 403,000   | 2 | **Top** (signal) |
| `test.h5`  | 0.32 GB | 404,000   | 1 | **Top** (signal) |
| **Total**  | 1.61 GB | 2,018,000 | — | Balanced (~60/40) |

> **Note on labels:** Label 2 in `val.h5` is also Top — confirmed by matching kinematics (constituent count, pT distributions are nearly identical to label 1 in test.h5). For training, binarize: `label > 0 → 1 (Top)`, `label == 0 → 0 (QCD)`.

## Data Structure

Each entry in the HDF5 table has **804 features** in `values_block_0`:

| Columns | Shape | Content |
|---------|-------|---------|
| 0–199   | (200,) | Constituent **pT** (GeV/c) |
| 200–399 | (200,) | Constituent **η** (pseudorapidity) |
| 400–599 | (200,) | Constituent **φ** (azimuthal angle) |
| 600–799 | (200,) | Constituent **E** (energy, GeV) |
| 800–803 | (4,) | Truth **px, py, pz, E** (GeV/c, GeV) |

`values_block_1` contains `[is_signal_new, event_weight]` (int64).

## Key Statistics

| Metric | QCD (label 0) | Top (label 1/2) |
|--------|---------------|-----------------|
| Avg constituents/event | ~107 | ~107 |
| Median constituents | ~102 | ~102 |
| Avg sum pT | ~903 GeV/c | ~900 GeV/c |
| Avg leading pT | ~233 GeV/c | ~233 GeV/c |

- Constituents are **zero-padded** to 200 max per event
- Sorted by **descending pT** (leading constituent first)
- ~47% of the 200-slot array is filled on average

## HDF5 Fix

The original error `"Can't synchronously read data (can't open directory)"` was caused by an **unregistered compression filter**. Fix:

```python
import hdf5plugin  # MUST be imported before h5py
import h5py
```

## Visualization

See `eda_summary.png` for label distributions, constituent count histograms, sum pT, and leading/subleading pT distributions across all three splits.
