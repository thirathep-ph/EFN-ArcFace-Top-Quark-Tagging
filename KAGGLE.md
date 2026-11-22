# Running ArcEFN training on Kaggle (free GPU)

Goal: reproduce EFN+ArcFace training for additional seeds (e.g. 67, 404) on
Kaggle's free GPU and ship the resulting checkpoints back.

## What's in `kaggle/`

- `train_arcefn.ipynb` — self-contained notebook (clone repo, download data,
  train one seed, zip checkpoint+history for download).
- `kernel-metadata.json` — metadata for `kaggle kernels push`.
- `make_notebook.py` — regenerates the notebook (edit here, then rerun).

## Prerequisites

1. Kaggle account with **phone verification** (required for GPU + internet).
2. Create an API token: Kaggle → Settings → API → "Create New Token"
   → saves `kaggle.json`. Place it at `~/.kaggle/kaggle.json` (Linux/macOS)
   or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
3. Install the CLI locally (optional, for push): `pip install kaggle`.

## Push the notebook

```bash
cd kaggle
# edit kernel-metadata.json "id" to <your-kaggle-username>/arcefn-train
kaggle kernels push -p .
```

## Run a seed

1. Open the kernel on Kaggle → "Edit".
2. In the config cell set `SEED = 67` (then a second run with `SEED = 404`).
3. Accelerator: **GPU T4 x2** (or P100). Internet: **On**.
4. "Run All". Expected: ~3–9 h for 50 epochs on 1.2M train jets.
   - For a sanity check first, set `MAX_EVENTS = 50000`.
5. When finished, download `arcefn_seed_<SEED>.zip` from the Output tab.

## Runtime notes / limits

- Kaggle notebook sessions cap at ~9 h (GPU) and ~20 GB disk. One seed
  (50 epochs) fits in one session; run the two seeds as two separate sessions.
- The notebook downloads the 3.4 GB Top Tagging H5 from Zenodo on first run
  (internet must be ON). Re-runs reuse the cached file if the session persists.
- Mixed precision (`--amp`) is enabled by default and uses CUDA AMP on Kaggle.
- Checkpoints are saved as `best val acc` only (`checkpoint.pt`) + `history.json`
  + `config.json`, matching the local convention.

## Bringing results back

Unzip into `experiments/seed_<SEED>/` in the local repo, then point analysis
scripts at it via `ARCEFN_MODEL_DIR=experiments/seed_<SEED>` (already
supported by `arcefn/utils/paths.py`). The training code itself is unchanged;
only paths are parameterised.

## Alternative: attach data as a Kaggle Dataset

If Zenodo is slow/blocked, upload `data/top_tagging/*.h5` as a private Kaggle
Dataset and add it under `dataset_sources` in `kernel-metadata.json`; then set
`ARCEFN_DATA_DIR=/kaggle/input/<dataset-slug>` in the config cell. Training
code reads `DATA_DIR` from that env var (see `arcefn/utils/paths.py`).

## Reproducibility

Same command as local: `python scripts/train_arcface.py --seed <SEED> --epochs 50 --amp`.
Seeds 42/123/7 are the existing ones; 67 and 404 are the new runs requested.
""")

print("written")