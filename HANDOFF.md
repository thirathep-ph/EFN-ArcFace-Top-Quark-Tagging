# HANDOFF — ArcEFN (code workstream)

> Work HERE for models, training, analysis scripts, experiments. Paper lives in `D:\research\ArcEFN-paper`, slides in `D:\research\ArcEFN-talks`. Old mixed repo `D:\research\jet_tagging` is READ-ONLY reference. NO remote configured — do not add/push anywhere without explicit instruction. Written 2026-10-07.

## What this repo is

`arcefn/` (EFN+ArcFace lib, 59k-param tagger, 91.84% acc) + `scripts/` (train/eval/~90 analysis files) + `experiments/*.json` (result SSOT the paper cites) + `docs/` (README, REPRODUCING, RESULTS gallery). HEAD `c079543` (docs prototype-first reframe, 2026-10-13). `.venv` Python 3.12.10 with locked deps + editable install. Running log: `LOG.md`. Contains NO paper sources, NO slides/media, NO logs, NO process docs.

## Key entry points

- Train: `scripts/train_arcface.py --scale 16 --seed 42` (Adam lr=1e-3 defaults, OneCycleLR peak 2e-3, 50 epochs, batch 128 eff. 1024; AMD DirectML box, never `--amp`).
- Canonical run: `experiments/robustness_s16_m05/` (`checkpoint.pt` + 64-D `embeddings.npz`, 404k test events).
- Figures/numbers: `scripts/analysis/compute_*.py` → `experiments/*.json` → `numbers.json` gate (50+ entries).
- Data: `data/` (3.4 GB, gitignored, via `scripts/download_data.py`).

## Rules

1. `experiments/*.json` are the paper's SSOT — reruns must reproduce gated values exactly or stop and report (spectral clustering uses unseeded eigsh; boundary events may flip between runs).
2. Figure scripts: PNG display text may change; JSON keys/values must stay byte-identical.
3. Others' files in the OLD repo (`infographic*`, `dlc*`, `pca_gate*`, `physics_sr` micro-sweeps, `run_*.ps1`) are NOT here and must never be staged/committed anywhere.
4. Private vs public: no hardware notes, machine paths, or secrets in code or commits.

## Handoff to paper

Paper does not rerun experiments — copy fresh `experiments/*.json` → `D:\research\ArcEFN-paper\experiments_frozen/` and note the date/commit. Then paper side rebuilds PDFs + runs its gate.

## Pending

- [ ] Nothing in flight. Next code work (if any): distance-based follow-ups feed paper's dilute story; feature-permuted retraining is future work per paper Limitations.
