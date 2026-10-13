# Research Log — ArcEFN (code workstream)

SSOT of the work: goal, current position, and what's done. Newest log entry on top; `Now` rewritten every session.

## Goal

59k-param IRC-safe EFN + ArcFace tagger (one class prototype per class on S⁶³) with every reported number traceable to `experiments/*.json`. Code side feeds the paper; paper cites nothing that isn't gated here.

## Now (2026-10-13, HEAD `942bd28`)

- Docs carry the prototype-first frame (`c079543`). `.venv` py3.12.10 + locked deps, pytest 33/33 green.
- `experiments/*.json` (56 entries) byte-identical to paper's frozen copy — nothing to sync.
- 3 figures restyled via new `--replot` modes (PNG-only, JSONs untouched); copies pushed to paper `paper/figures/`. Paper chat still must: rebuild PDFs+bundles, run gate, update `LOG.md` there.
- Next: only on demand (distance-based follow-ups; feature-permuted retraining is paper future work).

## Done

### 2026-10-13 (fig restyle, uncommitted → commit below)

- User: heatmap looked washed out, ROC/rejection used little space with tiny fonts (all printed at 0.48\textwidth). Restyled all three, display-only.
- `compute_subclass_composition.py`: new `render()` (two 2×2 panels Top|QCD, Blues, print-size fonts) + `--replot` from `subclass_composition.json`. Removed now-unused seaborn import.
- `compute_bootstrap_ci.py`: new `replot()` + `--replot`/`--scores-npz`/`--results-json`; ROC gains low-FPR inset + thicker lines + kept 95% CI bands (caption mentions them); rejection annotates gated 50%-TPR values 123.6/121.6/131.7× from archived JSON. Curve data honesty: cached scores in `experiments/paired_scores_cache.npz` (local gitignored copy of old-repo `paired_model_stats.npz`) reproduce all 3 gated AUCs to 10 decimals.
- Verified `git status experiments/` shows only the 2 scripts modified, zero JSON diff. PNGs copied to `D:\research\ArcEFN-paper\paper\figures/`.

### 2026-10-13 (`c079543`, `942bd28`)

- Docs reframe to prototype-first: `README.md` + `docs/RESULTS.md` (12+12 lines). Class prototypes = primary goal; Core/Edge subclasses = secondary finding en route.
- Created `.venv` (Python 3.12.10), installed `requirements-locked.txt`, `pip install -e .`. `pytest arcefn/tests`: 33 passed.
- Started this research log (`LOG.md`); refreshed `HANDOFF.md` state.
