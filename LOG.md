# Research Log — ArcEFN (code workstream)

SSOT of the work: goal, current position, and what's done. Newest log entry on top; `Now` rewritten every session.

## Goal

59k-param IRC-safe EFN + ArcFace tagger (one class prototype per class on S⁶³) with every reported number traceable to `experiments/*.json`. Code side feeds the paper; paper cites nothing that isn't gated here.

## Now (2026-10-13)

- Docs carry the prototype-first frame (`c079543`). `.venv` py3.12.10 + locked deps, pytest 33/33 green.
- `experiments/*.json` (56 entries) byte-identical to paper's frozen copy — nothing to sync.
- Rejection labels de-overlapped; deta/lund generators emit Dense/Dilute filenames (JSON keys untouched); gallery renamed to match. 3 fresh PNGs pushed to paper `paper/figures/`.
- Next: only on demand (distance-based follow-ups; feature-permuted retraining is paper future work).

## Done

### 2026-10-14: ROC legend outside-axes + revert-restore round-trip

- User asked revert-to-pre-capture then changed mind back; both done via git revert (history intact). Net new vs yesterday: ROC legend moved below axes (ncol=3, AUC to caption), inset xticks [0, 0.1], print 0.65 — verified clean at print size. Replotted (zero JSON diff), PNGs pushed to paper + gallery.

### 2026-10-13 p.m.: figure print-size fix (3 replot-capable figs)

- ROC/rejection natives → 14/12/12/13pt; composition → 15/17/20pt with print 0.48→0.90 textwidth. Effective print sizes now ~8-13pt (was 4-8pt). Zero JSON diff, PNGs pushed to paper + gallery.
- Root causes logged for data machine: most generators skip apply_style() (Sans vs Serif mix); print-size contract added to figure_style.py docstring (targets: labels/annots ≥9pt, ticks/legend ≥8pt effective).

### 2026-10-13 p.m.: rejection overlap fix + Dense/Dilute filenames

- Rejection 50%-TPR labels (123.6/121.6/131.7×) staggered to (0.60,260)/(0.62,150)/(0.62,85); text-box no-overlap verified programmatically (caution: `Annotation.get_window_extent` unions the leader arrow — check `Text.get_window_extent`). Replot: zero JSON diff.
- `compute_deta_dphi_planes.py` + `compute_subclass_lund_planes.py`: display-stem (`fstem`) map core/edge → dense/dilute for output filenames; `results["panels"]` keys unchanged. `py_compile` OK (data absent, no full run).
- Renamed 14 tracked PNGs in `docs/figures/` + all `docs/RESULTS.md` refs (table, captions, alt-text) to Dense/Dilute; PNG-internal labels were already Dense/Dilute.

### 2026-10-13 (fig restyle, `5359b63`)

- User: heatmap looked washed out, ROC/rejection used little space with tiny fonts (all printed at 0.48\textwidth). Restyled all three, display-only.
- `compute_subclass_composition.py`: new `render()` (two 2×2 panels Top|QCD, Blues, print-size fonts) + `--replot` from `subclass_composition.json`. Removed now-unused seaborn import.
- `compute_bootstrap_ci.py`: new `replot()` + `--replot`/`--scores-npz`/`--results-json`; ROC gains low-FPR inset + thicker lines + kept 95% CI bands (caption mentions them); rejection annotates gated 50%-TPR values 123.6/121.6/131.7× from archived JSON. Curve data honesty: cached scores in `experiments/paired_scores_cache.npz` (local gitignored copy of old-repo `paired_model_stats.npz`) reproduce all 3 gated AUCs to 10 decimals.
- Verified `git status experiments/` shows only the 2 scripts modified, zero JSON diff. PNGs copied to `D:\research\ArcEFN-paper\paper\figures/`.

### 2026-10-13 (`c079543`, `942bd28`)

- Docs reframe to prototype-first: `README.md` + `docs/RESULTS.md` (12+12 lines). Class prototypes = primary goal; Core/Edge subclasses = secondary finding en route.
- Created `.venv` (Python 3.12.10), installed `requirements-locked.txt`, `pip install -e .`. `pytest arcefn/tests`: 33 passed.
- Started this research log (`LOG.md`); refreshed `HANDOFF.md` state.
