# Research Log — ArcEFN (code workstream)

SSOT of the work: goal, current position, and what's done. Newest log entry on top; `Now` rewritten every session.

## Goal

59k-param IRC-safe EFN + ArcFace tagger (one class prototype per class on S⁶³) with every reported number traceable to `experiments/*.json`. Code side feeds the paper; paper cites nothing that isn't gated here.

## Now (2026-10-13, HEAD `942bd28`)

- Docs carry the prototype-first frame (`c079543`). `.venv` py3.12.10 + locked deps, pytest 33/33 green.
- `experiments/*.json` (56 entries) byte-identical to paper's frozen copy — nothing to sync.
- Next: only on demand (distance-based follow-ups; feature-permuted retraining is paper future work).

## Done

### 2026-10-13 (`c079543`, `942bd28`)

- Docs reframe to prototype-first: `README.md` + `docs/RESULTS.md` (12+12 lines). Class prototypes = primary goal; Core/Edge subclasses = secondary finding en route.
- Created `.venv` (Python 3.12.10), installed `requirements-locked.txt`, `pip install -e .`. `pytest arcefn/tests`: 33 passed.
- Started this research log (`LOG.md`); refreshed `HANDOFF.md` state.
