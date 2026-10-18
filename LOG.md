# Research Log — ArcEFN (code workstream)

SSOT of the work: goal, current position, and what's done. Newest log entry on top; `Now` rewritten every session.

## Goal

59k-param IRC-safe EFN + ArcFace tagger (one class prototype per class on S⁶³) with every reported number traceable to `experiments/*.json`. Code side feeds the paper; paper cites nothing that isn't gated here.

## Now (2026-10-14)

- Deep naming sweep DONE: JSON keys renamed Dense/Dilute everywhere (`experiments/*.json`, numbers.json names+pointers, SUBCLASS_NAMES, panels keys, slopes maps, cos_* fields). Values untouched. 59 files (37 JSON, 20 .py), parse-verified.
- Paper `experiments_frozen/` mirror renamed identically (40 files); paper gate re-run ALL GREEN (66/66 numbers, 63/63 refs, 39/39 floats).
- LOG.md historical entries scrubbed of legacy naming too (this log is now fully Dense/Dilute wording).

## Done

### 2026-10-14 p.m.: deep JSON-key rename to Dense/Dilute (user: go deep to JSON keys)

- User chose deepest scope: keys themselves renamed, values never touched. Full deterministic sweep (ordered REPL list, temp script; QCD: inner endpoint→Dense, outer→Dilute; Top: inner→Dilute, outer→Dense; axis/field names carry endpoint order).
- Code repo: 59 files (37 JSON + 20 .py), all JSON parse OK, all touched .py py_compile OK. Paper repo: 40 files (37 JSON), gate ALL GREEN after rename.
- numbers.json pointers/fnames auto-renamed; zero legacy residue outside LOG.md history; infographic regenerated.
- LOG.md is historical narration — legacy wording in entries below untouched by design.

### 2026-10-14 p.m.: repo renamed + naming/verification rounds

- User renamed GitHub repo → `EFN-ArcFace-Top-Quark-Tagging`; updated local remote URL + README badge/bib example + REPRODUCING git-hygiene URL + infographic footer (commit `9233643`); infographic PNG regenerated with Dense/Dilute labels (`808c51f`). All pushes --force-with-lease, remote main synced.
- PS 5.1 `Set-Content` corrupted non-ASCII files (BOM+mojibake) → lesson: textual edits via `.venv python` pathlib one-liners only, newline=''.
- Scan for oddities: 72 tracked files contain machine paths (benign provenance in result JSONs); LICENSE present; git identity matches paper correspondence email. Paper repo pushed private to `thirathep-ph/ArcEFN-paper` (feat/py-input) separately.
- Still open user-side: revoke 2 leaked PATs, dismiss alerts #1/#2, check Security log, then delete prepurge bundle in Temp.

### 2026-10-14: purged 2 leaked GitHub PATs from history (secret-scanning alerts #1 #2)

- Tokens lived in early notebook commits (be316dc, a6e560c…), gone from worktree since 9c04807 but present in pushed history. Purged via filter-repo replace-text (`ghp_*`/`github_pat_*`/`ghs_*`/`ghu_*` → REMOVED) across all 208 commits; verified zero `ghp_` in any blob (`log -S` + full-tree grep). ALL commit hashes rewritten; backup bundle kept at Temp (arcEFN-prepurge.bundle, 85MB) + pre-purge remote copy.
- Self-inflicted wounds logged: (1) piped long-run output through `Select -First`, which killed filter-repo mid-run — always redirect to log file; (2) filter-repo strips `origin` remote — re-add before pushing.
- Pushed purged `main` via --force-with-lease. USER STILL MUST: revoke both tokens in GitHub settings (check whether auto-revoked), review Security log for misuse, dismiss/resolve alerts #1 #2 in repo Security tab.

### 2026-10-14: pushed to public GitHub (user: wipe + replace)

- Pre-push hygiene verified: tree clean, .venv/data/checkpoints untracked, no secrets in tracked files (two .png binary false positives only), pack ~83MB. Machine-local paths inside result JSONs kept as-is (benign, already in history; SSOT integrity wins over scrubbing).
- `origin` added (URL = paper citation URL), fetched baseline, branched `main` at HEAD, `push --force-with-lease` (remote had 8 unrelated commits, 0 stars/forks — wiped per explicit order). `feat/py-input` stays local-only.

### 2026-10-14: ROC legend outside-axes + revert-restore round-trip

- User asked revert-to-pre-capture then changed mind back; both done via git revert (history intact). Net new vs yesterday: ROC legend moved below axes (ncol=3, AUC to caption), inset xticks [0, 0.1], print 0.65 — verified clean at print size. Replotted (zero JSON diff), PNGs pushed to paper + gallery.

### 2026-10-13 p.m.: figure print-size fix (3 replot-capable figs)

- ROC/rejection natives → 14/12/12/13pt; composition → 15/17/20pt with print 0.48→0.90 textwidth. Effective print sizes now ~8-13pt (was 4-8pt). Zero JSON diff, PNGs pushed to paper + gallery.
- Root causes logged for data machine: most generators skip apply_style() (Sans vs Serif mix); print-size contract added to figure_style.py docstring (targets: labels/annots ≥9pt, ticks/legend ≥8pt effective).

### 2026-10-13 p.m.: rejection overlap fix + Dense/Dilute filenames

- Rejection 50%-TPR labels (123.6/121.6/131.7×) staggered to (0.60,260)/(0.62,150)/(0.62,85); text-box no-overlap verified programmatically (caution: `Annotation.get_window_extent` unions the leader arrow — check `Text.get_window_extent`). Replot: zero JSON diff.
- `compute_deta_dphi_planes.py` + `compute_subclass_lund_planes.py`: display-stem (`fstem`) map subclass output filenames to dense/dilute; `results["panels"]` keys unchanged. `py_compile` OK (data absent, no full run).
- Renamed 14 tracked PNGs in `docs/figures/` + all `docs/RESULTS.md` refs (table, captions, alt-text) to Dense/Dilute; PNG-internal labels were already Dense/Dilute.

### 2026-10-13 (fig restyle, `5359b63`)

- User: heatmap looked washed out, ROC/rejection used little space with tiny fonts (all printed at 0.48\textwidth). Restyled all three, display-only.
- `compute_subclass_composition.py`: new `render()` (two 2×2 panels Top|QCD, Blues, print-size fonts) + `--replot` from `subclass_composition.json`. Removed now-unused seaborn import.
- `compute_bootstrap_ci.py`: new `replot()` + `--replot`/`--scores-npz`/`--results-json`; ROC gains low-FPR inset + thicker lines + kept 95% CI bands (caption mentions them); rejection annotates gated 50%-TPR values 123.6/121.6/131.7× from archived JSON. Curve data honesty: cached scores in `experiments/paired_scores_cache.npz` (local gitignored copy of old-repo `paired_model_stats.npz`) reproduce all 3 gated AUCs to 10 decimals.
- Verified `git status experiments/` shows only the 2 scripts modified, zero JSON diff. PNGs copied to `D:\research\ArcEFN-paper\paper\figures/`.

### 2026-10-13 (`c079543`, `942bd28`)

- Docs reframe to prototype-first: `README.md` + `docs/RESULTS.md` (12+12 lines). Class prototypes = primary goal; Dense/Dilute subclasses = secondary finding en route.
- Created `.venv` (Python 3.12.10), installed `requirements-locked.txt`, `pip install -e .`. `pytest arcefn/tests`: 33 passed.
- Started this research log (`LOG.md`); refreshed `HANDOFF.md` state.
