"""Aggregate SR eval cells into a summary artifact for the numbers checker.

Usage: python -m scripts.analysis.summarize_sr_cells <tag-prefix> <out.json>
  e.g. norm  -> matches eval_norm_*_full.json, groups by norm x seed
  e.g. stgB  -> matches eval_stgB_*_full.json, groups by config x norm
Reads best_auc_equation.complexity, finds its metrics in equations[].
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SR = ROOT / "experiments" / "physics_sr"


def cell_metrics(path):
    d = json.loads(path.read_text())
    c = d["best_auc_equation"]["complexity"]
    eq = next(e for e in d["equations"] if e["complexity"] == c)
    s = eq["equation"]
    return {
        "auc": round(eq["metrics"]["auc"], 4),
        "acc": round(eq["metrics"]["acc"], 4),
        "complexity": c,
        "mass_led": "Mass_over_pT" in s,
        "has_log": "log(" in s,
        "has_div": bool(re.search(r"(?<!\w)/", s)),
    }


def main():
    prefix, out = sys.argv[1], SR / sys.argv[2]
    cells = {}
    for p in sorted(SR.glob(f"eval_{prefix}_*_full.json")):
        tag = p.stem[len("eval_"):-len("_full")]
        try:
            cells[tag] = cell_metrics(p)
        except Exception as e:  # noqa: BLE001
            cells[tag] = {"error": str(e)}
    out.write_text(json.dumps({"cells": cells}, indent=1) + "\n")
    ok = sum(1 for v in cells.values() if "error" not in v)
    print(f"[summarize] {ok}/{len(cells)} cells -> {out.name}")


if __name__ == "__main__":
    main()
