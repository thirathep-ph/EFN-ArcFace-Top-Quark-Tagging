"""Real SSOT check: resolve every pointer, compare to expect, flag drift.

Resolution is delegated to create_ssot.resolve so this file can never drift
from the generator. Exit code 0 iff every entry resolves and is within tol.
"""
import json
import sys

sys.path.insert(0, "scripts/analysis")
from create_ssot import resolve  # noqa: E402

SSOT = "experiments/ssot_paper_numbers.json"
data = json.load(open(SSOT, encoding="utf-8"))["entries"]

bad = 0
for e in data:
    name = e.get("name", "?")
    art = e.get("artifact") or ""
    ptr = e.get("pointer") or ""
    exp = e.get("expect")
    tol = float(e.get("tol", 1e-9))
    scale = float(e.get("scale", 1))
    if not ptr:
        print(f"NO-POINTER  {name}")
        bad += 1
        continue
    try:
        got = resolve(art, ptr, ".")
    except Exception as exc:
        print(f"UNRESOLVED  {name:32s} {art} :: {ptr}  "
              f"({type(exc).__name__}: {exc})")
        bad += 1
        continue
    if not isinstance(got, (int, float)):
        print(f"NON-NUMERIC {name:32s} {art} :: {ptr} -> {got!r}")
        bad += 1
        continue
    val = float(got) * scale
    if abs(val - float(exp)) > tol:
        print(f"DRIFT       {name:32s} expect {exp}  got {val:.6g}  "
              f"({art} :: {ptr})")
        bad += 1

print()
print(f"entries {len(data)}   problems {bad}")

# paper_text vs artifact: the manuscript may print an older version.
text_bad = 0
for e in data:
    t = e.get("paper_text")
    if not t:
        continue
    try:
        pv = float(t.replace("\\%", ""))
    except ValueError:
        continue
    if abs(pv - float(e["expect"])) > float(e.get("tol", 1e-9)):
        print(f"TEXT-STALE  {e['name']:32s} paper prints {t}  artifact "
              f"{e['expect']}")
        text_bad += 1
print(f"paper_text mismatches {text_bad}")

sys.exit(1 if (bad or text_bad) else 0)
