"""Multi-seed Pareto fronts: do all seeds reach the same closed form?

x = equation complexity, y = loss.  One line per PySR seed.
If the seeds converge to one form, the five fronts overlap: no seed finds a
better trade-off, and at a given complexity the equations look alike.

Right panel: for each complexity, the equation form found by each seed,
colour-coded by family, so identical cells across rows = same form.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arcefn.utils.figure_style import apply_style   # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

apply_style()

SR = ROOT / "experiments" / "physics_sr"
OUTDIR = Path("D:/research/ArcEFN-paper/paper/figures/review/pysr")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED_COLORS = ["#0053a6", "#c8171f", "#009e73", "#e69f00", "#cc79a7"]

ALL_VARS = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD", "sqrt_d12_over_pT",
            "sqrt_d23_over_pT", "Tau21", "Tau32", "zg", "theta_g"]
CORE = {"Mass_over_pT", "Tau32"}


def family(eq: str) -> str:
    e = eq.replace(" ", "")
    used = {v for v in ALL_VARS if v in e}
    extra = used - CORE
    has_tanh = "tanh(" in e
    has_mass, has_tau = "Mass_over_pT" in used, "Tau32" in used

    if has_tanh and re.search(r"tanh\([^()]*(?:\([^()]*)*exp\(", e):
        return "tanh(exp)"
    if has_tanh and re.search(r"tanh\([^()]*Mass_over_pT[^()]*-[\d.]*\*?Tau32", e):
        return "tanh(m/pT - k*tau32)"
    if has_tanh and re.search(
            r"tanh\([^()]*Mass_over_pT[^()]*\)[^()]*?[+\-/][^()]*?\*?Tau32", e):
        return "tanh(m/pT) - k*tau32" if not extra else "tanh(m/pT)+other"
    if has_tanh and has_mass and not extra:
        return "tanh(m/pT) only"
    if has_tanh and has_tau and not has_mass:
        return "tanh(tau32) only"
    if has_tanh:
        return "tanh(...) other"
    if has_mass and has_tau and not extra:
        return "linear(m/pT,tau32)"
    if has_mass and not extra:
        return "linear(m/pT)"
    if has_tau and not extra:
        return "linear(tau32)"
    if has_mass and has_tau:
        return "linear(m/pT,tau32)+other"
    if has_mass:
        return "linear(m/pT)+other"
    if has_tau:
        return "linear(tau32)+other"
    return "other"


def main() -> None:
    d = json.load(open(SR / "pysr_multiseed_fronts.json"))
    seeds = d["seeds"]
    runs = {int(k): v for k, v in d["runs"].items()}
    print("fronts:", {s: len(runs[s]) for s in seeds})

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # ── left: loss vs complexity, one line per seed ──────────────────────
    ax = axes[0]
    for i, s in enumerate(seeds):
        rows = sorted(runs[s], key=lambda r: r["complexity"])
        cx = np.array([r["complexity"] for r in rows], float)
        ls = np.array([r["loss"] for r in rows], float)
        ax.plot(cx, ls, "o-", color=SEED_COLORS[i % 5], lw=1.5, ms=5.5,
                alpha=0.9, label=f"seed {s}")
    ax.set_xlabel("equation complexity")
    ax.set_ylabel("loss (MSE on model logit)")
    ax.set_title("Five seeds, five Pareto fronts: they overlap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # ── right: form matrix (seed x complexity) ───────────────────────────
    ax = axes[1]
    cxs = sorted({r["complexity"] for s in seeds for r in runs[s]})
    fams = sorted({family(r["equation"]) for s in seeds for r in runs[s]})
    cmap = plt.cm.get_cmap("tab10", len(fams))
    lookup = {f: i for i, f in enumerate(fams)}
    grid = np.full((len(seeds), len(cxs)), np.nan)
    for i, s in enumerate(seeds):
        for r in runs[s]:
            j = cxs.index(r["complexity"])
            grid[i, j] = lookup[family(r["equation"])]

    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=-0.5,
              vmax=len(fams) - 0.5)
    ax.set_xticks(range(len(cxs)))
    ax.set_xticklabels(cxs)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel("equation complexity")
    ax.set_title("Same form at the same complexity")
    for i in range(len(seeds)):
        for j in range(len(cxs)):
            if np.isnan(grid[i, j]):
                continue
            eqs = [r for r in runs[seeds[i]] if r["complexity"] == cxs[j]]
            f = family(eqs[0]["equation"])
            ax.text(j, i, f.split("(")[0][:9], ha="center", va="center",
                    fontsize=6, color="white")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(lookup[f]),
                             label=f) for f in fams]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              bbox_to_anchor=(1.01, 1))

    fig.suptitle("Independent PySR seeds converge to the same closed-form "
                 "trade-off (404k jets, z-norm, self-identity nesting)")
    fig.tight_layout()
    out = OUTDIR / "fig13_multiseed_fronts.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ── report: agreement at matched complexity ───────────────────────────
    print("\nform agreement at matched complexity:")
    agree = tot = 0
    for c in cxs:
        fams_c = []
        for s in seeds:
            hit = [r for r in runs[s] if r["complexity"] == c]
            if hit:
                fams_c.append(family(hit[0]["equation"]))
        if len(fams_c) < 2:
            continue
        top = max(set(fams_c), key=fams_c.count)
        agree += fams_c.count(top)
        tot += len(fams_c)
        print(f"  c={c:3d}: {fams_c.count(top)}/{len(fams_c)} -> {top}")
    print(f"\noverall agreement: {agree}/{tot} = "
          f"{100*agree/tot:.0f}%")
    print(f"saved: {out}  {out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
