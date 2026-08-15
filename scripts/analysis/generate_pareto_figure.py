"""Generate SciPost-style Pareto front figure for symbolic regression results.

SciPost style: minimal, grayscale-friendly, LaTeX math labels, no gimmicky
shading or AI-style annotations. Standard scientific scatter with key
equations annotated in a legend-like inset.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.figure_style import add_grid, save_fig, LEGEND_KWARGS

FIT_PATH = EXPERIMENTS / "physics_sr" / "fit_full.json"
FIGURES = EXPERIMENTS.parent / "paper" / "figures"

SELECTED_C = 6


def has_nested(eq: str) -> bool:
    import re
    for fn in ("tanh", "exp", "log", "sqrt"):
        if re.search(rf"\b{fn}\s*\([^)]*\b{fn}\s*\(", eq):
            return True
    return False


def count_features(eq: str) -> int:
    names = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
             "sqrt_d12", "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]
    return max(1, sum(1 for f in names if f in eq))


def short_label(eq: str, max_len: int = 55) -> str:
    eq = eq.replace("Mass_over_pT", "m/p_T").replace("mSD_over_pT", "m_{SD}/p_T")
    eq = eq.replace("Tau32", "\\tau_{32}").replace("Tau21", "\\tau_{21}")
    eq = eq.replace("theta_g", "\\theta_g").replace("zg", "z_g")
    if len(eq) > max_len:
        eq = eq[:max_len - 3] + "..."
    return eq


def main():
    data = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    eqs = data["equations"]

    C = np.array([e["complexity"] for e in eqs])
    L = np.array([e["loss"] for e in eqs])
    nested = np.array([has_nested(e["equation"]) for e in eqs])
    nfeat = np.array([count_features(e["equation"]) for e in eqs])

    # --- SciPost-compliant styling ---
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Clean points (circle) and nested points (cross)
    clean_mask = ~nested
    dirty_mask = nested

    ms_clean = np.where(nfeat[clean_mask] <= 2, 32, 48)
    ms_nested = np.where(nfeat[dirty_mask] <= 2, 32, 48)

    ax.scatter(C[clean_mask], L[clean_mask], marker='o', s=ms_clean,
               facecolors='none', edgecolors='#2c3e50', linewidths=0.7,
               zorder=3, label='No nested functions')
    ax.scatter(C[dirty_mask], L[dirty_mask], marker='x', s=ms_nested,
               color='#922b21', linewidths=1.0,
               zorder=3, label='Contains nested')

    sel_idx = np.where(C == SELECTED_C)[0][0]
    ax.scatter([C[sel_idx]], [L[sel_idx]], marker='D', s=55,
               facecolors='#d4ac0d', edgecolors='black', linewidths=0.8,
               zorder=5, label=f'Selected ($C={SELECTED_C}$)')

    # Light connecting line
    ax.plot(C, L, color='#7f8c8d', alpha=0.35, linewidth=0.8, zorder=1, linestyle='-')

    # Annotations: only key equations, placed with care
    annotations = {
        2:  {"label": "$\\tanh(m/p_T)$", "dx": 0, "dy": 0.022, "ha": "center", "va": "bottom"},
        6:  {"label": "$\\tanh(m/p_T) - 0.35\\,\\tau_{32}$", "dx": 2.0, "dy": 0.008, "ha": "left", "va": "bottom"},
    }

    for c, ann in annotations.items():
        idx = np.where(C == c)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        x, y = C[i], L[i]
        ax.annotate(
            ann["label"],
            xy=(x, y), xytext=(x + ann["dx"], y + ann["dy"]),
            fontsize=8.5, ha=ann["ha"], va=ann["va"],
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.9, linewidth=0.5),
            arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.6),
            zorder=6,
        )

    ax.set_xlabel("Expression complexity $C$", fontsize=12)
    ax.set_ylabel("MSE loss (z-scored logit)", fontsize=12)
    ax.set_xlim(0, 24)
    ax.set_ylim(0.0, 0.30)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.12, which='major')
    ax.grid(True, alpha=0.06, which='minor', linestyle=':')

    ax.legend(loc='upper right', **LEGEND_KWARGS,
              handlelength=1.5, handletextpad=0.5)

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'pareto_front_sr.png')
    print(f"Saved: pareto_front_sr.png")


if __name__ == "__main__":
    main()
