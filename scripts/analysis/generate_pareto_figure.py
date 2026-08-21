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

    # --- Publication styling (SciPost) ---
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.9, "xtick.major.width": 0.9,
        "ytick.major.width": 0.9, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    # Clean points (circle) and nested points (cross)
    clean_mask = ~nested
    dirty_mask = nested

    ms_clean = np.where(nfeat[clean_mask] <= 2, 52, 68)
    ms_nested = np.where(nfeat[dirty_mask] <= 2, 52, 68)

    ax.scatter(C[clean_mask], L[clean_mask], marker='o', s=ms_clean,
               facecolors='none', edgecolors='#1f4a5e', linewidths=1.1,
               zorder=3, label='No nested functions')
    ax.scatter(C[dirty_mask], L[dirty_mask], marker='x', s=ms_nested,
               color='#922b21', linewidths=1.4,
               zorder=3, label='Contains nested')

    sel_idx = np.where(C == SELECTED_C)[0][0]
    ax.scatter([C[sel_idx]], [L[sel_idx]], marker='D', s=85,
               facecolors='#d4ac0d', edgecolors='black', linewidths=1.0,
               zorder=5, label=f'Selected ($C={SELECTED_C}$)')

    # Connecting line — slightly stronger for print
    ax.plot(C, L, color='#7f8c8d', alpha=0.55, linewidth=1.0, zorder=1, linestyle='-')

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
            fontsize=9.0, ha=ann["ha"], va=ann["va"],
            color="#1f2e3a",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#9aa4ad", alpha=0.96, linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color="#9aa4ad", lw=0.8),
            zorder=6,
        )

    ax.set_xlabel("Expression complexity $C$", fontsize=11.5)
    ax.set_ylabel("MSE loss (z-scored logit)", fontsize=11.5)
    ax.set_xlim(0, 24)
    ax.set_ylim(0.0, 0.30)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.grid(True, alpha=0.28, which='major', linestyle='-', linewidth=0.6)
    ax.grid(True, alpha=0.14, which='minor', linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', **{**LEGEND_KWARGS, "fontsize": 9, "framealpha": 0.97, "edgecolor": "#7f8c8d"},
              handlelength=1.6, handletextpad=0.6)
    save_fig(fig, 'pareto_front_sr.png')
    print(f"Saved: pareto_front_sr.png")


if __name__ == "__main__":
    main()
