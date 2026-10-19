"""
Shared Figure Composition Standards for Publication Figures
==========================================================
All paper figures should import and use these constants/functions
for visual consistency (font, color, legend, panel labels, grid).

UNIVERSAL FIGURE CONTRACT (the one rule for every paper figure)
---------------------------------------------------------------
Authored canvas width = printed width * AUTHOR_FACTOR (4/3), and the
native font sizes below are fixed.  Then EVERY figure prints its text at
the SAME physical size, whatever width the paper gives it:

    effective_pt = native_pt * printed_in / authored_in
                = native_pt * 3/4
    ->  labels/annotations 9.0pt, ticks/legend 8.25pt, titles 10.1pt

So a figure printed at 0.48\\textwidth and one printed at 1.0\\textwidth
are typographically identical, and re-sizing a figure in LaTeX can
never make its text illegible.  Use fig_size() to get the canvas:

    fig, ax = plt.subplots(figsize=fig_size(PRINTED_W_IN, aspect))

The gate script scripts/paper_checks/check_figures.py enforces this
for every \\includegraphics in the paper (effective font in band, spec
entry present, printed width matching the registry).

Every figure generator MUST call apply_style() (DejaVu Serif) --
plots rendered without it fall back to DejaVu Sans and break the
paper's single-font rule. Display-only refonts never touch JSON.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
PAPER_FIGURES = _REPO / "paper" / "figures"
EXPERIMENTS = _REPO / "experiments"

# ── Typography ──────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Serif"

# ── UNIVERSAL FIGURE CONTRACT ───────────────────────────────────────────────
# paper text width (article class, letter) = 345pt = 4.792in
PAPER_TEXTWIDTH_IN = 345.0 / 72.0
# authored canvas = printed width * 4/3  ->  effective font = native * 3/4
AUTHOR_FACTOR = 4.0 / 3.0
NATIVE_LABEL = 12.0     # -> 9.00 pt in print
NATIVE_TICK = 11.0      # -> 8.25 pt in print
NATIVE_LEGEND = 11.0    # -> 8.25 pt in print
NATIVE_TITLE = 13.5     # -> 10.13 pt in print
NATIVE_ANNOT = 11.0     # -> 8.25 pt in print

FONT_SIZE = NATIVE_TICK
TICK_SIZE = NATIVE_TICK
LEGEND_SIZE = NATIVE_LEGEND
LABEL_SIZE = NATIVE_LABEL
TITLE_SIZE = NATIVE_TITLE


def fig_size(printed_width_in, aspect=None, height_in=None):
    """Canvas size (inches) for a figure that will be printed at
    `printed_width_in` inches in the paper.

    aspect: width/height of the canvas (default 1.6)
    height_in: explicit canvas height (overrides aspect)

    The canvas is AUTHOR_FACTOR times the printed width so that the
    native font sizes above land at their target size in print.
    """
    w = printed_width_in * AUTHOR_FACTOR
    h = height_in if height_in is not None else w / (aspect or 1.6)
    return (w, h)


def printed_width(frac_textwidth):
    """Printed width in inches for a \\textwidth fraction."""
    return frac_textwidth * PAPER_TEXTWIDTH_IN

# ── Color Palette (muted, colorblind-safe) ─────────────────────────────────
COLORS = {
    "qcd":       "#377eb8",
    "top":       "#e41a1c",
    "qcd_dilute":  "#9ecae1",
    "top_dense":  "#fdae6b",
    "green":     "#4daf4a",
    "purple":    "#984ea3",
    "orange":    "#ff7f00",
    "brown":     "#a65628",
    "gray":      "#999999",
}
CLASS_COLORS = {"QCD": COLORS["qcd"], "Top": COLORS["top"]}
SUBCLASS_COLORS = {
    "QCD dense":  COLORS["qcd"],
    "QCD dilute":  COLORS["qcd_dilute"],
    "Top dilute":  COLORS["top"],
    "Top dense":  COLORS["top_dense"],
}

# ── Line / Marker Defaults ─────────────────────────────────────────────────
LINEWIDTH = 1.2
MARKERSIZE = 4
ALPHA = 0.7

# ── Legend ──────────────────────────────────────────────────────────────────
LEGEND_KWARGS = dict(
    frameon=True,
    fancybox=True,
    framealpha=0.92,
    edgecolor="0.45",
    facecolor="white",
    fontsize=LEGEND_SIZE,
    ncol=1,
)

# ── Grid ───────────────────────────────────────────────────────────────────
GRID_KWARGS = dict(alpha=0.15, linestyle="--", linewidth=0.4)

# ── Figure DPI ─────────────────────────────────────────────────────────────
DPI = 200


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def apply_style():
    """Call once at top of script to set global rcParams."""
    mpl.rcParams.update({
        "font.family":      FONT_FAMILY,
        "font.size":        FONT_SIZE,
        "axes.labelsize":   LABEL_SIZE,
        "axes.titlesize":   TITLE_SIZE,
        "xtick.labelsize":  TICK_SIZE,
        "ytick.labelsize":  TICK_SIZE,
        "legend.fontsize":  LEGEND_SIZE,
        "figure.dpi":       DPI,
        "savefig.dpi":      DPI,
        "savefig.bbox":     "tight",
        "axes.linewidth":   0.6,
        "lines.linewidth":  LINEWIDTH,
        "lines.markersize": MARKERSIZE,
    })


def panel_label(ax, label, fontsize=TITLE_SIZE, loc="left", pad=6):
    """Add a bold panel label like $\mathbf{(a)}$ to an axes."""
    title = f"$\\mathbf{{{label}}}$"
    if loc == "left":
        ax.set_title(title, fontsize=fontsize, loc="left", fontweight="bold", pad=pad)
    else:
        ax.set_title(title, fontsize=fontsize, loc=loc, pad=pad)


def style_legend(ax, **kwargs):
    """Apply consistent legend styling. Pass extra kwargs to override defaults."""
    kw = {**LEGEND_KWARGS, **kwargs}
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame(**{k: v for k, v in kw.items()
                         if k in ("frameon", "fancybox", "framealpha", "edgecolor", "facecolor")})
        if "fontsize" in kw:
            for t in leg.get_texts():
                t.set_fontsize(kw["fontsize"])
        if "ncol" in kw:
            leg._ncol = kw["ncol"]


def add_grid(ax, **kwargs):
    """Add grid with consistent styling."""
    kw = {**GRID_KWARGS, **kwargs}
    ax.grid(True, **kw)
    ax.set_axisbelow(True)


def save_fig(fig, name, to_paper=True, **kwargs):
    """Save figure to experiments/ and optionally copy to paper/figures/.
    Uses ARCEFN_MODEL_DIR env var for model-specific subdirectory, falls back to EXPERIMENTS.
    """
    import os
    from shutil import copy2
    model_dir_env = os.environ.get('ARCEFN_MODEL_DIR', '').strip()
    if model_dir_env:
        exp_path = Path(model_dir_env) / name
    else:
        exp_path = EXPERIMENTS / name
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(exp_path), **{"dpi": DPI, "bbox_inches": "tight", **kwargs})
    if to_paper:
        copy2(str(exp_path), str(PAPER_FIGURES / name))
    plt.close(fig)
    print(f"  Saved: {name}")
