"""Shared Figure Composition Standards for Publication Figures
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
import matplotlib as mpl

# ── Fonts ─────────────────────────────────────────────────────────────────
FONT_FAMILY = "Arial"
FONT_SIZE = 11
LABEL_SIZE = 12
TITLE_SIZE = 13.5
TICK_SIZE = 11
LEGEND_SIZE = 10
AUTHOR_FACTOR = 4.0 / 3.0

# ── Color Palette (CERN/HEP official) ──────────────────────────────────────
# CERN blue (#0053a6) for QCD/signal, CERN red (#c8171f) for Top/background.
# Background: white, Grid: light gray (#cccccc), Text: black.
# Follows CERN Graphics Guidelines and CMS/ATLAS plotting conventions.
COLORS = {
    "qcd":       "#0053a6",   # CERN blue (signal/QCD)
    "top":       "#c8171f",   # CERN red (background/Top)
    "qcd_dilute":  "#7aa6d9", # light CERN blue
    "top_dense":   "#e88a8f", # light CERN red
    "green":     "#009e73",   # colorblind-safe green (Okabe-Ito)
    "purple":    "#cc79a7",   # colorblind-safe purple (Okabe-Ito)
    "orange":    "#e69f00",   # colorblind-safe orange (Okabe-Ito)
    "brown":     "#a65628",   # colorblind-safe brown
    "gray":      "#cccccc",   # grid/axes gray
}
CLASS_COLORS = {"QCD": COLORS["qcd"], "Top": COLORS["top"]}
SUBCLASS_COLORS = {
    "QCD dense":  COLORS["qcd"],
    "QCD dilute":  COLORS["qcd_dilute"],
    "Top dilute":  COLORS["top"],
    "Top dense":  COLORS["top_dense"],
}

# ── Line / Marker Defaults ───────────────────────────────────────────────
LINEWIDTH = 1.2
MARKERSIZE = 4
ALPHA = 0.7

# ── Legend ─────────────────────────────────────────────────────────────────
PAPER_FIGURES = {}

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
GRID_KWARGS = dict(alpha=0.3, linestyle="--", linewidth=0.5, color="#c5c5c5")

# ── Figure DPI ─────────────────────────────────────────────────────────────
DPI = 200


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

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
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "grid.color":       "#c5c5c5",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })


def add_grid(ax, **kwargs):
    """Add consistent grid to an axes."""
    kw = {**GRID_KWARGS, **kwargs}
    ax.grid(**kw)


def save_fig(fig, path, dpi=None, to_paper=False, **kwargs):
    """Save figure with consistent settings."""
    import matplotlib.pyplot as plt
    if dpi is None:
        dpi = DPI
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax, label, fontsize=TITLE_SIZE, loc="left", pad=6):
    """Add a bold panel label like $\\mathbf{(a)}$ to an axes."""
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
            leg._ncols = kw["ncol"]


def fig_size(printed_w_in, aspect=1.0):
    """Return authored canvas size for a given printed width."""
    authored_w = printed_w_in * AUTHOR_FACTOR
    authored_h = authored_w * aspect
    return (authored_w, authored_h)
