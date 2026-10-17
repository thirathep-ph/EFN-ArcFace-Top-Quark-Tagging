"""README hero infographic: clean, simple pipeline.

Reads values from experiments/numbers.json (SSOT). Output: docs/infographic.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
NUMBERS = {e["name"]: e["expect"] for e in
           json.load(open(ROOT / "experiments" / "numbers.json"))["entries"]
           if "expect" in e}
OUT = ROOT / "docs" / "infographic.png"


def val(key):
    return NUMBERS[key]


NAVY = "#1b2a4a"
TEAL = "#0e7c7b"
AMBER = "#c77d0a"
MUTE = "#9aa5b1"

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(7, 6.45, "EFN + ArcFace Top Quark Tagging", ha="center", fontsize=24,
        fontweight="bold", color=NAVY)
ax.text(7, 6.0, "Nearest class prototype on a hypersphere — geometrically readable",
        ha="center", fontsize=12, color=MUTE)


def box(x, y, w, h, title, lines, color=TEAL, tsize=13, lsize=11):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14",
                       facecolor="white", edgecolor=color, linewidth=2.5)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.4, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold", color=color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.95 - 0.45 * i, ln, ha="center", va="top",
                fontsize=lsize, color="#222222")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, color=MUTE, linewidth=2))


# pipeline: fewer, bigger boxes, more whitespace
steps = [
    ("Constituents", ["(E, px, py, pz)", "up to 200 / jet"]),
    ("Phi MLP", ["[2 -> 128 x3]", "angles only"]),
    ("Rho MLP", ["[128 -> 128 -> 64]", "jet embedding"]),
    ("Hypersphere", ["L2 -> S^63", "ArcFace m=0.5"]),
    ("Prototypes", ["nearest of 2", "Top / QCD"]),
]
py, ph = 3.7, 1.5
sw = 2.35
gap = (14 - 2 * 0.6 - 5 * sw) / 4
for i, (t, ln) in enumerate(steps):
    x = 0.6 + i * (sw + gap)
    box(x, py, sw, ph, t, ln, color=TEAL if i < 4 else AMBER)
    if i < len(steps) - 1:
        arrow(x + sw + 0.04, py + ph / 2, x + sw + gap - 0.04, py + ph / 2)

# one results box
acc = val("arcface_acc_s16_pct")
aci = (val("bootstrap_ci_lower_pct"), val("bootstrap_ci_upper_pct"))
box(0.6, 1.25, 12.8, 1.75, "Results", [
    f"Accuracy {acc:.2f}%  [{aci[0]:.2f}-{aci[1]:.2f}]   AUC {val('auc_arcface'):.3f}   Rej@50% {val('rejection_tpr50'):.0f}x",
    f"Params 59,072  |  QCD-Dense / QCD-Dilute / Top-Dilute / Top-Dense (ARI 0.79)  |  Top-Dense non-identifiable",
], color=NAVY, tsize=13, lsize=11)

ax.text(7, 0.75, "github.com/thirathep-ph/EFN-ArcFace-Top-Quark-Tagging",
        ha="center", fontsize=10, color=TEAL)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT, OUT.stat().st_size, "bytes")
