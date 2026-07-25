"""Make combined POC figure for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

exp = Path(r"D:\research\jet_tagging\experiments\poc_injection_scan.png")
mass = Path(r"D:\research\jet_tagging\experiments\poc_mass_decoupled.png")

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))

ax = axes[0]
ax.imshow(mpimg.imread(str(exp)))
ax.axis('off')
ax.set_title('(a) Margin stability vs signal fraction', fontsize=11, fontweight='bold', pad=6)

ax = axes[1]
ax.imshow(mpimg.imread(str(mass)))
ax.axis('off')
ax.set_title('(b) Mass-decoupled accuracy', fontsize=11, fontweight='bold', pad=6)

plt.subplots_adjust(hspace=0.12)
out = Path(r"D:\research\jet_tagging\paper\figures\poc_combined.png")
fig.savefig(str(out), dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
