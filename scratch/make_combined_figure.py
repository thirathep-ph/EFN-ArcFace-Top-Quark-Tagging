"""Make combined POC figure for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

exp = Path(r"D:\Jet Tagging\experiments\poc_injection_scan.png")
mass = Path(r"D:\Jet Tagging\experiments\poc_mass_decoupled.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.imshow(mpimg.imread(str(exp)))
ax.axis('off')
ax.set_title('(a) Margin stability vs signal fraction', fontsize=12, fontweight='bold')

ax = axes[1]
ax.imshow(mpimg.imread(str(mass)))
ax.axis('off')
ax.set_title('(b) Mass-decoupled accuracy', fontsize=12, fontweight='bold')

fig.tight_layout()
out = Path(r"D:\Jet Tagging\paper\figures\poc_combined.png")
fig.savefig(str(out), dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
