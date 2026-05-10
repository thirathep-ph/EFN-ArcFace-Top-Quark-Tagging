"""
Full EDA + Plots for Top Tagging Dataset
- 3 files: train.h5 (1.2M), val.h5 (400K), test.h5 (400K)
- 804 features: 200 constituents x 4-momenta + 4 truth
- Target labels: 0=QCD, 1=Top (note: val.h5 has label=2, type?)
- NOTE: Dataset is pre-split by class across files
"""
import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 72)
print("  TOP TAGGING DATASET - EDA")
print("=" * 72)

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
fig.suptitle("Top Tagging Dataset - EDA", fontsize=16, fontweight='bold')

all_stats = {}

for idx, fn in enumerate(['train.h5', 'val.h5', 'test.h5']):
    f = h5py.File(fn, 'r')
    tbl = f['table']['table']
    n = tbl.shape[0]
    
    # ── Labels ──
    labels = tbl['values_block_1'][:, 0].astype(np.int8)
    unique, counts = np.unique(labels, return_counts=True)
    label_map = {0: 'QCD (bkg)', 1: 'Top (sig)', 2: 'Other'}
    print(f"\n  {fn}: {n:,} entries")
    for u, c in zip(unique, counts):
        print(f"    Label {int(u)} ({label_map.get(int(u), '?')}): {c:,} ({c/n*100:.1f}%)")
    
    # ── Stratified sample: 50K from each label class ──
    sample_pt = []
    sample_nconst = []
    sample_sumpt = []
    weights = []
    
    for label_val in unique:
        mask = labels == label_val
        n_label = int(mask.sum())
        take = min(50000, n_label)
        step = max(1, n_label // take)
        idxs = np.where(mask)[0][::step][:take]
        
        v0 = tbl['values_block_0'][idxs]
        w = tbl['values_block_1'][idxs, 1]
        pts = v0[:, 0:200]
        
        nconst = (pts > 0).sum(axis=1)
        sorted_pts = np.sort(pts, axis=1)[:, ::-1]
        
        sample_nconst.append(nconst)
        sample_sumpt.append(pts.sum(axis=1))
        sample_pt.append(sorted_pts[:, :5])  # top 5 per event
        weights.append(w)
    
    sample_nconst = np.concatenate(sample_nconst)
    sample_sumpt = np.concatenate(sample_sumpt)
    sample_pt = np.concatenate(sample_pt)
    
    all_stats[fn] = {
        'n': n, 'labels': (unique.tolist(), counts.tolist()),
        'nconst_mean': sample_nconst.mean(),
        'sumpt_mean': sample_sumpt.mean(),
    }
    
    f.close()
    
    # ── Row 0: Label distribution ──
    ax = axes[0, idx]
    colors = {'QCD': 'steelblue', 'Top': 'coral', '2': 'gold'}
    bar_labels = [f"Label {int(u)}" for u in unique]
    bar_colors = [colors.get(str(int(u)), 'gray') for u in unique]
    bars = ax.bar(bar_labels, counts, color=bar_colors, edgecolor='white')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{c:,}\n({c/n*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(f'{fn.replace(".h5","").upper()}\n{n:,} total', fontweight='bold')
    ax.set_ylabel('Count')
    
    # ── Row 1: Constituent count ──
    ax = axes[1, idx]
    ax.hist(sample_nconst, bins=80, color='mediumseagreen', edgecolor='white', alpha=0.8)
    ax.axvline(sample_nconst.mean(), color='darkgreen', ls='--', lw=2,
               label=f'Mean={sample_nconst.mean():.1f}')
    ax.set_xlabel('# Constituents')
    ax.set_ylabel('Events')
    ax.legend(fontsize=8)
    
    # ── Row 2: Sum pT ──
    ax = axes[2, idx]
    ax.hist(sample_sumpt, bins=100, color='mediumpurple', edgecolor='white', alpha=0.8)
    ax.axvline(sample_sumpt.mean(), color='indigo', ls='--', lw=2,
               label=f'Mean={sample_sumpt.mean():.1f}')
    ax.set_xlabel('Sum pT')
    ax.set_ylabel('Events')
    ax.legend(fontsize=8)
    
    # ── Row 3: Top 5 constituent pT (per-event) ──
    ax = axes[3, idx]
    for rank in range(5):
        ax.hist(sample_pt[:, rank], bins=80, alpha=0.5, label=f'#{rank+1}' if idx == 0 else None)
    ax.set_xlabel('pT')
    ax.set_ylabel('Events')
    if idx == 0:
        ax.legend(fontsize=8, title='Constituent rank', title_fontsize=9)

# ── Column 4: Summary ──
ax = axes[0, 3]
datasets = ['train.h5', 'val.h5', 'test.h5']
x = np.arange(len(datasets))
width = 0.35
for i, d in enumerate(datasets):
    labels_list, counts_list = all_stats[d]['labels']
    ax.bar(x[i] - width/2, counts_list[0] if len(counts_list) > 0 else 0, width,
           label=f'Label {int(labels_list[0])}', color=['steelblue', 'coral', 'gold'][i])
    if len(counts_list) > 1:
        ax.bar(x[i] + width/2, counts_list[1], width,
               label=f'Label {int(labels_list[1])}', color='gray')
ax.set_xticks(x)
ax.set_xticklabels(['Train', 'Val', 'Test'])
ax.set_ylabel('Count')
ax.set_title('Label Distribution by Split', fontweight='bold')

ax = axes[1, 3]
ax.set_axis_off()
summary_text = (
    "SUMMARY\n"
    "-------------\n"
    "Train:  1,211,000 (label 0)\n"
    "Val:      403,000 (label 2)\n"
    "Test:     404,000 (label 1)\n"
    "-------------\n"
    "Total: 2,018,000 entries\n\n"
    "Features (804 total):\n"
    " 200 const x 4-momenta\n"
    " (pT, eta, phi, E)\n"
    " + 4 truth momentum\n"
    " + 2 labels/weights\n\n"
    "5-momenta stored:\n"
    " px, py, pz, E, mass\n"
    " (or pT, eta, phi, E)\n\n"
    "File sizes:\n"
    " train: 0.97 GB\n"
    " val:   0.32 GB\n"
    " test:  0.32 GB"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace')

for r in [2, 3]:
    axes[r, 3].set_axis_off()

plt.tight_layout()
plt.savefig('eda_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Print detailed stats ──
print(f"\n{'='*72}")
for fn in datasets:
    s = all_stats[fn]
    print(f"  {fn}: {s['n']:,} entries, ~{s['nconst_mean']:.1f} const/event, avg sum pT={s['sumpt_mean']:.1f}")
print(f"{'='*72}")
print("  EDA COMPLETE -> eda_summary.png")
print(f"{'='*72}")
