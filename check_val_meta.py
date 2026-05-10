"""Check val.h5 metadata for label explanation."""
import h5py
import hdf5plugin
import numpy as np

f = h5py.File('val.h5', 'r')
g = f['table']
t = g['table']

print("=== Table columns ===")
print(t.dtype.names)

print("\n=== values_block_0 metadata ===")
print(t.dtype['values_block_0'].metadata)

print("\n=== values_block_1 metadata ===")
print(t.dtype['values_block_1'].metadata)

print("\n=== Table attributes ===")
for k, v in t.attrs.items():
    print(f"  {k}: {v}")

print("\n=== Group attributes ===")
for k, v in g.attrs.items():
    print(f"  {k}: {v}")

print("\n=== Root attributes ===")
for k, v in f.attrs.items():
    print(f"  {k}: {v}")

# Check label distribution
labels = t['values_block_1'][()]
print("\n=== Label distribution ===")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  label={u}: {c} events ({c/len(labels)*100:.1f}%)")

# Check if label=2 events look like Top (high mass)
data = t['values_block_0'][()]
# Column 0 = E of first particle, check a few events
print("\n=== Sample E of first particle for each label ===")
for label_val in [0, 1, 2]:
    mask = labels == label_val
    idx = np.where(mask)[0][:5]
    print(f"  label={label_val}: first particle E = {data[idx, 0]}")
