"""
Explore FastJet Python's internal modules for vectorized/multi-event API.
"""
import fastjet

print("=== _multievent ===")
print(dir(fastjet._multievent))
print()

print("=== _pyjet ===")
print(dir(fastjet._pyjet))
print()

print("=== _singleevent ===")
print(dir(fastjet._singleevent))
print()

print("=== _generalevent ===")
print(dir(fastjet._generalevent))
print()

# Check if there's a way to cluster multiple events at once
print("=== ClusterSequence docs ===")
help(fastjet.ClusterSequence.__init__)
print()

# Check if there's a batch_cluster_sequence or similar
print("=== Searching for batch/vectorized methods ===")
for name in dir(fastjet):
    obj = getattr(fastjet, name)
    if callable(obj) and ('batch' in name.lower() or 'vector' in name.lower() or 'multi' in name.lower()):
        print(f"  {name}")
print()

# Check the interfacemulti module
print("=== interfacemulti ===")
print(dir(fastjet._ext.interfacemulti))
print()

# Check output_wrapper
print("=== output_wrapper ===")
print(dir(fastjet._ext.output_wrapper))
