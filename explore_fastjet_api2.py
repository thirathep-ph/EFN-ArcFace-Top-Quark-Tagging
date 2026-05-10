"""
Explore the vectorized FastJet API with Awkward Arrays.
"""
import fastjet
import awkward as ak
import numpy as np
from to_awkward import load_awkward

print("=" * 60)
print("Exploring FastJet Vectorized API")
print("=" * 60)

# 1. Check AwkwardClusterSequence
print("\n--- AwkwardClusterSequence ---")
print(dir(fastjet._pyjet.AwkwardClusterSequence))
print()

# Check its __init__ signature
print("AwkwardClusterSequence.__init__:")
help(fastjet._pyjet.AwkwardClusterSequence.__init__)
print()

# 2. Check DaskAwkwardClusterSequence
print("\n--- DaskAwkwardClusterSequence ---")
print(dir(fastjet._pyjet.DaskAwkwardClusterSequence))
print()

# 3. Check output_wrapper methods in detail
print("\n--- output_wrapper methods ---")
ow = fastjet._ext.output_wrapper
methods = [m for m in dir(ow) if m.startswith('to_numpy')]
for m in methods:
    print(f"  {m}")
print()

# 4. Check _classmultievent
print("\n--- _classmultievent ---")
print(dir(fastjet._multievent._classmultievent))
print()

# 5. Check _classsingleevent
print("\n--- _classsingleevent ---")
print(dir(fastjet._singleevent._classsingleevent))
print()

# 6. Check _classgeneralevent
print("\n--- _classgeneralevent ---")
print(dir(fastjet._generalevent._classgeneralevent))
print()

# 7. Test the vectorized API with real data
print("\n--- Testing Vectorized API ---")
events, labels, weights = load_awkward('train.h5', max_events=100, trim_padding=True)
print(f"Loaded {len(events)} events")

# Convert to PseudoJet array
# The vectorized API expects an awkward array of PseudoJets
# Let's check what format it needs
print(f"Events type: {type(events)}")
print(f"Events[0] type: {type(events[0])}")
print(f"Events layout: {ak.type(events)}")
