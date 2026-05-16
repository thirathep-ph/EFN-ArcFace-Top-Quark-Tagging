import awkward as ak
import numpy as np
import fastjet
from jet_tagging.data.loader import load_awkward

# 1. Load a small event
events, _, _ = load_awkward('data/top_tagging/test.h5', max_events=1)
px, py, pz, E = events.px, events.py, events.pz, events.E
cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})

# 2. Cluster
jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 1.5)
cs = fastjet.ClusterSequence(cons, jet_def)

# 3. SoftDrop
sd_jets = cs.exclusive_jets_softdrop_grooming(njets=1, beta=0.0, symmetry_cut=0.1, R0=1.5)

# 4. Inspect fields
print("Available fields in sd_jets:")
print(ak.fields(sd_jets))

# Also try looking at the first jet's attributes if it's not in the fields
try:
    print(f"Sample msoftdrop: {sd_jets.msoftdrop}")
except:
    print("Could not access msoftdrop directly")
